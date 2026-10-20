"""
Offline sweep of the in-season current-vs-prior weight per (position, stat) (backlog M2), from the traces that
scripts/capture_stat_blend_traces.py recorded.  No builds: every variant is re-scored from the captured inputs.

    blended  = w * current_rate + (1 - w) * prior_rate,   w = games / (games + k_eff)     (k_eff carries K, role confidence
                                                                                          and the role-change acceleration)
    variant  : k_eff' = s * k_eff   (s < 1 trusts this season sooner, s > 1 leans on last season longer)
    stat'    = stat + pre_vacancy * (blended' / blended - 1)        (vacancy, capacity and the later steps are additive after)
    points   : the board raw total moves by weight * (stat' - stat) (yardage first goes back through top-end compression),
               then the points line, the season anchor and the Questionable cut are re-applied - the shipped late steps.

Exact for the volume stats, receptions and yards (the blend identity holds to rounding); the TD stats go through the xTD
path and are NOT re-scored here (backlog M3).  Unscored volume stats (targets, attempts, completions) are read at the stat
level only: in the shipped model they reach points only through capacity and vacancy, which this re-score does not redo.

Pools follow harness v2: per (year, week, position) the union of both arms' top-N (STARTABLE_N) over ALL rows, restricted
to players with a box score; weekly bootstrap for the CI.

    python scripts/eval_stat_k_sweep.py [--traces .sweeps/stat_blend_traces.parquet]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_projections import (QUESTIONABLE_WORKLOAD_FACTORS, SEASON_ANCHOR_S, SIDELINED_AVAILABILITY,  # noqa: E402
                                     TOP_END_YARDAGE)
from scripts.eval_calibration_schemes import WEIGHTS, anchor, load_dump, shipped_line, top_n_mask  # noqa: E402

KEYS = ['year', 'week', 'player', 'pos', 'team']
# the stats the blend identity holds for (TDs excluded), by position
GROUPS = {
    'QB': ('passing_attempts', 'passing_completions', 'passing_yards', 'rushing_attempts', 'rushing_yards'),
    'RB': ('rushing_attempts', 'rushing_yards', 'targets', 'receptions', 'receiving_yards'),
    'WR': ('targets', 'receptions', 'receiving_yards'),
    'TE': ('targets', 'receptions', 'receiving_yards'),
}
SCALES = (0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
BUCKETS = ((1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 20))
SCORED = set(WEIGHTS)                    # stats that carry points in the shipped line


def load_joined(trace_path, weeks=(3, 17)):
    """The board file written by the same builds as the traces (so the two always agree), joined to the trace rows."""
    d = load_dump(trace_path + '.board.csv')
    d = d[d['week'].between(*weeks)].drop_duplicates(KEYS, keep=False).reset_index(drop=True)
    d['wk_key'] = d['year'] * 100 + d['week']
    d['start_raw'] = top_n_mask(d, d['raw'].to_numpy())
    t = pd.read_parquet(trace_path)
    t = t[t['week'].between(*weeks) & t['build_path'].astype(str).str.startswith('direct')]
    t = t.merge(d[KEYS].assign(_row=np.arange(len(d))), on=KEYS, how='inner')
    t = t.drop_duplicates(KEYS + ['stat'], keep=False)
    return d, t


def late_steps(d, raw):
    """The shipped steps after the stat line, from a raw total: the points line, the season anchor, the Questionable cut."""
    dd = d.copy()
    dd['raw'] = np.clip(raw, 0.0, None)
    shown = anchor(shipped_line(dd), dd, SEASON_ANCHOR_S)
    q = d['inj'].astype(str).str.lower().str.startswith('questionable').to_numpy()
    for pos, f in QUESTIONABLE_WORKLOAD_FACTORS.items():
        m = q & (d['pos'] == pos).to_numpy() & (d['avail'] > SIDELINED_AVAILABILITY).to_numpy()
        shown[m] = np.round(shown[m] * f, 2)
    return shown


def stat_delta(t, scale):
    """Per trace row: change in the final stat when k_eff is scaled by `scale` (0 where nothing blends)."""
    g = t['current_games'].to_numpy(float)
    w = t['current_weight'].to_numpy(float)
    cur = np.nan_to_num(t['current_rate'].to_numpy(float))
    prior = np.nan_to_num(t['prior_rate'].to_numpy(float))
    blended = t['blended_rate'].to_numpy(float)
    ok = (g > 0) & (w > 0) & (w < 1) & (blended > 1e-9)
    ke = np.where(ok, g * (1 - w) / np.where(w > 0, w, 1.0), 0.0)
    w2 = np.where(ok, g / np.where(ok, g + scale * ke, 1.0), w)
    blended2 = w2 * cur + (1 - w2) * prior
    ratio = np.where(ok, blended2 / np.where(blended > 1e-9, blended, 1.0), 1.0)
    return t['pre_vacancy_projection'].to_numpy(float) * (ratio - 1.0)


def variant_board(d, t, scales):
    """The shipped points under `scales` = {(pos, stat): s}; rows of other groups stay as shipped. Top-end yardage is applied
    to the pre-top-end stat (final minus its traced delta), so a changed blend is compressed exactly as the board does."""
    raw = d['raw'].to_numpy(float).copy()
    for (pos, stat), s in scales.items():
        if s == 1.0:
            continue
        sub = t[(t['pos'] == pos) & (t['stat'] == stat)]
        if sub.empty:
            continue
        rows = sub['_row'].to_numpy()
        final = sub['final_projection'].to_numpy(float)
        new = np.clip(final + stat_delta(sub, s), 0.0, None)
        if stat in TOP_END_YARDAGE and pos in TOP_END_YARDAGE[stat][0]:
            _, knot, share = TOP_END_YARDAGE[stat]
            pre = final - np.nan_to_num(sub['top_end_delta'].to_numpy(float))
            pre2 = np.clip(pre + stat_delta(sub, s), 0.0, None)
            live = ((d['avail'].to_numpy(float)[rows] > SIDELINED_AVAILABILITY) & (d['raw'].to_numpy(float)[rows] > 0.005)
                    & (d['week'].to_numpy()[rows] >= 3))
            new = np.where(live & (pre2 > knot), pre2 - share * (pre2 - knot), pre2)
        raw[rows] += WEIGHTS.get(stat, 0.0) * (new - final)
    return late_steps(d, raw)


def paired_pool(d, a, b):
    return (top_n_mask(d, a) | top_n_mask(d, b)) & d['played'].to_numpy() & d['actual'].notna().to_numpy()


def week_sq(d, pred, mask):
    e2 = (pred - d['actual'].to_numpy(float)) ** 2
    return pd.DataFrame({'k': d['wk_key'].to_numpy()[mask], 'e2': e2[mask]}).groupby('k')['e2'].agg(['sum', 'count'])


def drmse(d, base, var, pos=None, n_boot=1000, seed=3):
    mask = paired_pool(d, base, var)
    if pos:
        mask &= (d['pos'] == pos).to_numpy()
    if mask.sum() == 0:
        return None
    b, v = week_sq(d, base, mask), week_sq(d, var, mask)
    n = b['count'].sum()
    delta = float(np.sqrt(v['sum'].sum() / n) - np.sqrt(b['sum'].sum() / n))
    rng = np.random.default_rng(seed)
    bs, vs, bc = b['sum'].to_numpy(), v['sum'].to_numpy(), b['count'].to_numpy()
    out = []
    for _ in range(n_boot):
        i = rng.choice(len(bs), len(bs))
        out.append(np.sqrt(vs[i].sum() / bc[i].sum()) - np.sqrt(bs[i].sum() / bc[i].sum()))
    yrs = []
    for y in np.unique(d['year'].to_numpy()[mask]):
        m = mask & (d['year'].to_numpy() == y)
        bb, vv = week_sq(d, base, m), week_sq(d, var, m)
        yrs.append(np.sqrt(vv['sum'].sum() / vv['count'].sum()) - np.sqrt(bb['sum'].sum() / bb['count'].sum()))
    return {'d': delta, 'lo': float(np.percentile(out, 2.5)), 'hi': float(np.percentile(out, 97.5)), 'n': int(n),
            'yrs_better': int(sum(x < 0 for x in yrs)), 'yrs': len(yrs)}


def start_rows(d, t, pos, stat):
    """Trace rows of one group for START-pool players with a box score."""
    sub = t[(t['pos'] == pos) & (t['stat'] == stat)]
    sub = sub[sub['played'] & sub['actual'].notna()]
    return sub[d['start_raw'].to_numpy()[sub['_row'].to_numpy()]]


def stat_level(d, t, pos, stat, scales=SCALES):
    """RMSE of the projected STAT itself on the base START pool (played), by scale; the only readout for volume stats."""
    sub = start_rows(d, t, pos, stat)
    out = {}
    for s in scales:
        fin = np.clip(sub['final_projection'].to_numpy(float) + stat_delta(sub, s), 0.0, None)
        out[s] = float(np.sqrt(((fin - sub['actual'].to_numpy(float)) ** 2).mean()))
    return out, len(sub)


def bucket_optimum(d, t, pos, stat):
    """Per games-played bucket, the scale of k_eff that minimises the stat squared error on the START pool."""
    sub = start_rows(d, t, pos, stat)
    sub = sub[sub['current_games'] > 0]
    grid = np.array([0.1, 0.2, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    one = int(np.where(grid == 1.0)[0][0])
    rows = []
    for lo, hi in BUCKETS:
        b = sub[sub['current_games'].between(lo, hi)]
        if len(b) < 40:
            continue
        act = b['actual'].to_numpy(float)
        mse = [float(((np.clip(b['final_projection'].to_numpy(float) + stat_delta(b, s), 0.0, None) - act) ** 2).mean())
               for s in grid]
        rows.append({'games': f'{lo}' if lo == hi else f'{lo}-{hi}', 'n': len(b), 'best scale': grid[int(np.argmin(mse))],
                     'mean w': float(b['current_weight'].mean()), 'mse gain %': 100 * (min(mse) / mse[one] - 1)})
    return pd.DataFrame(rows)


def group_year_table(d, t, base):
    """Squared error and rows of the position START pool, per (group, scale, year), for the scored stats."""
    years = sorted(d['year'].unique())
    act = d['actual'].to_numpy(float)
    rows = []
    for pos, stats in GROUPS.items():
        for stat in stats:
            if stat not in SCORED:
                continue
            for sc in SCALES:
                var = base if sc == 1.0 else variant_board(d, t, {(pos, stat): sc})
                mask = paired_pool(d, base, var) & (d['pos'] == pos).to_numpy()
                for y in years:
                    m = mask & (d['year'].to_numpy() == y)
                    rows.append({'pos': pos, 'stat': stat, 'scale': sc, 'year': y,
                                 'sse': float(((var[m] - act[m]) ** 2).sum()), 'n': int(m.sum())})
    return pd.DataFrame(rows)


def loyo_joint(d, t, base, table):
    """Leave one season out: pick each group scale on the other seasons, apply them all together to the held-out one."""
    out = []
    for y in sorted(d['year'].unique()):
        chosen = {}
        for (pos, stat), g in table.groupby(['pos', 'stat']):
            f = g[g['year'] != y].groupby('scale')[['sse', 'n']].sum()
            chosen[(pos, stat)] = float((f['sse'] / f['n']).idxmin())      # per row: each scale has its own paired pool
        var = variant_board(d, t, chosen)
        m = (d['year'].to_numpy() == y)
        mask = paired_pool(d, base, var) & m
        act = d['actual'].to_numpy(float)
        b = float(np.sqrt(((base[mask] - act[mask]) ** 2).mean()))
        v = float(np.sqrt(((var[mask] - act[mask]) ** 2).mean()))
        out.append({'held-out year': int(y), 'dRMSE START-ALL': v - b,
                    'chosen': ' '.join(f'{p}.{st[:4]}={sc:g}' for (p, st), sc in chosen.items() if sc != 1.0) or 'all 1'})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', default='.sweeps/stat_blend_traces.parquet')
    a = ap.parse_args()
    pd.set_option('display.width', 220)
    d, t = load_joined(a.traces)
    base = variant_board(d, t, {})
    gap = np.abs(base - d['final'].to_numpy(float))
    print(f"joined rows {len(d):,} player-weeks, {len(t):,} stat rows, years {sorted(d['year'].unique())}, "
          f"weeks {int(d['week'].min())}-{int(d['week'].max())}")
    print(f"sanity: late steps from the raw total vs the board final points - median gap {np.median(gap):.3f}, "
          f"95th pct {np.percentile(gap, 95):.3f}, max {gap.max():.3f}")
    chk = t.merge(d[['p_receptions']].assign(_row=np.arange(len(d))), on='_row')
    chk = chk[chk['stat'] == 'receptions']
    print(f"sanity: trace final receptions vs board p_receptions - max gap {(chk['final_projection'] - chk['p_receptions']).abs().max():.3f}")

    print('\n=== 1. optimal k_eff scale by games played (stat-level MSE, START pool; <1 = trust this season sooner) ===')
    for pos, stats in GROUPS.items():
        for stat in stats:
            tab = bucket_optimum(d, t, pos, stat)
            if tab.empty:
                continue
            print(f'\n{pos} {stat}')
            print(tab.round(3).to_string(index=False))

    print('\n=== 2. one group at a time: change in START RMSE (points) and in the stat own RMSE, by scale ===')
    rows = []
    for pos, stats in GROUPS.items():
        for stat in stats:
            lvl, n = stat_level(d, t, pos, stat)
            for s in SCALES:
                if s == 1.0:
                    continue
                row = {'pos': pos, 'stat': stat, 'scale': s, 'n': n, 'stat RMSE %': 100 * (lvl[s] / lvl[1.0] - 1)}
                if stat in SCORED:
                    var = variant_board(d, t, {(pos, stat): s})
                    r = drmse(d, base, var, pos)
                    if r:
                        row.update({'dPts RMSE': r['d'], 'ci': f"[{r['lo']:+.3f},{r['hi']:+.3f}]",
                                    'yrs better': f"{r['yrs_better']}/{r['yrs']}"})
                rows.append(row)
    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False))
    res.to_csv('.sweeps/stat_k_sweep_groups.csv', index=False)

    print('\n=== 3. leave-one-season-out: scales chosen on four seasons, applied jointly to the fifth ===')
    table = group_year_table(d, t, base)
    table.to_csv('.sweeps/stat_k_sweep_year_table.csv', index=False)
    loyo = loyo_joint(d, t, base, table)
    print(loyo.round(4).to_string(index=False))
    print(f"mean {loyo['dRMSE START-ALL'].mean():+.4f}, better in {(loyo['dRMSE START-ALL'] < 0).sum()} of {len(loyo)} seasons")


if __name__ == '__main__':
    main()
