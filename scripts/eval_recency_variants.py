"""
Calibration audit C9 (2026-10-20): within-season recency, RECENCY_DECAY 0.85 (a player's game k games back weighs
0.85 ** (k - 1) in his current-season rate), re-scored offline from
scripts/capture_stat_blend_traces.py --recency-variants --xtd-variants (no builds).

The blend's game count is the raw count, so a different decay moves only the current rate:
    blended' = blended + w * (cur' - cur)                    direct-rate stats
    out'     = out + c * w * (cur' - cur)                    rushing/receiving TDs, which pass through the expected-TD step
                                                             (c = events / (events + K_xtd), events = TDs + 0.7 x last season's)
Each row's stat moves by blended' / blended and points follow through the shipped late steps (paired START pools, weekly
bootstrap CI, leave-one-season-out). One decay for every position, then one position at a time, then by stat family.

    python scripts/eval_recency_variants.py [--traces .sweeps/recency_capture.parquet]
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

import data.weekly_projections as wp  # noqa: E402
from scripts.eval_blend_modifiers import board  # noqa: E402
from scripts.eval_stat_k_sweep import drmse, load_joined, paired_pool  # noqa: E402

GRID = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0)
SHIPPED = 0.85
KEYS = ['year', 'week', 'player', 'pos', 'stat']
XTD_STATS = ('rushing_tds', 'receiving_tds')
FAMILY = {'passing_attempts': 'volume', 'rushing_attempts': 'volume', 'targets': 'volume',
          'passing_completions': 'catches', 'receptions': 'catches',
          'passing_yards': 'yards', 'rushing_yards': 'yards', 'receiving_yards': 'yards',
          'passing_tds': 'tds', 'rushing_tds': 'tds', 'receiving_tds': 'tds', 'passing_interceptions': 'tds'}


def load(trace_path):
    d, t = load_joined(trace_path)
    t = t.copy()
    t['_i'] = np.arange(len(t))
    x = pd.read_parquet(trace_path + '.xtd.parquet').drop_duplicates(KEYS + ['team'], keep=False)
    k = x['stat'].map(wp.XTD_TD_EVENT_K).fillna(8.0)
    ev = x['cur_td'] + wp.XTD_PRIOR_TD_WEIGHT * x['pri_td']
    x['c'] = ev / (ev + k)
    t = t.merge(x[KEYS + ['team', 'c']], on=KEYS + ['team'], how='left')
    r = pd.read_parquet(trace_path + '.recency.parquet').drop_duplicates(KEYS, keep=False)
    r[f'cur_{SHIPPED:g}'] = r['shipped']
    j = t[['_i', 'team', 'current_weight', 'current_rate', 'blended_rate', 'c'] + KEYS].merge(r, on=KEYS, how='inner')
    td = j['stat'].isin(XTD_STATS).to_numpy()
    # an expected-TD row with no recorded c cannot be moved exactly: leave it alone
    j['slope'] = j['current_weight'].to_numpy(float) * np.where(td, j['c'].to_numpy(float), 1.0)
    j = j[np.isfinite(j['slope']) & np.isfinite(j['shipped'])].reset_index(drop=True)
    return d, t, j


def final_under(t, j, decay_for):
    """decay_for(pos, stat) -> decay for that row (SHIPPED leaves it unchanged)."""
    fn = t['final_projection'].to_numpy(float).copy()
    decays = np.array([decay_for(p, s) for p, s in zip(j['pos'], j['stat'])], float)
    cur_new = np.full(len(j), np.nan)
    for dcy in np.unique(decays):
        m = decays == dcy
        cur_new[m] = j[f'cur_{dcy:g}'].to_numpy(float)[m]
    delta = np.nan_to_num(cur_new - j['shipped'].to_numpy(float))
    blended = j['blended_rate'].to_numpy(float)
    new = np.clip(blended + j['slope'].to_numpy(float) * delta, 0.0, None)
    ratio = np.where(blended > 1e-9, new / np.where(blended > 1e-9, blended, 1.0), 1.0)
    idx = j['_i'].to_numpy()
    fn[idx] = fn[idx] + t['pre_vacancy_projection'].to_numpy(float)[idx] * (ratio - 1.0)
    return fn


def held_out(d, base, var, y):
    act = d['actual'].to_numpy(float)
    mask = paired_pool(d, base, var) & (d['year'].to_numpy() == y)
    return float(np.sqrt(((var[mask] - act[mask]) ** 2).mean()) - np.sqrt(((base[mask] - act[mask]) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', default='.sweeps/recency_capture.parquet')
    a = ap.parse_args()
    pd.set_option('display.width', 220)
    d, t, j = load(a.traces)
    gap = np.abs(j['shipped'] - j['current_rate'])
    print(f"rows moved {len(j):,} of {len(t):,} trace rows; recorded shipped rate vs traced current rate: "
          f"99th pct gap {np.nanpercentile(gap, 99):.4f} (3-decimal rounding)")
    base = board(d, t, t['final_projection'].to_numpy(float))
    act = d['actual'].to_numpy(float)
    years = d['year'].to_numpy()

    print('\n=== one decay for every position and stat (shipped 0.85) ===')
    rows = []
    for dcy in GRID:
        if dcy == SHIPPED:
            continue
        var = board(d, t, final_under(t, j, lambda p, s, x=dcy: x))
        r = drmse(d, base, var)
        row = {'decay': dcy, 'START-ALL': r['d'], 'ci': f"[{r['lo']:+.4f},{r['hi']:+.4f}]", 'yrs better': f"{r['yrs_better']}/{r['yrs']}"}
        for pos in ('QB', 'RB', 'WR', 'TE'):
            row[pos] = drmse(d, base, var, pos)['d']
        rows.append(row)
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print('\n=== one position at a time (others shipped): that position\'s START dRMSE ===')
    rows, boards = [], {}
    for pos in ('QB', 'RB', 'WR', 'TE'):
        for dcy in GRID:
            if dcy == SHIPPED:
                continue
            var = board(d, t, final_under(t, j, lambda p, s, x=dcy, q=pos: x if p == q else SHIPPED))
            boards[(pos, dcy)] = var
            r = drmse(d, base, var, pos)
            rows.append({'pos': pos, 'decay': dcy, 'pos START dRMSE': r['d'], 'ci': f"[{r['lo']:+.4f},{r['hi']:+.4f}]",
                         'yrs better': f"{r['yrs_better']}/{r['yrs']}", 'START-ALL': drmse(d, base, var)['d']})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print('\n=== one stat family at a time, every position (START-ALL) ===')
    rows = []
    for fam in ('volume', 'catches', 'yards', 'tds'):
        rec = {'family': fam}
        for dcy in GRID:
            if dcy == SHIPPED:
                continue
            var = board(d, t, final_under(t, j, lambda p, s, x=dcy, f=fam: x if FAMILY.get(s) == f else SHIPPED))
            rec[f'{dcy:g}'] = drmse(d, base, var)['d']
        rows.append(rec)
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print('\n=== leave one season out: each position\'s decay chosen on four seasons, applied together to the fifth ===')
    out = []
    for y in sorted(np.unique(years)):
        chosen = {}
        for pos in ('QB', 'RB', 'WR', 'TE'):
            best, best_r = SHIPPED, 0.0
            for (p2, dcy), var in boards.items():
                if p2 != pos:
                    continue
                mask = paired_pool(d, base, var) & (years != y) & (d['pos'] == pos).to_numpy()
                r_ = float(np.sqrt(((var[mask] - act[mask]) ** 2).mean()) - np.sqrt(((base[mask] - act[mask]) ** 2).mean()))
                if r_ < best_r:
                    best, best_r = dcy, r_
            chosen[pos] = best
        var = board(d, t, final_under(t, j, lambda p, s, c=chosen: c.get(p, SHIPPED)))
        out.append({'held-out': int(y), 'START-ALL': held_out(d, base, var, y),
                    'chosen': ' '.join(f'{p}={v:g}' for p, v in chosen.items())})
    loyo = pd.DataFrame(out)
    print(loyo.round(4).to_string(index=False))
    print(f"mean {loyo['START-ALL'].mean():+.4f}, better in {(loyo['START-ALL'] < 0).sum()} of {len(loyo)}")


if __name__ == '__main__':
    main()
