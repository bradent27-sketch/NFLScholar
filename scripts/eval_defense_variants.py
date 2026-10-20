"""
Calibration audit C2-C5 (2026-10-20): the defense-allowed profile re-scored offline from
scripts/capture_stat_blend_traces.py --defense-variants (no builds).

The capture recorded, for every (year, week, position channel, defense, stat), the overall defense rating as shipped and the
two profiles it is blended from under other settings:

    current season : neutral league-average games Kc in {4, 8, 12, 20} x per-game recency decay {0.75, 0.85, 0.95}
    last season    : neutral games Kp in {4, 12, 20} x late-season floor {0.6, 0.8, 1.0}
    rating         = a * current + (1 - a) * last,  a = n / (n + Kb)   (n = this season's defense games)

Shipped: Kc = Kp = Kb = 12 (one constant, DEFENSE_PRIOR_GAMES, does all three jobs), decay 0.85 (the player recency decay,
reused), floor 0.80. A player's matchup multiplier moves by the change in his opponent's rating (an upper bound where a role
rating also feeds it), is re-clipped to MATCHUP_CLIP, optionally scaled toward 1 by a strength s, and his stat moves by the
same ratio; points follow through the shipped late steps (paired START pools, weekly bootstrap CI, per-season signs).

Rows whose WR/TE targets/receptions/yards matchup comes from the PFF alignment profile (2024-25) are not moved: that is C1.
TE receiving rows of PFF seasons are left out too (their scheme multiplier is not traced).

    python scripts/eval_defense_variants.py [--traces .sweeps/defense_capture.parquet]
"""
import argparse
import itertools
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
from scripts.eval_stat_k_sweep import drmse, load_joined  # noqa: E402

SHIPPED = {'kc': 12.0, 'decay': 0.85, 'kp': 12.0, 'floor': 0.8, 'kb': 12.0, 'strength': 1.0}
KC, DECAY, KP, FLOOR = (4.0, 8.0, 12.0, 20.0), (0.75, 0.85, 0.95), (4.0, 12.0, 20.0), (0.6, 0.8, 1.0)
KB, STRENGTH = (6.0, 12.0, 20.0, 30.0), (0.5, 0.75, 1.0, 1.25)
ALIGNMENT_STATS = ('targets', 'receptions', 'receiving_yards')


def load(trace_path):
    d, t = load_joined(trace_path)
    v = pd.read_parquet(trace_path + '.defense.parquet')
    t = t.copy()
    t['opp'] = d['opp'].astype(str).to_numpy()[t['_row'].to_numpy()]
    pff_year = t['year'] >= 2024
    from_alignment = t['alignment_residual_available'].fillna(0).astype(bool)
    receiving = t['stat'].isin(ALIGNMENT_STATS + ('receiving_tds',))
    movable = ~((t['pos'].isin(['WR', 'TE']) & t['stat'].isin(ALIGNMENT_STATS) & from_alignment)
                | (t['pos'].eq('TE') & receiving & pff_year))
    t['movable'] = movable & (t['matchup_multiplier'].fillna(0) > 0)
    return d, t, v


def ratings(v, kc, decay, kp, floor, kb):
    c = v[f'cur_{kc:g}_{decay:g}']
    p = v[f'pri_{kp:g}_{floor:g}'].fillna(1.0)
    a = v['n'] / (v['n'] + kb)
    return pd.Series(np.where(c.notna(), a * c.fillna(1.0) + (1 - a) * p, p), index=v.index)


def final_under(t, v, key, cfg):
    """Each trace row's final stat when the defense profile uses ``cfg``."""
    r = ratings(v, cfg['kc'], cfg['decay'], cfg['kp'], cfg['floor'], cfg['kb'])
    delta = (r - v['shipped']).to_numpy()
    lookup = pd.Series(delta, index=key)
    rk = pd.MultiIndex.from_arrays([t['year'], t['week'], t['pos'], t['opp'], t['stat']])
    dlt = np.nan_to_num(lookup.reindex(rk).to_numpy(), nan=0.0)
    m = t['matchup_multiplier'].to_numpy(float)
    m2 = np.clip(m + dlt, *wp.MATCHUP_CLIP)
    m2 = np.clip(1.0 + cfg['strength'] * (m2 - 1.0), 0.0, None)
    ratio = np.where(t['movable'].to_numpy(), m2 / np.where(m > 0, m, 1.0), 1.0)
    return t['final_projection'].to_numpy(float) + t['pre_vacancy_projection'].to_numpy(float) * (ratio - 1.0)


def score(d, t, v, key, base, cfg, positions=('QB', 'RB', 'WR', 'TE')):
    var = board(d, t, final_under(t, v, key, cfg))
    r = drmse(d, base, var)
    out = {'START-ALL': r['d'], 'ci': f"[{r['lo']:+.4f},{r['hi']:+.4f}]", 'yrs better': f"{r['yrs_better']}/{r['yrs']}"}
    for pos in positions:
        out[pos] = drmse(d, base, var, pos)['d']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', default='.sweeps/defense_capture.parquet')
    ap.add_argument('--quick', action='store_true', help='skip the leave-one-season-out grid')
    a = ap.parse_args()
    pd.set_option('display.width', 230)
    d, t, v = load(a.traces)
    key = pd.MultiIndex.from_arrays([v['year'], v['week'], v['pos'], v['defense'].astype(str), v['stat']])
    v = v[~key.duplicated(keep=False)].reset_index(drop=True)
    key = pd.MultiIndex.from_arrays([v['year'], v['week'], v['pos'], v['defense'].astype(str), v['stat']])
    replay = ratings(v, 12.0, 0.85, 12.0, 0.8, 12.0)
    print(f"trace rows {len(t):,} ({t['movable'].mean():.0%} movable), defense ratings {len(v):,}; "
          f"replay of the shipped settings vs recorded: max gap {np.nanmax(np.abs(replay - v['shipped'])):.2e}")
    base = board(d, t, t['final_projection'].to_numpy(float))

    print('\n=== one knob at a time (others shipped) ===')
    rows = []
    for knob, grid in (('kc', KC), ('decay', DECAY), ('kp', KP), ('floor', FLOOR), ('kb', KB), ('strength', STRENGTH)):
        for val in grid:
            if val == SHIPPED[knob]:
                continue
            cfg = dict(SHIPPED, **{knob: val})
            rows.append({'knob': knob, 'value': val, **score(d, t, v, key, base, cfg)})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print('\n=== C2: the three jobs of DEFENSE_PRIOR_GAMES apart (decay 0.85, floor 0.8, strength 1) ===')
    rows = []
    for kc, kp, kb in itertools.product(KC, KP, KB):
        cfg = dict(SHIPPED, kc=kc, kp=kp, kb=kb)
        rows.append({'kc': kc, 'kp': kp, 'kb': kb, **score(d, t, v, key, base, cfg, positions=())})
    res = pd.DataFrame(rows).sort_values('START-ALL')
    print(res.head(12).round(4).to_string(index=False))
    res.to_csv('.sweeps/defense_variants_c2.csv', index=False)

    if a.quick:
        return
    print('\n=== leave one season out: the full grid chosen on four seasons, scored on the fifth ===')
    grid = [dict(kc=kc, decay=dc, kp=kp, floor=fl, kb=kb, strength=s)
            for kc, dc, kp, fl, kb, s in itertools.product(KC, DECAY, KP, FLOOR, KB, (0.75, 1.0))]
    finals = {i: final_under(t, v, key, cfg) for i, cfg in enumerate(grid)}
    boards = {i: board(d, t, fn) for i, fn in finals.items()}
    act = d['actual'].to_numpy(float)
    from scripts.eval_stat_k_sweep import paired_pool
    years = d['year'].to_numpy()
    out = []
    for y in sorted(np.unique(years)):
        best_i, best_r = None, None
        for i, var in boards.items():
            mask = paired_pool(d, base, var) & (years != y)
            r_ = float(np.sqrt(((var[mask] - act[mask]) ** 2).mean()) - np.sqrt(((base[mask] - act[mask]) ** 2).mean()))
            if best_r is None or r_ < best_r:
                best_i, best_r = i, r_
        var = boards[best_i]
        mask = paired_pool(d, base, var) & (years == y)
        held = float(np.sqrt(((var[mask] - act[mask]) ** 2).mean()) - np.sqrt(((base[mask] - act[mask]) ** 2).mean()))
        out.append({'held-out': int(y), 'dRMSE START-ALL': held, **{k: grid[best_i][k] for k in SHIPPED}})
    loyo = pd.DataFrame(out)
    print(loyo.round(4).to_string(index=False))
    print(f"mean {loyo['dRMSE START-ALL'].mean():+.4f}, better in {(loyo['dRMSE START-ALL'] < 0).sum()} of {len(loyo)}")


if __name__ == '__main__':
    main()
