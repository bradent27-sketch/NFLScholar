"""
Calibration audit C1 (2026-10-20): the PFF alignment WR/TE matchup (targets, receptions, receiving yards), re-scored offline
from scripts/capture_stat_blend_traces.py --alignment-variants (no builds). Only seasons with PFF weekly exports (2024-25).

Recorded per player call: the shipped multiplier and the unclipped one under every variant of
    this season's alignment-defense shrinkage  {1, 2, 4, 8, 12, 20}  (shipped 4 games)
    the prior-season blend weight w0           {0, 0.2, 0.4}         (shipped 0.2)
Offline, on top of those: the final clip {(0.75, 1.30) shipped, (0.82, 1.22), (0.88, 1.14)} and a strength s {0.5, 0.75, 1}
(m = 1 + s * (clip(raw) - 1)). A trace row is moved only when its matchup multiplier IS the alignment one (matched on year,
week, position, opponent, stat and the multiplier itself); a TE whose scheme multiplier replaced it is left alone.

    python scripts/eval_alignment_variants.py [--traces .sweeps/alignment_capture.parquet]
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

from scripts.eval_blend_modifiers import board  # noqa: E402
from scripts.eval_stat_k_sweep import drmse, load_joined, paired_pool  # noqa: E402

SHRINK, W0 = (1.0, 2.0, 4.0, 8.0, 12.0, 20.0), (0.0, 0.2, 0.4)
CLIPS = ((0.75, 1.30), (0.82, 1.22), (0.88, 1.14))
STRENGTH = (0.5, 0.75, 1.0)
SHIPPED = (4.0, 0.2, (0.75, 1.30), 1.0)
STAT_NAME = {'targets': 'targets', 'receptions': 'receptions', 'yards': 'receiving_yards'}


def load(trace_path):
    d, t = load_joined(trace_path)
    t = t.copy()
    t['opp'] = d['opp'].astype(str).to_numpy()[t['_row'].to_numpy()]
    al = pd.read_parquet(trace_path + '.alignment.parquet')
    al['stat'] = al['stat'].map(STAT_NAME)
    t['_i'] = np.arange(len(t))
    keys = ['year', 'week', 'player', 'pos', 'team', 'stat']
    al = al.drop_duplicates(keys, keep=False)
    j = t[['_i', 'matchup_multiplier'] + keys].merge(al, on=keys, how='inner')
    # move a row only where the board really used the alignment multiplier (a TE's scheme multiplier can replace it)
    j = j[(j['matchup_multiplier'] - j['shipped']).abs() < 6e-4].drop(columns='matchup_multiplier')
    return d, t, j


def final_under(t, j, shrink, w0, clip, strength):
    raw = j[f'raw_{shrink:g}_{w0:g}'].to_numpy(float)
    m_new = 1.0 + strength * (np.clip(raw, *clip) - 1.0)
    m_old = t['matchup_multiplier'].to_numpy(float)[j['_i'].to_numpy()]
    ratio = np.where(np.isfinite(m_new) & (m_old > 0), m_new / np.where(m_old > 0, m_old, 1.0), 1.0)
    fn = t['final_projection'].to_numpy(float).copy()
    idx = j['_i'].to_numpy()
    fn[idx] = fn[idx] + t['pre_vacancy_projection'].to_numpy(float)[idx] * (ratio - 1.0)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', default='.sweeps/alignment_capture.parquet')
    a = ap.parse_args()
    pd.set_option('display.width', 230)
    d, t, j = load(a.traces)
    elig = t['pos'].isin(['WR', 'TE']) & t['stat'].isin(STAT_NAME.values()) & t['alignment_residual_available'].fillna(0).astype(bool)
    print(f"WR/TE receiving rows whose matchup is the alignment multiplier: {int(elig.sum()):,}; matched to a variant record: "
          f"{len(j):,} ({len(j) / max(int(elig.sum()), 1):.0%})")
    base = board(d, t, t['final_projection'].to_numpy(float))
    rows = []
    for shrink, w0, clip, s in itertools.product(SHRINK, W0, CLIPS, STRENGTH):
        if (shrink, w0, clip, s) == SHIPPED:
            continue
        var = board(d, t, final_under(t, j, shrink, w0, clip, s))
        r = drmse(d, base, var)
        rows.append({'shrink': shrink, 'w0': w0, 'clip': f'{clip[0]}-{clip[1]}', 'strength': s, 'START-ALL': r['d'],
                     'ci': f"[{r['lo']:+.4f},{r['hi']:+.4f}]", 'yrs better': f"{r['yrs_better']}/{r['yrs']}",
                     'WR': drmse(d, base, var, 'WR')['d'], 'TE': drmse(d, base, var, 'TE')['d'], '_var': var})
    res = pd.DataFrame(rows)
    show = res.drop(columns='_var')
    print('\n=== one knob at a time (others shipped) ===')
    one = show[((show.w0 == 0.2) & (show['clip'] == '0.75-1.3') & (show.strength == 1.0))
               | ((show.shrink == 4.0) & (show['clip'] == '0.75-1.3') & (show.strength == 1.0))
               | ((show.shrink == 4.0) & (show.w0 == 0.2) & (show.strength == 1.0))
               | ((show.shrink == 4.0) & (show.w0 == 0.2) & (show['clip'] == '0.75-1.3'))]
    print(one.round(4).to_string(index=False))
    print('\n=== best 12 of the full grid ===')
    print(show.sort_values('START-ALL').head(12).round(4).to_string(index=False))
    show.to_csv('.sweeps/alignment_variants_grid.csv', index=False)
    print('\n=== choose on one PFF season, score on the other ===')
    act = d['actual'].to_numpy(float)
    years = d['year'].to_numpy()
    for y in (2024, 2025):
        fit = []
        for _, r in res.iterrows():
            mask = paired_pool(d, base, r['_var']) & (years != y) & (years >= 2024)
            fit.append(np.sqrt(((r['_var'][mask] - act[mask]) ** 2).mean()) - np.sqrt(((base[mask] - act[mask]) ** 2).mean()))
        best = res.iloc[int(np.argmin(fit))]
        mask = paired_pool(d, base, best['_var']) & (years == y)
        held = np.sqrt(((best['_var'][mask] - act[mask]) ** 2).mean()) - np.sqrt(((base[mask] - act[mask]) ** 2).mean())
        print(f"  held-out {y}: chosen shrink {best['shrink']:g} w0 {best['w0']:g} clip {best['clip']} strength {best['strength']:g}"
              f" -> START-ALL {held:+.4f}")


if __name__ == '__main__':
    main()
