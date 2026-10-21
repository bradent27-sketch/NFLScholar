"""
Calibration audit C1b (2026-10-21): the PFF man/zone scheme TE matchup (targets, receptions, receiving yards), re-scored offline
from scripts/capture_stat_blend_traces.py --scheme-variants (no builds). Only seasons with PFF weekly exports (2024-25).

Recorded per TE call: the shipped multiplier and the unclipped one (his man-route share x the defense's man allowed-ratio plus
the rest x its zone ratio) under every variant of
    this season's scheme-defense shrinkage  {1, 2, 4, 8, 12, 20}  (shipped 4 games)
    the prior-season blend weight w0        {0, 0.2, 0.4, 0.6}    (shipped 0.4)
Offline, on top of those: the clip {(0.90, 1.10) shipped, (0.85, 1.15), (0.80, 1.22), (0.95, 1.05)} and a strength s
{0.5, 1, 1.5} (m = 1 + s * (clip(raw) - 1)). A row moves only where the board used the scheme multiplier.

    python scripts/eval_scheme_variants.py [--traces .sweeps/scheme_capture.parquet]
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

SHRINK, W0 = (1.0, 2.0, 4.0, 8.0, 12.0, 20.0), (0.0, 0.2, 0.4, 0.6)
CLIPS = ((0.90, 1.10), (0.85, 1.15), (0.80, 1.22), (0.95, 1.05))
STRENGTH = (0.5, 1.0, 1.5)
SHIPPED = (4.0, 0.4, (0.90, 1.10), 1.0)
STAT_NAME = {'targets': 'targets', 'receptions': 'receptions', 'yards': 'receiving_yards'}
KEYS = ['year', 'week', 'player', 'pos', 'team', 'stat']


def load(trace_path):
    d, t = load_joined(trace_path)
    t = t.copy()
    t['_i'] = np.arange(len(t))
    sc = pd.read_parquet(trace_path + '.scheme.parquet')
    sc['stat'] = sc['stat'].map(STAT_NAME)
    sc = sc.drop_duplicates(KEYS, keep=False)
    j = t[['_i', 'matchup_multiplier'] + KEYS].merge(sc, on=KEYS, how='inner')
    j = j[(j['matchup_multiplier'] - j['shipped']).abs() < 6e-4].drop(columns='matchup_multiplier').reset_index(drop=True)
    return d, t, j


def final_under(t, j, shrink, w0, clip, strength):
    raw = j[f'raw_{shrink:g}_{w0:g}'].to_numpy(float)
    m_new = 1.0 + strength * (np.clip(raw, *clip) - 1.0)
    m_old = j['shipped'].to_numpy(float)
    ratio = np.where(np.isfinite(m_new) & (m_old > 0), m_new / np.where(m_old > 0, m_old, 1.0), 1.0)
    fn = t['final_projection'].to_numpy(float).copy()
    idx = j['_i'].to_numpy()
    fn[idx] = fn[idx] + t['pre_vacancy_projection'].to_numpy(float)[idx] * (ratio - 1.0)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', default='.sweeps/scheme_capture.parquet')
    a = ap.parse_args()
    pd.set_option('display.width', 230)
    d, t, j = load(a.traces)
    print(f"TE receiving rows on the scheme multiplier: {len(j):,}; replay of the shipped setting: max gap "
          f"{np.nanmax(np.abs(np.clip(j['raw_4_0.4'], *SHIPPED[2]) - j['shipped'])):.2e}")
    base = board(d, t, t['final_projection'].to_numpy(float))
    rows = []
    for shrink, w0, clip, s in itertools.product(SHRINK, W0, CLIPS, STRENGTH):
        if (shrink, w0, clip, s) == SHIPPED:
            continue
        var = board(d, t, final_under(t, j, shrink, w0, clip, s))
        r, rt = drmse(d, base, var), drmse(d, base, var, 'TE')
        rows.append({'shrink': shrink, 'w0': w0, 'clip': f'{clip[0]}-{clip[1]}', 'strength': s, 'START-TE': rt['d'],
                     'ci': f"[{rt['lo']:+.4f},{rt['hi']:+.4f}]", 'yrs better': f"{rt['yrs_better']}/{rt['yrs']}",
                     'START-ALL': r['d'], '_var': var})
    res = pd.DataFrame(rows)
    show = res.drop(columns='_var')
    print('\n=== one knob at a time (others shipped) ===')
    one = show[((show.w0 == 0.4) & (show['clip'] == '0.9-1.1') & (show.strength == 1.0))
               | ((show.shrink == 4.0) & (show['clip'] == '0.9-1.1') & (show.strength == 1.0))
               | ((show.shrink == 4.0) & (show.w0 == 0.4) & (show.strength == 1.0))
               | ((show.shrink == 4.0) & (show.w0 == 0.4) & (show['clip'] == '0.9-1.1'))]
    print(one.round(4).to_string(index=False))
    print('\n=== best 12 of the full grid (START-TE) ===')
    print(show.sort_values('START-TE').head(12).round(4).to_string(index=False))
    show.to_csv('.sweeps/scheme_variants_grid.csv', index=False)
    print('\n=== choose on one PFF season, score on the other (START-TE) ===')
    act = d['actual'].to_numpy(float)
    years = d['year'].to_numpy()
    te = (d['pos'] == 'TE').to_numpy()
    for y in (2024, 2025):
        fit = []
        for _, r in res.iterrows():
            mask = paired_pool(d, base, r['_var']) & (years != y) & (years >= 2024) & te
            fit.append(np.sqrt(((r['_var'][mask] - act[mask]) ** 2).mean()) - np.sqrt(((base[mask] - act[mask]) ** 2).mean()))
        best = res.iloc[int(np.argmin(fit))]
        mask = paired_pool(d, base, best['_var']) & (years == y) & te
        held = np.sqrt(((best['_var'][mask] - act[mask]) ** 2).mean()) - np.sqrt(((base[mask] - act[mask]) ** 2).mean())
        print(f"  held-out {y}: chosen shrink {best['shrink']:g} w0 {best['w0']:g} clip {best['clip']} strength {best['strength']:g}"
              f" -> START-TE {held:+.4f}")


if __name__ == '__main__':
    main()
