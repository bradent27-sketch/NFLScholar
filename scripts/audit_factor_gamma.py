"""
Multiplier strength audit (2026-10-23, adapted from CFB Scholar's scripts/week_review/factor_gamma.py): for each traced step of
the stat line, the strength it SHOULD have, by counterfactual.

    multiplier f:      stat' = stat x f^(g - 1)        (g = 1 is the board, g = 0 removes the step)
    additive delta d:  stat' = stat + (g - 1) x d       (vacancy, pass capacity)

scored as RMSE (mean-consistent, like the harness) on the rows where the step is active, two ways: every game the model expected
the player to play (no box-score line = 0) and box-score games only. Reported per (position, stat, step): rows, size of the step,
best g on every-game rows with its RMSE change, the change at g = 0 (the step switched off), the best g in each season and how many
seasons agree. An offline screen (downstream steps are not re-run): it ranks where a harness run is worth it; a step whose best g
is 0 or whose removal costs nothing is a simplification candidate, one far from 1 is mis-sized.

    python scripts/audit_factor_gamma.py [--traces .sweeps/qb_style_capture.parquet] [--out .sweeps/audit_factor_gamma.txt]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.eval_calibration_schemes import load_dump  # noqa: E402

KEYS = ['year', 'week', 'player', 'pos', 'team']
MULTIPLIERS = ('matchup_multiplier', 'pace_multiplier', 'script_multiplier', 'environment_multiplier',
               'participation_multiplier', 'weather_stat_multiplier')
DELTAS = ('vacancy_delta', 'pass_capacity_delta')
GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
MIN_ROWS = 1500


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else float('nan')


def load(trace_path):
    d = load_dump(trace_path + '.board.csv')
    d = d[d['week'].between(3, 17)].drop_duplicates(KEYS, keep=False)
    t = pd.read_parquet(trace_path)
    t = t[t['week'].between(3, 17)].drop(columns=['played', 'actual'], errors='ignore')
    t = t.drop_duplicates(KEYS + ['stat'], keep=False)
    t = t.merge(d[KEYS + ['avail', 'played']], on=KEYS, how='inner')
    actual = []
    acts = d.set_index(KEYS)
    for stat, g in t.groupby('stat'):
        col = 'a_' + stat
        if col in d.columns:
            a = acts[col].reindex(pd.MultiIndex.from_frame(g[KEYS])).to_numpy(dtype=float)
            actual.append(pd.Series(a, index=g.index))
    t['a'] = pd.concat(actual).reindex(t.index)
    t['a0'] = np.where(t['played'].astype(bool), t['a'].fillna(0.0), 0.0)
    t['expected'] = t['avail'] > 0.5
    return t


def audit(t):
    rows = []
    for (pos, stat), g in t.groupby(['pos', 'stat']):
        proj = g['final_projection'].to_numpy(dtype=float)
        years = g['year'].to_numpy()
        for step in MULTIPLIERS + DELTAS:
            if step not in g.columns:
                continue
            v = pd.to_numeric(g[step], errors='coerce').to_numpy(dtype=float)
            if step in DELTAS:
                active = np.isfinite(v) & (np.abs(v) > 1e-6)
                size = float(np.mean(np.abs(v[active]))) if active.any() else 0.0
            else:
                active = np.isfinite(v) & (v > 0) & (np.abs(v - 1.0) > 1e-6)
                size = float(np.std(np.log(v[active]))) if active.any() else 0.0
            for scope, mask, target in (('every game', g['expected'].to_numpy(), g['a0'].to_numpy(dtype=float)),
                                        ('box score', g['played'].astype(bool).to_numpy() & g['a'].notna().to_numpy(),
                                         g['a'].to_numpy(dtype=float))):
                m = active & mask & np.isfinite(proj)
                if m.sum() < MIN_ROWS:
                    continue

                def pred(gamma, k=m):
                    if step in DELTAS:
                        return np.clip(proj[k] + (gamma - 1.0) * v[k], 0.0, None)
                    return proj[k] * v[k] ** (gamma - 1.0)
                err = {gm: rmse(pred(gm) - target[m]) for gm in GRID}
                best = min(err, key=err.get)
                per = []
                for y in np.unique(years[m]):
                    k = m & (years == y)
                    e_y = {gm: rmse(pred(gm, k) - target[k]) for gm in GRID}
                    per.append(min(e_y, key=e_y.get))
                rows.append({'pos': pos, 'stat': stat, 'step': step.replace('_multiplier', '').replace('_delta', ' (delta)'),
                             'scope': scope, 'rows': int(m.sum()), 'size': size, 'best g': best,
                             'gain % at best': 100 * (err[best] / err[1.0] - 1), 'off (g=0) %': 100 * (err[0.0] / err[1.0] - 1),
                             'season best g': per, 'seasons agree': int(np.sum(np.array(per) == best))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', default='.sweeps/qb_style_capture.parquet')
    ap.add_argument('--out', default='.sweeps/audit_factor_gamma.txt')
    a = ap.parse_args()
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 500)
    t = load(a.traces)
    out = audit(t)
    out.to_csv(a.out.replace('.txt', '.csv'), index=False)
    ev = out[out['scope'] == 'every game'].sort_values(['step', 'pos', 'stat'])
    text = ['Every-game rows (no box-score line = 0), RMSE; g = 1 is the board, g = 0 the step switched off:',
            ev.round(3).to_string(index=False), '',
            'Box-score rows only (the harness scopes), same grid:',
            out[out['scope'] == 'box score'].sort_values(['step', 'pos', 'stat']).round(3).to_string(index=False)]
    print('\n'.join(text))
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(text) + '\n')


if __name__ == '__main__':
    main()
