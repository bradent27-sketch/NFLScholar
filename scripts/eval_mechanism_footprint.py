"""
What each mechanism does to the board, from a harness run's saved variant boards (2026-10-24, `--variant-boards`): for every
variant (the mechanism switched off) against the base, player by player -

  footprint   share of players whose points move by more than 0.05, the mean |change| among them, the largest, by position;
  stats       the mean |change| of each stat it moves, as a share of the stat's mean;
  accuracy    RMSE of each stat (and points) against what happened, base vs variant, on played games (the capture board's
              actuals), by position - where the mechanism's effect lands, which the harness's points scopes do not show.

    python scripts/eval_mechanism_footprint.py --boards .sweeps/harness_boards/mechanisms_2022-2025_wk3-17.parquet
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.eval_calibration_schemes import load_dump  # noqa: E402

ACTUALS = '.sweeps/qb_style_capture.parquet.board.csv'
KEYS = ['year', 'week', 'Player', 'Pos', 'Team']
STATS = ('passing_attempts', 'passing_yards', 'passing_tds', 'rushing_attempts', 'rushing_yards', 'rushing_tds', 'targets',
         'receptions', 'receiving_yards', 'receiving_tds')


def rmse(e):
    e = np.asarray(e, dtype=float)
    return float(np.sqrt(np.mean(e * e))) if len(e) else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boards', default='.sweeps/harness_boards/mechanisms_2022-2025_wk3-17.parquet')
    ap.add_argument('--out', default='.sweeps/eval_mechanism_footprint.txt')
    a = ap.parse_args()
    boards = pd.read_parquet(a.boards)
    base = boards[boards['variant'] == 'base'].drop(columns='variant')
    acts = load_dump(ACTUALS)
    acts = acts[acts['played'].astype(bool)].rename(columns={'player': 'Player', 'team': 'Team', 'pos': 'Pos'})
    lines = []
    for label in [v for v in boards['variant'].unique() if v != 'base']:
        var = boards[boards['variant'] == label].drop(columns='variant')
        m = base.merge(var, on=KEYS, suffixes=('', '_v'))
        d = m['Model Proj Pts_v'] - m['Model Proj Pts']
        moved = d.abs() > 0.05
        lines.append(f'=== {label} (switched off) - {len(m)} player-weeks')
        lines.append(f'  points: {moved.mean():.1%} of players move by more than 0.05; among them mean |change| '
                     f'{d[moved].abs().mean() if moved.any() else 0:.2f}, largest {d.abs().max():.2f}; mean change {d.mean():+.3f}')
        for pos, g in m.groupby('Pos'):
            dd = g['Model Proj Pts_v'] - g['Model Proj Pts']
            mv = dd.abs() > 0.05
            lines.append(f'    {pos}: {mv.mean():.1%} moved, mean |change| {dd[mv].abs().mean() if mv.any() else 0:.2f}, '
                         f'mean change {dd.mean():+.3f}')
        stat_moves = []
        for stat in STATS:
            if stat not in m.columns or stat + '_v' not in m.columns:
                continue
            ds = (m[stat + '_v'] - m[stat]).abs()
            if ds.max() > 1e-6:
                stat_moves.append(f'{stat} {ds.mean() / max(m[stat].abs().mean(), 1e-9):.2%}')
        lines.append('  stats moved (mean |change| / mean): ' + (', '.join(stat_moves) if stat_moves else 'none'))
        if not moved.any():
            lines.append('')
            continue
        j = m.merge(acts, on=['year', 'week', 'Player', 'Team'], how='inner', suffixes=('', '_a'))
        acc = []
        for pos, g in j.groupby('Pos'):
            for stat in ('points',) + STATS:
                col, act = ('Model Proj Pts', 'actual') if stat == 'points' else (stat, 'a_' + stat)
                if col + '_v' not in g.columns or act not in g.columns:
                    continue
                touched = (g[col + '_v'] - g[col]).abs() > 1e-6
                if touched.sum() < 200:
                    continue
                y = pd.to_numeric(g[act], errors='coerce')
                ok = touched & y.notna()
                b, v = rmse(g.loc[ok, col] - y[ok]), rmse(g.loc[ok, col + '_v'] - y[ok])
                acc.append(f'    {pos} {stat:16} n={int(ok.sum()):5d}  RMSE {b:.3f} -> {v:.3f} ({v - b:+.4f}, {100 * (v / b - 1):+.2f}%)')
        lines.append('  accuracy on the rows it moves, played games (positive = worse without it):')
        lines += acc or ['    (fewer than 200 moved rows at every position/stat)']
        lines.append('')
    text = '\n'.join(lines)
    print(text)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
