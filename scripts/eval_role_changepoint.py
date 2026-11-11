"""
Role-state change-point study (2026-11-11, audit D5, docs/audit_leads_2026-11-10.md). Offline, read only, nothing ships.

Question: does a change point detected in a player's snap-share series this season predict his next game's snap share better
than the model's own Expected Snap Share (the shipped blend), held out? If not, D5 stops here.

Data: the `base` variant of .sweeps/harness_boards/capacity_screen_2022-2025_wk3-17.parquet (the default model's board for
every 2022-2025 week 3-17, built 2026-11-03) against the box-score weekly_snap_pct. Rows: RB / WR / TE who played (snap share
> 0) in the projected week and had 2+ games for the same team earlier that season. Shares are in points of snap share (0-100).

Predictors, each from the player's earlier games this season for the same team only:
  model      the board's Expected Snap Share
  season     the season-to-date mean
  last3      the mean of the last three games
  cp         a single change point: the split of the series (each side min_seg+ games) with the lowest squared error; when
             it cuts the error by more than `penalty` x the series variance x n, predict the mean after the split, else the
             season mean
  model+cp   the model's share, replaced by the post-split mean only where a change point is detected (what "replace the
             role-change rules" would mean)
penalty and min_seg are tuned on 2022-2023 by MAE and scored once on 2024-2025.

    python scripts/eval_role_changepoint.py [--board PATH]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BOARD = '.sweeps/harness_boards/capacity_screen_2022-2025_wk3-17.parquet'
POSITIONS = ('RB', 'WR', 'TE')
TRAIN, TEST = (2022, 2023), (2024, 2025)
PENALTIES = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
MIN_SEGS = (1, 2, 3)


def change_point(x, penalty, min_seg):
    """(detected, post-split mean) for the series x."""
    n = len(x)
    full = float(((x - x.mean()) ** 2).sum())
    best, at = full, None
    for k in range(min_seg, n - min_seg + 1):
        a, b = x[:k], x[k:]
        sse = float(((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum())
        if sse < best:
            best, at = sse, k
    var = x.var() if n > 1 else 0.0
    if at is None or full - best <= penalty * var * n or full - best <= 1e-9:
        return False, float(x.mean())
    return True, float(x[at:].mean())


def load_actuals(years):
    from data.prediction_ledger import clean_name_exact
    from data.transforms import load_and_merge_data
    out = []
    for year in years:
        stats, team_col, name_col, _ = load_and_merge_data(year, 'Full PPR')
        s = stats[pd.to_numeric(stats['week'], errors='coerce').between(1, 17)]
        out.append(pd.DataFrame({'year': year, 'week': s['week'].astype(int), 'team': s[team_col].astype(str),
                                 'key': clean_name_exact(s[name_col]),
                                 'share': pd.to_numeric(s['weekly_snap_pct'], errors='coerce')}))
    return pd.concat(out, ignore_index=True).dropna(subset=['share']).drop_duplicates(['year', 'week', 'team', 'key'])


def build_rows(board, actuals):
    from data.prediction_ledger import clean_name_exact
    b = board[board['variant'].eq('base') & board['Pos'].isin(POSITIONS)].copy()
    b['key'] = clean_name_exact(b['Player'])
    b = b.rename(columns={'Team': 'team'})[['year', 'week', 'team', 'key', 'Pos', 'Expected Snap Share']]
    hist = {k: g.sort_values('week') for k, g in actuals[actuals['share'] > 0].groupby(['year', 'team', 'key'])}
    rows = []
    for r in b.itertuples(index=False):
        g = hist.get((r.year, r.team, r.key))
        if g is None:
            continue
        now = g[g['week'].eq(r.week)]
        before = g[g['week'] < r.week]['share'].to_numpy(float)
        if now.empty or len(before) < 2 or pd.isna(r[5]):
            continue
        rows.append({'year': r.year, 'week': r.week, 'pos': r.Pos, 'actual': float(now['share'].iloc[0]),
                     'model': float(r[5]) * 100.0, 'season': before.mean(), 'last3': before[-3:].mean(), 'series': before})
    return pd.DataFrame(rows)


def add_cp(rows, penalty, min_seg):
    res = [change_point(x, penalty, min_seg) for x in rows['series']]
    out = rows.assign(detected=[d for d, _ in res], cp=[m for _, m in res])
    out['model+cp'] = np.where(out['detected'], out['cp'], out['model'])
    return out


def mae(df, col):
    return float((df[col] - df['actual']).abs().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--board', default=BOARD)
    a = ap.parse_args()
    board = pd.read_parquet(a.board)
    rows = build_rows(board, load_actuals(sorted(board['year'].unique())))
    train, test = rows[rows['year'].isin(TRAIN)], rows[rows['year'].isin(TEST)]
    print(f'rows: train {len(train)}, test {len(test)}')
    grid = []
    for p in PENALTIES:
        for m in MIN_SEGS:
            t = add_cp(train, p, m)
            grid.append((mae(t, 'model+cp'), mae(t, 'cp'), p, m, float(t['detected'].mean())))
    grid.sort()
    best = grid[0]
    print('train grid (MAE model+cp, MAE cp, penalty, min_seg, detected share), best 5:')
    for g in grid[:5]:
        print('  ' + ', '.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in g))
    p, m = best[2], best[3]
    cols = ['model', 'season', 'last3', 'cp', 'model+cp']
    for name, part in (('train', train), ('test', test)):
        t = add_cp(part, p, m)
        print(f'\n{name} (penalty {p}, min_seg {m}; change detected on {t["detected"].mean():.1%}):')
        print('  MAE all       ' + '  '.join(f'{c} {mae(t, c):.2f}' for c in cols))
        d = t[t['detected']]
        if len(d):
            print(f'  MAE detected  ' + '  '.join(f'{c} {mae(d, c):.2f}' for c in cols) + f'  (n={len(d)})')
        for pos, g in t.groupby('pos'):
            print(f'  {pos:3s} n={len(g):5d}  ' + '  '.join(f'{c} {mae(g, c):.2f}' for c in cols))
        # paired: model+cp vs model, clustered by player-season week blocks is overkill here; per-row paired mean and SE
        diff = (t['model+cp'] - t['actual']).abs() - (t['model'] - t['actual']).abs()
        print(f'  model+cp minus model: {diff.mean():+.3f} pts (SE {diff.std() / np.sqrt(len(diff)):.3f}); '
              f'on detected rows {diff[t["detected"]].mean():+.3f}')


if __name__ == '__main__':
    main()
