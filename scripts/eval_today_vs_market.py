"""
Today's model on the 2026 live weeks against the market's numbers saved then (2026-10-24): rebuild each played week with the current
DEFAULT_FEATURES as of that week (the historical injury and reserve replay, as the harness does - not the FantasyPros report the live
board had), and score it beside the market total and FantasyPros saved on that week's pre-kickoff ledger rows, on the same players
(the ledger scorecard's paired start-worthy pool). The live record (scripts/eval_market_record.py) scores the model each week actually
ran; this scores the model as it is now.

    python scripts/eval_today_vs_market.py [--year 2026] [--weeks 3,4,5,6,7]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import ledger_scorecard as sc  # noqa: E402
from data.loaders import load_schedule  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402
from data.weekly_projections import DEFAULT_FEATURES, build_weekly_projections  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=2026)
    ap.add_argument('--weeks', default='3,4,5,6,7')
    a = ap.parse_args()
    stats_df, _t, name_col, _ = load_and_merge_data(a.year, 'Full PPR')
    frame, _info = sc.load_scored_ledger(a.year, stats_df, name_col, load_schedule(a.year))
    frame = frame[frame['pre_kickoff']].copy()
    feats = frozenset(DEFAULT_FEATURES | {'v2_historical_injury_replay', 'v2_historical_reserve_replay'})
    today = []
    for week in [int(w) for w in a.weeks.split(',')]:
        build_weekly_projections.clear()
        board, _meta = build_weekly_projections(a.year, week, 'Full PPR', as_of_week=week, apply_injury=False, features=feats)
        if board is None or board.empty:
            continue
        today.append(pd.DataFrame({'_key': clean_name_exact(board['Player']), 'week': week,
                                   'today_pts': pd.to_numeric(board['Model Proj Pts'], errors='coerce')}))
        print(f'built week {week}: {len(board)} players', flush=True)
    today = pd.concat(today, ignore_index=True).drop_duplicates(['_key', 'week'])
    m = frame.merge(today, on=['_key', 'week'], how='inner')
    m['model_pts_live'] = m['model_pts']
    lines = [f'{a.year} weeks {a.weeks}: today\'s model rebuilt as of each week vs the market / FantasyPros saved then', '']
    for bench, col in (('Market', 'market_pts'), ('FantasyPros', 'fp_pts')):
        for startable in (True, False):
            rows = []
            for week, wk in m.groupby('week'):
                sub = wk[wk['today_pts'].notna() & wk[col].notna()]
                if startable and not sub.empty:
                    keep = pd.Series(False, index=sub.index)
                    for p, g in sub.groupby('Pos'):
                        n = sc.STARTABLE_N.get(p, 30)
                        keep.loc[g['today_pts'].nlargest(n).index] = True
                        keep.loc[g[col].nlargest(n).index] = True
                    sub = sub[keep]
                rows.append(sub)
            allp = pd.concat(rows)
            e_t, e_l, e_b = allp['today_pts'] - allp['actual_pts'], allp['model_pts_live'] - allp['actual_pts'], allp[col] - allp['actual_pts']
            rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))  # noqa: E731
            weeks_better = sum(rmse(g['today_pts'] - g['actual_pts']) < rmse(g[col] - g['actual_pts']) for _w, g in allp.groupby('week'))
            lines.append(f"vs {bench:11} {'start-worthy' if startable else 'all players':12} n={len(allp):5d}  RMSE today {rmse(e_t):.3f}  "
                         f"live model {rmse(e_l):.3f}  {bench} {rmse(e_b):.3f}  bias today {e_t.mean():+.2f} / {bench} {e_b.mean():+.2f}  "
                         f"weeks today better {weeks_better} of {allp['week'].nunique()}")
    text = '\n'.join(lines)
    print(text)
    with open('.sweeps/eval_today_vs_market.txt', 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
