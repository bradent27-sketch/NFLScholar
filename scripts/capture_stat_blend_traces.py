"""
Capture, for EVERY projected player and EVERY stat, the in-season blend's inputs from the shipped model's as-of builds, so
the per-stat current-vs-prior weight (STAT_K / STAT_K_BY_POS) can be swept OFFLINE with no further builds.  (Backlog M2,
2026-10-19; the same idea as capture_qb_td_traces.py, which did it for QB passing TDs only.)

For each (year, week) it builds the board exactly as the harness base arm does (DEFAULT_FEATURES plus the historical
injury/reserve replay, apply_injury=False, as-of the target week) and writes one row per (player, stat):

    current_games, current_rate, prior_rate, current_weight, role_confidence, role_change_confidence   the blend's inputs
    blended_rate, pre_vacancy_projection, vacancy_delta, final_projection                                what it produced
    actual (the player's box-score value of the stat that week; NaN when he has no box score)           what happened

    blended = w * current_rate + (1 - w) * prior_rate,  w = games / (games + k_eff)
    -> any other w re-scores the pre-vacancy stat as  pre_vacancy' = pre_vacancy * blended' / blended

The same builds also write a board-level file, `<out>.board.csv`, with exactly the columns of a `calibration_lab.py --mode dump`
file (board points, actual points, `played`, projected and actual stats, the season anchor inputs), so it loads with
`eval_calibration_schemes.load_dump` and always matches the traces row for row (an older dump drifts from the current model).
Both files are rewritten after every week and the run resumes where it stopped.

    python scripts/capture_stat_blend_traces.py --years 2021,2022,2023,2024,2025 --weeks 3-17
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.transforms import load_and_merge_data  # noqa: E402
from data.weekly_projections import (DEFAULT_FEATURES, _ALL_PROJECTION_STATS, _played_weeks_before,  # noqa: E402
                                     build_weekly_projections)
from scripts.calibration_lab import CONTEXT_COLS, EXTRA_ACTUALS  # noqa: E402
from scripts.eval_weekly_model import _actual_points  # noqa: E402

REPLAY = frozenset({'v2_historical_injury_replay', 'v2_historical_reserve_replay'})
TRACE_KEYS = ('build_path', 'current_games', 'current_rate', 'prior_rate', 'raw_prior_rate', 'prior2_weight',
              'current_weight', 'blended_rate', 'role_confidence', 'role_change_confidence', 'matchup_multiplier',
              'pace_multiplier', 'script_multiplier', 'availability_multiplier', 'environment_multiplier',
              'participation_multiplier', 'weather_stat_multiplier', 'pre_vacancy_projection', 'vacancy_delta',
              'pass_capacity_delta', 'top_end_delta', 'final_projection', 'prior_source', 'expected_snap_share')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2021,2022,2023,2024,2025')
    ap.add_argument('--weeks', default='3-17')
    ap.add_argument('--out', default='.sweeps/stat_blend_traces.parquet')
    ap.add_argument('--scoring', default='Full PPR')
    a = ap.parse_args()
    years = [int(y) for y in a.years.split(',')]
    lo, hi = a.weeks.split('-') if '-' in a.weeks else (a.weeks, a.weeks)
    weeks = list(range(int(lo), int(hi) + 1))
    feats = frozenset(DEFAULT_FEATURES | REPLAY)
    scoring_col = 'fantasy_points_ppr' if a.scoring != 'Standard' else 'fantasy_points'
    board_path = a.out + '.board.csv'
    rows, boards, done = [], [], set()
    if os.path.exists(a.out) and os.path.exists(board_path):    # resume
        prev = pd.read_parquet(a.out)
        prev_board = pd.read_csv(board_path, low_memory=False)
        done = set(zip(prev['year'], prev['week'])) & set(zip(prev_board['year'], prev_board['week']))
        rows = [r for r in prev.to_dict('records') if (r['year'], r['week']) in done]
        boards = [prev_board[(prev_board['year'] == y) & (prev_board['week'] == w)] for y, w in sorted(done)]
        print(f'resuming: {len(done)} weeks already captured', flush=True)
    t0 = time.time()
    for year in years:
        stats_df, _team_col, name_col, _ = load_and_merge_data(year, a.scoring)
        stats_df = stats_df.copy()
        stats_df['_wk'] = pd.to_numeric(stats_df['week'], errors='coerce')
        for week in weeks:
            if (year, week) in done:
                continue
            wk = stats_df[stats_df['_wk'] == week]
            if wk.empty:
                continue
            cols = [s for s in _ALL_PROJECTION_STATS if s in wk.columns]
            agg = wk.groupby(name_col, observed=True)[cols].sum()
            a_cols = [c for c in (*_ALL_PROJECTION_STATS, *EXTRA_ACTUALS) if c in wk.columns]
            sa = wk.groupby(name_col, observed=True)[a_cols].sum()
            actual_pts = _actual_points(stats_df, name_col, week, scoring_col)
            build_weekly_projections.clear()
            board, meta = build_weekly_projections(year, week, a.scoring, as_of_week=week, apply_injury=False, features=feats)
            build_weekly_projections.clear()
            if board.empty:
                print(f'{year} w{week}: nothing ({meta.get("reason")})', flush=True)
                continue
            keep = [c for c in CONTEXT_COLS if c in board.columns]
            bd = board[keep].rename(columns=CONTEXT_COLS).copy()
            hist = _played_weeks_before(stats_df, week)
            if not hist.empty and 'fantasy_points' in hist.columns:
                g = pd.to_numeric(hist['fantasy_points'], errors='coerce').groupby(hist[name_col]).agg(['mean', 'count'])
                bd['anchor_mean'] = bd['player'].map(g['mean'])
                bd['anchor_games'] = bd['player'].map(g['count']).fillna(0).astype(int)
            else:
                bd['anchor_mean'], bd['anchor_games'] = np.nan, 0
            bd['actual'] = bd['player'].map(actual_pts)
            bd['played'] = bd['actual'].notna()
            for st in _ALL_PROJECTION_STATS:
                bd[f'p_{st}'] = pd.to_numeric(board[st], errors='coerce').to_numpy() if st in board.columns else np.nan
            for st in a_cols:
                bd[f'a_{st}'] = bd['player'].map(sa[st])
            bd['year'], bd['week'] = year, week
            boards.append(bd)
            n = 0
            for _, r in board.iterrows():
                ex = meta['explanations'].get((r['Player'], r['Pos'], r['Team']))
                if not ex:
                    continue
                played = r['Player'] in agg.index
                for stat, tr in ex['stats'].items():
                    row = {'year': year, 'week': week, 'player': r['Player'], 'pos': r['Pos'], 'team': r['Team'],
                           'stat': stat, 'played': played,
                           'actual': (agg.at[r['Player'], stat] if played and stat in agg.columns else None)}
                    for k in TRACE_KEYS:
                        row[k] = tr.get(k)
                    rows.append(row)
                    n += 1
            out = pd.DataFrame(rows)
            for c in out.columns:                               # mixed object columns break parquet; keep the numerics numeric
                if c not in ('player', 'pos', 'team', 'stat', 'build_path', 'prior_source', 'played'):
                    out[c] = pd.to_numeric(out[c], errors='coerce')
            out['prior_source'] = out['prior_source'].astype(str)
            out.to_parquet(a.out, index=False)
            pd.concat(boards, ignore_index=True).to_csv(board_path, index=False)
            print(f'{year} w{week}: {n} stat rows, {len(rows)} total, {time.time() - t0:.0f}s', flush=True)
    print('done', len(rows), flush=True)


if __name__ == '__main__':
    main()
