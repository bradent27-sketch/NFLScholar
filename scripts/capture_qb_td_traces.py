"""
Capture every QB's passing-TD blend inputs from the shipped model's as-of builds, so the in-season blend (K) and
its structure can be swept OFFLINE and exactly, with no further builds.  (2026-10-06: v2_qb_passing_td_k, K=15, came
back START-QB -0.010 with CI spanning 0; each harness arm costs ~1 hour, so a K sweep by harness is not affordable.)

For every (year, week) it builds the board exactly as the harness's base arm does (DEFAULT_FEATURES plus the
historical injury/reserve replay, apply_injury=False, as-of the target week) and records, for each projected QB:

    current_games, current_rate, prior_rate, current_weight, role_confidence   the blend's inputs
    blended_rate, matchup/pace/environment/script multipliers, final TD        what it produced
    raw points, calibrated points, projected attempts/yards/TDs                what the board showed
    actual points, TDs, attempts, yards                                        what happened

Because  blended = w*current + (1-w)*prior  and  final TD = blended * (product of the downstream multipliers), any
other w (another K, a different prior) re-scores exactly:  final' = final * blended' / blended.

The parquet is rewritten after every week, so a partial run is usable.

    python scripts/capture_qb_td_traces.py --years 2021,2022,2023,2024,2025 --weeks 3-17
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from data.transforms import load_and_merge_data  # noqa: E402
from data.weekly_projections import DEFAULT_FEATURES, build_weekly_projections  # noqa: E402

REPLAY = frozenset({'v2_historical_injury_replay', 'v2_historical_reserve_replay'})
TRACE_KEYS = ('current_games', 'current_rate', 'prior_rate', 'raw_prior_rate', 'prior_source', 'current_weight',
              'role_confidence', 'blended_rate', 'matchup_multiplier', 'pace_multiplier', 'environment_multiplier',
              'script_multiplier', 'availability_multiplier', 'participation_multiplier', 'script_neutral_multiplier',
              'weather_stat_multiplier', 'pre_vacancy_projection', 'final_projection', 'pass_capacity_delta',
              'vacancy_delta', 'qb1_workload_source')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2021,2022,2023,2024,2025')
    ap.add_argument('--weeks', default='3-17')
    ap.add_argument('--out', default='.sweeps/qb_td_traces.parquet')
    ap.add_argument('--scoring', default='Full PPR')
    a = ap.parse_args()
    years = [int(y) for y in a.years.split(',')]
    lo, hi = a.weeks.split('-') if '-' in a.weeks else (a.weeks, a.weeks)
    weeks = list(range(int(lo), int(hi) + 1))
    feats = frozenset(DEFAULT_FEATURES | REPLAY)
    scoring_col = 'fantasy_points_ppr' if a.scoring != 'Standard' else 'fantasy_points'
    rows = []
    t0 = time.time()
    for year in years:
        stats_df, _team_col, name_col, _ = load_and_merge_data(year, a.scoring)
        stats_df = stats_df.copy()
        stats_df['_wk'] = pd.to_numeric(stats_df['week'], errors='coerce')
        for week in weeks:
            wk = stats_df[stats_df['_wk'] == week]
            if wk.empty:
                continue
            agg = {c: wk.groupby(name_col, observed=True)[c].sum()
                   for c in (scoring_col, 'passing_tds', 'passing_attempts', 'passing_yards') if c in wk.columns}
            board, meta = build_weekly_projections(year, week, a.scoring, as_of_week=week, apply_injury=False,
                                                   features=feats)
            if board.empty:
                print(f'{year} w{week}: nothing ({meta.get("reason")})', flush=True)
                continue
            qbs = board[board['Pos'] == 'QB']
            for _, r in qbs.iterrows():
                ex = meta['explanations'].get((r['Player'], 'QB', r['Team']))
                if not ex:
                    continue
                tr = ex['stats'].get('passing_tds', {})
                row = {'year': year, 'week': week, 'player': r['Player'], 'team': r['Team'], 'opp': r['Opponent'],
                       'availability': r.get('Availability'), 'raw': r['Raw Model Proj Pts'],
                       'model': r['Model Proj Pts'], 'p_att': r.get('passing_attempts'),
                       'p_yds': r.get('passing_yards'), 'p_td': r.get('passing_tds'),
                       'p_int': r.get('passing_interceptions'), 'qb_starter': r.get('QB Projected Starter')}
                for k in TRACE_KEYS:
                    row[k] = tr.get(k)
                for c, s in agg.items():
                    row['a_' + ('pts' if c == scoring_col else c)] = s.get(r['Player'])
                rows.append(row)
            pd.DataFrame(rows).to_parquet(a.out)
            print(f'{year} w{week}: {len(qbs)} QBs, {len(rows)} rows total, {time.time() - t0:.0f}s', flush=True)
    print('done', len(rows), flush=True)


if __name__ == '__main__':
    main()
