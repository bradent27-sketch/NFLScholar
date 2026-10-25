"""Calibration lab (2026-10-08): one dump of the harness base arm, then every calibration scheme scored offline.

Why a new dump. `fit_seasonal_calibration.py --mode dump` builds with CALIBRATION_INPUT_FEATURES (DEFAULT_FEATURES minus
'calibration'), which since 2026-10-06 still contains `v2_season_anchor`, so its "raw" column would now be the anchored,
uncalibrated total; it also adds `v2_historical_ourlads`, which the harness base arm does not have. This dump builds
EXACTLY the harness v2 base arm (DEFAULT_FEATURES + the injury/reserve replay flags, `backtest_component.py
--add-features`), keeps every board row, and records each stage of the points pipeline so any scheme can be re-run
offline on top of the shipped stat lines:

    raw    'Raw Model Proj Pts'  - score(stat line) after vacancy/capacity; the calibration input
    final  'Model Proj Pts'      - what the harness scores (calibration line, sidelined/zero restore, season anchor)
    anchor_mean / anchor_games   - the player's mean fantasy points over his games before the target week, computed
                                   from the same `hist` frame the build hands apply_season_anchor
    p_<stat> / a_<stat>          - projected and actual stat line (actuals 0 when the player has no box score)
    actual / played              - the harness's actual (`_actual_points`, PPR sum); played = a box score or a silent game
                                   (actual 0; since 2026-10-25, the harness's verdict definition); played_box / silent
                                   say which
    plus context columns (games this season, availability, role confidence, snap share, injury status, ...) for
    segment diagnostics.

    python scripts/calibration_lab.py --mode dump --years 2021-2025 --weeks 1-18 --dump-path .sweeps/calibration_lab_v7.csv
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

from data.transforms import load_and_merge_data  # noqa: E402
from data.weekly_projections import (  # noqa: E402
    DEFAULT_FEATURES, _ALL_PROJECTION_STATS, _played_weeks_before, build_weekly_projections)
from data.silent_games import played_players, with_silent_games  # noqa: E402
from scripts.eval_weekly_model import _actual_points  # noqa: E402

REPLAY_FLAGS = frozenset({'v2_historical_injury_replay', 'v2_historical_reserve_replay'})
HARNESS_BASE_FEATURES = frozenset(DEFAULT_FEATURES | REPLAY_FLAGS)

# Actual-only scoring terms the projection does not model (they land in the points residual).
EXTRA_ACTUALS = ('rushing_fumbles_lost', 'receiving_fumbles_lost', 'sack_fumbles_lost',
                 'passing_2pt_conversions', 'rushing_2pt_conversions', 'receiving_2pt_conversions',
                 'special_teams_tds')

CONTEXT_COLS = {
    'Player': 'player', 'Pos': 'pos', 'Team': 'team', 'Opponent': 'opp',
    'Games This Season': 'games_season', 'Availability': 'avail', 'Workload If Active': 'workload',
    'Injury Status': 'inj', 'Role Confidence': 'role_conf', 'Expected Snap Share': 'snap',
    'Role Change Confidence': 'role_change_conf', 'QB Projected Starter': 'qb_starter',
    '_profile_target_earner_rank': 'target_rank', '_profile_source_kind': 'source_kind',
    '_profile_evidence_games': 'evidence_games', '_ps_elevation_candidate': 'ps_candidate',
    'Raw Model Proj Pts': 'raw', 'Model Proj Pts': 'final',
}


def do_dump(years, weeks, scoring, dump_path, features=HARNESS_BASE_FEATURES):
    scoring_col = 'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'
    rows = []
    for year in years:
        stats_df, _t, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns:
            print(f"{year}: no weekly data, skipped", flush=True)
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                continue
            wk = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week]
            a_cols = [s for s in (*_ALL_PROJECTION_STATS, *EXTRA_ACTUALS) if s in wk.columns]
            sa = wk.groupby(name_col, observed=True)[a_cols].sum()
            # 'played' counts a silent game (a snap with no box-score row, data/silent_games.py) as played with 0 points - the
            # harness's verdict definition since 2026-10-25; 'played_box' keeps the box-score-only flag
            silent_names = played_players(year, week, stats_df, name_col, _t) - set(actual.index.astype(str))
            hist_source = with_silent_games(stats_df, year, name_col, _t) if 'v2_silent_games' in features else stats_df
            build_weekly_projections.clear()
            proj, _meta = build_weekly_projections(year, week, scoring, as_of_week=week, apply_injury=False,
                                                   features=features)
            build_weekly_projections.clear()
            if proj.empty:
                print(f"{year} w{week} empty", flush=True)
                continue
            keep = [c for c in CONTEXT_COLS if c in proj.columns]
            d = proj[keep].rename(columns=CONTEXT_COLS).copy()
            # the season anchor's inputs, from the frame the build itself passes apply_season_anchor
            hist = _played_weeks_before(hist_source, week)
            if not hist.empty and 'fantasy_points' in hist.columns:
                g = pd.to_numeric(hist['fantasy_points'], errors='coerce').groupby(hist[name_col]).agg(['mean', 'count'])
                d['anchor_mean'] = d['player'].map(g['mean'])
                d['anchor_games'] = d['player'].map(g['count']).fillna(0).astype(int)
            else:
                d['anchor_mean'], d['anchor_games'] = np.nan, 0
            d['actual'] = d['player'].map(actual)
            d['played_box'] = d['actual'].notna()
            d['silent'] = ~d['played_box'] & d['player'].astype(str).isin(silent_names)
            d.loc[d['silent'], 'actual'] = 0.0
            d['played'] = d['played_box'] | d['silent']
            for s in _ALL_PROJECTION_STATS:
                d[f'p_{s}'] = pd.to_numeric(proj[s], errors='coerce').to_numpy() if s in proj.columns else np.nan
            for s in a_cols:
                d[f'a_{s}'] = d['player'].map(sa[s])
                d.loc[d['silent'], f'a_{s}'] = 0.0
            d['year'], d['week'] = year, week
            rows.append(d)
            print(f"{year} w{week} done  ({len(d)} rows, {int(d['played'].sum())} played)", flush=True)
            # write as we go so a partial run is still usable
            pd.concat(rows, ignore_index=True).to_csv(dump_path, index=False)
    if not rows:
        raise SystemExit("no data")
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(dump_path, index=False)
    print(f"\nwrote {len(out):,} player-weeks ({int(out['played'].sum()):,} played) -> {dump_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('dump',), required=True)
    ap.add_argument('--years', default='2021-2025')
    ap.add_argument('--weeks', default='1-18')
    ap.add_argument('--scoring', default='Full PPR')
    ap.add_argument('--dump-path', default=os.path.join('.sweeps', 'calibration_lab_v7.csv'))
    ap.add_argument('--extra-features', default='',
                    help="flags added to the harness base arm, e.g. v2_historical_ourlads for a week-1 dump: a "
                         "historical week-1 board without it has no depth chart (starting QBs project 0); the flag "
                         "only acts at a cold start and its archive covers 2022-2025")
    ap.add_argument('--remove-features', default='', help="flags taken out of the harness base arm")
    a = ap.parse_args()
    if a.mode == 'dump':
        y0, y1 = (int(x) for x in a.years.split('-'))
        w0, w1 = (int(x) for x in a.weeks.split('-'))
        years, weeks = list(range(y0, y1 + 1)), list(range(w0, w1 + 1))
        feats = frozenset((HARNESS_BASE_FEATURES | {f for f in a.extra_features.split(',') if f})
                          - {f for f in a.remove_features.split(',') if f})
        print(f"dump {years} wk{weeks[0]}-{weeks[-1]}  ({len(years) * len(weeks)} builds), "
              f"{len(feats)} flags (harness base arm{' + ' + a.extra_features if a.extra_features else ''})\n",
              flush=True)
        do_dump(years, weeks, a.scoring, a.dump_path, feats)


if __name__ == '__main__':
    main()
