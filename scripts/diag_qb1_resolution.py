"""
How often does the in-season QB1 resolver pick the QB who actually starts?
(2026-10-05, for 'v2_qb1_returning_starter'.)

Runs data.weekly_projections.resolve_inseason_qb1s directly - the same inputs
build_weekly_projections gives it (played-weeks history annotated for
interrupted games, the target week's official injury report + reserve list as
the unavailable set) - for every team-week, with and without the returning-
starter rule, and with NO manual overrides (the model's own logic; the
override file only exists for 2026). The actual starter is the team's QB with
the most pass attempts that week.

Usage:
    python scripts/diag_qb1_resolution.py --years 2022,2023,2024,2025 --weeks 3-17
    python scripts/diag_qb1_resolution.py --years 2026 --weeks 2-4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

import data.weekly_projections as wp  # noqa: E402
from data.availability_overrides import _ASSUME_OUT_STATUSES  # noqa: E402
from data.historical_availability import historical_injury_profiles, historical_reserve_profiles  # noqa: E402
from data.loaders import load_schedule  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402


def run(years, weeks):
    rows = []
    for year in years:
        stats_df, team_col, name_col, _ = load_and_merge_data(year, 'Full PPR')
        prior_stats, prior_team_col, prior_name_col, _ = load_and_merge_data(year - 1, 'Full PPR')
        schedule = load_schedule(year)
        prior_played = wp._all_played_weeks(prior_stats)
        prior_annotated = wp.annotate_player_history_participation(
            prior_played, prior_name_col, prior_team_col, load_schedule(year - 1))
        player_prior = prior_annotated[prior_annotated['_player_history_eligible']].copy()
        s = stats_df.copy()
        s['_w'] = pd.to_numeric(s['week'], errors='coerce')
        qbs_all = s[s['position'].astype(str).str.upper().eq('QB')].copy()
        qbs_all['_pa'] = pd.to_numeric(qbs_all['passing_attempts'], errors='coerce').fillna(0)
        qbs_all['_team'] = wp._clean_team_key(qbs_all['game_team'] if 'game_team' in qbs_all.columns else qbs_all[team_col])
        starters = (qbs_all.sort_values('_pa', ascending=False).drop_duplicates(['_w', '_team'])
                    .set_index(['_w', '_team'])[name_col])
        for week in weeks:
            if not (s['_w'] == week).any():
                continue
            hist = wp._played_weeks_before(stats_df, week)
            if hist.empty:
                continue
            hist_annotated = wp.annotate_player_history_participation(
                hist, name_col, team_col, schedule,
                prior_reference=wp._player_snap_seed(player_prior, prior_name_col))
            player_hist = hist_annotated[hist_annotated['_player_history_eligible']].copy()
            profiles = wp.merge_reserve_profiles(historical_injury_profiles(year, week, schedule),
                                                 historical_reserve_profiles(year, week))
            unavailable = {n for n, p in profiles.items()
                           if str(p.get('status', '')).strip().lower() in _ASSUME_OUT_STATUSES
                           or float(pd.to_numeric(pd.Series([p.get('plays_probability')]), errors='coerce').fillna(1.0).iloc[0]) <= 0.01}
            current_qbs = hist[hist['position'].astype(str).str.upper().eq('QB')].copy()
            current_qbs = (current_qbs.assign(_week=pd.to_numeric(current_qbs['week'], errors='coerce'))
                           .sort_values('_week').drop_duplicates(name_col, keep='last'))
            empty = pd.DataFrame(columns=wp.QB1_OVERRIDE_COLUMNS)
            returning = wp.qb1_returning_inputs(year, week, stats_df, name_col, team_col,
                                                player_prior, prior_name_col, prior_team_col, schedule)
            res = {}
            for arm, ret in (('base', None), ('flag', returning)):
                res[arm] = wp.resolve_inseason_qb1s(current_qbs, name_col, team_col, player_hist, name_col, team_col,
                                                    week, year, overrides=empty, unavailable_players=unavailable,
                                                    returning=ret)
            teams = sorted(set(starters.loc[week].index) if week in starters.index.get_level_values(0) else [])
            for team in teams:
                actual = starters.get((week, team))
                rec = {'year': year, 'week': week, 'team': team, 'actual': actual}
                for arm in ('base', 'flag'):
                    info = res[arm]['by_team'].get(team, {})
                    rec[f'{arm}_status'] = info.get('status', 'none')
                    rec[f'{arm}_pick'] = info.get('player')
                    rec[f'{arm}_ok'] = (info.get('player') is not None and actual is not None and
                                        clean_name_exact(pd.Series([info.get('player')])).iloc[0]
                                        == clean_name_exact(pd.Series([actual])).iloc[0])
                rows.append(rec)
            print(f"{year} wk{week} done", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2022,2023,2024,2025')
    ap.add_argument('--weeks', default='3-17')
    ap.add_argument('--out', default='.sweeps/diag_qb1_resolution.parquet')
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.weeks.split('-'))
    d = run([int(y) for y in a.years.split(',')], list(range(lo, hi + 1)))
    d.to_parquet(a.out, index=False)
    pd.set_option('display.width', 200)
    n = len(d)
    for arm in ('base', 'flag'):
        resolved = d[f'{arm}_pick'].notna()
        print(f"{arm}: team-weeks {n} | resolved {resolved.sum()} | correct {d[f'{arm}_ok'].sum()} "
              f"({d[f'{arm}_ok'].mean():.1%} of all) | wrong {(resolved & ~d[f'{arm}_ok']).sum()} | unresolved {(~resolved).sum()}")
    ch = d[d['base_pick'].fillna('-') != d['flag_pick'].fillna('-')]
    print(f"\nchanged picks: {len(ch)} | flag right & base wrong: {(ch['flag_ok'] & ~ch['base_ok']).sum()} | "
          f"base right & flag wrong: {(ch['base_ok'] & ~ch['flag_ok']).sum()} | both wrong: {(~ch['base_ok'] & ~ch['flag_ok']).sum()}")
    print(ch[['year', 'week', 'team', 'actual', 'base_pick', 'base_status', 'flag_pick', 'flag_status']].to_string(index=False))


if __name__ == '__main__':
    main()
