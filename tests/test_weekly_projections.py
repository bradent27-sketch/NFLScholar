"""
Offline tests for the weekly projection model's compute layer
(data/weekly_projections.py).

Same convention as tests/test_matchup_signals.py: hand-built fixtures, no
network, no real season files - what's pinned here is the vectorized logic
that would fail SILENTLY against real data (a groupby keyed wrong, a
shrinkage formula pulling the wrong direction, a sign flipped on a margin).
Functions that touch the network or another loader (_target_margins_by_team,
_injury_multipliers, _load_pff_receiving, build_weekly_projections itself)
are exercised live instead, by scripts/validate_weekly_projections.py and by
actually opening the Weekly Rankings tab - not here.

Runs two ways: `python tests/test_weekly_projections.py` needs nothing but
the app's own dependencies, and `pytest tests/` works if pytest is installed.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

import data.weekly_projections as wp  # noqa: E402


def weekly(rows):
    """A minimal weekly-stats frame in the shape load_and_merge_data emits."""
    frame = pd.DataFrame(rows)
    for col in ('targets', 'receptions', 'rushing_attempts', 'rushing_yards',
                'receiving_yards', 'receiving_tds', 'rushing_tds',
                'weekly_snap_pct'):
        if col not in frame.columns:
            frame[col] = 0.0
    if 'season_type' not in frame.columns:
        frame['season_type'] = 'REG'
    if 'position' not in frame.columns:
        frame['position'] = pd.Series(dtype=str)
    return frame


# --- no-leakage week filters --------------------------------------------

def test_played_weeks_before_excludes_target_week_and_later():
    df = weekly([{'week': w} for w in (1, 2, 3, 4)])
    out = wp._played_weeks_before(df, as_of_week=3)
    assert sorted(out['week']) == [1, 2]


def test_played_weeks_before_excludes_week_zero_placeholder_rows():
    # A roster-only season (no games played yet) leaves week=0 placeholder
    # rows - these must never count as "history".
    df = weekly([{'week': 0}, {'week': 1}])
    out = wp._played_weeks_before(df, as_of_week=5)
    assert list(out['week']) == [1]


def test_all_played_weeks_keeps_everything_real():
    df = weekly([{'week': w} for w in (1, 5, 18)])
    assert sorted(wp._all_played_weeks(df)['week']) == [1, 5, 18]


# --- season totals / recent rate -----------------------------------------

def test_season_totals_sums_per_player_and_counts_distinct_games():
    df = weekly([
        {'name': 'A', 'week': 1, 'team': 'KC', 'position': 'WR', 'targets': 5, 'receiving_yards': 40},
        {'name': 'A', 'week': 2, 'team': 'KC', 'position': 'WR', 'targets': 7, 'receiving_yards': 60},
        {'name': 'B', 'week': 1, 'team': 'BUF', 'position': 'WR', 'targets': 3, 'receiving_yards': 20},
    ])
    out = wp._season_totals(df, 'name', 'team', 'WR', ['targets', 'receiving_yards'])
    a = out[out['name'] == 'A'].iloc[0]
    assert a['Games'] == 2 and a['targets'] == 12 and a['receiving_yards'] == 100 and a['Team'] == 'KC'


def test_season_totals_uses_most_recent_team_after_a_trade():
    df = weekly([
        {'name': 'A', 'week': 1, 'team': 'KC', 'position': 'WR', 'targets': 5},
        {'name': 'A', 'week': 5, 'team': 'BUF', 'position': 'WR', 'targets': 5},
    ])
    out = wp._season_totals(df, 'name', 'team', 'WR', ['targets'])
    assert out.iloc[0]['Team'] == 'BUF'


# --- quality-adjusted matchup / weighted own-history rate -----------------

def test_quality_adjusted_matchup_centers_on_one_and_favors_the_tougher_defense():
    # Player A faces KC (allows him half his normal level) and BUF (allows
    # him his normal level) once each; a league-average defense should read
    # 1.0 and KC (the tougher matchup) should read below BUF.
    df = weekly([
        {'name': 'A', 'week': 1, 'position': 'WR', 'opponent_team': 'KC', 'receiving_yards': 40},
        {'name': 'A', 'week': 2, 'position': 'WR', 'opponent_team': 'BUF', 'receiving_yards': 80},
        {'name': 'A', 'week': 3, 'position': 'WR', 'opponent_team': 'BUF', 'receiving_yards': 80},
    ])
    out = wp.build_quality_adjusted_matchup(df, 'name', ['receiving_yards'], as_of_week=4)
    assert out.loc['KC', 'receiving_yards'] < out.loc['BUF', 'receiving_yards']


def test_quality_adjusted_matchup_empty_without_opponent_column():
    df = weekly([{'name': 'A', 'week': 1, 'position': 'WR', 'receiving_yards': 40}])
    assert wp.build_quality_adjusted_matchup(df, 'name', ['receiving_yards'], as_of_week=2).empty


def test_weighted_player_rates_weighs_recent_games_more():
    # Same player, an old low game and a recent high game - the weighted
    # rate should sit closer to the RECENT value than a flat average would.
    df = weekly([
        {'name': 'A', 'week': 1, 'position': 'WR', 'opponent_team': 'KC', 'targets': 2},
        {'name': 'A', 'week': 6, 'position': 'WR', 'opponent_team': 'BUF', 'targets': 10},
    ])
    out, _totals = wp._weighted_player_rates(df, 'name', ['targets'], as_of_week=7,
                                    matchup_matrix=pd.DataFrame(), upcoming_opponent={'A': 'MIA'})
    flat_avg = 6.0
    assert out.loc['A', 'targets'] > flat_avg


def test_weighted_player_rates_upweights_a_rematch_game():
    # Two otherwise-identical-recency games at different values; the one
    # against the SAME team as the upcoming opponent should pull the rate
    # toward itself more than an ordinary equally-recent game would.
    df_rematch = weekly([
        {'name': 'A', 'week': 5, 'position': 'WR', 'opponent_team': 'KC', 'targets': 10},
        {'name': 'A', 'week': 4, 'position': 'WR', 'opponent_team': 'BUF', 'targets': 2},
    ])
    with_rematch, _ = wp._weighted_player_rates(
        df_rematch, 'name', ['targets'], as_of_week=6,
        matchup_matrix=pd.DataFrame(), upcoming_opponent={'A': 'KC'})

    df_no_rematch = weekly([
        {'name': 'A', 'week': 5, 'position': 'WR', 'opponent_team': 'SEA', 'targets': 10},
        {'name': 'A', 'week': 4, 'position': 'WR', 'opponent_team': 'BUF', 'targets': 2},
    ])
    without_rematch, _ = wp._weighted_player_rates(
        df_no_rematch, 'name', ['targets'], as_of_week=6,
        matchup_matrix=pd.DataFrame(), upcoming_opponent={'A': 'KC'})

    assert with_rematch.loc['A', 'targets'] > without_rematch.loc['A', 'targets']


def test_weighted_player_rates_empty_input_returns_empty():
    # Returns (rates, weighted totals) - the totals ride along so an
    # efficiency RATIO can be formed from two weighted sums.
    rates, totals = wp._weighted_player_rates(weekly([]), 'name', ['targets'], as_of_week=2,
                                              matchup_matrix=pd.DataFrame(), upcoming_opponent={})
    assert rates.empty and totals.empty


# --- shrinkage --------------------------------------------------------------


def test_blended_rate_leans_on_prior_with_zero_games():
    # A player with no current-season games yet should land entirely on
    # the prior - w_current = 0/(0+K) = 0.
    out = wp._blended_rate(np.array([50.0]), np.array([0.0]), np.array([50.0]),
                           np.array([10.0]), 'targets', np.array([0.5]))
    assert abs(out[0] - 50.0) < 1e-9


def test_blended_rate_converges_toward_current_rate_with_more_games():
    # Same prior/current-rate gap, more games played -> closer to the
    # current-season rate. This is the literal "current season outweighs
    # the past as the sample grows" mechanism.
    few = wp._blended_rate(np.array([20.0]), np.array([1.0]), np.array([5.0]),
                           np.array([5.0]), 'targets', np.array([0.5]))
    many = wp._blended_rate(np.array([20.0]), np.array([10.0]), np.array([5.0]),
                            np.array([5.0]), 'targets', np.array([0.5]))
    assert many[0] > few[0]


def test_role_confidence_shrinks_k_less_for_a_confident_role():
    # Higher role_confidence -> smaller effective K -> more weight on the
    # player's own current rate for the SAME games-played count.
    low_conf = wp._blended_rate(np.array([20.0]), np.array([4.0]), np.array([5.0]),
                                np.array([5.0]), 'targets', np.array([0.0]))
    high_conf = wp._blended_rate(np.array([20.0]), np.array([4.0]), np.array([5.0]),
                                 np.array([5.0]), 'targets', np.array([1.0]))
    assert high_conf[0] > low_conf[0]


def test_blended_rate_falls_back_to_position_rate_with_no_prior_season():
    out = wp._blended_rate(np.array([20.0]), np.array([0.0]), np.array([np.nan]),
                           np.array([8.0]), 'targets', np.array([0.5]))
    assert abs(out[0] - 8.0) < 1e-9


# --- game script -----------------------------------------------------------

def test_team_week_margins_sign_convention():
    # Home team wins 30-10 (margin +20 for home, -20 for away).
    sched = pd.DataFrame([{'week': 1, 'home_team': 'KC', 'away_team': 'BUF',
                           'home_score': 30, 'away_score': 10}])
    out = wp._team_week_margins(sched)
    kc = out[out['Team'] == 'KC'].iloc[0]
    buf = out[out['Team'] == 'BUF'].iloc[0]
    assert kc['margin'] == 20 and buf['margin'] == -20


def test_team_week_margins_drops_unplayed_games():
    sched = pd.DataFrame([{'week': 1, 'home_team': 'KC', 'away_team': 'BUF',
                           'home_score': np.nan, 'away_score': np.nan}])
    assert wp._team_week_margins(sched).empty


def test_week_opponents_maps_both_directions_and_omits_byes():
    sched = pd.DataFrame([{'week': 3, 'home_team': 'KC', 'away_team': 'BUF'}])
    out = wp._week_opponents(sched, 3)
    assert out == {'KC': 'BUF', 'BUF': 'KC'}
    assert 'DEN' not in out  # DEN has no game this week -> on a bye


def test_vectorized_game_script_multiplier_neutral_without_enough_history():
    df = weekly([{'name': 'A', 'week': 1, 'team': 'KC', 'position': 'WR', 'receiving_yards': 50}])
    sched = pd.DataFrame([{'week': 1, 'home_team': 'KC', 'away_team': 'BUF',
                           'home_score': 20, 'away_score': 17}])
    out = wp._vectorized_game_script_multiplier(
        df, 'name', 'team', as_of_week=2, schedule_df=sched,
        target_margins=pd.Series({'A': 3.0}), stat='receiving_yards')
    # Only one game of history - below the 4-game floor - so no multiplier
    # is produced for this player at all.
    assert out.empty


# --- missing-data safety ----------------------------------------------------

def test_season_totals_empty_input_returns_empty_not_a_crash():
    out = wp._season_totals(weekly([]), 'name', 'team', 'WR', ['targets'])
    assert out.empty


def test_role_confidence_handles_no_snap_column_gracefully():
    df = pd.DataFrame({'name': ['A'], 'week': [1], 'position': ['WR']})
    out = wp._role_confidence(df, 'name', as_of_week=2, pos='WR', pff_rec=pd.DataFrame())
    assert out.empty


# --- expected snap share / role volume ---------------------------------------

def test_expected_snap_share_separates_a_backup_from_a_starter():
    # The measured failure this exists for: a backup's PER-GAME rate looks
    # like a starter's on a small sample, and only snap share separates them.
    df = weekly([
        {'name': 'Starter', 'week': w, 'position': 'QB', 'team': 'KC',
         'opponent_team': 'DEN', 'weekly_snap_pct': 100.0} for w in (1, 2, 3, 4, 5)
    ] + [
        {'name': 'Backup', 'week': w, 'position': 'QB', 'team': 'KC',
         'opponent_team': 'DEN', 'weekly_snap_pct': 12.0} for w in (2, 5)
    ])
    share = wp.expected_snap_share(df, 'name', 'team', as_of_week=6)
    assert share['Starter'] == 1.0
    assert 0.0 < share['Backup'] < 0.2


def test_expected_snap_share_reads_a_role_takeover_from_recent_games_only():
    # Tyler Shough's real 2025 shape: 4% -> 54% -> 90% -> 95%. A season
    # average would still call him a backup; a four-appearance window
    # shouldn't.
    df = weekly([
        {'name': 'Riser', 'week': 1, 'position': 'QB', 'team': 'NO',
         'opponent_team': 'ATL', 'weekly_snap_pct': 4.0},
        {'name': 'Riser', 'week': 2, 'position': 'QB', 'team': 'NO',
         'opponent_team': 'ATL', 'weekly_snap_pct': 54.0},
        {'name': 'Riser', 'week': 3, 'position': 'QB', 'team': 'NO',
         'opponent_team': 'ATL', 'weekly_snap_pct': 90.0},
        {'name': 'Riser', 'week': 4, 'position': 'QB', 'team': 'NO',
         'opponent_team': 'ATL', 'weekly_snap_pct': 95.0},
        {'name': 'Riser', 'week': 5, 'position': 'QB', 'team': 'NO',
         'opponent_team': 'ATL', 'weekly_snap_pct': 99.0},
    ])
    share = wp.expected_snap_share(df, 'name', 'team', as_of_week=6, lookback=4)
    assert share['Riser'] > 0.8  # weeks 2-5, not weeks 1-5


def test_expected_snap_share_does_not_punish_a_returning_starter():
    # Measured regression the team-weeks version caused: a starter who
    # missed two weeks must not read as a part-time player on his return.
    df = weekly([
        # 'Hurt' misses weeks 3-4 entirely; 'Healthy' plays all four.
        {'name': 'Hurt', 'week': 1, 'position': 'RB', 'team': 'SF',
         'opponent_team': 'LA', 'weekly_snap_pct': 85.0},
        {'name': 'Hurt', 'week': 2, 'position': 'RB', 'team': 'SF',
         'opponent_team': 'LA', 'weekly_snap_pct': 88.0},
    ] + [
        {'name': 'Healthy', 'week': w, 'position': 'RB', 'team': 'SF',
         'opponent_team': 'LA', 'weekly_snap_pct': 86.0} for w in (1, 2, 3, 4)
    ])
    share = wp.expected_snap_share(df, 'name', 'team', as_of_week=5)
    assert share['Hurt'] > 0.8
    assert abs(share['Hurt'] - share['Healthy']) < 0.06


# --- roles / role-conditioned matchup ----------------------------------------

def test_player_roles_split_receivers_by_depth_of_target():
    rows = []
    for i in range(12):
        # 12 receivers on an evenly-spread ADOT ladder from 4 to 15 yards
        adot = 4 + i
        rows.append({'name': f'W{i}', 'week': 1, 'position': 'WR', 'team': 'KC',
                     'opponent_team': 'DEN', 'targets': 30,
                     'receiving_air_yards': 30 * adot})
    roles = wp.build_player_roles(weekly(rows), 'name', 'WR')
    assert roles['W0'] == 'WR_SHORT'
    assert roles['W11'] == 'WR_DEEP'
    assert roles['W5'] in ('WR_SHORT', 'WR_MID')


def test_player_roles_park_a_thin_sample_in_the_middle_bucket():
    # Two targets is not a measured role, whatever the ADOT says.
    rows = [{'name': f'W{i}', 'week': 1, 'position': 'WR', 'team': 'KC',
             'opponent_team': 'DEN', 'targets': 30, 'receiving_air_yards': 30 * (4 + i)}
            for i in range(12)]
    rows.append({'name': 'Thin', 'week': 1, 'position': 'WR', 'team': 'KC',
                 'opponent_team': 'DEN', 'targets': 2, 'receiving_air_yards': 80})
    roles = wp.build_player_roles(weekly(rows), 'name', 'WR')
    assert roles['Thin'] == 'WR_MID'


def test_role_matchup_blend_shrinks_toward_the_overall_rating():
    overall = pd.DataFrame({'targets': [1.0]}, index=['DEN'])
    role_tables = {'WR_DEEP': pd.DataFrame({'targets': [1.4]}, index=['DEN'])}
    opponents = np.array(['DEN'])
    roles = np.array(['WR_DEEP'])
    thin = wp._role_adjusted_multiplier(overall, role_tables, {('DEN', 'WR_DEEP'): 1.0},
                                        opponents, roles, 'targets')
    thick = wp._role_adjusted_multiplier(overall, role_tables, {('DEN', 'WR_DEEP'): 40.0},
                                         opponents, roles, 'targets')
    # More role-specific evidence -> closer to the role rating, never past it.
    assert 1.0 < thin[0] < thick[0] <= 1.4


def test_role_matchup_falls_back_when_the_defense_never_faced_that_role():
    overall = pd.DataFrame({'targets': [1.2]}, index=['DEN'])
    out = wp._role_adjusted_multiplier(overall, {'WR_DEEP': pd.DataFrame()}, {},
                                       np.array(['DEN']), np.array(['WR_SHORT']), 'targets')
    assert abs(out[0] - 1.2) < 1e-9


# --- game environment ---------------------------------------------------------

def test_game_environment_sign_convention_favours_the_home_team():
    # spread_line is POSITIVE when the HOME team is favored - the one thing
    # that silently inverts this whole component if it's read backwards.
    sched = pd.DataFrame({'week': [1], 'home_team': ['KC'], 'away_team': ['DEN'],
                          'total_line': [46.0], 'spread_line': [7.0], 'roof': ['outdoors']})
    env = wp.game_environment(sched, 1)
    assert env['KC']['implied'] == 26.5
    assert env['DEN']['implied'] == 19.5
    assert env['KC']['indoor'] is False


def test_game_environment_multiplier_is_neutral_without_a_posted_line():
    sched = pd.DataFrame({'week': [1], 'home_team': ['KC'], 'away_team': ['DEN'],
                          'total_line': [np.nan], 'spread_line': [np.nan], 'roof': ['dome']})
    env = wp.game_environment(sched, 1)
    out = wp._game_env_multiplier(env, np.array(['KC']), 'QB', league_implied=22.0)
    # Venue still applies; the total does not, because there isn't one.
    assert abs(out[0] - wp.VENUE_MULT['QB']['indoor']) < 1e-9


# --- calibration --------------------------------------------------------------

def test_calibration_is_one_sided():
    slope, intercept = wp.WEEKLY_CALIBRATION['WR']
    crossover = intercept / (1 - slope)
    high, low = crossover + 10, crossover - 5
    assert min(high, intercept + slope * high) < high      # the top is shrunk
    assert min(low, intercept + slope * low) == low        # the bottom is left alone


# --- teammate vacancy ---------------------------------------------------------

def test_vacated_targets_move_to_healthy_teammates_in_proportion():
    frame = pd.DataFrame({
        'Player': ['Out', 'Big', 'Small', 'OtherTeam'],
        'Pos': ['WR', 'WR', 'WR', 'WR'],
        'Team': ['KC', 'KC', 'KC', 'DEN'],
        'targets': [0.0, 8.0, 4.0, 8.0],
        'receiving_yards': [0.0, 80.0, 40.0, 80.0],
    })
    # `_full_targets` is what the position loop stashes before the injury
    # discount zeroes an Out player's own line - without it there is nothing
    # left to redistribute, which is the whole point of carrying it.
    frame['_full_targets'] = [9.0, 8.0, 4.0, 8.0]
    out, n = wp.redistribute_vacated_usage(frame, {'Out': 0.0})
    assert n == 2
    # 9 vacated targets, 75% of them re-used, split 2:1 by existing usage.
    assert out.loc[1, 'targets'] > out.loc[2, 'targets']
    assert out.loc[1, 'targets'] > 8.0 and out.loc[2, 'targets'] > 4.0
    assert out.loc[3, 'targets'] == 8.0          # a different team is untouched
    # Dependent stats ride the same factor, so yards per target is unchanged.
    assert abs(out.loc[1, 'receiving_yards'] / out.loc[1, 'targets'] - 10.0) < 1e-6


def test_vacancy_growth_is_capped():
    frame = pd.DataFrame({
        'Player': ['Out', 'Only'],
        'Pos': ['RB', 'RB'],
        'Team': ['KC', 'KC'],
        'rushing_attempts': [20.0, 4.0],
        'rushing_yards': [80.0, 16.0],
    })
    frame['_full_rushing_attempts'] = [50.0, 4.0]
    out, _n = wp.redistribute_vacated_usage(frame, {'Out': 0.4})
    assert out.loc[1, 'rushing_attempts'] <= 4.0 * wp.VACANCY_MAX_GROWTH + 1e-9


def test_scoring_a_cross_position_frame_survives_missing_stat_columns():
    # Regression: the assembled frame is the UNION of four positions' stat
    # lists, so a receiver's row carries NaN for every passing stat.
    # score_projected_stats reads with .get(stat, 0), which returns the NaN
    # for a key that exists and is NaN - and max(0.0, nan) is 0.0, so a
    # whole position silently projected exactly zero points while showing a
    # sensible stat line. Anything re-scoring a mixed frame must fill first.
    from data.transforms import score_projected_stats
    row = {'receiving_yards': 80.0, 'receptions': 6.0, 'receiving_tds': 0.5,
           'passing_yards': float('nan'), 'passing_tds': float('nan')}
    assert np.isnan(score_projected_stats(row, 'Full PPR'))
    filled = {k: (0.0 if v != v else v) for k, v in row.items()}
    assert score_projected_stats(filled, 'Full PPR') > 10


def test_vacancy_is_a_no_op_with_nobody_out():
    frame = pd.DataFrame({'Player': ['A'], 'Pos': ['WR'], 'Team': ['KC'], 'targets': [6.0]})
    out, n = wp.redistribute_vacated_usage(frame, {'A': 1.0})
    assert n == 0 and out.loc[0, 'targets'] == 6.0


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failures.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
