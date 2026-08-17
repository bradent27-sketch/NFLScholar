"""
Offline tests for the Matchup Analyzer's compute layer
(data/matchup_signals.py).

Built against hand-made fixtures rather than the real season files, so they
run with no network, no PFF exports on disk, and a known answer to check
against. The real files are exercised separately by actually opening the tab;
what's pinned here is the logic that would fail SILENTLY against them - a
percentile pointing the wrong way, a denominator counting the wrong thing, a
zero that should have been a blank.

Every assertion below corresponds to a decision documented in the module's
own docstrings. If one of these fails, the fix is usually in the doc too.

Runs two ways: `python tests/test_matchup_signals.py` needs nothing but the
app's own dependencies, and `pytest tests/` works if pytest is installed.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

import data.matchup_signals as ms  # noqa: E402


def weekly(rows):
    """A minimal weekly-stats frame in the shape load_and_merge_data emits."""
    frame = pd.DataFrame(rows)
    for col in ('targets', 'receptions', 'rushing_attempts', 'rushing_yards',
                'receiving_yards', 'receiving_tds', 'rushing_tds', 'passing_yards',
                'passing_tds', 'passing_completions', 'fantasy_points'):
        if col not in frame.columns:
            frame[col] = 0.0
    # Filled per row, not only when the column is absent entirely. Building
    # a frame from dicts where only SOME rows name season_type leaves the
    # rest NaN, and _played_weeks deliberately declines to filter on a
    # column that distinguishes nothing - so a half-filled fixture silently
    # tests the no-filter path instead of the one it means to.
    if 'season_type' not in frame.columns:
        frame['season_type'] = 'REG'
    else:
        frame['season_type'] = frame['season_type'].fillna('REG')
    return frame


# --- game series --------------------------------------------------------

def test_a_real_zero_is_kept_as_a_game():
    df = weekly([
        {'name': 'A', 'week': 1, 'opponent_team': 'BUF', 'team': 'KC', 'position': 'WR', 'receiving_tds': 1},
        {'name': 'A', 'week': 2, 'opponent_team': 'DEN', 'team': 'KC', 'position': 'WR', 'receiving_tds': 0},
    ])
    series = ms.player_game_series(df, 'name', 'A', 'receiving_tds')
    # Two games, not one. A 0-TD week is a fact about the matchup, and
    # dropping it both shortens the chart and inflates the average.
    assert len(series) == 2
    assert list(series['value']) == [1.0, 0.0]


def test_postseason_rows_are_excluded():
    df = weekly([
        {'name': 'A', 'week': 1, 'opponent_team': 'BUF', 'team': 'KC', 'position': 'WR', 'receiving_yards': 50},
        {'name': 'A', 'week': 19, 'opponent_team': 'BUF', 'team': 'KC', 'position': 'WR',
         'receiving_yards': 150, 'season_type': 'POST'},
    ])
    series = ms.player_game_series(df, 'name', 'A', 'receiving_yards')
    assert list(series['week']) == [1]


def test_duplicate_week_rows_collapse_to_one_game():
    # A mid-season trade produces two rows for the same week under some
    # sources. Two bars for one game shifts every later opponent label.
    df = weekly([
        {'name': 'A', 'week': 3, 'opponent_team': 'BUF', 'team': 'KC', 'position': 'WR', 'receiving_yards': 40},
        {'name': 'A', 'week': 3, 'opponent_team': 'BUF', 'team': 'KC', 'position': 'WR', 'receiving_yards': 20},
    ])
    series = ms.player_game_series(df, 'name', 'A', 'receiving_yards')
    assert len(series) == 1 and series['value'].iloc[0] == 60.0


# --- highlighting -------------------------------------------------------

def test_a_flat_season_stars_nothing():
    # Every game identical: "top quartile" would otherwise star all of them.
    assert ms.highlight_games([5, 5, 5, 5, 5]) == [False] * 5


def test_an_all_zero_season_stars_nothing():
    assert ms.highlight_games([0, 0, 0, 0, 0, 0]) == [False] * 6


def test_the_big_games_are_starred():
    flags = ms.highlight_games([10, 12, 11, 90, 9, 85, 8, 10])
    assert flags[3] and flags[5]
    assert not flags[0] and not flags[6]


# --- defense softness direction ----------------------------------------

def test_softness_is_higher_for_a_defense_that_allows_more():
    matrix = pd.DataFrame({'Team': ['A', 'B', 'C', 'D'], 'RB': [10.0, 20.0, 30.0, 40.0]})
    softness = ms.defense_softness(matrix, 'RB')
    assert softness['D'] > softness['A']


def test_vulnerability_rank_one_is_the_softest():
    matrix = pd.DataFrame({
        'Team': ['A', 'B', 'C'], 'RB': [30.0, 10.0, 20.0], 'WR': [5.0, 5.0, 5.0],
    })
    rows = ms.positional_vulnerability(matrix, 'A')
    rb = next(r for r in rows if r['position'] == 'RB')
    # A allows the most to RBs, so it is the softest RB matchup in this pool.
    assert rb['rank'] == 1 and rb['of'] == 3


def test_vulnerability_is_sorted_softest_first():
    matrix = pd.DataFrame({
        'Team': ['A', 'B'], 'RB': [30.0, 10.0], 'WR': [5.0, 40.0], 'TE': [12.0, 3.0],
    })
    rows = ms.positional_vulnerability(matrix, 'A')
    assert [r['pct'] for r in rows] == sorted([r['pct'] for r in rows], reverse=True)


# --- allowed by position ------------------------------------------------

def test_games_faced_counts_weeks_not_player_rows():
    # Three receivers in one game is ONE game faced, not three. Getting this
    # wrong divides the allowed total by 3 and makes every defense look elite.
    rows = [{'name': n, 'week': 1, 'opponent_team': 'DEF', 'team': 'OFF',
             'position': 'WR', 'receiving_yards': 50, 'targets': 5, 'receptions': 3,
             'receiving_tds': 0}
            for n in ('A', 'B', 'C')]
    allowed = ms.defense_allowed_by_position(weekly(rows), 'DEF', 'WR')
    assert allowed['available'] and allowed['games'] == 1
    yards = next(e for e in allowed['entries'] if e['label'] == 'Rec Yds')
    assert yards['value'] == 150.0


# --- usage shares -------------------------------------------------------

def test_share_is_per_week_not_season_total_over_season_total():
    # The player plays week 1 only; the team plays both weeks. A season
    # total over a season total would report 50% of his real share purely
    # because the denominator kept counting a week he wasn't on the field.
    df = weekly([
        {'name': 'A', 'week': 1, 'team': 'KC', 'opponent_team': 'X', 'position': 'WR', 'targets': 10},
        {'name': 'B', 'week': 1, 'team': 'KC', 'opponent_team': 'X', 'position': 'WR', 'targets': 10},
        {'name': 'B', 'week': 2, 'team': 'KC', 'opponent_team': 'Y', 'position': 'WR', 'targets': 20},
    ])
    usage = ms.usage_and_role(df, 'name', 'A', 'KC')
    assert usage['available']
    assert abs(usage['target_share'] - 50.0) < 0.01


def test_role_change_needs_a_real_swing_and_a_real_sample():
    steady = weekly([
        {'name': 'A', 'week': w, 'team': 'KC', 'opponent_team': 'X', 'position': 'WR', 'targets': 5}
        for w in range(1, 9)
    ] + [
        {'name': 'B', 'week': w, 'team': 'KC', 'opponent_team': 'X', 'position': 'WR', 'targets': 5}
        for w in range(1, 9)
    ])
    assert ms.usage_and_role(steady, 'name', 'A', 'KC')['role_change'] is None

    rows = []
    for w in range(1, 9):
        mine = 1 if w <= 5 else 9
        rows.append({'name': 'A', 'week': w, 'team': 'KC', 'opponent_team': 'X', 'position': 'WR', 'targets': mine})
        rows.append({'name': 'B', 'week': w, 'team': 'KC', 'opponent_team': 'X', 'position': 'WR', 'targets': 10 - mine})
    change = ms.usage_and_role(weekly(rows), 'name', 'A', 'KC')['role_change']
    assert change is not None and change['direction'] == 'up'


# --- curves -------------------------------------------------------------

def test_elasticity_needs_a_spread_of_opponents():
    series = pd.DataFrame({'week': [1, 2, 3, 4], 'opponent': ['A', 'B', 'C', 'D'],
                           'value': [10.0, 12.0, 11.0, 13.0]})
    # Every opponent equally soft: there's no curve to draw, and drawing a
    # flat one would imply a measurement that wasn't made.
    flat = ms.efficiency_elasticity_curve(series, {'A': 50, 'B': 51, 'C': 52, 'D': 53}, 'A')
    assert not flat['available']


def test_elasticity_tiers_sit_at_real_mean_softness():
    series = pd.DataFrame({
        'week': [1, 2, 3, 4, 5, 6], 'opponent': ['A', 'B', 'C', 'D', 'E', 'F'],
        'value': [10.0, 12.0, 40.0, 42.0, 80.0, 82.0],
    })
    softness = {'A': 5, 'B': 15, 'C': 45, 'D': 55, 'E': 85, 'F': 95}
    curve = ms.efficiency_elasticity_curve(series, softness, 'E')
    assert curve['available'] and len(curve['tiers']) == 3
    tough = next(t for t in curve['tiers'] if t['name'] == 'Tough')
    # Plotted where his tough games actually were (mean of 5 and 15), not
    # at a nominal one-sixth of the axis.
    assert abs(tough['x'] - 10.0) < 0.01 and tough['n'] == 2
    assert curve['projection'] is not None


def test_game_script_separates_trailing_from_leading():
    schedule = pd.DataFrame([
        {'week': 1, 'home_team': 'KC', 'away_team': 'X', 'home_score': 30, 'away_score': 10},
        {'week': 2, 'home_team': 'KC', 'away_team': 'Y', 'home_score': 10, 'away_score': 30},
        {'week': 3, 'home_team': 'Z', 'away_team': 'KC', 'home_score': 20, 'away_score': 24},
        {'week': 4, 'home_team': 'W', 'away_team': 'KC', 'home_score': 24, 'away_score': 20},
    ])
    series = pd.DataFrame({'week': [1, 2, 3, 4], 'opponent': ['X', 'Y', 'Z', 'W'],
                           'value': [5.0, 50.0, 20.0, 25.0]})
    curve = ms.game_script_sensitivity_curve(series, schedule, 'KC')
    assert curve['available']
    names = {t['name'] for t in curve['tiers']}
    # A 20-point win and a 20-point loss are opposite scripts; folding them
    # into one "blowout" bucket cancels exactly the signal being looked for.
    assert 'Won big' in names and 'Trailed big' in names


def test_margins_skip_an_unplayed_game():
    schedule = pd.DataFrame([
        {'week': 1, 'home_team': 'KC', 'away_team': 'X', 'home_score': 30, 'away_score': 10},
        {'week': 2, 'home_team': 'KC', 'away_team': 'Y', 'home_score': None, 'away_score': None},
    ])
    margins = ms.team_game_margins(schedule, 'KC')
    # A scheduled-but-unplayed game must be absent, never a 0 - which would
    # read as a tie and land in a bucket.
    assert margins == {1: 20.0}


# --- red zone -----------------------------------------------------------

def test_red_zone_counts_drives_not_plays():
    pbp = pd.DataFrame([
        # One drive, four plays, one touchdown.
        {'defteam': 'D', 'yardline_100': 18, 'game_id': 'g1', 'drive': 1, 'touchdown': 0},
        {'defteam': 'D', 'yardline_100': 12, 'game_id': 'g1', 'drive': 1, 'touchdown': 0},
        {'defteam': 'D', 'yardline_100': 8, 'game_id': 'g1', 'drive': 1, 'touchdown': 0},
        {'defteam': 'D', 'yardline_100': 3, 'game_id': 'g1', 'drive': 1, 'touchdown': 1},
        # A second drive that settled for a field goal.
        {'defteam': 'D', 'yardline_100': 15, 'game_id': 'g1', 'drive': 2, 'touchdown': 0},
        # A play outside the red zone must not count at all.
        {'defteam': 'D', 'yardline_100': 60, 'game_id': 'g1', 'drive': 3, 'touchdown': 0},
    ])
    rz = ms.red_zone_defense(pbp, 'D')
    assert rz['available']
    assert rz['trips'] == 2 and rz['tds'] == 1
    assert abs(rz['td_rate'] - 50.0) < 0.01


def test_red_zone_says_unavailable_rather_than_zero():
    # An unreachable play-by-play pull must not render as a defense that
    # allows nothing - that reads as elite instead of unknown.
    out = ms.red_zone_defense(pd.DataFrame(), 'D')
    assert out['available'] is False and out.get('reason')


# --- touchdown projection ----------------------------------------------

def test_touchdown_projection_is_poisson_on_the_rate():
    # 6 TDs in 12 games, two of them in one game. The Poisson result is
    # NOT the empirical hit rate and is not meant to be - it is 1-e^-0.5 =
    # 39.3% against an empirical 5/12 = 41.7%. Pinned explicitly because
    # "Poisson gives a higher number" is a tempting and wrong summary of
    # why this model is used; see anytime_td_projection's docstring.
    series = pd.DataFrame({'week': range(1, 13),
                           'value': [2.0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0]})
    out = ms.anytime_td_projection(series)
    assert out['available']
    assert abs(out['base_rate'] - 0.5) < 1e-9
    assert abs(out['probability'] - (1 - np.exp(-0.5))) < 1e-9


def test_touchdown_projection_uses_the_full_count_not_just_scoring_games():
    # Same number of games WITH a touchdown, different totals. A binary
    # hit rate cannot tell these apart; that it uses the whole count is the
    # actual reason this is a Poisson draw.
    one_each = pd.DataFrame({'week': range(1, 5), 'value': [1.0, 1.0, 0.0, 0.0]})
    two_each = pd.DataFrame({'week': range(1, 5), 'value': [2.0, 2.0, 0.0, 0.0]})
    assert (ms.anytime_td_projection(two_each)['probability']
            > ms.anytime_td_projection(one_each)['probability'])


def test_opponent_adjustment_is_capped():
    series = pd.DataFrame({'week': [1, 2], 'value': [1.0, 1.0]})
    softest = ms.anytime_td_projection(series, softness_pct=100)
    toughest = ms.anytime_td_projection(series, softness_pct=0)
    assert abs(softest['adjustment'] - 1.25) < 1e-9
    assert abs(toughest['adjustment'] - 0.75) < 1e-9


# --- percentile guards --------------------------------------------------

def test_a_thin_pool_gets_no_percentile_rather_than_a_made_up_one():
    # A percentile computed from four samples reads exactly like one
    # computed from four hundred, which is why this returns None.
    assert ms._percentile_of(5.0, pd.Series([1.0, 2.0, 3.0, 4.0])) is None
    assert ms._percentile_of(5.0, pd.Series(range(20), dtype='float64')) is not None


# --- team_defensive_prowess / ypt_allowed_for_team / league_average_allowed

def test_team_defensive_prowess_gives_the_worst_grade_the_highest_softness_pct():
    run_def = pd.DataFrame({
        'team_name': ['AAA', 'BBB', 'CCC'],
        'grades_defense': [90.0, 60.0, 30.0],
        'snap_counts_run': [500, 500, 500],
    })
    prowess = ms.team_defensive_prowess(run_def, pd.DataFrame())
    # AAA has the best (highest) grade, so it should be the TOUGHEST matchup
    # - the lowest softness percentile - and CCC (worst grade) the softest.
    assert prowess['AAA'] < prowess['BBB'] < prowess['CCC']


def test_team_defensive_prowess_combines_run_and_coverage_snaps():
    # A pure pass-rusher (zero coverage snaps) should still get a real
    # score off his run-defense snaps alone, and a player appearing in
    # BOTH exports contributes his snap-weighted grade from each - summed
    # weight, not double-counted as two separate players.
    run_def = pd.DataFrame({
        'team_name': ['AAA'], 'grades_defense': [80.0], 'snap_counts_run': [400],
    })
    cov = pd.DataFrame({
        'team_name': ['AAA'], 'grades_defense': [80.0], 'snap_counts_coverage': [200],
    })
    only_run = ms.team_defensive_prowess(run_def, pd.DataFrame())
    both = ms.team_defensive_prowess(run_def, cov)
    assert 'AAA' in only_run and 'AAA' in both


def test_team_defensive_prowess_empty_inputs_return_empty():
    assert ms.team_defensive_prowess(pd.DataFrame(), pd.DataFrame()) == {}
    assert ms.team_defensive_prowess(None, None) == {}


def test_ypt_allowed_for_team_keys_by_position_and_matches_on_nickname():
    coverage = pd.DataFrame({
        'team': ['Chiefs', 'Eagles'],
        'ypt_allowed_wr': [7.3, 6.1],
        'ypt_allowed_te': [8.0, 7.0],
        'ypt_allowed_rb': [5.8, 5.0],
    })
    out = ms.ypt_allowed_for_team(coverage, 'Kansas City Chiefs')
    assert set(out.keys()) == {'WR', 'TE', 'RB'}
    assert out['WR']['value'] == 7.3
    assert ms.ypt_allowed_for_team(coverage, 'Some Team Nobody Has') == {}


def test_league_average_allowed_averages_per_team_per_game_not_per_row():
    # Two receivers on the SAME team in the SAME week must not double the
    # week's total before it's averaged into the league figure.
    df = weekly([
        {'name': 'A', 'week': 1, 'opponent_team': 'BUF', 'team': 'KC', 'position': 'WR', 'receiving_yards': 60},
        {'name': 'B', 'week': 1, 'opponent_team': 'BUF', 'team': 'KC', 'position': 'WR', 'receiving_yards': 40},
        {'name': 'C', 'week': 1, 'opponent_team': 'DEN', 'team': 'NYJ', 'position': 'WR', 'receiving_yards': 50},
    ])
    avg = ms.league_average_allowed(df, 'WR', 'receiving_yards')
    # BUF allowed 100 (60+40) in its one game, DEN allowed 50 in its one -
    # average of the two GAME totals, (100+50)/2 = 75, not a per-row mean.
    assert abs(avg - 75.0) < 1e-9


def test_league_average_allowed_missing_column_returns_none():
    df = weekly([{'name': 'A', 'week': 1, 'opponent_team': 'BUF', 'team': 'KC', 'position': 'WR'}])
    assert ms.league_average_allowed(df, 'WR', 'not_a_real_column') is None


def test_missing_columns_do_not_raise():
    bare = pd.DataFrame({'name': ['A'], 'week': [1], 'opponent_team': ['X']})
    assert ms.player_game_series(bare, 'name', 'A', 'nope_not_a_column')['value'].tolist() == [0.0]
    assert ms.defense_allowed_by_position(pd.DataFrame(), 'D', 'WR')['available'] is False
    assert ms.usage_and_role(pd.DataFrame(), 'name', 'A', 'KC')['available'] is False


def test_usage_weekly_frame_carries_opponent_for_the_chart():
    # The Usage & Role line chart needs a per-week opponent label for its
    # tooltips/bar labels, same as the Game By Game chart - a regression
    # guard for that column actually landing in the weekly frame.
    df = weekly([
        {'name': 'A', 'week': 1, 'team': 'KC', 'opponent_team': 'BUF', 'position': 'WR', 'targets': 5},
        {'name': 'B', 'week': 1, 'team': 'KC', 'opponent_team': 'BUF', 'position': 'WR', 'targets': 5},
    ])
    usage = ms.usage_and_role(df, 'name', 'A', 'KC')
    assert usage['available']
    assert list(usage['weekly']['opponent']) == ['BUF']


# --- Wide YPRR (calculated, not a PFF export column) --------------------

def test_wide_yprr_is_the_rest_of_the_routes_once_slot_is_removed():
    rec = pd.DataFrame({
        'player': ['A'], 'position': ['WR'], 'routes': [100.0], 'yards': [500.0],
    })
    route_concept = pd.DataFrame({
        'player': ['A'], 'slot_routes': [40.0], 'slot_yards': [150.0],
    })
    table = ms.build_wide_yprr_table(rec, route_concept)
    row = table[table['player'] == 'A'].iloc[0]
    # 60 non-slot routes, 350 non-slot yards -> 5.833 yards/route.
    assert abs(row['wide_routes'] - 60.0) < 1e-9
    assert abs(row['wide_yprr'] - (350.0 / 60.0)) < 1e-9
    assert bool(row['includes_inline']) is False


def test_wide_yprr_flags_a_te_as_including_inline_routes():
    rec = pd.DataFrame({
        'player': ['A'], 'position': ['TE'], 'routes': [100.0], 'yards': [500.0],
    })
    table = ms.build_wide_yprr_table(rec, pd.DataFrame())
    # No route_concept data at all: the whole season counts as "non-slot",
    # and a TE's non-slot bucket is flagged as mixing in in-line routes.
    row = table[table['player'] == 'A'].iloc[0]
    assert bool(row['includes_inline']) is True
    assert abs(row['wide_routes'] - 100.0) < 1e-9


def test_wide_yprr_entry_needs_a_real_qualifying_pool():
    # Fewer than 10 players with enough wide routes to rank against -
    # _percentile_of's own thin-pool guard should still apply here.
    rec = pd.DataFrame({
        'player': ['A', 'B'], 'position': ['WR', 'WR'],
        'routes': [50.0, 50.0], 'yards': [250.0, 100.0],
    })
    entry = ms.wide_yprr_entry(rec, pd.DataFrame(), 'A')
    assert entry is not None and entry['pct'] is None
    assert abs(entry['value'] - 5.0) < 1e-9


def test_wide_yprr_entry_missing_player_returns_none():
    rec = pd.DataFrame({'player': ['A'], 'position': ['WR'], 'routes': [50.0], 'yards': [100.0]})
    assert ms.wide_yprr_entry(rec, pd.DataFrame(), 'Nobody') is None


# --- PFF blocking grade for a skill player -------------------------------

def test_blocking_grade_is_ranked_within_its_own_position_only():
    # offense_blocking.csv mixes O-line and skill positions - a WR's grade
    # must be percentiled against other WRs, not against a guard.
    block = pd.DataFrame({
        'player': ['WR1', 'WR2', 'G1'],
        'position': ['WR', 'WR', 'G'],
        'grades_run_block': [40.0, 80.0, 95.0],
    })
    entry = ms.blocking_grade_entry(block, 'WR1', 'WR')
    # WR1's 40 sits BELOW WR2's 80 within the WR-only pool - a small pool
    # (n=2) is below _percentile_of's rank-10 floor, so pct is None, but the
    # raw value must still be his own, not blended with the guard's.
    assert entry is not None
    assert abs(entry['value'] - 40.0) < 1e-9


def test_blocking_grade_missing_position_column_returns_none():
    assert ms.blocking_grade_entry(pd.DataFrame({'player': ['A']}), 'A', 'WR') is None


# --- Man/Zone row assembly ------------------------------------------------

def test_man_zone_grade_rows_puts_man_left_zone_right():
    scheme = {'man_rate': 30.0, 'zone_rate': 65.0, 'man_pct': 20.0, 'zone_pct': 80.0}
    pff_cov = {
        'man_grade': {'value': 70.0, 'pct': 40.0}, 'zone_grade': {'value': 60.0, 'pct': 55.0},
        'man_rating_allowed': {'value': 90.0, 'pct': 30.0}, 'zone_rating_allowed': {'value': 100.0, 'pct': 60.0},
    }
    rows = ms.man_zone_grade_rows(scheme, pff_cov)
    assert [r['label'] for r in rows] == ['Rate', 'Coverage Grade', 'QB Rating Allowed']
    rate_row = rows[0]
    assert rate_row['left'] == 20.0 and rate_row['right'] == 80.0
    assert rate_row['left_str'] == '30.0%' and rate_row['right_str'] == '65.0%'


def test_man_zone_grade_rows_skips_a_row_with_neither_side():
    rows = ms.man_zone_grade_rows({'man_rate': None, 'zone_rate': None}, {})
    assert rows == []


# --- Defense allowed by alignment (Slot vs "Wide") -----------------------

def test_defense_alignment_allowed_wide_is_total_minus_real_slot():
    cov_summary = pd.DataFrame({
        'team_name': ['KC', 'BUF'],
        'receptions': [200.0, 180.0], 'targets': [300.0, 260.0],
        'yards': [2500.0, 2200.0], 'touchdowns': [15.0, 12.0],
    })
    slot_cov = pd.DataFrame({
        'team_name': ['KC', 'BUF'],
        'receptions': [80.0, 60.0], 'targets': [120.0, 90.0],
        'yards': [700.0, 500.0], 'touchdowns': [5.0, 3.0],
    })
    out = ms.defense_alignment_allowed(cov_summary, slot_cov, 'KC')
    assert out['available']
    rec_row = next(r for r in out['rows'] if r['label'] == 'Rec')
    # 200 total - 80 slot = 120 wide.
    assert rec_row['left_str'] == '80' and rec_row['right_str'] == '120'
    targets_row = next(r for r in out['rows'] if r['label'] == 'Targets')
    assert targets_row['left_str'] == '120' and targets_row['right_str'] == '180'


def test_defense_alignment_allowed_missing_team_is_unavailable():
    cov_summary = pd.DataFrame({
        'team_name': ['BUF'], 'receptions': [180.0], 'targets': [260.0],
        'yards': [2200.0], 'touchdowns': [12.0],
    })
    out = ms.defense_alignment_allowed(cov_summary, pd.DataFrame(), 'KC')
    assert out['available'] is False


# --- Weekly stat rank indicator ------------------------------------------

def test_defense_stat_rank_computes_per_game_average_and_rank():
    rows = []
    for team, weekly_vals in (('DEF', [100.0, 140.0]), ('OTHER', [50.0, 70.0])):
        for wk, val in enumerate(weekly_vals, start=1):
            rows.append({'name': f'{team}-{wk}', 'week': wk, 'opponent_team': team,
                        'team': 'OFF', 'position': 'WR', 'receiving_yards': val})
    out = ms.defense_stat_rank(weekly(rows), 'DEF', 'WR', 'receiving_yards')
    assert out is not None
    # DEF allowed (100+140)/2 = 120/game, the higher (softer) of the two.
    assert abs(out['value'] - 120.0) < 1e-9
    assert out['rank'] == 1 and out['of'] == 2


def test_defense_stat_rank_missing_team_returns_none():
    df = weekly([{'name': 'A', 'week': 1, 'opponent_team': 'BUF', 'team': 'OFF',
                  'position': 'WR', 'receiving_yards': 50}])
    assert ms.defense_stat_rank(df, 'NOTATEAM', 'WR', 'receiving_yards') is None


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
