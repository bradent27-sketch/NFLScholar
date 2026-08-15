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


def test_missing_columns_do_not_raise():
    bare = pd.DataFrame({'name': ['A'], 'week': [1], 'opponent_team': ['X']})
    assert ms.player_game_series(bare, 'name', 'A', 'nope_not_a_column')['value'].tolist() == [0.0]
    assert ms.defense_allowed_by_position(pd.DataFrame(), 'D', 'WR')['available'] is False
    assert ms.usage_and_role(pd.DataFrame(), 'name', 'A', 'KC')['available'] is False


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
