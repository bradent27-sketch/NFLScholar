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


def test_recent_rate_is_the_mean_of_the_trailing_window_only():
    df = weekly([{'name': 'A', 'week': w, 'team': 'KC', 'position': 'WR', 'targets': v}
                for w, v in zip(range(1, 6), [2, 4, 6, 8, 10])])
    out = wp._recent_rate(df, 'name', 'WR', ['targets'], n=2)
    # Trailing 2 games of a player with 5 games played (weeks 4,5 -> 8,10) -> mean 9.
    assert out.iloc[0]['targets'] == 9.0


# --- in-season rate / shrinkage -------------------------------------------

def test_in_season_rate_blends_recent_and_season_60_40():
    cur_total = np.array([40.0])
    cur_games = np.array([4.0])       # season avg = 10
    recent_avg = np.array([20.0])
    out = wp._in_season_rate(cur_total, cur_games, recent_avg)
    assert abs(out[0] - (0.6 * 20 + 0.4 * 10)) < 1e-9


def test_in_season_rate_falls_back_to_season_avg_when_no_recent_window():
    cur_total = np.array([40.0])
    cur_games = np.array([4.0])
    recent_avg = np.array([np.nan])
    out = wp._in_season_rate(cur_total, cur_games, recent_avg)
    assert abs(out[0] - 10.0) < 1e-9


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
