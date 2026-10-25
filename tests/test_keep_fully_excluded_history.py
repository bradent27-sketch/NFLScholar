"""v2_keep_fully_excluded_history (candidate 2026-10-25): a RB/WR/TE whose every game this season was excluded keeps them."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.weekly_projections as wp  # noqa: E402


def _annotated():
    return pd.DataFrame({
        'player': ['Rookie WR', 'Rookie WR', 'Starter WR', 'Starter WR', 'Relief QB', 'Fill-in TE'],
        'position': ['WR', 'WR', 'WR', 'WR', 'QB', 'TE'],
        'week': [1.0, 6.0, 1.0, 2.0, 3.0, 4.0],
        '_player_history_eligible': [False, False, True, False, False, False],
        '_player_history_reason': ['partial replacement after teammate exit'] * 2
                                  + ['', 'abrupt partial role after established workload',
                                     'partial replacement after teammate exit', 'partial replacement after teammate exit'],
    })


def test_flag_is_registered_and_not_default():
    assert 'v2_keep_fully_excluded_history' in wp.MODEL_FEATURES
    assert 'v2_keep_fully_excluded_history' not in wp.DEFAULT_FEATURES


def test_only_players_with_every_game_excluded_keep_them():
    annotated = _annotated()
    out = wp._keep_fully_excluded_players(annotated, 'player')
    # the rookie and the fill-in TE had nothing left: both keep every game; the starter keeps his own exclusion; the QB is left
    assert out['_player_history_eligible'].tolist() == [True, True, True, False, False, True]
    assert out['_player_history_reason'].tolist()[:2] == ['', '']
    assert out['_player_history_reason'].iloc[3] == 'abrupt partial role after established workload'
    assert annotated['_player_history_eligible'].tolist() == [False, False, True, False, False, False]   # input untouched


def test_nothing_to_keep_is_unchanged():
    annotated = _annotated().iloc[2:4]
    out = wp._keep_fully_excluded_players(annotated, 'player')
    pd.testing.assert_frame_equal(out, annotated)


# v2_keep_fully_excluded_no_prior (candidate 2026-10-25, the revision after the first flag's harness): only a player with no
# eligible prior-season game at his position keeps his excluded games; a veteran stays on his prior-season line.

def test_no_prior_flag_is_registered_and_not_default():
    assert 'v2_keep_fully_excluded_no_prior' in wp.MODEL_FEATURES
    assert 'v2_keep_fully_excluded_no_prior' not in wp.DEFAULT_FEATURES


def test_with_a_prior_season_role_the_excluded_games_stay_excluded():
    annotated = _annotated()
    prior = pd.DataFrame({'player': ['Fill-in TE', 'Fill-in TE', 'Rookie WR'], 'position': ['TE', 'TE', 'TE'],
                          'week': [1.0, 2.0, 3.0]})
    out = wp._keep_fully_excluded_players(annotated, 'player', prior, 'player')
    # the fill-in TE had TE games last season: his cut-short game stays out; the rookie's prior row is at another position
    assert out['_player_history_eligible'].tolist() == [True, True, True, False, False, False]
    assert out['_player_history_reason'].iloc[5] == 'partial replacement after teammate exit'


def test_an_empty_prior_season_keeps_every_fully_excluded_player():
    annotated = _annotated()
    out = wp._keep_fully_excluded_players(annotated, 'player', pd.DataFrame(), 'player')
    pd.testing.assert_frame_equal(out, wp._keep_fully_excluded_players(annotated, 'player'))
