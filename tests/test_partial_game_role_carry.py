"""v2_partial_game_role_carry (2026-10-26): an excluded abrupt-partial game still lowers the player's projected role."""
import numpy as np
import pandas as pd

from data.weekly_projections import (DEFAULT_FEATURES, MODEL_FEATURES, PARTIAL_ROLE_CARRY_MIN, PARTIAL_ROLE_CARRY_REASON,
                                     partial_game_role_multiplier)


def _annotated(snaps, reasons, player='A'):
    n = len(snaps)
    return pd.DataFrame({
        'player_display_name': [player] * n, 'recent_team': ['NE'] * n, 'week': list(range(1, n + 1)),
        'weekly_snap_pct': snaps,
        '_player_history_eligible': [r == '' for r in reasons],
        '_player_history_reason': reasons,
    })


def test_partial_game_lowers_the_role():
    h = _annotated([80, 80, 80, 20], ['', '', '', PARTIAL_ROLE_CARRY_REASON])
    m = partial_game_role_multiplier(h, 'player_display_name', 'recent_team', as_of_week=5)
    # with the game: mean(.8, .8, .8, .2) = .65; without: .8
    assert np.isclose(m['A'], 0.65 / 0.80)


def test_floor_and_untouched_players():
    h = pd.concat([_annotated([90, 90, 2], ['', '', PARTIAL_ROLE_CARRY_REASON]),
                   _annotated([60, 60, 60], ['', '', ''], player='B')], ignore_index=True)
    m = partial_game_role_multiplier(h, 'player_display_name', 'recent_team', as_of_week=4)
    assert m['A'] >= PARTIAL_ROLE_CARRY_MIN and 'B' not in m.index


def test_other_exclusions_are_not_counted():
    h = _annotated([80, 80, 80, 20], ['', '', '', 'severe blowout rest'])
    m = partial_game_role_multiplier(h, 'player_display_name', 'recent_team', as_of_week=5)
    assert m.empty


def test_game_rolls_out_of_the_window():
    h = _annotated([80, 80, 80, 20, 80, 80, 80, 80], ['', '', '', PARTIAL_ROLE_CARRY_REASON, '', '', '', ''])
    m = partial_game_role_multiplier(h, 'player_display_name', 'recent_team', as_of_week=9)
    assert m.empty


def test_flag_is_a_candidate():
    assert 'v2_partial_game_role_carry' in MODEL_FEATURES
    assert 'v2_partial_game_role_carry' not in DEFAULT_FEATURES
