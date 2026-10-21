"""v2_no_history_exclusions (candidate 2026-10-21, calibration audit C15): an on/off check of the partial-game history filter."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.weekly_projections as wp  # noqa: E402


def test_flag_is_registered_and_not_default():
    assert 'v2_no_history_exclusions' in wp.MODEL_FEATURES
    assert 'v2_no_history_exclusions' not in wp.DEFAULT_FEATURES


def test_every_game_becomes_eligible():
    annotated = pd.DataFrame({'player': ['A', 'A', 'B'], '_player_history_eligible': [True, False, False],
                              '_player_history_reason': ['', 'qb split', 'severe blowout rest']})
    out = wp._keep_every_history_game(annotated)
    assert out['_player_history_eligible'].tolist() == [True, True, True]
    assert out['_player_history_reason'].tolist() == ['', '', '']
    assert annotated['_player_history_eligible'].tolist() == [True, False, False]   # input untouched
