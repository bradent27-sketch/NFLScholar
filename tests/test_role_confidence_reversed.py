"""v2_role_confidence_reversed_wrte (candidate 2026-10-20, calibration audit C10): for WR/TE receiving volume an
established role leans more on last season and an uncertain one trusts this season sooner."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.weekly_projections as wp  # noqa: E402


def test_flag_is_registered_and_not_default():
    assert 'v2_role_confidence_reversed_wrte' in wp.MODEL_FEATURES
    assert 'v2_role_confidence_reversed_wrte' not in wp.DEFAULT_FEATURES
    assert wp.ROLE_CONFIDENCE_REVERSED_POSITIONS == {'WR', 'TE'}
    assert wp.ROLE_CONFIDENCE_REVERSED_STATS == {'targets', 'receptions', 'receiving_yards'}


def test_reversal_mirrors_the_range():
    games = np.array([4.0, 4.0, 4.0])
    conf = np.array([1.0, 0.5, 0.0])
    lo, hi = wp.K_EFFECTIVE_RANGE
    k = wp.STAT_K['receptions']
    shipped = wp._current_blend_weight(games, 'receptions', conf, pos='WR')
    mirrored = wp._current_blend_weight(games, 'receptions', conf, pos='WR', reverse_confidence=True)
    assert np.allclose(shipped, games / (games + k * np.array([lo, (lo + hi) / 2, hi])))
    assert np.allclose(mirrored, games / (games + k * np.array([hi, (lo + hi) / 2, lo])))
    assert mirrored[0] < shipped[0] and mirrored[2] > shipped[2] and np.isclose(mirrored[1], shipped[1])


def test_blended_rate_passes_the_switch_through():
    cur, prior = np.array([10.0]), np.array([6.0])
    a = wp._blended_rate(cur, np.array([4.0]), prior, prior, 'targets', np.array([1.0]), pos='WR')
    b = wp._blended_rate(cur, np.array([4.0]), prior, prior, 'targets', np.array([1.0]), pos='WR', reverse_confidence=True)
    assert b < a                                     # confident role: less weight on this season's higher rate
