"""v2_fallback_prior_fast_trust (candidate 2026-10-21, model audit lead L3): an RB/WR/TE whose prior is the position
fallback (no prior of his own) trusts this season's rate sooner, K x 1/3 on the non-TD stats."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.weekly_projections as wp  # noqa: E402


def test_flag_is_registered_and_not_default():
    assert 'v2_fallback_prior_fast_trust' in wp.MODEL_FEATURES
    assert 'v2_fallback_prior_fast_trust' not in wp.DEFAULT_FEATURES
    assert np.isclose(wp.FALLBACK_PRIOR_K_SCALE, 1.0 / 3.0)
    assert wp.FALLBACK_PRIOR_FAST_TRUST_POSITIONS == ('RB', 'WR', 'TE')


def test_k_scale_speeds_up_trust_only_where_given():
    games = np.array([4.0, 4.0])
    conf = np.array([0.5, 0.5])
    base = wp._current_blend_weight(games, 'targets', conf)
    scaled = wp._current_blend_weight(games, 'targets', conf, k_scale=np.array([1.0, 1.0 / 3.0]))
    assert np.isclose(scaled[0], base[0])
    k = games[1] * (1.0 - base[1]) / base[1]
    assert np.isclose(scaled[1], games[1] / (games[1] + k / 3.0))
    assert scaled[1] > base[1]


def test_blended_rate_moves_toward_this_season():
    cur, games, prior, pos_rate = np.array([7.0]), np.array([5.0]), np.array([np.nan]), np.array([5.0])
    conf = np.array([0.5])
    base = wp._blended_rate(cur, games, prior, pos_rate, 'targets', conf)
    fast = wp._blended_rate(cur, games, prior, pos_rate, 'targets', conf, k_scale=np.array([1.0 / 3.0]))
    assert 5.0 < base[0] < fast[0] < 7.0
