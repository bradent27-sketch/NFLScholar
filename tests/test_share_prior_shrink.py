"""v2_share_prior_shrink (2026-10-26): the in-season expected snap share pulled toward last season's share."""
import numpy as np
import pandas as pd

from data.weekly_projections import (DEFAULT_FEATURES, MODEL_FEATURES, SHARE_PRIOR_SHRINK_K,
                                     expected_snap_share, expected_snap_share_appearances, shrink_share_to_prior)


def _hist():
    return pd.DataFrame({
        'player_display_name': ['A'] * 6 + ['B'],
        'recent_team': ['CLE'] * 7,
        'week': [1, 2, 3, 4, 5, 6, 1],
        'weekly_snap_pct': [50, 60, 70, 80, 90, 100, 0],
    })


def test_appearances_count_the_window():
    n = expected_snap_share_appearances(_hist(), 'player_display_name', as_of_week=7)
    assert n['A'] == 4.0 and n['B'] == 1.0
    n3 = expected_snap_share_appearances(_hist(), 'player_display_name', as_of_week=3)
    assert n3['A'] == 2.0


def test_shrink_one_game_toward_prior():
    share = expected_snap_share(_hist(), 'player_display_name', 'recent_team', as_of_week=7)
    assert share['B'] == 0.0
    out, applied = shrink_share_to_prior([0.0], [0.265], [1.0], 'RB')
    k = SHARE_PRIOR_SHRINK_K['RB']
    assert applied[0] and np.isclose(out[0], k * 0.265 / (1 + k))


def test_shrink_is_symmetric_and_fades_with_games():
    hi, _ = shrink_share_to_prior([0.70], [0.30], [1.0], 'WR')
    assert 0.30 < hi[0] < 0.70
    one, _ = shrink_share_to_prior([0.2], [0.8], [1.0], 'WR')
    four, _ = shrink_share_to_prior([0.2], [0.8], [4.0], 'WR')
    assert abs(four[0] - 0.2) < abs(one[0] - 0.2)


def test_rows_left_alone():
    out, applied = shrink_share_to_prior([0.4, 0.4, 0.4], [np.nan, 0.8, 0.8], [2.0, 0.0, np.nan], 'TE')
    assert not applied.any() and np.allclose(out, 0.4)
    out, applied = shrink_share_to_prior([0.4], [0.8], [2.0], 'QB')
    assert not applied.any() and out[0] == 0.4


def test_flag_is_a_candidate():
    assert 'v2_share_prior_shrink' in MODEL_FEATURES
    assert 'v2_share_prior_shrink' not in DEFAULT_FEATURES


def test_recency_weights_the_latest_appearance():
    h = _hist()
    flat = expected_snap_share(h, 'player_display_name', 'recent_team', as_of_week=7)
    recent = expected_snap_share(h, 'player_display_name', 'recent_team', as_of_week=7, last_weight=2.0)
    # A's last four: .7 .8 .9 1.0 -> flat .85, latest x2 -> (0.7 + 0.8 + 0.9 + 2.0) / 5 = 0.88
    assert np.isclose(flat['A'], 0.85) and np.isclose(recent['A'], 0.88)
    assert recent['B'] == 0.0
    assert 'v2_share_recency' in MODEL_FEATURES and 'v2_share_recency' not in DEFAULT_FEATURES
