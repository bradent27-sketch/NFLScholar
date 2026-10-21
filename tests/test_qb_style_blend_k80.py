"""v2_qb_style_blend_k80 (candidate 2026-10-21, calibration audit C13 follow-up): a QB1's own pass/rush split shrinks
toward the league mean with K = 80 dropbacks instead of 10."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.qb_volume_blend as qvb  # noqa: E402
import data.weekly_projections as wp  # noqa: E402


def test_flag_is_registered_and_not_default():
    assert 'v2_qb_style_blend_k80' in wp.MODEL_FEATURES
    assert 'v2_qb_style_blend_k80' not in wp.DEFAULT_FEATURES


def test_candidate_constant():
    assert qvb.QB_STYLE_BLEND_K == 10.0
    assert qvb.QB_STYLE_BLEND_K_CANDIDATE == 80.0
    dropbacks = 600.0
    assert round(dropbacks / (dropbacks + qvb.QB_STYLE_BLEND_K_CANDIDATE), 2) == 0.88
