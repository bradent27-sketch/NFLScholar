"""v2_recency_calibration_wrte (shipped 2026-10-27, backlog M11): the WR/TE points lines weighted toward recent seasons - the
shipped lines plus RECENCY_CALIBRATION_DELTA; QB/RB, the low end and the season anchor unchanged."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.weekly_projections as wp  # noqa: E402

FLAG = 'v2_recency_calibration_wrte'


def test_flag_is_registered_and_shipped():
    assert FLAG in wp.MODEL_FEATURES and FLAG in wp.DEFAULT_FEATURES
    assert FLAG not in wp.POST_CALIBRATION_FEATURES                   # part of the line itself


def test_off_is_the_shipped_tables():
    assert wp.points_scale(set()) is None
    assert wp.points_scale(wp.DEFAULT_FEATURES - {FLAG, 'v2_points_lines_start_rows'}) is None   # the other line delta


def test_on_moves_only_the_wr_te_season_phase_lines():
    scale = wp.points_scale({FLAG})
    for pos in ('QB', 'RB'):
        for week in (2, 10):
            assert wp._weekly_calibration_for(pos, week, scale) == wp._weekly_calibration_for(pos, week)
    for pos in ('WR', 'TE'):
        for week, bucket in ((2, 'cold'), (10, 'rest')):
            sl, ic = wp._weekly_calibration_for(pos, week)
            d_sl, d_ic = wp.RECENCY_CALIBRATION_DELTA[bucket][pos]
            assert np.allclose(wp._weekly_calibration_for(pos, week, scale), (sl + d_sl, ic + d_ic))
    # the low end stays the shipped one, joined to the moved line at its knot; the anchor keeps the shipped s
    raw = np.array([1.0, 3.0, 8.0, 20.0])
    sl, ic = wp._weekly_calibration_for('WR', 10, scale)
    knot, a = wp.LOW_END_POINTS_LINE['WR']
    expected = np.where(raw < knot, ic + sl * knot + a * (raw - knot), ic + sl * raw)
    assert np.allclose(wp.points_line('WR', 10, raw, low_end=True, scale=scale), expected)
    assert 'anchor_s' not in scale
    # the shipped tables themselves are not modified
    assert wp.WEEKLY_CALIBRATION_BY_BUCKET['rest']['TE'] == (0.981, 0.790)


def test_top_wr_te_rest_of_season_rise_a_little():
    scale = wp.points_scale({FLAG})
    for pos in ('WR', 'TE'):
        shipped = float(wp.points_line(pos, 10, 22.0))
        on = float(wp.points_line(pos, 10, 22.0, scale=scale))
        assert 0.0 < on - shipped < 0.5


def test_a_fit_v6_scale_wins():
    original = wp.POINTS_SCALE_V6
    try:
        wp.POINTS_SCALE_V6 = {'lines': {'WR': (0.9, 1.0)}, 'by_bucket': {}}
        assert wp.points_scale({'v2_points_scale_v6', FLAG}) is wp.POINTS_SCALE_V6
    finally:
        wp.POINTS_SCALE_V6 = original
