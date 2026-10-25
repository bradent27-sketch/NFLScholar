"""v2_low_end_points_line (shipped 2026-10-22, backlog L7b): below a knot the points line is re-fit on every game; the
shipped v4 line everywhere above it."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.weekly_projections as wp  # noqa: E402


def test_flag_is_registered_and_shipped():
    assert 'v2_low_end_points_line' in wp.MODEL_FEATURES
    assert 'v2_low_end_points_line' in wp.DEFAULT_FEATURES          # shipped 2026-10-22
    assert set(wp.LOW_END_POINTS_LINE) == {'RB', 'WR', 'TE'}


def test_off_is_the_shipped_line_and_on_only_moves_the_low_end():
    raw = np.array([0.0, 2.0, 5.9, 6.0, 9.0, 20.0])
    for week in (3, 10):
        for pos in ('QB', 'RB', 'WR', 'TE'):
            sl, ic = wp._weekly_calibration_for(pos, week)
            assert np.allclose(wp.points_line(pos, week, raw), ic + sl * raw)
        sl, ic = wp._weekly_calibration_for('WR', week)
        on = wp.points_line('WR', week, raw, low_end=True)
        knot, a = wp.LOW_END_POINTS_LINE['WR']
        assert np.allclose(on[raw >= knot], (ic + sl * raw)[raw >= knot])                  # unchanged at and above the knot
        assert np.isclose(on[1], ic + sl * knot + a * (2.0 - knot))
        assert on[1] < ic + sl * 2.0                                                         # the low end comes down
        assert np.allclose(wp.points_line('QB', week, raw, low_end=True), wp.points_line('QB', week, raw))


def test_continuous_at_the_knot():
    for pos, (knot, _a) in wp.LOW_END_POINTS_LINE.items():
        just_below = float(wp.points_line(pos, 10, knot - 1e-9, low_end=True))
        at = float(wp.points_line(pos, 10, knot, low_end=True))
        assert abs(just_below - at) < 1e-6


def test_a_refit_scale_replaces_every_table_only_behind_its_flag():
    # v2_points_scale_v6 (2026-10-25): the lines, the buckets, the low end and the anchor's s come from one re-fit dict
    scale = {'lines': {'QB': (0.9, 1.0), 'RB': (0.95, 0.5), 'WR': (0.97, 0.4), 'TE': (0.98, 0.3)},
             'by_bucket': {'cold': {'WR': (0.9, 0.8)}, 'rest': {'WR': (0.99, 0.2)}},
             'low_end': {'WR': (5.0, 1.3)}, 'anchor_s': {'WR': 0.7}}
    assert wp._weekly_calibration_for('RB', 10, scale) == (0.95, 0.5)
    assert wp._weekly_calibration_for('WR', 2, scale) == (0.9, 0.8) and wp._weekly_calibration_for('WR', 10, scale) == (0.99, 0.2)
    assert np.isclose(float(wp.points_line('WR', 10, 2.0, low_end=True, scale=scale)), 0.2 + 0.99 * 5.0 + 1.3 * (2.0 - 5.0))
    original = wp.POINTS_SCALE_V6
    try:
        wp.POINTS_SCALE_V6 = scale
        assert wp.points_scale({'v2_points_scale_v6'}) is scale and wp.points_scale(set()) is None
        wp.POINTS_SCALE_V6 = None
        assert wp.points_scale({'v2_points_scale_v6'}) is None          # not fit yet: the shipped tables
    finally:
        wp.POINTS_SCALE_V6 = original
