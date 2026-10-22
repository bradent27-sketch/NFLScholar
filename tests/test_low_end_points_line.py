"""v2_low_end_points_line (candidate 2026-10-22, backlog L7b): below a knot the points line is re-fit on every game; the
shipped v4 line everywhere above it."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.weekly_projections as wp  # noqa: E402


def test_flag_is_registered_and_not_default():
    assert 'v2_low_end_points_line' in wp.MODEL_FEATURES
    assert 'v2_low_end_points_line' not in wp.DEFAULT_FEATURES
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
