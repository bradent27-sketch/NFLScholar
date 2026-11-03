"""The 2026-10-28 calibration audit's flag that still ships (scripts/audit_calibration.py, methodology 2026-10-28):
v2_points_lines_start_rows (every points line moved to its start-row fit). Its siblings v2_luck_te_refit, v2_anchor_s_refit and
v2_luck_qb were retired 2026-11-02 and removed from the model 2026-11-03: their tests live in test_retired_points_steps.py."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.weekly_projections as wp  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402



def test_start_row_lines_ship_and_the_anchor_and_luck_refits_are_gone():
    assert 'v2_points_lines_start_rows' in wp.MODEL_FEATURES and 'v2_points_lines_start_rows' in wp.DEFAULT_FEATURES
    # the line itself, inert in a calibration dump (no 'calibration' there)
    assert 'v2_points_lines_start_rows' not in wp.POST_CALIBRATION_FEATURES
    for flag in ('v2_luck_te_refit', 'v2_anchor_s_refit', 'v2_luck_qb'):
        assert flag not in wp.MODEL_FEATURES


def test_start_row_lines_move_every_line_by_the_delta():
    scale = wp.points_scale({'v2_points_lines_start_rows'})
    for pos in ('QB', 'RB'):
        sl, ic = wp.WEEKLY_CALIBRATION[pos]
        d_sl, d_ic = wp.START_ROW_LINE_DELTA['all'][pos]
        for week in (2, 10):
            assert np.allclose(wp._weekly_calibration_for(pos, week, scale), (sl + d_sl, ic + d_ic))
    for pos in ('WR', 'TE'):
        for week, bucket in ((2, 'cold'), (10, 'rest')):
            sl, ic = wp.WEEKLY_CALIBRATION_BY_BUCKET[bucket][pos]
            d_sl, d_ic = wp.START_ROW_LINE_DELTA[bucket][pos]
            assert np.allclose(wp._weekly_calibration_for(pos, week, scale), (sl + d_sl, ic + d_ic))
    assert wp.WEEKLY_CALIBRATION['QB'] == (0.738, 4.154)              # the shipped tables are not modified


def test_start_row_lines_stack_on_the_recency_delta():
    both = wp.points_scale({'v2_recency_calibration_wrte', 'v2_points_lines_start_rows'})
    sl, ic = wp.WEEKLY_CALIBRATION_BY_BUCKET['rest']['TE']
    r_sl, r_ic = wp.RECENCY_CALIBRATION_DELTA['rest']['TE']
    s_sl, s_ic = wp.START_ROW_LINE_DELTA['rest']['TE']
    assert np.allclose(wp._weekly_calibration_for('TE', 10, both), (sl + r_sl + s_sl, ic + r_ic + s_ic))
    assert 'anchor_s' not in both and 'low_end' not in both


def test_start_row_lines_shrink_the_early_season_top_and_keep_the_low_end_joined():
    on = wp.points_scale(wp.DEFAULT_FEATURES)
    off = wp.points_scale(wp.DEFAULT_FEATURES - {'v2_points_lines_start_rows'})
    for pos in ('WR', 'TE'):
        assert float(wp.points_line(pos, 3, 20.0, scale=on)) < float(wp.points_line(pos, 3, 20.0, scale=off))
    raw = np.array([1.0, 3.0, 8.0, 20.0])
    sl, ic = wp._weekly_calibration_for('WR', 3, on)
    knot, a = wp.LOW_END_POINTS_LINE['WR']
    expected = np.where(raw < knot, ic + sl * knot + a * (raw - knot), ic + sl * raw)
    assert np.allclose(wp.points_line('WR', 3, raw, low_end=True, scale=on), expected)
