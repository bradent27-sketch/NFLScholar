"""The 2026-10-28 calibration audit's flags, shipped that day (scripts/audit_calibration.py, methodology 2026-10-28):
v2_points_lines_start_rows (every points line moved to its start-row fit), v2_luck_te_refit (TE luck -0.16 -> -0.33),
v2_anchor_s_refit (the season anchor's s re-fit on today's model) and v2_luck_qb (the luck step at QB, centered on the
average QB's luck)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.weekly_projections as wp  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402

FLAGS = ('v2_points_lines_start_rows', 'v2_luck_te_refit', 'v2_anchor_s_refit', 'v2_luck_qb')
LUCK_ON = frozenset({'v2_expected_points_luck', 'v2_luck_te_refit', 'v2_luck_qb'})


def test_flags_are_registered_and_the_anchor_and_luck_refits_retired():
    for flag in FLAGS:
        assert flag in wp.MODEL_FEATURES
    # 2026-11-02: the anchor and luck re-fits retired with v2_points_from_stat_line; the start-row lines still ship
    assert 'v2_points_lines_start_rows' in wp.DEFAULT_FEATURES
    assert not ({'v2_luck_te_refit', 'v2_anchor_s_refit', 'v2_luck_qb'} & wp.DEFAULT_FEATURES)
    # the anchor and luck re-fits act after the points line (POST_CALIBRATION_FEATURES); the start-row lines are the line itself,
    # inert in a calibration dump (no 'calibration' there)
    assert {'v2_luck_te_refit', 'v2_anchor_s_refit', 'v2_luck_qb'} <= wp.POST_CALIBRATION_FEATURES
    assert 'v2_points_lines_start_rows' not in wp.POST_CALIBRATION_FEATURES


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


def test_luck_te_refit_factors():
    assert wp.expected_points_luck_factors(set()) == wp.EXPECTED_POINTS_LUCK
    on = wp.expected_points_luck_factors({'v2_luck_te_refit'})
    assert on['TE'] == wp.EXPECTED_POINTS_LUCK_TE_REFIT == -0.33 and on['WR'] == wp.EXPECTED_POINTS_LUCK['WR']
    board = pd.DataFrame({'Player': ['A TE'], 'Pos': ['TE'], 'Team': ['KC'], 'Availability': [1.0],
                          'Model Proj Pts': [10.0], 'Calibrated Model Proj Pts': [10.0]})
    table = pd.DataFrame({'games': [4], 'luck_pg': [3.0]}, index=clean_name_exact(pd.Series(['A TE'])))
    out, _ = wp.apply_expected_points_luck(board, table, factors=on)
    assert out['Model Proj Pts'].iloc[0] == 9.01                     # 10 - 0.33 * 3


def test_anchor_s_refit_factors():
    assert wp.season_anchor_factors(set()) is None                   # apply_season_anchor's default, SEASON_ANCHOR_S
    assert wp.season_anchor_factors({'v2_season_anchor', 'v2_anchor_s_refit'}) == wp.SEASON_ANCHOR_S_REFIT
    assert all(wp.SEASON_ANCHOR_S_REFIT[p] > wp.SEASON_ANCHOR_S[p] for p in wp.SEASON_ANCHOR_S)   # a weaker pull
    assert all(0.0 < s <= 1.0 for s in wp.SEASON_ANCHOR_S_REFIT.values())


def test_qb_luck_is_centered_on_the_average_qb():
    on = wp.expected_points_luck_factors(LUCK_ON)
    assert on['QB'] == wp.EXPECTED_POINTS_LUCK_QB == -0.11 and on['TE'] == -0.33 and on['WR'] == -0.18
    assert 'QB' not in wp.expected_points_luck_factors(LUCK_ON - {'v2_luck_qb'})
    board = pd.DataFrame({'Player': ['A QB', 'B QB', 'C QB', 'A WR'], 'Pos': ['QB', 'QB', 'QB', 'WR'], 'Team': ['KC'] * 4,
                          'Availability': [1.0] * 4, 'Model Proj Pts': [18.0] * 4, 'Calibrated Model Proj Pts': [18.0] * 4})
    table = pd.DataFrame({'games': [5, 5, 5, 5], 'luck_pg': [2.5, -0.5, -4.5, 3.0]},
                         index=clean_name_exact(pd.Series(['A QB', 'B QB', 'C QB', 'A WR'])))
    out, info = wp.apply_expected_points_luck(board, table, factors=on)
    pts = out.set_index('Player')['Model Proj Pts']
    assert pts['A QB'] == 17.67                                      # 18 - 0.11 * (2.5 - (-0.5))
    assert pts['B QB'] == 18.0                                       # the average QB's luck: no move
    assert pts['C QB'] == 18.44                                      # 18 - 0.11 * (-4.5 + 0.5)
    assert pts['A WR'] == 17.46                                      # WR is not centered: 18 - 0.18 * 3
    assert info[('A QB', 'QB', 'KC')]['center'] == -0.5 and info[('A WR', 'WR', 'KC')]['center'] == 0.0
