"""v2_points_from_stat_line: the points total is the displayed stat line scored (times the Questionable cut)."""
import pandas as pd

import data.weekly_projections as wp


def test_flag_is_a_candidate():
    assert 'v2_points_from_stat_line' in wp.MODEL_FEATURES and 'v2_points_from_stat_line' not in wp.DEFAULT_FEATURES


def test_total_is_the_stat_line_scored_with_the_questionable_factor():
    board = pd.DataFrame({'Player': ['A', 'B'], 'Pos': ['WR', 'RB'], 'Team': ['X', 'Y'], 'Model Proj Pts': [14.0, 9.0],
                          'targets': [8.0, 2.0], 'receptions': [5.0, 1.5], 'receiving_yards': [60.0, 10.0],
                          'receiving_tds': [0.4, 0.05], 'rushing_attempts': [0.0, 12.0], 'rushing_yards': [0.0, 50.0],
                          'rushing_tds': [0.0, 0.3]})
    q = {('A', 'WR', 'X'): {'factor': 0.83, 'before': 14.0, 'after': 11.6}}
    out, info = wp.apply_points_from_stat_line(board, 'Full PPR', q)
    a = 5.0 + 6.0 + 0.4 * 6
    b = 1.5 + 1.0 + 0.05 * 6 + 5.0 + 0.3 * 6
    assert abs(out.loc[out.Player == 'A', 'Model Proj Pts'].iloc[0] - round(round(a, 1) * 0.83, 2)) < 1e-9
    assert abs(out.loc[out.Player == 'B', 'Model Proj Pts'].iloc[0] - round(b, 1)) < 1e-9
    assert info[('B', 'RB', 'Y')]['before'] == 9.0
    assert board['Model Proj Pts'].tolist() == [14.0, 9.0]          # the input is not changed
