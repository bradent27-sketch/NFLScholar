"""v2_questionable_rb (candidate, 2026-11-11): a Questionable RB who is projected to play has his rushing line cut
(QUESTIONABLE_RB_STAT_FACTORS) and is scored from that line; his receiving line, his availability and every other row stay."""
import pandas as pd

import data.weekly_projections as wp


def _board():
    return pd.DataFrame({
        'Player': ['Q Back', 'Healthy Back', 'Q Wideout', 'Out Back'], 'Pos': ['RB', 'RB', 'WR', 'RB'],
        'Team': ['X', 'X', 'Y', 'Z'], 'Availability': [1.0, 1.0, 1.0, 0.0], 'Model Proj Pts': [14.0, 14.0, 12.0, 0.0],
        'rushing_attempts': [15.0, 15.0, 0.5, 0.0], 'rushing_yards': [65.0, 65.0, 3.0, 0.0], 'rushing_tds': [0.5, 0.5, 0.0, 0.0],
        'targets': [3.0, 3.0, 7.0, 0.0], 'receptions': [2.5, 2.5, 4.5, 0.0], 'receiving_yards': [18.0, 18.0, 55.0, 0.0],
        'receiving_tds': [0.1, 0.1, 0.35, 0.0]})


PROFILES = {'Q Back': {'status': 'questionable'}, 'Q Wideout': {'status': 'questionable'}, 'Out Back': {'status': 'questionable'}}


def test_flag_ships_after_calibration_with_the_flags_it_needs():
    assert 'v2_questionable_rb' in wp.MODEL_FEATURES and 'v2_questionable_rb' in wp.DEFAULT_FEATURES
    assert 'v2_points_from_stat_line' in wp.DEFAULT_FEATURES
    assert 'v2_questionable_rb' in wp.POST_CALIBRATION_FEATURES


def test_questionable_rows_takes_the_eligible_backs_only():
    rows = wp.questionable_rows(_board(), PROFILES, wp.QUESTIONABLE_RB_STAT_FACTORS)
    assert set(rows) == {('Q Back', 'RB', 'X')}          # not the WR, not the healthy back, not the sidelined one
    assert wp.questionable_rows(_board(), {}, wp.QUESTIONABLE_RB_STAT_FACTORS) == {}


def test_rushing_line_is_cut_and_receiving_left_alone():
    board = _board()
    rows = wp.questionable_rows(board, PROFILES, wp.QUESTIONABLE_RB_STAT_FACTORS)
    out, info = wp.apply_questionable_stat_cut(board, rows, factors=wp.QUESTIONABLE_RB_STAT_FACTORS)
    f = wp.QUESTIONABLE_RB_STAT_FACTORS['RB']
    q = out.loc[out.Player == 'Q Back'].iloc[0]
    assert abs(q['rushing_attempts'] - 15.0 * f['rushing_attempts']) < 1e-9
    assert abs(q['rushing_yards'] - 65.0 * f['rushing_yards']) < 1e-9
    assert abs(q['rushing_tds'] - 0.5 * f['rushing_tds']) < 1e-9
    assert (q['targets'], q['receptions'], q['receiving_yards'], q['receiving_tds']) == (3.0, 2.5, 18.0, 0.1)
    assert q['Availability'] == 1.0                       # still assumed to play
    h = out.loc[out.Player == 'Healthy Back'].iloc[0]
    assert (h['rushing_attempts'], h['rushing_yards'], h['rushing_tds']) == (15.0, 65.0, 0.5)
    assert set(info) == {('Q Back', 'RB', 'X')} and set(info[('Q Back', 'RB', 'X')]) == set(f)


def test_points_score_the_cut_rushing_line_once():
    board = _board()
    rows = wp.questionable_rows(board, PROFILES, wp.QUESTIONABLE_RB_STAT_FACTORS)
    out, _ = wp.apply_questionable_stat_cut(board, rows, factors=wp.QUESTIONABLE_RB_STAT_FACTORS)
    scored, _ = wp.apply_points_from_stat_line(out, 'Full PPR', {})
    q = scored.loc[scored.Player == 'Q Back'].iloc[0]
    h = scored.loc[scored.Player == 'Healthy Back'].iloc[0]
    expected = wp.score_projected_stats(out.loc[out.Player == 'Q Back'].iloc[0].to_dict(), 'Full PPR')
    assert abs(q['Model Proj Pts'] - expected) < 0.06
    assert 0.85 < q['Model Proj Pts'] / h['Model Proj Pts'] < 0.97     # the receiving share is not cut


def test_workload_cut_unchanged_by_the_shared_eligibility():
    out, info = wp.apply_questionable_workload(_board(), PROFILES)
    assert set(info) == {('Q Wideout', 'WR', 'Y')}
    assert abs(out.loc[out.Player == 'Q Wideout', 'Model Proj Pts'].iloc[0] - round(12.0 * 0.83, 2)) < 1e-9
    assert out.loc[out.Player == 'Q Back', 'Model Proj Pts'].iloc[0] == 14.0
