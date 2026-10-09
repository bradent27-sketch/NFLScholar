"""v2_questionable_workload (2026-10-08): a Questionable WR who is projected to play has his point total scaled by
QUESTIONABLE_WORKLOAD_FACTORS['WR']; nothing else moves, and he is not made less available."""
import contextlib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

from data.weekly_projections import (DEFAULT_FEATURES, MODEL_FEATURES, QUESTIONABLE_WORKLOAD_FACTORS,  # noqa: E402
                                     apply_questionable_workload)


def _board():
    rows = [
        # player, pos, team, availability, points
        ('Q Wideout', 'WR', 'AAA', 1.0, 14.0),
        ('Healthy Wideout', 'WR', 'AAA', 1.0, 14.0),
        ('Doubtful Wideout', 'WR', 'AAA', 0.0, 14.0),
        ('Q Back', 'RB', 'AAA', 1.0, 14.0),
        ('Q Tight End', 'TE', 'AAA', 1.0, 9.0),
        ('Q Passer', 'QB', 'AAA', 1.0, 18.0),
        ('Q Zero Wideout', 'WR', 'AAA', 1.0, 0.0),
    ]
    frame = pd.DataFrame(rows, columns=['Player', 'Pos', 'Team', 'Availability', 'Model Proj Pts'])
    frame['Raw Model Proj Pts'] = frame['Model Proj Pts']
    frame['Calibrated Model Proj Pts'] = frame['Model Proj Pts']
    return frame


def _profiles(**overrides):
    base = {'Q Wideout': {'status': 'questionable'}, 'Healthy Wideout': {'status': 'healthy'},
            'Doubtful Wideout': {'status': 'doubtful'}, 'Q Back': {'status': 'questionable'},
            'Q Tight End': {'status': 'questionable'}, 'Q Passer': {'status': 'questionable'},
            'Q Zero Wideout': {'status': 'questionable'}}
    base.update(overrides)
    return base


def test_flag_is_registered_and_shipped():
    assert 'v2_questionable_workload' in MODEL_FEATURES
    assert 'v2_questionable_workload' in DEFAULT_FEATURES


def test_only_a_questionable_wr_who_is_playing_is_scaled():
    board = _board()
    out, info = apply_questionable_workload(board, _profiles())
    f = QUESTIONABLE_WORKLOAD_FACTORS['WR']
    assert set(info) == {('Q Wideout', 'WR', 'AAA')}
    assert info[('Q Wideout', 'WR', 'AAA')] == {'factor': f, 'before': 14.0, 'after': round(14.0 * f, 2),
                                                'status': 'questionable'}
    scaled = out.set_index('Player')
    assert scaled.loc['Q Wideout', 'Model Proj Pts'] == round(14.0 * f, 2)
    assert scaled.loc['Q Wideout', 'Calibrated Model Proj Pts'] == round(14.0 * f, 2)
    assert scaled.loc['Q Wideout', 'Availability'] == 1.0                       # still counted as playing
    assert scaled.loc['Q Wideout', 'Raw Model Proj Pts'] == 14.0                # the uncalibrated total is kept
    others = ~out['Player'].eq('Q Wideout')
    pd.testing.assert_frame_equal(out[others].reset_index(drop=True), board[others].reset_index(drop=True))


def test_wr_only_other_positions_and_zero_rows_are_left_alone():
    assert set(QUESTIONABLE_WORKLOAD_FACTORS) == {'WR'}
    assert 0.7 < QUESTIONABLE_WORKLOAD_FACTORS['WR'] < 1.0
    out, info = apply_questionable_workload(_board(), _profiles())
    frame = out.set_index('Player')
    for name, pts in (('Q Back', 14.0), ('Q Tight End', 9.0), ('Q Passer', 18.0), ('Q Zero Wideout', 0.0),
                      ('Healthy Wideout', 14.0), ('Doubtful Wideout', 14.0)):
        assert frame.loc[name, 'Model Proj Pts'] == pts


def test_no_profiles_or_empty_board_is_a_no_op():
    board = _board()
    for profiles in (None, {}):
        out, info = apply_questionable_workload(board, profiles)
        assert info == {}
        pd.testing.assert_frame_equal(out, board)
    out, info = apply_questionable_workload(board.iloc[0:0], _profiles())
    assert info == {} and out.empty


def test_a_custom_factor_table_is_honoured_and_status_case_is_ignored():
    board = _board()
    out, info = apply_questionable_workload(board, _profiles(**{'Q Back': {'status': 'Questionable'}}),
                                            factors={'RB': 0.9})
    assert set(info) == {('Q Back', 'RB', 'AAA')}
    assert out.set_index('Player').loc['Q Back', 'Model Proj Pts'] == 12.6


class _FakeSt:
    def __init__(self):
        self.calls = []

    def markdown(self, text, *a, **k):
        self.calls.append(('markdown', text))

    def caption(self, text, *a, **k):
        self.calls.append(('caption', text))

    def dataframe(self, frame, *a, **k):
        self.calls.append(('dataframe', frame))

    def expander(self, label, expanded=False):
        self.calls.append(('expander', label))
        return contextlib.nullcontext()


def test_points_path_lists_the_questionable_step_last(monkeypatch):
    import ui.tabs.rankings as rk
    fake = _FakeSt()
    monkeypatch.setattr(rk, 'st', fake)
    rk._render_points_path({
        'raw_points': 15.0, 'calibrated_points': 10.4, 'target_week': 6,
        'calibration': {'enabled': True, 'slope': 0.9, 'intercept': 1.0, 'raw_points': 15.0, 'displayed_points': 14.5},
        'season_anchor': {'before': 14.5, 'after': 12.5, 'factor': 0.69, 'mean': 9.0, 'games': 5},
        'questionable_workload': {'before': 12.5, 'after': 10.4, 'factor': 0.83, 'status': 'questionable'},
    })
    styled = next(c[1] for c in fake.calls if c[0] == 'dataframe')
    frame = getattr(styled, 'data', styled)
    assert list(frame['Stage']) == ['Raw model', 'Calibrated', 'Season anchor', 'Questionable tag']
    assert frame['Points'].iloc[-1] == '10.40' and frame['Change'].iloc[-1] == '-2.10'
    assert '×0.83' in frame['Basis'].iloc[-1] and 'still counted as playing' in frame['Basis'].iloc[-1]
