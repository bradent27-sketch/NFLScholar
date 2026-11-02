"""v2_points_from_stat_line: the points total is the displayed stat line scored (times the Questionable cut)."""
import pandas as pd

import data.weekly_projections as wp


def test_flag_ships_with_the_anchor_and_luck_retired():
    assert 'v2_points_from_stat_line' in wp.MODEL_FEATURES and 'v2_points_from_stat_line' in wp.DEFAULT_FEATURES
    assert not ({'v2_season_anchor', 'v2_anchor_s_refit', 'v2_expected_points_luck', 'v2_luck_te_refit', 'v2_luck_qb'} & wp.DEFAULT_FEATURES)
    assert 'v2_points_from_stat_line' in wp.POST_CALIBRATION_FEATURES


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


def test_edges_flag_ships_for_rb_and_wr_and_leaves_qb_and_te_alone():
    assert 'v2_points_from_stat_line_edges' in wp.MODEL_FEATURES
    assert 'v2_points_from_stat_line_edges' in wp.DEFAULT_FEATURES
    assert wp.POINTS_FROM_STAT_LINE_EDGE_POSITIONS == ('RB', 'WR')
    board = pd.DataFrame({'Player': ['Q', 'R', 'T'], 'Pos': ['QB', 'RB', 'TE'], 'Team': ['X', 'Y', 'Y'],
                          'Model Proj Pts': [18.0, 9.0, 7.0],
                          'passing_yards': [250.0, 0.0, 0.0], 'passing_tds': [1.8, 0.0, 0.0], 'passing_interceptions': [0.8, 0.0, 0.0],
                          'rushing_yards': [20.0, 50.0, 0.0], 'rushing_tds': [0.2, 0.3, 0.0], 'receptions': [0.0, 1.5, 4.0],
                          'receiving_yards': [0.0, 10.0, 40.0], 'receiving_tds': [0.0, 0.05, 0.3]})
    out, info = wp.apply_points_from_stat_line(board, 'Full PPR', positions=wp.POINTS_FROM_STAT_LINE_EDGE_POSITIONS)
    assert out.loc[out.Player == 'Q', 'Model Proj Pts'].iloc[0] == 18.0       # QB keeps the points line
    assert out.loc[out.Player == 'T', 'Model Proj Pts'].iloc[0] == 7.0        # so does the TE
    assert abs(out.loc[out.Player == 'R', 'Model Proj Pts'].iloc[0] - round(5.0 + 1.8 + 1.5 + 1.0 + 0.3, 1)) < 1e-9
    assert set(info) == {('R', 'RB', 'Y')}


def test_points_path_shows_the_stat_line_step_last(monkeypatch):
    import contextlib
    import ui.components as components
    import ui.tabs.rankings as rk

    class _FakeSt:
        def __init__(self):
            self.calls = []
            self.session_state = {}

        def markdown(self, text, *a, **k):
            self.calls.append(('markdown', text))

        def caption(self, text, *a, **k):
            self.calls.append(('caption', text))

        def dataframe(self, frame, *a, **k):
            self.calls.append(('dataframe', frame))

        def expander(self, label, expanded=False, **kwargs):
            self.calls.append(('expander', label))
            return contextlib.nullcontext()

    fake = _FakeSt()
    monkeypatch.setattr(rk, 'st', fake)
    monkeypatch.setattr(components, 'st', fake)        # quiet_note (the dropdowns) draws through ui.components
    rk._render_points_path({
        'raw_points': 15.0, 'calibrated_points': 14.5, 'target_week': 6,
        'calibration': {'enabled': True, 'slope': 0.9, 'intercept': 1.0, 'raw_points': 15.0, 'displayed_points': 14.5},
        'points_from_stat_line': {'before': 14.5, 'after': 13.2},
    })
    styled = next(c[1] for c in fake.calls if c[0] == 'dataframe')
    frame = getattr(styled, 'data', styled)
    assert list(frame['Stage']) == ['Raw model', 'Calibrated', 'Stat line scored']
    assert frame['Points'].iloc[-1] == '13.20' and frame['Change'].iloc[-1] == '-1.30'
    assert len(frame['Basis'].iloc[-1]) < 100                                    # fits the cell
    dropdown = ' '.join(str(c[1]) for c in fake.calls if c[0] == 'markdown')
    assert 'points and stats agree' in dropdown
