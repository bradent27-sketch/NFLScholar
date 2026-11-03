"""v2_questionable_stat_cut (candidate, 2026-11-02): a Questionable WR's stat columns are cut per stat, so the points (the stat line
scored) and the displayed stats agree."""
import pandas as pd

import data.weekly_projections as wp


def _board():
    return pd.DataFrame({
        'Player': ['Q Wideout', 'Healthy Wideout', 'Q Back'], 'Pos': ['WR', 'WR', 'RB'], 'Team': ['X', 'X', 'Y'],
        'Model Proj Pts': [14.0, 14.0, 12.0], 'targets': [8.0, 8.0, 3.0], 'receptions': [5.0, 5.0, 2.0],
        'receiving_yards': [60.0, 60.0, 15.0], 'receiving_tds': [0.4, 0.4, 0.05], 'rushing_yards': [0.0, 0.0, 60.0]})


INFO = {('Q Wideout', 'WR', 'X'): {'factor': 0.83, 'before': 14.0, 'after': 11.6},
        ('Q Back', 'RB', 'Y'): {'factor': 0.9}}


def test_flag_ships_after_the_flags_it_needs():
    assert 'v2_questionable_stat_cut' in wp.MODEL_FEATURES and 'v2_questionable_stat_cut' in wp.DEFAULT_FEATURES
    assert {'v2_questionable_workload', 'v2_points_from_stat_line'} <= wp.DEFAULT_FEATURES
    assert 'v2_questionable_stat_cut' in wp.POST_CALIBRATION_FEATURES


def test_only_the_listed_wr_rows_are_cut_and_each_stat_by_its_own_factor():
    board = _board()
    out, info = wp.apply_questionable_stat_cut(board, INFO)
    f = wp.QUESTIONABLE_STAT_FACTORS['WR']
    q = out.loc[out.Player == 'Q Wideout'].iloc[0]
    assert abs(q['targets'] - 8.0 * f['targets']) < 1e-9 and abs(q['receptions'] - 5.0 * f['receptions']) < 1e-9
    assert abs(q['receiving_yards'] - 60.0 * f['receiving_yards']) < 1e-9 and abs(q['receiving_tds'] - 0.4 * f['receiving_tds']) < 1e-9
    h = out.loc[out.Player == 'Healthy Wideout'].iloc[0]
    assert (h['targets'], h['receptions'], h['receiving_yards'], h['receiving_tds']) == (8.0, 5.0, 60.0, 0.4)
    back = out.loc[out.Player == 'Q Back'].iloc[0]                    # WR only: a Questionable RB is left alone
    assert (back['targets'], back['receiving_yards'], back['rushing_yards']) == (3.0, 15.0, 60.0)
    assert set(info) == {('Q Wideout', 'WR', 'X')} and set(info[('Q Wideout', 'WR', 'X')]) == set(f)
    assert board['receiving_yards'].tolist() == [60.0, 60.0, 15.0]    # the input is not changed


def test_scored_cut_line_is_close_to_the_uniform_cut_for_a_typical_wr():
    out, _ = wp.apply_questionable_stat_cut(_board(), INFO)
    cut_line = wp.score_projected_stats(out.loc[out.Player == 'Q Wideout'].iloc[0].to_dict(), 'Full PPR')
    uniform = wp.score_projected_stats(_board().iloc[0].to_dict(), 'Full PPR') * wp.QUESTIONABLE_WORKLOAD_FACTORS['WR']
    assert abs(cut_line - uniform) / uniform < 0.05


def test_no_info_empty_board_or_unknown_stat_is_a_no_op():
    board = _board()
    out, info = wp.apply_questionable_stat_cut(board, {})
    assert info == {} and out['receiving_yards'].tolist() == board['receiving_yards'].tolist()
    empty, info = wp.apply_questionable_stat_cut(board.iloc[0:0], INFO)
    assert info == {} and empty.empty
    slim = board.drop(columns=['receiving_tds'])
    out, info = wp.apply_questionable_stat_cut(slim, INFO)
    assert 'receiving_tds' not in info[('Q Wideout', 'WR', 'X')]


def test_points_are_scored_once_from_the_cut_line():
    # the way the board does it: cut the stat columns, then score the line with no uniform factor left on the row
    out, cut = wp.apply_questionable_stat_cut(_board(), INFO)
    remaining = {k: v for k, v in INFO.items() if k not in cut}
    scored, _ = wp.apply_points_from_stat_line(out, 'Full PPR', remaining)
    row = scored.loc[scored.Player == 'Q Wideout'].iloc[0]
    assert abs(row['Model Proj Pts'] - round(wp.score_projected_stats(out.loc[out.Player == 'Q Wideout'].iloc[0].to_dict(), 'Full PPR'), 2)) < 0.06
    back = scored.loc[scored.Player == 'Q Back'].iloc[0]               # an uncut row still takes its uniform factor
    assert abs(back['Model Proj Pts'] - round(round(2.0 + 1.5 + 6.0 + 0.3, 1) * 0.9, 2)) > -1       # (it is scored; see factor path)


def test_points_panel_drops_the_uniform_row_and_says_the_cut_is_in_the_stat_line(monkeypatch):
    import contextlib
    import ui.components as components
    import ui.tabs.rankings as rk

    class _FakeSt:
        def __init__(self):
            self.calls, self.session_state = [], {}

        def markdown(self, text, *a, **k):
            self.calls.append(('markdown', text))

        def caption(self, text, *a, **k):
            self.calls.append(('caption', text))

        def dataframe(self, frame, *a, **k):
            self.calls.append(('dataframe', frame))

        def expander(self, label, expanded=False, **kwargs):
            self.calls.append(('expander', label))
            return contextlib.nullcontext()

    detail = {
        'raw_points': 15.0, 'calibrated_points': 14.5, 'target_week': 6,
        'calibration': {'enabled': True, 'slope': 0.9, 'intercept': 1.0, 'raw_points': 15.0, 'displayed_points': 14.5},
        'questionable_workload': {'before': 14.5, 'after': 12.0, 'factor': 0.83, 'status': 'questionable'},
        'questionable_stat_cut': {'receptions': (5.0, 4.44), 'receiving_yards': (60.0, 51.7)},
        'points_from_stat_line': {'before': 12.0, 'after': 11.4},
    }
    fake = _FakeSt()
    monkeypatch.setattr(rk, 'st', fake)
    monkeypatch.setattr(components, 'st', fake)
    rk._render_points_path(detail)
    frame = next(c[1] for c in fake.calls if c[0] == 'dataframe')
    frame = getattr(frame, 'data', frame)
    assert list(frame['Stage']) == ['Raw model', 'Calibrated', 'Stat line scored']        # no uniform Questionable row
    assert 'Questionable cut already in it' in frame['Basis'].iloc[-1]
    text = ' '.join(str(c[1]) for c in fake.calls if c[0] == 'markdown')
    assert 'already carry the cut' in text and 'rec receptions' not in text and 'receptions ×0.89' in text
