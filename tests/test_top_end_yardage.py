"""v2_top_end_yardage (2026-10-13): from week 3, projected yards above a knot count only in part, the raw points are
re-scored from the compressed line and the points calibration line is applied to them."""
import contextlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

from data.transforms import score_projected_stats  # noqa: E402
from data.weekly_projections import (CALIBRATION_INPUT_FEATURES, DEFAULT_FEATURES, MODEL_FEATURES,  # noqa: E402
                                     POST_CALIBRATION_FEATURES, TOP_END_YARDAGE, _ALL_PROJECTION_STATS,
                                     _weekly_calibration_for, apply_top_end_yardage)


def _board():
    rows = [
        # name, pos, team, avail, stat line
        ('Alpha', 'WR', 'AAA', 1.0, {'targets': 10.0, 'receptions': 7.0, 'receiving_yards': 110.0,
                                     'receiving_tds': 0.6, 'rushing_yards': 70.0}),
        ('Possession', 'WR', 'AAA', 1.0, {'targets': 7.0, 'receptions': 5.0, 'receiving_yards': 55.0,
                                          'receiving_tds': 0.3}),
        ('Seam', 'TE', 'AAA', 1.0, {'targets': 8.0, 'receptions': 6.0, 'receiving_yards': 80.0,
                                    'receiving_tds': 0.5}),
        ('Bell Cow', 'RB', 'AAA', 1.0, {'rushing_attempts': 21.0, 'rushing_yards': 110.0, 'rushing_tds': 0.8,
                                        'targets': 4.0, 'receptions': 3.0, 'receiving_yards': 25.0}),
        ('Starter', 'QB', 'AAA', 1.0, {'passing_attempts': 38.0, 'passing_yards': 310.0, 'passing_tds': 2.0,
                                       'passing_interceptions': 0.7, 'rushing_yards': 80.0}),
        ('Hurt Star', 'WR', 'AAA', 0.0, {'targets': 10.0, 'receiving_yards': 120.0}),
        ('Nobody', 'WR', 'AAA', 1.0, {}),
    ]
    recs = []
    for name, pos, team, avail, line in rows:
        rec = {'Player': name, 'Pos': pos, 'Team': team, 'Availability': avail}
        for s in _ALL_PROJECTION_STATS:
            rec[s] = float(line.get(s, 0.0))
        pts = max(0.0, score_projected_stats(line, 'Full PPR')) if avail > 0 else 0.0
        rec['Raw Model Proj Pts'] = round(pts, 2)
        rec['Model Proj Pts'] = rec['Raw Model Proj Pts']
        rec['Calibrated Model Proj Pts'] = rec['Model Proj Pts']
        recs.append(rec)
    return pd.DataFrame(recs)


def _expected(value, stat):
    _pos, knot, share = TOP_END_YARDAGE[stat]
    return round(value - share * max(0.0, value - knot), 2)


def test_flag_is_shipped_and_is_a_calibration_input():
    assert 'v2_top_end_yardage' in MODEL_FEATURES
    assert 'v2_top_end_yardage' in DEFAULT_FEATURES
    # it changes the stat line the points lines are fit on, so it is never a post-calibration step
    assert 'v2_top_end_yardage' not in POST_CALIBRATION_FEATURES
    assert 'v2_top_end_yardage' in CALIBRATION_INPUT_FEATURES


def test_table_shape():
    # QB passing is deliberately left out: it pushed the already under-projected top QBs further down
    assert set(TOP_END_YARDAGE) == {'receiving_yards', 'rushing_yards'}
    for stat, (positions, knot, share) in TOP_END_YARDAGE.items():
        assert stat in _ALL_PROJECTION_STATS
        assert 0.0 < share < 1.0 and knot > 0
        assert 'QB' not in positions
    assert TOP_END_YARDAGE['rushing_yards'][0] == ('RB',)


def test_weeks_one_and_two_are_untouched():
    board = _board()
    for week in (1, 2, None):
        out, info = apply_top_end_yardage(board, week, 'Full PPR')
        assert info == {}
        pd.testing.assert_frame_equal(out, board)


def test_week_three_compresses_only_yards_above_the_knot_for_listed_positions():
    board = _board()
    out, info = apply_top_end_yardage(board, 3, 'Full PPR', calibrate=False)
    o = out.set_index('Player')
    b = board.set_index('Player')
    assert o.at['Alpha', 'receiving_yards'] == _expected(110.0, 'receiving_yards')
    assert o.at['Alpha', 'rushing_yards'] == 70.0             # a WR's rushing is not listed
    assert o.at['Possession', 'receiving_yards'] == 55.0      # below the knot
    assert o.at['Seam', 'receiving_yards'] == _expected(80.0, 'receiving_yards')
    assert o.at['Bell Cow', 'rushing_yards'] == _expected(110.0, 'rushing_yards')
    assert o.at['Bell Cow', 'receiving_yards'] == 25.0
    assert o.at['Starter', 'passing_yards'] == 310.0          # QBs are not touched at all
    assert o.at['Starter', 'rushing_yards'] == 80.0
    assert o.at['Starter', 'Model Proj Pts'] == b.at['Starter', 'Model Proj Pts']
    # volume and TDs never move
    for stat in ('targets', 'receptions', 'receiving_tds', 'rushing_attempts', 'rushing_tds', 'passing_attempts',
                 'passing_tds'):
        assert (o[stat] == b[stat]).all()
    assert {k[0] for k in info} == {'Alpha', 'Seam', 'Bell Cow'}


def test_sidelined_and_zero_rows_are_left_alone():
    board = _board()
    out, _ = apply_top_end_yardage(board, 5, 'Full PPR')
    keep = board['Player'].isin(['Hurt Star', 'Nobody', 'Possession'])
    pd.testing.assert_frame_equal(out[keep].reset_index(drop=True), board[keep].reset_index(drop=True))


def test_raw_points_are_rescored_and_the_points_line_is_reapplied():
    board = _board()
    out, info = apply_top_end_yardage(board, 6, 'Full PPR', calibrate=True)
    for _, row in out[out['Player'].isin([k[0] for k in info])].iterrows():
        line = {s: float(row[s]) for s in _ALL_PROJECTION_STATS}
        raw = round(max(0.0, score_projected_stats(line, 'Full PPR')), 2)
        assert row['Raw Model Proj Pts'] == raw
        slope, intercept = _weekly_calibration_for(row['Pos'], 6)
        assert row['Model Proj Pts'] == round(max(0.0, intercept + slope * raw), 2)
        assert row['Calibrated Model Proj Pts'] == row['Model Proj Pts']
        rec = info[(row['Player'], row['Pos'], row['Team'])]
        before = board.set_index('Player').at[row['Player'], 'Raw Model Proj Pts']
        assert rec['points_before'] == before and rec['points_after'] == raw < before


def test_without_calibration_the_total_is_the_rescored_line():
    board = _board()
    out, info = apply_top_end_yardage(board, 4, 'Full PPR', calibrate=False)
    moved = out['Player'].isin([k[0] for k in info])
    assert (out.loc[moved, 'Model Proj Pts'] == out.loc[moved, 'Raw Model Proj Pts']).all()


def test_compression_is_monotone_and_never_raises_a_value():
    for stat, (_pos, knot, share) in TOP_END_YARDAGE.items():
        xs = np.linspace(0.0, knot * 2.5, 200)
        ys = np.array([_expected(x, stat) for x in xs])
        assert np.all(ys <= xs + 0.005 + 1e-9)       # rounding to 0.01 only
        assert np.all(np.diff(ys) >= -0.011)        # rounding to 0.01 only


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


def _stat(final, top_end=0.0, **extra):
    base = {'raw_prior_rate': final, 'defense_adjusted_prior_rate': final, 'blended_rate': final,
            'current_rate': 0.0, 'current_games': 0.0, 'current_weight': 0.0, 'matchup_multiplier': 1.0,
            'pre_vacancy_projection': final - top_end, 'pass_capacity_delta': 0.0, 'vacancy_delta': 0.0,
            'top_end_delta': top_end, 'opening_calibration_delta': 0.0, 'final_projection': final, 'projection': final}
    base.update(extra)
    return base


def test_the_main_stat_table_renders_a_top_end_stage(monkeypatch):
    import ui.tabs.rankings as rk
    fake = _FakeSt()
    monkeypatch.setattr(rk, 'st', fake)
    stats = {'receiving_yards': _stat(88.0, top_end=-20.0), 'receptions': _stat(7.0), 'targets': _stat(10.0),
             'receiving_tds': _stat(0.6)}
    rk._render_decomposition_primary_table({'stats': stats, 'season_year': 2026, 'scoring_mode': 'Full PPR'}, None)
    styled = next(c[1] for c in fake.calls if c[0] == 'dataframe')
    frame = getattr(styled, 'data', styled)
    cols = list(frame.columns)
    assert 'Top-end yardage Δ' in cols and 'Opening-week calibration Δ' not in cols
    assert cols.index('Vacancy Δ') < cols.index('Top-end yardage Δ') < cols.index('Projected value')
    rec = frame[frame['Stat'] == 'Receiving Yards'].iloc[0]
    assert rec['Top-end yardage Δ'].startswith('-20')
    points = frame[frame['Stat'] == 'Fantasy points at this stage'].iloc[0]
    assert points['Top-end yardage Δ'] == points['Projected value']
    assert points['Vacancy Δ'] != points['Projected value']           # 2 points of receiving yards earlier
    assert any('Top-end yardage' in str(c[1]) for c in fake.calls if c[0] == 'markdown')


def test_points_path_shows_the_top_end_step(monkeypatch):
    import ui.tabs.rankings as rk
    fake = _FakeSt()
    monkeypatch.setattr(rk, 'st', fake)
    rk._render_points_path({
        'raw_points': 20.0, 'calibrated_points': 19.4, 'target_week': 6,
        'calibration': {'enabled': True, 'slope': 0.969, 'intercept': 0.99, 'raw_points': 20.0, 'displayed_points': 20.37},
        'top_end_yardage': {'stats': {'receiving_yards': (110.0, 90.0)}, 'points_before': 22.0, 'points_after': 20.0},
    })
    styled = next(c[1] for c in fake.calls if c[0] == 'dataframe')
    frame = getattr(styled, 'data', styled)
    assert list(frame['Stage']) == ['Raw model', 'Top-end yardage', 'Calibrated']
    assert list(frame['Points']) == ['22.00', '20.00', '20.37']
    assert frame['Change'].iloc[1] == '-2.00'
    assert all(len(b) < 90 for b in frame['Basis'])
