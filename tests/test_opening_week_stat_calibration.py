"""v2_opening_week_stat_calibration (2026-10-08): in week 1 (RB/WR/TE) and week 2 (RB) each stat gets its own
calibration line, the points are re-scored from the calibrated line, and nothing else moves."""
import contextlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

from data.transforms import score_projected_stats  # noqa: E402
from data.weekly_projections import (DEFAULT_FEATURES, MODEL_FEATURES, OPENING_WEEK_STAT_CALIBRATION,  # noqa: E402
                                     _ALL_PROJECTION_STATS, apply_opening_week_stat_calibration)


def _board():
    rows = [
        # name, pos, team, avail, stat line
        ('Bell Cow', 'RB', 'AAA', 1.0, {'rushing_attempts': 18.0, 'rushing_yards': 80.0, 'rushing_tds': 0.6,
                                        'targets': 4.0, 'receptions': 3.0, 'receiving_yards': 24.0,
                                        'receiving_tds': 0.12}),
        ('Hurt Back', 'RB', 'AAA', 0.0, {'rushing_attempts': 0.0, 'rushing_yards': 0.0}),
        ('Alpha', 'WR', 'AAA', 1.0, {'targets': 9.0, 'receptions': 6.0, 'receiving_yards': 80.0,
                                     'receiving_tds': 0.5}),
        ('Seam', 'TE', 'AAA', 1.0, {'targets': 6.0, 'receptions': 4.5, 'receiving_yards': 45.0,
                                    'receiving_tds': 0.35}),
        ('Starter', 'QB', 'AAA', 1.0, {'passing_attempts': 34.0, 'passing_yards': 250.0, 'passing_tds': 1.7,
                                       'passing_interceptions': 0.8, 'rushing_yards': 15.0}),
        ('Nobody', 'WR', 'AAA', 1.0, {}),           # healthy, zero projected volume
    ]
    recs = []
    for name, pos, team, avail, line in rows:
        rec = {'Player': name, 'Pos': pos, 'Team': team, 'Availability': avail}
        for s in _ALL_PROJECTION_STATS:
            rec[s] = float(line.get(s, 0.0))
        pts = max(0.0, score_projected_stats(line, 'Full PPR'))
        rec['Raw Model Proj Pts'] = pts
        rec['Model Proj Pts'] = round(0.9 * pts + 1.0, 2) if pts > 0 else 0.0     # some points line
        rec['Calibrated Model Proj Pts'] = rec['Model Proj Pts']
        recs.append(rec)
    return pd.DataFrame(recs)


def test_flag_is_registered_and_shipped():
    assert 'v2_opening_week_stat_calibration' in MODEL_FEATURES
    assert 'v2_opening_week_stat_calibration' in DEFAULT_FEATURES


def test_post_calibration_steps_are_not_inputs_of_the_calibration_lines():
    # a calibration dump (CALIBRATION_INPUT_FEATURES) must record the totals BEFORE these four acted
    from data.weekly_projections import CALIBRATION_INPUT_FEATURES, POST_CALIBRATION_FEATURES
    assert POST_CALIBRATION_FEATURES == {'v2_season_anchor', 'v2_opening_week_stat_calibration',
                                         'v2_questionable_workload', 'v2_expected_points_luck'}
    assert POST_CALIBRATION_FEATURES <= DEFAULT_FEATURES
    assert not (CALIBRATION_INPUT_FEATURES & POST_CALIBRATION_FEATURES)
    assert 'calibration' not in CALIBRATION_INPUT_FEATURES
    assert CALIBRATION_INPUT_FEATURES == DEFAULT_FEATURES - {'calibration'} - POST_CALIBRATION_FEATURES


def test_week_three_and_later_is_untouched():
    board = _board()
    for week in (3, 5, 17, None):
        out, info = apply_opening_week_stat_calibration(board, week, 'Full PPR')
        assert info == {}
        pd.testing.assert_frame_equal(out, board)


def test_week_one_calibrates_rb_wr_te_stat_lines_and_rescores_points():
    board = _board()
    out, info = apply_opening_week_stat_calibration(board, 1, 'Full PPR')
    moved = {k[0] for k in info}
    assert moved == {'Bell Cow', 'Alpha', 'Seam'}
    for _, row in out[out['Player'].isin(moved)].iterrows():
        lines = OPENING_WEEK_STAT_CALIBRATION[1][row['Pos']]
        before = board.set_index('Player').loc[row['Player']]
        for stat, (slope, intercept) in lines.items():
            assert row[stat] == round(max(0.0, slope * before[stat] + intercept), 2)
        line = {s: float(row[s]) for s in _ALL_PROJECTION_STATS}
        assert row['Model Proj Pts'] == round(max(0.0, score_projected_stats(line, 'Full PPR')), 2)
        assert row['Calibrated Model Proj Pts'] == row['Model Proj Pts']
        assert row['Raw Model Proj Pts'] == before['Raw Model Proj Pts']       # the uncalibrated total is kept
        rec = info[(row['Player'], row['Pos'], row['Team'])]
        assert rec['points_before'] == before['Model Proj Pts'] and rec['points_after'] == row['Model Proj Pts']
        assert set(rec['stats']) == set(lines)
    # the cold-start high is what it removes: every calibrated starter projects lower
    assert (out.set_index('Player').loc[list(moved), 'Model Proj Pts']
            < board.set_index('Player').loc[list(moved), 'Raw Model Proj Pts']).all()


def test_week_one_leaves_qb_sidelined_and_zero_rows_alone():
    board = _board()
    out, _ = apply_opening_week_stat_calibration(board, 1, 'Full PPR')
    keep = board['Player'].isin(['Starter', 'Hurt Back', 'Nobody'])
    pd.testing.assert_frame_equal(out[keep].reset_index(drop=True), board[keep].reset_index(drop=True))


def test_week_two_moves_running_backs_only():
    board = _board()
    out, info = apply_opening_week_stat_calibration(board, 2, 'Full PPR')
    assert {k[0] for k in info} == {'Bell Cow'}
    others = ~board['Player'].eq('Bell Cow')
    pd.testing.assert_frame_equal(out[others].reset_index(drop=True), board[others].reset_index(drop=True))


def test_table_shape_and_coherence():
    for week, by_pos in OPENING_WEEK_STAT_CALIBRATION.items():
        assert week in (1, 2)
        assert 'QB' not in by_pos
        for pos, lines in by_pos.items():
            for stat, (slope, intercept) in lines.items():
                assert stat in _ALL_PROJECTION_STATS
                assert 0.2 < slope < 1.2 and intercept >= 0.0
            # the lines alone keep catches <= targets at any realistic catch rate
            t = np.linspace(0.0, 15.0, 61)
            for rate in (0.3, 0.6, 0.9):
                ts, ti = lines['targets']
                rs, ri = lines['receptions']
                assert np.all(rs * rate * t + ri <= ts * t + ti + 1e-9)


def test_a_back_who_catches_every_target_is_clamped_to_his_targets():
    board = _board()
    i = board.index[board['Player'].eq('Bell Cow')][0]
    board.loc[i, ['targets', 'receptions']] = 9.0
    out, _ = apply_opening_week_stat_calibration(board, 1, 'Full PPR')
    row = out.loc[out['Player'].eq('Bell Cow')].iloc[0]
    assert row['receptions'] <= row['targets']


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


def test_points_path_names_the_per_stat_stage(monkeypatch):
    import ui.tabs.rankings as rk
    fake = _FakeSt()
    monkeypatch.setattr(rk, 'st', fake)
    rk._render_points_path({
        'raw_points': 15.0, 'calibrated_points': 12.4, 'target_week': 1,
        'calibration': {'enabled': True, 'slope': 0.9, 'intercept': 1.0, 'raw_points': 15.0, 'displayed_points': 12.4},
        'opening_week_calibration': {'week': 1, 'points_before': 14.5, 'points_after': 12.4, 'stats': {}},
    })
    styled = next(c[1] for c in fake.calls if c[0] == 'dataframe')
    frame = getattr(styled, 'data', styled)
    assert list(frame['Stage']) == ['Raw model', 'Calibrated (per stat)']
    assert 'Replaces the position points line' in frame['Basis'].iloc[1]
    assert frame['Points'].iloc[1] == '12.40'


def _stat(final, opening=0.0, **extra):
    base = {'raw_prior_rate': final, 'defense_adjusted_prior_rate': final, 'blended_rate': final,
            'current_rate': 0.0, 'current_games': 0.0, 'current_weight': 0.0, 'matchup_multiplier': 1.0,
            'pre_vacancy_projection': final - opening, 'pass_capacity_delta': 0.0, 'vacancy_delta': 0.0,
            'opening_calibration_delta': opening, 'final_projection': final, 'projection': final}
    base.update(extra)
    return base


def _render_primary_table(monkeypatch, stats):
    """Run the decomposition dialog's main stat table against a recording stand-in for ``st``."""
    import ui.tabs.rankings as rk
    fake = _FakeSt()
    monkeypatch.setattr(rk, 'st', fake)
    rk._render_decomposition_primary_table(
        {'stats': stats, 'season_year': 2027, 'scoring_mode': 'Full PPR'}, None)
    styled = next(c[1] for c in fake.calls if c[0] == 'dataframe')
    return getattr(styled, 'data', styled), fake


def test_the_main_stat_table_renders_an_opening_week_stage(monkeypatch):
    # regression: the stage was added to the loop but not to the stage-totals dict, so every decomposition dialog
    # raised KeyError('after_opening') the moment the shipped flag was on
    stats = {'receiving_yards': _stat(44.0, opening=-7.0), 'receptions': _stat(3.5, opening=-0.6),
             'receiving_tds': _stat(0.25, opening=-0.05), 'targets': _stat(5.0, opening=-1.0)}
    frame, fake = _render_primary_table(monkeypatch, stats)
    assert 'Opening-week calibration Δ' in frame.columns
    assert list(frame.columns).index('Opening-week calibration Δ') < list(frame.columns).index('Projected value')
    receiving = frame[frame['Stat'] == 'Receiving Yards'].iloc[0]
    assert receiving['Opening-week calibration Δ'].startswith('-7')
    points = frame[frame['Stat'] == 'Fantasy points at this stage'].iloc[0]
    assert points['Opening-week calibration Δ'].endswith('pts') and points['Projected value'].endswith('pts')
    # the stage is the last step: points after it equal the points of the final stat line
    assert points['Opening-week calibration Δ'] == points['Projected value']
    assert any('Opening-week calibration' in str(c[1]) for c in fake.calls if c[0] == 'markdown')


def test_the_main_stat_table_has_no_opening_column_from_week_three(monkeypatch):
    frame, _ = _render_primary_table(monkeypatch, {'receiving_yards': _stat(60.0), 'receptions': _stat(5.0)})
    assert 'Opening-week calibration Δ' not in frame.columns
    assert 'Projected value' in frame.columns
