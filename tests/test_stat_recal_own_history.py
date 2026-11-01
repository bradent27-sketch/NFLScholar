"""v2_stat_recal_own_history / _edges (2026-11-01, backlog M13): the board's stat columns re-predicted from the board's stat and
the player's own history (weeks 3-17 a linear form, week 1 projection + last season, week 2 a pull toward last season), with
the point total untouched."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.prop_probability as pp  # noqa: E402
import data.weekly_projections as wp  # noqa: E402
from data.prediction_ledger import build_ledger_frame  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402

LINEAR = {'WR': {'targets': (0.5, 0.4, 0.1, 0.1, 0.1, 1.0, 0.2)}, 'QB': {'rushing_attempts': (0.0, 1.0, 0, 0, 0, 0, 0)}}


def _history(rows):
    """rows: (name, pos, h_games, h_ps_games, {stat: (s2d, l3, ps)})"""
    recs, idx = [], []
    for name, pos, g, pg, stats in rows:
        key = clean_name_exact(pd.Series([name])).iloc[0]
        rec = {'h_games': g, 'h_ps_games': pg}
        for stat, (s2d, l3, ps) in stats.items():
            rec[f'h_s2d_{stat}'], rec[f'h_l3_{stat}'], rec[f'h_ps_{stat}'] = s2d, l3, ps
        recs.append(rec)
        idx.append((key, pos))
    return pd.DataFrame(recs, index=pd.MultiIndex.from_tuples(idx, names=['_key', 'pos']))


def _board(rows):
    cols = ['Player', 'Pos', 'Team', 'targets', 'receptions', 'rushing_attempts', 'Availability', 'Model Proj Pts']
    return pd.DataFrame(rows, columns=cols)


def test_flags_are_shipped_after_the_calibration_lines():
    for flag in ('v2_stat_recal_own_history', 'v2_stat_recal_own_history_edges'):
        assert flag in wp.MODEL_FEATURES
        assert flag in wp.DEFAULT_FEATURES and flag in wp.POST_CALIBRATION_FEATURES


def test_table_by_week_and_flag():
    core, edges = {'v2_stat_recal_own_history'}, {'v2_stat_recal_own_history_edges'}
    both = core | edges
    assert wp.stat_recal_table(core, 3) == ('linear', wp.STAT_RECAL_OWN_HISTORY)
    assert wp.stat_recal_table(core, 17) == ('linear', wp.STAT_RECAL_OWN_HISTORY)
    assert wp.stat_recal_table(core, 1) == (None, {}) and wp.stat_recal_table(core, 2) == (None, {})
    assert wp.stat_recal_table(edges, 1) == ('week1', wp.STAT_RECAL_OWN_HISTORY_EDGES[1])
    assert wp.stat_recal_table(edges, 2) == ('pull', wp.STAT_RECAL_OWN_HISTORY_EDGES[2])
    assert wp.stat_recal_table(edges, 9) == (None, {})
    assert wp.stat_recal_table(both, 18) == (None, {})                     # week 18 is not touched
    assert wp.stat_recal_table(set(), 9) == (None, {})
    # QB passing and every TD stat are never in a table; each form carries its own coefficient count
    for stats in wp.STAT_RECAL_OWN_HISTORY.values():
        assert all(len(c) == 7 for c in stats.values())
    for stats in wp.STAT_RECAL_OWN_HISTORY_EDGES[1].values():
        assert all(len(c) == 3 for c in stats.values())
    for stats in wp.STAT_RECAL_OWN_HISTORY_EDGES[2].values():
        assert all(isinstance(k, float) and 0 < k < 1 for k in stats.values())
    for table in (wp.STAT_RECAL_OWN_HISTORY, *wp.STAT_RECAL_OWN_HISTORY_EDGES.values()):
        for stats in table.values():
            assert not any(s.startswith('passing') or s.endswith('_tds') for s in stats)


def test_linear_form_and_points_left_alone():
    board = _board([['Ace Wideout', 'WR', 'AAA', 10.0, 6.0, 0.0, 1.0, 15.0]])
    hist = _history([('Ace Wideout', 'WR', 4, 17, {'targets': (6.0, 7.0, 9.0)})])
    out, info = wp.apply_stat_recal_own_history(board, hist, 'linear', LINEAR)
    blend = (4 * 6.0 + 6 * 9.0) / 10
    expected = 0.5 + 0.4 * 10 + 0.1 * 6 + 0.1 * 7 + 0.1 * 9 + 0.2 * blend
    assert out.at[0, 'targets'] == pytest.approx(expected)
    assert out.at[0, 'Model Proj Pts'] == 15.0 and out.at[0, 'receptions'] == 6.0
    assert out.at[0, '_pre_recal_targets'] == 10.0
    assert '_pre_recal_receptions' not in out.columns                       # no table entry, no column
    rec = info[('Ace Wideout', 'WR', 'AAA')]
    assert rec['form'] == 'linear' and rec['games'] == 4
    assert rec['stats']['targets'] == (10.0, pytest.approx(round(expected, 3)))


def test_linear_form_without_last_season_on_the_if_active_level():
    # 10 targets if active, half a chance to play; no last season: the projection stands in and the flag fires
    board = _board([['Ace Wideout', 'WR', 'AAA', 5.0, 3.0, 0.0, 0.5, 7.0]])
    hist = _history([('Ace Wideout', 'WR', 5, 0, {'targets': (6.0, 7.0, np.nan)})])
    out, _ = wp.apply_stat_recal_own_history(board, hist, 'linear', LINEAR)
    if_active = 0.5 + 0.4 * 10 + 0.1 * 6 + 0.1 * 7 + 0.1 * 10 + 1.0 + 0.2 * 6.0
    assert out.at[0, 'targets'] == pytest.approx(0.5 * if_active)


def test_week1_form_needs_last_season():
    table = {'WR': {'targets': (0.2, 0.3, 0.6)}}
    board = _board([['Ace Wideout', 'WR', 'AAA', 8.0, 5.0, 0.0, 1.0, 12.0],
                    ['Thin Guy', 'WR', 'BBB', 8.0, 5.0, 0.0, 1.0, 12.0]])
    hist = _history([('Ace Wideout', 'WR', 0, 16, {'targets': (np.nan, np.nan, 6.0)}),
                     ('Thin Guy', 'WR', 0, 3, {'targets': (np.nan, np.nan, 6.0)})])
    out, info = wp.apply_stat_recal_own_history(board, hist, 'week1', table)
    assert out.at[0, 'targets'] == pytest.approx(0.2 + 0.3 * 8 + 0.6 * 6)
    assert out.at[1, 'targets'] == 8.0 and np.isnan(out.at[1, '_pre_recal_targets'])
    assert list(info) == [('Ace Wideout', 'WR', 'AAA')] and info[('Ace Wideout', 'WR', 'AAA')]['form'] == 'week1'


def test_catches_never_outrun_targets():
    # separate fits push receptions above targets: the board's own catch rate is kept on the new targets
    table = {'WR': {'targets': (0.0, 0.5, 0.0), 'receptions': (0.0, 0.5, 0.5)}}
    board = _board([['Ace Wideout', 'WR', 'AAA', 8.0, 6.0, 0.0, 1.0, 12.0]])
    hist = _history([('Ace Wideout', 'WR', 0, 16, {'targets': (np.nan, np.nan, 9.0), 'receptions': (np.nan, np.nan, 7.0)})])
    out, info = wp.apply_stat_recal_own_history(board, hist, 'week1', table)
    assert out.at[0, 'targets'] == pytest.approx(4.0)                       # 0.5 x 8
    assert out.at[0, 'receptions'] == pytest.approx(4.0 * 6.0 / 8.0)        # not 0.5 x 6 + 0.5 x 7 = 6.5
    assert info[('Ace Wideout', 'WR', 'AAA')]['stats']['receptions'] == (6.0, pytest.approx(3.0))
    # the shipped week-2 table moves an RB's catches only with his targets (neither, today)
    rb = wp.STAT_RECAL_OWN_HISTORY_EDGES[2]['RB']
    assert ('receptions' in rb) == ('targets' in rb)
    for window in (wp.STAT_RECAL_OWN_HISTORY, *wp.STAT_RECAL_OWN_HISTORY_EDGES.values()):
        for stats in window.values():
            assert ('receptions' in stats) <= ('targets' in stats)


def test_week2_pull_needs_a_game_this_season():
    table = {'WR': {'targets': 0.5}}
    board = _board([['Ace Wideout', 'WR', 'AAA', 10.0, 5.0, 0.0, 1.0, 12.0],
                    ['Back Guy', 'WR', 'BBB', 10.0, 5.0, 0.0, 1.0, 12.0],      # missed week 1: first game back
                    ['Thin Guy', 'WR', 'CCC', 10.0, 5.0, 0.0, 1.0, 12.0]])     # 3 games last season: no baseline
    hist = _history([('Ace Wideout', 'WR', 1, 17, {'targets': (9.0, 9.0, 6.0)}),
                     ('Back Guy', 'WR', 0, 17, {'targets': (np.nan, np.nan, 6.0)}),
                     ('Thin Guy', 'WR', 1, 3, {'targets': (9.0, 9.0, 6.0)})])
    out, info = wp.apply_stat_recal_own_history(board, hist, 'pull', table)
    assert out.at[0, 'targets'] == pytest.approx(6.0 + 0.5 * 4.0)           # one game: last season is the baseline
    assert out['targets'].iloc[1:].tolist() == [10.0, 10.0]
    assert list(info) == [('Ace Wideout', 'WR', 'AAA')]


def test_weeks_3_17_need_a_game_this_season_and_a_baseline():
    names = ['Back Guy', 'Two Games', 'Thin Guy', 'New Guy']
    board = _board([[n, 'WR', 'AAA', 10.0, 5.0, 0.0, 1.0, 12.0] for n in names])
    hist = _history([('Back Guy', 'WR', 0, 17, {'targets': (np.nan, np.nan, 6.0)}),   # first game back: skipped
                     ('Two Games', 'WR', 2, 17, {'targets': (6.0, 6.0, 6.0)}),        # last season is the baseline
                     ('Thin Guy', 'WR', 2, 3, {'targets': (6.0, 6.0, 6.0)}),          # no baseline: skipped
                     ('New Guy', 'WR', 3, 0, {'targets': (6.0, 6.0, np.nan)})])       # 3 games this season
    out, info = wp.apply_stat_recal_own_history(board, hist, 'linear', LINEAR)
    assert out['_pre_recal_targets'].notna().tolist() == [False, True, False, True]
    assert sorted(k[0] for k in info) == ['New Guy', 'Two Games']


def test_skips_sidelined_namesakes_backup_qbs_and_low_points():
    board = _board([['Rookie Guy', 'WR', 'AAA', 8.0, 5.0, 0.0, 1.0, 10.0],      # no history at all
                    ['Hurt Guy', 'WR', 'BBB', 8.0, 5.0, 0.0, 0.0, 0.0],         # sidelined
                    ['Same Name', 'WR', 'CCC', 8.0, 5.0, 0.0, 1.0, 10.0],
                    ['Same Name', 'WR', 'DDD', 8.0, 5.0, 0.0, 1.0, 10.0],
                    ['Low Guy', 'WR', 'GGG', 1.0, 0.5, 0.0, 1.0, 2.9],          # under 3 points: usually not playing
                    ['Backup Qb', 'QB', 'EEE', 0.0, 0.0, 4.0, 1.0, 5.0],
                    ['Starter Qb', 'QB', 'FFF', 0.0, 0.0, 6.0, 1.0, 18.0]])
    board['QB Projected Starter'] = [False] * 5 + [False, True]
    stats = {'targets': (4.0, 4.0, 4.0)}
    rush = {'rushing_attempts': (2.0, 2.0, 2.0)}
    hist = _history([('Hurt Guy', 'WR', 6, 17, stats), ('Same Name', 'WR', 6, 17, stats), ('Low Guy', 'WR', 6, 17, stats),
                     ('Backup Qb', 'QB', 6, 17, rush), ('Starter Qb', 'QB', 6, 17, rush)])
    out, info = wp.apply_stat_recal_own_history(board, hist, 'linear', LINEAR)
    assert out['targets'].iloc[:4].tolist() == [8.0] * 4 and out.at[4, 'targets'] == 1.0
    assert out.at[5, 'rushing_attempts'] == 4.0
    assert out.at[6, 'rushing_attempts'] == pytest.approx(6.0)               # LINEAR's QB entry keeps the projection
    assert list(info) == [('Starter Qb', 'QB', 'FFF')]
    assert out['_pre_recal_targets'].isna().all() if '_pre_recal_targets' in out else True


def test_shipped_tables_barely_move_a_player_whose_history_agrees():
    """A guard on coefficient order: when the board, season to date, last 3 and last season all say the same typical value,
    weeks 3-17 move it by under 8%."""
    typical = {'targets': {'RB': 3.5, 'WR': 6, 'TE': 4.5}, 'receptions': {'RB': 2.8, 'WR': 4, 'TE': 3.2},
               'receiving_yards': {'RB': 22, 'WR': 50, 'TE': 35}, 'rushing_attempts': {'QB': 4, 'RB': 12},
               'rushing_yards': {'QB': 20, 'RB': 50}}
    for pos, stats in wp.STAT_RECAL_OWN_HISTORY.items():
        for stat in stats:
            x = typical[stat][pos]
            board = pd.DataFrame({'Player': ['Ace Player'], 'Pos': [pos], 'Team': ['AAA'], stat: [x], 'Availability': [1.0],
                                  'Model Proj Pts': [12.0], 'QB Projected Starter': [True]})
            hist = _history([('Ace Player', pos, 6, 17, {stat: (x, x, x)})])
            out, _ = wp.apply_stat_recal_own_history(board, hist, 'linear', {pos: {stat: stats[stat]}})
            assert abs(out.at[0, stat] / x - 1) < 0.08, (pos, stat, out.at[0, stat], x)


def test_game_script_flag_is_shipped_on_the_linear_table():
    flag = 'v2_stat_recal_game_script'
    assert flag in wp.MODEL_FEATURES and flag in wp.DEFAULT_FEATURES and flag in wp.POST_CALIBRATION_FEATURES
    assert flag not in wp.CALIBRATION_INPUT_FEATURES
    feats = {'v2_stat_recal_own_history', 'v2_stat_recal_own_history_edges', flag}
    form, table = wp.stat_recal_table(feats, 9)
    assert form == 'linear'
    assert table['QB'] == wp.STAT_RECAL_GAME_SCRIPT['QB']                  # QB rushing replaced, passing added
    for pos in ('RB', 'WR', 'TE'):
        assert table[pos] == wp.STAT_RECAL_OWN_HISTORY[pos]
    assert wp.STAT_RECAL_OWN_HISTORY['QB']['rushing_yards'][0] == 1.4287   # the shipped table is not mutated
    assert all(len(c) == 10 for c in wp.STAT_RECAL_GAME_SCRIPT['QB'].values())
    # weeks 1, 2 and 18 and a build without the core flag are unchanged
    assert wp.stat_recal_table(feats, 1) == ('week1', wp.STAT_RECAL_OWN_HISTORY_EDGES[1])
    assert wp.stat_recal_table(feats, 2) == ('pull', wp.STAT_RECAL_OWN_HISTORY_EDGES[2])
    assert wp.stat_recal_table(feats, 18) == (None, {})
    assert wp.stat_recal_table({flag}, 9) == (None, {})
    # attempts and completions are not in it, so completions can never pass attempts
    assert not {'passing_attempts', 'passing_completions'} & set(wp.STAT_RECAL_GAME_SCRIPT['QB'])


def test_game_script_inputs_margin_and_total():
    env = {'AAA': {'implied': 30.0}, 'BBB': {'implied': 20.0}, 'CCC': {'implied': 22.0}, 'DDD': {'implied': 24.0},
           'EEE': {'implied': None}, 'FFF': {'implied': 21.0}}
    opponents = {'AAA': 'BBB', 'BBB': 'AAA', 'CCC': 'DDD', 'DDD': 'CCC', 'EEE': 'FFF', 'FFF': 'EEE'}
    out = wp.stat_recal_script_inputs(env, opponents)
    league = (30 + 20 + 22 + 24 + 21) / 5
    assert out['AAA'] == pytest.approx((10.0, 30 - league)) and out['BBB'] == pytest.approx((-10.0, 20 - league))
    assert out['CCC'] == pytest.approx((-2.0, 22 - league))
    assert 'EEE' not in out and 'FFF' not in out                          # the opponent has no line
    assert wp.stat_recal_script_inputs({}, opponents) == {}


def test_game_script_terms_and_the_knee():
    c = (1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -0.5)
    table = {'QB': {'passing_yards': c}, 'WR': {'targets': (0.0, 1.0, 0, 0, 0, 0, 0)}}
    board = pd.DataFrame({'Player': ['Big Fav', 'Small Dog', 'No Line', 'Wide Out'], 'Pos': ['QB', 'QB', 'QB', 'WR'],
                          'Team': ['AAA', 'BBB', 'CCC', 'AAA'], 'passing_yards': [260.0, 240.0, 250.0, 0.0],
                          'targets': [0.0, 0.0, 0.0, 7.0], 'Availability': 1.0, 'Model Proj Pts': [20.0, 18.0, 19.0, 12.0],
                          'QB Projected Starter': [True, True, True, False]})
    hist = _history([('Big Fav', 'QB', 6, 17, {'passing_yards': (250.0, 250.0, 250.0)}),
                     ('Small Dog', 'QB', 6, 17, {'passing_yards': (230.0, 230.0, 230.0)}),
                     ('No Line', 'QB', 6, 17, {'passing_yards': (240.0, 240.0, 240.0)}),
                     ('Wide Out', 'WR', 6, 17, {'targets': (7.0, 7.0, 7.0)})])
    script = {'AAA': (10.0, 4.0), 'BBB': (-3.0, -1.0)}
    out, info = wp.apply_stat_recal_own_history(board, hist, 'linear', table, script=script)
    assert out.at[0, 'passing_yards'] == pytest.approx(1 + 0.5 * 260 - 10 + 8 - 0.5 * 3)    # 3 points past the knee
    assert out.at[1, 'passing_yards'] == pytest.approx(1 + 0.5 * 240 + 3 - 2)               # under the knee
    assert out.at[2, 'passing_yards'] == pytest.approx(1 + 0.5 * 250)                       # no line: m = t = 0
    assert out.at[3, 'targets'] == pytest.approx(7.0)                                       # seven terms ignore script
    assert out.at[0, '_pre_recal_passing_yards'] == 260.0
    assert info[('Big Fav', 'QB', 'AAA')]['stats']['passing_yards'][0] == 260.0
    # without script every row reads 0, 0
    out0, _ = wp.apply_stat_recal_own_history(board, hist, 'linear', table)
    assert out0.at[0, 'passing_yards'] == pytest.approx(1 + 0.5 * 260)


def test_game_script_table_moves_a_typical_qb_in_a_neutral_game_little():
    """A guard on coefficient order: a QB whose board and history agree on a typical value (the panel's starter averages,
    weeks 3-17 2021-2026), in a pick'em game of average total, moves under 8%; a 10-point favorite in a game 4 points richer
    than average runs fewer rushing yards and carries than the same QB in a pick'em."""
    typical = {'passing_yards': 230.0, 'passing_tds': 1.46, 'passing_interceptions': 0.69, 'rushing_attempts': 4.0,
               'rushing_yards': 20.0}
    for stat, coef in wp.STAT_RECAL_GAME_SCRIPT['QB'].items():
        x = typical[stat]
        board = pd.DataFrame({'Player': ['Even Passer', 'Big Favorite'], 'Pos': 'QB', 'Team': ['AAA', 'BBB'],
                              stat: [x, x], 'Availability': 1.0, 'Model Proj Pts': 18.0, 'QB Projected Starter': True})
        hist = _history([('Even Passer', 'QB', 6, 17, {stat: (x, x, x)}), ('Big Favorite', 'QB', 6, 17, {stat: (x, x, x)})])
        out, _ = wp.apply_stat_recal_own_history(board, hist, 'linear', {'QB': {stat: coef}},
                                                 script={'AAA': (0.0, 0.0), 'BBB': (10.0, 4.0)})
        assert out[f'_pre_recal_{stat}'].notna().all()
        assert abs(out.at[0, stat] / x - 1) < 0.08, (stat, out.at[0, stat], x)
        if stat.startswith('rushing'):
            assert out.at[1, stat] < out.at[0, stat], stat


def test_no_table_is_a_no_op():
    board = _board([['Ace Wideout', 'WR', 'AAA', 10.0, 6.0, 0.0, 1.0, 15.0]])
    hist = _history([('Ace Wideout', 'WR', 4, 17, {'targets': (6.0, 7.0, 9.0)})])
    out, info = wp.apply_stat_recal_own_history(board, hist, None, {})
    assert out.equals(board) and info == {}


def test_player_history_counts_box_score_games_only():
    rows = [{'player_display_name': 'Ace Wideout', 'position': 'WR', 'week': w, 'targets': t, '_silent_game': silent}
            for w, t, silent in ((1, 6.0, False), (2, 0.0, True), (3, 8.0, False), (4, 10.0, False))]
    prior = [{'player_display_name': 'Ace Wideout', 'position': 'WR', 'week': w, 'targets': 5.0} for w in range(1, 6)]
    h = pp.player_history(pd.DataFrame(rows), 'player_display_name', 5, pd.DataFrame(prior), 'player_display_name').iloc[0]
    assert h['h_games'] == 3 and h['h_s2d_targets'] == pytest.approx(8.0)     # the silent week 2 is not a box-score game
    assert h['h_ps_games'] == 5


def test_player_history_with_no_games_this_season():
    prior = [{'player_display_name': 'Ace Wideout', 'position': 'WR', 'week': w, 'targets': 5.0} for w in range(1, 6)]
    h = pp.player_history(pd.DataFrame(columns=['player_display_name', 'position', 'week', 'targets']), 'player_display_name',
                          1, pd.DataFrame(prior), 'player_display_name')
    assert h.loc[('acewideout', 'WR'), 'h_ps_games'] == 5
    assert np.isnan(h.loc[('acewideout', 'WR'), 'h_s2d_targets'])


def test_prop_engine_reads_the_stat_before_the_recal(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, 'ENGINE_PATH', str(tmp_path / 'missing.json'))     # no engine: prop_mean = the projection read
    frame = pd.DataFrame({'_key': ['a', 'b'], 'Pos': ['WR', 'WR'], 'targets': [8.0, 7.0], '_pre_recal_targets': [10.0, np.nan],
                          'line': [6.5, 6.5]})
    out = pp.prop_probabilities(frame, 'targets')
    assert out['prop_mean'].tolist() == [10.0, 7.0]


def test_ledger_keeps_the_stat_before_the_recal():
    board = pd.DataFrame({'Player': ['Ace Wideout'], 'Pos': ['WR'], 'Team': ['AAA'], 'Model Proj Pts': [12.0],
                          'targets': [8.0], '_pre_recal_targets': [10.0]})
    out = build_ledger_frame(2026, 9, board)
    assert out.at[0, 'targets'] == 8.0 and out.at[0, '_pre_recal_targets'] == 10.0


def test_the_main_stat_table_shows_the_own_history_step(monkeypatch):
    import contextlib
    import ui.tabs.rankings as rk

    class FakeSt:
        def __init__(self):
            self.calls = []

        def markdown(self, text, *a, **k):
            self.calls.append(('markdown', text))

        def caption(self, text, *a, **k):
            self.calls.append(('caption', text))

        def dataframe(self, frame, *a, **k):
            self.calls.append(('dataframe', frame))

        def expander(self, label, expanded=False):
            return contextlib.nullcontext()

    def stat(final, own=0.0):
        pre = final - own
        return {'raw_prior_rate': pre, 'defense_adjusted_prior_rate': pre, 'blended_rate': pre, 'current_rate': 0.0,
                'current_games': 0.0, 'current_weight': 0.0, 'matchup_multiplier': 1.0, 'pre_vacancy_projection': pre,
                'pass_capacity_delta': 0.0, 'vacancy_delta': 0.0, 'top_end_delta': 0.0, 'opening_calibration_delta': 0.0,
                'own_history_delta': own, 'final_projection': final, 'projection': final}

    fake = FakeSt()
    monkeypatch.setattr(rk, 'st', fake)
    stats = {'targets': stat(7.0, own=-1.0), 'receptions': stat(4.5, own=-0.5), 'receiving_yards': stat(60.0),
             'receiving_tds': stat(0.4)}
    rk._render_decomposition_primary_table({'stats': stats, 'season_year': 2026, 'scoring_mode': 'Full PPR'}, None)
    styled = next(c[1] for c in fake.calls if c[0] == 'dataframe')
    frame = getattr(styled, 'data', styled)
    cols = list(frame.columns)
    assert 'Own-history Δ' in cols and cols.index('Own-history Δ') < cols.index('Projected value')
    assert frame[frame['Stat'] == 'Targets'].iloc[0]['Own-history Δ'].startswith('-1')
    assert any('Own-history' in str(c[1]) for c in fake.calls if c[0] == 'markdown')
