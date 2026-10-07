"""Tests for ui/box_view.py - the clickable Score cells and the in-dialog box score of the Weekly Rankings
decomposition - and the score fields data.box_score's resolver now returns."""
import os
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

import data.game_slate as game_slate  # noqa: E402
import ui.box_view as bv  # noqa: E402
from data.box_score import game_link_positions  # noqa: E402


def _slate():
    rows = [
        # Game Id, Week, Away, Home, Away Pts, Home Pts, Winner, Played, Neutral Site
        ('2026_01_DAL_NYG', 1, 'DAL', 'NYG', 28, 20, 'DAL', True, False),     # DAL (away) won 28-20
        ('2026_02_WAS_DAL', 2, 'WAS', 'DAL', 20, 37, 'DAL', True, False),     # DAL (home) won 37-20
        ('2026_03_BAL_DAL', 3, 'BAL', 'DAL', 34, 31, 'BAL', True, False),     # DAL lost 31-34
        ('2026_05_TB_DAL', 5, 'TB', 'DAL', float('nan'), float('nan'), None, False, False),   # not played
    ]
    return pd.DataFrame([{'Game Id': g, 'Week': w, 'Away': a, 'Home': h, 'Away Pts': ap, 'Home Pts': hp,
                          'Winner': win, 'Played': pl, 'Neutral Site': ne, 'Date Display': '10/01/2026'}
                         for g, w, a, h, ap, hp, win, pl, ne in rows])


@pytest.fixture
def slate(monkeypatch):
    monkeypatch.setattr(game_slate, 'season_slate', lambda season: (_slate(), None))
    return _slate()


@pytest.fixture
def state(monkeypatch):
    fake = {}
    monkeypatch.setattr(bv.st, 'session_state', fake)
    return fake


def test_link_entries_carry_the_score_from_the_subjects_side_and_the_result():
    log = pd.DataFrame({'game_id': ['2026_01_DAL_NYG', '2026_02_WAS_DAL', '2026_03_BAL_DAL', '2026_05_TB_DAL']})
    entries = game_link_positions(log, _slate(), team='DAL')
    assert [e['score'] for e in entries] == ['28-20', '37-20', '31-34', '']       # DAL's own points first
    assert [e['result'] for e in entries] == ['W', 'W', 'L', '']
    # the same games read from the other side
    other = game_link_positions(log.iloc[[0]], _slate(), team='NYG')
    assert other[0]['score'] == '20-28' and other[0]['result'] == 'L'
    # the existing fields are unchanged
    assert entries[0]['label'] == 'W1 @NYG' and entries[1]['label'] == 'W2 vs WAS' and entries[0]['won'] is True


def test_score_entries_resolves_by_game_id_and_falls_back_to_week_and_team(slate):
    by_id = bv.score_entries(pd.DataFrame({'game_id': ['2026_03_BAL_DAL', 'nope']}), 2026, 'DAL', 'week', 'game_id')
    assert by_id[0]['game_id'] == '2026_03_BAL_DAL' and by_id[1] is None          # an unknown id is dropped, not guessed
    # a defense-allowed log has no game id: (week, defense) is exact, and the score reads defense-first
    defense = pd.DataFrame({'_week_num': [1.0, 2.0, 3.0]})
    by_week = bv.score_entries(defense, 2026, 'NYG', '_week_num')
    assert by_week[0]['game_id'] == '2026_01_DAL_NYG' and by_week[0]['score'] == '20-28'
    assert by_week[1] is None                                                      # NYG did not play week 2
    assert bv.score_entries(pd.DataFrame(), 2026, 'DAL', 'week') == []


def test_score_texts_dash_for_unplayed_and_unresolved():
    entries = [{'score': '24-17'}, {'score': ''}, None]
    assert bv.score_texts(entries) == ['24-17', bv.NO_SCORE, bv.NO_SCORE]


def test_clicked_entry_only_counts_a_score_cell_on_a_real_game_row():
    entries = [{'game_id': 'g1'}, None, {'game_id': 'g3'}]           # row 3 (index 3) is the trailing AVG row
    sel = lambda *cells: SimpleNamespace(selection=SimpleNamespace(cells=list(cells)))
    assert bv.clicked_entry(sel((0, 'Score')), entries, 'Score') == {'game_id': 'g1'}
    assert bv.clicked_entry(sel((2, 'Score')), entries, 'Score') == {'game_id': 'g3'}
    assert bv.clicked_entry(sel((0, 'Opponent')), entries, 'Score') is None        # another column
    assert bv.clicked_entry(sel((1, 'Score')), entries, 'Score') is None           # a game that did not resolve
    assert bv.clicked_entry(sel((3, 'Score')), entries, 'Score') is None           # the AVG row
    assert bv.clicked_entry(sel(), entries, 'Score') is None
    # dict-style state, as session_state hands it back
    assert bv.clicked_entry({'selection': {'rows': [], 'columns': [], 'cells': [(2, 'Score')]}}, entries, 'Score') == {'game_id': 'g3'}
    assert bv.clicked_entry(None, entries, 'Score') is None


def test_box_view_state_belongs_to_one_decomposition_and_closing_resets_the_tables(state):
    owner = bv.owner_key({'player': 'CeeDee Lamb', 'team': 'DAL', 'target_week': 5, 'as_of_week': 5})
    other = bv.owner_key({'player': 'Dak Prescott', 'team': 'DAL', 'target_week': 5, 'as_of_week': 5})
    assert bv.active_box_view(owner) is None
    bv.open_box_view(owner, 2026, '2026_03_BAL_DAL')
    assert bv.active_box_view(owner) == {'owner': owner, 'season': 2026, 'game_id': '2026_03_BAL_DAL'}
    assert bv.active_box_view(other) is None             # a stale box can never show on someone else's breakdown
    gen = bv.table_generation()
    bv.close_box_view()
    assert bv.active_box_view(owner) is None
    assert bv.table_generation() == gen + 1              # new widget keys: no old selection reopens the box


def test_render_game_table_opens_the_box_on_a_score_click_and_not_otherwise(state, monkeypatch):
    calls = {'rerun': 0, 'frames': 0}
    owner = ('P', 'DAL', 5, 5)
    entries = [{'game_id': 'g1'}, {'game_id': 'g2'}]
    styled = pd.DataFrame({'Score': ['1-0', '2-1']})

    def fake_dataframe(frame, **kwargs):
        calls['frames'] += 1
        calls['kwargs'] = kwargs
        return calls['event']

    monkeypatch.setattr(bv.st, 'dataframe', fake_dataframe)
    monkeypatch.setattr(bv.st, 'rerun', lambda **k: calls.__setitem__('rerun', calls['rerun'] + 1))

    calls['event'] = SimpleNamespace(selection=SimpleNamespace(cells=[(1, 'Score')]))
    bv.render_game_table(styled, entries, score_col='Score', key='k', owner=owner, season=2026, height=100)
    assert bv.active_box_view(owner)['game_id'] == 'g2' and calls['rerun'] == 1
    assert calls['kwargs']['selection_mode'] == 'single-cell' and calls['kwargs']['on_select'] == 'rerun'
    assert calls['kwargs']['key'] == 'k_0'

    bv.close_box_view()
    calls['event'] = SimpleNamespace(selection=SimpleNamespace(cells=[]))
    bv.render_game_table(styled, entries, score_col='Score', key='k', owner=owner, season=2026, height=100)
    assert bv.active_box_view(owner) is None and calls['rerun'] == 1                # nothing selected: nothing opens
    assert calls['kwargs']['key'] == 'k_1'                                          # a fresh key after the close

    # a table with no resolvable game is a plain table: no selection widget at all
    bv.render_game_table(styled, [None, None], score_col='Score', key='k', owner=owner, season=2026, height=100)
    assert 'on_select' not in calls['kwargs'] and 'key' not in calls['kwargs']


def test_box_view_context_says_whose_game_it_is(slate):
    own = slate.iloc[1]            # WAS @ DAL, week 2
    detail = {'player': 'CeeDee Lamb', 'team': 'DAL', 'opponent': 'TB'}
    focus, highlight, text = bv.box_view_context(detail, own)
    assert (focus, highlight) == ('DAL', 'CeeDee Lamb') and text == "Week 2 WAS @ DAL · CeeDee Lamb's game"

    # a defense-allowed log game that does not involve his team: the defense's tab, nobody bolded, not called his
    foreign = pd.Series({'Away': 'GB', 'Home': 'TB', 'Week': 4, 'Neutral Site': False})
    focus, highlight, text = bv.box_view_context(detail, foreign)
    assert focus == 'TB' and highlight is None
    assert text.startswith('Week 4 GB @ TB') and "not CeeDee Lamb's own game" in text

    unrelated = pd.Series({'Away': 'GB', 'Home': 'CHI', 'Week': 4, 'Neutral Site': False})
    assert bv.box_view_context(detail, unrelated) == (None, None, 'Week 4 GB @ CHI')
    neutral = pd.Series({'Away': 'DAL', 'Home': 'NYG', 'Week': 1, 'Neutral Site': True})
    assert bv.box_view_context(detail, neutral)[2].startswith('Week 1 DAL vs NYG')
