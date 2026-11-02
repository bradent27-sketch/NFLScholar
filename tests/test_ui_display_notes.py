"""2026-11-02 UI / display notes: the role-specific matchup multiplier, the Deep Dive game-log columns (result + score, the
opponent's difficulty percentile, team plays), the table centering and the small text helpers."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

import data.weekly_projections as wp  # noqa: E402
import ui.styling as styling  # noqa: E402
import ui.tabs.rankings as rk  # noqa: E402
from ui.tabs import rankings_games  # noqa: E402


def _stat(projection, multiplier):
    return {'projection': projection, 'matchup_multiplier': multiplier}


def _meta(*players):
    return {'explanations': {(name, pos, team): {'stats': stats} for name, pos, team, stats in players}}


def _board(*rows):
    return pd.DataFrame([{'Player': n, 'Pos': p, 'Team': t, 'Model Proj Pts': pts} for n, p, t, pts in rows])


# ---- 1. the multiplier is the player's own: his stats, weighted by the points they score -----------------------------------

def test_a_running_qb_reads_the_rushing_multiplier_a_pocket_qb_does_not():
    # the same defense: soft against QB rushing (1.30), average against QB passing (1.00)
    runner = {'passing_yards': _stat(250.0, 1.00), 'rushing_yards': _stat(60.0, 1.30)}
    pocket = {'passing_yards': _stat(250.0, 1.00), 'rushing_yards': _stat(5.0, 1.30)}
    meta = _meta(('Runner', 'QB', 'AAA', runner), ('Pocket', 'QB', 'BBB', pocket))
    out = rk._role_matchup_by_player(meta, _board(('Runner', 'QB', 'AAA', 24.0), ('Pocket', 'QB', 'BBB', 17.0)), 'Full PPR')
    assert out[('Runner', 'QB', 'AAA')][0] > out[('Pocket', 'QB', 'BBB')][0] > 1.0
    # worked: points 250 * .04 + 60 * .1 = 16.0 against 250 * .04 + 60 * .1 / 1.3 = 14.615
    assert abs(out[('Runner', 'QB', 'AAA')][0] - 16.0 / (10.0 + 6.0 / 1.3)) < 1e-9
    assert out[('Runner', 'QB', 'AAA')][1] > out[('Pocket', 'QB', 'BBB')][1]


def test_a_slot_receiver_and_a_wide_receiver_get_their_own_allowed_multiplier():
    slot = {'receptions': _stat(6.0, 1.12), 'receiving_yards': _stat(60.0, 1.12)}
    wide = {'receptions': _stat(6.0, 0.94), 'receiving_yards': _stat(60.0, 0.94)}
    meta = _meta(('Slot', 'WR', 'AAA', slot), ('Wide', 'WR', 'BBB', wide))
    out = rk._role_matchup_by_player(meta, _board(('Slot', 'WR', 'AAA', 12.0), ('Wide', 'WR', 'BBB', 12.0)), 'Full PPR')
    assert abs(out[('Slot', 'WR', 'AAA')][0] - 1.12) < 1e-9 and abs(out[('Wide', 'WR', 'BBB')][0] - 0.94) < 1e-9
    assert out[('Slot', 'WR', 'AAA')][1] > out[('Wide', 'WR', 'BBB')][1]


def test_a_back_who_catches_passes_weighs_the_receiving_multiplier_by_the_scoring():
    back = {'rushing_yards': _stat(50.0, 0.90), 'receptions': _stat(5.0, 1.20), 'receiving_yards': _stat(40.0, 1.20)}
    meta = _meta(('Back', 'RB', 'AAA', back))
    full = rk._role_matchup_by_player(meta, _board(('Back', 'RB', 'AAA', 14.0)), 'Full PPR')[('Back', 'RB', 'AAA')][0]
    standard = rk._role_matchup_by_player(meta, _board(('Back', 'RB', 'AAA', 14.0)), 'Standard')[('Back', 'RB', 'AAA')][0]
    assert full > standard                      # a reception is worth a point only in PPR, so the receiving side weighs more


def test_a_defense_that_forces_interceptions_lowers_a_qbs_number():
    base = {'passing_yards': _stat(250.0, 1.0), 'passing_interceptions': _stat(0.8, 1.0)}
    picky = {'passing_yards': _stat(250.0, 1.0), 'passing_interceptions': _stat(0.8, 1.4)}
    meta = _meta(('Base', 'QB', 'AAA', base), ('Picky', 'QB', 'BBB', picky))
    out = rk._role_matchup_by_player(meta, _board(('Base', 'QB', 'AAA', 18.0), ('Picky', 'QB', 'BBB', 18.0)), 'Full PPR')
    assert out[('Picky', 'QB', 'BBB')][0] < 1.0 == out[('Base', 'QB', 'AAA')][0]


def test_the_percentile_ranks_players_of_the_same_position_who_matter():
    meta = _meta(*[(f'WR{i}', 'WR', 'AAA', {'receiving_yards': _stat(60.0, 0.9 + 0.05 * i)}) for i in range(5)],
                 ('Deep', 'WR', 'AAA', {'receiving_yards': _stat(8.0, 1.4)}),     # 0.8 points: left out of the pool
                 ('Back', 'RB', 'AAA', {'rushing_yards': _stat(60.0, 1.4)}))
    board = _board(*[(f'WR{i}', 'WR', 'AAA', 8.0) for i in range(5)], ('Deep', 'WR', 'AAA', 1.0), ('Back', 'RB', 'AAA', 8.0))
    out = rk._role_matchup_by_player(meta, board, 'Full PPR')
    pcts = [out[(f'WR{i}', 'WR', 'AAA')][1] for i in range(5)]
    assert pcts == sorted(pcts) and pcts[0] < 15 and pcts[-1] > 85
    assert out[('Deep', 'WR', 'AAA')][1] == 100.0                  # above everyone who matters
    assert 40 < out[('Back', 'RB', 'AAA')][1] <= 100               # a one-man pool reads itself


def test_a_player_with_no_scored_projection_has_no_multiplier():
    out = rk._role_matchup_by_player(_meta(('Nobody', 'WR', 'AAA', {'targets': _stat(5.0, 1.2)})), _board(), 'Full PPR')
    assert out == {}


def test_the_games_view_tone_is_the_players_own_percentile():
    board = _board(('A', 'QB', 'AAA', 20.0), ('B', 'QB', 'BBB', 20.0), ('C', 'QB', 'CCC', 20.0))
    board['Opponent'] = 'XXX'                                  # one opponent for all of them: the old key gave one number
    board['Availability'] = 1.0
    for stat in ('passing_yards',):
        board[stat] = 250.0
    out = rankings_games.games_board(board, None, {('A', 'QB', 'AAA'): (1.1, 90.0), ('B', 'QB', 'BBB'): (0.9, 10.0)},
                                     lambda *a, **k: a[0])
    assert list(out['Matchup Pct'].iloc[:2]) == [90.0, 10.0] and np.isnan(out['Matchup Pct'].iloc[2])


# ---- 6. the Deep Dive game log ---------------------------------------------------------------------------------------------

def test_result_and_score_are_one_cell():
    assert rk._result_score('W', '59-37') == 'W 59-37'
    assert rk._result_score('', '10-24') == 'L 10-24'          # no result from the schedule: read off the score
    assert rk._result_score('', '17-17') == 'T 17-17'
    assert rk._result_score('W', rk.NO_SCORE) == rk.NO_SCORE


def test_allowed_tone_is_green_when_the_defense_allowed_more_than_the_offenses_own_average():
    tone = rk._allowed_tone([12.0, 8.0, 10.0, 3.0, 5.0], [10.0, 10.0, 10.0, 0.0, np.nan])
    assert tone[0] > 1.0 > tone[1] and abs(tone[2] - 1.0) < 1e-9
    assert tone[3] is None and tone[4] is None                  # no baseline to compare with: no colour
    assert 0.8 < min(t for t in rk._allowed_tone([0.0, 100.0], [10.0, 10.0])) and max(rk._allowed_tone([0.0, 100.0], [10.0, 10.0])) < 1.2


def test_matchup_blend_text_says_the_weights_and_nothing_else():
    assert rk._matchup_blend_text({'current_weight': 0.37}, 2026) == '2026 37% + 2025 63%'
    assert rk._matchup_blend_text({'current_weight': 1.0}, 2026) == '2026 only'
    assert rk._matchup_blend_text({'current_weight': 0.0}, 2026) == '2025 only (no 2026 games yet)'
    assert rk._matchup_blend_text({'source': 'old text'}, 2026) == 'old text'


def _games(**extra):
    rows = [{'name': 'P', 'team': 'AAA', 'week': 1, 'opponent_team': 'XXX', 'passing_yards': 200.0, 'passing_interceptions': 1.0,
             'weekly_snap_pct': 98.0, 'game_id': '2026_01_XXX_AAA'},
            {'name': 'P', 'team': 'AAA', 'week': 2, 'opponent_team': 'YYY', 'passing_yards': 300.0, 'passing_interceptions': 0.0,
             'weekly_snap_pct': 90.0, 'game_id': '2026_02_AAA_YYY'}]
    return pd.DataFrame(rows)


def test_game_log_carries_the_opponents_difficulty_percentile_and_the_teams_plays():
    matrix = pd.DataFrame({'passing_yards': [1.2, 0.8, 1.0], 'passing_interceptions': [1.3, 0.7, 1.0]},
                          index=['XXX', 'YYY', 'ZZZ'])
    plays = pd.DataFrame({'team': ['AAA', 'AAA'], 'week': [1, 2], 'plays': [71.0, 58.0]})
    log = wp._build_player_stat_game_log(_games(), 'name', ['passing_yards', 'passing_interceptions'], matrix,
                                         team_col='team', team_plays=plays)['P']
    assert list(log['_matchup_pct_passing_yards']) == [100.0, 0.0]          # XXX the softest, YYY the toughest
    assert list(log['_matchup_pct_passing_interceptions']) == [0.0, 100.0]  # more interceptions forced = the TOUGHER matchup
    assert list(log['_team_plays']) == [71.0, 58.0]
    assert list(log['weekly_snap_pct']) == [98.0, 90.0]


def test_game_log_without_plays_or_a_matrix_still_builds():
    log = wp._build_player_stat_game_log(_games(), 'name', ['passing_yards'], pd.DataFrame(), team_col='team')['P']
    assert log['_team_plays'].isna().all() and '_matchup_pct_passing_yards' not in log.columns


# ---- 2/3. every table is centered -------------------------------------------------------------------------------------------

def test_centered_column_config_adds_alignment_and_respects_what_is_there():
    import streamlit as st
    frame = pd.DataFrame({'a': [1], 'b': ['x'], 'c': [2], 'd': [3], 'e': [4], 'f': [5]})
    config = {'a': st.column_config.NumberColumn(format='%.1f'), 'b': 'Bee', 'c': st.column_config.Column(alignment='right'),
              'd': st.column_config.ProgressColumn(min_value=0, max_value=9), 'e': None}
    out = styling._centered_column_config(frame, config)
    assert out['a']['alignment'] == 'center' and out['a']['type_config']['format'] == '%.1f'
    assert out['b'] == {'label': 'Bee', 'alignment': 'center'}
    assert out['c']['alignment'] == 'right'                                    # a chosen alignment stays
    assert 'alignment' not in out['d']                                        # a type with no alignment option is left alone
    assert out['e'] is None                                                   # a hidden column stays hidden
    assert out['f'] == {'alignment': 'center'}                                # no entry: one is made


def test_centered_column_config_reads_a_styler_and_leaves_the_index_alone():
    frame = pd.DataFrame({'a': [1.0]}, index=pd.Index(['Player One'], name='Player'))
    out = styling._centered_column_config(frame.style, None)
    assert out == {'a': {'alignment': 'center'}}


def test_st_dataframe_is_wrapped_once_and_still_passes_everything_through(monkeypatch):
    import streamlit as st
    seen = {}
    monkeypatch.setattr(st, 'dataframe', lambda data=None, *a, **k: seen.update(data=data, a=a, k=k))
    monkeypatch.setattr(st, 'html', lambda *a, **k: None)
    styling.install_table_defaults()
    wrapped = st.dataframe
    styling.install_table_defaults()
    assert st.dataframe is wrapped                                            # not wrapped twice
    frame = pd.DataFrame({'a': [1]})
    st.dataframe(frame, width='stretch', height=120, hide_index=True)
    assert seen['k']['column_config'] == {'a': {'alignment': 'center'}} and seen['k']['hide_index'] is True
    assert seen['k']['height'] == 120 and seen['data'] is frame


# ---- the rendered Deep Dive player log ----------------------------------------------------------------------------------------

def test_deep_dive_player_log_columns_and_average_row(monkeypatch):
    captured = []
    monkeypatch.setattr(rk, 'render_game_table', lambda styled, entries, **kw: captured.append((styled, kw)))
    monkeypatch.setattr(rk, 'score_entries', lambda *a, **k: [None, None, None])
    monkeypatch.setattr(rk, 'score_texts', lambda entries: ['—'] * len(entries))
    monkeypatch.setattr(rk.st, 'markdown', lambda *a, **k: None)
    monkeypatch.setattr(rk.st, 'caption', lambda *a, **k: None)
    games = [
        {'week': 1, 'opponent_team': 'HOU', '_team_score': 36, '_opp_score': 31, '_result': 'W', 'passing_yards': 250.0,
         '_defadj_passing_yards': 240.0, '_matchup_pct_passing_yards': 65.0, 'weekly_snap_pct': 100.0, '_team_plays': 52.0},
        {'week': 2, 'opponent_team': 'DET', '_team_score': 26, '_opp_score': 29, '_result': 'L', 'passing_yards': 300.0,
         '_defadj_passing_yards': 310.0, '_matchup_pct_passing_yards': 25.0, 'weekly_snap_pct': 80.0, '_team_plays': 70.0},
        {'week': 3, 'opponent_team': 'LAC', '_team_score': 10, '_opp_score': 10, '_result': 'T', 'passing_yards': 200.0,
         '_defadj_passing_yards': 200.0, '_matchup_pct_passing_yards': 45.0, 'weekly_snap_pct': 0.0, '_team_plays': 60.0},
    ]
    detail = {'player': 'P', 'team': 'AAA', 'opponent': 'BBB', 'target_week': 4, 'as_of_week': 4, 'season_year': 2026,
              'stats': {'passing_yards': {}}}
    rk._render_stat_deep_dive(detail, 'passing_yards', {2026: games}, {}, 2026)
    styled, kwargs = captured[0]
    frame = styled.data
    assert list(frame.columns) == ['Week', 'Opponent', 'Score', 'Matchup %', 'Snap %', 'Team plays', 'Raw Passing Yards',
                                   'Defense-adj Passing Yards', 'Included']
    assert list(frame['Score'][:3]) == ['W 36-31', 'L 26-29', 'T 10-10']             # the result leads the score
    assert list(frame['Matchup %'][:3]) == ['65', '25', '45'] and frame['Matchup %'].iloc[-1] == '45'
    assert list(frame['Snap %'][:3]) == ['100', '80', '—']                         # a recorded 0 is no data
    assert list(frame['Team plays'][:3]) == ['52', '70', '60'] and frame['Team plays'].iloc[-1] == '61'
    assert frame['Snap %'].iloc[-1] == '90' and frame['Week'].iloc[-1] == 'AVG'
    assert set(kwargs['column_config']) == {'Matchup %', 'Snap %', 'Team plays'}


def test_deep_dive_player_log_without_the_new_fields_keeps_the_old_layout(monkeypatch):
    captured = []
    monkeypatch.setattr(rk, 'render_game_table', lambda styled, entries, **kw: captured.append(styled))
    monkeypatch.setattr(rk, 'score_entries', lambda *a, **k: [None])
    monkeypatch.setattr(rk, 'score_texts', lambda entries: ['—'] * len(entries))
    monkeypatch.setattr(rk.st, 'markdown', lambda *a, **k: None)
    monkeypatch.setattr(rk.st, 'caption', lambda *a, **k: None)
    games = [{'week': 1, 'opponent_team': 'HOU', '_team_score': 36, '_opp_score': 31, '_result': 'W', 'passing_yards': 250.0,
              '_defadj_passing_yards': 240.0}]
    detail = {'player': 'P', 'team': 'AAA', 'opponent': 'BBB', 'target_week': 4, 'as_of_week': 4, 'season_year': 2026,
              'stats': {'passing_yards': {}}}
    rk._render_stat_deep_dive(detail, 'passing_yards', {2026: games}, {}, 2026)
    assert list(captured[0].data.columns) == ['Week', 'Opponent', 'Score', 'Raw Passing Yards', 'Defense-adj Passing Yards',
                                              'Included']


# ---- quiet_note: a dim dropdown, safe to repeat and to nest ----------------------------------------------------------------

class _NoteSt:
    def __init__(self, refuse_expanders=False):
        self.session_state = {}
        self.refuse = refuse_expanders
        self.keys, self.markdown_calls = [], []

    def expander(self, label, expanded=False, key=None, **kwargs):
        if self.refuse:
            raise RuntimeError('Expanders may not be nested inside other expanders.')
        self.keys.append(key)
        import contextlib
        return contextlib.nullcontext()

    def markdown(self, text, *a, **k):
        self.markdown_calls.append((text, k))

    def caption(self, text, *a, **k):
        self.markdown_calls.append((text, {}))


def test_quiet_note_keys_are_unique_when_the_same_note_repeats(monkeypatch):
    import ui.components as components
    fake = _NoteSt()
    monkeypatch.setattr(components, 'st', fake)
    for _ in range(3):
        components.quiet_note('The same long explanation.', 'Details')
    assert len(set(fake.keys)) == 3 and all(k.startswith('quiet_note_') for k in fake.keys)


def test_quiet_note_inside_an_expander_becomes_a_plain_details_block(monkeypatch):
    import ui.components as components
    fake = _NoteSt(refuse_expanders=True)
    monkeypatch.setattr(components, 'st', fake)
    components.quiet_note('Explain **this**.', 'What <b>it</b> means')
    text, kwargs = fake.markdown_calls[0]
    assert text.startswith('<details><summary>What &lt;b&gt;it&lt;/b&gt; means</summary>') and 'Explain **this**.' in text
    assert kwargs == {'unsafe_allow_html': True}
