"""v2_point_in_time_roster (backlog D6 / N4): a backtest reads the roster as of the target week."""
import numpy as np
import pandas as pd

import data.weekly_projections as wp


WEEKLY = pd.DataFrame([
    # week 4: Cooper on CLE; week 5 rows below are the ones read for a week-5 target
    {'season': 2024, 'week': 4, 'team': 'CLE', 'full_name': 'Amari Cooper', 'gsis_id': '00-1', 'position': 'WR', 'status': 'ACT'},
    {'season': 2024, 'week': 5, 'team': 'CLE', 'full_name': 'Amari Cooper', 'gsis_id': '00-1', 'position': 'WR', 'status': 'ACT'},
    {'season': 2024, 'week': 5, 'team': 'MIN', 'full_name': 'T.J. Hockenson', 'gsis_id': '00-2', 'position': 'TE', 'status': 'RES'},
    {'season': 2024, 'week': 5, 'team': 'NYG', 'full_name': 'Daniel Jones', 'gsis_id': '00-3', 'position': 'QB', 'status': 'INA'},
    {'season': 2024, 'week': 5, 'team': 'KC', 'full_name': 'Kareem Hunt', 'gsis_id': '00-4', 'position': 'RB', 'status': 'DEV'},
    # two offensive players share a name: no name match for either
    {'season': 2024, 'week': 5, 'team': 'BUF', 'full_name': 'Josh Allen', 'gsis_id': '00-5', 'position': 'QB', 'status': 'ACT'},
    {'season': 2024, 'week': 5, 'team': 'JAX', 'full_name': 'Josh Allen', 'gsis_id': '00-6', 'position': 'WR', 'status': 'ACT'},
])


def _pit(monkeypatch, week=5):
    monkeypatch.setattr(wp, 'load_weekly_rosters', lambda season: WEEKLY)
    return wp.point_in_time_roster(2024, week)


def test_flag_is_a_candidate():
    assert 'v2_point_in_time_roster' in wp.MODEL_FEATURES and 'v2_point_in_time_roster' not in wp.DEFAULT_FEATURES


def test_point_in_time_roster_reads_only_the_week_and_ina_reads_active(monkeypatch):
    pit = _pit(monkeypatch).set_index('gsis_id')
    assert len(pit) == 6
    assert pit.loc['00-1', 'team'] == 'CLE'
    assert pit.loc['00-3', 'status'] == 'ACT'      # gameday inactive: announced after the board
    assert pit.loc['00-2', 'status'] == 'RES'
    monkeypatch.setattr(wp, 'load_weekly_rosters', lambda season: pd.DataFrame())
    assert wp.point_in_time_roster(2024, 5).empty


def test_lookup_prefers_the_id_and_skips_shared_names(monkeypatch):
    pit = _pit(monkeypatch)
    frame = pd.DataFrame({'player_display_name': ['Amari Cooper', 'Daniel Jones', 'Josh Allen', 'Josh Allen', 'Nobody'],
                          'player_id': ['00-1', '', '00-5', '', '']})
    team, status = wp._point_in_time_lookup(frame, 'player_display_name', pit)
    assert team.tolist()[:3] == ['CLE', 'NYG', 'BUF']   # id; unique name; id beats the shared name
    assert pd.isna(team.iloc[3]) and pd.isna(team.iloc[4])
    assert status.iloc[1] == 'ACT'


def test_apply_keeps_each_past_game_on_its_team_and_later_rows_on_the_week_roster(monkeypatch):
    pit = _pit(monkeypatch)
    stats = pd.DataFrame({'player_display_name': ['Amari Cooper'] * 3 + ['Daniel Jones'] * 2,
                          'player_id': ['00-1'] * 3 + ['00-3'] * 2,
                          'week': [3, 4, 8, 4, np.nan],
                          'team': ['BUF', 'BUF', 'BUF', 'MIN', 'MIN'],          # the merge's end-of-season team
                          'game_team': ['CLE', 'CLE', 'BUF', 'NYG', ''],
                          'status': ['ACT', 'ACT', 'ACT', 'CUT', 'CUT']})
    out = wp.apply_point_in_time_roster(stats, 'player_display_name', 'team', 5, pit)
    assert out['team'].tolist() == ['CLE', 'CLE', 'CLE', 'NYG', 'NYG']
    assert out['status'].tolist() == ['ACT', 'ACT', 'ACT', 'ACT', 'ACT']
    assert stats['team'].tolist() == ['BUF', 'BUF', 'BUF', 'MIN', 'MIN']   # the input is not changed


def test_placeholder_needs_the_player_on_the_week_roster(monkeypatch):
    pit = _pit(monkeypatch)
    name, team = 'player_display_name', 'team'
    roster = pd.DataFrame({name: ['T.J. Hockenson', 'Kareem Hunt', 'Amari Cooper', 'Gone Guy'],
                           'player_id': ['00-2', '00-4', '00-1', '00-9'],
                           'position': ['TE', 'RB', 'WR', 'WR'], team: ['MIN', 'KC', 'BUF', 'SEA'],
                           'status': ['ACT', 'ACT', 'ACT', 'ACT'], 'week': [9, 4, np.nan, 12]})
    prior = pd.DataFrame({name: ['T.J. Hockenson', 'Kareem Hunt', 'Amari Cooper', 'Gone Guy'],
                          'player_id': ['00-2', '00-4', '00-1', '00-9'],
                          'position': ['TE', 'RB', 'WR', 'WR'], 'week': [1, 1, 1, 1]})
    cur = pd.DataFrame(columns=[name, 'Team', 'Games'])
    rows = {}
    for pos in ('TE', 'RB', 'WR'):
        off = wp._add_missing_returning_player_rows_core(cur, roster, prior, name, name, team, [], pos)
        on = wp._add_missing_returning_player_rows_core(cur, roster, prior, name, name, team, [], pos, point_in_time=pit)
        rows[pos] = (off, on)
    assert len(rows['TE'][0]) == 1 and rows['TE'][1].empty          # on reserve in week 5
    assert len(rows['RB'][0]) == 1 and rows['RB'][1].empty          # practice squad in week 5
    off_wr, on_wr = rows['WR']
    assert set(off_wr[name]) == {'Amari Cooper', 'Gone Guy'}
    assert on_wr[name].tolist() == ['Amari Cooper'] and on_wr['Team'].tolist() == ['CLE']
