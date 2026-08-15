"""
Offline tests for the box score and the game-link resolver
(data/box_score.py).

Split the same way the porting notes recommend: the RESOLVER tests protect
against a link that navigates to the wrong game (silent, and worse than no
link), and the BOX tests protect against numbers that are quietly wrong -
a total that double-counts, a team's rows attributed to its opponent, a
`nan` printed where a stat goes. None of these raise. All of them render.

Runs two ways: `python tests/test_box_score.py` needs nothing but the app's
own dependencies, and `pytest tests/` works if pytest is installed.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

from data.box_score import (  # noqa: E402
    PASSING_COLS, RECEIVING_COLS, game_link_rows, game_players, leading_performers,
    section_rows, team_totals,
)


def players(rows):
    frame = pd.DataFrame(rows)
    for col in ('passing_yards', 'passing_attempts', 'passing_completions', 'passing_tds',
                'passing_interceptions', 'passing_first_downs', 'rushing_yards',
                'rushing_attempts', 'rushing_tds', 'rushing_first_downs', 'receiving_yards',
                'receptions', 'targets', 'receiving_tds', 'receiving_first_downs',
                'sacks_suffered', 'penalties', 'penalty_yards', 'fumbles_lost_total'):
        if col not in frame.columns:
            frame[col] = 0.0
    return frame


def slate(rows):
    frame = pd.DataFrame(rows)
    for col, default in (('Neutral Site', False), ('Played', True), ('Winner', None),
                         ('Away Pts', None), ('Home Pts', None), ('Date Display', '')):
        if col not in frame.columns:
            frame[col] = default
    return frame


# --- the join ----------------------------------------------------------

def test_game_id_is_compared_as_a_string_on_both_sides():
    # The two real sources agree on the type today. This comparison failing
    # returns an EMPTY BOX WITH NO ERROR, which is the single most
    # expensive bug in the port this was translated from, so the cast is
    # pinned rather than trusted.
    frame = players([{'game_id': 2025010, 'team': 'KC', 'player_display_name': 'A', 'position': 'QB'}])
    assert len(game_players(frame, '2025010')) == 1
    assert len(game_players(frame, 2025010)) == 1


def test_a_missing_game_returns_empty_not_an_error():
    frame = players([{'game_id': 'g1', 'team': 'KC', 'player_display_name': 'A', 'position': 'QB'}])
    assert game_players(frame, 'nope').empty


# --- team totals -------------------------------------------------------

def test_first_downs_do_not_double_count_a_completed_pass():
    # A completion credits the QB with a passing first down AND the
    # receiver with a receiving first down. Summing every *_first_downs
    # column reports 9 for what is really 6.
    frame = players([
        {'game_id': 'g', 'team': 'KC', 'player_display_name': 'QB', 'position': 'QB',
         'passing_first_downs': 4},
        {'game_id': 'g', 'team': 'KC', 'player_display_name': 'WR', 'position': 'WR',
         'receiving_first_downs': 3},
        {'game_id': 'g', 'team': 'KC', 'player_display_name': 'RB', 'position': 'RB',
         'rushing_first_downs': 2},
    ])
    assert team_totals(frame, 'KC')['First Downs'] == 6.0


def test_totals_only_count_the_named_team():
    frame = players([
        {'game_id': 'g', 'team': 'KC', 'player_display_name': 'A', 'position': 'QB', 'passing_yards': 300},
        {'game_id': 'g', 'team': 'BUF', 'player_display_name': 'B', 'position': 'QB', 'passing_yards': 250},
    ])
    assert team_totals(frame, 'KC')['Pass Yds'] == 300.0
    assert team_totals(frame, 'BUF')['Pass Yds'] == 250.0


def test_total_yards_is_passing_plus_rushing():
    frame = players([
        {'game_id': 'g', 'team': 'KC', 'player_display_name': 'A', 'position': 'QB', 'passing_yards': 300},
        {'game_id': 'g', 'team': 'KC', 'player_display_name': 'B', 'position': 'RB', 'rushing_yards': 100},
        # Receiving yards must NOT be added - they are the same yards as
        # the passing yards, counted from the other end.
        {'game_id': 'g', 'team': 'KC', 'player_display_name': 'C', 'position': 'WR', 'receiving_yards': 200},
    ])
    assert team_totals(frame, 'KC')['Total Yds'] == 400.0


def test_turnovers_are_interceptions_plus_lost_fumbles():
    frame = players([
        {'game_id': 'g', 'team': 'KC', 'player_display_name': 'A', 'position': 'QB',
         'passing_interceptions': 2, 'fumbles_lost_total': 1},
    ])
    assert team_totals(frame, 'KC')['Turnovers'] == 3.0


# --- section tables ----------------------------------------------------

def test_a_section_excludes_players_with_nothing_in_it():
    frame = players([
        {'game_id': 'g', 'team': 'KC', 'player_display_name': 'Targeted', 'position': 'WR', 'targets': 5},
        {'game_id': 'g', 'team': 'KC', 'player_display_name': 'Blanked', 'position': 'WR', 'targets': 0},
    ])
    table = section_rows(frame, 'KC', RECEIVING_COLS, 'targets')
    assert list(table['Player']) == ['Targeted']


def test_a_section_is_sorted_by_its_headline_stat():
    frame = players([
        {'game_id': 'g', 'team': 'KC', 'player_display_name': 'Few', 'position': 'WR', 'targets': 2},
        {'game_id': 'g', 'team': 'KC', 'player_display_name': 'Many', 'position': 'WR', 'targets': 11},
    ])
    assert list(section_rows(frame, 'KC', RECEIVING_COLS, 'targets')['Player']) == ['Many', 'Few']


def test_a_section_never_prints_nan():
    frame = players([{'game_id': 'g', 'team': 'KC', 'player_display_name': 'A',
                      'position': 'QB', 'passing_attempts': 30, 'passing_yards': None}])
    table = section_rows(frame, 'KC', PASSING_COLS, 'passing_attempts')
    assert 'nan' not in table.to_string().lower()
    assert table['YDS'].iloc[0] == 0.0


def test_leaders_skip_a_phase_the_team_did_nothing_in():
    frame = players([{'game_id': 'g', 'team': 'KC', 'player_display_name': 'A',
                      'position': 'QB', 'passing_yards': 300, 'passing_attempts': 40,
                      'passing_completions': 28, 'passing_tds': 3}])
    phases = [entry['phase'] for entry in leading_performers(frame, 'KC')]
    # A zero-yard rushing "leader" is noise, not a leader.
    assert phases == ['Passing']


# --- the resolver ------------------------------------------------------

SLATE = slate([
    {'Game Id': '2025_01_KC_BUF', 'Away': 'KC', 'Home': 'BUF', 'Week': 1,
     'Away Pts': 27.0, 'Home Pts': 24.0, 'Winner': 'KC', 'Date Display': '09/07/2025'},
    {'Game Id': '2025_02_DEN_KC', 'Away': 'DEN', 'Home': 'KC', 'Week': 2,
     'Away Pts': 10.0, 'Home Pts': 30.0, 'Winner': 'KC', 'Date Display': '09/14/2025'},
])


def test_resolves_by_game_id():
    log = pd.DataFrame({'game_id': ['2025_01_KC_BUF'], 'week': [1]})
    assert [e['game_id'] for e in game_link_rows(log, SLATE, team='KC')] == ['2025_01_KC_BUF']


def test_falls_back_to_week_plus_team():
    # No id at all. In the NFL a team plays exactly one game per week, so
    # this fallback is exact - unlike the date+team fallback in the sport
    # this was ported from, where a date matches ~150 games.
    log = pd.DataFrame({'week': [2]})
    assert [e['game_id'] for e in game_link_rows(log, SLATE, team='KC')] == ['2025_02_DEN_KC']


def test_the_week_fallback_requires_a_team():
    log = pd.DataFrame({'week': [2]})
    assert game_link_rows(log, SLATE, team=None) == []


def test_an_unresolvable_row_is_dropped_not_emitted_with_a_null_id():
    log = pd.DataFrame({'game_id': ['not_a_game'], 'week': [99]})
    # A dead link that navigates nowhere is worse than no link.
    assert game_link_rows(log, SLATE, team='KC') == []


def test_duplicate_rows_collapse_to_one_chip():
    log = pd.DataFrame({'game_id': ['2025_01_KC_BUF', '2025_01_KC_BUF'], 'week': [1, 1]})
    assert len(game_link_rows(log, SLATE, team='KC')) == 1


def test_limit_keeps_the_most_recent_games():
    log = pd.DataFrame({'game_id': ['2025_01_KC_BUF', '2025_02_DEN_KC'], 'week': [1, 2]})
    kept = game_link_rows(log, SLATE, team='KC', limit=1)
    # A strip under a game log wants the LAST games, not the first.
    assert [e['week'] for e in kept] == [2]


def test_away_and_home_labels_differ_and_read_cleanly():
    log = pd.DataFrame({'game_id': ['2025_01_KC_BUF', '2025_02_DEN_KC'], 'week': [1, 2]})
    labels = [e['label'] for e in game_link_rows(log, SLATE, team='KC')]
    assert labels == ['W1 @BUF', 'W2 vs DEN']
    # With the space. "vsDEN" is unreadable.
    assert 'vs DEN' in labels[1]


def test_win_flag_is_from_the_subject_teams_point_of_view():
    log = pd.DataFrame({'game_id': ['2025_01_KC_BUF'], 'week': [1]})
    assert game_link_rows(log, SLATE, team='KC')[0]['won'] is True
    assert game_link_rows(log, SLATE, team='BUF')[0]['won'] is False


def test_an_unplayed_game_has_no_win_flag():
    unplayed = slate([{'Game Id': 'g', 'Away': 'KC', 'Home': 'BUF', 'Week': 5,
                       'Played': False, 'Winner': None}])
    log = pd.DataFrame({'game_id': ['g'], 'week': [5]})
    assert game_link_rows(log, unplayed, team='KC')[0]['won'] is None


def test_a_neutral_site_game_is_never_an_away_game():
    neutral = slate([{'Game Id': 'g', 'Away': 'KC', 'Home': 'BUF', 'Week': 5,
                      'Neutral Site': True, 'Winner': 'KC', 'Away Pts': 20.0, 'Home Pts': 10.0}])
    log = pd.DataFrame({'game_id': ['g'], 'week': [5]})
    assert game_link_rows(log, neutral, team='KC')[0]['label'] == 'W5 vs BUF'


def test_a_label_never_contains_nan():
    messy = slate([{'Game Id': 'g', 'Away': 'KC', 'Home': 'BUF', 'Week': 5,
                    'Played': True, 'Winner': None, 'Away Pts': None, 'Home Pts': None,
                    'Date Display': None}])
    log = pd.DataFrame({'game_id': ['g'], 'week': [5]})
    entry = game_link_rows(log, messy, team='KC')[0]
    assert 'nan' not in entry['label'].lower()
    assert 'nan' not in entry['help'].lower()


def test_a_slate_without_the_id_column_resolves_nothing():
    # Rather than raising - an older cached slate frame predating the
    # column must degrade to "no links", not take the tab down.
    bare = pd.DataFrame({'Away': ['KC'], 'Home': ['BUF'], 'Week': [1]})
    assert game_link_rows(pd.DataFrame({'week': [1]}), bare, team='KC') == []


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failures.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
