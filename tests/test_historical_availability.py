"""data.historical_availability: time-valid injury replay for backtesting."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.historical_availability as ha  # noqa: E402


def _reports(rows):
    return pd.DataFrame(rows)


def _schedule():
    # BAL hosts DEN at 2023-09-24 13:00 ET (17:00 UTC); CAR at 20:15 ET (2023-09-25 00:15 UTC).
    return pd.DataFrame([
        {'week': 3, 'home_team': 'BAL', 'away_team': 'DEN', 'gameday': '2023-09-24', 'gametime': '13:00'},
        {'week': 3, 'home_team': 'CAR', 'away_team': 'NYG', 'gameday': '2023-09-24', 'gametime': '20:15'},
    ])


def test_shape_matches_the_live_injury_profiles_contract(monkeypatch):
    reports = _reports([
        {'week': 3, 'team': 'BAL', 'gsis_id': '00-1', 'full_name': 'A Back',
         'report_status': 'Out', 'date_modified': '2023-09-22T18:00:00Z'},
        {'week': 3, 'team': 'BAL', 'gsis_id': '00-2', 'full_name': 'B Wideout',
         'report_status': 'Questionable', 'date_modified': '2023-09-22T18:00:00Z'},
    ])
    monkeypatch.setattr(ha, 'load_injury_reports', lambda season: reports)
    out = ha.historical_injury_profiles(2023, 3, _schedule())
    assert set(out) == {'A Back', 'B Wideout'}
    row = out['A Back']
    assert set(row) == {'status', 'plays_probability', 'workload_if_active',
                        'source_year', 'source', 'gsis_id', 'team'}
    assert row['plays_probability'] == 0.0 and row['workload_if_active'] == 1.0
    assert out['B Wideout']['plays_probability'] == 1.0   # Questionable -> healthy, same live policy


def test_out_doubtful_and_ir_are_unavailable_everything_else_is_healthy(monkeypatch):
    reports = _reports([
        {'week': 3, 'team': 'BAL', 'gsis_id': f'00-{i}', 'full_name': f'Player {i}',
         'report_status': status, 'date_modified': '2023-09-22T18:00:00Z'}
        for i, status in enumerate(['Out', 'Doubtful', 'IR', 'Questionable', None, 'Probable'])
    ])
    monkeypatch.setattr(ha, 'load_injury_reports', lambda season: reports)
    out = ha.historical_injury_profiles(2023, 3, _schedule())
    assert out['Player 0']['plays_probability'] == 0.0    # Out
    assert out['Player 1']['plays_probability'] == 0.0    # Doubtful
    assert out['Player 2']['plays_probability'] == 0.0    # IR
    assert out['Player 3']['plays_probability'] == 1.0    # Questionable
    assert out['Player 4']['plays_probability'] == 1.0    # blank/no designation
    assert out['Player 5']['plays_probability'] == 1.0    # Probable (not an out-status)


def test_a_row_filed_after_that_teams_kickoff_is_dropped(monkeypatch):
    # BAL's game kicks off 2023-09-24 13:00 ET = 17:00 UTC.
    reports = _reports([
        {'week': 3, 'team': 'BAL', 'gsis_id': '00-1', 'full_name': 'Pregame Scratch',
         'report_status': 'Out', 'date_modified': '2023-09-24T16:00:00Z'},   # 1h before kickoff - fine
        {'week': 3, 'team': 'BAL', 'gsis_id': '00-2', 'full_name': 'Postgame Note',
         'report_status': 'Out', 'date_modified': '2023-09-24T20:00:00Z'},   # 3h after kickoff - leak
    ])
    monkeypatch.setattr(ha, 'load_injury_reports', lambda season: reports)
    out = ha.historical_injury_profiles(2023, 3, _schedule())
    assert 'Pregame Scratch' in out
    assert 'Postgame Note' not in out


def test_a_team_with_no_resolvable_kickoff_keeps_its_rows(monkeypatch):
    # No schedule at all -> can't build a kickoff guard; degrade to "keep everything"
    # rather than silently dropping every row for a bye-week/unscheduled team.
    reports = _reports([
        {'week': 3, 'team': 'BAL', 'gsis_id': '00-1', 'full_name': 'A Back',
         'report_status': 'Out', 'date_modified': '2023-09-24T23:00:00Z'},
    ])
    monkeypatch.setattr(ha, 'load_injury_reports', lambda season: reports)
    out = ha.historical_injury_profiles(2023, 3, pd.DataFrame())
    assert 'A Back' in out


def test_duplicate_rows_for_one_player_keep_the_most_recently_modified(monkeypatch):
    reports = _reports([
        {'week': 3, 'team': 'BAL', 'gsis_id': '00-1', 'full_name': 'A Back',
         'report_status': 'Questionable', 'date_modified': '2023-09-22T12:00:00Z'},
        {'week': 3, 'team': 'BAL', 'gsis_id': '00-1', 'full_name': 'A Back',
         'report_status': 'Out', 'date_modified': '2023-09-24T15:00:00Z'},  # final, pre-kickoff
    ])
    monkeypatch.setattr(ha, 'load_injury_reports', lambda season: reports)
    out = ha.historical_injury_profiles(2023, 3, _schedule())
    assert out['A Back']['status'] == 'out'
    assert out['A Back']['plays_probability'] == 0.0


def test_empty_or_missing_week_returns_empty_dict(monkeypatch):
    monkeypatch.setattr(ha, 'load_injury_reports', lambda season: pd.DataFrame())
    assert ha.historical_injury_profiles(2023, 3, _schedule()) == {}

    reports = _reports([{'week': 5, 'team': 'BAL', 'gsis_id': '00-1', 'full_name': 'A Back',
                         'report_status': 'Out', 'date_modified': '2023-10-08T15:00:00Z'}])
    monkeypatch.setattr(ha, 'load_injury_reports', lambda season: reports)
    assert ha.historical_injury_profiles(2023, 3, _schedule()) == {}


def test_never_raises_on_a_broken_source(monkeypatch):
    def _boom(season):
        raise RuntimeError('feed down')
    monkeypatch.setattr(ha, 'load_injury_reports', _boom)
    assert ha.historical_injury_profiles(2023, 3, _schedule()) == {}


def test_significant_out_players_requires_a_real_starters_snap_share(monkeypatch):
    reports = _reports([
        {'week': 3, 'team': 'BAL', 'gsis_id': '00-1', 'full_name': 'Starter Out',
         'report_status': 'Out', 'date_modified': '2023-09-22T18:00:00Z'},
        {'week': 3, 'team': 'BAL', 'gsis_id': '00-2', 'full_name': 'Backup Out',
         'report_status': 'Out', 'date_modified': '2023-09-22T18:00:00Z'},
        {'week': 3, 'team': 'BAL', 'gsis_id': '00-3', 'full_name': 'Questionable Starter',
         'report_status': 'Questionable', 'date_modified': '2023-09-22T18:00:00Z'},
    ])
    monkeypatch.setattr(ha, 'load_injury_reports', lambda season: reports)
    stats = pd.DataFrame([
        {'name': 'Starter Out', 'week': 1, 'weekly_snap_pct': 90.0},
        {'name': 'Starter Out', 'week': 2, 'weekly_snap_pct': 92.0},
        {'name': 'Backup Out', 'week': 1, 'weekly_snap_pct': 10.0},
        {'name': 'Backup Out', 'week': 2, 'weekly_snap_pct': 12.0},
    ])
    out = ha.significant_out_players(2023, 3, _schedule(), stats, name_col='name')
    assert out == {'Starter Out': 'BAL'}   # Backup Out's own snap share is too low; Questionable isn't "out" at all


def test_significant_out_players_returns_empty_without_snap_data():
    reports = _reports([{'week': 3, 'team': 'BAL', 'gsis_id': '00-1', 'full_name': 'A Back',
                         'report_status': 'Out', 'date_modified': '2023-09-22T18:00:00Z'}])
    assert ha.significant_out_players(2023, 3, _schedule(), pd.DataFrame(), name_col='name') == {}
    assert ha.significant_out_players(2023, 3, _schedule(),
                                      pd.DataFrame({'name': ['x'], 'week': [1]}), name_col='name') == {}


def test_recipient_teammates_excludes_the_injured_player_and_other_teams():
    out_players = {'Starter Out': 'BAL', 'Other Out': 'KC'}
    board = {'BAL': {'Starter Out', 'Teammate A', 'Teammate B'},
            'KC': {'Other Out', 'KC Guy'},
            'SF': {'Uninvolved'}}
    recipients = ha.recipient_teammates(out_players, board)
    assert recipients == {'Teammate A', 'Teammate B', 'KC Guy'}


def test_load_injury_reports_caches_and_filters_to_regular_season(tmp_path, monkeypatch):
    monkeypatch.setattr(ha, 'CACHE_DIR', str(tmp_path))
    calls = {'n': 0}

    class _FakeFrame:
        def to_pandas(self):
            calls['n'] += 1
            return pd.DataFrame([
                {'season': 2023, 'game_type': 'REG', 'week': 3, 'gsis_id': '00-1',
                 'full_name': 'A Back', 'report_status': 'Out', 'date_modified': '2023-09-22T18:00:00Z'},
                {'season': 2023, 'game_type': 'POST', 'week': 19, 'gsis_id': '00-2',
                 'full_name': 'B Wideout', 'report_status': 'Out', 'date_modified': '2024-01-10T18:00:00Z'},
            ])

    class _FakeNflreadpy:
        @staticmethod
        def load_injuries(seasons):
            return _FakeFrame()

    monkeypatch.setitem(sys.modules, 'nflreadpy', _FakeNflreadpy)
    out = ha.load_injury_reports(2023)
    assert list(out['full_name']) == ['A Back']   # POST filtered out
    assert calls['n'] == 1

    # Second call reads the cache, not the source again.
    out2 = ha.load_injury_reports(2023)
    assert calls['n'] == 1
    assert list(out2['full_name']) == ['A Back']
