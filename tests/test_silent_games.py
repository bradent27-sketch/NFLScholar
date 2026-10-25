"""data/silent_games.py (2026-10-25): a snap game with no box-score row is a game played with every stat 0."""
import pandas as pd

from data import silent_games as sg


def _stats():
    return pd.DataFrame([
        {'player_display_name': 'Lead Receiver', 'pfr_id': 'LeadRe00', 'exact_name': 'leadreceiver', 'position': 'WR',
         'team': 'KC', 'week': 1.0, 'targets': 8.0, 'receptions': 6.0, 'fantasy_points_ppr': 15.0, 'weekly_snap_pct': 90.0},
        {'player_display_name': 'Lead Receiver', 'pfr_id': 'LeadRe00', 'exact_name': 'leadreceiver', 'position': 'WR',
         'team': 'KC', 'week': 2.0, 'targets': 7.0, 'receptions': 5.0, 'fantasy_points_ppr': 12.0, 'weekly_snap_pct': 88.0},
        {'player_display_name': 'Blocking End', 'pfr_id': 'BlocEn00', 'exact_name': 'blockingend', 'position': 'TE',
         'team': 'KC', 'week': 1.0, 'targets': 1.0, 'receptions': 1.0, 'fantasy_points_ppr': 2.0, 'weekly_snap_pct': 30.0},
    ]).astype({'position': 'category', 'team': 'category'})


def _snaps():
    return pd.DataFrame([
        # a box-score game (matched by PFR id): not silent
        {'pfr': 'LeadRe00', 'name': 'Lead Receiver', 'pos': 'WR', 'week': 1.0, 'team': 'KC', 'opponent': 'BUF', 'game_id': 'g1', 'share': 0.9},
        # the blocking TE's week 2: snaps, no box row -> silent, identity from his week-1 box row
        {'pfr': 'BlocEn00', 'name': 'Blocking End', 'pos': 'TE', 'week': 2.0, 'team': 'KC', 'opponent': 'LV', 'game_id': 'g2', 'share': 0.25},
        # a receiver with no box row all season: silent, the snap table's name
        {'pfr': 'DeepWr00', 'name': 'Deep Receiver', 'pos': 'WR', 'week': 2.0, 'team': 'KC', 'opponent': 'LV', 'game_id': 'g2', 'share': 0.05},
        # matched by name when the PFR id is missing from the box score
        {'pfr': 'Other00', 'name': 'Lead Receiver', 'pos': 'WR', 'week': 2.0, 'team': 'KC', 'opponent': 'LV', 'game_id': 'g2', 'share': 0.88},
    ])


def test_silent_rows_are_zero_stat_games_at_their_snap_share(monkeypatch):
    monkeypatch.setattr(sg, '_snap_games', lambda year: _snaps())
    rows = sg.silent_game_rows(2024, _stats(), 'player_display_name', 'team').set_index('player_display_name')
    assert set(rows.index) == {'Blocking End', 'Deep Receiver'}
    te = rows.loc['Blocking End']
    assert te['week'] == 2.0 and te['targets'] == 0.0 and te['fantasy_points_ppr'] == 0.0 and te['weekly_snap_pct'] == 25.0
    assert te['position'] == 'TE' and te['pfr_id'] == 'BlocEn00' and te['opponent_team'] == 'LV' and te[sg.SILENT_FLAG]
    assert rows.loc['Deep Receiver', 'exact_name'] == 'deepreceiver' and rows.loc['Deep Receiver', 'position'] == 'WR'


def test_with_silent_games_appends_and_keeps_categories(monkeypatch):
    monkeypatch.setattr(sg, '_snap_games', lambda year: _snaps())
    sg._MEMO.clear()
    stats = _stats()
    out = sg.with_silent_games(stats, 2024, 'player_display_name', 'team')
    assert len(out) == len(stats) + 2 and int(out[sg.SILENT_FLAG].sum()) == 2
    assert isinstance(out['position'].dtype, pd.CategoricalDtype) and isinstance(out['team'].dtype, pd.CategoricalDtype)
    assert len(stats) == 3 and sg.SILENT_FLAG not in stats.columns                       # the loader's frame is untouched
    te = out[out['player_display_name'] == 'Blocking End']
    assert te['targets'].mean() == 0.5                                                   # one target over two games played


def test_played_players_counts_a_silent_game(monkeypatch):
    monkeypatch.setattr(sg, '_snap_games', lambda year: _snaps())
    sg._MEMO.clear()
    played = sg.played_players(2024, 2, _stats(), 'player_display_name')
    assert played == {'Lead Receiver', 'Blocking End', 'Deep Receiver'}
    assert sg.played_players(2024, 1, _stats(), 'player_display_name') == {'Lead Receiver', 'Blocking End'}


def test_no_snaps_means_no_silent_games(monkeypatch):
    monkeypatch.setattr(sg, '_snap_games', lambda year: pd.DataFrame())
    sg._MEMO.clear()
    assert sg.silent_game_rows(2024, _stats(), 'player_display_name', 'team').empty
    assert len(sg.with_silent_games(_stats(), 2024, 'player_display_name', 'team')) == 3


def test_the_silent_games_candidates_pick_their_tables():
    import data.weekly_projections as wp
    for flag in ('v2_backup_te_bands_silent', 'v2_backup_te_bands_pieces', 'v2_rb_participation_060'):
        assert flag in wp.MODEL_FEATURES
    assert wp.backup_te_bands(set()) is wp.BACKUP_TE_ACTIVITY_BANDS
    assert wp.backup_te_bands({'v2_backup_te_bands_silent'}) is wp.BACKUP_TE_ACTIVITY_BANDS_SILENT
    assert wp.backup_te_bands({'v2_backup_te_bands_pieces'}) is wp.BACKUP_TE_ACTIVITY_BANDS_PIECES
    assert all(f <= 1.0 for _e, f in wp.BACKUP_TE_ACTIVITY_BANDS_PIECES)              # the pieces only discount
    assert {'v2_silent_games', 'v2_silent_participation'} <= wp.DEFAULT_FEATURES     # shipped 2026-10-25
