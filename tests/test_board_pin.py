"""ui/board_pin.py (2026-10-23): a kept board on disk, which teams a ruling touched, and the team merge.

The merge's exactness against a real full build (a team-only build swapped in gives the full board) was checked on 2025 weeks 7 and
12 and the live 2026 week 8 when it was built; these pin the pieces on small frames."""
import datetime
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from ui import board_pin


def _board(rows):
    return pd.DataFrame(rows, columns=['Player', 'Pos', 'Team', 'Model Proj Pts', 'Role Confidence'])


def _meta(board, ledger_teams=()):
    explanations = {(r.Player, r.Pos, r.Team): {'position': r.Pos, 'game_log_by_season': {}, 'tag': 'old'}
                    for r in board.itertuples()}
    return {'explanations': explanations, 'players': len(board), 'vacancy_adjusted': False,
            'vacancy_ledger': [{'team': t, 'source_player': 'x', 'recipients': []} for t in ledger_teams],
            'rb_allocation_ledger': [], 'pass_capacity_ledger': [{'team': t} for t in ledger_teams],
            'rb_carry_budget_ledger': [], 'league_context': {('pos_rate', 'WR', 'targets'): 5.0}}


class ChangedTeamsTest(unittest.TestCase):
    def test_teams_whose_rows_differ(self):
        old = {'KC': frozenset({('avail', 'a')}), 'LV': frozenset({('avail', 'b')})}
        new = {'KC': frozenset({('avail', 'a')}), 'LV': frozenset({('avail', 'c')}), 'DEN': frozenset({('avail', 'd')})}
        self.assertEqual(board_pin.changed_teams(old, new), {'LV', 'DEN'})

    def test_no_snapshot_or_a_row_without_a_team_means_rebuild(self):
        self.assertIsNone(board_pin.changed_teams(None, {'KC': frozenset()}))
        self.assertIsNone(board_pin.changed_teams({}, {'': frozenset({('avail', 'x')})}))

    def test_nothing_changed(self):
        snap = {'KC': frozenset({('avail', 'a')})}
        self.assertEqual(board_pin.changed_teams(snap, dict(snap)), set())


class MergeTeamBoardTest(unittest.TestCase):
    def setUp(self):
        self.board = _board([('A1', 'WR', 'KC', 15.0, 0.8), ('B1', 'WR', 'LV', 12.0, 0.8), ('A2', 'WR', 'KC', 9.0, 0.6),
                             ('B2', 'RB', 'LV', 11.0, 0.7)])
        self.meta = _meta(self.board, ledger_teams=('KC', 'LV'))
        self.part = _board([('A1', 'WR', 'KC', 8.0, 0.8), ('A2', 'WR', 'KC', 13.0, 0.6)])
        self.part_meta = _meta(self.part, ledger_teams=('KC',))
        for detail in self.part_meta['explanations'].values():
            detail['tag'] = 'new'

    def test_rows_of_the_team_are_replaced_and_the_board_resorted(self):
        merged, _meta_out = board_pin.merge_team_board(self.board, self.meta, self.part, self.part_meta, {'KC'})
        self.assertEqual(list(merged['Player']), ['A2', 'B1', 'B2', 'A1'])
        self.assertEqual(float(merged.loc[merged['Player'] == 'A1', 'Model Proj Pts'].iloc[0]), 8.0)
        self.assertEqual(float(merged.loc[merged['Player'] == 'B1', 'Model Proj Pts'].iloc[0]), 12.0)

    def test_explanations_and_ledgers_of_the_team_come_from_the_partial_build(self):
        _merged, meta = board_pin.merge_team_board(self.board, self.meta, self.part, self.part_meta, {'KC'})
        self.assertEqual(meta['explanations'][('A1', 'WR', 'KC')]['tag'], 'new')
        self.assertEqual(meta['explanations'][('B1', 'WR', 'LV')]['tag'], 'old')
        self.assertEqual(sorted(e['team'] for e in meta['vacancy_ledger']), ['KC', 'LV'])
        self.assertEqual(meta['players'], 4)
        self.assertEqual(meta['league_context'], {('pos_rate', 'WR', 'targets'): 5.0})

    def test_bands_use_the_rank_on_the_merged_board(self):
        seen = {}

        def fake_distribution(row, rank, detail):
            seen[row['Player']] = rank
            return None
        with mock.patch('data.weekly_projections.board_distribution', fake_distribution):
            board_pin.merge_team_board(self.board, self.meta, self.part, self.part_meta, {'KC'})
        # WR ranks on the merged board: A2 13.0, B1 12.0, A1 8.0
        self.assertEqual((seen['A2'], seen['B1'], seen['A1']), (1.0, 2.0, 3.0))


class SaveLoadTest(unittest.TestCase):
    def test_round_trip_and_a_changed_model_signature_drops_it(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(board_pin, 'CACHE_DIR', tmp):
            key = (2026, 8, 'Full PPR')
            entry = {'inputs': ('a', 'b'), 'df': _board([('A1', 'WR', 'KC', 15.0, 0.8)]), 'meta': {},
                     'built_at': datetime.datetime(2026, 10, 23, 14, 0)}
            with mock.patch.object(board_pin, 'model_signature', return_value='sig1'):
                board_pin.save(key, entry)
                loaded = board_pin.load(key)
                self.assertEqual(loaded['inputs'], ('a', 'b'))
                self.assertNotIn('signature', loaded)
                self.assertTrue(board_pin.has_saved(key))
            with mock.patch.object(board_pin, 'model_signature', return_value='sig2'):
                self.assertIsNone(board_pin.load(key))

    def test_an_old_file_is_not_loaded(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(board_pin, 'CACHE_DIR', tmp), \
                mock.patch.object(board_pin, 'model_signature', return_value='sig'):
            key = (2026, 8, 'Full PPR')
            board_pin.save(key, {'inputs': (), 'df': pd.DataFrame(), 'meta': {}, 'built_at': datetime.datetime.now()})
            old = os.path.getmtime(board_pin._path(key)) - 13 * 3600
            os.utime(board_pin._path(key), (old, old))
            self.assertIsNone(board_pin.load(key))

    def test_only_the_newest_files_are_kept(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(board_pin, 'CACHE_DIR', tmp), \
                mock.patch.object(board_pin, 'model_signature', return_value='sig'):
            for week in range(1, board_pin.MAX_SAVED_BOARDS + 3):
                board_pin.save((2026, week, 'Full PPR'), {'inputs': (), 'df': pd.DataFrame(), 'meta': {},
                                                         'built_at': datetime.datetime.now()})
                stamp = 1_000_000 + week
                os.utime(board_pin._path((2026, week, 'Full PPR')), (stamp, stamp))
            self.assertEqual(len(os.listdir(tmp)), board_pin.MAX_SAVED_BOARDS)


if __name__ == '__main__':
    unittest.main()
