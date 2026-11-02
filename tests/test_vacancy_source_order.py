"""v2_vacancy_source_order: the vacancy pass's split does not depend on the board's row order."""
import pandas as pd

import data.weekly_projections as wp
from data.rb_role_allocator import redistribute_rb_vacancy_with_allocator


ROWS = [('Out One', 'WR', 0.0, 8.0), ('Out Two', 'WR', 0.0, 6.0), ('Slot', 'WR', 5.0, 5.0), ('Deep', 'WR', 3.0, 3.0),
        ('Tight', 'TE', 4.0, 4.0), ('Fourth', 'WR', 1.5, 1.5), ('Twin', 'WR', 1.5, 1.5)]
INJURED = {'Out One': {'plays_probability': 0.0}, 'Out Two': {'plays_probability': 0.0}}


def _board():
    return pd.DataFrame([{'Player': p, 'Team': 'NYJ', 'Pos': pos, 'targets': t, 'receptions': t * .65,
                          'receiving_yards': t * 7.5, 'receiving_tds': t * .05, '_full_targets': f,
                          'Expected Snap Share': .5, 'Model Proj Pts': t} for p, pos, t, f in ROWS])


def _split(board, deterministic):
    out, _ = redistribute_rb_vacancy_with_allocator(board, INJURED, receiver_pecking_order=True,
                                                    deterministic_order=deterministic)
    return out.set_index('Player')['targets'].sort_index()


def test_flag_is_a_candidate():
    assert 'v2_vacancy_source_order' in wp.MODEL_FEATURES and 'v2_vacancy_source_order' not in wp.DEFAULT_FEATURES


def test_row_order_changes_the_split_without_the_flag():
    board = _board()
    flipped = board.iloc[[1, 0, 2, 3, 4, 6, 5]].reset_index(drop=True)
    assert not _split(board, False).equals(_split(flipped, False))


def test_split_is_the_same_in_every_row_order_with_the_flag():
    board = _board()
    want = _split(board, True)
    for seed in range(6):
        shuffled = board.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        pd.testing.assert_series_equal(_split(shuffled, True), want)
    # the targets the two out receivers vacated are handed out, not lost or doubled
    assert abs(want.sum() - _split(board, False).sum()) < 0.05
