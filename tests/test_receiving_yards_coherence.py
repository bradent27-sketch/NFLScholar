"""v2_receiving_yards_coherence (2026-11-24, accuracy idea 4): a team's receivers' yards move part of the way toward its QB1."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

import data.weekly_projections as wp  # noqa: E402


def board():
    # AAA: QB1 implies more than the receivers' total (raised). BBB: less (lowered). CCC: no QB (untouched).
    return pd.DataFrame({
        'Player': ['A QB', 'A QB2', 'A WR', 'A TE', 'B QB', 'B RB', 'B WR', 'C WR'],
        'Pos': ['QB', 'QB', 'WR', 'TE', 'QB', 'RB', 'WR', 'WR'],
        'Team': ['AAA', 'AAA', 'AAA', 'AAA', 'BBB', 'BBB', 'BBB', 'CCC'],
        'Availability': [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],
        'passing_attempts': [35.0, 2.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0],
        'passing_yards': [300.0, 10.0, 0.0, 0.0, 150.0, 0.0, 0.0, 0.0],
        'receiving_yards': [0.0, 0.0, 120.0, 80.0, 0.0, 40.0, 160.0, 70.0],
    })


def test_flag_shipped_after_calibration():
    assert 'v2_receiving_yards_coherence' in wp.MODEL_FEATURES
    assert 'v2_receiving_yards_coherence' in wp.DEFAULT_FEATURES
    assert 'v2_receiving_yards_coherence' in wp.POST_CALIBRATION_FEATURES   # the calibration needs no refit
    assert wp.RECEIVING_YARDS_COHERENCE_WEIGHT == 0.25
    assert wp.RECEIVING_YARDS_COHERENCE_K == 0.956


def test_moves_both_ways_and_keeps_shares():
    b = board()
    out, info = wp.apply_receiving_yards_coherence(b)
    w, k = wp.RECEIVING_YARDS_COHERENCE_WEIGHT, wp.RECEIVING_YARDS_COHERENCE_K
    # AAA: total = 120 + 0.5 x 80 = 160 (availability-weighted); QB1 is the 35-attempt QB, not the backup
    fa = ((1 - w) * 160 + w * 300 / k) / 160
    # BBB: total = 200; QB1 150
    fb = ((1 - w) * 200 + w * 150 / k) / 200
    assert fa > 1 > fb
    assert np.isclose(info['AAA'], fa) and np.isclose(info['BBB'], fb)
    got = out.set_index('Player')['receiving_yards']
    assert np.isclose(got['A WR'], 120 * fa) and np.isclose(got['A TE'], 80 * fa)
    assert np.isclose(got['B RB'], 40 * fb) and np.isclose(got['B WR'], 160 * fb)
    assert np.isclose(got['B WR'] / got['B RB'], 4.0)          # shares unchanged
    assert got['C WR'] == 70.0 and 'CCC' not in info              # no QB on the board: untouched
    assert (out['passing_yards'] == b['passing_yards']).all()     # the QB line itself never moves


def test_categorical_team_column_matches_plain():
    # the live board's Team is Categorical (the historical boards' is plain text); 2026-11-24 the live build crashed on it
    b = board()
    cat = b.assign(Team=b['Team'].astype('category'))
    out_plain, info_plain = wp.apply_receiving_yards_coherence(b)
    out_cat, info_cat = wp.apply_receiving_yards_coherence(cat)
    assert info_cat == info_plain
    assert np.allclose(out_cat['receiving_yards'].astype(float), out_plain['receiving_yards'].astype(float))


def test_team_with_no_receiving_yards_is_untouched():
    b = board()
    b.loc[b['Team'] == 'BBB', 'receiving_yards'] = 0.0
    out, info = wp.apply_receiving_yards_coherence(b)
    assert 'BBB' not in info
    assert (out.loc[out['Team'] == 'BBB', 'receiving_yards'] == 0.0).all()


def test_board_without_a_qb_line_is_untouched():
    # a board with no QBs carries no passing columns (the unit-test boards): nothing to reconcile toward
    b = board()[lambda d: d['Pos'] != 'QB'].drop(columns=['passing_yards', 'passing_attempts'])
    out, info = wp.apply_receiving_yards_coherence(b)
    assert info == {} and out['receiving_yards'].equals(b['receiving_yards'])


def test_missing_column_fails_fast():
    try:
        wp.apply_receiving_yards_coherence(board().drop(columns=['Availability']))
    except KeyError as err:
        assert 'Availability' in str(err)
    else:
        raise AssertionError('expected a KeyError for the missing column')
