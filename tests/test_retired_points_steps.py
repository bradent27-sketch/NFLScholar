"""The points-total steps retired from the model (data/retired_points_steps.py; docs/retired_flags.md): the season anchor, the
expected-points luck step and their refits. Nothing in the model calls them; the calibration scripts replay them, and they were
worth having, so they stay tested."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.retired_points_steps as rps  # noqa: E402
import data.weekly_projections as wp  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402

LUCK_ON = frozenset({'v2_expected_points_luck', 'v2_luck_te_refit', 'v2_luck_qb'})


def test_the_retired_flags_are_gone_from_the_model_but_the_code_is_reexported():
    for flag in ('v2_season_anchor', 'v2_anchor_s_refit', 'v2_expected_points_luck', 'v2_luck_te_refit', 'v2_luck_qb'):
        assert flag not in wp.MODEL_FEATURES and flag not in wp.DEFAULT_FEATURES and flag not in wp.POST_CALIBRATION_FEATURES
    for name in ('apply_season_anchor', 'apply_expected_points_luck', 'season_anchor_factors', 'expected_points_luck_factors',
                 'SEASON_ANCHOR_S', 'SEASON_ANCHOR_S_REFIT', 'SEASON_ANCHOR_MIN_GAMES', 'EXPECTED_POINTS_LUCK'):
        assert getattr(wp, name) is getattr(rps, name)               # the calibration scripts import them from here


def test_luck_te_refit_factors():
    assert rps.expected_points_luck_factors(set()) == rps.EXPECTED_POINTS_LUCK
    on = rps.expected_points_luck_factors({'v2_luck_te_refit'})
    assert on['TE'] == rps.EXPECTED_POINTS_LUCK_TE_REFIT == -0.33 and on['WR'] == rps.EXPECTED_POINTS_LUCK['WR']
    board = pd.DataFrame({'Player': ['A TE'], 'Pos': ['TE'], 'Team': ['KC'], 'Availability': [1.0],
                          'Model Proj Pts': [10.0], 'Calibrated Model Proj Pts': [10.0]})
    table = pd.DataFrame({'games': [4], 'luck_pg': [3.0]}, index=clean_name_exact(pd.Series(['A TE'])))
    out, _ = rps.apply_expected_points_luck(board, table, factors=on)
    assert out['Model Proj Pts'].iloc[0] == 9.01                     # 10 - 0.33 * 3


def test_anchor_s_refit_factors():
    assert rps.season_anchor_factors(set()) is None                   # apply_season_anchor's default, SEASON_ANCHOR_S
    assert rps.season_anchor_factors({'v2_season_anchor', 'v2_anchor_s_refit'}) == rps.SEASON_ANCHOR_S_REFIT
    assert all(rps.SEASON_ANCHOR_S_REFIT[p] > rps.SEASON_ANCHOR_S[p] for p in rps.SEASON_ANCHOR_S)   # a weaker pull
    assert all(0.0 < s <= 1.0 for s in rps.SEASON_ANCHOR_S_REFIT.values())


def test_qb_luck_is_centered_on_the_average_qb():
    on = rps.expected_points_luck_factors(LUCK_ON)
    assert on['QB'] == rps.EXPECTED_POINTS_LUCK_QB == -0.11 and on['TE'] == -0.33 and on['WR'] == -0.18
    assert 'QB' not in rps.expected_points_luck_factors(LUCK_ON - {'v2_luck_qb'})
    board = pd.DataFrame({'Player': ['A QB', 'B QB', 'C QB', 'A WR'], 'Pos': ['QB', 'QB', 'QB', 'WR'], 'Team': ['KC'] * 4,
                          'Availability': [1.0] * 4, 'Model Proj Pts': [18.0] * 4, 'Calibrated Model Proj Pts': [18.0] * 4})
    table = pd.DataFrame({'games': [5, 5, 5, 5], 'luck_pg': [2.5, -0.5, -4.5, 3.0]},
                         index=clean_name_exact(pd.Series(['A QB', 'B QB', 'C QB', 'A WR'])))
    out, info = rps.apply_expected_points_luck(board, table, factors=on)
    pts = out.set_index('Player')['Model Proj Pts']
    assert pts['A QB'] == 17.67                                      # 18 - 0.11 * (2.5 - (-0.5))
    assert pts['B QB'] == 18.0                                       # the average QB's luck: no move
    assert pts['C QB'] == 18.44                                      # 18 - 0.11 * (-4.5 + 0.5)
    assert pts['A WR'] == 17.46                                      # WR is not centered: 18 - 0.18 * 3
    assert info[('A QB', 'QB', 'KC')]['center'] == -0.5 and info[('A WR', 'WR', 'KC')]['center'] == 0.0
