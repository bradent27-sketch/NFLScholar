"""v2_qb_implied_td_boost (shipped 2026-10-22, model audit lead L1): a QB's passing TDs, rushing TDs and rushing yards
respond more to his team's implied total (+0.5 on the per-stat game-total exponent)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.weekly_projections as wp  # noqa: E402


def test_flag_is_registered_and_shipped():
    assert 'v2_qb_implied_td_boost' in wp.MODEL_FEATURES
    assert 'v2_qb_implied_td_boost' in wp.DEFAULT_FEATURES          # shipped 2026-10-22
    assert wp.QB_IMPLIED_ELASTICITY_BOOST == {'passing_tds': 0.5, 'rushing_tds': 0.5, 'rushing_yards': 0.5}


def test_boost_raises_only_the_named_exponents():
    env = {'KC': {'implied': 27.0}, 'NYG': {'implied': 18.0}}
    stats = ('passing_tds', 'passing_yards', 'rushing_yards')
    teams = np.array(['KC', 'NYG', 'BYE'])
    base = wp._game_total_stat_multipliers(env, teams, 'QB', 22.5, stats)
    boosted = wp._game_total_stat_multipliers(env, teams, 'QB', 22.5, stats, boost=wp.QB_IMPLIED_ELASTICITY_BOOST)
    betas = dict(wp._load_game_total_perstat()['QB'])
    assert np.allclose(boosted['passing_yards'], base['passing_yards'])
    assert np.isclose(boosted['passing_tds'][0], (27.0 / 22.5) ** (betas['passing_tds'] + 0.5))
    assert np.isclose(boosted['passing_tds'][1], (18.0 / 22.5) ** (betas['passing_tds'] + 0.5))
    lo, hi = wp.GAME_TOTAL_CLIP
    assert np.isclose(boosted['rushing_yards'][0], min((27.0 / 22.5) ** (betas['rushing_yards'] + 0.5), hi))
    assert boosted['passing_tds'][2] == 1.0 and boosted['rushing_yards'][2] == 1.0      # no posted line
    assert boosted['passing_tds'][0] > base['passing_tds'][0] and boosted['passing_tds'][1] < base['passing_tds'][1]
