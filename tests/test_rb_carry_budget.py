import numpy as np
import pandas as pd
import pytest

import data.rb_carry_budget as cb

PARAMS = {'intercept': 0.0, 'coefs': {'rate': 40.0, 'spread': 0.1, 'opp_rb': 0.5}, 'k_shrink': 4.0}


def _budgets(**by_team):
    return pd.DataFrame({'budget': by_team}).rename_axis('team')


def _board(rows):
    return pd.DataFrame(rows, columns=['Player', 'Team', 'Pos', 'Availability', 'rushing_attempts',
                                       'rushing_yards', 'rushing_tds'])


def test_team_game_frame_counts_plays_and_rb_carries_per_game():
    stats = pd.DataFrame([
        {'game_team': 'kc', 'game_opponent': 'den', 'week': 1, 'position': 'QB', 'passing_attempts': 30, 'rushing_attempts': 4},
        {'game_team': 'kc', 'game_opponent': 'den', 'week': 1, 'position': 'RB', 'passing_attempts': 0, 'rushing_attempts': 18},
        {'game_team': 'kc', 'game_opponent': 'den', 'week': 1, 'position': 'WR', 'passing_attempts': 0, 'rushing_attempts': 1},
        {'game_team': None, 'game_opponent': None, 'week': 1, 'position': 'RB', 'passing_attempts': 0, 'rushing_attempts': 0},
    ])
    g = cb.team_game_frame(stats)
    assert len(g) == 1
    row = g.iloc[0]
    assert row['team'] == 'KC' and row['opponent'] == 'DEN'
    assert row['plays'] == 53 and row['rb_carries'] == 18


def test_budget_features_shrink_toward_prior_and_read_the_spread_from_the_teams_side():
    prior = pd.DataFrame({'team': ['KC', 'DEN'] * 4, 'opponent': ['DEN', 'KC'] * 4, 'week': np.repeat([1, 2, 3, 4], 2),
                          'plays': 60.0, 'rb_carries': [24.0, 18.0] * 4})
    cur = pd.DataFrame({'team': ['KC'] * 4, 'opponent': ['LV'] * 4, 'week': [1, 2, 3, 4], 'plays': 60.0, 'rb_carries': 30.0})
    sched = pd.DataFrame([{'week': 5, 'home_team': 'KC', 'away_team': 'DEN', 'spread_line': 6.5}])
    f = cb.budget_features(cur, prior, cb._spread_by_team(sched, 5), k=4.0)
    assert f.loc['KC', 'spread'] == 6.5 and f.loc['DEN', 'spread'] == -6.5
    # 4 current games at 30/60 + 4 prior-weight games at 24/60 -> 27/60.
    assert f.loc['KC', 'rate'] == pytest.approx(27.0 / 60.0)
    assert f.loc['DEN', 'rate'] == pytest.approx(18.0 / 60.0)      # no current games: prior only
    assert f.loc['KC', 'lg_rb'] == pytest.approx(21.0)
    # DEN's defense allowed 24/game last season (KC's carries), no current games vs DEN.
    assert f.loc['KC', 'opp_rb_allowed'] == pytest.approx(24.0)


def test_team_rb_carry_budgets_combine_the_fitted_terms():
    prior = pd.DataFrame([{'game_team': t, 'game_opponent': o, 'week': w, 'position': 'RB', 'passing_attempts': 0,
                           'rushing_attempts': 20.0} for w in (1, 2) for t, o in (('KC', 'DEN'), ('DEN', 'KC'))]
                         + [{'game_team': t, 'game_opponent': o, 'week': w, 'position': 'QB', 'passing_attempts': 40.0,
                             'rushing_attempts': 0.0} for w in (1, 2) for t, o in (('KC', 'DEN'), ('DEN', 'KC'))])
    sched = pd.DataFrame([{'week': 3, 'home_team': 'KC', 'away_team': 'DEN', 'spread_line': 3.0}])
    b = cb.team_rb_carry_budgets(prior.iloc[0:0], prior, sched, 3, params=PARAMS)
    # Everyone at the league rate and league allowed: budget = lg_rb + 0.1 * spread.
    assert b.loc['KC', 'budget'] == pytest.approx(20.0 + 0.3)
    assert b.loc['DEN', 'budget'] == pytest.approx(20.0 - 0.3)
    assert cb.team_rb_carry_budgets(prior, prior, sched, 3, params={}).empty


def test_room_inside_the_one_carry_band_is_untouched():
    board = _board([['A', 'KC', 'RB', 1.0, 15.0, 70.0, 0.5], ['B', 'KC', 'RB', 1.0, 6.0, 25.0, 0.1]])
    out, ledger = cb.apply_rb_carry_budget(board, _budgets(KC=20.2), deadband=1.0)
    pd.testing.assert_frame_equal(out, board)
    assert ledger.iloc[0]['factor'] == 1.0


def test_over_budget_room_is_pulled_to_the_band_edge_uniformly_with_efficiency_kept():
    board = _board([['A', 'KC', 'RB', 1.0, 16.0, 80.0, 0.6], ['B', 'KC', 'RB', 1.0, 8.0, 32.0, 0.2],
                    ['W', 'KC', 'WR', 1.0, 1.0, 8.0, 0.0]])
    out, _ = cb.apply_rb_carry_budget(board, _budgets(KC=20.0), deadband=1.0)
    rb = out[out['Pos'] == 'RB']
    assert rb['rushing_attempts'].sum() == pytest.approx(21.0, abs=1e-2)     # edge, not the budget
    assert (rb['rushing_yards'] / rb['rushing_attempts']).tolist() == pytest.approx([5.0, 4.0], abs=1e-3)
    assert rb['rushing_attempts'].iloc[0] / rb['rushing_attempts'].iloc[1] == pytest.approx(2.0, abs=1e-3)
    assert out[out['Pos'] == 'WR']['rushing_attempts'].iloc[0] == 1.0      # non-RB carries untouched


def test_under_budget_room_is_raised_symmetrically():
    board = _board([['A', 'KC', 'RB', 1.0, 12.0, 50.0, 0.4], ['B', 'KC', 'RB', 1.0, 4.0, 16.0, 0.1]])
    out, ledger = cb.apply_rb_carry_budget(board, _budgets(KC=20.0), deadband=1.0)
    assert out['rushing_attempts'].sum() == pytest.approx(19.0, abs=1e-2)
    assert 'raised' in ledger.iloc[0]['reason']


def test_lead_weighted_trim_lands_mostly_on_the_lead_back():
    board = _board([['A', 'KC', 'RB', 1.0, 18.0, 80.0, 0.6], ['B', 'KC', 'RB', 1.0, 6.0, 25.0, 0.1]])
    uni, _ = cb.apply_rb_carry_budget(board, _budgets(KC=20.0), deadband=1.0)
    lead, _ = cb.apply_rb_carry_budget(board, _budgets(KC=20.0), deadband=1.0, lead_weighted=True)
    assert lead['rushing_attempts'].sum() == pytest.approx(21.0, abs=1e-2)
    cut_lead = board['rushing_attempts'] - lead['rushing_attempts']
    cut_uni = board['rushing_attempts'] - uni['rushing_attempts']
    assert cut_lead.iloc[0] == pytest.approx(3.0 * 324 / 360, abs=1e-2)      # carries^2 weights
    assert cut_lead.iloc[1] < cut_uni.iloc[1]


def test_sidelined_back_counts_at_full_claim_and_his_stash_is_scaled():
    board = _board([['A', 'KC', 'RB', 0.0, 0.0, 0.0, 0.0], ['B', 'KC', 'RB', 1.0, 8.0, 32.0, 0.2]])
    board['_full_rushing_attempts'] = [16.0, 8.0]
    board['_full_rushing_yards'] = [72.0, 32.0]
    out, ledger = cb.apply_rb_carry_budget(board, _budgets(KC=20.0), deadband=1.0)
    assert ledger.iloc[0]['claim'] == pytest.approx(24.0)        # not 8: vacancy owns A's carries
    f = 21.0 / 24.0
    assert out.loc[1, 'rushing_attempts'] == pytest.approx(8.0 * f, abs=1e-3)
    assert out.loc[0, '_full_rushing_attempts'] == pytest.approx(16.0 * f, abs=1e-3)
    assert out.loc[0, '_full_rushing_yards'] == pytest.approx(72.0 * f, abs=1e-3)
    assert out.loc[0, 'rushing_attempts'] == 0.0


def test_team_without_a_budget_is_left_alone():
    board = _board([['A', 'KC', 'RB', 1.0, 30.0, 120.0, 1.0]])
    out, ledger = cb.apply_rb_carry_budget(board, _budgets(DEN=20.0))
    pd.testing.assert_frame_equal(out, board)
    assert 'No budget' in ledger.iloc[0]['reason']


def test_shipped_params_are_sane():
    cb.load_rb_carry_budget_params.cache_clear()
    p = cb.load_rb_carry_budget_params()
    assert p is not None and set(p['coefs']) == {'rate', 'spread', 'opp_rb'}
    assert p['coefs']['rate'] > 0 and p['coefs']['opp_rb'] > 0 and 0 <= p['coefs']['spread'] < 0.3
    assert abs(p['intercept']) < 1.5
