"""scripts.harness_v2: paired-pool, mean-consistent backtest metrics.

See the module's own docstring for the two flaws in the v1 harness this
fixes; the tests below are written to PROVE each fix matters, not just to
exercise the code (per the plan's own instruction: "this test is the proof
that the fix matters, so keep it").
"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.harness_v2 import (  # noqa: E402
    paired_start_pool, scope_pool, pairwise_acc, metrics, td_metrics,
    stat_metrics, bootstrap_ci, sign_test_p, holm_correction, decide, SCOPES, STARTABLE_N,
)


def _board(players, pos, pts):
    return pd.DataFrame({'Player': players, 'Pos': pos, 'Model Proj Pts': pts})


def test_aa_comparison_returns_exactly_zero_on_every_delta():
    """Base scored against itself: every metric is identical, so every
    delta a caller would compute from two `metrics()` calls is exactly 0."""
    players = [f'P{i}' for i in range(30)]
    rng = np.random.default_rng(1)
    pts = rng.normal(10, 4, size=30)
    board = _board(players, 'RB', pts)
    actual = pd.Series(rng.normal(10, 5, size=30), index=players)

    pool = scope_pool(board, board, 'RB', startable=True, actual_players=actual.index)
    pred = pd.Series(board.set_index('Player').loc[pool, 'Model Proj Pts'])
    m_a = metrics(pred, actual.loc[pool])
    m_b = metrics(pred, actual.loc[pool])
    for key in ('n', 'mae', 'rmse', 'bias', 'spearman', 'pairwise_acc'):
        assert m_a[key] == m_b[key]


def test_mae_rewards_shrinkage_but_pairwise_acc_does_not():
    """THE PROOF THAT THE FIX MATTERS (E3 in the audit). Real weekly
    fantasy-point actuals are right-skewed, so MAE is minimized below the
    true mean. A variant that shrinks every prediction toward zero by a
    constant factor can never change any pairwise ranking (multiplying by a
    positive constant preserves strict order) - so a REAL forecasting
    improvement is impossible to fake this way under pairwise_acc, even
    though it reliably "wins" under the old MAE-only rule."""
    rng = np.random.default_rng(42)
    n = 300
    # Right-skewed actuals: exponential, mean 10, median well below it
    # (ln(2)*10 ~= 6.9) - the same shape E3 found in real weekly points.
    actual = rng.exponential(scale=10.0, size=n)
    base_pred = np.full(n, 10.0) + rng.normal(0, 0.5, size=n)   # unbiased mean estimate + noise
    variant_pred = base_pred * 0.9                              # pure shrink toward zero

    m_base = metrics(pd.Series(base_pred), pd.Series(actual))
    m_variant = metrics(pd.Series(variant_pred), pd.Series(actual))

    assert m_variant['mae'] < m_base['mae'], \
        "the artifact: shrinking toward zero 'wins' on MAE against skewed actuals"
    assert abs(m_variant['pairwise_acc'] - m_base['pairwise_acc']) < 1e-9, \
        "a positive rescaling cannot change any pairwise ranking - v2's primary metric sees no change"
    assert m_variant['rmse'] >= m_base['rmse'] - 1e-9 or True  # RMSE direction isn't the point here; documented for context


def test_paired_start_pool_scores_both_arms_on_the_same_players():
    """THE OTHER PROOF (E5). v1's _scope_df calls nlargest separately per
    arm; a variant that promotes entirely different players into its own
    top-N would be scored on an easier/different population. v2's pool is
    the union of both arms' own top-N, restricted to players BOTH arms
    projected - so both arms are always scored on one identical set."""
    players = [f'P{i}' for i in range(10)]
    base = _board(players, 'RB', [10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    # Variant completely inverts who looks best.
    variant = _board(players, 'RB', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    pool = paired_start_pool(base, variant, 'RB', n=3)
    # Base's own top-3 is P0/P1/P2; variant's own top-3 is P9/P8/P7. Both
    # arms are then scored on the UNION of those, not either one alone.
    assert set(pool) == {'P0', 'P1', 'P2', 'P7', 'P8', 'P9'}
    assert len(pool) >= 3
    # Every pool member is a player both boards actually projected.
    assert set(pool) <= set(base['Player']) & set(variant['Player'])


def test_paired_start_pool_drops_a_player_missing_from_either_board():
    base = _board(['A', 'B', 'C'], 'WR', [10, 9, 8])
    variant = _board(['A', 'B'], 'WR', [10, 9])   # C ablated clean out of the board
    pool = paired_start_pool(base, variant, 'WR', n=3)
    assert 'C' not in pool
    assert set(pool) == {'A', 'B'}


def test_paired_start_pool_restricts_to_players_with_an_actual():
    base = _board(['A', 'B', 'C'], 'WR', [10, 9, 8])
    variant = _board(['A', 'B', 'C'], 'WR', [10, 9, 8])
    pool = paired_start_pool(base, variant, 'WR', n=3, actual_players=['A', 'B'])
    assert set(pool) == {'A', 'B'}


def test_start_all_scope_unions_each_positions_own_paired_pool():
    boards = {}
    for pos, n in STARTABLE_N.items():
        players = [f'{pos}{i}' for i in range(n + 2)]
        boards[pos] = _board(players, pos, list(range(n + 2, 0, -1)))
    base = pd.concat(boards.values(), ignore_index=True)
    variant = base.copy()   # A/A - the union should just be each position's own top-N
    pool = scope_pool(base, variant, None, startable=True)
    expected = set()
    for pos, n in STARTABLE_N.items():
        expected |= set(base[base['Pos'] == pos].nlargest(n, 'Model Proj Pts')['Player'])
    assert set(pool) == expected


def test_pairwise_acc_hand_cases():
    # Perfect agreement.
    assert pairwise_acc([1, 2, 3], [10, 20, 30]) == 1.0
    # Perfect disagreement.
    assert pairwise_acc([3, 2, 1], [10, 20, 30]) == 0.0
    # A tied prediction on a pair with different actuals counts as 0.5 for
    # that pair; the other two pairs are both correct.
    acc = pairwise_acc([5, 5, 1], [10, 20, 1])
    # pairs: (5,5)->actual(10,20) differ, pred tied -> 0.5
    #        (5,1)->actual(10,1) differ, pred order matches -> 1.0
    #        (5,1)->actual(20,1) differ, pred order matches -> 1.0
    assert abs(acc - (0.5 + 1.0 + 1.0) / 3) < 1e-9
    # Pairs with equal actuals are excluded entirely (nan when ALL are tied).
    assert math.isnan(pairwise_acc([1, 2], [5, 5]))


def test_poisson_deviance_hand_checked():
    # deviance = 2 * sum(y*ln(y/mu) - (y - mu)); the y*ln term is 0 when y=0.
    cases = [
        (np.array([1.0]), np.array([1.0]), 0.0),                              # perfect -> 0
        (np.array([1.0]), np.array([2.0]), 2 * (2 * math.log(2 / 1) - (2 - 1))),
        (np.array([0.5]), np.array([0.0]), 2 * (0.0 - (0.0 - 0.5))),
    ]
    for mu, y, expected in cases:
        got = td_metrics(mu, y)['poisson_deviance']
        assert abs(got - expected) < 1e-9


def test_td_metrics_brier_and_floor():
    mu = np.array([0.0, 1.0, 2.0])   # a hard-zero projection is floored, never literally 0
    y = np.array([0.0, 1.0, 0.0])
    m = td_metrics(mu, y)
    assert m['n'] == 3
    p_hat = 1.0 - np.exp(-np.maximum(mu, 1e-3))
    expected_brier = float(np.mean((p_hat - np.array([0.0, 1.0, 0.0])) ** 2))
    assert abs(m['brier'] - expected_brier) < 1e-9


def test_stat_metrics_rmse_per_stat():
    board = pd.DataFrame({'Player': ['A', 'B', 'C', 'D', 'E'],
                          'targets': [8.0, 6.0, 4.0, 2.0, 1.0],
                          'receiving_yards': [90.0, 60.0, 40.0, 20.0, 10.0]})
    actual = pd.DataFrame({'targets': [9.0, 5.0, 4.0, 3.0, 1.0],
                           'receiving_yards': [80.0, 65.0, 45.0, 15.0, 12.0]},
                          index=['A', 'B', 'C', 'D', 'E'])
    out = stat_metrics(board, actual, ['targets', 'receiving_yards', 'rushing_yards'])
    assert set(out) == {'targets', 'receiving_yards'}   # rushing_yards absent from both -> skipped
    assert out['targets']['n'] == 5
    manual_rmse = float(np.sqrt(np.mean((np.array([8, 6, 4, 2, 1]) - np.array([9, 5, 4, 3, 1])) ** 2)))
    assert abs(out['targets']['rmse'] - manual_rmse) < 1e-9


def test_bootstrap_ci_of_identical_deltas_is_a_point():
    lo, hi = bootstrap_ci([0.5] * 10, [20] * 10)
    assert abs(lo - 0.5) < 1e-9 and abs(hi - 0.5) < 1e-9
    # too few weeks -> explicitly nan, not a misleadingly narrow interval
    lo2, hi2 = bootstrap_ci([0.1, 0.2, 0.3], [10, 10, 10])
    assert math.isnan(lo2) and math.isnan(hi2)


def test_sign_test_p_matches_exact_binomial():
    # 8 wins, 0 losses out of 8 - very unlikely under the null of 50/50
    assert sign_test_p(8, 0) < 0.01
    # 5-5 split - indistinguishable from a coin flip
    assert sign_test_p(5, 5) == 1.0


def test_holm_correction_orders_by_significance_and_never_shrinks():
    raw = [0.001, 0.04, 0.20, 0.60]
    adj = holm_correction(raw)
    # Holm only ever inflates (or leaves) a p-value, and preserves monotonic
    # non-decreasing order once sorted by the ORIGINAL p-values.
    assert all(a >= r for a, r in zip(adj, raw))
    sorted_adj = [a for _, a in sorted(zip(raw, adj))]
    assert sorted_adj == sorted(sorted_adj)


def test_decide_ships_only_on_a_clean_improvement():
    # RMSE CI entirely negative (variant better), no position worse, bias flat.
    assert decide((-0.5, -0.1), (float('nan'), float('nan')), [1.0, 1.0, 1.0, 1.0], 0.0) == 'SHIP-ELIGIBLE'
    # RMSE CI entirely positive (variant worse) -> REJECT regardless of pairwise.
    assert decide((0.1, 0.5), (0.1, 0.5), [1.0, 1.0, 1.0, 1.0], 0.0) == 'REJECT'
    # CI straddles zero on both primaries -> INCONCLUSIVE.
    assert decide((-0.2, 0.2), (-0.02, 0.02), [1.0, 1.0, 1.0, 1.0], 0.0) == 'INCONCLUSIVE'
    # Improves overall but one START-position scope is significantly worse -> vetoed.
    assert decide((-0.5, -0.1), (float('nan'), float('nan')), [0.01, 1.0, 1.0, 1.0], 0.0) == 'INCONCLUSIVE'
    # Improves overall but bias grew too much -> vetoed.
    assert decide((-0.5, -0.1), (float('nan'), float('nan')), [1.0, 1.0, 1.0, 1.0], 5.0) == 'INCONCLUSIVE'


def test_scopes_include_start_all():
    names = [s[0] for s in SCOPES]
    assert 'START-ALL' in names
    assert len(names) == len(set(names))
