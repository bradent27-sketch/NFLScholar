"""
Harness v2 - the paired-pool, mean-consistent replacement for the metrics
`scripts/backtest_component.py`/`scripts/sweep_model_constant.py`/
`scripts/eval_weekly_model.py` have used since 2026-08-26.

WHY THIS EXISTS (docs/model_improvement_plan_2026-09-23.md, item 1; evidence
E3 and E5 in that doc's audit appendix). Two independent flaws in the old
harness:

  1. POOL PAIRING. `backtest_component.py::_scope_df` picks a START-scope
     pool by calling `.nlargest(N, 'Model Proj Pts')` SEPARATELY on the base
     board and the variant board. A variant that projects a totally
     different top-N (dropping hard players, promoting easy ones) is then
     scored on ITS OWN easier population, not the same players base was
     scored on. The whole-pool scopes already intersect both boards
     correctly (`set(base) & set(variant)`); only the STARTABLE cut had this
     bug. `paired_start_pool` below fixes it: the pool is the UNION of each
     arm's own top-N, restricted to players both arms actually projected
     that week and who have a real actual to score against - both arms are
     then evaluated on that identical set.

  2. METRIC CHOICE. Every ship/reject decision to date used MAE. MAE is
     minimized by the CONDITIONAL MEDIAN, and real weekly fantasy-point
     distributions are right-skewed (weekly mean exceeds weekly median by
     ~+0.95-0.98 pts for startable RB/WR/TE, 2021-2025) with 87-94% of
     RB/WR/TE player-games at exactly zero receiving/rushing TDs. A variant
     that merely compresses projections toward zero "wins" on MAE with zero
     forecasting skill gained - see test_harness_v2.py's
     test_mae_rewards_shrinkage_but_pairwise_acc_does_not for the proof.
     `pairwise_acc` (ranking concordance - the thing a start/sit decision
     actually needs) and RMSE (mean-consistent, unlike MAE) are the PRIMARY
     metrics here; MAE is kept only as a secondary, informational column.

This module is deliberately metrics-only (paired pools, metrics, bootstrap
CI, Holm correction, the ship/reject/inconclusive rule) - it has no opinion
about how a caller builds `base_df`/`var_df`; every existing script keeps its
own build/ablation loop and just calls into this instead of
`_scope_df`/`_scope_metrics`/`_metrics`.
"""
import math

import numpy as np
import pandas as pd

# How many players per position anyone would actually consider starting -
# same table scripts/eval_weekly_model.py has used since that script was
# built; kept here too so a harness_v2-only caller need not import the v1
# script just for this constant.
STARTABLE_N = {'QB': 24, 'RB': 40, 'WR': 55, 'TE': 20}

SCOPES = [
    ('ALL', None, False),
    ('QB', 'QB', False), ('RB', 'RB', False), ('WR', 'WR', False), ('TE', 'TE', False),
    ('START-QB', 'QB', True), ('START-RB', 'RB', True),
    ('START-WR', 'WR', True), ('START-TE', 'TE', True),
    ('START-ALL', None, True),
]


def paired_start_pool(base_df, var_df, pos, n, actual_players=None):
    """The union of the top-n (by 'Model Proj Pts') from base_df and from
    var_df, restricted to players present in BOTH boards (a player either
    arm didn't project at all can't be compared) and, when given, in
    `actual_players` (an index/iterable of players who actually have a
    scoreable actual that week). Both arms are then scored on this identical
    set - the fix for the pairing bug in the module docstring's point 1."""
    b = base_df if pos is None else base_df[base_df['Pos'] == pos]
    v = var_df if pos is None else var_df[var_df['Pos'] == pos]
    if b.empty or v.empty:
        return []
    top_b = set(b.nlargest(min(n, len(b)), 'Model Proj Pts')['Player'])
    top_v = set(v.nlargest(min(n, len(v)), 'Model Proj Pts')['Player'])
    pool = (top_b | top_v) & set(b['Player']) & set(v['Player'])
    if actual_players is not None:
        pool &= set(actual_players)
    return sorted(pool)


def scope_pool(base_df, var_df, pos, startable, actual_players=None):
    """Dispatches on a SCOPES row. START-ALL (pos is None, startable=True) is
    the union of each position's own paired START pool, NOT an nlargest over
    the whole board - 'Model Proj Pts' isn't comparable across positions, so
    a single cross-position cut would just re-litigate roster construction."""
    if startable and pos is None:
        pool = set()
        for p, n in STARTABLE_N.items():
            pool |= set(paired_start_pool(base_df, var_df, p, n, actual_players))
        return sorted(pool)
    if startable:
        return paired_start_pool(base_df, var_df, pos, STARTABLE_N.get(pos, 30), actual_players)
    b = base_df if pos is None else base_df[base_df['Pos'] == pos]
    v = var_df if pos is None else var_df[var_df['Pos'] == pos]
    pool = set(b['Player']) & set(v['Player'])
    if actual_players is not None:
        pool &= set(actual_players)
    return sorted(pool)


def pairwise_acc(pred, actual):
    """Share of all pairs (i, j) with actual_i != actual_j where
    sign(pred_i - pred_j) == sign(actual_i - actual_j). A tied prediction
    (pred_i == pred_j) counts as 0.5 for that pair - it's neither right nor
    wrong about which of the two is better. This is what a start/sit call
    actually needs (did the model rank the better player higher), and unlike
    MAE it is invariant to any monotonic rescaling of the predictions - see
    the module docstring's point 2."""
    p = np.asarray(pred, dtype=float)
    a = np.asarray(actual, dtype=float)
    n = len(p)
    if n < 2:
        return float('nan')
    iu = np.triu_indices(n, k=1)
    da = a[iu[0]] - a[iu[1]]
    mask = da != 0
    if not mask.any():
        return float('nan')
    dp = p[iu[0]][mask] - p[iu[1]][mask]
    sign_a = np.sign(da[mask])
    sign_p = np.sign(dp)
    correct = np.where(sign_p == 0, 0.5, (sign_p == sign_a).astype(float))
    return float(correct.mean())


def metrics(pred, actual):
    """pred/actual: two pandas Series (any index; joined on it) or anything
    pd.Series(...) accepts. Returns None on fewer than 5 paired rows - the
    same "too small to mean anything" floor scripts/eval_weekly_model.py has
    always used."""
    pred = pred if isinstance(pred, pd.Series) else pd.Series(pred)
    actual = actual if isinstance(actual, pd.Series) else pd.Series(actual)
    joined = pd.concat([pred.rename('pred'), actual.rename('actual')], axis=1).dropna()
    if len(joined) < 5:
        return None
    err = joined['pred'] - joined['actual']
    return {
        'n': len(joined),
        'mae': float(err.abs().mean()),            # secondary / informational only - see module docstring
        'rmse': float(np.sqrt((err ** 2).mean())),
        'bias': float(err.mean()),
        'spearman': float(joined['pred'].rank().corr(joined['actual'].rank())),
        'pairwise_acc': pairwise_acc(joined['pred'].to_numpy(), joined['actual'].to_numpy()),
    }


def td_metrics(mu, y):
    """Poisson deviance and the Brier score of P(TD>=1) = 1 - e^-mu, for a
    TD-rate projection mu against an actual count y. mu is floored at 1e-3
    (an exact-zero projection would otherwise make the deviance undefined for
    any y>0, and a real player with any opportunity should never project a
    hard zero anyway - see docs/model_improvement_plan_2026-09-23.md item 6).
    y*ln(y/mu) is taken as 0 when y=0 (the standard Poisson-deviance
    convention: the limit of y*ln(y) as y->0 is 0)."""
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-3)
    y = np.asarray(y, dtype=float)
    y_safe = np.where(y > 0, y, 1.0)   # avoid log(0); the where() below discards this branch anyway
    log_term = np.where(y > 0, y * np.log(y_safe / mu), 0.0)
    deviance = float(2.0 * np.sum(log_term - (y - mu)))
    p_hat = 1.0 - np.exp(-mu)
    y_bin = (y >= 1).astype(float)
    n = len(y)
    return {
        'n': n,
        'poisson_deviance': deviance,
        'poisson_deviance_mean': deviance / n if n else float('nan'),
        'brier': float(np.mean((p_hat - y_bin) ** 2)) if n else float('nan'),
    }


def stat_metrics(board, actual_stats, stats, name_col='Player'):
    """RMSE/bias per projected stat column, on whatever pool `board` already
    carries (caller restricts `board` to a paired START pool first - this
    function doesn't re-derive one). `actual_stats` is a DataFrame indexed by
    player name with one column per stat in `stats` (typically
    data.transforms.load_and_merge_data's frame, grouped/summed to one row
    per player for the target week)."""
    out = {}
    for stat in stats:
        if stat not in board.columns or stat not in actual_stats.columns:
            continue
        pred = board[[name_col, stat]].dropna().set_index(name_col)[stat]
        joined = pd.concat([pred.rename('pred'), actual_stats[stat].rename('actual')], axis=1).dropna()
        if len(joined) < 5:
            continue
        err = joined['pred'] - joined['actual']
        out[stat] = {'n': len(joined), 'rmse': float(np.sqrt((err ** 2).mean())),
                     'bias': float(err.mean())}
    return out


def bootstrap_ci(deltas, weights, n_boot=3000, seed=0):
    """Week-cluster bootstrap: each element of `deltas`/`weights` is already
    ONE WEEK's (weighted) metric delta, so resampling elements resamples
    whole weeks - the unit a real season varies at - rather than treating
    individual player-weeks (correlated within a week: one blowout game
    moves everyone's script at once) as independent draws."""
    if len(deltas) < 4:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    deltas = np.asarray(deltas, dtype=float)
    weights = np.asarray(weights, dtype=float)
    idx = np.arange(len(deltas))
    means = np.empty(n_boot)
    for b in range(n_boot):
        samp = rng.choice(idx, size=len(idx), replace=True)
        means[b] = np.average(deltas[samp], weights=weights[samp])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def sign_test_p(wins, losses):
    n = wins + losses
    if n == 0:
        return float('nan')
    k = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def holm_correction(p_values):
    """Holm-Bonferroni step-down adjustment for a family of p-values (e.g.
    the four START-position 'is the variant worse' tests). Returns adjusted
    p-values in the SAME order as the input."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min((m - rank) * p[idx], 1.0)
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj


def decide(rmse_delta_ci, pairwise_delta_ci, start_position_worse_pvals_holm, bias_growth,
          bias_growth_max=0.3, alpha=0.05):
    """The §1 primary decision rule. All deltas are (variant - base):
      rmse_delta_ci      : (lo, hi) - lower is better, so an improvement is
                            hi < 0 (CI entirely negative).
      pairwise_delta_ci  : (lo, hi) - higher is better, so an improvement is
                            lo > 0 (CI entirely positive).
      start_position_worse_pvals_holm : Holm-adjusted p-values, one per
                            START-position scope, for "variant is worse on
                            RMSE at that scope" (one-sided sign test or
                            equivalent) - ANY of these below alpha vetoes a
                            ship even if START-ALL improved.
      bias_growth        : |START-ALL bias(variant)| - |START-ALL bias(base)|.

    Returns 'SHIP-ELIGIBLE', 'REJECT', or 'INCONCLUSIVE'. MAE plays no part
    in this decision - see the module docstring."""
    def _finite(ci):
        return math.isfinite(ci[0]) and math.isfinite(ci[1])

    rmse_worse = _finite(rmse_delta_ci) and rmse_delta_ci[0] > 0
    pairwise_worse = _finite(pairwise_delta_ci) and pairwise_delta_ci[1] < 0
    if rmse_worse or pairwise_worse:
        return 'REJECT'

    rmse_improves = _finite(rmse_delta_ci) and rmse_delta_ci[1] < 0
    pairwise_improves = _finite(pairwise_delta_ci) and pairwise_delta_ci[0] > 0
    any_position_worse = any(p < alpha for p in start_position_worse_pvals_holm if math.isfinite(p))

    if (rmse_improves or pairwise_improves) and not any_position_worse and bias_growth <= bias_growth_max:
        return 'SHIP-ELIGIBLE'
    return 'INCONCLUSIVE'
