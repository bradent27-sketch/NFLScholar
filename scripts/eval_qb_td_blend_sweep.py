"""
Offline sweep of how the model estimates a QB's passing TDs, using the traces from scripts/capture_qb_td_traces.py.

Everything is re-scored EXACTLY from the captured inputs, no builds:

    blended  = w * current_rate + (1 - w) * prior                      the in-season blend
    w        = games / (games + k_eff),  k_eff = K * (1.3 - 0.6 * role_confidence)   (K = 5 shipped)
    final TD = blended * (matchup * pace * environment * ... multipliers)   ->   final' = final * blended' / blended

and the QB points follow:  raw' = raw + 4 * (final' - final);  points = clip(0.738 * raw' + 4.154, 0), the shipped line.

Knobs (all within-model, no market input):
    K      the blend constant (shipped 5; the harness-tested candidate was 15)
    rho    regress the PRIOR rate toward the league per-game QB TD rate (the prior is one noisy season too)
    lam    pull the final TD toward what the QB's own projected passing YARDS imply at the league TD/yard rate
           (yards are far more stable than TDs, so a TD count anchored to yards is a slower, steadier read)

Scored on the harness's pool: the top 24 QBs per week by the shipped projection over ALL rows, then the players who
played; against the shipped model (K=5, rho=0, lam=0). Leave-one-year-out for choosing, weekly bootstrap for the CI.

    python scripts/eval_qb_td_blend_sweep.py [--traces .sweeps/qb_td_traces.parquet]
"""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.harness_v2 import pairwise_acc  # noqa: E402

QB_LINE = (0.738, 4.154)
STARTABLE_QB = 24
K_SHIPPED = 5.0
KS = (5, 8, 12, 15, 20, 30, 45, 70, 120, 1000)
RHOS = (0.0, 0.25, 0.5, 0.75, 1.0)
LAMS = (0.0, 0.25)


def load(path):
    t = pd.read_parquet(path)
    for c in ('current_games', 'current_rate', 'prior_rate', 'current_weight', 'role_confidence', 'blended_rate',
              'final_projection', 'raw', 'model', 'p_att', 'p_yds', 'p_td', 'a_pts', 'a_passing_tds',
              'a_passing_yards', 'a_passing_attempts', 'availability'):
        t[c] = pd.to_numeric(t[c], errors='coerce')
    t['played'] = t['a_pts'].notna()
    # harness pool: top-N by the shipped projection over all rows of the week
    t['rank'] = t.groupby(['year', 'week'])['model'].rank(ascending=False, method='first')
    t['start'] = t['rank'] <= STARTABLE_QB
    return t


def league_rates(t):
    s = t[t['played'] & (t['p_att'] > 15)]
    return (float(s['a_passing_tds'].mean()), float(s['a_passing_tds'].sum() / s['a_passing_yards'].sum()))


def emulate(t, K=K_SHIPPED, rho=0.0, lam=0.0, league_td=1.45, td_per_yd=0.0063):
    g = t['current_games'].to_numpy(dtype=float)
    conf = t['role_confidence'].fillna(0.5).to_numpy(dtype=float)
    ke = K * (1.3 - 0.6 * conf)
    w = np.where(g > 0, g / (g + ke), 0.0)
    prior = t['prior_rate'].to_numpy(dtype=float)
    prior_new = prior + rho * (league_td - prior)
    cur = np.nan_to_num(t['current_rate'].to_numpy(dtype=float))
    blended_old = t['blended_rate'].to_numpy(dtype=float)
    blended_new = w * cur + (1 - w) * np.nan_to_num(prior_new)
    ratio = np.where(blended_old > 1e-9, blended_new / np.where(blended_old > 1e-9, blended_old, 1.0), 1.0)
    final_old = t['final_projection'].to_numpy(dtype=float)
    final_new = final_old * ratio
    if lam:
        final_new = (1 - lam) * final_new + lam * td_per_yd * t['p_yds'].to_numpy(dtype=float)
    final_new = np.where(t['p_att'].to_numpy(dtype=float) > 1, final_new, final_old)   # nothing projected stays put
    raw_new = t['raw'].to_numpy(dtype=float) + 4.0 * (final_new - final_old)
    pts = np.clip(QB_LINE[0] * raw_new + QB_LINE[1], 0.0, None)
    # a QB with nothing projected (a backup, 0 attempts) shows 0, not the line's intercept
    # (v2_uncalibrate_zero_rows) - keep the board's own number for those rows
    pts = np.where(t['p_att'].to_numpy(dtype=float) > 1, pts, t['model'].to_numpy(dtype=float))
    return pts, final_new


def metrics(sub, pts, td):
    e = pts - sub['a_pts'].to_numpy()
    pw = []
    for _, g in sub.assign(_p=pts).groupby(['year', 'week']):
        if len(g) > 4:
            pw.append(pairwise_acc(g['_p'], g['a_pts']))
    starters = (sub['p_att'] > 15).to_numpy()
    td_rmse = float(np.sqrt(((td[starters] - sub['a_passing_tds'].to_numpy()[starters]) ** 2).mean()))
    return {'rmse': float(np.sqrt((e ** 2).mean())), 'bias': float(e.mean()), 'pw': float(np.nanmean(pw)), 'td_rmse': td_rmse}


def week_sq_err(sub, pts):
    e2 = (pts - sub['a_pts'].to_numpy()) ** 2
    return pd.DataFrame({'year': sub['year'].to_numpy(), 'week': sub['week'].to_numpy(), 'e2': e2}).groupby(['year', 'week'])['e2'].agg(['sum', 'count'])


def boot_ci(base_w, var_w, n=2000, seed=7):
    """95% CI of the pooled RMSE change, resampling weeks."""
    rng = np.random.default_rng(seed)
    bs, bc, vs = base_w['sum'].to_numpy(), base_w['count'].to_numpy(), var_w['sum'].to_numpy()
    idx = np.arange(len(bs))
    out = []
    for _ in range(n):
        i = rng.choice(idx, len(idx))
        out.append(np.sqrt(vs[i].sum() / bc[i].sum()) - np.sqrt(bs[i].sum() / bc[i].sum()))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', default='.sweeps/qb_td_traces.parquet')
    a = ap.parse_args()
    t = load(a.traces)
    league_td, td_per_yd = league_rates(t)
    pool = t[t['start'] & t['played']].copy()
    pd.set_option('display.width', 220)
    print(f"QB rows {len(t)}, START pool played rows {len(pool)}, years {sorted(t['year'].unique())}, "
          f"weeks {int(t['week'].min())}-{int(t['week'].max())}; league TD/game {league_td:.3f}, TD/yard {td_per_yd:.5f}")

    # sanity: the emulation at the shipped settings must give back the board
    base_pts, base_td = emulate(pool, league_td=league_td, td_per_yd=td_per_yd)
    gap = np.abs(base_pts - pool['model'].to_numpy())
    ke_check = pool['current_games'] * (1 - pool['current_weight']) / pool['current_weight'].where(pool['current_weight'] > 0)
    print(f"emulation check at K=5: max |points - board| {gap.max():.3f}, mean {gap.mean():.4f}; "
          f"traced k_eff vs K*(1.3-0.6*conf): max diff {(ke_check - 5 * (1.3 - 0.6 * pool['role_confidence'])).abs().max():.3f}")
    base = metrics(pool, base_pts, base_td)
    base_w = week_sq_err(pool, base_pts)
    print(f"shipped: START-QB RMSE {base['rmse']:.4f} bias {base['bias']:+.3f} pairwise {base['pw']:.4f} TD-count RMSE {base['td_rmse']:.4f}")

    rows = []
    for K, rho, lam in itertools.product(KS, RHOS, LAMS):
        if (K, rho, lam) == (K_SHIPPED, 0.0, 0.0):
            continue
        pts, td = emulate(pool, K, rho, lam, league_td, td_per_yd)
        m = metrics(pool, pts, td)
        by_year = {}
        for y, sub in pool.groupby('year'):
            ix = pool.index.get_indexer(sub.index)
            by_year[int(y)] = float(np.sqrt(((pts[ix] - sub['a_pts'].to_numpy()) ** 2).mean())
                                    - np.sqrt(((base_pts[ix] - sub['a_pts'].to_numpy()) ** 2).mean()))
        rows.append({'K': K, 'rho': rho, 'lam': lam, 'dRMSE': m['rmse'] - base['rmse'], 'dBias': m['bias'] - base['bias'],
                     'dPairwise': m['pw'] - base['pw'], 'TD RMSE': m['td_rmse'],
                     'years better': sum(v < 0 for v in by_year.values()), 'of': len(by_year),
                     'worst year': max(by_year.values())})
    res = pd.DataFrame(rows).sort_values('dRMSE')
    surface = res[res.lam == 0].pivot(index='K', columns='rho', values='dRMSE')
    print('\nSTART-QB RMSE change vs shipped, K (rows) x rho (columns), lam 0 (negative = better):')
    print(surface.round(4).to_string())
    print('\nbest 15 of the grid by START-QB RMSE change vs shipped (negative = better):')
    print(res.head(15).round(4).to_string(index=False))
    print('\nK alone (rho 0, lam 0):')
    print(res[(res.rho == 0) & (res.lam == 0)].sort_values('K').round(4).to_string(index=False))
    print('\nlam alone (K 5, rho 0):')
    print(res[(res.K == 5) & (res.rho == 0)].sort_values('lam').round(4).to_string(index=False))
    print('\nrho alone (K 5, lam 0):')
    print(res[(res.K == 5) & (res.lam == 0)].sort_values('rho').round(4).to_string(index=False))

    # confidence for the winners
    print('\nweekly-bootstrap 95% CI of the RMSE change for the top 5 and the K=15 reference:')
    picks = [tuple(r) for r in res.head(5)[['K', 'rho', 'lam']].to_numpy()] + [(15, 0.0, 0.0)]
    for K, rho, lam in picks:
        pts, td = emulate(pool, K, rho, lam, league_td, td_per_yd)
        lo, hi = boot_ci(base_w, week_sq_err(pool, pts))
        print(f"  K {K:g} rho {rho} lam {lam}: dRMSE {metrics(pool, pts, td)['rmse'] - base['rmse']:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]")

    # leave-one-year-out: choose on the other years, score on the held-out year
    print('\nleave-one-year-out (parameters chosen on the other years, scored on the held-out year), change vs shipped:')
    tot = []
    for y in sorted(pool['year'].unique()):
        fit, test = pool[pool['year'] != y], pool[pool['year'] == y]
        best, best_r = None, None
        for K, rho, lam in itertools.product(KS, RHOS, LAMS):
            pts, _ = emulate(fit, K, rho, lam, league_td, td_per_yd)
            r = float(np.sqrt(((pts - fit['a_pts'].to_numpy()) ** 2).mean()))
            if best_r is None or r < best_r:
                best, best_r = (K, rho, lam), r
        pts_t, _ = emulate(test, *best, league_td, td_per_yd)
        pts_b, _ = emulate(test, K_SHIPPED, 0.0, 0.0, league_td, td_per_yd)
        d = float(np.sqrt(((pts_t - test['a_pts'].to_numpy()) ** 2).mean()) - np.sqrt(((pts_b - test['a_pts'].to_numpy()) ** 2).mean()))
        tot.append(d)
        print(f"  {int(y)}: chosen K {best[0]:g} rho {best[1]} lam {best[2]} -> {d:+.4f}")
    print(f"  mean {np.mean(tot):+.4f}, better in {sum(x < 0 for x in tot)} of {len(tot)} held-out years")


if __name__ == '__main__':
    main()
