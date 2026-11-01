"""
Is the Poisson that data/market_devig.py inverts a posted count line with too narrow? (2026-11-01, user: "look into the too
narrow poisson shape for attempts targets and receptions"; bar in .sweeps/count_shape_bar.md, set before this ran.)

For each count market, 2026 weeks 3-7: the last priced line before the player's team kicked off (Pinnacle, then DraftKings;
de-vigged P(over)), joined to the result (deep-audit panel, played rows). The market's implied mean is solved under a Poisson and
under a negative binomial with size r (var = mu + mu^2 / r), r fit by maximum likelihood leave-one-week-out. Scores per market:
log-likelihood per line, implied-mean MSE and bias against the result, and the share of results inside the central 50% interval.
Writes .sweeps/count_shape/: lines.csv, weeks.csv, summary.csv.

    python scripts/eval_count_line_shape.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.deep_audit import PANEL, PROP_SPECS, posted_join, posted_lines_2026  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.sweeps', 'count_shape')
MARKETS = {'passing_attempts': ('QB',), 'passing_completions': ('QB',), 'rushing_attempts': ('QB', 'RB'),
           'targets': ('RB', 'WR', 'TE'), 'receptions': ('RB', 'WR', 'TE')}
R_GRID = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 70.0, 100.0, 150.0, 250.0,
          500.0, 1000.0, np.inf)


def upper_tail(k, mu, r):
    """P(X >= k) under Poisson (r = inf) or NB(mu, r), arrays."""
    if np.isinf(r):
        return poisson.sf(k - 1, mu)
    return nbinom.sf(k - 1, r, r / (r + mu))


def implied_mean(line, p_over, r, iters=80):
    """mu with P(X > line) = p_over: the over wins at floor(line) + 1 or more (an x.5 line or a whole one with a push)."""
    k = np.floor(line) + 1
    p = np.clip(p_over, 1e-6, 1 - 1e-6)
    lo = np.full(len(k), 1e-6)
    hi = np.maximum(5 * k, k + 60.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = upper_tail(k, mid, r) < p
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def logpmf(y, mu, r):
    return poisson.logpmf(y, mu) if np.isinf(r) else nbinom.logpmf(y, r, r / (r + mu))


def mid_pit(y, mu, r):
    """F(y - 1) + pmf(y) / 2: where the result sits in the distribution (0.25-0.75 = inside the central half)."""
    if np.isinf(r):
        return poisson.cdf(y - 1, mu) + 0.5 * poisson.pmf(y, mu)
    q = r / (r + mu)
    return nbinom.cdf(y - 1, r, q) + 0.5 * nbinom.pmf(y, r, q)


def load_lines():
    p = pd.read_parquet(PANEL)
    p['played'] = p['played'].astype(bool)
    p = p[(p['year'] == 2026) & p['played']].reset_index(drop=True)
    arc, kick = posted_lines_2026()
    rows = []
    for stat, positions in MARKETS.items():
        d = p[p['pos'].isin(positions) & p[f'y_{stat}'].notna()].reset_index(drop=True)
        for _, L in posted_join(d, PROP_SPECS[stat], arc, kick).iterrows():
            i = int(L['i'])
            rows.append({'stat': stat, 'pos': d.at[i, 'pos'], 'player': d.at[i, 'player'], 'week': int(L['week']),
                         'book': L['book'], 'line': float(L['line']), 'p_over': float(L['p_book']),
                         'y': float(d.at[i, f'y_{stat}']), 'model': float(d.at[i, f'p_{stat}'])})
    lines = pd.DataFrame(rows)
    return lines[(lines['p_over'] > 0) & (lines['p_over'] < 1) & (lines['line'] >= 1.0)].reset_index(drop=True)


def best_r(g):
    """The grid r with the highest total log-likelihood on g."""
    ll = [logpmf(g['y'].to_numpy(), implied_mean(g['line'].to_numpy(), g['p_over'].to_numpy(), r), r).sum() for r in R_GRID]
    return R_GRID[int(np.argmax(ll))]


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = load_lines()
    print(f'{len(lines)} priced lines; whole-number lines: {int((lines["line"] % 1 == 0).sum())}')
    weeks = sorted(lines['week'].unique())
    scored, week_rows = [], []
    for stat in MARKETS:
        s = lines[lines['stat'] == stat]
        for w in weeks:
            te, tr = s[s['week'] == w].copy(), s[s['week'] != w]
            if not len(te) or len(tr) < 50:
                continue
            r = best_r(tr)
            r_pos = {ps: best_r(tr[tr['pos'] == ps]) if (tr['pos'] == ps).sum() >= 50 else r for ps in te['pos'].unique()}
            y, ln, po = te['y'].to_numpy(), te['line'].to_numpy(), te['p_over'].to_numpy()
            mu_p = implied_mean(ln, po, np.inf)
            mu_n = implied_mean(ln, po, r)
            rp = te['pos'].map(r_pos).to_numpy(float)
            mu_np = np.array([implied_mean(np.array([a]), np.array([b]), c)[0] for a, b, c in zip(ln, po, rp)])
            te['r'] = r
            te['mu_poisson'], te['mu_nb'], te['mu_nb_pos'] = mu_p, mu_n, mu_np
            te['ll_poisson'], te['ll_nb'] = logpmf(y, mu_p, np.inf), logpmf(y, mu_n, r)
            te['ll_nb_pos'] = [logpmf(np.array([a]), np.array([b]), c)[0] for a, b, c in zip(y, mu_np, rp)]
            te['pit_poisson'], te['pit_nb'] = mid_pit(y, mu_p, np.inf), mid_pit(y, mu_n, r)
            scored.append(te)
            week_rows.append({'stat': stat, 'week': w, 'n': len(te), 'r': r, 'r_by_pos': str(r_pos),
                              'll_poisson': te['ll_poisson'].mean(), 'll_nb': te['ll_nb'].mean(), 'll_nb_pos': te['ll_nb_pos'].mean(),
                              'mse_poisson': np.mean((y - mu_p) ** 2), 'mse_nb': np.mean((y - mu_n) ** 2)})
    sc = pd.concat(scored, ignore_index=True)
    sc.to_csv(os.path.join(OUT, 'lines.csv'), index=False)
    wk = pd.DataFrame(week_rows)
    wk.to_csv(os.path.join(OUT, 'weeks.csv'), index=False)
    out = []
    for stat, g in sc.groupby('stat', sort=False):
        w = wk[wk['stat'] == stat]
        y = g['y'].to_numpy()
        rec = {'stat': stat, 'n': len(g), 'r_all_weeks': best_r(g),
               'r_by_pos_all_weeks': str({ps: best_r(g[g['pos'] == ps]) for ps in sorted(g['pos'].unique()) if (g['pos'] == ps).sum() >= 50}),
               'll_poisson': g['ll_poisson'].mean(), 'll_nb': g['ll_nb'].mean(), 'll_nb_pos': g['ll_nb_pos'].mean(),
               'weeks_nb_better': int((w['ll_nb'] > w['ll_poisson']).sum()), 'weeks': len(w),
               'weeks_pos_beats_one_r': int((w['ll_nb_pos'] > w['ll_nb']).sum()),
               'mse_poisson': np.mean((y - g['mu_poisson']) ** 2), 'mse_nb': np.mean((y - g['mu_nb']) ** 2),
               'bias_poisson': np.mean(g['mu_poisson'] - y), 'bias_nb': np.mean(g['mu_nb'] - y),
               'bias_line': np.mean(g['line'] - y), 'mean_mu_shift': np.mean(g['mu_nb'] - g['mu_poisson']),
               'cover50_poisson': np.mean(g['pit_poisson'].between(0.25, 0.75)),
               'cover50_nb': np.mean(g['pit_nb'].between(0.25, 0.75))}
        rec['passes'] = bool(rec['ll_nb'] > rec['ll_poisson'] and rec['weeks_nb_better'] >= 4 and rec['mse_nb'] <= rec['mse_poisson'])
        out.append(rec)
    summ = pd.DataFrame(out)
    summ.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(summ.round(4).to_string(index=False))
        print(wk.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
