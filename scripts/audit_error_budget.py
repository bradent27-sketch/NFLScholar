"""
Error budget (model audit step 1, 2026-10-21): where does the current model's START-pool error come from?

Each scored row's projected stat line is split into opportunities (targets, carries, pass attempts) and per-opportunity rates,
and the opportunities further into the team total and the player's share of it:

    stat = team opportunities x share x rate            (team total and share over the team's board rows, same players for both)

One piece at a time is swapped for what actually happened (an "oracle") and the row is re-scored. The drop in squared error is
the most that piece could ever be worth: an upper bound, not a forecastable gain. Pieces:

    team    the team's total targets / carries / pass attempts (pace, run/pass mix, game script)
    share   the player's share of them (role, injuries around him, snaps)
    opps    both (the player's own opportunities)
    eff     non-TD rates: catch rate, yards per target / carry / attempt, completion and interception rates
    tds     TDs per opportunity
    all     every scored stat (what is left is fumbles, two-point plays and return TDs)

Scoring: Full PPR points from the stat line, then one least-squares line per position fit to the actuals on the same rows, for
the base model and for every oracle alike (in-sample, so the board's own calibration is not a handicap). The second table is
the Poisson floor per stat: if every projection were the true mean, Poisson randomness alone would leave this share of the
observed squared error (real counts are over-dispersed, so the true floor is higher and the reducible part smaller).

Pool: harness v2 START (top STARTABLE_N per year/week/position by the board's points, played, box score). The board is the
latest capture's board dump (current DEFAULT_FEATURES; the capture verified it identical to a plain build).

    python scripts/audit_error_budget.py [--board .sweeps/qb_style_capture.parquet.board.csv] [--out .sweeps/audit_error_budget.txt]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.eval_calibration_schemes import WEIGHTS, load_dump, top_n_mask  # noqa: E402

KEYS = ['year', 'week', 'player', 'pos', 'team']
POSITIONS = ('QB', 'RB', 'WR', 'TE')
# opportunity stat -> the rate stats measured per opportunity (TD stats last)
FAMILIES = {
    'passing_attempts': ('passing_completions', 'passing_yards', 'passing_interceptions', 'passing_tds'),
    'rushing_attempts': ('rushing_yards', 'rushing_tds'),
    'targets': ('receptions', 'receiving_yards', 'receiving_tds'),
}
TD_STATS = {'passing_tds', 'rushing_tds', 'receiving_tds'}
ORACLES = {
    'base': (),
    'team': ('team',),
    'share': ('share',),
    'opps': ('team', 'share'),
    'eff': ('eff',),
    'tds': ('tds',),
    'opps+eff': ('team', 'share', 'eff'),
    'opps+eff+tds': ('team', 'share', 'eff', 'tds'),
}


def load(path):
    d = load_dump(path)
    d = d[d['week'].between(3, 17)].drop_duplicates(KEYS, keep=False).reset_index(drop=True)
    d['wk_key'] = d['year'] * 100 + d['week']
    for opp, rates in FAMILIES.items():
        for s in (opp,) + rates:
            for pre in ('p_', 'a_'):
                d[pre + s] = pd.to_numeric(d[pre + s], errors='coerce').fillna(0.0)
    d['pool'] = top_n_mask(d, d['final'].to_numpy(float)) & d['played'].to_numpy() & d['actual'].notna().to_numpy()
    return d


def team_totals(d, col):
    return d.groupby(['year', 'week', 'team'])[col].transform('sum').to_numpy(float)


def oracle_line(d, parts):
    """{stat: array} with the named pieces taken from the actuals, everything else as projected."""
    out = {s: d['p_' + s].to_numpy(float).copy() for s in WEIGHTS}
    for opp, rates in FAMILIES.items():
        p, a = d['p_' + opp].to_numpy(float), d['a_' + opp].to_numpy(float)
        P, A = team_totals(d, 'p_' + opp), team_totals(d, 'a_' + opp)
        n = p.copy()
        if 'team' in parts and 'share' in parts:
            n = a.copy()
        elif 'team' in parts:
            n = np.where(P > 0, p * A / np.where(P > 0, P, 1.0), p)
        elif 'share' in parts:
            n = np.where((A > 0) & (P > 0), a * P / np.where(A > 0, A, 1.0), p)
        for s in rates:
            if s not in WEIGHTS:
                continue
            ps, as_ = d['p_' + s].to_numpy(float), d['a_' + s].to_numpy(float)
            live = (p > 0) & (d['played'].to_numpy())
            lg = {pos: (d.loc[live & (d['pos'] == pos).to_numpy(), 'a_' + s].sum()
                        / max(d.loc[live & (d['pos'] == pos).to_numpy(), 'a_' + opp].sum(), 1e-9)) for pos in POSITIONS}
            r_p = np.where(p > 0, ps / np.where(p > 0, p, 1.0), d['pos'].map(lg).to_numpy(float))
            use_actual = ('tds' in parts) if s in TD_STATS else ('eff' in parts)
            r = np.where(use_actual & (a > 0), as_ / np.where(a > 0, a, 1.0), r_p)
            stat = n * r
            # rows with no projected opportunities keep their projected stat unless an oracle moved them
            out[s] = np.where((p > 0) | (n > 0), stat, ps)
    return out


def points(d, line):
    return np.clip(sum(w * line[s] for s, w in WEIGHTS.items()), 0.0, None)


def fitted_rmse(x, y):
    """RMSE of y on the least-squares line in x (one line per position, in-sample)."""
    if len(x) < 3 or np.std(x) == 0:
        return float(np.sqrt(np.mean((y - np.mean(y)) ** 2)))
    b, a = np.polyfit(x, y, 1)
    return float(np.sqrt(np.mean((y - (a + b * x)) ** 2)))


def budget(d, preds, mask_fn):
    rows = []
    for pos in POSITIONS + ('ALL',):
        m = d['pool'].to_numpy() & mask_fn(d)
        if pos != 'ALL':
            m &= (d['pos'] == pos).to_numpy()
        y = d['actual'].to_numpy(float)
        row = {'pos': pos, 'n': int(m.sum()), 'shipped': float(np.sqrt(np.mean((d['final'].to_numpy(float)[m] - y[m]) ** 2)))}
        for name, pred in preds.items():
            if pos == 'ALL':
                se = []
                for p in POSITIONS:
                    mm = m & (d['pos'] == p).to_numpy()
                    se.append(fitted_rmse(pred[mm], y[mm]) ** 2 * mm.sum())
                row[name] = float(np.sqrt(sum(se) / m.sum()))
            else:
                row[name] = fitted_rmse(pred[m], y[m])
        rows.append(row)
    t = pd.DataFrame(rows).set_index('pos')
    share = t.copy()
    for c in preds:
        share[c] = 1.0 - t[c] ** 2 / t['base'] ** 2
    return t, share


def poisson_floor(d):
    rows = []
    m0 = d['pool'].to_numpy()
    for pos in POSITIONS:
        m = m0 & (d['pos'] == pos).to_numpy()
        for opp, rates in FAMILIES.items():
            for s in (opp,) + rates:
                if s in ('passing_completions',):
                    continue
                p, a = d['p_' + s].to_numpy(float)[m], d['a_' + s].to_numpy(float)[m]
                if p.mean() < 0.05:
                    continue
                mse = np.mean((a - p) ** 2)
                rows.append({'pos': pos, 'stat': s, 'mean proj': p.mean(), 'mean actual': a.mean(), 'RMSE': np.sqrt(mse),
                             'poisson floor share': min(p.mean() / mse, 9.99),
                             'pts weight': WEIGHTS.get(s, 0.0),
                             'pts MSE (stat x weight)': mse * WEIGHTS.get(s, 0.0) ** 2})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--board', default='.sweeps/qb_style_capture.parquet.board.csv')
    ap.add_argument('--out', default='.sweeps/audit_error_budget.txt')
    a = ap.parse_args()
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 30)
    d = load(a.board)
    preds = {name: points(d, oracle_line(d, parts)) for name, parts in ORACLES.items()}
    preds['all'] = np.clip(sum(w * d['a_' + s].to_numpy(float) for s, w in WEIGHTS.items()), 0.0, None)
    check = np.abs(preds['base'] - d['raw'].to_numpy(float))[d['pool'].to_numpy()]
    lines = [f"board {a.board}: {len(d):,} rows, START pool {int(d['pool'].sum()):,} (2021-2025 weeks 3-17)",
             f"base line rebuilt from the stat line vs the board's raw: median |diff| {np.median(check):.3f}, "
             f"99th pct {np.percentile(check, 99):.3f}", '']
    t, share = budget(d, preds, lambda x: np.ones(len(x), bool))
    lines += ['RMSE on the START pool, each column re-fit by one line per position (shipped = the board as scored):',
              t.round(3).to_string(), '',
              'Share of the base squared error removed by knowing that piece exactly (upper bound per piece):',
              share.drop(columns=['n', 'shipped', 'base']).round(3).to_string(), '']
    for y in sorted(d['year'].unique()):
        _, sh = budget(d, preds, lambda x, y=y: (x['year'] == y).to_numpy())
        lines.append(f"{y}: " + '  '.join(f"{c} {sh.loc['ALL', c]:.3f}" for c in ('team', 'share', 'opps', 'eff', 'tds', 'all')))
    lines += ['', 'Per stat on the START pool (Poisson floor share = share of the observed squared error that pure Poisson noise '
              'around a perfect mean would leave; >1 means under-dispersed vs Poisson):',
              poisson_floor(d).round(3).to_string(index=False)]
    text = '\n'.join(lines)
    print(text)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
