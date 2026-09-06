"""Phase 1 of the game-total-per-stat backtest (2026-09-04).

Fits a SEPARATE implied-game-total elasticity for every (position, projected
stat), replacing the one flat per-position number
``data.weekly_projections.GAME_TOTAL_ELASTICITY`` applies to every stat alike.

The model the fitted number plugs into is multiplicative:

    E[game_stat] = player_baseline_rate * (implied_team_total / league_avg) ** beta

so the estimator that matches it exactly is a Poisson / log-link rate model
with ``log(player_season_mean)`` as a fixed offset and ``log(implied_ratio)``
as the single regressor - one hand-rolled Newton-Raphson fit per (pos, stat),
no statsmodels dependency. It handles the pile of zeros in the TD / INT
columns natively, which a log-log OLS on ``game_stat / season_mean`` cannot.

Alongside it, a BINNED weighted least-squares fit on implied-ratio deciles is
reported as an interpretable cross-check and to carry the quadratic term
(does the response actually bend in log space, or is one elasticity enough?).

FIT WINDOW 2016-2023 - deliberately outside the 2024-2025 window
``scripts/gte_perstat_confirm.py`` scores the candidate on, the same
discipline ``GAME_TOTAL_ELASTICITY`` itself was built with.

    python scripts/fit_game_total_elasticity_perstat.py --mode fit --years 2016-2023
    python scripts/fit_game_total_elasticity_perstat.py --mode report   # re-print last CSV

`fit` writes:
  .sweeps/gte_perstat_fit_2016-2023.csv    - full table, both estimators, CIs, quad term
  .sweeps/gte_perstat_candidate.json       - {pos: {stat: poisson_beta}}, what the
                                             flag loads at build time until the
                                             numbers are hardcoded on ship
and prints a paste-ready GAME_TOTAL_ELASTICITY_BY_STAT dict + per-bin shape tables.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.transforms import load_and_merge_data  # noqa: E402
from data.weekly_projections import game_environment, load_schedule  # noqa: E402

STATS = ('passing_yards', 'passing_attempts', 'passing_completions', 'passing_tds',
         'passing_interceptions', 'rushing_yards', 'rushing_attempts', 'rushing_tds',
         'targets', 'receptions', 'receiving_yards', 'receiving_tds')
POSITIONS = ('QB', 'RB', 'WR', 'TE')
# Which stats are worth fitting for which position - a WR passing_yards row is
# noise, and its handful of trick-play games would fit an absurd elasticity.
POS_STATS = {
    'QB': ('passing_yards', 'passing_attempts', 'passing_completions', 'passing_tds',
           'passing_interceptions', 'rushing_yards', 'rushing_attempts', 'rushing_tds'),
    'RB': ('rushing_yards', 'rushing_attempts', 'rushing_tds', 'targets', 'receptions',
           'receiving_yards', 'receiving_tds'),
    'WR': ('targets', 'receptions', 'receiving_yards', 'receiving_tds'),
    'TE': ('targets', 'receptions', 'receiving_yards', 'receiving_tds'),
}
MIN_GAMES = 6          # a real season, not a cameo
MIN_SNAP_PCT = 25.0    # a game where he had a role
CSV_PATH = os.path.join('.sweeps', 'gte_perstat_fit_2016-2023.csv')
JSON_PATH = os.path.join('.sweeps', 'gte_perstat_candidate.json')

from data.weekly_projections import GAME_TOTAL_ELASTICITY  # noqa: E402


def _implied_ratio_by_team_week(year, weeks):
    """{(week, team): implied_total / league_avg_implied_that_week}."""
    sched = load_schedule(year)
    out = {}
    for wk in weeks:
        env = game_environment(sched, wk)
        vals = [e['implied'] for e in env.values() if e.get('implied')]
        if not vals:
            continue
        league = float(np.mean(vals))
        if league <= 0:
            continue
        for team, e in env.items():
            imp = e.get('implied')
            if imp and imp > 0:
                out[(wk, str(team))] = imp / league
    return out


def build_panel(years, weeks, scoring='Full PPR'):
    """One row per (player, season, week): his game line for every stat plus
    his own full-season mean of that stat (over games where he had a real
    role) and his team's implied-total ratio for the week."""
    frames = []
    for year in years:
        stats_df, _tc, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns or 'position' not in stats_df.columns:
            print(f"{year}: no weekly data, skipped", flush=True)
            continue
        df = stats_df.copy()
        df['_pos'] = df['position'].astype(str).str.upper()
        df = df[df['_pos'].isin(POSITIONS)].copy()
        df['_week'] = pd.to_numeric(df['week'], errors='coerce')
        df = df[df['_week'].isin(list(weeks))]
        for s in STATS:
            if s not in df.columns:
                df[s] = 0.0
            df[s] = pd.to_numeric(df[s], errors='coerce').fillna(0.0)
        snap = pd.to_numeric(df.get('weekly_snap_pct', 0.0), errors='coerce').fillna(0.0)
        df['_role_game'] = snap >= MIN_SNAP_PCT

        ratio = _implied_ratio_by_team_week(year, weeks)
        team_col = 'recent_team' if 'recent_team' in df.columns else (
            'team' if 'team' in df.columns else 'Team')
        df['_ratio'] = [ratio.get((int(w), str(t))) if pd.notna(w) else None
                        for w, t in zip(df['_week'], df[team_col])]
        df = df[df['_ratio'].notna()].copy()
        if df.empty:
            continue

        # Season mean per (player, stat) over role games only; fall back to all
        # games for a player who never cleared the snap bar but still has a
        # stable line (rare, mostly goal-line backs).
        gkey = [name_col, '_pos']
        role = df[df['_role_game']]
        n_role = role.groupby(gkey, observed=True)['_week'].nunique()
        keep_players = set(n_role[n_role >= MIN_GAMES].index)
        base = role if len(keep_players) else df
        means = base.groupby(gkey, observed=True)[list(STATS)].mean()
        df['_key'] = list(zip(df[name_col].astype(str), df['_pos']))
        for s in STATS:
            df[f'{s}__mean'] = df['_key'].map(
                {k: means.loc[k, s] for k in means.index if k in keep_players})
        df['_keep'] = df['_key'].map(lambda k: k in keep_players)
        df = df[df['_keep']].copy()
        df['_year'] = year
        frames.append(df[[name_col, '_pos', '_year', '_week', '_ratio', '_role_game']
                         + list(STATS) + [f'{s}__mean' for s in STATS]]
                      .rename(columns={name_col: 'player'}))
    if not frames:
        raise SystemExit("no panel rows built")
    return pd.concat(frames, ignore_index=True)


def poisson_beta(y, offset_log, x, iters=25, tol=1e-8):
    """MLE of b in  y ~ Poisson(exp(offset_log + a + b*x))  by Newton-Raphson
    on (a, b). Returns (a, b, converged)."""
    a, b = 0.0, 0.0
    x = np.asarray(x, float); y = np.asarray(y, float); off = np.asarray(offset_log, float)
    for _ in range(iters):
        mu = np.exp(off + a + b * x)
        # gradient / Hessian of the log-likelihood wrt (a, b)
        g = np.array([np.sum(y - mu), np.sum(x * (y - mu))])
        h00 = -np.sum(mu); h01 = -np.sum(x * mu); h11 = -np.sum(x * x * mu)
        H = np.array([[h00, h01], [h01, h11]])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return a, b, False
        a -= step[0]; b -= step[1]
        if np.max(np.abs(step)) < tol:
            return a, b, True
    return a, b, False


def binned_wls(ratio, val, mean, n_bins=10):
    """Rate-ratio vs implied-ratio on quantile bins. Returns (slope, quad,
    quad_ok, bin_rows) where slope is the linear elasticity and quad is the
    x^2 coefficient from a separate quadratic fit."""
    r = np.asarray(ratio, float); v = np.asarray(val, float); m = np.asarray(mean, float)
    ok = np.isfinite(r) & (r > 0) & np.isfinite(v) & np.isfinite(m) & (m > 0)
    r, v, m = r[ok], v[ok], m[ok]
    if len(r) < 200:
        return float('nan'), float('nan'), False, []
    edges = np.unique(np.quantile(r, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 4:
        return float('nan'), float('nan'), False, []
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (r >= lo) & (r <= hi)
        if b.sum() < 30 or m[b].sum() <= 0 or v[b].sum() <= 0:
            continue
        rows.append((np.log(r[b].mean()), np.log(v[b].sum() / m[b].sum()), int(b.sum())))
    if len(rows) < 4:
        return float('nan'), float('nan'), False, rows
    X = np.array([x for x, _, _ in rows]); Y = np.array([y for _, y, _ in rows])
    W = np.array([w for _, _, w in rows], float)
    slope = np.polyfit(X, Y, 1, w=np.sqrt(W))[0]
    quad = float('nan'); quad_ok = False
    if len(rows) >= 5:
        c = np.polyfit(X, Y, 2, w=np.sqrt(W))
        quad, quad_ok = float(c[0]), True
    return float(slope), quad, quad_ok, rows


def bootstrap_ci(ratio, val, mean, n=400, seed=0):
    rng = np.random.default_rng(seed)
    r = np.asarray(ratio, float); v = np.asarray(val, float); m = np.asarray(mean, float)
    idx = np.arange(len(r)); betas = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        b = binned_wls(r[s], v[s], m[s])[0]
        if np.isfinite(b):
            betas.append(b)
    if len(betas) < 30:
        return float('nan'), float('nan')
    return float(np.percentile(betas, 2.5)), float(np.percentile(betas, 97.5))


def do_fit(years, weeks):
    panel = build_panel(years, weeks)
    print(f"panel: {len(panel):,} player-games  "
          f"years={sorted(panel['_year'].unique())}  "
          f"implied-ratio range {panel['_ratio'].min():.2f}-{panel['_ratio'].max():.2f}\n", flush=True)

    out_rows = []
    candidate = {p: {} for p in POSITIONS}
    for pos in POSITIONS:
        pp = panel[panel['_pos'] == pos]
        print(f"{'=' * 78}\n{pos}   (n player-games = {len(pp):,})\n{'=' * 78}")
        for stat in POS_STATS[pos]:
            v = pp[stat].to_numpy(float)
            m = pp[f'{stat}__mean'].to_numpy(float)
            r = pp['_ratio'].to_numpy(float)
            ok = np.isfinite(m) & (m > 0) & np.isfinite(r) & (r > 0)
            v, m, r = v[ok], m[ok], r[ok]
            n = len(v)
            flat = GAME_TOTAL_ELASTICITY.get(pos, 0.0)
            if n < 400:
                print(f"  {stat:22s} n={n:<6d} too few, keeping flat {flat:+.3f}")
                candidate[pos][stat] = flat
                out_rows.append(dict(pos=pos, stat=stat, n=n, poisson_beta=flat,
                                     wls_slope=float('nan'), ci_lo=float('nan'),
                                     ci_hi=float('nan'), quad_coef=float('nan'),
                                     flat_elasticity=flat, note='n<400 kept flat'))
                continue
            a, b, conv = poisson_beta(v, np.log(m), np.log(r))
            wls_slope, quad, _qok, bins = binned_wls(r, v, m)
            lo, hi = bootstrap_ci(r, v, m)
            candidate[pos][stat] = round(float(b), 4)
            excl0 = (np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))
            print(f"  {stat:22s} n={n:<6d} poisson b={b:+.3f}{'' if conv else '!'}  "
                  f"wls={wls_slope:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]{' *' if excl0 else '  '}  "
                  f"quad={quad:+.3f}   (flat {flat:+.3f})")
            out_rows.append(dict(pos=pos, stat=stat, n=n, poisson_beta=round(float(b), 4),
                                 poisson_converged=conv, wls_slope=round(float(wls_slope), 4),
                                 ci_lo=round(lo, 4), ci_hi=round(hi, 4), ci_excludes_0=excl0,
                                 quad_coef=round(quad, 4), flat_elasticity=flat,
                                 delta_vs_flat=round(float(b) - flat, 4)))
            # per-bin shape, so a weird beta can be eyeballed
            for x, y, w in bins:
                out_rows.append(dict(pos=pos, stat=f'  bin:{stat}', n=w,
                                     poisson_beta=round(float(np.exp(x)), 4),   # implied ratio
                                     wls_slope=round(float(np.exp(y)), 4)))     # rate ratio
        print()

    os.makedirs('.sweeps', exist_ok=True)
    pd.DataFrame(out_rows).to_csv(CSV_PATH, index=False)
    with open(JSON_PATH, 'w', encoding='utf-8') as fh:
        json.dump(candidate, fh, indent=2)
    print(f"wrote {CSV_PATH}\nwrote {JSON_PATH}\n")
    print("GAME_TOTAL_ELASTICITY_BY_STAT = {")
    for pos in POSITIONS:
        inner = ', '.join(f"'{s}': {candidate[pos][s]:+.3f}" for s in POS_STATS[pos])
        print(f"    '{pos}': {{{inner}}},")
    print("}")


def do_report():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"no {CSV_PATH} - run --mode fit first")
    df = pd.read_csv(CSV_PATH)
    with pd.option_context('display.max_rows', None, 'display.width', 200):
        print(df[~df['stat'].astype(str).str.contains('bin:')].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['fit', 'report'])
    ap.add_argument('--years', default='2016-2023')
    ap.add_argument('--weeks', default='1-18')
    args = ap.parse_args()
    if args.mode == 'report':
        do_report(); return
    y0, y1 = (int(x) for x in args.years.split('-'))
    w0, w1 = (int(x) for x in args.weeks.split('-'))
    do_fit(list(range(y0, y1 + 1)), list(range(w0, w1 + 1)))


if __name__ == '__main__':
    main()
