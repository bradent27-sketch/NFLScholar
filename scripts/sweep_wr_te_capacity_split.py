"""v2_wr_te_capacity_split backtest + TE_MARGINAL_TARGET_WEIGHT sweep (2026-09-07).

WHAT SHIPPED, UNVALIDATED, ON 2026-09-07. data.pass_capacity_allocator's
team pass-capacity conservation used ONE uniform factor across a team's whole
WR/TE room. At cold start that room is off its pass-attempt budget almost
entirely because of an offseason WR-room change the model hasn't re-primed -
a wideout left and his targets weren't reclaimed (room under budget -> the
tight end gets scaled UP with the WRs: LAC Gadsden after Keenan Allen, GB
Kraft after Doubs+Wicks, TB Otton after Evans), or a high-volume wideout was
added (room over budget -> the tight end is docked WITH the WRs: IND Warren
after Keenan Allen arrived). `v2_wr_te_capacity_split` (weeks 1-2 only) fits
the WR and TE sub-rooms to SEPARATE budgets that divide the signed
off-budget delta by TE_MARGINAL_TARGET_WEIGHT (default 0.20), so a WR-room
swing lands ~80% on the WRs. Each tight end keeps his own prior-year target
rate as the anchor; only the reconciliation is damped.

This measures it. Paired A/B, weeks 1-2 (the only weeks the flag fires),
2022-2025 (the years with a frozen pre-Week-1 Ourlads archive). Base is the
shipped set with the flag REMOVED; each variant is the shipped set with the
flag on and pass_capacity_allocator.TE_MARGINAL_TARGET_WEIGHT monkeypatched
to a swept value. Both arms carry v2_historical_ourlads so the frozen chart
is actually read at cold start. Base is built once per (year, week) and
reused across every weight.

Reports dMAE = variant - base (NEGATIVE = the split is MORE accurate) on
ALL / QB / RB / WR / TE / START-WR / START-TE with a bootstrap 95% CI and a
weekly win-loss, per-stat startable WR and TE (targets/receptions/
receiving_yards), and a per-player ledger of every WR/TE the split moved on
a real board so the Gadsden / Kraft / Otton / Warren shapes can be judged
on 2022-2025 cases rather than an aggregate.

    python scripts/sweep_wr_te_capacity_split.py --years 2022,2023,2024,2025 --weeks 1-2
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data.pass_capacity_allocator as pca  # noqa: E402
from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from scripts.eval_weekly_model import _metrics, _weighted, STARTABLE_N, _actual_points  # noqa: E402

BASE_FEATS = frozenset((DEFAULT_FEATURES - {'v2_wr_te_capacity_split'}) | {'v2_historical_ourlads'})
VAR_FEATS = frozenset(DEFAULT_FEATURES | {'v2_historical_ourlads'})
WEIGHTS = (0.10, 0.15, 0.20, 0.25, 0.30)
SCOPES = [('ALL', None, False), ('QB', 'QB', False), ('RB', 'RB', False),
          ('WR', 'WR', False), ('TE', 'TE', False),
          ('START-WR', 'WR', True), ('START-TE', 'TE', True)]
PER_STAT = ('targets', 'receptions', 'receiving_yards')
STAT_SCOPES = (('START-TE', 'TE'), ('START-WR', 'WR'))


def _scope_df(df, pos, startable):
    d = df if pos is None else df[df['Pos'] == pos]
    if startable:
        d = d.nlargest(STARTABLE_N.get(pos, 30), 'Model Proj Pts')
    return d


def _boot_ci(deltas, weights, n=3000, seed=0):
    if len(deltas) < 4:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, float); w = np.asarray(weights, float)
    idx = np.arange(len(d)); out = np.empty(n)
    for b in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        out[b] = np.average(d[s], weights=w[s])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _sign_p(wins, losses):
    n = wins + losses
    if n == 0:
        return float('nan')
    k = max(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n))


def _stat_actuals(stats_df, name_col, week):
    rows = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week]
    present = [s for s in PER_STAT if s in rows.columns]
    return rows.groupby(name_col, observed=True)[present].sum()


def run(years, weeks, scoring, ledger_csv):
    scoring_col = 'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'
    scope_pairs = {w: {s[0]: [] for s in SCOPES} for w in WEIGHTS}
    stat_pairs = {w: {sc[0]: {st: [] for st in PER_STAT} for sc in STAT_SCOPES} for w in WEIGHTS}
    ledger = []

    for year in years:
        stats_df, _tc, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns:
            print(f"{year}: no weekly data, skipped", flush=True)
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                continue
            sa = _stat_actuals(stats_df, name_col, week)

            build_weekly_projections.clear()
            base, bmeta = build_weekly_projections(
                year, week, scoring, as_of_week=week, apply_injury=False, features=BASE_FEATS)
            build_weekly_projections.clear()
            if base.empty:
                print(f"{year} w{week}: base empty ({bmeta.get('reason')})", flush=True)
                continue

            for w in WEIGHTS:
                pca.TE_MARGINAL_TARGET_WEIGHT = w
                var, vmeta = build_weekly_projections(
                    year, week, scoring, as_of_week=week, apply_injury=False, features=VAR_FEATS)
                pca.TE_MARGINAL_TARGET_WEIGHT = 0.20
                build_weekly_projections.clear()
                if var.empty:
                    print(f"{year} w{week} [w={w}]: variant empty", flush=True)
                    continue
                pool = sorted(set(base['Player']) & set(var['Player']))
                if len(pool) < 20:
                    continue
                b = base[base['Player'].isin(pool)].set_index('Player')
                v = var[var['Player'].isin(pool)].set_index('Player')

                for scope, pos, startable in SCOPES:
                    bd = _scope_df(b.reset_index(), pos, startable)
                    vd = _scope_df(v.reset_index(), pos, startable)
                    mb = _metrics(pd.Series(bd['Model Proj Pts'].to_numpy(), index=bd['Player']), actual)
                    mv = _metrics(pd.Series(vd['Model Proj Pts'].to_numpy(), index=vd['Player']), actual)
                    if mb and mv:
                        scope_pairs[w][scope].append((mb, mv))

                for sc_name, sc_pos in STAT_SCOPES:
                    bd = _scope_df(b.reset_index(), sc_pos, True).set_index('Player')
                    for stat in PER_STAT:
                        if stat not in b.columns or stat not in sa.columns:
                            continue
                        common = bd.index.intersection(v.index).intersection(sa.index)
                        if len(common) < 3:
                            continue
                        act = pd.to_numeric(sa.loc[common, stat], errors='coerce')
                        eb = (pd.to_numeric(bd.loc[common, stat], errors='coerce') - act).abs().mean()
                        ev = (pd.to_numeric(v.loc[common, stat], errors='coerce') - act).abs().mean()
                        if np.isfinite(eb) and np.isfinite(ev):
                            stat_pairs[w][sc_name][stat].append((float(eb), float(ev), len(common)))

                moved = b.index[b['Pos'].isin(['WR', 'TE'])].intersection(v.index)
                for player in moved:
                    bp = float(b.loc[player, 'Model Proj Pts'])
                    vp = float(v.loc[player, 'Model Proj Pts'])
                    if abs(vp - bp) < 1e-6:
                        continue
                    av = float(actual.get(player, np.nan))
                    ledger.append(dict(
                        te_marginal_weight=w, year=year, week=week, player=player,
                        pos=str(b.loc[player, 'Pos']), team=str(b.loc[player, 'Team']),
                        base_proj=round(bp, 2), var_proj=round(vp, 2), shift=round(vp - bp, 3),
                        base_targets=round(float(b.loc[player, 'targets']), 2) if 'targets' in b.columns else None,
                        var_targets=round(float(v.loc[player, 'targets']), 2) if 'targets' in v.columns else None,
                        actual=round(av, 2) if np.isfinite(av) else None,
                        base_error=round(abs(bp - av), 2) if np.isfinite(av) else None,
                        var_error=round(abs(vp - av), 2) if np.isfinite(av) else None))
            print(f"{year} w{week}: base pool {len(base)}  swept {len(WEIGHTS)} weights", flush=True)

    _report(scope_pairs, stat_pairs, ledger, years, weeks)
    if ledger_csv and ledger:
        os.makedirs(os.path.dirname(ledger_csv) or '.', exist_ok=True)
        pd.DataFrame(ledger).sort_values(['te_marginal_weight', 'year', 'week', 'pos']).to_csv(ledger_csv, index=False)
        print(f"\nwrote {ledger_csv}  ({len(ledger)} moved (weight, WR/TE, week) rows)")


def _agg(pairs):
    if not pairs:
        return None
    mb = [p[0] for p in pairs]; mv = [p[1] for p in pairs]
    nt = sum(m['n'] for m in mb)
    base_mae, var_mae = _weighted(mb, 'mae'), _weighted(mv, 'mae')
    deltas = [x['mae'] - y['mae'] for y, x in zip(mb, mv)]
    wts = [y['n'] for y in mb]
    lo, hi = _boot_ci(deltas, wts)
    wins = sum(1 for d in deltas if d < 0); losses = sum(1 for d in deltas if d > 0)
    return nt, base_mae, var_mae, var_mae - base_mae, lo, hi, wins, losses


def _report(scope_pairs, stat_pairs, ledger, years, weeks):
    bar = '=' * 104
    print(f"\n{bar}\nv2_wr_te_capacity_split  x  TE_MARGINAL_TARGET_WEIGHT   years={years} weeks={weeks}\n{bar}")
    print("dMAE = variant - base.  NEGATIVE = the split is MORE accurate.  * = bootstrap 95% CI excludes 0.\n")
    for w in WEIGHTS:
        print(f"--- TE_MARGINAL_TARGET_WEIGHT = {w} ---")
        print(f"{'scope':<12}{'n':>7}{'MAE base':>11}{'MAE var':>11}{'dMAE':>10}{'95% CI':>22}{'wk W-L':>9}{'sign p':>8}")
        for scope, _p, _s in SCOPES:
            a = _agg(scope_pairs[w][scope])
            if a is None:
                continue
            nt, bmae, vmae, dmae, lo, hi, wins, losses = a
            star = '*' if (np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)) else ' '
            print(f"{scope:<12}{nt:>7}{bmae:>11.3f}{vmae:>11.3f}{dmae:>+10.3f}"
                  f"  [{lo:+.3f},{hi:+.3f}]{star}{f'{wins}-{losses}':>9}{_sign_p(wins, losses):>8.3f}")
        for sc_name, _pos in STAT_SCOPES:
            parts = []
            for stat in PER_STAT:
                rows = stat_pairs[w][sc_name][stat]
                if not rows:
                    parts.append(f"{stat}: -")
                    continue
                eb = np.array([r[0] for r in rows]); ev = np.array([r[1] for r in rows])
                ww = np.array([r[2] for r in rows], float)
                parts.append(f"{stat} {float(np.average(ev, weights=ww) - np.average(eb, weights=ww)):+.3f}")
            print(f"  per-stat {sc_name:<9} " + "   ".join(parts))
        print()

    if ledger:
        led = pd.DataFrame(ledger)
        print(f"{bar}\nWHO THE SPLIT MOVES (at TE_MARGINAL_TARGET_WEIGHT=0.20, rows with an actual)\n{bar}")
        # NB: 'shift' is a DataFrame method, so p.shift / r.shift resolve to
        # the bound method, not the column - always subscript it.
        sub = led[(led.te_marginal_weight == 0.20) & led.actual.notna()].copy()
        for pos in ('TE', 'WR'):
            p = sub[sub.pos == pos]
            if p.empty:
                continue
            helped = (p.var_error < p.base_error - 0.05).sum()
            hurt = (p.var_error > p.base_error + 0.05).sum()
            print(f"  {pos}: {len(p)} moved | helped {helped} / hurt {hurt} | "
                  f"mean |err| {p.base_error.mean():.2f} -> {p.var_error.mean():.2f} | "
                  f"mean shift {p['shift'].mean():+.2f}")
        print("\n  Largest TE moves (shift, most negative = docked hardest):")
        _te = sub[sub.pos == 'TE']
        te = _te.reindex(_te['shift'].abs().sort_values(ascending=False).index)
        for _, r in te.head(16).iterrows():
            print(f"   {r['year']} w{int(r['week'])} {r['player']:<22}{r['team']:<4} base={r['base_proj']:>6.2f} "
                  f"var={r['var_proj']:>6.2f} shift={r['shift']:>+6.2f}  actual={r['actual']}"
                  f"  err {r['base_error']}->{r['var_error']}")
    print("\n* = bootstrap 95% CI on the pooled weekly dMAE excludes zero.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2022,2023,2024,2025')
    ap.add_argument('--weeks', default='1-2')
    ap.add_argument('--scoring', default='Full PPR')
    ap.add_argument('--ledger-csv', default='.sweeps/wr_te_capacity_split_ledger.csv')
    a = ap.parse_args()
    years = [int(x) for x in a.years.replace(' ', '').split(',')]
    if '-' in a.weeks:
        w0, w1 = (int(x) for x in a.weeks.split('-'))
    else:
        w0 = w1 = int(a.weeks)
    run(years, list(range(w0, w1 + 1)), a.scoring, a.ledger_csv)


if __name__ == '__main__':
    main()
