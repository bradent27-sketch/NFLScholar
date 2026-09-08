"""v2_receiver_cold_start_vacancy cross-position bleed sweep (2026-09-07).

The shipped vacancy flag is a hard same-position silo: 100% of a departed
WR's vacated role goes to the remaining WRs, nothing to the TEs. This sweeps
RECEIVER_COLD_START_VACANCY_CROSS_POS_FRACTION - the share that instead
bleeds to the TE room (LAC/GB/TB "a wideout left and the targets landed on
the tight end" is the motivating shape, so the question is whether a *small*
deliberate TE cut reads better than 0).

BASE = today's default stack (vacancy ON, fraction 0.0). VAR = same stack
with the module constant monkeypatched to each fraction. Week 1 only - the
flag is cold-start gated. dMAE = variant - base; NEGATIVE = the bleed helps.

    python scripts/sweep_receiver_cold_start_vacancy_crosspos.py --years 2022,2023,2024,2025 --weeks 1-1
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

import data.weekly_projections as wp  # noqa: E402
from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from scripts.eval_weekly_model import _metrics, _weighted, STARTABLE_N, _actual_points  # noqa: E402

FEATS = frozenset(DEFAULT_FEATURES | {'v2_historical_ourlads'})   # vacancy already in DEFAULT
FRACTIONS = (0.10, 0.20, 0.30)
SCOPES = [('ALL', None, False), ('QB', 'QB', False), ('RB', 'RB', False),
          ('WR', 'WR', False), ('TE', 'TE', False),
          ('START-WR', 'WR', True), ('START-TE', 'TE', True)]
STAT_SCOPES = [('START-WR', 'WR'), ('START-TE', 'TE')]
PER_STAT = ('targets', 'receptions', 'receiving_yards')


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


def _build(year, week, scoring):
    build_weekly_projections.clear()
    df, meta = build_weekly_projections(
        year, week, scoring, as_of_week=week, apply_injury=False, features=FEATS)
    build_weekly_projections.clear()
    return df, meta


def run(years, weeks, scoring, ledger_csv):
    scoring_col = 'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'
    scope_pairs = {f: {s[0]: [] for s in SCOPES} for f in FRACTIONS}
    stat_pairs = {f: {sc[0]: {st: [] for st in PER_STAT} for sc in STAT_SCOPES} for f in FRACTIONS}
    ledger = []
    _orig = wp.RECEIVER_COLD_START_VACANCY_CROSS_POS_FRACTION

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

            wp.RECEIVER_COLD_START_VACANCY_CROSS_POS_FRACTION = 0.0
            base, bmeta = _build(year, week, scoring)
            if base.empty:
                print(f"{year} w{week}: base empty ({bmeta.get('reason')})", flush=True)
                continue

            for f in FRACTIONS:
                wp.RECEIVER_COLD_START_VACANCY_CROSS_POS_FRACTION = f
                var, vmeta = _build(year, week, scoring)
                wp.RECEIVER_COLD_START_VACANCY_CROSS_POS_FRACTION = _orig
                if var.empty:
                    print(f"{year} w{week} [f={f}]: variant empty", flush=True)
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
                        scope_pairs[f][scope].append((mb, mv))

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
                            stat_pairs[f][sc_name][stat].append((float(eb), float(ev), len(common)))

                moved = b.index[b['Pos'].isin(['WR', 'TE'])].intersection(v.index)
                for player in moved:
                    bp = float(b.loc[player, 'Model Proj Pts'])
                    vp = float(v.loc[player, 'Model Proj Pts'])
                    if abs(vp - bp) < 1e-6:
                        continue
                    av = float(actual.get(player, np.nan))
                    ledger.append(dict(
                        fraction=f, year=year, week=week, player=player,
                        pos=str(b.loc[player, 'Pos']), team=str(b.loc[player, 'Team']),
                        base_proj=round(bp, 2), var_proj=round(vp, 2), shift=round(vp - bp, 3),
                        actual=round(av, 2) if np.isfinite(av) else None,
                        base_error=round(abs(bp - av), 2) if np.isfinite(av) else None,
                        var_error=round(abs(vp - av), 2) if np.isfinite(av) else None))
            print(f"{year} w{week}: base pool {len(base)}  swept {len(FRACTIONS)} fractions", flush=True)

    wp.RECEIVER_COLD_START_VACANCY_CROSS_POS_FRACTION = _orig
    _report(scope_pairs, stat_pairs, ledger, years, weeks)
    if ledger_csv and ledger:
        os.makedirs(os.path.dirname(ledger_csv) or '.', exist_ok=True)
        pd.DataFrame(ledger).sort_values(['fraction', 'year', 'week', 'pos']).to_csv(ledger_csv, index=False)
        print(f"\nwrote {ledger_csv}  ({len(ledger)} moved (fraction, WR/TE, week) rows)")


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
    bar = '=' * 100
    print(f"\n{bar}\nv2_receiver_cold_start_vacancy  x  CROSS_POS_FRACTION   years={years} weeks={weeks}\n{bar}")
    print("BASE = shipped default (vacancy ON, cross-pos fraction 0.0 - hard same-position silo).")
    print("dMAE = variant - base.  NEGATIVE = bleeding some WR vacancy to the TE room helps.  * = 95% CI excludes 0.\n")
    for f in FRACTIONS:
        print(f"--- CROSS_POS_FRACTION = {f}  (departed WR pool: {100*(1-f):.0f}% -> WR room, {100*f:.0f}% -> TE room) ---")
        print(f"{'scope':<12}{'n':>7}{'MAE base':>11}{'MAE var':>11}{'dMAE':>10}{'95% CI':>22}{'wk W-L':>9}{'sign p':>8}")
        for scope, _p, _s in SCOPES:
            a = _agg(scope_pairs[f][scope])
            if a is None:
                continue
            nt, bmae, vmae, dmae, lo, hi, wins, losses = a
            star = '*' if (np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)) else ' '
            print(f"{scope:<12}{nt:>7}{bmae:>11.3f}{vmae:>11.3f}{dmae:>+10.3f}"
                  f"  [{lo:+.3f},{hi:+.3f}]{star}{f'{wins}-{losses}':>9}{_sign_p(wins, losses):>8.3f}")
        for sc_name, _pos in STAT_SCOPES:
            parts = []
            for stat in PER_STAT:
                rows = stat_pairs[f][sc_name][stat]
                if not rows:
                    parts.append(f"{stat}: -")
                    continue
                eb = np.array([r[0] for r in rows]); ev = np.array([r[1] for r in rows])
                ww = np.array([r[2] for r in rows], float)
                parts.append(f"{stat} {float(np.average(ev, weights=ww) - np.average(eb, weights=ww)):+.3f}")
            print(f"  per-stat {sc_name:<9} " + "   ".join(parts))
        sw, ste = _agg(scope_pairs[f]['START-WR']), _agg(scope_pairs[f]['START-TE'])
        if sw and ste:
            n_w, n_t = sw[0], ste[0]
            net = (sw[3] * n_w + ste[3] * n_t) / (n_w + n_t)
            print(f"  startable net (START-WR {sw[3]:+.3f}*n{n_w} + START-TE {ste[3]:+.3f}*n{n_t}) "
                  f"/ n{n_w + n_t} = {net:+.3f} MAE per startable player")
        print()

    if ledger:
        led = pd.DataFrame(ledger)
        print(f"{bar}\nWHO THE BLEED MOVES (rows with an actual)\n{bar}")
        for f in FRACTIONS:
            sub = led[(led.fraction == f) & led.actual.notna()]
            for pos in ('TE', 'WR'):
                p = sub[sub.pos == pos]
                if p.empty:
                    continue
                helped = (p.var_error < p.base_error - 0.05).sum()
                hurt = (p.var_error > p.base_error + 0.05).sum()
                print(f"  f={f}  {pos}: {len(p):>4} moved | helped {helped} / hurt {hurt} | "
                      f"mean |err| {p.base_error.mean():.2f} -> {p.var_error.mean():.2f} | "
                      f"mean shift {p['shift'].mean():+.2f}")
    print("\n* = bootstrap 95% CI on the pooled weekly dMAE excludes zero.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2022,2023,2024,2025')
    ap.add_argument('--weeks', default='1-1')
    ap.add_argument('--scoring', default='Full PPR')
    ap.add_argument('--ledger-csv', default='.sweeps/receiver_cold_start_vacancy_crosspos_ledger.csv')
    a = ap.parse_args()
    years = [int(x) for x in a.years.replace(' ', '').split(',')]
    if '-' in a.weeks:
        w0, w1 = (int(x) for x in a.weeks.split('-'))
    else:
        w0 = w1 = int(a.weeks)
    run(years, list(range(w0, w1 + 1)), a.scoring, a.ledger_csv)


if __name__ == '__main__':
    main()
