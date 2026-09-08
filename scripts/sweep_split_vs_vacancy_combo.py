"""split vs vacancy vs both - confirmation run (2026-09-07).

Two cold-start receiver-room levers, each already backtested alone:

  v2_wr_te_capacity_split        currently a DEFAULT. Full weight sweep
                                 (.sweeps/wr_te_capacity_split_2022-2025.txt)
                                 says wash on ALL, significant WR / START-WR
                                 cost at every weight - no weight rescues it.
  v2_receiver_cold_start_vacancy  currently OFF. Week-1 backtest
                                 (.sweeps/receiver_cold_start_vacancy_wk1_2022-2025.txt)
                                 says START-WR -0.47 (4-0), START-TE +0.29 -
                                 the better-shaped version of the same fix.

Neither was measured with the OTHER flag toggled. This run pins the BASE at
today's shipped default stack (split ON, vacancy OFF) and scores four
configs against it over weeks 1-2, so a swap vs. a keep-both decision has a
combined number behind it:

  BASE     split ON,  vacancy OFF   (today)
  swap     split OFF, vacancy ON
  combo    split ON,  vacancy ON
  neither  split OFF, vacancy OFF

All four carry v2_historical_ourlads so the frozen pre-Week-1 archive is
available for a historical year. dMAE = config - BASE; NEGATIVE beats today.

    python scripts/sweep_split_vs_vacancy_combo.py --years 2022,2023,2024,2025 --weeks 1-2
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

from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from scripts.eval_weekly_model import _metrics, _weighted, STARTABLE_N, _actual_points  # noqa: E402

_SPLIT = 'v2_wr_te_capacity_split'
_VAC = 'v2_receiver_cold_start_vacancy'
_OURLADS = 'v2_historical_ourlads'

BASE_FEATS = frozenset(DEFAULT_FEATURES | {_OURLADS})              # split ON, vacancy OFF
CONFIGS = {
    'swap':    frozenset((BASE_FEATS - {_SPLIT}) | {_VAC}),        # split OFF, vacancy ON
    'combo':   frozenset(BASE_FEATS | {_VAC}),                     # split ON,  vacancy ON
    'neither': frozenset(BASE_FEATS - {_SPLIT}),                   # split OFF, vacancy OFF
}

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


def _build(year, week, scoring, feats):
    build_weekly_projections.clear()
    df, meta = build_weekly_projections(
        year, week, scoring, as_of_week=week, apply_injury=False, features=feats)
    build_weekly_projections.clear()
    return df, meta


def run(years, weeks, scoring, ledger_csv):
    scoring_col = 'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'
    # scope_pairs[cfg][scope] = list of (metrics_base, metrics_cfg) per week
    scope_pairs = {c: {s[0]: [] for s in SCOPES} for c in CONFIGS}
    stat_pairs = {c: {sc[0]: {st: [] for st in PER_STAT} for sc in STAT_SCOPES} for c in CONFIGS}
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

            base, bmeta = _build(year, week, scoring, BASE_FEATS)
            if base.empty:
                print(f"{year} w{week}: base empty ({bmeta.get('reason')})", flush=True)
                continue
            variants = {}
            for cfg, feats in CONFIGS.items():
                v, vmeta = _build(year, week, scoring, feats)
                if v.empty:
                    print(f"{year} w{week} [{cfg}]: empty ({vmeta.get('reason')})", flush=True)
                variants[cfg] = v

            for cfg, v in variants.items():
                if v.empty:
                    continue
                pool = sorted(set(base['Player']) & set(v['Player']))
                if len(pool) < 20:
                    continue
                b = base[base['Player'].isin(pool)].set_index('Player')
                vv = v[v['Player'].isin(pool)].set_index('Player')

                for scope, pos, startable in SCOPES:
                    bd = _scope_df(b.reset_index(), pos, startable)
                    vd = _scope_df(vv.reset_index(), pos, startable)
                    mb = _metrics(pd.Series(bd['Model Proj Pts'].to_numpy(), index=bd['Player']), actual)
                    mv = _metrics(pd.Series(vd['Model Proj Pts'].to_numpy(), index=vd['Player']), actual)
                    if mb and mv:
                        scope_pairs[cfg][scope].append((mb, mv))

                for sc_name, sc_pos in STAT_SCOPES:
                    bd = _scope_df(b.reset_index(), sc_pos, True).set_index('Player')
                    for stat in PER_STAT:
                        if stat not in b.columns or stat not in sa.columns:
                            continue
                        common = bd.index.intersection(vv.index).intersection(sa.index)
                        if len(common) < 3:
                            continue
                        act = pd.to_numeric(sa.loc[common, stat], errors='coerce')
                        eb = (pd.to_numeric(bd.loc[common, stat], errors='coerce') - act).abs().mean()
                        ev = (pd.to_numeric(vv.loc[common, stat], errors='coerce') - act).abs().mean()
                        if np.isfinite(eb) and np.isfinite(ev):
                            stat_pairs[cfg][sc_name][stat].append((float(eb), float(ev), len(common)))

                moved = b.index[b['Pos'].isin(['WR', 'TE'])].intersection(vv.index)
                for player in moved:
                    bp = float(b.loc[player, 'Model Proj Pts'])
                    vp = float(vv.loc[player, 'Model Proj Pts'])
                    if abs(vp - bp) < 1e-6:
                        continue
                    av = float(actual.get(player, np.nan))
                    ledger.append(dict(
                        config=cfg, year=year, week=week, player=player,
                        pos=str(b.loc[player, 'Pos']), team=str(b.loc[player, 'Team']),
                        base_proj=round(bp, 2), var_proj=round(vp, 2), shift=round(vp - bp, 3),
                        actual=round(av, 2) if np.isfinite(av) else None,
                        base_error=round(abs(bp - av), 2) if np.isfinite(av) else None,
                        var_error=round(abs(vp - av), 2) if np.isfinite(av) else None))
            print(f"{year} w{week}: base pool {len(base)}  built {len(CONFIGS)} configs", flush=True)

    _report(scope_pairs, stat_pairs, ledger, years, weeks)
    if ledger_csv and ledger:
        os.makedirs(os.path.dirname(ledger_csv) or '.', exist_ok=True)
        pd.DataFrame(ledger).sort_values(['config', 'year', 'week', 'pos']).to_csv(ledger_csv, index=False)
        print(f"\nwrote {ledger_csv}  ({len(ledger)} moved (config, WR/TE, week) rows)")


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
    print(f"\n{bar}\nsplit vs vacancy vs both   years={years} weeks={weeks}\n{bar}")
    print("BASE = today's default stack (v2_wr_te_capacity_split ON, v2_receiver_cold_start_vacancy OFF).")
    print("dMAE = config - BASE.  NEGATIVE = the config beats today.  * = bootstrap 95% CI excludes 0.\n")
    for cfg in CONFIGS:
        print(f"--- {cfg}  ({'split OFF' if _SPLIT not in CONFIGS[cfg] else 'split ON'}, "
              f"{'vacancy ON' if _VAC in CONFIGS[cfg] else 'vacancy OFF'}) ---")
        print(f"{'scope':<12}{'n':>7}{'MAE base':>11}{'MAE cfg':>11}{'dMAE':>10}{'95% CI':>22}{'wk W-L':>9}{'sign p':>8}")
        for scope, _p, _s in SCOPES:
            a = _agg(scope_pairs[cfg][scope])
            if a is None:
                continue
            nt, bmae, vmae, dmae, lo, hi, wins, losses = a
            star = '*' if (np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)) else ' '
            print(f"{scope:<12}{nt:>7}{bmae:>11.3f}{vmae:>11.3f}{dmae:>+10.3f}"
                  f"  [{lo:+.3f},{hi:+.3f}]{star}{f'{wins}-{losses}':>9}{_sign_p(wins, losses):>8.3f}")
        for sc_name, _pos in STAT_SCOPES:
            parts = []
            for stat in PER_STAT:
                rows = stat_pairs[cfg][sc_name][stat]
                if not rows:
                    parts.append(f"{stat}: -")
                    continue
                eb = np.array([r[0] for r in rows]); ev = np.array([r[1] for r in rows])
                ww = np.array([r[2] for r in rows], float)
                parts.append(f"{stat} {float(np.average(ev, weights=ww) - np.average(eb, weights=ww)):+.3f}")
            print(f"  per-stat {sc_name:<9} " + "   ".join(parts))
        # population-weighted startable bottom line (START-WR + START-TE)
        sw, ste = _agg(scope_pairs[cfg]['START-WR']), _agg(scope_pairs[cfg]['START-TE'])
        if sw and ste:
            n_w, n_t = sw[0], ste[0]
            net = (sw[3] * n_w + ste[3] * n_t) / (n_w + n_t)
            print(f"  startable net (START-WR {sw[3]:+.3f}*n{n_w} + START-TE {ste[3]:+.3f}*n{n_t}) "
                  f"/ n{n_w + n_t} = {net:+.3f} MAE per startable player")
        print()

    print(f"{bar}\nWHO EACH CONFIG MOVES (WR/TE rows with an actual)\n{bar}")
    if ledger:
        led = pd.DataFrame(ledger)
        for cfg in CONFIGS:
            sub = led[(led.config == cfg) & led.actual.notna()]
            for pos in ('TE', 'WR'):
                p = sub[sub.pos == pos]
                if p.empty:
                    continue
                helped = (p.var_error < p.base_error - 0.05).sum()
                hurt = (p.var_error > p.base_error + 0.05).sum()
                print(f"  {cfg:<8} {pos}: {len(p):>4} moved | helped {helped} / hurt {hurt} | "
                      f"mean |err| {p.base_error.mean():.2f} -> {p.var_error.mean():.2f} | "
                      f"mean shift {p['shift'].mean():+.2f}")
    print("\n* = bootstrap 95% CI on the pooled weekly dMAE excludes zero.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2022,2023,2024,2025')
    ap.add_argument('--weeks', default='1-2')
    ap.add_argument('--scoring', default='Full PPR')
    ap.add_argument('--ledger-csv', default='.sweeps/split_vs_vacancy_combo_ledger.csv')
    a = ap.parse_args()
    years = [int(x) for x in a.years.replace(' ', '').split(',')]
    if '-' in a.weeks:
        w0, w1 = (int(x) for x in a.weeks.split('-'))
    else:
        w0 = w1 = int(a.weeks)
    run(years, list(range(w0, w1 + 1)), a.scoring, a.ledger_csv)


if __name__ == '__main__':
    main()
