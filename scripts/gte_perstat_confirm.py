"""Phase 2 of the game-total-per-stat backtest (2026-09-04).

Held-out confirm for 'v2_game_total_elasticity_perstat' - the per-(position,
stat) implied-total elasticity fitted by
scripts/fit_game_total_elasticity_perstat.py (which must have run first; this
reads the .sweeps/gte_perstat_candidate.json it writes, via the flag's own
loader).

Paired weekly A/B, same discipline as scripts/backtest_component.py:

  A (base)      = DEFAULT_FEATURES  (ships the FLAT per-position elasticity)
  B (variant)   = DEFAULT_FEATURES + v2_game_total_elasticity_perstat

scored every week on the INTERSECTION of the two player pools, so a variant
that merely drops hard players never gets undeserved credit.

Two readouts:

  1. MODEL PROJ PTS MAE by scope - ALL, per position, and startable
     (top QB24/RB40/WR55/TE20 by projection, the pool a lineup/prop is
     actually chosen from - the GATE metric). Bootstrap 95% CI on the pooled
     per-week dMAE + an exact sign test on weeks won/lost.

  2. PER-STAT MAE - for each of the 12 projected stats, |proj - actual| for
     that stat alone, base vs variant, over the startable pool. This is the
     point of the whole exercise: which stats does a per-stat elasticity
     actually make more accurate, and which does it leave alone.

    python scripts/gte_perstat_confirm.py --years 2024,2025 --weeks 3-18 \
        --outliers-csv .sweeps/gte_perstat_outliers.csv
    python scripts/gte_perstat_confirm.py --years 2021,2022,2023 --weeks 3-18
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from scripts.eval_weekly_model import _metrics, _weighted, STARTABLE_N, _actual_points  # noqa: E402

VARIANT = frozenset(DEFAULT_FEATURES | {'v2_game_total_elasticity_perstat'})
SCOPES = [('ALL', None, False),
          ('QB', 'QB', False), ('RB', 'RB', False), ('WR', 'WR', False), ('TE', 'TE', False),
          ('START-QB', 'QB', True), ('START-RB', 'RB', True),
          ('START-WR', 'WR', True), ('START-TE', 'TE', True)]
PER_STAT = ('passing_yards', 'passing_attempts', 'passing_completions', 'passing_tds',
            'passing_interceptions', 'rushing_yards', 'rushing_attempts', 'rushing_tds',
            'targets', 'receptions', 'receiving_yards', 'receiving_tds')
# Which startable position pool each stat's per-stat MAE is measured over.
STAT_SCOPE = {
    'passing_yards': ('QB',), 'passing_attempts': ('QB',), 'passing_completions': ('QB',),
    'passing_tds': ('QB',), 'passing_interceptions': ('QB',),
    'rushing_yards': ('QB', 'RB'), 'rushing_attempts': ('QB', 'RB'), 'rushing_tds': ('QB', 'RB'),
    'targets': ('RB', 'WR', 'TE'), 'receptions': ('RB', 'WR', 'TE'),
    'receiving_yards': ('RB', 'WR', 'TE'), 'receiving_tds': ('RB', 'WR', 'TE'),
}


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
    g = rows.groupby(name_col, observed=True)[present].sum()
    return g


def run(years, weeks, scoring, outliers_csv):
    scoring_col = 'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'
    # scope -> list of (metrics_base, metrics_var) per week
    scope_pairs = {s[0]: [] for s in SCOPES}
    # per stat -> list of (mae_base, mae_var, n) per week, startable pool
    stat_pairs = {s: [] for s in PER_STAT}
    outliers = []

    for year in years:
        stats_df, _tc, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns:
            print(f"{year}: no weekly data, skipped", flush=True)
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                continue
            base, bmeta = build_weekly_projections(
                year, week, scoring, as_of_week=week, apply_injury=False, features=DEFAULT_FEATURES)
            var, vmeta = build_weekly_projections(
                year, week, scoring, as_of_week=week, apply_injury=False, features=VARIANT)
            if base.empty or var.empty:
                print(f"{year} w{week}: empty ({bmeta.get('reason') or vmeta.get('reason')})")
                continue
            pool = sorted(set(base['Player']) & set(var['Player']))
            if len(pool) < 20:
                continue
            b = base[base['Player'].isin(pool)]
            v = var[var['Player'].isin(pool)]

            for scope, pos, startable in SCOPES:
                bd, vd = _scope_df(b, pos, startable), _scope_df(v, pos, startable)
                mb = _metrics(pd.Series(bd['Model Proj Pts'].to_numpy(), index=bd['Player']), actual)
                mv = _metrics(pd.Series(vd['Model Proj Pts'].to_numpy(), index=vd['Player']), actual)
                if mb and mv:
                    scope_pairs[scope].append((mb, mv))

            sa = _stat_actuals(stats_df, name_col, week)
            for stat in PER_STAT:
                if stat not in b.columns or stat not in sa.columns:
                    continue
                bd = pd.concat([_scope_df(b, p, True) for p in STAT_SCOPE.get(stat, ('RB', 'WR', 'TE'))])
                vd = v[v['Player'].isin(bd['Player'])]
                bd = bd.set_index('Player'); vd = vd.set_index('Player')
                common = bd.index.intersection(vd.index).intersection(sa.index)
                if len(common) < 5:
                    continue
                act = pd.to_numeric(sa.loc[common, stat], errors='coerce')
                eb = (pd.to_numeric(bd.loc[common, stat], errors='coerce') - act).abs().mean()
                ev = (pd.to_numeric(vd.loc[common, stat], errors='coerce') - act).abs().mean()
                if np.isfinite(eb) and np.isfinite(ev):
                    stat_pairs[stat].append((float(eb), float(ev), len(common)))

            # outlier ledger: variant's own startable misses + how far the
            # per-stat flag moved each row from base
            for pos in ('QB', 'RB', 'WR', 'TE'):
                vd = _scope_df(v, pos, True).set_index('Player')
                bd = b.set_index('Player')
                for player, r in vd.iterrows():
                    if player not in actual.index:
                        continue
                    pv, av = float(r['Model Proj Pts']), float(actual[player])
                    base_pp = float(bd.loc[player, 'Model Proj Pts']) if player in bd.index else np.nan
                    outliers.append(dict(year=year, week=week, player=player, pos=pos,
                                         base_proj=round(base_pp, 2), var_proj=round(pv, 2),
                                         actual=round(av, 2), var_error=round(pv - av, 2),
                                         perstat_shift=round(pv - base_pp, 3),
                                         abs_error=round(abs(pv - av), 2)))
            print(f"{year} w{week}: pool {len(pool)}  base/var built", flush=True)

    _report(scope_pairs, stat_pairs, years, weeks)
    if outliers_csv:
        os.makedirs(os.path.dirname(outliers_csv) or '.', exist_ok=True)
        pd.DataFrame(outliers).sort_values('abs_error', ascending=False).to_csv(outliers_csv, index=False)
        print(f"\nwrote outlier ledger -> {outliers_csv}")


def _report(scope_pairs, stat_pairs, years, weeks):
    bar = '=' * 92
    print(f"\n{bar}\nGAME-TOTAL PER-STAT ELASTICITY - held-out confirm   years={years} weeks={weeks}\n{bar}")
    print("dMAE = variant - base.  NEGATIVE = per-stat elasticity is MORE accurate than the flat one.\n")
    print(f"{'scope':<12}{'n':>8}{'MAE base':>11}{'MAE var':>11}{'dMAE':>10}{'95% CI':>22}"
          f"{'wk W-L':>10}{'sign p':>9}")
    for scope, _p, _s in SCOPES:
        pairs = scope_pairs.get(scope) or []
        if not pairs:
            continue
        mb = [p[0] for p in pairs]; mv = [p[1] for p in pairs]
        nt = sum(m['n'] for m in mb)
        base_mae, var_mae = _weighted(mb, 'mae'), _weighted(mv, 'mae')
        deltas = [mv_i['mae'] - mb_i['mae'] for mb_i, mv_i in zip(mb, mv)]
        wts = [mb_i['n'] for mb_i in mb]
        lo, hi = _boot_ci(deltas, wts)
        wins = sum(1 for d in deltas if d < 0)   # variant better that week
        losses = sum(1 for d in deltas if d > 0)
        star = ' *' if (np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)) else '  '
        print(f"{scope:<12}{nt:>8}{base_mae:>11.3f}{var_mae:>11.3f}{var_mae - base_mae:>+10.3f}"
              f"  [{lo:+.3f},{hi:+.3f}]{star}{f'{wins}-{losses}':>10}{_sign_p(wins, losses):>9.3f}")

    print(f"\n{bar}\nPER-STAT MAE (startable pool)   NEGATIVE dMAE = per-stat elasticity helps that stat\n{bar}")
    print(f"{'stat':<24}{'wk n':>7}{'obs':>9}{'MAE base':>11}{'MAE var':>11}{'dMAE':>10}{'95% CI':>22}{'W-L':>9}")
    for stat in PER_STAT:
        rows = stat_pairs.get(stat) or []
        if not rows:
            continue
        eb = np.array([r[0] for r in rows]); ev = np.array([r[1] for r in rows])
        w = np.array([r[2] for r in rows], float)
        base_mae = float(np.average(eb, weights=w)); var_mae = float(np.average(ev, weights=w))
        deltas = list(ev - eb)
        lo, hi = _boot_ci(deltas, list(w))
        wins = int((ev < eb).sum()); losses = int((ev > eb).sum())
        star = ' *' if (np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)) else '  '
        print(f"{stat:<24}{len(rows):>7}{int(w.sum()):>9}{base_mae:>11.3f}{var_mae:>11.3f}"
              f"{var_mae - base_mae:>+10.3f}  [{lo:+.3f},{hi:+.3f}]{star}{f'{wins}-{losses}':>9}")
    print("\n* = bootstrap 95% CI on the pooled weekly dMAE excludes zero.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2024,2025')
    ap.add_argument('--weeks', default='3-18')
    ap.add_argument('--scoring', default='Full PPR')
    ap.add_argument('--outliers-csv', default=None)
    a = ap.parse_args()
    years = [int(x) for x in a.years.replace(' ', '').split(',')]
    w0, w1 = (int(x) for x in a.weeks.split('-'))
    run(years, list(range(w0, w1 + 1)), a.scoring, a.outliers_csv)


if __name__ == '__main__':
    main()
