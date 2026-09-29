"""
Backtest `v2_xtd` (docs/model_improvement_plan_2026-09-23.md item 6) with
Poisson deviance and Brier score as PRIMARY, per the plan's own point 5:
MAE (and by extension a naive points-RMSE-only read) rewards a mostly-zero
TD count for staying low, so the standard scripts/backtest_component.py
points table is printed only as SECONDARY context here, never as the
ship/reject signal.

Reuses `build_weekly_projections`/`scripts.harness_v2.td_metrics` directly
rather than extending backtest_component.py's shared ablation harness - a
per-STAT rate metric (not a per-player POINTS metric) needs a different
pairing (actual per-player STAT count that week, not fantasy points), so
this is its own small script rather than a mode bolted onto that one.

Usage:
    python scripts/backtest_xtd.py --years 2022-2025 --weeks 2-6
    python scripts/backtest_xtd.py --years 2022-2025 --weeks 7-17
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from scripts import harness_v2  # noqa: E402

STATS = ('receiving_tds', 'rushing_tds')


def _actual_stat_totals(stats_df, name_col, week, stat):
    wk = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week]
    if wk.empty or stat not in wk.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(wk[stat], errors='coerce').fillna(0.0).groupby(wk[name_col]).sum()


def run(years, weeks, scoring='Full PPR'):
    variant_feats = DEFAULT_FEATURES | {'v2_xtd'}
    # rows[stat] = list of (mu_base, mu_variant, actual) arrays, one per
    # scored (year, week) - concatenated once at the end for the pooled
    # deviance/Brier, and kept per-week too for a bootstrap CI on the delta.
    rows = {stat: [] for stat in STATS}
    points_rows = []

    for year in years:
        stats_df, _team_col, name_col, _ = load_and_merge_data(year, scoring)
        if stats_df.empty or 'week' not in stats_df.columns:
            print(f"{year}: no weekly data, skipped")
            continue
        for week in weeks:
            base_proj, base_meta = build_weekly_projections(
                year, week, scoring, as_of_week=week, apply_injury=False, features=DEFAULT_FEATURES)
            if base_proj.empty:
                continue
            var_proj, var_meta = build_weekly_projections(
                year, week, scoring, as_of_week=week, apply_injury=False, features=variant_feats)
            if var_proj.empty:
                continue
            pool = sorted(set(base_proj['Player']) & set(var_proj['Player']))
            if len(pool) < 20:
                continue

            actual_points = pd.to_numeric(
                stats_df.loc[pd.to_numeric(stats_df['week'], errors='coerce') == week]
                .groupby(name_col)['fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'].sum(),
                errors='coerce')
            base_idx = base_proj.set_index('Player')['Model Proj Pts']
            var_idx = var_proj.set_index('Player')['Model Proj Pts']
            common_actual = actual_points.reindex(pool).dropna()
            if len(common_actual) >= 20:
                mb = harness_v2.metrics(base_idx.reindex(common_actual.index), common_actual)
                mv = harness_v2.metrics(var_idx.reindex(common_actual.index), common_actual)
                if mb and mv:
                    points_rows.append((mb, mv))

            for stat in STATS:
                if stat not in base_proj.columns or stat not in var_proj.columns:
                    continue
                actual = _actual_stat_totals(stats_df, name_col, week, stat).reindex(pool)
                base_stat = base_proj.set_index('Player')[stat].reindex(pool)
                var_stat = var_proj.set_index('Player')[stat].reindex(pool)
                joined = pd.DataFrame({'base': base_stat, 'var': var_stat, 'actual': actual}).dropna()
                if len(joined) < 20:
                    continue
                rows[stat].append({
                    'year': year, 'week': week,
                    'mu_base': joined['base'].to_numpy(dtype=float),
                    'mu_var': joined['var'].to_numpy(dtype=float),
                    'y': joined['actual'].to_numpy(dtype=float),
                })

    print(f"years={years} weeks={weeks[0]}-{weeks[-1]} scoring={scoring}\n")

    for stat in STATS:
        entries = rows[stat]
        if not entries:
            print(f"{stat}: no data")
            continue
        all_mu_base = np.concatenate([e['mu_base'] for e in entries])
        all_mu_var = np.concatenate([e['mu_var'] for e in entries])
        all_y = np.concatenate([e['y'] for e in entries])
        m_base = harness_v2.td_metrics(all_mu_base, all_y)
        m_var = harness_v2.td_metrics(all_mu_var, all_y)

        # Week-cluster bootstrap CI on the pooled-deviance delta (resample
        # WEEKS, not player-rows, so within-week correlation isn't treated
        # as independent evidence - same discipline as the points harness).
        deviance_deltas = []
        brier_deltas = []
        weights = []
        for e in entries:
            mb = harness_v2.td_metrics(e['mu_base'], e['y'])
            mv = harness_v2.td_metrics(e['mu_var'], e['y'])
            deviance_deltas.append(mv['poisson_deviance_mean'] - mb['poisson_deviance_mean'])
            brier_deltas.append(mv['brier'] - mb['brier'])
            weights.append(mb['n'])
        deviance_ci = harness_v2.bootstrap_ci(deviance_deltas, weights)
        brier_ci = harness_v2.bootstrap_ci(brier_deltas, weights)

        print(f"=== {stat} (n={m_base['n']}) - PRIMARY metrics ===")
        print(f"  Poisson deviance/obs: {m_base['poisson_deviance_mean']:.4f} -> "
              f"{m_var['poisson_deviance_mean']:.4f}  (d{m_var['poisson_deviance_mean']-m_base['poisson_deviance_mean']:+.4f} "
              f"CI[{deviance_ci[0]:+.4f},{deviance_ci[1]:+.4f}])  [lower is better]")
        print(f"  Brier (P(TD>=1)):     {m_base['brier']:.4f} -> {m_var['brier']:.4f}  "
              f"(d{m_var['brier']-m_base['brier']:+.4f} CI[{brier_ci[0]:+.4f},{brier_ci[1]:+.4f}])  [lower is better]")
        zero_frac_base = float(np.mean(all_mu_base <= 1e-6))
        zero_frac_var = float(np.mean(all_mu_var <= 1e-6))
        print(f"  Exact-zero rate: {zero_frac_base:.3f} -> {zero_frac_var:.3f}  "
              f"(v2_xtd's own point 7: should fall toward 0 for anyone with real opportunity)\n")

    if points_rows:
        mb_list = [p[0] for p in points_rows]
        mv_list = [p[1] for p in points_rows]
        n_total = sum(m['n'] for m in mb_list)
        rmse_b = float(np.average([m['rmse'] for m in mb_list], weights=[m['n'] for m in mb_list]))
        rmse_v = float(np.average([m['rmse'] for m in mv_list], weights=[m['n'] for m in mv_list]))
        print(f"=== Model Proj Pts (n={n_total}) - SECONDARY context only, per the plan's own"
              f" MAE/points caution for TD stats ===")
        print(f"  RMSE: {rmse_b:.3f} -> {rmse_v:.3f} (d{rmse_v-rmse_b:+.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2022-2025')
    ap.add_argument('--weeks', default='2-6')
    ap.add_argument('--scoring', default='Full PPR')
    args = ap.parse_args()
    lo, hi = args.years.split('-')
    years = list(range(int(lo), int(hi) + 1))
    wlo, whi = args.weeks.split('-')
    weeks = list(range(int(wlo), int(whi) + 1))
    run(years, weeks, args.scoring)


if __name__ == '__main__':
    main()
