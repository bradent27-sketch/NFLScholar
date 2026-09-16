"""
Sweep DEFENSE_BLOWOUT_WEIGHT_DISCOUNT for 'v2_defense_blowout_discount' - a
component NOT in DEFAULT_FEATURES (see docs/weekly_projections_methodology.md's
2026-09-15 entry: the shipped-constant guess of 0.5 backtested as a wash
overall, a real-looking win for START-QB/RB/TE, and a real-looking loss for
START-WR - built, backtested, rejected at that ONE hand-picked value). This
finds out whether a differently-tuned discount changes that verdict, and
whether WR genuinely can't be helped at any single global value before any
per-position/per-stat differentiation gets built.

Neither existing sweep tool covers this by itself:
  - scripts/sweep_model_constant.py sweeps a constant, but assumes its flag is
    ALREADY in DEFAULT_FEATURES (both base and variant build with
    features=DEFAULT_FEATURES; only the constant differs).
  - scripts/backtest_component.py's --add compares DEFAULT_FEATURES against
    DEFAULT_FEATURES+flag for an unshipped candidate, but only at whatever
    value the module constant currently holds - it does not sweep it.

This does both at once: base = DEFAULT_FEATURES (flag off, built ONCE per
week and reused across every swept value, same efficiency principle as
backtest_component.py); each variant = DEFAULT_FEATURES + the flag, with
DEFENSE_BLOWOUT_WEIGHT_DISCOUNT monkeypatched to one swept value for that
build only. Same bootstrap-CI / sign-test discipline as backtest_component.py
(imported directly, not reimplemented) so a value's apparent edge over the
shipped 0.5 can be told apart from noise at this sample size.

build_weekly_projections is @st.cache_data-keyed on its call args, which do
NOT include the module constant - cleared before every variant build (not
before base, which never reads this constant since its flag is off) to avoid
silently serving one discount value's cached result under another's key.

Usage:
    python scripts/sweep_defense_blowout_discount.py --values 0,0.25,0.5,0.75,1.0 \
        --years 2025 --weeks 5-17
    python scripts/sweep_defense_blowout_discount.py --values 0.3,0.4 \
        --years 2024,2025 --weeks 5-17   # confirm run on the coarse winner(s)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data.weekly_projections as wp  # noqa: E402
from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from scripts.eval_weekly_model import _actual_points, _weighted  # noqa: E402
from scripts.backtest_component import SCOPES, _scope_metrics, _bootstrap_ci, _sign_test_p  # noqa: E402

FLAG = 'v2_defense_blowout_discount'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--values', default='0.0,0.25,0.5,0.75,1.0',
                    help="comma-separated DEFENSE_BLOWOUT_WEIGHT_DISCOUNT values to test "
                         "(1.0 is a sanity check - no discount at all, should read ~0 delta)")
    ap.add_argument('--years', default='2024,2025')
    ap.add_argument('--weeks', default='5-17')
    ap.add_argument('--scoring', default='Full PPR')
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(',')]
    if '-' in args.weeks:
        lo, hi = args.weeks.split('-')
        weeks = list(range(int(lo), int(hi) + 1))
    else:
        weeks = [int(w) for w in args.weeks.split(',')]
    values = [float(v) for v in args.values.split(',') if v.strip() != '']
    scoring_col = 'fantasy_points_ppr' if args.scoring != 'Standard' else 'fantasy_points'
    variant_features = frozenset(set(DEFAULT_FEATURES) | {FLAG})
    shipped = wp.DEFENSE_BLOWOUT_WEIGHT_DISCOUNT

    print(f"years={years} weeks={weeks[0]}-{weeks[-1]} scoring={args.scoring}")
    print(f"shipped DEFENSE_BLOWOUT_WEIGHT_DISCOUNT={shipped}  sweeping={values}\n")

    # rows[value_label][scope] = list of (metrics_base, metrics_variant) weekly pairs
    rows = {str(v): {} for v in values}

    for year in years:
        stats_df, _tc, name_col, _ = load_and_merge_data(year, args.scoring)
        if 'week' not in stats_df.columns:
            print(f"{year}: no weekly data, skipped")
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                continue
            base_proj, base_meta = build_weekly_projections(
                year, week, args.scoring, as_of_week=week, apply_injury=False,
                features=DEFAULT_FEATURES)
            if base_proj.empty:
                print(f"{year} w{week} base: nothing ({base_meta.get('reason')})")
                continue

            for v in values:
                wp.DEFENSE_BLOWOUT_WEIGHT_DISCOUNT = v
                build_weekly_projections.clear()
                var_proj, var_meta = build_weekly_projections(
                    year, week, args.scoring, as_of_week=week, apply_injury=False,
                    features=variant_features)
                wp.DEFENSE_BLOWOUT_WEIGHT_DISCOUNT = shipped
                build_weekly_projections.clear()
                if var_proj.empty:
                    print(f"{year} w{week} discount={v}: nothing ({var_meta.get('reason')})")
                    continue
                pool = sorted(set(base_proj['Player']) & set(var_proj['Player']))
                if len(pool) < 20:
                    continue
                b = base_proj[base_proj['Player'].isin(pool)]
                vv = var_proj[var_proj['Player'].isin(pool)]
                for scope, pos, startable in SCOPES:
                    mb = _scope_metrics(b, actual, pos, startable)
                    mv = _scope_metrics(vv, actual, pos, startable)
                    if mb and mv:
                        rows[str(v)].setdefault(scope, []).append((mb, mv))
            print(f"{year} w{week} done", flush=True)

    print(f"\n{'=' * 78}\n{FLAG}  DEFENSE_BLOWOUT_WEIGHT_DISCOUNT sweep vs base=DEFAULT_FEATURES\n{'=' * 78}")
    for v in values:
        label = str(v)
        print(f"\n--- discount={label} ---")
        for scope, _pos, _st in SCOPES:
            pairs = rows[label].get(scope)
            if not pairs:
                continue
            mb_list = [p[0] for p in pairs]
            mv_list = [p[1] for p in pairs]
            n = sum(m['n'] for m in mb_list)
            mae_b = _weighted(mb_list, 'mae')
            mae_v = _weighted(mv_list, 'mae')
            rho_b = _weighted(mb_list, 'rank_corr')
            rho_v = _weighted(mv_list, 'rank_corr')
            d_mae = mae_v - mae_b
            d_rho = rho_v - rho_b
            wins = sum(1 for a, b in pairs if b['mae'] < a['mae'])
            losses = sum(1 for a, b in pairs if a['mae'] < b['mae'])
            deltas = [b['mae'] - a['mae'] for a, b in pairs]
            weights = [a['n'] for a in mb_list]
            clo, chi = _bootstrap_ci(deltas, weights)
            p = _sign_test_p(wins, losses)
            sig = ('  [CI excludes 0]' if (np.isfinite(clo) and (clo > 0 or chi < 0))
                  else '  [CI includes 0 -> not distinguishable from noise]' if np.isfinite(clo)
                  else '  [too few weeks for a CI]')
            print(f"  {scope:<10} n={n:<6} MAE {mae_b:.3f}->{mae_v:.3f}  dMAE {d_mae:+.3f}  "
                  f"dRho {d_rho:+.3f}  weeks won {wins}-{losses} (p={p:.2f})  "
                  f"CI[{clo:+.3f},{chi:+.3f}]{sig}")

    print("\n(dMAE negative = this discount value beats the untouched base. "
          "discount=1.0 is the no-op sanity check and should read ~0 everywhere.)")


if __name__ == '__main__':
    main()
