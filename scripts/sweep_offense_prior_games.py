"""
Sweep OFFENSE_PRIOR_GAMES for 'v2_offense_prior_blend' - a component NOT in
DEFAULT_FEATURES (see docs/weekly_projections_methodology.md's 2026-09-16
entry). Built to test the user's early-season concern: a team with only 1-2
games on the books has a near-meaningless "own average" baseline, and
_team_game_quality_profile had no credibility guard on the OFFENSE side of
that ratio (DEFENSE_PRIOR_GAMES only ever protects a thin DEFENSE sample).
OFFENSE_PRIOR_GAMES blends a thin offense's own baseline toward the league
average, n/(n+K) - this sweeps K.

Unlike the blowout-discount sweep, this component's hypothesized benefit is
concentrated in EARLY weeks specifically (a team with 4+ games already has a
fairly credible own-average even at moderate K) - week 1 is excluded by
default since it's a true cold start that falls back entirely to prior-season
data (a different code path this flag never touches; see build_weekly_
projections' COLD START section), so weeks 2-4 is the real target window.
A second, optional pass against the standard wk5-17 window checks the blend
decays away harmlessly rather than leaving a same-strength drag once teams
have real evidence.

Same base-once-per-week / cache-clearing / bootstrap-CI structure as
scripts/sweep_defense_blowout_discount.py (imports directly from
backtest_component.py and eval_weekly_model.py rather than reimplementing).

Usage:
    python scripts/sweep_offense_prior_games.py --values 0,3,6,9,12,18 \
        --years 2022,2023,2024,2025 --weeks 2-4
    python scripts/sweep_offense_prior_games.py --values 6 \
        --years 2024,2025 --weeks 5-17   # does the shrink decay away cleanly?
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

FLAG = 'v2_offense_prior_blend'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--values', default='0,3,6,9,12,18',
                    help="comma-separated OFFENSE_PRIOR_GAMES values to test "
                         "(0 is a sanity check - n/(n+0)=1 always, no blend "
                         "at all, should read exactly 0 delta)")
    ap.add_argument('--years', default='2022,2023,2024,2025')
    ap.add_argument('--weeks', default='2-4')
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
    shipped = wp.OFFENSE_PRIOR_GAMES

    print(f"years={years} weeks={weeks[0]}-{weeks[-1]} scoring={args.scoring}")
    print(f"shipped OFFENSE_PRIOR_GAMES={shipped}  sweeping={values}\n")

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
                wp.OFFENSE_PRIOR_GAMES = v
                build_weekly_projections.clear()
                var_proj, var_meta = build_weekly_projections(
                    year, week, args.scoring, as_of_week=week, apply_injury=False,
                    features=variant_features)
                wp.OFFENSE_PRIOR_GAMES = shipped
                build_weekly_projections.clear()
                if var_proj.empty:
                    print(f"{year} w{week} K={v}: nothing ({var_meta.get('reason')})")
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

    print(f"\n{'=' * 78}\n{FLAG}  OFFENSE_PRIOR_GAMES sweep vs base=DEFAULT_FEATURES\n{'=' * 78}")
    for v in values:
        label = str(v)
        print(f"\n--- K={label} ---")
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

    print("\n(dMAE negative = this K beats the untouched base. K=0 is the no-op "
          "sanity check and should read ~0 everywhere.)")


if __name__ == '__main__':
    main()
