"""
Sweep WHICH WR stat(s) 'v2_defense_blowout_discount' actually applies to, to
find out whether the confirmed WR/START-WR regression (docs/weekly_
projections_methodology.md's 2026-09-16 entries; full 26-week confirm,
+0.001/+0.027 MAE, CI excludes 0) is driven by one specific stat rather than
the whole channel uniformly, per the user's own hypothesis ("maybe receptions
and receiving yards are different").

WR's channel is ['targets', 'receptions', 'receiving_yards', 'receiving_tds']
(data.transforms.OFFENSE_PROJECTION_STATS['WR']). Tests the full set (today's
shipped-everywhere-but-WR shape, included here as a reference point since it
IS the confirmed-loss result) against each individual stat alone and a couple
of natural pairings, using the blowout_stats plumbing added to
build_team_game_quality_adjusted_matchup 2026-09-16.

WR is excluded from DEFAULT_FEATURES' shipped 'v2_defense_blowout_discount'
(see DEFENSE_BLOWOUT_DISCOUNT_POSITIONS), so this monkeypatches BOTH that
position gate (to {'WR'} for the duration of the sweep) and
DEFENSE_BLOWOUT_DISCOUNT_STATS (to the swept subset) around each variant
build - same base-once-per-week / cache-clearing structure as
scripts/sweep_defense_blowout_discount.py and
scripts/sweep_offense_prior_games.py.

This is the standard game-script question (not an early-season one), so it
uses the standard wk5-17 confirm window, not sweep_offense_prior_games.py's
early-season one.

Usage:
    python scripts/sweep_defense_blowout_wr_stats.py \
        --stat-sets "all,targets,receptions,receiving_yards,receiving_tds,receptions+receiving_yards" \
        --years 2024,2025 --weeks 5-17
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data.weekly_projections as wp  # noqa: E402
from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data, OFFENSE_PROJECTION_STATS  # noqa: E402
from scripts.eval_weekly_model import _actual_points, _weighted  # noqa: E402
from scripts.backtest_component import SCOPES, _scope_metrics, _bootstrap_ci, _sign_test_p  # noqa: E402

FLAG = 'v2_defense_blowout_discount'
WR_STATS = set(OFFENSE_PROJECTION_STATS['WR'])  # targets, receptions, receiving_yards, receiving_tds


def _parse_stat_sets(spec):
    """'all,targets,receptions+receiving_yards' -> [('all', None), ('targets', {'targets'}), ...].
    'all' (or 'none', included as a second sanity check) are special-cased;
    everything else is a '+'-joined subset of WR_STATS."""
    sets = []
    for label in spec.split(','):
        label = label.strip()
        if not label:
            continue
        if label == 'all':
            sets.append((label, None))  # None -> blowout_stats=None, i.e. every stat (today's shipped shape)
            continue
        if label == 'none':
            sets.append((label, frozenset()))  # empty set -> discount fires for no stat, a no-op sanity check
            continue
        stats = frozenset(s.strip() for s in label.split('+'))
        unknown = stats - WR_STATS
        if unknown:
            raise SystemExit(f"unknown WR stat(s) in --stat-sets: {unknown} (know: {sorted(WR_STATS)})")
        sets.append((label, stats))
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stat-sets',
                    default='all,none,targets,receptions,receiving_yards,receiving_tds,'
                            'receptions+receiving_yards')
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
    stat_sets = _parse_stat_sets(args.stat_sets)
    scoring_col = 'fantasy_points_ppr' if args.scoring != 'Standard' else 'fantasy_points'
    variant_features = frozenset(set(DEFAULT_FEATURES) | {FLAG})

    print(f"years={years} weeks={weeks[0]}-{weeks[-1]} scoring={args.scoring}")
    print(f"WR stat-sets swept: {[label for label, _ in stat_sets]}\n")

    rows = {label: {} for label, _ in stat_sets}
    original_positions = wp.DEFENSE_BLOWOUT_DISCOUNT_POSITIONS
    original_stats_map = wp.DEFENSE_BLOWOUT_DISCOUNT_STATS

    try:
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

                for label, stat_subset in stat_sets:
                    # ADD 'WR' to whatever's already shipped (RB/QB/TE) rather
                    # than replace it - v2_defense_blowout_discount is already
                    # in DEFAULT_FEATURES (2026-09-16), so `base_proj` above
                    # already has RB/QB/TE's discount active. Replacing the
                    # set instead of extending it silently DISABLED RB/QB/TE's
                    # already-shipped discount for every variant build - a
                    # confound that leaked into the WR/START-WR comparison via
                    # pass_capacity_allocator's cross-position volume
                    # reconciliation (RB's own discount changes RB's matchup
                    # rating, which changes RB's projected volume, which
                    # changes how much passing volume gets reconciled onto
                    # WR/TE) even though this sweep never touches WR's
                    # discount status directly in the 'none' arm.
                    wp.DEFENSE_BLOWOUT_DISCOUNT_POSITIONS = original_positions | {'WR'}
                    wp.DEFENSE_BLOWOUT_DISCOUNT_STATS = {'WR': stat_subset} if stat_subset is not None else {}
                    build_weekly_projections.clear()
                    var_proj, var_meta = build_weekly_projections(
                        year, week, args.scoring, as_of_week=week, apply_injury=False,
                        features=variant_features)
                    wp.DEFENSE_BLOWOUT_DISCOUNT_POSITIONS = original_positions
                    wp.DEFENSE_BLOWOUT_DISCOUNT_STATS = original_stats_map
                    build_weekly_projections.clear()
                    if var_proj.empty:
                        print(f"{year} w{week} stats={label}: nothing ({var_meta.get('reason')})")
                        continue
                    pool = sorted(set(base_proj['Player']) & set(var_proj['Player']))
                    if len(pool) < 20:
                        continue
                    b = base_proj[base_proj['Player'].isin(pool)]
                    vv = var_proj[var_proj['Player'].isin(pool)]
                    for scope, pos, startable in SCOPES:
                        if pos not in (None, 'WR'):
                            continue  # only WR/ALL/START-WR are meaningful - the change is WR-only
                        mb = _scope_metrics(b, actual, pos, startable)
                        mv = _scope_metrics(vv, actual, pos, startable)
                        if mb and mv:
                            rows[label].setdefault(scope, []).append((mb, mv))
                print(f"{year} w{week} done", flush=True)
    finally:
        wp.DEFENSE_BLOWOUT_DISCOUNT_POSITIONS = original_positions
        wp.DEFENSE_BLOWOUT_DISCOUNT_STATS = original_stats_map

    print(f"\n{'=' * 78}\n{FLAG}  WR-only, per-stat sweep vs base=DEFAULT_FEATURES\n{'=' * 78}")
    for label, _stat_subset in stat_sets:
        print(f"\n--- stats={label} ---")
        for scope, _pos, _st in SCOPES:
            if _pos not in (None, 'WR'):
                continue
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

    print("\n(dMAE negative = this stat-subset beats the untouched base. "
          "'none' is the no-op sanity check and should read ~0 everywhere; "
          "'all' reproduces the confirmed WR/START-WR loss as a reference point.)")


if __name__ == '__main__':
    main()
