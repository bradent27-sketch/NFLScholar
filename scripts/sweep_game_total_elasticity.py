"""
Sweep the strength of the implied-game-total elasticity, per position.

data/weekly_projections.py applies, when 'v2_game_total_elasticity' is on:

    mult = clip( (team_implied_points / league_avg_implied) ** e , *GAME_TOTAL_CLIP )

with e = GAME_TOTAL_ELASTICITY[pos] = {QB 0.42, RB 0.17, WR 0.14, TE 0.30}
(measured 2019-2023). The team-implied number already comes from the Vegas
total/spread on the schedule feed (game_environment()), which is posted for
future weeks - so this is only asking HOW STRONG the exponent should be, not
where the line comes from.

This scales the whole dict by k and, for each k, reports base (flag OFF, no
elasticity at all) vs variant (flag ON at k*shipped) per position, with the
paired bootstrap CI + sign test from backtest_component. GAME_TOTAL_CLIP is
widened in step with k so a stronger exponent is not silently clamped away.

Usage:
    python scripts/sweep_game_total_elasticity.py --k 0.5,1.0,1.5,2.0,3.0 --years 2025 --weeks 4-17
    python scripts/sweep_game_total_elasticity.py --k 1.0,2.0 --years 2024,2025 --weeks 2-18
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.weekly_projections as wp  # noqa: E402
from scripts.backtest_component import evaluate_components, print_flag_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', required=True, help='comma-separated scale factors on GAME_TOTAL_ELASTICITY')
    ap.add_argument('--years', default='2025')
    ap.add_argument('--weeks', default='4-17')
    ap.add_argument('--scoring', default='Full PPR')
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(',')]
    if '-' in args.weeks:
        lo, hi = args.weeks.split('-')
        weeks = list(range(int(lo), int(hi) + 1))
    else:
        weeks = [int(w) for w in args.weeks.split(',')]
    ks = [float(x) for x in args.k.split(',') if x.strip()]

    shipped_e = dict(wp.GAME_TOTAL_ELASTICITY)
    shipped_clip = wp.GAME_TOTAL_CLIP
    print(f"years={years} weeks={weeks[0]}-{weeks[-1]}  shipped elasticity={shipped_e}  clip={shipped_clip}")
    print("base = 'v2_game_total_elasticity' OFF (no game-total scaling at all).\n")

    variant = frozenset(wp.DEFAULT_FEATURES | {'v2_game_total_elasticity'})
    try:
        for k in ks:
            wp.GAME_TOTAL_ELASTICITY = {p: round(e * k, 4) for p, e in shipped_e.items()}
            # widen the clip band around 1.0 in proportion to k (k=1 -> shipped)
            lo = 1.0 - (1.0 - shipped_clip[0]) * max(k, 1.0)
            hi = 1.0 + (shipped_clip[1] - 1.0) * max(k, 1.0)
            wp.GAME_TOTAL_CLIP = (round(max(lo, 0.5), 3), round(min(hi, 1.6), 3))
            wp.build_weekly_projections.clear()
            label = f'game_total_elasticity_x{k:g}'
            print(f"### {label}  elasticity={wp.GAME_TOTAL_ELASTICITY}  clip={wp.GAME_TOTAL_CLIP}")
            rows, _ = evaluate_components(years, weeks, {label: variant}, args.scoring)
            print_flag_report(label, rows[label], mode='add')
            print()
    finally:
        wp.GAME_TOTAL_ELASTICITY = shipped_e
        wp.GAME_TOTAL_CLIP = shipped_clip
        wp.build_weekly_projections.clear()


if __name__ == '__main__':
    main()
