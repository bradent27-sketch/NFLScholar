"""
Sweep DEFENSE_PRIOR_GAMES (currently 4.0, driving every position's defense
matchup shrinkage since it was set - never itself backtested against
alternate values). Base = DEFAULT_FEATURES exactly as shipped; variant =
DEFAULT_FEATURES + 'v2_defense_prior_games_override', which only takes
effect because DEFENSE_PRIOR_GAMES_OVERRIDE is set to a real value here -
see that constant's own note in data/weekly_projections.py.

Same cache-clearing requirement as the other sweep scripts tonight.

Usage:
    python scripts/sweep_defense_prior_games.py --values 2,3,4,6,8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.weekly_projections as wp  # noqa: E402
from scripts.backtest_component import evaluate_components, print_flag_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--values', required=True, help="comma-separated DEFENSE_PRIOR_GAMES values to test")
    ap.add_argument('--years', default='2024,2025')
    ap.add_argument('--weeks', default='2-18')
    ap.add_argument('--scoring', default='Full PPR')
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(',')]
    if '-' in args.weeks:
        lo, hi = args.weeks.split('-')
        weeks = list(range(int(lo), int(hi) + 1))
    else:
        weeks = [int(w) for w in args.weeks.split(',')]
    values = [float(v) for v in args.values.split(',') if v.strip()]

    print(f"years={years} weeks={weeks[0]}-{weeks[-1]} values={values} (shipped default: {wp.DEFENSE_PRIOR_GAMES})")
    variant = frozenset(wp.DEFAULT_FEATURES | {'v2_defense_prior_games_override'})

    for v in values:
        wp.DEFENSE_PRIOR_GAMES_OVERRIDE = v
        wp.build_weekly_projections.clear()
        label = f'defense_prior_games_{v:.1f}'
        rows, _outliers = evaluate_components(years, weeks, {label: variant}, args.scoring)
        print_flag_report(label, rows[label], mode='add')

    wp.DEFENSE_PRIOR_GAMES_OVERRIDE = None
    wp.build_weekly_projections.clear()


if __name__ == '__main__':
    main()
