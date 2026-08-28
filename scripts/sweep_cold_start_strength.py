"""
Sweep COLD_START_MULTIPLIER_REGRESSION's strength (how far each cold-start
multiplier pulls back toward neutral 1.0) across Week 1 of every available
season, to check whether the 0.25 default's real START-QB win (dMAE -0.227,
CI excludes 0, 2021-2025 Week 1 - see docs/overnight_backtest_log_*.md) is a
genuine dose-response or a thin-sample (n=5 weeks) fluke. If stronger
regression keeps helping and weaker regression helps less, that is a real
signal; if it is non-monotonic, treat the whole result skeptically regardless
of any one value's own p-value.

Same cache-clearing requirement as scripts/sweep_scheme_blend_weight.py - the
`features` cache key does not change across strength values, so clearing
data.weekly_projections.build_weekly_projections between them is REQUIRED,
not optional.

Usage:
    python scripts/sweep_cold_start_strength.py --strengths 0.10,0.25,0.40,0.60
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.weekly_projections as wp  # noqa: E402
from scripts.backtest_component import evaluate_components, print_flag_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strengths', required=True, help="comma-separated regression strengths in [0,1]")
    ap.add_argument('--years', default='2021,2022,2023,2024,2025')
    ap.add_argument('--weeks', default='1-1')
    ap.add_argument('--scoring', default='Full PPR')
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(',')]
    if '-' in args.weeks:
        lo, hi = args.weeks.split('-')
        weeks = list(range(int(lo), int(hi) + 1))
    else:
        weeks = [int(w) for w in args.weeks.split(',')]
    strengths = [float(s) for s in args.strengths.split(',') if s.strip()]

    print(f"years={years} weeks={weeks[0]}-{weeks[-1]} strengths={strengths}")
    variant = frozenset(wp.DEFAULT_FEATURES | {'v2_cold_start_regression'})
    original = wp.COLD_START_MULTIPLIER_REGRESSION

    for s in strengths:
        wp.COLD_START_MULTIPLIER_REGRESSION = s
        wp.build_weekly_projections.clear()
        label = f'cold_start_strength_{s:.2f}'
        rows, _outliers = evaluate_components(years, weeks, {label: variant}, args.scoring)
        print_flag_report(label, rows[label], mode='add')

    wp.COLD_START_MULTIPLIER_REGRESSION = original
    wp.build_weekly_projections.clear()


if __name__ == '__main__':
    main()
