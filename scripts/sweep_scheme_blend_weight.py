"""
Sweep a FIXED alignment/scheme blend ratio per position, instead of the
default evidence-weighted blend - built 2026-08-27 per explicit request: TE's
evidence-weighted blend (scripts/backtest_stat_level.py --add
v2_scheme_alignment_blend) was real but weaker than scheme-alone's own
START-TE win, i.e. alignment was "diluting, not hurting" - the user wants TE
to keep alignment context regardless, just weighted less than the ~50/50 the
evidence-weighted default happened to land near. This finds where that
tradeoff actually lands rather than guessing at one ratio (e.g. "75/25").

Mechanics: sets data.weekly_projections.SCHEME_ALIGNMENT_BLEND_FIXED_WEIGHT[
position] to each weight in turn, clears build_weekly_projections' cache
(REQUIRED - the cache key is `features`, which does not change across weight
values, so skipping this would silently serve a stale result), and runs the
same per-stat paired A/B as backtest_stat_level.py for each weight against
the untouched base (DEFAULT_FEATURES, no blend at all).

Usage:
    python scripts/sweep_scheme_blend_weight.py --position TE --weights 0.5,0.65,0.75,0.85,0.9,1.0
    python scripts/sweep_scheme_blend_weight.py --position WR --weights 0.3,0.5,0.7
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.weekly_projections as wp  # noqa: E402
from scripts.backtest_stat_level import evaluate_stat_level, print_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--position', required=True, choices=['WR', 'TE'])
    ap.add_argument('--weights', required=True,
                    help="comma-separated scheme-side weights in [0,1], e.g. 0.5,0.75,1.0")
    ap.add_argument('--stats', default='targets,receptions,receiving_yards')
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
    raw_stats = [s.strip() for s in args.stats.split(',') if s.strip()]
    weights = [float(w) for w in args.weights.split(',') if w.strip()]

    print(f"years={years} weeks={weeks[0]}-{weeks[-1]} position={args.position} weights={weights}")
    variant = frozenset(wp.DEFAULT_FEATURES | {'v2_scheme_alignment_blend'})

    for w in weights:
        wp.SCHEME_ALIGNMENT_BLEND_FIXED_WEIGHT.clear()
        wp.SCHEME_ALIGNMENT_BLEND_FIXED_WEIGHT[args.position] = w
        wp.build_weekly_projections.clear()
        label = f'{args.position}_scheme_w{w:.2f}'
        rows = evaluate_stat_level(years, weeks, {label: variant}, raw_stats, args.scoring)
        # Both positions print - the one NOT under test still uses the
        # default evidence-weighted blend (SCHEME_ALIGNMENT_BLEND_FIXED_
        # WEIGHT.get() returns None for it), so its numbers are redundant
        # with the earlier evidence-weighted backtest, not wrong - ignore
        # them when reading this, they'll be identical across every weight.
        print_report(label, rows[label], mode='add')

    # Leave no global state behind for anything run after this script in the
    # same process/session.
    wp.SCHEME_ALIGNMENT_BLEND_FIXED_WEIGHT.clear()
    wp.build_weekly_projections.clear()


if __name__ == '__main__':
    main()
