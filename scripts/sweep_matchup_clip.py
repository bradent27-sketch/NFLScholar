"""
FIRST-PASS scan of MATCHUP_CLIP - the [low, high] bound on the forward DEFENSE
matchup multiplier (data/weekly_projections.py, currently (0.75, 1.30), clamped
in _overall_matchup_multiplier / _role_adjusted_multiplier /
_continuous_role_adjusted_multiplier). Never swept; the asymmetric -25%/+30%
range was hand-set, and a user note ("a defense listed 13th-toughest vs RB yet
every one of its multipliers pinned at the 0.75 floor") suggests the LOW bound
may be too aggressive.

METHOD (deliberately no model code change for a first look): MATCHUP_CLIP is a
plain module tuple read at call time. For each candidate tuple this rebinds
wp.MATCHUP_CLIP, clears the build cache, and runs the FULL model over the
window, reporting absolute whole-pool + per-position MAE / rank-corr. Because
there is no paired base-vs-variant flag, there is no bootstrap CI here - read
the ABSOLUTE MAE trend across settings, and if one clearly wins, wire it in
with a proper flag and run scripts/backtest_component.py for the CI.

Usage:
    python scripts/sweep_matchup_clip.py --clips "0.75,1.30;0.80,1.25;0.82,1.22;0.85,1.20;0.85,1.15;0.70,1.35" --years 2025 --weeks 4-17
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data.weekly_projections as wp  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from scripts.eval_weekly_model import _metrics, _actual_points, STARTABLE_N  # noqa: E402

SCOPES = [('ALL', None, False), ('QB', 'QB', False), ('RB', 'RB', False),
          ('WR', 'WR', False), ('TE', 'TE', False),
          ('START-RB', 'RB', True), ('START-WR', 'WR', True), ('START-TE', 'TE', True)]


def _scope(df, actual, pos, startable):
    d = df if pos is None else df[df['Pos'] == pos]
    if startable:
        d = d.nlargest(STARTABLE_N.get(pos, 30), 'Model Proj Pts')
    pred = pd.Series(d['Model Proj Pts'].to_numpy(), index=d['Player'])
    return _metrics(pred, actual)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', required=True, help='semicolon-separated low,high pairs')
    ap.add_argument('--years', default='2025')
    ap.add_argument('--weeks', default='4-17')
    ap.add_argument('--scoring', default='Full PPR')
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(',')]
    lo, hi = args.weeks.split('-')
    weeks = list(range(int(lo), int(hi) + 1))
    clips = [tuple(float(x) for x in p.split(',')) for p in args.clips.split(';')]
    shipped = wp.MATCHUP_CLIP
    print(f"years={years} weeks={weeks[0]}-{weeks[-1]}  shipped MATCHUP_CLIP={shipped}\n")

    per_year_stats = {}
    for y in years:
        sdf, _tc, ncol, _ = load_and_merge_data(y, args.scoring)
        per_year_stats[y] = (sdf, ncol)

    try:
        for clip in clips:
            wp.MATCHUP_CLIP = clip
            wp.build_weekly_projections.clear()
            agg = {s[0]: {'mae_w': [], 'n_w': [], 'rho_w': []} for s in SCOPES}
            for y in years:
                sdf, ncol = per_year_stats[y]
                if 'week' not in sdf.columns:
                    continue
                scol = 'fantasy_points_ppr' if args.scoring != 'Standard' else 'fantasy_points'
                for wk in weeks:
                    actual = _actual_points(sdf, ncol, wk, scol)
                    if actual.empty:
                        continue
                    proj, meta = wp.build_weekly_projections(
                        y, wk, args.scoring, as_of_week=wk, apply_injury=False)
                    if proj.empty:
                        continue
                    for name, pos, st in SCOPES:
                        m = _scope(proj, actual, pos, st)
                        if m:
                            agg[name]['mae_w'].append(m['mae'] * m['n'])
                            agg[name]['n_w'].append(m['n'])
                            agg[name]['rho_w'].append(m.get('rank_corr', np.nan) * m['n'])
            tag = 'SHIPPED' if clip == shipped else ''
            print(f"--- MATCHUP_CLIP = {clip}  {tag}")
            for name, _p, _s in SCOPES:
                n = sum(agg[name]['n_w'])
                if n <= 0:
                    continue
                mae = sum(agg[name]['mae_w']) / n
                rho = np.nansum(agg[name]['rho_w']) / n
                print(f"    {name:9s} n={n:<6d} MAE {mae:.4f}   rankcorr {rho:.4f}")
            print()
    finally:
        wp.MATCHUP_CLIP = shipped
        wp.build_weekly_projections.clear()


if __name__ == '__main__':
    main()
