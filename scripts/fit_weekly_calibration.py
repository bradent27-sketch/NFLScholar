"""
Fit the per-position calibration line for data.weekly_projections.

WHAT THIS MEASURES AND WHY IT'S NEEDED. A projection is supposed to be a
conditional expectation: among every player projected for 20 points, the
average one should score 20. This model's is not, and the direction is the
one selection always produces - the players a noisy projection ranks highest
are disproportionately the ones its noise pushed up, so the top of every
position comes in over. Measured on 2024-2025 before this fix: the top 15%
of each position was projected +2.6 (QB), +2.0 (RB), +2.3 (WR) and +0.5 (TE)
above what they actually scored, and a regression of actual on projected had
a slope well under 1 at every position - the signature of an over-dispersed
projection rather than a biased one.

Shrinking toward the positional mean fixes it. The shrink is a LINE per
position, `actual ~ a + b * projected`, and the whole point of this script
is that the line is fitted on seasons that are NOT the seasons the model is
evaluated on. Fitting it on 2024-2025 and then reporting an improvement on
2024-2025 would be fitting the test set - the exact mistake
scripts/validate_weekly_projections.py's own header warns about.

Fit window: 2021-2023. Evaluation window: 2024-2025. No overlap.

    python scripts/fit_weekly_calibration.py --years 2021,2022,2023

Copy the printed constants into WEEKLY_CALIBRATION in
data/weekly_projections.py. Re-run it if the model's components change
enough to move the fit - the calibration describes THIS model's dispersion,
not a universal constant.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_projections import build_weekly_projections, CALIBRATION_INPUT_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402

# FITTED ON THE WHOLE POOL, and applied ONE-SIDED (see WEEKLY_CALIBRATION in
# data/weekly_projections.py). Fitting on the startable pool alone was tried
# first, on the reasoning that the defect lives entirely at the top, and the
# line it produced is unusable as a general transform: slopes of 0.58-0.65
# with intercepts of 3.4-6.1, which is a sensible description of the top 40
# players and turns a 2-point bench receiver into a 5-point one. The line
# that describes the whole pool is the right line; restricting WHERE it is
# allowed to act is the right restriction.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2021,2022,2023')
    ap.add_argument('--weeks', default='5-17')
    ap.add_argument('--scoring', default='Full PPR')
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(',')]
    lo, hi = args.weeks.split('-')
    weeks = list(range(int(lo), int(hi) + 1))
    scoring_col = 'fantasy_points_ppr' if args.scoring != 'Standard' else 'fantasy_points'

    rows = []
    for year in years:
        stats_df, _t, name_col, _ = load_and_merge_data(year, args.scoring)
        if 'week' not in stats_df.columns:
            print(f"{year}: no weekly data, skipped")
            continue
        for week in weeks:
            actual = (stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week]
                      .groupby(name_col, observed=True)[scoring_col].sum())
            if actual.empty:
                continue
            # Fitted against the model WITHOUT calibration, obviously - the
            # line being fitted is the one that will be applied to it.
            proj, _meta = build_weekly_projections(
                year, week, args.scoring, as_of_week=week, apply_injury=False,
                features=CALIBRATION_INPUT_FEATURES)
            if proj.empty:
                continue
            d = proj[['Player', 'Pos', 'Model Proj Pts']].copy()
            d['actual'] = d['Player'].map(actual)
            rows.append(d.dropna(subset=['actual']))
    if not rows:
        raise SystemExit("no data")
    df = pd.concat(rows, ignore_index=True)

    print(f"fit window: years={years} weeks={weeks[0]}-{weeks[-1]}  n={len(df)}")
    print("\nWEEKLY_CALIBRATION = {")
    for pos in ('QB', 'RB', 'WR', 'TE'):
        sub = df[df['Pos'] == pos]
        if len(sub) < 200:
            continue
        slope, intercept = np.polyfit(sub['Model Proj Pts'], sub['actual'], 1)
        pred = intercept + slope * sub['Model Proj Pts']
        print(f"    '{pos}': ({slope:.3f}, {intercept:.3f}),"
              f"   # n={len(sub)}  MAE {abs(sub['Model Proj Pts'] - sub['actual']).mean():.3f}"
              f" -> {abs(pred - sub['actual']).mean():.3f}"
              f"  bias {(sub['Model Proj Pts'] - sub['actual']).mean():+.3f} -> {(pred - sub['actual']).mean():+.3f}")
    print("}")


if __name__ == '__main__':
    main()
