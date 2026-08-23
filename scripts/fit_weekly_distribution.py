"""
Fit empirical boom/bust percentile bands for data.weekly_distribution.

WHAT THIS MEASURES. A single point projection is a median-ish guess, not the
whole story - the same 12.0-point projection means something different for a
high-floor possession receiver than for a boom/bust deep threat. This script
measures, for each position x usage tier (the same Elite/Starter/Flex/
Streamer buckets scripts/backtest_five_weeks.py reports on), the empirical
distribution of actual/calibrated ratio across a real backtest: at P10, what
fraction of the projection did the actual outcome come in at; at P90; etc.
That ratio ladder is the "shape" - multiply it onto any player's own
calibrated points to get THEIR band, no separate fit per player needed.

RATIO, NOT RAW DELTA, so the band scales with the size of the projection - a
bust week for a 20-point projection and a 5-point projection are different in
absolute points but can share a shape in ratio terms.

OUT-OF-SAMPLE, SAME DISCIPLINE AS CALIBRATION. Fit window 2021-2023, matching
scripts/fit_weekly_calibration.py exactly - both describe this model's own
dispersion and neither should be fit on the seasons it's evaluated against.
Uses the CALIBRATED points (WEEKLY_CALIBRATION already applied) as the
denominator, since that's what a user actually sees as "the projection" -
the band is describing dispersion AROUND the number on screen, not around an
intermediate raw value nobody sees.

    python scripts/fit_weekly_distribution.py --years 2021,2022,2023

Copy the printed constants into WEEKLY_DISTRIBUTION_BANDS in
data/weekly_distribution.py. Re-run it whenever WEEKLY_CALIBRATION is
re-fit (see that script's own header) - a stale calibration line changes
what "calibrated points" means, which changes the ratio this script measures.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_projections import build_weekly_projections  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402

TIERS = [('Elite', 1, 6), ('Starter', 7, 15), ('Flex', 16, 30), ('Streamer/Bench', 31, 10_000)]
PERCENTILES = (10, 25, 50, 75, 90)
MIN_DENOMINATOR = 1.0  # a near-zero projection makes the ratio meaningless, not informative


def _actual_points(stats_df, name_col, week, scoring_col='fantasy_points_ppr'):
    rows = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week]
    if scoring_col not in rows.columns:
        return pd.Series(dtype=float)
    return rows.groupby(name_col, observed=True)[scoring_col].sum()


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

    ratios = {pos: {tier: [] for tier, _lo, _hi in TIERS} for pos in ('QB', 'RB', 'WR', 'TE')}

    for year in years:
        stats_df, _t, name_col, _ = load_and_merge_data(year, args.scoring)
        if 'week' not in stats_df.columns:
            print(f"{year}: no weekly data, skipped")
            continue
        for week in weeks:
            proj, meta = build_weekly_projections(year, week, args.scoring, as_of_week=week,
                                                   apply_injury=False, model_version='v2')
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if proj.empty or actual.empty:
                continue
            for pos in ('QB', 'RB', 'WR', 'TE'):
                sub = proj[proj['Pos'] == pos].sort_values('Model Proj Pts', ascending=False).reset_index(drop=True)
                for tier_name, tlo, thi in TIERS:
                    tsub = sub.iloc[tlo - 1:thi]
                    if tsub.empty:
                        continue
                    joined = tsub.set_index('Player')['Model Proj Pts'].to_frame('proj')
                    joined['actual'] = joined.index.map(actual)
                    joined = joined.dropna()
                    joined = joined[joined['proj'] >= MIN_DENOMINATOR]
                    if joined.empty:
                        continue
                    ratios[pos][tier_name].extend((joined['actual'] / joined['proj']).tolist())

    print(f"fit window: years={years} weeks={weeks[0]}-{weeks[-1]}\n")
    print("WEEKLY_DISTRIBUTION_BANDS = {")
    for pos in ('QB', 'RB', 'WR', 'TE'):
        print(f"    '{pos}': {{")
        for tier_name, _lo, _hi in TIERS:
            sample = np.array(ratios[pos][tier_name], dtype=float)
            if len(sample) < 30:
                print(f"        # '{tier_name}': skipped, only n={len(sample)} (<30)")
                continue
            pcts = {p: float(np.percentile(sample, p)) for p in PERCENTILES}
            band_str = ", ".join(f"{p}: {v:.3f}" for p, v in pcts.items())
            print(f"        '{tier_name}': {{{band_str}}},  # n={len(sample)}")
        print("    },")
    print("}")


if __name__ == '__main__':
    main()
