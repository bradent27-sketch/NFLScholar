"""
Two-year, every-week, every-modelled-player residual dump for
data.weekly_projections.build_weekly_projections, run with the CURRENT
DEFAULT_FEATURES (calibration off, per the 2026-08-23 removal - see that
block's comment in data/weekly_projections.py).

WHY THIS EXISTS ALONGSIDE eval_weekly_model.py. That script pools weeks into
one MAE/bias number per variant/scope - the right tool for "does component X
help", the wrong tool for "where specifically is the model wrong". This
script keeps every (year, week, player) row so the bias can be sliced by
projection decile, by cold-start vs in-season, or any other cut, the way
fit_weekly_calibration.py's old fit did but over the model's OWN measured
years instead of a held-out prior window - the point of this run is to look
at 2024-2025 directly, not to protect a downstream fit on a different window.

Injuries are OFF (apply_injury=False) for the same reason eval_weekly_model.py
turns them off: the injury feed carries no historical week granularity, see
build_weekly_projections' own docstring.

Usage:

    python scripts/backtest_weekly_residuals.py
    python scripts/backtest_weekly_residuals.py --years 2024,2025 --weeks 1-18 \
        --out scratch/weekly_residuals_2024_2025.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_projections import build_weekly_projections  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402


def _actual_points(stats_df, name_col, week, scoring_col):
    rows = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week]
    if scoring_col not in rows.columns:
        return pd.Series(dtype=float)
    return rows.groupby(name_col, observed=True)[scoring_col].sum()


def collect(years, weeks, scoring='Full PPR'):
    scoring_col = 'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'
    frames = []
    for year in years:
        stats_df, _team_col, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns:
            print(f"{year}: no weekly data, skipped")
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                print(f"{year} w{week}: no actuals, skipped")
                continue
            proj, meta = build_weekly_projections(
                year, week, scoring, as_of_week=week, apply_injury=False, features=None)
            if proj.empty:
                print(f"{year} w{week}: nothing ({meta.get('reason')})")
                continue
            d = proj[['Player', 'Pos', 'Team', 'Opponent', 'Model Proj Pts']].copy()
            d['year'] = year
            d['week'] = week
            d['cold_start'] = bool(meta.get('cold_start'))
            d['actual'] = d['Player'].map(actual)
            d = d.dropna(subset=['actual'])
            frames.append(d)
            print(f"{year} w{week}: {len(d)} paired rows"
                  f"{' [cold start]' if d['cold_start'].iloc[0] else ''}")
    if not frames:
        raise SystemExit("no data collected")
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={'Model Proj Pts': 'projected'})
    df['error'] = df['projected'] - df['actual']
    df['abs_error'] = df['error'].abs()
    return df


def _decile_report(df):
    print("\n=== Bias by projection decile, per position ===")
    for pos in ('QB', 'RB', 'WR', 'TE'):
        sub = df[df['Pos'] == pos].copy()
        if len(sub) < 200:
            continue
        sub['decile'] = pd.qcut(sub['projected'], 10, labels=False, duplicates='drop')
        slope, intercept = np.polyfit(sub['projected'], sub['actual'], 1)
        print(f"\n-- {pos}  (n={len(sub)}, slope={slope:.3f}, intercept={intercept:+.3f}, "
              f"overall bias={sub['error'].mean():+.3f}, MAE={sub['abs_error'].mean():.3f}) --")
        header = f"{'decile':>7}{'n':>7}{'proj lo':>9}{'proj hi':>9}{'mean proj':>11}{'mean actual':>13}{'bias':>9}{'MAE':>8}"
        print(header)
        for dec, g in sub.groupby('decile'):
            print(f"{dec:>7}{len(g):>7}{g['projected'].min():>9.2f}{g['projected'].max():>9.2f}"
                  f"{g['projected'].mean():>11.2f}{g['actual'].mean():>13.2f}"
                  f"{g['error'].mean():>+9.3f}{g['abs_error'].mean():>8.3f}")


def _cold_start_report(df):
    print("\n=== Cold start (week 1, prior-season fallback) vs in-season ===")
    for pos in ('QB', 'RB', 'WR', 'TE'):
        sub = df[df['Pos'] == pos]
        cold = sub[sub['cold_start']]
        warm = sub[~sub['cold_start']]
        if cold.empty:
            continue
        print(f"{pos:<4} cold n={len(cold):>5} bias={cold['error'].mean():>+7.3f} MAE={cold['abs_error'].mean():>6.3f}"
              f"   |   warm n={len(warm):>5} bias={warm['error'].mean():>+7.3f} MAE={warm['abs_error'].mean():>6.3f}")


def _week_trend_report(df):
    print("\n=== Bias by week-of-season, pooled across years, per position ===")
    for pos in ('QB', 'RB', 'WR', 'TE'):
        sub = df[df['Pos'] == pos]
        if sub.empty:
            continue
        g = sub.groupby('week').agg(n=('error', 'size'), bias=('error', 'mean'), mae=('abs_error', 'mean'))
        print(f"\n-- {pos} --")
        print(f"{'week':>5}{'n':>6}{'bias':>9}{'MAE':>8}")
        for wk, row in g.iterrows():
            print(f"{wk:>5}{int(row['n']):>6}{row['bias']:>+9.3f}{row['mae']:>8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2024,2025')
    ap.add_argument('--weeks', default='1-18')
    ap.add_argument('--scoring', default='Full PPR')
    ap.add_argument('--out', default='weekly_residuals_2024_2025.csv')
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(',')]
    lo, hi = args.weeks.split('-')
    weeks = list(range(int(lo), int(hi) + 1))

    print(f"years={years} weeks={weeks[0]}-{weeks[-1]} scoring={args.scoring} features=DEFAULT_FEATURES (no calibration)")

    df = collect(years, weeks, args.scoring)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df)} rows to {args.out}")

    _decile_report(df)
    _cold_start_report(df)
    _week_trend_report(df)


if __name__ == '__main__':
    main()
