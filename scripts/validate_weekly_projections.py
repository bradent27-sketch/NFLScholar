"""
One-pass validation for data.weekly_projections.build_weekly_projections.

WHAT THIS IS AND ISN'T. This is a single honest backtest, not a tuned or
repeated one - the model's constants (STAT_K, the clip ranges) were set from
reasoning documented in weekly_projections.py's own module docstring, then
checked here once, not iterated against this script's own output. Iterating
the model against this exact backtest would be fitting to the test set.

WHAT IT MEASURES. For each week in a season (using ONLY data from strictly
earlier weeks - the model's own as_of_week parameter, so this is a true
no-leakage backtest, not a same-week correlation), compares:
  - this app's own model (data.weekly_projections.build_weekly_projections)
  - a naive baseline: each player's own trailing recent-form average
    (data.transforms.build_recent_form_rank's own n_weeks window) - the
    simplest defensible thing to compare a real model against
against what actually happened that week, and reports MAE / bias / rank
correlation, overall and by position.

Run it on a normal network:

    python scripts/validate_weekly_projections.py --year 2025 --weeks 5-17
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_projections import build_weekly_projections  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402


def _actual_points(stats_df, name_col, week, scoring_col='fantasy_points_ppr'):
    rows = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week]
    if scoring_col not in rows.columns:
        return pd.Series(dtype=float)
    return rows.groupby(name_col)[scoring_col].sum()


def _naive_baseline(stats_df, name_col, as_of_week, scoring_col='fantasy_points_ppr', n_weeks=4):
    hist = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') < as_of_week]
    if hist.empty or scoring_col not in hist.columns:
        return pd.Series(dtype=float)
    recent = hist.sort_values('week').groupby(name_col).tail(n_weeks)
    return recent.groupby(name_col)[scoring_col].mean()


def _metrics(pred, actual, label):
    joined = pd.concat([pred.rename('pred'), actual.rename('actual')], axis=1).dropna()
    if len(joined) < 5:
        return None
    mae = float((joined['pred'] - joined['actual']).abs().mean())
    bias = float((joined['pred'] - joined['actual']).mean())
    # Spearman rank correlation via Pearson-on-ranks - this project has no
    # scipy dependency (see data.draft_sources.normal_sf's own docstring on
    # the same point), and Pearson correlation of two rank series is exactly
    # what Spearman's rho is.
    corr = float(joined['pred'].rank().corr(joined['actual'].rank()))
    return {'label': label, 'n': len(joined), 'mae': round(mae, 3),
            'bias': round(bias, 3), 'rank_corr': round(corr, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=2025)
    ap.add_argument('--weeks', type=str, default='5-17', help='e.g. 5-17 or 5,6,7')
    ap.add_argument('--scoring', type=str, default='Full PPR')
    args = ap.parse_args()

    if '-' in args.weeks:
        lo, hi = args.weeks.split('-')
        weeks = list(range(int(lo), int(hi) + 1))
    else:
        weeks = [int(w) for w in args.weeks.split(',')]

    scoring_col = 'fantasy_points_ppr' if args.scoring != 'Standard' else 'fantasy_points'
    stats_df, team_col, name_col, _ = load_and_merge_data(args.year, args.scoring)

    model_rows, naive_rows = [], []
    for week in weeks:
        # apply_injury=False: see build_weekly_projections' own docstring -
        # the injury feed has no historical week granularity, so validating
        # a past week with it on would score every player against whatever
        # designation they carried MONTHS after that week, not during it.
        proj, meta = build_weekly_projections(args.year, week, args.scoring, as_of_week=week, apply_injury=False)
        actual = _actual_points(stats_df, name_col, week, scoring_col)
        naive_full = _naive_baseline(stats_df, name_col, week, scoring_col)
        if proj.empty:
            print(f"week {week}: model produced nothing ({meta.get('reason')})")
            continue
        # SAME PLAYER POOL for both sides - the model only outputs QB/RB/WR/
        # TE rows meeting its own minimum-data bar (~60-80/week), while a
        # naive average over the WHOLE stats file includes DST/K/backups/
        # one-game call-ups (~900+/week) that are trivially easy to predict
        # near zero. Comparing against that unrestricted population makes
        # the naive baseline look artificially strong - restricting BOTH
        # sides to the players the model actually projected is the fair,
        # apples-to-apples comparison.
        pool = proj['Player']
        pred = pd.Series(proj['Model Proj Pts'].to_numpy(), index=pool)
        naive = naive_full.reindex(pool)

        m = _metrics(pred, actual, f'model w{week}')
        nm = _metrics(naive, actual, f'naive w{week}')
        if m:
            m['week'] = week
            m['pos'] = 'ALL'
            model_rows.append(m)
            for pos in ('QB', 'RB', 'WR', 'TE'):
                sub = proj[proj['Pos'] == pos]
                pred_pos = pd.Series(sub['Model Proj Pts'].to_numpy(), index=sub['Player'])
                mp = _metrics(pred_pos, actual, f'model w{week} {pos}')
                if mp:
                    mp['week'] = week
                    mp['pos'] = pos
                    model_rows.append(mp)
        if nm:
            nm['week'] = week
            nm['pos'] = 'ALL'
            naive_rows.append(nm)
        print(f"week {week}: model n={m['n'] if m else 0}, naive n={nm['n'] if nm else 0}")

    model_df = pd.DataFrame(model_rows)
    naive_df = pd.DataFrame(naive_rows)

    print("\n=== MODEL, overall (ALL positions, pooled across weeks) ===")
    overall = model_df[model_df['pos'] == 'ALL']
    print(f"n={overall['n'].sum()}  MAE={np.average(overall['mae'], weights=overall['n']):.3f}  "
          f"bias={np.average(overall['bias'], weights=overall['n']):+.3f}  "
          f"rank_corr={np.average(overall['rank_corr'], weights=overall['n']):.3f}")

    print("\n=== NAIVE BASELINE (recent-form avg), overall ===")
    print(f"n={naive_df['n'].sum()}  MAE={np.average(naive_df['mae'], weights=naive_df['n']):.3f}  "
          f"bias={np.average(naive_df['bias'], weights=naive_df['n']):+.3f}  "
          f"rank_corr={np.average(naive_df['rank_corr'], weights=naive_df['n']):.3f}")

    print("\n=== MODEL, by position (pooled across weeks) ===")
    for pos in ('QB', 'RB', 'WR', 'TE'):
        sub = model_df[model_df['pos'] == pos]
        if sub.empty:
            continue
        print(f"{pos}: n={sub['n'].sum()}  MAE={np.average(sub['mae'], weights=sub['n']):.3f}  "
              f"bias={np.average(sub['bias'], weights=sub['n']):+.3f}  "
              f"rank_corr={np.average(sub['rank_corr'], weights=sub['n']):.3f}")


if __name__ == '__main__':
    main()
