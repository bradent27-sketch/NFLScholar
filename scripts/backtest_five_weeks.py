"""
Five-checkpoint, no-leakage backtest of the V2 model against 2025, plus a
TD-dependency variance breakdown by position.

WHAT THIS ANSWERS (two separate questions the user asked together):

1. "Run the new model against 5 separate weeks of last season" - each
   checkpoint projects week N using build_weekly_projections(as_of_week=N,
   model_version='v2'), which by construction only reads prior-season history
   plus weeks < N of the current season (see that function's own docstring).
   That is exactly "a mock projection based on the previous season and the
   weeks leading up to it" - no result ever leaks into its own projection.
   Reported per week AND pooled, broken into rank-within-position tiers
   (Starter/Streamer-Bench, TIER_BOUNDS in data/weekly_distribution.py) using
   the MODEL'S OWN projected order that week, since that is the population a
   real start/sit call is made from.

2. "The TE bias may be a weekly-variance/TD-dependency issue, dig into it" -
   independent of the model entirely: for every skill-position player-game in
   the 2025 season, splits actual fantasy points into TD points vs non-TD
   (yardage + reception) points, using this app's own scoring formula
   (data/transforms.py apply_scoring_and_percentiles). Reports, by position:
   the mean share of a game's points that came from TDs, and - the sharper
   test - each player's own week-to-week STANDARD DEVIATION of TD points vs
   non-TD points, averaged within position. A position whose week-to-week
   swings are mostly TD-driven has a variance problem a single-number mean
   projection cannot solve (that is what the boom/bust distribution feature
   is for); a position whose TD component is also predictably OFF-CENTER
   (nonzero pooled bias, not just spread) has a real calibration bug.

Usage:
    python scripts/backtest_five_weeks.py
    python scripts/backtest_five_weeks.py --weeks 4,7,10,13,16 --year 2025
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_projections import build_weekly_projections  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from data.weekly_distribution import TIER_BOUNDS  # noqa: E402


def _actual_points(stats_df, name_col, week, scoring_col='fantasy_points_ppr'):
    rows = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week]
    if scoring_col not in rows.columns:
        return pd.Series(dtype=float)
    return rows.groupby(name_col, observed=True)[scoring_col].sum()


def _naive_baseline(stats_df, name_col, as_of_week, scoring_col='fantasy_points_ppr', n_weeks=4):
    hist = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') < as_of_week]
    if hist.empty or scoring_col not in hist.columns:
        return pd.Series(dtype=float)
    recent = hist.sort_values('week').groupby(name_col, observed=True).tail(n_weeks)
    return recent.groupby(name_col, observed=True)[scoring_col].mean()


def _metrics(pred, actual):
    joined = pd.concat([pred.rename('pred'), actual.rename('actual')], axis=1).dropna()
    if len(joined) < 3:
        return None
    err = joined['pred'] - joined['actual']
    return {'n': len(joined), 'mae': float(err.abs().mean()), 'bias': float(err.mean()),
            'rank_corr': float(joined['pred'].rank().corr(joined['actual'].rank()))}


def _weighted(rows, key):
    if not rows:
        return float('nan')
    n = np.array([r['n'] for r in rows], dtype=float)
    v = np.array([r[key] for r in rows], dtype=float)
    ok = np.isfinite(v)
    return float(np.average(v[ok], weights=n[ok])) if ok.any() else float('nan')


def run_backtest(year, weeks, scoring='Full PPR'):
    scoring_col = 'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'
    stats_df, _team_col, name_col, _ = load_and_merge_data(year, scoring)

    # model_rows[pos][tier] = list of per-week metric dicts; naive_rows mirrors it at ALL only.
    model_rows, naive_rows = {}, {'ALL': []}
    per_week_lines = []

    for week in weeks:
        proj, meta = build_weekly_projections(year, week, scoring, as_of_week=week,
                                               apply_injury=False, model_version='v2')
        actual = _actual_points(stats_df, name_col, week, scoring_col)
        if proj.empty or actual.empty:
            print(f"week {week}: skipped ({meta.get('reason', 'no actuals')})")
            continue
        naive = _naive_baseline(stats_df, name_col, week, scoring_col).reindex(proj['Player'])

        pred_all = pd.Series(proj['Model Proj Pts'].to_numpy(), index=proj['Player'])
        m_all = _metrics(pred_all, actual)
        n_all = _metrics(naive, actual)
        if m_all:
            model_rows.setdefault('ALL', {}).setdefault('ALL', []).append(m_all)
        if n_all:
            naive_rows['ALL'].append(n_all)

        for pos in ('QB', 'RB', 'WR', 'TE'):
            sub = proj[proj['Pos'] == pos].sort_values('Model Proj Pts', ascending=False).reset_index(drop=True)
            for tier_name, lo, hi in TIER_BOUNDS:
                tsub = sub.iloc[lo - 1:hi]
                if tsub.empty:
                    continue
                pred = pd.Series(tsub['Model Proj Pts'].to_numpy(), index=tsub['Player'])
                m = _metrics(pred, actual)
                if m:
                    model_rows.setdefault(pos, {}).setdefault(tier_name, []).append(m)
            pred_pos = pd.Series(sub['Model Proj Pts'].to_numpy(), index=sub['Player'])
            m_pos = _metrics(pred_pos, actual)
            if m_pos:
                model_rows.setdefault(pos, {}).setdefault('ALL', []).append(m_pos)

        per_week_lines.append((week, m_all, n_all))

    return model_rows, naive_rows, per_week_lines


def print_backtest(model_rows, naive_rows, per_week_lines, weeks):
    print(f"\n=== Backtest: V2 model vs naive 4-week trailing average, weeks {weeks} ===")
    print(f"{'week':>6}{'model MAE':>12}{'model bias':>12}{'model rho':>11}"
          f"{'naive MAE':>12}{'naive bias':>12}{'naive rho':>11}")
    for week, m, n in per_week_lines:
        if not m or not n:
            continue
        print(f"{week:>6}{m['mae']:>12.3f}{m['bias']:>+12.3f}{m['rank_corr']:>11.3f}"
              f"{n['mae']:>12.3f}{n['bias']:>+12.3f}{n['rank_corr']:>11.3f}")

    pooled_m = model_rows.get('ALL', {}).get('ALL', [])
    pooled_n = naive_rows.get('ALL', [])
    print(f"\n{'POOLED':>6}{_weighted(pooled_m, 'mae'):>12.3f}{_weighted(pooled_m, 'bias'):>+12.3f}"
          f"{_weighted(pooled_m, 'rank_corr'):>11.3f}{_weighted(pooled_n, 'mae'):>12.3f}"
          f"{_weighted(pooled_n, 'bias'):>+12.3f}{_weighted(pooled_n, 'rank_corr'):>11.3f}")

    print(f"\n{'--- by position x tier (model only, pooled across the 5 weeks) ---'}")
    print(f"{'pos':<6}{'tier':<16}{'n':>6}{'MAE':>8}{'bias':>9}{'rank_corr':>11}")
    for pos in ('QB', 'RB', 'WR', 'TE'):
        for tier_name, _lo, _hi in list(TIER_BOUNDS) + [('ALL', 0, 0)]:
            rows = model_rows.get(pos, {}).get(tier_name, [])
            if not rows:
                continue
            print(f"{pos:<6}{tier_name:<16}{sum(r['n'] for r in rows):>6}"
                  f"{_weighted(rows, 'mae'):>8.3f}{_weighted(rows, 'bias'):>+9.3f}"
                  f"{_weighted(rows, 'rank_corr'):>11.3f}")


TD_POINT_VALUE = {'rushing_tds': 6, 'receiving_tds': 6, 'passing_tds': 4}
NON_TD_COLS = {  # (column, per-unit value) pairs contributing non-TD points
    'rushing_yards': 0.1, 'receiving_yards': 0.1, 'passing_yards': 0.04,
    'receptions': 1.0,  # Full PPR
}


def td_dependency_report(year, scoring='Full PPR'):
    stats_df, _team_col, name_col, _ = load_and_merge_data(year, scoring)
    reg = stats_df[pd.to_numeric(stats_df['week'], errors='coerce').notna()].copy()
    for col in list(TD_POINT_VALUE) + list(NON_TD_COLS):
        if col not in reg.columns:
            reg[col] = 0.0
        reg[col] = pd.to_numeric(reg[col], errors='coerce').fillna(0.0)
    if 'passing_interceptions' in reg.columns:
        reg['int_penalty'] = pd.to_numeric(reg['passing_interceptions'], errors='coerce').fillna(0.0) * -2.0
    else:
        reg['int_penalty'] = 0.0

    reg['td_points'] = sum(reg[c] * v for c, v in TD_POINT_VALUE.items())
    reg['non_td_points'] = sum(reg[c] * v for c, v in NON_TD_COLS.items()) + reg['int_penalty']
    reg['total_points'] = reg['td_points'] + reg['non_td_points']

    if 'position' not in reg.columns:
        print("no position column found; skipping TD-dependency report")
        return

    print(f"\n=== TD-dependency, {year} regular season, actual results only (model not involved) ===")
    print(f"{'pos':<6}{'games':>8}{'mean pts':>10}{'TD-pt share':>13}"
          f"{'wk-to-wk std(TD)':>18}{'wk-to-wk std(nonTD)':>21}{'TD share of variance':>22}")
    for pos in ('QB', 'RB', 'WR', 'TE'):
        sub = reg[reg['position'].astype(str).str.upper() == pos]
        sub = sub[sub['total_points'] > 2.0]  # meaningful games only, not garbage-time zeros
        if sub.empty:
            continue
        mean_pts = sub['total_points'].mean()
        td_share = (sub['td_points'] / sub['total_points'].replace(0, np.nan)).clip(0, 1).mean()

        # Per-player week-to-week std, then averaged across players with >=4 games -
        # this is the "how lumpy is MY week-to-week output" question, distinct from
        # the cross-sectional td_share above.
        by_player = sub.groupby(name_col, observed=True)
        counts = by_player['total_points'].count()
        qualified = counts[counts >= 4].index
        std_td = by_player['td_points'].std().reindex(qualified).mean()
        std_nontd = by_player['non_td_points'].std().reindex(qualified).mean()
        var_td, var_nontd = std_td ** 2, std_nontd ** 2
        td_var_share = var_td / (var_td + var_nontd) if (var_td + var_nontd) > 0 else float('nan')

        print(f"{pos:<6}{len(sub):>8}{mean_pts:>10.2f}{td_share:>12.1%}"
              f"{std_td:>18.2f}{std_nontd:>21.2f}{td_var_share:>21.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=2025)
    ap.add_argument('--weeks', type=str, default='4,7,10,13,16')
    ap.add_argument('--scoring', type=str, default='Full PPR')
    args = ap.parse_args()
    weeks = [int(w) for w in args.weeks.split(',')]

    model_rows, naive_rows, per_week_lines = run_backtest(args.year, weeks, args.scoring)
    print_backtest(model_rows, naive_rows, per_week_lines, weeks)
    td_dependency_report(args.year, args.scoring)


if __name__ == '__main__':
    main()
