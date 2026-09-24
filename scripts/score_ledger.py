"""
Score the prediction ledger (data.prediction_ledger) against real outcomes -
model vs. market vs. FantasyPros, per position, points and stat, using
harness_v2's metrics on the SAME paired pool for all three sources. This is
the live counterpart to scripts/backtest_component.py: that script rebuilds
historical boards; this one scores whatever real live boards were actually
saved to data/ledger/ (see docs/model_improvement_plan_2026-09-23.md item 2c
- the ledger exists precisely because the live market/FP numbers for a week
that has already happened are gone once that week's board is overwritten).

Uses the LATEST ledger file for each (year, week) - data.prediction_ledger's
own "one build per hour" throttle already keeps that close to the last
pre-kickoff build; a week is only scored once real actuals exist for it.

Usage:
    python scripts/score_ledger.py --year 2026
    python scripts/score_ledger.py --year 2026 --weeks 1-2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.prediction_ledger import latest_ledger_paths, STAT_COLS  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402
from scripts import harness_v2  # noqa: E402

SOURCES = {
    'Model': 'Model Proj Pts',
    'Market': 'Mkt Market Pts',
    'FantasyPros': None,   # resolved per-row below - the PPR/Half/Standard column varies by scoring
}


def _fp_points_col(ledger):
    for c in ledger.columns:
        if c.startswith('FP Proj Pts'):
            return c
    return None


def _score_week(year, week, ledger, stats_df, name_col, scoring_col):
    actual = (stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week]
             .groupby(name_col, observed=True)[scoring_col].sum())
    if actual.empty:
        return None
    ledger = ledger.copy()
    ledger['_key'] = clean_name_exact(ledger['Player'])
    actual_by_key = actual.copy()
    actual_by_key.index = clean_name_exact(pd.Series(actual.index))
    ledger = ledger.set_index('_key')

    fp_col = _fp_points_col(ledger)
    source_cols = {'Model': 'Model Proj Pts', 'Market': 'Mkt Market Pts', 'FantasyPros': fp_col}

    print(f"\n{'=' * 78}\n{year} week {week}  (ledger n={len(ledger)}, actuals n={len(actual_by_key)})"
          f"\n{'=' * 78}")

    for pos in (None, 'QB', 'RB', 'WR', 'TE'):
        sub = ledger if pos is None else ledger[ledger['Pos'] == pos]
        if sub.empty:
            continue
        pool_all = sorted(set(sub.index) & set(actual_by_key.index))
        if len(pool_all) < 5:
            continue
        n = harness_v2.STARTABLE_N.get(pos, len(pool_all)) if pos else len(pool_all)
        start_pool = (sub.loc[pool_all, 'Model Proj Pts'].nlargest(min(n, len(pool_all))).index.tolist()
                     if pos else pool_all)
        label = pos or 'ALL'
        scopes_to_score = [(label, pool_all)]
        if pos:
            scopes_to_score.append((f'START-{label}', start_pool))
        for scope_name, pool in scopes_to_score:
            if len(pool) < 5:
                continue
            line = f"  {scope_name:<10}"
            for source, col in source_cols.items():
                if not col or col not in sub.columns:
                    continue
                pred = sub.loc[pool, col]
                m = harness_v2.metrics(pred, actual_by_key.loc[pool])
                if not m:
                    continue
                line += (f"  {source}: n={m['n']} RMSE={m['rmse']:.2f} MAE={m['mae']:.2f} "
                        f"bias={m['bias']:+.2f} pairwiseAcc={m['pairwise_acc']:.3f}")
            print(line)

    # Per-stat RMSE, model vs market vs FP, on the whole (non-startable) pool.
    print("  -- per-stat RMSE (model / market / FP), whole pool --")
    for stat in STAT_COLS:
        if stat not in stats_df.columns or stat not in ledger.columns:
            continue
        actual_stat = (stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week]
                       .groupby(name_col, observed=True)[stat].sum())
        actual_stat.index = clean_name_exact(pd.Series(actual_stat.index))
        pool = sorted(set(ledger.index) & set(actual_stat.index))
        if len(pool) < 5:
            continue
        cells = []
        for label, col in (('model', stat), ('market', f'Mkt {stat}'), ('FP', f'FP {stat}')):
            if col not in ledger.columns:
                continue
            pred = pd.to_numeric(ledger.loc[pool, col], errors='coerce')
            joined = pd.concat([pred.rename('pred'), actual_stat.loc[pool].rename('actual')], axis=1).dropna()
            if len(joined) < 5:
                continue
            rmse = float(np.sqrt(((joined['pred'] - joined['actual']) ** 2).mean()))
            cells.append(f"{label} RMSE={rmse:.2f} (n={len(joined)})")
        if cells:
            print(f"    {stat:<22}" + '  '.join(cells))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--weeks', default=None, help="e.g. 1-3; default is every ledger week found")
    ap.add_argument('--scoring', default='Full PPR')
    args = ap.parse_args()

    scoring_col = 'fantasy_points_ppr' if args.scoring != 'Standard' else 'fantasy_points'
    stats_df, _team_col, name_col, _ = load_and_merge_data(args.year, args.scoring)
    if 'week' not in stats_df.columns:
        raise SystemExit(f"no weekly stats available for {args.year}")

    latest = latest_ledger_paths(args.year)
    weeks = sorted(w for (_y, w) in latest)
    if args.weeks:
        if '-' in args.weeks:
            lo, hi = args.weeks.split('-')
            requested = set(range(int(lo), int(hi) + 1))
        else:
            requested = {int(w) for w in args.weeks.split(',')}
        weeks = [w for w in weeks if w in requested]

    if not weeks:
        raise SystemExit(f"no ledger files found for {args.year} under data/ledger/ "
                         "(nothing to score yet - build a board first)")

    scored_any = False
    for week in weeks:
        path = latest[(args.year, week)]
        ledger = pd.read_parquet(path)
        if _score_week(args.year, week, ledger, stats_df, name_col, scoring_col):
            scored_any = True
        else:
            print(f"\n{args.year} week {week}: no actuals yet (not played, or stats not loaded) - skipped")

    if not scored_any:
        print("\nnothing scoreable yet - every ledger week found is still upcoming")


if __name__ == '__main__':
    main()
