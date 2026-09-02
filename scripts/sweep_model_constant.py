"""Paired A/B sweep of a single hand-set model constant that has never been
backtested. Same paired-pool + bootstrap-CI discipline as
scripts/backtest_component.py, but instead of toggling a feature flag it
monkeypatches a module-level constant in data.weekly_projections and clears
the build cache between values (the cache key is `features`, which does not
change when a constant does - skipping the clear would serve a stale board).

    python scripts/sweep_model_constant.py --target RECENCY_DECAY --mode set \
        --values 0.75,0.80,0.90,0.95 --years 2024,2025 --weeks 3-16
    python scripts/sweep_model_constant.py --target STAT_K --mode scale \
        --values 0.5,0.75,1.5,2.0 --years 2024,2025 --weeks 3-16

--mode set    : the value replaces the constant (float / int constants).
--mode scale  : every numeric leaf of the constant is multiplied by the value
                (dict constants like STAT_K).
--mode tuple2 : "lo|hi" pairs replace a 2-tuple constant (e.g. a clip range).
"""
import argparse
import copy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data.weekly_projections as wp  # noqa: E402
from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from scripts.eval_weekly_model import _metrics, _weighted, STARTABLE_N, _actual_points  # noqa: E402
from scripts.backtest_component import _scope_df, _bootstrap_ci, _sign_test_p, SCOPES  # noqa: E402


def _scope_metrics(df, actual, pos, startable):
    d = _scope_df(df, pos, startable)
    return _metrics(pd.Series(d['Model Proj Pts'].to_numpy(), index=d['Player']), actual)


def _apply(target, mode, value):
    """Return (old_value, new_value). Mutates wp.<target> to new_value."""
    old = getattr(wp, target)
    if mode == 'set':
        new = type(old)(value) if isinstance(old, (int, float)) else value
    elif mode == 'scale':
        if isinstance(old, dict):
            new = {k: (type(v)(v * value) if isinstance(v, (int, float)) else v)
                   for k, v in old.items()}
        elif isinstance(old, (int, float)):
            new = type(old)(old * value)
        else:
            raise SystemExit(f"--mode scale needs a dict/number constant, {target} is {type(old)}")
    elif mode == 'tuple2':
        lo, hi = (float(x) for x in str(value).split('|'))
        new = (lo, hi)
    else:
        raise SystemExit(f"unknown --mode {mode}")
    setattr(wp, target, new)
    return old, new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, help="module-level name in data.weekly_projections")
    ap.add_argument('--mode', default='set', choices=['set', 'scale', 'tuple2'])
    ap.add_argument('--values', required=True, help="comma-separated values to test vs the shipped one")
    ap.add_argument('--years', default='2024,2025')
    ap.add_argument('--weeks', default='3-16')
    ap.add_argument('--scoring', default='Full PPR')
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(',')]
    if '-' in args.weeks:
        lo, hi = args.weeks.split('-')
        weeks = list(range(int(lo), int(hi) + 1))
    else:
        weeks = [int(w) for w in args.weeks.split(',')]
    raw_values = [v.strip() for v in args.values.split(',') if v.strip()]
    values = [float(v) if args.mode in ('set', 'scale') else v for v in raw_values]

    if not hasattr(wp, args.target):
        raise SystemExit(f"data.weekly_projections has no attribute {args.target}")
    shipped = copy.deepcopy(getattr(wp, args.target))
    scoring_col = 'fantasy_points_ppr' if args.scoring != 'Standard' else 'fantasy_points'

    print(f"target={args.target}  shipped={shipped}")
    print(f"mode={args.mode}  values={values}")
    print(f"years={years} weeks={weeks[0]}-{weeks[-1]} scoring={args.scoring}\n")

    # rows[label][scope] = list of (metrics_base, metrics_variant) weekly pairs
    rows = {str(v): {} for v in values}

    for year in years:
        stats_df, _tc, name_col, _ = load_and_merge_data(year, args.scoring)
        if 'week' not in stats_df.columns:
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                continue
            setattr(wp, args.target, copy.deepcopy(shipped))
            build_weekly_projections.clear()
            base_proj, base_meta = build_weekly_projections(
                year, week, args.scoring, as_of_week=week, apply_injury=False, features=DEFAULT_FEATURES)
            if base_proj.empty:
                print(f"{year} w{week}: base empty ({base_meta.get('reason')})")
                continue

            for v in values:
                _apply(args.target, args.mode, v)
                build_weekly_projections.clear()
                var_proj, _vm = build_weekly_projections(
                    year, week, args.scoring, as_of_week=week, apply_injury=False,
                    features=DEFAULT_FEATURES)
                setattr(wp, args.target, copy.deepcopy(shipped))
                build_weekly_projections.clear()
                if var_proj.empty:
                    continue
                pool = sorted(set(base_proj['Player']) & set(var_proj['Player']))
                if len(pool) < 20:
                    continue
                b = base_proj[base_proj['Player'].isin(pool)]
                vv = var_proj[var_proj['Player'].isin(pool)]
                for scope, pos, startable in SCOPES:
                    mb = _scope_metrics(b, actual, pos, startable)
                    mv = _scope_metrics(vv, actual, pos, startable)
                    if mb and mv:
                        rows[str(v)].setdefault(scope, []).append((mb, mv))
            print(f"{year} w{week} done", flush=True)

    print(f"\n{'=' * 78}\n{args.target}  ({args.mode})  vs shipped {shipped}\n{'=' * 78}")
    for v in values:
        label = str(v)
        print(f"\n--- value {label} ---")
        for scope, _pos, _st in SCOPES:
            pairs = rows[label].get(scope)
            if not pairs:
                continue
            mb_list = [p[0] for p in pairs]
            mv_list = [p[1] for p in pairs]
            n = sum(m['n'] for m in mb_list)
            mae_b = _weighted(mb_list, 'mae')
            mae_v = _weighted(mv_list, 'mae')
            d = mae_v - mae_b
            wins = sum(1 for a, b in pairs if b['mae'] < a['mae'])
            losses = sum(1 for a, b in pairs if a['mae'] < b['mae'])
            deltas = [b['mae'] - a['mae'] for a, b in pairs]
            weights = [a['n'] for a in mb_list]
            clo, chi = _bootstrap_ci(deltas, weights)
            p = _sign_test_p(wins, losses)
            flag = ' *' if (np.isfinite(clo) and (clo > 0 or chi < 0)) else ''
            print(f"  {scope:<10} n={n:<6} MAE {mae_b:.3f}->{mae_v:.3f}  dMAE {d:+.3f}  "
                  f"w-l {wins}-{losses} (p={p:.2f})  CI[{clo:+.3f},{chi:+.3f}]{flag}")

    setattr(wp, args.target, shipped)
    print("\n(dMAE negative = the swept value beats the shipped constant. "
          "* = 95% CI excludes 0.)")


if __name__ == '__main__':
    main()
