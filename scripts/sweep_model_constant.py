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
    python scripts/sweep_model_constant.py --target STAT_K_BY_POS --mode scale_keys \
        --keys RB:rushing_attempts,RB:rushing_yards --values 0.33,0.5,0.75,1,1.5 \
        --add-features v2_stat_k_by_pos --years 2019,2020,2021,2022 --weeks 2-6

--mode set        : the value replaces the constant (float / int constants).
--mode scale      : every numeric leaf of the constant is multiplied by the
                    value (a flat dict constant like STAT_K).
--mode scale_keys : only the leaves named in --keys ("pos:stat", comma-
                    separated) of a NESTED dict constant (STAT_K_BY_POS) are
                    multiplied by the value; every other leaf stays at its
                    shipped/seed setting. Needed because a per-position K
                    sweep tunes one group (e.g. RB volume) at a time while
                    holding the rest fixed - see docs/model_improvement_plan_
                    2026-09-23.md item 4's protocol.
--mode tuple2     : "lo|hi" pairs replace a 2-tuple constant (e.g. a clip range).

--add-features    : comma-separated MODEL_FEATURES names OR'd into BOTH the
                    base and variant build's feature set (on top of
                    DEFAULT_FEATURES) - needed whenever the constant being
                    swept is only even consulted behind its own flag (e.g.
                    STAT_K_BY_POS is dead weight unless 'v2_stat_k_by_pos' is
                    active - see _current_blend_weight's own docstring).

--harness v2 (default): scripts.harness_v2's paired-pool, RMSE/pairwise_acc-
    primary metrics (see that module's docstring for why v1's MAE-only,
    unpaired-START-pool harness is the wrong tool for judging a real
    accuracy change). --harness v1 keeps the original MAE-only report for
    reproducing old numbers.
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
from scripts import harness_v2  # noqa: E402


def _scope_metrics(df, actual, pos, startable):
    d = _scope_df(df, pos, startable)
    return _metrics(pd.Series(d['Model Proj Pts'].to_numpy(), index=d['Player']), actual)


def _apply(target, mode, value, keys=None):
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
    elif mode == 'scale_keys':
        if not (isinstance(old, dict) and all(isinstance(v, dict) for v in old.values())):
            raise SystemExit(f"--mode scale_keys needs a nested dict constant (pos -> {{stat: K}}), "
                             f"{target} is {type(old)}")
        if not keys:
            raise SystemExit("--mode scale_keys needs --keys pos:stat[,pos:stat...]")
        new = copy.deepcopy(old)
        for key in keys:
            pos, stat = key.split(':')
            if pos not in new or stat not in new[pos]:
                raise SystemExit(f"--keys entry {key!r} not found in {target}")
            new[pos][stat] = type(new[pos][stat])(new[pos][stat] * value)
    elif mode == 'tuple2':
        lo, hi = (float(x) for x in str(value).split('|'))
        new = (lo, hi)
    else:
        raise SystemExit(f"unknown --mode {mode}")
    setattr(wp, target, new)
    return old, new


def _run_sweep_v1(target, mode, values, keys, years, weeks, scoring, add_features):
    scoring_col = 'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'
    shipped = copy.deepcopy(getattr(wp, target))
    feats = frozenset(DEFAULT_FEATURES | add_features)
    rows = {str(v): {} for v in values}

    for year in years:
        stats_df, _tc, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns:
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                continue
            setattr(wp, target, copy.deepcopy(shipped))
            build_weekly_projections.clear()
            base_proj, base_meta = build_weekly_projections(
                year, week, scoring, as_of_week=week, apply_injury=False, features=feats)
            if base_proj.empty:
                print(f"{year} w{week}: base empty ({base_meta.get('reason')})")
                continue

            for v in values:
                _apply(target, mode, v, keys)
                build_weekly_projections.clear()
                var_proj, _vm = build_weekly_projections(
                    year, week, scoring, as_of_week=week, apply_injury=False, features=feats)
                setattr(wp, target, copy.deepcopy(shipped))
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

    print(f"\n{'=' * 78}\n{target}  ({mode})  vs shipped {shipped}  [harness v1]\n{'=' * 78}")
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

    setattr(wp, target, shipped)
    print("\n(dMAE negative = the swept value beats the shipped constant. "
          "* = 95% CI excludes 0.)")


def _run_sweep_v2(target, mode, values, keys, years, weeks, scoring, add_features):
    scoring_col = 'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'
    shipped = copy.deepcopy(getattr(wp, target))
    feats = frozenset(DEFAULT_FEATURES | add_features)
    # rows[value][scope] = list of (metrics_base, metrics_variant) weekly pairs
    rows = {str(v): {} for v in values}
    # Stat-level RMSE for whatever stats are actually being swept (derived
    # from --keys, e.g. RB:rushing_attempts -> 'rushing_attempts') - the
    # plan's own ask: a points-only view can hide a real per-stat move.
    # stat_rows[value][stat] = list of (metrics_base, metrics_variant).
    report_stats = sorted({k.split(':')[1] for k in keys}) if keys else []
    stat_rows = {str(v): {} for v in values}

    for year in years:
        stats_df, _tc, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns:
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                continue
            setattr(wp, target, copy.deepcopy(shipped))
            build_weekly_projections.clear()
            base_proj, base_meta = build_weekly_projections(
                year, week, scoring, as_of_week=week, apply_injury=False, features=feats)
            if base_proj.empty:
                print(f"{year} w{week}: base empty ({base_meta.get('reason')})")
                continue
            base_idx = base_proj.set_index('Player')['Model Proj Pts']

            for v in values:
                _apply(target, mode, v, keys)
                build_weekly_projections.clear()
                var_proj, _vm = build_weekly_projections(
                    year, week, scoring, as_of_week=week, apply_injury=False, features=feats)
                setattr(wp, target, copy.deepcopy(shipped))
                build_weekly_projections.clear()
                if var_proj.empty:
                    continue
                var_idx = var_proj.set_index('Player')['Model Proj Pts']
                for scope, pos, startable in harness_v2.SCOPES:
                    pool = harness_v2.scope_pool(base_proj, var_proj, pos, startable,
                                                 actual_players=actual.index)
                    min_pool = max(8, harness_v2.STARTABLE_N.get(pos, 40) // 2) if (startable and pos) else 20
                    if len(pool) < min_pool:
                        continue
                    mb = harness_v2.metrics(base_idx.reindex(pool), actual.loc[pool])
                    mv = harness_v2.metrics(var_idx.reindex(pool), actual.loc[pool])
                    if mb and mv:
                        rows[str(v)].setdefault(scope, []).append((mb, mv))

                if report_stats:
                    week_stats = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week]
                    all_pool = harness_v2.scope_pool(base_proj, var_proj, None, False,
                                                     actual_players=actual.index)
                    for stat in report_stats:
                        if stat not in week_stats.columns:
                            continue
                        actual_stat = week_stats.groupby(name_col, observed=True)[stat].sum()
                        base_sub = base_proj[base_proj['Player'].isin(all_pool)]
                        var_sub = var_proj[var_proj['Player'].isin(all_pool)]
                        actual_stat_df = actual_stat.to_frame(stat)
                        msb = harness_v2.stat_metrics(base_sub, actual_stat_df, [stat])
                        msv = harness_v2.stat_metrics(var_sub, actual_stat_df, [stat])
                        if stat in msb and stat in msv:
                            stat_rows[str(v)].setdefault(stat, []).append((msb[stat], msv[stat]))
            print(f"{year} w{week} done", flush=True)

    print(f"\n{'=' * 78}\n{target}  ({mode})  vs shipped {shipped}  [harness v2]\n{'=' * 78}")
    for v in values:
        label = str(v)
        print(f"\n--- value {label} ---")
        scope_summary = {}
        for scope, _pos, _st in harness_v2.SCOPES:
            pairs = rows[label].get(scope)
            if not pairs:
                continue
            mb_list = [p[0] for p in pairs]
            mv_list = [p[1] for p in pairs]
            n = sum(m['n'] for m in mb_list)
            rmse_b, rmse_v = _weighted(mb_list, 'rmse'), _weighted(mv_list, 'rmse')
            pw_b, pw_v = _weighted(mb_list, 'pairwise_acc'), _weighted(mv_list, 'pairwise_acc')
            mae_b, mae_v = _weighted(mb_list, 'mae'), _weighted(mv_list, 'mae')
            bias_b, bias_v = _weighted(mb_list, 'bias'), _weighted(mv_list, 'bias')

            rmse_deltas = [mv['rmse'] - mb['rmse'] for mb, mv in pairs]
            weights = [mb['n'] for mb, mv in pairs]
            rmse_ci = _bootstrap_ci(rmse_deltas, weights)
            pw_deltas = [mv['pairwise_acc'] - mb['pairwise_acc'] for mb, mv in pairs]
            pw_ci = _bootstrap_ci(pw_deltas, weights)
            rmse_wins = sum(1 for mb, mv in pairs if mv['rmse'] < mb['rmse'])
            rmse_losses = sum(1 for mb, mv in pairs if mb['rmse'] < mv['rmse'])
            scope_summary[scope] = {'rmse_ci': rmse_ci, 'pw_ci': pw_ci, 'bias_b': bias_b, 'bias_v': bias_v}

            print(f"  {scope:<10} n={n:<6} RMSE {rmse_b:.3f}->{rmse_v:.3f} (Δ{rmse_v-rmse_b:+.3f} "
                  f"CI[{rmse_ci[0]:+.3f},{rmse_ci[1]:+.3f}])  "
                  f"pairwiseAcc {pw_b:.3f}->{pw_v:.3f} (Δ{pw_v-pw_b:+.3f} CI[{pw_ci[0]:+.3f},{pw_ci[1]:+.3f}])  "
                  f"bias {bias_b:+.3f}->{bias_v:+.3f}  MAE(secondary) {mae_b:.3f}->{mae_v:.3f}  "
                  f"weeks-var-better(RMSE) {rmse_wins}-{rmse_losses}")

        if all(f'START-{p}' in scope_summary for p in ('QB', 'RB', 'WR', 'TE')) and 'START-ALL' in scope_summary:
            primary = scope_summary['START-ALL']
            bias_growth = abs(primary['bias_v']) - abs(primary['bias_b'])
            # No per-position Holm veto here (unlike backtest_component.py) -
            # a constant sweep is read for its DOSE-RESPONSE across values,
            # not a single ship/reject call; print the primary CI/bias and
            # let a human read the curve across all swept values together.
            print(f"  [START-ALL bias growth {bias_growth:+.3f}]")

        for stat in report_stats:
            pairs = stat_rows[label].get(stat)
            if not pairs:
                continue
            mb_list = [p[0] for p in pairs]
            mv_list = [p[1] for p in pairs]
            n = sum(m['n'] for m in mb_list)
            rmse_b, rmse_v = _weighted(mb_list, 'rmse'), _weighted(mv_list, 'rmse')
            print(f"  stat:{stat:<20} n={n:<6} RMSE {rmse_b:.3f}->{rmse_v:.3f} (Δ{rmse_v-rmse_b:+.3f})")

    setattr(wp, target, shipped)
    print("\n(RMSE/pairwiseAcc CI excluding 0 = distinguishable from noise at this sample. "
          "MAE is secondary/informational only - see scripts/harness_v2.py.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, help="module-level name in data.weekly_projections")
    ap.add_argument('--mode', default='set', choices=['set', 'scale', 'scale_keys', 'tuple2'])
    ap.add_argument('--keys', default=None,
                    help="comma-separated pos:stat leaves to scale, for --mode scale_keys "
                    "(e.g. RB:rushing_attempts,RB:rushing_yards)")
    ap.add_argument('--values', required=True, help="comma-separated values to test vs the shipped one")
    ap.add_argument('--years', default='2024,2025')
    ap.add_argument('--weeks', default='3-16')
    ap.add_argument('--scoring', default='Full PPR')
    ap.add_argument('--add-features', default='',
                    help="comma-separated MODEL_FEATURES names OR'd into DEFAULT_FEATURES for both "
                    "arms - needed when the swept constant is dead weight without its own flag "
                    "(e.g. --target STAT_K_BY_POS needs --add-features v2_stat_k_by_pos)")
    ap.add_argument('--harness', default='v2', choices=['v1', 'v2'])
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(',')]
    if '-' in args.weeks:
        lo, hi = args.weeks.split('-')
        weeks = list(range(int(lo), int(hi) + 1))
    else:
        weeks = [int(w) for w in args.weeks.split(',')]
    raw_values = [v.strip() for v in args.values.split(',') if v.strip()]
    values = [float(v) if args.mode in ('set', 'scale', 'scale_keys') else v for v in raw_values]
    keys = [k.strip() for k in args.keys.split(',')] if args.keys else None
    add_features = frozenset(f.strip() for f in args.add_features.split(',') if f.strip())

    if not hasattr(wp, args.target):
        raise SystemExit(f"data.weekly_projections has no attribute {args.target}")
    from data.weekly_projections import MODEL_FEATURES
    unknown = add_features - set(MODEL_FEATURES)
    if unknown:
        raise SystemExit(f"--add-features not in MODEL_FEATURES: {sorted(unknown)}")

    print(f"target={args.target}  shipped={getattr(wp, args.target)}")
    print(f"mode={args.mode}  keys={keys}  values={values}  add_features={sorted(add_features)}")
    print(f"years={years} weeks={weeks[0]}-{weeks[-1]} scoring={args.scoring} harness={args.harness}\n")

    runner = _run_sweep_v2 if args.harness == 'v2' else _run_sweep_v1
    runner(args.target, args.mode, values, keys, years, weeks, args.scoring, add_features)


if __name__ == '__main__':
    main()
