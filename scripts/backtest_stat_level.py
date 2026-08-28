"""
Per-RAW-STAT extension of scripts/backtest_component.py, for questions like
"does this component help targets specifically, even if receiving_yards is a
wash" - the fantasy-points MAE in backtest_component.py is a single number
that several raw stats' errors can offset inside of, which can hide a real
effect on one stat behind noise on another.

Same paired A/B discipline (paired player pool per week, bootstrap CI, exact
sign-test p-value) as backtest_component.py, but scores each of
--stats (default: targets, receptions, receiving_yards) against its own
ACTUAL box-score value instead of scoring 'Model Proj Pts' against fantasy
points. Restricted to WR/TE - the only positions v2_scheme_matchup and
v2_pff_alignment_matchup touch at all.

Usage:
    python scripts/backtest_stat_level.py --add v2_scheme_matchup --years 2024,2025 --weeks 2-18
    python scripts/backtest_stat_level.py --flags v2_pff_alignment_matchup --stats targets,receiving_yards
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES, MODEL_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from scripts.eval_weekly_model import _metrics, _weighted, STARTABLE_N  # noqa: E402
from scripts.backtest_component import _bootstrap_ci, _sign_test_p  # noqa: E402

POSITIONS = ('WR', 'TE')


def _actual_stat(stats_df, name_col, week, stat_col):
    rows = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week]
    if stat_col not in rows.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(rows.groupby(name_col, observed=True)[stat_col].sum(), errors='coerce')


def evaluate_stat_level(years, weeks, variant_defs, raw_stats, scoring='Full PPR'):
    # rows[label][pos][stat][scope] = list of (metrics_base, metrics_variant)
    rows = {label: {pos: {stat: {} for stat in raw_stats} for pos in POSITIONS} for label in variant_defs}

    for year in years:
        stats_df, _team_col, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns:
            print(f"{year}: no weekly data, skipped")
            continue
        for week in weeks:
            actuals = {stat: _actual_stat(stats_df, name_col, week, stat) for stat in raw_stats}
            if all(a.empty for a in actuals.values()):
                continue
            base_proj, base_meta = build_weekly_projections(
                year, week, scoring, as_of_week=week, apply_injury=False, features=DEFAULT_FEATURES)
            if base_proj.empty:
                continue

            for label, variant_feats in variant_defs.items():
                var_proj, var_meta = build_weekly_projections(
                    year, week, scoring, as_of_week=week, apply_injury=False, features=variant_feats)
                if var_proj.empty:
                    continue
                pool = sorted(set(base_proj['Player']) & set(var_proj['Player']))
                if len(pool) < 20:
                    continue
                base_sub = base_proj[base_proj['Player'].isin(pool)]
                var_sub = var_proj[var_proj['Player'].isin(pool)]

                for pos in POSITIONS:
                    base_pos = base_sub[base_sub['Pos'] == pos]
                    var_pos = var_sub[var_sub['Pos'] == pos]
                    base_top = base_pos.nlargest(STARTABLE_N.get(pos, 30), 'Model Proj Pts')
                    var_top = var_pos.nlargest(STARTABLE_N.get(pos, 30), 'Model Proj Pts')

                    for stat in raw_stats:
                        actual = actuals[stat]
                        if actual.empty or stat not in base_pos.columns:
                            continue
                        for scope, base_df, var_df in (
                            (f'{pos}', base_pos, var_pos),
                            (f'START-{pos}', base_top, var_top),
                        ):
                            pred_b = pd.Series(base_df[stat].to_numpy(dtype=float), index=base_df['Player'])
                            pred_v = pd.Series(var_df[stat].to_numpy(dtype=float), index=var_df['Player'])
                            mb = _metrics(pred_b, actual)
                            mv = _metrics(pred_v, actual)
                            if mb and mv:
                                rows[label][pos][stat].setdefault(scope, []).append((mb, mv))
    return rows


def print_report(label, rows, mode='ablate'):
    print(f"\n{'=' * 74}\n{label}  [{mode} mode]  per-stat breakdown\n{'=' * 74}")
    for pos in POSITIONS:
        for stat, scopes in rows[pos].items():
            for scope in (pos, f'START-{pos}'):
                pairs = scopes.get(scope)
                if not pairs:
                    continue
                mb_list = [p[0] for p in pairs]
                mv_list = [p[1] for p in pairs]
                n_total = sum(m['n'] for m in mb_list)
                mae_base = _weighted(mb_list, 'mae')
                mae_variant = _weighted(mv_list, 'mae')
                d_mae = mae_variant - mae_base
                wins = sum(1 for mb, mv in pairs if mv['mae'] < mb['mae'])
                losses = sum(1 for mb, mv in pairs if mb['mae'] < mv['mae'])
                deltas = [mv['mae'] - mb['mae'] for mb, mv in pairs]
                weights = [mb['n'] for mb, mv in pairs]
                lo, hi = _bootstrap_ci(deltas, weights)
                p = _sign_test_p(wins, losses)
                if math.isnan(lo):
                    sig = "[too few weeks]"
                elif lo > 0 or hi < 0:
                    sig = "[CI excludes 0]"
                else:
                    sig = "[CI includes 0]"
                print(f"  {scope:<10} {stat:<16} n={n_total:<6} MAE {mae_base:.3f} vs {mae_variant:.3f}  "
                      f"dMAE(var-base) {d_mae:+.3f}  weeks {wins}-{losses} (p={p:.2f})  "
                      f"CI[{lo:+.3f},{hi:+.3f}] {sig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flags', default='', help="comma-separated DEFAULT_FEATURES names to ABLATE")
    ap.add_argument('--add', default='', help="comma-separated MODEL_FEATURES names to ADD as a candidate")
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

    ablate_flags = [f.strip() for f in args.flags.split(',') if f.strip()]
    add_flags = [f.strip() for f in args.add.split(',') if f.strip()]
    if not ablate_flags and not add_flags:
        raise SystemExit("pass --flags (ablate) and/or --add (candidate addition)")

    variant_defs = {}
    modes = {}
    for flag in ablate_flags:
        variant_defs[flag] = frozenset(DEFAULT_FEATURES - {flag})
        modes[flag] = 'ablate'
    for flag in add_flags:
        variant_defs[flag] = frozenset(DEFAULT_FEATURES | {flag})
        modes[flag] = 'add'

    print(f"years={years} weeks={weeks[0]}-{weeks[-1]} scoring={args.scoring} stats={raw_stats}")
    rows = evaluate_stat_level(years, weeks, variant_defs, raw_stats, args.scoring)
    for label in variant_defs:
        print_report(label, rows[label], mode=modes[label])


if __name__ == '__main__':
    main()
