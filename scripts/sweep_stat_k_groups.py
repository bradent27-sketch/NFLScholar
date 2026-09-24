"""Item 4 driver (docs/model_improvement_plan_2026-09-23.md): sweeps all four
STAT_K_BY_POS groups from that item's protocol in ONE process, building each
week's base board once and reusing it across every group's variants -
scripts/sweep_model_constant.py run four times separately would rebuild that
same base board four times over for nothing, since none of the four groups
touch each other's keys.

Groups (each holds every other STAT_K_BY_POS leaf at its shipped/seed value,
scaling only its own keys together by one shared factor per test point):
    RB volume      : rushing_attempts, rushing_yards, targets, receptions
    WR/TE yardage  : receiving_yards (both positions)
    WR/TE TDs      : receiving_tds (both positions)
    QB rushing     : rushing_attempts, rushing_yards
(QB passing volume is deliberately excluded - the plan defers it until item
5's script-neutral volume work lands.)

Usage (fit window per the plan):
    python scripts/sweep_stat_k_groups.py --years 2019,2020,2021,2022 --weeks 2-6
    python scripts/sweep_stat_k_groups.py --years 2023,2024,2025 --weeks 2-6   # confirm
    python scripts/sweep_stat_k_groups.py --years 2023,2024,2025 --weeks 7-17  # mid-season check
"""
import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

import data.weekly_projections as wp  # noqa: E402
from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from scripts.eval_weekly_model import _actual_points, _weighted  # noqa: E402
from scripts.sweep_model_constant import _apply  # noqa: E402
from scripts import harness_v2  # noqa: E402

GROUPS = {
    'RB_volume': {'keys': ['RB:rushing_attempts', 'RB:rushing_yards', 'RB:targets', 'RB:receptions'],
                 'values': [0.33, 0.5, 0.75, 1.0, 1.5]},
    'WR_TE_yardage': {'keys': ['WR:receiving_yards', 'TE:receiving_yards'],
                      'values': [0.75, 1.0, 1.5, 2.0]},
    'WR_TE_tds': {'keys': ['WR:receiving_tds', 'TE:receiving_tds'],
                 'values': [1.0, 1.5, 2.0, 3.0]},
    'QB_rushing': {'keys': ['QB:rushing_attempts', 'QB:rushing_yards'],
                  'values': [1.0, 1.5, 2.0]},
}
FEATS = frozenset(DEFAULT_FEATURES | {'v2_stat_k_by_pos'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2019,2020,2021,2022')
    ap.add_argument('--weeks', default='2-6')
    ap.add_argument('--scoring', default='Full PPR')
    ap.add_argument('--groups', default=','.join(GROUPS),
                    help="comma-separated subset of: " + ','.join(GROUPS))
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(',')]
    if '-' in args.weeks:
        lo, hi = args.weeks.split('-')
        weeks = list(range(int(lo), int(hi) + 1))
    else:
        weeks = [int(w) for w in args.weeks.split(',')]
    groups = [g.strip() for g in args.groups.split(',') if g.strip()]
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise SystemExit(f"unknown group(s) {sorted(unknown)}; known: {sorted(GROUPS)}")
    scoring_col = 'fantasy_points_ppr' if args.scoring != 'Standard' else 'fantasy_points'

    shipped = copy.deepcopy(wp.STAT_K_BY_POS)
    rows = {g: {str(v): {} for v in GROUPS[g]['values']} for g in groups}
    stat_rows = {g: {str(v): {} for v in GROUPS[g]['values']} for g in groups}
    report_stats = {g: sorted({k.split(':')[1] for k in GROUPS[g]['keys']}) for g in groups}

    print(f"groups={groups}  years={years}  weeks={weeks[0]}-{weeks[-1]}  scoring={args.scoring}")
    for g in groups:
        print(f"  {g}: keys={GROUPS[g]['keys']} values={GROUPS[g]['values']}")
    print()

    for year in years:
        stats_df, _tc, name_col, _ = load_and_merge_data(year, args.scoring)
        if 'week' not in stats_df.columns:
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                continue
            wp.STAT_K_BY_POS = copy.deepcopy(shipped)
            build_weekly_projections.clear()
            base_proj, base_meta = build_weekly_projections(
                year, week, args.scoring, as_of_week=week, apply_injury=False, features=FEATS)
            if base_proj.empty:
                print(f"{year} w{week}: base empty ({base_meta.get('reason')})")
                continue
            base_idx = base_proj.set_index('Player')['Model Proj Pts']
            week_stats = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week]

            for group in groups:
                spec = GROUPS[group]
                for v in spec['values']:
                    _apply('STAT_K_BY_POS', 'scale_keys', v, spec['keys'])
                    build_weekly_projections.clear()
                    var_proj, _vm = build_weekly_projections(
                        year, week, args.scoring, as_of_week=week, apply_injury=False, features=FEATS)
                    wp.STAT_K_BY_POS = copy.deepcopy(shipped)
                    build_weekly_projections.clear()
                    if var_proj.empty:
                        continue
                    var_idx = var_proj.set_index('Player')['Model Proj Pts']
                    for scope, pos, startable in harness_v2.SCOPES:
                        pool = harness_v2.scope_pool(base_proj, var_proj, pos, startable,
                                                     actual_players=actual.index)
                        min_pool = (max(8, harness_v2.STARTABLE_N.get(pos, 40) // 2)
                                   if (startable and pos) else 20)
                        if len(pool) < min_pool:
                            continue
                        mb = harness_v2.metrics(base_idx.reindex(pool), actual.loc[pool])
                        mv = harness_v2.metrics(var_idx.reindex(pool), actual.loc[pool])
                        if mb and mv:
                            rows[group][str(v)].setdefault(scope, []).append((mb, mv))

                    all_pool = harness_v2.scope_pool(base_proj, var_proj, None, False,
                                                     actual_players=actual.index)
                    base_sub = base_proj[base_proj['Player'].isin(all_pool)]
                    var_sub = var_proj[var_proj['Player'].isin(all_pool)]
                    for stat in report_stats[group]:
                        if stat not in week_stats.columns:
                            continue
                        actual_stat = week_stats.groupby(name_col, observed=True)[stat].sum().to_frame(stat)
                        msb = harness_v2.stat_metrics(base_sub, actual_stat, [stat])
                        msv = harness_v2.stat_metrics(var_sub, actual_stat, [stat])
                        if stat in msb and stat in msv:
                            stat_rows[group][str(v)].setdefault(stat, []).append((msb[stat], msv[stat]))
            print(f"{year} w{week} done", flush=True)

    for group in groups:
        print(f"\n{'=' * 78}\nGROUP {group}   keys={GROUPS[group]['keys']}\n{'=' * 78}")
        for v in GROUPS[group]['values']:
            label = str(v)
            print(f"\n--- value {label} ---")
            scope_summary = {}
            for scope, _pos, _st in harness_v2.SCOPES:
                pairs = rows[group][label].get(scope)
                if not pairs:
                    continue
                mb_list = [p[0] for p in pairs]
                mv_list = [p[1] for p in pairs]
                n = sum(m['n'] for m in mb_list)
                rmse_b, rmse_v = _weighted(mb_list, 'rmse'), _weighted(mv_list, 'rmse')
                pw_b, pw_v = _weighted(mb_list, 'pairwise_acc'), _weighted(mv_list, 'pairwise_acc')
                mae_b, mae_v = _weighted(mb_list, 'mae'), _weighted(mv_list, 'mae')
                bias_b, bias_v = _weighted(mb_list, 'bias'), _weighted(mv_list, 'bias')
                rmse_ci = harness_v2.bootstrap_ci([mv['rmse'] - mb['rmse'] for mb, mv in pairs],
                                                  [mb['n'] for mb, mv in pairs])
                pw_ci = harness_v2.bootstrap_ci([mv['pairwise_acc'] - mb['pairwise_acc'] for mb, mv in pairs],
                                                [mb['n'] for mb, mv in pairs])
                rmse_wins = sum(1 for mb, mv in pairs if mv['rmse'] < mb['rmse'])
                rmse_losses = sum(1 for mb, mv in pairs if mb['rmse'] < mv['rmse'])
                scope_summary[scope] = {'rmse_ci': rmse_ci, 'bias_b': bias_b, 'bias_v': bias_v}
                print(f"  {scope:<10} n={n:<6} RMSE {rmse_b:.3f}->{rmse_v:.3f} "
                      f"(Δ{rmse_v-rmse_b:+.3f} CI[{rmse_ci[0]:+.3f},{rmse_ci[1]:+.3f}])  "
                      f"pairwiseAcc {pw_b:.3f}->{pw_v:.3f} (Δ{pw_v-pw_b:+.3f} "
                      f"CI[{pw_ci[0]:+.3f},{pw_ci[1]:+.3f}])  bias {bias_b:+.3f}->{bias_v:+.3f}  "
                      f"MAE(secondary) {mae_b:.3f}->{mae_v:.3f}  weeks-var-better(RMSE) {rmse_wins}-{rmse_losses}")
            if 'START-ALL' in scope_summary:
                p = scope_summary['START-ALL']
                print(f"  [START-ALL bias growth {abs(p['bias_v']) - abs(p['bias_b']):+.3f}]")
            for stat in report_stats[group]:
                pairs = stat_rows[group][label].get(stat)
                if not pairs:
                    continue
                mb_list = [p[0] for p in pairs]
                mv_list = [p[1] for p in pairs]
                n = sum(m['n'] for m in mb_list)
                rmse_b, rmse_v = _weighted(mb_list, 'rmse'), _weighted(mv_list, 'rmse')
                print(f"  stat:{stat:<20} n={n:<6} RMSE {rmse_b:.3f}->{rmse_v:.3f} (Δ{rmse_v-rmse_b:+.3f})")

    print("\n(RMSE/pairwiseAcc CI excluding 0 = distinguishable from noise at this sample. "
          "Pick each group's best value here, then confirm the combined setting on "
          "2023-2025 against DEFAULT_FEATURES with backtest_component.py --add v2_stat_k_by_pos "
          "once STAT_K_BY_POS itself is updated to the chosen values.)")


if __name__ == '__main__':
    main()
