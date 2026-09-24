"""Item 3, experiment 1 (docs/model_improvement_plan_2026-09-23.md): does
`v2_historical_injury_replay` actually help - and does it help MOST where it
should (a player whose team just lost a real starter, per
data.historical_availability.significant_out_players)?

Per the plan's own instruction: "It should improve RECIPIENT a lot, ALL a
little. If it doesn't, stop and debug before going further" - run this
FIRST, before ablating v2_vacancy/pecking-order/etc. with replay on in both
arms (those are separate follow-up experiments once this one confirms the
replay data itself is good).

Usage:
    python scripts/backtest_injury_replay.py --years 2022,2023,2024,2025 --weeks 3-17
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from data.loaders import load_schedule  # noqa: E402
from data.historical_availability import significant_out_players, recipient_teammates  # noqa: E402
from scripts.eval_weekly_model import _actual_points, _weighted  # noqa: E402
from scripts import harness_v2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2022,2023,2024,2025')
    ap.add_argument('--weeks', default='3-17')
    ap.add_argument('--scoring', default='Full PPR')
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(',')]
    if '-' in args.weeks:
        lo, hi = args.weeks.split('-')
        weeks = list(range(int(lo), int(hi) + 1))
    else:
        weeks = [int(w) for w in args.weeks.split(',')]
    scoring_col = 'fantasy_points_ppr' if args.scoring != 'Standard' else 'fantasy_points'

    base_feats = DEFAULT_FEATURES
    variant_feats = frozenset(DEFAULT_FEATURES | {'v2_historical_injury_replay'})
    print(f"years={years} weeks={weeks[0]}-{weeks[-1]} scoring={args.scoring}")
    print(f"base=DEFAULT_FEATURES ({len(base_feats)})  "
          f"variant=DEFAULT_FEATURES+v2_historical_injury_replay ({len(variant_feats)})\n")

    rows = {}   # scope -> [(metrics_base, metrics_variant), ...]
    n_recipient_weeks = 0
    n_recipients_total = 0

    for year in years:
        stats_df, _tc, name_col, _ = load_and_merge_data(year, args.scoring)
        if 'week' not in stats_df.columns:
            continue
        schedule = load_schedule(year)
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                continue
            base_proj, base_meta = build_weekly_projections(
                year, week, args.scoring, as_of_week=week, apply_injury=False, features=base_feats)
            if base_proj.empty:
                print(f"{year} w{week}: base empty ({base_meta.get('reason')})")
                continue
            var_proj, var_meta = build_weekly_projections(
                year, week, args.scoring, as_of_week=week, apply_injury=False, features=variant_feats)
            if var_proj.empty:
                print(f"{year} w{week}: variant empty ({var_meta.get('reason')})")
                continue

            out_players = significant_out_players(year, week, schedule, stats_df, name_col=name_col)
            board_players_by_team = base_proj.groupby('Team')['Player'].apply(set).to_dict()
            recipients = recipient_teammates(out_players, board_players_by_team)
            if recipients:
                n_recipient_weeks += 1
                n_recipients_total += len(recipients)

            base_idx = base_proj.set_index('Player')['Model Proj Pts']
            var_idx = var_proj.set_index('Player')['Model Proj Pts']
            for scope, pos, startable in harness_v2.SCOPES:
                pool = harness_v2.scope_pool(base_proj, var_proj, pos, startable, actual_players=actual.index)
                min_pool = max(8, harness_v2.STARTABLE_N.get(pos, 40) // 2) if (startable and pos) else 20
                if len(pool) < min_pool:
                    continue
                mb = harness_v2.metrics(base_idx.reindex(pool), actual.loc[pool])
                mv = harness_v2.metrics(var_idx.reindex(pool), actual.loc[pool])
                if mb and mv:
                    rows.setdefault(scope, []).append((mb, mv))

            if recipients:
                pool = sorted(recipients & set(base_proj['Player']) & set(var_proj['Player'])
                             & set(actual.index))
                if len(pool) >= 5:
                    mb = harness_v2.metrics(base_idx.reindex(pool), actual.loc[pool])
                    mv = harness_v2.metrics(var_idx.reindex(pool), actual.loc[pool])
                    if mb and mv:
                        rows.setdefault('RECIPIENT', []).append((mb, mv))
            print(f"{year} w{week} done (significant-out this week: {len(out_players)}, "
                 f"recipients: {len(recipients)})", flush=True)

    print(f"\nweeks with >=1 qualifying RECIPIENT pool: {n_recipient_weeks}, "
         f"total recipient player-weeks: {n_recipients_total}\n")

    print(f"{'=' * 78}\nv2_historical_injury_replay  [add mode]\n{'=' * 78}")
    for scope in ['RECIPIENT'] + [s for s, _p, _st in harness_v2.SCOPES]:
        pairs = rows.get(scope)
        if not pairs:
            print(f"  {scope:<10} (no weeks cleared the pool-size floor)")
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
        wins = sum(1 for mb, mv in pairs if mv['rmse'] < mb['rmse'])
        losses = sum(1 for mb, mv in pairs if mb['rmse'] < mv['rmse'])
        print(f"  {scope:<10} weeks={len(pairs):<4} n={n:<6} "
             f"RMSE {rmse_b:.3f}->{rmse_v:.3f} (Δ{rmse_v-rmse_b:+.3f} CI[{rmse_ci[0]:+.3f},{rmse_ci[1]:+.3f}])  "
             f"pairwiseAcc {pw_b:.3f}->{pw_v:.3f} (Δ{pw_v-pw_b:+.3f} CI[{pw_ci[0]:+.3f},{pw_ci[1]:+.3f}])  "
             f"bias {bias_b:+.3f}->{bias_v:+.3f}  MAE(secondary) {mae_b:.3f}->{mae_v:.3f}  "
             f"weeks-var-better(RMSE) {wins}-{losses}")

    print("\n(Expect: RECIPIENT improves a lot, ALL a little. If RECIPIENT doesn't clearly "
          "improve, stop and debug the replay data/wiring before running the follow-up "
          "ablations in item 3's protocol - see docs/model_improvement_plan_2026-09-23.md.)")


if __name__ == '__main__':
    main()
