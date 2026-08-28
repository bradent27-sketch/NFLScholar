"""
Reusable paired A/B isolation harness for ONE OR MORE DEFAULT_FEATURES
components at a time - built for the 2026-08-26 component backtest program
(see docs/weekly_projections_methodology.md). This is what
scripts/eval_weekly_model.py's one-off invocation for v2_pff_alignment_matchup
was generalized into, so every future "does component X help" question is a
one-line rerun instead of a bespoke script.

Same paired-pool discipline as eval_weekly_model.py (every variant scored on
the SAME weeks, on the intersection of player pools every variant projected
that week - a component that just drops hard players never gets undeserved
credit), extended with two things that script does not do:

  1. A bootstrap confidence interval on the pooled weekly MAE delta, plus an
     exact two-sided sign-test p-value on the week win/loss count. A single
     point-estimate delta (e.g. "-0.018 MAE") cannot tell you whether that is
     a real effect or noise at ~17-34 correlated weekly samples - see
     eval_weekly_model.py's own docstring on this. When the 95% CI straddles
     zero, this script says so explicitly instead of implying a clean result.
  2. An outlier ledger: the largest absolute misses (predicted vs actual) in
     the SHIPPED model's (DEFAULT_FEATURES, no ablation) own startable-pool
     projections across the backtest window. This is not a model-comparison
     tool - it is raw material for a human to apply context a backtest
     cannot see (injury, blowout, role change, coaching decision) to specific
     player-weeks. A component test that reports "no measurable effect" is
     ambiguous between "doesn't matter" and "matters occasionally, in ways
     this aggregate MAE view averages away" - the ledger is the place to
     check for the second case.

The shipped-model (DEFAULT_FEATURES) projection is built ONCE per week and
reused across every flag under test in the same run (rather than once per
flag), so testing N flags together costs 1 + N build calls per week, not 2*N.

Usage:
    python scripts/backtest_component.py --flags v2_continuous_roles
    python scripts/backtest_component.py --flags v2_continuous_roles,v2_defense_prior,v2_channel_matchups
    python scripts/backtest_component.py --flags v2_qb_volume_blend --years 2025 --weeks 2-18 --outliers-csv out.csv
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from scripts.eval_weekly_model import _metrics, _weighted, STARTABLE_N, _actual_points  # noqa: E402

SCOPES = [
    ('ALL', None, False),
    ('QB', 'QB', False), ('RB', 'RB', False), ('WR', 'WR', False), ('TE', 'TE', False),
    ('START-QB', 'QB', True), ('START-RB', 'RB', True),
    ('START-WR', 'WR', True), ('START-TE', 'TE', True),
]


def _scope_df(df, pos, startable):
    d = df if pos is None else df[df['Pos'] == pos]
    if startable:
        d = d.nlargest(STARTABLE_N.get(pos, 30), 'Model Proj Pts')
    return d


def _scope_metrics(df, actual, pos, startable):
    d = _scope_df(df, pos, startable)
    pred = pd.Series(d['Model Proj Pts'].to_numpy(), index=d['Player'])
    return _metrics(pred, actual)


def evaluate_components(years, weeks, variant_defs, scoring='Full PPR'):
    """``variant_defs`` maps a label -> the full feature set to test against
    DEFAULT_FEATURES (base). Works for both ablation (label's set =
    DEFAULT_FEATURES minus a shipped flag) and addition (label's set =
    DEFAULT_FEATURES plus an unproven candidate not yet shipped) - the caller
    decides which by what it puts in the set; this function only ever
    compares "base" against "whatever's in variant_defs[label]"."""
    scoring_col = 'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'
    # flag_rows[label][scope] = list of (metrics_base, metrics_variant) weekly pairs
    flag_rows = {label: {} for label in variant_defs}
    outliers = []

    for year in years:
        stats_df, _team_col, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns:
            print(f"{year}: no weekly data, skipped")
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                continue
            base_proj, base_meta = build_weekly_projections(
                year, week, scoring, as_of_week=week, apply_injury=False, features=DEFAULT_FEATURES)
            if base_proj.empty:
                print(f"{year} w{week} base: nothing ({base_meta.get('reason')})")
                continue

            # Outlier ledger: shipped model's own startable cut, independent
            # of any ablation - one flag or seven, this part is identical.
            for pos in ('QB', 'RB', 'WR', 'TE'):
                top = _scope_df(base_proj, pos, True)
                for _, r in top.iterrows():
                    player = r['Player']
                    if player not in actual.index:
                        continue
                    pred_v = float(r['Model Proj Pts'])
                    actual_v = float(actual[player])
                    err = pred_v - actual_v
                    outliers.append({
                        'year': year, 'week': week, 'player': player, 'pos': pos,
                        'pred': pred_v, 'actual': actual_v, 'error': err, 'abs_error': abs(err),
                    })

            for label, variant_feats in variant_defs.items():
                var_proj, var_meta = build_weekly_projections(
                    year, week, scoring, as_of_week=week, apply_injury=False, features=variant_feats)
                if var_proj.empty:
                    print(f"{year} w{week} {label} (variant): nothing ({var_meta.get('reason')})")
                    continue
                pool = sorted(set(base_proj['Player']) & set(var_proj['Player']))
                if len(pool) < 20:
                    continue
                base_sub = base_proj[base_proj['Player'].isin(pool)]
                var_sub = var_proj[var_proj['Player'].isin(pool)]

                for scope, pos, startable in SCOPES:
                    mb = _scope_metrics(base_sub, actual, pos, startable)
                    mv = _scope_metrics(var_sub, actual, pos, startable)
                    if mb and mv:
                        flag_rows[label].setdefault(scope, []).append((mb, mv))
    return flag_rows, outliers


def _bootstrap_ci(deltas, weights, n_boot=3000, seed=0):
    if len(deltas) < 4:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    deltas = np.asarray(deltas, dtype=float)
    weights = np.asarray(weights, dtype=float)
    idx = np.arange(len(deltas))
    means = np.empty(n_boot)
    for b in range(n_boot):
        samp = rng.choice(idx, size=len(idx), replace=True)
        means[b] = np.average(deltas[samp], weights=weights[samp])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _sign_test_p(wins, losses):
    n = wins + losses
    if n == 0:
        return float('nan')
    k = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def print_flag_report(label, rows, mode='ablate'):
    """``mode`` only changes the printed sign-convention note: 'ablate' means
    variant = DEFAULT_FEATURES minus a shipped flag (positive dMAE => the
    flag helps); 'add' means variant = DEFAULT_FEATURES plus an unproven
    candidate (positive dMAE => the candidate HURTS - it made the variant
    worse than the untouched base)."""
    print(f"\n{'=' * 70}\n{label}  [{mode} mode]\n{'=' * 70}")
    for scope, _pos, _startable in SCOPES:
        pairs = rows.get(scope)
        if not pairs:
            continue
        mb_list = [p[0] for p in pairs]
        mv_list = [p[1] for p in pairs]
        n_total = sum(m['n'] for m in mb_list)
        mae_base = _weighted(mb_list, 'mae')
        mae_variant = _weighted(mv_list, 'mae')
        rho_base = _weighted(mb_list, 'rank_corr')
        rho_variant = _weighted(mv_list, 'rank_corr')
        d_mae = mae_variant - mae_base   # sign meaning depends on `mode` - see docstring
        d_rho = rho_variant - rho_base
        wins = sum(1 for mb, mv in pairs if mv['mae'] < mb['mae'])   # variant beat base
        losses = sum(1 for mb, mv in pairs if mb['mae'] < mv['mae'])

        deltas = [mv['mae'] - mb['mae'] for mb, mv in pairs]
        weights = [mb['n'] for mb, mv in pairs]
        lo, hi = _bootstrap_ci(deltas, weights)
        p = _sign_test_p(wins, losses)

        if math.isnan(lo):
            sig = "  [too few weeks for a CI]"
        elif lo > 0 or hi < 0:
            sig = "  [CI excludes 0 -> distinguishable from noise at this sample]"
        else:
            sig = "  [CI includes 0 -> NOT distinguishable from noise at this sample]"

        print(f"  {scope:<10} n={n_total:<6} MAE base {mae_base:.3f} vs variant {mae_variant:.3f}  "
              f"dMAE(var-base) {d_mae:+.3f}  dRho(var-base) {d_rho:+.3f}  "
              f"weeks variant-won {wins}-{losses} (sign-test p={p:.2f})  "
              f"boot95%CI[{lo:+.3f},{hi:+.3f}]{sig}")


def print_outliers(outliers, top_n=25):
    if not outliers:
        print("\nno outlier data collected")
        return
    df = pd.DataFrame(outliers).sort_values('abs_error', ascending=False)
    print(f"\n{'=' * 70}\nTOP {top_n} LARGEST MISSES - shipped model (DEFAULT_FEATURES), "
          f"startable cut, {len(df)} player-weeks scored\n{'=' * 70}")
    print(f"{'year':>5}{'wk':>4}{'pos':>5}  {'player':<24}{'pred':>7}{'actual':>8}{'error':>8}  note")
    for _, r in df.head(top_n).iterrows():
        note = 'OVERPROJECTED (bust)' if r['error'] > 0 else 'UNDERPROJECTED (boom)'
        print(f"{int(r['year']):>5}{int(r['week']):>4}{r['pos']:>5}  {r['player']:<24}"
              f"{r['pred']:>7.1f}{r['actual']:>8.1f}{r['error']:>+8.1f}  {note}")

    print(f"\n-- top 8 per position --")
    for pos in ('QB', 'RB', 'WR', 'TE'):
        pos_df = df[df['pos'] == pos].head(8)
        if pos_df.empty:
            continue
        print(f"\n  {pos}:")
        for _, r in pos_df.iterrows():
            note = 'bust' if r['error'] > 0 else 'boom'
            print(f"    {int(r['year'])} w{int(r['week']):<3}{r['player']:<24}"
                  f"pred {r['pred']:>6.1f}  actual {r['actual']:>6.1f}  err {r['error']:>+6.1f}  ({note})")
    return df


def main():
    from data.weekly_projections import MODEL_FEATURES

    ap = argparse.ArgumentParser()
    ap.add_argument('--flags', default='', help="comma-separated DEFAULT_FEATURES names to ABLATE "
                    "(variant = DEFAULT_FEATURES minus this flag)")
    ap.add_argument('--add', default='', help="comma-separated MODEL_FEATURES names to ADD as an unproven "
                    "candidate (variant = DEFAULT_FEATURES plus this flag) - for a component NOT yet shipped")
    ap.add_argument('--years', default='2024,2025')
    ap.add_argument('--weeks', default='2-18')
    ap.add_argument('--scoring', default='Full PPR')
    ap.add_argument('--top-outliers', type=int, default=25)
    ap.add_argument('--outliers-csv', default=None, help="optional path to dump the full outlier ledger")
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(',')]
    if '-' in args.weeks:
        lo, hi = args.weeks.split('-')
        weeks = list(range(int(lo), int(hi) + 1))
    else:
        weeks = [int(w) for w in args.weeks.split(',')]

    ablate_flags = [f.strip() for f in args.flags.split(',') if f.strip()]
    add_flags = [f.strip() for f in args.add.split(',') if f.strip()]
    if not ablate_flags and not add_flags:
        raise SystemExit("pass --flags (ablate) and/or --add (candidate addition)")
    unknown_ablate = set(ablate_flags) - set(DEFAULT_FEATURES)
    if unknown_ablate:
        raise SystemExit(f"--flags not in DEFAULT_FEATURES: {sorted(unknown_ablate)}")
    unknown_add = set(add_flags) - set(MODEL_FEATURES)
    if unknown_add:
        raise SystemExit(f"--add not in MODEL_FEATURES: {sorted(unknown_add)}")
    already_shipped = set(add_flags) & set(DEFAULT_FEATURES)
    if already_shipped:
        raise SystemExit(f"--add flag(s) already in DEFAULT_FEATURES, use --flags to ablate instead: {sorted(already_shipped)}")

    variant_defs = {}
    modes = {}
    for flag in ablate_flags:
        variant_defs[flag] = frozenset(DEFAULT_FEATURES - {flag})
        modes[flag] = 'ablate'
    for flag in add_flags:
        variant_defs[flag] = frozenset(DEFAULT_FEATURES | {flag})
        modes[flag] = 'add'

    print(f"years={years} weeks={weeks[0]}-{weeks[-1]} scoring={args.scoring}")
    print(f"ablating: {ablate_flags}")
    print(f"adding as candidate: {add_flags}")
    print(f"DEFAULT_FEATURES size: {len(DEFAULT_FEATURES)}")

    flag_rows, outliers = evaluate_components(years, weeks, variant_defs, args.scoring)
    for label in variant_defs:
        print_flag_report(label, flag_rows[label], mode=modes[label])
    df = print_outliers(outliers, args.top_outliers)
    if args.outliers_csv and df is not None:
        df.to_csv(args.outliers_csv, index=False)
        print(f"\nfull outlier ledger written to {args.outliers_csv}")


if __name__ == '__main__':
    main()
