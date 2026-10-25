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
from data.silent_games import played_players  # noqa: E402
from scripts.eval_weekly_model import _metrics, _weighted, STARTABLE_N, _actual_points  # noqa: E402
from scripts import harness_v2  # noqa: E402

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
            build_weekly_projections.clear()          # see evaluate_components_v2: a week's builds are never reused
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


def _one_sided_worse_p(variant_wins, losses):
    """P(Binomial(n, 0.5) <= variant_wins) - a one-sided sign-test p-value
    for 'the variant is worse'. Small when the variant rarely beat base on a
    metric across weeks. Unlike the two-sided _sign_test_p below (which
    flags a lopsided split in EITHER direction), this never vetoes a scope
    the variant significantly IMPROVED - see print_flag_report_v2's use of
    it as the per-position veto in the §1 decision rule."""
    n = variant_wins + losses
    if n == 0:
        return float('nan')
    return sum(math.comb(n, i) for i in range(0, variant_wins + 1)) / (2 ** n)


# Columns of each week's base board kept by --boards-out (2026-10-22): enough to diff two runs player by player.
BOARD_KEEP_COLUMNS = ('Player', 'Pos', 'Team', 'Opponent', 'Availability', 'Games This Season', 'Expected Snap Share', 'Raw Model Proj Pts',
                      'Model Proj Pts', 'passing_attempts', 'passing_yards', 'passing_tds', 'rushing_attempts',
                      'rushing_yards', 'rushing_tds', 'targets', 'receptions', 'receiving_yards', 'receiving_tds')


def evaluate_components_v2(years, weeks, variant_defs, scoring='Full PPR', base_feats=None, weekly=None, boards=None,
                           variant_boards=False):
    """Same shape as evaluate_components (flag_rows[label][scope] = list of
    (metrics_base, metrics_variant) weekly pairs, plus the outlier ledger),
    built on scripts.harness_v2's paired-pool + pairwise_acc/RMSE metrics
    instead of _scope_df/_metrics - see harness_v2's own module docstring for
    exactly what this fixes and why it matters. Used when --harness v2
    (the default). ``weekly`` (a list), when given, gets one dict per
    (variant, scope, year, week) pair with both arms' metrics - the per-week
    record a faster screen is checked against (2026-10-21). ``boards`` (a list), when given, gets each week's base board
    (BOARD_KEEP_COLUMNS) so two runs can be compared player by player (2026-10-22: one season per run came out differently);
    with ``variant_boards`` every variant's board too, under a 'variant' column ('base' for the base), so how far a mechanism moves
    the projections can be read player by player (2026-10-24)."""
    scoring_col = 'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'
    flag_rows = {label: {} for label in variant_defs}
    outliers = []
    base_feats = base_feats if base_feats is not None else DEFAULT_FEATURES

    for year in years:
        stats_df, _team_col, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns:
            print(f"{year}: no weekly data, skipped")
            continue
        for week in weeks:
            # Each build stays in build_weekly_projections' in-memory cache for an hour (Streamlit's TTL cache, ~60 MB a
            # board with its explanations) and no build is reused across weeks: an hour of them was ~10 GB, and two full
            # runs died out of memory on 2026-10-23/24. Clearing per week keeps only this week's (results unchanged).
            build_weekly_projections.clear()
            box_actual = _actual_points(stats_df, name_col, week, scoring_col)
            if box_actual.empty:
                continue
            # the verdict scopes score everyone who PLAYED, a silent game as 0 (harness_v2.BOX_SCORE_SCOPES' comment)
            actual = harness_v2.played_actual(box_actual, played_players(year, week, stats_df, name_col, _team_col))
            base_proj, base_meta = build_weekly_projections(
                year, week, scoring, as_of_week=week, apply_injury=False, features=base_feats)
            if base_proj.empty:
                print(f"{year} w{week} base: nothing ({base_meta.get('reason')})")
                continue
            if boards is not None:
                keep = [c for c in BOARD_KEEP_COLUMNS if c in base_proj.columns]
                frame = base_proj[keep].assign(year=year, week=week)
                boards.append(frame.assign(variant='base') if variant_boards else frame)

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
                if boards is not None and variant_boards:
                    keep = [c for c in BOARD_KEEP_COLUMNS if c in var_proj.columns]
                    boards.append(var_proj[keep].assign(year=year, week=week, variant=label))
                base_idx = base_proj.set_index('Player')['Model Proj Pts']
                var_idx = var_proj.set_index('Player')['Model Proj Pts']
                for scope, pos, startable in harness_v2.SCOPES:
                    pool = harness_v2.scope_pool(base_proj, var_proj, pos, startable,
                                                 actual_players=actual.index)
                    # A per-position START pool (TE=20, QB=24) can't clear a
                    # flat 20-player floor most weeks once it's also required
                    # to be the INTERSECTION of both arms' own top-N and to
                    # have a real actual - that flat floor silently starved
                    # START-TE/START-QB of every week in a smoke test. Scale
                    # the floor to the scope's own size instead; whole-pool
                    # and START-ALL scopes keep the original 20.
                    min_pool = max(8, harness_v2.STARTABLE_N.get(pos, 40) // 2) if (startable and pos) else 20
                    if len(pool) < min_pool:
                        continue
                    mb = harness_v2.metrics(base_idx.reindex(pool), actual.loc[pool])
                    mv = harness_v2.metrics(var_idx.reindex(pool), actual.loc[pool])
                    if mb and mv:
                        flag_rows[label].setdefault(scope, []).append((mb, mv))
                        if weekly is not None:
                            weekly.append({'variant': label, 'scope': scope, 'year': year, 'week': week,
                                           **{f'base_{k}': v for k, v in mb.items()},
                                           **{f'var_{k}': v for k, v in mv.items()}})
                # box-score scopes (harness_v2.BOX_SCORE_SCOPES): the verdict scopes as they were before 2026-10-25
                for scope, pos, startable in harness_v2.BOX_SCORE_SCOPES:
                    pool = harness_v2.scope_pool(base_proj, var_proj, pos, startable, actual_players=box_actual.index)
                    min_pool = max(8, harness_v2.STARTABLE_N.get(pos, 40) // 2) if (startable and pos) else 20
                    if len(pool) < min_pool:
                        continue
                    mb = harness_v2.metrics(base_idx.reindex(pool), box_actual.loc[pool])
                    mv = harness_v2.metrics(var_idx.reindex(pool), box_actual.loc[pool])
                    if mb and mv:
                        flag_rows[label].setdefault(scope, []).append((mb, mv))
                        if weekly is not None:
                            weekly.append({'variant': label, 'scope': scope, 'year': year, 'week': week,
                                           **{f'base_{k}': v for k, v in mb.items()},
                                           **{f'var_{k}': v for k, v in mv.items()}})
                # every-game scopes (harness_v2.EVERY_GAME_SCOPES): no box-score line counts as 0
                for scope, pos, startable in harness_v2.EVERY_GAME_SCOPES:
                    pool = harness_v2.every_game_pool(base_proj, var_proj, pos, startable)
                    min_pool = max(8, harness_v2.STARTABLE_N.get(pos, 40) // 2) if (startable and pos) else 20
                    if len(pool) < min_pool:
                        continue
                    act = harness_v2.every_game_actual(box_actual, pool)
                    mb = harness_v2.metrics(base_idx.reindex(pool), act)
                    mv = harness_v2.metrics(var_idx.reindex(pool), act)
                    if mb and mv:
                        flag_rows[label].setdefault(scope, []).append((mb, mv))
                        if weekly is not None:
                            weekly.append({'variant': label, 'scope': scope, 'year': year, 'week': week,
                                           **{f'base_{k}': v for k, v in mb.items()},
                                           **{f'var_{k}': v for k, v in mv.items()}})
    return flag_rows, outliers


def print_flag_report_v2(label, rows, mode='ablate'):
    """v2 equivalent of print_flag_report: pairwise_acc and RMSE are
    primary, MAE is printed as a secondary column only (see harness_v2's
    module docstring), and a final verdict line applies the §1 decision rule
    (scripts.harness_v2.decide) using START-ALL as the primary scope and the
    four START-position scopes' Holm-corrected significance as the veto."""
    print(f"\n{'=' * 78}\n{label}  [{mode} mode, harness v2]\n{'=' * 78}")
    scope_summary = {}
    for scope, _pos, _startable in harness_v2.SCOPES + harness_v2.BOX_SCORE_SCOPES + harness_v2.EVERY_GAME_SCOPES:
        if scope == harness_v2.BOX_SCORE_SCOPES[0][0] and rows.get(scope):
            print("  -- box-score games only (the verdict scopes before 2026-10-25, silent games left out); not part of the verdict --")
        if scope == harness_v2.EVERY_GAME_SCOPES[0][0] and rows.get(scope):
            print("  -- every game the model expected him to play (no box-score line = 0 points); not part of the verdict --")
        pairs = rows.get(scope)
        if not pairs:
            continue
        mb_list = [p[0] for p in pairs]
        mv_list = [p[1] for p in pairs]
        n_total = sum(m['n'] for m in mb_list)
        rmse_b, rmse_v = _weighted(mb_list, 'rmse'), _weighted(mv_list, 'rmse')
        pw_b, pw_v = _weighted(mb_list, 'pairwise_acc'), _weighted(mv_list, 'pairwise_acc')
        mae_b, mae_v = _weighted(mb_list, 'mae'), _weighted(mv_list, 'mae')
        bias_b, bias_v = _weighted(mb_list, 'bias'), _weighted(mv_list, 'bias')

        rmse_deltas = [mv['rmse'] - mb['rmse'] for mb, mv in pairs]
        rmse_weights = [mb['n'] for mb, mv in pairs]
        rmse_ci = _bootstrap_ci(rmse_deltas, rmse_weights)
        pw_deltas = [mv['pairwise_acc'] - mb['pairwise_acc'] for mb, mv in pairs]
        pw_ci = _bootstrap_ci(pw_deltas, rmse_weights)

        rmse_wins = sum(1 for mb, mv in pairs if mv['rmse'] < mb['rmse'])
        rmse_losses = sum(1 for mb, mv in pairs if mb['rmse'] < mv['rmse'])
        # One-sided: P(a fair coin gives the variant this few wins or fewer)
        # - small only when the variant rarely beat base, i.e. evidence it's
        # WORSE specifically (unlike a two-sided test, this never vetoes a
        # scope the variant significantly IMPROVED).
        p_worse = _one_sided_worse_p(rmse_wins, rmse_losses)

        scope_summary[scope] = {'rmse_ci': rmse_ci, 'pw_ci': pw_ci, 'p_worse': p_worse,
                                'bias_b': bias_b, 'bias_v': bias_v}

        print(f"  {scope:<10} n={n_total:<6} RMSE {rmse_b:.3f}->{rmse_v:.3f} (Δ{rmse_v-rmse_b:+.3f} "
              f"CI[{rmse_ci[0]:+.3f},{rmse_ci[1]:+.3f}])  "
              f"pairwiseAcc {pw_b:.3f}->{pw_v:.3f} (Δ{pw_v-pw_b:+.3f} CI[{pw_ci[0]:+.3f},{pw_ci[1]:+.3f}])  "
              f"bias {bias_b:+.3f}->{bias_v:+.3f}  MAE(secondary) {mae_b:.3f}->{mae_v:.3f}  "
              f"weeks-var-better(RMSE) {rmse_wins}-{rmse_losses}")

    if 'START-ALL' in scope_summary and all(f'START-{p}' in scope_summary for p in ('QB', 'RB', 'WR', 'TE')):
        primary = scope_summary['START-ALL']
        position_p = [scope_summary[f'START-{p}']['p_worse'] for p in ('QB', 'RB', 'WR', 'TE')]
        position_p_holm = harness_v2.holm_correction(position_p)
        bias_growth = abs(primary['bias_v']) - abs(primary['bias_b'])
        verdict = harness_v2.decide(primary['rmse_ci'], primary['pw_ci'], position_p_holm, bias_growth)
        print(f"  VERDICT ({mode}): {verdict}   [bias growth {bias_growth:+.3f}, "
              f"Holm-adjusted per-position p {[round(p, 3) for p in position_p_holm]}]")
        print(f"  RECOMMENDATION: {_recommendation(label, mode, verdict)}")
    else:
        print("  VERDICT: (not enough scopes to apply the §1 rule - need START-ALL and all four "
              "START-position scopes)")


def _recommendation(label, mode, verdict):
    """VERDICT is phrased in terms of 'the variant' (ablate mode: DEFAULT
    minus the flag; add mode: DEFAULT plus the flag) - a SHIP-ELIGIBLE
    ablation means the variant WITHOUT the flag won, i.e. the flag looks
    like it's hurting and should be REMOVED, the opposite of what
    'SHIP-ELIGIBLE' reads as at a glance. Spell the actual action out so
    nobody has to re-derive the sign convention from the mode word alone."""
    if mode == 'ablate':
        return {
            'SHIP-ELIGIBLE': f"'{label}' looks like it's HURTING (removing it measurably improved "
                             f"START-ALL) - candidate to REMOVE from DEFAULT_FEATURES.",
            'REJECT': f"'{label}' is CONFIRMED HELPING (removing it measurably hurt START-ALL) - "
                      f"keep it shipped.",
            'INCONCLUSIVE': f"no significant effect from removing '{label}' at this scope/window - "
                            f"keep it shipped as-is (status quo); a narrower or position-specific "
                            f"test may still find one.",
        }[verdict]
    return {
        'SHIP-ELIGIBLE': f"candidate '{label}' measurably HELPED when added - consider promoting it "
                         f"into DEFAULT_FEATURES.",
        'REJECT': f"candidate '{label}' measurably HURT when added - stays unshipped, rejection "
                  f"confirmed under harness v2.",
        'INCONCLUSIVE': f"no significant effect from adding '{label}' - stays unshipped, no evidence "
                        f"to promote it.",
    }[verdict]


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
    ap.add_argument('--ablate-group', default='',
                    help="comma-separated DEFAULT_FEATURES names removed TOGETHER as one variant (e.g. a confirming run "
                    "for several flags shipped at once, each tested alone). Several groups: 'name=a,b;name2=c,d', each "
                    "its own variant 'group:name' sharing the base build")
    ap.add_argument('--add-group', default='',
                    help="comma-separated MODEL_FEATURES names added TOGETHER as one candidate variant (e.g. two "
                    "overlapping candidates also tested one by one with --add)")
    ap.add_argument('--variant', default='',
                    help="named variants that remove AND add flags at once, e.g. a leaner model: 'lean=-a,-b,+c;lean2=-d,+e' "
                    "('-' a DEFAULT_FEATURES flag removed, '+' a MODEL_FEATURES flag added); each its own variant "
                    "'variant:name', judged as an ablation (is the leaner model at least as good)")
    ap.add_argument('--add', default='', help="comma-separated MODEL_FEATURES names to ADD as an unproven "
                    "candidate (variant = DEFAULT_FEATURES plus this flag) - for a component NOT yet shipped")
    ap.add_argument('--add-features', default='',
                    help="comma-separated MODEL_FEATURES names OR'd into BOTH arms (base AND every "
                    "variant) - a precondition for the test, not the thing being tested. E.g. "
                    "--flags v2_vacancy --add-features v2_historical_injury_replay tests v2_vacancy "
                    "with a time-valid historical injury feed active in both arms, since v2_vacancy "
                    "is otherwise inert with apply_injury=False (every backtest).")
    ap.add_argument('--years', default='2024,2025')
    ap.add_argument('--weeks', default='2-18')
    ap.add_argument('--scoring', default='Full PPR')
    ap.add_argument('--top-outliers', type=int, default=25)
    ap.add_argument('--outliers-csv', default=None, help="optional path to dump the full outlier ledger")
    ap.add_argument('--boards-out', default='auto',
                    help="v2 only: every week's base board (key columns) to this parquet; 'auto' (default) writes "
                    ".sweeps/harness_boards/<flags>_<years>_wk<weeks>.parquet, '' skips it")
    ap.add_argument('--variant-boards', action='store_true',
                    help="with --boards-out, also write every variant's board (a 'variant' column; 'base' for the base)")
    ap.add_argument('--weekly-csv', default='auto',
                    help="v2 only: per-week paired metrics (variant, scope, year, week, both arms) to this CSV; "
                    "'auto' (default) writes .sweeps/harness_weekly/<flags>_<years>_wk<weeks>.csv, '' skips it")
    ap.add_argument('--harness', default='v2', choices=['v1', 'v2'],
                    help="v2 (default): paired START pools + RMSE/pairwise_acc primary, MAE secondary "
                    "(scripts.harness_v2 - fixes the pool-pairing and MAE-bias flaws in v1). "
                    "v1: the original MAE-only harness, kept so old numbers can be reproduced exactly.")
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(',')]
    if '-' in args.weeks:
        lo, hi = args.weeks.split('-')
        weeks = list(range(int(lo), int(hi) + 1))
    else:
        weeks = [int(w) for w in args.weeks.split(',')]

    ablate_flags = [f.strip() for f in args.flags.split(',') if f.strip()]
    add_flags = [f.strip() for f in args.add.split(',') if f.strip()]
    add_features = frozenset(f.strip() for f in getattr(args, 'add_features').split(',') if f.strip())
    ablate_groups = {}
    for chunk in [c.strip() for c in args.ablate_group.split(';') if c.strip()]:
        name, _, flags = chunk.rpartition('=') if '=' in chunk else ('', '', chunk)
        members = [f.strip() for f in flags.split(',') if f.strip()]
        ablate_groups[name.strip() or '+'.join(members)] = members
    group_flags = [f for members in ablate_groups.values() for f in members]
    unknown_group = set(group_flags) - set(DEFAULT_FEATURES)
    if unknown_group:
        raise SystemExit(f"--ablate-group not in DEFAULT_FEATURES: {sorted(unknown_group)}")
    add_group_flags = [f.strip() for f in args.add_group.split(',') if f.strip()]
    bad_add_group = (set(add_group_flags) - set(MODEL_FEATURES)) | (set(add_group_flags) & set(DEFAULT_FEATURES))
    if bad_add_group:
        raise SystemExit(f"--add-group must be MODEL_FEATURES not yet in DEFAULT_FEATURES: {sorted(bad_add_group)}")
    mixed_variants = {}
    for chunk in [c.strip() for c in args.variant.split(';') if c.strip()]:
        name, _, flags = chunk.partition('=')
        items = [f.strip() for f in flags.split(',') if f.strip()]
        removed = {f[1:] for f in items if f.startswith('-')}
        added = {f[1:] for f in items if f.startswith('+')}
        if not name.strip() or len(removed) + len(added) != len(items):
            raise SystemExit(f"--variant needs 'name=-flag,+flag,...': {chunk!r}")
        bad = (removed - set(DEFAULT_FEATURES)) | (added - set(MODEL_FEATURES)) | (added & set(DEFAULT_FEATURES))
        if bad:
            raise SystemExit(f"--variant {name}: removed flags must be in DEFAULT_FEATURES, added ones in MODEL_FEATURES "
                             f"and not shipped: {sorted(bad)}")
        mixed_variants[name.strip()] = (removed, added)
    if not ablate_flags and not add_flags and not group_flags and not add_group_flags and not mixed_variants:
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
    unknown_add_features = add_features - set(MODEL_FEATURES)
    if unknown_add_features:
        raise SystemExit(f"--add-features not in MODEL_FEATURES: {sorted(unknown_add_features)}")
    if add_features and args.harness == 'v1':
        raise SystemExit("--add-features only applies to --harness v2 (v1's evaluate_components "
                         "does not thread it through - it would silently be ignored)")

    base_feats = frozenset(DEFAULT_FEATURES | add_features)
    variant_defs = {}
    modes = {}
    if add_group_flags:
        label = 'addgroup:' + '+'.join(add_group_flags)
        variant_defs[label] = frozenset(DEFAULT_FEATURES | set(add_group_flags) | add_features)
        modes[label] = 'add'
    for name, members in ablate_groups.items():
        label = 'group:' + name
        variant_defs[label] = frozenset((DEFAULT_FEATURES - set(members)) | add_features)
        modes[label] = 'ablate'
    for name, (removed, added) in mixed_variants.items():
        label = 'variant:' + name
        variant_defs[label] = frozenset((DEFAULT_FEATURES - removed) | added | add_features)
        modes[label] = 'ablate'
    for flag in ablate_flags:
        variant_defs[flag] = frozenset((DEFAULT_FEATURES - {flag}) | add_features)
        modes[flag] = 'ablate'
    for flag in add_flags:
        variant_defs[flag] = frozenset(DEFAULT_FEATURES | {flag} | add_features)
        modes[flag] = 'add'

    print(f"years={years} weeks={weeks[0]}-{weeks[-1]} scoring={args.scoring} harness={args.harness}")
    print(f"ablating: {ablate_flags}")
    print(f"adding as candidate: {add_flags}")
    print(f"add-features (forced on in both arms): {sorted(add_features)}")
    print(f"DEFAULT_FEATURES size: {len(DEFAULT_FEATURES)}")

    if args.harness == 'v2':
        weekly, boards = [], ([] if args.boards_out else None)
        flag_rows, outliers = evaluate_components_v2(years, weeks, variant_defs, args.scoring,
                                                      base_feats=base_feats, weekly=weekly, boards=boards,
                                                      variant_boards=args.variant_boards)
        for label in variant_defs:
            print_flag_report_v2(label, flag_rows[label], mode=modes[label])
        weekly_csv = args.weekly_csv
        if weekly_csv == 'auto':
            tag = ''.join(c if c.isalnum() or c in '_+.-' else '_' for c in '+'.join(sorted(variant_defs)))[:120]
            weekly_csv = os.path.join('.sweeps', 'harness_weekly',
                                      f"{tag}_{years[0]}-{years[-1]}_wk{weeks[0]}-{weeks[-1]}.csv")
        if weekly_csv and weekly:
            os.makedirs(os.path.dirname(weekly_csv) or '.', exist_ok=True)
            pd.DataFrame(weekly).to_csv(weekly_csv, index=False)
            print(f"\nper-week paired metrics written to {weekly_csv}")
        boards_out = args.boards_out
        if boards_out == 'auto':
            tag = ''.join(c if c.isalnum() or c in '_+.-' else '_' for c in '+'.join(sorted(variant_defs)))[:120]
            boards_out = os.path.join('.sweeps', 'harness_boards',
                                      f"{tag}_{years[0]}-{years[-1]}_wk{weeks[0]}-{weeks[-1]}.parquet")
        if boards_out and boards:
            os.makedirs(os.path.dirname(boards_out) or '.', exist_ok=True)
            frame = pd.concat(boards, ignore_index=True)
            for col in frame.columns:
                if frame[col].dtype == object:
                    frame[col] = frame[col].astype(str)
            frame.to_parquet(boards_out, index=False)
            print(f"base boards written to {boards_out}")
    else:
        flag_rows, outliers = evaluate_components(years, weeks, variant_defs, args.scoring)
        for label in variant_defs:
            print_flag_report(label, flag_rows[label], mode=modes[label])
    df = print_outliers(outliers, args.top_outliers)
    if args.outliers_csv and df is not None:
        df.to_csv(args.outliers_csv, index=False)
        print(f"\nfull outlier ledger written to {args.outliers_csv}")


if __name__ == '__main__':
    main()
