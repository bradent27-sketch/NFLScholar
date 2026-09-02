"""FINAL VERDICT on defensive-coaching-aware defense priors.

Five prior tests found no shippable signal, but all of them answered the
question "did THIS config beat baseline on whole-pool MAE?". That is the
wrong question twice over:

  1. Whole-pool MAE is not the objective. Start/sit, DFS and props are
     decided on the STARTABLE subset and on the ORDERING within it. A
     change can be a whole-pool wash and still be worth shipping if it
     makes the top-20 TEs measurably better ranked.
  2. A null result on one config does not rule out a mechanism. "We tried
     seven tables and none was significant" is not the same statement as
     "no table could ever be worth more than X".

This script is built to close the question rather than add a sixth null.
Four legs, each answering something the earlier runs could not:

  LEG A - CHANNEL BUDGET (the bound that makes this decisive).
      Ablate v2_defense_prior entirely: how much is the WHOLE prior-season
      defense channel worth on startables? The coaching cohort idea does
      not add information - it only re-weights how fast that one channel
      decays, and only for the ~40% of team-seasons that changed staff. So
      whatever the full channel is worth is a hard ceiling on the coaching
      refinement of it, and realistically it can capture only a fraction.
      If the whole channel is worth 0.05 startable MAE, no cohort table can
      be worth 0.10, and the question is closed by arithmetic.

  LEG B - MECHANISM ORACLE (cheating ceiling).
      Score every cohort table in the grid, then pick the winner PER
      POSITION using the very data we scored on. That is deliberate
      overfitting with perfect hindsight - it cannot be beaten by any
      honest fitting procedure. It is the best this mechanism could EVER
      do. A small oracle number is a permanent answer; only a large one
      would justify further work.

  LEG C - HONEST HOLDOUT.
      Pick the table on FIT_YEARS, freeze it, score on TEST_YEARS which
      were never looked at. The gap between Leg B and Leg C is the
      overfitting tax, and quantifies how much of every earlier "promising"
      result was selection.

  LEG D - POWER, and DECISION metrics.
      MDE: the smallest startable dMAE this design could reliably detect at
      80% power. Converts "not significant" into "we can rule out anything
      bigger than X", which is the actually useful statement going forward.
      Decision metrics, because MAE is not what a lineup or a prop cares
      about:
        - Spearman rank correlation within the startable pool (ordering).
        - Start/sit flip accuracy: of the player pairs whose ORDER this
          change flips, how often does the new order match reality? 50% is
          a coin flip and worth nothing regardless of MAE.
        - Prop-direction accuracy: treating the base projection as the
          line, when the variant moves a player up/down, does the actual
          land on that side?

Scored on weeks 2-10 by design: that is where the prior-season defense
carries real weight (by week ~12 it is blended down to noise), so it is
the only window where this mechanism has leverage at all. If it is dead
here it is dead everywhere.

    python scripts/coaching_final_verdict.py --years 2016-2025 --weeks 2-10
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

POSITIONS = ('QB', 'RB', 'WR', 'TE')
COACH_FLAG = 'v2_coaching_aware_defense_prior'

# Split for Leg C. FIT years choose the table; TEST years are never consulted
# until the table is frozen.
FIT_YEARS = (2016, 2017, 2018, 2019, 2020, 2021)
TEST_YEARS = (2022, 2023, 2024, 2025)

# The cohort-table grid. Each entry is the POS_COHORT_PRIOR_GAMES env string
# consumed by data.coaching_changes._pos_cohort_prior_games(). Positions are
# scored independently, so ONE build per config covers all four at once.
#
# Directions come from the year-over-year persistence work: a defense that
# reset its staff should get a SHORTER prior leash (trust last season less),
# one that only promoted a coordinator under a retained HC arguably a LONGER
# one. The grid spans "barely any taper" to "prior nearly discarded on
# reset", so a real effect of any plausible size has a config that would
# catch it.
def _tbl(none, dc_only, hc_only, both):
    return ','.join(
        f'{p}:{k}={v}'
        for p in POSITIONS
        for k, v in (('none', none), ('dc_only', dc_only),
                     ('hc_only', hc_only), ('both', both)))


GRID = {
    'flat (control)':       _tbl(12, 12, 12, 12),
    'reset-mild':           _tbl(12, 12, 8, 8),
    'reset-strong':         _tbl(12, 12, 4, 4),
    'reset-extreme':        _tbl(12, 12, 2, 2),
    'stable-long':          _tbl(16, 24, 12, 12),
    'both-directions':      _tbl(16, 24, 4, 6),
}


def _scope_startable(df, pos):
    d = df[df['Pos'] == pos]
    return d.nlargest(STARTABLE_N.get(pos, 30), 'Model Proj Pts')


def _bootstrap_ci(deltas, weights, n_boot=4000, seed=0):
    if len(deltas) < 4:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    d, w = np.asarray(deltas, float), np.asarray(weights, float)
    idx = np.arange(len(d))
    means = np.array([np.average(d[s], weights=w[s])
                      for s in (rng.choice(idx, len(idx), replace=True)
                                for _ in range(n_boot))])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _sign_test_p(wins, losses):
    n = wins + losses
    if n == 0:
        return float('nan')
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def _mde(deltas, power=0.80, alpha=0.05):
    """Minimum detectable effect: the smallest true weekly-mean dMAE this
    design would catch at `power`, given the observed week-to-week spread.

    Two-sided z approximation - the week count here (~90) is comfortably
    into the regime where that is fine, and the point of this number is its
    order of magnitude, not its third decimal.
    """
    d = np.asarray(deltas, float)
    d = d[np.isfinite(d)]
    if len(d) < 4:
        return float('nan')
    return float((1.959964 + 0.8416212) * d.std(ddof=1) / math.sqrt(len(d)))


def _rank_corr(pred, actual):
    """Spearman between a projection and reality over the same players."""
    common = [p for p in pred.index if p in actual.index]
    if len(common) < 5:
        return float('nan')
    a = pd.Series(pred).loc[common].rank()
    b = pd.Series(actual).loc[common].rank()
    if a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def _flip_accuracy(base_pred, var_pred, actual):
    """Of the player PAIRS whose relative order this change flips, how often
    is the new order the one reality agreed with?

    This is the start/sit question stated exactly: a change only matters if
    it reorders somebody, and it only helps if the reordering is right more
    often than wrong. 50% means the change is churn, whatever it does to MAE.
    """
    common = [p for p in base_pred.index if p in var_pred.index and p in actual.index]
    if len(common) < 3:
        return 0, 0
    b = pd.Series(base_pred).loc[common].to_numpy()
    v = pd.Series(var_pred).loc[common].to_numpy()
    a = pd.Series(actual).loc[common].to_numpy()
    right = wrong = 0
    n = len(common)
    for i in range(n):
        for j in range(i + 1, n):
            b_order = b[i] - b[j]
            v_order = v[i] - v[j]
            if b_order == 0 or v_order == 0:
                continue
            if (b_order > 0) == (v_order > 0):
                continue                      # order unchanged - not a flip
            a_order = a[i] - a[j]
            if a_order == 0:
                continue
            if (v_order > 0) == (a_order > 0):
                right += 1
            else:
                wrong += 1
    return right, wrong


def _prop_direction(base_pred, var_pred, actual, min_move=0.05):
    """Treat the BASE projection as the posted line. When the variant moves a
    player off it, does the actual land on the side it moved toward?

    This is the prop-betting read of the same change, and it is scored only
    on players the change actually moved by more than `min_move` points -
    a move too small to act on should not be counted as a call.
    """
    common = [p for p in base_pred.index if p in var_pred.index and p in actual.index]
    right = wrong = 0
    for p in common:
        line = float(base_pred[p])
        move = float(var_pred[p]) - line
        if abs(move) < min_move:
            continue
        act = float(actual[p])
        if act == line:
            continue
        if (move > 0) == (act > line):
            right += 1
        else:
            wrong += 1
    return right, wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2016-2025')
    ap.add_argument('--weeks', default='2-10')
    ap.add_argument('--scoring', default='Full PPR')
    ap.add_argument('--skip-ablation', action='store_true',
                    help='skip Leg A (the v2_defense_prior channel-budget ablation)')
    args = ap.parse_args()

    y0, y1 = (int(x) for x in args.years.split('-'))
    years = list(range(y0, y1 + 1))
    w0, w1 = (int(x) for x in args.weeks.split('-'))
    weeks = list(range(w0, w1 + 1))
    scoring_col = 'fantasy_points_ppr' if args.scoring != 'Standard' else 'fantasy_points'

    # Every config that needs a build. Base is DEFAULT_FEATURES as shipped.
    configs = {}
    if not args.skip_ablation:
        configs['ABLATE defense_prior'] = (
            frozenset(DEFAULT_FEATURES - {'v2_defense_prior'}), None)
    for label, table in GRID.items():
        configs[label] = (frozenset(DEFAULT_FEATURES | {COACH_FLAG}), table)

    print(f"years={years[0]}-{years[-1]} weeks={weeks[0]}-{weeks[-1]} "
          f"scoring={args.scoring}")
    print(f"configs={len(configs)} (+1 base) -> {(len(configs) + 1) * len(years) * len(weeks)} builds")
    print(f"FIT={FIT_YEARS}  TEST={TEST_YEARS}\n", flush=True)

    # rows[(label, pos)] = list of per-week dicts
    rows = {}

    for year in years:
        stats_df, _tc, name_col, _ = load_and_merge_data(year, args.scoring)
        if 'week' not in stats_df.columns:
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                continue

            os.environ.pop('POS_COHORT_PRIOR_GAMES', None)
            build_weekly_projections.clear()
            base, bmeta = build_weekly_projections(
                year, week, args.scoring, as_of_week=week,
                apply_injury=False, features=DEFAULT_FEATURES)
            if base.empty:
                print(f"{year} w{week}: base empty ({bmeta.get('reason')})", flush=True)
                continue

            built = {}
            for label, (feats, table) in configs.items():
                if table is None:
                    os.environ.pop('POS_COHORT_PRIOR_GAMES', None)
                else:
                    os.environ['POS_COHORT_PRIOR_GAMES'] = table
                build_weekly_projections.clear()
                var, _vm = build_weekly_projections(
                    year, week, args.scoring, as_of_week=week,
                    apply_injury=False, features=feats)
                if not var.empty:
                    built[label] = var
            os.environ.pop('POS_COHORT_PRIOR_GAMES', None)
            build_weekly_projections.clear()

            for label, var in built.items():
                # Paired pool: score both sides on the same players, so a
                # variant that merely drops hard players earns nothing.
                pool = sorted(set(base['Player']) & set(var['Player']))
                if len(pool) < 20:
                    continue
                b_all = base[base['Player'].isin(pool)]
                v_all = var[var['Player'].isin(pool)]
                for pos in POSITIONS:
                    # Startable set is defined by the BASE model's own top-N,
                    # held fixed across both sides. Letting each side pick its
                    # own top-N would compare two different player sets and
                    # call the difference an improvement.
                    b_pos = _scope_startable(b_all, pos)
                    names = [p for p in b_pos['Player'] if p in actual.index]
                    if len(names) < 5:
                        continue
                    bp = pd.Series(b_pos.set_index('Player')['Model Proj Pts']).reindex(names)
                    vp = pd.Series(v_all.set_index('Player')['Model Proj Pts']).reindex(names)
                    vp = vp.fillna(bp)
                    act = actual.reindex(names)

                    mb = _metrics(bp, act)
                    mv = _metrics(vp, act)
                    if not (mb and mv):
                        continue
                    fr, fw = _flip_accuracy(bp, vp, act)
                    pr, pw = _prop_direction(bp, vp, act)
                    rows.setdefault((label, pos), []).append({
                        'year': year, 'week': week, 'n': mb['n'],
                        'mae_b': mb['mae'], 'mae_v': mv['mae'],
                        'd': mv['mae'] - mb['mae'],
                        'rho_b': _rank_corr(bp, act), 'rho_v': _rank_corr(vp, act),
                        'flip_right': fr, 'flip_wrong': fw,
                        'prop_right': pr, 'prop_wrong': pw,
                    })
            print(f"{year} w{week} done ({len(built)} configs)", flush=True)

    _report(rows, configs)


def _agg(recs):
    """One config x position, over a set of weeks."""
    if not recs:
        return None
    n = sum(r['n'] for r in recs)
    w = [r['n'] for r in recs]
    d = [r['d'] for r in recs]
    mae_b = np.average([r['mae_b'] for r in recs], weights=w)
    mae_v = np.average([r['mae_v'] for r in recs], weights=w)
    lo, hi = _bootstrap_ci(d, w)
    wins = sum(1 for r in recs if r['d'] < 0)
    losses = sum(1 for r in recs if r['d'] > 0)
    rb = [r['rho_b'] for r in recs if np.isfinite(r['rho_b'])]
    rv = [r['rho_v'] for r in recs if np.isfinite(r['rho_v'])]
    fr = sum(r['flip_right'] for r in recs)
    fw = sum(r['flip_wrong'] for r in recs)
    pr = sum(r['prop_right'] for r in recs)
    pw = sum(r['prop_wrong'] for r in recs)
    return {
        'n': n, 'weeks': len(recs), 'mae_b': mae_b, 'mae_v': mae_v,
        'd': mae_v - mae_b, 'lo': lo, 'hi': hi,
        'sig': np.isfinite(lo) and (lo > 0 or hi < 0),
        'wins': wins, 'losses': losses, 'p': _sign_test_p(wins, losses),
        'drho': (np.mean(rv) - np.mean(rb)) if (rb and rv) else float('nan'),
        'flip_right': fr, 'flip_wrong': fw,
        'flip_acc': fr / (fr + fw) if (fr + fw) else float('nan'),
        'prop_acc': pr / (pr + pw) if (pr + pw) else float('nan'),
        'prop_n': pr + pw,
        'mde': _mde(d),
    }


def _report(rows, configs):
    labels = list(configs)
    line = '=' * 108

    print(f"\n\n{line}\nLEG A - CHANNEL BUDGET: what is the WHOLE prior-season "
          f"defense channel worth on startables?\n{line}")
    print("Removing v2_defense_prior deletes the entire channel the coaching idea "
          "re-weights.\nWhatever that costs is a hard ceiling on any refinement of it.\n")
    print(f"{'pos':<6}{'n':>7}{'MAE base':>10}{'MAE no-prior':>14}{'dMAE':>9}"
          f"{'  95% CI':>20}")
    for pos in POSITIONS:
        a = _agg(rows.get(('ABLATE defense_prior', pos), []))
        if not a:
            continue
        print(f"{pos:<6}{a['n']:>7}{a['mae_b']:>10.3f}{a['mae_v']:>14.3f}"
              f"{a['d']:>+9.3f}{f'  [{a['lo']:+.3f},{a['hi']:+.3f}]':>20}"
              f"{' *' if a['sig'] else ''}")
    print("\n(positive dMAE = removing the prior-season channel HURTS, i.e. that is "
          "the channel's\n value. The coaching adjustment competes for a slice of "
          "this number, never more.)")

    grid_labels = [l for l in labels if l != 'ABLATE defense_prior']

    print(f"\n\n{line}\nFULL GRID - every cohort table, startable subsets only "
          f"(the objective that matters)\n{line}")
    for pos in POSITIONS:
        print(f"\n--- START-{pos} ---")
        print(f"{'config':<20}{'n':>7}{'dMAE':>9}{'  95% CI':>20}{'  w-l':>9}"
              f"{'  dRho':>9}{'  flip-acc':>11}{'  prop-acc':>11}")
        for label in grid_labels:
            a = _agg(rows.get((label, pos), []))
            if not a:
                continue
            fa = f"{a['flip_acc']:.3f}" if np.isfinite(a['flip_acc']) else '   -  '
            pa = f"{a['prop_acc']:.3f}" if np.isfinite(a['prop_acc']) else '   -  '
            print(f"{label:<20}{a['n']:>7}{a['d']:>+9.3f}"
                  f"{f'  [{a['lo']:+.3f},{a['hi']:+.3f}]':>20}"
                  f"{f'  {a['wins']}-{a['losses']}':>9}{a['drho']:>+9.4f}"
                  f"{fa:>11}{pa:>11}{' *' if a['sig'] else ''}")

    print(f"\n\n{line}\nLEG B - MECHANISM ORACLE: best config per position, chosen "
          f"WITH HINDSIGHT on this same data\n{line}")
    print("Deliberate overfitting. No honest procedure can beat it, so it is the "
          "ceiling of\nthis mechanism. A small number here closes the question "
          "permanently.\n")
    print(f"{'pos':<6}{'oracle config':<20}{'dMAE':>9}{'  95% CI':>20}"
          f"{'  flip-acc':>11}")
    for pos in POSITIONS:
        best, best_a = None, None
        for label in grid_labels:
            a = _agg(rows.get((label, pos), []))
            if a and (best_a is None or a['d'] < best_a['d']):
                best, best_a = label, a
        if best_a is None:
            continue
        fa = f"{best_a['flip_acc']:.3f}" if np.isfinite(best_a['flip_acc']) else '   -  '
        print(f"{pos:<6}{best:<20}{best_a['d']:>+9.3f}"
              f"{f'  [{best_a['lo']:+.3f},{best_a['hi']:+.3f}]':>20}{fa:>11}")

    print(f"\n\n{line}\nLEG C - HONEST HOLDOUT: table chosen on {FIT_YEARS[0]}-"
          f"{FIT_YEARS[-1]}, frozen, scored on {TEST_YEARS[0]}-{TEST_YEARS[-1]}\n{line}")
    print(f"{'pos':<6}{'picked on FIT':<20}{'FIT dMAE':>10}{'TEST dMAE':>11}"
          f"{'  TEST 95% CI':>22}{'  flip-acc':>11}")
    for pos in POSITIONS:
        best, best_fit = None, None
        for label in grid_labels:
            recs = [r for r in rows.get((label, pos), []) if r['year'] in FIT_YEARS]
            a = _agg(recs)
            if a and (best_fit is None or a['d'] < best_fit['d']):
                best, best_fit = label, a
        if best is None:
            continue
        test = _agg([r for r in rows.get((best, pos), []) if r['year'] in TEST_YEARS])
        if not test:
            continue
        fa = f"{test['flip_acc']:.3f}" if np.isfinite(test['flip_acc']) else '   -  '
        print(f"{pos:<6}{best:<20}{best_fit['d']:>+10.3f}{test['d']:>+11.3f}"
              f"{f'  [{test['lo']:+.3f},{test['hi']:+.3f}]':>22}{fa:>11}"
              f"{' *' if test['sig'] else ''}")
    print("\n(FIT minus TEST is the overfitting tax - how much of a 'promising' "
          "result was selection.)")

    print(f"\n\n{line}\nLEG D - POWER: what could this design have detected?\n{line}")
    print("MDE = smallest true startable dMAE detectable at 80% power, 5% two-sided.\n"
          "Any effect smaller than this is beyond what more runs of this design can "
          "resolve.\n")
    print(f"{'pos':<6}{'weeks':>7}{'MDE (pts)':>12}{'  as % of startable MAE':>26}")
    for pos in POSITIONS:
        a = _agg(rows.get((grid_labels[0], pos), []))
        if not a or not np.isfinite(a['mde']):
            continue
        pct = 100.0 * a['mde'] / a['mae_b'] if a['mae_b'] else float('nan')
        print(f"{pos:<6}{a['weeks']:>7}{a['mde']:>12.4f}{pct:>25.2f}%")

    print("\n\nREAD: ship only if a config is negative AND CI-excludes-0 in LEG C "
          "(honest holdout),\nwith flip-acc > 0.50. LEG B being small means no "
          "future tuning can rescue it.\nLEG A being small means the whole channel "
          "was never worth much to begin with.")


if __name__ == '__main__':
    main()
