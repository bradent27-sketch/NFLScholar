"""Startable-pool calibration: is the WR/TE "under-projection" real, and would
correcting it help or hurt?

THE FINDING THAT PROMPTED THIS (scripts/analyze_startable_te_bias.py,
2021-2025 wk4-17, shipped DEFAULT_FEATURES, startable cut):

    pos   mean(pred-act)   median(pred-act)
    QB         +0.117           +0.320
    RB         -0.321           +0.700
    WR         -1.363           -0.300
    TE         -1.188           -0.000

Read as means alone that says the model under-projects startable WR by 10.7%
and TE by 11.4%, and the obvious fix is to add ~1.2-1.4 points.

THE MEAN IS THE WRONG STATISTIC HERE, AND THE FIX IS PROBABLY BACKWARDS:

  * Mean and median disagree wildly (TE: -1.19 vs -0.00; RB: -0.32 vs +0.70,
    which flips sign). That gap IS the boom tail - a right-skewed error
    distribution, exactly what the outlier ledger shows (its top misses are
    almost entirely "UNDERPROJECTED (boom)": Beckham 8.5 -> 76.8, Tank Dell
    8.3 -> 59.2, Achane 9.3 -> 54.0).
  * MAE is minimised by predicting the conditional MEDIAN, not the mean. A
    startable TE pool with a median bias of -0.000 is therefore already
    essentially MAE-optimal, and adding +1.2 to every TE would move every
    projection AWAY from the MAE-optimal point. It should make MAE worse.
  * A model that refuses to chase ceiling games is behaving correctly. The
    mean "bias" is the price of not projecting every TE for a 40-burger.

So this script does not assume the correction is good and fit its size. It
tests whether ANY correction helps, and reports the shape of the tradeoff:

  MEAN-matching correction   - zeroes mean bias. Right for EXPECTED VALUE
                               (props, DFS pricing, season totals).
  MEDIAN-matching correction - zeroes median bias. Right for MAE / typical
                               week accuracy.
  MAE-OPTIMAL correction     - grid-searched offset that minimises MAE
                               directly, per tier. The empirical answer,
                               which should land near the median one.

and scores each on: MAE, mean bias, median bias, rank correlation, and
prop-direction accuracy. Cross-validated - offsets are fit on FIT_YEARS and
scored on TEST_YEARS, never on the same rows.

ONE STRUCTURAL POINT THE REPORT MAKES EXPLICIT: a per-tier ADDITIVE offset is
monotone in projected rank, so it CANNOT reorder players within a position.
It therefore cannot improve start/sit decisions within a position at all. It
can only move (a) MAE, (b) cross-position/flex comparisons, and (c) the level
a prop line is judged against. If MAE gets worse, there is nothing left to
justify it.

TWO PASSES, ONE MODEL BUILD:

    python scripts/fit_startable_calibration.py --mode dump --years 2019-2025
    python scripts/fit_startable_calibration.py --mode fit

`dump` is the only expensive part (one build per week, no variant builds at
all). It writes .sweeps/startable_predictions.csv; `fit` reads that and is
instant, so the correction form can be re-explored without re-running a
single projection.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from scripts.eval_weekly_model import STARTABLE_N, _actual_points  # noqa: E402

DUMP_PATH = os.path.join('.sweeps', 'startable_predictions.csv')
POSITIONS = ('QB', 'RB', 'WR', 'TE')

FIT_YEARS = (2019, 2020, 2021, 2022)
TEST_YEARS = (2023, 2024, 2025)

# Rank tiers inside each position's startable pool. The bias is NOT flat - TE
# splits +0.40 / -0.77 / -1.60 across these - so a single per-position number
# would be wrong at both ends simultaneously.
TIERS = {
    'QB': [(1, 6), (7, 12), (13, 24)],
    'RB': [(1, 6), (7, 12), (13, 24), (25, 40)],
    'WR': [(1, 6), (7, 12), (13, 24), (25, 40), (41, 55)],
    'TE': [(1, 3), (4, 6), (7, 12), (13, 20)],
}


def _tier_of(pos, rank):
    for lo, hi in TIERS[pos]:
        if lo <= rank <= hi:
            return f'{lo}-{hi}'
    return None


def do_dump(years, weeks, scoring):
    scoring_col = 'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'
    out = []
    for year in years:
        stats_df, _tc, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns:
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                continue
            build_weekly_projections.clear()
            proj, meta = build_weekly_projections(
                year, week, scoring, as_of_week=week,
                apply_injury=False, features=DEFAULT_FEATURES)
            if proj.empty:
                print(f"{year} w{week}: empty ({meta.get('reason')})", flush=True)
                continue
            for pos in POSITIONS:
                # Rank is the projected order, so it is assigned BEFORE any
                # filtering to players with a recorded actual - dropping a
                # bye/inactive player first would silently promote everyone
                # below him and shift the whole tier structure.
                pool = proj[proj['Pos'] == pos].nlargest(
                    STARTABLE_N.get(pos, 30), 'Model Proj Pts').reset_index(drop=True)
                pool = pool[['Player', 'Model Proj Pts']].copy()
                pool['rank'] = np.arange(1, len(pool) + 1)
                pool = pool[pool['Player'].isin(actual.index)]
                for player, pred, rank in zip(pool['Player'],
                                              pool['Model Proj Pts'],
                                              pool['rank']):
                    out.append({
                        'year': year, 'week': week, 'pos': pos, 'rank': int(rank),
                        'tier': _tier_of(pos, int(rank)), 'player': player,
                        'pred': float(pred), 'actual': float(actual[player]),
                    })
            print(f"{year} w{week} done", flush=True)
    df = pd.DataFrame(out)
    os.makedirs('.sweeps', exist_ok=True)
    df.to_csv(DUMP_PATH, index=False)
    print(f"\nwrote {len(df):,} player-weeks -> {DUMP_PATH}")


def _mae(pred, actual):
    return float(np.mean(np.abs(pred - actual)))


def _best_offset(pred, actual, lo=-4.0, hi=4.0, step=0.05):
    """Offset that directly minimises MAE. Grid, not calculus - the objective
    is piecewise-linear and this is a few thousand cheap evaluations."""
    grid = np.arange(lo, hi + step, step)
    maes = [_mae(pred + g, actual) for g in grid]
    return float(grid[int(np.argmin(maes))])


def _prop_direction(pred, actual, offset, min_move=0.05):
    """If the offset moves a projection off the (uncorrected) line, does the
    actual land on the side it moved toward? The prop read of the change."""
    if abs(offset) < min_move:
        return float('nan'), 0
    moved_up = offset > 0
    live = actual != pred
    right = int(np.sum((actual[live] > pred[live]) == moved_up))
    n = int(np.sum(live))
    return (right / n if n else float('nan')), n


def do_fit():
    if not os.path.exists(DUMP_PATH):
        raise SystemExit(f"no dump at {DUMP_PATH} - run --mode dump first")
    df = pd.read_csv(DUMP_PATH)
    df = df[df['tier'].notna()]
    print(f"loaded {len(df):,} player-weeks  "
          f"years={sorted(df['year'].unique())}  weeks={df['week'].min()}-{df['week'].max()}\n")

    fit = df[df['year'].isin(FIT_YEARS)]
    test = df[df['year'].isin(TEST_YEARS)]
    print(f"FIT  {FIT_YEARS} n={len(fit):,}")
    print(f"TEST {TEST_YEARS} n={len(test):,}\n")
    if fit.empty or test.empty:
        raise SystemExit("need both FIT and TEST years present in the dump")

    bar = '=' * 104
    print(f"{bar}\nSTEP 1 - IS IT SKEW? mean vs median bias per tier (FIT years)\n{bar}")
    print(f"{'pos':<5}{'tier':<9}{'n':>7}{'mean bias':>11}{'median bias':>13}"
          f"{'  skew gap':>11}{'  MAE':>8}")
    for pos in POSITIONS:
        for lo, hi in TIERS[pos]:
            t = f'{lo}-{hi}'
            g = fit[(fit['pos'] == pos) & (fit['tier'] == t)]
            if len(g) < 30:
                continue
            e = g['pred'] - g['actual']
            gap = e.mean() - e.median()
            print(f"{pos:<5}{t:<9}{len(g):>7}{e.mean():>+11.3f}{e.median():>+13.3f}"
                  f"{gap:>+11.3f}{_mae(g['pred'].to_numpy(), g['actual'].to_numpy()):>8.3f}")
    print("\n(a large negative skew gap = the mean is being dragged by boom games;\n"
          " the median is what MAE actually cares about.)")

    # Three candidate corrections, all fit on FIT only.
    offsets = {'mean-match': {}, 'median-match': {}, 'MAE-optimal': {}}
    for pos in POSITIONS:
        for lo, hi in TIERS[pos]:
            t = f'{lo}-{hi}'
            g = fit[(fit['pos'] == pos) & (fit['tier'] == t)]
            if len(g) < 30:
                continue
            e = (g['pred'] - g['actual']).to_numpy()
            key = (pos, t)
            offsets['mean-match'][key] = float(-e.mean())
            offsets['median-match'][key] = float(-np.median(e))
            offsets['MAE-optimal'][key] = _best_offset(
                g['pred'].to_numpy(), g['actual'].to_numpy())

    print(f"\n\n{bar}\nSTEP 2 - FITTED OFFSETS (from FIT years only)\n{bar}")
    print(f"{'pos':<5}{'tier':<9}{'mean-match':>13}{'median-match':>15}{'MAE-optimal':>14}")
    for pos in POSITIONS:
        for lo, hi in TIERS[pos]:
            t = f'{lo}-{hi}'
            k = (pos, t)
            if k not in offsets['mean-match']:
                continue
            print(f"{pos:<5}{t:<9}{offsets['mean-match'][k]:>+13.2f}"
                  f"{offsets['median-match'][k]:>+15.2f}{offsets['MAE-optimal'][k]:>+14.2f}")

    print(f"\n\n{bar}\nSTEP 3 - HELD-OUT RESULT on {TEST_YEARS} (never used for fitting)\n{bar}")
    print(f"{'pos':<5}{'correction':<14}{'n':>7}{'MAE base':>10}{'MAE corr':>10}"
          f"{'dMAE':>9}{'  mean bias':>12}{'  med bias':>11}{'  prop-acc':>11}")
    verdict = {}
    for pos in POSITIONS:
        g = test[test['pos'] == pos]
        if len(g) < 50:
            continue
        pred = g['pred'].to_numpy()
        act = g['actual'].to_numpy()
        base_mae = _mae(pred, act)
        for name, table in offsets.items():
            off = np.array([table.get((pos, t), 0.0) for t in g['tier']], dtype=float)
            corr = pred + off
            e = corr - act
            pacc, _pn = _prop_direction(pred, act, float(np.mean(off)))
            d = _mae(corr, act) - base_mae
            verdict.setdefault(pos, {})[name] = d
            ps = f"{pacc:.3f}" if np.isfinite(pacc) else '   -  '
            print(f"{pos:<5}{name:<14}{len(g):>7}{base_mae:>10.3f}{_mae(corr, act):>10.3f}"
                  f"{d:>+9.3f}{e.mean():>+12.3f}{np.median(e):>+11.3f}{ps:>11}")
        print()

    print(f"{bar}\nSTEP 4 - VERDICT\n{bar}")
    print("A per-tier additive offset is monotone in rank, so it CANNOT reorder "
          "players within\na position - it cannot improve start/sit at all. Its "
          "only possible payoffs are MAE,\ncross-position flex comparisons, and "
          "prop levels. So: if dMAE is positive, there is\nnothing left to "
          "justify shipping it.\n")
    for pos in POSITIONS:
        if pos not in verdict:
            continue
        best = min(verdict[pos], key=lambda k: verdict[pos][k])
        d = verdict[pos][best]
        if d < -0.005:
            call = f"best = {best} ({d:+.3f}) - worth a real backtest"
        elif d < 0.005:
            call = f"no correction helps (best {best} {d:+.3f}) - model already calibrated for MAE"
        else:
            call = f"EVERY correction HURTS (best {best} {d:+.3f}) - do not ship"
        print(f"  {pos:<5} {call}")
    print("\nIf mean-match hurts MAE but you want expected value for props/DFS, the "
          "answer is a\nSEPARATE mean-calibrated output alongside the MAE-optimal "
          "projection - not a\nreplacement for it. The two objectives genuinely "
          "disagree on skewed outcomes.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['dump', 'fit'])
    ap.add_argument('--years', default='2019-2025')
    ap.add_argument('--weeks', default='4-17')
    ap.add_argument('--scoring', default='Full PPR')
    args = ap.parse_args()

    if args.mode == 'dump':
        y0, y1 = (int(x) for x in args.years.split('-'))
        w0, w1 = (int(x) for x in args.weeks.split('-'))
        years = list(range(y0, y1 + 1))
        weeks = list(range(w0, w1 + 1))
        print(f"dumping shipped-model startable predictions: "
              f"{years[0]}-{years[-1]} wk{weeks[0]}-{weeks[-1]}")
        print(f"builds = {len(years) * len(weeks)} (base only, no variants)\n", flush=True)
        do_dump(years, weeks, args.scoring)
    else:
        do_fit()


if __name__ == '__main__':
    main()
