"""FINAL calibration fit for data.weekly_projections (the "v3" calibration).

Supersedes scripts/fit_weekly_calibration.py, which fits one line per position
on 2021-2023 and prints it. That script is not wrong, but it cannot answer the
three questions this one exists to answer, and the answers turned out to
matter:

  1. HOW MUCH DATA?  The old fit used 3 seasons (2021-2023). Weekly stats
     exist back to 2019, so 2 more seasons are available and were never used.

  2. IS THE ONE-SIDED CLAMP COSTING ANYTHING?  The line is applied as
     np.minimum(raw, a + b*raw) - it can only push a projection DOWN, never
     up. Solving a + b*r < r for the shipped constants shows where each
     position's line actually bites:

         QB  above 15.4 pts     RB  above 13.6 pts
         WR  above 29.2 pts     TE  above 25.8 pts

     So WR and TE are, in practice, almost UNCALIBRATED - their lines sit
     above the identity for every realistic projection. That is very likely
     why a tier analysis of the calibrated output
     (scripts/fit_startable_calibration.py) still finds WR 1-6 over-projected
     by +1.68 and WR 41-55 under-projected by -2.20: the top correction is
     too weak to reach them and the bottom correction is blocked outright.

  3. IS ONE LINE PER POSITION THE RIGHT SHAPE?  A single line assumes the
     miscalibration is uniform in projection level. The tier evidence says it
     is not - it flips sign between the top and the tail of the same position.

TWO PHASES, so the expensive part is paid once:

    python scripts/fit_weekly_calibration_v3.py --mode dump --years 2019-2025
    python scripts/fit_weekly_calibration_v3.py --mode fit

`dump` builds UNCALIBRATED whole-pool projections (CALIBRATION_INPUT_FEATURES)
and writes .sweeps/calibration_predictions.csv. `fit` reads that and is
instant, so candidate forms can be compared without re-running a projection.

EVERY CANDIDATE IS FITTED ON FIT_YEARS AND SCORED ON TEST_YEARS. Fitting and
reporting on the same seasons is the mistake the original script's header
warns about, and it is easy to make again when comparing several forms - the
more flexible form always wins in-sample.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_projections import (  # noqa: E402
    build_weekly_projections, CALIBRATION_INPUT_FEATURES, WEEKLY_CALIBRATION)
from data.transforms import load_and_merge_data  # noqa: E402
from scripts.eval_weekly_model import STARTABLE_N  # noqa: E402

DUMP_PATH = os.path.join('.sweeps', 'calibration_predictions.csv')
POSITIONS = ('QB', 'RB', 'WR', 'TE')

# Five seasons to fit, three to test - a wider fit window than the old
# script's three seasons, and the test window is still untouched by it.
FIT_YEARS = (2019, 2020, 2021, 2022)
TEST_YEARS = (2023, 2024, 2025)


def do_dump(years, weeks, scoring):
    scoring_col = 'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'
    rows = []
    for year in years:
        stats_df, _t, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns:
            print(f"{year}: no weekly data, skipped", flush=True)
            continue
        for week in weeks:
            actual = (stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week]
                      .groupby(name_col, observed=True)[scoring_col].sum())
            if actual.empty:
                continue
            build_weekly_projections.clear()
            proj, _meta = build_weekly_projections(
                year, week, scoring, as_of_week=week, apply_injury=False,
                features=CALIBRATION_INPUT_FEATURES)
            if proj.empty:
                continue
            d = proj[['Player', 'Pos', 'Model Proj Pts']].copy()
            d.columns = ['player', 'pos', 'raw']
            d['actual'] = d['player'].map(actual)
            d['year'], d['week'] = year, week
            rows.append(d.dropna(subset=['actual']))
            print(f"{year} w{week} done", flush=True)
    if not rows:
        raise SystemExit("no data")
    out = pd.concat(rows, ignore_index=True)
    os.makedirs('.sweeps', exist_ok=True)
    out.to_csv(DUMP_PATH, index=False)
    print(f"\nwrote {len(out):,} player-weeks -> {DUMP_PATH}")


# --- candidate forms -------------------------------------------------------
# Each returns a function raw -> calibrated, fitted on the frame it is given.

def _fit_line(sub, damp=1.0):
    slope, intercept = np.polyfit(sub['raw'], sub['actual'], 1)
    if damp != 1.0:
        # Half-strength damping, the shipped convention: move the line only
        # part of the way from the identity toward the fitted line.
        slope = 1.0 + damp * (slope - 1.0)
        intercept = damp * intercept
    return float(slope), float(intercept)


def _apply(raw, slope, intercept, one_sided):
    line = intercept + slope * raw
    return np.clip(np.minimum(raw, line) if one_sided else line, 0.0, None)


def _fit_tiered(sub, n_bins=5, damp=1.0):
    """A line per projection-level BIN, not one line per position.

    Bins are quantiles of the projection, so each carries the same number of
    player-weeks rather than the same point range - the top of a position is
    thinly populated and equal-width bins would fit it on almost nothing.
    """
    edges = np.unique(np.quantile(sub['raw'], np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return None
    fits = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (sub['raw'] >= lo) & (sub['raw'] <= hi)
        chunk = sub[m]
        if len(chunk) < 150:
            fits.append((lo, hi, 1.0, 0.0))
            continue
        s, i = _fit_line(chunk, damp)
        fits.append((lo, hi, s, i))
    return edges, fits


def _apply_tiered(raw, fitted, one_sided):
    edges, fits = fitted
    out = np.asarray(raw, dtype=float).copy()
    for lo, hi, s, i in fits:
        m = (out >= lo) & (out <= hi)
        if m.any():
            out[m] = _apply(out[m], s, i, one_sided)
    return np.clip(out, 0.0, None)


def _startable_mask(df):
    """Top-N by PROJECTION within each (year, week, pos) - the pool a lineup
    is actually chosen from."""
    keep = np.zeros(len(df), dtype=bool)
    for (_y, _w, pos), chunk in df.groupby(['year', 'week', 'pos'], observed=True):
        n = STARTABLE_N.get(pos, 30)
        idx = chunk.nlargest(min(n, len(chunk)), 'raw').index
        keep[df.index.get_indexer(idx)] = True
    return keep


def _score(name, pred, act, start_mask):
    mae = float(np.mean(np.abs(pred - act)))
    smae = float(np.mean(np.abs(pred[start_mask] - act[start_mask]))) if start_mask.any() else float('nan')
    return {'name': name, 'mae': mae, 'start_mae': smae,
            'bias': float(np.mean(pred - act)),
            'start_bias': float(np.mean(pred[start_mask] - act[start_mask])) if start_mask.any() else float('nan')}


def do_fit(damp):
    if not os.path.exists(DUMP_PATH):
        raise SystemExit(f"no dump at {DUMP_PATH} - run --mode dump first")
    df = pd.read_csv(DUMP_PATH)
    fit_df = df[df['year'].isin(FIT_YEARS)].reset_index(drop=True)
    test_df = df[df['year'].isin(TEST_YEARS)].reset_index(drop=True)
    print(f"loaded {len(df):,} player-weeks  years={sorted(df['year'].unique())}")
    print(f"FIT  {FIT_YEARS} n={len(fit_df):,}")
    print(f"TEST {TEST_YEARS} n={len(test_df):,}   (damp={damp})\n")
    if fit_df.empty or test_df.empty:
        raise SystemExit("need both windows present in the dump")

    bar = '=' * 100
    print(f"{bar}\nHELD-OUT COMPARISON - every form fitted on FIT years, scored on TEST years\n{bar}")
    print("MAE = whole pool. START-MAE = top-N by projection, the pool a lineup comes from.\n")

    recommended = {}
    for pos in POSITIONS:
        f = fit_df[fit_df['pos'] == pos]
        t = test_df[test_df['pos'] == pos].reset_index(drop=True)
        if len(f) < 400 or len(t) < 200:
            continue
        raw_t = t['raw'].to_numpy(dtype=float)
        act_t = t['actual'].to_numpy(dtype=float)
        smask = _startable_mask(t)

        cands = [_score('uncalibrated', raw_t, act_t, smask)]

        # the shipped constants, as they stand today
        if pos in WEEKLY_CALIBRATION:
            s, i = WEEKLY_CALIBRATION[pos]
            cands.append(_score(f'SHIPPED ({s:.3f},{i:.3f}) 1-sided',
                                _apply(raw_t, s, i, True), act_t, smask))

        forms = {}
        for lbl, dmp in (('full', 1.0), ('half', 0.5)):
            s, i = _fit_line(f, dmp)
            forms[f'line {lbl}-strength'] = (s, i)
            for side_lbl, one in (('1-sided', True), ('2-sided', False)):
                cands.append(_score(f'line {lbl} {side_lbl} ({s:.3f},{i:.3f})',
                                    _apply(raw_t, s, i, one), act_t, smask))
        tier = _fit_tiered(f, damp=damp)
        if tier is not None:
            for side_lbl, one in (('1-sided', True), ('2-sided', False)):
                cands.append(_score(f'tiered x{len(tier[1])} {side_lbl}',
                                    _apply_tiered(raw_t, tier, one), act_t, smask))

        base = cands[0]
        print(f"--- {pos}   (TEST n={len(t):,}, startable n={int(smask.sum()):,})")
        print(f"{'form':<34}{'MAE':>8}{'dMAE':>9}{'START-MAE':>11}{'dSTART':>9}{'bias':>9}{'s-bias':>9}")
        best = None
        for c in cands:
            d = c['mae'] - base['mae']
            ds = c['start_mae'] - base['start_mae']
            print(f"{c['name']:<34}{c['mae']:>8.3f}{d:>+9.3f}{c['start_mae']:>11.3f}"
                  f"{ds:>+9.3f}{c['bias']:>+9.3f}{c['start_bias']:>+9.3f}")
            if c['name'] != 'uncalibrated' and (best is None or c['start_mae'] < best['start_mae']):
                best = c
        recommended[pos] = (best, forms)
        print()

    print(f"{bar}\nREAD\n{bar}")
    print("Ranked on START-MAE, not whole-pool MAE: the whole pool is dominated by\n"
          "bench players whose projection nobody acts on, and a form can win there\n"
          "while losing where lineups are actually set.\n")
    for pos, (best, _forms) in recommended.items():
        print(f"  {pos:<4} best held-out form: {best['name']}   "
              f"START-MAE {best['start_mae']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['dump', 'fit'])
    ap.add_argument('--years', default='2019-2025')
    ap.add_argument('--weeks', default='5-17')
    ap.add_argument('--scoring', default='Full PPR')
    ap.add_argument('--damp', type=float, default=1.0,
                    help='damping for the tiered form (1.0 = full fitted strength)')
    args = ap.parse_args()
    if args.mode == 'dump':
        y0, y1 = (int(x) for x in args.years.split('-'))
        w0, w1 = (int(x) for x in args.weeks.split('-'))
        years, weeks = list(range(y0, y1 + 1)), list(range(w0, w1 + 1))
        print(f"dumping UNCALIBRATED projections {years[0]}-{years[-1]} "
              f"wk{weeks[0]}-{weeks[-1]}  ({len(years) * len(weeks)} builds)\n", flush=True)
        do_dump(years, weeks, args.scoring)
    else:
        do_fit(args.damp)


if __name__ == '__main__':
    main()
