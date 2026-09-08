"""Season-phase calibration research (2026-09-07).

The shipped WEEKLY_CALIBRATION is ONE (slope, intercept) per position, fitted
on weeks 5-17. The cold-start signed-bias check (wk1-2, 2022-25) found every
position projecting HIGH there - startable TE +2.85 pts, 8/8 weeks - while
the wk5-17 hold-out has the model projecting LOW. A single line cannot serve
a regime that flips sign. This script:

  --mode dump      build raw-vs-actual for EVERY week 1-18, points AND the
                   projected stats, with the shipped CALIBRATION_INPUT_FEATURES
                   (+ v2_historical_ourlads so a historical cold start resolves).
  --mode analyze   per-(pos, week) and per-(pos, phase-bucket) line fits +
                   signed bias, so the week trajectory is visible; then a
                   verdict on (a) how to bucket the season and (b) whether a
                   per-STAT calibration buys anything a points line doesn't.
  --mode emit      fit the SHIPPING scheme - QB/RB one line, WR/TE a 2-bucket
                   split (weeks 1-4 "cold" vs 5-18 "rest", the bake-off's
                   cold4_rest) - and print the WEEKLY_CALIBRATION /
                   WEEKLY_CALIBRATION_BY_BUCKET dicts ready to paste, plus a
                   held-out 2025 startable MAE/|bias| check of the new scheme
                   vs the shipped single line.

    python scripts/fit_seasonal_calibration.py --mode dump --years 2021-2025 --weeks 1-18
    python scripts/fit_seasonal_calibration.py --mode analyze
    python scripts/fit_seasonal_calibration.py --mode emit
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_projections import (  # noqa: E402
    build_weekly_projections, CALIBRATION_INPUT_FEATURES, WEEKLY_CALIBRATION)
from data.transforms import load_and_merge_data  # noqa: E402
from scripts.eval_weekly_model import STARTABLE_N  # noqa: E402

DUMP_PATH = os.path.join('.sweeps', 'seasonal_calibration_predictions.csv')
POSITIONS = ('QB', 'RB', 'WR', 'TE')
FIT_YEARS = (2021, 2022, 2023)
TEST_YEARS = (2024, 2025)
FEATS = frozenset(CALIBRATION_INPUT_FEATURES | {'v2_historical_ourlads'})

# Projected stats to also dump raw-vs-actual for, per position family.
STAT_COLS = ('targets', 'receptions', 'receiving_yards', 'receiving_tds',
             'carries', 'rushing_yards', 'rushing_tds',
             'attempts', 'passing_yards', 'passing_tds', 'passing_interceptions')

# Candidate season-phase bucketings to score against per-week and single-line.
BUCKETS = {
    'single':      {w: 'all' for w in range(1, 19)},
    'cold_rest':   {w: ('cold' if w <= 2 else 'rest') for w in range(1, 19)},
    'cold4_rest':  {w: ('cold' if w <= 4 else 'rest') for w in range(1, 19)},
    'early_mid_late': {w: ('early' if w <= 4 else 'late' if w >= 15 else 'mid')
                       for w in range(1, 19)},
    'e2_mid_late': {w: ('early' if w <= 2 else 'late' if w >= 15 else 'mid')
                    for w in range(1, 19)},
}


def do_dump(years, weeks, scoring):
    scoring_col = 'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'
    rows = []
    for year in years:
        stats_df, _t, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns:
            print(f"{year}: no weekly data, skipped", flush=True)
            continue
        for week in weeks:
            wk = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week]
            actual = wk.groupby(name_col, observed=True)[scoring_col].sum()
            if actual.empty:
                continue
            sa = wk.groupby(name_col, observed=True)[
                [s for s in STAT_COLS if s in wk.columns]].sum()
            build_weekly_projections.clear()
            proj, _meta = build_weekly_projections(
                year, week, scoring, as_of_week=week, apply_injury=False, features=FEATS)
            build_weekly_projections.clear()
            if proj.empty:
                print(f"{year} w{week} empty", flush=True)
                continue
            d = proj[['Player', 'Pos', 'Model Proj Pts']].copy()
            d.columns = ['player', 'pos', 'raw']
            d['actual'] = d['player'].map(actual)
            for s in STAT_COLS:
                if s in proj.columns:
                    d[f'p_{s}'] = pd.to_numeric(proj[s], errors='coerce').to_numpy()
                    d[f'a_{s}'] = d['player'].map(sa[s]) if s in sa.columns else np.nan
            d['year'], d['week'] = year, week
            rows.append(d.dropna(subset=['actual']))
            print(f"{year} w{week} done  ({len(d)} rows)", flush=True)
    if not rows:
        raise SystemExit("no data")
    out = pd.concat(rows, ignore_index=True)
    os.makedirs('.sweeps', exist_ok=True)
    out.to_csv(DUMP_PATH, index=False)
    print(f"\nwrote {len(out):,} player-weeks -> {DUMP_PATH}")


def _fit_half(raw, actual):
    """Half-strength line, the shipped convention: identity + 0.5*(fit-identity)."""
    if len(raw) < 30:
        return None
    s, i = np.polyfit(raw, actual, 1)
    return 1.0 + 0.5 * (s - 1.0), 0.5 * i


def _startable_mask(df):
    keep = np.zeros(len(df), dtype=bool)
    for (_y, _w, pos), chunk in df.groupby(['year', 'week', 'pos'], observed=True):
        n = STARTABLE_N.get(pos, 30)
        idx = chunk.nlargest(n, 'raw').index
        keep[df.index.get_indexer(idx)] = True
    return keep


def do_analyze():
    df = pd.read_csv(DUMP_PATH)
    df['start'] = _startable_mask(df)
    fit = df[df['year'].isin(FIT_YEARS)].copy()
    test = df[df['year'].isin(TEST_YEARS)].copy()
    print(f"loaded {len(df):,}  fit={sorted(fit['year'].unique())} "
          f"test={sorted(test['year'].unique())}  weeks {df['week'].min()}-{df['week'].max()}\n")

    # ---- 1. per-(pos, week) raw signed bias + fitted slope, startable pool ----
    print("=" * 96)
    print("PER-WEEK, startable pool: raw signed bias (pred-actual) and the half-strength slope")
    print("=" * 96)
    for pos in POSITIONS:
        p = fit[(fit['pos'] == pos) & fit['start']]
        line = [f"{pos}"]
        for w in range(1, 19):
            sub = p[p['week'] == w]
            if len(sub) < 20:
                line.append(f"w{w}:  --  ")
                continue
            bias = (sub['raw'] - sub['actual']).mean()
            fh = _fit_half(sub['raw'].to_numpy(), sub['actual'].to_numpy())
            sl = fh[0] if fh else float('nan')
            line.append(f"w{w}:{bias:+5.1f}/{sl:.2f}")
        print("  " + "  ".join(line))
    print("\n(each cell = raw_bias / fitted_slope.  bias>0 = model HIGH.  slope<1 = shrink.)\n")

    # ---- 2. score each bucketing: fit on FIT_YEARS, measure on TEST_YEARS ----
    print("=" * 96)
    print("BUCKETING BAKE-OFF  -  startable MAE and |bias| on the held-out test years")
    print("=" * 96)
    print(f"{'scheme':<16}{'pos':<5}{'buckets':<8}{'START-MAE':>11}{'START-|bias|':>13}"
          f"{'d-MAE vs single':>17}")
    per_scheme = {}
    for scheme, wk2b in BUCKETS.items():
        for pos in POSITIONS:
            f_pos = fit[(fit['pos'] == pos)]
            t_pos = test[(test['pos'] == pos)].copy()
            if t_pos.empty:
                continue
            coeffs = {}
            for b in set(wk2b.values()):
                fb = f_pos[f_pos['week'].map(wk2b) == b]
                c = _fit_half(fb['raw'].to_numpy(), fb['actual'].to_numpy())
                if c:
                    coeffs[b] = c
            if not coeffs:
                continue
            b_series = t_pos['week'].map(wk2b)
            sl = b_series.map(lambda b: coeffs.get(b, (1.0, 0.0))[0]).to_numpy()
            ic = b_series.map(lambda b: coeffs.get(b, (1.0, 0.0))[1]).to_numpy()
            t_pos['cal'] = np.clip(sl * t_pos['raw'].to_numpy() + ic, 0.0, None)
            ts = t_pos[t_pos['start']]
            mae = (ts['cal'] - ts['actual']).abs().mean()
            bias = abs((ts['cal'] - ts['actual']).mean())
            per_scheme.setdefault(pos, {})[scheme] = mae
            d = mae - per_scheme[pos].get('single', mae)
            print(f"{scheme:<16}{pos:<5}{len(set(wk2b.values())):<8}{mae:>11.3f}{bias:>13.3f}"
                  f"{d:>+17.3f}")
        print()

    # ---- 3. per-STAT: does calibrating stats beat calibrating points? --------
    print("=" * 96)
    print("PER-STAT vs POINTS  -  startable, cold weeks (1-2) only, held-out test years")
    print("=" * 96)
    stat_families = {'QB': ('attempts', 'passing_yards', 'passing_tds', 'passing_interceptions'),
                     'RB': ('carries', 'rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards'),
                     'WR': ('targets', 'receptions', 'receiving_yards', 'receiving_tds'),
                     'TE': ('targets', 'receptions', 'receiving_yards', 'receiving_tds')}
    for pos in POSITIONS:
        fc = fit[(fit['pos'] == pos) & (fit['week'] <= 2) & fit['start']]
        tc = test[(test['pos'] == pos) & (test['week'] <= 2) & test['start']].copy()
        if len(tc) < 30:
            print(f"  {pos}: too few cold-week test rows"); continue
        parts = []
        for s in stat_families[pos]:
            pc, ac = f'p_{s}', f'a_{s}'
            if pc not in fc.columns or fc[pc].notna().sum() < 30:
                continue
            c = _fit_half(fc[pc].dropna().to_numpy(),
                          fc.loc[fc[pc].notna(), ac].to_numpy())
            if not c:
                continue
            raw_b = (tc[pc] - tc[ac]).mean()
            cal_b = ((c[0] * tc[pc] + c[1]) - tc[ac]).mean()
            parts.append(f"{s}: bias {raw_b:+.2f}->{cal_b:+.2f} (slope {c[0]:.2f})")
        print(f"  {pos} cold-week per-stat:  " + "   ".join(parts))
    print("\nDONE.  Read the per-week grid for the shape, the bake-off for the bucketing, "
          "the per-stat block for whether stats earn their own line.")


# Shipping scheme: only WR and TE earn a season-phase split. The bake-off
# (analyze mode, held out on 2024-25) put QB at d-MAE +0.010 and RB at +0.002
# for every bucketing - i.e. a wash-to-worse - so they keep one line all year.
COLD_MAX_WEEK = 4
BUCKETED_POS = ('WR', 'TE')


def _half_fit_or_identity(sub, pcol='raw', acol='actual'):
    c = _fit_half(sub[pcol].to_numpy(dtype=float), sub[acol].to_numpy(dtype=float))
    return c if c else (1.0, 0.0)


def _apply(raw, coeff):
    return np.clip(coeff[0] * np.asarray(raw, dtype=float) + coeff[1], 0.0, None)


def do_emit():
    df = pd.read_csv(DUMP_PATH)
    df['start'] = _startable_mask(df)
    all_years = sorted(df['year'].unique())
    hold = max(all_years)                       # 2025 - never in any fit here
    ship_fit = df[df['year'] < hold]            # 2021-2024 -> the held-out check
    full_fit = df                               # 2021-2025 -> the values we ship

    def fit_scheme(frame):
        single, cold, rest = {}, {}, {}
        for pos in POSITIONS:
            p = frame[frame['pos'] == pos]
            single[pos] = _half_fit_or_identity(p)
            if pos in BUCKETED_POS:
                cold[pos] = _half_fit_or_identity(p[p['week'] <= COLD_MAX_WEEK])
                rest[pos] = _half_fit_or_identity(p[p['week'] > COLD_MAX_WEEK])
        return single, cold, rest

    # ---- held-out 2025 check: shipped single line vs the new scheme ----------
    s_single, s_cold, s_rest = fit_scheme(ship_fit)
    t = df[df['year'] == hold].copy()
    from data.weekly_projections import WEEKLY_CALIBRATION as SHIPPED
    print("=" * 88)
    print(f"HELD-OUT {hold}: startable MAE / |bias|   (fit on {min(all_years)}-{hold - 1})")
    print("=" * 88)
    print(f"{'pos':<5}{'shipped single':>20}{'refit single':>20}{'WR/TE 2-bucket':>20}")
    for pos in POSITIONS:
        tp = t[t['pos'] == pos].copy()
        ts = tp[tp['start']]
        raw, act = ts['raw'].to_numpy(dtype=float), ts['actual'].to_numpy(dtype=float)

        def mb(pred):
            e = pred - act
            return f"{np.abs(e).mean():.3f}/{e.mean():+.3f}"

        shipped = _apply(raw, SHIPPED.get(pos, (1.0, 0.0)))
        refit = _apply(raw, s_single[pos])
        if pos in BUCKETED_POS:
            wk = ts['week'].to_numpy()
            bkt = np.where(wk <= COLD_MAX_WEEK,
                           _apply(raw, s_cold[pos]), _apply(raw, s_rest[pos]))
        else:
            bkt = refit
        print(f"{pos:<5}{mb(shipped):>20}{mb(refit):>20}{mb(bkt):>20}")

    # ---- values to ship: fit on everything -----------------------------------
    f_single, f_cold, f_rest = fit_scheme(full_fit)
    print("\n" + "=" * 88)
    print(f"SHIP THESE  (fit on {min(all_years)}-{max(all_years)}, half-strength, two-sided)")
    print("=" * 88)
    print("WEEKLY_CALIBRATION = {")
    for pos in POSITIONS:
        s, i = f_single[pos]
        print(f"    {pos!r:<6}: ({s:.3f}, {i:.3f}),")
    print("}")
    print(f"\nWEEKLY_CALIBRATION_COLD_MAX_WEEK = {COLD_MAX_WEEK}")
    print("WEEKLY_CALIBRATION_BY_BUCKET = {")
    for bname, bdict in (('cold', f_cold), ('rest', f_rest)):
        print(f"    {bname!r}: {{")
        for pos in BUCKETED_POS:
            s, i = bdict[pos]
            print(f"        {pos!r:<6}: ({s:.3f}, {i:.3f}),")
        print("    },")
    print("}")
    print("\n(QB/RB fall through to WEEKLY_CALIBRATION - no phase split. WR/TE 'rest'"
          "\n bucket is the in-season line; 'cold' is weeks 1-4.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('dump', 'analyze', 'emit'), required=True)
    ap.add_argument('--years', default='2021-2025')
    ap.add_argument('--weeks', default='1-18')
    ap.add_argument('--scoring', default='Full PPR')
    a = ap.parse_args()
    if a.mode == 'dump':
        y0, y1 = (int(x) for x in a.years.split('-'))
        w0, w1 = (int(x) for x in a.weeks.split('-'))
        years, weeks = list(range(y0, y1 + 1)), list(range(w0, w1 + 1))
        print(f"dump {years} wk{weeks[0]}-{weeks[-1]}  ({len(years) * len(weeks)} builds)\n", flush=True)
        do_dump(years, weeks, a.scoring)
    elif a.mode == 'emit':
        do_emit()
    else:
        do_analyze()


if __name__ == '__main__':
    main()
