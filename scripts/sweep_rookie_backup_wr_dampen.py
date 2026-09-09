"""Strength sweep for v2_rookie_backup_wr_dampen (2026-09-08).

At full strength (1.0) the flag pulls a no-prior slot-2 rookie WR to
ROOKIE_BACKUP_WR_SHARE (0.06) and zeroes his cold-start vacancy weight; it
lost the backtest marginally (TE +0.018*, nothing better). This sweeps
ROOKIE_BACKUP_WR_DAMPEN_STRENGTH - at 0.5 the pull stops halfway from the 0.16
floor (~0.11) and half the vacancy weight is kept - to find a lighter dock
that lands neutral.

    python scripts/sweep_rookie_backup_wr_dampen.py --years 2022,2023,2024,2025 --weeks 1 \
        --strengths 0.3,0.5,0.7,1.0
"""
import argparse
import math
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

import data.weekly_projections as wp  # noqa: E402
from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from scripts.eval_weekly_model import _metrics, _weighted, STARTABLE_N, _actual_points  # noqa: E402

BASE = frozenset(DEFAULT_FEATURES | {'v2_historical_ourlads'})
VAR = frozenset(BASE | {'v2_rookie_backup_wr_dampen'})
SCOPES = [('ALL', None, False), ('WR', 'WR', False), ('TE', 'TE', False),
          ('START-WR', 'WR', True), ('START-TE', 'TE', True),
          ('START-QB', 'QB', True), ('START-RB', 'RB', True)]


def _scope_df(df, pos, startable):
    d = df if pos is None else df[df['Pos'] == pos]
    if startable:
        d = d.nlargest(STARTABLE_N.get(pos, 30), 'Model Proj Pts')
    return d


def _boot_ci(deltas, weights, n=3000, seed=0):
    if len(deltas) < 4:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, float); w = np.asarray(weights, float)
    idx = np.arange(len(d)); out = np.empty(n)
    for b in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        out[b] = np.average(d[s], weights=w[s])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def run(years, weeks, scoring, strengths):
    scoring_col = 'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'
    by_strength = {s: {sc[0]: [] for sc in SCOPES} for s in strengths}
    sturdy = {s: [] for s in strengths}   # WR-target shifts for the eye-test column
    # per docked player at each strength: (targets_delta, actual_pts, base_abs_err, var_abs_err)
    docked = {s: [] for s in strengths}
    # per-player abs-err pairs on the startable WR pool, for a robust MEDIAN view
    startwr_err = {s: [] for s in strengths}   # list of (base_abs_err, var_abs_err)

    for year in years:
        stats_df, _tc, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns:
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                continue
            build_weekly_projections.clear()
            b, _ = build_weekly_projections(year, week, scoring, as_of_week=week,
                                            apply_injury=False, features=BASE)
            if b.empty:
                continue
            for s in strengths:
                wp.ROOKIE_BACKUP_WR_DAMPEN_STRENGTH = float(s)
                build_weekly_projections.clear()
                v, _ = build_weekly_projections(year, week, scoring, as_of_week=week,
                                                apply_injury=False, features=VAR)
                if v.empty:
                    continue
                common = sorted(set(b['Player']) & set(v['Player']))
                if len(common) < 20:
                    continue
                bi = b[b['Player'].isin(common)].set_index('Player')
                vi = v[v['Player'].isin(common)].set_index('Player')
                for scope, pos, startable in SCOPES:
                    bd = _scope_df(bi.reset_index(), pos, startable)
                    vd = _scope_df(vi.reset_index(), pos, startable)
                    mb = _metrics(pd.Series(bd['Model Proj Pts'].to_numpy(), index=bd['Player']), actual)
                    mv = _metrics(pd.Series(vd['Model Proj Pts'].to_numpy(), index=vd['Player']), actual)
                    if mb and mv:
                        by_strength[s][scope].append((mb, mv))
                start_wr = set(_scope_df(bi.reset_index(), 'WR', True)['Player'])
                for p in common:
                    if bi.loc[p, 'Pos'] != 'WR':
                        continue
                    av = float(actual.get(p, np.nan))
                    be = abs(float(bi.loc[p, 'Model Proj Pts']) - av) if np.isfinite(av) else np.nan
                    ve = abs(float(vi.loc[p, 'Model Proj Pts']) - av) if np.isfinite(av) else np.nan
                    if p in start_wr and np.isfinite(be):
                        startwr_err[s].append((be, ve))
                    d = float(vi.loc[p, 'targets']) - float(bi.loc[p, 'targets'])
                    if d < -0.15:
                        sturdy[s].append(d)
                        if np.isfinite(av):
                            docked[s].append((d, av, be, ve))
            print(f"{year} w{week}: done", flush=True)
    wp.ROOKIE_BACKUP_WR_DAMPEN_STRENGTH = 1.0
    _report(by_strength, sturdy, docked, startwr_err, years, weeks, strengths)


def _report(by_strength, sturdy, docked, startwr_err, years, weeks, strengths):
    bar = '=' * 104
    print(f"\n{bar}\nv2_rookie_backup_wr_dampen STRENGTH SWEEP   years={years} weeks={weeks}\n{bar}")
    print("dMAE = var - base (mean).  NEGATIVE = more accurate.  * = bootstrap 95% CI excludes 0.\n")
    keyscopes = ['ALL', 'WR', 'TE', 'START-WR', 'START-TE']
    hdr = f"{'strength':>9}"
    for sc in keyscopes:
        hdr += f"{sc:>12}"
    hdr += f"{'START-WR dMdAE':>15}"
    print(hdr)
    for s in strengths:
        row = f"{s:>9.2f}"
        for sc in keyscopes:
            pr = by_strength[s].get(sc) or []
            if not pr:
                row += f"{'-':>12}"
                continue
            mb = [x[0] for x in pr]; mv = [x[1] for x in pr]
            d = _weighted(mv, 'mae') - _weighted(mb, 'mae')
            deltas = [x[1]['mae'] - x[0]['mae'] for x in pr]
            lo, hi = _boot_ci(deltas, [x[0]['n'] for x in pr])
            star = '*' if (np.isfinite(lo) and (lo > 0 or hi < 0)) else ''
            row += f"{f'{d:+.3f}{star}':>12}"
        er = startwr_err[s]
        if er:
            be = np.array([x[0] for x in er]); ve = np.array([x[1] for x in er])
            row += f"{np.median(ve) - np.median(be):>+15.3f}"
        print(row)
    print("\n(START-WR dMdAE = change in MEDIAN abs error on the startable-WR pool - robust to breakout outliers)")

    print(f"\n{bar}\nWHAT THE DOCK ACTUALLY HITS (players whose targets fell > 0.15)\n{bar}")
    print(f"{'strength':>9}{'n':>6}{'mean act':>10}{'median act':>12}{'% under 5':>11}"
          f"{'sum base|err|':>14}{'sum var|err|':>14}")
    for s in strengths:
        dd = docked[s]
        if not dd:
            continue
        act = np.array([r[1] for r in dd]); be = np.array([r[2] for r in dd]); ve = np.array([r[3] for r in dd])
        print(f"{s:>9.2f}{len(dd):>6}{act.mean():>10.2f}{np.median(act):>12.2f}"
              f"{100.0 * (act < 5).mean():>10.0f}%{be.sum():>14.1f}{ve.sum():>14.1f}")
    print("\n(if 'median act' is ~0 and '% under 5' is high, the dock is right on the TYPICAL case even "
          "\n where mean |err| is dragged by a rare breakout - the point of looking past MAE)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2022,2023,2024,2025')
    ap.add_argument('--weeks', default='1')
    ap.add_argument('--scoring', default='Full PPR')
    ap.add_argument('--strengths', default='0.3,0.5,0.7,1.0')
    a = ap.parse_args()
    years = [int(x) for x in a.years.replace(' ', '').split(',')]
    if '-' in a.weeks:
        w0, w1 = (int(x) for x in a.weeks.split('-'))
        weeks = list(range(w0, w1 + 1))
    else:
        weeks = [int(a.weeks)]
    strengths = [float(x) for x in a.strengths.replace(' ', '').split(',')]
    run(years, weeks, a.scoring, strengths)


if __name__ == '__main__':
    main()
