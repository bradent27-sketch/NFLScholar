"""Strength sweep for v2_td_volume_shrink (2026-09-08).

At full strength (1.0) the flag regressed cold-start WR/TE receiving_tds hard
toward a league TD-per-target rate on projected volume and LOST the backtest
(ALL +0.024*, TE +0.099*, startable receiving_tds MAE +0.030*). This sweeps
TD_VOLUME_SHRINK_STRENGTH - a 0..1 dial on how far the shrink moves the rate
off the player's own two-year-blended number - to see whether a lighter
"slight regression" version lands neutral.

    python scripts/sweep_td_volume_shrink.py --years 2022,2023,2024,2025 --weeks 1 \
        --strengths 0.25,0.4,0.55,0.7,1.0
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
VAR = frozenset(BASE | {'v2_td_volume_shrink'})
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


def _sign_p(w, l):
    n = w + l
    if n == 0:
        return float('nan')
    k = max(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n))


def run(years, weeks, scoring, strengths):
    scoring_col = 'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'
    # per strength: {scope: [(mb, mv), ...]}  and a receiving_tds MAE pair list
    by_strength = {s: {sc[0]: [] for sc in SCOPES} for s in strengths}
    td_by_strength = {s: [] for s in strengths}
    # per-player |err| pairs on receiving_tds, startable WR/TE - for a MEDIAN view
    td_err = {s: [] for s in strengths}

    for year in years:
        stats_df, _tc, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns:
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                continue
            wk = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week]
            td_actual = wk.groupby(name_col)['receiving_tds'].sum() if 'receiving_tds' in wk.columns else None
            build_weekly_projections.clear()
            b, _ = build_weekly_projections(year, week, scoring, as_of_week=week,
                                            apply_injury=False, features=BASE)
            if b.empty:
                continue
            for s in strengths:
                wp.TD_VOLUME_SHRINK_STRENGTH = float(s)
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
                if td_actual is not None:
                    pool = pd.concat([_scope_df(bi.reset_index(), p, True) for p in ('WR', 'TE')])['Player']
                    idx = [p for p in pool if p in td_actual.index]
                    if len(idx) >= 5:
                        a = pd.to_numeric(td_actual.reindex(idx), errors='coerce')
                        peb = (pd.to_numeric(bi.reindex(idx)['receiving_tds'], errors='coerce') - a).abs()
                        pev = (pd.to_numeric(vi.reindex(idx)['receiving_tds'], errors='coerce') - a).abs()
                        if np.isfinite(peb.mean()) and np.isfinite(pev.mean()):
                            td_by_strength[s].append((float(peb.mean()), float(pev.mean()), len(idx)))
                            for xb, xv in zip(peb.to_numpy(), pev.to_numpy()):
                                if np.isfinite(xb) and np.isfinite(xv):
                                    td_err[s].append((float(xb), float(xv)))
            print(f"{year} w{week}: done", flush=True)
    wp.TD_VOLUME_SHRINK_STRENGTH = 1.0
    _report(by_strength, td_by_strength, td_err, years, weeks, strengths)


def _report(by_strength, td_by_strength, td_err, years, weeks, strengths):
    bar = '=' * 100
    print(f"\n{bar}\nv2_td_volume_shrink STRENGTH SWEEP   years={years} weeks={weeks}\n{bar}")
    print("dMAE = var - base.  NEGATIVE = more accurate.  * = bootstrap 95% CI excludes 0.\n")
    keyscopes = ['ALL', 'WR', 'TE', 'START-WR', 'START-TE']
    hdr = f"{'strength':>9}"
    for sc in keyscopes:
        hdr += f"{sc:>13}"
    hdr += f"{'recv_tds':>13}{'recvTD-medn':>13}"
    print(hdr)
    for s in strengths:
        row = f"{s:>9.2f}"
        for sc in keyscopes:
            pr = by_strength[s].get(sc) or []
            if not pr:
                row += f"{'-':>13}"
                continue
            mb = [x[0] for x in pr]; mv = [x[1] for x in pr]
            d = _weighted(mv, 'mae') - _weighted(mb, 'mae')
            deltas = [x[1]['mae'] - x[0]['mae'] for x in pr]
            lo, hi = _boot_ci(deltas, [x[0]['n'] for x in pr])
            star = '*' if (np.isfinite(lo) and (lo > 0 or hi < 0)) else ''
            row += f"{f'{d:+.3f}{star}':>13}"
        tp = td_by_strength[s]
        if tp:
            eb = np.array([r[0] for r in tp]); ev = np.array([r[1] for r in tp])
            w = np.array([r[2] for r in tp], float)
            d = float(np.average(ev, weights=w) - np.average(eb, weights=w))
            lo, hi = _boot_ci(list(ev - eb), list(w))
            star = '*' if (np.isfinite(lo) and (lo > 0 or hi < 0)) else ''
            row += f"{f'{d:+.3f}{star}':>13}"
        te = td_err[s]
        if te:
            be = np.array([x[0] for x in te]); ve = np.array([x[1] for x in te])
            row += f"{np.median(ve) - np.median(be):>+13.3f}"
        print(row)
    print("\n(recv_tds = receiving_tds MEAN abs err on startable WR/TE; last col = MEDIAN abs err change)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2022,2023,2024,2025')
    ap.add_argument('--weeks', default='1')
    ap.add_argument('--scoring', default='Full PPR')
    ap.add_argument('--strengths', default='0.25,0.4,0.55,0.7,1.0')
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
