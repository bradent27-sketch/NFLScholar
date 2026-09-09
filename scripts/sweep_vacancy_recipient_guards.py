"""Backtest the 2026-09-08 surgical vacancy-recipient guards.

Each arm is DEFAULT_FEATURES plus one or more of:
  v2_vacancy_bump_cap     - no ONE recipient gains > RECEIVER_COLD_START_VACANCY_MAX_BUMP
  v2_vacancy_growth_cap   - post-vacancy share <= pre x GROWTH_CAP (MIN_ABS_GAIN floor)
  v2_vacancy_chart_split  - split the pool by Ourlads chart rank

All three shape only the ALLOCATION of the vacated pool - never its size,
never a non-recipient - so the blast radius should be far smaller than the
already-rejected v2_cold_start_room_budget / pool-dampening.

Question: does any arm pull the LAC-Gadsden-shaped over-projection down
(returning moderate-role receiver rocketed to the 0.92 cap) without a
significant MAE cost on the decision pools?

    python scripts/sweep_vacancy_recipient_guards.py --years 2022,2023,2024,2025 --weeks 1
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

from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from scripts.eval_weekly_model import _metrics, _weighted, STARTABLE_N, _actual_points  # noqa: E402

BASE = frozenset(DEFAULT_FEATURES | {'v2_historical_ourlads'})
ARMS = {
    'bump_cap':     frozenset(BASE | {'v2_vacancy_bump_cap'}),
    'growth_cap':   frozenset(BASE | {'v2_vacancy_growth_cap'}),
    'bump+growth':  frozenset(BASE | {'v2_vacancy_bump_cap', 'v2_vacancy_growth_cap'}),
    'all3':         frozenset(BASE | {'v2_vacancy_bump_cap', 'v2_vacancy_growth_cap',
                                      'v2_vacancy_chart_split'}),
}
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


def _sign_p(wins, losses):
    n = wins + losses
    if n == 0:
        return float('nan')
    k = max(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n))


def run(years, weeks, scoring):
    scoring_col = 'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'
    pairs = {arm: {s[0]: [] for s in SCOPES} for arm in ARMS}
    ledger = {arm: [] for arm in ARMS}

    for year in years:
        stats_df, _tc, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns:
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                continue
            build_weekly_projections.clear()
            base_df, bmeta = build_weekly_projections(
                year, week, scoring, as_of_week=week, apply_injury=False, features=BASE)
            arm_dfs = {}
            for arm, feats in ARMS.items():
                build_weekly_projections.clear()
                arm_dfs[arm], _m = build_weekly_projections(
                    year, week, scoring, as_of_week=week, apply_injury=False, features=feats)
            if base_df.empty or any(d.empty for d in arm_dfs.values()):
                print(f"{year} w{week}: empty", flush=True)
                continue
            common = set(base_df['Player'])
            for d in arm_dfs.values():
                common &= set(d['Player'])
            if len(common) < 20:
                continue
            b = base_df[base_df['Player'].isin(common)].set_index('Player')
            for arm, d in arm_dfs.items():
                v = d[d['Player'].isin(common)].set_index('Player')
                for scope, pos, startable in SCOPES:
                    bd = _scope_df(b.reset_index(), pos, startable)
                    vd = _scope_df(v.reset_index(), pos, startable)
                    mb = _metrics(pd.Series(bd['Model Proj Pts'].to_numpy(), index=bd['Player']), actual)
                    mv = _metrics(pd.Series(vd['Model Proj Pts'].to_numpy(), index=vd['Player']), actual)
                    if mb and mv:
                        pairs[arm][scope].append((mb, mv))
                touched = b.index[b['Pos'].isin(['WR', 'TE'])].intersection(v.index)
                for p in touched:
                    bs = float(b.loc[p, 'Expected Snap Share']); vs = float(v.loc[p, 'Expected Snap Share'])
                    if abs(vs - bs) < 0.03:
                        continue
                    bp = float(b.loc[p, 'Model Proj Pts']); vp = float(v.loc[p, 'Model Proj Pts'])
                    av = float(actual.get(p, np.nan))
                    ledger[arm].append(dict(
                        year=year, week=week, team=str(b.loc[p, 'Team']), pos=str(b.loc[p, 'Pos']),
                        player=p, bs=round(bs, 3), vs=round(vs, 3), bp=round(bp, 2), vp=round(vp, 2),
                        actual=round(av, 2) if np.isfinite(av) else None,
                        d_err=round(abs(vp - av) - abs(bp - av), 2) if np.isfinite(av) else None))
            print(f"{year} w{week}: common {len(common)}", flush=True)

    _report(pairs, ledger, years, weeks)


def _report(pairs, ledger, years, weeks):
    bar = '=' * 96
    for arm in ARMS:
        print(f"\n{bar}\n{arm}  vs base   years={years} weeks={weeks}\n{bar}")
        print("dMAE = arm - base.  NEGATIVE = the guard is MORE accurate.\n")
        print(f"{'scope':<12}{'n':>8}{'MAE base':>11}{'MAE arm':>11}{'dMAE':>10}{'95% CI':>22}{'wk W-L':>10}{'sign p':>9}")
        for scope, _p, _s in SCOPES:
            pr = pairs[arm].get(scope) or []
            if not pr:
                continue
            mb = [x[0] for x in pr]; mv = [x[1] for x in pr]
            nt = sum(m['n'] for m in mb)
            base_mae, arm_mae = _weighted(mb, 'mae'), _weighted(mv, 'mae')
            deltas = [b['mae'] - a['mae'] for a, b in zip(mb, mv)]  # arm - base
            wts = [a['n'] for a in mb]
            lo, hi = _boot_ci(deltas, wts)
            wins = sum(1 for d in deltas if d < 0); losses = sum(1 for d in deltas if d > 0)
            star = ' *' if (np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)) else '  '
            print(f"{scope:<12}{nt:>8}{base_mae:>11.3f}{arm_mae:>11.3f}{arm_mae - base_mae:>+10.3f}"
                  f"  [{lo:+.3f},{hi:+.3f}]{star}{f'{wins}-{losses}':>10}{_sign_p(wins, losses):>9.3f}")
        led = ledger[arm]
        big = sorted((r for r in led if r['bs'] - r['vs'] > 0.08), key=lambda r: -(r['bs'] - r['vs']))
        print(f"\n  recipients pulled DOWN > 0.08 snap ({len(big)}; sum d_err = "
              f"{round(sum(r['d_err'] for r in big if r['d_err'] is not None), 2)}):")
        for r in big[:40]:
            print(f"    {r['year']} w{r['week']} {r['team']:<4}{r['pos']:<3}{r['player']:<22} "
                  f"{r['bs']:.2f}->{r['vs']:.2f}  proj {r['bp']:>5.1f}->{r['vp']:>5.1f}  act={r['actual']}  d_err={r['d_err']}")
    print("\n* = bootstrap 95% CI on the pooled weekly dMAE excludes zero.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2022,2023,2024,2025')
    ap.add_argument('--weeks', default='1')
    ap.add_argument('--scoring', default='Full PPR')
    a = ap.parse_args()
    years = [int(x) for x in a.years.replace(' ', '').split(',')]
    if '-' in a.weeks:
        w0, w1 = (int(x) for x in a.weeks.split('-'))
        weeks = list(range(w0, w1 + 1))
    else:
        weeks = [int(a.weeks)]
    run(years, weeks, a.scoring)


if __name__ == '__main__':
    main()
