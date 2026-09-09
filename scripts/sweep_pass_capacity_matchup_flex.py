"""In-season backtest for v2_pass_capacity_matchup_flex (2026-09-08).

apply_pass_capacity_conservation runs LAST on the target channel, after each
player's +-22% forward matchup multiplier is baked into `targets`, and
reconciles it back out through (a) a hard prior-season RB/(WR+TE) split and
(b) a targets-denominated deadband. The flag relaxes both, bounded:
  band = PASS_CAPACITY_RB_SHARE_BAND   - RB share may follow this week's mix
                                          within +-band of the prior share
  dead = PASS_CAPACITY_FACTOR_DEADBAND - a group within a +-dead MULTIPLICATIVE
                                          band of its budget is left alone
band=0 / dead=0 are exact no-ops for their arm, so one --arms entry can test
either alone or both together.

    python scripts/sweep_pass_capacity_matchup_flex.py --years 2023,2024 --weeks 5-12 \
        --arms "0.05,0;0.08,0;0,0.10;0,0.15;0.05,0.10"
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
import data.pass_capacity_allocator as pca  # noqa: E402
from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from scripts.eval_weekly_model import _metrics, _weighted, STARTABLE_N, _actual_points  # noqa: E402

BASE = frozenset(DEFAULT_FEATURES)
VAR = frozenset(BASE | {'v2_pass_capacity_matchup_flex'})
SCOPES = [('ALL', None, False), ('RB', 'RB', False), ('WR', 'WR', False), ('TE', 'TE', False),
          ('START-RB', 'RB', True), ('START-WR', 'WR', True), ('START-TE', 'TE', True)]
RB_RECV_STATS = ['targets', 'receptions', 'receiving_yards']


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


def _team_target_ratio(df):
    """Sum(RB+WR+TE targets) / Sum(QB passing_attempts), per team, then mean."""
    d = df.copy()
    d['targets'] = pd.to_numeric(d.get('targets', 0.0), errors='coerce').fillna(0.0)
    d['passing_attempts'] = pd.to_numeric(d.get('passing_attempts', 0.0), errors='coerce').fillna(0.0)
    catch = d[d['Pos'].isin(['RB', 'WR', 'TE'])].groupby('Team')['targets'].sum()
    att = d[d['Pos'].eq('QB')].groupby('Team')['passing_attempts'].sum()
    j = pd.concat([catch, att], axis=1, keys=['t', 'a']).dropna()
    j = j[j['a'] > 5.0]
    return float((j['t'] / j['a']).mean()) if not j.empty else float('nan')


_SHIPPED_BAND = pca.PASS_CAPACITY_RB_SHARE_BAND
_SHIPPED_DEAD = pca.PASS_CAPACITY_FACTOR_DEADBAND


def run(years, weeks, scoring, arms):
    scoring_col = 'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'
    per = {a: {sc[0]: [] for sc in SCOPES} for a in arms}       # (mb, mv) fantasy-pts metrics
    rb_recv = {a: {s: [] for s in RB_RECV_STATS} for a in arms}  # (base|err|, var|err|) startable RB
    ratio = {a: [] for a in arms}                                # (base_ratio, var_ratio) per slate
    lifted = {a: [] for a in arms}                               # (base_signed_pts_err, var_signed_pts_err) for RBs the flag raised

    for year in years:
        stats_df, _tc, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns:
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                continue
            wk = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week]
            act_tgt = wk.groupby(name_col)['targets'].sum() if 'targets' in wk.columns else None
            act_rec = wk.groupby(name_col)['receptions'].sum() if 'receptions' in wk.columns else None
            act_ry = wk.groupby(name_col)['receiving_yards'].sum() if 'receiving_yards' in wk.columns else None
            act_map = {'targets': act_tgt, 'receptions': act_rec, 'receiving_yards': act_ry}
            build_weekly_projections.clear()
            b, _ = build_weekly_projections(year, week, scoring, as_of_week=week,
                                            apply_injury=False, features=BASE)
            if b.empty:
                continue
            b_ratio = _team_target_ratio(b)
            for a in arms:
                band, dead = a
                # These module globals are read at CALL time inside
                # apply_pass_capacity_conservation (dial args left None), so a
                # direct rebind here takes effect; an os.environ set would not
                # (the _env_float read happens once at import).
                pca.PASS_CAPACITY_RB_SHARE_BAND = float(band)
                pca.PASS_CAPACITY_FACTOR_DEADBAND = float(dead)
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
                for scope, pos, st in SCOPES:
                    bd = _scope_df(bi.reset_index(), pos, st)
                    vd = _scope_df(vi.reset_index(), pos, st)
                    mb = _metrics(pd.Series(bd['Model Proj Pts'].to_numpy(), index=bd['Player']), actual)
                    mv = _metrics(pd.Series(vd['Model Proj Pts'].to_numpy(), index=vd['Player']), actual)
                    if mb and mv:
                        per[a][scope].append((mb, mv))
                ratio[a].append((b_ratio, _team_target_ratio(v)))
                # RB receiving-stat abs err on the PASS-CATCHING backs (top 24
                # by projected targets, not points - a top-40-by-points RB pool
                # is mostly pure runners with ~1 target and washes the signal).
                _rb_all = bi.reset_index()
                _rb_all = _rb_all[_rb_all['Pos'] == 'RB'].nlargest(24, 'targets')
                idx = [p for p in _rb_all['Player'] if act_tgt is not None and p in act_tgt.index]
                for stat in RB_RECV_STATS:
                    am = act_map[stat]
                    if am is None:
                        continue
                    for p in idx:
                        a_val = float(am.get(p, np.nan))
                        if not np.isfinite(a_val):
                            continue
                        eb = abs(float(bi.loc[p, stat]) - a_val)
                        ev = abs(float(vi.loc[p, stat]) - a_val)
                        rb_recv[a][stat].append((eb, ev))
                # RBs the flag actually LIFTED (var targets up > 0.1): did the
                # extra volume reduce their systematic points under-projection?
                for p in common:
                    if bi.loc[p, 'Pos'] != 'RB':
                        continue
                    if float(vi.loc[p, 'targets']) - float(bi.loc[p, 'targets']) > 0.1:
                        av = float(actual.get(p, np.nan))
                        if np.isfinite(av):
                            lifted[a].append((float(bi.loc[p, 'Model Proj Pts']) - av,
                                              float(vi.loc[p, 'Model Proj Pts']) - av))
            print(f"{year} w{week}: done", flush=True)
    pca.PASS_CAPACITY_RB_SHARE_BAND = _SHIPPED_BAND
    pca.PASS_CAPACITY_FACTOR_DEADBAND = _SHIPPED_DEAD
    _report(per, rb_recv, ratio, lifted, years, weeks, arms)


def _report(per, rb_recv, ratio, lifted, years, weeks, arms):
    bar = '=' * 108
    print(f"\n{bar}\nv2_pass_capacity_matchup_flex   years={years} weeks={weeks[0]}-{weeks[-1]}\n{bar}")
    print("dMAE = var - base (fantasy pts, weighted).  NEGATIVE = more accurate.  * = bootstrap 95% CI excludes 0.\n")
    keyscopes = ['ALL', 'RB', 'START-RB', 'WR', 'START-WR', 'TE', 'START-TE']
    hdr = f"{'band':>6}{'dead':>6}"
    for sc in keyscopes:
        hdr += f"{sc:>11}"
    hdr += f"{'RBtgtMAE':>10}{'RBrecMAE':>10}{'RBryMAE':>10}{'Σtgt/att b>v':>14}"
    print(hdr)
    for a in arms:
        band, dead = a
        row = f"{band:>6.2f}{dead:>6.2f}"
        for sc in keyscopes:
            pr = per[a].get(sc) or []
            if not pr:
                row += f"{'-':>11}"
                continue
            mb = [x[0] for x in pr]; mv = [x[1] for x in pr]
            d = _weighted(mv, 'mae') - _weighted(mb, 'mae')
            deltas = [x[1]['mae'] - x[0]['mae'] for x in pr]
            lo, hi = _boot_ci(deltas, [x[0]['n'] for x in pr])
            star = '*' if (np.isfinite(lo) and (lo > 0 or hi < 0)) else ''
            row += f"{f'{d:+.3f}{star}':>11}"
        for stat in RB_RECV_STATS:
            pairs = rb_recv[a][stat]
            if pairs:
                eb = np.array([x[0] for x in pairs]); ev = np.array([x[1] for x in pairs])
                d = float(ev.mean() - eb.mean())
                lo, hi = _boot_ci(list(ev - eb), [1.0] * len(pairs))
                star = '*' if (np.isfinite(lo) and (lo > 0 or hi < 0)) else ''
                row += f"{f'{d:+.3f}{star}':>10}"
            else:
                row += f"{'-':>10}"
        rr = ratio[a]
        if rr:
            rb_ = np.mean([x[0] for x in rr]); rv_ = np.mean([x[1] for x in rr])
            row += f"{f'{rb_:.3f}>{rv_:.3f}':>14}"
        print(row)
    print("\n(RB*MAE = mean abs err on startable-RB targets / receptions / receiving_yards; "
          "Σtgt/att = mean team catcher-targets / QB attempts, base>var - must stay ~<=1.0)")

    print(f"\n{bar}\nRBs THE FLAG LIFTED (var targets up > 0.1): mean SIGNED points error, base -> var\n{bar}")
    print(f"{'band':>6}{'dead':>6}{'n':>7}{'mean signed err b':>20}{'mean signed err v':>20}{'|err| b':>10}{'|err| v':>10}")
    for a in arms:
        ll = lifted[a]
        if not ll:
            print(f"{a[0]:>6.2f}{a[1]:>6.2f}{0:>7}")
            continue
        sb = np.array([x[0] for x in ll]); sv = np.array([x[1] for x in ll])
        print(f"{a[0]:>6.2f}{a[1]:>6.2f}{len(ll):>7}{sb.mean():>20.3f}{sv.mean():>20.3f}"
              f"{np.abs(sb).mean():>10.3f}{np.abs(sv).mean():>10.3f}")
    print("\n(if 'mean signed err' is NEGATIVE at base - model under-projects these RBs - and moves toward 0 at var,\n"
          " the flex is recovering a real matchup edge the allocator was washing out)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2023,2024')
    ap.add_argument('--weeks', default='5-12')
    ap.add_argument('--scoring', default='Full PPR')
    ap.add_argument('--arms', default='0.05,0;0.08,0;0,0.10;0,0.15;0.05,0.10')
    a = ap.parse_args()
    years = [int(x) for x in a.years.replace(' ', '').split(',')]
    lo, hi = a.weeks.split('-')
    weeks = list(range(int(lo), int(hi) + 1))
    arms = [tuple(float(x) for x in p.split(',')) for p in a.arms.split(';')]
    run(years, weeks, a.scoring, arms)


if __name__ == '__main__':
    main()
