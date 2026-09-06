"""Phase 3 of the game-total-per-stat backtest (2026-09-04, reconfirm).

Two follow-ups requested after the J2/J3 confirm read as a wash on blended
fantasy points but showed real per-stat structure:

  1. POOLED 5-YEAR MAE - J2 (2024-25) and J3 (2021-23) each showed the same
     direction for passing_yards/passing_attempts/passing_completions/
     rushing_attempts but neither window alone cleared significance on most
     of them. Re-running years=2021..2025 in one pool (instead of two
     separate windows) roughly doubles n on those specific stats - the
     natural next step before deciding whether they're real.

  2. TD CALIBRATION - passing_tds/rushing_tds/receiving_tds had by far the
     largest fitted elasticities (+0.27 to +0.40 vs +0.03 to +0.14 for
     volume stats) but showed ~0 MAE movement in J2/J3. TD counts are
     near-binary per player-week, so MAE on the point estimate is the wrong
     instrument - it can't distinguish an expected-TD move from 0.35 to
     0.38. This reuses the SAME base/var builds already being made for the
     MAE table and additionally scores each as a Poisson-implied P(TD>=1)
     (lambda = the projected mean stat), via Brier score and log-loss
     against the actual binary outcome. This is the metric that matters for
     an anytime-TD / first-TD prop, which MAE cannot see.

Both run in ONE pass over the data (same base/var builds reused for both
purposes) rather than two separate sweeps, since each build is expensive.

    python scripts/gte_perstat_reconfirm.py --years 2021,2022,2023,2024,2025 --weeks 3-18
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

from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from scripts.eval_weekly_model import _metrics, _weighted, STARTABLE_N, _actual_points  # noqa: E402
from scripts.gte_perstat_confirm import (  # noqa: E402
    VARIANT, SCOPES, PER_STAT, STAT_SCOPE, _scope_df, _boot_ci, _sign_p, _stat_actuals,
)

TD_STATS = ('passing_tds', 'rushing_tds', 'receiving_tds')
EPS = 1e-6


def _calib_pairs(bd, vd, sa, stat):
    common = bd.index.intersection(vd.index).intersection(sa.index)
    if len(common) < 5:
        return None
    act = pd.to_numeric(sa.loc[common, stat], errors='coerce')
    lam_b = pd.to_numeric(bd.loc[common, stat], errors='coerce').clip(lower=0.0)
    lam_v = pd.to_numeric(vd.loc[common, stat], errors='coerce').clip(lower=0.0)
    ok = act.notna() & lam_b.notna() & lam_v.notna()
    if ok.sum() < 5:
        return None
    act, lam_b, lam_v = act[ok], lam_b[ok], lam_v[ok]
    outcome = (act >= 0.5).astype(float).to_numpy()
    p_b = np.clip(1.0 - np.exp(-lam_b.to_numpy()), EPS, 1.0 - EPS)
    p_v = np.clip(1.0 - np.exp(-lam_v.to_numpy()), EPS, 1.0 - EPS)
    brier_b = float(np.mean((p_b - outcome) ** 2))
    brier_v = float(np.mean((p_v - outcome) ** 2))
    ll_b = float(-np.mean(outcome * np.log(p_b) + (1 - outcome) * np.log(1 - p_b)))
    ll_v = float(-np.mean(outcome * np.log(p_v) + (1 - outcome) * np.log(1 - p_v)))
    return brier_b, brier_v, ll_b, ll_v, len(common), float(outcome.mean())


def run(years, weeks, scoring, outliers_csv):
    scoring_col = 'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'
    scope_pairs = {s[0]: [] for s in SCOPES}
    stat_pairs = {s: [] for s in PER_STAT}
    calib_pairs = {s: [] for s in TD_STATS}
    outliers = []

    for year in years:
        stats_df, _tc, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns:
            print(f"{year}: no weekly data, skipped", flush=True)
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                continue
            base, bmeta = build_weekly_projections(
                year, week, scoring, as_of_week=week, apply_injury=False, features=DEFAULT_FEATURES)
            var, vmeta = build_weekly_projections(
                year, week, scoring, as_of_week=week, apply_injury=False, features=VARIANT)
            if base.empty or var.empty:
                print(f"{year} w{week}: empty ({bmeta.get('reason') or vmeta.get('reason')})")
                continue
            pool = sorted(set(base['Player']) & set(var['Player']))
            if len(pool) < 20:
                continue
            b = base[base['Player'].isin(pool)]
            v = var[var['Player'].isin(pool)]

            for scope, pos, startable in SCOPES:
                bd, vd = _scope_df(b, pos, startable), _scope_df(v, pos, startable)
                mb = _metrics(pd.Series(bd['Model Proj Pts'].to_numpy(), index=bd['Player']), actual)
                mv = _metrics(pd.Series(vd['Model Proj Pts'].to_numpy(), index=vd['Player']), actual)
                if mb and mv:
                    scope_pairs[scope].append((mb, mv))

            sa = _stat_actuals(stats_df, name_col, week)
            for stat in PER_STAT:
                if stat not in b.columns or stat not in sa.columns:
                    continue
                bd = pd.concat([_scope_df(b, p, True) for p in STAT_SCOPE.get(stat, ('RB', 'WR', 'TE'))])
                vd = v[v['Player'].isin(bd['Player'])]
                bd = bd.set_index('Player'); vd = vd.set_index('Player')
                common = bd.index.intersection(vd.index).intersection(sa.index)
                if len(common) < 5:
                    continue
                act = pd.to_numeric(sa.loc[common, stat], errors='coerce')
                eb = (pd.to_numeric(bd.loc[common, stat], errors='coerce') - act).abs().mean()
                ev = (pd.to_numeric(vd.loc[common, stat], errors='coerce') - act).abs().mean()
                if np.isfinite(eb) and np.isfinite(ev):
                    stat_pairs[stat].append((float(eb), float(ev), len(common)))

                if stat in TD_STATS:
                    calib = _calib_pairs(bd, vd, sa, stat)
                    if calib is not None:
                        calib_pairs[stat].append(calib)

            for pos in ('QB', 'RB', 'WR', 'TE'):
                vd = _scope_df(v, pos, True).set_index('Player')
                bd = b.set_index('Player')
                for player, r in vd.iterrows():
                    if player not in actual.index:
                        continue
                    pv, av = float(r['Model Proj Pts']), float(actual[player])
                    base_pp = float(bd.loc[player, 'Model Proj Pts']) if player in bd.index else np.nan
                    outliers.append(dict(year=year, week=week, player=player, pos=pos,
                                         base_proj=round(base_pp, 2), var_proj=round(pv, 2),
                                         actual=round(av, 2), var_error=round(pv - av, 2),
                                         perstat_shift=round(pv - base_pp, 3),
                                         abs_error=round(abs(pv - av), 2)))
            print(f"{year} w{week}: pool {len(pool)}  base/var built", flush=True)

    _report(scope_pairs, stat_pairs, calib_pairs, years, weeks)
    if outliers_csv:
        os.makedirs(os.path.dirname(outliers_csv) or '.', exist_ok=True)
        pd.DataFrame(outliers).sort_values('abs_error', ascending=False).to_csv(outliers_csv, index=False)
        print(f"\nwrote outlier ledger -> {outliers_csv}")


def _report(scope_pairs, stat_pairs, calib_pairs, years, weeks):
    bar = '=' * 92
    print(f"\n{bar}\nGAME-TOTAL PER-STAT ELASTICITY - POOLED reconfirm   years={years} weeks={weeks}\n{bar}")
    print("dMAE = variant - base.  NEGATIVE = per-stat elasticity is MORE accurate than the flat one.\n")
    print(f"{'scope':<12}{'n':>8}{'MAE base':>11}{'MAE var':>11}{'dMAE':>10}{'95% CI':>22}"
          f"{'wk W-L':>10}{'sign p':>9}")
    for scope, _p, _s in SCOPES:
        pairs = scope_pairs.get(scope) or []
        if not pairs:
            continue
        mb = [p[0] for p in pairs]; mv = [p[1] for p in pairs]
        nt = sum(m['n'] for m in mb)
        base_mae, var_mae = _weighted(mb, 'mae'), _weighted(mv, 'mae')
        deltas = [mv_i['mae'] - mb_i['mae'] for mb_i, mv_i in zip(mb, mv)]
        wts = [mb_i['n'] for mb_i in mb]
        lo, hi = _boot_ci(deltas, wts)
        wins = sum(1 for d in deltas if d < 0)
        losses = sum(1 for d in deltas if d > 0)
        star = ' *' if (np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)) else '  '
        print(f"{scope:<12}{nt:>8}{base_mae:>11.3f}{var_mae:>11.3f}{var_mae - base_mae:>+10.3f}"
              f"  [{lo:+.3f},{hi:+.3f}]{star}{f'{wins}-{losses}':>10}{_sign_p(wins, losses):>9.3f}")

    print(f"\n{bar}\nPER-STAT MAE (startable pool), POOLED   NEGATIVE dMAE = helps\n{bar}")
    print(f"{'stat':<24}{'wk n':>7}{'obs':>9}{'MAE base':>11}{'MAE var':>11}{'dMAE':>10}{'95% CI':>22}{'W-L':>9}")
    for stat in PER_STAT:
        rows = stat_pairs.get(stat) or []
        if not rows:
            continue
        eb = np.array([r[0] for r in rows]); ev = np.array([r[1] for r in rows])
        w = np.array([r[2] for r in rows], float)
        base_mae = float(np.average(eb, weights=w)); var_mae = float(np.average(ev, weights=w))
        deltas = list(ev - eb)
        lo, hi = _boot_ci(deltas, list(w))
        wins = int((ev < eb).sum()); losses = int((ev > eb).sum())
        star = ' *' if (np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)) else '  '
        print(f"{stat:<24}{len(rows):>7}{int(w.sum()):>9}{base_mae:>11.3f}{var_mae:>11.3f}"
              f"{var_mae - base_mae:>+10.3f}  [{lo:+.3f},{hi:+.3f}]{star}{f'{wins}-{losses}':>9}")
    print("\n* = bootstrap 95% CI on the pooled weekly dMAE excludes zero.")

    print(f"\n{bar}\nTD CALIBRATION (startable pool) - Poisson-implied P(stat>=1), Brier + log-loss\n"
          f"NEGATIVE d = variant BETTER CALIBRATED (more accurate anytime-TD probability)\n{bar}")
    print(f"{'stat':<16}{'wk n':>7}{'obs':>9}{'rate':>7}{'Brier base':>12}{'Brier var':>11}{'dBrier':>10}"
          f"{'95% CI':>22}{'W-L':>9}")
    for stat in TD_STATS:
        rows = calib_pairs.get(stat) or []
        if not rows:
            continue
        bb = np.array([r[0] for r in rows]); bv = np.array([r[1] for r in rows])
        w = np.array([r[4] for r in rows], float)
        rate = float(np.average([r[5] for r in rows], weights=w))
        base_b = float(np.average(bb, weights=w)); var_b = float(np.average(bv, weights=w))
        deltas = list(bv - bb)
        lo, hi = _boot_ci(deltas, list(w))
        wins = int((bv < bb).sum()); losses = int((bv > bb).sum())
        star = ' *' if (np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)) else '  '
        print(f"{stat:<16}{len(rows):>7}{int(w.sum()):>9}{rate:>7.3f}{base_b:>12.4f}{var_b:>11.4f}"
              f"{var_b - base_b:>+10.4f}  [{lo:+.4f},{hi:+.4f}]{star}{f'{wins}-{losses}':>9}")
    print(f"\n{'stat':<16}{'wk n':>7}{'LL base':>10}{'LL var':>10}{'dLogLoss':>11}{'95% CI':>22}{'W-L':>9}")
    for stat in TD_STATS:
        rows = calib_pairs.get(stat) or []
        if not rows:
            continue
        lb = np.array([r[2] for r in rows]); lv = np.array([r[3] for r in rows])
        w = np.array([r[4] for r in rows], float)
        base_ll = float(np.average(lb, weights=w)); var_ll = float(np.average(lv, weights=w))
        deltas = list(lv - lb)
        lo, hi = _boot_ci(deltas, list(w))
        wins = int((lv < lb).sum()); losses = int((lv > lb).sum())
        star = ' *' if (np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)) else '  '
        print(f"{stat:<16}{len(rows):>7}{base_ll:>10.4f}{var_ll:>10.4f}{var_ll - base_ll:>+11.4f}"
              f"  [{lo:+.4f},{hi:+.4f}]{star}{f'{wins}-{losses}':>9}")
    print("\n* = bootstrap 95% CI on the pooled weekly delta excludes zero.  'rate' = actual P(stat>=1) in the pool.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2021,2022,2023,2024,2025')
    ap.add_argument('--weeks', default='3-18')
    ap.add_argument('--scoring', default='Full PPR')
    ap.add_argument('--outliers-csv', default=None)
    a = ap.parse_args()
    years = [int(x) for x in a.years.replace(' ', '').split(',')]
    w0, w1 = (int(x) for x in a.weeks.split('-'))
    run(years, list(range(w0, w1 + 1)), a.scoring, a.outliers_csv)


if __name__ == '__main__':
    main()
