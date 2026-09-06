"""v2_receiver_cold_start_vacancy backtest (2026-09-04).

Live-board case that surfaced this: Green Bay's 2025 WR corps (Doubs 77.6%,
Watson 67.8%, Golden 53.2%, Reed 53.0%, Wicks 47.2%, Williams 11.5%) summed
to ~310% of team snap share. Doubs and Wicks are both gone for 2026. The
cold-start board's remaining corps sums to only ~160% - the vacated role
simply evaporates instead of flowing to Watson/Golden/Reed, because
redistribute_v2_vacated_usage only reacts to a player still IN the pool and
marked OUT by the live injury feed; a player who left the roster entirely
never appears in the pool at all.

v2_receiver_cold_start_vacancy (apply_cold_start_receiver_vacancy) finds,
per team/position at cold start, any player present in last season's role
reference who is absent from this season's pool entirely, and redistributes
RECEIVER_COLD_START_VACANCY_SURVIVAL (0.70) of his prior share to the
remaining WR/TE corps, weighted by each recipient's own current share.

Both base and variant carry v2_historical_ourlads so the frozen pre-Week-1
archive and prior-season role reference are actually available for a
historical year. Week 1 only (cold start).

    python scripts/sweep_receiver_cold_start_vacancy.py --years 2022,2023,2024,2025 --weeks 1
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

BASE_FEATS = frozenset(DEFAULT_FEATURES | {'v2_historical_ourlads'})
VAR_FEATS = frozenset(BASE_FEATS | {'v2_receiver_cold_start_vacancy'})
SCOPES = [('ALL', None, False), ('WR', 'WR', False), ('TE', 'TE', False),
          ('START-WR', 'WR', True), ('START-TE', 'TE', True),
          ('START-QB', 'QB', True), ('START-RB', 'RB', True)]
PER_STAT = ('targets', 'receptions', 'receiving_yards')


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


def _stat_actuals(stats_df, name_col, week):
    rows = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week]
    present = [s for s in PER_STAT if s in rows.columns]
    return rows.groupby(name_col, observed=True)[present].sum()


def run(years, weeks, scoring):
    scoring_col = 'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'
    scope_pairs = {s[0]: [] for s in SCOPES}
    stat_pairs = {s: [] for s in PER_STAT}
    ledger = []

    for year in years:
        stats_df, _tc, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns:
            print(f"{year}: no weekly data, skipped", flush=True)
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                continue
            build_weekly_projections.clear()
            base, bmeta = build_weekly_projections(
                year, week, scoring, as_of_week=week, apply_injury=False, features=BASE_FEATS)
            build_weekly_projections.clear()
            var, vmeta = build_weekly_projections(
                year, week, scoring, as_of_week=week, apply_injury=False, features=VAR_FEATS)
            if base.empty or var.empty:
                print(f"{year} w{week}: empty ({bmeta.get('reason') or vmeta.get('reason')})")
                continue
            pool = sorted(set(base['Player']) & set(var['Player']))
            if len(pool) < 20:
                continue
            b = base[base['Player'].isin(pool)].set_index('Player')
            v = var[var['Player'].isin(pool)].set_index('Player')

            for scope, pos, startable in SCOPES:
                bd = _scope_df(b.reset_index(), pos, startable)
                vd = _scope_df(v.reset_index(), pos, startable)
                mb = _metrics(pd.Series(bd['Model Proj Pts'].to_numpy(), index=bd['Player']), actual)
                mv = _metrics(pd.Series(vd['Model Proj Pts'].to_numpy(), index=vd['Player']), actual)
                if mb and mv:
                    scope_pairs[scope].append((mb, mv))

            sa = _stat_actuals(stats_df, name_col, week)
            for stat in PER_STAT:
                if stat not in b.columns or stat not in sa.columns:
                    continue
                bd = pd.concat([_scope_df(b.reset_index(), p, True) for p in ('WR', 'TE')]).set_index('Player')
                common = bd.index.intersection(v.index).intersection(sa.index)
                if len(common) < 5:
                    continue
                act = pd.to_numeric(sa.loc[common, stat], errors='coerce')
                eb = (pd.to_numeric(bd.loc[common, stat], errors='coerce') - act).abs().mean()
                ev = (pd.to_numeric(v.loc[common, stat], errors='coerce') - act).abs().mean()
                if np.isfinite(eb) and np.isfinite(ev):
                    stat_pairs[stat].append((float(eb), float(ev), len(common)))

            touched = b.index[(b['Pos'].isin(['WR', 'TE']))].intersection(v.index)
            for player in touched:
                bp = float(b.loc[player, 'Model Proj Pts'])
                vp = float(v.loc[player, 'Model Proj Pts'])
                if abs(vp - bp) < 1e-6:
                    continue
                av = float(actual.get(player, np.nan))
                ledger.append(dict(
                    year=year, week=week, player=player, pos=str(b.loc[player, 'Pos']),
                    team=str(b.loc[player, 'Team']) if 'Team' in b.columns else '',
                    base_proj=round(bp, 2), var_proj=round(vp, 2), shift=round(vp - bp, 3),
                    actual=round(av, 2) if np.isfinite(av) else None,
                    base_error=round(abs(bp - av), 2) if np.isfinite(av) else None,
                    var_error=round(abs(vp - av), 2) if np.isfinite(av) else None,
                ))
            print(f"{year} w{week}: pool {len(pool)}  touched {sum(1 for r in ledger if r['year'] == year and r['week'] == week)}",
                  flush=True)

    _report(scope_pairs, stat_pairs, ledger, years, weeks)


def _report(scope_pairs, stat_pairs, ledger, years, weeks):
    bar = '=' * 92
    print(f"\n{bar}\nv2_receiver_cold_start_vacancy   years={years} weeks={weeks}\n{bar}")
    print("dMAE = variant - base.  NEGATIVE = redistributing the vacancy is MORE accurate.\n")
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
        wins = sum(1 for d in deltas if d < 0); losses = sum(1 for d in deltas if d > 0)
        star = ' *' if (np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)) else '  '
        print(f"{scope:<12}{nt:>8}{base_mae:>11.3f}{var_mae:>11.3f}{var_mae - base_mae:>+10.3f}"
              f"  [{lo:+.3f},{hi:+.3f}]{star}{f'{wins}-{losses}':>10}{_sign_p(wins, losses):>9.3f}")

    print(f"\n{'stat (startable WR/TE)':<24}{'wk n':>7}{'obs':>9}{'MAE base':>11}{'MAE var':>11}{'dMAE':>10}{'95% CI':>22}{'W-L':>9}")
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

    print(f"\n{bar}\nEVERY PLAYER-WEEK TOUCHED BY THE FLAG ({len(ledger)} total)\n{bar}")
    if not ledger:
        print("none - no team had a qualifying departure (prior share >= "
              "RECEIVER_COLD_START_VACANCY_MIN_SHARE) this window.")
    else:
        for r in sorted(ledger, key=lambda r: (r['year'], r['week'])):
            print(f"  {r['year']} w{r['week']:<3} {r['team']:<4} {r['pos']:<3} {r['player']:<24} "
                  f"base={r['base_proj']:>6.2f} var={r['var_proj']:>6.2f} shift={r['shift']:>+6.2f}"
                  f"  actual={r['actual']}  base_err={r['base_error']}  var_err={r['var_error']}")
    print("\n* = bootstrap 95% CI on the pooled weekly dMAE excludes zero.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2022,2023,2024,2025')
    ap.add_argument('--weeks', default='1-1')
    ap.add_argument('--scoring', default='Full PPR')
    a = ap.parse_args()
    years = [int(x) for x in a.years.replace(' ', '').split(',')]
    w0, w1 = (int(x) for x in a.weeks.split('-'))
    run(years, list(range(w0, w1 + 1)), a.scoring)


if __name__ == '__main__':
    main()
