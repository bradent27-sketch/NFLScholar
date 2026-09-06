"""TE2 buried-vet dock backtest (2026-09-04).

apply_buried_veteran_dock docks a PROVEN vet (prior snap share > 0.20) whom
the CURRENT depth chart now lists as a backup. For WR the dock fires at
Ourlads slot rank 2 (RECEIVER_BURIED_VET_BACKUP_SLOT_RANK). For TE it fires
one slot later, at rank 3 (RECEIVER_BURIED_VET_BACKUP_SLOT_RANK_TE) - a
charted TE-2 is exempted entirely, on the theory that TE-2 is often the real
receiving tight end in heavy-12-personnel offenses (Gesicki-behind-Sample).

User's suspicion (2026-09-04): that TE-2 exemption may have been naive - it
means a proven vet TE now buried at TE-2 on a NEW team/situation gets ZERO
role discount at cold start, purely because "TE2 can be a real role
somewhere" is true in general, not because it's true for THIS team.

This backtests dropping the exemption: RECEIVER_BURIED_VET_BACKUP_SLOT_RANK_TE
3 -> 2 (TE now docked exactly like WR at slot rank 2), scored ONLY at cold
start (Week 1, where depth_chart_decay=1.0 and the effect is strongest) on
the years with a frozen pre-Week-1 Ourlads archive. Both base and variant
carry v2_historical_ourlads so the historical chart is actually read (plain
DEFAULT_FEATURES does not consult the frozen archive - see
weekly_rankings_backlog.md's "inert" precedent for why this matters).

Because the dock only fires for a specific, fairly rare case (a proven vet
NOW charted at TE rank 2), n is small - this also writes a per-player ledger
of every case where the two variants actually produced a different number,
so the verdict can be read off real cases, not just an aggregate that a
single big miss could dominate.

    python scripts/sweep_te_buried_vet_slot.py --years 2022,2023,2024,2025 --weeks 1
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

FEATS = frozenset(DEFAULT_FEATURES | {'v2_historical_ourlads'})
SCOPES = [('ALL', None, False), ('TE', 'TE', False), ('START-TE', 'TE', True),
          ('WR', 'WR', False), ('START-WR', 'WR', True),
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


def run(years, weeks, scoring, ledger_csv):
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
                year, week, scoring, as_of_week=week, apply_injury=False, features=FEATS)
            if base.empty:
                print(f"{year} w{week}: base empty ({bmeta.get('reason')})")
                continue
            wp.RECEIVER_BURIED_VET_BACKUP_SLOT_RANK_TE = 2
            build_weekly_projections.clear()
            var, vmeta = build_weekly_projections(
                year, week, scoring, as_of_week=week, apply_injury=False, features=FEATS)
            wp.RECEIVER_BURIED_VET_BACKUP_SLOT_RANK_TE = 3
            build_weekly_projections.clear()
            if var.empty:
                print(f"{year} w{week}: variant empty ({vmeta.get('reason')})")
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
                bd = _scope_df(b.reset_index(), 'TE', True).set_index('Player')
                common = bd.index.intersection(v.index).intersection(sa.index)
                if len(common) < 3:
                    continue
                act = pd.to_numeric(sa.loc[common, stat], errors='coerce')
                eb = (pd.to_numeric(bd.loc[common, stat], errors='coerce') - act).abs().mean()
                ev = (pd.to_numeric(v.loc[common, stat], errors='coerce') - act).abs().mean()
                if np.isfinite(eb) and np.isfinite(ev):
                    stat_pairs[stat].append((float(eb), float(ev), len(common)))

            # Ledger every TE actually touched by the constant change.
            te_common = b.index[(b['Pos'] == 'TE')].intersection(v.index)
            for player in te_common:
                bp = float(b.loc[player, 'Model Proj Pts'])
                vp = float(v.loc[player, 'Model Proj Pts'])
                if abs(vp - bp) < 1e-6:
                    continue
                av = float(actual.get(player, np.nan))
                ledger.append(dict(
                    year=year, week=week, player=player,
                    base_proj=round(bp, 2), var_proj=round(vp, 2), shift=round(vp - bp, 3),
                    actual=round(av, 2) if np.isfinite(av) else None,
                    base_error=round(abs(bp - av), 2) if np.isfinite(av) else None,
                    var_error=round(abs(vp - av), 2) if np.isfinite(av) else None,
                    base_targets=round(float(b.loc[player, 'targets']), 2) if 'targets' in b.columns else None,
                    var_targets=round(float(v.loc[player, 'targets']), 2) if 'targets' in v.columns else None,
                ))
            print(f"{year} w{week}: pool {len(pool)}  TE rows touched this week: "
                  f"{sum(1 for r in ledger if r['year'] == year and r['week'] == week)}", flush=True)

    _report(scope_pairs, stat_pairs, ledger, years, weeks)
    if ledger_csv and ledger:
        os.makedirs(os.path.dirname(ledger_csv) or '.', exist_ok=True)
        pd.DataFrame(ledger).sort_values(['year', 'week']).to_csv(ledger_csv, index=False)
        print(f"\nwrote {ledger_csv}  ({len(ledger)} touched TE player-weeks)")
    elif ledger_csv:
        print(f"\nno TE player-weeks were touched by the constant change - "
              f"the dock's gate (proven vet now charted TE-2) never fired in this window.")


def _report(scope_pairs, stat_pairs, ledger, years, weeks):
    bar = '=' * 92
    print(f"\n{bar}\nTE BURIED-VET SLOT RANK 3->2 (dock TE-2 like WR-2)   years={years} weeks={weeks}\n{bar}")
    print("dMAE = variant - base.  NEGATIVE = docking TE-2 is MORE accurate.\n")
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

    print(f"\n{'stat (startable TE)':<24}{'wk n':>7}{'obs':>9}{'MAE base':>11}{'MAE var':>11}{'dMAE':>10}{'95% CI':>22}{'W-L':>9}")
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

    print(f"\n{bar}\nEVERY TE PLAYER-WEEK ACTUALLY TOUCHED BY THE CHANGE ({len(ledger)} total)\n{bar}")
    if not ledger:
        print("none - the dock's gate (proven vet, prior share>0.20, now charted TE rank 2) never fired.")
    else:
        for r in sorted(ledger, key=lambda r: (r['year'], r['week'])):
            print(f"  {r['year']} w{r['week']:<3} {r['player']:<24} base={r['base_proj']:>6.2f} "
                  f"var={r['var_proj']:>6.2f} shift={r['shift']:>+6.2f}  actual={r['actual']}"
                  f"  base_err={r['base_error']}  var_err={r['var_error']}")
    print("\n* = bootstrap 95% CI on the pooled weekly dMAE excludes zero.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2022,2023,2024,2025')
    ap.add_argument('--weeks', default='1-1')
    ap.add_argument('--scoring', default='Full PPR')
    ap.add_argument('--ledger-csv', default='.sweeps/te_buried_vet_slot_ledger.csv')
    a = ap.parse_args()
    years = [int(x) for x in a.years.replace(' ', '').split(',')]
    w0, w1 = (int(x) for x in a.weeks.split('-'))
    run(years, list(range(w0, w1 + 1)), a.scoring, a.ledger_csv)


if __name__ == '__main__':
    main()
