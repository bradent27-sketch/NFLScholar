"""TE buried-vet dock GRADED grid (2026-09-07).

Follow-up to scripts/sweep_te_buried_vet_slot.py, which flipped
RECEIVER_BURIED_VET_BACKUP_SLOT_RANK_TE 3 -> 2 (dock TE-2 exactly like WR-2:
"at slot" keeps 0.5, "deeper" is hard-capped to ~0) and found it a wash on
START-TE bought at a real WR / whole-board cost.

User's next ask: make the dock GRADED instead of binary, and sweep it. Two
dials (data.weekly_projections):
  RECEIVER_BURIED_VET_KEEP_FRACTION_TE       - what a charted TE-2 proven vet keeps
  RECEIVER_BURIED_VET_DEEP_KEEP_FRACTION_TE  - what a charted TE-3+ proven vet keeps
                                              (of its OWN current share, not ~0)
This runs the 3x3 grid  TE2 in {0.8, 0.6, 0.4}  x  TE3 in {0.6, 0.4, 0.2}
(9 combinations), every one with SLOT_RANK_TE = 2 so both tiers actually
engage, scored at cold start (Week 1) on the years with a frozen pre-Week-1
Ourlads archive. Base (the shipped exemption: SLOT_RANK_TE = 3, TE-2 docked
by nothing) is built ONCE per year and reused across all nine.

Both base and every variant carry v2_historical_ourlads so the frozen chart
is actually read (plain DEFAULT_FEATURES does not consult the archive).

    python scripts/sweep_te_buried_vet_slot_grid.py --years 2022,2023,2024,2025 --weeks 1

Reads off `dMAE = variant - base`; NEGATIVE = that combo is MORE accurate.
Writes a per-(combo, player-week) ledger so a combo can be judged on real
cases, not just an aggregate a single blow-up could dominate.
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
          ('WR', 'WR', False), ('START-WR', 'WR', True)]
PER_STAT = ('targets', 'receptions', 'receiving_yards')
TE2_KEEPS = (0.8, 0.6, 0.4)
TE3_KEEPS = (0.6, 0.4, 0.2)


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


def _build(year, week, scoring):
    build_weekly_projections.clear()
    board, meta = build_weekly_projections(
        year, week, scoring, as_of_week=week, apply_injury=False, features=FEATS)
    return board, meta


def _set_variant(te2_keep, te3_keep):
    wp.RECEIVER_BURIED_VET_BACKUP_SLOT_RANK_TE = 2
    wp.RECEIVER_BURIED_VET_KEEP_FRACTION_TE = te2_keep
    wp.RECEIVER_BURIED_VET_DEEP_KEEP_FRACTION_TE = te3_keep


def _reset_shipped():
    wp.RECEIVER_BURIED_VET_BACKUP_SLOT_RANK_TE = 3
    wp.RECEIVER_BURIED_VET_KEEP_FRACTION_TE = None
    wp.RECEIVER_BURIED_VET_DEEP_KEEP_FRACTION_TE = None


def run(years, weeks, scoring, ledger_csv):
    # combo -> {scope -> [(base_metrics, var_metrics) per week]}, plus per-stat.
    combos = [(a, b) for a in TE2_KEEPS for b in TE3_KEEPS]
    scope_pairs = {c: {s[0]: [] for s in SCOPES} for c in combos}
    stat_pairs = {c: {s: [] for s in PER_STAT} for c in combos}
    ledger = []

    for year in years:
        stats_df, _tc, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns:
            print(f"{year}: no weekly data, skipped", flush=True)
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week,
                                    'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points')
            if actual.empty:
                continue
            sa = _stat_actuals(stats_df, name_col, week)

            _reset_shipped()
            base, bmeta = _build(year, week, scoring)
            if base.empty:
                print(f"{year} w{week}: base empty ({bmeta.get('reason')})", flush=True)
                continue
            b_all = base.set_index('Player')

            for (te2_keep, te3_keep) in combos:
                _set_variant(te2_keep, te3_keep)
                var, vmeta = _build(year, week, scoring)
                _reset_shipped()
                build_weekly_projections.clear()
                if var.empty:
                    print(f"{year} w{week} [{te2_keep}/{te3_keep}]: variant empty", flush=True)
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
                        scope_pairs[(te2_keep, te3_keep)][scope].append((mb, mv))

                bd = _scope_df(b.reset_index(), 'TE', True).set_index('Player')
                for stat in PER_STAT:
                    if stat not in b.columns or stat not in sa.columns:
                        continue
                    common = bd.index.intersection(v.index).intersection(sa.index)
                    if len(common) < 3:
                        continue
                    act = pd.to_numeric(sa.loc[common, stat], errors='coerce')
                    eb = (pd.to_numeric(bd.loc[common, stat], errors='coerce') - act).abs().mean()
                    ev = (pd.to_numeric(v.loc[common, stat], errors='coerce') - act).abs().mean()
                    if np.isfinite(eb) and np.isfinite(ev):
                        stat_pairs[(te2_keep, te3_keep)][stat].append((float(eb), float(ev), len(common)))

                te_common = b.index[(b['Pos'] == 'TE')].intersection(v.index)
                for player in te_common:
                    bp = float(b.loc[player, 'Model Proj Pts'])
                    vpx = float(v.loc[player, 'Model Proj Pts'])
                    if abs(vpx - bp) < 1e-6:
                        continue
                    av = float(actual.get(player, np.nan))
                    ledger.append(dict(
                        te2_keep=te2_keep, te3_keep=te3_keep, year=year, week=week, player=player,
                        base_proj=round(bp, 2), var_proj=round(vpx, 2), shift=round(vpx - bp, 3),
                        actual=round(av, 2) if np.isfinite(av) else None,
                        base_error=round(abs(bp - av), 2) if np.isfinite(av) else None,
                        var_error=round(abs(vpx - av), 2) if np.isfinite(av) else None))
            print(f"{year} w{week}: base pool {len(base)}  ran {len(combos)} combos", flush=True)

    _report(scope_pairs, stat_pairs, combos, years, weeks)
    if ledger_csv and ledger:
        os.makedirs(os.path.dirname(ledger_csv) or '.', exist_ok=True)
        pd.DataFrame(ledger).sort_values(['te2_keep', 'te3_keep', 'year', 'week']).to_csv(ledger_csv, index=False)
        print(f"\nwrote {ledger_csv}  ({len(ledger)} touched (combo, TE, week) rows)")


def _agg(pairs):
    """(n, base_mae, var_mae, dmae, ci_lo, ci_hi, wins, losses) for one scope's week list."""
    if not pairs:
        return None
    mb = [p[0] for p in pairs]; mv = [p[1] for p in pairs]
    nt = sum(m['n'] for m in mb)
    base_mae, var_mae = _weighted(mb, 'mae'), _weighted(mv, 'mae')
    deltas = [x['mae'] - y['mae'] for y, x in zip(mb, mv)]
    wts = [y['n'] for y in mb]
    lo, hi = _boot_ci(deltas, wts)
    wins = sum(1 for d in deltas if d < 0); losses = sum(1 for d in deltas if d > 0)
    return nt, base_mae, var_mae, var_mae - base_mae, lo, hi, wins, losses


def _report(scope_pairs, stat_pairs, combos, years, weeks):
    bar = '=' * 100
    print(f"\n{bar}\nTE BURIED-VET GRADED DOCK GRID   years={years} weeks={weeks}\n{bar}")
    print("dMAE = variant - base.  NEGATIVE = graded dock is MORE accurate.  "
          "* = bootstrap 95% CI excludes 0.\n")
    hdr = f"{'TE2 keep':>9}{'TE3 keep':>10} | "
    for scope, _p, _s in SCOPES:
        hdr += f"{scope:>13}"
    print(hdr)
    print('-' * len(hdr))
    best = None
    for combo in combos:
        row = f"{combo[0]:>9.2f}{combo[1]:>10.2f} | "
        st_te = None
        for scope, _p, _s in SCOPES:
            a = _agg(scope_pairs[combo][scope])
            if a is None:
                row += f"{'-':>13}"
                continue
            _, _, _, dmae, lo, hi, w, l = a
            star = '*' if (np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)) else ' '
            row += f"{dmae:>+11.3f}{star} "
            if scope == 'START-TE':
                st_te = dmae
        print(row)
        if st_te is not None and (best is None or st_te < best[1]):
            best = (combo, st_te)

    print(f"\nPer-stat startable-TE dMAE (targets / receptions / receiving_yards):")
    print(f"{'TE2':>6}{'TE3':>6} | " + ''.join(f"{s:>18}" for s in PER_STAT))
    for combo in combos:
        row = f"{combo[0]:>6.2f}{combo[1]:>6.2f} | "
        for stat in PER_STAT:
            rows = stat_pairs[combo][stat]
            if not rows:
                row += f"{'-':>18}"
                continue
            eb = np.array([r[0] for r in rows]); ev = np.array([r[1] for r in rows])
            w = np.array([r[2] for r in rows], float)
            row += f"{float(np.average(ev, weights=w) - np.average(eb, weights=w)):>+18.3f}"
        print(row)

    if best is not None:
        print(f"\nBest START-TE dMAE: TE2 keep {best[0][0]}, TE3 keep {best[0][1]}  "
              f"(dMAE {best[1]:+.3f}). Read its WR / ALL columns above before shipping - "
              "the binary 3->2 test's START-TE gain came with a significant WR cost.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2022,2023,2024,2025')
    ap.add_argument('--weeks', default='1-1')
    ap.add_argument('--scoring', default='Full PPR')
    ap.add_argument('--ledger-csv', default='.sweeps/te_buried_vet_slot_grid_ledger.csv')
    a = ap.parse_args()
    years = [int(x) for x in a.years.replace(' ', '').split(',')]
    w0, w1 = (int(x) for x in a.weeks.split('-')) if '-' in a.weeks else (int(a.weeks), int(a.weeks))
    run(years, list(range(w0, w1 + 1)), a.scoring, a.ledger_csv)


if __name__ == '__main__':
    main()
