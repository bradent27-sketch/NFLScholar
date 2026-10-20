"""
Calibration audit C10 and C11 (2026-10-20): the two modifiers on the current-vs-prior blend weight, re-scored offline from a
stat-blend capture (scripts/capture_stat_blend_traces.py) with no builds.

    k_eff = K[pos, stat] * (hi - (hi - lo) * role_confidence) * (1 - reduction * role_change_confidence)
            (the last factor only for targets and rushing_attempts; RB uses reduction 0 under v2_role_change_by_stat)
    w     = games / (games + k_eff)

C10 varies (lo, hi) = K_EFFECTIVE_RANGE (shipped 0.7, 1.3): how far role confidence moves trust in this season's rate.
C11 varies the reduction (ROLE_CHANGE_K_REDUCTION, shipped 0.5) for QB/WR/TE: how much a detected role change speeds it up.

C10 moves every directly blended stat, so it is scored on points through the shipped late steps (same engine as
scripts/eval_stat_k_sweep.py: paired START pools, weekly bootstrap CI, leave-one-season-out). C11 moves only targets and
rushing attempts, which reach points through the capacity and vacancy steps this re-score does not redo, so it is read at
the stat level only (a harness run is the points test).

    python scripts/eval_blend_modifiers.py [--traces .sweeps/defense_capture.parquet]
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

import data.weekly_projections as wp  # noqa: E402
from scripts.eval_calibration_schemes import WEIGHTS  # noqa: E402
from scripts.eval_stat_k_sweep import drmse, late_steps, load_joined  # noqa: E402

ROLE_CHANGE_STATS = ('targets', 'rushing_attempts')
RANGES = ((1.3, 0.7), (1.15, 0.85), (1.0, 1.0), (0.85, 1.15), (0.7, 1.3), (0.5, 1.5), (0.3, 1.7))
REDUCTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
BLEND_STATS = ('passing_attempts', 'passing_completions', 'passing_yards', 'rushing_attempts', 'rushing_yards',
               'targets', 'receptions', 'receiving_yards')


def base_k(t):
    return np.array([wp.STAT_K_BY_POS.get(p, {}).get(s, wp.STAT_K.get(s, 3)) for p, s in zip(t['pos'], t['stat'])], float)


def k_eff(t, lo=0.7, hi=1.3, reduction=None):
    """k_eff for every trace row under a (lo, hi) range and a QB/WR/TE role-change reduction (None = shipped 0.5)."""
    conf = np.clip(t['role_confidence'].to_numpy(float), 0.0, 1.0)
    change = np.clip(np.nan_to_num(t['role_change_confidence'].to_numpy(float)), 0.0, 1.0)
    k = base_k(t) * (hi - (hi - lo) * conf)
    red = np.where(t['pos'].eq('RB'), wp.ROLE_CHANGE_K_REDUCTION_RB_CARRY,
                   wp.ROLE_CHANGE_K_REDUCTION if reduction is None else reduction)
    applies = t['stat'].isin(ROLE_CHANGE_STATS).to_numpy()
    return np.where(applies, k * (1.0 - red * change), k)


def new_final(t, k_new):
    """Each row's final stat when its blend weight uses ``k_new`` (rows with no games or a degenerate blend unchanged)."""
    g = t['current_games'].to_numpy(float)
    w = t['current_weight'].to_numpy(float)
    blended = t['blended_rate'].to_numpy(float)
    ok = (g > 0) & (w > 0) & (w < 1) & (blended > 1e-9) & np.isfinite(k_new) & (k_new > 0)
    w2 = np.where(ok, g / np.where(ok, g + k_new, 1.0), w)
    blended2 = w2 * np.nan_to_num(t['current_rate'].to_numpy(float)) + (1 - w2) * np.nan_to_num(t['prior_rate'].to_numpy(float))
    ratio = np.where(ok, blended2 / np.where(blended > 1e-9, blended, 1.0), 1.0)
    return t['final_projection'].to_numpy(float) + t['pre_vacancy_projection'].to_numpy(float) * (ratio - 1.0)


def board(d, t, final_new):
    """Points with each trace row's final stat replaced (top-end yardage re-applied to the pre-top-end stat)."""
    raw = d['raw'].to_numpy(float).copy()
    final = t['final_projection'].to_numpy(float)
    new = np.clip(final_new, 0.0, None)
    for stat, (positions, knot, share) in wp.TOP_END_YARDAGE.items():
        m = (t['stat'].eq(stat) & t['pos'].isin(positions)).to_numpy()
        if not m.any():
            continue
        rows = t['_row'].to_numpy()[m]
        pre = final[m] - np.nan_to_num(t['top_end_delta'].to_numpy(float)[m])
        pre2 = np.clip(pre + (final_new[m] - final[m]), 0.0, None)
        live = ((d['avail'].to_numpy(float)[rows] > wp.SIDELINED_AVAILABILITY) & (d['raw'].to_numpy(float)[rows] > 0.005)
                & (d['week'].to_numpy()[rows] >= wp.TOP_END_YARDAGE_FIRST_WEEK))
        new[m] = np.where(live & (pre2 > knot), pre2 - share * (pre2 - knot), pre2)
    np.add.at(raw, t['_row'].to_numpy(), t['stat'].map(WEIGHTS).fillna(0.0).to_numpy() * (new - final))
    return late_steps(d, raw) + d['_late_offset'].to_numpy(float)


def stat_rmse(t, final_new, stat, positions):
    m = (t['stat'].eq(stat) & t['pos'].isin(positions) & t['played'] & t['actual'].notna()).to_numpy()
    a = t['actual'].to_numpy(float)[m]
    return float(np.sqrt(((np.clip(final_new[m], 0, None) - a) ** 2).mean())), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', default='.sweeps/defense_capture.parquet')
    a = ap.parse_args()
    pd.set_option('display.width', 220)
    d, t = load_joined(a.traces)
    t = t[t['stat'].isin(BLEND_STATS)].reset_index(drop=True)
    g = t['current_games'].to_numpy(float)
    w = t['current_weight'].to_numpy(float)
    ok = (g > 0) & (w > 0) & (w < 1)
    traced = np.where(ok, g * (1 - w) / np.where(ok, w, 1.0), np.nan)
    rebuilt = k_eff(t)
    gap = np.abs(rebuilt - traced)[ok]
    print(f"rows {len(t):,}; k_eff rebuilt from the formula vs traced: median gap {np.median(gap):.4f}, "
          f"99th pct {np.percentile(gap, 99):.4f} (should be ~0)")
    base = board(d, t, t['final_projection'].to_numpy(float))

    print('\n=== C10: role-confidence range (lo, hi) on K; shipped (0.7, 1.3) ===')
    rows = []
    for lo, hi in RANGES:
        fn = new_final(t, k_eff(t, lo, hi))
        var = board(d, t, fn)
        r = drmse(d, base, var)
        row = {'range': f'({lo}, {hi})', 'dRMSE START-ALL': r['d'], 'ci': f"[{r['lo']:+.4f},{r['hi']:+.4f}]",
               'yrs better': f"{r['yrs_better']}/{r['yrs']}"}
        for pos in ('QB', 'RB', 'WR', 'TE'):
            row[pos] = drmse(d, base, var, pos)['d']
        rows.append(row)
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print('\n=== C11: role-change reduction for QB/WR/TE (targets, rushing attempts); shipped 0.5; stat-level RMSE ===')
    rows = []
    for red in REDUCTIONS:
        fn = new_final(t, k_eff(t, reduction=red))
        rec = {'reduction': red}
        for stat, positions in (('targets', ('WR', 'TE')), ('rushing_attempts', ('QB',))):
            rmse, n = stat_rmse(t, fn, stat, positions)
            rec[f'{stat} {"/".join(positions)} RMSE'] = rmse
        rows.append(rec)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    share = float((t['stat'].isin(ROLE_CHANGE_STATS) & (t['role_change_confidence'].fillna(0) > 0.05)).mean())
    print(f"(rows of targets/rushing attempts with a role-change confidence above 0.05: {share:.1%} of all blend rows)")


if __name__ == '__main__':
    main()
