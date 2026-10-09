"""Offline held-out test of a Questionable-tag production factor (2026-10-08).

Reads the calibration-lab dump (harness base arm, shipped `final` per player-week) and the official injury report, and
scores candidate factors the way harness v2 scores a flag: paired START pools over all rows, played rows, RMSE /
bias / pairwise, a week-bootstrap CI. Each season is predicted with factors fit on the OTHER four (leave-one-season-out),
so the number is out of sample even though the harness window and the fit window overlap.

Designs (all multiply the shipped final points of rows tagged Questionable, Availability > 0.01, final > 0):
  wr_pts       WR only, one factor on the points total          (the finding)
  all_pts      every position, one factor
  pos_pts      a factor per position, shrunk halfway to 1       (the finding without the WR-only choice)
  wr_stat      WR only, a factor per scored receiving stat       (receptions / yards / TDs) then re-scored
  wr_raw       WR only, a factor on the RAW total before the points lines and the anchor (a volume-level 'workload')

    python scripts/eval_questionable_factor.py --dump-path .sweeps/calibration_lab_v7.csv --week1-path .sweeps/calibration_lab_v7_wk1_ourlads.csv
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

import scripts.eval_calibration_schemes as ecs  # noqa: E402
from scripts.diag_questionable_effect import attach_reports  # noqa: E402

POS = ecs.POS


def ls_factor(pred, actual):
    """Least-squares factor through the origin, actual ~ f * pred."""
    den = float((pred * pred).sum())
    return float((pred * actual).sum() / den) if den > 0 else 1.0


def fit_factors(d, train, design):
    q = train & d['isq'].to_numpy() & d['played'].to_numpy()
    out = {}
    if design in ('wr_pts', 'wr_raw', 'wr_stat'):
        m = q & (d['pos'] == 'WR').to_numpy()
        col = 'raw' if design == 'wr_raw' else 'final'
        if design == 'wr_raw':
            # the factor on RAW that, through the shipped line and anchor, best matches actual: search it
            best = None
            for f in np.arange(0.70, 1.001, 0.01):
                p = scored_raw(d, m, f)
                err = float(((p - d['actual'].to_numpy()[m]) ** 2).sum())
                if best is None or err < best[0]:
                    best = (err, f)
            out['WR'] = best[1]
        else:
            out['WR'] = ls_factor(d[col].to_numpy(dtype=float)[m], d['actual'].to_numpy()[m])
    elif design == 'all_pts':
        m = q
        f = ls_factor(d['final'].to_numpy(dtype=float)[m], d['actual'].to_numpy()[m])
        out = {p: f for p in POS}
    elif design == 'pos_pts':
        for p in POS:
            m = q & (d['pos'] == p).to_numpy()
            f = ls_factor(d['final'].to_numpy(dtype=float)[m], d['actual'].to_numpy()[m]) if m.sum() >= 30 else 1.0
            out[p] = 1.0 + 0.5 * (f - 1.0)
    return out


def scored_raw(d, mask, f):
    """Final points of the masked rows if the RAW total were scaled by f first (line, sidelined rule, anchor re-applied)."""
    sub = d[mask]
    raw = sub['raw'].to_numpy(dtype=float) * f
    cal = np.empty(len(sub))
    for i, (pos, wk) in enumerate(zip(sub['pos'], sub['week'])):
        sl, ic = ecs._weekly_calibration_for(pos, int(wk))
        cal[i] = max(0.0, sl * raw[i] + ic)
    s = sub['pos'].map(ecs.SEASON_ANCHOR_S).to_numpy(dtype=float)
    ok = (~np.isnan(s) & sub['anchor_mean'].notna().to_numpy() & (sub['anchor_games'] >= ecs.SEASON_ANCHOR_MIN_GAMES).to_numpy()
          & (cal > 0.005))
    m = sub['anchor_mean'].to_numpy(dtype=float)
    out = cal.copy()
    out[ok] = np.clip(m[ok] + s[ok] * (cal[ok] - m[ok]), 0.0, None)
    return out


def apply_factors(d, factors, design):
    pred = d['final'].to_numpy(dtype=float).copy()
    q = d['isq'].to_numpy()
    for pos, f in factors.items():
        m = q & (d['pos'] == pos).to_numpy()
        if design == 'wr_raw':
            pred[m] = scored_raw(d, m, f)
        else:
            pred[m] = np.round(pred[m] * f, 2)
    return pred


def stat_design(d, train):
    """WR per-stat factors: receptions, receiving yards, receiving TDs, from the same Q START-free pool."""
    out = {}
    q = train & d['isq'].to_numpy() & d['played'].to_numpy() & (d['pos'] == 'WR').to_numpy()
    for s in ('receptions', 'receiving_yards', 'receiving_tds'):
        out[s] = ls_factor(d[f'p_{s}'].to_numpy(dtype=float)[q], d[f'a_{s}'].to_numpy(dtype=float)[q])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-path', default=os.path.join('.sweeps', 'calibration_lab_v7.csv'))
    ap.add_argument('--week1-path', default='')
    a = ap.parse_args()
    d = ecs.load_dump(a.dump_path, a.week1_path)
    d = attach_reports(d)
    d['isq'] = (d['report_status'].fillna('') == 'Questionable').to_numpy() & (d['avail'] > 0.01).to_numpy() & (d['final'] > 0.005).to_numpy()
    d = d.reset_index(drop=True)
    d['wk_key'] = d['year'] * 100 + d['week']
    years = sorted(d['year'].unique())
    base = d['final'].to_numpy(dtype=float)
    print(f"Q-tagged live rows: {int(d['isq'].sum())}  (WR {int((d['isq'] & (d['pos'] == 'WR')).sum())}); join misses are untagged")

    preds, fits = {}, {}
    for design in ('wr_pts', 'all_pts', 'pos_pts', 'wr_raw'):
        pred = base.copy()
        fits[design] = {}
        for y in years:
            test = (d['year'] == y).to_numpy()
            f = fit_factors(d, ~test, design)
            fits[design][y] = f
            p = apply_factors(d, f, design)
            pred[test] = p[test]
        preds[design] = pred
    # per-stat WR design
    pred = base.copy()
    fits['wr_stat'] = {}
    for y in years:
        test = (d['year'] == y).to_numpy()
        f = stat_design(d, ~test)
        fits['wr_stat'][y] = f
        m = test & d['isq'].to_numpy() & (d['pos'] == 'WR').to_numpy()
        line = {s: d[f'p_{s}'].to_numpy(dtype=float).copy() for s in ecs.WEIGHTS}
        for s, fac in f.items():
            line[s][m] = line[s][m] * fac
        pts = ecs.score_points(d, line)
        # keep the shipped points pipeline and shift it by the change this stat edit makes to the raw score
        delta = pts - ecs.score_points(d, {s: d[f'p_{s}'].to_numpy(dtype=float) for s in ecs.WEIGHTS})
        pred[m] = np.clip(base[m] + delta[m], 0.0, None)
    preds['wr_stat'] = pred

    print('\nfactors fit without each season (WR | all | per-pos shrunk):')
    for y in years:
        print(f"  without {y}: WR {fits['wr_pts'][y]['WR']:.3f}   all {fits['all_pts'][y]['WR']:.3f}   "
              + ' '.join(f"{p} {fits['pos_pts'][y][p]:.3f}" for p in POS)
              + f"   wr_raw {fits['wr_raw'][y]['WR']:.2f}   wr_stat " + ' '.join(f"{s[:8]} {v:.2f}" for s, v in fits['wr_stat'][y].items()))

    for label, window in (('HARNESS WINDOW 2022-2025 wk3-17', d['year'].between(2022, 2025) & d['week'].between(3, 17)),
                          ('ALL SEASONS 2021-2025 wk3-17', d['week'].between(3, 17)),
                          ('WEEKS 1-17 2021-2025', d['week'].between(1, 17))):
        ecs.report(d, preds, window, label)
    ecs.per_year(d, preds, d['week'].between(3, 17))

    # on the Q rows themselves
    print('\nQ-tagged rows only (played), START pool by the shipped board:')
    st = ecs.top_n_mask(d, base) & d['isq'].to_numpy() & d['played'].to_numpy() & d['week'].between(3, 17).to_numpy()
    rows = []
    for pos in POS + ('ALL',):
        m = st & ((d['pos'] == pos).to_numpy() if pos != 'ALL' else True)
        if m.sum() < 20:
            continue
        r = {'pos': pos, 'n': int(m.sum()), 'RMSE shipped': round(float(np.sqrt(((base[m] - d['actual'].to_numpy()[m]) ** 2).mean())), 3),
             'bias shipped': round(float((base[m] - d['actual'].to_numpy()[m]).mean()), 2)}
        for design, p in preds.items():
            r[f'RMSE {design}'] = round(float(np.sqrt(((p[m] - d['actual'].to_numpy()[m]) ** 2).mean())), 3)
            r[f'bias {design}'] = round(float((p[m] - d['actual'].to_numpy()[m]).mean()), 2)
        rows.append(r)
    pd.set_option('display.width', 250)
    print(pd.DataFrame(rows).to_string(index=False))

    # final factors on all seasons
    print('\nfactors fit on all five seasons:')
    print('  WR pts', round(fit_factors(d, np.ones(len(d), dtype=bool), 'wr_pts')['WR'], 3),
          '  all pts', round(fit_factors(d, np.ones(len(d), dtype=bool), 'all_pts')['WR'], 3),
          '  per-pos shrunk', {k: round(v, 3) for k, v in fit_factors(d, np.ones(len(d), dtype=bool), 'pos_pts').items()},
          '  wr_raw', round(fit_factors(d, np.ones(len(d), dtype=bool), 'wr_raw')['WR'], 2),
          '  wr_stat', {k: round(v, 3) for k, v in stat_design(d, np.ones(len(d), dtype=bool)).items()})


if __name__ == '__main__':
    main()
