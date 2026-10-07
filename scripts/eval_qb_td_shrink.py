"""
How much would a slower / flatter QB passing-TD estimate help QB points? (2026-10-06, after the v2_qb_passing_td_k
harness - K=15 in the in-season blend - came back START-QB -0.010, CI spanning 0.)

The QB's projected passing TDs regress on actual with slope ~0.55 in EVERY season phase, weeks 1-2 included, so it is
not only the in-season blend reacting too fast: the projected spread is about twice what the data supports. This asks
what the ceiling is if the TD count is pulled toward a stable anchor:

    new_td = anchor + lam * (projected_td - anchor)
    anchor 'att'  = fit-years league TD per attempt * projected attempts
    anchor 'yds'  = fit-years league TD per passing yard * projected passing yards
    anchor 'own'  = the model's own TD number for this QB with the in-season component removed is NOT available in
                    the dump, so it is not tested here.

Raw points move by 4 * (new_td - projected_td); then EITHER the shipped QB calibration line is applied unchanged, or
the line is refit (half strength, the shipping scheme) on the fit years - because compressing the TD part makes the
points line's own compression too strong. lam is picked on the fit years and scored on held-out years on the
harness's startable pool (played rows, weeks 3-17), against the shipped line and no TD change.

    python scripts/eval_qb_td_shrink.py --dump-path .sweeps/seasonal_calibration_allrows_v6.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import scripts.fit_seasonal_calibration as fsc  # noqa: E402
from data.weekly_projections import WEEKLY_CALIBRATION  # noqa: E402
from scripts.harness_v2 import pairwise_acc  # noqa: E402

LAMS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0)
SPLITS = [('fit 21-23 / test 24-25', (2021, 2022, 2023), (2024, 2025)),
          ('fit 21-24 / test 25', (2021, 2022, 2023, 2024), (2025,)),
          ('fit 22-25 / test 21', (2022, 2023, 2024, 2025), (2021,)),
          ('fit 21,23,25 / test 22,24', (2021, 2023, 2025), (2022, 2024))]


def anchor_rate(frame, kind):
    q = frame[(frame['pos'] == 'QB') & frame['played'] & (frame['p_passing_attempts'] >= 15)]
    if kind == 'att':
        return float(q['a_passing_tds'].sum() / q['a_passing_attempts'].sum())
    return float(q['a_passing_tds'].sum() / q['a_passing_yards'].sum())


def shrunk_raw(q, lam, kind, rate):
    base = rate * (q['p_passing_attempts'] if kind == 'att' else q['p_passing_yards'])
    new_td = base + lam * (q['p_passing_tds'] - base)
    return q['raw'] + 4.0 * (new_td - q['p_passing_tds'])


def score(raw, actual, line):
    pred = fsc._apply(raw, line)
    return pred


def rmse(pred, actual):
    return float(np.sqrt(((pred - actual) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-path', default='.sweeps/seasonal_calibration_allrows_v6.csv')
    a = ap.parse_args()
    d = pd.read_csv(a.dump_path)
    d['played'] = d['played'].astype(bool)
    d['start'] = fsc._startable_mask(d)
    for c in d.columns:
        if c.startswith('p_') or c.startswith('a_'):
            d[c] = pd.to_numeric(d[c], errors='coerce')
    q = d[(d['pos'] == 'QB') & d['played'] & d['week'].between(3, 17)].copy()
    q = q[q['p_passing_attempts'].notna() & q['p_passing_tds'].notna()]
    pool = q[q['start']]
    ship = WEEKLY_CALIBRATION['QB']
    pd.set_option('display.width', 220)
    print(f"QB played rows {len(q)}, START pool {len(pool)}; shipped QB line {ship}")

    for kind in ('att', 'yds'):
        print(f"\n===== anchor = league TD per {'attempt' if kind == 'att' else 'passing yard'} =====")
        verdict = {'ship': [], 'refit': []}
        for name, fy, ty in SPLITS:
            fit_all = d[d['year'].isin(fy)]
            rate = anchor_rate(fit_all, kind)
            fitq, testq = q[q['year'].isin(fy)], q[q['year'].isin(ty)]
            fit_pool, test_pool = fitq[fitq['start']], testq[testq['start']]
            base_pred = rmse(fsc._apply(test_pool['raw'], ship), test_pool['actual'])
            # lam picked on the fit years, once per line scheme
            out = {}
            for scheme in ('ship', 'refit'):
                best, best_r = None, None
                for lam in LAMS:
                    raw_f = shrunk_raw(fit_pool, lam, kind, rate)
                    line = ship if scheme == 'ship' else fsc._half_fit_or_identity(
                        pd.DataFrame({'raw': shrunk_raw(fitq, lam, kind, rate), 'actual': fitq['actual']}))
                    r = rmse(fsc._apply(raw_f, line), fit_pool['actual'])
                    if best_r is None or r < best_r:
                        best, best_r = (lam, line), r
                lam, line = best
                raw_t = shrunk_raw(test_pool, lam, kind, rate)
                out[scheme] = (lam, rmse(fsc._apply(raw_t, line), test_pool['actual']) - base_pred, line)
                verdict[scheme].append(out[scheme][1])
            print(f"{name}: rate {rate:.4f}  shipped-line RMSE {base_pred:.3f} | "
                  f"ship-line lam {out['ship'][0]:.1f} d {out['ship'][1]:+.4f} | "
                  f"refit-line lam {out['refit'][0]:.1f} d {out['refit'][1]:+.4f} (line {tuple(round(x, 3) for x in out['refit'][2])})")
        for k, v in verdict.items():
            print(f"  {k}: START-QB RMSE change by split {[round(x, 4) for x in v]} -> "
                  f"{'better in every split' if all(x < 0 for x in v) else 'not every split'}")

    rate = anchor_rate(d, 'att')
    print(f"\nin-sample on 2021-2025 (anchor TD/att {rate:.4f}), START-QB RMSE by lam, shipped line / refit line:")
    rows = []
    for lam in LAMS:
        raw = shrunk_raw(pool, lam, 'att', rate)
        line = fsc._half_fit_or_identity(pd.DataFrame({'raw': shrunk_raw(q, lam, 'att', rate), 'actual': q['actual']}))
        pw = [pairwise_acc(fsc._apply(shrunk_raw(g, lam, 'att', rate), line), g['actual'])
              for _, g in pool.groupby(['year', 'week']) if len(g) > 4]
        rows.append({'lam': lam, 'ship RMSE': round(rmse(fsc._apply(raw, ship), pool['actual']), 4),
                     'refit RMSE': round(rmse(fsc._apply(raw, line), pool['actual']), 4),
                     'refit line': tuple(round(x, 3) for x in line),
                     'pairwise (refit)': round(float(np.nanmean(pw)), 4) if pw else np.nan,
                     'bias (refit)': round(float((fsc._apply(raw, line) - pool['actual']).mean()), 3)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
