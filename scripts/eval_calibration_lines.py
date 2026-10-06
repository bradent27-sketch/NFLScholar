"""
Score calibration lines the way harness v2 does, straight from an --all-rows calibration dump
(2026-10-06). The dump already is the harness's base-arm board: replay flags on, every live player kept,
raw (uncalibrated) points. The harness pool is the top-N projections over ALL live rows, scored on the
rows with a box score only - exactly `played & start` here - so a set of lines can be compared in seconds
instead of a two-hour backtest.

  1. held-out: refit the shipping scheme (QB/RB one line; WR/TE cold/rest buckets; half strength) on
     some years, score it against the SHIPPED lines on the others, for three splits (fit 21-23 / test
     24-25, fit 21-24 / test 25, fit 22-25 / test 21).
  2. pooled 2022-2025 weeks 3-17 (the harness window): shipped lines vs the lines fit on every year,
     RMSE / MAE / signed bias / pairwise per scope. In-sample for the refit, so read it as "does it hurt
     the gate's own metric", not as evidence it generalises - the held-out block is that.

    python scripts/eval_calibration_lines.py --dump-path .sweeps/seasonal_calibration_allrows_v6.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import scripts.fit_seasonal_calibration as fsc  # noqa: E402
from data.weekly_projections import (WEEKLY_CALIBRATION, WEEKLY_CALIBRATION_BY_BUCKET,  # noqa: E402
                                     WEEKLY_CALIBRATION_COLD_MAX_WEEK)
from scripts.harness_v2 import metrics, pairwise_acc  # noqa: E402

POS = ('QB', 'RB', 'WR', 'TE')
BUCKETED = fsc.BUCKETED_POS


def shipped_lines():
    out = {}
    for pos in POS:
        out[(pos, 'all')] = WEEKLY_CALIBRATION[pos]
        if pos in BUCKETED:
            out[(pos, 'cold')] = WEEKLY_CALIBRATION_BY_BUCKET['cold'][pos]
            out[(pos, 'rest')] = WEEKLY_CALIBRATION_BY_BUCKET['rest'][pos]
    return out


def fit_lines(frame):
    """The shipping scheme, fit on played rows of `frame` (half strength, two-sided)."""
    out = {}
    for pos in POS:
        p = frame[frame['pos'] == pos]
        out[(pos, 'all')] = fsc._half_fit_or_identity(p)
        if pos in BUCKETED:
            out[(pos, 'cold')] = fsc._half_fit_or_identity(p[p['week'] <= WEEKLY_CALIBRATION_COLD_MAX_WEEK])
            out[(pos, 'rest')] = fsc._half_fit_or_identity(p[p['week'] > WEEKLY_CALIBRATION_COLD_MAX_WEEK])
    return out


def predict(df, lines, only=None):
    pred = np.zeros(len(df))
    raw = df['raw'].to_numpy(dtype=float)
    for pos in POS:
        m = (df['pos'] == pos).to_numpy()
        if only is not None and pos not in only:
            continue
        if pos in BUCKETED:
            cold = (df['week'] <= WEEKLY_CALIBRATION_COLD_MAX_WEEK).to_numpy()
            for sel, key in ((m & cold, 'cold'), (m & ~cold, 'rest')):
                pred[sel] = fsc._apply(raw[sel], lines[(pos, key)])
        else:
            pred[m] = fsc._apply(raw[m], lines[(pos, 'all')])
    return pred


def score(df, lines, label, weeks=(1, 18)):
    """Per-position START metrics (+ START-ALL) for played rows of the top-N pool."""
    d = df[(df['week'] >= weeks[0]) & (df['week'] <= weeks[1])].copy()
    d['pred'] = predict(d, lines)
    d = d[d['start'] & d['played'].astype(bool)]
    rows = {}
    for scope, sub in [(p, d[d['pos'] == p]) for p in POS] + [('ALL', d)]:
        m = metrics(sub['pred'], sub['actual'])
        pw = [pairwise_acc(g['pred'], g['actual']) for _, g in sub.groupby(['year', 'week'])]
        rows[scope] = {'n': m['n'], 'rmse': m['rmse'], 'mae': m['mae'], 'bias': m['bias'],
                       'pairwise': float(np.nanmean(pw))}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-path', default='.sweeps/seasonal_calibration_allrows_v6.csv')
    a = ap.parse_args()
    df = pd.read_csv(a.dump_path)
    df['start'] = fsc._startable_mask(df)            # pool over ALL live rows, before the played filter
    df['played'] = df['played'].astype(bool)
    shipped = shipped_lines()
    pd.set_option('display.width', 200)

    print('=' * 100)
    print('1. HELD-OUT: shipped lines vs lines refit on the other years (played rows only; START pools; weeks 1-18)')
    print('=' * 100)
    splits = [('fit 21-23 / test 24-25', (2021, 2022, 2023), (2024, 2025)),
              ('fit 21-24 / test 25', (2021, 2022, 2023, 2024), (2025,)),
              ('fit 22-25 / test 21', (2022, 2023, 2024, 2025), (2021,))]
    verdict = {p: [] for p in POS}
    for name, fy, ty in splits:
        refit = fit_lines(df[df['year'].isin(fy) & df['played']])
        test = df[df['year'].isin(ty)]
        s_old, s_new = score(test, shipped, 'shipped'), score(test, refit, 'refit')
        print(f"\n{name}")
        rows = []
        for scope in list(POS) + ['ALL']:
            o, n = s_old[scope], s_new[scope]
            rows.append({'scope': scope, 'n': o['n'],
                         'RMSE shipped': round(o['rmse'], 3), 'RMSE refit': round(n['rmse'], 3), 'dRMSE': round(n['rmse'] - o['rmse'], 3),
                         'dMAE': round(n['mae'] - o['mae'], 3),
                         'bias shipped': round(o['bias'], 3), 'bias refit': round(n['bias'], 3),
                         'dpairwise': round(n['pairwise'] - o['pairwise'], 4)})
            if scope in POS:
                verdict[scope].append((n['rmse'] - o['rmse'], abs(n['bias']) - abs(o['bias'])))
        print(pd.DataFrame(rows).to_string(index=False))
    print('\nper-position verdict (apply only if RMSE improves in EVERY split; |bias| noted):')
    for pos in POS:
        d_rmse = [v[0] for v in verdict[pos]]
        d_bias = [v[1] for v in verdict[pos]]
        ok = all(x < 0 for x in d_rmse)
        print(f"  {pos}: dRMSE by split {[round(x, 3) for x in d_rmse]}  d|bias| {[round(x, 3) for x in d_bias]}  -> {'IMPROVES in every split' if ok else 'not every split'}")

    print('\n' + '=' * 100)
    print('2. POOLED, harness window 2022-2025 weeks 3-17: shipped lines vs lines fit on all years (in-sample)')
    print('=' * 100)
    full = fit_lines(df[df['played']])
    win = df[df['year'].between(2022, 2025)]
    s_old, s_new = score(win, shipped, 'shipped', (3, 17)), score(win, full, 'refit', (3, 17))
    rows = []
    for scope in list(POS) + ['ALL']:
        o, n = s_old[scope], s_new[scope]
        rows.append({'scope': 'START-' + scope, 'n': o['n'], 'RMSE shipped': round(o['rmse'], 3), 'RMSE refit': round(n['rmse'], 3),
                     'dRMSE': round(n['rmse'] - o['rmse'], 3), 'bias shipped': round(o['bias'], 3), 'bias refit': round(n['bias'], 3),
                     'pairwise shipped': round(o['pairwise'], 4), 'pairwise refit': round(n['pairwise'], 4)})
    print(pd.DataFrame(rows).to_string(index=False))
    print('\nshipped lines:', {f'{k[0]}/{k[1]}': tuple(round(x, 3) for x in v) for k, v in shipped.items()})
    print('refit (all years):', {f'{k[0]}/{k[1]}': tuple(round(x, 3) for x in v) for k, v in full.items()})


if __name__ == '__main__':
    main()
