"""Where is the SHIPPED final projection still mis-calibrated? (2026-10-08)

Reads a `scripts/calibration_lab.py` dump (the harness base arm) and prints the signed error (final - actual) of the
shipped 'Model Proj Pts' by segment, on played rows: the harness START pool (top-N per position over all rows) and
the whole played pool. A segment is a calibration candidate only if its bias is (a) large next to its standard error,
(b) the same sign in most seasons, and (c) worth something in RMSE: removing a bias b from a share p of the pool cuts
MSE by about p * b^2.

    python scripts/diag_calibration_segments.py --dump-path .sweeps/calibration_lab_v7.csv --week1-path .sweeps/calibration_lab_v7_wk1_ourlads.csv
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

from scripts.eval_calibration_schemes import load_dump, top_n_mask  # noqa: E402

POS = ('QB', 'RB', 'WR', 'TE')


def _bucket(x, edges, labels):
    return pd.cut(pd.to_numeric(x, errors='coerce'), edges, labels=labels, right=False).astype(str)


def segments(d):
    seg = {}
    seg['phase'] = pd.cut(d['week'], [0, 1, 2, 4, 8, 12, 19], labels=['wk1', 'wk2', 'wk3-4', 'wk5-8', 'wk9-12', 'wk13-18']).astype(str)
    seg['games before'] = _bucket(d['anchor_games'], [0, 1, 2, 4, 8, 99], ['0', '1', '2-3', '4-7', '8+'])
    if 'inj' in d:
        inj = d['inj'].fillna('').astype(str).str.split(' ').str[0].replace({'': 'none', 'Nan': 'none'})
        seg['injury tag'] = inj.where(inj.isin(['none', 'Questionable', 'Doubtful', 'Out', 'Reserve', 'Practice']), 'other')
    if 'avail' in d:
        seg['availability'] = _bucket(d['avail'], [-1, 0.011, 0.5, 0.85, 0.99, 2], ['<=0.01', '<0.5', '0.5-0.85', '0.85-0.99', '1.0'])
    if 'snap' in d:
        seg['exp snap share'] = _bucket(d['snap'], [-1, 0.3, 0.5, 0.7, 0.85, 2], ['<30%', '30-50%', '50-70%', '70-85%', '85%+'])
    if 'role_conf' in d:
        seg['role confidence'] = d['role_conf'].fillna('none').astype(str)
    if 'source_kind' in d:
        seg['profile source'] = d['source_kind'].fillna('none').astype(str)
    if 'target_rank' in d:
        seg['target-earner rank'] = _bucket(d['target_rank'], [0, 2, 3, 4, 99], ['1', '2', '3', '4+'])
    # rank of the final projection inside its START pool: does the line bend at the top?
    r = pd.Series(d['final'].to_numpy(), index=d.index).groupby([d['wk_key'], d['pos']]).rank(method='first', ascending=False)
    frac = r / d['pos'].map({'QB': 24, 'RB': 40, 'WR': 55, 'TE': 20})
    seg['start-pool tier'] = pd.cut(frac, [0, 0.25, 0.5, 0.75, 1.0, 99], labels=['top 25%', '25-50%', '50-75%', '75-100%', 'outside']).astype(str)
    seg['year'] = d['year'].astype(str)
    return seg


def table(d, mask, seg_name, seg, pos):
    m = mask & (d['pos'] == pos).to_numpy() if pos != 'ALL' else mask
    e = (d['final'] - d['actual']).to_numpy()
    rows = []
    for lab in sorted(seg[m].unique()):
        mm = m & (seg == lab).to_numpy()
        n = int(mm.sum())
        if n < 40:
            continue
        ee = e[mm]
        yb = pd.Series(ee).groupby(d['year'].to_numpy()[mm]).mean()
        same = int((np.sign(yb) == np.sign(ee.mean())).sum())
        rows.append({'segment': lab, 'n': n, 'share': round(n / m.sum(), 3), 'bias': round(ee.mean(), 2),
                     't': round(ee.mean() / (ee.std(ddof=1) / np.sqrt(n)), 1),
                     'same sign yrs': f"{same}/{len(yb)}", 'RMSE': round(float(np.sqrt((ee ** 2).mean())), 2),
                     'MSE gain if fixed': round(n / m.sum() * ee.mean() ** 2, 3),
                     'by year': ' '.join(f"{v:+.1f}" for v in yb.to_numpy())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-path', default=os.path.join('.sweeps', 'calibration_lab_v7.csv'))
    ap.add_argument('--week1-path', default='')
    ap.add_argument('--weeks', default='1-18')
    a = ap.parse_args()
    d = load_dump(a.dump_path, a.week1_path)
    w0, w1 = (int(x) for x in a.weeks.split('-'))
    win = d['week'].between(w0, w1).to_numpy()
    played = d['played'].to_numpy()
    start = top_n_mask(d, d['final'].to_numpy(dtype=float))
    seg = segments(d)
    pd.set_option('display.width', 250)
    for pool_name, mask in (('START pool (harness), played', start & played & win), ('ALL played', played & win)):
        print('\n' + '#' * 110 + f"\n# {pool_name}, weeks {a.weeks}: signed error final - actual (+ = model HIGH)\n" + '#' * 110)
        for name, s in seg.items():
            for pos in POS + ('ALL',):
                t = table(d, mask, name, s, pos)
                if t.empty:
                    continue
                print(f"\n[{name}] {pos}")
                print(t.to_string(index=False))


if __name__ == '__main__':
    main()
