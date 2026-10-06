"""
Is the model's touchdown projection over-dispersed, and does compressing it improve points accuracy?
(2026-10-06.)  Offline, from an --all-rows calibration dump (the harness's base-arm board: replay on, raw
uncalibrated points, startable pool over all live rows, scored on rows with a box score).

Per TD channel (QB rushing, RB rushing, WR/TE receiving, RB receiving, QB passing) fit
    actual TDs ~ a + b * projected TDs        (least squares, played rows, fit years only)
and replace each projected TD count with a + b * projected (floored at 0). The raw points move by
scoring-weight x (new - old) TDs, then the points calibration lines are applied exactly as the model does,
and START-pool RMSE / bias / pairwise are scored against the same lines on the unchanged board, in the
three held-out splits scripts/eval_calibration_lines.py uses. Also scored with the lines REFIT on the
TD-calibrated board (what shipping it would require).

    python scripts/eval_td_calibration.py --dump-path .sweeps/seasonal_calibration_allrows_v6.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import scripts.fit_seasonal_calibration as fsc  # noqa: E402
from scripts.eval_calibration_lines import fit_lines, score, shipped_lines  # noqa: E402

PTS = {'rushing_tds': 6.0, 'receiving_tds': 6.0, 'passing_tds': 4.0}
# (label, position mask, stat)
CHANNELS = (('QB rush', ('QB',), 'rushing_tds'), ('RB rush', ('RB',), 'rushing_tds'),
            ('RB rec', ('RB',), 'receiving_tds'), ('WR rec', ('WR',), 'receiving_tds'),
            ('TE rec', ('TE',), 'receiving_tds'), ('QB pass', ('QB',), 'passing_tds'))


def fit_channels(frame):
    """{label: (a, b)} from played rows of `frame`."""
    out = {}
    for label, poss, stat in CHANNELS:
        s = frame[frame['pos'].isin(poss)]
        p = pd.to_numeric(s[f'p_{stat}'], errors='coerce')
        a = pd.to_numeric(s[f'a_{stat}'], errors='coerce')
        ok = p.notna() & a.notna()
        b, a0 = np.polyfit(p[ok], a[ok], 1)
        out[label] = (float(a0), float(b))
    return out


def apply_channels(df, coefs):
    new = df.copy()
    shift = np.zeros(len(new))
    for label, poss, stat in CHANNELS:
        a0, b = coefs[label]
        m = new['pos'].isin(poss).to_numpy()
        p = pd.to_numeric(new[f'p_{stat}'], errors='coerce').fillna(0.0).to_numpy()
        f = np.clip(a0 + b * p, 0.0, None)
        shift[m] += PTS[stat] * (f[m] - p[m])
    new['raw'] = new['raw'].to_numpy(dtype=float) + shift
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-path', default='.sweeps/seasonal_calibration_allrows_v6.csv')
    a = ap.parse_args()
    df = pd.read_csv(a.dump_path)
    df['start'] = fsc._startable_mask(df)
    df['played'] = df['played'].astype(bool)
    shipped = shipped_lines()
    pd.set_option('display.width', 220)

    full = fit_channels(df[df['played'] & (df['week'] >= 3)])
    print('TD calibration, played rows, wk3+, all years: actual = a + b * projected')
    for k, (a0, b) in full.items():
        print(f'  {k:<8} a {a0:+.4f}  b {b:.3f}')

    splits = [('fit 21-23 / test 24-25', (2021, 2022, 2023), (2024, 2025)),
              ('fit 21-24 / test 25', (2021, 2022, 2023, 2024), (2025,)),
              ('fit 22-25 / test 21', (2022, 2023, 2024, 2025), (2021,))]
    verdicts = {p: [] for p in ('QB', 'RB', 'WR', 'TE')}
    for name, fy, ty in splits:
        fit = df[df['year'].isin(fy) & df['played'] & (df['week'] >= 3)]
        coefs = fit_channels(fit)
        test = df[df['year'].isin(ty)]
        test_td = apply_channels(test, coefs)
        # points lines: (a) the shipped ones, (b) refit on the TD-calibrated fit years
        fit_td_all = apply_channels(df[df['year'].isin(fy)], coefs)
        refit = fit_lines(fit_td_all[fit_td_all['played']])
        base = score(test, shipped, 'base')
        a_ = score(test_td, shipped, 'td+shipped lines')
        b_ = score(test_td, refit, 'td+refit lines')
        print(f'\n{name}: coefs ' + ', '.join(f'{k} b={v[1]:.2f}' for k, v in coefs.items()))
        rows = []
        for scope in ('QB', 'RB', 'WR', 'TE', 'ALL'):
            rows.append({'scope': scope, 'n': base[scope]['n'], 'RMSE base': round(base[scope]['rmse'], 3),
                         'dRMSE +TDcal (shipped lines)': round(a_[scope]['rmse'] - base[scope]['rmse'], 3),
                         'dRMSE +TDcal (lines refit)': round(b_[scope]['rmse'] - base[scope]['rmse'], 3),
                         'bias base': round(base[scope]['bias'], 3), 'bias TDcal+refit': round(b_[scope]['bias'], 3),
                         'dpairwise (refit)': round(b_[scope]['pairwise'] - base[scope]['pairwise'], 4)})
            if scope != 'ALL':
                verdicts[scope].append((a_[scope]['rmse'] - base[scope]['rmse'], b_[scope]['rmse'] - base[scope]['rmse']))
        print(pd.DataFrame(rows).to_string(index=False))
    print('\nverdict by position (dRMSE per split: shipped lines | refit lines):')
    for pos, v in verdicts.items():
        print(f"  {pos}: shipped {[round(x[0], 3) for x in v]} | refit {[round(x[1], 3) for x in v]}")

    # the harness window, in-sample for the TD fit: START metrics, 2022-2025 wk3-17
    win = df[df['year'].between(2022, 2025)]
    td_all = apply_channels(win, full)
    refit_all = fit_lines(apply_channels(df, full)[df['played'].to_numpy()])
    b0, b1, b2 = score(win, shipped, 'b', (3, 17)), score(td_all, shipped, 't', (3, 17)), score(td_all, refit_all, 'r', (3, 17))
    print('\nHarness window 2022-2025 wk3-17 (TD fit on all years, so in-sample):')
    print(pd.DataFrame([{'scope': 'START-' + s, 'RMSE base': round(b0[s]['rmse'], 3), 'RMSE +TDcal shipped lines': round(b1[s]['rmse'], 3),
                         'RMSE +TDcal refit lines': round(b2[s]['rmse'], 3), 'bias base': round(b0[s]['bias'], 3),
                         'bias +TDcal refit': round(b2[s]['bias'], 3)} for s in ('QB', 'RB', 'WR', 'TE', 'ALL')]).to_string(index=False))


if __name__ == '__main__':
    main()
