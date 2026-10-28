"""Recency-weighted points lines (2026-10-27, user, from the CFB app: "calibration can be weighted more strongly towards recent
seasons as opposed to equal value for each past season").

Today every season in the calibration window counts the same: the points lines (WEEKLY_CALIBRATION, WR/TE cold/rest) are an
unweighted fit over 2021-2025. This re-fits the lines with season weights decay ** (T - 1 - season) and scores them FORWARD-CHAINED -
each test season calibrated from earlier seasons only, as the live 2026 board is from 2021-2025 - against the same forward fit at
equal weights. The low end and the season anchor stay as shipped. A harness run cannot judge this: the shipped lines were fit on
the harness's own seasons.

    python scripts/eval_recency_calibration.py [--dump-path .sweeps/calibration_lab_silent.csv] [--decays 0.7,0.5,0.3]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import scripts.fit_points_scale as fps  # noqa: E402
from data.weekly_projections import LOW_END_POINTS_LINE, SEASON_ANCHOR_S  # noqa: E402
from scripts.eval_calibration_schemes import MIN_ROWS, POS, load_dump, top_n_mask  # noqa: E402

SCOPES = ('START-ALL', 'START-QB', 'START-RB', 'START-WR', 'START-TE', 'EVERY-ALL')


def weighted_line(p, a, w, k):
    if len(p) < MIN_ROWS or np.var(p) < 1e-9:
        return 1.0, 0.0
    b, c = np.polyfit(p, a, 1, w=np.sqrt(w))      # polyfit weights the residual, so sqrt for a weighted least squares
    return 1.0 + k * (b - 1.0), k * c


def fit_lines(d, train, weights, k, pool):
    m = train & d['played'].to_numpy() & d['live'].to_numpy()
    if pool == 'start':
        m &= d['start_raw'].to_numpy()
    out = {}
    for pos in POS:
        for cold in (True, False):
            b = fps.bucket_of(pos, cold)
            sel = m & (d['pos'] == pos).to_numpy()
            if b != 'all':
                sel &= (d['cold'] == cold).to_numpy()
            out[(pos, b)] = weighted_line(d['raw'].to_numpy()[sel], d['actual'].to_numpy()[sel], weights[sel], k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-path', default='.sweeps/calibration_lab_silent.csv')
    ap.add_argument('--decays', default='0.7,0.5,0.3')
    ap.add_argument('--k', type=float, default=0.5)
    a = ap.parse_args()
    d = load_dump(a.dump_path)
    d = d[~((d['year'] == 2021) & (d['week'] == 1))].reset_index(drop=True)   # no depth chart behind that board (fit_points_scale)
    d['wk_key'] = d['year'] * 100 + d['week']
    d['start_raw'] = top_n_mask(d, d['raw'].to_numpy())
    d['actual0'] = d['actual'].fillna(0.0)
    d['expected'] = d['avail'] > fps.EXPECTED
    d['_late_offset'] = 0.0
    low, s = dict(LOW_END_POINTS_LINE), dict(SEASON_ANCHOR_S)
    d['_late_offset'] = d['final'].to_numpy(float) - fps.late(d, fps.apply_lines(d, fps.shipped_params(), low), s)
    yr = d['year'].to_numpy()
    years = sorted(d['year'].unique())
    tests = [y for y in years if y >= years[0] + 2]
    win = d['week'].between(3, 17).to_numpy() & np.isin(yr, tests)
    every = np.ones(len(d), dtype=bool)

    print(f'{a.dump_path}: {len(d):,} rows, seasons {years}; lines at k={a.k:g}\n\nEach season on its own:')
    for y in years:
        p = fit_lines(d, yr == y, every.astype(float), a.k, 'all')
        print(f'  {y}: ' + '  '.join(f'{pos}/{b} ({v[0]:.3f}, {v[1]:+.2f})' for (pos, b), v in p.items()))

    def forward(decay_by_pos, pool):
        out = d['final'].to_numpy(dtype=float).copy()
        for t in tests:
            train = yr < t
            params = fit_lines(d, train, train.astype(float), a.k, pool)
            for pos, dec in decay_by_pos.items():
                rec = fit_lines(d, train, np.where(train, dec ** (t - 1 - yr).clip(0), 0.0), a.k, pool)
                params.update({key: v for key, v in rec.items() if key[0] == pos})
            out[yr == t] = fps.late(d, fps.apply_lines(d, params, low), s)[yr == t]
        return out

    for pool in ('all', 'start'):
        base = forward({}, pool)
        print(f'\nForward-chained, test seasons {tests}, lines fit on {pool} rows: dRMSE vs equal weights (negative = recency better) '
              '[95% CI over weeks] weeks better-worse')
        for which in (POS, ('WR', 'TE')):
            for dec in [float(x) for x in a.decays.split(',')]:
                var = forward({pos: dec for pos in which}, pool)
                sc = fps.scores(d, base, var, win)
                print(f"  {'/'.join(which):11} decay {dec:g}: " + '  '.join(
                    f'{k} {sc[k][0]:+.4f}[{sc[k][1]:+.4f},{sc[k][2]:+.4f}] {sc[k][5]}-{sc[k][6]}' for k in SCOPES))
                print('      START-ALL by season: ' + '  '.join(
                    f"{t} {fps.scores(d, base, var, win & (yr == t))['START-ALL'][0]:+.4f}" for t in tests))


if __name__ == '__main__':
    main()
