"""
Is the model's yards-per-opportunity (receiving yards per target, QB yards per attempt) over-dispersed, and does
regressing it toward the position mean improve held-out points accuracy? (2026-10-06, from the week-5 board audit:
WR yards-per-catch above 17 projects 67 yards and gets 56; QB yards/attempt projections run 11-13 yards too extreme
at both ends.)

    new_yards = volume * (m + lam * (projected_yards_per_opportunity - m))      m = position mean, fit years

The shrink is applied to the projected YARDS only (volume untouched), the raw points move by the yard weight
(0.1 receiving, 0.04 passing, Full PPR), and the SHIPPED calibration lines are applied unchanged. lam is picked on
the fit years (smallest START RMSE for that position) and scored on the held-out years on the harness's pool
(startable top-N over all live rows, played rows, weeks 3-17), against lam = 1 (no change).

    python scripts/eval_efficiency_shrink.py --dump-path .sweeps/seasonal_calibration_allrows_v6.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import scripts.fit_seasonal_calibration as fsc  # noqa: E402
from scripts.eval_calibration_lines import predict, shipped_lines  # noqa: E402
from scripts.harness_v2 import pairwise_acc  # noqa: E402

LAMS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
CHANNELS = {  # pos -> (volume col, projected-yards col, yard weight in points)
    'WR': ('p_targets', 'p_receiving_yards', 0.1),
    'TE': ('p_targets', 'p_receiving_yards', 0.1),
    'RB': ('p_targets', 'p_receiving_yards', 0.1),
    'QB': ('p_passing_attempts', 'p_passing_yards', 0.04),
}
MIN_VOLUME = {'WR': 1.0, 'TE': 1.0, 'RB': 1.0, 'QB': 10.0}


def position_mean(frame, pos):
    vol, yds, _ = CHANNELS[pos]
    s = frame[(frame['pos'] == pos) & (frame[vol] >= MIN_VOLUME[pos])]
    return float(s[yds].sum() / s[vol].sum())


def shrink(frame, lam, means):
    """Frame with 'raw' moved for every channel at its own lam (dict pos -> lam)."""
    out = frame.copy()
    for pos, (vol, yds, w) in CHANNELS.items():
        l = lam.get(pos, 1.0)
        if l == 1.0:
            continue
        m = (out['pos'] == pos) & (out[vol] >= MIN_VOLUME[pos])
        eff = out.loc[m, yds] / out.loc[m, vol]
        new = out.loc[m, vol] * (means[pos] + l * (eff - means[pos]))
        out.loc[m, 'raw'] = out.loc[m, 'raw'] + w * (new - out.loc[m, yds])
    return out


def rmse(frame, lines, pos=None):
    sub = frame if pos is None else frame[frame['pos'] == pos]
    pred = predict(sub, lines)
    return float(np.sqrt(((pred - sub['actual'].to_numpy()) ** 2).mean()))


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
    pool = d[d['start'] & d['played'] & d['week'].between(3, 17)].copy()
    lines = shipped_lines()
    pd.set_option('display.width', 220)

    print('START-pool RMSE by single-channel lam (shipped lines, all of 2021-2025, in-sample), lam=1 is the model today')
    means_all = {p: position_mean(d[d['year'].between(2021, 2025)], p) for p in CHANNELS}
    print('position means (yards per opportunity):', {p: round(v, 3) for p, v in means_all.items()})
    rows = []
    for pos in CHANNELS:
        base = rmse(pool, lines, pos)
        row = {'pos': pos, 'n': int((pool['pos'] == pos).sum())}
        for lam in LAMS:
            row[f'{lam:.1f}'] = round(rmse(shrink(pool, {pos: lam}, means_all), lines, pos) - base, 4)
        rows.append(row)
    print('change in START RMSE vs lam=1.0 (negative = better):')
    print(pd.DataFrame(rows).to_string(index=False))

    splits = [('fit 21-23 / test 24-25', (2021, 2022, 2023), (2024, 2025)),
              ('fit 21-24 / test 25', (2021, 2022, 2023, 2024), (2025,)),
              ('fit 22-25 / test 21', (2022, 2023, 2024, 2025), (2021,)),
              ('fit 21,23,25 / test 22,24', (2021, 2023, 2025), (2022, 2024))]
    print('\nheld-out: lam chosen on the fit years per position, scored on the test years (change vs lam=1)')
    verdict = {p: [] for p in list(CHANNELS) + ['ALL']}
    for name, fy, ty in splits:
        fit_all = d[d['year'].isin(fy)]
        means = {p: position_mean(fit_all, p) for p in CHANNELS}
        fit_pool, test_pool = pool[pool['year'].isin(fy)], pool[pool['year'].isin(ty)]
        best = {}
        for pos in CHANNELS:
            scores = {lam: rmse(shrink(fit_pool, {pos: lam}, means), lines, pos) for lam in LAMS}
            best[pos] = min(scores, key=scores.get)
        shrunk = shrink(test_pool, best, means)
        row = {'split': name, 'lam': ' '.join(f'{p}{best[p]:.1f}' for p in CHANNELS)}
        for pos in list(CHANNELS) + ['ALL']:
            sub_b = test_pool if pos == 'ALL' else test_pool[test_pool['pos'] == pos]
            sub_s = shrunk.loc[sub_b.index]
            delta = rmse(sub_s, lines) - rmse(sub_b, lines)
            row[pos] = round(delta, 4)
            verdict[pos].append(delta)
        print(row)
    print('\nverdict (negative everywhere = robust):')
    for k, v in verdict.items():
        print(f"  {k}: {[round(x, 4) for x in v]} -> {'better in every split' if all(x < 0 for x in v) else 'not every split'}")
    full = {p: min(LAMS, key=lambda lam: rmse(shrink(pool, {p: lam}, means_all), lines, p)) for p in CHANNELS}
    print('\nlam fit on all years:', full)


if __name__ == '__main__':
    main()
