"""
Where is the model weakest? (2026-10-06.)  From an --all-rows calibration dump (the harness base-arm board):
model (shipped calibration lines applied) against two naive benchmarks built from the same dump -
the player's own to-date mean points this season and his last-3-games mean - on the harness's own pool
(startable top-N by projection over all live rows, scored on rows with a box score, weeks 3-17, 2022-2025).

A segment where the model barely beats a naive average is where its inputs add least; a segment with a large
signed bias is where the level is off. Reports RMSE, skill (1 - RMSE_model / RMSE_naive), signed bias and
pairwise accuracy by position x projection tier, by season phase, and by within-team role.

    python scripts/analyze_model_weak_spots.py --dump-path .sweeps/seasonal_calibration_allrows_v6.csv
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


def add_benchmarks(d):
    """to-date mean and last-3 mean of this player's PLAYED games earlier in the same season."""
    d = d.sort_values(['year', 'player', 'week']).copy()
    played_pts = d['actual'].where(d['played'])
    g = played_pts.groupby([d['year'], d['player']])
    d['bm_mean'] = g.transform(lambda s: s.shift(1).expanding().mean())
    d['bm_last3'] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    d['bm_n'] = g.transform(lambda s: s.shift(1).expanding().count())
    return d


def summarize(x, label):
    e_m = x['pred'] - x['actual']
    out = {'segment': label, 'n': len(x), 'RMSE model': np.sqrt((e_m ** 2).mean()), 'bias': e_m.mean()}
    for name, col in (('to-date mean', 'bm_mean'), ('last-3', 'bm_last3')):
        y = x.dropna(subset=[col])
        out[f'skill vs {name}'] = (1 - np.sqrt(((y['pred'] - y['actual']) ** 2).mean()) / np.sqrt(((y[col] - y['actual']) ** 2).mean())) if len(y) > 30 else np.nan
    pw = [pairwise_acc(g['pred'], g['actual']) for _, g in x.groupby(['year', 'week']) if len(g) > 4]
    out['pairwise'] = float(np.nanmean(pw)) if pw else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-path', default='.sweeps/seasonal_calibration_allrows_v6.csv')
    a = ap.parse_args()
    d = pd.read_csv(a.dump_path)
    d['played'] = d['played'].astype(bool)
    d['start'] = fsc._startable_mask(d)
    d = add_benchmarks(d)
    d['pred'] = predict(d, shipped_lines())
    x = d[d['start'] & d['played'] & d['year'].between(2022, 2025) & d['week'].between(3, 17)].copy()
    pd.set_option('display.width', 230)
    pd.set_option('display.float_format', lambda v: f'{v:.3f}')
    print(f"pool: {len(x)} startable player-games (2022-2025 wk3-17), {x['bm_mean'].notna().sum()} with a to-date benchmark")

    rows = [summarize(x[x['pos'] == p], p) for p in ('QB', 'RB', 'WR', 'TE')] + [summarize(x, 'ALL')]
    print('\n== by position =='); print(pd.DataFrame(rows).to_string(index=False))

    # projection tier within position-week (rank of the model's own projection)
    x['rk'] = x.groupby(['year', 'week', 'pos'])['pred'].rank(ascending=False, method='first')
    tiers = {'QB': [(1, 6), (7, 12), (13, 24)], 'RB': [(1, 8), (9, 20), (21, 40)], 'WR': [(1, 12), (13, 30), (31, 55)], 'TE': [(1, 4), (5, 10), (11, 20)]}
    rows = []
    for pos, bins in tiers.items():
        for lo, hi in bins:
            rows.append(summarize(x[(x['pos'] == pos) & (x['rk'] >= lo) & (x['rk'] <= hi)], f'{pos} rank {lo}-{hi}'))
    print('\n== by position x projection tier (rank of the model projection that week) =='); print(pd.DataFrame(rows).to_string(index=False))

    rows = []
    for pos in ('QB', 'RB', 'WR', 'TE'):
        for lab, lo, hi in (('wk3-5', 3, 5), ('wk6-10', 6, 10), ('wk11-17', 11, 17)):
            rows.append(summarize(x[(x['pos'] == pos) & x['week'].between(lo, hi)], f'{pos} {lab}'))
    print('\n== by position x season phase =='); print(pd.DataFrame(rows).to_string(index=False))

    # within-team role: rank of the model projection among the team's players of that position that week
    d2 = d[d['played'] | True].copy()
    if 'team' in x.columns:
        x['trk'] = x.groupby(['year', 'week', 'team', 'pos'])['pred'].rank(ascending=False, method='first')
        rows = []
        for pos in ('RB', 'WR', 'TE'):
            for lab, lo, hi in (('team #1', 1, 1), ('team #2', 2, 2), ('team #3+', 3, 9)):
                s = x[(x['pos'] == pos) & x['trk'].between(lo, hi)]
                if len(s) > 40:
                    rows.append(summarize(s, f'{pos} {lab}'))
        print('\n== by within-team role (rank among the team\'s startable players at the position) =='); print(pd.DataFrame(rows).to_string(index=False))

    # boom / bust: share of player-games above 2x projection and below 0.25x (starters, proj >= 8)
    s = x[x['pred'] >= 8]
    print(f"\nplayer-games projected >= 8 pts: n={len(s)}; actual > 2x projection {(s['actual'] > 2 * s['pred']).mean():.1%}; "
          f"actual < 25% of projection {(s['actual'] < 0.25 * s['pred']).mean():.1%}; "
          f"the two tails explain {((s['pred'] - s['actual']) ** 2)[(s['actual'] > 2 * s['pred']) | (s['actual'] < 0.25 * s['pred'])].sum() / ((s['pred'] - s['actual']) ** 2).sum():.0%} of squared error")


if __name__ == '__main__':
    main()
