"""
First game back after missed games (2026-10-13): is a "first game back" factor on the point total worth it for RB/WR/TE?
scripts/diag_returning_players.py found skill players in their first game after missing their team's previous 4+ games
running about 20% under their projection (START pool +2.55 pts, 5/5 seasons), 2-3 missed about 8%.

Same method as the Questionable factor (scripts/eval_questionable_factor.py): on the shipped board (lab dump + the shipped
late steps, see diag_returning_players.shipped_board), each segment's factor is least squares through the origin of
actual on shown over the fit seasons' played rows with a projection of 5+, applied to every row of that segment in the
held-out season (leave-one-season-out), and scored like harness v2 (paired START pools, played rows, bootstrap CI over
weeks) against the shipped board.

    python scripts/eval_returning_factor.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_projections import SIDELINED_AVAILABILITY  # noqa: E402
from scripts.diag_returning_players import classify, season_status, shipped_board  # noqa: E402
from scripts.eval_calibration_schemes import boot_ci, load_dump, paired_scores  # noqa: E402

SEGMENT_SETS = {
    'back-4+': ('back-4+',),
    'back-2-3': ('back-2-3',),
    'back-2+ (one factor)': ('back-2-3', 'back-4+'),
}


def factor_preds(d, segments, positions, by_pos=True, min_shown=5.0):
    pred = d['final'].to_numpy(float).copy()
    params = []
    played = d['played'].to_numpy() & d['actual'].notna().to_numpy() & d['week'].between(3, 17).to_numpy()
    live = (d['avail'] > SIDELINED_AVAILABILITY).to_numpy() & (pred > 0.005)
    seg = d['segment'].isin(segments).to_numpy()
    for y in sorted(d['year'].unique()):
        fit = played & (d['year'] != y).to_numpy() & (pred >= min_shown) & seg
        tgt = (d['year'] == y).to_numpy() & seg & live
        groups = [(p,) for p in positions] if by_pos else [tuple(positions)]
        for grp in groups:
            pm = d['pos'].isin(grp).to_numpy()
            f = fit & pm
            if f.sum() < 10:
                continue
            k = float((d['actual'].to_numpy()[f] * pred[f]).sum() / (pred[f] ** 2).sum())
            k = float(np.clip(k, 0.5, 1.2))
            params.append((y, '/'.join(grp), round(k, 3), int(f.sum())))
            pred[tgt & pm] = np.round(pred[tgt & pm] * k, 2)
    return pred, params


def main():
    d = load_dump('.sweeps/calibration_lab_v7.csv', '.sweeps/calibration_lab_v7_wk1_ourlads.csv')
    d['final'] = shipped_board(d)
    d = d.join(season_status(d, sorted(d['year'].unique())))
    d['segment'] = d.apply(classify, axis=1)
    window = (d['week'].between(3, 17) & (d['year'] >= 2022)).to_numpy()
    base = d['final'].to_numpy(float)
    pd.set_option('display.width', 240)
    rows = []
    for label, segs in SEGMENT_SETS.items():
        for positions, by_pos, plabel in ((('RB', 'WR', 'TE'), True, 'per position'),
                                          (('RB', 'WR', 'TE'), False, 'RB/WR/TE pooled'),
                                          (('RB', 'WR'), True, 'RB, WR only')):
            pred, params = factor_preds(d, segs, positions, by_pos)
            sc = paired_scores(d, base, pred, window)
            r = {'segments': label, 'positions': plabel}
            for scope in ('START-ALL', 'START-RB', 'START-WR', 'START-TE', 'ALL'):
                x = sc[scope]
                dr = x['rmse_v'] - x['rmse_b']
                if scope in ('START-ALL', 'ALL'):
                    lo, hi = boot_ci(x['weeks'])
                    r[scope] = f'{dr:+.4f} [{lo:+.3f},{hi:+.3f}]'
                else:
                    r[scope] = f'{dr:+.4f}'
            sa = sc['START-ALL']
            r['bias'] = f"{sa['bias_b']:+.3f}->{sa['bias_v']:+.3f}"
            r['W-L'] = f"{sa['wins']}-{sa['losses']}"
            yrs = []
            for y in (2022, 2023, 2024, 2025):
                s2 = paired_scores(d, base, pred, window & (d['year'] == y).to_numpy())['START-ALL']
                yrs.append(s2['rmse_v'] - s2['rmse_b'])
            r['yrs better'] = f"{sum(v < 0 for v in yrs)}/4"
            factors = {}
            for _y, grp, k, _n in params:
                factors.setdefault(grp, []).append(k)
            r['factor (held-out range)'] = ' '.join(f'{g} {min(v):.2f}-{max(v):.2f}' for g, v in factors.items())
            rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

    print('\nRows each segment touches (harness window, live rows, any projection):')
    m = window & (d['avail'] > SIDELINED_AVAILABILITY).to_numpy() & (base > 0.005)
    print(d[m].groupby(['segment', 'pos']).size().unstack(fill_value=0).to_string())


if __name__ == '__main__':
    main()
