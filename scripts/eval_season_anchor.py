"""
Does shrinking the calibrated projection toward the player's OWN to-date average improve held-out accuracy?
(2026-10-06, from scripts/analyze_model_weak_spots.py: the model's departures from a player's to-date mean carry
real signal, corr 0.27-0.36, but WR/RB departures are ~30% too large, worst after week 6.)

    final = bm + s(pos, phase) * (calibrated - bm)         bm = mean of the player's played games earlier this
                                                           season (>= MIN_GAMES of them; else no change)

s is fit by least squares of (actual - bm) on (calibrated - bm) on the FIT years, per position and season-phase
bucket, clipped to [0.3, 1.0], and scored on held-out years on the harness's pool (startable top-N over all live
rows, played rows, weeks 3-17), against the shipped calibration alone. Everything uses only games before the
target week, so it is implementable live. A within-model anchor: no market or FantasyPros input.

    python scripts/eval_season_anchor.py --dump-path .sweeps/seasonal_calibration_allrows_v6.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import scripts.fit_seasonal_calibration as fsc  # noqa: E402
from scripts.analyze_model_weak_spots import add_benchmarks  # noqa: E402
from scripts.eval_calibration_lines import predict, shipped_lines  # noqa: E402
from scripts.harness_v2 import pairwise_acc  # noqa: E402

POS = ('QB', 'RB', 'WR', 'TE')
PHASES = {'wk3-6': (3, 6), 'wk7-11': (7, 11), 'wk12-17': (12, 17)}
MIN_GAMES = 2


def phase_of(week):
    for k, (lo, hi) in PHASES.items():
        if lo <= week <= hi:
            return k
    return 'wk12-17'


def fit_s(frame, by_phase=True):
    """{(pos, phase): s} from rows with a benchmark."""
    out = {}
    f = frame[frame['bm_n'] >= MIN_GAMES]
    for pos in POS:
        for ph in (PHASES if by_phase else ['all']):
            s = f[(f['pos'] == pos) & ((f['phase'] == ph) if by_phase else True)]
            x, y = (s['pred'] - s['bm_mean']).to_numpy(), (s['actual'] - s['bm_mean']).to_numpy()
            out[(pos, ph)] = float(np.clip((x * y).sum() / (x * x).sum(), 0.3, 1.0)) if len(s) > 60 else 1.0
    return out


def apply_s(frame, s, by_phase=True):
    new = frame.copy()
    k = [s.get((p, ph if by_phase else 'all'), 1.0) for p, ph in zip(new['pos'], new['phase'])]
    has = (new['bm_n'] >= MIN_GAMES) & new['bm_mean'].notna()
    anchored = new['bm_mean'] + np.array(k) * (new['pred'] - new['bm_mean'])
    new['pred'] = np.where(has, anchored, new['pred'])
    return new


def score_pool(x):
    out = {}
    for scope, sub in [(p, x[x['pos'] == p]) for p in POS] + [('ALL', x)]:
        e = sub['pred'] - sub['actual']
        pw = [pairwise_acc(g['pred'], g['actual']) for _, g in sub.groupby(['year', 'week']) if len(g) > 4]
        out[scope] = {'n': len(sub), 'rmse': float(np.sqrt((e ** 2).mean())), 'bias': float(e.mean()), 'pw': float(np.nanmean(pw))}
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
    d['phase'] = d['week'].map(phase_of)
    pool = d[d['start'] & d['played'] & d['week'].between(3, 17)]
    pd.set_option('display.width', 220)
    splits = [('fit 21-23 / test 24-25', (2021, 2022, 2023), (2024, 2025)),
              ('fit 21-24 / test 25', (2021, 2022, 2023, 2024), (2025,)),
              ('fit 22-25 / test 21', (2022, 2023, 2024, 2025), (2021,)),
              ('fit 21,23,25 / test 22,24', (2021, 2023, 2025), (2022, 2024))]
    verdict = {p: [] for p in POS + ('ALL',)}
    for name, fy, ty in splits:
        for by_phase in (True, False):
            s = fit_s(pool[pool['year'].isin(fy)], by_phase)
            test = pool[pool['year'].isin(ty)]
            base, new = score_pool(test), score_pool(apply_s(test, s, by_phase))
            if by_phase:
                rows = [{'scope': k, 'n': base[k]['n'], 'RMSE base': round(base[k]['rmse'], 3), 'dRMSE anchor': round(new[k]['rmse'] - base[k]['rmse'], 3),
                         'bias base': round(base[k]['bias'], 3), 'bias anchor': round(new[k]['bias'], 3),
                         'dpairwise': round(new[k]['pw'] - base[k]['pw'], 4)} for k in POS + ('ALL',)]
                print(f"\n{name}  (s by position x phase: " + ', '.join(f"{p}/{ph} {v:.2f}" for (p, ph), v in s.items() if p in ('WR', 'RB')) + ')')
                print(pd.DataFrame(rows).to_string(index=False))
                for k in POS + ('ALL',):
                    verdict[k].append(new[k]['rmse'] - base[k]['rmse'])
            else:
                print(f"   single s per position: " + ', '.join(f"{p} {v:.2f}" for (p, _), v in s.items()) +
                      f" -> dRMSE ALL {new['ALL']['rmse'] - base['ALL']['rmse']:+.3f}")
    print('\ndRMSE by split (by position x phase), negative = better:')
    for k, v in verdict.items():
        print(f"  {k}: {[round(x, 3) for x in v]}  -> {'IMPROVES in every split' if all(x < 0 for x in v) else 'not every split'}")
    full = fit_s(pool[pool['year'].between(2021, 2025)], True)
    print('\nfit on all years (the values a ship would use):')
    print(pd.DataFrame([{'pos': p, **{ph: round(full[(p, ph)], 2) for ph in PHASES}} for p in POS]).to_string(index=False))


if __name__ == '__main__':
    main()
