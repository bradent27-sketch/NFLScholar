"""
Low end of the points line, re-fit on every game (backlog L7 option b, 2026-10-22; user: "yes start the points line refit").

The v4 points lines were fit on games with a box-score line only. A low projection that records a stat scores more than its raw
number, so the line lifts every low projection (TE raw 0.53 -> shown 1.20 against 0.60 actual once every expected game counts).
The v5 all-rows re-fit of the WHOLE line was reverted on 2026-10-01 because the box-score harness scored it worse, so this keeps the
v4 line wherever a player is startable and re-fits only below a knot k:

    shown = L(raw)                       raw >= k        (L = the shipped v4 line for that position and season phase)
    shown = L(k) + a * (raw - k)         raw <  k        (continuous at k; a fit on every game, zeros counted)

a per position (RB, WR, TE; QB has its own starter rules), chosen by grid search on four seasons and scored on the fifth, through the
shipped late steps (season anchor, Questionable cut, the post-calibration offset such as the luck step). Scored three ways, held
out by season: every game (players expected to play, no box-score line = 0) for the whole position, the every-game START pool, and
the box-score START pool (the harness's own).

    python scripts/eval_low_end_line.py [--board .sweeps/qb_style_capture.parquet.board.csv] [--knots 4,6,8]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_projections import (QUESTIONABLE_WORKLOAD_FACTORS, SEASON_ANCHOR_S, SIDELINED_AVAILABILITY,  # noqa: E402
                                     _weekly_calibration_for)
from scripts.eval_calibration_schemes import anchor, keep_raw_where_not_live, load_dump, top_n_mask  # noqa: E402

KEYS = ['year', 'week', 'player', 'pos', 'team']
POSITIONS = ('RB', 'WR', 'TE')
SLOPES = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6)
EXPECTED = 0.5


def line(d, knot=None, slopes=None):
    """The shipped v4 points line, with the low end replaced below ``knot`` by L(knot) + a * (raw - knot)."""
    raw = d['raw'].to_numpy(dtype=float)
    out = raw.copy()
    for pos in ('QB', 'RB', 'WR', 'TE'):
        for cold in (True, False):
            m = ((d['pos'] == pos) & (d['cold'] == cold)).to_numpy()
            sl, ic = _weekly_calibration_for(pos, 1 if cold else 10)
            val = sl * raw[m] + ic
            if knot is not None and slopes and pos in slopes:
                low = raw[m] < knot
                val = np.where(low, sl * knot + ic + slopes[pos] * (raw[m] - knot), val)
            out[m] = np.clip(val, 0.0, None)
    return keep_raw_where_not_live(d, out)


def late(d, cal):
    shown = anchor(cal, d, SEASON_ANCHOR_S)
    q = d['inj'].astype(str).str.lower().str.startswith('questionable').to_numpy()
    for pos, f in QUESTIONABLE_WORKLOAD_FACTORS.items():
        m = q & (d['pos'] == pos).to_numpy() & (d['avail'] > SIDELINED_AVAILABILITY).to_numpy()
        shown[m] = np.round(shown[m] * f, 2)
    return shown + d['_late_offset'].to_numpy(dtype=float)


def load(path):
    d = load_dump(path)
    d = d[d['week'].between(3, 17)].drop_duplicates(KEYS, keep=False).reset_index(drop=True)
    d['wk_key'] = d['year'] * 100 + d['week']
    d['_late_offset'] = d['final'].to_numpy(float) - late(d.assign(_late_offset=0.0), line(d))
    d['actual0'] = d['actual'].fillna(0.0)
    d['expected'] = d['avail'] > EXPECTED
    return d


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else float('nan')


def pools(d, base, var):
    """Masks: every game (expected), every-game START (union of top-N among expected rows), box-score START (played)."""
    exp = d['expected'].to_numpy()
    played = d['played'].to_numpy() & d['actual'].notna().to_numpy()
    top_e = (top_n_mask(d, np.where(exp, base, -1)) | top_n_mask(d, np.where(exp, var, -1))) & exp
    top_p = (top_n_mask(d, base) | top_n_mask(d, var)) & played
    return exp, top_e, top_p


def score(d, base, var, mask, actual):
    e0, e1 = base[mask] - actual[mask], var[mask] - actual[mask]
    yrs = d['year'].to_numpy()[mask]
    seasons = sum(rmse(e1[yrs == y]) < rmse(e0[yrs == y]) for y in np.unique(yrs))
    return rmse(e1) - rmse(e0), f'{seasons}/{len(np.unique(yrs))}', float(np.mean(e0)), float(np.mean(e1))


def fit_slopes(d, train, knot):
    """Per position, the slope a minimising every-game RMSE of the final number on training rows below the knot."""
    out = {}
    for pos in POSITIONS:
        m = train & (d['pos'] == pos).to_numpy() & d['expected'].to_numpy() & (d['raw'].to_numpy() < knot)
        best, best_err = 1.0, np.inf
        for a in SLOPES:
            pred = late(d, line(d, knot, {pos: a}))
            err = rmse(pred[m] - d['actual0'].to_numpy()[m])
            if err < best_err:
                best, best_err = a, err
        out[pos] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--board', default='.sweeps/qb_style_capture.parquet.board.csv')
    ap.add_argument('--knots', default='4,6,8')
    ap.add_argument('--out', default='.sweeps/eval_low_end_line.txt')
    a = ap.parse_args()
    d = load(a.board)
    base = late(d, line(d))
    print('late steps reproduce the board:', round(float(np.abs(base - d['final'].to_numpy(float)).max()), 4))
    years = d['year'].to_numpy()
    lines = [f'{a.board}: {len(d):,} rows (2021-2025 weeks 3-17), held out by season', '']
    for knot in [float(k) for k in a.knots.split(',')]:
        var = base.copy()
        chosen = {}
        for y in np.unique(years):
            slopes = fit_slopes(d, years != y, knot)
            chosen[int(y)] = slopes
            var[years == y] = late(d, line(d, knot, slopes))[years == y]
        all_slopes = fit_slopes(d, np.ones(len(d), bool), knot)
        lines.append(f'knot {knot:g}: slope fit on all five seasons {all_slopes}; held-out choices {chosen}')
        exp, top_e, top_p = pools(d, base, var)
        act0, act = d['actual0'].to_numpy(float), d['actual'].to_numpy(float)
        for pos in POSITIONS + (None,):
            pm = np.ones(len(d), bool) if pos is None else (d['pos'] == pos).to_numpy()
            ev = score(d, base, var, exp & pm, act0)
            es = score(d, base, var, top_e & pm, act0)
            bs = score(d, base, var, top_p & pm, act)
            lines.append(f"   {pos or 'ALL':4s} every game {ev[0]:+.4f} ({ev[1]}, bias {ev[2]:+.2f} -> {ev[3]:+.2f})   "
                         f"every-game START {es[0]:+.4f} ({es[1]})   box-score START {bs[0]:+.4f} ({bs[1]})")
        lines.append('')
    text = '\n'.join(lines)
    print(text)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
