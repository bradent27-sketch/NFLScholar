"""
The points-scale re-fit (2026-10-25, user: "Refit the points scale"): the v4 points lines (WEEKLY_CALIBRATION, WR/TE cold/rest)
and the low end below the knots (LOW_END_POINTS_LINE) were fit before 7+ ships, the low end on its own day; this fits them again, together,
on the model as it now stands, from one scripts/calibration_lab.py dump of that model.

    lines      one (slope, intercept) per position, WR/TE split weeks 1-4 / 5-18 (the shipped structure), fit on PLAYED rows - a
               box-score row or a silent game (0 points), the harness's verdict definition since 2026-10-25 - over every live row
               (not sidelined, raw > 0), applied at strength k (shipped: k = 0.5, half way from the identity to the fit);
    low end    below the position's knot, L(knot) + a * (raw - knot), a fit on EVERY game the model expected the player to play
               (no game = 0), the target the user chose for it on 2026-10-22;
    anchor     the season anchor's s re-fit on the same training rows (eval_calibration_schemes.fit_s), or kept.

Held out by season: each season is predicted from a fit on the others, scored against the dump's own 'final' (today's
shipped pipeline), harness-style: PLAYED START pools (union of both arms' top-N), PLAYED ALL, EVERY ALL; 95% CI over weeks.
Then the constants from a fit on every season, ready to paste.

    python scripts/fit_points_scale.py --dump-path .sweeps/calibration_lab_silent.csv [--k 0.5,1.0] [--knots 4,6,8]
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

from data.weekly_projections import (LOW_END_POINTS_LINE, QUESTIONABLE_WORKLOAD_FACTORS, SEASON_ANCHOR_S,  # noqa: E402
                                     SIDELINED_AVAILABILITY, WEEKLY_CALIBRATION, WEEKLY_CALIBRATION_BY_BUCKET)
from scripts.eval_calibration_schemes import (BUCKETED_SHIPPED, POS, _line, anchor, boot_ci, fit_s,  # noqa: E402
                                              keep_raw_where_not_live, load_dump, top_n_mask)

LOW_POSITIONS = ('RB', 'WR', 'TE')
SLOPES = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6)
EXPECTED = 0.5


def bucket_of(pos, cold):
    return ('cold' if cold else 'rest') if pos in BUCKETED_SHIPPED else 'all'


def shipped_params():
    out = {}
    for pos in POS:
        for cold in (True, False):
            b = bucket_of(pos, cold)
            out[(pos, b)] = WEEKLY_CALIBRATION_BY_BUCKET.get(b, {}).get(pos, WEEKLY_CALIBRATION[pos]) if b != 'all' \
                else WEEKLY_CALIBRATION[pos]
    return out


def apply_lines(d, params, low=None):
    """raw -> the points line per (position, bucket), with the low end L(knot) + a * (raw - knot) below each knot in `low`
    ({pos: (knot, a)}), floored at 0; a sidelined or zero row keeps its raw total."""
    raw = d['raw'].to_numpy(dtype=float)
    out = raw.copy()
    for pos in POS:
        for cold in (True, False):
            m = ((d['pos'] == pos) & (d['cold'] == cold)).to_numpy()
            sl, ic = params[(pos, bucket_of(pos, cold))]
            val = sl * raw[m] + ic
            if low and pos in low:
                knot, a = low[pos]
                val = np.where(raw[m] < knot, sl * knot + ic + a * (raw[m] - knot), val)
            out[m] = np.clip(val, 0.0, None)
    return keep_raw_where_not_live(d, out)


def late(d, cal, s):
    """The shipped late steps after the line: the season anchor (s by position), the Questionable cut, and the dump's own
    leftover offset (the luck step and anything else after the line, carried as final - reproduction)."""
    shown = anchor(cal, d, s)
    q = d['inj'].astype(str).str.lower().str.startswith('questionable').to_numpy()
    for pos, f in QUESTIONABLE_WORKLOAD_FACTORS.items():
        m = q & (d['pos'] == pos).to_numpy() & (d['avail'] > SIDELINED_AVAILABILITY).to_numpy()
        shown[m] = np.round(shown[m] * f, 2)
    return shown + d['_late_offset'].to_numpy(dtype=float)


def fit_lines(d, train, k):
    m = train & d['played'].to_numpy() & d['live'].to_numpy()
    params = {}
    for pos in POS:
        for cold in (True, False):
            b = bucket_of(pos, cold)
            sel = m & (d['pos'] == pos).to_numpy()
            if b != 'all':
                sel &= (d['cold'] == cold).to_numpy()
            params[(pos, b)] = _line(d['raw'].to_numpy()[sel], d['actual'].to_numpy()[sel], k)
    return params


def fit_low(d, train, params, s, knots, target='every'):
    """Per position, the (knot, slope) minimising the RMSE of the final number on training rows below the largest knot (the same rows
    for every knot, so the knots compare): `target` 'every' = every game he was expected to play (no game = 0), 'played' = games he
    played (silent = 0)."""
    out = {}
    act0 = d['actual0'].to_numpy()
    rows = d['expected'].to_numpy() if target == 'every' else (d['played'].to_numpy() & d['actual'].notna().to_numpy())
    for pos in LOW_POSITIONS:
        best, best_err = None, np.inf
        for knot in knots:
            m = train & (d['pos'] == pos).to_numpy() & rows & (d['raw'].to_numpy() < max(knots))
            for a in SLOPES:
                pred = late(d, apply_lines(d, params, {pos: (knot, a)}), s)
                err = float(np.sqrt(np.mean((pred[m] - act0[m]) ** 2)))
                if err < best_err:
                    best, best_err = (knot, a), err
        out[pos] = best
    return out


def scores(d, base, var, window):
    """Harness-style paired scopes: PLAYED START-<pos>/START-ALL (union of both arms' top-N), PLAYED ALL, EVERY ALL."""
    w = window
    played = d['played'].to_numpy() & d['actual'].notna().to_numpy() & w
    exp = d['expected'].to_numpy() & w
    act, act0 = d['actual'].to_numpy(), d['actual0'].to_numpy()
    top = top_n_mask(d, base) | top_n_mask(d, var)
    scopes = [(f'START-{p}', played & top & (d['pos'] == p).to_numpy(), act) for p in POS]
    scopes += [('START-ALL', played & top, act), ('ALL', played, act), ('EVERY-ALL', exp, act0)]
    out = {}
    for name, m, y in scopes:
        eb, ev = base[m] - y[m], var[m] - y[m]
        g = pd.DataFrame({'wk': d['wk_key'].to_numpy()[m], 'sb': eb ** 2, 'sv': ev ** 2, 'n': 1}).groupby('wk').sum()
        lo, hi = boot_ci(g)
        out[name] = (float(np.sqrt((ev ** 2).mean()) - np.sqrt((eb ** 2).mean())), lo, hi, float(eb.mean()), float(ev.mean()),
                     int((g['sv'] < g['sb']).sum()), int((g['sv'] > g['sb']).sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-path', required=True)
    ap.add_argument('--k', default='0.5,1.0')
    ap.add_argument('--knots', default='4,6,8')
    ap.add_argument('--out', default='.sweeps/fit_points_scale.txt')
    ap.add_argument('--drop-weeks', default='2021:1',
                    help="year:week pairs left out (2021 week 1: the depth-chart archive behind v2_historical_ourlads starts in 2022, "
                         "so that opening board has no starters)")
    a = ap.parse_args()
    d = load_dump(a.dump_path)
    for pair in [x for x in a.drop_weeks.split(',') if x]:
        y, w = (int(v) for v in pair.split(':'))
        d = d[~((d['year'] == y) & (d['week'] == w))]
    d = d.reset_index(drop=True)
    d['wk_key'] = d['year'] * 100 + d['week']
    d['start_raw'] = top_n_mask(d, d['raw'].to_numpy())
    d['actual0'] = d['actual'].fillna(0.0)
    d['expected'] = d['avail'] > EXPECTED
    d['_late_offset'] = 0.0
    shipped_low = {p: v for p, v in LOW_END_POINTS_LINE.items()}
    s_ship = dict(SEASON_ANCHOR_S)
    d['_late_offset'] = d['final'].to_numpy(float) - late(d, apply_lines(d, shipped_params(), shipped_low), s_ship)
    base = d['final'].to_numpy(dtype=float)
    years = sorted(d['year'].unique())
    knots = [float(x) for x in a.knots.split(',')]
    window = d['week'].between(3, 17).to_numpy()
    lines = [f"{a.dump_path}: {len(d):,} rows, years {years}; played {int(d['played'].sum()):,} "
             f"(silent {int(d.get('silent', pd.Series(False)).astype(bool).sum()):,}); scored on weeks 3-17, held out by season",
             f"reproduction of the shipped pipeline: max |diff| {float(np.abs(late(d, apply_lines(d, shipped_params(), shipped_low), s_ship) - base).max()):.4f}",
             '']
    for k in [float(x) for x in a.k.split(',')]:
        for refit_s, low_target in ((False, 'every'), (False, 'played'), (True, 'every')):
            var = base.copy()
            for y in years:
                train = (d['year'] != y).to_numpy()
                test = ~train
                params = fit_lines(d, train, k)
                s = fit_s(d, apply_lines(d, params, shipped_low), train) if refit_s else s_ship
                low = fit_low(d, train, params, s, knots, low_target)
                var[test] = late(d, apply_lines(d, params, low), s)[test]
            sc = scores(d, base, var, window)
            label = f"k={k:g}, low end fit on {low_target} games, anchor s {'re-fit' if refit_s else 'kept'}"
            lines.append(f"{label}: held-out dRMSE vs today's pipeline (negative = better)   [95% CI over weeks]   bias   weeks better-worse")
            for name, (dr, lo, hi, bb, bv, wins, losses) in sc.items():
                lines.append(f"   {name:10s} {dr:+.4f} [{lo:+.4f},{hi:+.4f}]   bias {bb:+.2f} -> {bv:+.2f}   {wins}-{losses}")
            train = np.ones(len(d), dtype=bool)
            params = fit_lines(d, train, k)
            s = fit_s(d, apply_lines(d, params, shipped_low), train) if refit_s else s_ship
            low = fit_low(d, train, params, s, knots, low_target)
            lines.append(f"   fit on every season: lines {{{', '.join(f'{p}/{b}: ({v[0]:.3f}, {v[1]:.3f})' for (p, b), v in params.items())}}}")
            lines.append(f"   low end {low}   anchor s {s}")
            lines.append('')
    text = '\n'.join(lines)
    print(text)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
