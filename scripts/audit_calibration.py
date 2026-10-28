"""Calibration audit (2026-10-28, user: "check and make sure we are calibrating to the best of our ability ... If you think of any
new ideas/changes do it for the other positions as well"). Pre-set bars: docs/weekly_projections_methodology.md, 2026-10-28.

The whole calibration step is replayed offline from a scripts/calibration_lab.py dump: raw -> the points line (per position and
season phase, the low end below the knot) -> the sidelined/zero restore -> the season anchor -> the Questionable cut -> the
expected-points luck step. The replay of the shipped constants must reproduce the dump's own 'final' (weeks 3-17); anything it
does not model is carried as a per-row offset.

    part 1  is today's calibration fit to today's model: reliability by projection band, the slope of actual on shown points by
            position and season phase, weeks 1-2 and 18, and each fitted constant set re-fit in sample on all five seasons;
    part 2  new ideas at every position, forward-chained (each of 2023-2025 from earlier seasons only), each idea re-fit against
            the shipped structure re-fit on the same seasons.

    python scripts/audit_calibration.py --dump-path .sweeps/calibration_lab_v8.csv [--part 1|2|all]
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

from data.expected_points import load_ff_opportunity_weekly, luck_per_game  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402
from data.weekly_projections import (EXPECTED_POINTS_LUCK, LOW_END_POINTS_LINE, QUESTIONABLE_WORKLOAD_FACTORS,  # noqa: E402
                                     SEASON_ANCHOR_MIN_GAMES, SEASON_ANCHOR_S, SIDELINED_AVAILABILITY, WEEKLY_CALIBRATION,
                                     WEEKLY_CALIBRATION_BY_BUCKET, recency_calibration_scale)
from scripts.eval_calibration_schemes import POS, boot_ci, load_dump, top_n_mask  # noqa: E402

MIN_ROWS = 60
EXPECTED = 0.5                        # availability above which a player was expected to play (EVERY scopes)
SCOPES = ('START-ALL', 'START-QB', 'START-RB', 'START-WR', 'START-TE', 'ALL', 'EVERY-ALL', 'EVERY-QB')
BANDS = (0, 3, 6, 9, 12, 15, 18, 21, 25, 99)


# ---------------------------------------------------------------------------------------------------------------- data
def add_luck(d):
    """luck_pg / luck_games as the build's luck step sees them (earlier games this season, Full PPR)."""
    luck, games = np.full(len(d), np.nan), np.zeros(len(d))
    ff_by_year = {}
    for (y, w), idx in d.groupby(['year', 'week']).groups.items():
        if y not in ff_by_year:
            ff_by_year[y] = load_ff_opportunity_weekly(int(y))
        t = luck_per_game(ff_by_year[y], int(w))
        if t.empty:
            continue
        keys = clean_name_exact(d.loc[idx, 'player'].astype(str))
        luck[idx] = keys.map(t['luck_pg']).to_numpy(dtype=float)
        games[idx] = keys.map(t['games']).fillna(0).to_numpy(dtype=float)
    d['luck_pg'], d['luck_games'] = luck, games
    return d


def prepare(path, extra=''):
    d = load_dump(path)
    if extra:                                     # e.g. the 2026 weeks rebuilt with today's code: a season no fit has seen
        d = pd.concat([d, load_dump(extra)], ignore_index=True)
    d = d[~((d['year'] == 2021) & (d['week'] == 1))].reset_index(drop=True)     # no depth chart behind that board
    d['start_raw'] = top_n_mask(d, d['raw'].to_numpy())
    d['wk_key'] = d['year'] * 100 + d['week']
    d['start_raw'] = top_n_mask(d, d['raw'].to_numpy())
    d['actual0'] = d['actual'].fillna(0.0)
    d['expected'] = d['avail'] > EXPECTED
    d['qtag'] = d['inj'].astype(str).str.lower().str.startswith('questionable')
    d = add_luck(d)
    # the points line is not what week 1 (RB/WR/TE) and week 2 (RB) show: v2_opening_week_stat_calibration replaces it there
    wk, pos = d['week'].to_numpy(), d['pos'].to_numpy()
    d['line_applies'] = ~(((wk == 1) & np.isin(pos, ('RB', 'WR', 'TE'))) | ((wk == 2) & (pos == 'RB')))
    return d


class Arr:
    """The dump's columns as numpy arrays (the replay runs thousands of times in the grid fits)."""
    def __init__(self, d):
        self.n = len(d)
        self.pos = d['pos'].to_numpy()
        self.week = d['week'].to_numpy()
        self.year = d['year'].to_numpy()
        self.raw = d['raw'].to_numpy(dtype=float)
        self.final = d['final'].to_numpy(dtype=float)
        self.actual = d['actual'].to_numpy(dtype=float)
        self.actual0 = d['actual0'].to_numpy(dtype=float)
        self.played = d['played'].to_numpy() & d['actual'].notna().to_numpy()
        self.live = d['live'].to_numpy()
        self.avail = d['avail'].to_numpy(dtype=float)
        self.expected = d['expected'].to_numpy()
        self.amean = d['anchor_mean'].to_numpy(dtype=float)
        self.agames = d['anchor_games'].to_numpy(dtype=float)
        self.qtag = d['qtag'].to_numpy()
        self.luck = d['luck_pg'].to_numpy(dtype=float)
        self.lgames = d['luck_games'].to_numpy(dtype=float)
        self.games = pd.to_numeric(d['games_season'], errors='coerce').fillna(0).to_numpy(dtype=float)
        self.applies = d['line_applies'].to_numpy()
        self.start_raw = d['start_raw'].to_numpy()
        self.wk_key = d['wk_key'].to_numpy()
        self.window = (self.week >= 3) & (self.week <= 17)
        self.posm = {p: self.pos == p for p in POS}


# ------------------------------------------------------------------------------------------------------------ pipeline
def shipped_bucket(pos, week):
    """The shipped phase split: WR/TE weeks 1-4 'cold', 5+ 'rest'; QB/RB one line."""
    return ('cold' if week <= 4 else 'rest') if pos in ('WR', 'TE') else 'all'


def phase_bucket(cold_max=4, positions=('WR', 'TE'), mid_max=None):
    def f(pos, week):
        if pos not in positions:
            return 'all'
        if week <= cold_max:
            return 'cold'
        if mid_max is not None and week <= mid_max:
            return 'mid'
        return 'rest'
    return f


def bucket_array(A, bucket):
    weeks = np.unique(A.week)
    out = np.empty(A.n, dtype=object)
    for p in POS:
        for w in weeks:
            out[A.posm[p] & (A.week == w)] = bucket(p, w)
    return out


def shipped_config(recency=True):
    scale = recency_calibration_scale() if recency else {'lines': WEEKLY_CALIBRATION, 'by_bucket': WEEKLY_CALIBRATION_BY_BUCKET}
    lines = {}
    for p in POS:
        lines[(p, 'all')] = scale['lines'][p]
        for b in ('cold', 'rest'):
            lines[(p, b)] = scale['by_bucket'].get(b, {}).get(p, scale['lines'][p])
    return {'bucket': shipped_bucket, 'lines': lines, 'low': dict(LOW_END_POINTS_LINE), 'top': {}, 'evid': None,
            'anchor_s': dict(SEASON_ANCHOR_S), 'q': dict(QUESTIONABLE_WORKLOAD_FACTORS), 'luck': dict(EXPECTED_POINTS_LUCK)}


def line_stage(A, C, barr=None):
    """raw -> the points line (low end below / top end above the knots), clipped at 0; a sidelined or zero row keeps raw."""
    barr = bucket_array(A, C['bucket']) if barr is None else barr
    out = A.raw.copy()
    for p in POS:
        pm = A.posm[p]
        for b in np.unique(barr[pm]):
            m = pm & (barr == b)
            sl, ic = C['lines'][(p, b)]
            r = A.raw[m]
            if C.get('evid') and (p, b) in C['evid']:
                # an evidence-dependent slope: a slope per games-this-season bin, the line pivots on its own mean raw
                (edges, slopes, pivot) = C['evid'][(p, b)]
                g = A.games[m]
                k = np.clip(np.searchsorted(edges, g, side='right') - 1, 0, len(slopes) - 1)
                val = ic + sl * pivot + np.asarray(slopes)[k] * (r - pivot)
            else:
                val = ic + sl * r
            if p in C['low']:
                knot, a = C['low'][p]
                val = np.where(r < knot, ic + sl * knot + a * (r - knot), val)
            if p in C.get('top', {}):
                knot, a = C['top'][p]
                val = np.where(r > knot, ic + sl * knot + a * (r - knot), val)
            out[m] = np.clip(val, 0.0, None)
    return np.where(A.live, out, A.raw)


def late_stage(A, cal, C):
    """The season anchor, the Questionable cut and the luck step, in the build's order and with its eligibility rules."""
    shown = cal.copy()
    for p, s in C['anchor_s'].items():
        m = (A.posm[p] & ~np.isnan(A.amean) & (A.agames >= SEASON_ANCHOR_MIN_GAMES) & (cal > 0.005)
             & (A.avail > SIDELINED_AVAILABILITY))
        if isinstance(s, tuple):                  # (edges, s per bin): s by the number of games in the player's mean (idea 11)
            edges, ss = s
            s = np.asarray(ss)[np.clip(np.searchsorted(edges, A.agames, side='right') - 1, 0, len(ss) - 1)][m]
        shown[m] = np.clip(A.amean[m] + s * (cal[m] - A.amean[m]), 0.0, None)
    for p, f in C['q'].items():
        m = A.posm[p] & A.qtag & (A.avail > SIDELINED_AVAILABILITY) & (shown > 0.005)
        shown[m] = shown[m] * f
    for p, b in C['luck'].items():
        m = A.posm[p] & ~np.isnan(A.luck) & (A.lgames >= 1) & (A.avail > SIDELINED_AVAILABILITY) & (shown > 0.005)
        shown[m] = np.clip(shown[m] + b * A.luck[m], 0.0, None)
    return shown


def final_stage(A, shown, C):
    """The final-stage line (idea 10): actual = a + b * shown per (position, phase) on the number every other step left."""
    fl = C.get('final_line')
    if not fl:
        return shown
    out = shown.copy()
    phase = C['final_phase']
    for (p, ph), (sl, ic) in fl.items():
        m = A.posm[p] & (phase(A.week) == ph) & A.live & (shown > 0.005)
        out[m] = np.clip(ic + sl * shown[m], 0.0, None)
    return out


def run(A, C, offset, barr=None):
    """The shown total under config C; rows where the points line does not apply keep the dump's own number."""
    pred = final_stage(A, late_stage(A, line_stage(A, C, barr), C) + offset, C)
    return np.where(A.applies, pred, A.final)


# ------------------------------------------------------------------------------------------------------------- fitting
def wls_line(p, a, w, k):
    if len(p) < MIN_ROWS or np.var(p) < 1e-9:
        return 1.0, 0.0
    b, c = np.polyfit(p, a, 1, w=np.sqrt(w))
    return 1.0 + k * (b - 1.0), k * c


def fit_rows(A, train, pool='all', where_applied=False, low=None):
    m = train & A.played & A.live
    if pool == 'start':
        m &= A.start_raw
    if where_applied:
        m &= A.applies
    out = {}
    for p in POS:
        mp = m & A.posm[p]
        if low and p in low and pool == 'above_knot':
            mp &= A.raw >= low[p][0]
        out[p] = mp
    return out


def fit_lines(A, train, bucket, k=0.5, decay=None, pool='all', where_applied=False, low=None, barr=None):
    """Lines per (position, bucket) on PLAYED live training rows; `decay` {pos: d} weights each season d ** (seasons back from
    the newest training season); pool 'all' / 'start' (top-N by raw) / 'above_knot' (rows the low end does not replace)."""
    barr = bucket_array(A, bucket) if barr is None else barr
    rows = fit_rows(A, train, pool, where_applied, low)
    newest = A.year[train].max()
    out = {}
    for p in POS:
        w = np.ones(A.n)
        if decay and p in decay:
            w = decay[p] ** (newest - A.year).clip(0)
        for b in np.unique(barr[A.posm[p]]):
            sel = rows[p] & (barr == b)
            out[(p, b)] = wls_line(A.raw[sel], A.actual[sel], w[sel], k)
    return out


def fit_end(A, train, C, offset, which, grids, target='every', positions=None, barr=None):
    """Per position, the (knot, slope) for the low ('low') or top ('top') end minimising the RMSE of the shown total on training
    rows beyond the extreme knot of the grid (the same rows for every knot, so knots compare). target 'every' = every game he was
    expected to play (no game = 0), 'played' = played games."""
    knots, slopes = grids
    out = dict(C[which])
    rows = (A.expected if target == 'every' else A.played) & train & A.applies
    y = A.actual0 if target == 'every' else np.nan_to_num(A.actual)
    for p in (positions or POS):
        region = (A.raw < max(knots)) if which == 'low' else (A.raw > min(knots))
        m = rows & A.posm[p] & region & A.live
        if m.sum() < MIN_ROWS:
            continue
        best = C[which].get(p)                                       # the config as it stands is the bar to beat
        best_err = float(np.mean((run(A, C, offset, barr)[m] - y[m]) ** 2))
        for knot in knots:
            for a in slopes:
                c = dict(C)
                c[which] = dict(C[which])
                c[which][p] = (knot, a)
                pred = run(A, c, offset, barr)
                err = float(np.mean((pred[m] - y[m]) ** 2))
                if err < best_err:
                    best, best_err = (knot, a), err
        out[p] = best
    return out


def fit_anchor_s(A, cal, train, positions):
    out = {}
    m0 = (train & A.start_raw & A.played & A.window & ~np.isnan(A.amean) & (A.agames >= SEASON_ANCHOR_MIN_GAMES)
          & (cal > 0.005) & (A.avail > SIDELINED_AVAILABILITY))
    for p in positions:
        m = m0 & A.posm[p]
        x, y = cal[m] - A.amean[m], A.actual[m] - A.amean[m]
        out[p] = float(np.clip((x * y).sum() / (x * x).sum(), 0.3, 1.2)) if m.sum() > MIN_ROWS else 1.0
    return out


def fit_anchor_s_by_games(A, cal, train, positions, edges=(2, 4, 7)):
    """fit_anchor_s within each games-in-the-mean bin: {pos: (edges, [s per bin])}."""
    out = {}
    k = np.clip(np.searchsorted(edges, A.agames, side='right') - 1, 0, len(edges) - 1)
    for j in range(len(edges)):
        s = fit_anchor_s(A, cal, train & (k == j), positions)
        for p in positions:
            out.setdefault(p, (np.asarray(edges, dtype=float), []))[1].append(s[p])
    return out


def fit_q(A, C, offset, train, positions):
    """Least squares through the origin of actual on the shown total before the cut, Q-tagged played START rows."""
    c = dict(C, q={}, luck={})
    before = late_stage(A, line_stage(A, c), c)
    out = {}
    for p in positions:
        m = train & A.posm[p] & A.qtag & A.played & A.window & A.start_raw & (A.avail > SIDELINED_AVAILABILITY) & (before > 0.005)
        out[p] = float((A.actual[m] * before[m]).sum() / (before[m] ** 2).sum()) if m.sum() >= 30 else 1.0
    return out, int(sum((train & A.posm[p] & A.qtag & A.played & A.window & A.start_raw).sum() for p in positions))


def fit_luck(A, C, offset, train, positions):
    """Slope of the miss (actual - shown before the luck step) on luck per game, played START rows with an earlier game."""
    c = dict(C, luck={})
    before = late_stage(A, line_stage(A, c), c) + offset
    out = {}
    for p in positions:
        m = train & A.posm[p] & A.played & A.window & A.start_raw & ~np.isnan(A.luck) & (A.lgames >= 1)
        if m.sum() < MIN_ROWS:
            out[p] = 0.0
            continue
        out[p] = float(np.polyfit(A.luck[m], A.actual[m] - before[m], 1)[0])
    return out


# ------------------------------------------------------------------------------------------------------------- scoring
def scores(A, base, var, window):
    """Harness-style paired scopes: PLAYED START-<pos>/START-ALL (union of both arms' top-N), PLAYED ALL, EVERY ALL/QB."""
    d_idx = pd.DataFrame({'wk_key': A.wk_key, 'pos': A.pos})
    top = top_n_mask(d_idx, base) | top_n_mask(d_idx, var)
    played = A.played & window
    exp = A.expected & window
    sc = [(f'START-{p}', played & top & A.posm[p], A.actual) for p in POS]
    sc += [('START-ALL', played & top, A.actual), ('ALL', played, A.actual), ('EVERY-ALL', exp, A.actual0),
           ('EVERY-QB', exp & A.posm['QB'], A.actual0)]
    out = {}
    for name, m, y in sc:
        eb, ev = base[m] - y[m], var[m] - y[m]
        g = pd.DataFrame({'wk': A.wk_key[m], 'sb': eb ** 2, 'sv': ev ** 2, 'n': 1}).groupby('wk').sum()
        lo, hi = boot_ci(g)
        out[name] = (float(np.sqrt((ev ** 2).mean()) - np.sqrt((eb ** 2).mean())), lo, hi, float(eb.mean()), float(ev.mean()),
                     int((g['sv'] < g['sb']).sum()), int((g['sv'] > g['sb']).sum()))
    return out


def fmt(sc, keys=SCOPES):
    return '  '.join(f'{k} {sc[k][0]:+.4f}[{sc[k][1]:+.4f},{sc[k][2]:+.4f}]' for k in keys)


def verdict(sc_all, by_season, scope='START'):
    """The pre-set bar (methodology 2026-10-28)."""
    s = sc_all
    if scope == 'START':
        ok = (s['START-ALL'][0] <= -0.001 and s['START-ALL'][2] < 0.001
              and all(s[f'START-{p}'][0] <= 0.002 for p in POS)
              and s['ALL'][0] <= 0.0005 and s['EVERY-ALL'][0] <= 0.0005
              and all(v <= 0.001 for v in by_season.values()))
    else:      # an idea aimed at the every-game QB scope (the QB low end)
        ok = s['EVERY-QB'][0] < 0 and s['EVERY-QB'][2] < 0 and s['ALL'][0] <= 0 and abs(s['START-ALL'][0]) < 0.0005
    return 'PASS' if ok else 'fail'


# ------------------------------------------------------------------------------------------------------------- part 1
def reliability(A, shown, out):
    m = A.played & A.window & A.live
    out.append('Reliability, weeks 3-17, played live rows: shown band -> n, mean shown, mean actual, bias (shown - actual)')
    for p in POS:
        parts = []
        for lo, hi in zip(BANDS[:-1], BANDS[1:]):
            mm = m & A.posm[p] & (shown >= lo) & (shown < hi)
            if mm.sum() < 30:
                continue
            parts.append(f'{lo}-{hi if hi < 99 else "+"}: {mm.sum()} {shown[mm].mean():.1f}/{A.actual[mm].mean():.1f} '
                         f'{shown[mm].mean() - A.actual[mm].mean():+.2f}')
        out.append(f'  {p}: ' + ' | '.join(parts))
    d_idx = pd.DataFrame({'wk_key': A.wk_key, 'pos': A.pos})
    top = top_n_mask(d_idx, shown)
    out.append('\nSlope of actual on shown (OLS) and bias, START pool (top-N by shown), played, by season phase:')
    phases = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 12), (13, 17), (18, 18))
    for p in POS:
        parts = []
        for w0, w1 in phases:
            mm = A.played & top & A.posm[p] & (A.week >= w0) & (A.week <= w1)
            if mm.sum() < 40:
                continue
            b = np.polyfit(shown[mm], A.actual[mm], 1)[0]
            parts.append(f'wk{w0}-{w1} n{mm.sum()} slope {b:.2f} bias {(shown[mm] - A.actual[mm]).mean():+.2f}')
        out.append(f'  {p}: ' + ' | '.join(parts))
    out.append('\nSame, by season (weeks 3-17):')
    for p in POS:
        parts = []
        for y in np.unique(A.year):
            mm = A.played & top & A.posm[p] & A.window & (A.year == y)
            b = np.polyfit(shown[mm], A.actual[mm], 1)[0]
            parts.append(f'{y} slope {b:.2f} bias {(shown[mm] - A.actual[mm]).mean():+.2f}')
        out.append(f'  {p}: ' + ' | '.join(parts))
    out.append('\nTop of the START pool by shown rank within position (weeks 3-17, played): bias shown - actual')
    rank = pd.Series(shown).groupby([A.wk_key, A.pos]).rank(method='first', ascending=False).to_numpy()
    for p in POS:
        parts = []
        for r0, r1 in ((1, 3), (4, 8), (9, 12), (13, 24), (25, 36)):
            mm = A.played & A.window & A.posm[p] & (rank >= r0) & (rank <= r1)
            if mm.sum() < 40:
                continue
            parts.append(f'#{r0}-{r1} n{mm.sum()} {(shown[mm] - A.actual[mm]).mean():+.2f}')
        out.append(f'  {p}: ' + ' | '.join(parts))


def part1(A, d, C_ship, offset, out):
    reliability(A, A.final, out)
    out.append('')
    full = np.ones(A.n, dtype=bool)
    barr = bucket_array(A, shipped_bucket)

    def report(label, C, scope_keys=SCOPES):
        var = run(A, C, offset, barr)
        sc = scores(A, A.final, var, A.window)
        out.append(f'  {label}: ' + fmt(sc, scope_keys))
        return sc

    out.append('In sample (fit on all five seasons) against today\'s constants, weeks 3-17 (dRMSE, negative = re-fit better):')
    # lines, the shipped structure: equal weights, then WR/TE at decay 0.5 (today's recency delta, re-derived)
    eq = fit_lines(A, full, shipped_bucket, barr=barr)
    rec = fit_lines(A, full, shipped_bucket, decay={'WR': 0.5, 'TE': 0.5}, barr=barr)
    C = dict(C_ship, lines=rec)
    report('lines re-fit (QB/RB equal, WR/TE decay 0.5)', C)
    out.append('     re-fit lines: ' + ', '.join(f'{p}/{b} ({v[0]:.3f}, {v[1]:.3f})' for (p, b), v in rec.items()))
    out.append('     shipped     : ' + ', '.join(f'{p}/{b} ({v[0]:.3f}, {v[1]:.3f})' for (p, b), v in C_ship['lines'].items()
                                                  if (p, b) in rec))
    delta = {k: (rec[k][0] - eq[k][0], rec[k][1] - eq[k][1]) for k in rec if k[0] in ('WR', 'TE')}
    out.append('     recency delta re-derived: ' + ', '.join(f'{p}/{b} ({v[0]:+.4f}, {v[1]:+.4f})' for (p, b), v in delta.items()))
    low = fit_end(A, full, C_ship, offset, 'low', ((4.0, 6.0, 8.0), (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)), 'every',
                  ('RB', 'WR', 'TE'), barr)
    out.append(f'     low end re-fit {low} (shipped {C_ship["low"]})')
    report('low end re-fit (every game)', dict(C_ship, low=low))
    s = fit_anchor_s(A, line_stage(A, C_ship, barr), full, ('RB', 'WR', 'TE'))
    out.append(f'     anchor s re-fit {s} (shipped {C_ship["anchor_s"]})')
    report('anchor s re-fit', dict(C_ship, anchor_s=s))
    q, nq = fit_q(A, C_ship, offset, full, ('WR',))
    out.append(f'     Questionable re-fit {q} on {nq} WR rows (shipped {C_ship["q"]})')
    report('Questionable WR re-fit', dict(C_ship, q=q))
    lk = fit_luck(A, C_ship, offset, full, ('WR', 'TE'))
    out.append(f'     luck re-fit {lk} (shipped {C_ship["luck"]})')
    report('luck WR/TE re-fit', dict(C_ship, luck=lk))
    C_all = dict(C_ship, lines=rec, low=low, anchor_s=s, q=q, luck=lk)
    report('everything re-fit at once', C_all)


# ------------------------------------------------------------------------------------------------------------- part 2
def forward(A, tests, offset, make):
    """Each test season predicted by `make(train) -> (config, bucket array)` fit on earlier seasons only."""
    out = A.final.copy()
    for t in tests:
        C, barr = make(A.year < t)
        out[A.year == t] = run(A, C, offset, barr)[A.year == t]
    return out


def ref_maker(A, C_ship, bucket=shipped_bucket, **kw):
    barr = bucket_array(A, bucket)

    def make(train):
        lines = fit_lines(A, train, bucket, decay=kw.get('decay', {'WR': 0.5, 'TE': 0.5}), k=kw.get('k', 0.5),
                          pool=kw.get('pool', 'all'), where_applied=kw.get('where_applied', False), low=C_ship['low'], barr=barr)
        C = dict(C_ship, bucket=bucket, lines=lines)
        if kw.get('refit_s'):
            C['anchor_s'] = fit_anchor_s(A, line_stage(A, C, barr), train, kw.get('s_positions', ('RB', 'WR', 'TE')))
        return C, barr
    return make


def part2(A, C_ship, offset, out, only=None, tests=(2023, 2024, 2025)):
    tests = list(tests)
    base = forward(A, tests, offset, ref_maker(A, C_ship))
    win = A.window & np.isin(A.year, tests)
    out.append(f'Forward-chained, test seasons {tests}, weeks 3-17. Reference: the shipped structure re-fit on the same seasons '
               '(QB/RB one line, WR/TE weeks 1-4 / 5+, WR/TE season weights 0.5, k 0.5, lines on every played live row).')
    sc_ref = scores(A, A.final, base, win)
    out.append('  reference vs today\'s constants (in sample): ' + fmt(sc_ref, ('START-ALL', 'ALL', 'EVERY-ALL')))
    out.append('dRMSE idea - reference (negative = idea better) [95% CI over weeks]; START-ALL by season; verdict on the pre-set bar')

    passed = []

    def test(label, make, scope='START', window=None):
        if only and not any(o in label for o in only):
            return None
        w = win if window is None else window
        var = forward(A, tests, offset, make)
        sc = scores(A, base, var, w)
        by = {t: scores(A, base, var, w & (A.year == t))['START-ALL'][0] for t in tests}
        v = verdict(sc, by, scope)
        out.append(f'  {label:58s} ' + fmt(sc) + '  | ' + ' '.join(f'{t} {x:+.4f}' for t, x in by.items()) + f'  -> {v}')
        if v == 'PASS':
            passed.append((label, make, w))
        return sc

    # 1. recency per position
    for dec in (0.7, 0.5):
        test(f'recency QB decay {dec}', ref_maker(A, C_ship, decay={'QB': dec, 'WR': 0.5, 'TE': 0.5}))
        test(f'recency RB decay {dec}', ref_maker(A, C_ship, decay={'RB': dec, 'WR': 0.5, 'TE': 0.5}))
    test('recency WR/TE off (equal weights everywhere)', ref_maker(A, C_ship, decay={}))
    # 2. fit where the line is applied
    test('fit on the weeks the line applies (not wk1 RB/WR/TE, wk2 RB)', ref_maker(A, C_ship, where_applied=True))
    test('fit above the low-end knot only (RB/WR/TE)', ref_maker(A, C_ship, pool='above_knot'))
    test('fit on START rows only', ref_maker(A, C_ship, pool='start'))
    # 3. strength
    for k in (0.75, 1.0):
        test(f'strength k {k} (anchor s re-fit)', ref_maker(A, C_ship, k=k, refit_s=True))
    test('strength k 0.5, anchor s re-fit (control for the row above)', ref_maker(A, C_ship, refit_s=True))
    # 4. season phases at every position, the boundary
    for cm in (4, 5, 6, 8):
        test(f'phase split at every position, cold <= wk{cm}', ref_maker(A, C_ship, bucket=phase_bucket(cm, POS)))
    for cm in (5, 6, 8):
        test(f'phase split WR/TE only, cold <= wk{cm}', ref_maker(A, C_ship, bucket=phase_bucket(cm, ('WR', 'TE'))))
    test('three phases every position (1-4 / 5-8 / 9+)', ref_maker(A, C_ship, bucket=phase_bucket(4, POS, mid_max=8)))
    test('three phases WR/TE (1-4 / 5-8 / 9+)', ref_maker(A, C_ship, bucket=phase_bucket(4, ('WR', 'TE'), mid_max=8)))
    # 5. top end: its own slope above a high knot, every position
    top_grid = {'QB': ((20.0, 23.0, 26.0), (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)),
                'RB': ((14.0, 17.0, 20.0), (0.6, 0.7, 0.8, 0.9, 1.0, 1.1)),
                'WR': ((14.0, 17.0, 20.0), (0.6, 0.7, 0.8, 0.9, 1.0, 1.1)),
                'TE': ((10.0, 13.0, 16.0), (0.6, 0.7, 0.8, 0.9, 1.0, 1.1))}
    ref = ref_maker(A, C_ship)

    def top_maker(positions, target):
        def make(train):
            C, barr = ref(train)
            top = {}
            for p in positions:
                top.update(fit_end(A, train, dict(C, top={}), offset, 'top', top_grid[p], target, (p,), barr))
            return dict(C, top={p: v for p, v in top.items() if v}), barr
        return make
    for p in POS:
        test(f'top-end knot {p} (fit on played games)', top_maker((p,), 'played'))
    test('top-end knot every position', top_maker(POS, 'played'))

    # 6. evidence: the slope by games this season (0-1 / 2-4 / 5+), every position, pivoting on the bucket's mean raw
    def evid_maker(positions, edges=(0, 2, 5)):
        def make(train):
            C, barr = ref(train)
            rows = fit_rows(A, train)
            ev = {}
            for p in positions:
                for b in np.unique(barr[A.posm[p]]):
                    sel = rows[p] & (barr == b)
                    sl, _ic = C['lines'][(p, b)]
                    pivot = float(A.raw[sel].mean())
                    k = np.clip(np.searchsorted(edges, A.games[sel], side='right') - 1, 0, len(edges) - 1)
                    slopes = []
                    for j in range(len(edges)):
                        mm = k == j
                        if mm.sum() < MIN_ROWS:
                            slopes.append(sl)
                            continue
                        bfit = float(np.polyfit(A.raw[sel][mm], A.actual[sel][mm], 1)[0])    # the bin's own slope, half strength
                        slopes.append(1.0 + 0.5 * (bfit - 1.0))
                    ev[(p, b)] = (np.asarray(edges, dtype=float), slopes, pivot)
            return dict(C, evid=ev), barr
        return make
    for p in POS:
        test(f'evidence slope by games this season {p}', evid_maker((p,)))
    test('evidence slope every position', evid_maker(POS))

    # 7. QB low end (every-game target)
    def qb_low_maker(train):
        C, barr = ref(train)
        low = fit_end(A, train, C, offset, 'low', ((4.0, 6.0, 8.0), (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)), 'every', ('QB',),
                      barr)
        return dict(C, low=low), barr
    test('QB low end (every game)', qb_low_maker, scope='EVERY')

    # 8. the post-line steps at the positions without them
    def post_maker(kind, positions):
        def make(train):
            C, barr = ref(train)
            if kind == 'anchor':
                s = fit_anchor_s(A, line_stage(A, C, barr), train, positions)
                C = dict(C, anchor_s={**C['anchor_s'], **s})
            elif kind == 'q':
                q, _n = fit_q(A, C, offset, train, positions)
                C = dict(C, q={**C['q'], **q})
            elif kind == 'luck':
                C = dict(C, luck={**C['luck'], **fit_luck(A, C, offset, train, positions)})
            return C, barr
        return make
    test('season anchor for QB', post_maker('anchor', ('QB',)))
    for p in ('RB', 'TE', 'QB'):
        test(f'Questionable cut for {p}', post_maker('q', (p,)))
    for p in ('RB', 'QB'):
        test(f'luck step for {p}', post_maker('luck', (p,)))
    test('anchor s re-fit forward (RB/WR/TE)', post_maker('anchor', ('RB', 'WR', 'TE')))
    test('luck re-fit forward (WR/TE)', post_maker('luck', ('WR', 'TE')))
    test('Questionable re-fit forward (WR)', post_maker('q', ('WR',)))

    # 10. the final-stage line: a line on the shown number itself, after every step, fit on START rows
    def final_maker(phase, k):
        def make(train):
            C, barr = ref(train)
            pred = run(A, C, offset, barr)
            top = top_n_mask(pd.DataFrame({'wk_key': A.wk_key, 'pos': A.pos}), pred)
            ph = phase(A.week)
            fl = {}
            for p in POS:
                for v in np.unique(ph[A.window]):
                    m = train & A.played & A.window & top & A.posm[p] & (ph == v)
                    if m.sum() >= MIN_ROWS:
                        b, c = np.polyfit(pred[m], A.actual[m], 1)
                        fl[(p, v)] = (1.0 + k * (b - 1.0), k * c)
            return dict(C, final_line=fl, final_phase=phase), barr
        return make
    one = lambda w: np.zeros_like(w)                                   # noqa: E731
    two = lambda w: (w >= 9).astype(int)                               # noqa: E731
    for k in (0.5, 1.0):
        test(f'final-stage line, all weeks, k {k}', final_maker(one, k))
        test(f'final-stage line, weeks 3-8 / 9+, k {k}', final_maker(two, k))

    # 11. the anchor's s by the number of games in the player's mean (2-3 / 4-6 / 7+)
    def anchor_games_maker(positions):
        def make(train):
            C, barr = ref(train)
            s = fit_anchor_s_by_games(A, line_stage(A, C, barr), train, positions)
            return dict(C, anchor_s={**C['anchor_s'], **s}), barr
        return make
    test('anchor s by games in the mean (RB/WR/TE)', anchor_games_maker(('RB', 'WR', 'TE')))
    test('anchor s by games in the mean (QB/RB/WR/TE)', anchor_games_maker(POS))

    # 9. week 18 (rested starters): a week-18 line at every position, scored on week 18 alone
    def w18_bucket(pos, week):
        return 'w18' if week == 18 else shipped_bucket(pos, week)
    w18 = A.played & (A.week == 18) & np.isin(A.year, tests)
    out.append('\nWeek 18 alone (n START-ALL rows small; reference as above):')
    test('week-18 line at every position', ref_maker(A, C_ship, bucket=w18_bucket), window=w18)

    # the shippable form of each passing idea: what it changes, moved onto today's constants (a line change as the delta
    # idea - reference, fit on earlier seasons only, as the recency delta was), scored against today's pipeline
    if passed:
        out.append("\nShippable form of each passing idea, forward-chained against today's pipeline (bar: START-ALL <= 0, no season "
                   'worse than +0.001):')
    for label, make, w in passed:
        def ship_make(train, make=make):
            C_i, barr_i = make(train)
            C_r, _barr_r = ref_maker(A, C_ship)(train)
            C = {k: (dict(v) if isinstance(v, dict) else v) for k, v in C_ship.items()}
            if C_i['bucket'] is C_r['bucket']:
                for key, (sl, ic) in C_r['lines'].items():
                    di = C_i['lines'][key]
                    C['lines'][key] = (C_ship['lines'][key][0] + di[0] - sl, C_ship['lines'][key][1] + di[1] - ic)
                barr = bucket_array(A, shipped_bucket)
            else:                                       # a new phase structure: its own re-fit lines
                C['bucket'], C['lines'], barr = C_i['bucket'], C_i['lines'], barr_i
            for key in ('low', 'top', 'evid', 'anchor_s', 'q', 'luck', 'final_line', 'final_phase'):
                if C_i.get(key) != C_r.get(key):
                    C[key] = C_i[key]
            return C, barr
        var = forward(A, tests, offset, ship_make)
        sc = scores(A, A.final, var, w)
        by = {t: scores(A, A.final, var, w & (A.year == t))['START-ALL'][0] for t in tests}
        ok = sc['START-ALL'][0] <= 0 and all(x <= 0.001 for x in by.values())
        out.append(f'  {label:58s} ' + fmt(sc) + '  | ' + ' '.join(f'{t} {x:+.4f}' for t, x in by.items())
                   + f"  -> {'PASS' if ok else 'fail'}")
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-path', default='.sweeps/calibration_lab_v8.csv')
    ap.add_argument('--part', default='all', choices=('1', '2', 'all'))
    ap.add_argument('--no-recency', action='store_true', help='the dump was built without v2_recency_calibration_wrte')
    ap.add_argument('--only', default='', help='part 2: comma-separated substrings of the idea labels to run')
    ap.add_argument('--extra-dump', default='', help='a second dump appended (the 2026 weeks rebuilt with the current code)')
    ap.add_argument('--tests', default='2023,2024,2025', help='part 2: the forward-chained test seasons')
    ap.add_argument('--out', default='.sweeps/audit_calibration.txt')
    a = ap.parse_args()
    d = prepare(a.dump_path, a.extra_dump)
    A = Arr(d)
    C_ship = shipped_config(recency=not a.no_recency)
    rep = run(A, C_ship, np.zeros(A.n))
    diff = (A.final - rep)
    win = A.window & A.applies
    out = [f'{a.dump_path}: {A.n:,} rows, seasons {sorted(np.unique(A.year))}, played {int(A.played.sum()):,}',
           f'replay of the shipped constants, weeks 3-17: max |final - replay| {np.abs(diff[win]).max():.3f}, '
           f'rows off by > 0.05: {int((np.abs(diff[win]) > 0.05).sum())} of {int(win.sum())}', '']
    offset = np.where(A.applies, diff, 0.0)
    if a.part in ('1', 'all'):
        part1(A, d, C_ship, offset, out)
        out.append('')
    if a.part in ('2', 'all'):
        part2(A, C_ship, offset, out, [s for s in a.only.split(',') if s], [int(t) for t in a.tests.split(',')])
    text = '\n'.join(out)
    print(text)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
