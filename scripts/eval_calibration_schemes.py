"""Score calibration SCHEMES offline against the shipped pipeline, the way harness v2 scores a flag (2026-10-08).

Input: a `scripts/calibration_lab.py --mode dump` file (the harness base-arm board with every pipeline stage
recorded). Every scheme replaces the stage between the shipped stat line and the final points total:

    shipped        the board's own 'Model Proj Pts' (v4 points lines -> sidelined/zero restore -> season anchor)
    line_*         the shipped STRUCTURE refit: one points line per position (WR/TE cold/rest), then the anchor
    stat_*         a line per (position, scoring stat) on the stat itself; points = score(calibrated stat line)
    comp_*         one regression of actual points on the projection's per-stat POINT components (a slope per
                   component - TD points can shrink more than yardage points), i.e. a per-stat calibration fit
                   directly to the points target

Every scheme is followed by the season anchor with `s` refit on the same training years, and keeps the shipped
rules: a sidelined row (Availability <= 0.01) or a zero row keeps its raw total. Fits use rows with a box score
(E[actual | played], the harness's criterion).

Held-out design: leave-one-year-out - each season is predicted by a fit on the other four, so every scored
prediction is out of sample. The shipped lines/anchor were fit on 2021-2025 (in sample everywhere), which favours
`shipped`. Scoring mirrors harness v2: per (year, week, position) START pool = union of both arms' top-N
(STARTABLE_N) over ALL rows, restricted to rows with a box score; ALL = every row with a box score; RMSE / bias /
pairwise; a 95% CI on dRMSE from a bootstrap over weeks.

    python scripts/eval_calibration_schemes.py --dump-path .sweeps/calibration_lab_v7.csv
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

from data.weekly_projections import (SEASON_ANCHOR_MIN_GAMES, SEASON_ANCHOR_S, SIDELINED_AVAILABILITY,  # noqa: E402
                                     WEEKLY_CALIBRATION_COLD_MAX_WEEK, _weekly_calibration_for)
from scripts.harness_v2 import STARTABLE_N, pairwise_acc  # noqa: E402

POS = ('QB', 'RB', 'WR', 'TE')
WEIGHTS = {'passing_yards': 0.04, 'passing_tds': 4.0, 'passing_interceptions': -2.0,
           'rushing_yards': 0.1, 'rushing_tds': 6.0,
           'receptions': 1.0, 'receiving_yards': 0.1, 'receiving_tds': 6.0}
# the scoring stats each position's line is fit on (others keep their raw value)
POS_STATS = {
    'QB': ('passing_yards', 'passing_tds', 'passing_interceptions', 'rushing_yards', 'rushing_tds'),
    'RB': ('rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards', 'receiving_tds'),
    'WR': ('receptions', 'receiving_yards', 'receiving_tds', 'rushing_yards'),
    'TE': ('receptions', 'receiving_yards', 'receiving_tds'),
}
BUCKETED_SHIPPED = ('WR', 'TE')
ANCHOR_POS = tuple(SEASON_ANCHOR_S)
MIN_ROWS = 60


# ------------------------------------------------------------------------------------------------ data
def load_dump(path, week1_path=''):
    """The dump, with its week-1 rows replaced by `week1_path` when given (a `--extra-features
    v2_historical_ourlads` week-1 dump: without that flag a historical opening board has no depth chart; it covers
    2022-2025, so 2021 week 1 is dropped)."""
    d = pd.read_csv(path, low_memory=False)
    if week1_path:
        w1 = pd.read_csv(week1_path, low_memory=False)
        d = pd.concat([d[d['week'] != 1], w1], ignore_index=True)
    d['played'] = d['played'].astype(bool)
    d['avail'] = pd.to_numeric(d.get('avail', 1.0), errors='coerce').fillna(1.0) if 'avail' in d else 1.0
    for s in WEIGHTS:
        for pre in ('p_', 'a_'):
            c = pre + s
            d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0.0) if c in d.columns else 0.0
    d['actual'] = pd.to_numeric(d['actual'], errors='coerce')
    d['cold'] = d['week'] <= WEEKLY_CALIBRATION_COLD_MAX_WEEK
    # rows a calibration may move: not sidelined, a non-zero raw total
    d['live'] = (d['avail'] > SIDELINED_AVAILABILITY) & (d['raw'] > 0.005)
    if 'anchor_mean' not in d.columns:                      # an older dump: derive from its own actuals
        from scripts.analyze_model_weak_spots import add_benchmarks
        d = add_benchmarks(d).rename(columns={'bm_mean': 'anchor_mean', 'bm_n': 'anchor_games'})
    d['anchor_games'] = pd.to_numeric(d['anchor_games'], errors='coerce').fillna(0)
    if 'final' not in d.columns:
        d['final'] = anchor(shipped_line(d), d, {p: SEASON_ANCHOR_S[p] for p in ANCHOR_POS})
    d = d.reset_index(drop=True)
    d['wk_key'] = d['year'] * 100 + d['week']
    d['start_raw'] = top_n_mask(d, d['raw'].to_numpy())
    return d


def top_n_mask(d, pred):
    """True for the top STARTABLE_N[pos] rows by `pred` within each (year, week, pos), over ALL rows."""
    r = pd.Series(pred, index=d.index).groupby([d['wk_key'], d['pos']]).rank(method='first', ascending=False)
    n = d['pos'].map(STARTABLE_N).fillna(30)
    return (r <= n).to_numpy()


def score_points(d, stat_values):
    """Points from a {stat: array} line (floored at 0), the projection's own scoring."""
    pts = np.zeros(len(d))
    for s, w in WEIGHTS.items():
        pts += w * stat_values[s]
    return np.clip(pts, 0.0, None)


def keep_raw_where_not_live(d, pred):
    return np.where(d['live'].to_numpy(), pred, d['raw'].to_numpy())


def shipped_line(d):
    out = d['raw'].to_numpy(dtype=float).copy()
    for pos in POS:
        for cold in (True, False):
            m = ((d['pos'] == pos) & (d['cold'] == cold)).to_numpy()
            sl, ic = _weekly_calibration_for(pos, 1 if cold else 10)
            out[m] = np.clip(sl * out[m] + ic, 0.0, None)
    return keep_raw_where_not_live(d, out)


def anchor(cal, d, s_by_pos):
    s = d['pos'].map(s_by_pos).to_numpy(dtype=float)
    ok = (~np.isnan(s) & d['anchor_mean'].notna().to_numpy() & (d['anchor_games'] >= SEASON_ANCHOR_MIN_GAMES).to_numpy()
          & (cal > 0.005) & (d['avail'] > SIDELINED_AVAILABILITY).to_numpy())
    m = d['anchor_mean'].to_numpy(dtype=float)
    out = cal.copy()
    out[ok] = np.clip(m[ok] + s[ok] * (cal[ok] - m[ok]), 0.0, None)
    return out


def fit_s(d, cal, train):
    """Least-squares s per anchored position on the training START pool (by raw), played, weeks 3-17 - the
    `eval_season_anchor.py` fit."""
    out = {}
    base = train & d['start_raw'].to_numpy() & d['played'].to_numpy() & d['week'].between(3, 17).to_numpy() \
        & d['anchor_mean'].notna().to_numpy() & (d['anchor_games'] >= SEASON_ANCHOR_MIN_GAMES).to_numpy()
    for pos in ANCHOR_POS:
        m = base & (d['pos'] == pos).to_numpy()
        x = cal[m] - d['anchor_mean'].to_numpy()[m]
        y = d['actual'].to_numpy()[m] - d['anchor_mean'].to_numpy()[m]
        out[pos] = float(np.clip((x * y).sum() / (x * x).sum(), 0.3, 1.0)) if m.sum() > MIN_ROWS else 1.0
    return out


# ------------------------------------------------------------------------------------------------ schemes
def _line(p, a, k, intercept=True):
    if len(p) < MIN_ROWS or np.var(p) < 1e-9:
        return 1.0, 0.0
    if intercept:
        b, c = np.polyfit(p, a, 1)
    else:
        b, c = float((p * a).sum() / (p * p).sum()), 0.0
    return 1.0 + k * (b - 1.0), k * c


def _fit_rows(d, train, pool):
    m = train & d['played'].to_numpy() & d['live'].to_numpy()
    if pool == 'start':
        m &= d['start_raw'].to_numpy()
    return m


def make_line_scheme(k=0.5, pool='all', buckets='shipped'):
    """The shipped structure: one points line per position (and phase bucket)."""
    def bucket_of(pos, cold):
        if buckets == 'all' or (buckets == 'shipped' and pos in BUCKETED_SHIPPED):
            return 'cold' if cold else 'rest'
        return 'all'

    def fit(d, train):
        m = _fit_rows(d, train, pool)
        params = {}
        for pos in POS:
            for cold in (True, False):
                b = bucket_of(pos, cold)
                sel = m & (d['pos'] == pos).to_numpy()
                if b != 'all':
                    sel &= (d['cold'] == cold).to_numpy()
                params[(pos, b)] = _line(d['raw'].to_numpy()[sel], d['actual'].to_numpy()[sel], k)
        return params

    def predict(d, params):
        out = d['raw'].to_numpy(dtype=float).copy()
        for pos in POS:
            for cold in (True, False):
                m = ((d['pos'] == pos) & (d['cold'] == cold)).to_numpy()
                sl, ic = params[(pos, bucket_of(pos, cold))]
                out[m] = np.clip(sl * out[m] + ic, 0.0, None)
        return keep_raw_where_not_live(d, out), None
    return fit, predict


def make_stat_scheme(k=0.5, pool='all', buckets='none', intercept=True):
    """A line per (position, scoring stat, bucket) on the stat itself; points re-scored from the calibrated line."""
    def b_of(cold):
        return ('cold' if cold else 'rest') if buckets == 'all' else 'all'

    def fit(d, train):
        m = _fit_rows(d, train, pool)
        params = {}
        for pos in POS:
            for cold in (True, False):
                sel = m & (d['pos'] == pos).to_numpy()
                if buckets == 'all':
                    sel &= (d['cold'] == cold).to_numpy()
                for s in POS_STATS[pos]:
                    params[(pos, b_of(cold), s)] = _line(d[f'p_{s}'].to_numpy()[sel], d[f'a_{s}'].to_numpy()[sel],
                                                         k, intercept)
        return params

    def predict(d, params):
        line = {s: d[f'p_{s}'].to_numpy(dtype=float).copy() for s in WEIGHTS}
        for pos in POS:
            for cold in (True, False):
                m = ((d['pos'] == pos) & (d['cold'] == cold)).to_numpy()
                for s in POS_STATS[pos]:
                    sl, ic = params[(pos, b_of(cold), s)]
                    line[s][m] = np.clip(sl * line[s][m] + ic, 0.0, None)
        live = d['live'].to_numpy()
        for s in line:
            line[s] = np.where(live, line[s], d[f'p_{s}'].to_numpy(dtype=float))
        return keep_raw_where_not_live(d, score_points(d, line)), line
    return fit, predict


def make_comp_scheme(k=0.5, pool='all', buckets='shipped', ridge=0.0):
    """actual points ~ a + sum_c b_c * (weight_c * projected stat_c), per position (and bucket); half strength
    pulls every b_c halfway to 1 and a halfway to 0, as the shipped lines do."""
    def b_of(pos, cold):
        if buckets == 'all' or (buckets == 'shipped' and pos in BUCKETED_SHIPPED):
            return 'cold' if cold else 'rest'
        return 'all'

    def comps(d, pos):
        return np.column_stack([WEIGHTS[s] * d[f'p_{s}'].to_numpy(dtype=float) for s in POS_STATS[pos]])

    def fit(d, train):
        m = _fit_rows(d, train, pool)
        params = {}
        for pos in POS:
            X_all = comps(d, pos)
            for cold in (True, False):
                b = b_of(pos, cold)
                sel = m & (d['pos'] == pos).to_numpy()
                if b != 'all':
                    sel &= (d['cold'] == cold).to_numpy()
                X = np.column_stack([np.ones(sel.sum()), X_all[sel]])
                y = d['actual'].to_numpy()[sel]
                if len(y) < MIN_ROWS:
                    params[(pos, b)] = np.r_[0.0, np.ones(X.shape[1] - 1)]
                    continue
                # a component with (almost) no projected volume in this slice keeps slope 1 and leaves the fit;
                # ridge pulls toward the identity (a=0, b_c=1), not toward zero
                prior = np.r_[0.0, np.ones(X.shape[1] - 1)]
                free = np.r_[True, (np.abs(X[:, 1:]).sum(0) > 1.0) & (X[:, 1:].std(0) > 1e-6)]
                y_adj = y - X[:, ~free] @ prior[~free]
                Xf = X[:, free]
                lam = ridge * len(y)
                pen = np.diag(np.r_[0.0, np.ones(Xf.shape[1] - 1)])
                beta_f = np.linalg.lstsq(Xf.T @ Xf + lam * pen, Xf.T @ y_adj + lam * pen @ prior[free], rcond=None)[0]
                beta = prior.copy()
                beta[free] = beta_f
                params[(pos, b)] = prior + k * (beta - prior)
        return params

    def predict(d, params):
        out = d['raw'].to_numpy(dtype=float).copy()
        line = {s: d[f'p_{s}'].to_numpy(dtype=float).copy() for s in WEIGHTS}
        for pos in POS:
            X_all = comps(d, pos)
            for cold in (True, False):
                m = ((d['pos'] == pos) & (d['cold'] == cold)).to_numpy()
                beta = params[(pos, b_of(pos, cold))]
                out[m] = np.clip(beta[0] + X_all[m] @ beta[1:], 0.0, None)
                for j, s in enumerate(POS_STATS[pos]):
                    line[s][m] = line[s][m] * beta[1 + j]
        return keep_raw_where_not_live(d, out), line
    return fit, predict


def make_opening_scheme(inner, groups):
    """The shipped lines everywhere EXCEPT the opening-week groups (e.g. [(1, 2), (3, 4)]), where `inner` is fit on
    those weeks alone (one fit per group) and applied there. The season anchor keeps its shipped s, so from the
    first week after the groups every row is exactly the shipped board."""
    fit_in, pred_in = inner

    def fit(d, train):
        return [fit_in(d, train & d['week'].between(lo, hi).to_numpy()) for lo, hi in groups]

    def predict(d, params):
        cal = shipped_line(d)
        line = {s: d[f'p_{s}'].to_numpy(dtype=float).copy() for s in WEIGHTS}
        has_line = False
        for (lo, hi), p in zip(groups, params):
            m = d['week'].between(lo, hi).to_numpy()
            c, ln = pred_in(d, p)
            cal[m] = c[m]
            if ln is not None:
                has_line = True
                for s in ln:
                    line[s][m] = ln[s][m]
        return cal, (line if has_line else None)
    return fit, predict


SCHEMES = {
    'open1_line_half':    make_opening_scheme(make_line_scheme(0.5, 'all', 'none'), [(1, 1)]),
    'open1_line_full':    make_opening_scheme(make_line_scheme(1.0, 'all', 'none'), [(1, 1)]),
    'open1_line_full_st': make_opening_scheme(make_line_scheme(1.0, 'start', 'none'), [(1, 1)]),
    'open1_stat_full':    make_opening_scheme(make_stat_scheme(1.0, 'all', 'none'), [(1, 1)]),
    'open1_stat_full_st': make_opening_scheme(make_stat_scheme(1.0, 'start', 'none'), [(1, 1)]),
    'open1_comp_full':    make_opening_scheme(make_comp_scheme(1.0, 'all', 'none'), [(1, 1)]),
    'open12_line_full':   make_opening_scheme(make_line_scheme(1.0, 'all', 'none'), [(1, 1), (2, 2)]),
    'open12_stat_full':   make_opening_scheme(make_stat_scheme(1.0, 'all', 'none'), [(1, 1), (2, 2)]),
    'open4_line_half':    make_opening_scheme(make_line_scheme(0.5, 'all', 'none'), [(1, 4)]),
    'open4_line_full':    make_opening_scheme(make_line_scheme(1.0, 'all', 'none'), [(1, 4)]),
    'open4_line_full_st': make_opening_scheme(make_line_scheme(1.0, 'start', 'none'), [(1, 4)]),
    'open2_line_full':    make_opening_scheme(make_line_scheme(1.0, 'all', 'none'), [(1, 2)]),
    'open_wk_line_full':  make_opening_scheme(make_line_scheme(1.0, 'all', 'none'), [(1, 1), (2, 2), (3, 4)]),
    'open4_stat_full':    make_opening_scheme(make_stat_scheme(1.0, 'all', 'none'), [(1, 4)]),
    'open_wk_stat_full':  make_opening_scheme(make_stat_scheme(1.0, 'all', 'none'), [(1, 1), (2, 2), (3, 4)]),
    'open4_stat_full_st': make_opening_scheme(make_stat_scheme(1.0, 'start', 'none'), [(1, 4)]),
    'open4_comp_full':    make_opening_scheme(make_comp_scheme(1.0, 'all', 'none'), [(1, 4)]),
    'line_half':          make_line_scheme(0.5, 'all', 'shipped'),
    'line_full':          make_line_scheme(1.0, 'all', 'shipped'),
    'line_half_start':    make_line_scheme(0.5, 'start', 'shipped'),
    'line_full_start':    make_line_scheme(1.0, 'start', 'shipped'),
    'stat_half':          make_stat_scheme(0.5, 'all', 'none'),
    'stat_full':          make_stat_scheme(1.0, 'all', 'none'),
    'stat_half_cold':     make_stat_scheme(0.5, 'all', 'all'),
    'stat_full_cold':     make_stat_scheme(1.0, 'all', 'all'),
    'stat_full_start':    make_stat_scheme(1.0, 'start', 'none'),
    'stat_full_prop':     make_stat_scheme(1.0, 'all', 'none', intercept=False),
    'comp_half':          make_comp_scheme(0.5, 'all', 'shipped'),
    'comp_full':          make_comp_scheme(1.0, 'all', 'shipped'),
    'comp_full_start':    make_comp_scheme(1.0, 'start', 'shipped'),
}


def loyo_predict(d, scheme, years, use_anchor=True):
    """Leave-one-year-out: (final points, calibrated stat line or None, per-year params) for every row."""
    fit, predict = SCHEMES[scheme]
    pred = np.full(len(d), np.nan)
    lines = None
    info = {}
    for y in years:
        test = (d['year'] == y).to_numpy()
        train = ~test
        params = fit(d, train)
        cal, line = predict(d, params)
        if scheme.startswith('open'):
            s = dict(SEASON_ANCHOR_S)
        else:
            s = fit_s(d, cal, train) if use_anchor else {}
        fin = anchor(cal, d, s) if use_anchor else cal
        pred[test] = fin[test]
        if line is not None:
            if lines is None:
                lines = {k: np.full(len(d), np.nan) for k in line}
            for k in line:
                lines[k][test] = line[k][test]
        info[y] = (params, s)
    return pred, lines, info


# ------------------------------------------------------------------------------------------------ scoring
def paired_scores(d, base, var, window):
    """Per scope: pooled RMSE base/var, bias, pairwise, and per-week SSE pairs for the bootstrap."""
    w = window.to_numpy() if hasattr(window, 'to_numpy') else window
    tb, tv = top_n_mask(d, base), top_n_mask(d, var)
    played = d['played'].to_numpy()
    act = d['actual'].to_numpy()
    out = {}
    scopes = [(f'START-{p}', (d['pos'] == p).to_numpy() & (tb | tv)) for p in POS]
    scopes.append(('START-ALL', (tb | tv)))
    scopes.append(('ALL', np.ones(len(d), dtype=bool)))
    for name, m in scopes:
        m = m & played & w
        if m.sum() < 10:
            continue
        eb, ev = base[m] - act[m], var[m] - act[m]
        wk = d['wk_key'].to_numpy()[m]
        g = pd.DataFrame({'wk': wk, 'sb': eb ** 2, 'sv': ev ** 2, 'n': 1}).groupby('wk').sum()
        pw_b, pw_v = [], []
        if name.startswith('START-') and name != 'START-ALL':
            for _, idx in pd.Series(np.arange(m.sum())).groupby(wk).groups.items():
                ii = np.asarray(idx)
                if len(ii) > 4:
                    pw_b.append(pairwise_acc(base[m][ii], act[m][ii]))
                    pw_v.append(pairwise_acc(var[m][ii], act[m][ii]))
        out[name] = {'n': int(m.sum()), 'rmse_b': float(np.sqrt((eb ** 2).mean())), 'rmse_v': float(np.sqrt((ev ** 2).mean())),
                     'bias_b': float(eb.mean()), 'bias_v': float(ev.mean()),
                     'pw_b': float(np.nanmean(pw_b)) if pw_b else np.nan, 'pw_v': float(np.nanmean(pw_v)) if pw_v else np.nan,
                     'weeks': g, 'wins': int((g['sv'] < g['sb']).sum()), 'losses': int((g['sv'] > g['sb']).sum())}
    return out


def boot_ci(g, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    sb, sv, n = g['sb'].to_numpy(), g['sv'].to_numpy(), g['n'].to_numpy()
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    d = np.sqrt(sv[idx].sum(1) / n[idx].sum(1)) - np.sqrt(sb[idx].sum(1) / n[idx].sum(1))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def report(d, preds, window, label):
    print('\n' + '=' * 118)
    print(f"{label}: held-out (leave-one-year-out) dRMSE vs SHIPPED   [CI over weeks]   bias shipped -> scheme")
    print('=' * 118)
    rows = []
    for name, pred in preds.items():
        sc = paired_scores(d, d['final'].to_numpy(dtype=float), pred, window)
        r = {'scheme': name}
        for scope in ('START-ALL', 'START-QB', 'START-RB', 'START-WR', 'START-TE', 'ALL'):
            if scope not in sc:
                continue
            x = sc[scope]
            dr = x['rmse_v'] - x['rmse_b']
            if scope in ('START-ALL', 'ALL'):
                lo, hi = boot_ci(x['weeks'])
                r[scope] = f"{dr:+.3f} [{lo:+.3f},{hi:+.3f}]"
            else:
                r[scope] = f"{dr:+.3f}"
        sa = sc['START-ALL']
        r['bias S-ALL'] = f"{sa['bias_b']:+.2f}->{sa['bias_v']:+.2f}"
        r['wk W-L'] = f"{sa['wins']}-{sa['losses']}"
        rows.append(r)
    base_sc = paired_scores(d, d['final'].to_numpy(dtype=float), d['final'].to_numpy(dtype=float), window)
    print('shipped RMSE: ' + '  '.join(f"{k} {v['rmse_b']:.3f} (bias {v['bias_b']:+.2f})" for k, v in base_sc.items()))
    pd.set_option('display.width', 250)
    print(pd.DataFrame(rows).to_string(index=False))


def per_year(d, preds, window):
    print('\nSTART-ALL dRMSE by held-out year (negative = scheme better):')
    years = sorted(d['year'].unique())
    rows = []
    for name, pred in preds.items():
        r = {'scheme': name}
        for y in years:
            w = window & (d['year'] == y)
            sc = paired_scores(d, d['final'].to_numpy(dtype=float), pred, w)
            if 'START-ALL' in sc:
                r[y] = round(sc['START-ALL']['rmse_v'] - sc['START-ALL']['rmse_b'], 3)
        r['better in'] = f"{sum(1 for y in years if r.get(y, 0) < 0)}/{len(years)}"
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))


def stat_report(d, lines_by_scheme, window):
    """Per (pos, stat): RMSE / bias of the shipped (raw) stat line vs the scheme's calibrated stat line, START pool by
    shipped final, played rows."""
    m0 = top_n_mask(d, d['final'].to_numpy(dtype=float)) & d['played'].to_numpy() & window.to_numpy()
    for name, lines in lines_by_scheme.items():
        if lines is None:
            continue
        rows = []
        for pos in POS:
            m = m0 & (d['pos'] == pos).to_numpy()
            for s in POS_STATS[pos]:
                a = d[f'a_{s}'].to_numpy()[m]
                p0 = d[f'p_{s}'].to_numpy()[m]
                p1 = lines[s][m]
                rows.append({'pos': pos, 'stat': s, 'n': int(m.sum()),
                             'RMSE raw': round(float(np.sqrt(((p0 - a) ** 2).mean())), 3),
                             'RMSE cal': round(float(np.sqrt(((p1 - a) ** 2).mean())), 3),
                             'bias raw': round(float((p0 - a).mean()), 3), 'bias cal': round(float((p1 - a).mean()), 3),
                             'actual/proj raw': round(float(a.sum() / max(p0.sum(), 1e-9)), 3)})
        print(f"\nstat line, {name} (START pool, played, held-out):")
        print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-path', default=os.path.join('.sweeps', 'calibration_lab_v7.csv'))
    ap.add_argument('--week1-path', default='', help='a week-1 dump that replaces the main dump\'s week 1')
    ap.add_argument('--schemes', default=','.join(SCHEMES))
    ap.add_argument('--stat-report', default='stat_full,comp_full')
    a = ap.parse_args()
    d = load_dump(a.dump_path, a.week1_path)
    years = sorted(d['year'].unique())
    print(f"loaded {len(d):,} rows ({int(d['played'].sum()):,} played), years {years}")
    if 'final' in d.columns:
        chk = shipped_line(d)
        chk = anchor(chk, d, {p: SEASON_ANCHOR_S[p] for p in ANCHOR_POS})
        diff = np.abs(chk - d['final'].to_numpy(dtype=float))
        print(f"shipped pipeline reproduced from raw: max |diff| {diff.max():.3f}, rows > 0.02: {int((diff > 0.02).sum())}")
    preds, lines_by_scheme, infos = {}, {}, {}
    for name in [s for s in a.schemes.split(',') if s]:
        preds[name], lines_by_scheme[name], infos[name] = loyo_predict(d, name, years)
    harness_window = d['year'].between(2022, 2025) & d['week'].between(3, 17)
    report(d, preds, harness_window, 'HARNESS WINDOW 2022-2025 wk3-17')
    per_year(d, preds, d['week'].between(3, 17))
    report(d, preds, d['week'].between(1, 4), 'COLD WEEKS 1-4, all years')
    per_year(d, preds, d['week'].between(1, 4))
    print('\nSTART-ALL dRMSE by week (all years pooled):')
    rows = []
    for name, pred in preds.items():
        r = {'scheme': name}
        for wk in (1, 2, 3, 4, 5):
            sc = paired_scores(d, d['final'].to_numpy(dtype=float), pred, d['week'] == wk)
            if 'START-ALL' in sc:
                r[f'wk{wk}'] = round(sc['START-ALL']['rmse_v'] - sc['START-ALL']['rmse_b'], 3)
                r[f'bias wk{wk}'] = f"{sc['START-ALL']['bias_b']:+.2f}->{sc['START-ALL']['bias_v']:+.2f}"
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
    stat_report(d, {k: v for k, v in lines_by_scheme.items() if k in a.stat_report.split(',')}, harness_window)
    for name in [s for s in a.stat_report.split(',') if s in infos]:
        params, s = infos[name][max(years)]
        print(f"\n{name} params (fit without {max(years)}): anchor s {s}")
        for group in (params if isinstance(params, list) else [params]):
            for k, v in group.items():
                print(f"   {k}: {np.round(v, 3).tolist() if hasattr(v, '__len__') else v}")


if __name__ == '__main__':
    main()
