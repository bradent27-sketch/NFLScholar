"""
Stage-attribution audit of the projection mechanism (2026-11-01; user: "audit the projection mechanism ... where is it not
properly tuned, where is it most inaccurate, where does it have redundant mechanisms, where can it be simplified").

Reads a capture of today's model (scripts/capture_stat_blend_traces.py: one row per player-stat with every stage's
multiplier / delta, and the board file with the points chain raw -> line -> season anchor -> Questionable cut -> luck) and asks
of every stage, on played games:

    beta = how much of what the stage moved came true.  actual = (final - c) + beta * c + e,  c = the stage's contribution
           to the final number (multiplicative stage m: c = final - final / m; additive stage: c = its delta).
           beta ~ 1: sized right.  beta ~ 0: the stage adds nothing (redundant with what is already there).
           beta < 0: it moves the number the wrong way.  beta > 1: too weak.
    held-out MSE change, by season (beta fit on the other seasons), of (a) removing the stage (c -> 0) and (b) rescaling it
           (c -> beta c), against the shipped number.

A multiplicative stage's c treats the downstream stages as proportional (true for the multipliers and the capacity factor;
approximate where an additive step follows). Played = a box-score line (silent games have no stat line, so they are out of
the stat part; the points part uses the board's `played`).

    python scripts/audit_stage_attribution.py --capture .sweeps/stage_capture.parquet [--capture .sweeps/stage_capture_2026.parquet]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.transforms import score_projected_stats  # noqa: E402
from scripts.eval_weekly_model import STARTABLE_N  # noqa: E402

MULT_STAGES = ('matchup_multiplier', 'pace_multiplier', 'script_multiplier', 'environment_multiplier',
               'participation_multiplier', 'weather_stat_multiplier')
ADD_STAGES = ('vacancy_delta', 'carry_budget_delta', 'pass_capacity_delta', 'top_end_delta', 'own_history_delta')
STATS = {
    'QB': ('passing_attempts', 'passing_yards', 'passing_tds', 'passing_interceptions', 'rushing_attempts', 'rushing_yards',
           'rushing_tds'),
    'RB': ('rushing_attempts', 'rushing_yards', 'rushing_tds', 'targets', 'receptions', 'receiving_yards'),
    'WR': ('targets', 'receptions', 'receiving_yards', 'receiving_tds'),
    'TE': ('targets', 'receptions', 'receiving_yards', 'receiving_tds'),
}
POINT_STEPS = (('line', 'raw', 'pts_line'), ('season_anchor', None, 'pts_season_anchor'),
               ('questionable', None, 'pts_questionable_workload'), ('luck', None, 'pts_expected_points_luck'))


def start_pool(b):
    """The harness START pool: each week's top STARTABLE_N projections per position."""
    rank = b.groupby(['year', 'week', 'pos'])['final'].rank(ascending=False, method='first')
    return b[rank <= b['pos'].map(STARTABLE_N).fillna(30)].copy()


def load(paths):
    traces, boards = [], []
    for p in paths:
        traces.append(pd.read_parquet(p))
        boards.append(pd.read_csv(p + '.board.csv', low_memory=False))
    return pd.concat(traces, ignore_index=True), pd.concat(boards, ignore_index=True)


def beta_fit(y, base, c):
    """Least-squares beta in y - base = beta * c (no intercept)."""
    den = float(np.dot(c, c))
    return float(np.dot(c, y - base) / den) if den > 0 else np.nan


def attribution(frame, final_col, actual_col, contrib, label, by='year'):
    """One row: rows touched, size of the stage, beta (pooled and per season), held-out MSE change of removing / rescaling."""
    d = frame[[final_col, actual_col, by]].copy()
    d['c'] = contrib
    d = d.dropna()
    touched = d['c'].abs() > 1e-6
    if touched.sum() < 30:
        return None
    y, f, c = d[actual_col].to_numpy(float), d[final_col].to_numpy(float), d['c'].to_numpy(float)
    base = f - c
    mse = np.mean((y - f) ** 2)
    beta = beta_fit(y, base, c)
    seasons = sorted(d[by].unique())
    per, held_rm, held_sc = [], [], []
    for s in seasons:
        te = (d[by] == s).to_numpy()
        tr = ~te
        per.append(beta_fit(y[te], base[te], c[te]))
        b_tr = beta_fit(y[tr], base[tr], c[tr])
        held_sc.append(np.sum((y[te] - (base[te] + b_tr * c[te])) ** 2) - np.sum((y[te] - f[te]) ** 2))
        held_rm.append(np.sum((y[te] - base[te]) ** 2) - np.sum((y[te] - f[te]) ** 2))
    n = len(d)
    return {'stage': label, 'n': n, 'touched': float(touched.mean()), 'sd_c': float(np.std(c[touched.to_numpy()])),
            'mean_c': float(np.mean(c)), 'beta': beta,
            'beta_lo_hi': f'{np.nanmin(per):+.2f}..{np.nanmax(per):+.2f}',
            'seasons_beta_gt_half': int(np.sum(np.array(per) > 0.5)), 'n_seasons': len(seasons),
            'rm_pct': 100 * np.sum(held_rm) / (mse * n), 'scale_pct': 100 * np.sum(held_sc) / (mse * n),
            'rm_seasons_worse': int(np.sum(np.array(held_rm) > 0)), 'sc_seasons_better': int(np.sum(np.array(held_sc) < 0))}


def stat_stages(tr, weeks=(3, 17)):
    t = tr[(tr['played'].astype(str).str.lower() == 'true') & tr['actual'].notna()
           & tr['week'].between(*weeks)].copy()
    t['final'] = pd.to_numeric(t['final_projection'], errors='coerce')
    out = []
    for pos, stats in STATS.items():
        for stat in stats:
            g = t[(t['pos'] == pos) & (t['stat'] == stat)]
            if pos == 'QB':
                g = g[g['final'] > 0]
            if len(g) < 200:
                continue
            for m in MULT_STAGES:
                mult = pd.to_numeric(g[m], errors='coerce')
                c = (g['final'] - g['final'] / mult.where(mult > 0)).fillna(0.0)
                r = attribution(g, 'final', 'actual', c, m.replace('_multiplier', '').replace('_stat', ''))
                if r:
                    out.append({'pos': pos, 'stat': stat, **r})
            for a in ADD_STAGES:
                if a not in g.columns:
                    continue
                c = pd.to_numeric(g[a], errors='coerce').fillna(0.0)
                r = attribution(g, 'final', 'actual', c, a.replace('_delta', ''))
                if r:
                    out.append({'pos': pos, 'stat': stat, **r})
    return pd.DataFrame(out)


def points_stages(bd, weeks=(3, 17), start=False):
    b = bd[bd['played'].astype(bool) & bd['week'].between(*weeks)].copy()
    if start:
        b = start_pool(b)
    out = []
    prev = b['raw'].astype(float)
    chain = [('line', b['pts_line'])]
    for name, col in (('season_anchor', 'pts_season_anchor'), ('questionable', 'pts_questionable_workload'),
                      ('luck', 'pts_expected_points_luck')):
        chain.append((name, b[col]))
    contribs = {}
    for name, after in chain:
        after = pd.to_numeric(after, errors='coerce')
        ran = after.notna()
        contribs[name] = (after - prev).where(ran, 0.0).fillna(0.0)
        prev = after.where(ran, prev)
    resid_check = float((prev - b['final']).abs().max())
    for pos in ('QB', 'RB', 'WR', 'TE', None):
        g = b if pos is None else b[b['pos'] == pos]
        for name, c in contribs.items():
            r = attribution(g, 'final', 'actual', c.loc[g.index], name)
            if r:
                out.append({'pos': pos or 'ALL', **r})
    return pd.DataFrame(out), resid_check


def points_from_stats(bd, weeks=(3, 17)):
    """Simplification check: the displayed (recal'd) stat line scored straight to points, with one forward-chained linear line
    per position, against the shipped points chain."""
    b = bd[bd['played'].astype(bool) & bd['week'].between(*weeks)].copy()
    cols = {s: f'p_{s}' for s in ('rushing_yards', 'rushing_tds', 'receiving_yards', 'receiving_tds', 'receptions',
                                  'passing_yards', 'passing_tds', 'passing_interceptions')}
    stat = pd.DataFrame({s: pd.to_numeric(b[c], errors='coerce').fillna(0.0) for s, c in cols.items() if c in b})
    b['from_stats'] = [score_projected_stats(r, 'Full PPR') for r in stat.to_dict('records')]
    rows = []
    for pos in ('QB', 'RB', 'WR', 'TE'):
        g = b[b['pos'] == pos]
        for yr in sorted(g['year'].unique()):
            tr, te = g[g['year'] < yr], g[g['year'] == yr]
            if len(tr) < 200:
                continue
            X = np.c_[np.ones(len(tr)), tr['from_stats']]
            coef = np.linalg.lstsq(X, tr['actual'].to_numpy(float), rcond=None)[0]
            pred = coef[0] + coef[1] * te['from_stats']
            for name, v in (('shipped', te['final']), ('raw', te['raw']), ('from_stats', te['from_stats']),
                            ('from_stats_line', pred)):
                rows.append({'pos': pos, 'year': yr, 'variant': name, 'n': len(te),
                             'rmse': float(np.sqrt(np.mean((te['actual'] - v) ** 2)))})
    return pd.DataFrame(rows)


def segment_table(d, seg, by='year'):
    """Bias, RMSE, slope (actual on projection) and seasons agreeing on the bias sign, per segment value."""
    rows = []
    sse_all = float(np.sum(d['resid'] ** 2))
    for val, g in d.groupby(seg, observed=True):
        if len(g) < 60:
            continue
        per = g.groupby(by)['resid'].mean()
        sign = np.sign(g['resid'].mean())
        x = g['final'].to_numpy(float)
        slope = float(np.polyfit(x, g['actual'].to_numpy(float), 1)[0]) if np.std(x) > 0 else np.nan
        rows.append({'segment': seg, 'value': str(val), 'n': len(g), 'row_share': len(g) / len(d),
                     'err_share': float(np.sum(g['resid'] ** 2)) / sse_all, 'proj': g['final'].mean(),
                     'bias': g['resid'].mean(), 'rmse': float(np.sqrt(np.mean(g['resid'] ** 2))), 'slope': slope,
                     'seasons_same_sign': f'{int(np.sum(np.sign(per) == sign))}/{len(per)}'})
    return rows


def error_map(bd, panel_path, tr):
    """Where today's model misses most: START pool, played, weeks 3-17, residual = actual - final points, by segment.
    Features come from the deep-audit panel (same player-weeks, joined on year/week/player/pos), stage firing from the capture."""
    b = bd[bd['played'].astype(bool) & bd['week'].between(3, 17)].copy()
    b = start_pool(b)
    b['resid'] = b['actual'] - b['final']
    if panel_path and os.path.exists(panel_path):
        keep = ['year', 'week', 'player', 'pos', 'h_games', 'rookie_like', 'spread', 'total', 'imp_team', 'home',
                'tm_out_pts_total', '_profile_source_kind', 'Role Confidence', '_profile_target_earner_rank',
                'h_ps_games', 'dev_s2d_points', 'h_sd_fantasy_points_ppr']
        p = pd.read_parquet(panel_path, columns=keep).drop_duplicates(['year', 'week', 'player', 'pos'])
        b = b.merge(p, on=['year', 'week', 'player', 'pos'], how='left')
    # stage firing per player-week from the traces (any stat)
    t = tr[tr['week'].between(3, 17)]
    fire = t.groupby(['year', 'week', 'player', 'pos']).agg(
        vac=('vacancy_delta', lambda s: float(np.nansum(np.abs(pd.to_numeric(s, errors='coerce'))))),
        cap=('pass_capacity_delta', lambda s: float(np.nansum(pd.to_numeric(s, errors='coerce')))),
        own=('own_history_delta', lambda s: float(np.nansum(np.abs(pd.to_numeric(s, errors='coerce')))))).reset_index()
    b = b.merge(fire, on=['year', 'week', 'player', 'pos'], how='left')
    b['tier'] = b.groupby('pos')['final'].transform(lambda s: pd.qcut(s, 5, labels=['q1 low', 'q2', 'q3', 'q4', 'q5 high']))
    b['window'] = pd.cut(b['week'], [2, 5, 9, 13, 17], labels=['wk3-5', 'wk6-9', 'wk10-13', 'wk14-17'])
    segs = {'tier': 'tier', 'window': 'window'}
    if 'h_games' in b:
        b['evidence'] = pd.cut(b['h_games'].fillna(0), [-1, 0, 2, 5, 20], labels=['0 games', '1-2', '3-5', '6+'])
        b['prior'] = np.where(b['h_ps_games'].fillna(0) > 0, 'own prior season', 'no prior season')
        b['rookie'] = np.where(b['rookie_like'].fillna(0) > 0, 'rookie-like', 'veteran')
        b['spread_b'] = pd.cut(b['spread'], [-40, -7.5, -3, 3, 7.5, 40],
                               labels=['dog 7.5+', 'dog 3-7.5', 'pick', 'fav 3-7.5', 'fav 7.5+'])
        b['teammates_out'] = pd.cut(b['tm_out_pts_total'].fillna(0), [-1, 0.5, 8, 16, 200],
                                    labels=['none', '<8 pts', '8-16', '16+'])
        b['earner_rank'] = b['_profile_target_earner_rank'].clip(upper=4).fillna(-1).astype(int).astype(str)
        b['form'] = pd.qcut(b['dev_s2d_points'].rank(method='first'), 5,
                            labels=['proj far below own avg', 'below', 'near', 'above', 'proj far above own avg'])
        segs.update({k: k for k in ('evidence', 'prior', 'rookie', 'spread_b', 'teammates_out', 'earner_rank', 'form')})
    b['vacancy_recipient'] = np.where(b['vac'].fillna(0) > 0.05, 'vacancy moved him', 'no vacancy')
    b['capacity_cut'] = pd.cut(b['cap'].fillna(0), [-100, -2, -0.5, 0.5, 100],
                               labels=['cut 2+', 'cut 0.5-2', 'about none', 'raised'])
    b['recal_moved'] = pd.cut(b['own'].fillna(0), [-1, 0.5, 3, 1000], labels=['<0.5', '0.5-3', '3+'])
    segs.update({k: k for k in ('vacancy_recipient', 'capacity_cut', 'recal_moved')})
    rows = []
    for pos in ('QB', 'RB', 'WR', 'TE'):
        g = b[b['pos'] == pos]
        for seg in segs:
            if seg in g:
                rows += [{'pos': pos, **r} for r in segment_table(g, seg)]
    return pd.DataFrame(rows), b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', action='append', required=True)
    ap.add_argument('--out', default='.sweeps/stage_audit')
    ap.add_argument('--panel', default='.sweeps/deep_audit/panel.parquet')
    a = ap.parse_args()
    tr, bd = load(a.capture)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 400)
    fmt = {'touched': '{:.0%}'.format, 'sd_c': '{:.3f}'.format, 'mean_c': '{:+.3f}'.format, 'beta': '{:+.2f}'.format,
           'rm_pct': '{:+.2f}'.format, 'scale_pct': '{:+.2f}'.format}
    s = stat_stages(tr)
    s.to_csv(os.path.join(a.out, 'stat_stages.csv'), index=False)
    print('STAT STAGES (played box-score games, weeks 3-17). rm/scale = held-out MSE change in % of the stat\'s MSE')
    print(s.to_string(index=False, formatters=fmt))
    for start in (False, True):
        p, chk = points_stages(bd, start=start)
        p.to_csv(os.path.join(a.out, f'points_stages{"_start" if start else ""}.csv'), index=False)
        print(f'\nPOINTS CHAIN ({"START pool" if start else "every played row"}; chain check max |diff| {chk:.3f})')
        print(p.to_string(index=False, formatters=fmt))
    f = points_from_stats(bd)
    if not f.empty:
        f.to_csv(os.path.join(a.out, 'points_from_stats.csv'), index=False)
        pooled = f.groupby(['pos', 'variant']).apply(
            lambda g: np.sqrt(np.sum(g['rmse'] ** 2 * g['n']) / g['n'].sum())).unstack('variant')
        print('\nPOINTS FROM THE DISPLAYED STAT LINE (forward-chained line), pooled RMSE over the held-out seasons')
        print(pooled.round(3).to_string())
    em, rows = error_map(bd, a.panel, tr)
    em.to_csv(os.path.join(a.out, 'error_map.csv'), index=False)
    rows.to_parquet(os.path.join(a.out, 'error_map_rows.parquet'), index=False)
    print('\nERROR MAP (START pool, played, weeks 3-17; resid = actual - final). err_share vs row_share = where the error sits')
    print(em.to_string(index=False, formatters={'row_share': '{:.0%}'.format, 'err_share': '{:.0%}'.format,
                                                'proj': '{:.2f}'.format, 'bias': '{:+.2f}'.format,
                                                'rmse': '{:.2f}'.format, 'slope': '{:.2f}'.format}))


if __name__ == '__main__':
    main()
