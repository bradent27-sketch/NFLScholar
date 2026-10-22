"""
Residual scan (model audit step 2, 2026-10-21): is the current model's miss predictable from what was known before kickoff?

Reads the feature table from scripts/audit_features.py. Residual = actual minus the board's number (points, or one stat).
Everything is held out by season: fit on four seasons, score the fifth, five times.

  1. Headroom: a gradient-boosted model of the residual on all features. Its held-out gain on the START pool (the board plus
     the predicted residual, at full and half strength) is roughly how much error the features we already have could still
     remove. Near zero means tuning existing inputs is exhausted; a clear gain says where to look.
  2. Which features carry it: permutation importance on the held-out seasons (how much of the gain is lost when a feature is
     shuffled), with the number of seasons where it matters.
  3. One feature at a time (transparent): quintile bins of the feature fit on four seasons (shrunk bin means), applied to the
     fifth; held-out START gain and seasons better; the five-season bin table for the top ones.
  4. The same headroom for the volume stats (targets, carries, pass attempts) in stat units, and for team totals.

Diagnostic only: nothing here ships. A lead becomes a named, mechanistic model flag and goes through the harness.

    python scripts/audit_residual_scan.py [--features .sweeps/audit_features.parquet] [--out .sweeps/audit_residual_scan.txt]
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
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

STARTABLE_N = {'QB': 24, 'RB': 40, 'WR': 55, 'TE': 20}
POSITIONS = ('QB', 'RB', 'WR', 'TE')
GBM = dict(learning_rate=0.04, max_iter=300, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=1.0,
           early_stopping=False, random_state=7)
BIN_SHRINK_K = 300.0
TEAM_FEATURES = ('f_spread', 'f_total', 'f_implied', 'f_home', 'f_rest_diff', 'f_dome', 'f_wind', 'f_temp', 'f_qb_change',
                 'f_team_qb_pass_yds', 'f_team_qb_games', 'f_vac_pts_team', 'f_week', 'f_hc_changed', 'f_oc_changed',
                 'f_opp_dc_changed', 'f_div_game')


def load(path):
    x = pd.read_parquet(path)
    for c in [c for c in x.columns if c.startswith('f_p_')]:
        x[c] = x[c].fillna(0.0)
    x['pos_code'] = x['pos'].map({p: i for i, p in enumerate(POSITIONS)}).astype(float)
    r = x.groupby(['wk_key', 'pos'])['f_final'].rank(method='first', ascending=False)
    x['wide'] = (r <= 1.5 * x['pos'].map(STARTABLE_N)).to_numpy() & x['played'].to_numpy() & x['actual'].notna().to_numpy()
    x['pool'] = x['pool'].astype(bool) & x['played'].astype(bool)
    return x


def feature_cols(x):
    return [c for c in x.columns if c.startswith('f_')] + ['pos_code']


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else float('nan')


def weekly_ci(wk, e0, e1, n_boot=1000, seed=3):
    g = pd.DataFrame({'k': wk, 'a': e0 ** 2, 'b': e1 ** 2}).groupby('k').agg(a=('a', 'sum'), b=('b', 'sum'), n=('a', 'size'))
    a, b, n = g['a'].to_numpy(), g['b'].to_numpy(), g['n'].to_numpy()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        i = rng.choice(len(a), len(a))
        out.append(np.sqrt(b[i].sum() / n[i].sum()) - np.sqrt(a[i].sum() / n[i].sum()))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def heldout_predictions(x, cols, target, train_mask, score_mask):
    """Per season: fit on the other seasons' train rows, predict this season's score rows. Returns pred (NaN elsewhere)."""
    pred = np.full(len(x), np.nan)
    models = {}
    years = x['year'].to_numpy()
    for y in sorted(np.unique(years)):
        tr = train_mask & (years != y) & np.isfinite(target)
        te = score_mask & (years == y)
        if tr.sum() < 500 or te.sum() == 0:
            continue
        cat = [cols.index('pos_code')] if 'pos_code' in cols else None
        m = HistGradientBoostingRegressor(categorical_features=cat, **GBM)
        m.fit(x.loc[tr, cols].to_numpy(float), target[tr])
        pred[te] = m.predict(x.loc[te, cols].to_numpy(float))
        models[y] = m
    return pred, models


def headroom_table(x, base, actual, pred, mask, label):
    rows = []
    for pos in POSITIONS + ('ALL',):
        m = mask & np.isfinite(pred)
        if pos != 'ALL':
            m &= (x['pos'] == pos).to_numpy()
        if m.sum() == 0:
            continue
        e0 = base[m] - actual[m]
        r = {'scope': pos, 'n': int(m.sum()), 'base RMSE': rmse(e0)}
        for s in (1.0, 0.5):
            e1 = base[m] + s * pred[m] - actual[m]
            r[f'd x{s:g}'] = rmse(e1) - rmse(e0)
        lo, hi = weekly_ci(x['wk_key'].to_numpy()[m], e0, base[m] + 0.5 * pred[m] - actual[m])
        r['CI x0.5'] = f'[{lo:+.3f}, {hi:+.3f}]'
        yrs = []
        for y in sorted(x['year'].unique()):
            mm = m & (x['year'] == y).to_numpy()
            if mm.sum():
                yrs.append(rmse(base[mm] + 0.5 * pred[mm] - actual[mm]) < rmse(base[mm] - actual[mm]))
        r['seasons better x0.5'] = f'{sum(yrs)}/{len(yrs)}'
        r['share of MSE x1'] = 1.0 - (r['base RMSE'] + r['d x1']) ** 2 / r['base RMSE'] ** 2
        rows.append(r)
    return f'{label}\n' + pd.DataFrame(rows).set_index('scope').round(4).to_string()


def permutation_importance(x, cols, models, base, actual, mask, n_rep=2, seed=11):
    """Loss of held-out gain (MSE units) when one feature is shuffled within the held-out season, per season."""
    rng = np.random.default_rng(seed)
    out = {c: [] for c in cols}
    years = x['year'].to_numpy()
    for y, m in models.items():
        te = mask & (years == y)
        X = x.loc[te, cols].to_numpy(float)
        b, a = base[te], actual[te]
        ref = np.mean((b + m.predict(X) - a) ** 2)
        for j, c in enumerate(cols):
            losses = []
            for _ in range(n_rep):
                Xp = X.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                losses.append(np.mean((b + m.predict(Xp) - a) ** 2) - ref)
            out[c].append(float(np.mean(losses)))
    t = pd.DataFrame({'feature': cols, 'mean loss': [np.mean(out[c]) for c in cols],
                      'seasons >0': [sum(v > 0 for v in out[c]) for c in cols]})
    return t.sort_values('mean loss', ascending=False)


def bin_scan(x, cols, resid, base, actual, train_mask, score_mask):
    rows = []
    years = x['year'].to_numpy()
    for pos in POSITIONS:
        pm = (x['pos'] == pos).to_numpy()
        for c in cols:
            if c == 'pos_code':
                continue
            v = x[c].to_numpy(float)
            corr = np.zeros(len(x))
            ok_any = False
            per_year = []
            for y in sorted(np.unique(years)):
                tr = train_mask & pm & (years != y) & np.isfinite(v) & np.isfinite(resid)
                te = score_mask & pm & (years == y)
                if tr.sum() < 300 or te.sum() == 0:
                    continue
                edges = np.unique(np.nanquantile(v[tr], [0.2, 0.4, 0.6, 0.8]))
                if len(edges) == 0:
                    continue
                btr = np.digitize(v[tr], edges)
                means = {}
                for b in np.unique(btr):
                    rr = resid[tr][btr == b]
                    means[b] = rr.mean() * len(rr) / (len(rr) + BIN_SHRINK_K)
                bte = np.digitize(np.nan_to_num(v[te], nan=np.nan), edges)
                adj = np.array([means.get(b, 0.0) for b in bte])
                adj[~np.isfinite(v[te])] = 0.0
                adj -= np.mean([means.get(b, 0.0) for b in btr])     # bins only; the level is not the feature's
                corr[te] = adj
                ok_any = True
                per_year.append(rmse(base[te] + adj - actual[te]) < rmse(base[te] - actual[te]))
            if not ok_any:
                continue
            m = score_mask & pm
            d = rmse(base[m] + corr[m] - actual[m]) - rmse(base[m] - actual[m])
            rows.append({'pos': pos, 'feature': c, 'dRMSE': d, 'seasons better': f'{sum(per_year)}/{len(per_year)}',
                         '_better': sum(per_year)})
    return pd.DataFrame(rows).sort_values('dRMSE')


def bin_table(x, c, pos, resid, mask):
    m = mask & (x['pos'] == pos).to_numpy() & np.isfinite(x[c].to_numpy(float))
    v = x.loc[m, c].to_numpy(float)
    edges = np.unique(np.nanquantile(v, [0.2, 0.4, 0.6, 0.8]))
    b = np.digitize(v, edges)
    t = pd.DataFrame({'bin': b, 'v': v, 'r': resid[m], 'y': x.loc[m, 'year'].to_numpy()})
    g = t.groupby('bin').agg(lo=('v', 'min'), hi=('v', 'max'), n=('r', 'size'), resid=('r', 'mean'), se=('r', 'sem'))
    g['seasons same sign'] = t.groupby('bin').apply(
        lambda s: int(max((s.groupby('y')['r'].mean() > 0).sum(), (s.groupby('y')['r'].mean() < 0).sum())), include_groups=False)
    return g.round(3).to_string()


def stat_headroom(x, cols, lines):
    specs = (('targets', ('WR', 'TE', 'RB')), ('rushing_attempts', ('RB',)), ('passing_attempts', ('QB',)),
             ('receiving_yards', ('WR', 'TE', 'RB')), ('rushing_yards', ('RB',)), ('passing_yards', ('QB',)))
    for stat, positions in specs:
        pm = x['pos'].isin(positions).to_numpy()
        base = x[f'f_p_{stat}'].to_numpy(float)
        act = x[f'a_{stat}'].to_numpy(float)
        resid = act - base
        pred, _ = heldout_predictions(x, cols, resid, x['wide'].to_numpy() & pm, x['pool'].to_numpy() & pm)
        lines.append(headroom_table(x, base, act, pred, x['pool'].to_numpy() & pm,
                                    f'-- {stat} ({"/".join(positions)}), stat units'))
        lines.append('')


def team_headroom(x, lines):
    x = x.assign(_p_pts=x['f_final'].where(x['played']), _a_pts=x['actual'].where(x['played']))
    agg = {'p_targets': ('f_p_targets', 'sum'), 'a_targets': ('a_targets', 'sum'),
           'p_car': ('f_p_rushing_attempts', 'sum'), 'a_car': ('a_rushing_attempts', 'sum'),
           'p_pts': ('_p_pts', 'sum'), 'a_pts': ('_a_pts', 'sum')}
    t = x.groupby(['year', 'week', 'team']).agg(**agg, **{c: (c, 'first') for c in TEAM_FEATURES if c in x}).reset_index()
    t['wk_key'] = t['year'] * 100 + t['week']
    t['pos_code'] = 0.0
    t['pos'] = 'ALL'
    t['played'] = True
    cols = [c for c in TEAM_FEATURES if c in t] + ['p_targets', 'p_car']
    for p_col, a_col, label in (('p_targets', 'a_targets', 'team targets (board players)'),
                                ('p_car', 'a_car', 'team carries (board players)'),
                                ('p_pts', 'a_pts', 'team points (board players who played)')):
        base, act = t[p_col].to_numpy(float), t[a_col].to_numpy(float)
        resid = act - base
        allm = np.ones(len(t), bool)
        pred, _ = heldout_predictions(t, cols, resid, allm, allm)
        m = np.isfinite(pred)
        e0, e1, eh = base[m] - act[m], base[m] + pred[m] - act[m], base[m] + 0.5 * pred[m] - act[m]
        yrs = [rmse(eh[t['year'].to_numpy()[m] == y]) < rmse(e0[t['year'].to_numpy()[m] == y]) for y in sorted(t['year'].unique())]
        lines.append(f'-- {label}: team-weeks {m.sum():,}, base RMSE {rmse(e0):.3f}, held-out x1 {rmse(e1) - rmse(e0):+.3f}, '
                     f'x0.5 {rmse(eh) - rmse(e0):+.3f}, seasons better x0.5 {sum(yrs)}/{len(yrs)}, mean miss {np.mean(-e0):+.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='.sweeps/audit_features.parquet')
    ap.add_argument('--out', default='.sweeps/audit_residual_scan.txt')
    ap.add_argument('--top', type=int, default=30)
    a = ap.parse_args()
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 30)
    x = load(a.features)
    cols = feature_cols(x)
    base, act = x['f_final'].to_numpy(float), x['actual'].to_numpy(float)
    resid = act - base
    wide, pool = x['wide'].to_numpy(), x['pool'].to_numpy()
    lines = [f'{len(x):,} rows, {len(cols)} features, START pool {int(pool.sum()):,}, training rows (top 1.5 x N, played) '
             f'{int(wide.sum()):,}; held out by season 2021-2025', '']

    pred, models = heldout_predictions(x, cols, resid, wide, pool)
    lines.append(headroom_table(x, base, act, pred, pool, '1. Headroom, points (board + predicted residual, held-out seasons):'))
    no_xfp = [c for c in cols if c not in ('f_xfp_td', 'f_luck_td', 'f_xfp_l3')]
    pred2, _ = heldout_predictions(x, no_xfp, resid, wide, pool)
    lines += ['', headroom_table(x, base, act, pred2, pool, '   without the in-sample expected-points features:'), '']
    pd.DataFrame({'year': x['year'], 'week': x['week'], 'player': x['player'], 'pos': x['pos'], 'team': x['team'],
                  'pred_resid': pred}).to_parquet(a.out.replace('.txt', '_pred.parquet'), index=False)

    imp = permutation_importance(x, cols, models, base, act, pool)
    lines += ['2. Permutation importance (held-out gain lost when shuffled, points^2; top):',
              imp.head(a.top).round(4).to_string(index=False), '']

    scan = bin_scan(x, cols, resid, base, act, wide, pool)
    lines += ['3. One feature at a time (quintile bins fit on four seasons, START pool of the fifth):',
              scan.drop(columns=['_better']).head(a.top).round(4).to_string(index=False), '']
    for _, r in scan[scan['_better'] >= 4].head(8).iterrows():
        lines += [f"   {r['pos']} {r['feature']} (five-season bins, training rows; resid = actual - board):",
                  bin_table(x, r['feature'], r['pos'], resid, wide), '']

    lines.append('4. Stat headroom (same model, residual of the stat itself):')
    stat_headroom(x, cols, lines)
    lines.append('   Team totals (team features only):')
    team_headroom(x, lines)
    text = '\n'.join(lines)
    print(text)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
