"""
Drill-down of the residual scan's leads (model audit step 2b, 2026-10-21), on the START pool of the audit feature table.
Each lead: the residual (actual - board) by the feature, by position, with the number of seasons that agree in sign, and the
held-out gain of the simplest correction (one linear term in the feature, fit on four seasons and scored on the fifth, against the board as it
stands; the points line refit is shown beside it so a lead that is only the line's level or slope can be told apart).

  A. Implied team total: level, and the week's implied total against the team's own to-date average (game environment
     separated from team quality), by stat (TDs, yards, receptions).
  B. Experience: rookies and second-year players by part of the season (does the gap grow as the season goes on?), and by stat.
  C. Last season's target share (WR/TE), by part of the season (does it fade as this season's data arrive?).
  D. Last game's snap share against the 3-game average (a role change the recency weights are slow to see).
  E. Level by position and season (points and stats).

    python scripts/audit_leads.py [--features .sweeps/audit_features.parquet] [--out .sweeps/audit_leads.txt]
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

POSITIONS = ('QB', 'RB', 'WR', 'TE')
PHASES = ((3, 6, 'wk 3-6'), (7, 11, 'wk 7-11'), (12, 17, 'wk 12-17'))
STAT_RESID = {'QB': ('passing_tds', 'passing_yards', 'rushing_yards'),
              'RB': ('rushing_tds', 'rushing_yards', 'rushing_attempts', 'targets', 'receiving_yards'),
              'WR': ('receiving_tds', 'receiving_yards', 'receptions', 'targets'),
              'TE': ('receiving_tds', 'receiving_yards', 'receptions', 'targets')}


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else float('nan')


def load(path):
    x = pd.read_parquet(path)
    x = x[x['pool'].astype(bool) & x['played'].astype(bool)].copy()
    x['resid'] = x['actual'] - x['f_final']
    x['phase'] = pd.cut(x['week'], [2, 6, 11, 17], labels=[p[2] for p in PHASES])
    x = x.sort_values(['year', 'team', 'week'])
    # the team's average implied total over its earlier games this season (pre-game; NaN in its first game)
    tm = x.drop_duplicates(['year', 'team', 'week'])[['year', 'team', 'week', 'f_implied']].sort_values(['year', 'team', 'week'])
    tm['implied_avg_td'] = tm.groupby(['year', 'team'])['f_implied'].transform(lambda s: s.shift(1).expanding().mean())
    x = x.merge(tm[['year', 'team', 'week', 'implied_avg_td']], on=['year', 'team', 'week'], how='left')
    x['implied_dev'] = x['f_implied'] - x['implied_avg_td']
    x['snap_dev'] = x['f_snap_last'] - x['f_snap_l3']
    x['exp_group'] = np.select([x['f_years_exp'] == 0, x['f_years_exp'] == 1, x['f_years_exp'] >= 2],
                               ['rookie', 'year 2', 'year 3+'], 'unknown')
    for pos, stats in STAT_RESID.items():
        for s in stats:
            x[f'r_{s}'] = x[f'a_{s}'] - x[f'f_p_{s}'].fillna(0.0)
    return x


def _heldout_fit(x, cols, target):
    corr = np.full(len(x), np.nan)
    for y in sorted(x['year'].unique()):
        tr = (x['year'] != y) & x[cols].notna().all(axis=1)
        te = (x['year'] == y)
        X = np.column_stack([np.ones(tr.sum())] + [x.loc[tr, c].to_numpy(float) for c in cols])
        beta, *_ = np.linalg.lstsq(X, x.loc[tr, target].to_numpy(float), rcond=None)
        Xt = np.column_stack([np.ones(te.sum())] + [x.loc[te, c].fillna(x.loc[tr, c].mean()).to_numpy(float) for c in cols])
        corr[te.to_numpy()] = Xt @ beta
    return corr


def heldout_linear_gain(x, feature, target='resid'):
    """Held-out (by season) RMSE change against the board of (a) target ~ a + b*final (the points line refit), and (b)
    target ~ a + c*(feature - its training mean): one term in the feature and nothing else, the size a mechanism acting
    through that feature alone could reach; seasons where (b) beats the board."""
    e0 = x[target].to_numpy(float)
    line = _heldout_fit(x, ['f_final'], target)
    feat = np.full(len(x), np.nan)
    for y in sorted(x['year'].unique()):
        tr = (x['year'] != y) & x[feature].notna()
        te = (x['year'] == y)
        v = x[feature].to_numpy(float)
        b = np.polyfit(v[tr.to_numpy()], e0[tr.to_numpy()], 1)[0]
        mu = np.nanmean(v[tr.to_numpy()])
        feat[te.to_numpy()] = b * np.nan_to_num(v[te.to_numpy()] - mu)
    d_line = rmse(e0 - line) - rmse(e0)
    d_feat = rmse(e0 - feat) - rmse(e0)
    yrs = sum(rmse((e0 - feat)[(x['year'] == y).to_numpy()]) < rmse(e0[(x['year'] == y).to_numpy()])
              for y in x['year'].unique())
    return d_line, d_feat, f'{yrs}/{x["year"].nunique()}'


def binned(x, feature, by=('pos',), q=5, targets=('resid',)):
    x = x[x[feature].notna()].copy()
    x['bin'] = x.groupby(list(by))[feature].transform(lambda s: pd.qcut(s.rank(method='first'), q, labels=False))
    rows = []
    for keys, g in x.groupby(list(by) + ['bin'], observed=True):
        r = dict(zip(list(by) + ['bin'], keys if isinstance(keys, tuple) else (keys,)))
        r.update({'lo': g[feature].min(), 'hi': g[feature].max(), 'n': len(g)})
        for t in targets:
            if t not in g:
                continue
            ys = g.groupby('year')[t].mean()
            r[t] = g[t].mean()
            r[f'{t} same sign'] = int(max((ys > 0).sum(), (ys < 0).sum()))
        rows.append(r)
    return pd.DataFrame(rows).round(3).to_string(index=False)


def grouped(x, keys, targets=('resid',)):
    rows = []
    for k, g in x.groupby(keys, observed=True):
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        r['n'] = len(g)
        for t in targets:
            if t not in g:
                continue
            ys = g.groupby('year')[t].mean()
            r[t] = g[t].mean()
            r[f'{t} same sign'] = int(max((ys > 0).sum(), (ys < 0).sum()))
        rows.append(r)
    return pd.DataFrame(rows).round(3).to_string(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='.sweeps/audit_features.parquet')
    ap.add_argument('--out', default='.sweeps/audit_leads.txt')
    a = ap.parse_args()
    pd.set_option('display.width', 240)
    pd.set_option('display.max_columns', 40)
    x = load(a.features)
    L = [f'START pool, played: {len(x):,} rows 2021-2025 weeks 3-17. resid = actual - board (points); r_<stat> = actual - '
         'projected stat. "same sign" = seasons (of 5) whose mean has the majority sign.', '']

    L.append('A. Implied team total. Held-out gain of one linear term (with the board number), by position:')
    for feat in ('f_implied', 'implied_dev', 'f_spread'):
        for pos in POSITIONS:
            p = x[x['pos'] == pos]
            dl, df, yrs = heldout_linear_gain(p, feat)
            L.append(f'   {feat:12s} {pos}: points line refit {dl:+.4f}; this term alone {df:+.4f} ({yrs} seasons better)')
    L += ['', '   by implied total (level):', binned(x, 'f_implied', targets=('resid',)), '',
          '   by implied total vs the team\'s own to-date average (environment, team quality removed):',
          binned(x, 'implied_dev', targets=('resid',)), '']
    for pos in POSITIONS:
        p = x[x['pos'] == pos]
        L += [f'   {pos} stats by implied_dev:', binned(p, 'implied_dev', by=('pos',),
                                                       targets=tuple(f'r_{s}' for s in STAT_RESID[pos])), '']

    L.append('B. Experience by part of the season:')
    L += [grouped(x, ['pos', 'exp_group', 'phase'], targets=('resid',)), '']
    for pos in ('RB', 'WR', 'TE'):
        p = x[x['pos'] == pos]
        L += [f'   {pos} stats by experience:', grouped(p, ['exp_group'], targets=tuple(f'r_{s}' for s in STAT_RESID[pos])), '']
    for pos in POSITIONS:
        p = x[x['pos'] == pos].assign(rookie=lambda d: (d['f_years_exp'] == 0).astype(float),
                                      year2=lambda d: (d['f_years_exp'] == 1).astype(float))
        for feat in ('rookie', 'year2'):
            dl, df, yrs = heldout_linear_gain(p, feat)
            L.append(f'   {feat:6s} {pos}: this term alone {df:+.4f} ({yrs} seasons better)')
    L.append('')

    L.append("C. Last season's target share (WR/TE) by part of the season:")
    for pos in ('WR', 'TE'):
        p = x[x['pos'] == pos]
        L += [binned(p, 'f_tshare_ly', by=('phase',), q=4, targets=('resid', 'r_targets')), '']
        dl, df, yrs = heldout_linear_gain(p, 'f_tshare_ly')
        L.append(f'   tshare_ly {pos}: this term alone {df:+.4f} ({yrs} seasons better)')
    L.append('')

    L.append("D. Last game's snap share minus its 3-game average:")
    L += [binned(x, 'snap_dev', targets=('resid', 'r_targets')), '']
    for pos in POSITIONS:
        p = x[x['pos'] == pos]
        dl, df, yrs = heldout_linear_gain(p, 'snap_dev')
        L.append(f'   snap_dev {pos}: this term alone {df:+.4f} ({yrs} seasons better)')
    L.append('')

    L.append('E. Level (mean of actual - board) by position and season, points and stats:')
    tg = ['resid', 'r_targets', 'r_receptions', 'r_receiving_yards', 'r_receiving_tds', 'r_rushing_attempts',
          'r_rushing_yards', 'r_rushing_tds', 'r_passing_yards', 'r_passing_tds']
    t = x.groupby(['pos', 'year'])[[c for c in tg if c in x]].mean().round(3)
    L += [t.to_string(), '']
    for pos in POSITIONS:
        dl, _, _ = heldout_linear_gain(x[x['pos'] == pos], 'f_final')
        L.append(f'   {pos}: a held-out points line on top of the board (level + slope) {dl:+.4f}')
    text = '\n'.join(L)
    print(text)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
