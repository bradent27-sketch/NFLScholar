"""
Does targets-per-route-run (TPRR) carry information about next week's targets that the model's target projection lacks?
(2026-10-19, backlog item N2; first read.)

targets = routes run x TPRR. Routes (volume) are stable; TPRR is noisy and should be regressed to the position mean, more
for a small sample. The model's target rate is a plain per-game blend, so a high-TPRR-so-far receiver may be projected as
if his rate were all talent. Test, WR/TE rows of the lab dump for 2024-2025 (the PFF weekly exports with `routes` we hold),
weeks 3-17, played:

    miss = actual targets - projected targets
    z    = routes_pg_to_date * (TPRR_to_date - position mean TPRR)     (the targets "above routes-average" so far, per game)

A negative slope on z = the model over-trusts a hot TPRR. Also the routes-per-game deviation from what the board's snap
share implies. Slopes by position with a bootstrap CI over weeks, and the same on points. First read only: no model change.

    python scripts/eval_routes_targets.py
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.utils import clean_name_exact  # noqa: E402
from scripts.diag_returning_players import shipped_board  # noqa: E402
from scripts.eval_calibration_schemes import load_dump  # noqa: E402

YEARS = (2024, 2025)


def load_pff_weekly(year):
    rows = []
    for path in glob.glob(os.path.join('pff_imports', str(year), 'weekly', '*', 'receiving_summary.csv')):
        week = int(os.path.basename(os.path.dirname(path)))
        df = pd.read_csv(path, usecols=lambda c: c in ('player', 'position', 'team_name', 'routes', 'targets', 'receptions',
                                                       'yards', 'route_rate', 'player_game_count'))
        df['week'], df['year'] = week, year
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out['key'] = clean_name_exact(out['player'].astype(str))
    return out[out['routes'] > 0]


def features(pff, min_routes=20):
    """Per (key, year, week): routes per earlier game, TPRR over earlier games, earlier games, and the position-mean TPRR
    of everyone's earlier games that season (so no future information enters)."""
    pff = pff.sort_values(['key', 'year', 'week']).copy()
    g = pff.groupby(['key', 'year'])
    for c in ('routes', 'targets'):
        pff['cum_' + c] = g[c].cumsum() - pff[c]
    pff['games_before'] = g.cumcount()
    out = []
    for (year), sub in pff.groupby('year'):
        for week in sorted(sub['week'].unique()):
            earlier = sub[sub['week'] < week]
            if earlier.empty:
                continue
            for grp, label in ((['WR'], 'WR'), (['TE'], 'TE')):
                e = earlier[earlier['position'].isin(grp)]
                mean_tprr = e['targets'].sum() / max(e['routes'].sum(), 1)
                cur = sub[(sub['week'] == week) & sub['position'].isin(grp)][['key', 'year', 'week', 'cum_routes', 'cum_targets', 'games_before']].copy()
                cur['mean_tprr'] = mean_tprr
                out.append(cur)
    f = pd.concat(out, ignore_index=True)
    f = f[(f['games_before'] >= 2) & (f['cum_routes'] >= min_routes)]
    f['routes_pg'] = f['cum_routes'] / f['games_before']
    f['tprr'] = f['cum_targets'] / f['cum_routes']
    f['z'] = f['routes_pg'] * (f['tprr'] - f['mean_tprr'])
    return f[['key', 'year', 'week', 'games_before', 'routes_pg', 'tprr', 'mean_tprr', 'z']]


def main():
    pd.set_option('display.width', 220)
    d = load_dump('.sweeps/calibration_lab_v7.csv', '.sweeps/calibration_lab_v7_wk1_ourlads.csv')
    d['shown'] = shipped_board(d)
    d['key'] = clean_name_exact(d['player'].astype(str))
    f = features(pd.concat([load_pff_weekly(y) for y in YEARS], ignore_index=True)).drop_duplicates(['key', 'year', 'week'], keep=False)
    d = d.merge(f, on=['key', 'year', 'week'], how='left')
    m = d[(d['pos'].isin(['WR', 'TE'])) & d['played'] & d['actual'].notna() & d['week'].between(3, 17) & d['year'].isin(YEARS)
          & d['z'].notna() & (d['p_targets'] > 1.0)].copy()
    m['miss_t'] = m['a_targets'] - m['p_targets']
    m['miss_pts'] = m['actual'] - m['shown']
    print(f"WR/TE rows 2024-2025 weeks 3-17 with PFF routes history: {len(m):,}")
    rows = []
    for pos in ('WR', 'TE', 'WR+TE'):
        s = m if pos == 'WR+TE' else m[m['pos'] == pos]
        for ycol in ('miss_t', 'miss_pts'):
            x, y = s['z'].to_numpy(float), s[ycol].to_numpy(float)
            b = np.polyfit(x, y, 1)
            keyed = (s['year'] * 100 + s['week']).to_numpy()
            groups = {k: np.where(keyed == k)[0] for k in np.unique(keyed)}
            ks = list(groups)
            rng = np.random.default_rng(0)
            boots = []
            for _ in range(500):
                ix = np.concatenate([groups[ks[i]] for i in rng.choice(len(ks), len(ks))])
                boots.append(np.polyfit(x[ix], y[ix], 1)[0])
            by_year = [np.polyfit(g['z'], g[ycol], 1)[0] for _, g in s.groupby('year')]
            rows.append({'rows': pos, 'y': ycol, 'n': len(s), 'slope on z': b[0], 'ci': f'[{np.percentile(boots, 2.5):+.2f},{np.percentile(boots, 97.5):+.2f}]',
                         'by year': ' '.join(f'{v:+.2f}' for v in by_year), 'sd z': x.std(), 'corr': np.corrcoef(x, y)[0, 1]})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print('\nz = routes per game x (TPRR to date - position mean TPRR); a negative slope means a hot TPRR is over-trusted.')
    print('mean TPRR (WR %.3f, TE %.3f); mean routes per game (WR %.1f, TE %.1f)' % (
        m[m.pos == 'WR'].tprr.mean(), m[m.pos == 'TE'].tprr.mean(), m[m.pos == 'WR'].routes_pg.mean(), m[m.pos == 'TE'].routes_pg.mean()))


if __name__ == '__main__':
    main()
