"""
Does a team's pass rate over expected (PROE: actual pass rate minus nflverse's play-by-play `xpass`, which models the
pass probability from down, distance, field position, score and time) carry information about the team's NEXT-game
passing volume that the model's team pass-attempt projection lacks? (2026-10-19, backlog item N6; first read.)

Team-week test, 2021-2025 weeks 3-17: miss = (team's actual pass attempts) - (sum of the board's projected pass attempts of
its QBs, the lab dump). Features use EARLIER weeks of the same season only (>= 2 games): proe_to_date (percentage points),
and the prior season's PROE. Slope by season with a bootstrap CI over weeks; also the correlation of PROE with the miss.
First read only: no model change.

    python scripts/eval_proe.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.loaders import load_pbp  # noqa: E402
from scripts.eval_calibration_schemes import load_dump  # noqa: E402

YEARS = tuple(range(2020, 2026))


def team_game_proe(year):
    pbp = load_pbp(year)
    r = pbp[pbp['play_type'].isin(['pass', 'run']) & pbp['xpass'].notna() & pbp['posteam'].notna()].copy()
    if 'season_type' in r.columns:
        r = r[r['season_type'].astype(str).str.upper() == 'REG']
    r['pass_flag'] = (r['play_type'] == 'pass').astype(float)
    g = r.groupby(['posteam', 'week']).agg(plays=('pass_flag', 'size'), pass_rate=('pass_flag', 'mean'), xpass=('xpass', 'mean')).reset_index()
    g['year'] = year
    g['proe_game'] = 100.0 * (g['pass_rate'] - g['xpass'])
    return g


def features(tg):
    tg = tg.sort_values(['posteam', 'year', 'week']).copy()
    rows = []
    for (team, year), g in tg.groupby(['posteam', 'year']):
        g = g.reset_index(drop=True)
        cum = (g['proe_game'] * g['plays']).cumsum() - g['proe_game'] * g['plays']
        cp = g['plays'].cumsum() - g['plays']
        for i, r in g.iterrows():
            rows.append({'team': team, 'year': year, 'week': int(r['week']), 'games_before': i,
                         'proe_to_date': cum[i] / cp[i] if cp[i] > 0 else np.nan})
    f = pd.DataFrame(rows)
    season = tg.groupby(['posteam', 'year']).apply(lambda g: (g['proe_game'] * g['plays']).sum() / g['plays'].sum(), include_groups=False).rename('proe_season').reset_index()
    season['year'] += 1
    season = season.rename(columns={'posteam': 'team', 'proe_season': 'proe_prior'})
    return f.merge(season, on=['team', 'year'], how='left')


def main():
    pd.set_option('display.width', 220)
    tg = pd.concat([team_game_proe(y) for y in YEARS], ignore_index=True)
    f = features(tg)
    d = load_dump('.sweeps/calibration_lab_v7.csv', '.sweeps/calibration_lab_v7_wk1_ourlads.csv')
    q = d[(d['pos'] == 'QB') & d['week'].between(3, 17)].copy()
    q['p_att'] = q['p_passing_attempts'].fillna(0.0)
    q['a_att'] = q['a_passing_attempts'].fillna(0.0)
    team = q.groupby(['year', 'week', 'team']).agg(p_att=('p_att', 'sum'), a_att=('a_att', 'sum'), played=('played', 'any')).reset_index()
    team = team[team['played'] & (team['p_att'] > 15)]
    team = team.merge(f, on=['year', 'week', 'team'], how='left')
    team = team[(team['games_before'] >= 2) & team['proe_to_date'].notna()].copy()
    team['miss'] = team['a_att'] - team['p_att']
    print(f"team-weeks: {len(team):,}  mean projected {team.p_att.mean():.1f} actual {team.a_att.mean():.1f}")
    rows = []
    for label, s in [('all', team)] + [(str(y), team[team.year == y]) for y in range(2021, 2026)]:
        for col in ('proe_to_date', 'proe_prior'):
            t = s[s[col].notna()]
            if len(t) < 60:
                continue
            x, y = t[col].to_numpy(float), t['miss'].to_numpy(float)
            b = np.polyfit(x, y, 1)
            keyed = (t['year'] * 100 + t['week']).to_numpy()
            groups = {k: np.where(keyed == k)[0] for k in np.unique(keyed)}
            ks = list(groups)
            rng = np.random.default_rng(0)
            boots = []
            for _ in range(500):
                ix = np.concatenate([groups[ks[i]] for i in rng.choice(len(ks), len(ks))])
                boots.append(np.polyfit(x[ix], y[ix], 1)[0])
            rows.append({'sample': label, 'feature': col, 'n': len(t), 'slope (att per PROE pt)': b[0],
                         'ci': f'[{np.percentile(boots, 2.5):+.2f},{np.percentile(boots, 97.5):+.2f}]', 'corr': np.corrcoef(x, y)[0, 1], 'sd of feature': x.std()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
