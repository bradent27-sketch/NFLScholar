"""
A1 offline study (2026-11-11, docs/audit_leads_2026-11-10.md): does the final practice status (full / limited / did not
participate) say more than the game status the model reads (Questionable, Doubtful, none)?

Panel: the harness's base boards 2022-2025 weeks 3-17 (.sweeps/harness_boards/capacity_screen_2022-2025_wk3-17.parquet,
variant 'base' - the default before the 2026-11-03 RB/TE capacity relief; the injury handling is the same), joined to the
nflverse injury report (data.historical_availability.load_injury_reports: the week's LAST filing only, so this is the final
practice status, not the day-by-day pattern), the box scores and the snap counts.

Per position, cell (game status x practice status) and season: rows, the model's mean Availability, the share who played (an
offensive snap), and the bias (actual minus projection) on points, the main stats and snap share - over every row (expected
value: a player who sat scores 0) and over the played rows - each minus the same season and position's unlisted players' bias,
so a cell reads against players the report never named. A cell is a lead when its relative bias has the same sign in 4 of 4
seasons (the harness has no 5th; the live 2026 ledger can be the 5th later).

    python scripts/study_practice_status.py [--out .sweeps/study_practice_status.txt]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.historical_availability import load_injury_reports  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402

BOARD = os.path.join('.sweeps', 'harness_boards', 'capacity_screen_2022-2025_wk3-17.parquet')
SEASONS = (2022, 2023, 2024, 2025)
POSITIONS = ('QB', 'RB', 'WR', 'TE')
STATS = {'pts': 'Model Proj Pts', 'receptions': 'receptions', 'receiving_yards': 'receiving_yards',
         'rushing_yards': 'rushing_yards', 'passing_yards': 'passing_yards'}
PRACTICE = {'Full Participation in Practice': 'full', 'Limited Participation in Practice': 'limited',
            'Did Not Participate In Practice': 'DNP'}
MIN_ROWS = 20


def panel():
    board = pd.read_parquet(BOARD)
    board = board[(board['variant'] == 'base') & board['Pos'].isin(POSITIONS)].copy()
    board['_key'] = clean_name_exact(board['Player'].astype(str))
    frames = []
    for season in SEASONS:
        b = board[board['year'] == season]
        inj = load_injury_reports(season)
        inj = inj[inj['week'].between(3, 17)].copy()
        inj['_key'] = clean_name_exact(inj['full_name'].astype(str))
        inj = inj.drop_duplicates(['week', '_key', 'team'], keep='last')[['week', '_key', 'team', 'report_status',
                                                                          'practice_status']]
        b = b.merge(inj.rename(columns={'team': 'Team'}), on=['week', '_key', 'Team'], how='left')
        stats, _tc, name_col, _ = load_and_merge_data(season, 'Full PPR')
        stats = stats[stats['season'] == season] if 'season' in stats.columns else stats
        act = stats[['week', name_col, 'fantasy_points_ppr'] + [s for s in STATS.values() if s in stats.columns
                                                                  and s != 'Model Proj Pts']].copy()
        act['_key'] = clean_name_exact(act[name_col].astype(str))
        act = act.drop(columns=[name_col]).drop_duplicates(['week', '_key'])
        act = act.rename(columns={c: f'act_{c}' for c in act.columns if c not in ('week', '_key')})
        b = b.merge(act, on=['week', '_key'], how='left')
        snaps = pd.read_parquet(os.path.join('data', 'cache', f'snap_counts_{season}.parquet'))
        snaps = snaps[snaps['game_type'].astype(str).eq('REG')].copy()
        snaps['_key'] = clean_name_exact(snaps['player'].astype(str))
        snaps = snaps.groupby(['week', '_key'], as_index=False)[['offense_snaps', 'offense_pct']].max()
        b = b.merge(snaps, on=['week', '_key'], how='left')
        frames.append(b)
    p = pd.concat(frames, ignore_index=True)
    p['played'] = p['offense_snaps'].fillna(0) > 0
    p['game'] = p['report_status'].fillna('none').replace({'Out': 'Out'})
    unlisted = pd.Series(np.where(p['report_status'].isna() & p['practice_status'].isna(), 'unlisted', 'none'), index=p.index)
    p['practice'] = p['practice_status'].map(PRACTICE).fillna(unlisted)
    p['cell'] = np.where(p['practice'].eq('unlisted'), 'unlisted', p['game'] + ' x ' + p['practice'])
    p['act_pts'] = p['act_fantasy_points_ppr']
    for k, col in STATS.items():
        a = f'act_{col}' if k != 'pts' else 'act_pts'
        actual = pd.to_numeric(p[a], errors='coerce').where(p['played'], 0.0).fillna(0.0)
        p[f'bias_{k}'] = actual - pd.to_numeric(p[col], errors='coerce')
    p['bias_snap'] = p['offense_pct'].where(p['played'], 0.0).fillna(0.0) - pd.to_numeric(p['Expected Snap Share'],
                                                                                          errors='coerce')
    return p


def cell_table(p):
    """Per (Pos, cell, season): rows, model availability, played share, relative bias (cell minus unlisted) for every row and
    for played rows."""
    out = []
    keys = ['Pos', 'season_', 'cell']
    p = p.assign(season_=p['year'])
    base_all = p[p['cell'] == 'unlisted'].groupby(['Pos', 'season_'])
    base_play = p[(p['cell'] == 'unlisted') & p['played']].groupby(['Pos', 'season_'])
    cols = [c for c in p.columns if c.startswith('bias_')]
    b_all, b_play = base_all[cols].mean(), base_play[cols].mean()
    for (pos, season, cell), g in p.groupby(keys):
        if cell in ('unlisted',) or cell.startswith('Out') or len(g) < 1:
            continue
        rec = {'Pos': pos, 'season': season, 'cell': cell, 'rows': len(g), 'model avail': g['Availability'].mean(),
               'played': g['played'].mean()}
        for c in cols:
            rec[f'{c[5:]} all'] = g[c].mean() - b_all.at[(pos, season), c]
            gp = g[g['played']]
            rec[f'{c[5:]} played'] = gp[c].mean() - b_play.at[(pos, season), c] if len(gp) else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def leads(t, measures=('pts all', 'pts played', 'snap played', 'receptions played', 'rushing_yards played')):
    """Pooled per (Pos, cell): rows, mean played share and model availability, the pooled relative bias per measure and how many
    seasons share its sign; 'lead' when 4 of 4 with MIN_ROWS+ rows a season."""
    out = []
    for (pos, cell), g in t.groupby(['Pos', 'cell']):
        if g['rows'].min() < MIN_ROWS or g['season'].nunique() < len(SEASONS):
            continue
        w = g['rows']
        rec = {'Pos': pos, 'cell': cell, 'rows': int(w.sum()), 'played': np.average(g['played'], weights=w),
               'model avail': np.average(g['model avail'], weights=w)}
        flags = []
        for m in measures:
            v = g[m].to_numpy(float)
            pooled = np.nansum(v * w) / w[np.isfinite(v)].sum() if np.isfinite(v).any() else np.nan
            same = int((np.sign(v) == np.sign(pooled)).sum())
            rec[m] = pooled
            rec[f'{m} seasons'] = same
            if same == len(SEASONS):
                flags.append(m)
        rec['same sign 4/4'] = ', '.join(flags)
        out.append(rec)
    return pd.DataFrame(out).sort_values(['Pos', 'rows'], ascending=[True, False])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join('.sweeps', 'study_practice_status.txt'))
    a = ap.parse_args()
    p = panel()
    t = cell_table(p)
    fmt = lambda v: f'{v:.3f}'  # noqa: E731
    cells = p[p['Pos'].isin(POSITIONS)].groupby('cell').size().sort_values(ascending=False)
    text = '\n'.join([
        "A1: practice status against the game status, base harness boards 2022-2025 wk3-17 "
        f"({len(p):,} player-weeks; injury report = the week's last filing). Bias = actual minus projection, as the cell's "
        "mean minus the unlisted players' mean (same position and season); 'all' counts a player who sat as 0, 'played' only "
        "rows with an offensive snap. snap = offense share minus Expected Snap Share.",
        '', 'Rows per cell:', cells.to_string(), '',
        f'1. LEADS: pooled per position and cell ({MIN_ROWS}+ rows in every season); seasons = how many share the pooled sign',
        leads(t).to_string(index=False, float_format=fmt), '',
        '2. BY SEASON (points and snaps)',
        t[['Pos', 'cell', 'season', 'rows', 'model avail', 'played', 'pts all', 'pts played', 'snap played']]
        .sort_values(['Pos', 'cell', 'season']).to_string(index=False, float_format=fmt)])
    print(text)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')
    t.to_csv(a.out.replace('.txt', '_cells.csv'), index=False)


if __name__ == '__main__':
    main()
