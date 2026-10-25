"""
How big are silent games (2026-10-25, user: "Count silent games ... essential for making sure the model is accurate and that our
backtests are accurate ... especially for players who get less volume")? A silent game is a regular-season game the snap counts put a
player on the field for (offensive snaps > 0) with no box-score row: nflverse's weekly stats are stat-triggered, so that game is
missing from his history (his per-game rates are averages over the games he DID record a stat) and from every backtest pool that
reads "played" off the box score.

Per season and position: snap appearances, the share with no box-score row, how that share runs by snap share, how many players
it touches, and how far it inflates a per-game rate (targets / carries per game over box-score games vs over every snap game).

Matching: snap rows carry PFR ids (pfr_player_id), box-score rows carry the same id (pfr_id); a snap row whose player has no PFR id
match that week falls back to the cleaned name.

    python scripts/diag_silent_games.py [--years 2021,2022,2023,2024,2025]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.loaders import _load_snap_counts_raw  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402

SKILL = ('QB', 'RB', 'WR', 'TE')
SNAP_POS = {'QB': 'QB', 'RB': 'RB', 'FB': 'RB', 'WR': 'WR', 'TE': 'TE'}
SHARE_BINS = [0, 0.05, 0.15, 0.30, 0.50, 0.70, 1.01]


def season_frame(year):
    stats, _team_col, name_col, _ = load_and_merge_data(year, 'Full PPR')
    box = stats[pd.to_numeric(stats['week'], errors='coerce').between(1, 18)].copy()
    box['week'] = pd.to_numeric(box['week'], errors='coerce').astype(int)
    box = box[box['position'].isin(SKILL)]
    snaps = _load_snap_counts_raw(year)
    snaps = snaps[(snaps['game_type'] == 'REG') & (pd.to_numeric(snaps['offense_snaps'], errors='coerce') > 0)].copy()
    snaps['pos'] = snaps['position'].map(SNAP_POS)
    snaps = snaps[snaps['pos'].notna()]
    snaps['week'] = snaps['week'].astype(int)
    snaps['share'] = pd.to_numeric(snaps['offense_pct'], errors='coerce')
    snaps['_key'] = clean_name_exact(snaps['player'])
    by_id = set(zip(box['pfr_id'].astype(str), box['week']))
    by_name = set(zip(box['exact_name'].astype(str), box['week']))
    snaps['boxed'] = [(str(i), w) in by_id or (k, w) in by_name
                      for i, k, w in zip(snaps['pfr_player_id'], snaps['_key'], snaps['week'])]
    snaps['year'] = year
    return snaps, box, name_col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2021,2022,2023,2024,2025')
    a = ap.parse_args()
    frames = []
    rate_rows = []
    for year in [int(y) for y in a.years.split(',')]:
        snaps, box, _name_col = season_frame(year)
        frames.append(snaps)
        # rate inflation: per player-season, volume per box-score game vs per snap game (silent games add 0 volume)
        vol = {'RB': 'carries', 'WR': 'targets', 'TE': 'targets'}
        for pos, stat in vol.items():
            b = box[box['position'] == pos]
            col = 'carries' if stat == 'carries' and 'carries' in b.columns else ('rushing_attempts' if stat == 'carries' else 'targets')
            if col not in b.columns:
                continue
            per = b.groupby('pfr_id').agg(volume=(col, 'sum'), box_games=('week', 'nunique'))
            sg = snaps[(snaps['pos'] == pos)].groupby('pfr_player_id').agg(snap_games=('week', 'nunique'),
                                                                            silent=('boxed', lambda s: int((~s).sum())))
            j = per.join(sg, how='inner')
            j = j[j['box_games'] >= 2]
            j['rate_box'] = j['volume'] / j['box_games']
            j['rate_all'] = j['volume'] / (j['box_games'] + j['silent'])
            for tier, part in (('all players', j), ('rate under 3/game', j[j['rate_box'] < 3]),
                               ('rate 3+/game', j[j['rate_box'] >= 3])):
                if part.empty:
                    continue
                rate_rows.append({'year': year, 'pos': pos, 'stat': stat, 'tier': tier, 'players': len(part),
                                  'with silent games': int((part['silent'] > 0).sum()),
                                  'rate over box games': part['rate_box'].mean(), 'rate over snap games': part['rate_all'].mean(),
                                  'inflation': part['rate_box'].sum() / part['rate_all'].sum() - 1})
    allsnaps = pd.concat(frames, ignore_index=True)
    out = ['SILENT GAMES: offensive-snap games with no box-score row (regular season)', '']
    t = (allsnaps.groupby(['year', 'pos']).agg(snap_games=('boxed', 'size'), silent=('boxed', lambda s: int((~s).sum())))
         .assign(share=lambda d: d['silent'] / d['snap_games']))
    out.append(t.to_string(float_format=lambda v: f'{v:.3f}'))
    out.append('')
    allsnaps['share_bin'] = pd.cut(allsnaps['share'], SHARE_BINS, right=False)
    t2 = (allsnaps.groupby(['pos', 'share_bin'], observed=True)
          .agg(snap_games=('boxed', 'size'), silent=('boxed', lambda s: int((~s).sum())))
          .assign(share=lambda d: d['silent'] / d['snap_games']))
    out.append('by snap share that game (2021-2025 pooled)')
    out.append(t2.to_string(float_format=lambda v: f'{v:.3f}'))
    out.append('')
    out.append('per-game rate over box-score games vs over every snap game (player-seasons with 2+ box games)')
    out.append(pd.DataFrame(rate_rows).to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    text = '\n'.join(out)
    print(text)
    with open('.sweeps/diag_silent_games.txt', 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
