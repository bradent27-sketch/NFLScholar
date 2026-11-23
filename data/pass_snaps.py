"""Pass-play snaps per player-game: on the field for a dropback (accuracy idea 1/2, 2026-11-23).

Two sources, one definition (his snaps on the team's dropbacks; team_db = the team's dropbacks that game):
  - nflverse participation (``load_participation``, 2016-2025): the offensive players on every play, joined to play-by-play
    dropbacks. Cached as data/cache/pass_snaps_{year}.parquet (the derived table, a few thousand rows a season).
  - the weekly PFF receiving exports' ``pass_plays`` column (2024 on; the only source for 2026, which participation does not
    cover yet), read through data.route_participation.weekly_route_rows. team_db there = the most pass plays any of the
    team's receivers was on the field for.
``pass_snap_rows`` picks participation when the season has it, else PFF. Pass-play snaps are not routes: a back or tight end
in pass protection counts. Participation has no per-player route flag, so this is the one definition both sources share.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.utils import clean_name_exact

PARTICIPATION_SEASONS = range(2016, 2026)
_COLUMNS = ['week', 'Team', '_key', 'pass_snaps', 'team_db']


def _participation_derived(year: int) -> pd.DataFrame:
    """[season, week, Team, gsis_id, _key, pass_snaps, team_db] for one season's regular-season dropbacks."""
    import nflreadpy

    from data.historical_availability import load_weekly_rosters
    from data.loaders import load_pbp
    from data.pff_alignment import _canonical_team_key
    part = nflreadpy.load_participation(seasons=[int(year)]).to_pandas()
    part = part.rename(columns={'nflverse_game_id': 'game_id'})[['game_id', 'play_id', 'offense_players']]
    part['play_id'] = pd.to_numeric(part['play_id'], errors='coerce')
    pbp = load_pbp(int(year))
    if pbp is None or pbp.empty or part.empty:
        return pd.DataFrame()
    pbp = pbp.loc[(pbp['season_type'] == 'REG') & (pd.to_numeric(pbp['qb_dropback'], errors='coerce') == 1),
                  ['game_id', 'play_id', 'week', 'posteam']]
    m = pbp.merge(part, on=['game_id', 'play_id'], how='inner')
    m = m[m['offense_players'].astype(str).str.len() > 5]
    if m.empty:
        return pd.DataFrame()
    team_db = m.groupby(['week', 'posteam']).size().rename('team_db')
    ex = m[['week', 'posteam', 'offense_players']].assign(gsis_id=m['offense_players'].astype(str).str.split(';'))
    ex = ex.explode('gsis_id')
    pc = ex.groupby(['week', 'posteam', 'gsis_id']).size().rename('pass_snaps').reset_index()
    pc = pc.join(team_db, on=['week', 'posteam'])
    ros = load_weekly_rosters(int(year))
    names = (ros[['gsis_id', 'full_name']].dropna().drop_duplicates('gsis_id')
             if ros is not None and not ros.empty else pd.DataFrame(columns=['gsis_id', 'full_name']))
    pc = pc.merge(names, on='gsis_id', how='inner')
    pc['_key'] = clean_name_exact(pc['full_name'].astype(str)).to_numpy()
    pc['Team'] = pc['posteam'].map(_canonical_team_key)
    pc['season'] = int(year)
    return pc[['season', 'week', 'Team', 'gsis_id', '_key', 'pass_snaps', 'team_db']]


def participation_pass_snaps(year: int) -> pd.DataFrame:
    """One participation season's pass-play snaps (cached; data.loaders._cached_season_download's contract: never raises,
    empty when the season is not available)."""
    if int(year) not in PARTICIPATION_SEASONS:
        return pd.DataFrame()
    from data.loaders import _cached_season_download
    return _cached_season_download('pass_snaps', int(year), lambda: _participation_derived(int(year)))


def pass_snap_rows(year: int, as_of_week: int) -> pd.DataFrame:
    """[week, Team, _key, pass_snaps, team_db] for every game before ``as_of_week``: participation when the season has it,
    else the weekly PFF exports. Empty when neither has the season (the features that read it are then inert)."""
    d = participation_pass_snaps(year)
    if d is not None and not d.empty:
        d = d[pd.to_numeric(d['week'], errors='coerce') < int(as_of_week)]
        return d[_COLUMNS].reset_index(drop=True)
    from data.route_participation import weekly_route_rows
    r = weekly_route_rows(int(year), int(as_of_week))
    if r is None or r.empty:
        return pd.DataFrame(columns=_COLUMNS)
    r = r[r['pass_plays'] > 0].rename(columns={'pass_plays': 'pass_snaps'})
    return r[_COLUMNS].reset_index(drop=True)


def pass_snap_trend(rows: pd.DataFrame, min_before: int = 2) -> pd.DataFrame:
    """Per (Team, _key) from ``pass_snap_rows``: his pass-snap share (snaps / team dropbacks) in his latest game (ps_last),
    over the last three (ps_l3, pooled), over every game before the latest (ps_before, pooled) and over all of them (ps_s2d,
    pooled); the latest game's week and how many earlier games. Players with fewer than ``min_before`` earlier games for the
    team are left out."""
    cols = ['Team', '_key', 'last_week', 'ps_last', 'ps_l3', 'ps_before', 'ps_s2d', 'games_before']
    if rows is None or rows.empty:
        return pd.DataFrame(columns=cols)
    d = rows[rows['team_db'] > 0].sort_values('week')
    out = []
    for (team, key), g in d.groupby(['Team', '_key'], sort=False):
        if len(g) < min_before + 1:
            continue
        s, t = g['pass_snaps'].to_numpy(float), g['team_db'].to_numpy(float)
        out.append({'Team': team, '_key': key, 'last_week': int(g['week'].iloc[-1]),
                    'ps_last': s[-1] / t[-1], 'ps_l3': s[-3:].sum() / t[-3:].sum(),
                    'ps_before': s[:-1].sum() / t[:-1].sum(), 'ps_s2d': s.sum() / t.sum(), 'games_before': len(g) - 1})
    return pd.DataFrame(out, columns=cols) if out else pd.DataFrame(columns=cols)


TREND_CLIP = (-0.8, 1.5)


def trend_ratios(trend: pd.DataFrame, team_last_week: dict) -> dict:
    """{(Team, _key): (last, l3)} from ``pass_snap_trend``: last = ps_last / ps_before - 1, l3 = ps_l3 / ps_s2d - 1, each
    clipped to TREND_CLIP. Ratios, so a source's constant level offset cancels (the PFF exports' pass-play share runs about 7%
    above participation's, 2024-2025, correlation 0.98). Only players whose latest game is their team's latest game
    (``team_last_week``: {canonical team: week}); a missing export or a game he sat leaves him out (the terms read 0)."""
    if trend is None or trend.empty:
        return {}
    out = {}
    for team, key, week, ps_last, ps_l3, ps_before, ps_s2d in zip(
            trend['Team'], trend['_key'], trend['last_week'], trend['ps_last'], trend['ps_l3'], trend['ps_before'],
            trend['ps_s2d']):
        if team_last_week.get(team) != week or ps_before <= 0 or ps_s2d <= 0:
            continue
        out[(team, key)] = (float(np.clip(ps_last / ps_before - 1.0, *TREND_CLIP)),
                            float(np.clip(ps_l3 / ps_s2d - 1.0, *TREND_CLIP)))
    return out
