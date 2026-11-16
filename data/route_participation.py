"""Route participation from the weekly PFF receiving exports (2026-11-16, v2_te_route_rise).

Route participation = routes run / the team's dropbacks in that game (a team's dropbacks = the most pass plays any of its
receivers was on the field for). The board does not read routes anywhere else; this module only answers "did a tight end's
(or receiver's) route participation jump in his latest game", which scripts/study_role_alignment_drift.py found the board lags (2024-2026).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.pff_alignment import DEFAULT_PFF_ROOT, _canonical_team_key, discover_weekly_alignment_exports
from data.utils import clean_name_exact

_NUMERIC = ('routes', 'pass_plays')


def weekly_route_rows(year: int, as_of_week: int, pff_root: str | Path = DEFAULT_PFF_ROOT) -> pd.DataFrame:
    """[week, Team, _key, position, routes, pass_plays, team_db] for every eligible weekly export before ``as_of_week``.
    Empty on any missing / unreadable source (the rule is then inert)."""
    try:
        found = discover_weekly_alignment_exports(year, as_of_week, pff_root)
        archives = found.archives
    except Exception:
        return pd.DataFrame()
    if archives is None or archives.empty:
        return pd.DataFrame()
    rows = []
    for arch in archives.loc[archives['eligible_as_of'].astype(bool)].itertuples(index=False):
        path = getattr(arch, 'summary_path', '')
        week = int(getattr(arch, 'week'))
        if not path or week >= as_of_week:
            continue
        try:
            d = pd.read_csv(path, low_memory=False, usecols=lambda c: c in ('player', 'position', 'team_name') + _NUMERIC)
        except Exception:
            continue
        if not {'player', 'team_name', 'routes', 'pass_plays'} <= set(d.columns):
            continue
        d = d.assign(week=week)
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    d = pd.concat(rows, ignore_index=True)
    d['Team'] = d['team_name'].map(_canonical_team_key)
    d['_key'] = clean_name_exact(d['player'].astype(str))
    for c in _NUMERIC:
        d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0.0)
    d = d.join(d.groupby(['week', 'Team'])['pass_plays'].max().rename('team_db'), on=['week', 'Team'])
    return d[['week', 'Team', '_key', 'position', 'routes', 'pass_plays', 'team_db']]


def latest_game_moves(rows: pd.DataFrame, min_before: int = 2) -> pd.DataFrame:
    """Per (Team, _key): route participation in his latest exported game (rp_last), over his earlier games this season for
    the same team (rp_before, pooled routes / pooled dropbacks), the latest game's week and how many earlier games."""
    if rows is None or rows.empty:
        return pd.DataFrame(columns=['Team', '_key', 'last_week', 'rp_last', 'rp_before', 'games_before'])
    out = []
    for (team, key), g in rows.groupby(['Team', '_key'], sort=False):
        g = g.sort_values('week')
        if len(g) < min_before + 1:
            continue
        last, before = g.iloc[-1], g.iloc[:-1]
        db_before = before['team_db'].sum()
        if last['team_db'] <= 0 or db_before <= 0:
            continue
        out.append({'Team': team, '_key': key, 'last_week': int(last['week']),
                    'rp_last': float(last['routes'] / last['team_db']),
                    'rp_before': float(before['routes'].sum() / db_before), 'games_before': len(before)})
    return pd.DataFrame(out)


def route_rise_factors(result: pd.DataFrame, moves: pd.DataFrame, team_last_week: dict, params: dict,
                       max_factor: float) -> dict:
    """{(Player, Pos, Team): {'factor', 'rp_last', 'rp_before', 'week'}} for the rows of the positions ``params`` lists
    ({pos: (threshold, exponent)}) whose latest game's route participation is threshold+ above his earlier level, and whose
    latest exported game is his team's latest game (a missing weekly export leaves the rule off rather than reading an old
    game as the latest). factor = min((rp_last / rp_before) ** exponent, max_factor)."""
    if result is None or result.empty or moves is None or moves.empty:
        return {}
    index = {(t, k): r for t, k, r in zip(moves['Team'], moves['_key'], moves.itertuples(index=False))}
    keys = clean_name_exact(result['Player'].astype(str))
    out = {}
    for player, pos, team, key in zip(result['Player'], result['Pos'], result['Team'], keys):
        if pos not in params:
            continue
        threshold, exponent = params[pos]
        team_key = _canonical_team_key(team)
        m = index.get((team_key, key))
        if m is None or m.rp_before <= 0 or team_last_week.get(team_key) != m.last_week:
            continue
        if m.rp_last - m.rp_before < threshold:
            continue
        factor = float(min((m.rp_last / m.rp_before) ** exponent, max_factor))
        if not np.isfinite(factor) or factor <= 1.0:
            continue
        out[(player, pos, team)] = {'factor': factor, 'rp_last': m.rp_last, 'rp_before': m.rp_before, 'week': m.last_week}
    return out
