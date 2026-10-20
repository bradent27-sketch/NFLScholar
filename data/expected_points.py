"""Expected fantasy points per player-game (nflverse ff_opportunity) and the "luck" derived from them.

ff_opportunity gives, for every player-game, the fantasy points his touches were worth given their context (target depth,
field position, down and distance ...) next to the points he scored. Actual minus expected is "luck": a receiver whose
results ran ahead of his usage tends to give some of it back, one who ran behind it tends to rebound. On the shipped board the
miss (actual - projected) moves against earlier-games luck by about -0.18 points per point for WR and -0.16 for TE
(2021-2025, every season for WR); see docs/weekly_projections_methodology.md, 2026-10-19 and 2026-10-20, and backlog N1.

This module only supplies the data: a cached weekly table per season and a per-player luck table as of a target week.
The adjustment itself is data.weekly_projections.apply_expected_points_luck behind 'v2_expected_points_luck'.

Caveat: nflverse's expectation model is trained on all seasons, so its numbers for a past game are in-sample for the
EXPECTATION model (not for the player's later results). Weeks of the live season are the clean check.
"""
import os

import pandas as pd

from data.historical_availability import CACHE_DIR, _partial_season_cache_is_stale
from data.utils import clean_name_exact

FF_COLUMNS = ('season', 'week', 'posteam', 'player_id', 'full_name', 'position', 'receptions', 'receptions_exp',
              'total_fantasy_points', 'total_fantasy_points_exp')


def _cache_path(season):
    return os.path.join(CACHE_DIR, f'ff_opportunity_{int(season)}.parquet')


def load_ff_opportunity_weekly(season):
    """Weekly expected-points rows for one season, cached to data/cache/ff_opportunity_{season}.parquet. A partial
    season's cache is refreshed when older than historical_availability.CACHE_MAX_AGE_HOURS (a failed refresh keeps the
    cached copy). Never raises: an unreachable source with no cache returns an empty frame."""
    path = _cache_path(season)
    cached = None
    if os.path.exists(path):
        try:
            cached = pd.read_parquet(path)
        except Exception:
            cached = None
        if cached is not None and not _partial_season_cache_is_stale(cached, path):
            return cached
    try:
        import nflreadpy
        df = nflreadpy.load_ff_opportunity(seasons=[int(season)], stat_type='weekly').to_pandas()
    except Exception:
        return cached if cached is not None else pd.DataFrame()
    if df is None or df.empty:
        return cached if cached is not None else pd.DataFrame()
    df = df[[c for c in FF_COLUMNS if c in df.columns]].copy()
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_parquet(path, index=False)
    except Exception:
        pass
    return df


def _reception_weight(scoring_mode):
    text = str(scoring_mode or '')
    return 1.0 if 'Full' in text else (0.5 if 'Half' in text else 0.0)


def luck_per_game(ff, as_of_week, scoring_mode='Full PPR'):
    """DataFrame indexed by clean player name with ``games`` (earlier games with an expected-points row this season) and
    ``luck_pg`` (mean of actual minus expected points over them), using regular-season weeks BEFORE ``as_of_week`` only.
    The feed scores a reception as a full point; under Half-PPR or Standard that part of the luck is scaled down to the
    mode's reception value. A name that appears twice within one week of the feed is ambiguous and is left out."""
    cols = {'week', 'full_name', 'total_fantasy_points', 'total_fantasy_points_exp'}
    if ff is None or len(ff) == 0 or not cols.issubset(ff.columns) or as_of_week is None:
        return pd.DataFrame(columns=['games', 'luck_pg'])
    f = ff.copy()
    f['week'] = pd.to_numeric(f['week'], errors='coerce')
    f = f[f['week'].between(1, 18) & (f['week'] < float(as_of_week))]
    if f.empty:
        return pd.DataFrame(columns=['games', 'luck_pg'])
    f['key'] = clean_name_exact(f['full_name'].astype(str))
    luck = pd.to_numeric(f['total_fantasy_points'], errors='coerce') - pd.to_numeric(f['total_fantasy_points_exp'], errors='coerce')
    rec_gap = 1.0 - _reception_weight(scoring_mode)
    if rec_gap and {'receptions', 'receptions_exp'}.issubset(f.columns):
        luck = luck - rec_gap * (pd.to_numeric(f['receptions'], errors='coerce') - pd.to_numeric(f['receptions_exp'], errors='coerce')).fillna(0.0)
    f['luck'] = luck
    f = f.dropna(subset=['luck'])
    dup = f.duplicated(['key', 'week'], keep=False)
    ambiguous = set(f.loc[dup, 'key'])
    f = f[~f['key'].isin(ambiguous)]
    g = f.groupby('key')['luck'].agg(['size', 'mean'])
    return g.rename(columns={'size': 'games', 'mean': 'luck_pg'})


def expected_points_luck_table(year, as_of_week, scoring_mode='Full PPR'):
    """The luck table for a build of ``year`` as of ``as_of_week`` (empty when the feed is unavailable)."""
    return luck_per_game(load_ff_opportunity_weekly(year), as_of_week, scoring_mode)
