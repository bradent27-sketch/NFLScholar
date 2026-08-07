"""
Week-to-week scoring behaviour: how a player's points arrive, not just how
many of them there are.

WHY THIS EXISTS. Everything else on this board is a SEASON total. Ceiling,
Floor and Risk look like they describe volatility but they don't - they are
percentiles of a season-long finish drawn from rank uncertainty, so they
answer "how wrong might this projection be", not "what do his weeks look
like". Those are different questions and only the second one decides
matchups.

A fantasy season is seventeen head-to-head games. Two players projected for
identical season totals are not identical assets if one scores 14 every week
and the other alternates 4 and 24: against a typical opponent the steady one
wins more weeks, and in a league that pays out on wins that is the whole
ballgame. Nothing in this app could see that difference before.

TWO THINGS ARE BUILT HERE

  Player metrics - Start%, Boom% and weekly spread, computed from a player's
  own week-by-week history where he has enough of it, and from the players
  who historically finished at his projected rank where he doesn't (rookies,
  role changes). Every number is scored under YOUR league's settings, so a
  TE-premium league genuinely reshuffles who counts as consistent.

  Weekly score pools - the resampling material the season simulator draws
  from (data/draft_season_sim.py). Bootstrapping real weeks rather than
  drawing from a fitted normal is deliberate: weekly fantasy scoring is
  right-skewed and fat-tailed - a running back's 40-point game exists in a
  way no symmetric distribution around his mean will ever produce - and it
  is exactly those weeks that decide matchups.
"""
import numpy as np
import pandas as pd
import streamlit as st

from data.draft_board import (
    CURVE_SEASONS, MODERN_SEASON_GAMES, DRAFTABLE_POSITIONS, score_stats,
)

# Ranks either side of a player's own projected rank whose weeks get pooled
# with his. Widening the window trades specificity for sample size; +/-2 puts
# roughly 400 real weeks behind each rank, which is enough for a stable tail
# without smearing the RB1 distribution into the RB8 one.
RANK_WINDOW = 2

# Own-history games needed before a player is described by his OWN weeks
# rather than by the players who finished where he's projected. Two full
# seasons of a starter's workload, near enough - below this the sample says
# more about which weeks he happened to be healthy than about how he scores.
MIN_OWN_GAMES = 20

# A "boom" week is a top-5 finish at the position that week - the kind of
# week that wins a matchup on its own rather than merely contributing.
BOOM_RANK = 5


def _weekly_frame(scoring, latest_season, n_seasons=CURVE_SEASONS):
    """Regular-season weeks, scored under this league's settings."""
    from data.loaders import load_weekly_stats_history
    hist = load_weekly_stats_history()
    if hist is None or hist.empty or 'season' not in hist.columns:
        return pd.DataFrame(), None
    df = hist[pd.to_numeric(hist['week'], errors='coerce').fillna(0) > 0].copy()
    if 'season_type' in df.columns:
        df = df[df['season_type'].astype(str).str.upper().isin(['REG', 'REGULAR'])]
    # Weeks 18+ are regular season in the modern schedule but the fantasy
    # season is over, and week 18 rest-day benchings would otherwise read as
    # a durability problem rather than the scheduling artifact they are.
    df = df[pd.to_numeric(df['week'], errors='coerce') <= MODERN_SEASON_GAMES]
    name_col = 'player_display_name' if 'player_display_name' in df.columns else 'player_name'
    if name_col not in df.columns or 'position' not in df.columns:
        return pd.DataFrame(), None
    seasons = sorted([s for s in df['season'].dropna().unique() if s <= latest_season])[-n_seasons:]
    if not seasons:
        return pd.DataFrame(), None
    df = df[df['season'].isin(seasons)].copy()
    df['_points'] = score_stats(df, scoring)
    df['_pos'] = df['position'].astype(str).str.upper()
    df['_name'] = df[name_col].astype(str).str.strip()
    return df, name_col


@st.cache_data(show_spinner=False)
def build_weekly_pools(scoring, latest_season, n_seasons=CURVE_SEASONS):
    """
    {(position, finish rank): array of weekly scores} for every draftable
    rank, plus the weekly thresholds a score is judged against.

    A rank's pool is every week played by the players who FINISHED within
    RANK_WINDOW of that rank, in any of the sampled seasons. Finish rank
    rather than preseason rank because the question being answered is "what
    do the weeks of an RB5-caliber season look like", and using preseason
    rank would fold in the entirely separate question of how often a
    projection is wrong - which Ceiling and Floor already cover.

    Only games PLAYED are pooled. A missed game is not a zero-point week for
    your team, it's a week you start someone else, and the simulator handles
    absence separately (see data/draft_season_sim.py). Folding zeros in here
    would double-count injuries and make every player look streakier than he
    is.
    """
    df, _ = _weekly_frame(scoring, latest_season, n_seasons)
    if df.empty:
        return {}, {}

    # Season totals decide finish rank; a player needs to have actually
    # played to be ranked, or a one-game cameo lands mid-pack on a per-game
    # basis and pollutes the pool.
    totals = df.groupby(['season', '_pos', '_name'])['_points'].agg(['sum', 'size'])
    totals = totals[totals['size'] >= 4].reset_index()
    totals['rank'] = totals.groupby(['season', '_pos'])['sum'].rank(ascending=False, method='first')

    keyed = df.merge(totals[['season', '_pos', '_name', 'rank']],
                     on=['season', '_pos', '_name'], how='inner')

    pools = {}
    for pos in DRAFTABLE_POSITIONS:
        rows = keyed[keyed['_pos'] == pos]
        if rows.empty:
            continue
        max_rank = int(rows['rank'].max())
        ranks = rows['rank'].to_numpy()
        points = rows['_points'].to_numpy(dtype=float)
        for rank in range(1, min(max_rank, 90) + 1):
            window = (ranks >= rank - RANK_WINDOW) & (ranks <= rank + RANK_WINDOW)
            sample = points[window]
            if len(sample) >= 25:
                pools[(pos, rank)] = sample

    # What a startable week looked like, per position per week: the score of
    # the last player a 12-team league would have started that week. Used to
    # turn a raw point total into "would this have been a usable week".
    thresholds = {}
    starters = {'QB': 12, 'RB': 30, 'WR': 40, 'TE': 12, 'K': 12, 'DST': 12}
    for (season, week, pos), group in df.groupby(['season', 'week', '_pos']):
        if pos not in starters:
            continue
        scores = np.sort(group['_points'].to_numpy(dtype=float))[::-1]
        cut = starters[pos] - 1
        thresholds[(int(season), int(week), pos)] = {
            'startable': float(scores[cut]) if len(scores) > cut else 0.0,
            'boom': float(scores[BOOM_RANK - 1]) if len(scores) >= BOOM_RANK else float('inf'),
        }
    return pools, thresholds


@st.cache_data(show_spinner=False)
def build_consistency(scoring, latest_season, n_seasons=CURVE_SEASONS):
    """
    Per-player weekly behaviour: {name: {...}} with

      start_pct  - share of weeks played that cleared the startable bar at
                   his position. A reliability read: 90% means he was worth
                   starting nearly every week he was active.
      boom_pct   - share of weeks that finished top-5 at his position. The
                   matchup-winning weeks.
      weekly_sd  - standard deviation of his weekly scores.
      weekly_cv  - that spread relative to his own mean, which is the version
                   that compares across positions (a quarterback's 6-point
                   swing and a tight end's are not the same thing).
      games      - how many weeks the numbers rest on.

    Own history only. Players without MIN_OWN_GAMES are absent from this map
    and callers fall back to their projected rank's pool, which is what
    build_rank_consistency provides.
    """
    df, _ = _weekly_frame(scoring, latest_season, n_seasons)
    if df.empty:
        return {}
    _, thresholds = build_weekly_pools(scoring, latest_season, n_seasons)
    if not thresholds:
        return {}

    keys = list(zip(df['season'].astype(int), df['week'].astype(int), df['_pos']))
    startable = np.array([thresholds.get(k, {}).get('startable', np.nan) for k in keys])
    boom = np.array([thresholds.get(k, {}).get('boom', np.nan) for k in keys])
    work = df.assign(_startable=df['_points'].to_numpy() >= startable,
                     _boom=df['_points'].to_numpy() >= boom)

    out = {}
    work = work[work['_pos'].isin(DRAFTABLE_POSITIONS)]
    for (name, pos), group in work.groupby(['_name', '_pos']):
        if len(group) < MIN_OWN_GAMES:
            continue
        points = group['_points'].to_numpy(dtype=float)
        mean = float(points.mean())
        out[(name, pos)] = {
            'start_pct': round(float(group['_startable'].mean()) * 100, 1),
            'boom_pct': round(float(group['_boom'].mean()) * 100, 1),
            'weekly_sd': round(float(points.std(ddof=1)) if len(points) > 1 else 0.0, 1),
            'weekly_cv': round(float(points.std(ddof=1) / mean) if mean > 0 and len(points) > 1 else 0.0, 3),
            'games': int(len(group)),
            'basis': 'own',
        }
    return out


@st.cache_data(show_spinner=False)
def build_rank_consistency(scoring, latest_season, n_seasons=CURVE_SEASONS):
    """
    The same three numbers computed for a RANK rather than a player, so a
    rookie projected as the RB18 still gets a consistency read - the one
    belonging to how RB18 seasons have actually behaved.
    """
    df, _ = _weekly_frame(scoring, latest_season, n_seasons)
    if df.empty:
        return {}
    _, thresholds = build_weekly_pools(scoring, latest_season, n_seasons)
    if not thresholds:
        return {}

    totals = df.groupby(['season', '_pos', '_name'])['_points'].agg(['sum', 'size'])
    totals = totals[totals['size'] >= 4].reset_index()
    totals['rank'] = totals.groupby(['season', '_pos'])['sum'].rank(ascending=False, method='first')
    keyed = df.merge(totals[['season', '_pos', '_name', 'rank']],
                     on=['season', '_pos', '_name'], how='inner')

    keys = list(zip(keyed['season'].astype(int), keyed['week'].astype(int), keyed['_pos']))
    startable = np.array([thresholds.get(k, {}).get('startable', np.nan) for k in keys])
    boom = np.array([thresholds.get(k, {}).get('boom', np.nan) for k in keys])
    keyed = keyed.assign(_startable=keyed['_points'].to_numpy() >= startable,
                         _boom=keyed['_points'].to_numpy() >= boom)

    out = {}
    for pos in DRAFTABLE_POSITIONS:
        rows = keyed[keyed['_pos'] == pos]
        if rows.empty:
            continue
        ranks = rows['rank'].to_numpy()
        for rank in range(1, min(int(rows['rank'].max()), 90) + 1):
            window = rows[(ranks >= rank - RANK_WINDOW) & (ranks <= rank + RANK_WINDOW)]
            if len(window) < 25:
                continue
            points = window['_points'].to_numpy(dtype=float)
            mean = float(points.mean())
            out[(pos, rank)] = {
                'start_pct': round(float(window['_startable'].mean()) * 100, 1),
                'boom_pct': round(float(window['_boom'].mean()) * 100, 1),
                'weekly_sd': round(float(points.std(ddof=1)), 1),
                'weekly_cv': round(float(points.std(ddof=1) / mean) if mean > 0 else 0.0, 3),
                'games': int(len(window)),
                'basis': 'rank',
            }
    return out


def attach_consistency(board, settings):
    """
    Add Start %, Boom % and Wk SD to the board.

    Own history where a player has enough of it, his projected rank's history
    where he doesn't, and the source is recorded per row rather than hidden -
    "this rookie scores like a typical RB18" and "this veteran has scored
    like this for three years" are different claims and shouldn't look
    identical on the board.
    """
    if board is None or board.empty or 'Proj Pts' not in board.columns:
        return board

    scoring = settings['scoring']
    season = settings.get('baseline_season', 2025)
    own = build_consistency(scoring, season)
    by_rank = build_rank_consistency(scoring, season)
    if not own and not by_rank:
        return board

    out = board.copy()
    # Projected finish rank within position, which is the key into the rank
    # pools - NOT consensus rank, since the pools are keyed on where players
    # actually finished.
    pos_upper = out['Pos'].astype(str).str.upper()
    proj_rank = out.groupby(pos_upper)['Proj Pts'].rank(ascending=False, method='first')

    start, boom, spread, basis = [], [], [], []
    for name, pos, rank in zip(out['Player'].astype(str), pos_upper, proj_rank):
        record = own.get((name, pos))
        if record is None and pd.notna(rank):
            record = by_rank.get((pos, int(rank)))
        if record is None:
            start.append(np.nan); boom.append(np.nan)
            spread.append(np.nan); basis.append(None)
            continue
        start.append(record['start_pct'])
        boom.append(record['boom_pct'])
        spread.append(record['weekly_sd'])
        basis.append(record['basis'])

    out['Start %'] = start
    out['Boom %'] = boom
    out['Wk SD'] = spread
    out['consistency_basis'] = basis
    return out
