"""
Week-to-week scoring behaviour: how a player's points arrive, not just how
many of them there are.

WHAT THIS BUILDS. Weekly score pools - the resampling material the season
simulator draws from (data/draft_season_sim.py). Bootstrapping real weeks
rather than drawing from a fitted normal is deliberate: weekly fantasy
scoring is right-skewed and fat-tailed - a running back's 40-point game
exists in a way no symmetric distribution around his mean will ever produce
- and it is exactly those weeks that decide matchups.

WHY IT MATTERS. Everything else on the board is a SEASON total. Ceiling,
Floor and Risk look like they describe volatility but they don't - they are
percentiles of a season-long finish drawn from rank uncertainty, so they
answer "how wrong might this projection be", not "what do his weeks look
like". A fantasy season is seventeen head-to-head games, and two players
projected for identical totals are not identical assets if one scores 14
every week and the other alternates 4 and 24.

THIS MODULE USED TO ALSO PUBLISH per-player Start%/Boom%/weekly-spread
columns on the board itself, from a player's own history where he had
enough of it and from same-rank finishers where he didn't. They were built,
shipped, used and cut: the numbers were sound but they didn't change a
single pick, because by the time you are choosing between two players you
have already narrowed to players whose weekly shapes are similar. The
simulator consumes the same weekly history to answer the question those
columns were reaching for - how often this roster actually wins - and it
answers it in wins rather than in percentages a drafter has to translate.
Recover them from git history if a use ever appears.
"""
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
    rank.

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

    return pools
