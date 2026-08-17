"""
Vegas game lines, from a source that is free, uncapped, and needs no key.

WHY THIS SOURCE. Every sportsbook and odds aggregator is either paywalled,
rate-limited, or behind bot detection. nflverse mirrors the closing spread
and total for every NFL game into a plain CSV on GitHub, updated through the
season, going back to 1999. No key, no request budget, no scraping, no terms
to violate - and it is the same market number the books are posting.

That matters here specifically because the existing Odds API key is capped at
1,000 requests a month, which is a real constraint during a season. This
costs nothing and can be pulled as often as you like.

WHAT IT IS GOOD FOR, AND WHAT IT IS NOT. The market's implied points for a
team is a genuinely informative number - it is what the money says an offense
will score, and a drafter reading a board wants to know that one player's
offense is priced at 27 points a game and another's at 19.

It is NOT, on this evidence, a way to improve player projections, and the
measurement is in this module rather than in a commit message because the
idea is obvious enough that someone will try it again. See
TEAM_MULTIPLIER_MEASUREMENT below.

PRESEASON COVERAGE IS PARTIAL. Books post the first few weeks as lookahead
lines and fill the rest in as the season runs - in early August 2026 that was
52 of 272 games, about three per team. So a team's implied scoring here is an
average over a handful of posted games, not a full-season figure, and it
carries opponent bias that a full slate would average out. `games_posted`
travels with every row so a thin estimate is visible rather than implied.
"""
import io

import pandas as pd
import requests
import streamlit as st

# nflverse's game file. Raw GitHub rather than the nflverse-data release
# assets that nflreadpy uses: the release assets live on github.com proper,
# which some networks block while still allowing raw.githubusercontent.com -
# and this is the same data either way.
GAMES_URLS = [
    'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv',
    'https://raw.githubusercontent.com/nflverse/nfldata/main/data/games.csv',
]
FETCH_TTL = 3600
FETCH_TIMEOUT = 30

# A REGULAR-SEASON slate is 17 games per team. Used to turn implied points
# per game into an implied season total.
SEASON_GAMES = 17


# MEASURED, AND THE ANSWER WAS NO.
#
# The obvious use for this data is to scale a player's projection by his
# team's implied scoring - a receiver on an offense the market prices at 27
# points a game should beat the same receiver on a 19-point offense. It is a
# reasonable idea and it does not survive a backtest.
#
# Test: project season N from season N-1 per-game fantasy production, with and
# without multiplying by (team implied points / league average) ** alpha.
# 748 player-seasons across 2023-2025, players with 8+ games in both years.
#
#   alpha    MAE (wks 1-3 lines)    r        MAE (full-season lines)   r
#   0.00           2.830          0.7850            2.830           0.7850
#   0.25           2.839          0.7874            2.819           0.7915
#   0.50           2.870          0.7874            2.826           0.7955
#   1.00           3.010          0.7807            2.925           0.7967
#
# The preseason-available signal (weeks 1-3, the only lines posted in August)
# makes the projection WORSE at every strength. Full-season lines - which
# require hindsight and are useless for a draft - buy a MAE improvement of
# 0.011 points per game, which is nothing.
#
# By position, at alpha 0.5 on the preseason signal, only running backs
# improved (MAE 3.290 -> 3.220, n=187) and the other three degraded (QB 3.202
# -> 3.326, TE 2.051 -> 2.109, WR 2.900 -> 2.976). One position out of four
# on a modest sample is what noise looks like, not a finding.
#
# WHY IT FAILS, most likely: a player's fantasy output is his SHARE of an
# offense times that offense's output, and the share moves far more between
# seasons than the team total does. Scaling by team while holding share fixed
# corrects the smaller term and adds variance to the larger one. It is also
# double-counting - a player's own usage history already encodes the quality
# of the offense he plays in.
#
# So the implied total is surfaced as INFORMATION on the board, and nothing
# in the projection path multiplies by it.
TEAM_MULTIPLIER_MEASUREMENT = 'see module docstring; not implemented'


def _fetch_csv(urls):
    """First URL that answers, as a DataFrame. Returns (df, error)."""
    errors = []
    for url in urls:
        try:
            resp = requests.get(url, timeout=FETCH_TIMEOUT)
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            continue
        if resp.status_code == 200:
            try:
                return pd.read_csv(io.StringIO(resp.text), low_memory=False), None
            except Exception as exc:
                errors.append(f"{url}: parsed as CSV badly ({exc})")
                continue
        errors.append(f"{url}: HTTP {resp.status_code}")
    return pd.DataFrame(), "Couldn't fetch Vegas game lines. " + "; ".join(errors)


@st.cache_data(ttl=FETCH_TTL, show_spinner=False)
def fetch_game_lines(season):
    """
    Regular-season games for one season, with whatever lines are posted.

    Returns (frame, meta). The frame is one row per GAME; use
    implied_team_points for the per-team view.
    """
    games, err = _fetch_csv(GAMES_URLS)
    if err:
        return pd.DataFrame(), {'error': err}
    if 'season' not in games.columns:
        return pd.DataFrame(), {'error': "Game file has no 'season' column."}

    season_games = games[(games['season'] == int(season))
                         & (games['game_type'].astype(str).str.upper() == 'REG')].copy()
    if season_games.empty:
        return pd.DataFrame(), {'error': f"No {season} regular-season games in the file."}

    for col in ('spread_line', 'total_line'):
        if col not in season_games.columns:
            return pd.DataFrame(), {'error': f"Game file has no '{col}' column."}
        season_games[col] = pd.to_numeric(season_games[col], errors='coerce')

    posted = int(season_games['total_line'].notna().sum())
    return season_games, {
        'error': None,
        'games': int(len(season_games)),
        'posted': posted,
        'weeks_posted': sorted(
            int(w) for w in season_games.loc[season_games['total_line'].notna(), 'week']
            .dropna().unique()),
    }


def implied_team_points(games):
    """
    One row per team-game with the market's implied points for that team.

    THE SIGN CONVENTION IS THE ONLY THING THAT CAN GO WRONG HERE, and getting
    it backwards would invert every number while still looking plausible.
    nflverse's `spread_line` is positive when the HOME team is favored, so:

        home = total/2 + spread/2      away = total/2 - spread/2

    Verified against a real row: NE at SEA, total 44.5, spread 3.5, and Seattle
    the home favorite at -192 on the moneyline. That gives SEA 24.0 and NE
    20.5 - summing to the total, with the favorite higher. Both checks have to
    pass, and only the correct convention passes the second one.
    """
    if games is None or games.empty:
        return pd.DataFrame(columns=['season', 'week', 'team', 'opponent', 'home',
                                     'implied_points', 'implied_allowed', 'total_line'])
    total = pd.to_numeric(games['total_line'], errors='coerce')
    spread = pd.to_numeric(games['spread_line'], errors='coerce')
    home_points = total / 2 + spread / 2
    away_points = total / 2 - spread / 2

    home = pd.DataFrame({
        'season': games['season'], 'week': games['week'],
        'team': games['home_team'], 'opponent': games['away_team'], 'home': 1,
        'implied_points': home_points, 'implied_allowed': away_points,
        'total_line': total,
    })
    away = pd.DataFrame({
        'season': games['season'], 'week': games['week'],
        'team': games['away_team'], 'opponent': games['home_team'], 'home': 0,
        'implied_points': away_points, 'implied_allowed': home_points,
        'total_line': total,
    })
    both = pd.concat([home, away], ignore_index=True)
    return both.dropna(subset=['implied_points']).reset_index(drop=True)


def team_scoring_environment(season):
    """
    Per-team market scoring view: implied points per game, and how thin the
    estimate is.

    Returns (frame, meta) with columns team / implied_ppg / implied_allowed_ppg
    / games_posted / implied_season_points / vs_league.

    `vs_league` is the ratio to the league average, which is the readable form
    - 1.14 means the market prices this offense 14% above average. It is
    displayed, never multiplied into a projection (see the module docstring).
    """
    games, meta = fetch_game_lines(season)
    if meta.get('error'):
        return pd.DataFrame(), meta

    per_game = implied_team_points(games)
    if per_game.empty:
        return pd.DataFrame(), {**meta, 'error': f"No {season} games have lines posted yet."}

    grouped = per_game.groupby('team').agg(
        implied_ppg=('implied_points', 'mean'),
        implied_allowed_ppg=('implied_allowed', 'mean'),
        games_posted=('implied_points', 'size'),
    ).reset_index()
    grouped['implied_season_points'] = (grouped['implied_ppg'] * SEASON_GAMES).round(1)
    league = grouped['implied_ppg'].mean()
    grouped['vs_league'] = (grouped['implied_ppg'] / league).round(3) if league else 1.0
    grouped['implied_ppg'] = grouped['implied_ppg'].round(2)
    grouped['implied_allowed_ppg'] = grouped['implied_allowed_ppg'].round(2)
    return grouped.sort_values('implied_ppg', ascending=False).reset_index(drop=True), {
        **meta, 'teams': int(len(grouped)), 'league_avg_ppg': round(float(league), 2),
    }


SCORING_MATCHUP_CLIP = (0.75, 1.3)
# Games-count shrinkage constant for the offense/defense baselines below -
# same role as data.weekly_projections.STAT_K, just one constant instead of
# a per-stat dict since there's only one thing being shrunk here (points).
# A team with 3 posted games should still lean heavily on the league
# average; one with 10 should mostly trust its own measured number.
BASELINE_SHRINK_K = 4


def _shrunk_baseline(measured, games_posted, league_avg):
    """weight = games / (games + K); blends `measured` toward `league_avg`
    as `games_posted` grows - the same shrinkage pattern
    data.weekly_projections._blended_rate uses for player rates, applied
    here to a team's own scoring/allowed level."""
    if league_avg is None or pd.isna(league_avg):
        return measured
    if measured is None or pd.isna(measured) or not games_posted:
        return league_avg
    w = games_posted / (games_posted + BASELINE_SHRINK_K)
    return w * measured + (1 - w) * league_avg


def estimate_full_season_scoring(season):
    """
    A fuller-season team-scoring estimate that fills in the games with NO
    posted Vegas line using each team's own measured scoring level against
    that specific week's SCHEDULED opponent's measured defense - not just
    averaging whatever handful of games happen to have a market number.

    WHY THIS EXISTS. team_scoring_environment's implied PPG is honest but
    can be a thin, biased slice - in early August 2026 that was 3 games per
    team, and which 3 is not representative of the other 14 (see that
    function's own docstring). A team that happened to draw two soft
    defenses in its first three posted games reads as a much better offense
    than a full slate would show, and there is no way to tell from the
    number alone. This fills the REST of the known schedule (every
    matchup is set well before every line is posted) with a matchup-
    adjusted estimate instead of silently omitting those games from the
    average.

    THE METHOD, deliberately the same shape as every other matchup
    adjustment in this app (data.transforms.build_stat_allowed_matrix,
    data.weekly_projections' opponent-allowed multiplier): a rate times an
    opponent-allowed ratio versus league average, clipped so one extreme
    defense can't blow the estimate out:

        estimated_points(team, week) =
            offense_baseline[team] * (defense_baseline[opponent] / league_avg_allowed)

    Both baselines are the team's own posted-game average (from
    team_scoring_environment), SHRUNK toward the league average by how many
    games are actually posted (`_shrunk_baseline`) - a team with one posted
    game leans almost entirely on the league average defense/offense level
    rather than treating one data point as its whole identity.

    For a game that DOES have a posted line, the market's own implied
    points are used as-is - the model estimate never overrides a real
    number, only fills the gap where one doesn't exist yet.

    Returns (frame, meta). frame has one row per team: `posted_ppg` (market-
    only average, same as team_scoring_environment's implied_ppg),
    `games_posted`, `full_season_ppg` (blends real posted-line games with
    the matchup estimate for the rest, across every scheduled game), and
    `full_season_points` (that rate x 17). Never feeds a player
    projection - same INFORMATION-ONLY convention as Vegas PPG, for the
    same measured reason (module docstring).
    """
    games, meta = fetch_game_lines(season)
    if meta.get('error'):
        return pd.DataFrame(), meta
    per_game = implied_team_points(games)
    if per_game.empty:
        return pd.DataFrame(), {**meta, 'error': f"No {season} games have lines posted yet."}

    env = per_game.groupby('team').agg(
        posted_ppg=('implied_points', 'mean'),
        posted_allowed_ppg=('implied_allowed', 'mean'),
        games_posted=('implied_points', 'size'),
    )
    league_avg_ppg = float(env['posted_ppg'].mean())
    league_avg_allowed = float(env['posted_allowed_ppg'].mean())
    env['off_baseline'] = env.apply(
        lambda r: _shrunk_baseline(r['posted_ppg'], r['games_posted'], league_avg_ppg), axis=1)
    env['def_baseline'] = env.apply(
        lambda r: _shrunk_baseline(r['posted_allowed_ppg'], r['games_posted'], league_avg_allowed), axis=1)

    lo, hi = SCORING_MATCHUP_CLIP
    posted_teams = set(games.loc[games['total_line'].notna(), 'home_team']) | set(
        games.loc[games['total_line'].notna(), 'away_team'])

    def _est(offense, defense):
        off_base = env['off_baseline'].get(offense, league_avg_ppg)
        def_base = env['def_baseline'].get(defense, league_avg_allowed)
        if league_avg_allowed <= 0:
            return off_base
        mult = min(max(def_base / league_avg_allowed, lo), hi)
        return off_base * mult

    rows = []
    for _, g in games.iterrows():
        has_line = pd.notna(g.get('total_line'))
        home, away = g['home_team'], g['away_team']
        home_pts = None
        away_pts = None
        if has_line:
            total, spread = float(g['total_line']), float(g['spread_line'])
            home_pts = total / 2 + spread / 2
            away_pts = total / 2 - spread / 2
        else:
            home_pts = _est(home, away)
            away_pts = _est(away, home)
        rows.append({'team': home, 'points': home_pts, 'posted': has_line})
        rows.append({'team': away, 'points': away_pts, 'posted': has_line})

    full = pd.DataFrame(rows)
    grouped = full.groupby('team').agg(
        full_season_ppg=('points', 'mean'), games=('points', 'size')).reset_index()
    grouped['full_season_points'] = (grouped['full_season_ppg'] * SEASON_GAMES).round(1)
    grouped['full_season_ppg'] = grouped['full_season_ppg'].round(2)
    grouped = grouped.merge(
        env[['posted_ppg', 'games_posted']].reset_index().rename(columns={'posted_ppg': 'posted_ppg'}),
        on='team', how='left')
    grouped['posted_ppg'] = grouped['posted_ppg'].round(2)
    grouped['games_posted'] = grouped['games_posted'].fillna(0).astype(int)
    grouped['modeled_games'] = grouped['games'] - grouped['games_posted']
    return grouped.sort_values('full_season_ppg', ascending=False).reset_index(drop=True), {
        **meta, 'teams': int(len(grouped)), 'league_avg_ppg': round(league_avg_ppg, 2),
    }


def attach_team_environment(board, environment):
    """
    Add the market's implied scoring for each player's team to the board.

    Adds `Vegas PPG` (implied points per game for his offense) and
    `Vegas O Rank` (1 = best-priced offense). INFORMATION ONLY - nothing
    downstream multiplies a projection by it, deliberately, because the
    backtest in this module's docstring says doing so makes projections worse.

    Team codes come from the same nflverse vocabulary the rest of this app
    uses, so no standardization is needed - but it goes through
    standardize_team anyway, because a silent join failure here would show as
    an empty column rather than as an error.
    """
    if board is None or board.empty or environment is None or environment.empty:
        return board
    from data.odds_sources import standardize_team

    env = environment.copy()
    env['_team'] = env['team'].map(standardize_team)
    env = env[env['_team'].ne('')]
    if env.empty:
        return board
    env['rank'] = env['implied_ppg'].rank(ascending=False, method='min').astype(int)

    out = board.copy()
    keys = out['Team'].map(standardize_team)
    out['Vegas PPG'] = keys.map(dict(zip(env['_team'], env['implied_ppg'])))
    out['Vegas O Rank'] = keys.map(dict(zip(env['_team'], env['rank']))).astype('Int64')
    return out
