"""
The draft valuation engine: turns a consensus ranking plus your league's
settings into projected points, value over replacement, tiers, and - the
part that actually decides picks - the probability a player survives to
your next pick.

WHY NOT JUST RANK BY PROJECTED POINTS: because a ranking answers the wrong
question. On the clock you are not asking "who is best?", you are asking
"who will still be here in 22 picks?" A player 4 points better who is
certain to last another round is a worse pick right now than a player 4
points worse who will not. Everything below exists to make that comparison
computable, and it is the single biggest gap between this and a plain
sorted cheat sheet.

THE PROJECTION MODEL, PLAINLY STATED
Expert consensus tells us the ORDER players are expected to finish in. It
says nothing about how many points that order is worth - and the answer
depends entirely on your league (a TE-premium 14-team superflex values the
same ordering completely differently than a 10-team standard league). So:

  1. From this app's own local weekly data, compute what the player who
     ACTUALLY FINISHED as the Nth-best at each position scored, under YOUR
     scoring settings, averaged over recent seasons. That's a value curve
     per position.
  2. A player expected to finish 5th does not reliably finish 5th. Rather
     than reading his value straight off curve[5], spread him across the
     finishes he plausibly lands on and take the mean. The spread is taken
     from how much the experts themselves disagree about him (FantasyPros
     publishes each player's best/worst/stdev rank), so a player nobody
     agrees on gets a genuinely wider outcome distribution than a consensus
     lock.
  3. That same distribution, read at its tails instead of its mean, is what
     the Ceiling and Floor columns are. They are not a separate model or a
     vibe - they are the 85th and 15th percentile of the same curve, which
     is why they stay internally consistent with the projection.

Step 2 matters more than it looks: reading value straight off the realized
curve systematically overvalues the top of every position, because the
player who finished #1 is by definition whoever got luckiest, and nobody is
reliably that guy in advance. Smearing across plausible finishes removes
that bias instead of baking it into every pick.
"""
import math

import numpy as np
import pandas as pd
import streamlit as st

from data.draft_sources import vectorized_normal_sf, normal_sf

# Positions this board drafts. IDP is deliberately out of scope here (same
# call the rest of the app already makes for its fantasy-facing tabs), but
# K and DST are in, because leaving them out means the last two rounds of
# every real draft happen off-board.
DRAFTABLE_POSITIONS = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
FLEX_ELIGIBLE = ['RB', 'WR', 'TE']
SUPERFLEX_ELIGIBLE = ['QB', 'RB', 'WR', 'TE']

DEFAULT_SCORING = {
    'pass_yd': 0.04, 'pass_td': 4.0, 'pass_int': -2.0, 'pass_2pt': 2.0,
    'rush_yd': 0.1, 'rush_td': 6.0, 'rush_att': 0.0, 'rush_2pt': 2.0,
    'rec': 1.0, 'rec_yd': 0.1, 'rec_td': 6.0, 'rec_2pt': 2.0,
    # A separate per-reception bonus for TEs only. This is a real and
    # increasingly common setting, and it is not cosmetic - it changes
    # where the TE cliff falls, which changes replacement level, which
    # changes every position's value, not just TE's.
    'te_premium': 0.0,
    'fumble_lost': -2.0,
    'fg_0_39': 3.0, 'fg_40_49': 4.0, 'fg_50_plus': 5.0, 'pat': 1.0,
    # Per-GAME yardage milestone bonuses. All default to 0 (off).
    #
    # These are the one scoring rule that cannot be computed from season
    # totals, and getting that wrong is not a rounding error: a back with
    # 1,200 yards spread evenly over 17 games earns no 100-yard bonuses at
    # all, while a back with the same 1,200 yards concentrated in eight big
    # games earns eight of them. Season totals are identical; scoring is
    # not. Everything downstream of score_stats therefore has to feed it
    # WEEKLY rows and sum afterwards - see build_positional_value_curves.
    'bonus_rush_100': 0.0, 'bonus_rush_150': 0.0, 'bonus_rush_200': 0.0, 'bonus_rush_250': 0.0,
    'bonus_rec_100': 0.0, 'bonus_rec_150': 0.0, 'bonus_rec_200': 0.0, 'bonus_rec_250': 0.0,
    'bonus_pass_300': 0.0, 'bonus_pass_400': 0.0, 'bonus_pass_500': 0.0, 'bonus_pass_600': 0.0,
    # 'cumulative' awards every threshold cleared (a 210-yard rushing game
    # triggers the 100, 150 AND 200 bonuses) - this is how Sleeper and ESPN
    # apply stacked milestone bonuses when you configure more than one.
    # 'highest' awards only the largest threshold cleared. Leagues genuinely
    # differ, and on a 250-yard game the two conventions can differ by more
    # than a touchdown, so it's a setting rather than an assumption.
    'bonus_mode': 'cumulative',
    # Team defense. Scored from real team-week defensive stats plus points
    # allowed (see build_dst_value_curve), not approximated - a DST slot is
    # a real roster slot and leaving it unvalued means the last two rounds
    # of every draft happen with no guidance at all.
    'dst_sack': 1.0, 'dst_int': 2.0, 'dst_fumble_rec': 2.0,
    'dst_td': 6.0, 'dst_safety': 2.0,
}

# Points-allowed buckets for DST scoring. Left as a constant rather than a
# per-league input because essentially every platform ships this same
# ladder as its default and it's rarely customized - unlike the per-event
# values above, which genuinely do vary league to league.
DST_POINTS_ALLOWED_TIERS = [
    (0, 10.0), (6, 7.0), (13, 4.0), (20, 1.0), (27, 0.0), (34, -1.0),
]
DST_POINTS_ALLOWED_WORST = -4.0

DEFAULT_ROSTER = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'SUPERFLEX': 0,
                  'K': 1, 'DST': 1, 'BENCH': 6}

# How many seasons of realized finishes feed the positional value curves.
# Long enough to average out one weird year, short enough that the curves
# reflect the current scoring environment rather than a decade-old one -
# league-wide passing rates and the resulting positional distributions have
# moved enough that 2015 curves would actively mislead.
CURVE_SEASONS = 5

# Games in a modern season, used to put pre-2021 (16-game) seasons on the
# same footing as current ones when building the curves.
MODERN_SEASON_GAMES = 17

# The games basis every projected stat line is expressed on. Lives here
# rather than in data.draft_projections because draft_projections already
# imports from this module, and both the projection side and the valuation
# side need the same number to agree.
#
# 17.0 WAS STRUCTURALLY IMPOSSIBLE, before injuries entered the argument.
# This app scores weeks 1-17, and since 2021 a team plays 17 games across
# EIGHTEEN weeks - so weeks 1-17 hold at most SIXTEEN of them for a normal
# player. Checked against every season in local history: the most games any
# player recorded inside weeks 1-17 is 16 in 2021-24, and the 17s in 2019 and
# 2025 are mid-season trades collecting two teams' bye weeks. Projecting on a
# 17-game basis inflated every stat line by ~6% for nobody's benefit.
PACE_GAMES = 16.0

# Expected games from a player who holds a starting role, per position,
# MEASURED and then rounded UP.
#
# Cohort: last season's top-24 at the position - the players actually being
# drafted as starters - kept only if they held a starting role the following
# season, judged by per-game usage in the games they played (QB 20+ pass
# attempts, RB 10+ touches, WR 5+ targets, TE 3.5+ targets). That filter is
# the point: an injured starter keeps starter usage in the games he DID play,
# while a demoted one doesn't, so this separates "hurt" from "lost the job".
# Measured over 2022-25 only, since 2019-20 were 16-game seasons.
#
#         n    mean    95% CI          median   rounded up
#   QB   85   12.99   [12.24, 13.75]    15.0        13
#   RB   87   13.85   [13.20, 14.47]    15.0        14
#   WR   95   13.56   [12.94, 14.16]    15.0        14
#   TE   74   13.51   [12.88, 14.11]    14.0        14
#
# THE MEDIAN IS 15 AND THE MEAN IS 13.5, which is the whole shape of this:
# the typical starter misses one game, and a minority who lose half a season
# drag the average down. Drafting off the mean would be drafting out of fear
# of injury, which is not how anyone should pick. The rounded-up mean lands
# between the two and is the number used.
#
# QUARTERBACK IS 14 RATHER THAN ITS OWN ROUNDED-UP 13, which is the one
# number here not taken straight from the table. Two reasons, and they pull
# the same way:
#
#   The QB measurement is contaminated downward in a way the usage filter
#   cannot fix. A quarterback benched in week 9 after a bad run still shows
#   20+ attempts per game in the games he started, so he reads as an
#   available starter who missed eight. That is a lost job, not a lost
#   season, and it is why QBs measured LOWER (12.99) than backs and
#   receivers despite being the more durable position.
#
#   Holding all four positions to one number keeps the games assumption from
#   becoming a positional thumb on the scale. Any per-position difference
#   here shifts value between positions, and that difference should be
#   earned by a measurement worth trusting - which this one isn't.
#
# An earlier version shipped 16 here, on the argument that a starting QB
# should simply be assumed to play. It was measured and dropped: it inflated
# QB surplus about 8% and moved quarterbacks 0-3 board slots, while pushing
# projected passing yards to 1.29x the actual league total against ~1.04 for
# every skill stat. Most of that inflation cancelled in VORP - replacement is
# computed from the same inflated pool - which is exactly why the aggregate
# number looked far worse than the draft-value effect actually was.
EXPECTED_GAMES = {'QB': 14.0, 'RB': 14.0, 'WR': 14.0, 'TE': 14.0}
# Kickers and defenses are not injury-priced here: a kicker is replaced by
# another kicker and a DST is the whole unit, so neither carries the
# individual availability risk this table is about.
DEFAULT_EXPECTED_GAMES = 16.0


def expected_games(position):
    """Games a starter at this position is projected to play, weeks 1-17."""
    return EXPECTED_GAMES.get(str(position).upper(), DEFAULT_EXPECTED_GAMES)


# The raw local weekly exports and the canonical names data.loaders renames
# them to are NOT the same vocabulary - the on-disk nflverse files call rush
# attempts 'carries' and pass attempts 'attempts', while load_year_data
# hands the rest of the app 'rushing_attempts'. This module reads BOTH (the
# raw history files for the value curves, canonicalized frames elsewhere),
# so every lookup goes through an alias list. Without this, points-per-carry
# scoring silently contributed zero for every player.
_STAT_ALIASES = {
    'rushing_attempts': ['rushing_attempts', 'carries'],
    'passing_attempts': ['passing_attempts', 'attempts'],
    'passing_interceptions': ['passing_interceptions', 'interceptions'],
}


def _col(df, name):
    """Column if present (under any known alias), else a zero series - the
    local weekly exports vary in which optional stat columns they carry
    across seasons, and in what they call the ones they do."""
    for candidate in _STAT_ALIASES.get(name, [name]):
        if candidate in df.columns:
            return pd.to_numeric(df[candidate], errors='coerce').fillna(0)
    return pd.Series(0.0, index=df.index)


def score_stats(df, scoring, position_col='position'):
    """
    Fantasy points for a frame of (already-aggregated or per-week) stat rows
    under arbitrary league scoring.

    Kept separate from the existing apply_scoring_and_percentiles in
    data.transforms on purpose: that one implements the app's three fixed
    presets (Full/Half/Standard) for the general stat tabs, and hardcodes
    the rest. A draft board has to honor whatever a real league actually
    uses - 6-point passing TDs, .5 points per carry, TE premium, negative
    points for fumbles, per-game yardage bonuses - so it needs a fully
    parameterized version.

    Every stat is pulled by explicit name via _col rather than by selecting
    columns with a prefix filter. That is deliberate: 'receptions' does not
    start with 'receiving_', so a startswith(('passing_','rushing_',
    'receiving_',...)) selection silently drops it and every curve gets
    built with PPR contributing exactly zero - a failure that computes
    cleanly, renders cleanly, and just quietly prices WRs like it's a
    standard league. Naming each stat is what makes that class of bug
    impossible.
    """
    pts = (
        _col(df, 'passing_yards') * scoring['pass_yd'] +
        _col(df, 'passing_tds') * scoring['pass_td'] +
        _col(df, 'passing_interceptions') * scoring['pass_int'] +
        _col(df, 'passing_2pt_conversions') * scoring['pass_2pt'] +
        _col(df, 'rushing_yards') * scoring['rush_yd'] +
        _col(df, 'rushing_tds') * scoring['rush_td'] +
        _col(df, 'rushing_attempts') * scoring['rush_att'] +
        _col(df, 'rushing_2pt_conversions') * scoring['rush_2pt'] +
        _col(df, 'receptions') * scoring['rec'] +
        _col(df, 'receiving_yards') * scoring['rec_yd'] +
        _col(df, 'receiving_tds') * scoring['rec_td'] +
        _col(df, 'receiving_2pt_conversions') * scoring['rec_2pt'] +
        (_col(df, 'rushing_fumbles_lost') + _col(df, 'receiving_fumbles_lost')) * scoring['fumble_lost'] +
        (_col(df, 'fg_made_0_19') + _col(df, 'fg_made_20_29') + _col(df, 'fg_made_30_39')) * scoring['fg_0_39'] +
        _col(df, 'fg_made_40_49') * scoring['fg_40_49'] +
        (_col(df, 'fg_made_50_59') + _col(df, 'fg_made_60_')) * scoring['fg_50_plus'] +
        _col(df, 'pat_made') * scoring['pat']
    )
    if scoring.get('te_premium') and position_col in df.columns:
        is_te = df[position_col].astype(str).str.upper() == 'TE'
        pts = pts + _col(df, 'receptions') * float(scoring['te_premium']) * is_te.astype(float)

    pts = pts + _yardage_bonus_points(df, scoring)
    return pts


# (stat column, threshold, scoring key) for every per-game yardage milestone.
YARDAGE_BONUSES = [
    ('rushing_yards', 100, 'bonus_rush_100'), ('rushing_yards', 150, 'bonus_rush_150'),
    ('rushing_yards', 200, 'bonus_rush_200'), ('rushing_yards', 250, 'bonus_rush_250'),
    ('receiving_yards', 100, 'bonus_rec_100'), ('receiving_yards', 150, 'bonus_rec_150'),
    ('receiving_yards', 200, 'bonus_rec_200'), ('receiving_yards', 250, 'bonus_rec_250'),
    ('passing_yards', 300, 'bonus_pass_300'), ('passing_yards', 400, 'bonus_pass_400'),
    ('passing_yards', 500, 'bonus_pass_500'), ('passing_yards', 600, 'bonus_pass_600'),
]


def has_yardage_bonuses(scoring):
    return any(float(scoring.get(key, 0) or 0) != 0 for _, _, key in YARDAGE_BONUSES)


def _yardage_bonus_points(df, scoring):
    """
    Per-game milestone bonus points for one frame of GAME rows.

    Caller contract: the rows handed in must be per-game, not season totals.
    Applying a 100-yard bonus to a 1,400-yard season total would award it
    once for the season instead of once per qualifying game - an order-of-
    magnitude error in exactly the direction that flatters big-yardage
    players. Every caller in this module scores weekly rows and sums after.
    """
    if not has_yardage_bonuses(scoring):
        return pd.Series(0.0, index=df.index)

    highest_only = str(scoring.get('bonus_mode', 'cumulative')).lower().startswith('high')
    total = pd.Series(0.0, index=df.index)
    if highest_only:
        # Walk thresholds descending and award only the first (largest) one
        # cleared, tracking which rows have already been paid.
        by_stat = {}
        for stat, threshold, key in YARDAGE_BONUSES:
            by_stat.setdefault(stat, []).append((threshold, key))
        for stat, tiers in by_stat.items():
            values = _col(df, stat)
            already_paid = pd.Series(False, index=df.index)
            for threshold, key in sorted(tiers, key=lambda t: -t[0]):
                points = float(scoring.get(key, 0) or 0)
                qualifies = (values >= threshold) & ~already_paid
                if points:
                    total = total + qualifies.astype(float) * points
                already_paid = already_paid | (values >= threshold)
    else:
        for stat, threshold, key in YARDAGE_BONUSES:
            points = float(scoring.get(key, 0) or 0)
            if points:
                total = total + (_col(df, stat) >= threshold).astype(float) * points
    return total


def _season_points_from_weekly(weekly_df, name_col, scoring):
    """
    Season fantasy-point totals, scored WEEK BY WEEK and then summed.

    Scoring per week and summing (rather than summing stats and scoring
    once) is required for per-game yardage milestone bonuses to be counted
    at all - see _yardage_bonus_points. It's identical arithmetic for every
    other scoring rule, since those are all linear in the stats, so there's
    no cost to always taking this path and no second code path to keep in
    sync.
    """
    scored = weekly_df.copy()
    scored['_points'] = score_stats(scored, scoring)
    return (scored.groupby(['season', name_col], observed=True)
            .agg(points=('_points', 'sum'), position=('position', 'first'))
            .reset_index())


@st.cache_data(show_spinner=False)
def build_positional_value_curves(scoring, latest_season, n_seasons=CURVE_SEASONS):
    """
    curve[pos] = array where index i holds the season points scored by the
    player who finished (i+1)th at that position, averaged across recent
    seasons and under THIS league's scoring.

    This is the piece that makes the whole board settings-aware. It is not a
    ranking translated into points by some fixed constant - it is measured,
    from this app's own local weekly data, at each league's actual scoring
    settings. Change PPR from 1.0 to 0.5 and the WR curve genuinely
    flattens, the RB curve doesn't, and every downstream number (VORP,
    tier breaks) moves accordingly.

    Pre-2021 seasons are scaled by 17/16 so a 16-game season's totals don't
    drag the curve down relative to current ones.
    """
    from data.loaders import load_weekly_stats_history
    hist = load_weekly_stats_history()
    if hist is None or hist.empty or 'season' not in hist.columns:
        return {}

    df = hist.copy()
    if 'season_type' in df.columns:
        df = df[df['season_type'].astype(str).str.upper().isin(['REG', 'REGULAR', ''])]
    df = df[pd.to_numeric(df['week'], errors='coerce').fillna(0) > 0]
    seasons = sorted([s for s in df['season'].dropna().unique() if s <= latest_season])[-n_seasons:]
    if not seasons:
        return {}
    df = df[df['season'].isin(seasons)]

    name_col = 'player_display_name' if 'player_display_name' in df.columns else 'player_name'
    if name_col not in df.columns or 'position' not in df.columns:
        return {}

    season_totals = _season_points_from_weekly(df, name_col, scoring)

    # 16-game seasons scaled up so they're comparable to 17-game ones.
    games_scale = season_totals['season'].map(lambda s: MODERN_SEASON_GAMES / 16.0 if s < 2021 else 1.0)
    season_totals['points'] = season_totals['points'] * games_scale

    curves = {}
    for pos in DRAFTABLE_POSITIONS:
        pos_rows = season_totals[season_totals['position'].astype(str).str.upper() == pos]
        if pos_rows.empty:
            continue
        per_season = []
        for season in seasons:
            vals = pos_rows.loc[pos_rows['season'] == season, 'points'].sort_values(ascending=False).to_numpy()
            if len(vals) >= 10:
                per_season.append(vals)
        if not per_season:
            continue
        depth = max(len(v) for v in per_season)
        # Pad each season out to the deepest one by repeating its own tail
        # value rather than zero-filling: a season that only had 60 rostered
        # TEs shouldn't drag rank 61+ to zero and invent a cliff that isn't
        # there.
        padded = np.vstack([np.concatenate([v, np.full(depth - len(v), v[-1])]) for v in per_season])
        curve = padded.mean(axis=0)
        # Light smoothing + enforced monotonicity. The curve is a rank-
        # ordered quantity so it must be non-increasing; sampling noise at
        # the deep end occasionally violates that, which would otherwise
        # show up as a nonsensical "the WR84 slot is worth more than WR83".
        if len(curve) >= 5:
            kernel = np.ones(3) / 3.0
            smoothed = np.convolve(curve, kernel, mode='same')
            smoothed[0], smoothed[-1] = curve[0], curve[-1]
            curve = smoothed
        curve = np.minimum.accumulate(curve)
        curves[pos] = curve

    dst_curve = build_dst_value_curve(scoring, seasons)
    if dst_curve is not None and len(dst_curve):
        curves['DST'] = dst_curve
    return curves


@st.cache_data(show_spinner=False)
def build_dst_value_curve(scoring, seasons):
    """
    The same realized-finish value curve as every other position, but for
    team defenses - which don't exist in player-level stats at all and so
    have to be built from team-week defensive stats plus points allowed.

    Points allowed comes from the schedule (the opponent's final score),
    not from a defensive box score, because that's what it actually is.
    Returns None when the source is unreachable, in which case DSTs are
    carried on the board as untradeable-for-value rows rather than
    fabricated - see build_draft_board's handling of positions with no
    curve.
    """
    try:
        import nflreadpy
        seasons = [int(s) for s in seasons]
        team_stats = nflreadpy.load_team_stats(seasons, summary_level='week').to_pandas()
        sched = nflreadpy.load_schedules(seasons).to_pandas()
    except Exception:
        return None
    if team_stats is None or team_stats.empty or sched is None or sched.empty:
        return None

    # Points allowed per team per week, read off the schedule from the other
    # side of each game.
    scored = []
    for _, g in sched.iterrows():
        if pd.isna(g.get('home_score')) or pd.isna(g.get('away_score')):
            continue
        scored.append({'season': g['season'], 'week': g['week'], 'team': g['home_team'],
                       'points_allowed': g['away_score']})
        scored.append({'season': g['season'], 'week': g['week'], 'team': g['away_team'],
                       'points_allowed': g['home_score']})
    if not scored:
        return None
    pa = pd.DataFrame(scored)

    ts = team_stats.copy()
    if 'season_type' in ts.columns:
        ts = ts[ts['season_type'].astype(str).str.upper().isin(['REG', 'REGULAR'])]
    ts = ts.merge(pa, on=['season', 'week', 'team'], how='inner')
    if ts.empty:
        return None

    def _pa_points(val):
        for threshold, pts in DST_POINTS_ALLOWED_TIERS:
            if val <= threshold:
                return pts
        return DST_POINTS_ALLOWED_WORST

    ts['dst_points'] = (
        _col(ts, 'def_sacks') * scoring.get('dst_sack', 1.0) +
        _col(ts, 'def_interceptions') * scoring.get('dst_int', 2.0) +
        _col(ts, 'fumble_recovery_opp') * scoring.get('dst_fumble_rec', 2.0) +
        (_col(ts, 'def_tds') + _col(ts, 'special_teams_tds')) * scoring.get('dst_td', 6.0) +
        _col(ts, 'def_safeties') * scoring.get('dst_safety', 2.0) +
        ts['points_allowed'].map(_pa_points)
    )

    per_season = []
    for season in sorted(ts['season'].unique()):
        totals = ts[ts['season'] == season].groupby('team')['dst_points'].sum()
        scale = MODERN_SEASON_GAMES / 16.0 if season < 2021 else 1.0
        vals = (totals * scale).sort_values(ascending=False).to_numpy()
        if len(vals) >= 20:
            per_season.append(vals)
    if not per_season:
        return None
    depth = max(len(v) for v in per_season)
    padded = np.vstack([np.concatenate([v, np.full(depth - len(v), v[-1])]) for v in per_season])
    return np.minimum.accumulate(padded.mean(axis=0))


# Fallback rank-uncertainty slopes (sd of finish position = intercept +
# slope x consensus rank), used when the local history needed to calibrate
# empirically isn't available. Ordered the way the measured data comes out:
# QB and TE are far more predictable year to year than RB and WR, which is
# a real and well-known asymmetry, not noise.
FALLBACK_RANK_SD = {
    'QB': (5.0, 0.22), 'RB': (9.0, 0.30), 'WR': (10.0, 0.32),
    'TE': (5.5, 0.28), 'K': (9.0, 0.35), 'DST': (7.0, 0.35),
}

# Consensus rankings beat naive "he'll repeat last year's finish"
# persistence, which is what the calibration below actually measures - the
# experts know about the trade, the new OC, the torn ACL. So the measured
# dispersion is scaled down before use. 0.75 is a judgment call, not a
# fitted value: it can't be fitted without a history of preseason consensus
# rankings to check against, which no free source publishes. It is exposed
# as the "Projection uncertainty" control in the UI precisely because it's
# the one number here that is a considered estimate rather than a
# measurement.
CONSENSUS_SKILL_SHRINK = 0.75


@st.cache_data(show_spinner=False)
def calibrate_rank_uncertainty(scoring, latest_season, n_seasons=CURVE_SEASONS):
    """
    Measure, from this app's own history, how far players actually land from
    where they were ranked - as a function of position and rank.

    The honest version of this would regress final finish against PRESEASON
    CONSENSUS rank across many years, but no free source publishes an
    archive of historical preseason ECR. What is available locally is each
    player's finish in year N-1 and year N, so this measures the dispersion
    of year-over-year finish instead and then shrinks it by
    CONSENSUS_SKILL_SHRINK to account for consensus being better-informed
    than pure persistence.

    Why bother instead of picking a constant: the measured spread is not
    remotely uniform across positions. Top-6 QBs and TEs land within about
    10 slots of where they were; top-6 RBs and WRs scatter more than twice
    as far. A board that treats those as equally certain will systematically
    overrate the reliability of early RB/WR picks and underrate the
    stability of elite QB/TE - which is exactly the sort of quiet, coherent
    wrongness that's hardest to notice on draft night.

    Returns {pos: (intercept, slope)} for sd_of_finish = intercept + slope * rank.
    """
    from data.loaders import load_weekly_stats_history
    hist = load_weekly_stats_history()
    if hist is None or hist.empty or 'season' not in hist.columns:
        return dict(FALLBACK_RANK_SD)

    df = hist[pd.to_numeric(hist['week'], errors='coerce').fillna(0) > 0].copy()
    if 'season_type' in df.columns:
        df = df[df['season_type'].astype(str).str.upper().isin(['REG', 'REGULAR'])]
    name_col = 'player_display_name' if 'player_display_name' in df.columns else 'player_name'
    if name_col not in df.columns or 'position' not in df.columns:
        return dict(FALLBACK_RANK_SD)

    seasons = sorted([s for s in df['season'].dropna().unique() if s <= latest_season])[-(n_seasons + 1):]
    df = df[df['season'].isin(seasons)]
    totals = _season_points_from_weekly(df, name_col, scoring)
    totals['pos_rank'] = totals.groupby(['season', 'position'])['points'].rank(ascending=False, method='first')

    out = dict(FALLBACK_RANK_SD)
    for pos in DRAFTABLE_POSITIONS:
        pos_rows = totals[totals['position'].astype(str).str.upper() == pos]
        pairs = []
        for season in seasons[1:]:
            prev = pos_rows[pos_rows['season'] == season - 1].set_index(name_col)['pos_rank']
            cur = pos_rows[pos_rows['season'] == season].set_index(name_col)['pos_rank']
            joined = pd.concat([prev.rename('prev'), cur.rename('cur')], axis=1).dropna()
            if not joined.empty:
                pairs.append(joined)
        if not pairs:
            continue
        joined = pd.concat(pairs)
        joined = joined[joined['prev'] <= 60]
        if len(joined) < 30:
            continue
        # sd of the year-over-year move, measured in rank buckets, then fit
        # linearly against rank so it extrapolates smoothly past the buckets.
        buckets, sds = [], []
        for lo, hi in [(1, 6), (7, 12), (13, 24), (25, 36), (37, 60)]:
            b = joined[(joined['prev'] >= lo) & (joined['prev'] <= hi)]
            if len(b) >= 12:
                buckets.append((lo + hi) / 2.0)
                sds.append(float((b['cur'] - b['prev']).std()))
        if len(buckets) < 2:
            continue
        slope, intercept = np.polyfit(buckets, sds, 1)
        intercept *= CONSENSUS_SKILL_SHRINK
        slope *= CONSENSUS_SKILL_SHRINK
        # A negative or absurd fit means the buckets were too noisy to trust;
        # keep the fallback rather than propagating a nonsense model.
        if intercept > 0.5 and 0 <= slope < 2.0:
            out[pos] = (float(intercept), float(slope))
    return out


def _rank_uncertainty(board, calibration=None):
    """
    Per-player spread (in positional finish slots) around his consensus rank.

    Two ingredients, doing different jobs:
      - The calibrated position/rank baseline says how uncertain a RB12 is
        in general. That's the measured part.
      - Each player's own expert disagreement then scales that baseline up
        or down. A player the analysts are unusually split on gets a wider
        band than a typical player at his rank; a consensus lock gets a
        narrower one.

    The scaling is relative to what's NORMAL at that rank (fitted per
    position), not absolute, because raw expert spread grows with rank
    automatically - without normalizing, every late-round player would look
    "high risk" purely for being late, which tells you nothing.
    """
    calibration = calibration or FALLBACK_RANK_SD
    # A missing Pos Rank falls to WORSE than the deepest real rank on the
    # board, never to 1 (which used to make an unranked player look like a
    # rock-solid RB1/WR1 - the tightest uncertainty band this formula can
    # produce, the opposite of what "no real consensus rank" should mean).
    # Same fix, same reasoning, as data.draft_projections.project_stat_lines'
    # rank_idx - see that call site's comment for the confirmed real board
    # this was caught on.
    raw_rank = pd.to_numeric(board['Pos Rank'], errors='coerce')
    worst_known = raw_rank.max()
    fallback_rank = float(worst_known) + 1.0 if pd.notna(worst_known) else 1.0
    pos_rank = raw_rank.fillna(fallback_rank).to_numpy(dtype=float)
    positions = board['Pos'].astype(str).str.upper().to_numpy()

    base = np.empty(len(board), dtype=float)
    for i, (pos, rank) in enumerate(zip(positions, pos_rank)):
        intercept, slope = calibration.get(pos, FALLBACK_RANK_SD.get(pos, (8.0, 0.3)))
        base[i] = intercept + slope * rank

    sd = pd.to_numeric(board.get('ECR SD'), errors='coerce')
    best = pd.to_numeric(board.get('ECR Best'), errors='coerce')
    worst = pd.to_numeric(board.get('ECR Worst'), errors='coerce')
    # (worst - best)/4 approximates a standard deviation for a roughly
    # bell-shaped spread; take whichever of that and the published stdev is
    # larger, so a tight stdev with one wild outlier still reads as risky.
    expert = pd.concat([sd, (worst - best) / 4.0], axis=1).max(axis=1)

    ratio = np.ones(len(board), dtype=float)
    for pos in np.unique(positions):
        mask = positions == pos
        vals = expert[mask].to_numpy(dtype=float)
        ranks = pos_rank[mask]
        ok = np.isfinite(vals) & (vals > 0)
        if ok.sum() < 8:
            continue
        slope, intercept = np.polyfit(ranks[ok], vals[ok], 1)
        expected = np.maximum(intercept + slope * ranks, 1e-6)
        with np.errstate(invalid='ignore', divide='ignore'):
            r = np.where(np.isfinite(vals) & (expected > 0), vals / expected, 1.0)
        ratio[mask] = np.clip(np.nan_to_num(r, nan=1.0), 0.65, 1.5)

    return np.maximum(base * ratio, 1.0)


def _balanced_finish_weights(weights, iters=40):
    """
    Make the finish-probability matrix conserve total points within a
    position, by enforcing that finish slots get claimed exactly as often as
    they exist.

    THE PROBLEM THIS FIXES, because it is subtle and it distorted the board
    across positions: spreading each player over the finishes he might land
    on necessarily pulls the top of a position DOWN, since the curve is
    steeply convex there and a projected RB1 lands at RB4 as easily as he
    stays at RB1. That correction is right. But the size of it scales with
    how uncertain the position is - and RB/WR carry roughly twice the
    measured rank uncertainty of QB/TE. So the raw version deflated elite
    RB/WR about twice as hard as elite QBs, which is not a real football
    fact, it's an artifact of the smearing. The visible symptom was elite
    QBs floating up the overall board past where any real draft takes them.

    The fix is a conservation law. There are N players and N finish slots;
    every player finishes somewhere, and every slot is filled by exactly one
    player. So the matrix should be doubly stochastic - rows sum to 1
    (each player lands somewhere) AND columns sum to their share (each slot
    is claimed once). Alternately normalizing rows and columns (Sinkhorn)
    converges to the closest such matrix. Total projected points per
    position then equals the total the position actually scored, so the
    smearing redistributes value within a position instead of quietly
    destroying it - and it destroys none of it more aggressively at RB than
    at QB.
    """
    w = np.array(weights, dtype=float)
    n_players, n_slots = w.shape
    if n_players == 0 or n_slots == 0:
        return w
    # Each slot should carry the same total mass, and the masses must add up
    # to one whole player per row.
    col_target = n_players / float(n_slots)
    for _ in range(iters):
        row_sums = w.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        w /= row_sums
        col_sums = w.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        w *= col_target / col_sums
    # End on a row normalization so each player's distribution is a proper
    # one; after convergence the columns are already within rounding of
    # target, so this costs nothing.
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return w / row_sums


def project_points(board, curves, rank_sd_scale=1.0, calibration=None):
    """
    Attach Proj Pts / Ceiling / Floor to a board carrying Pos + Pos Rank.

    Each player's value is the expected value of the positional value curve
    under the distribution of finishes he could plausibly have (see the
    module docstring, step 2). Ceiling and Floor are the 85th/15th
    percentiles of that same distribution, so a volatile player's ceiling
    rises and floor falls together, out of one coherent model, instead of
    being two independently-invented numbers.
    """
    if board.empty or not curves:
        return board

    out = board.copy()
    out['Proj Pts'] = np.nan
    out['Ceiling'] = np.nan
    out['Floor'] = np.nan
    sd_all = _rank_uncertainty(out, calibration) * float(rank_sd_scale)

    for pos, curve in curves.items():
        mask = (out['Pos'].astype(str).str.upper() == pos).to_numpy()
        if not mask.any():
            continue
        ranks = pd.to_numeric(out.loc[mask, 'Pos Rank'], errors='coerce').fillna(len(curve)).to_numpy()
        sds = np.maximum(sd_all[mask], 0.75)

        finishes = np.arange(1, len(curve) + 1, dtype=float)
        # weights[i, k] = P(player i finishes at slot k)
        z = (finishes[None, :] - ranks[:, None]) / sds[:, None]
        weights = _balanced_finish_weights(np.exp(-0.5 * z ** 2))

        out.loc[mask, 'Proj Pts'] = weights @ curve
        cdf = np.cumsum(weights, axis=1)
        # Curve is descending, so a HIGH finish percentile is a LOW slot
        # index: the ceiling is the 15th percentile of finish position.
        ceil_idx = np.argmax(cdf >= 0.15, axis=1)
        floor_idx = np.argmax(cdf >= 0.85, axis=1)
        out.loc[mask, 'Ceiling'] = curve[ceil_idx]
        out.loc[mask, 'Floor'] = curve[floor_idx]

    for c in ('Proj Pts', 'Ceiling', 'Floor'):
        out[c] = pd.to_numeric(out[c], errors='coerce').round(1)
    return out


def add_outcome_range_from_projections(board, calibration=None, rank_sd_scale=1.0):
    """
    Ceiling and Floor for a board whose Proj Pts already came from somewhere
    else (the volume-based stat-line projections, or an FFA import).

    The projection itself is left completely untouched - only the range
    around it is added. The implied value curve is simply that position's own
    projected points sorted descending, so the outcome range is expressed in
    the same units as the projection that produced it rather than against a
    historical curve the projection never saw. A player is then spread over
    the finishes he could plausibly land on using the same measured rank
    uncertainty as everywhere else, and Ceiling/Floor are read off the 15th
    and 85th percentiles of that spread.

    Kept separate from project_points because these are now two different
    jobs. project_points DERIVES a projection from a rank; this one only
    describes the uncertainty around a projection that already exists.
    """
    if board.empty or 'Proj Pts' not in board.columns:
        return board

    out = board.copy()
    out['Ceiling'] = np.nan
    out['Floor'] = np.nan
    sd_all = _rank_uncertainty(out, calibration) * float(rank_sd_scale)
    pos_upper = out['Pos'].astype(str).str.upper()

    for pos in pos_upper.unique():
        mask = (pos_upper == pos) & out['Proj Pts'].notna()
        if mask.sum() < 3:
            out.loc[mask, 'Ceiling'] = out.loc[mask, 'Proj Pts']
            out.loc[mask, 'Floor'] = out.loc[mask, 'Proj Pts']
            continue
        subset = out.loc[mask]
        curve = np.sort(subset['Proj Pts'].to_numpy(dtype=float))[::-1]
        ranks = subset['Proj Pts'].rank(ascending=False, method='first').to_numpy(dtype=float)
        sds = np.maximum(sd_all[mask.to_numpy()], 0.75)

        finishes = np.arange(1, len(curve) + 1, dtype=float)
        z = (finishes[None, :] - ranks[:, None]) / sds[:, None]
        # Row-normalized, NOT Sinkhorn-balanced. The balancing exists to
        # conserve a position's total points when the smearing is producing
        # the projection itself; here the projection already exists and all
        # that's wanted is each player's own distribution over finishes.
        # Balancing it drags probability mass off the top of the board to
        # satisfy the column constraint, which pushed the best player's
        # "ceiling" several slots BELOW his own projection - a ceiling that
        # is worse than the expectation is obviously wrong.
        weights = np.exp(-0.5 * z ** 2)
        totals = weights.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        weights = weights / totals
        cdf = np.cumsum(weights, axis=1)
        ceiling = curve[np.argmax(cdf >= 0.15, axis=1)]
        floor = curve[np.argmax(cdf >= 0.85, axis=1)]
        # A player at the very top of his position has nowhere above him to
        # regress toward, so clamp rather than let the discretized curve
        # hand back a ceiling under his own projection.
        projected = subset['Proj Pts'].to_numpy(dtype=float)
        out.loc[mask, 'Ceiling'] = np.maximum(ceiling, projected)
        out.loc[mask, 'Floor'] = np.minimum(floor, projected)

    for column in ('Ceiling', 'Floor'):
        out[column] = pd.to_numeric(out[column], errors='coerce').round(1)
    return out


def blend_with_recent_production(board, recent_df, weight):
    """
    Optionally pull projections toward each player's own most recent
    per-game production, extrapolated to a full season.

    Defaults to off (weight 0) because consensus rank already prices in
    everything last year's box score cannot know - a new offensive
    coordinator, an offseason trade, a rookie who hasn't played a snap, a
    back returning from an ACL. Last year's stats are a check on the
    market, not a replacement for it, which is exactly how the slider is
    framed in the UI. It earns its place for someone who thinks the market
    is asleep on a player whose underlying volume was already elite.
    """
    if board.empty or recent_df is None or recent_df.empty or weight <= 0:
        return board
    out = board.copy()
    prod = recent_df.set_index('Player')['Proj Pts Recent'] if 'Proj Pts Recent' in recent_df.columns else None
    if prod is None:
        return out
    mapped = out['Player'].map(prod)
    have = mapped.notna() & out['Proj Pts'].notna()
    out.loc[have, 'Proj Pts'] = (
        out.loc[have, 'Proj Pts'] * (1 - weight) + mapped[have] * weight
    ).round(1)
    return out


def compute_starter_demand(board, settings):
    """
    How many players at each position actually get STARTED across the whole
    league - which is what replacement level really means.

    Done by simulating the league's starting lineups being filled greedily
    from the top of the board, rather than the usual shortcut of splitting
    each FLEX slot evenly across RB/WR/TE. That shortcut is wrong in a way
    that matters: flex slots do not go to positions in equal shares, they
    go to whoever is actually better at the margin. In a PPR league flex
    slots skew heavily WR; in a TE-premium league TEs start taking them;
    in a superflex league QBs take essentially all of them. Splitting
    evenly hardcodes an answer that changes with exactly the settings this
    board is supposed to be sensitive to.

    Returns {pos: number of players started league-wide}.
    """
    n_teams = int(settings['num_teams'])
    roster = settings['roster']
    dedicated = {pos: n_teams * int(roster.get(pos, 0)) for pos in DRAFTABLE_POSITIONS}
    flex_open = n_teams * int(roster.get('FLEX', 0))
    superflex_open = n_teams * int(roster.get('SUPERFLEX', 0))

    started = {pos: 0 for pos in DRAFTABLE_POSITIONS}
    pool = board.dropna(subset=['Proj Pts']).sort_values('Proj Pts', ascending=False)
    for pos, pts in zip(pool['Pos'].astype(str).str.upper(), pool['Proj Pts']):
        if pos not in started:
            continue
        if started[pos] < dedicated.get(pos, 0):
            started[pos] += 1
        elif flex_open > 0 and pos in FLEX_ELIGIBLE:
            started[pos] += 1
            flex_open -= 1
        elif superflex_open > 0 and pos in SUPERFLEX_ELIGIBLE:
            started[pos] += 1
            superflex_open -= 1
        if (sum(started.values()) >= sum(dedicated.values()) + n_teams * (int(roster.get('FLEX', 0)) + int(roster.get('SUPERFLEX', 0)))):
            break
    return started


# How a league's BENCH gets spent, as relative shares per position. Starters
# are counted separately (compute_starter_demand); this is only the backups.
#
# WHY THIS EXISTS: the streaming measurement needs to know how many players
# at a position are ROSTERED, because that is what decides how deep the free
# pool is. It was being handed the STARTER count instead - so a 12-team
# league was modelled as rostering twelve tight ends, when it really rosters
# about twenty-two. The free pool looked twice as deep as it is, and the
# measured value of streaming was correspondingly too high.
#
# That single wrong input was the whole reason tight end replacement sat at
# TE6. Fed the real rostered depth, streaming a TE returns 136.1 against
# TE12's 142.1 - it no longer clears the bar, and TE replacement returns to
# TE12 where it belongs.
#
# The ratios are how real rosters are built rather than an even split: a
# manager carries one backup quarterback and one backup tight end at most,
# often none, and spends everything else on running backs and receivers
# because those are the positions with startable depth and injury churn.
# Kickers and defenses get no backup at all.
BENCH_SHARE = {'QB': 1.0, 'RB': 2.5, 'WR': 2.5, 'TE': 1.0, 'K': 0.0, 'DST': 0.0}


def rostered_depth(started, settings):
    """
    {pos: players rostered league-wide} = starters + a share of the bench.

    Settings-sensitive for free, which is the point of scaling to the REAL
    bench rather than hardcoding a depth: a deeper bench thins the waiver
    wire, which lowers what streaming returns, which correctly raises the
    surplus of the players who are actually scarce.
    """
    n_teams = int(settings['num_teams'])
    bench_spots = n_teams * int(settings['roster'].get('BENCH', 0))
    total_share = sum(BENCH_SHARE.values()) or 1.0
    return {pos: int(round(started.get(pos, 0)
                           + bench_spots * (share / total_share)))
            for pos, share in BENCH_SHARE.items()}


# Positions where the waiver wire stays genuinely deep all season, because
# all 32 NFL teams field one and a 12-team league rosters far fewer. These
# are the positions where "replacement" is not the last rostered player -
# it's whoever you stream, which is a much higher bar.
#
# TIGHT END BELONGS HERE, and leaving it out was distorting the whole back
# half of the board. The original reasoning ("leagues roster 60+ of them")
# is true of running backs and receivers and simply false of tight ends: a
# 12-team league that starts one TE rosters roughly fifteen, against 32 NFL
# starters, so ~17 startable tight ends sit free all season. That is the
# same structure as quarterback, and it has the same consequence - the real
# cost of passing on tight ends is far lower than "the 13th best one's
# season total", which is what the standard baseline assumes.
#
# The symptom this fixes: with replacement set at TE13, every tight end from
# roughly TE8 to TE14 carried positive VORP, floated 30-40 picks above his
# ADP, and the board went on recommending another one deep into the draft -
# a roster nobody would ever build, since only one of them can start.
#
# RB AND WR STAY OUT, AND THE ORIGINAL REASON GIVEN HERE WAS WRONG. It used
# to say streaming them measures below their last rostered starter, so
# listing them would change nothing. That was only true while the free pool
# was being computed from starter counts. Fed the real rostered depth,
# streaming measures 207.8 at RB against RB24's 171.7 - it would bind, and it
# would put running back replacement at RB13.
#
# That number is not believable, and the simulation below says why: a player
# who is free in week 1 stays free ALL SEASON, because the rostered tier is
# fixed by last year's finish. So a breakout rookie back is on the wire in
# week 1, gets picked up in week 3, and is started every week after. That is
# not streaming - it is being handed the season's best waiver add with nobody
# competing for him. Quarterback and tight end tolerate the assumption
# because breakouts there are rarer and shallower; running back is exactly
# where it breaks, since that is where mid-season starters are created and
# where the wire is cleared within hours.
#
# So the exclusion is right and the justification is different: RB/WR are out
# until the simulation models waiver competition (remove a player from the
# pool once he is picked up, or cap how long he can be held). Do that and one
# rule can safely cover every position.
#
# This is settings-sensitive without special-casing. build_streaming_replacement
# only ever RAISES the bar, and it takes the rostered count as input - so a
# TE-premium or 2-TE league rosters more tight ends, the free pool thins,
# the measured value drops, and tight ends correctly regain their surplus.
STREAMABLE_POSITIONS = ['QB', 'TE', 'K', 'DST']

# Positions universally drafted in the last couple of rounds. See
# demote_late_round_positions for why this is a fact about draft structure
# rather than a judgement about their value.
LATE_ROUND_POSITIONS = ['K', 'DST']

# Spare picks beyond the K/DST slots themselves before those positions start
# being suggested. One, so with a kicker and a defense still to fill they
# appear with three picks left - late enough to match how drafts actually go,
# early enough that you can't run out of room.
LATE_ROUND_GRACE = 1

# How many free options a streamer realistically rotates between, chosen on
# prior form. Averaging the top 2 rather than always taking the single best
# reflects that the obvious pickup isn't always startable (bye, injury,
# someone else grabbed him first).
STREAMING_CHOICE_DEPTH = 2

# A streamer needs some evidence before picking someone up, so the
# simulation below only considers free players with at least this many
# games already played that season.
STREAMING_MIN_PRIOR_GAMES = 2


@st.cache_data(show_spinner=False)
def build_streaming_replacement(scoring, latest_season, rostered_counts, n_seasons=CURVE_SEASONS):
    """
    Measure what streaming a position off waivers actually returns over a
    full season, instead of assuming replacement is the last rostered
    player's season total.

    WHY THIS EXISTS: standard VORP puts QB replacement at QB12 in a 12-team
    1QB league, which then prices an elite quarterback as a top-12 overall
    pick. Real drafts - especially sharp ones - never take one there, and
    they're right. The reason is that QB12's SEASON TOTAL is not what you
    get by passing on quarterbacks. All 32 teams start one, only 12 are
    rostered, so every week there are ~20 free quarterbacks and you start
    the best matchup among them. Cherry-picking weekly from a deep free
    pool returns far more than any single unrostered player's season total,
    which means the true replacement bar is much higher and the surplus of
    an elite QB much smaller.

    Measured by simulating the strategy WITHOUT hindsight, which is the only
    way this number means anything. Each week, the free players are ranked
    by how they'd scored up to that point in the season - information a real
    streamer actually has - and the best of them is started. What he then
    scored THAT week is what gets banked. Summing across the season gives
    what streaming really returned.

    The naive version of this measurement (rank free players by what they
    scored in the week being simulated, then start the winner) produces
    numbers well above the position's own #1 finisher, because it is
    cherry-picking a weekly maximum out of ~20 candidates after the results
    are in. That is not a baseline, it's a leak.

    The same logic self-corrects for superflex without special-casing: pass
    a rostered count of 24 instead of 12 and the free pool that's left is
    genuinely bad, so the measured streaming value drops and elite QBs
    regain their surplus - which is exactly right for that format.

    Returns {pos: season points}, empty when local history is unavailable.
    """
    from data.loaders import load_weekly_stats_history
    hist = load_weekly_stats_history()
    if hist is None or hist.empty or 'season' not in hist.columns:
        return {}

    df = hist[pd.to_numeric(hist['week'], errors='coerce').fillna(0) > 0].copy()
    if 'season_type' in df.columns:
        df = df[df['season_type'].astype(str).str.upper().isin(['REG', 'REGULAR'])]
    name_col = 'player_display_name' if 'player_display_name' in df.columns else 'player_name'
    if name_col not in df.columns or 'position' not in df.columns:
        return {}
    seasons = sorted([s for s in df['season'].dropna().unique() if s <= latest_season])[-n_seasons:]
    if not seasons:
        return {}
    df = df[df['season'].isin(seasons)]
    df['_points'] = score_stats(df, scoring)

    out = {}
    for pos, n_rostered in dict(rostered_counts).items():
        if pos not in STREAMABLE_POSITIONS:
            continue
        if pos == 'DST':
            # Team defenses never appear in player-level data, so the loop
            # below has nothing to work with. Streaming a defense is the most
            # common streaming there is, so it gets measured from team stats
            # rather than skipped.
            value = _stream_dst(scoring, seasons, int(n_rostered))
            if value:
                out['DST'] = value
            continue
        pos_rows = df[df['position'].astype(str).str.upper() == pos]
        if pos_rows.empty:
            continue
        per_season = []
        for season in seasons:
            season_rows = pos_rows[pos_rows['season'] == season]
            if season_rows.empty:
                continue
            # WHO IS ROSTERED is decided by the PREVIOUS season's finish -
            # draft-time information - not by how this season turned out.
            #
            # This distinction inverts the answer. Defining the rostered tier
            # by final season totals leaves a free pool consisting precisely
            # of the players who ended up worst, because every breakout has
            # been retroactively removed from it. Streaming then measures as
            # worse than the last starting player at every position, which is
            # both obviously false and the opposite of why nobody drafts a
            # kicker in round four. Ranking by the prior season puts this
            # year's breakouts back in the pool, where a real waiver wire
            # actually has them.
            prior_rows = pos_rows[pos_rows['season'] == season - 1]
            if not prior_rows.empty:
                prior_totals = prior_rows.groupby(name_col)['_points'].sum().sort_values(ascending=False)
                rostered = set(prior_totals.head(int(n_rostered)).index)
            else:
                totals = season_rows.groupby(name_col)['_points'].sum().sort_values(ascending=False)
                rostered = set(totals.head(int(n_rostered)).index)
            free_rows = season_rows[~season_rows[name_col].isin(rostered)]
            if free_rows.empty:
                continue

            banked = 0.0
            weeks = sorted(free_rows['week'].dropna().unique())
            for week in weeks:
                prior = free_rows[free_rows['week'] < week]
                this_week = free_rows[free_rows['week'] == week]
                if prior.empty or this_week.empty:
                    continue
                form = prior.groupby(name_col)['_points'].agg(['mean', 'count'])
                form = form[form['count'] >= STREAMING_MIN_PRIOR_GAMES]
                if form.empty:
                    continue
                # Pick on prior form only - no knowledge of this week's result.
                candidates = form.nlargest(STREAMING_CHOICE_DEPTH, 'mean').index
                actual = this_week[this_week[name_col].isin(candidates)]['_points']
                if not actual.empty:
                    banked += float(actual.mean())
            if banked <= 0:
                continue
            if season < 2021:
                banked *= MODERN_SEASON_GAMES / 16.0
            per_season.append(banked)
        if per_season:
            out[pos] = float(np.mean(per_season))
    return out


def _stream_dst(scoring, seasons, n_rostered):
    """Streaming value for team defenses, measured from team-week stats."""
    try:
        import nflreadpy
        team_stats = nflreadpy.load_team_stats(list(seasons), summary_level='week').to_pandas()
        sched = nflreadpy.load_schedules(list(seasons)).to_pandas()
    except Exception:
        return None
    if team_stats is None or team_stats.empty or sched is None or sched.empty:
        return None

    rows = []
    for _, g in sched.iterrows():
        if pd.isna(g.get('home_score')) or pd.isna(g.get('away_score')):
            continue
        rows.append({'season': g['season'], 'week': g['week'], 'team': g['home_team'],
                     'points_allowed': g['away_score']})
        rows.append({'season': g['season'], 'week': g['week'], 'team': g['away_team'],
                     'points_allowed': g['home_score']})
    if not rows:
        return None

    ts = team_stats.copy()
    if 'season_type' in ts.columns:
        ts = ts[ts['season_type'].astype(str).str.upper().isin(['REG', 'REGULAR'])]
    ts = ts.merge(pd.DataFrame(rows), on=['season', 'week', 'team'], how='inner')
    if ts.empty:
        return None

    def _pa_points(val):
        for threshold, pts in DST_POINTS_ALLOWED_TIERS:
            if val <= threshold:
                return pts
        return DST_POINTS_ALLOWED_WORST

    ts['_points'] = (
        _col(ts, 'def_sacks') * scoring.get('dst_sack', 1.0) +
        _col(ts, 'def_interceptions') * scoring.get('dst_int', 2.0) +
        _col(ts, 'fumble_recovery_opp') * scoring.get('dst_fumble_rec', 2.0) +
        (_col(ts, 'def_tds') + _col(ts, 'special_teams_tds')) * scoring.get('dst_td', 6.0) +
        _col(ts, 'def_safeties') * scoring.get('dst_safety', 2.0) +
        ts['points_allowed'].map(_pa_points)
    )

    per_season = []
    for season in sorted(ts['season'].unique()):
        season_rows = ts[ts['season'] == season]
        prior = ts[ts['season'] == season - 1]
        if prior.empty:
            continue
        rostered = set(prior.groupby('team')['_points'].sum()
                       .sort_values(ascending=False).head(int(n_rostered)).index)
        free_rows = season_rows[~season_rows['team'].isin(rostered)]
        if free_rows.empty:
            continue
        banked = 0.0
        for week in sorted(free_rows['week'].dropna().unique()):
            prior_weeks = free_rows[free_rows['week'] < week]
            this_week = free_rows[free_rows['week'] == week]
            if prior_weeks.empty or this_week.empty:
                continue
            form = prior_weeks.groupby('team')['_points'].agg(['mean', 'count'])
            form = form[form['count'] >= STREAMING_MIN_PRIOR_GAMES]
            if form.empty:
                continue
            candidates = form.nlargest(STREAMING_CHOICE_DEPTH, 'mean').index
            actual = this_week[this_week['team'].isin(candidates)]['_points']
            if not actual.empty:
                banked += float(actual.mean())
        if banked > 0:
            if season < 2021:
                banked *= MODERN_SEASON_GAMES / 16.0
            per_season.append(banked)
    return float(np.mean(per_season)) if per_season else None


# Bye week: one guaranteed absence per player per season, on top of injury.
_BYE_WEEKS = 1.0

# What share of a streamable position's bench value survives the fact that
# you could just pick someone up instead. Not zero - a rostered backup is
# occasionally the better option, and in a superflex league he's a genuine
# asset - but small, because the free pool at those positions is deep all
# season. See expected_start_share.
_STREAMABLE_BENCH_COVERAGE = 0.2

# Fallback absence rates (share of a 17-game season a draftable player at
# each position misses) for when local history can't be read. Close to the
# measured values so the model degrades quietly rather than changing shape.
_DEFAULT_ABSENCE = {'QB': 0.26, 'RB': 0.21, 'WR': 0.20, 'TE': 0.22, 'K': 0.10, 'DST': 0.0}


@st.cache_data(show_spinner=False)
def measure_absence_rates(latest_season, n_seasons=CURVE_SEASONS):
    """
    What share of a season a player you'd actually DRAFT ends up missing,
    per position.

    Measured the only way that's draft-relevant: take the players who
    finished near the top of a position in one season - the ones who'd be
    drafted the following August - and count how many of the next season's
    games they actually played. That folds in injury, benching, holdouts and
    retirement, which is the real question ("will this roster spot produce"),
    rather than the narrower one an injury feed answers.

    The result is deliberately NOT used to mark players down. It feeds the
    depth model below, where the question is how often a BACKUP gets pressed
    into your starting lineup.
    """
    from data.loaders import load_weekly_stats_history
    hist = load_weekly_stats_history()
    if hist is None or hist.empty or 'season' not in hist.columns:
        return dict(_DEFAULT_ABSENCE)
    df = hist[pd.to_numeric(hist['week'], errors='coerce').fillna(0) > 0].copy()
    if 'season_type' in df.columns:
        df = df[df['season_type'].astype(str).str.upper().isin(['REG', 'REGULAR'])]
    name_col = 'player_display_name' if 'player_display_name' in df.columns else 'player_name'
    points_col = 'fantasy_points_ppr' if 'fantasy_points_ppr' in df.columns else None
    if name_col not in df.columns or 'position' not in df.columns or not points_col:
        return dict(_DEFAULT_ABSENCE)

    seasons = sorted([s for s in df['season'].dropna().unique() if s <= latest_season])[-n_seasons:]
    if len(seasons) < 2:
        return dict(_DEFAULT_ABSENCE)

    # How deep into each position a 12-team league's drafters actually go.
    draftable_depth = {'QB': 24, 'RB': 48, 'WR': 60, 'TE': 24, 'K': 14, 'DST': 14}
    out = dict(_DEFAULT_ABSENCE)
    for pos, depth in draftable_depth.items():
        rates = []
        for season in seasons[1:]:
            prior = df[(df['season'] == season - 1) &
                       (df['position'].astype(str).str.upper() == pos)]
            current = df[(df['season'] == season) &
                         (df['position'].astype(str).str.upper() == pos)]
            if prior.empty or current.empty:
                continue
            drafted = prior.groupby(name_col)[points_col].sum().nlargest(depth).index
            played = current[current[name_col].isin(drafted)].groupby(name_col).size()
            # A drafted player who never appears played zero games, and
            # dropping him would measure only the players who stayed healthy.
            games = np.array([int(played.get(p, 0)) for p in drafted], dtype=float)
            if len(games):
                rates.append(1.0 - games.mean() / MODERN_SEASON_GAMES)
        if rates:
            out[pos] = float(np.clip(np.mean(rates), 0.0, 0.6))
    return out


def lineup_slots_per_team(board, settings):
    """
    How many starting slots ONE roster devotes to each position, flex
    included.

    DERIVED FROM SETTINGS, NOT FROM THE BOARD IN FRONT OF IT. An earlier
    version divided compute_starter_demand by the league size, which made the
    flex share pleasingly empirical - but that function allocates flex by
    walking the pool it is handed, and mid-draft that pool has had its best
    receivers taken. Ten rounds in it was reporting that tight ends occupy
    1.75 starting slots and receivers 2.25, which is an artifact of who was
    left rather than a fact about the league, and the depth model downstream
    then priced a third tight end as most of a starter.

    Flex is split across the eligible positions in proportion to their
    dedicated starting slots, which is stable, moves correctly with settings
    (a 3-WR league sends more flex to receivers, superflex sends it to
    quarterbacks) and cannot drift as players come off the board.

    `board` is kept in the signature for callers and for symmetry; nothing
    reads it.
    """
    roster = settings['roster']
    slots = {pos: float(roster.get(pos, 0)) for pos in DRAFTABLE_POSITIONS}
    for slot_name, eligible in (('FLEX', FLEX_ELIGIBLE), ('SUPERFLEX', SUPERFLEX_ELIGIBLE)):
        count = float(roster.get(slot_name, 0))
        if count <= 0:
            continue
        weights = {pos: max(float(roster.get(pos, 0)), 0.0) for pos in eligible}
        total = sum(weights.values())
        if total <= 0:
            weights = {pos: 1.0 for pos in eligible}
            total = float(len(eligible))
        for pos, weight in weights.items():
            slots[pos] = slots.get(pos, 0.0) + count * weight / total
    return slots


def contingency_value(points, ceiling, pos, share, waiver_points):
    """
    What a player who does NOT crack your starting lineup is worth: an OPTION
    on him beating whatever you could have picked up free, in the weeks he'd
    actually be startable.

    An option is valued on its upside. You are not stuck with a late-round
    pick who busts - you drop him and stream the position - so the downside is
    bounded near zero while the upside is a starter you didn't pay for. That
    is why high-variance late running backs are worth more than their
    projections suggest and why a safe, low-ceiling backup is worth less than
    it looks. E[max(X - waiver, 0)] with X normal around the projection, its
    spread taken from the board's own Ceiling.

    Lives here rather than in data.draft_intel because both the board's
    recommendations and the positional value-add panel need it, and
    draft_intel already imports this module.
    """
    if points is None or pd.isna(points) or share <= 0:
        return 0.0
    mean = float(points)
    waiver = float(waiver_points or 0.0)
    if ceiling is not None and pd.notna(ceiling) and float(ceiling) > mean:
        # Ceiling is the 85th percentile outcome, 1.036 standard deviations up.
        sigma = max(1.0, (float(ceiling) - mean) / 1.036)
    else:
        sigma = max(1.0, 0.25 * abs(mean))
    z = (waiver - mean) / sigma
    density = float(np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi))
    expected_excess = sigma * density + (mean - waiver) * normal_sf(z)
    return max(0.0, float(share) * expected_excess)


def expected_start_share(pos, depth, slots, absence):
    """
    The share of the season the `depth`-th best player you own at `pos`
    spends in your STARTING lineup (depth is 1-indexed: 1 is your best).

    THIS IS THE NUMBER THAT ANSWERS "should I take another one of these",
    and it is why a second tight end and a third receiver are not
    remotely the same pick even when the board rates them alike.

    Within the starting slots a position occupies, a player starts whenever
    he's available. Past them he is a contingency asset: he only enters the
    lineup in a week where enough of the players above him are out, which is
    a binomial tail on the measured absence rate. The number of slots is
    therefore doing all the work, and it differs enormously by position:

        TE, 1 slot   depth 1 = 1.00   depth 2 = 0.22   depth 3 = 0.00
        WR, ~2.9     depth 1 = 1.00   depth 2 = 1.00   depth 3 = 0.90
        RB, ~2.4     depth 1 = 1.00   depth 2 = 1.00   depth 3 = 0.40

    A third receiver still fills a flex slot nearly every week. A second
    tight end plays about three weeks a year, and a third plays never -
    which is exactly why nobody drafts one, and why a board that priced
    them off league-wide surplus alone kept recommending them.

    Returned relative to a full starter (depth 1 = 1.0), so it composes as a
    plain multiplier on a player's marginal value without double-counting
    the games a starter misses - those are already in his projection.
    """
    slot_count = float(slots.get(pos, 1.0) or 0.0)
    rate = float(absence.get(pos, 0.2) or 0.0)
    # Bye weeks are an absence the injury measurement can't see: it counts
    # games played out of 17, and a bye isn't a missed game.
    rate = float(np.clip(rate + _BYE_WEEKS / MODERN_SEASON_GAMES, 0.0, 0.95))
    depth = int(max(1, depth))
    if slot_count <= 0:
        return 0.0
    if depth <= slot_count:
        # Whole slot, or the fractional tail of one (a 2.9-slot position
        # gives its third player 90% of a starting job).
        return float(min(1.0, slot_count - depth + 1))

    # Bench. He needs at least `needed` of the `n` starters above him out.
    needed = int(np.ceil(depth - slot_count))
    n = max(1, int(round(slot_count)))
    if needed > n:
        return 0.0
    tail = sum(math.comb(n, i) * rate ** i * (1 - rate) ** (n - i)
               for i in range(needed, n + 1))

    # At a streamable position you don't need to have ROSTERED the cover: the
    # week your quarterback is out there are twenty free ones and you start
    # the best matchup among them. So the handful of weeks a backup would
    # nominally be needed are mostly weeks the waiver wire already covers,
    # and holding him costs a roster spot for nothing.
    #
    # Without this the panel kept naming the same backup quarterback round
    # after round in a 1QB league - the one recommendation no real drafter
    # would ever act on. It applies only to bench depth; a position's
    # STARTER is unaffected, which is why elite quarterbacks keep their
    # value here.
    if pos in STREAMABLE_POSITIONS:
        tail *= _STREAMABLE_BENCH_COVERAGE
    return float(np.clip(tail, 0.0, 1.0))


def depth_multiplier(pos, my_roster, slots, absence):
    """
    Multiplier on what one more player at `pos` is worth to THIS roster,
    given how many you already hold.

    Thin wrapper over expected_start_share that works out the depth for you.
    Kept separate so callers never have to get the off-by-one right: the
    player being considered would be the (n+1)th at his position.
    """
    held = sum(1 for p in (my_roster or [])
               if str(p.get('Pos', '')).upper() == str(pos).upper())
    return expected_start_share(str(pos).upper(), held + 1, slots, absence)


def manpower_demand(board, settings, absence=None):
    """
    How many players at each position a league actually CONSUMES over a
    season, as opposed to how many starting slots it has.

    The standard replacement bar answers "who is the last player started
    league-wide this week" and sets replacement at, say, RB24 in a 12-team
    league. That quietly assumes every starter plays every week. They don't -
    a draftable back misses about a fifth of the season - so those 24 starting
    slots have to be filled roughly 24 x 17 times by a pool that is only
    available 79% of the time, and the league really burns through closer to
    thirty running backs. The bar belongs where the last of THOSE sits.

    This is the man-games baseline, the third option alongside VOLS (the last
    dedicated starter) and VORP (the last player started anywhere). It matters
    most where it differs between positions, and its practical effect on this
    board is to lift running backs and receivers relative to quarterbacks and
    tight ends - not because backs are better, but because a season eats more
    of them, which is exactly the reasoning real drafters apply and pure
    starter-count VBD cannot express.

    STREAMABLE POSITIONS ARE DELIBERATELY EXCLUDED, and getting this wrong
    the first time is instructive. Applied naively the adjustment sets
    quarterback replacement at QB16 instead of QB12 and hands every starting
    QB about twenty points of surplus he hasn't earned - undoing the entire
    elite-QB discount this board is careful about. The flaw is in the premise:
    the extra man-games at quarterback are not covered by rostering a
    sixteenth one, they're covered off waivers, because thirty-two teams start
    a QB and twelve are owned. A position whose absences are met from the free
    pool does not need a deeper DRAFTED pool, and its bar is already set by
    what streaming returns.

    What survives is the part with real content: a season consumes far more
    running backs and receivers than it has starting slots, and their absences
    genuinely do have to be covered from your own bench.
    """
    started = compute_starter_demand(board, settings)
    if absence is None:
        absence = measure_absence_rates(settings.get('baseline_season', 2025))
    demand = {}
    for pos, count in started.items():
        if pos in STREAMABLE_POSITIONS:
            demand[pos] = count
            continue
        rate = float(np.clip(absence.get(pos, 0.0) or 0.0, 0.0, 0.6))
        # Bye weeks are excluded on purpose. Everyone has exactly one, so a
        # bye is a slot the league covers by rostering depth it already has,
        # not a reason the player pool has to be deeper.
        demand[pos] = count / max(1e-6, 1.0 - rate)
    return demand


def add_value_over_replacement(board, settings):
    """
    VORP (vs. the first player who would NOT be started anywhere in the
    league) and VOLS (vs. the last player who fills a dedicated starting
    slot at his own position).

    Both are shown because they answer different questions and disagree in
    an informative way. VORP is the honest measure of surplus in a vacuum;
    VOLS is closer to how drafts actually behave, since drafters fill their
    own dedicated slots before they think about flex. A player with high
    VOLS and low VORP is one the room will draft earlier than he's worth.
    """
    if board.empty or 'Proj Pts' not in board.columns:
        return board, {}

    out = board.copy()
    started = compute_starter_demand(out, settings)
    n_teams = int(settings['num_teams'])
    roster = settings['roster']

    # The bar sits at the last player the league actually consumes across a
    # season, not the last one it starts in a given week - see
    # manpower_demand. Switchable so the older behaviour stays reachable, and
    # so the two can be compared rather than argued about.
    demand = dict(started)
    if settings.get('use_manpower_baseline', True):
        demand = {pos: int(round(value))
                  for pos, value in manpower_demand(out, settings).items()}

    replacement, last_starter = {}, {}
    for pos in DRAFTABLE_POSITIONS:
        pos_pts = out.loc[out['Pos'].astype(str).str.upper() == pos, 'Proj Pts'].dropna().sort_values(ascending=False).to_numpy()
        if len(pos_pts) == 0:
            continue
        r_idx = min(demand.get(pos, 0), len(pos_pts) - 1)
        replacement[pos] = float(pos_pts[r_idx])
        ls_idx = min(max(n_teams * int(roster.get(pos, 0)) - 1, 0), len(pos_pts) - 1)
        last_starter[pos] = float(pos_pts[ls_idx])

    # Raise the bar at QB/K/DST to what streaming actually returns. Only
    # ever raises it - if the measured streaming value comes out below the
    # last rostered player (which happens in superflex, where the free pool
    # really is barren), the standard baseline is already the harder test
    # and is kept.
    streaming_used = {}
    if settings.get('use_streaming_baseline', True):
        # ROSTERED depth, not starter count. What decides how deep the waiver
        # wire is, is how many players a league holds - starters plus the
        # bench spent on that position - and passing the starter count made
        # every free pool look about twice as deep as it is.
        depth = rostered_depth(started, settings)
        streaming = build_streaming_replacement(
            settings['scoring'], settings.get('baseline_season', 2025),
            tuple(sorted((p, int(depth.get(p, started.get(p, 0))))
                         for p in STREAMABLE_POSITIONS)),
        )
        for pos, value in (streaming or {}).items():
            if pos in replacement and value > replacement[pos]:
                streaming_used[pos] = round(value, 1)
                replacement[pos] = value

    pos_upper = out['Pos'].astype(str).str.upper()
    out['VORP'] = (out['Proj Pts'] - pos_upper.map(replacement)).round(1)
    out['VOLS'] = (out['Proj Pts'] - pos_upper.map(last_starter)).round(1)
    out['Ceiling VORP'] = (out['Ceiling'] - pos_upper.map(replacement)).round(1)
    meta = {'started': started, 'demand': demand, 'replacement': replacement,
            'last_starter': last_starter, 'streaming_replacement': streaming_used}
    return out, meta


def add_handcuff_value(board, settings, replacement=None):
    """
    Insurance value for a clear backup running back sitting directly behind
    a real starting workload on the same team.

    VORP prices a backup at the touches he's actually expected to get, which
    for a depth-chart #2 back is little enough to land below replacement -
    correct as an expected-role estimate, and the reason a real dart throw
    like a handcuff running back still reads by VORP alone as one of the
    worst players on the board even though ADP keeps drafting him as a
    bench stash. Measured on the August 2026 board: a pure-model rank vs ADP
    comparison showed running backs undervalued by a median 17 spots
    league-wide, growing from +6 inside the realistically-drafted top 10
    rounds to +18 past the bench-stash line - almost entirely this failure
    mode, concentrated in exactly this tier (see docs/draft_hq_methodology.md).

    What VORP alone misses is the same kind of option value
    `contingency_value` already prices for a bench spot on YOUR OWN roster
    (used by data.draft_intel.positional_value_add) - except here the
    "waiver pickup" being insured against is a role transplant, not a
    stream. An NFL backfield concentrates touches on one back far more than
    a passing game concentrates targets, so the back who inherits a
    starter's job doesn't get a marginal bump, he gets a new season. This
    reuses the exact same E[max(X - waiver, 0)] math, pointed at the
    STARTER's own outcome distribution instead of a generic waiver-wire
    player, scaled by the position's measured season-absence rate instead
    of a personal roster's depth.

    Deliberately RB-only, and deliberately restricted to the SINGLE
    next-best back on the team (not the whole depth chart down to the
    fullback). A real handcuff situation has one clear heir; pricing the
    same insurance event again for the 3rd- and 4th-string bodies behind
    him would count the same injury multiple times over. No floor on the
    starter's own workload, either - the E[max(X - waiver, 0)] math below
    already scales down on its own when the starter's own projection is
    modest, so a hard replacement-level gate on him only did one thing in
    practice: it zeroed out backups on a mediocre committee backfield (a
    Rachaad White/Jacory Croskey-Merritt- caliber lead back, real but not
    replacement-clearing) for no principled reason - the backup behind a
    real-if-unspectacular starter still deserves SOME insurance value, just
    a smaller one, which the formula already produces unaided.

    The backup himself still has to be worth nothing on his OWN current
    role (below replacement) - a genuine 1A/1B committee, where both backs
    already clear replacement on their own workload, is already priced
    fairly by VORP alone and isn't what this is for. The identical
    measurement run on WR never showed the underlying pattern at all: a
    WR2 stepping in for an injured WR1 gets a target bump, not a role
    transplant, because passing volume is shared across a receiving corps
    in a way rushing volume isn't concentrated on one back.
    """
    if board.empty or 'VORP' not in board.columns or 'Team' not in board.columns:
        return board
    out = board.copy()
    out['Handcuff Value'] = 0.0
    pos_upper = out['Pos'].astype(str).str.upper()
    rb_mask = pos_upper == 'RB'
    if not rb_mask.any():
        return out

    rb_replacement = float((replacement or {}).get('RB', 0.0))
    absence = measure_absence_rates(
        settings.get('baseline_season', 2025)).get('RB', _DEFAULT_ABSENCE['RB'])

    handcuff_values = {}
    for team, group in out[rb_mask].groupby('Team'):
        grp = group.dropna(subset=['Proj Pts']).sort_values('Proj Pts', ascending=False)
        if len(grp) < 2 or not team:
            continue
        starter, backup = grp.iloc[0], grp.iloc[1]
        # Only a genuine "worth nothing on his own current role" backup. A
        # real committee 1B (already above replacement on his own touches)
        # is already priced fairly by VORP alone, and stacking an insurance
        # value on top of an already-positive VORP would double-count the
        # workload he's already getting.
        if backup['Proj Pts'] >= rb_replacement:
            continue
        value = contingency_value(
            float(starter['Proj Pts']), starter.get('Ceiling'), 'RB',
            share=absence, waiver_points=float(backup['Proj Pts']),
        )
        handcuff_values[backup.name] = round(value, 1)

    if handcuff_values:
        idx = pd.Index(handcuff_values.keys())
        out.loc[idx, 'Handcuff Value'] = pd.Series(handcuff_values)
        out['VORP'] = (out['VORP'] + out['Handcuff Value']).round(1)
    return out


def _kmeans_1d(values, k, iters=60):
    """
    Deterministic 1-D k-means, seeded at evenly spaced quantiles.

    Deterministic seeding matters more than it sounds: tiers that shuffle
    between reruns because of random init would make the board feel broken
    during a draft, when the user is re-reading the same screen between
    picks and needs it to say the same thing each time.
    """
    values = np.asarray(values, dtype=float)
    k = int(max(1, min(k, len(values))))
    if k == 1 or len(values) <= 1:
        return np.zeros(len(values), dtype=int)
    centers = np.quantile(values, np.linspace(0, 1, k))
    labels = np.zeros(len(values), dtype=int)
    for _ in range(iters):
        new_labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            members = values[labels == i]
            if len(members):
                centers[i] = members.mean()
    # Relabel so tier 1 is the most valuable cluster.
    order = np.argsort(-centers)
    remap = {old: new + 1 for new, old in enumerate(order)}
    return np.array([remap[l] for l in labels])


def assign_tiers(board, tiers_per_position=8):
    """
    Cluster each position's DRAFTABLE players into tiers by projected points.

    Tiers are the most-used feature on every good draft board for a reason:
    they convert a continuous ranking into the actual decision. Being 3
    spots apart inside one tier is noise you should ignore in favour of
    positional need; being 3 spots apart across a tier break is a real cliff
    worth reaching for. Clustering on projected POINTS rather than on rank
    is what makes the breaks land where value actually drops instead of
    every N players.

    Only players at or above replacement level get clustered. This matters
    more than it sounds: the board carries ~82 QBs, of whom maybe 20 are
    ever drafted. Clustering all 82 spends the entire cluster budget
    separating grades of worthless backup, and squashes every QB anyone
    would actually draft into one or two tiers - which is precisely where
    the resolution needs to be. Everyone below replacement is swept into a
    single trailing tier, since the distinction between QB58 and QB61 is not
    a draft decision anyone makes.
    """
    if board.empty or 'Proj Pts' not in board.columns:
        return board
    out = board.copy()
    out['Tier'] = np.nan
    pos_upper = out['Pos'].astype(str).str.upper()
    has_vorp = 'VORP' in out.columns and out['VORP'].notna().any()

    for pos in pos_upper.unique():
        pos_mask = (pos_upper == pos) & out['Proj Pts'].notna()
        if not pos_mask.any():
            continue
        if has_vorp:
            relevant = pos_mask & (out['VORP'] >= 0)
        else:
            relevant = pos_mask
        # Guard against a position where nothing clears replacement (can
        # happen for K/DST in odd roster configs) - fall back to the whole
        # position rather than producing a board with no tiers at all.
        if relevant.sum() < 3:
            relevant = pos_mask

        vals = out.loc[relevant, 'Proj Pts'].to_numpy()
        k = int(np.clip(len(vals) // 3, 3, tiers_per_position))
        labels = _kmeans_1d(vals, k)
        out.loc[relevant, 'Tier'] = labels
        below = pos_mask & ~relevant
        if below.any():
            out.loc[below, 'Tier'] = int(labels.max()) + 1

    out['Tier'] = pd.to_numeric(out['Tier'], errors='coerce')
    return out


def tier_by_position(df, value_col, pos_col='Pos', max_tiers=6):
    """
    Per-position tiers by clustering `value_col`, wherever a significant
    cutoff actually occurs - the same `_kmeans_1d` clustering assign_tiers
    uses for the draft board, generalized for any per-position points
    table rather than one that also needs VORP/replacement-level filtering.

    Unlike assign_tiers, every row with a real value participates (there is
    no "below replacement" concept for a single week's projected pool - a
    streaming DST or QB2 spot start is a legitimate tier, not noise to sweep
    into a trailing bucket).

    Returns a Series of tier ints aligned to df's index, NaN where `pos_col`
    or `value_col` is missing for that row.
    """
    if df.empty or value_col not in df.columns or pos_col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    out = pd.Series(np.nan, index=df.index)
    pos_upper = df[pos_col].astype(str).str.upper()
    for pos in pos_upper.unique():
        mask = (pos_upper == pos) & df[value_col].notna()
        if not mask.any():
            continue
        vals = df.loc[mask, value_col].to_numpy()
        k = int(np.clip(len(vals) // 3, 1, max_tiers))
        out.loc[mask] = _kmeans_1d(vals, k)
    return out


def attach_adp(board, adp_df, market_weight=0.0):
    """
    Join market ADP onto the board and derive the value gap.

    'Value vs ADP' is positive when the market drafts a player LATER than
    this board ranks him - i.e. he'll be there when you want him, and
    taking him at his ADP is free surplus. That framing (board rank minus
    market pick) is deliberately the one that answers "is this a value?",
    not the reverse sign convention some sites use.
    """
    if board.empty:
        return board
    out = board.copy()
    # A player whose position has no value curve (DST when the team-stats
    # source is unreachable) still needs a board rank so he stays draftable
    # and trackable - he just sorts behind everyone who could be valued,
    # ordered among his peers by consensus rank. Int64 (nullable) rather
    # than int because the fallback can still be empty on a board with no
    # ECR at all, and a hard astype(int) there raises instead of degrading.
    if 'VORP' in out.columns and out['VORP'].notna().any():
        primary = out['VORP'].rank(ascending=False, method='first')
        n_valued = int(out['VORP'].notna().sum())
        fallback = out['ECR'].rank(method='first') + n_valued
        out['Board Rank'] = primary.fillna(fallback)
    else:
        out['Board Rank'] = out['ECR'].rank(method='first')
    out['Board Rank'] = out['Board Rank'].round().astype('Int64')

    # Positional rank as a readable label ("RB3", "QB1"). Raw Proj Pts are
    # not comparable across positions - every QB outscores every running
    # back, so a points column alone makes the board look like it thinks
    # the QB1 is the best player alive. This is the cheapest possible guard
    # against reading it that way: it puts each projection back in the only
    # context where it means anything.
    out['Pos Rk'] = out['Pos'].astype(str).str.upper() + out['Pos Rank'].astype(str)

    if adp_df is None or adp_df.empty:
        # No market data. ADP and Value-vs-ADP stay genuinely blank - there
        # is no market to compare against and inventing one would fake the
        # single most useful column on the board.
        #
        # But the AVAILABILITY model still needs some estimate of when each
        # player comes off the board, and without one the whole board becomes
        # availability-blind - which is the specific failure that makes
        # value-based drafting rank an elite quarterback around pick 7. His
        # surplus over the last starting QB is real; what VORP cannot see is
        # that a quarterback worth 90% as much is still sitting there in
        # round 9, so the true cost of waiting is a fraction of that surplus.
        # Consensus rank is a decent proxy for draft order (it is, after all,
        # what most of the room is reading), so it fills in here - flagged as
        # an estimate, with a wider spread than measured ADP would carry.
        out['ADP'] = np.nan
        out['ADP SD'] = np.nan
        out['Value vs ADP'] = np.nan
        out['Expected Pick'] = out['ECR'].rank(method='first')
        out['Expected Pick SD'] = np.maximum(4.0, 0.32 * out['Expected Pick'])
        out['Pick Source'] = 'ECR estimate'
        out = apply_market_blend(out, market_weight)
        return out

    from data.utils import clean_name_exact, clean_name_for_merge
    adp = adp_df.copy()
    adp['_exact'] = clean_name_exact(adp['Player'])
    adp['_loose'] = clean_name_for_merge(adp['Player'])
    exact_map = dict(zip(adp['_exact'], adp['ADP']))
    loose_map = dict(zip(adp['_loose'], adp['ADP']))
    sd_exact = dict(zip(adp['_exact'], adp['ADP SD'])) if 'ADP SD' in adp.columns else {}
    sd_loose = dict(zip(adp['_loose'], adp['ADP SD'])) if 'ADP SD' in adp.columns else {}

    b_exact = clean_name_exact(out['Player'])
    b_loose = clean_name_for_merge(out['Player'])
    out['ADP'] = b_exact.map(exact_map).fillna(b_loose.map(loose_map))
    out['ADP SD'] = b_exact.map(sd_exact).fillna(b_loose.map(sd_loose)) if sd_exact else np.nan

    # Real ADP where the market covers a player, consensus rank as the
    # estimate where it doesn't - a 262-player ADP feed leaves the back half
    # of the board with no draft-position estimate at all otherwise, and
    # those are exactly the late-round players whose availability decides
    # whether you can wait on them.
    ecr_rank = out['ECR'].rank(method='first')
    has_adp = pd.to_numeric(out['ADP'], errors='coerce').notna()
    out['Expected Pick'] = pd.to_numeric(out['ADP'], errors='coerce').fillna(ecr_rank)
    published = pd.to_numeric(out['ADP SD'], errors='coerce')
    # Two different spreads, because they describe two different things. A
    # real ADP carries roughly 0.22x its own value in observed stdev across
    # the published feeds - that's measured, and it's the number the
    # availability model wants. A player with no ADP at all is being placed
    # by consensus rank standing in for a draft slot, which is a strictly
    # vaguer claim, so it gets a visibly wider band rather than borrowing the
    # measured one and inheriting confidence it hasn't earned.
    estimated = np.where(has_adp,
                         np.maximum(2.5, 0.22 * out['Expected Pick']),
                         np.maximum(4.0, 0.32 * out['Expected Pick']))
    estimated = pd.Series(estimated, index=out.index)
    out['Expected Pick SD'] = published.where(published.notna() & (published > 0), estimated)
    out['Pick Source'] = np.where(pd.to_numeric(out['ADP'], errors='coerce').notna(),
                                  'ADP', 'ECR estimate')

    out = apply_market_blend(out, market_weight)
    out['Value vs ADP'] = (out['ADP'] - out['Board Rank']).round(1)
    return out


def apply_market_blend(board, market_weight):
    """
    Pull the board's ordering toward the market by a chosen amount.

    Value-based drafting and the market disagree most sharply at
    quarterback: with replacement set at the last starting QB, VBD prices an
    elite QB as a top-15 overall pick in a 1QB league, and real drafts -
    sharp ones especially - take him twenty-plus picks later. That gap is
    genuine disagreement, not an error in either direction. Analysts have
    argued the elite-QB discount is a market inefficiency for years, and the
    market has gone on discounting them anyway.

    This board does not pretend to settle it. VORP stays exactly what the
    model says - it is never blended - and this only moves the ORDER the
    board is presented in, so you can decide how much deference the market
    gets. At 0 you draft the model; at 100 you draft ADP; in between you get
    a board that respects value while refusing to reach three rounds ahead
    of the room for it.

    Blending happens in rank space rather than on points because the two
    inputs have no common unit - projected points and average draft position
    aren't convertible, but their orderings are directly comparable.
    """
    out = board.copy()
    weight = float(market_weight or 0)
    if weight <= 0 or 'Board Rank' not in out.columns:
        return out
    # Blend toward whatever market signal exists: real ADP when we have it,
    # consensus rank otherwise. Falling back to ECR matters because the
    # blend is the board's main defence against out-thinking the entire
    # analyst industry on the strength of one model, and switching it off
    # entirely whenever the ADP feed is down is exactly backwards.
    market = pd.to_numeric(out.get('Expected Pick'), errors='coerce')
    if market.isna().all():
        market = pd.to_numeric(out.get('ADP'), errors='coerce')
    if market.notna().sum() < 10:
        return out

    model_rank = out['Board Rank'].astype(float)
    adp_rank = market.rank(method='first').fillna(model_rank)

    blended = (1 - weight) * model_rank + weight * adp_rank
    out['Board Rank'] = blended.rank(method='first').round().astype('Int64')
    return out


def demote_late_round_positions(board, settings):
    """
    Sort kickers and team defenses behind every offensive player.

    NOT a fudge to make the board look conventional - it corrects a real
    mismatch between what VORP measures and what a draft actually offers.
    Measured honestly, the best kicker really does carry ~40 points of
    surplus over the last startable one, which is comparable to a WR3's
    surplus, and value-based drafting will therefore slot him around pick
    50. Every measurement thrown at this agrees: streaming a kicker returns
    LESS than rostering a good one, and kicker year-over-year persistence
    (0.23) is no worse than receiver (0.16). The surplus is real.

    What makes it undraftable at pick 50 is the draft's structure, not the
    projection: you are never choosing between the best kicker and a third
    receiver, because the kicker will still be available 130 picks later and
    the receiver will not. Ranking them against each other answers a
    question no drafter ever faces.

    So their value stays fully visible in the VORP and VONA columns - the
    model is not being overruled, just told when the choice is actually
    live. Positions the league doesn't even start are dropped further still.
    """
    if board.empty or 'Board Rank' not in board.columns:
        return board
    out = board.copy()
    roster = settings.get('roster', {})
    pos_upper = out['Pos'].astype(str).str.upper()

    offset = pd.Series(0.0, index=out.index)
    horizon = float(len(out))
    for pos in LATE_ROUND_POSITIONS:
        mask = pos_upper == pos
        if not mask.any():
            continue
        # A position the league never starts isn't merely late, it's
        # undraftable - pushed behind even the ones that are.
        offset[mask] = horizon * (1.0 if int(roster.get(pos, 0)) > 0 else 2.0)

    out['Board Rank'] = (out['Board Rank'].astype(float) + offset).rank(
        method='first').round().astype('Int64')
    return out


def effective_adp_sd(board):
    """
    Standard deviation of a player's actual draft slot, for the availability
    model.

    Uses the market's own observed stdev where the ADP source publishes it
    (Fantasy Football Calculator does, from real drafts - that's the honest
    number). Where it's missing, it's estimated as growing with the expected
    pick, because early picks are far more predictable than late ones: nobody
    is unsure whether the consensus 1.01 goes in round one, but a player
    expected around 120 routinely goes anywhere from 90 to undrafted.
    """
    if 'Expected Pick SD' in board.columns:
        sd = pd.to_numeric(board['Expected Pick SD'], errors='coerce')
        if sd.notna().any():
            return sd
    adp = pd.to_numeric(board.get('ADP'), errors='coerce')
    published = pd.to_numeric(board.get('ADP SD'), errors='coerce')
    estimated = np.maximum(2.5, 0.22 * adp)
    return published.where(published.notna() & (published > 0), estimated)


def _conditional_survival(z_next, z_now):
    """
    P(draft slot >= next pick | draft slot >= the pick on the clock).

    The plain ratio of the two survival probabilities, computed in log space
    once both are deep enough in the tail that the ratio would otherwise be
    0/0. For large z the normal survival function is well approximated by
    phi(z)/z, so the ratio becomes (z_now/z_next) * exp(-(z_next^2 - z_now^2)/2),
    which stays finite exactly where the direct division stops being.
    """
    p_next = vectorized_normal_sf(z_next)
    p_now = vectorized_normal_sf(z_now)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(p_now > 1e-9, p_next / np.where(p_now > 0, p_now, 1.0), np.nan)
        tail_ok = (z_now > 0) & (z_next > 0)
        tail = np.exp(np.clip(-(z_next ** 2 - z_now ** 2) / 2.0, -700, 0)) * np.where(
            tail_ok, z_now / np.where(z_next > 0, z_next, 1.0), 0.0)
        ratio = np.where(np.isfinite(ratio), ratio, np.where(tail_ok, tail, 0.0))
    return np.clip(ratio, 0.0, 1.0)


def add_availability(board, next_pick, current_pick=None):
    """
    P(this player is still on the board at your next pick).

    Models each player's draft slot as normally distributed around his ADP
    with the spread from effective_adp_sd, so the probability he lasts to
    pick N is the chance his draft slot lands at or beyond N.

    CONDITIONED ON WHERE THE DRAFT ACTUALLY IS, which is the part that makes
    it usable rather than decorative. The unconditional form asks "what are
    the odds this player goes after pick N", and for anyone whose ADP is
    already past N that answer is ~100% no matter how much of the draft has
    happened - so ten rounds into a draft the entire remaining board reads
    100% and the column tells you nothing. What a drafter is actually asking
    is "given he's STILL HERE at pick 109, does he last to my pick at 121",
    and that conditional is a completely different (and much sharper) number:
    every pick he survives is evidence about where he'll go, and the model
    should spend it.

    This is the column that changes how you draft. Two players with nearly
    identical value are not the same pick if one is 85% to come back to you
    and the other is 12%; taking the 85% one now is how you end up with
    both, and no amount of staring at a ranked list tells you that.
    """
    if board.empty or next_pick is None:
        out = board.copy()
        out['Avail Next %'] = np.nan
        return out
    out = board.copy()
    expected = pd.to_numeric(out.get('Expected Pick'), errors='coerce')
    if expected.isna().all():
        expected = pd.to_numeric(out.get('ADP'), errors='coerce')
    sd = effective_adp_sd(out).replace(0, np.nan)

    z_next = ((float(next_pick) - expected) / sd).to_numpy(dtype=float)
    if current_pick is None or float(current_pick) <= 1:
        survival = vectorized_normal_sf(z_next)
    else:
        # The pick on the clock is where "still available" is measured from.
        # min() with next_pick guards the degenerate case of a next pick that
        # has already passed, which would otherwise report a probability
        # above 1.
        now = min(float(current_pick), float(next_pick))
        z_now = ((now - expected) / sd).to_numpy(dtype=float)
        survival = _conditional_survival(z_next, z_now)
    out['Avail Next %'] = (survival * 100).round(0)
    return out


def refresh_pick_context(board, next_pick, current_pick=None, value_col='VORP'):
    """
    Recompute the two columns that depend on where the draft is, for a board
    that was assembled earlier.

    The board itself is expensive and cached against league settings; these
    two are cheap and go stale every single pick. Splitting them out is what
    lets a mock draft in round 10 show honest availability against a board
    that was built before the draft started - previously the whole surface
    inherited the pick numbers from whenever the board happened to be
    assembled, which in a mock meant it never advanced at all.

    Expects `board` to already have drafted players removed, since both
    columns are statements about who is left.
    """
    if board is None or board.empty:
        return board
    out = add_availability(board, next_pick, current_pick=current_pick)
    return compute_vona(out, next_pick, value_col=value_col)


def compute_vona(board, next_pick, value_col='VORP'):
    """
    Value Over Next Available: what you actually lose by NOT taking this
    player right now.

    For each position, the expected best player still available at your next
    pick is computed as a proper expectation - walking the position's board
    in value order and weighting each player by the chance he is both
    available AND the best one available (i.e. everyone above him is gone).
    A player's VONA is his value minus that expectation.

    This is the number that resolves the classic draft-room paralysis. A
    stud RB with a VONA of 40 and a WR with a VONA of 4 is not a close call
    even if the WR grades out a few points higher in isolation: the WR (or
    someone within four points of him) is coming back to you, and the RB is
    not. Positional runs, tier cliffs and scarcity all fall out of this one
    calculation instead of needing to be eyeballed separately.
    """
    if board.empty or value_col not in board.columns:
        return board
    out = board.copy()
    out['VONA'] = np.nan
    if next_pick is None or 'Avail Next %' not in out.columns:
        return out
    # Without availability data every player's survival probability is zero,
    # which makes "expected best available at your next pick" zero, which
    # makes VONA collapse to exactly VORP for every row. That's not a
    # degraded answer, it's a wrong one wearing a different column name -
    # it would read as "nobody comes back to you" when the truth is "we
    # don't know". Left blank instead, and the UI says why.
    if out['Avail Next %'].notna().sum() == 0:
        return out

    for pos in out['Pos'].astype(str).str.upper().unique():
        mask = (out['Pos'].astype(str).str.upper() == pos) & out[value_col].notna()
        if not mask.any():
            continue
        sub = out.loc[mask].sort_values(value_col, ascending=False)
        p_avail = (pd.to_numeric(sub['Avail Next %'], errors='coerce').fillna(0) / 100.0).to_numpy()
        vals = sub[value_col].to_numpy(dtype=float)
        # P(i is the best available) = P(i available) * prod_{j<i} P(j gone)
        gone_cumprod = np.cumprod(np.concatenate([[1.0], 1.0 - p_avail[:-1]]))
        weights = p_avail * gone_cumprod
        expected_next = float((weights * vals).sum())
        # Whatever probability mass is left over is "nobody at this position
        # survives", which is worth the replacement-level value of 0 by
        # construction (VORP is already measured against replacement), so no
        # extra term is needed here.
        out.loc[sub.index, 'VONA'] = (vals - expected_next).round(1)
    return out


def add_risk_labels(board):
    """
    Upside / Bust / Risk, all derived from the same outcome distribution the
    projection came from rather than invented separately.

    Upside and Bust are the gap from the projection to the 85th/15th
    percentile outcomes, expressed as points. Risk is the width of that
    band relative to the projection - a normalized read on how much of this
    pick is a coin flip, which is what actually differs between a boring
    reliable veteran and a rookie the industry can't agree on.
    """
    if board.empty or 'Ceiling' not in board.columns:
        return board
    out = board.copy()
    out['Upside'] = (out['Ceiling'] - out['Proj Pts']).round(1)
    out['Bust'] = (out['Proj Pts'] - out['Floor']).round(1)
    spread = (out['Ceiling'] - out['Floor'])
    denom = out['Proj Pts'].abs().clip(lower=1)
    out['Risk'] = (spread / denom * 100).round(0)
    return out


def build_draft_board(ecr_board, settings, adp_df=None, next_pick=None,
                      recent_df=None, recent_weight=0.0, tiers_per_position=8,
                      rank_sd_scale=1.0, latest_season=2025, market_weight=0.0):
    """
    The whole pipeline, in the order the pieces depend on each other.

    Ordering is load-bearing: projections must exist before replacement
    level can be computed (replacement level is defined in terms of
    projected points), replacement level must exist before VORP, VORP
    before VONA, and ADP must be attached before availability - which VONA
    in turn needs. Reordering any of these
    silently produces a board full of NaN rather than an error, which is
    exactly the kind of failure that would go unnoticed until draft night.
    """
    if ecr_board is None or ecr_board.empty:
        return pd.DataFrame(), {}

    settings = {**settings, 'baseline_season': latest_season}
    calibration = calibrate_rank_uncertainty(settings['scoring'], latest_season)

    # A board arriving with Proj Pts already filled in came from the
    # volume-based stat-line projections (data.draft_projections) or an FFA
    # import, and re-deriving points from consensus rank here would throw
    # that away and replace a real projection with an analogy. In that case
    # only the outcome RANGE is added on top.
    has_projection = 'Proj Pts' in ecr_board.columns and ecr_board['Proj Pts'].notna().any()
    if has_projection:
        board = add_outcome_range_from_projections(
            ecr_board, calibration=calibration, rank_sd_scale=rank_sd_scale)
    else:
        curves = build_positional_value_curves(settings['scoring'], latest_season)
        board = project_points(ecr_board, curves, rank_sd_scale=rank_sd_scale, calibration=calibration)
    board = blend_with_recent_production(board, recent_df, recent_weight)
    # Tiering runs AFTER replacement level, not before, because it needs to
    # know who is actually draftable - see assign_tiers on why clustering
    # the undraftable tail wrecks resolution where it matters.
    board, meta = add_value_over_replacement(board, settings)
    board = add_handcuff_value(board, settings, replacement=meta.get('replacement'))
    board = assign_tiers(board, tiers_per_position=tiers_per_position)
    board = add_risk_labels(board)
    board = attach_adp(board, adp_df, market_weight=market_weight)
    board = add_availability(board, next_pick)
    board = compute_vona(board, next_pick)
    board = demote_late_round_positions(board, settings)
    meta['has_curves'] = True
    meta['projection_source'] = 'stat line' if has_projection else 'rank curve'
    meta['rank_sd_calibration'] = calibration
    return board.sort_values('VORP', ascending=False).reset_index(drop=True), meta


# ---------------------------------------------------------------------------
# Roster construction & recommendations
# ---------------------------------------------------------------------------

def roster_needs(my_roster, settings):
    """
    Which starting slots are still empty, given who you've already drafted.

    Slots are filled in the order that leaves the most flexibility: a drafted
    RB fills a dedicated RB slot before it fills FLEX, so that a roster with
    RB/RB/WR doesn't misreport its flex as already spoken for while a real
    starting slot sits empty.
    """
    roster = settings['roster']
    counts = {}
    for pos in DRAFTABLE_POSITIONS:
        counts[pos] = sum(1 for p in my_roster if str(p.get('Pos', '')).upper() == pos)

    needs = {}
    leftover = {}
    for pos in DRAFTABLE_POSITIONS:
        want = int(roster.get(pos, 0))
        have = counts.get(pos, 0)
        needs[pos] = max(0, want - have)
        leftover[pos] = max(0, have - want)

    flex_need = int(roster.get('FLEX', 0)) - sum(leftover.get(p, 0) for p in FLEX_ELIGIBLE)
    superflex_need = int(roster.get('SUPERFLEX', 0)) - max(0, sum(leftover.get(p, 0) for p in SUPERFLEX_ELIGIBLE) - int(roster.get('FLEX', 0)))
    needs['FLEX'] = max(0, flex_need)
    needs['SUPERFLEX'] = max(0, superflex_need)
    return needs


# Bench copies of a position anyone would actually carry, beyond their
# starting slots. Mirrors data.draft_sim.BENCH_DEPTH_ALLOWANCE so the advice
# and the simulated opponents obey one rule: one backup quarterback and one
# backup tight end, none at kicker or defense, and real depth only at running
# back and receiver where injuries and byes are covered from your own bench.
BENCH_DEPTH_ALLOWANCE = {'QB': 1, 'RB': 4, 'WR': 4, 'TE': 1, 'K': 0, 'DST': 0}


def _roster_cap(pos, settings):
    """The most of `pos` anyone would sensibly roster in this league."""
    roster = settings['roster']
    starters = float(roster.get(pos, 0))
    if pos in SUPERFLEX_ELIGIBLE:
        starters += float(roster.get('SUPERFLEX', 0))
    if pos in FLEX_ELIGIBLE:
        # A flex slot is one claim shared across three positions, so it
        # counts as a third of a slot each rather than a whole one for all.
        starters += float(roster.get('FLEX', 0)) / len(FLEX_ELIGIBLE)
    return max(1, int(round(starters + BENCH_DEPTH_ALLOWANCE.get(pos, 1))))


def _need_multiplier(pos, needs, settings, depth_share=None):
    """
    How much to weight a position by what your roster still needs.

    Deliberately gentle while a slot is still open (a 25% nudge, not a hard
    filter). Positional need is real but drafting purely for need is how
    people end up reaching two rounds early for a kicker in round 9 - the
    value model should still be driving, with need breaking ties between
    comparable players.

    ONCE THE SLOT IS FULL the gentleness stops, because a flat 0.92 there
    was wrong in a way that showed: with a starting quarterback already
    rostered, the second-best QB on the board kept out-scoring every real
    option and the panel recommended the same backup quarterback in eight
    consecutive rounds. Past the starting lineup the question isn't "how
    good is he" but "how often would he be in my lineup", and
    expected_start_share answers that - roughly a third for a QB2, a
    quarter for a TE2, nothing at all for a third tight end.
    """
    direct = needs.get(pos, 0)
    flexable = needs.get('FLEX', 0) if pos in FLEX_ELIGIBLE else 0
    superflexable = needs.get('SUPERFLEX', 0) if pos in SUPERFLEX_ELIGIBLE else 0
    if direct > 0:
        return 1.25
    if flexable > 0 or superflexable > 0:
        return 1.10
    if depth_share is None:
        return 0.92
    # Floored just above zero rather than at it, so a position that can
    # never start again still sorts by value among its own kind instead of
    # collapsing into an arbitrary tie.
    return float(max(0.02, min(0.92, depth_share)))


def recommend_picks(board, my_roster, settings, next_pick=None, top_n=6, value_col='VONA',
                    allow_late_round=False):
    """
    Rank the available players by what they're worth to YOUR roster right
    now, and say why.

    The score is the value model (VONA if a next pick is known, VORP
    otherwise) adjusted for what your lineup still needs, whether the pick
    would stack a bye week you're already thin on, and whether the player
    sits at the bottom of a tier - i.e. the last chance to get this quality
    before a real cliff.

    Every adjustment is surfaced in the returned 'Why' string. A
    recommendation you can't audit is one you won't trust at pick 4 when it
    disagrees with you, and a draft tool you don't trust is one you ignore.
    """
    if board.empty:
        return pd.DataFrame()

    needs = roster_needs(my_roster, settings)
    my_byes = [p.get('Bye') for p in my_roster if p.get('Bye') and not pd.isna(p.get('Bye'))]
    bye_counts = pd.Series(my_byes).value_counts().to_dict() if my_byes else {}

    avail = board.copy()
    base_col = value_col if value_col in avail.columns and avail[value_col].notna().any() else 'VORP'
    avail = avail[avail[base_col].notna()]
    if avail.empty:
        return pd.DataFrame()

    # Kickers and defenses are filtered out of suggestions entirely until
    # every other starting slot is filled. Suggesting a defense as your third
    # best option is never right: it will still be there in twelve rounds,
    # and the receiver it's competing with will not.
    # allow_late_round is set when the caller has explicitly asked for a
    # kicker or defense (the K/DST position filter). Suppressing them there
    # would leave the panel permanently empty for those buttons, which is a
    # worse answer than the one the suppression exists to prevent.
    starters_left = sum(n for pos, n in needs.items()
                        if pos not in LATE_ROUND_POSITIONS and n > 0)

    # Kickers and defenses are held back until the LAST FEW PICKS, not merely
    # until the skill starters are filled.
    #
    # The old test - "are any non-K/DST starting slots still open" - goes
    # false around round 8 in a normal 15-round draft, and from that moment a
    # kicker outscored everything left on the board: his slot is empty so he
    # gets the open-slot bonus, and his surplus over replacement is large next
    # to the bench receivers he's competing with. A recommendation-driven mock
    # took Brandon Aubrey at pick 101 and the Texans defense at 116, five
    # rounds before anyone sane takes either.
    #
    # The simulator's bots have always known better (LATE_ROUND_ONLY in
    # data.draft_sim); this brings the advice into line with them. The gate
    # opens once you have barely enough picks left to fill those slots, which
    # is exactly when a real drafter starts thinking about them.
    total_rounds = sum(int(settings['roster'].get(k, 0)) for k in
                       ['QB', 'RB', 'WR', 'TE', 'K', 'DST', 'FLEX', 'SUPERFLEX', 'BENCH'])
    picks_left = max(0, total_rounds - len(my_roster or []))
    late_slots_needed = sum(int(needs.get(pos, 0)) for pos in LATE_ROUND_POSITIONS)
    late_round_time = picks_left <= late_slots_needed + LATE_ROUND_GRACE
    # Blocked outright once the slot is filled, rather than merely discounted.
    # A second kicker is never the right pick in any league that starts one -
    # you stream the position - and with a full roster in the final round
    # every position's depth share collapses to about the same small number,
    # so nothing else was left to stop it. A mock took Brandon Aubrey in round
    # 13 and Jake Bates in round 15.
    blocked = [pos for pos in LATE_ROUND_POSITIONS
               if int(needs.get(pos, 0)) <= 0 or starters_left > 0 or not late_round_time]

    # Nobody rosters more of a position than they could conceivably use, and
    # the simulator's bots have always known it (_position_limits in
    # data.draft_sim). The advice didn't, and drifted past the same line the
    # bots respect: recommendation-driven mocks came out holding three
    # quarterbacks in a one-QB league, because with a full roster every
    # position's option value shrinks to roughly the same small number and
    # the ordering between them becomes noise. Capping is the honest fix -
    # the model genuinely cannot tell those picks apart, so the constraint
    # has to come from roster construction rather than from valuation.
    held = pd.Series([str(p.get('Pos', '')).upper() for p in (my_roster or [])],
                     dtype=object).value_counts().to_dict()
    for pos in DRAFTABLE_POSITIONS:
        if held.get(pos, 0) >= _roster_cap(pos, settings):
            blocked.append(pos)

    if blocked and not allow_late_round:
        avail = avail[~avail['Pos'].astype(str).str.upper().isin(set(blocked))]
        if avail.empty:
            return pd.DataFrame()

    # How much of a season one more player at each position would actually
    # spend in this roster's starting lineup. Computed once for the whole
    # loop rather than per row, since it depends only on what you hold.
    slots = lineup_slots_per_team(board, settings)
    absence = measure_absence_rates(settings.get('baseline_season', 2025))
    depth_shares = {p: depth_multiplier(p, my_roster, slots, absence)
                    for p in DRAFTABLE_POSITIONS}

    # Positions you effectively cannot play another of are dropped outright,
    # not merely scaled down. Scaling alone wasn't enough: VONA measures what
    # you lose by WAITING, and for the last decent quarterback on the board
    # that number is large - nobody comparable is coming back - so a heavily
    # discounted QB2 still out-scored every real option and the panel named a
    # backup quarterback in six straight rounds of a 1QB league.
    #
    # The premise VONA rests on is false there: you lose nothing by waiting
    # on a player who would never enter your lineup. Any position with a real
    # starting need is exempt, the filter is skipped when it would empty the
    # panel, and asking for a position explicitly (the QB/TE buttons) still
    # shows you its best available.
    if not allow_late_round:
        playable = avail[avail['Pos'].astype(str).str.upper().map(
            lambda p: needs.get(p, 0) > 0 or depth_shares.get(p, 1.0) >= 0.1)]
        if not playable.empty:
            avail = playable

    # What a free pickup at each position is worth, for the option value
    # below. Measured off the pool actually left rather than a constant, so
    # it falls through the draft exactly as the waiver wire does.
    waiver_points = {}
    for pos in DRAFTABLE_POSITIONS:
        pos_points = board.loc[board['Pos'].astype(str).str.upper() == pos, 'Proj Pts'].dropna()
        if pos_points.empty:
            continue
        ordered = pos_points.sort_values(ascending=False).to_numpy()
        index = min(int(round(slots.get(pos, 1) * settings['num_teams'])), len(ordered) - 1)
        waiver_points[pos] = float(ordered[index])

    # Once every starting slot at a position is full the question stops being
    # "what do I lose by waiting" and becomes "what is this bench spot worth",
    # and those need different arithmetic. Left on VONA, the model kept
    # naming backup quarterbacks: with a full roster every position's depth
    # share collapses to roughly the same small number, the multiplier stops
    # discriminating, and whoever has the biggest raw VONA wins - which late
    # in a draft is the best remaining QB, a player who would never start.
    # Option value separates them properly, because it prices the gap to a
    # free replacement rather than to the next man at his own position, and
    # that gap is tiny at a streamable position and real at receiver.
    use_option_value = starters_left == 0

    rows = []
    for _, r in avail.nlargest(min(60, len(avail)), base_col).iterrows():
        pos = str(r['Pos']).upper()
        base = float(r[base_col])
        reasons = []

        mult = _need_multiplier(pos, needs, settings, depth_shares.get(pos))
        if use_option_value and needs.get(pos, 0) <= 0:
            score = contingency_value(r.get('Proj Pts'), r.get('Ceiling'), pos,
                                      depth_shares.get(pos, 0.0),
                                      waiver_points.get(pos, 0.0))
            base = max(base, 1.0)      # only used to scale the nudges below
        else:
            # Scaling a NEGATIVE value by a fraction makes it bigger, which
            # would promote exactly the players the multiplier is meant to
            # bury. Applied around zero instead, so shrinking always moves a
            # score toward zero whichever side of it the player sits on.
            score = base * mult if base >= 0 else base / max(mult, 0.02)
        if mult > 1.0:
            reasons.append(f"fills an open {pos} slot" if needs.get(pos, 0) > 0 else "flex-eligible need")
        elif mult < 0.5:
            reasons.append(f"{pos} full — a backup starts ~{depth_shares.get(pos, 0):.0%} of weeks")
        elif mult < 1.0:
            reasons.append(f"{pos} already covered")

        avail_pct = r.get('Avail Next %')
        if pd.notna(avail_pct):
            if avail_pct < 25:
                score += abs(base) * 0.12
                reasons.append(f"only {avail_pct:.0f}% to last to your next pick")
            elif avail_pct > 75:
                score -= abs(base) * 0.10
                reasons.append(f"{avail_pct:.0f}% likely to come back to you")

        bye = r.get('Bye')
        if pd.notna(bye) and bye_counts.get(bye, 0) >= 2:
            score -= abs(base) * 0.05
            reasons.append(f"3rd player on bye {int(bye)}")

        tier = r.get('Tier')
        if pd.notna(tier):
            same_tier = avail[(avail['Pos'].astype(str).str.upper() == pos) & (avail['Tier'] == tier)]
            if len(same_tier) == 1:
                score += abs(base) * 0.08
                reasons.append(f"last player in {pos} tier {int(tier)}")

        v_adp = r.get('Value vs ADP')
        if pd.notna(v_adp) and v_adp >= 12:
            reasons.append(f"going {v_adp:.0f} picks later than he's worth")

        rows.append({
            'Player': r['Player'], 'Pos': pos, 'Team': r.get('Team'),
            'Tier': tier, 'Proj Pts': r.get('Proj Pts'), 'VORP': r.get('VORP'),
            'VONA': r.get('VONA'), 'ADP': r.get('ADP'), 'Avail Next %': avail_pct,
            'Fit Score': round(score, 1), 'Why': "; ".join(reasons) if reasons else "best value available",
        })

    return pd.DataFrame(rows).sort_values('Fit Score', ascending=False).head(top_n).reset_index(drop=True)


def snake_pick_numbers(draft_slot, num_teams, rounds, draft_type='Snake'):
    """
    Every overall pick number you own, given your slot.

    Needed for the availability model to know what "your next pick" is, and
    it's genuinely different between formats: in a snake draft from slot 3
    of 12 you pick 3, 22, 27, 46... - the gap alternates between 18 and 6,
    so "will he last?" has a wildly different answer depending on which side
    of the turn you're on. That asymmetry is the whole reason turn strategy
    exists, and a tool that assumed even gaps would get it backwards half
    the time.
    """
    slot = int(draft_slot)
    n = int(num_teams)
    picks = []
    for rnd in range(1, int(rounds) + 1):
        if draft_type == 'Snake' and rnd % 2 == 0:
            picks.append((rnd - 1) * n + (n - slot + 1))
        else:
            picks.append((rnd - 1) * n + slot)
    return picks


def next_pick_for(picks, picks_made):
    """The first of your pick numbers that hasn't happened yet."""
    for p in picks:
        if p > picks_made:
            return p
    return None
