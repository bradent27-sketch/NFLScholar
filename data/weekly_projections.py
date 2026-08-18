"""
This app's own weekly fantasy-points projection model - not FantasyPros'
number, and not a bare average of past fantasy points.

WHAT THIS IS FOR. Player Search already has a next-game projection
(`data.transforms.build_player_projection`) for looking up ONE player at a
time. This module projects the WHOLE skill-position pool for one week at
once, for the Weekly Rankings tab, and goes further than that single-player
model in the inputs it blends: current-vs-prior-season usage (weighted
toward the current season as its own sample grows), snap-share/route-share
role confidence, opponent-allowed rates, pace, and a game-script read
against the Vegas-implied spread for the target week - not just a recent-
form/season-average blend times a matchup multiplier.

DELIBERATELY BUILT BY COMPOSITION, NOT FROM SCRATCH. Every external signal
here reuses a primitive this app already has, tested and in production
elsewhere, rather than re-deriving it:
  - pace: data.loaders.load_team_pace (same source)
  - game script: the SAME bucket edges data.matchup_signals.
    game_script_sensitivity_curve uses (Trailed big / Lost close / Won
    close / Won big, split at +/-7.5), computed here in one vectorized pass
    across the whole player pool instead of that function's one-player-at-
    a-time loop (see _vectorized_game_script_multiplier's docstring - this
    is the exact class of mistake data/draft_projections.py's own history
    already flags: "Vectorized; the per-player loop version dominated the
    whole build").
  - injuries: data.draft_sources.fetch_injury_report (same feed the draft
    board already uses)

DELIBERATELY NOT a flat "multiply by Vegas implied team total" adjustment.
data/odds_market.py's own module docstring documents a real backtest
(748 player-seasons, 2023-2025) where scaling a player's SEASON projection
by his team's implied points made it WORSE at every strength tested - "a
player's own usage history already encodes the quality of the offense he
plays in", and scaling on top of that adds variance rather than signal. The
game-script piece here is a different technique answering a different
question: not "is this offense good", but "does the market expect THIS
game to be close or a blowout, and does THIS player's own history show his
role changing when games go that way" - a personalized read-off of the
player's own measured relationship to game margin (the same "interpolate
this player's own curve at the target value" pattern
`data.matchup_signals.efficiency_elasticity_curve` already uses for
opponent softness), not a league-wide flat multiplier. It's applied only to
volume stats, capped at +/-15%, and silently skipped (multiplier 1.0) for
any player without enough game-margin history or without a posted market
line for the target week - see SCRIPT_CLIP and _vectorized_game_script_multiplier.

ONE SHRINKAGE MECHANISM, REUSED FOR EVERYTHING. Rather than a separate
"blend seasons" step and a separate "regress efficiency toward the mean"
step, every per-game rate (targets, carries, receiving yards, TDs, ...)
goes through the SAME weighted blend in _blended_rate:

    w_current = games_this_season / (games_this_season + K[stat])
    blended   = w_current * this_season_rate + (1 - w_current) * prior

`prior` is the player's OWN prior-season rate when he has one, else the
position's current-season average rate (so a rookie with zero history
lands on the position baseline, not on nothing). K is bigger for noisier
stats (touchdowns, interceptions) so those get shrunk harder on a small
sample - the reason a receiver's third career target doesn't get scored
like an every-down role. This is exactly "the current season outweighs the
past as the sample grows": w_current climbs from 0 toward 1 as
games_this_season grows, automatically, no separate schedule to tune.

ROLE CONFIDENCE (snap share + routes run) doesn't get its own multiplier -
it shrinks or widens K itself (see _role_confidence / K_EFFECTIVE_RANGE): a
player whose recent snap share and PFF route rate both say "every-down
role" gets LESS shrinkage (his own small-sample rate is trusted sooner);
a thin, uncertain role gets MORE. That is what "informs the model" means
here - it changes how much weight the model puts on a player's own numbers,
not a bolt-on adjustment layered on top of an unrelated calculation.

THIS SEASON'S OWN GAME LOG IS ITSELF RECENCY- AND MATCHUP-WEIGHTED, not a
flat season-to-date average feeding the shrinkage above
(`build_quality_adjusted_matchup` / `_weighted_player_rates`, replacing an
earlier flat 60% trailing-4-game / 40% season-average split). Three stacked
adjustments on every past game, per explicit request:

  1. RECENCY - most recent game weight 1.0, decaying by RECENCY_DECAY per
     game back, a smooth curve rather than a hard 4-game cutoff.
  2. MATCHUP STRENGTH - a big game against a bad defense is scaled DOWN
     before it's averaged into the player's rate, and a quiet game against
     a good defense is scaled UP - so a huge day against a defense that
     gives that up to everyone doesn't inflate a player's real level, and a
     quiet day against a tough defense doesn't understate it. The defense
     rating used for this is ITSELF quality-adjusted and recency-weighted -
     see build_quality_adjusted_matchup's own docstring for the "Baltimore
     vs. a good slot receiver" reasoning this exists to capture, on both
     the offense side (here) and the defense side (the same matrix is what
     prices the upcoming opponent's matchup_mult below).
  3. REMATCH - a past game against the SAME opponent the player faces again
     this week gets its weight multiplied up further (REMATCH_WEIGHT_MULT)
     - not overriding everything else, but a real rematch is the best
     single data point available for what's about to happen.

This mechanically pushes the model toward USAGE AND EFFICIENCY over bare
box-score totals, which was also asked for directly: a raw counting stat is
now never averaged on its own terms, only after being leveled for who it
came against and how long ago - a talent/usage read, not a highlight-reel
average.

NEVER LEAKS THE TARGET WEEK'S OWN RESULT. Every input (usage rates,
opponent-allowed matrix, snap trend, game-script buckets) is computed off
games with week < as_of_week (defaults to the target week itself). This is
what lets the exact same function double as the backtest harness in
docs/weekly_projections_methodology.md - project week N with only what was
known before week N, then compare to what actually happened.
"""
import numpy as np
import pandas as pd
import streamlit as st

from data.transforms import load_and_merge_data, OFFENSE_PROJECTION_STATS, score_projected_stats
from data.loaders import load_team_pace, load_schedule
from data.utils import clean_name_exact

DRAFTABLE_POSITIONS = ('QB', 'RB', 'WR', 'TE')

# Bigger K = more shrinkage toward the prior for a given sample size - see
# the module docstring's _blended_rate explanation. Counting/volume stats
# (targets, carries, yardage) are self-correlated week to week (a player's
# role doesn't reset each Sunday), so a modest K lets his own small sample
# dominate quickly. Scoring stats are lumpy - a 2-TD game is common, a
# 2-TD-per-game RATE is not - so they get pulled toward the prior harder.
STAT_K = {
    'targets': 3, 'receptions': 3, 'receiving_yards': 3,
    'rushing_attempts': 3, 'rushing_yards': 3,
    'passing_attempts': 3, 'passing_completions': 3, 'passing_yards': 3,
    'receiving_tds': 6, 'rushing_tds': 6, 'passing_tds': 5, 'passing_interceptions': 6,
}

# role_confidence in [0, 1] scales K by this range - a confident every-down
# role shrinks K toward the low end (own rate trusted sooner), a thin/
# uncertain role stretches it toward the high end (leans harder on the
# prior). Bounded so role confidence can move K at most 40% either way -
# it's a nudge on top of sample size, not a replacement for it.
K_EFFECTIVE_RANGE = (0.7, 1.3)

# Per-game recency decay for a player's OWN within-season history (see the
# module docstring's "THIS SEASON'S OWN GAME LOG..." section) - most recent
# played game weight 1.0, each game further back multiplied by this again.
# 0.85 means a game 8 weeks back (games_ago=8) carries 0.85**7 =~ 0.32x a
# just-played game's weight, and one 14 weeks back carries =~0.11x - still
# present, never dominant. Chosen to be gentler than a hard cutoff (the old
# 60/40 trailing-4/season split effectively zeroed anything before game 5)
# while still making "recent weeks count more" true of every game, not just
# a binary in/out of a fixed window. Also used, identically, to recency-
# weight a DEFENSE's own matchup history in build_quality_adjusted_matchup -
# one constant, one meaning ("how fast does old evidence fade"), reused for
# both sides of the same idea per explicit request ("implement this for the
# defense as well").
RECENCY_DECAY = 0.85

# Extra weight multiplier on a past game whose opponent is the SAME team a
# player faces again this week, stacked on top of its ordinary recency
# weight - "there should be some weight allocated to it (not too much but a
# decent amount)" per explicit request. 1.6x a same-recency ordinary game:
# enough to matter (a recent rematch data point can materially move a rate)
# without letting one game override the rest of a season's evidence, which
# is what an override (rather than a weight bump) would risk on a small
# sample.
REMATCH_WEIGHT_MULT = 1.6

MATCHUP_CLIP = (0.75, 1.3)
# Narrower than MATCHUP_CLIP - used only to retroactively adjust a single
# PAST game's value in _weighted_player_rates, not the forward-looking
# projection multiplier. A one-game matchup rating is a noisier estimate
# than the blended rate it feeds into, so a per-game correction is damped
# relative to the multiplier applied once to the whole projection.
# MEASURED, HONESTLY: scripts/validate_weekly_projections.py (2025 weeks
# 5-17) puts this whole recency/matchup/rematch reweighting within noise of
# the pre-existing flat 60/40 trailing-4/season split it replaced - overall
# rank-corr 0.655 either way, MAE 4.65-4.67 across every variant tried
# (MATCHUP_CLIP here, no adjustment at all, this narrower clip). QB and TE
# improved a little, RB and WR moved a little the other way, none of it
# outside what looks like backtest noise at n=300-350/week. This constant
# is the best of the variants actually tried, not a constant validated to
# clearly help - kept because the mechanism is directly what was asked for
# (matchup context on both sides, recency, same-opponent rematches) and
# measurably does not hurt, not because it moved the aggregate numbers on
# its own. Don't read a future small backtest delta on this range as
# meaningful without a bigger sample (see the same season's naive baseline
# sitting inside the same noise band as this whole family of variants).
HISTORY_MATCHUP_CLIP = (0.85, 1.15)
PACE_CLIP = (0.85, 1.15)
SCRIPT_CLIP = (0.85, 1.15)
# Which raw stats the game-script read applies to - VOLUME only. Touchdowns
# are excluded: too sparse per player-game to bucket reliably by margin
# without the noise swamping the signal, and pass-rate/completion-quality
# stats for QBs are left alone for the same reason plus the passing_yards
# figure already carries most of the same information.
SCRIPT_ELIGIBLE_STATS = {'targets', 'receptions', 'receiving_yards', 'rushing_attempts', 'rushing_yards'}
# Same bucket edges as data.matchup_signals.game_script_sensitivity_curve -
# this is the same measurement, not a different one (see module docstring).
SCRIPT_BUCKETS = [(-999, -7.5, -12.5), (-7.5, 0, -3.75), (0, 7.5, 3.75), (7.5, 999, 12.5)]

INJURY_MULTIPLIER = {'out': 0.0, 'ir': 0.0, 'doubtful': 0.4, 'questionable': 0.85, 'suspended': 0.0}


def _played_weeks_before(stats_df, as_of_week):
    """Regular-season rows strictly before as_of_week - the no-leakage gate
    every CURRENT-season input in this module goes through."""
    if stats_df.empty or 'week' not in stats_df.columns:
        return stats_df.iloc[0:0]
    weeks = pd.to_numeric(stats_df['week'], errors='coerce')
    return stats_df[(weeks > 0) & (weeks < as_of_week)].copy()


def _all_played_weeks(stats_df):
    """Every real regular-season row, no cutoff - for the PRIOR season side
    of the blend, which is always used in full (there's no leakage risk in
    a season that already ended)."""
    if stats_df.empty or 'week' not in stats_df.columns:
        return stats_df.iloc[0:0]
    weeks = pd.to_numeric(stats_df['week'], errors='coerce')
    return stats_df[weeks > 0].copy()


def _season_totals(stats_df, name_col, team_col, pos, stats):
    """
    Per-player totals + games played for one position's stat list, one
    groupby - not a per-player loop.

    Also carries Team (most recent team that season - a player's row can
    move teams on a trade) and Games, since every caller needs both.
    """
    rows = stats_df[stats_df['position'].astype(str).str.upper() == pos]
    if rows.empty:
        return pd.DataFrame(columns=[name_col, 'Games', team_col] + stats)
    agg = {s: 'sum' for s in stats if s in rows.columns}
    agg['week'] = 'nunique'
    grouped = rows.groupby(name_col, as_index=False).agg(agg).rename(columns={'week': 'Games'})
    last_team = rows.sort_values('week').groupby(name_col)[team_col].last()
    grouped = grouped.merge(last_team.rename('Team'), left_on=name_col, right_index=True, how='left')
    return grouped


def build_quality_adjusted_matchup(hist_pos, name_col, stats, as_of_week):
    """
    Recency-weighted, OPPONENT-QUALITY-ADJUSTED defensive matchup rating for
    one position: how much MORE or LESS than a typical opponent's own normal
    level a defense allowed - not how many raw yards/catches/etc it allowed.

    WHY THIS EXISTS, PER EXPLICIT REQUEST. A flat "average stat allowed per
    game" gives a defense identical blame for 130 yards allowed to an elite
    target and 130 allowed to a fringe roster player, and those are not the
    same signal about the defense at all: if Baltimore gives up 130 yards to
    a receiver who does that to everyone (a strong slot target like Amon-Ra
    St. Brown), that's barely below his normal level and shouldn't hurt
    Baltimore's rating much; the same 130 to a WR3 who usually does 40 is a
    real breakdown. This compares what each opponent did against a defense
    to what that SAME PLAYER does on average against everyone, then averages
    THAT ratio per defense - a measure of the defense specifically, with the
    opposing talent leveled out.

    Recency-weighted the same way the offense side of this module is (see
    RECENCY_DECAY): a defense's more recent matchups count more, same
    principle applied to both sides of the same idea per explicit request
    ("implement this for the defense as well").

    Returns a DataFrame indexed by opponent_team, one column per stat, values
    centered near 1.0 (a league-average defense reads 1.0; >1 allows more
    than a typical opponent's own level, <1 allows less) - drop-in compatible
    with the same MATCHUP_CLIP scale build_stat_allowed_matrix's ratio used.

    APPROXIMATION, NOTED HONESTLY: a player's own baseline is his mean
    INCLUDING the very game being compared (no leave-one-out split) - a
    small self-inclusion bias that mutes the signal slightly (an outlier
    game pulls its own baseline toward itself), accepted the same way this
    app already accepts "season total / games" as a baseline everywhere
    else rather than a leave-one-out estimator. There's also an inherent,
    accepted circularity in any single-pass strength-of-schedule adjustment
    like this one: a player's baseline here isn't itself matchup-adjusted.
    A fully rigorous version would iterate offense and defense ratings
    against each other to convergence; one pass is the standard practical
    approximation and what every other adjustment in this app already uses.
    """
    if hist_pos.empty or 'opponent_team' not in hist_pos.columns:
        return pd.DataFrame()
    stats = [s for s in stats if s in hist_pos.columns]
    if not stats:
        return pd.DataFrame()
    df = hist_pos.copy()
    weeks = pd.to_numeric(df['week'], errors='coerce')
    games_ago = (as_of_week - weeks).clip(lower=1)
    w = RECENCY_DECAY ** (games_ago - 1)

    baseline = df.groupby(name_col)[stats].transform('mean').replace(0, np.nan)
    ratio = df[stats].div(baseline)
    valid = ratio.notna()
    num = ratio.fillna(0.0).mul(w, axis=0)
    den = valid.astype(float).mul(w, axis=0)
    # .astype(str) - opponent_team is categorical upstream (see HANDOFF.md
    # gotcha #10) and a categorical groupby key/lookup silently propagates
    # the categorical dtype through .map() into places (.clip() below,
    # cur['Opponent'].map(...) at the call site) that then raise on an
    # unordered-categorical comparison. Every opponent-keyed lookup in this
    # function and _weighted_player_rates casts to plain str for this
    # reason - never index by the raw categorical column.
    num['_opponent'] = df['opponent_team'].astype(str).to_numpy()
    den['_opponent'] = df['opponent_team'].astype(str).to_numpy()

    num_sum = num.groupby('_opponent')[stats].sum()
    den_sum = den.groupby('_opponent')[stats].sum()
    result = num_sum.div(den_sum.replace(0, np.nan))

    # Re-center so the league-average defense reads 1.0 - a player's own
    # mean-of-his-games baseline is unbiased in expectation but a finite
    # sample of defenses won't average to exactly 1.0 by chance, and every
    # multiplier downstream (MATCHUP_CLIP etc.) assumes 1.0 = average.
    league_mean = result.mean()
    return result.div(league_mean.replace(0, np.nan))


def _weighted_player_rates(hist_pos, name_col, stats, as_of_week, matchup_matrix, upcoming_opponent):
    """
    Per player, per stat: a recency-weighted, opponent-quality-adjusted rate
    - the WITHIN-season half of the cross-season blend (see _blended_rate),
    replacing an earlier flat 60% trailing-4-game / 40% season-average
    split with the three stacked adjustments the module docstring's "THIS
    SEASON'S OWN GAME LOG..." section describes (recency, matchup strength,
    rematch).

    `matchup_matrix` is build_quality_adjusted_matchup's output - the SAME
    defense ratings used to price the upcoming opponent below, applied
    retroactively to the player's own past games instead of only forward.
    `upcoming_opponent` is {player: this week's opponent}, for the rematch
    weight bump.

    Returns a DataFrame indexed by name_col: one '{stat}' rate column per
    stat, plus 'weight_sum' - NOT plugged into _blended_rate's games-played
    shrinkage (that stays the RAW game count, unchanged, so the existing
    current-vs-prior-season trust calibration isn't disturbed by this
    change) - kept only in case a caller wants the effective sample size.
    """
    if hist_pos.empty:
        return pd.DataFrame()
    stats = [s for s in stats if s in hist_pos.columns]
    if not stats:
        return pd.DataFrame()
    df = hist_pos.copy()
    weeks = pd.to_numeric(df['week'], errors='coerce')
    games_ago = (as_of_week - weeks).clip(lower=1)
    w = RECENCY_DECAY ** (games_ago - 1)

    # .astype(str) - see build_quality_adjusted_matchup's comment on why a
    # raw categorical opponent_team can't be used for a downstream
    # .map()/.clip() chain.
    opponent = (df['opponent_team'].astype(str) if 'opponent_team' in df.columns
               else pd.Series('', index=df.index))
    is_rematch = opponent.eq(df[name_col].map(upcoming_opponent))
    w = w * np.where(is_rematch, REMATCH_WEIGHT_MULT, 1.0)
    df['_w'] = w

    num_cols = []
    for stat in stats:
        raw = pd.to_numeric(df[stat], errors='coerce').fillna(0.0)
        mult = pd.Series(1.0, index=df.index)
        if matchup_matrix is not None and not matchup_matrix.empty and stat in matchup_matrix.columns:
            looked_up = opponent.map(matchup_matrix[stat])
            mult = looked_up.fillna(1.0).clip(*HISTORY_MATCHUP_CLIP)
        col = f'_num_{stat}'
        df[col] = (raw / mult) * df['_w']
        num_cols.append(col)

    agg = {c: (c, 'sum') for c in num_cols}
    agg['weight_sum'] = ('_w', 'sum')
    grouped = df.groupby(name_col).agg(**agg)
    weight_sum = grouped['weight_sum'].replace(0, np.nan)
    rates = grouped[num_cols].div(weight_sum, axis=0)
    rates.columns = [c[len('_num_'):] for c in rates.columns]
    rates['weight_sum'] = grouped['weight_sum']
    return rates


def _blended_rate(cur_rate, cur_games, prior_rate, pos_rate, stat, role_confidence):
    """
    The one shrinkage formula every stat in this module goes through - see
    the module docstring. All arguments are numpy arrays (vectorized over
    the whole player pool for one stat at a time), never scalars in a loop.

    `cur_rate` is this season's recency-weighted rate (_in_season_rate's
    output), not a bare season total/games - see that function's docstring.
    """
    cur_games = np.maximum(cur_games, 0)
    prior = np.where(np.isnan(prior_rate), pos_rate, prior_rate)
    prior = np.where(np.isnan(prior), 0.0, prior)

    base_k = STAT_K.get(stat, 3)
    lo, hi = K_EFFECTIVE_RANGE
    # role_confidence=1 -> multiplier `lo` (less shrinkage, own rate trusted
    # sooner); role_confidence=0 -> multiplier `hi`.
    k_mult = hi - (hi - lo) * np.clip(role_confidence, 0.0, 1.0)
    k_eff = base_k * k_mult

    w_current = cur_games / (cur_games + k_eff)
    return w_current * cur_rate + (1 - w_current) * prior


def _role_confidence(stats_df, name_col, as_of_week, pos, pff_rec):
    """
    0-1 per player: how much of an every-down role his RECENT snaps and (for
    pass-catchers) PFF's season-to-date route rate say he holds. Feeds
    _blended_rate's K, not a separate multiplier - see module docstring.

    Recent snap share is the last up to 3 PLAYED games before as_of_week,
    not the season average - a role change is exactly what a season average
    would smear out.
    """
    hist = _played_weeks_before(stats_df[stats_df['position'].astype(str).str.upper() == pos], as_of_week)
    if hist.empty or 'weekly_snap_pct' not in hist.columns:
        return pd.Series(dtype=float, name='role_confidence')
    recent = (hist.sort_values('week').groupby(name_col).tail(3)
              .groupby(name_col)['weekly_snap_pct'].mean() / 100.0).clip(0, 1)

    if pos in ('WR', 'TE') and pff_rec is not None and not pff_rec.empty and 'player' in pff_rec.columns:
        routes = pff_rec.copy()
        routes['_key'] = clean_name_exact(routes['player'])
        route_rate = routes.set_index('_key')['route_rate']
        route_rate = pd.to_numeric(route_rate, errors='coerce') / 100.0
        keyed_recent = recent.copy()
        keyed_recent.index = clean_name_exact(pd.Series(keyed_recent.index))
        combined = pd.concat([keyed_recent.rename('snap'), route_rate.rename('route')], axis=1)
        blended = combined.mean(axis=1, skipna=True)
        # Reindex back onto real player names via the same exact-key bridge,
        # falling back to snap share alone for anyone with no PFF route row.
        exact_keys = clean_name_exact(pd.Series(recent.index))
        out = pd.Series(blended.reindex(exact_keys.values).to_numpy(), index=recent.index)
        out = out.fillna(recent)
        return out.rename('role_confidence')

    return recent.rename('role_confidence')


def _team_week_margins(schedule_df):
    """[Team, week, margin] for every team-game in the schedule, vectorized
    (home and away rows built at once, not one team at a time) - the whole-
    pool sibling of data.matchup_signals.team_game_margins."""
    needed = {'week', 'home_team', 'away_team', 'home_score', 'away_score'}
    if schedule_df is None or schedule_df.empty or not needed.issubset(schedule_df.columns):
        return pd.DataFrame(columns=['Team', 'week', 'margin'])
    home_score = pd.to_numeric(schedule_df['home_score'], errors='coerce')
    away_score = pd.to_numeric(schedule_df['away_score'], errors='coerce')
    home = pd.DataFrame({'Team': schedule_df['home_team'], 'week': schedule_df['week'],
                         'margin': home_score - away_score})
    away = pd.DataFrame({'Team': schedule_df['away_team'], 'week': schedule_df['week'],
                         'margin': away_score - home_score})
    both = pd.concat([home, away], ignore_index=True).dropna(subset=['margin'])
    both['week'] = pd.to_numeric(both['week'], errors='coerce')
    return both


def _vectorized_game_script_multiplier(stats_df, name_col, team_col, as_of_week, schedule_df,
                                       target_margins, stat):
    """
    Whole-pool version of data.matchup_signals.game_script_sensitivity_curve
    for one stat: bucket every player's PLAYED games by that game's real
    margin (merged in once, not looked up per player), average by
    (player, bucket) in a single groupby, then read off each player's own
    curve at his team's MARKET-IMPLIED margin for the target week.

    Returns a Series of multipliers indexed by player name, 1.0 (neutral)
    for anyone without enough bucketed history or without a target margin.
    """
    hist = _played_weeks_before(stats_df, as_of_week)
    if hist.empty or stat not in hist.columns or target_margins is None or target_margins.empty:
        return pd.Series(dtype=float)
    margins = _team_week_margins(schedule_df)
    if margins.empty:
        return pd.Series(dtype=float)
    merged = hist.merge(margins, left_on=[team_col, 'week'], right_on=['Team', 'week'], how='inner')
    if merged.empty:
        return pd.Series(dtype=float)

    def _bucket(m):
        for lo, hi, mid in SCRIPT_BUCKETS:
            if lo < m <= hi:
                return mid
        return np.nan
    merged['_bucket'] = merged['margin'].map(_bucket)
    merged = merged.dropna(subset=['_bucket'])
    if merged.empty:
        return pd.Series(dtype=float)

    bucket_means = merged.groupby([name_col, '_bucket'])[stat].mean().reset_index()
    game_counts = merged.groupby(name_col)['week'].nunique()
    season_avg = merged.groupby(name_col)[stat].mean()

    out = {}
    for player, group in bucket_means.groupby(name_col):
        if game_counts.get(player, 0) < 4 or len(group) < 2:
            continue
        target = target_margins.get(player)
        if target is None or pd.isna(target):
            continue
        avg = season_avg.get(player)
        if not avg or avg <= 0:
            continue
        xs, ys = group['_bucket'].to_numpy(), group[stat].to_numpy()
        order = np.argsort(xs)
        projected = float(np.interp(target, xs[order], ys[order]))
        out[player] = float(np.clip(projected / avg, *SCRIPT_CLIP))
    return pd.Series(out, dtype=float)


def _target_margins_by_team(year, week):
    """
    Market-implied margin (from THIS team's own point of view - positive
    means favored) for every team's game in `week`, or {} if no lines are
    posted yet for that far out (the normal case beyond the first few
    weeks - see data/odds_market.py's own docstring on posted coverage).
    Never raises; a missing line just means the game-script read sits out
    for that player, same as every other best-effort signal in this app.
    """
    try:
        from data.odds_market import fetch_game_lines, implied_team_points
        games, meta = fetch_game_lines(year)
        if meta.get('error') or games.empty:
            return {}
        week_games = games[pd.to_numeric(games['week'], errors='coerce') == week]
        per_team = implied_team_points(week_games)
        if per_team.empty:
            return {}
        return dict(zip(per_team['team'], per_team['implied_points'] - per_team['implied_allowed']))
    except Exception:
        return {}


def _week_opponents(schedule_df, week):
    """{team: opponent} for one week, from the real schedule - teams absent
    from this dict are on a bye. Vectorized dict build, not a per-team scan."""
    if schedule_df is None or schedule_df.empty:
        return {}
    wk = schedule_df[pd.to_numeric(schedule_df['week'], errors='coerce') == week]
    if wk.empty:
        return {}
    out = dict(zip(wk['home_team'], wk['away_team']))
    out.update(dict(zip(wk['away_team'], wk['home_team'])))
    return out


def _injury_multipliers(year, week):
    """{player: multiplier} from the injury report AS OF `week` - Out/IR/
    Suspended zero the projection out, Doubtful/Questionable discount it.
    Best-effort: an unreachable feed just means no discount is applied,
    same degrade-gracefully convention as every other live source here.

    Passing `week` (not just `year`) matters for anything other than the
    live current week: fetch_injury_report defaults to each player's LATEST
    designation for the whole season, which for a PAST week is that
    player's end-of-season status, not his status that week - a receiver
    who tore an ACL in week 15 would show 'Out' (and get zeroed) on a week
    3 projection otherwise. Harmless for the live/current week (there's no
    future data to exclude either way).
    """
    try:
        from data.draft_sources import fetch_injury_report
        injuries, _err = fetch_injury_report(year, as_of_week=week)
    except Exception:
        return {}
    if injuries is None or injuries.empty:
        return {}
    status = injuries['Injury Status'].astype(str).str.strip().str.lower()
    mult = status.map(INJURY_MULTIPLIER)
    return dict(zip(injuries['Player'], mult.dropna()))


def _cold_start_pool(stats_df, name_col, team_col, as_of_week):
    """
    Player identity (name/team/position) for a COLD START - projecting a
    week with zero real games played this season yet to compare against
    (week 1, or the whole preseason before a weekly stats file exists at
    all), where the normal player pool (built off this season's own played
    games) has nothing to read.

    TWO DIFFERENT KINDS OF "NOTHING TO READ" HERE, AND THEY GET DIFFERENT
    TREATMENT so team/position is never accidentally read from a LATER week
    than the one being projected:

      - No 'week' column at all (e.g. 2026 before kickoff) - load_year_data's
        roster-only fallback (HANDOFF.md gotcha #6) still carries this
        season's real team/position from the roster file (offseason trades,
        cuts, depth-chart moves already reflected), just zero real stat
        rows. Every row here IS the current snapshot - nothing to leak.
      - A 'week' column exists (a real, possibly-already-played season) but
        `as_of_week` itself has no rows yet (the live "haven't kicked off
        week 1" case, or testing as_of_week=1 against a completed season).
        Using the UNFILTERED stats_df for team/position here would pull in
        a later week's post-trade team and leak it backward into what's
        supposed to be a week-1-only read - so this reads ONLY rows from
        `as_of_week` itself when they exist, and only falls back to the
        unfiltered frame when even those don't exist yet (nothing legitimate
        to leak in that case either - the season hasn't started).
    """
    if stats_df.empty or name_col not in stats_df.columns:
        return pd.DataFrame(columns=[name_col, team_col, 'position'])
    cols = [c for c in (name_col, team_col, 'position') if c in stats_df.columns]
    if 'week' in stats_df.columns:
        this_week = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == as_of_week]
        pool = this_week[cols] if not this_week.empty else stats_df[cols]
    else:
        pool = stats_df[cols]
    return pool.dropna(subset=[name_col]).drop_duplicates(subset=[name_col])


def _load_pff_receiving(year):
    try:
        from data.loaders import load_pff_data_with_fallback
        pff, source_year = load_pff_data_with_fallback(year)
        return pff.get('rec', pd.DataFrame())
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def build_weekly_projections(year, week, scoring_mode='Full PPR', as_of_week=None, apply_injury=True):
    """
    This app's own projected stat line + fantasy points for every QB/RB/WR/
    TE with usable history, for one week.

    `as_of_week` defaults to `week` (project week N off weeks < N of the
    same season - the real-time use). Passed explicitly and lower than
    `week` by the backtest harness in docs/weekly_projections_methodology.md
    to validate against a week that's already happened without leaking its
    own result into the projection that's supposed to predict it.

    `apply_injury=False` skips the injury-status discount entirely - THE
    BACKTEST NEEDS THIS. fetch_injury_report always returns each player's
    MOST RECENT designation, not a designation as of `week` - fine (correct,
    even) for a live current-week projection, but when validating a past
    week months or years later it's reading each player's LAST reported
    status of that entire season (often 'Out' - end-of-season IR - or a
    routine 'Questionable' from an unrelated later week), applied to every
    week tested regardless of relevance. Measured impact: this alone was
    quietly zeroing out or discounting roughly 1,000 of ~2,000 skill-
    position players in the 2025 backtest, which was the dominant source of
    this model's apparent bias before it was isolated - see
    docs/weekly_projections_methodology.md.

    Returns (DataFrame, meta). Empty DataFrame with meta['reason'] set only
    when there is truly nothing anywhere to build even a cold-start
    projection from (see COLD START below) - meta['cold_start'] is True
    when the returned board leans entirely on prior-season data because
    this season has no games played yet.

    COLD START: WEEK 1 (OR A SEASON WITH NO WEEKLY FILE YET) FALLS BACK TO
    PRIOR-SEASON DATA RATHER THAN PROJECTING NOTHING. The player pool
    (who's on which team) still has to come from somewhere real for THIS
    season - see _cold_start_pool - but every rate in the projection itself
    already had a well-defined answer for a player with zero current-season
    games: _blended_rate's own cross-season shrinkage (w_current =
    games/(games+K)) already lands ENTIRELY on the prior-season rate when
    games=0, which is exactly "based on past season stats" for the offense
    side. The one piece that previously had no prior-season equivalent was
    the DEFENSE side - the opponent-allowed matchup multiplier normally
    read off THIS season's own games - so a cold start computes
    build_quality_adjusted_matchup off last season's full year instead,
    anchored so last season's FINAL week reads as "most recent" (see the
    cold-start branch below). This is deliberately not a good projection -
    a whole season has changed since the numbers it's built from - but it
    is a real one, which is the ask: "albeit the projection won't be great"
    beats no projection at all. FantasyPros' own weekly projection
    (data.draft_sources.fetch_fantasypros_weekly_projections) still has
    real season-opener-specific analyst input this app doesn't and remains
    the better source where it's available; this is the fallback for when
    it isn't, or for seeing this app's own read.
    """
    if as_of_week is None:
        as_of_week = week
    stats_df, team_col, name_col, _ = load_and_merge_data(year, scoring_mode)
    if stats_df.empty:
        return pd.DataFrame(), {'reason': f'No roster or weekly data for {year} yet.'}

    hist = _played_weeks_before(stats_df, as_of_week) if 'week' in stats_df.columns else stats_df.iloc[0:0]
    cold_start = hist.empty

    prior_stats, prior_team_col, prior_name_col, _ = pd.DataFrame(), team_col, name_col, None
    try:
        prior_stats, prior_team_col, prior_name_col, _ = load_and_merge_data(year - 1, scoring_mode)
    except Exception:
        prior_stats = pd.DataFrame()

    if cold_start and (prior_stats is None or prior_stats.empty):
        return pd.DataFrame(), {'reason': f'Week {as_of_week} of {year} has no games played yet this '
                                          f'season, and {year - 1} has no data to fall back on either - '
                                          'nothing to build even a rough cold-start projection from.'}

    schedule_df = load_schedule(year)
    opponents = _week_opponents(schedule_df, week)
    target_margins = _target_margins_by_team(year, week)
    injury_mult = _injury_multipliers(year, week) if apply_injury else {}
    pff_rec = _load_pff_receiving(year)
    pace = load_team_pace(year)
    if cold_start and (pace is None or pace.empty):
        # This season's own pace data doesn't exist yet either (same reason
        # as everything else in a cold start) - last season's team pace is
        # a far better estimate of week 1 than the neutral 1.0 fallback
        # below would be.
        pace = load_team_pace(year - 1)
    league_pace = pace['def_pace'].mean() if pace is not None and not pace.empty and 'def_pace' in pace.columns else None

    cold_pool = _cold_start_pool(stats_df, name_col, team_col, as_of_week) if cold_start else pd.DataFrame()
    prior_played = _all_played_weeks(prior_stats)
    prior_max_week = None
    if cold_start and not prior_played.empty:
        prior_weeks_numeric = pd.to_numeric(prior_played['week'], errors='coerce')
        prior_max_week = prior_weeks_numeric.max() if prior_weeks_numeric.notna().any() else None

    all_rows = []
    for pos in DRAFTABLE_POSITIONS:
        stats = OFFENSE_PROJECTION_STATS[pos]
        if cold_start:
            pool_pos = cold_pool[cold_pool['position'].astype(str).str.upper() == pos] if not cold_pool.empty else cold_pool
            if pool_pos.empty:
                continue
            cur = pool_pos[[name_col, team_col]].rename(columns={team_col: 'Team'}).drop_duplicates(subset=[name_col])
            cur['Games'] = 0
            for stat in stats:
                cur[stat] = 0.0
        else:
            cur = _season_totals(hist, name_col, team_col, pos, stats)
            if cur.empty:
                continue
        prior = (_season_totals(prior_played, prior_name_col, prior_team_col, pos, stats)
                 if not prior_stats.empty else pd.DataFrame())
        prior_rates = pd.DataFrame({s: prior[s] / prior['Games'].replace(0, 1) for s in stats}) if not prior.empty else pd.DataFrame()
        if not prior.empty:
            # Keyed by clean_name_exact, not a raw column-name/value match -
            # `prior_name_col` isn't even guaranteed to be a real column in
            # THIS season's frame (a roster-only current season uses
            # 'player_name', a real prior season uses 'player_display_name'
            # - different column AND, on other year pairs, potentially
            # different formatting of the same name), so joining on the
            # literal column name silently KeyErrors or joins nothing.
            prior_rates['_key'] = clean_name_exact(prior[prior_name_col])

        role_conf = _role_confidence(stats_df, name_col, as_of_week, pos, pff_rec)
        cur = cur.merge(role_conf.rename('role_confidence'), left_on=name_col, right_index=True, how='left')
        cur['role_confidence'] = cur['role_confidence'].fillna(0.5)

        cur['Opponent'] = cur['Team'].map(opponents)
        cur = cur[cur['Opponent'].notna()].copy()  # bye-week teams drop out entirely
        if cur.empty:
            continue
        # .astype(str) AFTER the notna() filter above, never before - Team
        # can be categorical upstream (HANDOFF.md gotcha #10) and casting a
        # real NaN to str first turns it into the literal string "nan",
        # which then survives a .notna() bye-week filter it should have
        # been dropped by. Needed downstream for a clean .map() against
        # matchup_matrix's plain-str index (build_quality_adjusted_matchup).
        cur['Opponent'] = cur['Opponent'].astype(str)
        cur['target_margin'] = cur['Team'].map(target_margins)

        # Recency + matchup-strength + rematch weighted own-history rate,
        # and the SAME defense ratings reused below to price the upcoming
        # matchup - see build_quality_adjusted_matchup's docstring and the
        # module docstring's "THIS SEASON'S OWN GAME LOG..." section.
        if cold_start:
            # No games THIS season to build a matchup rating from - fall
            # back to PRIOR season's full year, anchored so its FINAL week
            # reads as "most recent" (prior_max_week + 1 -> games_ago=1 for
            # that week), so the recency decay within last season is still
            # real (its second half predicts this year's week 1 better than
            # its opener does) even though the whole thing is a year old.
            # The offense side needs no equivalent override: _blended_rate's
            # own w_current=0 (cur_games=0 here) already lands entirely on
            # prior_rate below without any extra machinery.
            prior_pos_rows = (prior_played[prior_played['position'].astype(str).str.upper() == pos]
                              if not prior_played.empty else prior_played)
            matchup_matrix = (build_quality_adjusted_matchup(
                                  prior_pos_rows, prior_name_col, stats, prior_max_week + 1)
                              if prior_max_week is not None and not pd.isna(prior_max_week)
                              else pd.DataFrame())
        else:
            pos_rows = hist[hist['position'].astype(str).str.upper() == pos]
            upcoming_opponent_map = dict(zip(cur[name_col], cur['Opponent']))
            matchup_matrix = build_quality_adjusted_matchup(pos_rows, name_col, stats, as_of_week)
            weighted_rates = _weighted_player_rates(
                pos_rows, name_col, stats, as_of_week, matchup_matrix, upcoming_opponent_map)
        if cold_start:
            # No current-season rows to weight at all - every player's
            # in_season_rate below falls through to its own season_avg
            # fallback (0/0=0 for a cold-start cur), which is fine: cur_games
            # is 0 for everyone here too, so _blended_rate ignores it anyway
            # and lands entirely on prior_rate.
            weighted_rates = pd.DataFrame()

        proj_cols = {}
        for stat in stats:
            if stat not in cur.columns:
                continue
            cur_total = cur[stat].to_numpy(dtype=float)
            cur_games = cur['Games'].to_numpy(dtype=float)
            # The season-long flat average is kept only as a FALLBACK for a
            # player weighted_rates has no usable row for (e.g. every one of
            # his historical games had an unratable opponent) - the primary
            # value is the recency/matchup/rematch-weighted rate above it.
            season_avg = np.divide(cur_total, np.maximum(cur_games, 1),
                                   out=np.zeros_like(cur_total, dtype=float), where=cur_games > 0)
            if stat in weighted_rates.columns:
                in_season_rate = cur[name_col].map(weighted_rates[stat]).to_numpy(dtype=float)
            else:
                in_season_rate = np.full(len(cur), np.nan)
            in_season_rate = np.where(np.isnan(in_season_rate), season_avg, in_season_rate)

            if not prior_rates.empty and stat in prior_rates.columns:
                prior_map = pd.Series(prior_rates[stat].to_numpy(), index=prior_rates['_key'])
                prior_rate = clean_name_exact(cur[name_col]).map(prior_map).to_numpy(dtype=float)
            else:
                prior_rate = np.full(len(cur), np.nan)
            # Games-weighted, not a plain mean of each player's own rate - a
            # one-game emergency start would otherwise count exactly as much
            # as a nine-game starter's rate in setting the rookie/no-prior
            # baseline, which understates it (confirmed real in the backtest
            # write-up in docs/weekly_projections_methodology.md).
            #
            # cur_total/cur_games are all zero for everyone in a cold start
            # (Games=0 by construction) - the position average has to come
            # from PRIOR season's totals instead, or a rookie with zero
            # career history (prior_rate also NaN) would fall through to a
            # bare 0.0 baseline instead of "what a typical player at this
            # position does," the same fallback he'd get in-season.
            if cold_start and not prior.empty and stat in prior.columns:
                prior_games_total = prior['Games'].sum()
                pos_rate = float(prior[stat].sum() / prior_games_total) if prior_games_total > 0 else 0.0
            else:
                games_total = cur_games.sum()
                pos_rate = float(cur_total.sum() / games_total) if games_total > 0 else 0.0
            pos_rate_arr = np.full(len(cur), pos_rate)

            # cur_games (RAW game count, not the recency-weighted
            # weight_sum) still drives the current-vs-prior-season shrinkage
            # below, deliberately - see _weighted_player_rates' docstring on
            # why that calibration is left undisturbed by this change.
            blended = _blended_rate(in_season_rate, cur_games, prior_rate, pos_rate_arr, stat,
                                    cur['role_confidence'].to_numpy(dtype=float))

            matchup_mult = np.ones(len(cur))
            if matchup_matrix is not None and not matchup_matrix.empty and stat in matchup_matrix.columns:
                opp_val = cur['Opponent'].map(matchup_matrix[stat])
                matchup_mult = np.clip(opp_val.fillna(1.0).to_numpy(dtype=float), *MATCHUP_CLIP)

            script_mult = np.ones(len(cur))
            if stat in SCRIPT_ELIGIBLE_STATS:
                script_series = _vectorized_game_script_multiplier(
                    stats_df, name_col, team_col, as_of_week, schedule_df, cur.set_index(name_col)['target_margin'], stat)
                if not script_series.empty:
                    script_mult = cur[name_col].map(script_series).fillna(1.0).to_numpy(dtype=float)

            proj_cols[stat] = blended * matchup_mult * script_mult

        pace_mult = pd.Series(1.0, index=cur.index)
        if league_pace and league_pace > 0:
            opp_pace = cur['Opponent'].map(pace['def_pace'])
            pace_mult = np.clip(opp_pace.fillna(league_pace) / league_pace, *PACE_CLIP)

        inj_mult = cur[name_col].map(injury_mult).fillna(1.0)

        for stat in proj_cols:
            # Floored at zero - every one of these is a real-world COUNT
            # (yards, attempts, catches) and none are physically negative,
            # even though a single past GAME legitimately can be (a kneel-
            # heavy or stuffed rushing line). A negative blended rate only
            # shows up for a low-volume player whose weighted history
            # happens to be dominated by one such game - same class of
            # small-sample artifact already floored in data/draft_projections.py
            # (see that file's matching comment), applied here too.
            proj_cols[stat] = np.clip(
                proj_cols[stat] * pace_mult.to_numpy() * inj_mult.to_numpy(), 0.0, None)

        out = pd.DataFrame({
            'Player': cur[name_col], 'Pos': pos, 'Team': cur['Team'], 'Opponent': cur['Opponent'],
            'Games This Season': cur['Games'].astype(int),
            'Role Confidence': cur['role_confidence'].round(2),
        })
        for stat, values in proj_cols.items():
            out[stat] = np.round(values, 2)
        proj_dicts = out[[s for s in stats if s in out.columns]].to_dict('records')
        # Floored at zero, same reasoning as the raw per-stat floor above -
        # a projection is an expectation, and no real player has a negative
        # one. Score_projected_stats subtracts for interceptions, so a
        # near-zero-volume passer (a WR/RB with a trace INT rate and
        # otherwise nothing projected) can otherwise net a small negative
        # total even with every raw stat already non-negative.
        out['Model Proj Pts'] = [max(0.0, score_projected_stats(d, scoring_mode)) for d in proj_dicts]
        out['Injury Status'] = out['Player'].map(lambda p: 'Out/Doubtful' if injury_mult.get(p, 1.0) < 0.9 else '')
        all_rows.append(out)

    if not all_rows:
        return pd.DataFrame(), {'reason': f'No projectable players found for week {week}.'}
    result = pd.concat(all_rows, ignore_index=True).sort_values('Model Proj Pts', ascending=False).reset_index(drop=True)
    meta = {'reason': None, 'year': year, 'week': week, 'as_of_week': as_of_week,
            'players': int(len(result)), 'scoring': scoring_mode, 'cold_start': cold_start}
    return result, meta
