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

# Every raw stat any position projects, in one flat list - the union the
# post-loop passes (teammate vacancy) need to re-score a whole-league frame
# whose columns came from four different per-position stat lists.
_ALL_PROJECTION_STATS = sorted({s for stats in OFFENSE_PROJECTION_STATS.values() for s in stats})

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

# ---------------------------------------------------------------------------
# OPTIONAL MODEL COMPONENTS
#
# Every entry here is a named, individually switchable piece of the
# projection, and every one of them shipped only after
# scripts/eval_weekly_model.py measured it against the same weeks with the
# same paired player pool. The switch is not decoration: it is what makes a
# component falsifiable, and this file's own history (HISTORY_MATCHUP_CLIP's
# comment, docs/draft_hq_methodology.md's three built-measured-and-rejected
# changes) is the reason the project works this way. A component that can't
# be turned off can't be shown to help.
#
# DEFAULT_FEATURES is what the app actually runs. A name here but not in
# DEFAULT_FEATURES was built, measured, and left off - with the measurement
# written down next to it in docs/weekly_projections_methodology.md.
# ---------------------------------------------------------------------------
MODEL_FEATURES = (
    'volume_efficiency',  # opportunities x per-opportunity rate, not a flat per-game rate
    'role_volume',        # baselines scaled by expected SNAP SHARE, not per game played
    'role_matchup',       # defense ratings conditioned on the player's own ROLE
    'redzone_tds',        # touchdowns from red-zone opportunity, not a raw TD rate
    'role_trend',         # a step change in snap share, not a decayed average of it
    'volume_faced',       # opponent pass/rush volume faced, split (pace is one number)
    'game_env',           # market total, roof/venue, wind, rest
    'teammate_vacancy',   # an OUT teammate's usage redistributed (live only)
    'calibration',        # shrink toward the positional mean, fitted out-of-sample
)
# What the app actually runs. `volume_efficiency` and `game_env` are NOT in
# it: both were built, measured on the same 8,107 paired player-weeks, and
# left off because they did not help (volume_efficiency +0.051 MAE / -0.005
# rank-corr, winning 5 of 26 weeks; game_env +0.012 MAE at the measured
# elasticity and +0.006 at half of it, winning 10-11 of 26). The code stays,
# switchable, with the measurement written next to each - see
# docs/weekly_projections_methodology.md.
DEFAULT_FEATURES = frozenset({'role_volume', 'role_matchup', 'teammate_vacancy', 'calibration'})

# The feature set the calibration line was FITTED against - i.e. everything
# that ships except calibration itself, and except teammate_vacancy (which
# only ever fires off a live injury feed and is inert in a backtest).
# scripts/fit_weekly_calibration.py builds the model with exactly this set,
# so the line describes the dispersion of the model it is applied to. If the
# shipping component set changes, the line has to be re-fitted - that is
# what makes it a measurement rather than a magic number.
CALIBRATION_INPUT_FEATURES = frozenset({'role_volume', 'role_matchup'})


# ---------------------------------------------------------------------------
# CALIBRATION
#
# A projection is supposed to be a conditional expectation: among every
# player projected for 20 points, the average one should score 20. This
# model's was not, and the direction is the one selection always produces -
# the players a noisy projection ranks highest are disproportionately the
# ones its own noise pushed up. Measured on 2024-2025, the top 15% of each
# position came in +2.6 (QB), +2.0 (RB), +2.3 (WR) and +0.5 (TE) above what
# they actually scored, and regressing actual on projected gave a slope well
# under 1 at every position - over-dispersion, not bias.
#
# (slope, intercept) per position, FITTED ON 2021-2023 - deliberately
# outside the 2024-2025 window every model change here is evaluated on, so
# this is a measurement rather than a curve fitted to its own test. Produced
# by scripts/fit_weekly_calibration.py against CALIBRATION_INPUT_FEATURES;
# re-run it if the shipping component set changes, since the line describes
# the dispersion of the model it is applied to.
#
# APPLIED ONE-SIDED: `min(projection, line(projection))`. The line is fitted
# over the whole pool and crosses the identity at roughly 13 (QB), 10 (RB),
# 8 (WR) and 7 (TE) points, so it shrinks above that and INFLATES below it -
# and the bulk does not need inflating. Measured: the two-sided version
# bought every startable gain below and cost +0.116 whole-pool MAE, winning
# 1 week of 26, entirely from lifting several hundred near-zero bench rows.
# Clipping it to the shrink half keeps the correction where the defect is.
#
# WHAT IT DOES AND DOESN'T BUY, honestly: it is a per-position MONOTONE
# transform, so it cannot change the order of players within a position and
# does not pretend to. What it changes is the LEVEL - which is what a
# projected point total is read for when it sits next to FantasyPros' and
# the market's numbers on the same row, and what decides whether the
# startable tier is systematically over-promised. Measured on 2024-2025 the
# startable bias it removes is +0.80 (QB), +0.81 (RB), +0.87 (WR).
# ---------------------------------------------------------------------------
WEEKLY_CALIBRATION = {
    'QB': (0.689, 4.115),
    'RB': (0.810, 1.825),
    'WR': (0.786, 1.652),
    'TE': (0.799, 1.389),
}


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


# ---------------------------------------------------------------------------
# ROLE-CONDITIONED MATCHUPS
#
# The explicit ask this implements: "Baltimore vs. a good slot receiver" is
# already handled (build_quality_adjusted_matchup levels for WHO the
# production came against), but "a defense that is soft to a possession
# receiver and airtight deep", "a receiving back vs. a high-volume runner",
# and "a high-completion QB vs. a high-ADOT low-completion QB" are all a
# different question: not how good the defense is against the position, but
# against a player who does THIS FOR A LIVING.
#
# One mechanism covers all of them. Every player gets a ROLE label derived
# from his own measured season-to-date profile (never from a hand-assigned
# list, which would go stale the week a role changes), the defense gets a
# separate rating per role, and a player is priced against the rating for
# players like him - shrunk hard toward the defense's overall rating,
# because a defense plays ~9 games and splitting those three ways leaves
# 2-4 observations per bucket.
#
# Labels are TERCILES of the qualifying pool, not fixed thresholds. A fixed
# "ADOT > 9 is a deep passer" cutoff silently redefines itself every time
# the league's passing environment moves (league ADOT has drifted most of a
# yard across the seasons this app carries); terciles measure the same
# thing - "downfield relative to his peers this season" - in every one of
# them without a constant to re-tune.
# ---------------------------------------------------------------------------

# Games of role-specific evidence for a 50/50 blend against the defense's
# overall rating. A defense sees a given role in roughly a third of its
# games, so 4 is deliberately larger than the sample usually available -
# the role rating is a NUDGE on the overall rating for most matchups and
# only takes over when a defense really has faced this role repeatedly.
ROLE_MATCHUP_K = 10.0

# Below this many qualifying events a player has no measurable profile and
# lands in the middle bucket rather than being labelled off two targets.
ROLE_MIN_EVENTS = {'QB': 40, 'RB': 20, 'WR': 15, 'TE': 12}

ROLE_LABELS = {
    # Ordered low -> high on the role metric below.
    'QB': ('QB_QUICK', 'QB_BALANCED', 'QB_DOWNFIELD'),
    'RB': ('RB_RUSHER', 'RB_BALANCED', 'RB_RECEIVER'),
    'WR': ('WR_SHORT', 'WR_MID', 'WR_DEEP'),
    'TE': ('TE_SHORT', 'TE_MID', 'TE_DEEP'),
}


def _role_metric(hist_pos, name_col, pos):
    """
    (metric, events) per player - the one number his role label is a tercile
    of, plus how much evidence stands behind it.

      QB  - ADOT (air yards per attempt). Separates a quick-game, high-
            completion passer from a low-completion downfield thrower, which
            is exactly the pair the request named. Completion rate is
            deliberately NOT a second axis: it is largely a CONSEQUENCE of
            ADOT (the two correlate strongly), so splitting on both would
            mostly re-cut the same players and thin every bucket for nothing.
      RB  - share of his own touches that are targets. A receiving back and
            a two-down grinder face genuinely different defenses: one is
            priced by a linebacker's coverage, the other by a front seven.
      WR  - ADOT. The honest proxy this app can compute for every season for
            "possession/slot vs. field-stretcher"; PFF's real slot rate only
            exists for the seasons whose exports are on disk, and nothing
            in the free feeds carries alignment at all. Named as a proxy
            here rather than sold as alignment data.
      TE  - same as WR.
    """
    if hist_pos.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    g = hist_pos.groupby(name_col, observed=True)

    def _sum(col):
        return g[col].sum() if col in hist_pos.columns else None

    if pos == 'QB':
        att, ay = _sum('passing_attempts'), _sum('passing_air_yards')
        if att is None or ay is None:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        return ay / att.replace(0, np.nan), att
    if pos == 'RB':
        tgt, car = _sum('targets'), _sum('rushing_attempts')
        if tgt is None or car is None:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        touches = (tgt + car).replace(0, np.nan)
        return tgt / touches, touches
    tgt, ay = _sum('targets'), _sum('receiving_air_yards')
    if tgt is None or ay is None:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return ay / tgt.replace(0, np.nan), tgt


def build_player_roles(hist_pos, name_col, pos):
    """
    {player: role label} for one position, from each player's own
    season-to-date profile. Anyone under ROLE_MIN_EVENTS, or with an
    undefined metric, lands in the MIDDLE label - "no measurable role" is
    not the same as "average role", but the middle bucket is the only
    honest place to put a player the model can't characterise, and it's
    also where the role multiplier does the least.
    """
    labels = ROLE_LABELS.get(pos)
    if labels is None:
        return {}
    metric, events = _role_metric(hist_pos, name_col, pos)
    if metric.empty:
        return {}
    min_events = ROLE_MIN_EVENTS.get(pos, 15)
    qualified = metric[(events >= min_events) & metric.notna()]
    if len(qualified) < 9:
        return {p: labels[1] for p in metric.index}
    lo, hi = qualified.quantile([1 / 3, 2 / 3]).tolist()
    out = {}
    for player, value in metric.items():
        if player not in qualified.index or not np.isfinite(value):
            out[player] = labels[1]
        elif value <= lo:
            out[player] = labels[0]
        elif value >= hi:
            out[player] = labels[2]
        else:
            out[player] = labels[1]
    return out


def build_role_matchup(hist_pos, name_col, stats, as_of_week, roles):
    """
    build_quality_adjusted_matchup, cut by the OPPONENT'S ROLE as well as by
    defense: {role: DataFrame(index=opponent_team, columns=stats)} plus
    {(opponent, role): effective sample size}.

    Identical arithmetic to the overall rating - every past game compared to
    that player's OWN baseline first, recency-weighted, then centered so the
    league average reads 1.0 - just grouped one level finer. Centering is
    done WITHIN each role, which matters: deep receivers post a different
    absolute yards-per-game level than possession receivers, and a rating
    centered across all of them would read every defense as "good against
    deep guys" purely because deep targets bust more often.
    """
    if hist_pos.empty or 'opponent_team' not in hist_pos.columns or not roles:
        return {}, {}
    stats = [s for s in stats if s in hist_pos.columns]
    if not stats:
        return {}, {}
    df = hist_pos.copy()
    weeks = pd.to_numeric(df['week'], errors='coerce')
    w = RECENCY_DECAY ** ((as_of_week - weeks).clip(lower=1) - 1)

    baseline = df.groupby(name_col, observed=True)[stats].transform('mean').replace(0, np.nan)
    ratio = df[stats].div(baseline)
    valid = ratio.notna()
    num = ratio.fillna(0.0).mul(w, axis=0)
    den = valid.astype(float).mul(w, axis=0)
    # .astype(str) for the same categorical reason build_quality_adjusted_matchup
    # documents - never key a lookup off the raw categorical column.
    opponent = df['opponent_team'].astype(str).to_numpy()
    role = df[name_col].map(roles).fillna('').to_numpy()
    num['_opp'], num['_role'] = opponent, role
    den['_opp'], den['_role'] = opponent, role

    num_sum = num.groupby(['_opp', '_role'], observed=True)[stats].sum()
    den_sum = den.groupby(['_opp', '_role'], observed=True)[stats].sum()
    result = num_sum.div(den_sum.replace(0, np.nan))
    evidence = den.groupby(['_opp', '_role'], observed=True)[stats].sum().mean(axis=1)

    out, sizes = {}, {}
    for role_label, block in result.groupby(level='_role'):
        block = block.droplevel('_role')
        league_mean = block.mean()
        out[role_label] = block.div(league_mean.replace(0, np.nan))
    for (opp, role_label), size in evidence.items():
        sizes[(opp, role_label)] = float(size)
    return out, sizes


def _role_adjusted_multiplier(overall, role_tables, role_sizes, opponents, player_roles, stat):
    """
    The forward-looking matchup multiplier for one stat, blending the
    defense's role-specific rating toward its overall rating by how much
    role-specific evidence there actually is:

        w = n_role / (n_role + ROLE_MATCHUP_K)

    Same `evidence / (evidence + K)` shape as _blended_rate's cross-season
    blend and draft_projections' stickiness weight - one shrinkage idea,
    reused, rather than a third hand-tuned schedule.

    Applied ONLY forward, to the upcoming opponent - not retroactively to
    each past game the way the overall rating is. A retroactive role
    correction would be correcting a single game by a rating built partly
    from that same game at a sample size where that self-reference is no
    longer negligible; the overall rating, built off every opponent a
    defense faced, is the right instrument for that job and already does it.

    `opponents` and `player_roles` are numpy arrays aligned to the player
    frame. Returns the per-player multiplier array (already clipped).
    """
    base = np.ones(len(opponents))
    if overall is not None and not overall.empty and stat in overall.columns:
        base = pd.Series(opponents).map(overall[stat]).fillna(1.0).to_numpy(dtype=float)
    if not role_tables:
        return np.clip(base, *MATCHUP_CLIP)

    role_vals = np.full(len(opponents), np.nan)
    weights = np.zeros(len(opponents))
    for i, (opp, role_label) in enumerate(zip(opponents, player_roles)):
        table = role_tables.get(role_label)
        if table is None or stat not in table.columns or opp not in table.index:
            continue
        value = table.at[opp, stat]
        if not np.isfinite(value):
            continue
        n = role_sizes.get((opp, role_label), 0.0)
        role_vals[i] = value
        weights[i] = n / (n + ROLE_MATCHUP_K)
    blended = np.where(np.isnan(role_vals), base, weights * role_vals + (1 - weights) * base)
    return np.clip(blended, *MATCHUP_CLIP)


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
        return pd.DataFrame(), pd.DataFrame()
    stats = [s for s in stats if s in hist_pos.columns]
    if not stats:
        return pd.DataFrame(), pd.DataFrame()
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
    # The weighted TOTALS ride along beside the rates because an efficiency
    # RATIO (yards per target, catch rate, TD per carry) has to be formed
    # from two weighted sums, not from the quotient of two weighted rates -
    # the per-game rate divides both by the same weight_sum, so the quotient
    # is arithmetically identical here, but only as long as both stats have
    # the same weights, which stops being true the moment a stat is missing
    # for some games. Carrying the sums makes the ratio path independent of
    # that assumption instead of quietly depending on it.
    totals = grouped[num_cols].copy()
    totals.columns = [c[len('_num_'):] for c in totals.columns]
    return rates, totals


# ---------------------------------------------------------------------------
# VOLUME x EFFICIENCY
#
# WHY THIS EXISTS. Measured, not assumed: on 8,107 paired player-weeks across
# 2024-2025 (scripts/eval_weekly_model.py + the calibration decomposition in
# docs/weekly_projections_methodology.md), the flat per-game-rate model
# over-projected the top 15% of every position on EVERY counting stat at
# once - WR targets +11%, receptions +16%, receiving yards +18%; RB carries
# +8%, rushing yards +9%, receiving yards +18%; QB attempts, yards and TDs
# all ~+11-14%. A uniform over-projection of both volume AND the yardage
# built on it is the signature of one thing: a player's own observed rate
# being trusted too far, with the error compounding because yardage was
# being projected as its own independent per-game rate rather than as
# (opportunities x efficiency).
#
# The fix is structural, and it is what every published projection pipeline
# does: project OPPORTUNITY first (attempts, carries, targets), then apply a
# per-opportunity efficiency to it. Two things fall out of that which the
# flat version could not express:
#
#   1. Efficiency evidence is counted in OPPORTUNITIES, not games. A
#      receiver with 90 targets has a far better-measured yards-per-target
#      than one with 12, and both have the same number of games. The flat
#      model shrank both by the same games-based K.
#   2. Efficiency and volume regress at genuinely different speeds. Usage is
#      a decision a coaching staff makes and repeats; yards per target is
#      mostly the outcome of a handful of contested balls. Splitting them
#      lets the second be shrunk hard without also flattening the first.
#
# The K values below are OPPORTUNITY COUNTS at which a player's own rate
# carries half the weight, set from the published stabilization ranges for
# each rate (catch rate and yards per target stabilize fastest; touchdown
# rates and interception rates slowest, since both are rare events). They
# are NOT fitted to the backtest this change is measured on.
# ---------------------------------------------------------------------------

# (numerator, denominator, opportunities for a 50/50 blend).
EFFICIENCY_RATIOS = {
    'QB': [('passing_completions', 'passing_attempts', 120),
           ('passing_yards', 'passing_attempts', 150),
           ('passing_tds', 'passing_attempts', 250),
           ('passing_interceptions', 'passing_attempts', 250),
           ('rushing_yards', 'rushing_attempts', 90),
           ('rushing_tds', 'rushing_attempts', 150)],
    'RB': [('rushing_yards', 'rushing_attempts', 90),
           ('rushing_tds', 'rushing_attempts', 150),
           ('receptions', 'targets', 50),
           ('receiving_yards', 'targets', 70),
           ('receiving_tds', 'targets', 120)],
    'WR': [('receptions', 'targets', 50),
           ('receiving_yards', 'targets', 70),
           ('receiving_tds', 'targets', 120)],
}
EFFICIENCY_RATIOS['TE'] = EFFICIENCY_RATIOS['WR']

# The defense's effect on a ratio is its rating for the NUMERATOR divided by
# its rating for the DENOMINATOR - "does this defense give up more yards per
# target, over and above giving up more targets". Narrower than MATCHUP_CLIP
# because it is a ratio OF two clipped ratings and would otherwise be able to
# reach 1.3/0.75 = 1.73x on its own.
EFFICIENCY_MATCHUP_CLIP = (0.88, 1.14)


def _shrunk_ratio(own_num, own_den, prior_num, prior_den, pos_num, pos_den, evidence, k):
    """
    One efficiency ratio per player: his own (recency- and matchup-weighted)
    rate, shrunk toward a baseline by how many real OPPORTUNITIES stand
    behind it.

        w    = evidence / (evidence + k)
        rate = w * own + (1 - w) * baseline

    Baseline is his own PRIOR-SEASON ratio when he has one, else the
    position's current pooled ratio - the same "own history first, position
    baseline as the floor" ladder _blended_rate already uses for volume, so
    a rookie lands on the position and a veteran lands on himself.

    `evidence` is the RAW opportunity count, not the recency-weighted one:
    the weighted sum is the right estimate of the rate and the wrong measure
    of how much was observed (the decay shrinks it by ~40% for a full
    season). Same split _blended_rate already makes between `cur_rate`
    (weighted) and `cur_games` (raw), for the same reason.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        own = np.divide(own_num, own_den, out=np.full_like(own_num, np.nan, dtype=float),
                        where=own_den > 0)
        prior = np.divide(prior_num, prior_den, out=np.full_like(prior_num, np.nan, dtype=float),
                          where=prior_den > 0)
    pos_ratio = float(pos_num / pos_den) if pos_den > 0 else 0.0
    baseline = np.where(np.isfinite(prior), prior, pos_ratio)
    own = np.where(np.isfinite(own), own, baseline)
    w = evidence / (evidence + k)
    return w * own + (1 - w) * baseline


def _efficiency_matchup(overall, role_tables, role_sizes, opponents, player_roles, num_stat, den_stat):
    """clip(rating[numerator] / rating[denominator]) - the part of a
    defense's effect that is NOT already carried by the volume it allows."""
    num_mult = _role_adjusted_multiplier(overall, role_tables, role_sizes,
                                         opponents, player_roles, num_stat)
    den_mult = _role_adjusted_multiplier(overall, role_tables, role_sizes,
                                         opponents, player_roles, den_stat)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.divide(num_mult, den_mult, out=np.ones_like(num_mult), where=den_mult > 0)
    return np.clip(ratio, *EFFICIENCY_MATCHUP_CLIP)


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


def expected_snap_share(stats_df, name_col, team_col, as_of_week, lookback=4):
    """
    Per player: the share of his TEAM'S snaps he should be expected to play
    in the upcoming game - averaged over his team's last `lookback` played
    weeks, counting a week he did not appear in as a real ZERO rather than
    skipping it.

    THAT ZERO IS THE ENTIRE POINT, and it is what a per-game rate cannot
    express. A backup quarterback's per-game passing rate is computed over
    the games he actually appeared in - a garbage-time drive or two, in
    which he really did throw the ball - so it looks like a starter's rate
    at a small sample size, and every shrinkage path in this module then
    pulls him toward the POSITION's average per-game production, which is a
    starter's workload. Measured on the 2024-2025 backtest, that is by a
    wide margin the model's single worst failure mode: of the 25 largest
    upgrades this model made over a trailing-average baseline, sixteen were
    backup QBs (Joe Milton, Joshua Dobbs, Kedon Slovis, Jalen Milroe, Tyson
    Bagent, Taylor Heinicke, ...) projected 12-17 points for a week they
    would spend holding a clipboard. Snap share separates them cleanly and
    without a judgment call: those five sat at 2-21% of their team's snaps
    while a real starter sits at 100%.

    It also catches the opposite case for free, which is the one worth
    getting right: Tyler Shough went 4% -> 54% -> 90% -> 82% -> 95% -> 99%
    over six weeks of 2025. A four-week window reads him as a starter three
    weeks before a season-long average would.

    MEASURED AS SHARE-WHEN-ACTIVE, NOT SHARE-OF-TEAM-WEEKS, and that
    distinction was itself decided on measurement rather than taste. The
    first version averaged over the player's TEAM's last four games, scoring
    a week he missed as a real zero - which reads a backup correctly and a
    RETURNING STARTER completely wrong: a running back who missed two weeks
    hurt came back at a third of his own role, and the backtest showed it
    (whole-pool MAE improved either way, but the startable-RB pool got
    materially worse: MAE +0.28, rank correlation -0.09, losing 19 of 26
    weeks). Averaging over his last four APPEARANCES answers the question a
    start/sit call actually asks - "how big is his role when he is out
    there" - and leaves "is he playing at all" to the injury feed, which is
    the input that actually knows. Both readings still separate the backups
    this exists to catch: over their last four appearances Joe Milton sits
    at 21% of snaps and Joshua Dobbs at 18%, against 100% for a starter.
    """
    hist = _played_weeks_before(stats_df, as_of_week)
    if hist.empty or 'weekly_snap_pct' not in hist.columns:
        return pd.Series(dtype=float)
    hist = hist.copy()
    hist['week'] = pd.to_numeric(hist['week'], errors='coerce')
    snap = (pd.to_numeric(hist['weekly_snap_pct'], errors='coerce') / 100.0).clip(0, 1)
    hist = hist.assign(_snap=snap).dropna(subset=['_snap'])
    if hist.empty:
        return pd.Series(dtype=float)
    recent = hist.sort_values('week').groupby(name_col, observed=True).tail(lookback)
    return recent.groupby(name_col, observed=True)['_snap'].mean().rename('expected_snap_share')


def season_snap_share(stats_df, name_col, team_col=None):
    """
    A whole-season snap share, in one of TWO deliberately different senses:

      team_col=None  - mean share across the games he APPEARED in: "how big
                       was his role when he played". This is the unit a
                       per-game rate is denominated in, so it is the right
                       denominator for scaling a prior-season per-game rate
                       to a changed role.
      team_col given - total share divided by his team's whole season: "how
                       much of the year was he actually out there". This is
                       the right reading for a COLD START, where there is no
                       live injury feed to separately answer "is he playing"
                       and a part-time body would otherwise inherit a
                       starter's baseline off a handful of appearances.

    The distinction is not cosmetic: a QB3 who mopped up three blowouts at
    40% of snaps reads as 0.40 the first way and 0.07 the second, and week 1
    wants the second.
    """
    if stats_df.empty or 'weekly_snap_pct' not in stats_df.columns:
        return pd.Series(dtype=float)
    snap = (pd.to_numeric(stats_df['weekly_snap_pct'], errors='coerce') / 100.0).clip(0, 1)
    if team_col is None or team_col not in stats_df.columns or 'week' not in stats_df.columns:
        return snap.groupby(stats_df[name_col], observed=True).mean()
    weeks_by_team = (stats_df[[team_col, 'week']].astype({team_col: str})
                     .drop_duplicates().groupby(team_col)['week'].nunique())
    total = snap.groupby(stats_df[name_col], observed=True).sum()
    last_team = (stats_df.assign(_w=pd.to_numeric(stats_df['week'], errors='coerce'))
                 .sort_values('_w').groupby(name_col, observed=True)[team_col].last().astype(str))
    denom = last_team.reindex(total.index).map(weeks_by_team)
    return (total / denom.replace(0, np.nan)).clip(0.0, 1.0)


# How far a prior-season per-game rate may be scaled by a role change. A
# player whose snap share doubled really is a different player this year,
# but the rate being scaled was measured in a different offense with
# different teammates, so the upside is capped well below the raw ratio.
ROLE_VOLUME_CLIP = (0.0, 1.5)


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

    The per-player read-off is done on a PIVOTED numpy array, not by
    iterating a pandas groupby: the groupby-chop version of this loop was
    measured at 3.2s of a 4.4s whole-model build (~73% of the entire
    projection), all of it pandas slicing overhead rather than arithmetic.
    Same interpolation, same output (asserted equal on real 2025 data when
    this was changed), ~20x less time - which is what makes iterating on
    the model's own components affordable at all.
    """
    hist = _played_weeks_before(stats_df, as_of_week)
    if hist.empty or stat not in hist.columns or target_margins is None or len(target_margins) == 0:
        return pd.Series(dtype=float)
    margins = _team_week_margins(schedule_df)
    if margins.empty:
        return pd.Series(dtype=float)
    merged = hist.merge(margins, left_on=[team_col, 'week'], right_on=['Team', 'week'], how='inner')
    if merged.empty:
        return pd.Series(dtype=float)

    edges = np.array([lo for lo, _hi, _mid in SCRIPT_BUCKETS], dtype=float)
    mids = np.array([mid for _lo, _hi, mid in SCRIPT_BUCKETS], dtype=float)
    idx = np.searchsorted(edges, merged['margin'].to_numpy(dtype=float), side='left') - 1
    ok = (idx >= 0) & (idx < len(mids))
    merged = merged[ok].copy()
    if merged.empty:
        return pd.Series(dtype=float)
    merged['_bucket'] = mids[idx[ok]]

    bucket_means = merged.groupby([name_col, '_bucket'], observed=True)[stat].mean().unstack('_bucket')
    bucket_means = bucket_means.reindex(columns=mids)
    game_counts = merged.groupby(name_col, observed=True)['week'].nunique()
    season_avg = merged.groupby(name_col, observed=True)[stat].mean()

    players = bucket_means.index
    values = bucket_means.to_numpy(dtype=float)
    valid = ~np.isnan(values)
    targets = pd.Series(players).map(target_margins).to_numpy(dtype=float)
    counts = game_counts.reindex(players).to_numpy(dtype=float)
    avgs = season_avg.reindex(players).to_numpy(dtype=float)

    eligible = (counts >= 4) & (valid.sum(axis=1) >= 2) & np.isfinite(targets) & (avgs > 0)
    out = {}
    for i in np.flatnonzero(eligible):
        mask = valid[i]
        projected = float(np.interp(targets[i], mids[mask], values[i][mask]))
        out[players[i]] = float(np.clip(projected / avgs[i], *SCRIPT_CLIP))
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


# ---------------------------------------------------------------------------
# GAME ENVIRONMENT
#
# Two multipliers, both MEASURED on 2019-2023 - five seasons deliberately
# outside the 2024-2025 window every model change in this module is
# evaluated on, so these constants are not fitted to their own test.
# 21,330 player-games, restricted to players with 6+ games and a real
# scoring baseline, each game expressed as a ratio to that player's own
# season average so a good player in a good offense doesn't masquerade as a
# game-environment effect.
#
# 1. IMPLIED TEAM TOTAL. Log-log elasticity of that ratio to the team's
#    market-implied points (total_line/2 +/- spread_line/2):
#        QB 0.416   TE 0.301   RB 0.168   WR 0.140
#    Quarterbacks and tight ends live on team scoring; running backs and
#    receivers much less so, because a chunk of their production is volume
#    that survives a low-scoring game. Note this is a WEEKLY read and is not
#    in tension with data/odds_market.py's finding that a SEASON-long
#    implied-points multiplier made season projections worse: over a full
#    season a team's own usage history already encodes how good its offense
#    is, so the multiplier double-counts. For one specific week against one
#    specific opponent it is new information the player's own history does
#    not contain.
#
# 2. VENUE. Indoor (dome or closed roof) vs outdoors, same ratio measure:
#        QB 1.070   TE 1.052   WR 1.040   RB 1.001
#    Normalised below so the league-frequency-weighted average is 1.0, since
#    the baseline being multiplied is a player own mixed-venue average.
#    RB is left at exactly 1.0 - the measured effect was 0.1%, which is not
#    a signal, and a rounding artifact should not get a multiplier.
#
# DELIBERATELY NOT USED: WIND, despite being the LARGEST measured effect in
# the whole study (outdoor games, 15+ mph: QB 0.880 vs 1.017 in calm air,
# TE 0.907, WR 0.895, RB unaffected at 0.965 - teams run more into a wind,
# which is exactly the right shape for the effect to be real). It is not
# used because nflverse populates `wind` and `temp` AFTER a game is played,
# not when the schedule is published: a backtest would happily consume it
# and report an improvement the live model could never reproduce, because on
# the Thursday you actually set a lineup that column is empty. Recorded here
# so the next person to spot the wind column knows it was measured, and why
# it was left out anyway. A real forecast feed would make this usable.
# ---------------------------------------------------------------------------
GAME_TOTAL_ELASTICITY = {'QB': 0.42, 'RB': 0.17, 'WR': 0.14, 'TE': 0.30}
GAME_TOTAL_CLIP = (0.88, 1.15)
INDOOR_ROOFS = ('dome', 'closed')
# league-frequency-weighted to 1.0 at the ~28% of games played indoors
VENUE_MULT = {
    'QB': {'indoor': 1.049, 'outdoor': 0.981},
    'TE': {'indoor': 1.037, 'outdoor': 0.986},
    'WR': {'indoor': 1.028, 'outdoor': 0.989},
    'RB': {'indoor': 1.000, 'outdoor': 1.000},
}


def game_environment(schedule_df, week):
    """
    {team: {'implied': market-implied points, 'indoor': bool}} for one week,
    straight off the schedule feed - which carries `spread_line`,
    `total_line` and `roof` for every game, posted, for future weeks as well
    as played ones.

    Read here rather than through data.odds_market.fetch_game_lines even
    though both ultimately come from the same nflverse games file: this
    module already loads the schedule for opponents and margins, and the
    roof column is not exposed by that other path at all.

    SIGN CONVENTION IS THE ONE THING THAT CAN SILENTLY INVERT THIS:
    `spread_line` is positive when the HOME team is favored (documented in
    data/odds_market.py, and the same trap that file flags). Home implied is
    therefore total/2 + spread/2 and away is total/2 - spread/2.
    """
    if schedule_df is None or schedule_df.empty:
        return {}
    needed = {'week', 'home_team', 'away_team', 'total_line', 'spread_line'}
    if not needed.issubset(schedule_df.columns):
        return {}
    wk = schedule_df[pd.to_numeric(schedule_df['week'], errors='coerce') == week]
    if wk.empty:
        return {}
    total = pd.to_numeric(wk['total_line'], errors='coerce')
    spread = pd.to_numeric(wk['spread_line'], errors='coerce')
    roof = wk['roof'].astype(str).str.lower() if 'roof' in wk.columns else pd.Series('', index=wk.index)
    indoor = roof.isin(INDOOR_ROOFS)
    out = {}
    for team, imp, ind in zip(wk['home_team'], total / 2 + spread / 2, indoor):
        out[str(team)] = {'implied': float(imp) if pd.notna(imp) else None, 'indoor': bool(ind)}
    for team, imp, ind in zip(wk['away_team'], total / 2 - spread / 2, indoor):
        out[str(team)] = {'implied': float(imp) if pd.notna(imp) else None, 'indoor': bool(ind)}
    return out


def _game_env_multiplier(env, teams, pos, league_implied):
    """Per-player game-environment multiplier: implied-total elasticity x
    venue. 1.0 for any team without a posted line (the normal case more than
    a few weeks out), same degrade-gracefully convention as every other
    best-effort signal here."""
    elasticity = GAME_TOTAL_ELASTICITY.get(pos, 0.0)
    venue = VENUE_MULT.get(pos, {'indoor': 1.0, 'outdoor': 1.0})
    out = np.ones(len(teams))
    if not env:
        return out
    for i, team in enumerate(teams):
        entry = env.get(str(team))
        if not entry:
            continue
        mult = 1.0
        implied = entry.get('implied')
        if implied and league_implied and league_implied > 0 and implied > 0 and elasticity:
            mult *= float(np.clip((implied / league_implied) ** elasticity, *GAME_TOTAL_CLIP))
        mult *= venue['indoor'] if entry.get('indoor') else venue['outdoor']
        out[i] = mult
    return out


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


# ---------------------------------------------------------------------------
# TEAMMATE VACANCY
#
# When a team's WR1 is ruled out, his targets do not evaporate - they go to
# the other receivers, and the model's own history-based rates cannot know
# that, because every game in that history was played with him on the field.
# This is the single largest adjustment human analysts make week to week and
# the most common reason a market projection moves without any news about
# the player it moved for.
#
# HONESTLY FLAGGED: this component is NOT measured by
# scripts/eval_weekly_model.py and cannot be. It fires only off the live
# injury feed, and the backtest runs with apply_injury=False for the reason
# build_weekly_projections' docstring gives - nflverse's injury data carries
# no historical week granularity, so a backtest would be reading each
# player's LAST designation of the season and applying it to every week.
# There is no honest way to score it against 2024-2025 here, which also
# means it does not touch any number this pass reports: with injuries off it
# is inert, so the shipping model's measured results are the same with it on
# or off.
#
# It ships ON anyway, and that is a judgment call rather than a measurement:
# a receiver's targets demonstrably do not disappear when he is inactive,
# ignoring that is knowably wrong rather than merely unmeasured, and the
# component only fires on an explicit Out/Doubtful designation. The
# constants below are set conservatively for the same reason - an unmeasured
# adjustment should be small. Switchable off by dropping it from
# DEFAULT_FEATURES.
#
# The constants are conservative on purpose. Vacated usage does not all land
# on the remaining skill players (some of it becomes a different play call
# entirely, or goes to a body the model isn't projecting), and no single
# healthy teammate absorbs an unbounded share of it.
# ---------------------------------------------------------------------------
VACANCY_ABSORB = 0.75      # share of a sidelined player's volume that is re-used at all
VACANCY_MAX_GROWTH = 1.40  # no one teammate's own volume may grow by more than this
VACANCY_OUT_THRESHOLD = 0.5  # injury multiplier at or below which a player counts as out


def redistribute_vacated_usage(result, injury_mult):
    """
    Move a sidelined player's projected TARGETS and CARRIES onto his healthy
    teammates, proportional to each one's own projected volume, then rescale
    the stats that ride on that volume.

    Proportional to EXISTING volume, not evenly: the second receiver on a
    depth chart absorbs more of a missing first receiver than the fifth
    does, and "who already gets the ball" is the only measured proxy for
    that this function has. Yards, receptions and touchdowns are scaled by
    the same factor as the opportunity they come from rather than
    re-projected - a player getting 20% more targets is the model's claim
    here, and his per-target efficiency is not asserted to change with it.

    Returns (frame, n_players_adjusted). A team with nobody out, or nobody
    healthy left to absorb, comes back untouched.
    """
    if result.empty or not injury_mult or 'Team' not in result.columns:
        return result, 0
    out = result.copy()
    mult = out['Player'].map(injury_mult).fillna(1.0)
    sidelined = mult <= VACANCY_OUT_THRESHOLD
    if not sidelined.any():
        return out, 0

    groups = {'targets': ('receptions', 'receiving_yards', 'receiving_tds'),
              'rushing_attempts': ('rushing_yards', 'rushing_tds')}
    adjusted = set()
    for volume_col, dependents in groups.items():
        if volume_col not in out.columns:
            continue
        volume = pd.to_numeric(out[volume_col], errors='coerce').fillna(0.0)
        # A sidelined player's own projection has already been multiplied
        # down by the injury discount - to zero, for anyone actually ruled
        # Out - so what he vacates has to come from the PRE-injury volume the
        # position loop stashed for exactly this. Falls back to his post-
        # discount volume if that column isn't there (a caller assembling a
        # frame by hand, e.g. a unit test).
        full_col = f'_full_{volume_col}'
        pre_injury = (pd.to_numeric(out[full_col], errors='coerce').fillna(0.0)
                      if full_col in out.columns else volume)
        vacated = pd.Series(np.where(sidelined, np.maximum(pre_injury, volume), 0.0),
                            index=out.index)
        by_team = vacated.groupby(out['Team'].astype(str)).sum() * VACANCY_ABSORB
        healthy = (~sidelined) & (volume > 0)
        if not healthy.any():
            continue
        share_base = volume.where(healthy, 0.0)
        team_base = share_base.groupby(out['Team'].astype(str)).transform('sum')
        team_vacated = out['Team'].astype(str).map(by_team).fillna(0.0)
        gain = np.where(team_base > 0, share_base / team_base.replace(0, np.nan) * team_vacated, 0.0)
        gain = np.nan_to_num(gain)
        new_volume = np.minimum(volume + gain, volume * VACANCY_MAX_GROWTH)
        with np.errstate(divide='ignore', invalid='ignore'):
            factor = np.divide(new_volume, volume, out=np.ones(len(out)), where=volume > 0)
        factor = np.where(healthy, factor, 1.0)
        if not (factor > 1.0001).any():
            continue
        adjusted.update(out.index[factor > 1.0001])
        out[volume_col] = np.round(volume * factor, 2)
        for dep in dependents:
            if dep in out.columns:
                out[dep] = np.round(pd.to_numeric(out[dep], errors='coerce').fillna(0.0) * factor, 2)
    return out, len(adjusted)


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
def build_weekly_projections(year, week, scoring_mode='Full PPR', as_of_week=None, apply_injury=True,
                             features=None):
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
    feats = DEFAULT_FEATURES if features is None else frozenset(features)
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
    env = game_environment(schedule_df, week) if 'game_env' in feats else {}
    league_implied = None
    if env:
        implied_vals = [e['implied'] for e in env.values() if e.get('implied')]
        # League average for the WEEK being projected, not a fixed constant -
        # scoring environments move year to year, and this multiplier is
        # supposed to say "richer than a typical game", not "richer than 2019".
        league_implied = float(np.mean(implied_vals)) if implied_vals else None
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

    # Game-script curves are a WHOLE-POOL read (they never look at position),
    # so they are built once per stat here rather than once per
    # position-and-stat inside the loop below - the same curve was being
    # rebuilt up to four times, and this function was measured at ~73% of the
    # entire model build before that was fixed.
    all_target_margins = {}
    for _t, _m in target_margins.items():
        all_target_margins[_t] = _m
    player_margins = None
    script_by_stat = {}
    if all_target_margins and 'week' in stats_df.columns:
        player_margins = stats_df.drop_duplicates(subset=[name_col]).set_index(name_col)[team_col] \
            .astype(str).map(all_target_margins)
        for _stat in SCRIPT_ELIGIBLE_STATS:
            script_by_stat[_stat] = _vectorized_game_script_multiplier(
                stats_df, name_col, team_col, as_of_week, schedule_df, player_margins, _stat)

    # Expected snap share for the upcoming game, and the prior season's own
    # share to scale a prior-season per-game rate against - see
    # expected_snap_share's docstring for the measured failure this fixes.
    exp_share = pd.Series(dtype=float)
    prior_share = pd.Series(dtype=float)
    if 'role_volume' in feats:
        exp_share = expected_snap_share(stats_df, name_col, team_col, as_of_week)
        prior_share = season_snap_share(prior_played, prior_name_col) if not prior_played.empty \
            else pd.Series(dtype=float)
        if exp_share.empty and not prior_played.empty:
            # Cold start: nothing this season to read a role off, so last
            # season's stands in - the same "based on last season" fallback
            # every other input already makes for week 1. Measured over the
            # WHOLE season (see season_snap_share's team_col branch), not
            # over his appearances: with no games played there is no injury
            # feed answer to "is he even the starter", and the appearance
            # reading hands a mop-up QB3 a starter's baseline off three
            # blowouts. Confirmed on real 2026 rosters - it was putting a
            # third-string quarterback at QB5 overall.
            exp_share = season_snap_share(prior_played, prior_name_col, prior_team_col)

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
        role_tables, role_sizes, player_roles = {}, {}, {}
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
            anchor_week = (prior_max_week + 1) if (prior_max_week is not None
                                                   and not pd.isna(prior_max_week)) else None
            matchup_matrix = (build_quality_adjusted_matchup(
                                  prior_pos_rows, prior_name_col, stats, anchor_week)
                              if anchor_week is not None else pd.DataFrame())
            # Cold start reads ROLES off last season too, and keys them by
            # the player's own name in THAT season's frame - which is why the
            # lookup below re-keys through clean_name_exact rather than
            # assuming both seasons spell (or even name the column) the same
            # way (gotcha #35).
            if 'role_matchup' in feats and anchor_week is not None and not prior_pos_rows.empty:
                prior_roles = build_player_roles(prior_pos_rows, prior_name_col, pos)
                role_tables, role_sizes = build_role_matchup(
                    prior_pos_rows, prior_name_col, stats, anchor_week, prior_roles)
                roles_by_key = {clean_name_exact(pd.Series([p])).iloc[0]: r
                                for p, r in prior_roles.items()}
                player_roles = {p: roles_by_key.get(clean_name_exact(pd.Series([p])).iloc[0])
                                for p in cur[name_col]}
        else:
            pos_rows = hist[hist['position'].astype(str).str.upper() == pos]
            upcoming_opponent_map = dict(zip(cur[name_col], cur['Opponent']))
            matchup_matrix = build_quality_adjusted_matchup(pos_rows, name_col, stats, as_of_week)
            weighted_rates, weighted_totals = _weighted_player_rates(
                pos_rows, name_col, stats, as_of_week, matchup_matrix, upcoming_opponent_map)
            if 'role_matchup' in feats:
                player_roles = build_player_roles(pos_rows, name_col, pos)
                role_tables, role_sizes = build_role_matchup(
                    pos_rows, name_col, stats, as_of_week, player_roles)
        if cold_start:
            # No current-season rows to weight at all - every player's
            # in_season_rate below falls through to its own season_avg
            # fallback (0/0=0 for a cold-start cur), which is fine: cur_games
            # is 0 for everyone here too, so _blended_rate ignores it anyway
            # and lands entirely on prior_rate.
            weighted_rates, weighted_totals = pd.DataFrame(), pd.DataFrame()

        # Snap-share arrays for this position's frame. `share_ref` is the
        # position's own snap-weighted game count, which turns the position
        # baseline from "per game played" into "per FULL-SNAP game" - the
        # unit a player's expected share can then be multiplied against.
        use_role_volume = 'role_volume' in feats and not exp_share.empty
        if use_role_volume:
            player_share = cur[name_col].map(exp_share).to_numpy(dtype=float)
            if cold_start:
                # A cold-start pool is keyed off a roster file whose names
                # need not match last season's stats file exactly (gotcha
                # #35) - re-key through clean_name_exact rather than losing
                # every share to a spelling difference.
                share_keyed = pd.Series(exp_share.to_numpy(),
                                        index=clean_name_exact(pd.Series(exp_share.index)))
                share_keyed = share_keyed[~share_keyed.index.duplicated()]
                player_share = np.where(
                    np.isfinite(player_share),
                    player_share,
                    clean_name_exact(cur[name_col]).map(share_keyed).to_numpy(dtype=float))
            # A player with NO measured role anywhere - an undrafted rookie,
            # a practice-squad call-up - must not default to a full-time
            # one. np.nan_to_num(..., nan=1.0) did exactly that, and it put
            # three UDFA running backs at the very top of a week-1 board
            # (Jacory Croskey-Merritt at 24.7 projected points) purely
            # because "no snap data" was being read as "every snap". The
            # position's own median share is the honest stand-in.
            measured = player_share[np.isfinite(player_share)]
            default_share = float(np.median(measured)) if measured.size else 0.5
            player_share = np.where(np.isfinite(player_share), player_share, default_share)
            keys_rv = clean_name_exact(cur[name_col])
            prior_share_keyed = (pd.Series(prior_share.to_numpy(),
                                           index=clean_name_exact(pd.Series(prior_share.index)))
                                 if not prior_share.empty else pd.Series(dtype=float))
            player_prior_share = (keys_rv.map(prior_share_keyed).to_numpy(dtype=float)
                                  if not prior_share_keyed.empty else np.full(len(cur), np.nan))
            if cold_start:
                pos_rows_for_share = (prior_played[prior_played['position'].astype(str).str.upper() == pos]
                                      if not prior_played.empty else prior_played)
                share_name_col = prior_name_col
            else:
                pos_rows_for_share = hist[hist['position'].astype(str).str.upper() == pos]
                share_name_col = name_col
            pos_share_sum = 0.0
            if not pos_rows_for_share.empty and 'weekly_snap_pct' in pos_rows_for_share.columns:
                pos_share_sum = float(
                    (pd.to_numeric(pos_rows_for_share['weekly_snap_pct'], errors='coerce')
                     .fillna(0.0) / 100.0).clip(0, 1).sum())
        else:
            player_share = np.ones(len(cur))
            player_prior_share = np.full(len(cur), np.nan)
            pos_share_sum = 0.0

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
            if use_role_volume:
                # The position baseline is a STARTER's per-game production.
                # Handing it to a 5%-snap backup as his fallback is the bug
                # expected_snap_share exists to fix: re-denominate it per
                # full-snap game, then give each player his own expected
                # share of one.
                if pos_share_sum > 0:
                    source = prior if (cold_start and not prior.empty and stat in prior.columns) else cur
                    if stat in source.columns:
                        pos_rate_arr = np.full(
                            len(cur), float(source[stat].sum()) / pos_share_sum) * player_share
                # A prior-season per-game rate was measured at LAST year's
                # role. Scale it to this year's - a benched former starter
                # shouldn't carry his starter rate forward, and a promoted
                # backup shouldn't be held to his old one.
                role_scale = np.where(
                    np.isfinite(player_prior_share) & (player_prior_share > 0.02)
                    & np.isfinite(player_share),
                    np.clip(np.divide(player_share, player_prior_share,
                                      out=np.ones_like(player_share),
                                      where=player_prior_share > 0.02), *ROLE_VOLUME_CLIP),
                    1.0)
                prior_rate = prior_rate * role_scale

            # cur_games (RAW game count, not the recency-weighted
            # weight_sum) still drives the current-vs-prior-season shrinkage
            # below, deliberately - see _weighted_player_rates' docstring on
            # why that calibration is left undisturbed by this change.
            blended = _blended_rate(in_season_rate, cur_games, prior_rate, pos_rate_arr, stat,
                                    cur['role_confidence'].to_numpy(dtype=float))

            matchup_mult = _role_adjusted_multiplier(
                matchup_matrix, role_tables, role_sizes,
                cur['Opponent'].to_numpy(),
                cur[name_col].map(player_roles).fillna('').to_numpy(),
                stat)

            script_series = script_by_stat.get(stat)
            script_mult = np.ones(len(cur))
            if script_series is not None and not script_series.empty:
                script_mult = cur[name_col].map(script_series).fillna(1.0).to_numpy(dtype=float)

            proj_cols[stat] = blended * matchup_mult * script_mult

        pace_mult = pd.Series(1.0, index=cur.index)
        if league_pace and league_pace > 0:
            opp_pace = cur['Opponent'].map(pace['def_pace'])
            pace_mult = np.clip(opp_pace.fillna(league_pace) / league_pace, *PACE_CLIP)

        inj_mult = cur[name_col].map(injury_mult).fillna(1.0)
        env_mult = (_game_env_multiplier(env, cur['Team'].astype(str).to_numpy(), pos, league_implied)
                    if env else np.ones(len(cur)))

        # The PRE-INJURY volume, kept before the discount below multiplies it
        # away. A player ruled Out has an injury multiplier of exactly 0.0,
        # so his projected volume is 0 and there is nothing left to divide
        # back out of - and he is precisely the player whose usage the
        # vacancy pass exists to redistribute. Stashing it here is the only
        # place it still exists. (Caught by a unit test, not by inspection:
        # the first version tried to recover it as volume/multiplier and
        # silently redistributed zero for every Out player, which is the one
        # case that matters.)
        vacancy_volume = {}
        if 'teammate_vacancy' in feats:
            for stat in ('targets', 'rushing_attempts'):
                if stat in proj_cols:
                    vacancy_volume[stat] = np.clip(
                        proj_cols[stat] * pace_mult.to_numpy() * env_mult, 0.0, None)

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
                proj_cols[stat] * pace_mult.to_numpy() * inj_mult.to_numpy() * env_mult, 0.0, None)

        # VOLUME x EFFICIENCY. Every dependent stat is REBUILT here off the
        # opportunity count already projected above rather than kept as its
        # own independent per-game rate - see the EFFICIENCY_RATIOS block's
        # header for the measurement that motivated this. Deliberately after
        # the pace/game-script/injury scaling, not before: those all act on
        # OPPORTUNITY (a faster game is more snaps, a blowout is more passes,
        # a hobbled player plays less), so a stat derived from an already-
        # scaled opportunity count inherits them exactly once. Applying them
        # again here would square them.
        if 'volume_efficiency' in feats:
            opponents_arr = cur['Opponent'].to_numpy()
            roles_arr = cur[name_col].map(player_roles).fillna('').to_numpy()
            keys = clean_name_exact(cur[name_col])
            for num_stat, den_stat, k_opp in EFFICIENCY_RATIOS.get(pos, []):
                if num_stat not in proj_cols or den_stat not in proj_cols:
                    continue
                if weighted_totals is not None and not weighted_totals.empty \
                        and num_stat in weighted_totals.columns and den_stat in weighted_totals.columns:
                    own_num = cur[name_col].map(weighted_totals[num_stat]).to_numpy(dtype=float)
                    own_den = cur[name_col].map(weighted_totals[den_stat]).to_numpy(dtype=float)
                else:
                    own_num = cur[num_stat].to_numpy(dtype=float) if num_stat in cur.columns else np.zeros(len(cur))
                    own_den = cur[den_stat].to_numpy(dtype=float) if den_stat in cur.columns else np.zeros(len(cur))
                if not prior.empty and num_stat in prior.columns and den_stat in prior.columns:
                    prior_keyed = prior.set_index(clean_name_exact(prior[prior_name_col]))
                    prior_num = keys.map(prior_keyed[num_stat]).to_numpy(dtype=float)
                    prior_den = keys.map(prior_keyed[den_stat]).to_numpy(dtype=float)
                else:
                    prior_num = np.full(len(cur), np.nan)
                    prior_den = np.full(len(cur), np.nan)
                # Position baseline from whichever season actually has rows -
                # a cold start has zeros for everyone in `cur` by
                # construction, same reasoning as the volume side above.
                if cold_start and not prior.empty and num_stat in prior.columns and den_stat in prior.columns:
                    pos_num, pos_den = float(prior[num_stat].sum()), float(prior[den_stat].sum())
                elif num_stat in cur.columns and den_stat in cur.columns:
                    pos_num, pos_den = float(cur[num_stat].sum()), float(cur[den_stat].sum())
                else:
                    continue
                evidence = (cur[den_stat].to_numpy(dtype=float) if den_stat in cur.columns
                            else np.zeros(len(cur)))
                ratio = _shrunk_ratio(own_num, own_den, prior_num, prior_den,
                                      pos_num, pos_den, evidence, k_opp)
                eff_mult = _efficiency_matchup(matchup_matrix, role_tables, role_sizes,
                                               opponents_arr, roles_arr, num_stat, den_stat)
                proj_cols[num_stat] = np.clip(proj_cols[den_stat] * ratio * eff_mult, 0.0, None)

        out = pd.DataFrame({
            'Player': cur[name_col], 'Pos': pos, 'Team': cur['Team'], 'Opponent': cur['Opponent'],
            'Games This Season': cur['Games'].astype(int),
            'Role Confidence': cur['role_confidence'].round(2),
        })
        for stat, values in proj_cols.items():
            out[stat] = np.round(values, 2)
        for stat, values in vacancy_volume.items():
            out[f'_full_{stat}'] = np.round(values, 2)
        proj_dicts = out[[s for s in stats if s in out.columns]].to_dict('records')
        # Floored at zero, same reasoning as the raw per-stat floor above -
        # a projection is an expectation, and no real player has a negative
        # one. Score_projected_stats subtracts for interceptions, so a
        # near-zero-volume passer (a WR/RB with a trace INT rate and
        # otherwise nothing projected) can otherwise net a small negative
        # total even with every raw stat already non-negative.
        out['Model Proj Pts'] = [max(0.0, score_projected_stats(d, scoring_mode)) for d in proj_dicts]
        if 'calibration' in feats and pos in WEEKLY_CALIBRATION:
            # Applied to the POINT TOTAL only, never to the individual stat
            # line above it. The stat line is what the projection actually
            # claims will happen on the field and it stays internally
            # consistent (a receiving-yards figure that matches its own
            # reception count); the calibration is a statement about this
            # model's dispersion, not about football, so pushing it back
            # into the yards and carries would corrupt a line that is
            # displayed and read on its own terms.
            slope, intercept = WEEKLY_CALIBRATION[pos]
            raw = out['Model Proj Pts'].to_numpy(dtype=float)
            out['Model Proj Pts'] = np.round(
                np.clip(np.minimum(raw, intercept + slope * raw), 0.0, None), 2)
        out['Injury Status'] = out['Player'].map(lambda p: 'Out/Doubtful' if injury_mult.get(p, 1.0) < 0.9 else '')
        all_rows.append(out)

    if not all_rows:
        return pd.DataFrame(), {'reason': f'No projectable players found for week {week}.'}
    result = pd.concat(all_rows, ignore_index=True).sort_values('Model Proj Pts', ascending=False).reset_index(drop=True)

    vacancy_adjusted = 0
    if 'teammate_vacancy' in feats and injury_mult:
        # Runs on the assembled frame, not inside the position loop: a
        # missing tight end's targets go to receivers and backs too, so the
        # redistribution has to see a whole TEAM at once.
        result, vacancy_adjusted = redistribute_vacated_usage(result, injury_mult)
        result = result.drop(columns=[c for c in result.columns if c.startswith('_full_')])
        if vacancy_adjusted:
            stat_cols = [c for c in _ALL_PROJECTION_STATS if c in result.columns]
            recomputed = [max(0.0, score_projected_stats(d, scoring_mode))
                          for d in result[stat_cols].to_dict('records')]
            if 'calibration' in feats:
                slopes = result['Pos'].map(lambda p: WEEKLY_CALIBRATION.get(p, (1.0, 0.0)))
                recomputed = [min(v, sl * v + ic) for v, (sl, ic) in zip(recomputed, slopes)]
            result['Model Proj Pts'] = np.round(np.clip(recomputed, 0.0, None), 2)
            result = result.sort_values('Model Proj Pts', ascending=False).reset_index(drop=True)

    result = result.drop(columns=[c for c in result.columns if c.startswith('_full_')])
    meta = {'reason': None, 'year': year, 'week': week, 'as_of_week': as_of_week,
            'players': int(len(result)), 'scoring': scoring_mode, 'cold_start': cold_start,
            'features': sorted(feats), 'vacancy_adjusted': vacancy_adjusted}
    return result, meta
