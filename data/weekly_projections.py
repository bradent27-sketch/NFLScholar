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
(`build_team_game_quality_adjusted_matchup` / `_weighted_player_rates`, replacing an
earlier flat 60% trailing-4-game / 40% season-average split). Three stacked
adjustments on every past game, per explicit request:

  1. RECENCY - most recent game weight 1.0, decaying by RECENCY_DECAY per
     game back, a smooth curve rather than a hard 4-game cutoff.
  2. MATCHUP STRENGTH - a big game against a bad defense is scaled DOWN
     before it's averaged into the player's rate, and a quiet game against
     a good defense is scaled UP - so a huge day against a defense that
     gives that up to everyone doesn't inflate a player's real level, and a
     quiet day against a tough defense doesn't understate it. The defense
     rating used for this is itself a recency-weighted, offense-position
     team-game observed/expected profile. It controls for the overall
     opponent unit without allowing a backup player's tiny personal average
     to become a defense-wide signal; the same matrix prices the upcoming
     opponent below.
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
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from data.transforms import (load_and_merge_data, OFFENSE_PROJECTION_STATS,
                             score_projected_stats)
from data.loaders import load_team_pace, load_team_weekly_plays, load_schedule, CACHE_TTL_SECONDS
from data.utils import clean_name_exact
from data.ourlads_depth_charts import (
    load_ourlads_snapshot, build_ourlads_projection_signal,
    apply_ourlads_starter_roster_overlay,
)
from data.rb_role_allocator import (
    classify_functional_position, derive_preseason_rb_capacities,
    allocate_preseason_rb_roles, redistribute_rb_vacancy_with_allocator,
    analyze_rb_role_segments, derive_rb_allocator_segment_fields,
    INELIGIBLE_ROSTER_STATUSES,
)
from data.player_aliases import canonical_player_key, stable_roster_identity_keys
from data.availability_overrides import (
    load_availability_overrides, resolve_target_week_availability,
)
from data.pass_capacity_allocator import apply_pass_capacity_conservation
from data.qb_volume_blend import blend_qb1_volume
from data.fantasypros_availability import load_fantasypros_availability
from data.matchup_signals import defense_stat_rank
from data.weekly_distribution import player_distribution
from data.pff_alignment import (
    load_weekly_alignment_profiles, load_season_alignment_prior,
    lookup_alignment_profile, load_weekly_alignment_defense_profiles,
    alignment_defense_residual_multiplier, ALIGNMENT_DEFENSE_SUPPORTED_POSITIONS,
    load_weekly_scheme_profiles, lookup_scheme_profile,
    load_weekly_scheme_defense_profiles, scheme_defense_residual_multiplier,
    SCHEME_DEFENSE_SUPPORTED_POSITIONS,
    blend_alignment_profile_toward_prior2,
    _attach_offense_leave_one_out_baselines,
)

# Maps this file's stat names onto pff_alignment.py's ALIGNMENT_DEFENSE_STATS
# naming ('yards', not 'receiving_yards') - the only three stats the defense
# residual preview computes. Touchdowns stay neutral on purpose: too sparse
# per slot/non-slot split to trust, and alignment_defense_residual_multiplier
# itself refuses to score them (see its own touchdown guard) even if asked.
ALIGNMENT_SCORING_STAT_MAP = {
    'targets': 'targets', 'receptions': 'receptions', 'receiving_yards': 'yards',
}

# Optional FIXED mixing weight (scheme's own share) per position, overriding
# 'v2_scheme_alignment_blend''s default evidence-weighted blend - empty by
# default, meaning every position uses that default. Built 2026-08-27 per
# explicit request: the component backtest program's evidence-weighted blend
# showed a real, positive pattern for BOTH positions but with opposite
# implications - WR's blend beat scheme-alone outright (best of every variant
# tested), while TE's blend was real but weaker than scheme-alone's own
# START-TE win, i.e. alignment context was "diluting, not hurting" TE's
# result. The user wants to KEEP alignment context for TE regardless (not
# drop it for a pure scheme-wins-outright design) while weighting it less
# than the ~50/50 the evidence-weighted default happened to land near - this
# lets scripts/sweep_scheme_blend_weight.py test specific fixed ratios (e.g.
# {'TE': 0.75}) to find where that tradeoff actually lands, rather than
# guessing at one number. Set by a sweep script via monkeypatch + explicit
# build_weekly_projections.clear() between runs (the cache key is `features`,
# which does not change across weight values - mutating this dict alone
# would silently serve a stale cached result otherwise); never set here by
# default.
SCHEME_ALIGNMENT_BLEND_FIXED_WEIGHT = {}

# One past the last real week of a completed prior season, so every one of
# its weeks passes discover_weekly_alignment_exports' strict
# `week < as_of_week` eligibility check when this app cold-starts a new
# season's Week 1 alignment-defense evidence off that prior season's full
# archive (see build_weekly_projections' cold_start branch around its
# load_weekly_alignment_defense_profiles call, further down this file).
# Raised from 19 to 23 on 2026-08-25 when postseason weeks (19 WC / 20 DIV /
# 21 CONF / 22 SB - same nflverse week numbering used elsewhere in this app)
# were added to the weekly PFF archive and that call site started passing
# include_postseason=True: the old sentinel of exactly 19 excluded every
# postseason week (`week < 19` fails for week 19 itself), silently
# defeating the whole point of adding them.
PFF_ALIGNMENT_DEFENSE_COLD_START_AS_OF_WEEK = 23

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
# a binary in/out of a fixed window. It is also used for a CURRENT-SEASON
# defense profile. A completed prior-season defense uses the broad
# cross-season floor below instead, because last January's finale should not
# be treated as last Sunday.
RECENCY_DECAY = 0.85

# When a prior season is the only defensive sample (Week 1) or is being
# blended into an early-season defense, do NOT let its final few games behave
# as if they happened yesterday.  The ordinary 0.85 decay would make an early
# prior-season game worth roughly one tenth of the finale, which is too sharp
# across an offseason and can turn a few late-season game scripts into a
# large Week 1 carry/pass matchup boost.  This floor means every game keeps
# an 80% full-season baseline while the remaining 20% retains a modest
# late-season preference: final game = 1.00, eight games earlier ~= 0.86,
# fourteen games earlier ~= 0.82. Raised from an original 0.75/25% split per
# explicit user preference (weight the late season a little less, the whole
# season a little more) - a deliberately small nudge, not a rework of the
# mechanism; does NOT touch the separate current-season/prior-season blend
# weight (how fast a defense profile shifts off last year onto this year's
# games), which is its own knob elsewhere and was left exactly as it was.
PRIOR_SEASON_DEFENSE_RECENCY_FLOOR = 0.80

# CANDIDATE, built 2026-08-27 - see 'v2_cold_start_regression' in
# MODEL_FEATURES. At true cold start (Week 1, no current-season games yet -
# see `cold_start = hist.empty`), every matchup-side multiplier this model
# computes (the defense/role matchup, and the script/pace/availability/
# environment "context" group) is built entirely from LAST season's evidence,
# with no way yet to tell whether a defense - or a game environment read off
# a still-thin market - will actually repeat this year. Per explicit request:
# a defense that graded as elite or as terrible last season is not equally
# likely to repeat that exact grade, and treating the Week 1 read at full
# strength overvalues both tails. This pulls every such multiplier a fixed
# fraction of the way back toward neutral (1.0) ONLY at cold start - 0.25
# means 75% of the computed multiplier's own deviation from 1.0 survives, 25%
# regresses to "assume average until proven otherwise." Applied identically
# to the defense-side multiplier and each context factor (not the product of
# all of them at once) - in the near-1.0 range these multipliers normally
# live in, the two are nearly indistinguishable, and per-factor is far
# simpler to apply correctly across this file's several separate multiplier
# sites. NOT YET BACKTESTED - built and gated, first measurement pending.
COLD_START_MULTIPLIER_REGRESSION = 0.25

# Preseason QB workload selection deliberately does not reuse the generated
# Depth Charts table.  That table is useful for browsing a roster, but it
# ranks candidates from inherited snap data and therefore cannot reliably know
# an announced new-season starter.  ``data/qb1_overrides.csv`` is a small,
# explicit, user-maintained layer for those ambiguous rooms instead.
QB1_OVERRIDE_PATH = Path(__file__).with_name('qb1_overrides.csv')
QB1_OVERRIDE_COLUMNS = ('year', 'team', 'player')
# A QB who handled at least 65% of his prior team's full-season offensive
# snaps is a clear incumbent for this narrow preseason-workload purpose.
# It intentionally does not bless a player who only started part of last year:
# those rooms require an explicit current-season choice.
QB1_AUTO_INCUMBENT_MIN_SHARE = 0.65

# A player-game is excluded from *that player's* full-game rate history only
# when the recorded snaps give strong evidence that it was interrupted rather
# than an ordinary lower-workload outing.  There is no historical injury
# timestamp in the local feed, so these are deliberately high-precision
# guards, not a claim to identify every injury or benching.  Missing/zero
# snap data is never classified.
PARTIAL_GAME_REFERENCE_APPEARANCES = 3
PARTIAL_GAME_MIN_REFERENCE_APPEARANCES = 2
PARTIAL_GAME_ESTABLISHED_SNAP_SHARE = 0.65
PARTIAL_GAME_ABSOLUTE_MAX_SNAP_SHARE = 0.50
PARTIAL_GAME_RELATIVE_MAX_SHARE = 0.60
PARTIAL_REPLACEMENT_MIN_SNAP_SHARE = 0.20
PARTIAL_REPLACEMENT_MAX_SNAP_SHARE = 0.75
PARTIAL_REPLACEMENT_MAX_PRIOR_SHARE = 0.30
QB_SPLIT_MIN_SNAP_SHARE = 0.20
QB_SPLIT_MAX_SNAP_SHARE = 0.80
QB_SPLIT_MIN_COMBINED_SHARE = 0.90
QB_SPLIT_MAX_COMBINED_SHARE = 1.10
# A final score alone cannot prove when a player sat.  This only acts when a
# proven full-time player also logged a sharply reduced share in an extreme,
# winning blowout.
SEVERE_BLOWOUT_MARGIN = 28.0
SEVERE_BLOWOUT_MAX_SNAP_SHARE = 0.65
SEVERE_BLOWOUT_RELATIVE_MAX_SHARE = 0.75

# In season, a single recent full-snap QB is enough to be an automatic
# upcoming starter.  Anything less clear stays explicitly unresolved until
# the user selects a QB1 rather than granting live volume to every backup.
QB1_INSEASON_MIN_SNAP_SHARE = 0.70
QB1_INSEASON_MIN_LEAD = 0.20
# An old starter's last active appearance should not compete with the QB who
# has actually handled the club's most recent games.  Two team games leaves
# room for a bye or one missed outing but does not revive a September starter
# in December merely because his own last appearances were full-snap games.
QB1_INSEASON_MAX_STALE_TEAM_GAMES = 2

# A cold start has no current-year injury/participation evidence. For an
# established skill player who is still on the same team, last season's
# *active-game* role is more relevant than the fraction of the calendar he
# happened to miss. These guards deliberately exclude rookies, sparse fill-in
# backups, and team changes, where a full-season participation share remains
# the safer preseason reading.
COLD_START_RETURNING_ROLE_MIN_GAMES = {'RB': 8, 'WR': 6, 'TE': 6}
COLD_START_RETURNING_ROLE_MIN_ACTIVE_SHARE = 0.60
COLD_START_RETURNING_ROLE_CAP = 0.95

# Two-season cold-start blend, added 2026-08-24 at explicit request: a down
# year, an abbreviated/injury-shortened season, or a genuinely thin sample
# should not fully override what a healthy prior-prior season showed. 8
# games is this app's own "roughly half a season" line (COLD_START_
# RETURNING_ROLE_MIN_GAMES uses the same idea per-position); a player at or
# above it gets a flat, modest look back at the older season, one below it
# gets progressively more of it as his 2025 sample thins out. Capped at 0.55
# (not 1.0) even at zero 2025 games with SOME 2025 read still present - a
# true "no 2025 row at all" case bypasses this weight entirely and uses 2024
# outright (see _blend_with_prior2's only_prior2 branch), so this cap only
# governs a real but thin 2025 sample, not a total absence.
PRIOR2_BLEND_FULL_SEASON_GAMES = 8.0
PRIOR2_BLEND_BASE_WEIGHT = 0.20
PRIOR2_BLEND_MAX_WEIGHT = 0.55

# Asymmetric dampening + full-season decay, added 2026-08-24 per follow-up
# request: stay bullish on an ascending player. A blend that would LOWER a
# share/rate (his 2024 was worse than his 2025 - e.g. a 2025 breakout off a
# thin or absent 2024 role) is cut to roughly a third weight; a blend that
# RAISES it (a genuine regression - Lamar Jackson/Jayden Daniels: strong
# 2024, an injury-shortened or down 2025) keeps the full weight above. See
# _blend_with_prior2's "pulls_down" branch.
PRIOR2_BLEND_DECREASE_DAMPENING = 0.35

# 2024's influence fades to zero as THIS player accumulates his own 2026
# games - not a calendar-week cutoff, so an early-season injury absence
# doesn't burn the fade down before he's actually played. Reuses the same
# "8 games = a full season of evidence" line as PRIOR2_BLEND_FULL_SEASON_
# GAMES above, just applied to the CURRENT season's sample instead of the
# immediately-prior one. Used for prior_share (role_scale's denominator,
# read every week) - never for exp_share, which only reads 2024 at cold
# start, since once real 2026 games exist it already IS the observed role.
PRIOR2_DECAY_GAMES_2026 = 8.0

# A 2024 row under this many games is a practice-squad/inactive-all-year
# entry, not a season - dropped from every two-season blend below, same as
# no row at all, rather than treated as a real "he was bad" read.
PRIOR2_BLEND_MIN_GAMES = 1.0

# The per-stat two-season rate blend (see prior_rate's construction in the
# main per-position loop) only applies to counting/volume stats - the same
# STAT_K==3 bucket _blended_rate already trusts with the least shrinkage.
# TD rate has its own dedicated, more conservative mechanism (blend_
# comparable_td_priors, opportunity-gated, fixed weight) and interceptions
# get no two-year blend at all - both stay on their existing behavior.
PRIOR2_RATE_BLEND_STATS = frozenset(stat for stat, k in STAT_K.items() if k == 3)

# A locally imported Ourlads depth chart is evidence that a player belongs in
# a current formation / rotation; it is not a source of actual snap counts.
# These are deliberately modest *floors*, applied only to players changing
# teams or otherwise carrying thin/no prior role evidence.  A listed WR
# starter can be a two-TE-package player, so none of these means "100% of
# snaps" or an equal share among LWR/RWR/SWR.  Rank 2 recognizes a visible
# rotational role without inventing meaningful work for deeper reserves.
OURLADS_PRESEASON_ROLE_FLOORS = {
    'RB': {1: 0.55, 2: 0.20},
    'WR': {1: 0.45, 2: 0.16},
    'TE': {1: 0.50, 2: 0.18},
}
OURLADS_LOW_EVIDENCE_PRIOR_SHARE = 0.20

# Deep-bench WR/TE receiving-volume cutoff. Added 2026-08-25 per the user's
# own read of a real defect: a team's real WR1-4/TE1-3 each get a credible
# share, but every player ranked below them on the SAME team still carried a
# small nonzero share too - individually modest, but data.pass_capacity_
# allocator's team-target fit (see its own module docstring) splits each
# team's LEFTOVER budget proportionally across its whole tail, so a pile of
# should-be-negligible WR6/WR7/WR8/TE4+ claims measurably shrinks what's
# left for the real WR5/committee-TE names sharing that same tail - "team
# capacity delta" docking legitimate players. Per the user: "the 8th WR
# doesn't need a projection...WR5 may get about 1 a week but past that
# probably need to cut it off...past TE3 don't need projected receiving
# volume." Ranked by each player's own current ``player_share`` within his
# team (already reflects any Ourlads role floor/pull applied above), so this
# reads the model's own best current understanding of role rather than a
# separate guess, and works identically at cold start and in-season alike.
WR_DEPTH_RANK_SMALL_ROLE = 5              # "about 1 a week" - real but minor
WR_DEPTH_RANK_SMALL_ROLE_SHARE_CAP = 0.05
WR_DEPTH_RANK_CUTOFF = 6                  # rank 6 and deeper - cut off
TE_DEPTH_RANK_CUTOFF = 4                  # rank 4 and deeper - cut off
RECEIVER_DEPTH_CUTOFF_SHARE_CAP = 0.01

# How many weeks into a season the Ourlads role floor above keeps ANY pull,
# once real snaps exist (cold_start=False). Per the user: "the depth charts
# are mostly for the first week to get a gauge on new players and not for
# later into the season...it is unlikely new versions will be uploaded
# throughout the season as the snap counts and snaps will speak more than
# depth charts." A hard cliff at week 2 already guarantees the second half -
# a stale chart can never outrank real snap data - but a single-game sample
# by week 2-3 is still thin for exactly the new/thin-evidence players this
# floor exists to help, so its PULL (not its binary application) fades
# linearly to zero by this many weeks rather than vanishing in one step.
EARLY_SEASON_DEPTH_CHART_DECAY_WEEKS = 4

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
    'role_matchup',       # defense ratings conditioned on the player's own ROLE -
                           # superseded by 'v2_continuous_roles' in DEFAULT_FEATURES
                           # (both gate the same branch; v2_continuous_roles adds a
                           # continuous rather than tiered role read on top)
    'redzone_tds',        # touchdowns from red-zone opportunity, not a raw TD rate
    'role_trend',         # a step change in snap share, not a decayed average of it
    'volume_faced',       # opponent pass/rush volume faced, split (pace is one number)
    'game_env',           # market total, roof/venue, wind, rest
    'teammate_vacancy',   # an OUT teammate's usage redistributed (live only) -
                           # superseded by 'v2_vacancy' in DEFAULT_FEATURES (same
                           # gate; v2_vacancy adds the ledger tracking on top)
    'qb1_override',        # expected-QB1 selection / backup-volume gate
    'v2_output_contract',  # V2's output-column contract
    'v2_as_of_guard',      # strict as-of-week cutoff guard
    'v2_adaptive_volume',  # sample-size-adaptive volume blend
    'v2_td_two_year_prior',  # two-year, not one-year, TD-rate prior
    'v2_defense_prior',    # defense rating prior/shrinkage revision
    'v2_continuous_roles', # continuous (not tiered) role-share read; see role_matchup above
    'v2_channel_matchups', # per-route-channel matchup read
    'v2_alignment_contract',  # PFF alignment data contract/validation
    'v2_pff_alignment_matchup',  # WR/TE slot/non-slot defense residual - built and
                                  # measured on 2025 weeks 2-18: lost on the startable
                                  # WR/TE pool (see DEFAULT_FEATURES's comment below).
                                  # Kept ON in DEFAULT_FEATURES anyway at the user's
                                  # explicit request (2026-08-26) so it stays live and
                                  # inspectable on the board while the mechanism is
                                  # re-studied, rather than gated behind a second model.
    'v2_availability',     # availability/injury resolver revision; see v2_fantasypros_availability
    'v2_vacancy',           # see teammate_vacancy above
    'v2_preseason_rb_allocator',  # team-constrained cold-start RB roles
    'v2_pass_capacity',    # team-constrained WR/TE/RB target conservation
    'v2_qb_volume_blend',  # QB1 volume: team dropbacks x evidence-weighted player style
    'v2_fantasypros_availability',  # FantasyPros-sourced injury signal, healthy by default
    'calibration',        # shrink toward the positional mean, fitted out-of-sample
    'v2_role_change_by_stat',  # per-stat ROLE_CHANGE_K_REDUCTION instead of one
                                # shared constant - dampens RB targets AND
                                # rushing_attempts's role-change trust-
                                # acceleration. SHIPPED 2026-08-27 after a
                                # corrected re-test measured a real RB win;
                                # see ROLE_CHANGE_K_REDUCTION_RB_CARRY's own
                                # note for the full history (a first attempt
                                # that tested inert, and why)
    'v2_scheme_matchup',    # WR/TE man/zone allowed-by-scheme multiplier -
                             # built 2026-08-27 as a redesigned, wired-in
                             # version of the man/zone "scheme_defense"
                             # pipeline that had sat preview-only since the
                             # 2026-08-23 commit that added it. See
                             # scheme_defense_residual_multiplier's own
                             # 2026-08-27 redesign note in data/pff_alignment.py
                             # (it inherited alignment's pre-redesign flaws
                             # and was fixed to match before ever being
                             # backtested) and this file's own note below.
                             # Not yet in DEFAULT_FEATURES - first backtest
                             # pending.
    'v2_scheme_alignment_blend',  # evidence-weighted blend of alignment and
                             # scheme instead of either replacing the broad
                             # matchup alone or one outright replacing the
                             # other - see the note above
                             # 'v2_scheme_alignment_blend' in this file's
                             # per-stat loop for the full reasoning. Mutually
                             # exclusive with 'v2_scheme_matchup' in practice
                             # (both test different combination strategies);
                             # if both are set, the blend wins.
    'v2_cold_start_regression',  # pull every matchup/context multiplier 25%
                             # of the way back toward neutral 1.0, but ONLY
                             # at true cold start (Week 1) - see
                             # COLD_START_MULTIPLIER_REGRESSION's own note.
                             # Not yet backtested.
    'v2_game_total_elasticity',  # implied game-total scaling ALONE, unbundled
                             # from 'game_env' (rejected as a bundle, +0.012
                             # MAE) - see _game_env_multiplier's own note.
                             # Not yet backtested standalone.
    'v2_venue_mult',        # indoor/outdoor venue scaling ALONE, same
                             # unbundling as above. Not yet backtested
                             # standalone.
    'v2_defense_prior_games_override',  # sweep hook for DEFENSE_PRIOR_GAMES
                             # (currently 4.0, never itself backtested) - see
                             # that constant's own note and
                             # scripts/sweep_defense_prior_games.py. A no-op
                             # unless DEFENSE_PRIOR_GAMES_OVERRIDE is also set.
)
# What the app actually runs - the single standard model. Until 2026-08-26
# this file offered two configurations: this set (then called "V1, released
# baseline") and a separate, larger "V2, experimental" set the UI let you
# opt into. That toggle was retired 2026-08-26 at the user's explicit
# request once V2 had been evaluated long enough to become the standard -
# see git history (this file, ui/tabs/rankings.py) for the old dual-model
# UI if it's ever needed again. DEFAULT_FEATURES is now exactly the former
# V2_EXPERIMENTAL_FEATURES set plus 'calibration' (see that name's own note
# below for why it's added back in). `volume_efficiency` and `game_env` are
# NOT in it: both were built, measured on the same 8,107 paired
# player-weeks, and left off because they did not help (volume_efficiency
# +0.051 MAE / -0.005 rank-corr, winning 5 of 26 weeks; game_env +0.012 MAE
# at the measured elasticity and +0.006 at half of it, winning 10-11 of 26).
# The code stays, switchable, with the measurement written next to each -
# see docs/weekly_projections_methodology.md. ``qb1_override`` is a
# participation-correctness layer for both cold starts and in-season QB
# rooms, not a claim of a new fitted backtest improvement.
#
# ``calibration`` was never part of the old V2_EXPERIMENTAL_FEATURES set (it
# shipped uncalibrated), which the 2026-08-26 retirement carried forward by
# default - the user was asked explicitly and chose to ADD calibration to
# the new standard rather than ship uncalibrated, since it's a measured fix
# for a real top-of-pool over-projection bug (see WEEKLY_CALIBRATION's own
# comment). Because the shipping component set changed substantially that
# same day (the former V2-only components joined it), the calibration line
# was RE-FITTED against the new CALIBRATION_INPUT_FEATURES via
# scripts/fit_weekly_calibration.py - re-run it again any time DEFAULT_FEATURES
# changes, since the line describes the dispersion of the model it's applied to.
#
# ``v2_pff_alignment_matchup``'s WR/TE allowed-by-alignment multiplier was
# BUILT AND MEASURED 2026-08-24 as an INCREMENTAL residual multiplied on top
# of the broad role/defense matchup, paired A/B against the rest of this set
# on 2025 weeks 2-18 (week 1 excluded - cold start / season-prior fallback,
# a different code path). Whole-pool WR looked like a rounding-error win
# (-0.003 MAE) but that pool is dominated by bench players the eval script's
# own docstring calls "trivially easy to rank." The startable subset lost on
# both stats: START-WR +0.022 MAE, START-TE +0.082 MAE (worst result of any
# scope measured). Likely cause at the time: only one season (2025) of
# weekly-grain PFF data existed, and the incremental design divided by a
# position-normal alignment mix on top of an already-thin per-alignment
# sample - two compounding sources of noise.
#
# REDESIGNED 2026-08-26 per explicit request (see alignment_defense_residual_
# multiplier's own docstring in data/pff_alignment.py): the position-normal-
# mix division and a separate confidence-based shrink-to-1.0 were both
# removed, and this component now REPLACES the broad role/defense matchup
# for WR/TE targets/receptions/receiving_yards (see the alignment_player_
# factor block just below) rather than multiplying on top of it - directly
# addressing the "two independent opinions of the same matchup" redundancy
# and the "normalizes too aggressively toward league average" complaint that
# prompted this change. Same day, a full 2024 weekly-grain PFF archive
# (pff_imports/2024/weekly/) was also confirmed present, which the original
# rejection's own "likely cause" pointed at as the fix this needed.
#
# RE-MEASURED 2026-08-26 (same day as the redesign) via scripts/eval_weekly_model.py's
# paired A/B harness, isolating exactly 'v2_pff_alignment_matchup' (DEFAULT_FEATURES
# with vs. without that one flag) on 2025 weeks 2-18 - same window as the original
# 2026-08-24 measurement above, now against the redesigned mechanism with 2024+2025
# evidence. Result: START-WR - the worst scope in the original rejection - REVERSES,
# now WINNING by -0.069 MAE / +0.024 rank-corr (11 of 17 weeks). START-TE moves the
# other way, a small MAE loss (+0.025) though its rank-corr still edges in alignment's
# favor (+0.002). Whole-pool WR/TE and ALL/QB/RB are a wash (as expected - QB/RB aren't
# a targeted stat group). Kept ON per explicit user request either way ("if it doesn't
# pass don't just remove it, just report the statistics") - this is not a pass/fail
# gate, just the current honest numbers. Re-run this A/B again after any future change
# to alignment_defense_residual_multiplier or its inputs.
DEFAULT_FEATURES = frozenset({
    'role_volume', 'qb1_override',
    'v2_output_contract',
    'v2_as_of_guard',
    'v2_adaptive_volume',
    'v2_td_two_year_prior',
    'v2_defense_prior',
    'v2_continuous_roles',
    'v2_channel_matchups',
    'v2_alignment_contract',
    'v2_pff_alignment_matchup',
    'v2_availability',
    'v2_vacancy',
    'v2_preseason_rb_allocator',
    'v2_pass_capacity',
    'v2_qb_volume_blend',
    'v2_fantasypros_availability',
    'calibration',
    'v2_role_change_by_stat',
})


def resolve_model_features(features=None):
    """Return one explicit feature set for a reproducible model run.

    ``features`` always wins so the evaluation harness (and calibration fit
    script) can test an isolated component combination. Otherwise this
    returns DEFAULT_FEATURES - the single standard model; there is no longer
    a second "model_version" switch (retired 2026-08-26, see
    DEFAULT_FEATURES's own comment).
    """
    if features is not None:
        return frozenset(features)
    return DEFAULT_FEATURES

# The feature set the calibration line was FITTED against - i.e. everything
# that ships except calibration itself. scripts/fit_weekly_calibration.py
# builds the model with exactly this set, so the line describes the
# dispersion of the model it is applied to. If the shipping component set
# changes, the line has to be re-fitted - that is what makes it a
# measurement rather than a magic number (see DEFAULT_FEATURES's own
# comment for the 2026-08-26 re-fit this triggered).
CALIBRATION_INPUT_FEATURES = frozenset(DEFAULT_FEATURES - {'calibration'})


# ---------------------------------------------------------------------------
# CALIBRATION
#
# A projection is supposed to be a conditional expectation: among every
# player projected for 20 points, the average one should score 20. This
# model's was not, and the direction is the one selection always produces -
# the players a noisy projection ranks highest are disproportionately the
# ones its own noise pushed up.
#
# RE-DERIVED 2026-08-23 from a full two-year (2024-2025, weeks 1-18)
# every-modelled-player-week residual backtest, run specifically to check
# this wasn't standing in for a fixable upstream bug before re-enabling it -
# see docs/weekly_projections_methodology.md for the full investigation.
# Correlating the top-of-pool miss against role confidence, defense sample
# size, the matchup multiplier itself, current/prior blend weight, and
# player experience found essentially nothing (|r| < 0.18 everywhere,
# matchup multiplier for the worst busts statistically indistinguishable
# from the rest of the cohort) - ruling out a mis-rated defense or a stale
# role read as the driver. What the busts share instead: realized volume
# far below projected (WR/TE busts averaged 58-59% of their projected
# targets; RB 71%) concentrated in games that came in worse than even the
# PREGAME MARKET LINE expected. Since the model's own script adjustment is
# built from that same pregame line, this is real, un-forecastable-in-advance
# variance, not a parameter to tune - exactly the selection-effect
# over-dispersion calibration exists to absorb, not a symptom of a broken
# input.
#
# (slope, intercept) per position, FITTED ON 2021-2023 - deliberately
# outside the 2024-2025 window every model change here is evaluated on, so
# this is a measurement rather than a curve fitted to its own test. Produced
# by scripts/fit_weekly_calibration.py against CALIBRATION_INPUT_FEATURES;
# re-run it if the shipping component set changes, since the line describes
# the dispersion of the model it is applied to.
#
# RE-FITTED 2026-08-26 when the separate "V1 released baseline" / "V2
# experimental" toggle was retired and DEFAULT_FEATURES grew to become
# exactly the former V2_EXPERIMENTAL_FEATURES set plus calibration (see that
# name's own comment) - CALIBRATION_INPUT_FEATURES moved with it, so the
# previous fit (below what shipped 2026-08-23, against the smaller pre-V2
# DEFAULT_FEATURES) no longer described the model it was being applied to.
# The RAW out-of-sample fit against the NEW, larger feature set (n=11,761,
# 2021-2023 weeks 5-17):
#
#   'QB': (0.467, 8.218),   'RB': (0.852, 2.038),
#   'WR': (0.935, 1.947),   'TE': (0.945, 1.521),
#
# (previous 2026-08-23 fit, for reference: QB (0.522, 7.403), RB (0.830,
# 1.704), WR (0.818, 1.409), TE (0.827, 1.242) - QB's raw slope moved
# further from 1 this round, the others stayed in a similar range.)
#
# APPLIED AT HALF STRENGTH, DELIBERATELY - explicit request to keep the
# board legible as this model's own signal rather than a full statistical
# shrink toward the mean, even where the fitted line alone would be more
# "accurate" by MAE. Each stored constant below blends the raw fit halfway
# toward the identity line (b_slope = 1 + 0.5*(slope-1), b_intercept =
# 0.5*intercept) before the one-sided clip is applied - same formula and
# same 0.5 strength as before, just applied to the refreshed raw fit above.
# NOTE: the 2026-08-23 startable-tier holdout recheck described below (the
# +-0.43 bias bound and the MAE-vs-uncalibrated comparison) was NOT re-run
# against this new fit as part of the 2026-08-26 refactor - only the raw
# fit + half-strength dampening was refreshed to match the new shipping
# feature set. Re-run that startable-tier holdout check (see
# docs/weekly_projections_methodology.md) before trusting those specific
# numbers again; the dampening RATIONALE (why half strength, why one-sided)
# still applies unchanged.
#
# APPLIED ONE-SIDED: `min(projection, line(projection))`. A slope under 1
# shrinks above the line's identity crossover and would INFLATE below it -
# and the bulk of the pool does not need inflating. Measured (pre-damping):
# the two-sided version bought every startable gain below and cost +0.116
# whole-pool MAE, winning 1 week of 26, entirely from lifting several
# hundred near-zero bench rows. Clipping it to the shrink half keeps the
# correction where the defect is.
#
# WHAT IT DOES AND DOESN'T BUY, honestly: it is a per-position MONOTONE
# transform, so it cannot change the order of players within a position and
# does not pretend to. What it changes is the LEVEL - which is what a
# projected point total is read for when it sits next to FantasyPros' and
# the market's numbers on the same row, and what decides whether the
# startable tier is systematically over-promised.
#
# QB RE-INCLUDED, 2026-08-23. Previously dropped the same day for "running
# the wrong direction" on a whole-pool re-check - that check was itself an
# artifact: QB's whole pool is dominated by backup/committee QBs clustered
# near a zero projection, and that bottom-heavy population was dragging the
# whole-pool regression, not the startable tier a calibration line is meant
# to correct. Restricting to the startable-24 pool showed QB with the same
# top-of-pool over-projection shape as every other position (bias flips
# from -2.0 whole-pool to +0.11 startable-only, and the top decile within
# that startable cut still over-projects by +2.86) - re-fit and re-included
# rather than left off on a stale finding. Re-run the isolation check in
# docs/weekly_projections_methodology.md if the shipping feature set moves
# again, on the STARTABLE pool specifically, not the whole pool - that
# distinction is what produced the wrong call the first time.
#
# RE-FITTED 2026-08-27 when 'v2_role_change_by_stat' joined DEFAULT_FEATURES
# (scripts/fit_weekly_calibration.py, same 2021-2023 weeks 5-17 window,
# n=11,761). Raw fit barely moved - QB/WR/TE identical to 3 decimals, RB
# slope 0.852->0.853 - consistent with that component's own measured effect
# being small and RB-only. Recorded per this file's own "re-fit any time
# DEFAULT_FEATURES changes" rule, not because anything shifted meaningfully.
# ---------------------------------------------------------------------------
WEEKLY_CALIBRATION = {
    'QB': (0.734, 4.109),
    'RB': (0.926, 1.018),
    'WR': (0.968, 0.974),
    'TE': (0.972, 0.760),
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


def player_identity_keys(frame, name_col):
    """Stable cross-season identity keys, with names only as a last resort.

    This is deliberately the same identifier hierarchy used by the Ourlads
    resolver and preseason cold-pool dedupe: GSIS/player/PFF IDs first, then
    a reviewed canonical full-name fallback.  Keeping one implementation
    prevents an overlay from finding ``gsis_id:00-...`` while a later role
    lookup only sees a display-name alias for that same player.
    """
    return stable_roster_identity_keys(frame, name_col)


def _identity_by_name(history, name_col):
    """Map a source's exact display name to its preferred stable key."""
    if history is None or history.empty or name_col not in history.columns:
        return pd.Series(dtype=object)
    table = pd.DataFrame({
        '_name_key': canonical_player_key(history[name_col]),
        '_identity_key': player_identity_keys(history, name_col),
    })
    # Prefer a true ID when any row has one; otherwise retain the exact name
    # fallback.  Sorting makes this deterministic across a roster merge.
    # Stable keys are namespaced by their provider (``gsis_id:``,
    # ``pff_id:``, ...), while only the reviewed-name fallback begins
    # ``name:``.  Prefer any stable namespace when duplicate display rows
    # exist rather than relying on the old, no-longer-used ``id:`` prefix.
    table['_is_id'] = ~table['_identity_key'].astype(str).str.startswith('name:')
    table = table.sort_values(['_name_key', '_is_id'], kind='stable').drop_duplicates('_name_key', keep='last')
    return table.set_index('_name_key')['_identity_key']


def attach_player_identity(totals, history, name_col):
    """Attach ``_identity_key`` to an aggregate created by _season_totals."""
    if totals is None or totals.empty:
        return totals
    output = totals.copy()
    by_name = _identity_by_name(history, name_col)
    names = canonical_player_key(output[name_col])
    output['_identity_key'] = names.map(by_name).fillna('name:' + names).to_numpy(dtype=object)
    return output


def identity_indexed_series(values, history, name_col):
    """Reindex a name-indexed stat/share Series by stable source identity."""
    if values is None or values.empty:
        return pd.Series(dtype=float)
    by_name = _identity_by_name(history, name_col)
    names = canonical_player_key(pd.Series(values.index))
    identities = names.map(by_name).fillna('name:' + names)
    out = pd.Series(values.to_numpy(), index=identities)
    return out[~out.index.duplicated(keep='last')]


def defense_recency_weights(weeks, as_of_week, recency_floor=0.0):
    """Defense-history weights with an optional cross-season baseline floor."""
    games_ago = (as_of_week - pd.to_numeric(weeks, errors='coerce')).clip(lower=1)
    floor = float(np.clip(recency_floor, 0.0, 1.0))
    return floor + (1.0 - floor) * RECENCY_DECAY ** (games_ago - 1)


def _clean_team_key(values):
    """Plain, upper-case NFL team keys without categorical/NaN surprises."""
    series = pd.Series(values).astype(object)
    series = series.where(series.notna(), '').astype(str).str.strip().str.upper()
    series = series.replace({'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'})
    return series.where(~series.isin(('', 'NAN', 'NONE', '<NA>')), '')


def _team_game_plays_lookup(plays_df):
    """Normalize a (team, week, plays) table to the ['_offense', '_week',
    '_plays'] merge key _position_team_games' output already uses, so
    _team_game_quality_profile can divide a single game's raw stat total by
    that game's own play count. Team abbreviations are run through the same
    _clean_team_key normalization as _historical_game_team, so an old code
    (e.g. OAK) in a play-count source still joins against a game row already
    relabeled under its current franchise (LV).
    """
    if plays_df is None or plays_df.empty:
        return pd.DataFrame(columns=['_offense', '_week', '_plays'])
    out = plays_df.rename(columns={'team': '_offense', 'week': '_week', 'plays': '_plays'})
    out = out[['_offense', '_week', '_plays']].copy()
    out['_offense'] = _clean_team_key(out['_offense'])
    out['_week'] = pd.to_numeric(out['_week'], errors='coerce')
    out['_plays'] = pd.to_numeric(out['_plays'], errors='coerce')
    out = out.dropna(subset=['_week'])
    return out.groupby(['_offense', '_week'], observed=True, as_index=False)['_plays'].sum()


def _historical_game_opponent(frame):
    """The defense actually faced, preferring immutable raw weekly context."""
    source = (frame['game_opponent'] if 'game_opponent' in frame.columns
              else frame['opponent_team'] if 'opponent_team' in frame.columns
              else pd.Series('', index=frame.index, dtype=object))
    return _clean_team_key(source)


def _historical_game_team(frame, team_col):
    """Return the offense that actually played each historical game.

    A roster merge intentionally keeps a player's *latest* team for player
    cards and current projections.  It must not rewrite a prior CLE game as
    a CIN game after a mid-season trade, though.  ``game_team`` is preserved
    from the raw weekly feed by ``load_year_data``.  For cached/older frames
    that predate that column, a game id plus the opponent reconstructs the
    offense exactly whenever the standard nflverse ``YYYY_WW_AWAY_HOME`` id
    is available.  Only then do we fall back to the merged team column.
    """
    has_explicit_game_team = 'game_team' in frame.columns
    fallback = (_clean_team_key(frame['game_team']) if has_explicit_game_team
                else _clean_team_key(frame[team_col]) if team_col in frame.columns
                else pd.Series('', index=frame.index, dtype=object))
    # The raw weekly feed's team is authoritative. Game-id reconstruction is
    # deliberately a fallback for pre-contract cached frames only: old game
    # ids retain historical aliases (for example OAK) that should not replace
    # the raw feed's current NFL abbreviation (LV).
    if 'game_id' not in frame.columns or not (
            'game_opponent' in frame.columns or 'opponent_team' in frame.columns):
        return fallback
    parts = frame['game_id'].astype(str).str.rsplit('_', n=2, expand=True)
    if parts.shape[1] < 3:
        return fallback
    side_a = _clean_team_key(parts.iloc[:, -2]).set_axis(frame.index)
    side_b = _clean_team_key(parts.iloc[:, -1]).set_axis(frame.index)
    defense = _historical_game_opponent(frame)
    inferred = pd.Series(
        np.where(defense.eq(side_a), side_b,
                 np.where(defense.eq(side_b), side_a, '')),
        index=frame.index,
    )
    # Prefer a real raw game team row-by-row. Roster-only placeholder rows
    # can carry the column but no value, where game-id inference remains a
    # useful fallback; old cached frames with no explicit column prefer the
    # inferred game team over their current-roster fallback.
    if has_explicit_game_team:
        return fallback.where(fallback.ne(''), inferred)
    return inferred.where(inferred.ne(''), fallback)


def _offense_defense_game_universe(history, team_col):
    """Every real offense-defense game available in a weekly history frame."""
    required = {'opponent_team', 'week'}
    if history is None or history.empty or not required.issubset(history.columns):
        return pd.DataFrame(columns=['_offense', '_defense', '_week'])
    cols = ['opponent_team', 'week']
    for col in (team_col, 'game_team', 'game_opponent', 'game_id'):
        if col in history.columns:
            cols.append(col)
    frame = history.loc[:, list(dict.fromkeys(cols))].copy()
    frame['_offense'] = _historical_game_team(frame, team_col)
    frame['_defense'] = _historical_game_opponent(frame).to_numpy()
    frame['_week'] = pd.to_numeric(frame['week'], errors='coerce')
    frame = frame[(frame['_offense'] != '') & (frame['_defense'] != '') & frame['_week'].notna()]
    return frame[['_offense', '_defense', '_week']].drop_duplicates().reset_index(drop=True)


def _position_team_games(hist_pos, team_col, stats, name_col=None, roles=None,
                         game_universe=None):
    """One offense-position (optionally role) total per real defense game.

    This is the defensive-profile unit used everywhere in Weekly Rankings.
    Player rows are bookkeeping, not independent defensive observations: one
    backup replacement, injury fill-in, or statless relief appearance must
    not receive the same voting weight as an entire offense's game.
    """
    required = {'opponent_team', 'week'}
    if hist_pos.empty or not required.issubset(hist_pos.columns):
        return pd.DataFrame(), []
    stats = [stat for stat in stats if stat in hist_pos.columns]
    if not stats:
        return pd.DataFrame(), []
    cols = ['opponent_team', 'week'] + stats
    if team_col in hist_pos.columns:
        cols.append(team_col)
    if 'game_team' in hist_pos.columns:
        cols.append('game_team')
    if 'game_opponent' in hist_pos.columns:
        cols.append('game_opponent')
    if 'game_id' in hist_pos.columns:
        cols.append('game_id')
    if roles is not None and name_col in hist_pos.columns:
        cols.append(name_col)
    frame = hist_pos.loc[:, list(dict.fromkeys(cols))].copy()
    frame['_offense'] = _historical_game_team(frame, team_col)
    frame['_defense'] = _historical_game_opponent(frame).to_numpy()
    frame['_week'] = pd.to_numeric(frame['week'], errors='coerce')
    frame = frame[(frame['_offense'] != '') & (frame['_defense'] != '') & frame['_week'].notna()].copy()
    if frame.empty:
        return pd.DataFrame(), []
    for stat in stats:
        frame[stat] = pd.to_numeric(frame[stat], errors='coerce').fillna(0.0)

    group_keys = ['_offense', '_defense', '_week']
    if roles is not None:
        if name_col not in frame.columns:
            return pd.DataFrame(), []
        direct = frame[name_col].map(roles)
        if direct.isna().any():
            keyed_roles = {
                clean_name_exact(pd.Series([player])).iloc[0]: role
                for player, role in roles.items()
            }
            direct = direct.fillna(clean_name_exact(frame[name_col]).map(keyed_roles))
        frame['_role'] = direct.fillna('').astype(str)
        frame = frame[frame['_role'].ne('')].copy()
        if frame.empty:
            return pd.DataFrame(), []
        group_keys.append('_role')

    game = frame.groupby(group_keys, as_index=False, observed=True)[stats].sum()
    # A player weekly file is stat-triggered: an offense can play a real
    # game while a position records no row at all. For broad position
    # profiles, preserve that 0-output team game rather than dropping it and
    # quietly treating "no TE production" as no defensive evidence. Role
    # profiles deliberately do not fill role-absent games—the absence may be
    # a personnel decision/injury, not a defense result.
    if roles is None and game_universe is not None:
        universe = _offense_defense_game_universe(game_universe, team_col)
        if not universe.empty:
            game = universe.merge(game, on=group_keys, how='left')
            for stat in stats:
                game[stat] = pd.to_numeric(game[stat], errors='coerce').fillna(0.0)
    # NFL QB box-score rushing includes kneels. A team-QB total can therefore
    # be negative even though a player's forward rushing expectation is
    # floored at zero downstream. Do not let a negative historical baseline
    # create a sign-flipped defense ratio; it is absent usable evidence for
    # this purpose, not proof of a defense that gives up "negative rushing".
    if 'rushing_yards' in game.columns:
        game['rushing_yards'] = game['rushing_yards'].clip(lower=0.0)
    return game, group_keys


def _build_player_stat_game_log(hist_annotated_pos, name_col, stats, matchup_matrix,
                                schedule_df=None, team_col='team'):
    """Per-game raw + defense-adjusted values for one position's game
    history, keyed by player name (as spelled in ``name_col``).

    Factored out so the Deep Dive can build this identically for a CURRENT
    season and a PRIOR season, from either the in-season or cold-start
    branch - one definition of "what does a per-game log row look like"
    instead of copy-pasting it per season/branch combination. The defense
    adjustment is raw / matchup multiplier, WITHOUT _weighted_player_rates'
    recency/rematch weighting - that belongs in the averaged rate, not in a
    per-game log meant to show each game as-is. Sourced from an *_annotated
    frame (not the eligibility-filtered player_hist/player_prior), so
    partial-game-screen-excluded games are still present here, correctly
    flagged via _player_history_eligible/_reason.

    ``schedule_df`` (the same year's real schedule, home/away score) is
    optional and joined on ``game_id`` only for display context - final
    score and win/loss - never as a projection input. A missing/unjoinable
    game (a schedule this app doesn't have, or a synthetic test fixture)
    just leaves those columns blank rather than failing the whole log.
    """
    if hist_annotated_pos is None or hist_annotated_pos.empty:
        return {}
    log = hist_annotated_pos.copy()
    opponent = (log['opponent_team'].astype(str) if 'opponent_team' in log.columns
                else pd.Series('', index=log.index))
    for stat in stats:
        if stat not in log.columns:
            continue
        raw = pd.to_numeric(log[stat], errors='coerce').fillna(0.0)
        mult = pd.Series(1.0, index=log.index)
        if matchup_matrix is not None and not matchup_matrix.empty and stat in matchup_matrix.columns:
            mult = opponent.map(matchup_matrix[stat]).fillna(1.0).clip(*HISTORY_MATCHUP_CLIP)
        log[f'_defadj_{stat}'] = raw / mult
    log['_team_score'] = np.nan
    log['_opp_score'] = np.nan
    log['_result'] = ''
    if (schedule_df is not None and not schedule_df.empty and 'game_id' in log.columns
            and {'home_team', 'away_team', 'home_score', 'away_score'}.issubset(schedule_df.columns)
            and team_col in log.columns):
        sched = schedule_df[['game_id', 'home_team', 'away_team', 'home_score', 'away_score']].drop_duplicates('game_id')
        merged = log[['game_id']].merge(sched, on='game_id', how='left')
        is_home = log[team_col].astype(str).to_numpy() == merged['home_team'].astype(str).to_numpy()
        team_score = np.where(is_home, merged['home_score'], merged['away_score'])
        opp_score = np.where(is_home, merged['away_score'], merged['home_score'])
        log['_team_score'] = pd.to_numeric(pd.Series(team_score, index=log.index), errors='coerce')
        log['_opp_score'] = pd.to_numeric(pd.Series(opp_score, index=log.index), errors='coerce')
        log['_result'] = np.select(
            [log['_team_score'] > log['_opp_score'], log['_team_score'] < log['_opp_score'],
             log['_team_score'].notna() & log['_opp_score'].notna()],
            ['W', 'L', 'T'], default='')
    return {name: g for name, g in log.groupby(name_col)}


def _eligible_fantasy_points(*sources) -> list:
    """Real per-game fantasy points from one or more game-log sources -
    DataFrames (with '_player_history_eligible'/'fantasy_points' columns,
    same shape _build_player_stat_game_log produces) or lists of record
    dicts (the shape 'game_log_by_season' stores) - filtered to eligible
    games only. Feeds data.weekly_distribution.player_width_scale's real-
    variance signal; a player with too little combined history here just
    falls through to that function's own role_confidence-based fallback."""
    out = []
    for source in sources:
        if isinstance(source, pd.DataFrame):
            if source.empty or 'fantasy_points' not in source.columns:
                continue
            eligible = source
            if '_player_history_eligible' in source.columns:
                eligible = source[source['_player_history_eligible'].astype(bool)]
            out.extend(pd.to_numeric(eligible['fantasy_points'], errors='coerce').dropna().tolist())
        elif isinstance(source, list):
            for row in source:
                if not isinstance(row, dict) or not row.get('_player_history_eligible', True):
                    continue
                value = row.get('fantasy_points')
                if value is not None:
                    try:
                        out.append(float(value))
                    except (TypeError, ValueError):
                        pass
    return out


def _defense_adjusted_prior_average(player_game_log_prior, stats):
    """Per-player, per-stat mean of the prior season's own per-game
    ``_defadj_{stat}`` values (ELIGIBLE games only) - literally an average
    of the SAME numbers the Deep Dive's per-game "Defense-adj" column
    already shows for that season, not a new/separate adjustment. Added
    2026-08-25 for the decomposition table's "Season average (adj)" column
    ("this is based on the defenses a player has played" - the user's own
    framing), distinct from `raw_prior_rate` (the plain, unadjusted
    average) and from `blended_rate` (which is about the UPCOMING
    opponent, not games already played). Display-only - never an input to
    `_blended_rate` or anything downstream of it. Two-years-back is
    deliberately excluded: no quality-adjusted matchup matrix is built for
    that season (see `_build_player_stat_game_log`'s own comment), so its
    "Defense-adj" already just repeats the raw value and would only dilute
    this number toward the unadjusted one.
    """
    out = {stat: np.full(len(player_game_log_prior), np.nan) for stat in stats}
    for i, log in enumerate(player_game_log_prior):
        if not isinstance(log, pd.DataFrame) or log.empty:
            continue
        eligible = (log[log['_player_history_eligible'].astype(bool)]
                   if '_player_history_eligible' in log.columns else log)
        if eligible.empty:
            continue
        for stat in stats:
            col = f'_defadj_{stat}'
            if col in eligible.columns:
                val = pd.to_numeric(eligible[col], errors='coerce').mean()
                if pd.notna(val):
                    out[stat][i] = float(val)
    return out


def _build_defense_weekly_log(pos_rows, team_col, stats, game_universe, as_of_week):
    """Per-week opponent-allowed rows for one position, keyed by defense team.

    The raw offense-defense-week totals _position_team_games builds and
    _team_game_quality_profile immediately collapses into a single matchup
    multiplier - kept here too (one more cheap call with the identical
    arguments) so the Deep Dive can show the arithmetic behind that
    multiplier instead of just the final number. Shared by both season
    choices in both branches, same reasoning as _build_player_stat_game_log.
    """
    game, _keys = _position_team_games(pos_rows, team_col, stats, game_universe=game_universe)
    if game.empty:
        return {}
    game = game.copy()
    baseline = game.groupby('_offense', observed=True)[stats].transform('mean')
    for stat in stats:
        game[f'_baseline_{stat}'] = baseline[stat]
    game['_weight'] = defense_recency_weights(game['_week'], as_of_week)
    return {team: g for team, g in game.groupby('_defense')}


def _team_game_quality_profile(game, stats, as_of_week, recency_floor=0.0,
                               partition_keys=(), plays=None):
    """Pooled observed/expected team-game defense factor.

    A defense's profile is a *ratio of weighted totals*, not an equal-weight
    average of per-game ratios. A 35-target receiving game consequently has
    more evidence than a 4-target game, while a large number of player rows
    in either game has no effect. Four league-average neutral games provide
    a small, explicit empirical-Bayes prior for every stat; that is most
    useful for TDs/INTs, where zeroes are common, and modest after a full
    season of evidence.

    ``plays`` is an optional ['_offense', '_week', '_plays'] table (see
    as_of_team_weekly_plays/load_team_weekly_plays). When supplied, every
    game's raw stat total is divided by THAT game's own play count before
    the observed/expected ratio is built, so the ratio reflects per-play
    defensive quality rather than volume. This exists because the
    standalone pace multiplier applied later in build_weekly_projections
    (opponent_defensive_pace / league_pace) already re-applies volume, from
    the upcoming opponent's SEASON pace level; without this normalization a
    defense that merely faces a lot of plays reads as a bad matchup twice -
    once here, from raw per-game totals correlating with that game's play
    count, and again through pace_mult. A game whose play count is unknown
    is dropped from the ratio (no evidence) rather than left at raw scale,
    which would silently mix per-play and per-game units in the same sum.
    """
    if game.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    stats = [stat for stat in stats if stat in game.columns]
    if not stats:
        return pd.DataFrame(), pd.Series(dtype=float)
    partition_keys = list(partition_keys)
    if partition_keys:
        profiles, evidence_blocks = [], []
        for key, block in game.groupby(partition_keys, observed=True):
            profile, evidence = _team_game_quality_profile(
                block, stats, as_of_week, recency_floor=recency_floor, plays=plays)
            if profile.empty:
                continue
            labels = key if isinstance(key, tuple) else (key,)
            labeled = profile.copy()
            for column, label in zip(partition_keys, labels):
                labeled[column] = label
            profiles.append(labeled.set_index(partition_keys, append=True))
            if not evidence.empty:
                frame = evidence.rename('_evidence').reset_index()
                for column, label in zip(partition_keys, labels):
                    frame[column] = label
                evidence_blocks.append(frame.set_index(['_defense'] + partition_keys)['_evidence'])
        if not profiles:
            return pd.DataFrame(), pd.Series(dtype=float)
        profile_out = pd.concat(profiles).sort_index()
        evidence_out = pd.concat(evidence_blocks).sort_index() if evidence_blocks else pd.Series(dtype=float)
        return profile_out, evidence_out

    if plays is not None and not plays.empty:
        game = game.merge(plays, on=['_offense', '_week'], how='left')
        known_plays = pd.to_numeric(game['_plays'], errors='coerce')
        has_plays = known_plays > 0
        game = game.loc[has_plays].copy()
        if game.empty:
            return pd.DataFrame(), pd.Series(dtype=float)
        game[stats] = game[stats].div(known_plays[has_plays].to_numpy(), axis=0)

    baseline_keys = ['_offense'] + partition_keys
    baseline = game.groupby(baseline_keys, observed=True)[stats].transform('mean')
    weights = defense_recency_weights(game['_week'], as_of_week, recency_floor)
    observed = game[stats].mul(weights, axis=0)
    expected = baseline.mul(weights, axis=0)
    observed['_defense'] = game['_defense'].to_numpy()
    expected['_defense'] = game['_defense'].to_numpy()
    observed_sum = observed.groupby('_defense', observed=True)[stats].sum()
    expected_sum = expected.groupby('_defense', observed=True)[stats].sum()
    # A neutral prior is expressed in the same units as each stat: four
    # league-average offense-position games at the profile's average recency
    # weight. Adding it to both numerator and denominator pulls a sparse
    # observed/expected ratio toward 1.0 without inventing a direction.
    league_expected = baseline.mean().clip(lower=0.0)
    prior = league_expected * float(DEFENSE_PRIOR_GAMES) * float(weights.mean())
    result = observed_sum.add(prior, axis='columns').div(
        expected_sum.add(prior, axis='columns').replace(0, np.nan))
    result = result.dropna(axis=1, how='all')
    if result.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    # Re-center so 1.0 stays the league-average forward multiplier.
    result = result.div(result.mean().replace(0, np.nan))

    evidence_frame = game.loc[:, ['_defense']].copy()
    evidence_frame['_weight'] = weights.to_numpy(dtype=float)
    evidence = evidence_frame.groupby('_defense', observed=True)['_weight'].sum()
    return result, evidence


def build_team_game_quality_adjusted_matchup(hist_pos, team_col, stats, as_of_week,
                                             recency_floor=0.0, game_universe=None,
                                             plays=None):
    """Robust defense profile for one projected position channel.

    For every statistic, first sum *all players at that position* into one
    offense-versus-defense game. Compare that total with the offense's own
    season-average total for the same position, recency-weight the resulting
    one-game residual, then re-center the league at 1.0. QB rushing, RB
    rushing, RB receiving, WR receiving, and TE receiving are all built in
    separate calls, so their defensive channels never overlap.

    This supersedes the former player-row / player-season-average estimator
    in the projection path. A spot starter can change one position team's
    total for one game, but can no longer supply a huge ratio from a tiny
    personal baseline or count as a second independent defensive game.

    ``plays`` is optional and forwarded to _team_game_quality_profile - see
    its docstring for why the ratio is normalized by each game's own play
    count rather than left as a raw-total volume that pace_mult would then
    double-apply.
    """
    game, group_keys = _position_team_games(
        hist_pos, team_col, stats, game_universe=game_universe)
    if game.empty:
        return pd.DataFrame()
    result, _evidence = _team_game_quality_profile(
        game, stats, as_of_week, recency_floor=recency_floor,
        partition_keys=group_keys[3:], plays=plays,
    )
    return result


def build_quality_adjusted_matchup(hist_pos, name_col, stats, as_of_week,
                                   recency_floor=0.0):
    """Legacy individual-player comparator retained for regression tests.

    Weekly Rankings no longer calls this function.  Its player-row design is
    useful as a pinned counterexample for the old spot-start sensitivity,
    while all production defense profiles use
    ``build_team_game_quality_adjusted_matchup`` instead.
    """
    if hist_pos.empty or 'opponent_team' not in hist_pos.columns:
        return pd.DataFrame()
    stats = [s for s in stats if s in hist_pos.columns]
    if not stats:
        return pd.DataFrame()
    df = hist_pos.copy()
    w = defense_recency_weights(df['week'], as_of_week, recency_floor)
    baseline = df.groupby(name_col, observed=True)[stats].transform('mean').replace(0, np.nan)
    ratio = df[stats].div(baseline)
    valid = ratio.notna()
    num = ratio.fillna(0.0).mul(w, axis=0)
    den = valid.astype(float).mul(w, axis=0)
    num['_opponent'] = _clean_team_key(df['opponent_team']).to_numpy()
    den['_opponent'] = num['_opponent'].to_numpy()
    result = num.groupby('_opponent', observed=True)[stats].sum().div(
        den.groupby('_opponent', observed=True)[stats].sum().replace(0, np.nan))
    return result.div(result.mean().replace(0, np.nan))


QB_PASSING_MATCHUP_STATS = frozenset({
    'passing_attempts', 'passing_completions', 'passing_yards', 'passing_tds',
    'passing_interceptions',
})


def build_qb_passing_quality_adjusted_matchup(hist_qbs, team_col, stats, as_of_week,
                                               recency_floor=0.0, plays=None):
    """Backward-compatible QB-passing wrapper around the common estimator."""
    passing_stats = [stat for stat in stats if stat in QB_PASSING_MATCHUP_STATS]
    return build_team_game_quality_adjusted_matchup(
        hist_qbs, team_col, passing_stats, as_of_week, recency_floor=recency_floor, plays=plays)


def build_qb_quality_adjusted_matchup(hist_qbs, name_col, team_col, stats, as_of_week,
                                      recency_floor=0.0, plays=None):
    """Backward-compatible QB wrapper; passing and rushing stay separate
    only because callers build QB as its own position channel, not because
    rushing falls back to a player-row defensive estimator."""
    del name_col  # kept in the public signature for existing callers
    return build_team_game_quality_adjusted_matchup(
        hist_qbs, team_col, stats, as_of_week, recency_floor=recency_floor, plays=plays)


# Four defense games are enough for the current season to move a profile
# materially, but not enough to erase last year's evidence.  This is
# intentionally lighter than player-stat shrinkage: coordinator, personnel,
# and scheme changes make a defense less stable across an offseason.
DEFENSE_PRIOR_GAMES = 4.0
# Sweep hook, built 2026-08-27 - never set outside a sweep script. This
# constant has driven every position's defense matchup number since it was
# set, never itself backtested against alternate values. None (default)
# means "use DEFENSE_PRIOR_GAMES as always"; a float means "use this instead,
# but ONLY when 'v2_defense_prior_games_override' is in feats" - gated so a
# plain DEFAULT_FEATURES build is never affected by this constant being
# nonzero, only a deliberate sweep variant is. See
# scripts/sweep_defense_prior_games.py.
DEFENSE_PRIOR_GAMES_OVERRIDE = None


def _defense_game_evidence(hist_pos, game_universe=None, team_col=None):
    """Effective defense-game count indexed by opponent team.

    A player-row count would say a defense that happened to face many WRs
    supplied more independent evidence than one that faced fewer.  The game
    is the independent defensive sample, so each opponent/week pair counts
    once regardless of how many player rows it generated.
    """
    if game_universe is not None and team_col is not None:
        games = _offense_defense_game_universe(game_universe, team_col)
        if not games.empty:
            return games.groupby('_defense', observed=True)['_week'].nunique().astype(float)
    if hist_pos.empty or not {'opponent_team', 'week'}.issubset(hist_pos.columns):
        return pd.Series(dtype=float)
    rows = hist_pos[['opponent_team', 'week']].dropna().copy()
    rows['opponent_team'] = rows['opponent_team'].astype(str)
    return rows.drop_duplicates().groupby('opponent_team', observed=True)['week'].nunique().astype(float)


def blend_defense_prior(current, prior, evidence, prior_games=DEFENSE_PRIOR_GAMES):
    """Blend a time-valid current defense profile into its prior-year one.

    ``current`` and ``prior`` are already opponent-quality-adjusted ratings
    centered around 1.0.  Missing current evidence falls back to prior; a
    completely unknown defense is neutral.  The returned frame is therefore
    safe to use as a direct matchup multiplier and carries no assumption that
    a season-total source existed before the historical target week.
    """
    current = current if current is not None else pd.DataFrame()
    prior = prior if prior is not None else pd.DataFrame()
    columns = list(dict.fromkeys(list(current.columns) + list(prior.columns)))
    index = current.index.union(prior.index) if columns else pd.Index([])
    if not len(index) or not columns:
        return pd.DataFrame()
    cur = current.reindex(index=index, columns=columns)
    old = prior.reindex(index=index, columns=columns)
    n = pd.Series(evidence, dtype=float).reindex(index).fillna(0.0).clip(lower=0.0)
    alpha = n / (n + float(prior_games))
    out = cur.copy()
    for col in columns:
        c = pd.to_numeric(cur[col], errors='coerce')
        p = pd.to_numeric(old[col], errors='coerce').fillna(1.0)
        # A team with no current sample gets its prior; a current profile
        # with a missing stat likewise degrades safely rather than reading
        # missing as a favorable matchup.
        out[col] = np.where(c.notna(), alpha * c.fillna(1.0) + (1.0 - alpha) * p, p)
    return out


def _as_of_team_game_plays(stats_df, team_col, as_of_week):
    """Cutoff-safe per-(team, week) play count, the shared raw ingredient
    behind both as_of_team_pace's season-level proxy and
    as_of_team_weekly_plays' per-game table - one measurement of "plays
    that game", not two that could quietly drift apart.
    """
    hist = _played_weeks_before(stats_df, as_of_week)
    required = {team_col, 'opponent_team', 'week'}
    if hist.empty or not required.issubset(hist.columns):
        return pd.DataFrame(columns=[team_col, 'opponent_team', 'week', '_plays'])
    play_cols = [c for c in ('passing_attempts', 'rushing_attempts') if c in hist.columns]
    if not play_cols:
        return pd.DataFrame(columns=[team_col, 'opponent_team', 'week', '_plays'])
    frame = hist[[team_col, 'opponent_team', 'week'] + play_cols].copy()
    frame['_plays'] = frame[play_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).sum(axis=1)
    return frame.groupby([team_col, 'opponent_team', 'week'], observed=True)['_plays'].sum().reset_index()


def as_of_team_pace(stats_df, team_col, as_of_week):
    """A cutoff-safe pace proxy from weekly player box scores.

    ``load_team_pace`` is a whole-season source, which is correct for a live
    upcoming slate but leaks later games in a historical test.  The available
    weekly player data gives a useful, reproducible substitute: team pass
    attempts plus team rushing attempts per completed game.  It omits sacks,
    so it is deliberately called a proxy and used only in V2 historical
    runs; it is still much more honest than reading the future.
    """
    games = _as_of_team_game_plays(stats_df, team_col, as_of_week)
    if games.empty:
        return pd.DataFrame(columns=['off_pace', 'def_pace'])
    off = games.groupby(team_col, observed=True)['_plays'].mean().rename('off_pace')
    defense = games.groupby('opponent_team', observed=True)['_plays'].mean().rename('def_pace')
    defense.index.name = off.index.name
    return pd.concat([off, defense], axis=1)


def as_of_team_weekly_plays(stats_df, team_col, as_of_week):
    """Cutoff-safe per-(team, week) play count - the per-game counterpart to
    as_of_team_pace's season-average proxy, used only in V2 historical runs
    for the same leakage reason.

    Feeds the defense-matchup ratio's own pace normalization
    (_team_game_quality_profile's ``plays`` argument): a single game's raw
    stat total is divided by that game's own play count before it is
    compared to a defense's expected total, so the standalone pace
    multiplier (built from as_of_team_pace/load_team_pace's SEASON-level
    estimate) is the only place game volume gets applied. Without this, a
    defense's ratio would already carry its own pace, and pace_mult would
    apply it a second time.
    """
    games = _as_of_team_game_plays(stats_df, team_col, as_of_week)
    if games.empty:
        return pd.DataFrame(columns=['team', 'week', 'plays'])
    return games.rename(columns={team_col: 'team', '_plays': 'plays'})[['team', 'week', 'plays']]


# ---------------------------------------------------------------------------
# ROLE-CONDITIONED MATCHUPS
#
# The broad profile already levels each offense's overall positional output.
# The further ask here is "a defense that is soft to a possession
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


def build_continuous_role_profiles(history, name_col, team_col, pos):
    """Return auditable mixed-role inputs without assigning a hard label.

    The weekly box-score feed supplies target/carry share, ADOT, and snaps
    as-of the target week.  It does *not* supply time-valid slot/inline
    alignment; those fields are intentionally reported as missing rather
    than backfilled from a PFF season-total export.  The caller can use the
    profile today for target-earner and RB/QB channels, and the explanation
    makes the alignment gap visible instead of silently pretending it was
    solved.
    """
    columns = ['adot', 'target_share', 'carry_share', 'receiving_back_share',
               'snap_share', 'target_earner_rank', 'target_earner_score',
               'alignment_available', 'evidence_games']
    if history.empty or name_col not in history.columns:
        return pd.DataFrame(columns=columns)
    source = history.copy()
    if 'position' in source.columns:
        frame = source[source['position'].astype(str).str.upper() == str(pos).upper()].copy()
    else:
        frame = source.copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    for col in ('targets', 'rushing_attempts', 'receiving_air_yards', 'weekly_snap_pct'):
        if col not in frame.columns:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors='coerce').fillna(0.0)
        if col not in source.columns:
            source[col] = 0.0
        source[col] = pd.to_numeric(source[col], errors='coerce').fillna(0.0)
    grouped = frame.groupby(name_col, observed=True)
    out = pd.DataFrame(index=grouped.size().index)
    targets = grouped['targets'].sum()
    carries = grouped['rushing_attempts'].sum()
    air = grouped['receiving_air_yards'].sum()
    out['adot'] = air / targets.replace(0, np.nan)
    out['snap_share'] = (grouped['weekly_snap_pct'].mean() / 100.0).clip(0.0, 1.0)
    out['evidence_games'] = grouped['week'].nunique() if 'week' in frame.columns else 0
    if team_col in frame.columns:
        group_keys = [team_col] + (['week'] if 'week' in source.columns else [])
        team_target_total = source.groupby(group_keys, observed=True)['targets'].sum()
        team_carry_total = source.groupby(group_keys, observed=True)['rushing_attempts'].sum()
        frame['_target_total'] = pd.MultiIndex.from_frame(frame[group_keys]).map(team_target_total)
        frame['_carry_total'] = pd.MultiIndex.from_frame(frame[group_keys]).map(team_carry_total)
        frame['_target_share_game'] = np.divide(frame['targets'], frame['_target_total'],
                                                 out=np.zeros(len(frame)), where=frame['_target_total'] > 0)
        frame['_carry_share_game'] = np.divide(frame['rushing_attempts'], frame['_carry_total'],
                                                out=np.zeros(len(frame)), where=frame['_carry_total'] > 0)
        out['target_share'] = frame.groupby(name_col, observed=True)['_target_share_game'].mean()
        out['carry_share'] = frame.groupby(name_col, observed=True)['_carry_share_game'].mean()
        team_for_player = frame.sort_values('week').groupby(name_col, observed=True)[team_col].last()
        out['_team'] = team_for_player
        # Continuous score: the highest target earner on a team is 1.0;
        # low-volume teammates approach 0.0.  This preserves WR1/WR2 context
        # without making a brittle binary label.
        out['target_earner_rank'] = out.groupby('_team', observed=True)['target_share'].rank(
            ascending=False, method='min')
        team_size = out.groupby('_team', observed=True)['target_share'].transform('size')
        out['target_earner_score'] = np.where(
            team_size > 1, 1.0 - (out['target_earner_rank'] - 1.0) / (team_size - 1.0), 1.0)
        out = out.drop(columns=['_team'])
    else:
        out['target_share'] = np.nan
        out['carry_share'] = np.nan
        out['target_earner_rank'] = np.nan
        out['target_earner_score'] = np.nan
    out['receiving_back_share'] = targets / (targets + carries).replace(0, np.nan)
    out['alignment_available'] = False
    return out.reindex(columns=columns)


def build_continuous_role_weights(hist_pos, name_col, pos):
    """Soft weights over the legacy role tables, for V2's no-cliff matcher."""
    labels = ROLE_LABELS.get(pos)
    metric, evidence = _role_metric(hist_pos, name_col, pos)
    if labels is None or metric.empty:
        return pd.DataFrame()
    result = pd.DataFrame(0.0, index=metric.index, columns=labels)
    qualified = metric[(evidence >= ROLE_MIN_EVENTS.get(pos, 15)) & metric.notna()]
    if len(qualified) < 9:
        result[labels[1]] = 1.0
        return result
    lo, hi = qualified.quantile([1 / 3, 2 / 3]).tolist()
    midpoint = (float(lo) + float(hi)) / 2.0
    for player, value in metric.items():
        if player not in qualified.index or not np.isfinite(value):
            result.at[player, labels[1]] = 1.0
        elif value <= lo:
            result.at[player, labels[0]] = 1.0
        elif value < midpoint:
            weight_mid = (value - lo) / max(midpoint - lo, 0.01)
            result.at[player, labels[0]] = 1.0 - weight_mid
            result.at[player, labels[1]] = weight_mid
        elif value < hi:
            weight_deep = (value - midpoint) / max(hi - midpoint, 0.01)
            result.at[player, labels[1]] = 1.0 - weight_deep
            result.at[player, labels[2]] = weight_deep
        else:
            result.at[player, labels[2]] = 1.0
    return result


def build_role_matchup(hist_pos, name_col, team_col, stats, as_of_week, roles,
                       recency_floor=0.0, plays=None):
    """Role-conditioned team-game defense profiles.

    A role is still assigned from a player's own measured usage, but once
    assigned, all same-role players on an offense are summed into *one*
    offense-defense-week total before comparison to that offense-role's own
    baseline. Thus a 12-snap fill-in does not acquire a giant defense signal
    merely because his personal season average is tiny. The returned role
    tables retain the existing public contract, including evidence-weighted
    shrinkage in the forward multiplier.

    ``plays`` is optional and forwarded to _team_game_quality_profile - same
    per-game pace normalization as build_team_game_quality_adjusted_matchup,
    kept consistent here so a role-conditioned rating and the overall rating
    it blends toward are on the same (pace-free) scale.
    """
    if hist_pos.empty or not roles:
        return {}, {}
    game, group_keys = _position_team_games(
        hist_pos, team_col, stats, name_col=name_col, roles=roles)
    if game.empty or '_role' not in game.columns:
        return {}, {}
    result, evidence = _team_game_quality_profile(
        game, stats, as_of_week, recency_floor=recency_floor,
        partition_keys=group_keys[3:], plays=plays,
    )
    if result.empty:
        return {}, {}
    out, sizes = {}, {}
    for role_label, block in result.groupby(level='_role', observed=True):
        out[str(role_label)] = block.droplevel('_role')
    for (opp, role_label), size in evidence.items():
        sizes[(str(opp), str(role_label))] = float(size)
    return out, sizes


def _overall_matchup_multiplier(overall, opponents, stat):
    """Direct position-channel defense multiplier without a role overlay."""
    base = np.ones(len(opponents))
    if overall is not None and not overall.empty and stat in overall.columns:
        base = pd.Series(opponents).map(overall[stat]).fillna(1.0).to_numpy(dtype=float)
    return np.clip(base, *MATCHUP_CLIP)


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
    base = _overall_matchup_multiplier(overall, opponents, stat)
    if not role_tables:
        return base

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


def _continuous_role_adjusted_multiplier(overall, role_tables, role_sizes, opponents, role_weights, stat):
    """V2 counterpart of ``_role_adjusted_multiplier`` without role cliffs.

    The underlying defense tables remain evidence-shrunk (a defense has only
    a few games against a given style), but a player's mix is a weighted
    blend across nearby profiles.  A QB or WR near a boundary therefore does
    not jump from one defense rating to another because one target changed
    his tercile label.
    """
    base = _overall_matchup_multiplier(overall, opponents, stat)
    if role_weights is None or role_weights.empty or not role_tables:
        return base
    aligned = role_weights.reindex(columns=list(role_tables), fill_value=0.0).fillna(0.0)
    result = base.copy()
    used = np.zeros(len(opponents))
    for label in aligned.columns:
        table = role_tables.get(label)
        if table is None or stat not in table.columns:
            continue
        component = pd.Series(opponents).map(table[stat]).to_numpy(dtype=float)
        evidence = np.asarray([role_sizes.get((str(opp), label), 0.0) for opp in opponents], dtype=float)
        evidence_weight = evidence / (evidence + ROLE_MATCHUP_K)
        blended = np.where(np.isfinite(component),
                           evidence_weight * component + (1.0 - evidence_weight) * base,
                           base)
        weights = aligned[label].to_numpy(dtype=float)
        result += weights * (blended - base)
        used += weights
    # A player with no measurable profile is neutral relative to the overall
    # defense.  Do not normalize a partial/missing alignment into confidence.
    return np.clip(np.where(used > 0, result, base), *MATCHUP_CLIP)


def _weighted_player_rates(hist_pos, name_col, stats, as_of_week, matchup_matrix, upcoming_opponent):
    """
    Per player, per stat: a recency-weighted, opponent-quality-adjusted rate
    - the WITHIN-season half of the cross-season blend (see _blended_rate),
    replacing an earlier flat 60% trailing-4-game / 40% season-average
    split with the three stacked adjustments the module docstring's "THIS
    SEASON'S OWN GAME LOG..." section describes (recency, matchup strength,
    rematch).

    `matchup_matrix` is build_team_game_quality_adjusted_matchup's output - the SAME
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

    # .astype(str) - opponent_team can be categorical upstream, so never
    # carry its raw categorical dtype through a downstream .map()/.clip()
    # chain.
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


def _efficiency_matchup(overall, role_tables, role_sizes, opponents, player_roles, num_stat, den_stat,
                        position=None):
    """clip(rating[numerator] / rating[denominator]) - the part of a
    defense's effect that is NOT already carried by the volume it allows."""
    # QB passing's forward multiplier bypasses role buckets. One QB's
    # passing line is the offense's team passing line, so a role table adds
    # no independent signal and previously re-opened the small-player-row
    # sensitivity fixed by the team-game profile.
    is_qb_passing = (str(position).upper() == 'QB' and
                     num_stat in QB_PASSING_MATCHUP_STATS and
                     den_stat in QB_PASSING_MATCHUP_STATS)
    if is_qb_passing:
        num_mult = _overall_matchup_multiplier(overall, opponents, num_stat)
        den_mult = _overall_matchup_multiplier(overall, opponents, den_stat)
    else:
        num_mult = _role_adjusted_multiplier(overall, role_tables, role_sizes,
                                             opponents, player_roles, num_stat)
        den_mult = _role_adjusted_multiplier(overall, role_tables, role_sizes,
                                             opponents, player_roles, den_stat)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.divide(num_mult, den_mult, out=np.ones_like(num_mult), where=den_mult > 0)
    return np.clip(ratio, *EFFICIENCY_MATCHUP_CLIP)


ROLE_CHANGE_K_REDUCTION = 0.50
# CANDIDATE, built 2026-08-27 - see 'v2_role_change_by_stat' in
# MODEL_FEATURES. Built to test whether dampening RB's role-change trust-
# acceleration fixes v2_adaptive_volume's measured RB harm (component
# backtest program, scripts/backtest_component.py: START-RB dMAE -0.016, 95%
# CI [-0.033,-0.003], excludes zero; 2024-2025 weeks 2-18).
#
# FIRST ATTEMPT (rushing_attempts only) backtested EXACTLY INERT - traced
# and confirmed real (not a wiring bug: _blended_rate correctly received the
# reduction and correctly moved the raw rushing_attempts number, e.g.
# 11.98 -> 11.94 for one real player-week), but rushing_attempts barely
# matters: rushing_yards/rushing_tds are independently blended from their
# OWN historical rate and stayed bit-identical in that same comparison.
#
# A direct with/without-v2_adaptive_volume diff on real RBs (name-indexed,
# not position-indexed - row order is not stable across separate
# build_weekly_projections calls) found the actual channel: targets,
# receptions, AND receiving_yards all move together when the flag is
# toggled (e.g. Chuba Hubbard, 2024 w5: targets 2.667->2.625, receptions
# 2.378->2.418, receiving_yards 17.64->17.934), while rushing stays fixed.
# So the measured RB harm lives on the RECEIVING side, not the rushing side
# - this candidate dampens 'targets' (RB only) alongside 'rushing_attempts',
# not rushing_attempts alone.
#
# RE-BACKTESTED 2026-08-27 with the corrected scope (--add
# v2_role_change_by_stat, 2024-2025 weeks 2-18): real RB win this time. RB
# whole-pool dMAE -0.002, 95% CI [-0.004,-0.001] (excludes zero). START-RB
# dMAE -0.013, CI [-0.030,+0.000] (same direction, right at the edge of
# significance - not fully confirmed at this sample size). WR/QB untouched
# as designed (gated to pos=='RB' only); TE showed a small opposite wiggle
# on only 2 decisive weeks, almost certainly noise since TE isn't touched by
# this mechanism at all. SHIPPED into DEFAULT_FEATURES 2026-08-27 on this
# result, per explicit request.
ROLE_CHANGE_K_REDUCTION_RB_CARRY = 0.0
TD_TWO_YEAR_OLDER_WEIGHT = 0.30
TD_OPPORTUNITY_STAT = {
    'passing_tds': 'passing_attempts',
    'rushing_tds': 'rushing_attempts',
    'receiving_tds': 'targets',
}


def projection_channel(position, stat):
    """Name the position-specific offense/defense channel for a stat.

    The projection loop builds every matchup matrix inside a position filter,
    so QB rushing and RB rushing never share a defensive table.  Exposing the
    channel here makes that invariant testable and visible in V2's trace.
    """
    pos = str(position).upper()
    if pos == 'QB':
        return 'QB rushing' if stat.startswith('rushing_') else 'QB passing'
    if pos == 'RB':
        return 'RB receiving' if stat.startswith(('targets', 'receptions', 'receiving_')) else 'RB rushing'
    if pos == 'WR':
        return 'WR receiving'
    if pos == 'TE':
        return 'TE receiving'
    return f'{pos} other'


def _current_blend_weight(cur_games, stat, role_confidence, role_change_confidence=None,
                          role_change_reduction=ROLE_CHANGE_K_REDUCTION):
    """The current-season weight used by ``_blended_rate`` (also traced).

    ``role_change_reduction`` lets a caller use a smaller (or zero) trust-
    acceleration than the shared ``ROLE_CHANGE_K_REDUCTION`` default for one
    stat - see 'v2_role_change_by_stat' in DEFAULT_FEATURES's own comment for
    why RB rushing_attempts gets its own value.
    """
    games = np.maximum(np.asarray(cur_games, dtype=float), 0.0)
    confidence = np.clip(np.asarray(role_confidence, dtype=float), 0.0, 1.0)
    lo, hi = K_EFFECTIVE_RANGE
    k_eff = STAT_K.get(stat, 3) * (hi - (hi - lo) * confidence)
    if role_change_confidence is not None and stat in {'targets', 'rushing_attempts'}:
        change = np.clip(np.asarray(role_change_confidence, dtype=float), 0.0, 1.0)
        k_eff = k_eff * (1.0 - role_change_reduction * change)
    return games / (games + k_eff)


def blend_comparable_td_priors(last_year_rate, two_year_rate,
                               last_year_opportunity, two_year_opportunity):
    """Use a second TD-rate prior only for a genuinely comparable role.

    Touchdown rate is noisy enough to benefit from more history, but an old
    season from a materially different opportunity role is not helpful.  A
    player needs a usable sample in both seasons and roughly similar
    opportunities per game (within +/-40%) before the older season receives
    a modest 30% weight.  This never affects target/carry volume.
    """
    recent = np.asarray(last_year_rate, dtype=float)
    older = np.asarray(two_year_rate, dtype=float)
    recent_opp = np.asarray(last_year_opportunity, dtype=float)
    older_opp = np.asarray(two_year_opportunity, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.divide(older_opp, recent_opp, out=np.full_like(recent_opp, np.nan),
                          where=recent_opp > 0)
    comparable = (np.isfinite(recent) & np.isfinite(older) & (recent_opp >= 0.8)
                  & (older_opp >= 0.8) & (ratio >= 0.60) & (ratio <= 1.67))
    blended = (1.0 - TD_TWO_YEAR_OLDER_WEIGHT) * recent + TD_TWO_YEAR_OLDER_WEIGHT * older
    return np.where(comparable, blended, recent), comparable


def prior2_blend_weight(games_2025, games_2024, current, prior2_value, games_2026=None):
    """Vectorized two-season (2024) blend weight for one share/rate component.

    Pulled out of build_weekly_projections's ``_blend_with_prior2`` closure
    so the weight math itself - the part with real behavioral rules, not
    the identity-index plumbing around it - is unit-testable directly. All
    arguments broadcast as numpy arrays (or scalars).

    Base weight rises from PRIOR2_BLEND_BASE_WEIGHT (>=8 games in 2025) to
    PRIOR2_BLEND_MAX_WEIGHT (0 games in 2025). Cut to PRIOR2_BLEND_
    DECREASE_DAMPENING of that wherever blending would LOWER ``current``
    (2024 was worse - stay bullish on an ascending player); left at full
    weight wherever it would RAISE it (2024 was better - Lamar Jackson/
    Jayden Daniels: an injury-shortened or down 2025 off a strong 2024).
    ``games_2026``, if given, additionally fades the whole weight to zero
    as this player accumulates his own current-season games (not a
    calendar-week cutoff - a mid-season return from injury has barely
    played, so his 2024 read shouldn't already be gone). A 2024 read under
    PRIOR2_BLEND_MIN_GAMES games (or missing entirely) returns weight 0.
    """
    games_2025 = np.asarray(games_2025, dtype=float)
    games_2024 = np.asarray(games_2024, dtype=float)
    current = np.asarray(current, dtype=float)
    prior2_value = np.asarray(prior2_value, dtype=float)
    fraction_missing = np.clip(
        (PRIOR2_BLEND_FULL_SEASON_GAMES - games_2025) / PRIOR2_BLEND_FULL_SEASON_GAMES, 0.0, 1.0)
    fraction_missing = np.where(np.isnan(games_2025), 1.0, fraction_missing)
    weight = (PRIOR2_BLEND_BASE_WEIGHT
             + (PRIOR2_BLEND_MAX_WEIGHT - PRIOR2_BLEND_BASE_WEIGHT) * fraction_missing)
    pulls_down = prior2_value < current
    weight = np.where(pulls_down, weight * PRIOR2_BLEND_DECREASE_DAMPENING, weight)
    if games_2026 is not None:
        games_2026 = np.asarray(games_2026, dtype=float)
        decay = np.clip(1.0 - games_2026 / PRIOR2_DECAY_GAMES_2026, 0.0, 1.0)
        decay = np.where(np.isnan(games_2026), 1.0, decay)
        weight = weight * decay
    invalid = np.isnan(prior2_value) | np.isnan(games_2024) | (games_2024 < PRIOR2_BLEND_MIN_GAMES)
    return np.where(invalid, 0.0, weight)


def _blended_rate(cur_rate, cur_games, prior_rate, pos_rate, stat, role_confidence,
                  role_change_confidence=None, role_change_reduction=ROLE_CHANGE_K_REDUCTION):
    """
    The one shrinkage formula every stat in this module goes through - see
    the module docstring. All arguments are numpy arrays (vectorized over
    the whole player pool for one stat at a time), never scalars in a loop.

    `cur_rate` is this season's recency-weighted rate (_in_season_rate's
    output), not a bare season total/games - see that function's docstring.
    """
    prior = np.where(np.isnan(prior_rate), pos_rate, prior_rate)
    prior = np.where(np.isnan(prior), 0.0, prior)
    w_current = _current_blend_weight(cur_games, stat, role_confidence, role_change_confidence,
                                      role_change_reduction)
    return w_current * cur_rate + (1 - w_current) * prior


def confirmed_role_change_signal(hist_pos, name_col, team_col, pos, as_of_week):
    """0--1 evidence that a WR/RB's recent opportunity is a new role.

    Three games alone are not enough: the signal requires an established
    current-season comparison window, a usable snap/opportunity floor, and
    an increase in targets (WR) or combined carries/targets (RB).  It is a
    confidence modifier for volume shrinkage, not a multiplier applied to a
    projection, so it cannot manufacture usage a player has not shown.
    """
    if pos not in ('WR', 'RB') or hist_pos.empty or 'week' not in hist_pos.columns:
        return pd.Series(dtype=float)
    frame = hist_pos.copy()
    frame['_week'] = pd.to_numeric(frame['week'], errors='coerce')
    frame = frame[frame['_week'] < as_of_week].sort_values('_week')
    if frame.empty:
        return pd.Series(dtype=float)
    for col in ('targets', 'rushing_attempts', 'weekly_snap_pct'):
        if col not in frame.columns:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors='coerce').fillna(0.0)
    opportunity = frame['targets'] if pos == 'WR' else frame['targets'] + frame['rushing_attempts']
    frame['_opportunity'] = opportunity
    out = {}
    for player, rows in frame.groupby(name_col, observed=True):
        recent = rows.tail(3)
        old = rows.iloc[:-3]
        if len(recent) < 3 or len(old) < 1:
            out[player] = 0.0
            continue
        recent_opp = float(recent['_opportunity'].mean())
        old_opp = float(old['_opportunity'].mean())
        recent_snap = float(recent['weekly_snap_pct'].mean() / 100.0)
        # A role need not raise snap share if a full-time WR changes from a
        # decoy to a first-read target.  A high current snap rate plus a
        # durable opportunity jump is still confirmation.
        opportunity_gain = (recent_opp - old_opp) / max(old_opp, 1.0)
        opportunity_support = np.clip((opportunity_gain - 0.20) / 0.40, 0.0, 1.0)
        snap_support = np.clip((recent_snap - 0.40) / 0.35, 0.0, 1.0)
        out[player] = float(min(opportunity_support, snap_support))
    return pd.Series(out, dtype=float, name='role_change_confidence')


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
    # The visible roster team can be a player's latest destination after a
    # trade.  A whole-season participation denominator belongs to the team
    # that actually played the historical game, never to that later roster.
    game_team = _historical_game_team(stats_df, team_col)
    context = pd.DataFrame({'_team': game_team, '_week': stats_df['week']}, index=stats_df.index)
    weeks_by_team = (context[context['_team'].ne('')].drop_duplicates()
                     .groupby('_team', observed=True)['_week'].nunique())
    total = snap.groupby(stats_df[name_col], observed=True).sum()
    last_team = (pd.DataFrame({'_player': stats_df[name_col], '_team': game_team,
                               '_w': pd.to_numeric(stats_df['week'], errors='coerce')}, index=stats_df.index)
                 .sort_values('_w').groupby('_player', observed=True)['_team'].last().astype(str))
    denom = last_team.reindex(total.index).map(weeks_by_team)
    return (total / denom.replace(0, np.nan)).clip(0.0, 1.0)


def recent_active_share(stats_df, name_col, lookback=8):
    """Active-game snap share over a player's most recent ``lookback``
    appearances within one season - a recency-weighted alternative to
    season_snap_share's WHOLE-season active average, added 2026-08-24 for
    the RB allocator's eligibility gate specifically. A flat season average
    treats a role that grew significantly late in the year identically to
    one that never did: measured on real 2025 data, Tank Bigsby's whole-
    season active share (14.3%) and Will Shipley's (12.6%) sit within 2
    points of each other despite very different roles by year's end - Bigsby
    was up to 39%/59% snap shares in his last two appearances, Shipley's
    finale (41%) was an isolated outlier in an otherwise single-digit-to-
    low-teens season. Their last-8-appearance shares separate them cleanly
    (19.6% vs 14.5%). Deliberately still an ACTIVE-game share (games he
    didn't appear in are skipped, not scored zero) - "is he playing at all"
    stays the injury feed's job, same convention as season_snap_share and
    expected_snap_share."""
    if stats_df.empty or 'weekly_snap_pct' not in stats_df.columns or 'week' not in stats_df.columns:
        return pd.Series(dtype=float)
    frame = stats_df.copy()
    frame['_snap'] = (pd.to_numeric(frame['weekly_snap_pct'], errors='coerce') / 100.0).clip(0.0, 1.0)
    frame['_week'] = pd.to_numeric(frame['week'], errors='coerce')
    frame = frame[frame['_week'].notna() & frame['_snap'].notna()]
    if frame.empty:
        return pd.Series(dtype=float)
    recent = frame.sort_values('_week', kind='stable').groupby(name_col, observed=True).tail(lookback)
    return recent.groupby(name_col, observed=True)['_snap'].mean()


def pre_absence_role_summary(stats_df, name_col, team_col):
    """Evidence for a sustained role that ended in a missed-game stretch.

    This is deliberately conservative: it needs at least four real snap rows
    and three subsequent team games without a player row.  It does not infer
    an injury label; it merely prevents an obvious late absence from becoming
    proof that a player's prior active role permanently disappeared.
    """
    columns = ['pre_absence_snap_share', 'interrupted_season',
               'terminal_gap_weeks', 'pre_absence_games']
    if (stats_df is None or stats_df.empty or 'weekly_snap_pct' not in stats_df.columns
            or 'week' not in stats_df.columns or name_col not in stats_df.columns):
        return pd.DataFrame(columns=columns)
    frame = stats_df.copy()
    frame['_team'] = _historical_game_team(frame, team_col)
    frame['_week'] = pd.to_numeric(frame['week'], errors='coerce')
    frame['_snap'] = (pd.to_numeric(frame['weekly_snap_pct'], errors='coerce') / 100.0).clip(0.0, 1.0)
    usable = frame['_team'].ne('') & frame['_week'].notna() & frame['_snap'].gt(0.0)
    if 'has_snap_match' in frame.columns:
        matched = frame['has_snap_match']
        if pd.api.types.is_bool_dtype(matched):
            usable &= matched.fillna(False).astype(bool)
        else:
            usable &= matched.astype(str).str.lower().isin({'1', 'true', 't', 'yes', 'y'})
    played = frame.loc[usable, [name_col, '_team', '_week', '_snap']].copy()
    if played.empty:
        return pd.DataFrame(columns=columns)
    team_last = frame.loc[frame['_team'].ne('') & frame['_week'].notna()].groupby('_team', observed=True)['_week'].max()
    rows = []
    for player, group in played.sort_values('_week').groupby(name_col, observed=True):
        last = group.iloc[-1]
        terminal_week = team_last.get(last['_team'], np.nan)
        missed_after = terminal_week - last['_week'] if pd.notna(terminal_week) else 0.0
        interrupted = bool(len(group) >= 4 and missed_after >= 3)
        rows.append({name_col: player,
                     'pre_absence_snap_share': float(group.tail(4)['_snap'].mean()),
                     'interrupted_season': interrupted,
                     # Retain the raw gap even when there are only three
                     # proven high-snap appearances.  The normal returning
                     # role rule still requires its larger sample, but a
                     # live charted starter can use this audit field for a
                     # separate bounded short-sample recovery rather than
                     # being silently treated as a 17%-snap backup.
                     'terminal_gap_weeks': float(max(0.0, missed_after)),
                     'pre_absence_games': int(len(group))})
    return pd.DataFrame(rows).set_index(name_col)[columns]


def restore_cold_start_returning_role_share(whole_season_share, active_game_share,
                                            prior_games, prior_team, current_team, pos,
                                            pre_absence_share=None, depth_rank=None,
                                            terminal_gap_weeks=None,
                                            return_details=False):
    """Continuously recover a proven returning skill player's active role.

    This is not a blanket "every rostered player is healthy" switch. It only
    fires for RB/WR/TE players with a meaningful prior sample, a strong role
    while active, and the same team in the coming season. A player who played
    seven games at a 90% snap rate after an injury should not be projected as
    a 37% player solely because those ten missed games are in last year's
    denominator; a three-game backup or a free-agent signing still should.

    Earlier code used a hard 60% active-share threshold.  That made a 59%
    returning lead role receive no credit while a 61% one jumped all the way
    to its active-game share.  The recovery is now evidence-weighted and
    continuous; a depth-chart rank can modestly increase confidence but never
    force a workload by itself.

    A narrow live-preseason exception handles a proven charted starter with
    only three to five eligible high-snap games before a clear terminal gap.
    It exists for an injury-shortened season such as Nabers's, not for every
    three-game fill-in: it requires same-team source rank 1, a 3+ team-game
    gap, and a high active role, and is capped below a normal full-sample
    restoration.  ``return_details`` preserves the legacy two-value API for
    existing callers while exposing the audited recovery reason to V2.
    """
    whole = np.asarray(whole_season_share, dtype=float)
    active = np.asarray(active_game_share, dtype=float)
    games = np.asarray(prior_games, dtype=float)
    pre_absence = (np.asarray(pre_absence_share, dtype=float)
                   if pre_absence_share is not None else active.copy())
    ranks = (np.asarray(depth_rank, dtype=float)
             if depth_rank is not None else np.full(len(whole), np.nan))
    prior_team = pd.Series(prior_team).astype(str).str.strip().str.upper().to_numpy()
    current_team = pd.Series(current_team).astype(str).str.strip().str.upper().to_numpy()
    gaps = (np.asarray(terminal_gap_weeks, dtype=float)
            if terminal_gap_weeks is not None else np.zeros(len(whole), dtype=float))
    minimum_games = COLD_START_RETURNING_ROLE_MIN_GAMES.get(str(pos).upper())
    if minimum_games is None:
        empty = np.zeros(len(whole), dtype=bool)
        if return_details:
            return whole, empty, np.full(len(whole), 'not applicable', dtype=object)
        return whole, empty
    role_signal = np.fmax(np.nan_to_num(active, nan=0.0), np.nan_to_num(pre_absence, nan=0.0))
    restored = np.clip(role_signal, 0.0, COLD_START_RETURNING_ROLE_CAP)
    standard_eligible = (
        np.isfinite(whole) & np.isfinite(active) & np.isfinite(games)
        & (games >= minimum_games)
        & (role_signal >= 0.35)
        & (prior_team == current_team)
        & (restored > whole + 0.05)
    )
    # See the docstring: three high-snap games plus a terminal multi-game
    # absence are still too little to call a generic full workload, but a
    # current, unambiguous charted WR/RB1 should not be forced to the
    # fraction of the calendar he happened to miss.  This path is deliberately
    # unavailable without all of the independently meaningful checks.
    charted_short_sample = (
        (ranks == 1)
        & np.isfinite(games) & (games >= 3) & (games < minimum_games)
        & np.isfinite(gaps) & (gaps >= 3)
        & (role_signal >= 0.75)
        & (prior_team == current_team)
        & (restored > whole + 0.05)
    )
    eligible = standard_eligible | charted_short_sample
    evidence = np.clip((games - minimum_games + 1.0) / 10.0, 0.20, 0.70)
    rank_bonus = np.where(ranks == 1, 0.08, np.where(ranks == 2, 0.03, 0.0))
    standard_alpha = np.clip(0.28 + 0.42 * evidence + rank_bonus, 0.25, 0.78)
    # Three qualifying games receive a 52% pull toward the observed active
    # role; it rises gently with more eligible evidence and can never exceed
    # a 75% snap projection on this short-sample path.
    short_alpha = np.clip(0.52 + 0.035 * (games - 3.0), 0.52, 0.60)
    alpha = np.where(charted_short_sample, short_alpha, standard_alpha)
    recovered = whole + alpha * (restored - whole)
    recovered = np.where(charted_short_sample, np.minimum(recovered, 0.75), recovered)
    values = np.where(eligible, recovered, whole)
    reasons = np.where(
        charted_short_sample, 'charted short-sample returning-starter recovery',
        np.where(standard_eligible, 'continuous returning active/pre-absence role recovery', 'none'),
    )
    if return_details:
        return values, eligible, reasons
    return values, eligible


def apply_ourlads_preseason_role_floor(player_share, player_prior_share, prior_team,
                                       current_team, player_names, pos, skill_roles):
    """Use a local depth chart only as a conservative Week-1 role floor.

    This function deliberately does not allocate total team snaps or targets,
    and never reduces an existing model role.  It only prevents a matched
    listed starter / second-string player from falling to a generic
    low-evidence baseline when he is new to the club or has little meaningful
    prior workload.  That lets a rookie or new signing enter the board while
    retaining the continuous role model for established players.

    Literal Ourlads depth order is retained.  If a source RB2 cannot be
    matched to the roster, source RB3 remains rank 3 rather than being
    re-enumerated as rank 2.  An Ourlads status colour is carried as audit
    metadata; target-week availability is resolved separately and is the
    only input allowed to zero a V2 player's role.
    """
    shares = np.asarray(player_share, dtype=float).copy()
    prior_shares = np.asarray(player_prior_share, dtype=float)
    size = len(shares)
    applied = np.zeros(size, dtype=bool)
    role_rank = np.full(size, np.nan)
    role_floor = np.full(size, np.nan)
    role_label = np.full(size, '', dtype=object)
    if pos not in OURLADS_PRESEASON_ROLE_FLOORS or skill_roles is None or skill_roles.empty:
        return shares, applied, role_rank, role_floor, role_label
    required = {'team', 'position', 'position_label', 'matched_player_key'}
    if not required.issubset(skill_roles.columns):
        return shares, applied, role_rank, role_floor, role_label

    roles = skill_roles[skill_roles['position'].astype(str).str.upper().eq(pos)].copy()
    if roles.empty:
        return shares, applied, role_rank, role_floor, role_label
    roles['_team'] = _clean_team_key(roles['team']).to_numpy()
    roles['_key'] = clean_name_exact(roles['matched_player_key'])
    source_rows = roles['source_row'] if 'source_row' in roles.columns else pd.Series(0, index=roles.index)
    source_slots = (roles['source_slot'] if 'source_slot' in roles.columns
                    else roles['source_rank'] if 'source_rank' in roles.columns
                    else roles['depth_rank'] if 'depth_rank' in roles.columns
                    else pd.Series(0, index=roles.index))
    roles['_row'] = pd.to_numeric(source_rows, errors='coerce').fillna(0)
    roles['_slot'] = pd.to_numeric(source_slots, errors='coerce').fillna(0)
    if 'position_occurrence' in roles.columns:
        # RB/TE overflow rows are an extension of the same source position,
        # not independent starting formations.  The primary row contains the
        # only ranks we use for a modest Week-1 floor.
        roles = roles[pd.to_numeric(roles['position_occurrence'], errors='coerce').fillna(0).eq(0)].copy()
    if roles.empty:
        return shares, applied, role_rank, role_floor, role_label

    role_lookup = {}
    for (team, label), group in roles.groupby(['_team', 'position_label'], observed=True):
        ordered = group.sort_values(['_row', '_slot'], kind='stable').drop_duplicates('_key', keep='first')
        for _, row in ordered.iterrows():
            literal_rank = int(max(1, row['_slot']))
            # The modest role floor only exists for the top two, but the
            # literal chart rank is useful for the V2 team allocator all the
            # way down the primary row.  Keeping RB3 visible is what stops
            # unlisted reserves from displacing him in an ambiguous room.
            floor = OURLADS_PRESEASON_ROLE_FLOORS[pos].get(literal_rank, np.nan)
            key = (team, row['_key'])
            candidate = (literal_rank, float(floor), str(label))
            existing = role_lookup.get(key)
            # A player can theoretically appear in multiple WR formations;
            # retain the stronger (lower-rank) evidence rather than summing
            # formations into an impossible snap share.
            if existing is None or candidate[0] < existing[0]:
                role_lookup[key] = candidate

    current_teams = _clean_team_key(pd.Series(current_team)).to_numpy()
    prior_teams = _clean_team_key(pd.Series(prior_team)).to_numpy()
    keys = clean_name_exact(pd.Series(player_names)).to_numpy()
    for index, (team, key) in enumerate(zip(current_teams, keys)):
        candidate = role_lookup.get((team, key))
        if candidate is None:
            continue
        rank, floor, label = candidate
        role_rank[index] = rank
        role_floor[index] = floor
        role_label[index] = label
        new_team = prior_teams[index] == '' or prior_teams[index] != team
        thin_prior = not np.isfinite(prior_shares[index]) or prior_shares[index] <= OURLADS_LOW_EVIDENCE_PRIOR_SHARE
        if np.isfinite(floor) and (new_team or thin_prior) and shares[index] < floor:
            shares[index] = floor
            applied[index] = True
    return shares, applied, role_rank, role_floor, role_label


def ourlads_player_audit_arrays(matches, teams, identity_keys, player_names):
    """Return source/identity audit fields aligned to a projection player array.

    Ourlads can list one player in more than one formation.  The lowest
    literal source rank is the strongest role evidence, while the full source
    ledger remains available in the data contract.  This helper only exposes
    the one compact row needed in a player popup; it never creates an
    availability or workload decision from source styling.
    """
    size = len(player_names)
    empty = {
        'source_rank': np.full(size, np.nan),
        'source_status': np.full(size, '', dtype=object),
        'source_status_warning': np.full(size, '', dtype=object),
        'identity_match_method': np.full(size, '', dtype=object),
        'identity_match_confidence': np.full(size, '', dtype=object),
        'identity_match_warning': np.full(size, '', dtype=object),
        'source_name': np.full(size, '', dtype=object),
        'roster_identity_key': np.full(size, '', dtype=object),
    }
    if matches is None or matches.empty or 'team' not in matches.columns:
        return empty
    rows = matches.copy()
    if 'matched_identity_key' not in rows.columns:
        rows['matched_identity_key'] = ''
    if 'matched_player_key' not in rows.columns:
        rows['matched_player_key'] = ''
    rows['_team'] = _clean_team_key(rows['team']).to_numpy()
    rows['_identity'] = rows['matched_identity_key'].fillna('').astype(str)
    rows['_name'] = clean_name_exact(rows['matched_player_key'])
    rank_source = (rows['source_rank'] if 'source_rank' in rows.columns
                   else rows['source_depth_rank'] if 'source_depth_rank' in rows.columns
                   else pd.Series(np.nan, index=rows.index))
    rank = pd.to_numeric(rank_source, errors='coerce')
    rows['_rank'] = rank.fillna(9999.0)
    rows = rows.sort_values(['_team', '_identity', '_name', '_rank', 'source_row', 'source_slot'], kind='stable')
    by_identity = rows[rows['_identity'].ne('')].drop_duplicates(['_team', '_identity'], keep='first')
    by_name = rows.drop_duplicates(['_team', '_name'], keep='first')
    identity_lookup = {
        (str(row['_team']), str(row['_identity'])): row
        for _, row in by_identity.iterrows()
    }
    name_lookup = {
        (str(row['_team']), str(row['_name'])): row
        for _, row in by_name.iterrows() if str(row['_name'])
    }
    def _audit_text(value):
        if value is None:
            return ''
        try:
            if bool(pd.isna(value)):
                return ''
        except (TypeError, ValueError):
            pass
        return str(value)

    for index, (team, identity, player) in enumerate(zip(
            _clean_team_key(pd.Series(teams)), pd.Series(identity_keys).astype(str),
            clean_name_exact(pd.Series(player_names)))):
        row = identity_lookup.get((str(team), str(identity)))
        if row is None:
            row = name_lookup.get((str(team), str(player)))
        if row is None:
            continue
        source_rank = pd.to_numeric(pd.Series([row.get('source_rank', row.get('source_depth_rank'))]),
                                    errors='coerce').iloc[0]
        empty['source_rank'][index] = float(source_rank) if pd.notna(source_rank) else np.nan
        empty['source_status'][index] = _audit_text(row.get('source_status', ''))
        empty['source_status_warning'][index] = _audit_text(row.get('source_status_warning', ''))
        empty['identity_match_method'][index] = _audit_text(row.get('match_method', ''))
        empty['identity_match_confidence'][index] = _audit_text(row.get('match_confidence', ''))
        empty['identity_match_warning'][index] = _audit_text(row.get('match_warning', ''))
        empty['source_name'][index] = _audit_text(row.get('player', ''))
        empty['roster_identity_key'][index] = _audit_text(row.get('matched_identity_key', ''))
    return empty


def _read_qb1_override_table(path=QB1_OVERRIDE_PATH):
    """Read the small, explicit preseason-QB1 file without guessing.

    The file is intentionally simple so it can be updated in the app or in a
    spreadsheet:

    ``year,team,player``

    A malformed file is reported to the caller instead of being silently
    replaced.  That keeps a typo from changing a quarterback's workload.
    """
    empty = pd.DataFrame(columns=QB1_OVERRIDE_COLUMNS)
    target = Path(path)
    if not target.exists():
        return empty, None
    try:
        table = pd.read_csv(target, dtype=str, keep_default_na=False)
    except Exception as exc:
        return empty, f'Could not read {target.name}: {exc}'
    normalized = {str(col).strip().lower(): col for col in table.columns}
    missing = [col for col in QB1_OVERRIDE_COLUMNS if col not in normalized]
    if missing:
        return empty, f'{target.name} is missing required column(s): {", ".join(missing)}.'
    table = table.rename(columns={normalized[col]: col for col in QB1_OVERRIDE_COLUMNS})
    table = table.loc[:, list(QB1_OVERRIDE_COLUMNS)].copy()
    table['year'] = pd.to_numeric(table['year'], errors='coerce')
    table['team'] = table['team'].astype(str).str.strip().str.upper()
    table['player'] = table['player'].astype(str).str.strip()
    table = table[(table['team'] != '') & (table['player'] != '') & table['year'].notna()]
    return table, None


def load_qb1_overrides(year, path=QB1_OVERRIDE_PATH):
    """Return user-maintained QB1 choices for one season plus any file error."""
    table, problem = _read_qb1_override_table(path)
    if problem or table.empty:
        return pd.DataFrame(columns=QB1_OVERRIDE_COLUMNS), problem
    selected = table[table['year'].astype(int).eq(int(year))].copy()
    if selected.empty:
        return pd.DataFrame(columns=QB1_OVERRIDE_COLUMNS), None
    # Multiple rows for one club are intentionally retained here so the
    # resolver can mark the room ambiguous and expose an actionable warning.
    return selected.reset_index(drop=True), None


def _invalidate_weekly_projection_cache():
    """Ensure a saved QB1 choice is reflected on the very next app rerun."""
    builder = globals().get('build_weekly_projections')
    clear = getattr(builder, 'clear', None)
    if callable(clear):
        clear()


def save_qb1_override(year, team, player, path=QB1_OVERRIDE_PATH):
    """Persist one user-selected preseason QB1, replacing that team's row."""
    table, problem = _read_qb1_override_table(path)
    if problem:
        raise ValueError(problem)
    team = str(team).strip().upper()
    player = str(player).strip()
    if not team or not player:
        raise ValueError('Both team and player are required for a QB1 override.')
    keep = ~((pd.to_numeric(table['year'], errors='coerce').eq(int(year))) & table['team'].eq(team))
    table = pd.concat([
        table.loc[keep, list(QB1_OVERRIDE_COLUMNS)],
        pd.DataFrame([{'year': int(year), 'team': team, 'player': player}]),
    ], ignore_index=True)
    table = table.sort_values(['year', 'team'], kind='stable').reset_index(drop=True)
    table.to_csv(Path(path), index=False)
    _invalidate_weekly_projection_cache()


def clear_qb1_override(year, team, path=QB1_OVERRIDE_PATH):
    """Remove one manual selection; an unambiguous incumbent can still win."""
    table, problem = _read_qb1_override_table(path)
    if problem:
        raise ValueError(problem)
    team = str(team).strip().upper()
    keep = ~((pd.to_numeric(table['year'], errors='coerce').eq(int(year))) & table['team'].eq(team))
    table.loc[keep, list(QB1_OVERRIDE_COLUMNS)].to_csv(Path(path), index=False)
    _invalidate_weekly_projection_cache()


def resolve_preseason_qb1s(current_qbs, current_name_col, current_team_col,
                            prior_played, prior_name_col, prior_team_col,
                            year, overrides=None, ourlads_qb1s=None,
                            unavailable_players=None):
    """Resolve cold-start QB workload sources from explicit, auditable inputs.

    Precedence is deliberately narrow: a manual selection wins, then a
    matched healthy first available QB from a locally imported Ourlads chart,
    then a single clear prior-season incumbent.  A partial season, rookie/new
    arrival without chart evidence, or two former starters in the same room
    remains unresolved until the user selects ``year/team/player`` in
    ``qb1_overrides.csv``.  The generated app depth chart is never consulted.

    Returns a transparent payload rather than only a set, so both the model
    and the UI can distinguish an automatic incumbent from an unresolved
    decision.  The projection layer uses this single selection to grant QB
    volume; it never turns every roster QB into a starter just to hide
    ambiguity.
    """
    empty = {'selected': {}, 'by_team': {}, 'selection_required_teams': set(), 'warnings': []}
    required = {current_name_col, current_team_col}
    if current_qbs is None or current_qbs.empty or not required.issubset(current_qbs.columns):
        return empty
    current = current_qbs.loc[:, [current_name_col, current_team_col]].dropna(subset=[current_name_col]).copy()
    current['_team'] = current[current_team_col].astype(str).str.strip().str.upper()
    current['_player'] = current[current_name_col].astype(str).str.strip()
    current['_key'] = clean_name_exact(current['_player'])
    current = current[(current['_team'] != '') & (current['_key'] != '')]
    current = current.drop_duplicates(subset=['_team', '_key'])
    if current.empty:
        return empty
    unavailable = set(clean_name_exact(pd.Series(list(unavailable_players or []))))

    if overrides is None:
        overrides, file_problem = load_qb1_overrides(year)
    else:
        file_problem = None
        overrides = overrides.copy()
    warnings = []
    if file_problem:
        warnings.append(file_problem)
    if overrides is None:
        overrides = pd.DataFrame(columns=QB1_OVERRIDE_COLUMNS)
    for col in QB1_OVERRIDE_COLUMNS:
        if col not in overrides.columns:
            overrides[col] = ''
    overrides = overrides.loc[:, list(QB1_OVERRIDE_COLUMNS)].copy()
    overrides['team'] = overrides['team'].astype(str).str.strip().str.upper()
    overrides['player'] = overrides['player'].astype(str).str.strip()
    overrides = overrides[(overrides['team'] != '') & (overrides['player'] != '')]

    chart_qbs = pd.DataFrame(columns=['_team', '_key', '_player', '_source_slot', '_source_status'])
    if ourlads_qb1s is not None and not ourlads_qb1s.empty:
        needed = {'team', 'matched_player_key'}
        if needed.issubset(ourlads_qb1s.columns):
            chart_qbs = ourlads_qb1s.copy()
            chart_qbs['_team'] = _clean_team_key(chart_qbs['team']).to_numpy()
            chart_qbs['_key'] = clean_name_exact(chart_qbs['matched_player_key'])
            player_source = (chart_qbs['matched_player'] if 'matched_player' in chart_qbs.columns
                             else chart_qbs['_key'])
            chart_qbs['_player'] = player_source.astype(str).str.strip()
            chart_qbs['_source_slot'] = pd.to_numeric(
                chart_qbs.get('source_slot', pd.Series(1, index=chart_qbs.index)), errors='coerce').fillna(1).astype(int)
            chart_qbs['_source_status'] = (
                chart_qbs['status_class'].astype(str)
                if 'status_class' in chart_qbs.columns else ''
            )
            chart_qbs = chart_qbs[(chart_qbs['_team'] != '') & (chart_qbs['_key'] != '')]
            chart_qbs = chart_qbs.drop_duplicates(subset=['_team', '_key'])

    # Re-key the prior season by player, not by the old team.  A veteran who
    # plainly held a full workload and changed clubs remains an obvious
    # incumbent *unless* that new room contains another such incumbent.
    prior_shares = pd.Series(dtype=float)
    if (prior_played is not None and not prior_played.empty and
            {prior_name_col, prior_team_col, 'position'}.issubset(prior_played.columns)):
        prior_qbs = prior_played[prior_played['position'].astype(str).str.upper().eq('QB')]
        if not prior_qbs.empty:
            prior_shares = season_snap_share(prior_qbs, prior_name_col, prior_team_col)
    if not prior_shares.empty:
        prior_keyed = pd.DataFrame({
            '_key': clean_name_exact(pd.Series(prior_shares.index)),
            '_prior_share': pd.to_numeric(prior_shares.to_numpy(), errors='coerce'),
        }).groupby('_key', observed=True)['_prior_share'].max()
        current['_prior_share'] = current['_key'].map(prior_keyed).fillna(0.0)
    else:
        current['_prior_share'] = 0.0

    selected, by_team, requires_selection = {}, {}, set()
    for team, room in current.groupby('_team', observed=True):
        manual = overrides[overrides['team'].eq(team)]
        if len(manual) > 1:
            warnings.append(f'{team}: multiple QB1 override rows; choose exactly one before using a full starter workload.')
            requires_selection.add(team)
            by_team[team] = {'status': 'selection_required', 'reason': 'multiple manual choices'}
            continue
        if len(manual) == 1:
            manual_key = clean_name_exact(pd.Series([manual.iloc[0]['player']])).iloc[0]
            match = room[room['_key'].eq(manual_key)]
            if len(match) == 1:
                row = match.iloc[0]
                if row['_key'] in unavailable:
                    warnings.append(
                        f"{team}: QB1 override '{row['_player']}' is explicitly unavailable in the target-week availability source.")
                else:
                    selected[(team, row['_key'])] = 'manual_override'
                    by_team[team] = {
                        'status': 'manual_override', 'player': row['_player'],
                        'prior_snap_share': float(row['_prior_share']),
                        'reason': 'explicit preseason QB1 selection',
                    }
                    continue
            else:
                warnings.append(f"{team}: QB1 override '{manual.iloc[0]['player']}' does not match one current roster QB.")

        chart = chart_qbs[chart_qbs['_team'].eq(team) & ~chart_qbs['_key'].isin(unavailable)]
        if len(chart) == 1:
            chart_row = chart.iloc[0]
            match = room[room['_key'].eq(chart_row['_key'])]
            if len(match) == 1:
                row = match.iloc[0]
                selected[(team, row['_key'])] = 'ourlads_depth_chart'
                by_team[team] = {
                    'status': 'ourlads_depth_chart', 'player': row['_player'],
                    'prior_snap_share': float(row['_prior_share']),
                    'source_slot': int(chart_row['_source_slot']),
                    'source_status': str(chart_row['_source_status']),
                    'reason': 'first available QB on locally imported Ourlads depth chart',
                }
                continue
            warnings.append(
                f"{team}: imported Ourlads QB1 '{chart_row['_player']}' does not match one current roster QB.")
        elif len(chart) > 1:
            warnings.append(f'{team}: multiple imported Ourlads QB1 candidates; ignored for automatic selection.')

        incumbents = room[(~room['_key'].isin(unavailable))
                          & room['_prior_share'].ge(QB1_AUTO_INCUMBENT_MIN_SHARE)]
        if len(incumbents) == 1:
            row = incumbents.iloc[0]
            selected[(team, row['_key'])] = 'prior_season_incumbent'
            by_team[team] = {
                'status': 'prior_season_incumbent', 'player': row['_player'],
                'prior_snap_share': float(row['_prior_share']),
                'reason': f'at least {QB1_AUTO_INCUMBENT_MIN_SHARE:.0%} of prior-team season snaps',
            }
            continue

        requires_selection.add(team)
        if len(incumbents) > 1:
            reason = 'multiple full-season prior incumbents in this QB room'
        else:
            reason = f'no QB reached {QB1_AUTO_INCUMBENT_MIN_SHARE:.0%} of prior-team season snaps'
        by_team[team] = {'status': 'selection_required', 'reason': reason}

    return {
        'selected': selected,
        'by_team': by_team,
        'selection_required_teams': requires_selection,
        'warnings': warnings,
    }


def resolve_inseason_qb1s(current_qbs, current_name_col, current_team_col,
                           recent_history, history_name_col, history_team_col,
                           as_of_week, year, overrides=None,
                           unavailable_players=None):
    """Resolve one expected QB1 per team from explicit choices or real snaps.

    An in-season board should not turn a backup's one relief appearance into
    an independent 5--10 point projection.  A manual QB1 choice always wins.
    Without one, this function selects a QB only when the most recent
    snap-based role is clearly a full starter role.  An ambiguous room is
    intentionally left unresolved, so no QB receives invented starter
    volume until the user makes an auditable choice.
    """
    empty = {'selected': {}, 'by_team': {}, 'selection_required_teams': set(), 'warnings': []}
    required = {current_name_col, current_team_col}
    if current_qbs is None or current_qbs.empty or not required.issubset(current_qbs.columns):
        return empty
    current = current_qbs.loc[:, [current_name_col, current_team_col]].dropna(
        subset=[current_name_col]).copy()
    current['_team'] = _clean_team_key(current[current_team_col]).to_numpy()
    current['_player'] = current[current_name_col].astype(str).str.strip()
    current['_key'] = clean_name_exact(current['_player'])
    current = current[(current['_team'] != '') & (current['_key'] != '')]
    current = current.drop_duplicates(subset=['_team', '_key'])
    if current.empty:
        return empty

    if overrides is None:
        overrides, file_problem = load_qb1_overrides(year)
    else:
        file_problem = None
        overrides = overrides.copy()
    warnings = [file_problem] if file_problem else []
    if overrides is None:
        overrides = pd.DataFrame(columns=QB1_OVERRIDE_COLUMNS)
    for col in QB1_OVERRIDE_COLUMNS:
        if col not in overrides.columns:
            overrides[col] = ''
    overrides = overrides.loc[:, list(QB1_OVERRIDE_COLUMNS)].copy()
    overrides['team'] = _clean_team_key(overrides['team']).to_numpy()
    overrides['player'] = overrides['player'].astype(str).str.strip()
    overrides = overrides[(overrides['team'] != '') & (overrides['player'] != '')]

    current['_recent_snap_share'] = 0.0
    current['_last_snap_week'] = np.nan
    current['_team_latest_week'] = np.nan
    current['_recently_active'] = False
    if (recent_history is not None and not recent_history.empty
            and {history_name_col, history_team_col, 'position', 'week', 'weekly_snap_pct'}.issubset(
                recent_history.columns)):
        history = _played_weeks_before(recent_history, as_of_week)
        if not history.empty:
            context = history.copy()
            context['_team'] = _historical_game_team(context, history_team_col).to_numpy()
            context['_week'] = pd.to_numeric(context['week'], errors='coerce')
            team_latest = context[(context['_team'] != '') & context['_week'].notna()].groupby(
                '_team', observed=True)['_week'].max()
            recent_qbs = context[context['position'].astype(str).str.upper().eq('QB')].copy()
            recent_qbs['_snap'] = (pd.to_numeric(recent_qbs['weekly_snap_pct'], errors='coerce') / 100.0)
            real_snap = recent_qbs['_snap'].gt(0.0) & recent_qbs['_snap'].le(1.0)
            if 'has_snap_match' in recent_qbs.columns:
                raw_match = recent_qbs['has_snap_match']
                if pd.api.types.is_bool_dtype(raw_match):
                    matched = raw_match.fillna(False).astype(bool)
                else:
                    matched = raw_match.astype(str).str.strip().str.lower().isin(
                        {'1', 'true', 't', 'yes', 'y'})
                real_snap &= matched
            recent_qbs = recent_qbs[real_snap & (recent_qbs['_team'] != '')].copy()
            if not recent_qbs.empty:
                recent_qbs['_key'] = clean_name_exact(recent_qbs[history_name_col])
                recent_qbs = recent_qbs.sort_values(['_team', '_key', '_week'])
                tail = recent_qbs.groupby(['_team', '_key'], observed=True).tail(
                    PARTIAL_GAME_REFERENCE_APPEARANCES)
                signals = tail.groupby(['_team', '_key'], observed=True).agg(
                    # The most recent eligible game is the timely starter
                    # signal.  A new starter should not need three games to
                    # overcome an injured predecessor's old full-snap starts.
                    _recent_snap_share=('_snap', 'last'),
                    _last_snap_week=('_week', 'max'),
                )
                current_index = pd.MultiIndex.from_arrays([current['_team'], current['_key']])
                current['_recent_snap_share'] = signals['_recent_snap_share'].reindex(
                    current_index).fillna(0.0).to_numpy(dtype=float)
                current['_last_snap_week'] = signals['_last_snap_week'].reindex(
                    current_index).to_numpy(dtype=float)
            current['_team_latest_week'] = current['_team'].map(team_latest).to_numpy(dtype=float)
            current['_recently_active'] = (
                current['_last_snap_week'].notna()
                & current['_team_latest_week'].notna()
                & current['_last_snap_week'].ge(
                    current['_team_latest_week'] - (QB1_INSEASON_MAX_STALE_TEAM_GAMES - 1))
            )

    unavailable = set()
    if unavailable_players:
        unavailable = set(clean_name_exact(pd.Series(list(unavailable_players))))
    selected, by_team, requires_selection = {}, {}, set()
    for team, room in current.groupby('_team', observed=True):
        manual = overrides[overrides['team'].eq(team)]
        if len(manual) > 1:
            warnings.append(f'{team}: multiple QB1 override rows; choose exactly one before projecting QB volume.')
            requires_selection.add(team)
            by_team[team] = {'status': 'selection_required', 'reason': 'multiple manual choices'}
            continue
        if len(manual) == 1:
            manual_key = clean_name_exact(pd.Series([manual.iloc[0]['player']])).iloc[0]
            match = room[room['_key'].eq(manual_key)]
            if len(match) == 1:
                row = match.iloc[0]
                selected[(team, row['_key'])] = 'manual_override'
                by_team[team] = {
                    'status': 'manual_override', 'player': row['_player'],
                    'recent_snap_share': float(row['_recent_snap_share']),
                    'reason': 'explicit upcoming-game QB1 selection',
                }
                continue
            warnings.append(f"{team}: QB1 override '{manual.iloc[0]['player']}' does not match one current QB.")

        available = room[~room['_key'].isin(unavailable)].copy()
        recent_available = available[available['_recently_active']].copy()
        if recent_available.empty:
            requires_selection.add(team)
            by_team[team] = {
                'status': 'selection_required',
                'reason': 'no recently active QB with a clear starter role',
            }
            continue
        ordered = recent_available.sort_values('_recent_snap_share', ascending=False, kind='stable')
        top = ordered.iloc[0]
        runner = float(ordered.iloc[1]['_recent_snap_share']) if len(ordered) > 1 else 0.0
        top_share = float(top['_recent_snap_share'])
        if (top_share >= QB1_INSEASON_MIN_SNAP_SHARE
                and top_share - runner >= QB1_INSEASON_MIN_LEAD):
            selected[(team, top['_key'])] = 'observed_current_starter'
            by_team[team] = {
                'status': 'observed_current_starter', 'player': top['_player'],
                'recent_snap_share': top_share,
                'reason': (f'at least {QB1_INSEASON_MIN_SNAP_SHARE:.0%} recent snaps and '
                           f'{QB1_INSEASON_MIN_LEAD:.0%} lead over next QB'),
            }
        else:
            requires_selection.add(team)
            by_team[team] = {
                'status': 'selection_required',
                'reason': 'no single recent full-snap QB starter',
                'top_recent_snap_share': top_share,
            }
    return {
        'selected': selected,
        'by_team': by_team,
        'selection_required_teams': requires_selection,
        'warnings': warnings,
    }


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


def _role_confidence_detail(stats_df, name_col, as_of_week, pos, pff_rec):
    """Audit-only companion to `_role_confidence`: the raw ingredients that
    formula actually reads, not the 0-1 output itself. Deliberately a
    near-duplicate of `_role_confidence` rather than a refactor of it - this
    function is read only by the UI's decomposition audit, so a bug here
    must never be able to change a scored role_confidence value. Returns a
    DataFrame indexed like `_role_confidence`'s own Series, columns:
    `recent_snap_pct` (mean of the last up to 3 played games' snap share),
    `games_sampled` (how many of those games were available, 0-3),
    `route_rate` (PFF season-to-date route rate, WR/TE only, NaN if no PFF
    row matched), `method` (which ingredient(s) actually fed the number
    _role_confidence returned for this player)."""
    hist = _played_weeks_before(stats_df[stats_df['position'].astype(str).str.upper() == pos], as_of_week)
    if hist.empty or 'weekly_snap_pct' not in hist.columns:
        return pd.DataFrame(columns=['recent_snap_pct', 'games_sampled', 'route_rate', 'method'])
    tail_games = hist.sort_values('week').groupby(name_col).tail(3)
    recent = (tail_games.groupby(name_col)['weekly_snap_pct'].mean() / 100.0).clip(0, 1)
    games_sampled = tail_games.groupby(name_col)['weekly_snap_pct'].count()
    route_rate = pd.Series(dtype=float, index=recent.index)
    method = pd.Series('snap share only (last up to 3 games)', index=recent.index)

    if pos in ('WR', 'TE') and pff_rec is not None and not pff_rec.empty and 'player' in pff_rec.columns:
        routes = pff_rec.copy()
        routes['_key'] = clean_name_exact(routes['player'])
        route_by_key = pd.to_numeric(
            routes.set_index('_key')['route_rate'], errors='coerce') / 100.0
        exact_keys = clean_name_exact(pd.Series(recent.index))
        route_rate = pd.Series(
            route_by_key.reindex(exact_keys.values).to_numpy(), index=recent.index)
        method = pd.Series(
            np.where(route_rate.notna(),
                     'snap share + PFF route rate (averaged)',
                     'snap share only (no PFF route row matched)'),
            index=recent.index)

    return pd.DataFrame({
        'recent_snap_pct': recent,
        'games_sampled': games_sampled.reindex(recent.index),
        'route_rate': route_rate,
        'method': method,
    })


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


def annotate_player_history_participation(history, name_col, team_col, schedule_df=None):
    """Mark only clearly interrupted player-games as unusable rate evidence.

    Weekly box scores do not include a trustworthy timestamped injury/bench
    field.  This helper therefore does *not* infer an injury from a quiet
    stat line, or treat every modest snap dip as a partial game.  It excludes
    a player-game from that player's own full-game production history only
    when measured snaps support one of these narrow descriptions:

    * a proven full-time player abruptly logged roughly half a game or less;
    * two QBs split a game's offensive snaps (starter exit, relief, or
      garbage-time handoff), so neither per-game passing line is a normal
      upcoming-starter sample;
    * a previously fringe player took part of the same-position work after
      that proven teammate's abrupt exit; or
    * a proven starter had sharply reduced work in a 28+ point win, a clear
      late-game-rest situation rather than an ordinary lower-workload game.

    Rows with missing or zero snap data remain eligible.  The annotation is
    intentionally for *player baseline/rate* inputs only.  Defense profiles
    must continue to use the raw team-game history: an injured starter and
    his replacement still describe what the defense faced that day.
    """
    annotated = history.copy()
    annotated['_player_history_eligible'] = True
    annotated['_player_history_reason'] = ''
    required = {name_col, team_col, 'position', 'week', 'weekly_snap_pct'}
    if annotated.empty or not required.issubset(annotated.columns):
        return annotated

    snap = pd.to_numeric(annotated['weekly_snap_pct'], errors='coerce') / 100.0
    usable_snap = snap.gt(0.0) & snap.le(1.0)
    # ``weekly_snap_pct`` is deliberately filled with 0 when a source has no
    # week-level snap export.  If an explicit match flag is available, honor
    # it so an unavailable source can never masquerade as participation.
    if 'has_snap_match' in annotated.columns:
        raw_match = annotated['has_snap_match']
        if pd.api.types.is_bool_dtype(raw_match):
            matched = raw_match.fillna(False).astype(bool)
        else:
            matched = raw_match.astype(str).str.strip().str.lower().isin(
                {'1', 'true', 't', 'yes', 'y'})
        usable_snap &= matched
    if not usable_snap.any():
        return annotated

    frame = annotated.loc[usable_snap].copy()
    frame['_row'] = np.flatnonzero(usable_snap.to_numpy())
    frame['_snap'] = snap.loc[usable_snap].to_numpy(dtype=float)
    frame['_player'] = clean_name_exact(frame[name_col])
    frame['_offense'] = _historical_game_team(frame, team_col).to_numpy()
    frame['_position'] = frame['position'].astype(str).str.upper()
    frame['_week'] = pd.to_numeric(frame['week'], errors='coerce')
    frame = frame[(frame['_player'] != '') & (frame['_offense'] != '')
                  & frame['_week'].notna()
                  & frame['_position'].isin(DRAFTABLE_POSITIONS)].copy()
    if frame.empty:
        return annotated

    if 'game_id' in frame.columns:
        game_id = frame['game_id'].astype(str).str.strip()
        fallback = (frame['_offense'].astype(str) + '|' + frame['_week'].astype(str) + '|'
                    + _historical_game_opponent(frame).astype(str))
        frame['_game_key'] = game_id.where(~game_id.isin(('', 'nan', 'None')), fallback)
    else:
        frame['_game_key'] = (frame['_offense'].astype(str) + '|' + frame['_week'].astype(str) + '|'
                              + _historical_game_opponent(frame).astype(str))

    # A player's reference is only the prior real appearances known at that
    # point in history.  This avoids reading future snaps to label a past
    # game, which keeps both live projections and historical backtests clean.
    frame = frame.sort_values(['_player', '_week', '_game_key', '_row']).copy()
    reference = np.full(len(frame), np.nan)
    appearances = np.zeros(len(frame), dtype=int)
    for _player, positions in frame.groupby('_player', observed=True).indices.items():
        prior_snaps = []
        for position in positions:
            recent = prior_snaps[-PARTIAL_GAME_REFERENCE_APPEARANCES:]
            appearances[position] = len(recent)
            if recent:
                reference[position] = float(np.median(recent))
            prior_snaps.append(float(frame.iloc[position]['_snap']))
    frame['_prior_snap_reference'] = reference
    frame['_prior_snap_appearances'] = appearances
    frame['_established_role'] = (
        frame['_prior_snap_appearances'].ge(PARTIAL_GAME_MIN_REFERENCE_APPEARANCES)
        & frame['_prior_snap_reference'].ge(PARTIAL_GAME_ESTABLISHED_SNAP_SHARE)
    )
    frame['_abrupt_partial'] = (
        frame['_established_role']
        & frame['_snap'].gt(0.0)
        & frame['_snap'].le(PARTIAL_GAME_ABSOLUTE_MAX_SNAP_SHARE)
        & frame['_snap'].le(PARTIAL_GAME_RELATIVE_MAX_SHARE * frame['_prior_snap_reference'])
    )

    reasons = np.full(len(annotated), '', dtype=object)

    def mark(rows, reason):
        for row in rows:
            row_id = int(row)
            if reasons[row_id] == '':
                reasons[row_id] = reason

    # QBs have one shared passing workload.  A 52/48 or 75/25 split is
    # inherently not a normal per-game starter observation for either QB.
    qb_rows = frame[frame['_position'].eq('QB')]
    for _key, group in qb_rows.groupby(['_offense', '_game_key'], observed=True):
        split = group[(group['_snap'].ge(QB_SPLIT_MIN_SNAP_SHARE))
                      & (group['_snap'].le(QB_SPLIT_MAX_SNAP_SHARE))]
        combined = float(split['_snap'].sum())
        if len(split) >= 2 and QB_SPLIT_MIN_COMBINED_SHARE <= combined <= QB_SPLIT_MAX_COMBINED_SHARE:
            mark(group['_row'].to_numpy(), 'QB split/relief game')

    # A QB row where he barely touched the field is not a start by any
    # reading, established role or not - _abrupt_partial above only fires
    # for a PROVEN player's sudden drop, which structurally can never catch
    # a rookie/first-time player's own opening token appearance (no prior
    # snaps exist yet to compare against). Found 2026-08-25 on Jaxson Dart's
    # 2025 weeks 2-3: 4%/5% snap share, 0 passing attempts, a single
    # 1-carry garbage/gadget rush each, sitting in his rate history at full
    # weight right next to his real starts once he actually took over in
    # week 4 - dragging his season average down toward a game he functionally
    # didn't play. Gated on BOTH a snap share under the same 20% floor that
    # defines a legitimate split-start above AND zero passing attempts (when
    # that column exists) so a real low-snap package/committee passer who
    # still threw the ball in his limited reps is not swept in alongside him.
    qb_token_appearance = frame['_position'].eq('QB') & frame['_snap'].lt(QB_SPLIT_MIN_SNAP_SHARE)
    if 'passing_attempts' in frame.columns:
        pass_att = pd.to_numeric(frame['passing_attempts'], errors='coerce').fillna(0.0)
        qb_token_appearance &= pass_att.le(0.0)
    mark(frame.loc[qb_token_appearance, '_row'].to_numpy(),
         'QB token appearance (under 20% snaps, zero pass attempts)')

    mark(frame.loc[frame['_abrupt_partial'], '_row'].to_numpy(),
         'abrupt partial role after established workload')

    # A partial replacement gets the same treatment only when it occurs in
    # the exact game as the proven starter's abrupt exit and the incoming
    # player had no established role.  This avoids treating normal RB/WR
    # committees as injuries merely because several players share snaps.
    for _key, group in frame.groupby(['_offense', '_position', '_game_key'], observed=True):
        if not group['_abrupt_partial'].any():
            continue
        incoming = group[
            ~group['_abrupt_partial']
            & group['_snap'].ge(PARTIAL_REPLACEMENT_MIN_SNAP_SHARE)
            & group['_snap'].le(PARTIAL_REPLACEMENT_MAX_SNAP_SHARE)
            & ((group['_prior_snap_appearances'] < PARTIAL_GAME_MIN_REFERENCE_APPEARANCES)
               | group['_prior_snap_reference'].le(PARTIAL_REPLACEMENT_MAX_PRIOR_SHARE))
        ]
        mark(incoming['_row'].to_numpy(), 'partial replacement after teammate exit')

    # Margin is supporting evidence only.  A score cannot tell us when a
    # player sat, so use it solely for a huge winning margin plus an already
    # established player's materially reduced measured participation.
    margins = _team_week_margins(schedule_df)
    if not margins.empty:
        margin_frame = margins.copy()
        margin_frame['_offense'] = _clean_team_key(margin_frame['Team'])
        margin_frame['_week'] = pd.to_numeric(margin_frame['week'], errors='coerce')
        margin_map = margin_frame.dropna(subset=['_week']).drop_duplicates(
            ['_offense', '_week'], keep='last').set_index(['_offense', '_week'])['margin']
        lookup = pd.MultiIndex.from_frame(frame[['_offense', '_week']])
        frame['_final_margin'] = margin_map.reindex(lookup).to_numpy(dtype=float)
        blowout_rest = (
            frame['_established_role']
            & frame['_final_margin'].ge(SEVERE_BLOWOUT_MARGIN)
            & frame['_snap'].gt(0.0)
            & frame['_snap'].le(SEVERE_BLOWOUT_MAX_SNAP_SHARE)
            & frame['_snap'].le(SEVERE_BLOWOUT_RELATIVE_MAX_SHARE * frame['_prior_snap_reference'])
        )
        mark(frame.loc[blowout_rest, '_row'].to_numpy(), 'severe blowout rest')

    excluded = reasons != ''
    annotated['_player_history_eligible'] = ~excluded
    annotated['_player_history_reason'] = reasons
    return annotated


def _player_history_exclusion_summary(annotated, name_col):
    """Per-player count/reasons for the projection decomposition."""
    if (annotated is None or annotated.empty
            or '_player_history_eligible' not in annotated.columns
            or name_col not in annotated.columns):
        return pd.DataFrame(columns=['excluded_games', 'excluded_reasons'])
    excluded = annotated[~annotated['_player_history_eligible'].astype(bool)]
    if excluded.empty:
        return pd.DataFrame(columns=['excluded_games', 'excluded_reasons'])
    summary = excluded.groupby(name_col, observed=True).agg(
        excluded_games=('_player_history_eligible', 'size'),
        excluded_reasons=('_player_history_reason',
                          lambda values: '; '.join(sorted({str(v) for v in values if str(v)}))),
    )
    return summary


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


def _game_env_multiplier(env, teams, pos, league_implied, use_total=True, use_venue=True):
    """Per-player game-environment multiplier: implied-total elasticity x
    venue. 1.0 for any team without a posted line (the normal case more than
    a few weeks out), same degrade-gracefully convention as every other
    best-effort signal here.

    ``use_total``/``use_venue`` let a caller apply just ONE of the two
    factors - added 2026-08-27 for the component backtest program: the
    BUNDLED 'game_env' feature was built, measured, and rejected (+0.012 MAE,
    see DEFAULT_FEATURES's own comment), but a bundle that fails as a whole
    can still be hiding one real, useful piece diluted by a non-working one -
    exactly what happened with the original alignment mechanism before its
    redesign. 'v2_game_total_elasticity' and 'v2_venue_mult' isolate each
    piece for its own backtest rather than re-testing the same bundle again.
    """
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
        if use_total and implied and league_implied and league_implied > 0 and implied > 0 and elasticity:
            mult *= float(np.clip((implied / league_implied) ** elasticity, *GAME_TOTAL_CLIP))
        if use_venue:
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
    # draft_sources intentionally falls back to the last available injury
    # season for draft research.  That is useful there but unsafe for a live
    # next-season weekly projection: a 2025 IR designation must never create
    # a 2026 Week-1 vacancy.  The source contract records the fallback year
    # in ``attrs`` specifically so consumers can reject it here.
    source_year = injuries.attrs.get('season', year)
    if int(source_year) != int(year):
        return {}
    status = injuries['Injury Status'].astype(str).str.strip().str.lower()
    mult = status.map(INJURY_MULTIPLIER)
    return dict(zip(injuries['Player'], mult.dropna()))


def _injury_profiles(year, week):
    """Live availability probability separated from workload-if-active.

    There is no time-correct historical injury archive in the current data
    stack, so callers must disable this for historical tests.  Unlike V1,
    a Questionable player is not assumed to become a worse player if he
    plays: uncertainty belongs in availability, while conditional workload
    stays at his normal projection until player-type-specific evidence says
    otherwise.
    """
    try:
        from data.draft_sources import fetch_injury_report
        injuries, _err = fetch_injury_report(year, as_of_week=week)
    except Exception:
        return {}
    if injuries is None or injuries.empty or 'Player' not in injuries.columns:
        return {}
    source_year = injuries.attrs.get('season', year)
    if int(source_year) != int(year):
        return {}
    statuses = injuries.get('Injury Status', pd.Series('', index=injuries.index))
    out = {}
    for player, status in zip(injuries['Player'], statuses):
        label = str(status).strip().lower()
        if label in {'out', 'ir', 'suspended'}:
            availability = 0.0
        elif label == 'doubtful':
            availability = 0.25
        elif label == 'questionable':
            availability = 0.85
        else:
            availability = 1.0
        out[player] = {
            'status': label or 'unknown',
            'plays_probability': availability,
            'workload_if_active': 1.0,
            'source_year': int(source_year),
            'source': 'target-season injury report',
        }
    return out


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

# (dependent, opportunity) pairs that are physically impossible to invert -
# a reception requires a target, a receiving/rushing TD requires a
# reception/carry. Each stat on this board is independently blended from
# its OWN historical rate (see the module docstring), so nothing upstream
# actually enforces this; small-sample noise can occasionally let a
# dependent's own rate land fractionally above its opportunity's.
DEPENDENT_STAT_CLAMPS = (
    ('receptions', 'targets'),
    ('receiving_tds', 'receptions'),
    ('rushing_tds', 'rushing_attempts'),
)


def clamp_dependent_stats(result):
    """Enforce receptions<=targets and TDs<=catches/carries on the final board.

    Only ever pulls a DEPENDENT stat down to match its opportunity count -
    the opportunity side (targets, receptions-as-TD-opportunity, carries) is
    never adjusted to fit a dependent, since the opportunity count is the
    better-evidenced, more fundamental quantity of the two. A no-op for any
    row that is already consistent, which is the large majority of them.
    """
    if result is None or result.empty:
        return result
    out = result.copy()
    for dependent, opportunity in DEPENDENT_STAT_CLAMPS:
        if dependent not in out.columns or opportunity not in out.columns:
            continue
        dep_values = pd.to_numeric(out[dependent], errors='coerce')
        opp_values = pd.to_numeric(out[opportunity], errors='coerce')
        out[dependent] = np.where(
            dep_values.notna() & opp_values.notna() & (dep_values > opp_values),
            opp_values, out[dependent])
    return out


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


V2_VACANCY_SURVIVAL = {'passing_attempts': 0.85, 'rushing_attempts': 0.80, 'targets': 0.80}
V2_VACANCY_MAX_GROWTH = 2.00


def redistribute_v2_vacated_usage(result, injury_profiles, skip_rb=False, skip_receivers=False):
    """Role-specific, ledgered vacancy redistribution for the V2 experiment.

    This is intentionally narrower than the old all-skill-player allocator:
    an absent WR/TE reallocates only to active WR/TE target earners, an RB's
    carries and targets use separate RB recipient pools, and an unavailable
    QB's pass volume goes to one best available projected QB.  If a credible
    replacement is absent from the as-of player pool, the volume remains
    explicitly unallocated instead of being invented for another position.
    """
    if result.empty or not injury_profiles or 'Team' not in result.columns:
        return result, 0, []
    out = result.copy()
    availability = out['Player'].map(
        lambda p: float(injury_profiles.get(p, {}).get('plays_probability', 1.0)))
    sidelined = availability <= 0.01
    if not sidelined.any():
        return out, 0, []
    out['_v2_availability'] = availability
    adjusted, ledger = set(), []

    def _allocate(volume_col, dependent_cols, source_positions, recipient_positions, one_recipient=False):
        if volume_col not in out.columns:
            return
        volume = pd.to_numeric(out[volume_col], errors='coerce').fillna(0.0)
        full_col = f'_full_{volume_col}'
        pre = pd.to_numeric(out.get(full_col, volume), errors='coerce').fillna(0.0)
        source = sidelined & out['Pos'].isin(source_positions)
        if not source.any():
            return
        for team in out.loc[source, 'Team'].astype(str).unique():
            source_rows = out.index[source & out['Team'].astype(str).eq(team)]
            vacated = float(pre.loc[source_rows].sum())
            reusable = vacated * V2_VACANCY_SURVIVAL[volume_col]
            recipient_mask = (out['Team'].astype(str).eq(team) & ~sidelined
                              & out['Pos'].isin(recipient_positions))
            # QB volume has one expected starter recipient.  An unresolved
            # room deliberately has none: vacancy accounting should surface
            # that volume as unallocated rather than quietly recreate a
            # backup projection the QB1 gate removed above.
            if one_recipient and 'QB Projected Starter' in out.columns:
                recipient_mask &= out['QB Projected Starter'].fillna(False).astype(bool)
            candidates = out.index[recipient_mask]
            if not len(candidates) or reusable <= 0:
                ledger.append({'team': team, 'volume': volume_col, 'vacated': vacated,
                               'allocated': 0.0, 'unallocated': reusable,
                               'reason': 'No projected, active role-compatible replacement.'})
                continue
            weights = volume.loc[candidates].clip(lower=0.0)
            snap_values = (out.loc[candidates, 'Expected Snap Share']
                           if 'Expected Snap Share' in out.columns
                           else pd.Series(0.0, index=candidates))
            snaps = pd.to_numeric(snap_values, errors='coerce').fillna(0.0)
            if one_recipient:
                # Expected snap share is a time-valid depth proxy when it is
                # measured.  It prevents a QB with one garbage-time drive
                # from winning merely because his per-appearance rate was
                # high; ties fall back to existing projected volume.
                winner = (snaps * 10.0 + weights).idxmax()
                weights = pd.Series(0.0, index=candidates)
                weights.loc[winner] = 1.0
            elif weights.sum() <= 0:
                weights = snaps.clip(lower=0.0)
            if weights.sum() <= 0:
                ledger.append({'team': team, 'volume': volume_col, 'vacated': vacated,
                               'allocated': 0.0, 'unallocated': reusable,
                               'reason': 'Role-compatible replacements have no as-of opportunity/depth evidence.'})
                continue
            requested = weights / weights.sum() * reusable
            allowed = (volume.loc[candidates] * V2_VACANCY_MAX_GROWTH - volume.loc[candidates]).clip(lower=0.0)
            gain = np.minimum(requested, allowed)
            allocated = float(gain.sum())
            factor = (volume.loc[candidates] + gain) / volume.loc[candidates].replace(0, np.nan)
            # A just-promoted replacement can legitimately have zero prior
            # volume.  Use a finite factor based on its candidate allocation
            # and scale only dependent stats when there was a prior rate.
            factor = factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            out.loc[candidates, volume_col] = (volume.loc[candidates] + gain).round(2)
            for dep in dependent_cols:
                if dep in out.columns:
                    base = pd.to_numeric(out.loc[candidates, dep], errors='coerce').fillna(0.0)
                    out.loc[candidates, dep] = (base * factor).round(2)
            adjusted.update(candidates[gain > 0])
            ledger.append({'team': team, 'volume': volume_col, 'vacated': vacated,
                           'allocated': allocated, 'unallocated': max(0.0, reusable - allocated),
                           'reason': 'Role-compatible as-of projected recipients.'})

    _allocate('passing_attempts',
              ('passing_completions', 'passing_yards', 'passing_tds', 'passing_interceptions'),
              {'QB'}, {'QB'}, one_recipient=True)
    if not skip_rb:
        _allocate('rushing_attempts', ('rushing_yards', 'rushing_tds'), {'RB'}, {'RB'})
    if not skip_receivers:
        if not skip_rb:
            _allocate('targets', ('receptions', 'receiving_yards', 'receiving_tds'), {'RB'}, {'RB'})
        _allocate('targets', ('receptions', 'receiving_yards', 'receiving_tds'), {'WR', 'TE'}, {'WR', 'TE'})
    return out.drop(columns=['_v2_availability']), len(adjusted), ledger


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
    # Roster/weekly source merges can retain duplicate metadata labels (most
    # often player_id).  Label selection on such a frame returns *both*
    # physical columns even when a name appears only once in ``cols``, which
    # later turns a supposedly one-dimensional allocator input into a 2-D
    # array.  The first source column is the established loader precedence;
    # normalize the cold roster view once before selecting metadata.
    source = stats_df.loc[:, ~stats_df.columns.duplicated()].copy()
    # Keep stable identifiers and narrow roster metadata through the cold
    # pool.  The prior implementation immediately reduced this to
    # name/team/position, forcing every cross-season join back through a
    # brittle name string and making a roster FB look indistinguishable from
    # a normal RB.  These fields are projection inputs, not UI decoration.
    identity_cols = (
        name_col, team_col, 'position', 'player_id', 'gsis_id', 'pff_id',
        'depth_chart_position', 'status', 'roster_position_group',
        'draft_number', 'is_rookie_flag', 'years_exp', 'entry_year',
    )
    cols = list(dict.fromkeys(c for c in identity_cols if c in source.columns))
    if 'week' in source.columns:
        this_week = source[pd.to_numeric(source['week'], errors='coerce') == as_of_week]
        pool = this_week[cols] if not this_week.empty else source[cols]
    else:
        pool = source[cols]
    # A preseason overlay can legitimately present the same player under a
    # source spelling such as ``James Cook III`` while the roster says
    # ``James Cook``.  Deduplicating display names would leave both rows in
    # the pool (or, worse, keep the wrong one after a later sort).  Stable
    # roster identity is authoritative whenever it exists; exact canonical
    # name is only the deliberate no-ID fallback.  This keeps two genuinely
    # distinct same-name players separate while making source spelling a
    # non-event for allocation.
    pool = pool.dropna(subset=[name_col]).copy()
    pool['_cold_pool_identity'] = player_identity_keys(pool, name_col)
    pool = pool.drop_duplicates(subset=['_cold_pool_identity'], keep='first')
    pool = pool.drop(columns=['_cold_pool_identity'])
    if 'status' in pool.columns:
        # A strict hard-status filter is appropriate before a Week-1 player
        # pool is built.  It does not guess at questionable/injury status;
        # those remain the live-injury feed's responsibility.
        # .astype(object) FIRST, not after: 'status' can arrive as a pandas
        # Categorical column (from the roster source), and .fillna('') on a
        # Categorical raises TypeError unless '' is already one of its
        # defined categories - found 2026-08-27, broke every true cold start
        # (Week 1) outright. Casting to object first drops the category
        # restriction while still preserving real NaN for fillna to catch.
        status = pool['status'].astype(object).fillna('').astype(str).str.strip().str.upper()
        pool = pool[~status.isin({'RET', 'CUT', 'RES', 'FA'})].copy()
    return pool


def _load_pff_receiving(year, allow_season_totals=True):
    """Load PFF receiving data only when its coverage is time-valid.

    The local export is season-total data without a week/timestamp column.
    It is useful for a live, upcoming slate whose observed stats end before
    the target week, but it would leak every later game into a historical
    projection.  V2 calls this with ``False`` for that latter case.
    """
    if not allow_season_totals:
        return pd.DataFrame()
    try:
        from data.loaders import load_pff_data_with_fallback
        pff, source_year = load_pff_data_with_fallback(year)
        return pff.get('rec', pd.DataFrame())
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def build_weekly_projections(year, week, scoring_mode='Full PPR', as_of_week=None, apply_injury=True,
                             features=None, availability_fingerprint=None):
    """
    This app's own projected stat line + fantasy points for every QB/RB/WR/
    TE with usable history, for one week.

    `as_of_week` defaults to `week` (project week N off weeks < N of the
    same season - the real-time use). Passed explicitly and lower than
    `week` by the backtest harness in docs/weekly_projections_methodology.md
    to validate against a week that's already happened without leaking its
    own result into the projection that's supposed to predict it.

    `availability_fingerprint` is never read inside this function - it
    exists ONLY so the caller can pass
    data.availability_overrides.availability_fingerprint(year, week, ...)
    and get a real cache-key change out of an availability edit, instead of
    the caller having to call this function's own .clear() (which wipes
    every cached year/week/scoring/model combination, not just the edited
    one - see that helper's own docstring). Injury/override files are still
    read fresh from disk below regardless of what's passed here; this
    parameter's only job is making that freshness visible to
    @st.cache_data.

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
    build_team_game_quality_adjusted_matchup off last season's full year instead,
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
    feats = resolve_model_features(features)
    stats_df, team_col, name_col, _ = load_and_merge_data(year, scoring_mode)
    if stats_df.empty:
        return pd.DataFrame(), {'reason': f'No roster or weekly data for {year} yet.'}

    observed_weeks = (pd.to_numeric(stats_df.get('week', pd.Series(dtype=float)), errors='coerce')
                      .dropna())
    latest_observed_week = int(observed_weeks[observed_weeks > 0].max()) \
        if (observed_weeks > 0).any() else None
    # If the loaded season includes the target week or a later one, this is
    # necessarily a historical evaluation.  Sources that only expose a
    # season total (PFF alignment, full-season pace) and live injury/odds
    # feeds cannot be treated as known then.
    historical_target = latest_observed_week is not None and latest_observed_week >= as_of_week
    use_v2_guard = 'v2_as_of_guard' in feats
    source_contract = {
        'as_of_week': int(as_of_week),
        'latest_observed_week': latest_observed_week,
        'historical_target': historical_target,
        'pff_season_totals': 'eligible_live_only',
        'pace': 'weekly_box_score_proxy' if use_v2_guard and historical_target else 'season_loader',
        'injury': 'disabled_historical' if use_v2_guard and historical_target else 'live_report',
        'market_script': 'disabled_historical' if use_v2_guard and historical_target else 'live_market',
        'prior_defense_recency': (
            f'{int(PRIOR_SEASON_DEFENSE_RECENCY_FLOOR * 100)}% full-season baseline + '
            f'{int((1.0 - PRIOR_SEASON_DEFENSE_RECENCY_FLOOR) * 100)}% late-season tilt'
        ),
    }

    hist = _played_weeks_before(stats_df, as_of_week) if 'week' in stats_df.columns else stats_df.iloc[0:0]
    cold_start = hist.empty

    prior_stats, prior_team_col, prior_name_col, _ = pd.DataFrame(), team_col, name_col, None
    try:
        prior_stats, prior_team_col, prior_name_col, _ = load_and_merge_data(year - 1, scoring_mode)
    except Exception:
        prior_stats = pd.DataFrame()
    prior2_stats, prior2_team_col, prior2_name_col = pd.DataFrame(), team_col, name_col
    if 'v2_td_two_year_prior' in feats:
        try:
            prior2_stats, prior2_team_col, prior2_name_col, _ = load_and_merge_data(year - 2, scoring_mode)
        except Exception:
            prior2_stats = pd.DataFrame()

    if cold_start and (prior_stats is None or prior_stats.empty):
        return pd.DataFrame(), {'reason': f'Week {as_of_week} of {year} has no games played yet this '
                                          f'season, and {year - 1} has no data to fall back on either - '
                                          'nothing to build even a rough cold-start projection from.'}

    schedule_df = load_schedule(year)
    opponents = _week_opponents(schedule_df, week)
    env = game_environment(schedule_df, week) if (
        'game_env' in feats or 'v2_game_total_elasticity' in feats or 'v2_venue_mult' in feats
    ) else {}
    league_implied = None
    if env:
        implied_vals = [e['implied'] for e in env.values() if e.get('implied')]
        # League average for the WEEK being projected, not a fixed constant -
        # scoring environments move year to year, and this multiplier is
        # supposed to say "richer than a typical game", not "richer than 2019".
        league_implied = float(np.mean(implied_vals)) if implied_vals else None
    target_margins = (_target_margins_by_team(year, week)
                      if not (use_v2_guard and historical_target) else {})
    # Keep the raw target-season feed separate until we have the current
    # roster pool.  Availability data frequently has a display-name variant;
    # resolving it directly onto the live pool prevents a source spelling
    # from silently missing (or being confused with a same-name player).
    raw_injury_profiles = {}
    if apply_injury and not (use_v2_guard and historical_target):
        if 'v2_fantasypros_availability' in feats:
            # FantasyPros-sourced, healthy by default: an empty dict here
            # (no file uploaded yet, or nothing filed for this year/week) is
            # not "unknown", it IS the player's availability - see
            # data/fantasypros_availability.py. This deliberately does not
            # fall back to nflverse; that is the whole point of the switch.
            raw_injury_profiles, _fp_error = load_fantasypros_availability(year, week)
        elif 'v2_availability' in feats:
            raw_injury_profiles = _injury_profiles(year, week)
    injury_profiles = {}
    availability_resolution_warnings = []
    pff_rec = _load_pff_receiving(
        year, allow_season_totals=not (use_v2_guard and historical_target))
    # The weekly archive is a separately audited PFF source.  It never
    # scrapes PFF and only uses reports strictly before the target week.
    # The foundation is intentionally neutral in scoring until its
    # defense-alignment residual clears its own backtest gate; meanwhile it
    # makes actual slot/non-slot player evidence visible in the popup.
    pff_alignment_profiles = pd.DataFrame()
    pff_alignment_contract = {
        'status': 'feature disabled', 'source_kind': 'none', 'included_weeks': [],
        'issues': [], 'adjustment': 'neutral (no alignment matchup multiplier applied)',
    }
    # Two-years-back (e.g. 2024 for a 2026 target) alignment archive, used
    # ONLY to lightly ground an in-season 2026 read toward the player's own
    # prior-year tendency - see blend_alignment_profile_toward_prior2. Loaded
    # once here, looked up per player alongside pff_alignment_profiles below.
    # Empty/unused at cold start (week 1 has no in-season read yet to
    # ground) and unless 'v2_td_two_year_prior' - the SAME flag already
    # gating every other stat's 2024 grounding - is also active.
    prior2_alignment_profiles = pd.DataFrame()
    pff_alignment_defense_profiles = pd.DataFrame()
    # Raw per-week, per-alignment defense evidence behind
    # pff_alignment_defense_profiles' season-aggregate, shrunk candidate
    # multipliers - kept so the UI can show what a defense actually allowed
    # to slot/wide/inline (and the whole position, alignment-blind) by week,
    # not just the final heavily-shrunk-toward-1.0 number. See
    # load_weekly_alignment_defense_profiles's AlignmentDefenseLoadResult -
    # its own docstring already flags team_games as meant for exactly this,
    # it just wasn't threaded through yet.
    pff_alignment_defense_team_games = pd.DataFrame()
    pff_alignment_defense_contract = {
        'status': 'feature disabled', 'source_kind': 'none', 'included_weeks': [],
        'issues': [], 'adjustment': 'neutral (defense residual preview only; no scoring multiplier applied)',
        'defender_slot_coverage_used': False,
    }
    if 'v2_pff_alignment_matchup' in feats:
        try:
            if cold_start:
                # include_postseason_weeks folds the source season's own
                # WC/DIV/CONF/SB weekly archive (weeks 19-22) in as
                # supplementary evidence on top of the regular-season total -
                # added 2026-08-25 alongside the 2025 postseason weekly
                # archive itself; disabled automatically inside the loader
                # for a historical backtest (see its own docstring).
                alignment_result = load_season_alignment_prior(
                    year - 1, year, week,
                    historical_backtest=bool(historical_target),
                    include_postseason_weeks=True,
                )
                if not alignment_result.available:
                    # No reviewed season_manifest.csv for the prior year (a
                    # season-total file with no manifest is refused above,
                    # by design - e.g. 2024's own season-total receiving
                    # export includes real playoff production for every
                    # team that made the postseason, confirmed 2026-08-25 by
                    # diffing it against real regular-season-only box
                    # scores, so it is correctly NOT marked regular_season
                    # in a manifest). Falls back to aggregating that prior
                    # year's own full WEEKLY archive instead - the same
                    # source the defense side below always uses, and, once
                    # a full season of weekly exports exists, a strictly
                    # more precise source than a season total anyway (real
                    # per-game measurement, not a single lump sum). Reuses
                    # PFF_ALIGNMENT_DEFENSE_COLD_START_AS_OF_WEEK purely as
                    # "large enough to admit every real week 1-22" - the
                    # postseason opt-in below is the actual gate, not this
                    # sentinel.
                    fallback_result = load_weekly_alignment_profiles(
                        year - 1, PFF_ALIGNMENT_DEFENSE_COLD_START_AS_OF_WEEK,
                        include_postseason=not bool(historical_target))
                    if fallback_result.available:
                        alignment_result = fallback_result
            else:
                alignment_result = load_weekly_alignment_profiles(year, as_of_week)
            pff_alignment_profiles = alignment_result.profiles
            pff_alignment_contract = {
                'status': 'time-valid local alignment profiles available'
                if alignment_result.available else 'no eligible local alignment profiles',
                'source_kind': alignment_result.metadata.get('source_kind', 'weekly_archive'),
                'included_weeks': list(alignment_result.metadata.get('included_weeks', ())),
                'postseason_weeks_included': list(alignment_result.metadata.get('postseason_weeks_included', ())),
                'issues': list(alignment_result.issues),
                'adjustment': 'neutral (no alignment matchup multiplier applied)',
            }
            # No longer gated behind `not cold_start` (removed 2026-08-26 per
            # explicit request to include 2024 "to an extent" even at Week
            # 1) - see ALIGNMENT_PRIOR2_MAX_WEIGHT's own comment in
            # pff_alignment.py for why this lands at a light ~10% pull at
            # cold start rather than the formula's higher in-season ceiling.
            if 'v2_td_two_year_prior' in feats:
                prior2_result = load_weekly_alignment_profiles(
                    year - 2, PFF_ALIGNMENT_DEFENSE_COLD_START_AS_OF_WEEK,
                    include_postseason=not bool(historical_target))
                if prior2_result.available:
                    prior2_alignment_profiles = prior2_result.profiles
        except Exception as exc:
            # A malformed optional local export must not break the rankings
            # board.  Its diagnostic remains visible in the data contract.
            pff_alignment_contract = {
                'status': 'alignment archive unavailable', 'source_kind': 'weekly_archive',
                'included_weeks': [], 'issues': [f'{type(exc).__name__}: {exc}'],
                'adjustment': 'neutral (no alignment matchup multiplier applied)',
            }
        try:
            # Cold start (a new season, nothing played yet) mirrors the
            # player-side prior above: the CURRENT year's weekly archive is
            # always empty at that point (no games exist to export), so
            # without this branch the defense-allowed-by-alignment residual
            # could never activate until the current season had already
            # played several weeks. Falls back to the full completed PRIOR
            # season archive instead - every one of its weeks is eligible
            # (source_year < target week's own season, so the as-of-week
            # cutoff that protects against future leakage doesn't apply to
            # data that's a full year old), mapped through that SAME prior
            # season's own schedule (never the new season's schedule, which
            # would misattribute the offense/defense matchups). Added
            # 2026-08-24 alongside the wide/inline 3-way blend, so the
            # defense side actually has real 2025 evidence to work with for
            # a 2026 Week 1 board. include_postseason=True added 2026-08-25
            # alongside the WC/DIV/CONF/SB weekly archive (weeks 19-22) -
            # see PFF_ALIGNMENT_DEFENSE_COLD_START_AS_OF_WEEK's own comment
            # for why the sentinel had to move from 19 to 23 at the same time.
            if cold_start:
                alignment_defense_result = load_weekly_alignment_defense_profiles(
                    year - 1, PFF_ALIGNMENT_DEFENSE_COLD_START_AS_OF_WEEK,
                    load_schedule(year - 1, include_postseason=True),
                    include_postseason=True)
            else:
                alignment_defense_result = load_weekly_alignment_defense_profiles(
                    year, as_of_week, schedule_df)
            pff_alignment_defense_profiles = alignment_defense_result.profiles
            pff_alignment_defense_team_games = alignment_defense_result.team_games
            pff_alignment_defense_contract = {
                'status': ('time-valid offensive-weekly alignment defense profiles available'
                           if alignment_defense_result.available
                           else 'no eligible local alignment defense profiles'),
                'source_kind': alignment_defense_result.metadata.get(
                    'source_kind', 'weekly_offensive_alignment_archive'),
                'included_weeks': list(alignment_defense_result.metadata.get('included_weeks', ())),
                'issues': list(alignment_defense_result.issues),
                # This branch only runs when 'v2_pff_alignment_matchup' is in
                # feats - never true for DEFAULT_FEATURES, true for
                # V2_EXPERIMENTAL_FEATURES only as a diagnostic reactivation
                # (see that set's own comment): the candidate_multiplier this
                # profile feeds was BUILT, MEASURED, AND REJECTED 2026-08-24.
                # Whenever this branch does run it IS multiplied into the
                # WR/TE targets/receptions/receiving_yards matchup step - see
                # ALIGNMENT_SCORING_STAT_MAP's application site - which is
                # what this contract's flag reports. The underlying profile's
                # own 'scoring_active' flag stays False
                # (that describes whether pff_alignment.py itself would apply
                # it without a caller's gate, a separate question from what
                # this specific run did).
                'adjustment': 'applied (WR/TE targets/receptions/receiving_yards matchup residual)',
                'scoring_active': True,
                'profile_scoring_active': bool(alignment_defense_result.metadata.get('scoring_active', False)),
                'requires_backtest_before_activation': bool(
                    alignment_defense_result.metadata.get('requires_backtest_before_activation', True)),
                'defender_slot_coverage_used': bool(
                    alignment_defense_result.metadata.get('defender_slot_coverage_used', False)),
                'team_game_rows': int(alignment_defense_result.metadata.get('team_game_rows', 0)),
                'profile_rows': int(alignment_defense_result.metadata.get('profile_rows', 0)),
            }
        except Exception as exc:
            # This source is explanatory only until an explicit OOS release
            # gate is met; malformed local files must not stop rankings.
            pff_alignment_defense_contract = {
                'status': 'alignment-defense archive unavailable',
                'source_kind': 'weekly_offensive_alignment_archive',
                'included_weeks': [],
                'issues': [f'{type(exc).__name__}: {exc}'],
                'adjustment': 'neutral (defense residual preview only; no scoring multiplier applied)',
                'defender_slot_coverage_used': False,
            }
    # Man/zone scheme evidence - same weekly archive, same dormant/audit-only
    # posture, same feature gate as the slot/wide/inline pair above (one
    # lever for "show this PFF weekly evidence in the audit panel", not two).
    # No season-prior equivalent exists yet for the PLAYER side (that would
    # need its own load_season_alignment_prior-style manifest/rollover
    # logic); at cold start pff_scheme_profiles honestly stays empty rather
    # than guessing. The DEFENSE side needs no such branch - it is always
    # keyed to the CURRENT year's archive with an as_of_week cutoff, exactly
    # like pff_alignment_defense_profiles above, and is simply empty/
    # unavailable at Week 1 when nothing has been played yet.
    pff_scheme_profiles = pd.DataFrame()
    pff_scheme_contract = {
        'status': 'feature disabled', 'source_kind': 'none', 'included_weeks': [],
        'issues': [], 'adjustment': 'neutral (no scheme matchup multiplier applied)',
    }
    pff_scheme_defense_profiles = pd.DataFrame()
    pff_scheme_defense_contract = {
        'status': 'feature disabled', 'source_kind': 'none', 'included_weeks': [],
        'issues': [], 'adjustment': 'neutral (defense residual preview only; no scoring multiplier applied)',
    }
    if 'v2_pff_alignment_matchup' in feats:
        if not cold_start:
            try:
                scheme_result = load_weekly_scheme_profiles(year, as_of_week)
                pff_scheme_profiles = scheme_result.profiles
                pff_scheme_contract = {
                    'status': 'time-valid local scheme profiles available'
                    if scheme_result.available else 'no eligible local scheme profiles',
                    'source_kind': scheme_result.metadata.get('source_kind', 'weekly_archive'),
                    'included_weeks': list(scheme_result.metadata.get('included_weeks', ())),
                    'issues': list(scheme_result.issues),
                    'adjustment': 'neutral (no scheme matchup multiplier applied)',
                }
            except Exception as exc:
                pff_scheme_contract = {
                    'status': 'scheme archive unavailable', 'source_kind': 'weekly_archive',
                    'included_weeks': [], 'issues': [f'{type(exc).__name__}: {exc}'],
                    'adjustment': 'neutral (no scheme matchup multiplier applied)',
                }
        try:
            scheme_defense_result = load_weekly_scheme_defense_profiles(
                year, as_of_week, schedule_df)
            pff_scheme_defense_profiles = scheme_defense_result.profiles
            pff_scheme_defense_contract = {
                'status': ('time-valid offensive-weekly scheme defense profiles available'
                           if scheme_defense_result.available
                           else 'no eligible local scheme defense profiles'),
                'source_kind': scheme_defense_result.metadata.get(
                    'source_kind', 'weekly_offensive_scheme_archive'),
                'included_weeks': list(scheme_defense_result.metadata.get('included_weeks', ())),
                'issues': list(scheme_defense_result.issues),
                'adjustment': 'neutral (defense residual preview only; no scoring multiplier applied)',
                'scoring_active': bool(scheme_defense_result.metadata.get('scoring_active', False)),
                'team_game_rows': int(scheme_defense_result.metadata.get('team_game_rows', 0)),
                'profile_rows': int(scheme_defense_result.metadata.get('profile_rows', 0)),
            }
        except Exception as exc:
            pff_scheme_defense_contract = {
                'status': 'scheme-defense archive unavailable',
                'source_kind': 'weekly_offensive_scheme_archive',
                'included_weeks': [], 'issues': [f'{type(exc).__name__}: {exc}'],
                'adjustment': 'neutral (defense residual preview only; no scoring multiplier applied)',
            }
    source_contract['pff_alignment'] = pff_alignment_contract
    source_contract['pff_alignment_defense'] = pff_alignment_defense_contract
    source_contract['pff_scheme'] = pff_scheme_contract
    source_contract['pff_scheme_defense'] = pff_scheme_defense_contract
    pace = (as_of_team_pace(stats_df, team_col, as_of_week)
            if use_v2_guard and historical_target else load_team_pace(year))
    if cold_start and (pace is None or pace.empty):
        # This season's own pace data doesn't exist yet either (same reason
        # as everything else in a cold start) - last season's team pace is
        # a far better estimate of week 1 than the neutral 1.0 fallback
        # below would be.
        pace = load_team_pace(year - 1)
    league_pace = pace['def_pace'].mean() if pace is not None and not pace.empty and 'def_pace' in pace.columns else None

    # Per-game play counts for the defense-matchup ratio's own pace
    # normalization (see _team_game_quality_profile's ``plays`` docstring) -
    # same live-vs-cutoff-safe source split as ``pace`` immediately above,
    # just kept at per-(team, week) granularity instead of collapsed to a
    # season average. ``current_plays`` covers this season's games (used by
    # the in-season branch's own-year matchup/role tables); ``prior_plays``
    # covers last season in full (used by every prior-season matchup/role
    # table, at cold start and mid-season alike) - last season is already
    # complete, so it carries no leakage risk and needs no as-of cutoff.
    current_plays = _team_game_plays_lookup(
        as_of_team_weekly_plays(stats_df, team_col, as_of_week)
        if use_v2_guard and historical_target else load_team_weekly_plays(year))
    prior_plays = _team_game_plays_lookup(load_team_weekly_plays(year - 1))

    cold_pool = _cold_start_pool(stats_df, name_col, team_col, as_of_week) if cold_start else pd.DataFrame()
    prior_played = _all_played_weeks(prior_stats)
    prior2_played = _all_played_weeks(prior2_stats)

    # Ourlads is a personal, locally imported preseason source.  It is never
    # fetched over the network here, and a current-season snapshot must never
    # leak into a historical Week 1 backtest.  The chart is informative when
    # it identifies a current role, not a replacement for in-season snaps.
    ourlads_signal = {
        'matches': pd.DataFrame(), 'qb_starters': pd.DataFrame(), 'skill_roles': pd.DataFrame(),
        'warnings': [], 'matched_teams': [],
    }
    ourlads_source_contract = {
        'status': 'not applicable outside a live preseason cold start',
        'snapshot_teams': 0, 'matched_teams': 0, 'warnings': [],
    }
    if cold_start and not historical_target:
        ourlads_snapshot, ourlads_problem = load_ourlads_snapshot(year)
        if ourlads_problem:
            ourlads_source_contract['status'] = f'ignored: {ourlads_problem}'
            ourlads_source_contract['warnings'] = [ourlads_problem]
        elif ourlads_snapshot.empty:
            ourlads_source_contract['status'] = 'no local Ourlads snapshot imported for this season'
        elif cold_pool.empty:
            ourlads_source_contract['status'] = 'snapshot available but current cold-start roster was empty'
        else:
            # An lc_red (inactive) Ourlads row is treated as a warning, not
            # dropped outright - the availability resolver downstream is what
            # actually discounts it. (Until 2026-08-26 the retired V1 mode
            # filtered these rows out here instead; that behavioral split no
            # longer exists now that this is the only model.)
            snapshot_for_model = ourlads_snapshot
            cold_pool, roster_overlay, overlay_warnings = apply_ourlads_starter_roster_overlay(
                snapshot_for_model, cold_pool, name_col, team_col,
                prior_played, prior_name_col, prior_team_col,
            )
            ourlads_signal = build_ourlads_projection_signal(
                snapshot_for_model, cold_pool, name_col, team_col)
            ourlads_source_contract = {
                'status': 'active local preseason role signal',
                'snapshot_teams': int(ourlads_snapshot['team'].nunique()),
                'matched_teams': len(ourlads_signal['matched_teams']),
                'matched_players': len(ourlads_signal['matches']),
                'matched_qb1s': len(ourlads_signal['qb_starters']),
                'roster_overlay_changes': roster_overlay,
                'warnings': list(overlay_warnings) + list(ourlads_signal['warnings']),
            }
    if (cold_start and not historical_target and not cold_pool.empty
            and 'v2_preseason_rb_allocator' in feats):
        # Functional position is intentionally separate from the broad fantasy
        # roster position.  An FB stays visible in the fantasy-RB display
        # pool so its own tiny historic touch rate can be shown, but it is
        # marked FB before the allocator/vacancy paths and therefore never
        # receives a generic RB fallback.  Current depth-chart metadata is
        # live-preseason evidence only, never a historical-backtest input.
        if not ourlads_signal['matches'].empty:
            matched = ourlads_signal['matches'].copy()
            if {'matched_player_key', 'functional_position'}.issubset(matched.columns):
                roles = matched[['matched_player_key', 'functional_position']].dropna().copy()
                roles['_is_fb'] = roles['functional_position'].astype(str).str.upper().eq('FB')
                roles = (roles.sort_values(['matched_player_key', '_is_fb'], kind='stable')
                         .drop_duplicates('matched_player_key', keep='last'))
                mapped = clean_name_exact(cold_pool[name_col]).map(
                    roles.set_index('matched_player_key')['functional_position'])
                cold_pool['ourlads_position'] = mapped.fillna('').to_numpy(dtype=object)
        cold_pool['functional_position'] = classify_functional_position(cold_pool)
        broad_projection = cold_pool.get('position', pd.Series('', index=cold_pool.index)).astype(str).str.upper()
        # The rankings table has QB/RB/WR/TE display buckets.  A fullback
        # belongs under RB for a transparent low-usage line, while its
        # separate ``functional_position`` retains the football distinction.
        cold_pool['projection_position'] = broad_projection.replace({'HB': 'RB', 'TB': 'RB', 'FB': 'RB'})
    source_contract['ourlads_preseason_depth_chart'] = ourlads_source_contract

    # A chart's color communicates a source flag, not a target-week medical
    # designation.  Resolve actual availability only from the current report
    # and the small explicit override file, after the roster overlay has
    # supplied the current identity/team context.  This is V2-only: V1 keeps
    # its established direct injury-map behavior as a control.
    manual_availability = pd.DataFrame()
    manual_availability_problem = None
    if (apply_injury and not (use_v2_guard and historical_target)
            and ('v2_availability' in feats or 'v2_fantasypros_availability' in feats)):
        manual_availability, manual_availability_problem = load_availability_overrides(year, week)
        availability_roster = (cold_pool if cold_start and not cold_pool.empty
                               else stats_df.loc[:, ~stats_df.columns.duplicated()].copy())
        injury_profiles, availability_resolution_warnings = resolve_target_week_availability(
            raw_injury_profiles, manual_availability, availability_roster, name_col, team_col)
        if manual_availability_problem:
            availability_resolution_warnings.append(manual_availability_problem)
        source_contract['availability'] = {
            'policy': ('manual target-week override > target-season injury report; '
                       'Ourlads status is warning-only'),
            'injury_source': ('FantasyPros injury report; healthy by default until uploaded'
                              if 'v2_fantasypros_availability' in feats
                              else 'nflverse live report (most recent designation)'),
            'resolved_profiles': len(injury_profiles),
            'manual_overrides': len(manual_availability),
            'warnings': list(availability_resolution_warnings),
        }
    else:
        source_contract['availability'] = {
            'policy': ('disabled for historical/as-of run' if historical_target
                       else 'V1 direct injury map; no V2 availability resolver'),
            'resolved_profiles': 0, 'manual_overrides': 0, 'warnings': [],
        }
    if injury_profiles:
        injury_mult = {p: v['plays_probability'] * v['workload_if_active']
                       for p, v in injury_profiles.items()}
    elif 'v2_fantasypros_availability' in feats:
        # An empty resolved profile here means exactly what it says: no
        # FantasyPros report has been uploaded for this week (or nothing on
        # it makes anyone unavailable), so nobody gets an injury discount.
        # Falling through to nflverse below would silently reintroduce the
        # source this flag exists to replace.
        injury_mult = {}
    else:
        injury_mult = (_injury_multipliers(year, week)
                       if apply_injury and not (use_v2_guard and historical_target) else {})

    # Keep the current roster's functional subposition available after a
    # player history is condensed into season totals.  In particular, a
    # roster feed may list a fullback broadly as RB but correctly expose
    # ``depth_chart_position=FB``.  Losing that field after Week 1 would let
    # the in-season V2 vacancy pool reclassify him as a core RB.  This map is
    # V2-only so V1 retains its original broad-position control path.
    current_functional_position_by_identity = pd.Series(dtype=object)
    if 'v2_preseason_rb_allocator' in feats and not stats_df.empty:
        functional_columns = [name_col, 'position', 'depth_chart_position']
        functional_columns.extend(
            column for column in ('player_id', 'gsis_id', 'pff_id')
            if column in stats_df.columns and column not in functional_columns
        )
        # A few loader combinations can preserve duplicate metadata labels
        # after a roster merge.  Functional classification needs one scalar
        # series per field, so retain the first deterministic copy here.
        functional_base = stats_df.loc[:, ~stats_df.columns.duplicated()]
        functional_source = functional_base.loc[:, [column for column in functional_columns
                                                     if column in functional_base.columns]].copy()
        if name_col in functional_source.columns:
            functional_source['_identity_key'] = player_identity_keys(functional_source, name_col)
            functional_source['_functional_position'] = classify_functional_position(functional_source)
            # Any exact current roster match is stronger than a broad
            # historical fantasy label.  If several source rows exist, the
            # final nonempty functional value is a deterministic fallback.
            functional_source['_functional_position'] = (
                functional_source['_functional_position'].astype(str).str.upper())
            functional_source = functional_source[
                functional_source['_functional_position'].isin({'QB', 'RB', 'WR', 'TE', 'FB'})]
            if not functional_source.empty:
                current_functional_position_by_identity = (
                    functional_source.drop_duplicates('_identity_key', keep='last')
                    .set_index('_identity_key')['_functional_position']
                )

    # Keep the raw histories for every defense profile.  The player-history
    # copies below exclude only clearly interrupted individual games before
    # computing a player's full-game rate, role trend, or snap expectation.
    # An injury replacement still belongs in the opponent's team-game result.
    hist_annotated = annotate_player_history_participation(
        hist, name_col, team_col, schedule_df)
    player_hist = hist_annotated[hist_annotated['_player_history_eligible']].copy()
    prior_schedule_df = load_schedule(year - 1) if not prior_played.empty else pd.DataFrame()
    prior_annotated = annotate_player_history_participation(
        prior_played, prior_name_col, prior_team_col, prior_schedule_df)
    player_prior = prior_annotated[prior_annotated['_player_history_eligible']].copy()
    prior2_schedule_df = (load_schedule(year - 2)
                          if not prior2_played.empty else pd.DataFrame())
    prior2_annotated = annotate_player_history_participation(
        prior2_played, prior2_name_col, prior2_team_col, prior2_schedule_df)
    player_prior2 = prior2_annotated[prior2_annotated['_player_history_eligible']].copy()
    source_contract['partial_game_history_filter'] = (
        'exclude only snap-confirmed QB splits, abrupt established-role exits, '
        'paired partial replacements, and severe winning-blowout rest games'
    )
    source_contract['partial_game_history_exclusions'] = {
        'current_season': int((~hist_annotated['_player_history_eligible']).sum()),
        'prior_season': int((~prior_annotated['_player_history_eligible']).sum()),
        'two_year_prior': int((~prior2_annotated['_player_history_eligible']).sum()),
    }
    current_history_exclusions = _player_history_exclusion_summary(hist_annotated, name_col)
    prior_history_exclusions = _player_history_exclusion_summary(prior_annotated, prior_name_col)

    qb1_resolution = {'selected': {}, 'by_team': {}, 'selection_required_teams': set(), 'warnings': []}
    unavailable_qbs = {
        player for player, multiplier in injury_mult.items()
        if float(multiplier) <= 0.01
    }
    if 'qb1_override' in feats:
        if cold_start:
            current_qbs = (cold_pool[cold_pool['position'].astype(str).str.upper().eq('QB')]
                           if not cold_pool.empty and 'position' in cold_pool.columns else pd.DataFrame())
            qb1_resolution = resolve_preseason_qb1s(
                current_qbs, name_col, team_col,
                player_prior, prior_name_col, prior_team_col, year,
                ourlads_qb1s=ourlads_signal['qb_starters'],
                unavailable_players=unavailable_qbs,
            )
            qb1_resolution['warnings'].extend(
                ourlads_source_contract.get('warnings', ourlads_signal['warnings']))
        else:
            current_qbs = hist[hist['position'].astype(str).str.upper().eq('QB')].copy()
            if not current_qbs.empty:
                current_qbs = (current_qbs.assign(_week=pd.to_numeric(current_qbs['week'], errors='coerce'))
                                .sort_values('_week').drop_duplicates(name_col, keep='last'))
            qb1_resolution = resolve_inseason_qb1s(
                current_qbs, name_col, team_col,
                player_hist, name_col, team_col, as_of_week, year,
                unavailable_players=unavailable_qbs,
            )
    if cold_start:
        source_contract['qb_starter_source'] = (
            ('manual_qb1_overrides_plus_healthy_local_ourlads_qb1_plus_unambiguous_prior_season_incumbents'
             if not ourlads_signal['qb_starters'].empty
             else 'manual_qb1_overrides_plus_unambiguous_prior_season_incumbents')
            if 'qb1_override' in feats else 'qb1_override_feature_disabled'
        )
        source_contract['qb1_auto_incumbent_min_prior_snap_share'] = QB1_AUTO_INCUMBENT_MIN_SHARE
        source_contract['qb1_selection_required_teams'] = sorted(qb1_resolution['selection_required_teams'])
        source_contract['qb1_override_warnings'] = list(qb1_resolution['warnings'])
        source_contract['preseason_skill_role_policy'] = (
            'same-team RB/WR/TE only: restore active-game snap role when prior games and snaps meet evidence guards'
        )
    else:
        source_contract['qb_starter_source'] = (
            'manual_qb1_overrides_plus_unambiguous_recent_full_snap_starters'
            if 'qb1_override' in feats else 'qb1_override_feature_disabled'
        )
        source_contract['qb1_selection_required_teams'] = sorted(qb1_resolution['selection_required_teams'])
        source_contract['qb1_override_warnings'] = list(qb1_resolution['warnings'])
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
        player_margins = player_hist.drop_duplicates(subset=[name_col]).set_index(name_col)[team_col] \
            .astype(str).map(all_target_margins)
        for _stat in SCRIPT_ELIGIBLE_STATS:
            script_by_stat[_stat] = _vectorized_game_script_multiplier(
                player_hist, name_col, team_col, as_of_week, schedule_df, player_margins, _stat)

    # Expected snap share for the upcoming game, and the prior season's own
    # share to scale a prior-season per-game rate against - see
    # expected_snap_share's docstring for the measured failure this fixes.
    exp_share = pd.Series(dtype=float)
    prior_share = pd.Series(dtype=float)
    if 'role_volume' in feats:
        exp_share = expected_snap_share(player_hist, name_col, team_col, as_of_week)
        prior_share = season_snap_share(player_prior, prior_name_col) if not player_prior.empty \
            else pd.Series(dtype=float)
        if exp_share.empty and not player_prior.empty:
            # Cold start: nothing this season to read a role off, so last
            # season's stands in - the same "based on last season" fallback
            # every other input already makes for week 1. Measured over the
            # WHOLE season (see season_snap_share's team_col branch), not
            # over his appearances: with no games played there is no injury
            # feed answer to "is he even the starter", and the appearance
            # reading hands a mop-up QB3 a starter's baseline off three
            # blowouts. Confirmed on real 2026 rosters - it was putting a
            # third-string quarterback at QB5 overall.
            exp_share = season_snap_share(player_prior, prior_name_col, prior_team_col)

    role_history = player_prior if cold_start else player_hist
    role_history_name = prior_name_col if cold_start else name_col
    exp_share_identity = identity_indexed_series(exp_share, role_history, role_history_name)
    prior_share_identity = identity_indexed_series(prior_share, player_prior, prior_name_col)
    # Pre-blend snapshot, kept for the RB allocator's SEASON-SCOPED
    # eligibility gate below (data.rb_role_allocator.allocate_preseason_
    # rb_roles): that gate is deliberately about "does 2025 alone clear a
    # bar", a different question from "what's the best point-estimate role"
    # the blend just below answers. Always defined (even when the blend
    # never runs) so the allocator_input construction further down never
    # references an undefined name.
    prior_active_snap_share_2025_only = prior_share_identity.copy()
    recent_active_share_2025_identity = (
        identity_indexed_series(recent_active_share(player_prior, prior_name_col), player_prior, prior_name_col)
        if cold_start and not player_prior.empty else pd.Series(dtype=float)
    )
    games_2025_identity = pd.Series(dtype=float)
    games_2024_identity = pd.Series(dtype=float)
    games_2026_identity = pd.Series(dtype=float)
    prior2_active_identity = pd.Series(dtype=float)

    # TWO-SEASON BLEND, added 2026-08-24, extended same day to run every
    # week (not just cold start) and to cover per-stat rates, not just
    # share - see PRIOR2_BLEND_DECREASE_DAMPENING and PRIOR2_DECAY_GAMES_
    # 2026 above. A cold start previously read ONLY the immediately-prior
    # season - exactly the wrong read for a down year, a season interrupted
    # by injury, or an abbreviated sample (Malik Nabers's 2025), all of
    # which shows up as a genuinely thin or depressed 2025 share with no
    # mechanism to check whether 2024 told a different story. Deliberately
    # a WEIGHTED BLEND here (continuous), not the RB allocator's separate
    # binary eligibility gate later - this changes the VALUE for every
    # RB/WR/TE (and QB, though qb1_override usually dominates for him
    # anyway), not just RB eligibility. Operates on the IDENTITY-indexed
    # series (built just above) rather than the raw name-indexed ones, so
    # it can join 2024 evidence by stable player identity and hand back a
    # Series in the exact shape every downstream reader (restoration, the
    # RB allocator, role_scale) already expects.
    have_two_season_data = 'role_volume' in feats and not player_prior.empty and not player_prior2.empty
    if have_two_season_data:
        games_2025 = (
            player_prior.groupby(prior_name_col, observed=True)['week'].nunique()
            if 'week' in player_prior.columns else pd.Series(dtype=float)
        )
        games_2025_identity = identity_indexed_series(games_2025, player_prior, prior_name_col)
        games_2024 = (
            player_prior2.groupby(prior2_name_col, observed=True)['week'].nunique()
            if 'week' in player_prior2.columns else pd.Series(dtype=float)
        )
        games_2024_identity = identity_indexed_series(games_2024, player_prior2, prior2_name_col)
        games_2026 = (
            player_hist.groupby(name_col, observed=True)['week'].nunique()
            if 'week' in player_hist.columns and not player_hist.empty else pd.Series(dtype=float)
        )
        games_2026_identity = identity_indexed_series(games_2026, player_hist, name_col)
        prior2_active_identity = identity_indexed_series(
            season_snap_share(player_prior2, prior2_name_col), player_prior2, prior2_name_col)
        prior2_whole_identity = identity_indexed_series(
            season_snap_share(player_prior2, prior2_name_col, prior2_team_col), player_prior2, prior2_name_col)

        def _blend_with_prior2(current_identity, prior2_by_identity, decay_by_2026_games=False):
            # Identity-index plumbing lives here; the actual weight rule
            # (base-by-2025-games, asymmetric dampening, optional 2026-games
            # decay, 2024-min-games gate) is prior2_blend_weight - a plain
            # module-level function, unit-tested on its own.
            current_identity = current_identity.astype(float)
            index = current_identity.index
            prior2_vals = pd.to_numeric(index.to_series().map(prior2_by_identity), errors='coerce')
            games = pd.to_numeric(index.to_series().map(games_2025_identity), errors='coerce')
            prior2_games = pd.to_numeric(index.to_series().map(games_2024_identity), errors='coerce')
            games_2026_played = (
                pd.to_numeric(index.to_series().map(games_2026_identity), errors='coerce')
                if decay_by_2026_games else None
            )
            weight = pd.Series(
                prior2_blend_weight(games.to_numpy(), prior2_games.to_numpy(),
                                    current_identity.to_numpy(), prior2_vals.to_numpy(),
                                    games_2026=(games_2026_played.to_numpy() if decay_by_2026_games else None)),
                index=index)
            both = current_identity.notna() & prior2_vals.notna()
            blended = current_identity.copy()
            blended[both] = (1.0 - weight[both]) * current_identity[both] + weight[both] * prior2_vals[both]
            # No 2025 read at all (a genuinely missing identity, not a thin
            # one) - 2024 is strictly better than the position-median
            # fallback further down, so use it outright rather than blend
            # against nothing.
            only_prior2 = current_identity.isna() & prior2_vals.notna()
            blended[only_prior2] = prior2_vals[only_prior2]
            return blended, weight

        if cold_start:
            exp_share_identity, _ = _blend_with_prior2(exp_share_identity, prior2_whole_identity)
        # prior_share is role_scale's denominator and is read every week
        # (not just cold start), so its 2024 context keeps mattering as the
        # season goes - decayed by this player's own 2026 games played.
        # exp_share above is NOT extended the same way: once real 2026
        # games exist it already IS the observed role, so blending it
        # toward a two-year-old number would corrupt real signal.
        prior_share_identity, _ = _blend_with_prior2(
            prior_share_identity, prior2_active_identity, decay_by_2026_games=True)
    prior_pre_absence = pre_absence_role_summary(player_prior, prior_name_col, prior_team_col)
    prior_pre_absence_identity = identity_indexed_series(
        prior_pre_absence.get('pre_absence_snap_share', pd.Series(dtype=float)),
        player_prior, prior_name_col)
    prior_terminal_gap_identity = identity_indexed_series(
        prior_pre_absence.get('terminal_gap_weeks', pd.Series(dtype=float)),
        player_prior, prior_name_col)
    prior_pre_absence_games_identity = identity_indexed_series(
        prior_pre_absence.get('pre_absence_games', pd.Series(dtype=float)),
        player_prior, prior_name_col)
    prior_interrupted_identity = identity_indexed_series(
        prior_pre_absence.get('interrupted_season', pd.Series(dtype=float)).astype(float)
        if not prior_pre_absence.empty else pd.Series(dtype=float), player_prior, prior_name_col)
    # Team continuity is a player-role input, so it needs the team that
    # actually played the final historical game—not the latest roster team
    # that a later merge may have written onto every old row after a trade.
    # Defense profiles and snap denominators already use ``game_team``; keep
    # this small identity-indexed companion for the cold-start role/restoration
    # decision as well.
    prior_last_game_team_by_identity = pd.Series(dtype=object)
    if not player_prior.empty and prior_name_col in player_prior.columns:
        prior_team_context = pd.DataFrame({
            '_identity_key': player_identity_keys(player_prior, prior_name_col),
            '_game_team': _historical_game_team(player_prior, prior_team_col),
            '_week': pd.to_numeric(player_prior.get('week', pd.Series(np.nan, index=player_prior.index)),
                                   errors='coerce'),
        }, index=player_prior.index)
        prior_team_context = prior_team_context[
            prior_team_context['_identity_key'].astype(str).ne('')
            & prior_team_context['_game_team'].astype(str).ne('')
        ]
        if not prior_team_context.empty:
            prior_last_game_team_by_identity = (
                prior_team_context.sort_values('_week', kind='stable')
                .groupby('_identity_key', observed=True)['_game_team'].last()
            )
    rb_role_segments, rb_teammate_context = pd.DataFrame(), pd.DataFrame()
    if cold_start and 'v2_preseason_rb_allocator' in feats and not prior_played.empty:
        try:
            # This consumes the raw prior season rather than the player-rate
            # filtered copy.  A replacement's real team-game work belongs in
            # the segment context, while the player-rate filter continues to
            # govern only rate averages.
            rb_role_segments, rb_teammate_context = analyze_rb_role_segments(
                prior_played, player_col=prior_name_col, team_col=prior_team_col)
            source_contract['rb_role_segments'] = {
                'status': 'prior-season immutable team-week role segments',
                'clear_interrupted_returners': int(
                    rb_role_segments.get('interrupted_season', pd.Series(dtype=bool)).fillna(False).sum()),
                'teammate_context_rows': int(len(rb_teammate_context)),
            }
        except Exception as exc:
            # A segment is extra evidence, never a reason to break the full
            # week-one projection.  The visible contract records why it was
            # unavailable instead of silently treating a failed parser as a
            # football conclusion.
            source_contract['rb_role_segments'] = {
                'status': 'unavailable', 'warning': f'{type(exc).__name__}: {exc}',
            }
    else:
        source_contract['rb_role_segments'] = {
            'status': 'not applicable outside a V2 preseason cold start',
        }
    rb_team_capacities = (derive_preseason_rb_capacities(prior_played, prior_team_col)
                          if cold_start and 'v2_preseason_rb_allocator' in feats else pd.DataFrame())

    all_rows, explanations = [], {}
    rb_allocation_ledger = []
    for pos in DRAFTABLE_POSITIONS:
        stats = OFFENSE_PROJECTION_STATS[pos]
        if cold_start:
            pool_position = (cold_pool.get('projection_position', cold_pool.get('position', pd.Series('', index=cold_pool.index)))
                             .astype(str).str.upper())
            pool_pos = cold_pool[pool_position == pos] if not cold_pool.empty else cold_pool
            if pool_pos.empty:
                continue
            carry_cols = [name_col, team_col] + [c for c in (
                'player_id', 'gsis_id', 'pff_id', 'depth_chart_position', 'status',
                'draft_number', 'is_rookie_flag', 'years_exp', 'ourlads_position',
                'functional_position', 'projection_position') if c in pool_pos.columns]
            cur = (pool_pos[carry_cols].rename(columns={team_col: 'Team'})
                   .drop_duplicates(subset=[name_col]))
            cur['Games'] = 0
            for stat in stats:
                cur[stat] = 0.0
            cur['_identity_key'] = player_identity_keys(cur, name_col)
        else:
            cur = attach_player_identity(
                _season_totals(player_hist, name_col, team_col, pos, stats), player_hist, name_col)
            if cur.empty:
                continue
        # Keep a non-display functional-position flag through the assembled
        # frame.  In a live preseason cold start this is the roster/Ourlads
        # classification (so an FB cannot masquerade as a broad RB); in an
        # in-season player-total frame the selected fantasy position remains
        # the conservative fallback because a current roster depth code may
        # no longer describe an earlier played game.
        cur['_functional_position'] = cur.get(
            'functional_position', cur.get('projection_position', pd.Series(pos, index=cur.index))
        ).fillna(pos).astype(str).str.upper()
        if (not cold_start and 'v2_preseason_rb_allocator' in feats
                and not current_functional_position_by_identity.empty):
            mapped_functional = cur['_identity_key'].map(current_functional_position_by_identity)
            cur['_functional_position'] = mapped_functional.fillna(cur['_functional_position']).astype(str).str.upper()
        prior = (attach_player_identity(
                    _season_totals(player_prior, prior_name_col, prior_team_col, pos, stats),
                    player_prior, prior_name_col)
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
            prior_rates['_identity_key'] = prior['_identity_key'].to_numpy(dtype=object)
        # A current fantasy roster can call a player RB while the prior
        # season correctly records him as FB.  Preserve that player's *own*
        # low-volume historical rate for a live preseason display, without
        # adding FB production to the ordinary RB population baseline or
        # core-RB capacity.  If there is no own FB history, the later blend
        # deliberately uses zero rather than a generic RB fallback.
        prior_fullbacks = pd.DataFrame()
        fullback_prior_rates = pd.DataFrame()
        if (pos == 'RB' and 'v2_preseason_rb_allocator' in feats
                and not prior_stats.empty):
            prior_fullbacks = attach_player_identity(
                _season_totals(player_prior, prior_name_col, prior_team_col, 'FB', stats),
                player_prior, prior_name_col)
            if not prior_fullbacks.empty:
                fullback_prior_rates = pd.DataFrame({
                    stat: prior_fullbacks[stat] / prior_fullbacks['Games'].replace(0, 1)
                    for stat in stats
                })
                fullback_prior_rates['_identity_key'] = prior_fullbacks['_identity_key'].to_numpy(dtype=object)
        prior_role_reference = (
            pd.concat([prior, prior_fullbacks], ignore_index=True)
            .drop_duplicates('_identity_key', keep='last')
            if not prior_fullbacks.empty else prior
        )
        if cold_start and not prior_role_reference.empty and not prior_last_game_team_by_identity.empty:
            prior_role_reference = prior_role_reference.copy()
            prior_role_reference['Team'] = prior_role_reference['_identity_key'].map(
                prior_last_game_team_by_identity).fillna(prior_role_reference['Team'])
        older = (attach_player_identity(
                    _season_totals(player_prior2, prior2_name_col, prior2_team_col, pos, stats),
                    player_prior2, prior2_name_col)
                 if not prior2_stats.empty else pd.DataFrame())
        older_rates = (pd.DataFrame({s: older[s] / older['Games'].replace(0, 1) for s in stats})
                       if not older.empty else pd.DataFrame())
        if not older.empty:
            older_rates['_identity_key'] = older['_identity_key'].to_numpy(dtype=object)

        if cold_start and prior_max_week is not None and not pd.isna(prior_max_week):
            # player_hist (THIS season) is empty at cold start, so the normal
            # path below would return an empty Series and every player would
            # fall through to the same 0.5 neutral default - which is exactly
            # what was happening: role_confidence, and therefore
            # player_distribution's width_scale, was identically 1.0 for
            # EVERY player at Week 1 of a season, making the range-of-
            # outcomes chart's relative shape indistinguishable between
            # players (only its point-scale differed) even though the chart
            # itself was rendering correctly off genuinely different inputs.
            # Substitute last season's own last-3-games snap share, the same
            # "prior season stands in for missing current-season signal"
            # pattern already used for matchup_matrix/roles just below - and
            # re-key through clean_name_exact for the same reason documented
            # there (gotcha #35): a prior-season frame's name column is not
            # guaranteed to even be called the same thing as this year's.
            prior_role_conf = _role_confidence(
                player_prior, prior_name_col, int(prior_max_week) + 1, pos, pff_rec)
            prior_role_by_key = pd.Series(
                prior_role_conf.to_numpy(),
                index=clean_name_exact(pd.Series(prior_role_conf.index)))
            cur['role_confidence'] = clean_name_exact(cur[name_col]).map(prior_role_by_key).to_numpy()
            role_conf_detail = _role_confidence_detail(
                player_prior, prior_name_col, int(prior_max_week) + 1, pos, pff_rec)
            role_conf_detail = role_conf_detail.set_index(
                clean_name_exact(pd.Series(role_conf_detail.index)))
            keyed_names = clean_name_exact(cur[name_col])
            for col in ('recent_snap_pct', 'games_sampled', 'route_rate', 'method'):
                cur[f'_role_confidence_{col}'] = keyed_names.map(
                    role_conf_detail[col] if col in role_conf_detail.columns else pd.Series(dtype=object)
                ).to_numpy()
        else:
            role_conf = _role_confidence(player_hist, name_col, as_of_week, pos, pff_rec)
            cur = cur.merge(role_conf.rename('role_confidence'), left_on=name_col, right_index=True, how='left')
            role_conf_detail = _role_confidence_detail(player_hist, name_col, as_of_week, pos, pff_rec)
            cur = cur.merge(
                role_conf_detail.add_prefix('_role_confidence_'),
                left_on=name_col, right_index=True, how='left')
        cur['role_confidence'] = cur['role_confidence'].fillna(0.5)
        for col in ('_role_confidence_recent_snap_pct', '_role_confidence_games_sampled',
                    '_role_confidence_route_rate'):
            if col not in cur.columns:
                cur[col] = np.nan
        if '_role_confidence_method' not in cur.columns:
            cur['_role_confidence_method'] = ''
        cur['_role_confidence_method'] = cur['_role_confidence_method'].fillna(
            'no recent-snap history — default 0.5 used')
        role_profiles = pd.DataFrame()
        role_change = pd.Series(dtype=float)
        if not cold_start:
            pos_history = player_hist[player_hist['position'].astype(str).str.upper() == pos]
            role_profiles = build_continuous_role_profiles(player_hist, name_col, team_col, pos)
            role_change = confirmed_role_change_signal(pos_history, name_col, team_col, pos, as_of_week)
        # Overlay only actual time-valid PFF alignment fields.  The free-data
        # continuous role profile remains the source for ADOT/share evidence;
        # PFF contributes slot/non-slot information without pretending its
        # non-slot TE measure is an inline measure or changing a stat line.
        pff_alignment_for_cur = pd.DataFrame(index=cur.index)
        if 'v2_pff_alignment_matchup' in feats:
            alignment_rows = []
            pff_ids = (cur.get('pff_id', pd.Series('', index=cur.index))
                       if 'pff_id' in cur.columns else pd.Series('', index=cur.index))
            for row_index, player, team, pff_id in zip(
                    cur.index, cur[name_col], cur['Team'], pff_ids):
                profile = lookup_alignment_profile(
                    pff_alignment_profiles, player_id=pff_id,
                    player=str(player), team=str(team), position=pos,
                )
                if not prior2_alignment_profiles.empty:
                    prior2_profile = lookup_alignment_profile(
                        prior2_alignment_profiles, player_id=pff_id,
                        player=str(player), team=str(team), position=pos,
                    )
                    profile = blend_alignment_profile_toward_prior2(profile, prior2_profile)
                alignment_rows.append(profile)
            if alignment_rows:
                pff_alignment_for_cur = pd.DataFrame(alignment_rows, index=cur.index)
        # Same overlay, man/zone route share instead of slot/wide/inline -
        # see pff_scheme_profiles' construction above for why this is empty
        # (not an error) at cold start.
        pff_scheme_for_cur = pd.DataFrame(index=cur.index)
        if 'v2_pff_alignment_matchup' in feats and not pff_scheme_profiles.empty:
            scheme_rows = []
            pff_ids = (cur.get('pff_id', pd.Series('', index=cur.index))
                       if 'pff_id' in cur.columns else pd.Series('', index=cur.index))
            for player, team, pff_id in zip(cur[name_col], cur['Team'], pff_ids):
                scheme_rows.append(lookup_scheme_profile(
                    pff_scheme_profiles, player_id=pff_id,
                    player=str(player), team=str(team), position=pos,
                ))
            if scheme_rows:
                pff_scheme_for_cur = pd.DataFrame(scheme_rows, index=cur.index)
        cur['role_change_confidence'] = cur[name_col].map(role_change).fillna(0.0)
        if not current_history_exclusions.empty:
            cur['_current_history_excluded_games'] = cur[name_col].map(
                current_history_exclusions['excluded_games']).fillna(0.0)
            cur['_current_history_exclusion_reasons'] = cur[name_col].map(
                current_history_exclusions['excluded_reasons']).fillna('')
        else:
            cur['_current_history_excluded_games'] = 0.0
            cur['_current_history_exclusion_reasons'] = ''
        if not prior_history_exclusions.empty:
            prior_exclusion_by_key = pd.DataFrame({
                '_key': clean_name_exact(pd.Series(prior_history_exclusions.index)),
                'excluded_games': prior_history_exclusions['excluded_games'].to_numpy(),
                'excluded_reasons': prior_history_exclusions['excluded_reasons'].to_numpy(),
            }).drop_duplicates('_key', keep='last').set_index('_key')
            keys = clean_name_exact(cur[name_col])
            cur['_prior_history_excluded_games'] = keys.map(
                prior_exclusion_by_key['excluded_games']).fillna(0.0).to_numpy(dtype=float)
            cur['_prior_history_exclusion_reasons'] = keys.map(
                prior_exclusion_by_key['excluded_reasons']).fillna('').to_numpy(dtype=object)
        else:
            cur['_prior_history_excluded_games'] = 0.0
            cur['_prior_history_exclusion_reasons'] = ''

        cur['Opponent'] = cur['Team'].map(opponents)
        cur = cur[cur['Opponent'].notna()].copy()  # bye-week teams drop out entirely
        if cur.empty:
            continue
        # .astype(str) AFTER the notna() filter above, never before - Team
        # can be categorical upstream (HANDOFF.md gotcha #10) and casting a
        # real NaN to str first turns it into the literal string "nan",
        # which then survives a .notna() bye-week filter it should have
        # been dropped by. Needed downstream for a clean .map() against
        # matchup_matrix's plain-str index (team-game defense profile).
        cur['Opponent'] = cur['Opponent'].astype(str)
        cur['target_margin'] = cur['Team'].map(target_margins)

        # Make the time-valid, offense-side PFF defense evidence inspectable
        # beside a player's slot/non-slot mix.  These are candidate residuals
        # only: the helper's applied multiplier is deliberately 1.0 until a
        # predeclared out-of-sample test authorizes scoring use.
        alignment_defense_previews = {}
        if 'v2_pff_alignment_matchup' in feats:
            pff_alignment_for_cur = pff_alignment_for_cur.reindex(cur.index)
            preview_columns = {
                'targets': [], 'receptions': [], 'yards': [],
            }
            preview_reasons = []
            preview_available = []
            preview_blend_modes = []
            # Per-alignment raw shrunk_allowed_ratio, one series per
            # (stat, alignment) pair - added 2026-08-26 so the "Alignment
            # mix" UI section can show its own worked calculation (per-
            # alignment multiplier x this player's own per-alignment rate)
            # instead of only the already-blended candidate_multiplier.
            preview_ratio_columns = {
                f'{stat}_{align}': []
                for stat in preview_columns for align in ('slot', 'wide', 'inline', 'non_slot')
            }
            for row_index, opponent in zip(cur.index, cur['Opponent']):
                slot_rate = (pff_alignment_for_cur.at[row_index, 'slot_alignment_rate']
                             if 'slot_alignment_rate' in pff_alignment_for_cur.columns else None)
                # Real 3-way slot/wide/inline blend when this player's own
                # wide/inline rates are on the profile (added 2026-08-24) -
                # alignment_defense_residual_multiplier falls back to the
                # original slot/non-slot blend on its own whenever either is
                # missing or this defense lacks wide/inline comparison
                # evidence, so passing them here is always safe.
                wide_rate = (pff_alignment_for_cur.at[row_index, 'wide_alignment_rate']
                             if 'wide_alignment_rate' in pff_alignment_for_cur.columns else None)
                inline_rate = (pff_alignment_for_cur.at[row_index, 'inline_alignment_rate']
                               if 'inline_alignment_rate' in pff_alignment_for_cur.columns else None)
                previews = {
                    stat: alignment_defense_residual_multiplier(
                        pff_alignment_defense_profiles,
                        defense_team=opponent,
                        position=pos,
                        player_slot_rate=slot_rate,
                        player_wide_rate=wide_rate,
                        player_inline_rate=inline_rate,
                        stat=stat,
                    )
                    for stat in preview_columns
                }
                alignment_defense_previews[row_index] = previews
                for stat in preview_columns:
                    preview_columns[stat].append(previews[stat].get('candidate_multiplier', 1.0))
                    for align, profile_key in (
                        ('slot', 'slot_profile'), ('wide', 'wide_profile'),
                        ('inline', 'inline_profile'), ('non_slot', 'non_slot_profile'),
                    ):
                        profile = previews[stat].get(profile_key) or {}
                        preview_ratio_columns[f'{stat}_{align}'].append(profile.get('shrunk_allowed_ratio'))
                preview_reasons.append(previews['targets'].get('reason', ''))
                preview_available.append(bool(previews['targets'].get('candidate_available', False)))
                preview_blend_modes.append(previews['targets'].get('blend_mode', 'slot_non_slot'))
            for stat, values in preview_columns.items():
                pff_alignment_for_cur[f'alignment_defense_{stat}_candidate_multiplier'] = values
            for key, values in preview_ratio_columns.items():
                pff_alignment_for_cur[f'alignment_defense_{key}_ratio'] = values
            pff_alignment_for_cur['alignment_defense_candidate_available'] = preview_available
            pff_alignment_for_cur['alignment_defense_reason'] = preview_reasons
            pff_alignment_for_cur['alignment_defense_blend_mode'] = preview_blend_modes
            # True only for WR/TE (RB/QB remain audit-only, matching
            # ALIGNMENT_DEFENSE_SUPPORTED_POSITIONS - the candidate is
            # already forced to 1.0/unavailable for them regardless) and
            # only meaningful when this whole block ran, i.e. the caller
            # explicitly requested 'v2_pff_alignment_matchup' - rejected
            # 2026-08-24, see DEFAULT_FEATURES's own comment, so this is
            # never true for the DEFAULT_FEATURES board; true for a V2 board
            # only as the user's own diagnostic reactivation (see
            # V2_EXPERIMENTAL_FEATURES's comment).
            pff_alignment_for_cur['alignment_defense_scoring_active'] = pos in ALIGNMENT_DEFENSE_SUPPORTED_POSITIONS

        # Same defense-evidence preview, man/zone instead of slot/non-slot.
        scheme_defense_previews = {}
        if 'v2_pff_alignment_matchup' in feats:
            pff_scheme_for_cur = pff_scheme_for_cur.reindex(cur.index)
            scheme_preview_columns = {'targets': [], 'receptions': [], 'yards': []}
            scheme_preview_reasons = []
            scheme_preview_available = []
            for row_index, opponent in zip(cur.index, cur['Opponent']):
                man_rate = (pff_scheme_for_cur.at[row_index, 'man_route_share']
                           if 'man_route_share' in pff_scheme_for_cur.columns else None)
                previews = {
                    stat: scheme_defense_residual_multiplier(
                        pff_scheme_defense_profiles,
                        defense_team=opponent,
                        position=pos,
                        player_man_rate=man_rate,
                        stat=stat,
                    )
                    for stat in scheme_preview_columns
                }
                scheme_defense_previews[row_index] = previews
                for stat in scheme_preview_columns:
                    scheme_preview_columns[stat].append(previews[stat].get('candidate_multiplier', 1.0))
                scheme_preview_reasons.append(previews['targets'].get('reason', ''))
                scheme_preview_available.append(bool(previews['targets'].get('candidate_available', False)))
            for stat, values in scheme_preview_columns.items():
                pff_scheme_for_cur[f'scheme_defense_{stat}_candidate_multiplier'] = values
            pff_scheme_for_cur['scheme_defense_candidate_available'] = scheme_preview_available
            pff_scheme_for_cur['scheme_defense_reason'] = scheme_preview_reasons
            pff_scheme_for_cur['scheme_defense_scoring_active'] = False

        # Recency + matchup-strength + rematch weighted own-history rate,
        # and the SAME defense ratings reused below to price the upcoming
        # matchup - see build_team_game_quality_adjusted_matchup's docstring and the
        # module docstring's "THIS SEASON'S OWN GAME LOG..." section.
        role_tables, role_sizes, player_roles, continuous_role_weights = {}, {}, {}, pd.DataFrame()
        defense_current_evidence = pd.Series(dtype=float)
        defense_prior_evidence = pd.Series(dtype=float)
        if cold_start:
            # No games THIS season to build a matchup rating from - fall
            # back to PRIOR season's full year.  It keeps a modest late-season
            # preference but mostly reflects the full season; an offseason is
            # not a continuation of last year's final few game scripts.
            # The offense side needs no equivalent override: _blended_rate's
            # own w_current=0 (cur_games=0 here) already lands entirely on
            # prior_rate below without any extra machinery.
            prior_pos_rows = (prior_played[prior_played['position'].astype(str).str.upper() == pos]
                              if not prior_played.empty else prior_played)
            defense_prior_evidence = _defense_game_evidence(
                prior_pos_rows, game_universe=prior_played, team_col=prior_team_col)
            anchor_week = (prior_max_week + 1) if (prior_max_week is not None
                                                   and not pd.isna(prior_max_week)) else None
            matchup_matrix = (
                build_team_game_quality_adjusted_matchup(
                    prior_pos_rows, prior_team_col, stats, anchor_week,
                    recency_floor=PRIOR_SEASON_DEFENSE_RECENCY_FLOOR,
                    game_universe=prior_played, plays=prior_plays,
                ) if anchor_week is not None else pd.DataFrame()
            )
            # Deep Dive: no CURRENT-season games exist yet at cold start, so
            # that side of the season selector is honestly empty - but last
            # season's full log is exactly what this branch already loaded
            # to build the projection itself, so build it here too rather
            # than showing a blank tab for the entire preseason/Week 1
            # window. Re-keyed through clean_name_exact, same gotcha as the
            # role lookup just below (#35): a prior-season frame's name
            # column is not guaranteed to even be spelled the same way.
            player_game_log_current = np.full(len(cur), np.nan, dtype=object)
            opponent_defense_log_current = np.full(len(cur), np.nan, dtype=object)
            prior_annotated_pos = (prior_annotated[prior_annotated['position'].astype(str).str.upper() == pos]
                                   if not prior_annotated.empty else prior_annotated)
            game_log_by_name_prior = _build_player_stat_game_log(
                prior_annotated_pos, prior_name_col, stats, matchup_matrix,
                schedule_df=prior_schedule_df, team_col=prior_team_col)
            game_log_by_key_prior = {clean_name_exact(pd.Series([p])).iloc[0]: g
                                     for p, g in game_log_by_name_prior.items()}
            player_game_log_prior = clean_name_exact(cur[name_col]).map(game_log_by_key_prior).to_numpy()
            defense_log_by_team_prior = _build_defense_weekly_log(
                prior_pos_rows, prior_team_col, stats, prior_played,
                anchor_week if anchor_week is not None else as_of_week)
            opponent_defense_log_prior = cur['Opponent'].astype(str).map(defense_log_by_team_prior).to_numpy()
            # Two-years-back ("prior2") Deep Dive log, same construction as
            # the prior-season pair just above, minus a quality-adjusted
            # matchup matrix (not built for prior2 - only its raw per-game
            # rate feeds the model, never a matchup multiplier) - so its
            # Defense-adj column intentionally reads the same as its Raw
            # column. Honestly empty (empty dict/list per player) when
            # prior2_played has no data, same convention as the current-
            # season pair at cold start.
            prior2_pos_rows = (prior2_played[prior2_played['position'].astype(str).str.upper() == pos]
                               if not prior2_played.empty else prior2_played)
            prior2_annotated_pos = (
                prior2_annotated[prior2_annotated['position'].astype(str).str.upper() == pos]
                if not prior2_annotated.empty else prior2_annotated)
            game_log_by_name_prior2 = _build_player_stat_game_log(
                prior2_annotated_pos, prior2_name_col, stats, pd.DataFrame(),
                schedule_df=prior2_schedule_df, team_col=prior2_team_col)
            game_log_by_key_prior2 = {clean_name_exact(pd.Series([p])).iloc[0]: g
                                      for p, g in game_log_by_name_prior2.items()}
            player_game_log_prior2 = clean_name_exact(cur[name_col]).map(game_log_by_key_prior2).to_numpy()
            defense_log_by_team_prior2 = _build_defense_weekly_log(
                prior2_pos_rows, prior2_team_col, stats, prior2_played, as_of_week)
            opponent_defense_log_prior2 = cur['Opponent'].astype(str).map(defense_log_by_team_prior2).to_numpy()
            # Cold start reads ROLES off last season too, and keys them by
            # the player's own name in THAT season's frame - which is why the
            # lookup below re-keys through clean_name_exact rather than
            # assuming both seasons spell (or even name the column) the same
            # way (gotcha #35).
            if ('role_matchup' in feats or 'v2_continuous_roles' in feats) and anchor_week is not None and not prior_pos_rows.empty:
                player_prior_pos_rows = player_prior[
                    player_prior['position'].astype(str).str.upper() == pos]
                prior_roles = build_player_roles(player_prior_pos_rows, prior_name_col, pos)
                role_tables, role_sizes = build_role_matchup(
                    prior_pos_rows, prior_name_col, prior_team_col, stats, anchor_week, prior_roles,
                    recency_floor=PRIOR_SEASON_DEFENSE_RECENCY_FLOOR, plays=prior_plays)
                roles_by_key = {clean_name_exact(pd.Series([p])).iloc[0]: r
                                for p, r in prior_roles.items()}
                player_roles = {p: roles_by_key.get(clean_name_exact(pd.Series([p])).iloc[0])
                                for p in cur[name_col]}
                if 'v2_continuous_roles' in feats:
                    prior_weights = build_continuous_role_weights(
                        player_prior_pos_rows, prior_name_col, pos)
                    keyed_weights = prior_weights.copy()
                    keyed_weights.index = clean_name_exact(pd.Series(keyed_weights.index))
                    continuous_role_weights = pd.DataFrame(
                        [keyed_weights.loc[clean_name_exact(pd.Series([p])).iloc[0]].to_dict()
                         if clean_name_exact(pd.Series([p])).iloc[0] in keyed_weights.index else {}
                         for p in cur[name_col]], index=cur.index).fillna(0.0)
        else:
            pos_rows = hist[hist['position'].astype(str).str.upper() == pos]
            player_pos_rows = player_hist[
                player_hist['position'].astype(str).str.upper() == pos]
            defense_current_evidence = _defense_game_evidence(
                pos_rows, game_universe=hist, team_col=team_col)
            upcoming_opponent_map = dict(zip(cur[name_col], cur['Opponent']))
            matchup_matrix = build_team_game_quality_adjusted_matchup(
                pos_rows, team_col, stats, as_of_week, game_universe=hist, plays=current_plays)
            # prior_pos_rows/prior_anchor/prior_matrix are computed
            # unconditionally (not just under v2_defense_prior) because the
            # Deep Dive's prior-season selector needs them regardless of
            # whether that feature flag lets them affect SCORING - only the
            # blend into matchup_matrix below stays feature-gated.
            prior_pos_rows = (prior_played[prior_played['position'].astype(str).str.upper() == pos]
                              if not prior_played.empty else prior_played)
            defense_prior_evidence = _defense_game_evidence(
                prior_pos_rows, game_universe=prior_played, team_col=prior_team_col)
            prior_anchor = (prior_max_week + 1 if prior_max_week is not None else None)
            if prior_anchor is None and not prior_pos_rows.empty:
                max_prior = pd.to_numeric(prior_pos_rows['week'], errors='coerce').max()
                prior_anchor = int(max_prior) + 1 if pd.notna(max_prior) else None
            prior_matrix = (
                build_team_game_quality_adjusted_matchup(
                    prior_pos_rows, prior_team_col, stats, prior_anchor,
                    recency_floor=PRIOR_SEASON_DEFENSE_RECENCY_FLOOR,
                    game_universe=prior_played, plays=prior_plays,
                ) if prior_anchor is not None else pd.DataFrame()
            )
            if 'v2_defense_prior' in feats:
                prior_games_ = (DEFENSE_PRIOR_GAMES_OVERRIDE
                                if ('v2_defense_prior_games_override' in feats
                                    and DEFENSE_PRIOR_GAMES_OVERRIDE is not None)
                                else DEFENSE_PRIOR_GAMES)
                matchup_matrix = blend_defense_prior(
                    matchup_matrix, prior_matrix, _defense_game_evidence(
                        pos_rows, game_universe=hist, team_col=team_col),
                    prior_games=prior_games_)

            # Deep Dive per-game/per-week detail for BOTH seasons, threaded
            # through to the decomposition dialog via explanations[...]
            # below. Current-season frames are cheap: they reuse data
            # _weighted_player_rates and _position_team_games already
            # compute for this exact position group once per board build
            # (not once per player) - "stop discarding the intermediate
            # frame", not new computation. The prior-season pair costs one
            # more of the same cheap calls, using prior_matrix (now always
            # built above) instead of the current-season matchup_matrix, so
            # the season selector always has a real last-season log to show
            # even mid-season, not only at cold start.
            game_log_pos = hist_annotated[hist_annotated['position'].astype(str).str.upper() == pos]
            game_log_by_name = _build_player_stat_game_log(
                game_log_pos, name_col, stats, matchup_matrix,
                schedule_df=schedule_df, team_col=team_col)
            player_game_log_current = cur[name_col].map(game_log_by_name).to_numpy()

            prior_annotated_pos = (prior_annotated[prior_annotated['position'].astype(str).str.upper() == pos]
                                   if not prior_annotated.empty else prior_annotated)
            game_log_by_name_prior = _build_player_stat_game_log(
                prior_annotated_pos, prior_name_col, stats, prior_matrix,
                schedule_df=prior_schedule_df, team_col=prior_team_col)
            game_log_by_key_prior = {clean_name_exact(pd.Series([p])).iloc[0]: g
                                     for p, g in game_log_by_name_prior.items()}
            player_game_log_prior = clean_name_exact(cur[name_col]).map(game_log_by_key_prior).to_numpy()

            defense_log_by_team = _build_defense_weekly_log(pos_rows, team_col, stats, hist, as_of_week)
            opponent_defense_log_current = cur['Opponent'].astype(str).map(defense_log_by_team).to_numpy()
            defense_log_by_team_prior = _build_defense_weekly_log(
                prior_pos_rows, prior_team_col, stats, prior_played,
                prior_anchor if prior_anchor is not None else as_of_week)
            opponent_defense_log_prior = cur['Opponent'].astype(str).map(defense_log_by_team_prior).to_numpy()

            # Two-years-back ("prior2") Deep Dive log - same convention as
            # the cold-start branch's own prior2 log just above: no quality-
            # adjusted matchup matrix, so Defense-adj reads the same as Raw.
            prior2_pos_rows = (prior2_played[prior2_played['position'].astype(str).str.upper() == pos]
                               if not prior2_played.empty else prior2_played)
            prior2_annotated_pos = (
                prior2_annotated[prior2_annotated['position'].astype(str).str.upper() == pos]
                if not prior2_annotated.empty else prior2_annotated)
            game_log_by_name_prior2 = _build_player_stat_game_log(
                prior2_annotated_pos, prior2_name_col, stats, pd.DataFrame(),
                schedule_df=prior2_schedule_df, team_col=prior2_team_col)
            game_log_by_key_prior2 = {clean_name_exact(pd.Series([p])).iloc[0]: g
                                      for p, g in game_log_by_name_prior2.items()}
            player_game_log_prior2 = clean_name_exact(cur[name_col]).map(game_log_by_key_prior2).to_numpy()
            defense_log_by_team_prior2 = _build_defense_weekly_log(
                prior2_pos_rows, prior2_team_col, stats, prior2_played, as_of_week)
            opponent_defense_log_prior2 = cur['Opponent'].astype(str).map(defense_log_by_team_prior2).to_numpy()

            weighted_rates, weighted_totals = _weighted_player_rates(
                player_pos_rows, name_col, stats, as_of_week, matchup_matrix, upcoming_opponent_map)
            if 'role_matchup' in feats or 'v2_continuous_roles' in feats:
                player_roles = build_player_roles(player_pos_rows, name_col, pos)
                role_tables, role_sizes = build_role_matchup(
                    pos_rows, name_col, team_col, stats, as_of_week, player_roles, plays=current_plays)
                if 'v2_continuous_roles' in feats:
                    continuous_role_weights = build_continuous_role_weights(player_pos_rows, name_col, pos)
                    continuous_role_weights = continuous_role_weights.reindex(cur[name_col]).fillna(0.0)
        if cold_start:
            # No current-season rows to weight at all - every player's
            # in_season_rate below falls through to its own season_avg
            # fallback (0/0=0 for a cold-start cur), which is fine: cur_games
            # is 0 for everyone here too, so _blended_rate ignores it anyway
            # and lands entirely on prior_rate.
            weighted_rates, weighted_totals = pd.DataFrame(), pd.DataFrame()
            # player_game_log_current/opponent_defense_log_current were
            # already set to the honest "no games this season" NaN stub
            # above, alongside the real prior-season pair - same fail-to-
            # empty discipline as weighted_rates above, not fabricated.

        # Snap-share arrays for this position's frame. `share_ref` is the
        # position's own snap-weighted game count, which turns the position
        # baseline from "per game played" into "per FULL-SNAP game" - the
        # unit a player's expected share can then be multiplied against.
        use_role_volume = 'role_volume' in feats and not exp_share.empty
        name_keys_rv = clean_name_exact(cur[name_col])
        identity_keys_rv = cur.get('_identity_key', player_identity_keys(cur, name_col)).astype(str)
        team_keys_rv = cur['Team'].astype(str).str.strip().str.upper()
        qb1_workload_override = np.zeros(len(cur), dtype=bool)
        qb1_selection_required = np.zeros(len(cur), dtype=bool)
        qb1_workload_source = np.full(len(cur), 'Not applicable', dtype=object)
        qb_projected_starter = np.ones(len(cur), dtype=bool)
        qb_nonstarter_volume_factor = np.ones(len(cur), dtype=float)
        returning_role_restored = np.zeros(len(cur), dtype=bool)
        returning_role_reason = np.full(len(cur), 'none', dtype=object)
        preseason_role_source = np.full(len(cur), 'Not applicable', dtype=object)
        ourlads_role_floor_applied = np.zeros(len(cur), dtype=bool)
        ourlads_role_rank = np.full(len(cur), np.nan)
        ourlads_role_floor = np.full(len(cur), np.nan)
        ourlads_role_label = np.full(len(cur), '', dtype=object)
        ourlads_audit = ourlads_player_audit_arrays(
            ourlads_signal.get('matches', pd.DataFrame()), team_keys_rv,
            identity_keys_rv, cur[name_col].to_numpy(dtype=object))
        ourlads_source_status = ourlads_audit['source_status']
        ourlads_source_status_warning = ourlads_audit['source_status_warning']
        ourlads_identity_match_method = ourlads_audit['identity_match_method']
        ourlads_identity_match_confidence = ourlads_audit['identity_match_confidence']
        ourlads_identity_match_warning = ourlads_audit['identity_match_warning']
        ourlads_source_name = ourlads_audit['source_name']
        ourlads_matched_identity_key = ourlads_audit['roster_identity_key']
        rb_allocator_applied = np.zeros(len(cur), dtype=bool)
        rb_core = np.zeros(len(cur), dtype=bool)
        rb_carry_allocation = np.full(len(cur), np.nan)
        rb_target_allocation = np.full(len(cur), np.nan)
        rb_snap_capacity = np.full(len(cur), np.nan)
        rb_carry_capacity = np.full(len(cur), np.nan)
        rb_target_capacity = np.full(len(cur), np.nan)
        rb_other_snap = np.full(len(cur), np.nan)
        rb_other_carries = np.full(len(cur), np.nan)
        rb_other_targets = np.full(len(cur), np.nan)
        rb_allocation_source = np.full(len(cur), '', dtype=object)
        rb_allocation_eligibility_reason = np.full(len(cur), '', dtype=object)
        rb_established_incumbent_backstop = np.zeros(len(cur), dtype=bool)
        rb_carry_rate_scale = np.ones(len(cur), dtype=float)
        rb_target_rate_scale = np.ones(len(cur), dtype=float)
        rb_segment_status = np.full(len(cur), '', dtype=object)
        rb_segment_pre_absence_snap_share = np.full(len(cur), np.nan)
        rb_segment_gap_games = np.full(len(cur), np.nan)
        rb_segment_return_snap_share = np.full(len(cur), np.nan)
        rb_interrupted_incumbent_credit = np.zeros(len(cur), dtype=float)
        rb_shared_healthy_lead_score = np.zeros(len(cur), dtype=float)
        rb_replacement_only_downweight = np.zeros(len(cur), dtype=float)
        if pos == 'QB':
            if 'qb1_override' in feats:
                selected_qb1s = qb1_resolution['selected']
                required_teams = qb1_resolution['selection_required_teams']
                qb1_workload_override = np.asarray([
                    (team, key) in selected_qb1s
                    for team, key in zip(team_keys_rv, name_keys_rv)
                ], dtype=bool)
                qb1_selection_required = np.asarray([
                    team in required_teams for team in team_keys_rv
                ], dtype=bool)
                qb1_workload_source = np.asarray([
                    ('Manual QB1 selection' if selected_qb1s.get((team, key)) == 'manual_override'
                     else 'Imported Ourlads depth-chart QB1' if selected_qb1s.get((team, key)) == 'ourlads_depth_chart'
                     else 'Automatic prior-season incumbent' if selected_qb1s.get((team, key)) == 'prior_season_incumbent'
                     else 'Automatic recent full-snap starter' if selected_qb1s.get((team, key)) == 'observed_current_starter'
                     else 'QB1 selection required' if team in required_teams
                     else 'QB non-starter')
                    for team, key in zip(team_keys_rv, name_keys_rv)
                ], dtype=object)
                qb_projected_starter = qb1_workload_override.copy()
                qb_nonstarter_volume_factor = qb_projected_starter.astype(float)
            else:
                qb1_workload_source = np.full(len(cur), 'QB1 selection feature disabled', dtype=object)
        role_scale = np.ones(len(cur), dtype=float)
        if use_role_volume:
            # Stable IDs are the primary cross-season bridge.  The exact-name
            # fallback remains for older/local rows with no identifier, but a
            # name change such as Kenny -> Kenneth Gainwell no longer loses
            # a real prior role simply because its display string changed.
            player_share = identity_keys_rv.map(exp_share_identity).to_numpy(dtype=float)
            # A player with NO measured role anywhere - an undrafted rookie,
            # a practice-squad call-up - must not default to a full-time
            # one. np.nan_to_num(..., nan=1.0) did exactly that, and it put
            # three UDFA running backs at the very top of a week-1 board
            # (Jacory Croskey-Merritt at 24.7 projected points) purely
            # because "no snap data" was being read as "every snap".
            #
            # The position's own MEDIAN share was the original stand-in, but
            # a real-data audit (2026-08-24) found the median (a "typical
            # rostered contributor") is itself far too generous for a true
            # unknown: three UDFA WRs with zero career snaps each landed a
            # 30.8% projected share on a real board, which was enough to
            # crowd legitimate committee-role RBs (Kenneth Walker III,
            # Derrick Henry) out of their own team's pass-catcher budget in
            # data.pass_capacity_allocator's team-target fit. The 15th
            # percentile of the same measured population is still position-
            # and week-aware (not a hardcoded number) and still nonzero (a
            # real call-up is not zeroed), but no longer treats a total
            # unknown as an average rostered player. ``fell_back_to_default``
            # marks exactly who received it, so the Ourlads-rank refinement
            # just below can shrink it further for a player a real depth
            # chart shows buried at 3rd string or deeper.
            measured = player_share[np.isfinite(player_share)]
            default_share = float(np.percentile(measured, 15)) if measured.size else 0.5
            fell_back_to_default = ~np.isfinite(player_share)
            player_share = np.where(np.isfinite(player_share), player_share, default_share)
            player_whole_share = player_share.copy()
            # A named QB1 (or one unambiguous full-season incumbent) is a
            # full-game starter, not last season's fraction of team weeks.
            # This changes workload only; availability remains the injury
            # input and passing efficiency/matchup remain normal model inputs.
            player_share = np.where(qb1_workload_override, 1.0, player_share)
            player_prior_share = (identity_keys_rv.map(prior_share_identity).to_numpy(dtype=float)
                                  if not prior_share_identity.empty else np.full(len(cur), np.nan))
            # Unblended 2025-only active share + 2024 season evidence, for
            # the RB allocator's season-scoped eligibility gate below - see
            # prior_active_snap_share_2025_only's own comment for why this
            # has to be separate from the (now blended) player_prior_share.
            player_prior_share_2025_only = (
                identity_keys_rv.map(prior_active_snap_share_2025_only).to_numpy(dtype=float)
                if not prior_active_snap_share_2025_only.empty else np.full(len(cur), np.nan))
            player_prior2_games = (identity_keys_rv.map(games_2024_identity).to_numpy(dtype=float)
                                   if not games_2024_identity.empty else np.full(len(cur), np.nan))
            player_prior2_active_share = (
                identity_keys_rv.map(prior2_active_identity).to_numpy(dtype=float)
                if not prior2_active_identity.empty else np.full(len(cur), np.nan))
            player_recent_active_share_2025 = (
                identity_keys_rv.map(recent_active_share_2025_identity).to_numpy(dtype=float)
                if not recent_active_share_2025_identity.empty else np.full(len(cur), np.nan))
            player_prior_teams = np.full(len(cur), '', dtype=object)
            prior_games = np.full(len(cur), np.nan)
            player_pre_absence_share = (identity_keys_rv.map(prior_pre_absence_identity).to_numpy(dtype=float)
                                        if not prior_pre_absence_identity.empty else np.full(len(cur), np.nan))
            player_terminal_gap_weeks = (identity_keys_rv.map(prior_terminal_gap_identity).to_numpy(dtype=float)
                                         if not prior_terminal_gap_identity.empty else np.full(len(cur), np.nan))
            player_pre_absence_games = (identity_keys_rv.map(prior_pre_absence_games_identity).to_numpy(dtype=float)
                                         if not prior_pre_absence_games_identity.empty else np.full(len(cur), np.nan))
            player_interrupted_season = (identity_keys_rv.map(prior_interrupted_identity).fillna(0.0)
                                         .to_numpy(dtype=float) > 0.5
                                         if not prior_interrupted_identity.empty else np.zeros(len(cur), dtype=bool))
            if cold_start and not prior_role_reference.empty:
                prior_role_keyed = prior_role_reference.copy()
                prior_role_keyed = prior_role_keyed.drop_duplicates('_identity_key', keep='last').set_index('_identity_key')
                prior_games = pd.to_numeric(
                    identity_keys_rv.map(prior_role_keyed.get('Games', pd.Series(dtype=float))), errors='coerce'
                ).to_numpy(dtype=float)
                mapped_prior_teams = identity_keys_rv.map(
                    prior_role_keyed.get('Team', pd.Series(dtype=object)))
                # ``Team`` can be categorical upstream.  Casting before
                # fillna avoids trying to insert an empty string as a new
                # category when a newly overlaid player has no prior row.
                player_prior_teams = pd.Series(
                    mapped_prior_teams, index=cur.index, dtype=object).fillna('').to_numpy(dtype=object)
            # Resolve the local depth chart first so its literal rank can be
            # an audited, bounded confidence input to returning-role recovery.
            # It never supplies availability: a red source flag remains a
            # warning unless the manual/current injury layer says the player
            # is actually unavailable.
            depth_chart_decay = 1.0 if cold_start else max(
                0.0, 1.0 - (int(as_of_week) - 1) / EARLY_SEASON_DEPTH_CHART_DECAY_WEEKS)
            if depth_chart_decay > 0.0 and not historical_target and pos in OURLADS_PRESEASON_ROLE_FLOORS:
                (floor_share, ourlads_role_floor_applied, ourlads_role_rank,
                 ourlads_role_floor, ourlads_role_label) = apply_ourlads_preseason_role_floor(
                    player_share, player_prior_share, player_prior_teams,
                    team_keys_rv.to_numpy(dtype=object), cur[name_col].to_numpy(dtype=object),
                    pos, ourlads_signal['skill_roles'],
                )
                # A hard floor at Week 1; a fading PULL toward that same
                # floor for a few weeks after, never a fresh binary cliff.
                player_share = np.where(
                    ourlads_role_floor_applied,
                    player_share + depth_chart_decay * (floor_share - player_share),
                    player_share)
                # The shared identity resolver is authoritative.  Keep its
                # literal rank as the fallback even if a legacy name-key
                # caller cannot see a display-name variant.
                ourlads_role_rank = np.where(
                    np.isfinite(ourlads_role_rank), ourlads_role_rank,
                    ourlads_audit['source_rank'])
                # Shrink a percentile-FALLBACK share (never a player's own
                # measured one) when the real chart shows him buried 3rd
                # string or deeper, or shows a real chart for this team/
                # position that simply never lists him at all - added
                # 2026-08-24 per the user's own read of a real case (Bam
                # Knight, ARI's chart RB4, behind Love/Allgeier/Conner:
                # "should be projected less than 10% if not less"). A
                # generic no-evidence player and a chart-confirmed deep
                # reserve are not the same thing once a real chart exists;
                # only the former still deserves the ordinary percentile
                # default. Same fading-pull shape as the role floor above,
                # just pulling down instead of up.
                skill_roles = ourlads_signal.get('skill_roles')
                if skill_roles is not None and not skill_roles.empty and 'position' in skill_roles.columns:
                    charted_teams_this_pos = set(_clean_team_key(
                        skill_roles.loc[skill_roles['position'].astype(str).str.upper().eq(pos), 'team']))
                    team_has_chart_arr = np.isin(
                        team_keys_rv.to_numpy(dtype=object), list(charted_teams_this_pos))
                    deep_or_unlisted = fell_back_to_default & (
                        (np.isfinite(ourlads_role_rank) & (ourlads_role_rank >= 3))
                        | (~np.isfinite(ourlads_role_rank) & team_has_chart_arr)
                    )
                    deep_bench_target = np.minimum(player_share, min(default_share, 0.03))
                    player_share = np.where(
                        deep_or_unlisted,
                        player_share + depth_chart_decay * (deep_bench_target - player_share),
                        player_share)
            player_availability = cur[name_col].map(
                lambda player: injury_profiles.get(player, {}).get('plays_probability', 1.0)
            ).to_numpy(dtype=float)
            if (cold_start and pos in COLD_START_RETURNING_ROLE_MIN_GAMES
                    and not prior_role_reference.empty):
                player_share, returning_role_restored, returning_role_reason = restore_cold_start_returning_role_share(
                    player_share, player_prior_share, prior_games, player_prior_teams,
                    team_keys_rv.to_numpy(dtype=object), pos,
                    pre_absence_share=player_pre_absence_share,
                    depth_rank=np.where(player_availability > 0.01, ourlads_role_rank, np.nan),
                    terminal_gap_weeks=player_terminal_gap_weeks,
                    return_details=True,
                )
                preseason_role_source = np.where(returning_role_restored, returning_role_reason,
                                                 'prior-season whole-team participation')
            elif cold_start and pos in COLD_START_RETURNING_ROLE_MIN_GAMES:
                preseason_role_source = np.full(
                    len(cur), 'prior-season whole-team participation', dtype=object,
                )
            elif not cold_start:
                preseason_role_source = np.full(len(cur), 'observed current-season role', dtype=object)
            if ourlads_role_floor_applied.any():
                role_messages = np.asarray([
                    (f'Ourlads {label} listed role floor (depth {int(rank)})'
                     if used else source)
                    for source, used, label, rank in zip(
                        preseason_role_source, ourlads_role_floor_applied,
                        ourlads_role_label, ourlads_role_rank)
                ], dtype=object)
                preseason_role_source = role_messages
            if pos in ('WR', 'TE'):
                depth_rank_within_team = (
                    pd.Series(player_share, index=cur.index)
                    .groupby(team_keys_rv.to_numpy(dtype=object))
                    .rank(method='first', ascending=False)
                    .to_numpy(dtype=float)
                )
                if pos == 'WR':
                    share_cap = np.where(
                        depth_rank_within_team >= WR_DEPTH_RANK_CUTOFF, RECEIVER_DEPTH_CUTOFF_SHARE_CAP,
                        np.where(depth_rank_within_team == WR_DEPTH_RANK_SMALL_ROLE,
                                 WR_DEPTH_RANK_SMALL_ROLE_SHARE_CAP, np.inf))
                else:
                    share_cap = np.where(
                        depth_rank_within_team >= TE_DEPTH_RANK_CUTOFF, RECEIVER_DEPTH_CUTOFF_SHARE_CAP, np.inf)
                player_share = np.minimum(player_share, share_cap)
            if cold_start:
                pos_rows_for_share = (player_prior[player_prior['position'].astype(str).str.upper() == pos]
                                      if not player_prior.empty else player_prior)
                share_name_col = prior_name_col
            else:
                pos_rows_for_share = player_hist[player_hist['position'].astype(str).str.upper() == pos]
                share_name_col = name_col
            pos_share_sum = 0.0
            if not pos_rows_for_share.empty and 'weekly_snap_pct' in pos_rows_for_share.columns:
                pos_share_sum = float(
                    (pd.to_numeric(pos_rows_for_share['weekly_snap_pct'], errors='coerce')
                     .fillna(0.0) / 100.0).clip(0, 1).sum())
            role_scale = np.where(
                np.isfinite(player_prior_share) & (player_prior_share > 0.02)
                & np.isfinite(player_share),
                np.clip(np.divide(player_share, player_prior_share,
                                  out=np.ones_like(player_share),
                                  where=player_prior_share > 0.02), *ROLE_VOLUME_CLIP),
                1.0)
            # The prior rate is already a per-game rate from the QB's active
            # appearances. A resolved QB1 receives that full rate rather than
            # being shrunk for old missed games or boosted by partial snaps.
            role_scale = np.where(qb1_workload_override, 1.0, role_scale)
        else:
            player_share = np.ones(len(cur))
            player_whole_share = player_share.copy()
            player_prior_share = np.full(len(cur), np.nan)
            player_pre_absence_share = np.full(len(cur), np.nan)
            player_terminal_gap_weeks = np.full(len(cur), np.nan)
            player_pre_absence_games = np.full(len(cur), np.nan)
            player_interrupted_season = np.zeros(len(cur), dtype=bool)
            pos_share_sum = 0.0
            preseason_role_source = np.full(len(cur), 'role-volume feature disabled', dtype=object)

        if (pos == 'RB' and not cold_start and use_role_volume
                and 'v2_preseason_rb_allocator' in feats):
            # The preseason allocator is deliberately not reapplied after
            # games have been observed.  Its vacancy handoff *is* still the
            # V2 role-aware redistribution path, though, so establish a
            # current core-RB recipient pool from observed expected snaps.
            # Otherwise every live RB would retain the initialization value
            # ``_rb_core=False`` and an OUT RB could not redistribute any
            # carries or targets at all.  This is a recipient eligibility
            # label, not a second preseason capacity forecast.
            # ``cur`` is an aggregated in-season frame whose source
            # position was preserved in this private column before the
            # aggregation.  Prefer it explicitly: the generic classifier's
            # public-column fallback is intentionally conservative and would
            # otherwise see no position label on this reduced frame.
            live_functional = cur.get(
                '_functional_position', classify_functional_position(cur)
            ).astype(str).str.upper().to_numpy()
            live_status = cur.get('status', pd.Series('', index=cur.index)).astype(str).str.upper().to_numpy()
            rb_core = (
                (live_functional == 'RB')
                & np.isfinite(player_share)
                & (np.asarray(player_share, dtype=float) >= 0.05)
                & ~np.isin(live_status, tuple(INELIGIBLE_ROSTER_STATUSES))
            )
            rb_allocation_source = np.where(
                rb_core,
                'in-season observed-role core-RB recipient pool',
                'not an in-season eligible core-RB recipient',
            )

        if (pos == 'RB' and cold_start and use_role_volume
                and 'v2_preseason_rb_allocator' in feats):
            # Reconcile preseason RB roles against finite team capacities
            # BEFORE per-stat matchup/pace factors.  This is a role baseline,
            # not a claim that every matchup has identical carry volume.
            cap_by_team = (rb_team_capacities.set_index('team')
                           if not rb_team_capacities.empty else pd.DataFrame())
            def _prior_rate_for(stat):
                if prior.empty or stat not in prior.columns:
                    return np.full(len(cur), np.nan)
                values = pd.Series(
                    (prior[stat] / prior['Games'].replace(0, np.nan)).to_numpy(dtype=float),
                    index=prior['_identity_key'])
                return identity_keys_rv.map(values).to_numpy(dtype=float)

            prior_carry_rate = _prior_rate_for('rushing_attempts')
            prior_target_rate = _prior_rate_for('targets')
            allocator_input = pd.DataFrame({
                'team': team_keys_rv.to_numpy(dtype=object),
                'Player': cur[name_col].to_numpy(dtype=object),
                'player_id': cur.get('player_id', pd.Series('', index=cur.index)).to_numpy(dtype=object),
                'gsis_id': cur.get('gsis_id', pd.Series('', index=cur.index)).to_numpy(dtype=object),
                'pff_id': cur.get('pff_id', pd.Series('', index=cur.index)).to_numpy(dtype=object),
                'position': 'RB',
                'depth_chart_position': cur.get('depth_chart_position', pd.Series('', index=cur.index)).to_numpy(dtype=object),
                'ourlads_position': cur.get('ourlads_position', pd.Series('', index=cur.index)).to_numpy(dtype=object),
                'ourlads_depth_rank': ourlads_role_rank,
                'base_snap_share': player_share,
                'prior_active_snap_share': player_prior_share,
                'prior_whole_snap_share': player_whole_share,
                # Season-scoped inputs for the eligibility gate ONLY - see
                # allocate_preseason_rb_roles' own comment on strong_evidence.
                'season_active_snap_share_2025': player_prior_share_2025_only,
                'season_recent8_snap_share_2025': player_recent_active_share_2025,
                'prior2_games': player_prior2_games,
                'prior2_active_snap_share': player_prior2_active_share,
                'pre_absence_snap_share': player_pre_absence_share,
                'interrupted_season': player_interrupted_season,
                'prior_carries_per_game': prior_carry_rate,
                'prior_targets_per_game': prior_target_rate,
                'prior_games': prior_games,
                # Keep this separate from ``base_snap_share``.  The latter
                # can be a position-median placeholder for an unobserved
                # rookie/reserve; it must not make that player credible in a
                # team without an imported chart.
                'has_observed_prior_role': (
                    np.isfinite(player_prior_share)
                    | (np.isfinite(prior_games) & (prior_games > 0))
                ),
                # Both arrays are already ordered like ``cur``.  Compare
                # positionally: ``team_keys_rv`` may retain duplicate
                # source indexes after a cold-pool filter, and Series.eq
                # would otherwise outer-align indexes and double the input.
                'same_team': (
                    _clean_team_key(pd.Series(player_prior_teams)).to_numpy(dtype=object)
                    == team_keys_rv.to_numpy(dtype=object)
                ),
                # A source spelling/matching problem must never make an
                # active, same-team workhorse disappear from a Week-1 pool.
                # This is intentionally an eligibility backstop only; the
                # allocator still reconciles his finite team capacity with
                # the rest of the credible backfield.
                'established_incumbent_backstop': (
                    (_clean_team_key(pd.Series(player_prior_teams)).to_numpy(dtype=object)
                     == team_keys_rv.to_numpy(dtype=object))
                    & np.isfinite(player_prior_share)
                    & (np.asarray(player_prior_share, dtype=float) >= 0.45)
                    & np.isfinite(prior_games)
                    & (np.asarray(prior_games, dtype=float) >= 8)
                ),
                'draft_capital': pd.to_numeric(cur.get('draft_number', pd.Series(np.nan, index=cur.index)), errors='coerce').to_numpy(dtype=float),
                'is_rookie': cur.get('is_rookie_flag', pd.Series(False, index=cur.index)).to_numpy(),
                'status': cur.get('status', pd.Series('', index=cur.index)).to_numpy(dtype=object),
                'availability': cur[name_col].map(
                    lambda player: injury_profiles.get(player, {}).get('plays_probability', 1.0)).to_numpy(dtype=float),
            }, index=cur.index)
            for column, default in (
                ('core_rb_snap_capacity', 1.0), ('rb_carry_capacity', 21.0), ('rb_target_capacity', 5.0),
            ):
                allocator_input[column] = (team_keys_rv.map(cap_by_team[column]).fillna(default).to_numpy(dtype=float)
                                           if not cap_by_team.empty and column in cap_by_team.columns
                                           else np.full(len(cur), default))
            allocator_input = derive_rb_allocator_segment_fields(
                allocator_input, rb_role_segments, rb_teammate_context,
                player_col='Player', team_col='team')
            # A clear internal absence/return has a true pre-gap role.  Use
            # it in preference to the terminal-only generic summary before
            # the allocator scores the player, then blend carry/target
            # evidence toward that era rather than letting the backup's
            # replacement-only season volume become a permanent equal split.
            segment_pre_snap = pd.to_numeric(
                allocator_input.get('rb_segment_pre_absence_snap_share', pd.Series(np.nan, index=allocator_input.index)),
                errors='coerce')
            segment_interrupted = allocator_input.get(
                'rb_segment_interrupted_season', pd.Series(False, index=allocator_input.index))
            segment_interrupted = pd.Series(segment_interrupted, index=allocator_input.index).fillna(False).astype(bool)
            use_segment = segment_interrupted & segment_pre_snap.notna()
            allocator_input.loc[use_segment, 'pre_absence_snap_share'] = segment_pre_snap.loc[use_segment]
            allocator_input.loc[use_segment, 'interrupted_season'] = True
            credit = pd.to_numeric(
                allocator_input.get('interrupted_incumbent_role_credit', pd.Series(0.0, index=allocator_input.index)),
                errors='coerce').fillna(0.0).clip(0.0, 1.0)
            for raw_column, segment_column in (
                ('prior_carries_per_game', 'rb_segment_pre_absence_carries_per_game'),
                ('prior_targets_per_game', 'rb_segment_pre_absence_targets_per_game'),
            ):
                historic = pd.to_numeric(allocator_input[raw_column], errors='coerce')
                pre_gap = pd.to_numeric(
                    allocator_input.get(segment_column, pd.Series(np.nan, index=allocator_input.index)),
                    errors='coerce')
                # 50% maximum blend leaves the complete season informative;
                # interruption credit determines how strongly the healthy
                # lead-back era can correct a replacement-distorted average.
                blend = (0.50 * credit).where(pre_gap.notna(), 0.0)
                allocator_input[raw_column] = (
                    historic.where(historic.notna(), pre_gap).fillna(0.0) * (1.0 - blend)
                    + pre_gap.fillna(historic).fillna(0.0) * blend
                )
            allocation, allocation_ledger = allocate_preseason_rb_roles(allocator_input)
            allocation = allocation.reindex(cur.index)
            if not allocation.empty:
                player_share = allocation['expected_snap_share'].to_numpy(dtype=float)
                rb_allocator_applied = allocation['eligible_core_rb'].fillna(False).to_numpy(dtype=bool)
                # ``core_rb`` in the allocator means the broad functional
                # position; the V2 projection/vacancy recipient pool must
                # instead use the *credible allocated* subset.  Otherwise a
                # chart-unlisted reserve could appear as Core RB in the UI
                # (and become a fragile zero-share vacancy recipient).
                rb_core = rb_allocator_applied.copy()
                rb_carry_allocation = allocation['allocated_carries'].to_numpy(dtype=float)
                rb_target_allocation = allocation['allocated_targets'].to_numpy(dtype=float)
                rb_allocation_source = allocation['allocation_source'].fillna('').to_numpy(dtype=object)
                rb_allocation_eligibility_reason = allocation.get(
                    'allocation_eligibility_reason', pd.Series('', index=allocation.index)
                ).fillna('').to_numpy(dtype=object)
                rb_established_incumbent_backstop = allocation.get(
                    'established_incumbent_backstop', pd.Series(False, index=allocation.index)
                ).fillna(False).astype(bool).to_numpy(dtype=bool)
                rb_snap_capacity = allocator_input['core_rb_snap_capacity'].to_numpy(dtype=float)
                rb_carry_capacity = allocator_input['rb_carry_capacity'].to_numpy(dtype=float)
                rb_target_capacity = allocator_input['rb_target_capacity'].to_numpy(dtype=float)
                # Feed the same audited segment fields into the decomposition
                # and later rate trace.  The actual allocation remains the
                # single source of expected Week-1 opportunity.
                player_pre_absence_share = allocator_input['pre_absence_snap_share'].to_numpy(dtype=float)
                player_interrupted_season = pd.Series(
                    allocator_input['interrupted_season'], index=allocator_input.index).fillna(False).astype(bool).to_numpy()
                rb_segment_status = allocator_input.get(
                    'rb_segment_rb_segment_status', allocator_input.get(
                        'rb_segment_status', pd.Series('', index=allocator_input.index))
                ).fillna('').astype(str).to_numpy(dtype=object)
                rb_segment_pre_absence_snap_share = segment_pre_snap.to_numpy(dtype=float)
                rb_segment_gap_games = pd.to_numeric(
                    allocator_input.get('rb_segment_absence_team_games', pd.Series(np.nan, index=allocator_input.index)),
                    errors='coerce').to_numpy(dtype=float)
                rb_segment_return_snap_share = pd.to_numeric(
                    allocator_input.get('rb_segment_return_recovery_snap_share', pd.Series(np.nan, index=allocator_input.index)),
                    errors='coerce').to_numpy(dtype=float)
                rb_interrupted_incumbent_credit = credit.to_numpy(dtype=float)
                rb_shared_healthy_lead_score = pd.to_numeric(
                    allocator_input.get('shared_healthy_lead_score', pd.Series(0.0, index=allocator_input.index)),
                    errors='coerce').fillna(0.0).to_numpy(dtype=float)
                rb_replacement_only_downweight = pd.to_numeric(
                    allocator_input.get('replacement_only_era_downweight', pd.Series(0.0, index=allocator_input.index)),
                    errors='coerce').fillna(0.0).to_numpy(dtype=float)
                if not allocation_ledger.empty:
                    ledger_by_resource = allocation_ledger.set_index(['team', 'resource'])
                    rb_other_snap = np.asarray([
                        float(ledger_by_resource.loc[(team, 'core_rb_snaps'), 'unallocated'])
                        if (team, 'core_rb_snaps') in ledger_by_resource.index else np.nan
                        for team in team_keys_rv], dtype=float)
                    rb_other_carries = np.asarray([
                        float(ledger_by_resource.loc[(team, 'rb_carries'), 'unallocated'])
                        if (team, 'rb_carries') in ledger_by_resource.index else np.nan
                        for team in team_keys_rv], dtype=float)
                    rb_other_targets = np.asarray([
                        float(ledger_by_resource.loc[(team, 'rb_targets'), 'unallocated'])
                        if (team, 'rb_targets') in ledger_by_resource.index else np.nan
                        for team in team_keys_rv], dtype=float)
                    rb_allocation_ledger.extend(allocation_ledger.to_dict('records'))
                # Do not apply the old all-stat snap ratio to a carry/target
                # capacity we just reconciled.  Dependent rushing and
                # receiving stats receive their own opportunity-specific
                # scale below, while non-opportunity fallbacks retain the
                # snap-share scale for a sensible rookie baseline.
                role_scale = np.where(
                    np.isfinite(player_prior_share) & (player_prior_share > 0.02),
                    np.clip(np.divide(player_share, player_prior_share,
                                      out=np.ones_like(player_share), where=player_prior_share > 0.02),
                            *ROLE_VOLUME_CLIP), 1.0)
                rb_carry_rate_scale = np.where(
                    np.isfinite(prior_carry_rate) & (prior_carry_rate > 0.05),
                    np.clip(np.divide(rb_carry_allocation, prior_carry_rate,
                                      out=np.ones_like(player_share), where=prior_carry_rate > 0.05),
                            0.20, 2.00), role_scale)
                rb_target_rate_scale = np.where(
                    np.isfinite(prior_target_rate) & (prior_target_rate > 0.05),
                    np.clip(np.divide(rb_target_allocation, prior_target_rate,
                                      out=np.ones_like(player_share), where=prior_target_rate > 0.05),
                            0.20, 2.00), role_scale)
                preseason_role_source = np.where(
                    rb_allocator_applied,
                    'team-constrained preseason core-RB allocation', preseason_role_source)

        # Team dropbacks x evidence-weighted personal style, computed once
        # per position pass and applied per-stat below. Empty/NaN for every
        # row except a selected QB1 - see data/qb_volume_blend.py for why
        # this specific seam (the QB1's own PRIOR rate, before it enters
        # _blended_rate) lets the fix fade out on its own as his current-
        # season starts accumulate real evidence.
        qb_blend_rates, qb_blend_audit = {}, {}
        if pos == 'QB' and 'v2_qb_volume_blend' in feats:
            # player_prior/player_prior2 (not raw prior_stats/prior2_stats):
            # the personal rush-share/efficiency this blend reads must skip
            # the same partial/split games every other stat's prior-season
            # rate already skips (player_prior is built at ~line 4220 via
            # annotate_player_history_participation) - passing the raw,
            # unfiltered frame silently let a QB-split relief game drag a
            # QB1's own rate toward that thin sample (Jayden Daniels' 2025:
            # 180.3 raw YPG including 2 split games vs. 205.6 excluding them,
            # found 2026-08-25). Team dropback capacity still wants the RAW
            # team-game total, so the unfiltered frame is passed separately
            # as prior_history_team - see blend_qb1_volume's docstring.
            qb_blend_rates, qb_blend_audit = blend_qb1_volume(
                identity_keys_rv.to_numpy(dtype=object), team_keys_rv.to_numpy(dtype=object),
                qb1_workload_override, player_prior, prior2_history=player_prior2,
                prior_history_team=prior_stats)

        # Display-only "Season average (adj)" ingredient - see
        # _defense_adjusted_prior_average's own docstring. player_game_log_
        # prior is set by both the cold_start and in-season branches above,
        # aligned one-per-row to `cur`.
        defense_adjusted_prior = _defense_adjusted_prior_average(player_game_log_prior, stats)

        proj_cols, stat_trace = {}, {}
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

            stat_prior2_weight = np.full(len(cur), np.nan)
            if not prior_rates.empty and stat in prior_rates.columns:
                prior_map = pd.Series(prior_rates[stat].to_numpy(), index=prior_rates['_identity_key'])
                if (have_two_season_data and stat in PRIOR2_RATE_BLEND_STATS
                        and not older_rates.empty and stat in older_rates.columns):
                    # No separate season-long decay here (contrast
                    # prior_share_identity above): prior_rate already fades
                    # out on its own via _blended_rate's w_current as this
                    # player accumulates 2026 games, so 2024's slice of it
                    # fades for free without a second, compounding discount.
                    older_map = pd.Series(older_rates[stat].to_numpy(), index=older_rates['_identity_key'])
                    prior_map, prior2_weight_map = _blend_with_prior2(prior_map, older_map)
                    stat_prior2_weight = identity_keys_rv.map(prior2_weight_map).to_numpy(dtype=float)
                prior_rate = identity_keys_rv.map(prior_map).to_numpy(dtype=float)
            else:
                prior_rate = np.full(len(cur), np.nan)
            fullback_mask = (
                (pos == 'RB')
                & ('v2_preseason_rb_allocator' in feats)
                & cur['_functional_position'].astype(str).str.upper().eq('FB').to_numpy()
            )
            if fullback_mask.any() and not fullback_prior_rates.empty and stat in fullback_prior_rates.columns:
                fullback_map = pd.Series(
                    fullback_prior_rates[stat].to_numpy(),
                    index=fullback_prior_rates['_identity_key'],
                )
                fullback_rate = identity_keys_rv.map(fullback_map).to_numpy(dtype=float)
                prior_rate = np.where(fullback_mask, fullback_rate, prior_rate)
            td_two_year_used = np.zeros(len(cur), dtype=bool)
            opportunity_stat = TD_OPPORTUNITY_STAT.get(stat)
            if ('v2_td_two_year_prior' in feats and opportunity_stat and not older_rates.empty
                    and stat in older_rates.columns and opportunity_stat in older.columns
                    and not prior.empty and opportunity_stat in prior.columns):
                older_map = pd.Series(older_rates[stat].to_numpy(), index=older_rates['_identity_key'])
                older_rate = identity_keys_rv.map(older_map).to_numpy(dtype=float)
                prior_opp_map = pd.Series(
                    (prior[opportunity_stat] / prior['Games'].replace(0, np.nan)).to_numpy(),
                    index=prior['_identity_key'])
                older_opp_map = pd.Series(
                    (older[opportunity_stat] / older['Games'].replace(0, np.nan)).to_numpy(),
                    index=older['_identity_key'])
                current_prior_opp = identity_keys_rv.map(prior_opp_map).to_numpy(dtype=float)
                older_prior_opp = identity_keys_rv.map(older_opp_map).to_numpy(dtype=float)
                prior_rate, td_two_year_used = blend_comparable_td_priors(
                    prior_rate, older_rate, current_prior_opp, older_prior_opp)
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
            prior_rate_before_role = prior_rate.copy()
            prior_source_is_player = np.isfinite(prior_rate)
            stat_role_scale = role_scale
            if pos == 'RB' and cold_start and 'v2_preseason_rb_allocator' in feats:
                if stat in {'rushing_attempts', 'rushing_yards', 'rushing_tds'}:
                    stat_role_scale = rb_carry_rate_scale
                elif stat in {'targets', 'receptions', 'receiving_yards', 'receiving_tds'}:
                    stat_role_scale = rb_target_rate_scale
            if fullback_mask.any():
                # A functional FB is intentionally excluded from the core-RB
                # allocator, which gives it zero core capacity/share.  That
                # exclusion must not also erase the player's own documented
                # historic touches.  Keep an own-rate-only line at its active
                # game rate; population fallbacks remain disabled below.
                stat_role_scale = np.where(fullback_mask, 1.0, stat_role_scale)
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
                prior_rate = prior_rate * stat_role_scale

            if pos == 'QB' and 'v2_qb_volume_blend' in feats and stat in qb_blend_rates:
                # Replaces the raw personal prior rate for the selected QB1
                # only; a nonstarter (qb1_workload_override False) or a QB1
                # with no usable prior-season history (NaN blended rate) is
                # untouched and falls through to the existing logic above.
                blended_stat_rate = qb_blend_rates[stat]
                replace_prior = qb1_workload_override & np.isfinite(blended_stat_rate)
                prior_rate = np.where(replace_prior, blended_stat_rate, prior_rate)

            if fullback_mask.any():
                # A fullback is visible only through his own historic usage.
                # He never falls through to a starter-level RB population
                # rate merely because a roster feed grouped him as RB.
                in_season_rate = np.where(fullback_mask, np.nan_to_num(in_season_rate, nan=0.0), in_season_rate)
                prior_rate = np.where(fullback_mask, np.nan_to_num(prior_rate, nan=0.0), prior_rate)
                pos_rate_arr = np.where(fullback_mask, 0.0, pos_rate_arr)

            # cur_games (RAW game count, not the recency-weighted
            # weight_sum) still drives the current-vs-prior-season shrinkage
            # below, deliberately - see _weighted_player_rates' docstring on
            # why that calibration is left undisturbed by this change.
            role_change_reduction = (
                ROLE_CHANGE_K_REDUCTION_RB_CARRY
                if (pos == 'RB' and stat in ('rushing_attempts', 'targets')
                    and 'v2_role_change_by_stat' in feats)
                else ROLE_CHANGE_K_REDUCTION)
            blended = _blended_rate(in_season_rate, cur_games, prior_rate, pos_rate_arr, stat,
                                    cur['role_confidence'].to_numpy(dtype=float),
                                    (cur['role_change_confidence'].to_numpy(dtype=float)
                                     if 'v2_adaptive_volume' in feats else None),
                                    role_change_reduction)
            # The allocator's carry/target capacity is the preseason role
            # baseline.  For the two opportunity stats themselves, make that
            # conservation exact before the normal defense/pace projection
            # applies.  A rookie therefore receives his allocated opportunity
            # rather than a generic positional fallback, while yards/TDs keep
            # their player-specific efficiency history.
            if pos == 'RB' and cold_start and 'v2_preseason_rb_allocator' in feats:
                if stat == 'rushing_attempts':
                    blended = np.where(rb_core, rb_carry_allocation, blended)
                    prior_source_is_player = rb_core
                elif stat == 'targets':
                    blended = np.where(rb_core, rb_target_allocation, blended)
                    prior_source_is_player = rb_core

            # QB passing is intentionally direct overall-defense evidence.
            # One QB's passing line is a team passing line; role buckets do
            # not add an independent sample beyond the robust team-game
            # profile. QB rushing remains a separate QB-only channel.
            if pos == 'QB' and stat in QB_PASSING_MATCHUP_STATS:
                if matchup_matrix.empty or stat not in matchup_matrix.columns:
                    matchup_mult = np.ones(len(cur))
                else:
                    matchup_mult = np.clip(
                        cur['Opponent'].map(matchup_matrix[stat]).fillna(1.0).to_numpy(dtype=float),
                        *MATCHUP_CLIP,
                    )
                role_overlay = 'bypassed for QB passing; team-game overall profile'
            elif 'v2_continuous_roles' in feats:
                matchup_mult = _continuous_role_adjusted_multiplier(
                    matchup_matrix, role_tables, role_sizes,
                    cur['Opponent'].to_numpy(), continuous_role_weights, stat)
                role_overlay = 'continuous team-game role blend'
            else:
                matchup_mult = _role_adjusted_multiplier(
                    matchup_matrix, role_tables, role_sizes,
                    cur['Opponent'].to_numpy(),
                    cur[name_col].map(player_roles).fillna('').to_numpy(),
                    stat)
                role_overlay = 'team-game role blend'

            # WR/TE allowed-by-alignment multiplier: player's own slot/wide/
            # inline (or slot/non-slot) mix weighted against the opponent's
            # own allowed-by-alignment rate for each alignment (computed
            # above by alignment_defense_residual_multiplier, per-player-row,
            # into alignment_defense_previews).
            #
            # REDESIGNED 2026-08-26 per explicit request: this used to be
            # MULTIPLIED alongside the broad role/defense matchup below as an
            # incremental correction - two independent opinions of the same
            # WR/TE matchup stacked together, which is exactly the
            # "redundancy problem" that request named. It now REPLACES the
            # broad matchup outright for a player/defense/stat with
            # available alignment evidence (matchup_mult is overwritten, not
            # multiplied) - alignment_defense_residual_multiplier's own
            # candidate_multiplier is already a complete, correctly-scaled
            # matchup multiplier for this player's specific alignment mix
            # (see that function's docstring for the normal-mix-division and
            # confidence-shrink removal that made this replacement sound).
            # Falls back to the broad matchup untouched wherever alignment
            # evidence isn't available for that row (non-WR/TE, a defense/
            # stat combo with no comparison games, etc).
            #
            # Originally built 2026-08-24 as the incremental version above,
            # BACKTESTED AND REJECTED that same day (see
            # docs/weekly_projections_methodology.md for the original -0.003
            # to +0.082 MAE numbers) - that backtest predates both this
            # redesign and the 2024 weekly archive now available, so it does
            # not describe current behavior and needs re-running before this
            # is trusted at face value again.
            alignment_player_factor = np.ones(len(cur))
            alignment_residual_available = np.zeros(len(cur), dtype=bool)
            alignment_scoring_stat = ALIGNMENT_SCORING_STAT_MAP.get(stat)
            if ('v2_pff_alignment_matchup' in feats and pos in ALIGNMENT_DEFENSE_SUPPORTED_POSITIONS
                    and alignment_scoring_stat is not None and alignment_defense_previews):
                previews = [alignment_defense_previews.get(idx, {}).get(alignment_scoring_stat, {})
                            for idx in cur.index]
                alignment_player_factor = np.array(
                    [p.get('candidate_multiplier', 1.0) for p in previews], dtype=float)
                alignment_residual_available = np.array(
                    [bool(p.get('candidate_available', False)) for p in previews], dtype=bool)
                matchup_mult = np.where(alignment_residual_available, alignment_player_factor, matchup_mult)

            # Man/zone allowed-by-scheme multiplier - CANDIDATE, built
            # 2026-08-27 for the first-ever backtest of this mechanism (see
            # 'v2_scheme_matchup' in MODEL_FEATURES). Same REPLACE pattern as
            # alignment immediately above, and deliberately applied AFTER it:
            # where BOTH alignment and scheme have available evidence for the
            # same player/defense/stat, scheme's own opinion wins, since the
            # question this flag exists to answer is "does letting man/zone
            # evidence have the final say, wherever it has evidence, help" -
            # not a claim that scheme is more trustworthy in general. Falls
            # back to whatever matchup_mult already held (broad matchup, or
            # alignment's replacement) wherever scheme evidence is unavailable.
            scheme_player_factor = np.ones(len(cur))
            scheme_residual_available = np.zeros(len(cur), dtype=bool)
            scheme_scoring_stat = ALIGNMENT_SCORING_STAT_MAP.get(stat)
            scheme_wanted = ('v2_scheme_matchup' in feats or 'v2_scheme_alignment_blend' in feats)
            if (scheme_wanted and pos in SCHEME_DEFENSE_SUPPORTED_POSITIONS
                    and scheme_scoring_stat is not None and scheme_defense_previews):
                scheme_previews_row = [scheme_defense_previews.get(idx, {}).get(scheme_scoring_stat, {})
                                       for idx in cur.index]
                scheme_player_factor = np.array(
                    [p.get('candidate_multiplier', 1.0) for p in scheme_previews_row], dtype=float)
                scheme_residual_available = np.array(
                    [bool(p.get('candidate_available', False)) for p in scheme_previews_row], dtype=bool)

            # Evidence-weighted blend of alignment and scheme - CANDIDATE,
            # built 2026-08-27 to test combining both rather than either
            # replacing the broad matchup alone or one outright replacing the
            # other, given the two are correlated at r~0.72
            # (scripts/check_alignment_scheme_overlap.py). Deliberately a
            # NAIVE weighted average - it does NOT discount for that shared
            # variance - built as the cheap baseline to check empirically
            # whether the correlation caution actually costs anything in
            # practice, before building anything more elaborate (e.g. a
            # joint alignment x scheme interaction profile, which corrects
            # for the overlap by construction instead of by adjustment).
            # Weight per side is that side's own effect_weight (the weaker of
            # its two per-bucket confidences, already in each preview dict) -
            # zero for a side with no evidence, so this reduces to
            # "whichever one has evidence" when only one does (~1% of rows;
            # see the overlap script's "both or neither" finding for why this
            # is rarely exercised in practice).
            if 'v2_scheme_alignment_blend' in feats:
                fixed_w = SCHEME_ALIGNMENT_BLEND_FIXED_WEIGHT.get(pos)
                if fixed_w is not None:
                    # DELIBERATE FIXED RATIO, not evidence-weighted - a
                    # per-position override for the component backtest
                    # program's weight sweep (see
                    # scripts/sweep_scheme_blend_weight.py). fixed_w is
                    # scheme's own share; alignment gets (1 - fixed_w). Still
                    # collapses to "whichever one has evidence" when only one
                    # side is available, same as the evidence-weighted path.
                    align_w = np.where(alignment_residual_available, 1.0 - fixed_w, 0.0)
                    scheme_w = np.where(scheme_residual_available, fixed_w, 0.0)
                else:
                    align_conf = np.array([
                        (alignment_defense_previews.get(idx, {}).get(alignment_scoring_stat, {}).get('effect_weight', 0.0)
                         if alignment_scoring_stat is not None else 0.0)
                        for idx in cur.index], dtype=float)
                    scheme_conf = np.array([
                        (scheme_defense_previews.get(idx, {}).get(scheme_scoring_stat, {}).get('effect_weight', 0.0)
                         if scheme_scoring_stat is not None else 0.0)
                        for idx in cur.index], dtype=float)
                    align_w = np.where(alignment_residual_available, np.maximum(align_conf, 0.0), 0.0)
                    scheme_w = np.where(scheme_residual_available, np.maximum(scheme_conf, 0.0), 0.0)
                total_w = align_w + scheme_w
                blend_available = total_w > 0
                blended_factor = np.divide(
                    align_w * alignment_player_factor + scheme_w * scheme_player_factor,
                    total_w, out=np.ones(len(cur)), where=blend_available)
                matchup_mult = np.where(blend_available, blended_factor, matchup_mult)
            elif 'v2_scheme_matchup' in feats:
                matchup_mult = np.where(scheme_residual_available, scheme_player_factor, matchup_mult)

            if cold_start and 'v2_cold_start_regression' in feats:
                matchup_mult = 1.0 + (1.0 - COLD_START_MULTIPLIER_REGRESSION) * (matchup_mult - 1.0)

            role_change_reduction = (
                ROLE_CHANGE_K_REDUCTION_RB_CARRY
                if (pos == 'RB' and stat in ('rushing_attempts', 'targets')
                    and 'v2_role_change_by_stat' in feats)
                else ROLE_CHANGE_K_REDUCTION)
            current_weight = _current_blend_weight(
                cur_games, stat, cur['role_confidence'].to_numpy(dtype=float),
                (cur['role_change_confidence'].to_numpy(dtype=float)
                 if 'v2_adaptive_volume' in feats else None),
                role_change_reduction)
            stat_trace[stat] = {
                'build_path': np.full(len(cur), 'direct rate'),
                'current_rate': in_season_rate,
                'raw_prior_rate': prior_rate_before_role,
                'defense_adjusted_prior_rate': defense_adjusted_prior.get(
                    stat, np.full(len(cur), np.nan)),
                # Weight actually given to 2024 (`older_map`) in the blend
                # just above - NaN when this stat isn't in
                # PRIOR2_RATE_BLEND_STATS (TD-type stats use a separate
                # two-year path, blend_comparable_td_priors, exposed via
                # 'two_year_td_prior' instead) or no usable 2024 read exists
                # for this player. Added 2026-08-25 so the decomposition's
                # "100% {season}" note can't be misread as "0% 2024" - that
                # note is about CURRENT vs PRIOR SEASON weight, an
                # independent axis from this one.
                'prior2_weight': stat_prior2_weight,
                'prior_rate': np.where(np.isnan(prior_rate), pos_rate_arr, prior_rate),
                'prior_source': np.where(prior_source_is_player, 'player prior', 'position fallback'),
                'role_scale': stat_role_scale,
                'expected_snap_share': player_share,
                'prior_snap_share': player_prior_share,
                'pre_absence_snap_share': player_pre_absence_share,
                'interrupted_season': player_interrupted_season,
                'rb_allocator_applied': rb_allocator_applied,
                'rb_core': rb_core,
                'rb_carry_allocation': rb_carry_allocation,
                'rb_target_allocation': rb_target_allocation,
                'rb_snap_capacity': rb_snap_capacity,
                'rb_carry_capacity': rb_carry_capacity,
                'rb_target_capacity': rb_target_capacity,
                'rb_other_snap': rb_other_snap,
                'rb_other_carries': rb_other_carries,
                'rb_other_targets': rb_other_targets,
                'rb_allocation_source': rb_allocation_source,
                'rb_allocation_eligibility_reason': rb_allocation_eligibility_reason,
                'rb_established_incumbent_backstop': rb_established_incumbent_backstop,
                'rb_segment_status': rb_segment_status,
                'rb_segment_pre_absence_snap_share': rb_segment_pre_absence_snap_share,
                'rb_segment_gap_games': rb_segment_gap_games,
                'rb_segment_return_snap_share': rb_segment_return_snap_share,
                'rb_interrupted_incumbent_credit': rb_interrupted_incumbent_credit,
                'rb_shared_healthy_lead_score': rb_shared_healthy_lead_score,
                'rb_replacement_only_downweight': rb_replacement_only_downweight,
                'qb1_workload_override': qb1_workload_override,
                'qb1_workload_source': qb1_workload_source,
                'qb1_selection_required': qb1_selection_required,
                'qb_projected_starter': qb_projected_starter,
                'qb_nonstarter_volume_factor': qb_nonstarter_volume_factor,
                'qb1_blend_applied': (qb1_workload_override & np.isfinite(qb_blend_rates.get(stat, np.full(len(cur), np.nan))))
                if pos == 'QB' else np.zeros(len(cur), dtype=bool),
                'qb1_blend_personal_dropbacks': qb_blend_audit.get('personal_dropbacks', np.full(len(cur), np.nan)),
                'qb1_blend_personal_rush_share': qb_blend_audit.get('personal_rush_share', np.full(len(cur), np.nan)),
                'qb1_blend_league_rush_share': qb_blend_audit.get('league_rush_share', np.full(len(cur), np.nan)),
                'qb1_blend_evidence_weight': qb_blend_audit.get('evidence_weight', np.full(len(cur), np.nan)),
                'qb1_blend_team_dropback_capacity': qb_blend_audit.get('team_dropback_capacity', np.full(len(cur), np.nan)),
                'qb1_blend_prior2_weight': qb_blend_audit.get('prior2_weight', np.full(len(cur), np.nan)),
                'qb1_blend_personal_dropbacks_2024': qb_blend_audit.get('personal_dropbacks_2024', np.full(len(cur), np.nan)),
                'returning_role_restored': returning_role_restored,
                'returning_role_reason': returning_role_reason,
                'preseason_role_source': preseason_role_source,
                'ourlads_role_floor_applied': ourlads_role_floor_applied,
                'ourlads_role_available_rank': ourlads_role_rank,
                'ourlads_role_floor': ourlads_role_floor,
                'ourlads_role_position_label': ourlads_role_label,
                'ourlads_source_status': ourlads_source_status,
                'ourlads_source_status_warning': ourlads_source_status_warning,
                'ourlads_identity_match_method': ourlads_identity_match_method,
                'ourlads_identity_match_confidence': ourlads_identity_match_confidence,
                'ourlads_identity_match_warning': ourlads_identity_match_warning,
                'ourlads_source_name': ourlads_source_name,
                'ourlads_matched_identity_key': ourlads_matched_identity_key,
                'current_games': cur_games,
                'current_history_excluded_games': cur['_current_history_excluded_games'].to_numpy(dtype=float),
                'current_history_exclusion_reasons': cur['_current_history_exclusion_reasons'].to_numpy(dtype=object),
                'prior_history_excluded_games': cur['_prior_history_excluded_games'].to_numpy(dtype=float),
                'prior_history_exclusion_reasons': cur['_prior_history_exclusion_reasons'].to_numpy(dtype=object),
                'role_confidence': cur['role_confidence'].to_numpy(dtype=float),
                'role_change_confidence': cur['role_change_confidence'].to_numpy(dtype=float),
                'current_weight': current_weight,
                'blended_rate': blended,
                'matchup_multiplier': matchup_mult,
                'alignment_residual_multiplier': alignment_player_factor,
                'alignment_residual_available': alignment_residual_available,
                'defense_profile': np.full(
                    len(cur),
                    ('prior season: position-channel team-game, broad full-season recency'
                     if cold_start else ('current season + prior team-game bridge'
                                         if 'v2_defense_prior' in feats
                                         else 'current season position-channel team-game')),
                ),
                'defense_estimator': np.full(
                    len(cur),
                    'offense-position team-game normalized production',
                ),
                'defense_current_games': cur['Opponent'].map(
                    defense_current_evidence).fillna(0.0).to_numpy(dtype=float),
                'defense_prior_games': cur['Opponent'].map(
                    defense_prior_evidence).fillna(0.0).to_numpy(dtype=float),
                'role_overlay': np.full(len(cur), role_overlay),
                'target_margin': cur['target_margin'].to_numpy(dtype=float),
                'two_year_td_prior': td_two_year_used,
            }

            script_series = script_by_stat.get(stat)
            script_mult = np.ones(len(cur))
            if script_series is not None and not script_series.empty:
                script_mult = cur[name_col].map(script_series).fillna(1.0).to_numpy(dtype=float)
            if cold_start and 'v2_cold_start_regression' in feats:
                script_mult = 1.0 + (1.0 - COLD_START_MULTIPLIER_REGRESSION) * (script_mult - 1.0)
            stat_trace[stat]['script_multiplier'] = script_mult
            stat_trace[stat]['script_status'] = (
                'modeled' if stat in SCRIPT_ELIGIBLE_STATS and script_series is not None and not script_series.empty
                else ('not modeled for this stat' if stat not in SCRIPT_ELIGIBLE_STATS else 'no usable market/history')
            )

            proj_cols[stat] = blended * matchup_mult * script_mult

        pace_mult = pd.Series(1.0, index=cur.index)
        opp_pace = pd.Series(np.nan, index=cur.index)
        if league_pace and league_pace > 0:
            opp_pace = cur['Opponent'].map(pace['def_pace'])
            pace_mult = np.clip(opp_pace.fillna(league_pace) / league_pace, *PACE_CLIP)

        inj_mult = cur[name_col].map(injury_mult).fillna(1.0)
        use_total = 'game_env' in feats or 'v2_game_total_elasticity' in feats
        use_venue = 'game_env' in feats or 'v2_venue_mult' in feats
        env_mult = (_game_env_multiplier(env, cur['Team'].astype(str).to_numpy(), pos, league_implied,
                                         use_total=use_total, use_venue=use_venue)
                    if env else np.ones(len(cur)))

        if cold_start and 'v2_cold_start_regression' in feats:
            pace_mult = 1.0 + (1.0 - COLD_START_MULTIPLIER_REGRESSION) * (pace_mult - 1.0)
            inj_mult = 1.0 + (1.0 - COLD_START_MULTIPLIER_REGRESSION) * (inj_mult - 1.0)
            env_mult = 1.0 + (1.0 - COLD_START_MULTIPLIER_REGRESSION) * (env_mult - 1.0)

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
        if 'teammate_vacancy' in feats or 'v2_vacancy' in feats:
            for stat in ('passing_attempts', 'targets', 'rushing_attempts'):
                if stat in proj_cols:
                    # qb_nonstarter_volume_factor is included here too (it's a
                    # QB1 gate: 1.0 for the starter and every non-QB row, hard
                    # 0.0 for a backup QB) even though the discount it stands
                    # in for is applied to proj_cols much later (see "sits
                    # after optional efficiency rebuilding" below). Without it,
                    # a benched backup's blended relief-appearance rate - his
                    # real per-game share is correctly zero, he is never going
                    # to play over a healthy starter - still looked like
                    # legitimate "vacated" volume the moment the injury feed
                    # marked him Out, and got redistributed onto the ALREADY-
                    # STARTING QB1 on top of his own complete rate (caught via
                    # Lamar Jackson projecting 324.79 passing yards in a normal
                    # matchup - traced to backup Skylar Thompson's fictional
                    # 11.26-attempt "full" volume). The starter's OWN injury
                    # discount must still reach vacancy_volume undimmed (that's
                    # the whole point of this snapshot, see the comment above)
                    # - this factor is 1.0 for him regardless of his own
                    # injury status, so that path is untouched.
                    vacancy_volume[stat] = np.clip(
                        proj_cols[stat] * pace_mult.to_numpy() * env_mult
                        * qb_nonstarter_volume_factor, 0.0, None)

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
            trace = stat_trace.get(stat)
            if trace is not None:
                trace['pace_multiplier'] = pace_mult.to_numpy(dtype=float)
                trace['opponent_defensive_pace'] = opp_pace.to_numpy(dtype=float)
                trace['league_pace'] = np.full(len(cur), league_pace if league_pace else np.nan)
                trace['availability_multiplier'] = inj_mult.to_numpy(dtype=float)
                trace['environment_multiplier'] = np.asarray(env_mult, dtype=float)
                trace['environment_status'] = np.full(
                    len(cur), 'modeled' if 'game_env' in feats and env else
                    ('feature enabled; no usable line' if 'game_env' in feats else 'feature disabled'),
                )

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
            keys = identity_keys_rv
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
                    prior_keyed = prior.drop_duplicates('_identity_key', keep='last').set_index('_identity_key')
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
                                               opponents_arr, roles_arr, num_stat, den_stat,
                                               position=pos)
                proj_cols[num_stat] = np.clip(proj_cols[den_stat] * ratio * eff_mult, 0.0, None)
                trace = stat_trace.get(num_stat)
                if trace is not None:
                    trace['build_path'] = np.full(len(cur), f'derived from {den_stat}')
                    trace['efficiency_denominator'] = np.full(len(cur), den_stat)
                    trace['efficiency_rate'] = ratio
                    trace['efficiency_evidence'] = evidence
                    trace['efficiency_matchup_multiplier'] = eff_mult
                    trace['efficiency_defense_estimator'] = np.full(
                        len(cur),
                        ('offense-team game normalized QB passing'
                         if pos == 'QB' and num_stat in QB_PASSING_MATCHUP_STATS
                         and den_stat in QB_PASSING_MATCHUP_STATS
                         else 'position-team game normalized role blend'),
                    )

        # A QB who is not the expected starter has no normal offensive
        # workload to project.  Snap share alone is insufficient because a
        # backup's relief-game per-appearance rate can still look startable;
        # one selected QB1 receives volume and every other QB on that team is
        # held at zero until he becomes the selected starter.  The gate sits
        # after optional efficiency rebuilding so both passing and rushing
        # stats reconcile to the same explicit participation decision.
        if pos == 'QB' and 'qb1_override' in feats:
            for stat in proj_cols:
                proj_cols[stat] = np.asarray(proj_cols[stat], dtype=float) * qb_nonstarter_volume_factor
                trace = stat_trace.get(stat)
                if trace is not None:
                    trace['qb_projected_starter'] = qb_projected_starter
                    trace['qb_nonstarter_volume_factor'] = qb_nonstarter_volume_factor

        # The direct-rate factors above describe the calculation before an
        # optional volume×efficiency rebuild.  Capture the final pre-vacancy
        # stat here, after every enabled branch, so a decomposition can always
        # reconcile with the board rather than narrate an intermediate value.
        for stat, trace in stat_trace.items():
            if stat in proj_cols:
                trace['pre_vacancy_projection'] = np.asarray(proj_cols[stat], dtype=float)

        out = pd.DataFrame({
            'Player': cur[name_col], 'Pos': pos, 'Team': cur['Team'], 'Opponent': cur['Opponent'],
            '_functional_position': cur['_functional_position'],
            'Games This Season': cur['Games'].astype(int),
            'Role Confidence': cur['role_confidence'].round(2),
            'Expected Snap Share': np.round(player_share, 3),
            'Role Change Confidence': cur['role_change_confidence'].round(2),
            'Partial-Game Exclusions': cur['_current_history_excluded_games'].astype(int),
            'QB1 Workload Override': qb1_workload_override,
            'QB1 Workload Source': qb1_workload_source,
            'QB1 Selection Required': qb1_selection_required,
            'QB Projected Starter': qb_projected_starter,
        })
        out['_role_confidence_recent_snap_pct'] = cur['_role_confidence_recent_snap_pct'].to_numpy()
        out['_role_confidence_games_sampled'] = cur['_role_confidence_games_sampled'].to_numpy()
        out['_role_confidence_route_rate'] = cur['_role_confidence_route_rate'].to_numpy()
        out['_role_confidence_method'] = cur['_role_confidence_method'].to_numpy()
        if pos == 'RB' and 'v2_preseason_rb_allocator' in feats:
            # Internal fields remain on the assembled board until the V2
            # vacancy pass has consumed them.  They are deliberately not
            # rendered as ranking-table columns; the explanation payload
            # below exposes the same information in a readable form.
            out['_rb_core'] = rb_core
            out['_rb_carry_allocation_share'] = np.divide(
                rb_carry_allocation, rb_carry_capacity,
                out=np.zeros(len(cur), dtype=float), where=np.isfinite(rb_carry_capacity) & (rb_carry_capacity > 0))
            out['_rb_target_allocation_share'] = np.divide(
                rb_target_allocation, rb_target_capacity,
                out=np.zeros(len(cur), dtype=float), where=np.isfinite(rb_target_capacity) & (rb_target_capacity > 0))
        if not role_profiles.empty:
            for profile_col in ('adot', 'target_share', 'carry_share', 'receiving_back_share',
                                'snap_share', 'target_earner_rank', 'target_earner_score',
                                'alignment_available', 'evidence_games'):
                if profile_col in role_profiles.columns:
                    out[f'_profile_{profile_col}'] = cur[name_col].map(role_profiles[profile_col])
        if not pff_alignment_for_cur.empty:
            for profile_col in (
                'slot_alignment_rate', 'non_slot_alignment_rate', 'wide_alignment_rate',
                'inline_alignment_rate', 'alignment_sample_weight', 'alignment_confidence',
                'alignment_available', 'alignment_experimental', 'alignment_effect_weight',
                'alignment_matchup_multiplier', 'alignment_semantics', 'source_kind',
                'source_year', 'source_weeks', 'source_week_count', 'source_regular_season',
                'source_time_valid', 'source_confidence', 'source_notes',
                'alignment_defense_targets_candidate_multiplier',
                'alignment_defense_receptions_candidate_multiplier',
                'alignment_defense_yards_candidate_multiplier',
                'alignment_defense_candidate_available', 'alignment_defense_reason',
                'alignment_defense_scoring_active', 'alignment_defense_blend_mode',
                'alignment_prior2_weight', 'alignment_prior2_sample_weight',
                'alignment_defense_targets_slot_ratio', 'alignment_defense_targets_wide_ratio',
                'alignment_defense_targets_inline_ratio', 'alignment_defense_targets_non_slot_ratio',
                'alignment_defense_receptions_slot_ratio', 'alignment_defense_receptions_wide_ratio',
                'alignment_defense_receptions_inline_ratio', 'alignment_defense_receptions_non_slot_ratio',
                'alignment_defense_yards_slot_ratio', 'alignment_defense_yards_wide_ratio',
                'alignment_defense_yards_inline_ratio', 'alignment_defense_yards_non_slot_ratio',
            ):
                if profile_col in pff_alignment_for_cur.columns:
                    out[f'_profile_{profile_col}'] = pff_alignment_for_cur[profile_col].to_numpy()
        if not pff_scheme_for_cur.empty:
            for profile_col in (
                'man_route_share', 'zone_route_share', 'scheme_sample_weight', 'scheme_confidence',
                'man_catch_rate', 'man_yards_per_target', 'man_yprr',
                'zone_catch_rate', 'zone_yards_per_target', 'zone_yprr',
                'scheme_available', 'scheme_semantics', 'source_weeks', 'source_week_count',
                'scheme_defense_targets_candidate_multiplier',
                'scheme_defense_receptions_candidate_multiplier',
                'scheme_defense_yards_candidate_multiplier',
                'scheme_defense_candidate_available', 'scheme_defense_reason',
                'scheme_defense_scoring_active',
            ):
                if profile_col in pff_scheme_for_cur.columns:
                    out[f'_scheme_profile_{profile_col}'] = pff_scheme_for_cur[profile_col].to_numpy()
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
        out['Raw Model Proj Pts'] = [max(0.0, score_projected_stats(d, scoring_mode)) for d in proj_dicts]
        out['Model Proj Pts'] = out['Raw Model Proj Pts']
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
        out['Calibrated Model Proj Pts'] = out['Model Proj Pts']
        if 'v2_availability' in feats:
            out['Availability'] = out['Player'].map(
                lambda p: injury_profiles.get(p, {}).get('plays_probability', 1.0)).astype(float).round(2)
            out['Workload If Active'] = out['Player'].map(
                lambda p: injury_profiles.get(p, {}).get('workload_if_active', 1.0)).astype(float).round(2)
            out['Injury Status'] = out['Player'].map(
                lambda p: injury_profiles.get(p, {}).get('status', '')).replace('unknown', '')
            out['_availability_source'] = out['Player'].map(
                lambda p: injury_profiles.get(p, {}).get('source', 'no current availability source'))
            out['_availability_match_method'] = out['Player'].map(
                lambda p: injury_profiles.get(p, {}).get('match_method', 'not matched'))
            out['_availability_note'] = out['Player'].map(
                lambda p: injury_profiles.get(p, {}).get('note', ''))
        else:
            out['Availability'] = out['Player'].map(lambda p: injury_mult.get(p, 1.0)).astype(float).round(2)
            out['Workload If Active'] = 1.0
            out['Injury Status'] = out['Player'].map(lambda p: 'Out/Doubtful' if injury_mult.get(p, 1.0) < 0.9 else '')
            out['_availability_source'] = 'V1 legacy injury multiplier'
            out['_availability_match_method'] = 'legacy'
            out['_availability_note'] = ''

        # Keep the explanation payload outside the visible dataframe.  This
        # preserves a compact table while giving the dialog every input it
        # needs without re-running projection math in the UI.
        def _trace_value(trace, key, index, default=None):
            values = trace.get(key)
            if values is None:
                return default
            try:
                value = values[index]
            except (IndexError, KeyError, TypeError):
                return default
            if isinstance(value, np.generic):
                value = value.item()
            return default if pd.isna(value) else value

        def _trace_number(trace, key, index, default=None):
            value = _trace_value(trace, key, index, default)
            try:
                return round(float(value), 3) if value is not None else None
            except (TypeError, ValueError):
                return default

        # "Overall defense matchup" - one composite number + rank per opponent,
        # not per stat (defense_stat_rank's OWN rank is 1 = allows the MOST;
        # the UI flips this into a "toughness rank" before display - see
        # _open_projection_dialog's own comment on why). fantasy_points is
        # this app's OWN scoring-
        # weighted blend of targets/receptions/yards/TDs (apply_scoring_and_
        # percentiles), so ranking defenses on points allowed to this position
        # already "encompasses all stats against that position" without a new
        # hand-weighted composite. Computed ONCE per position here (32 teams),
        # not once per row - defense_stat_rank's own groupby is not free.
        # Leakage guard: sourced from _played_weeks_before(stats_df, as_of_week),
        # the SAME as-of cutoff every other current-season input in this
        # function goes through, so a historical-week decomposition can't see
        # that week's own (or a later week's) result either.
        #
        # Early season (as_of_week close to 1) has no current-season games to
        # rank teams on at all - falls back to the FULL prior season
        # (prior_stats, already loaded above for the same reason every other
        # V2 "cold start" input falls back to it). A full completed season is
        # always strictly "before" any week of the current one, so this needs
        # no additional as-of cutoff of its own. source is stamped on the
        # result so the dialog can say which one it's showing rather than
        # silently blending a stale prior-year read with a live one.
        defense_matchup_by_opponent = {}
        asof_stats = (_played_weeks_before(stats_df, as_of_week)
                     if 'week' in stats_df.columns else stats_df.iloc[0:0])
        prior_available = (not prior_stats.empty and 'opponent_team' in prior_stats.columns
                          and 'week' in prior_stats.columns)
        if not asof_stats.empty or prior_available:
            for opp_team in out['Opponent'].dropna().astype(str).unique():
                rank_info = (defense_stat_rank(asof_stats, opp_team, pos, 'fantasy_points')
                            if not asof_stats.empty else None)
                if rank_info:
                    rank_info['source'] = f'{year} season, through Week {int(as_of_week) - 1}'
                elif prior_available:
                    rank_info = defense_stat_rank(prior_stats, opp_team, pos, 'fantasy_points')
                    if rank_info:
                        rank_info['source'] = f'{year - 1} full season (no {year} games played yet)'
                if rank_info:
                    defense_matchup_by_opponent[opp_team] = rank_info

        # Rank within THIS position/week, by the same displayed number the
        # decomposition dialog shows (post-calibration), so a player's boom/
        # bust tier always matches the number the user is actually looking
        # at rather than a raw pre-calibration ordering.
        position_rank = out['Calibrated Model Proj Pts'].rank(ascending=False, method='min')

        for i, (_out_index, row) in enumerate(out.iterrows()):
            player = row['Player']
            role = {k.replace('_profile_', ''): (None if pd.isna(row[k]) else row[k])
                    for k in out.columns if k.startswith('_profile_')}
            stat_detail = {}
            for stat, trace in stat_trace.items():
                stat_detail[stat] = {
                    'channel': projection_channel(pos, stat),
                    'build_path': _trace_value(trace, 'build_path', i, 'direct rate'),
                    'current_rate': _trace_number(trace, 'current_rate', i),
                    'raw_prior_rate': _trace_number(trace, 'raw_prior_rate', i),
                    'defense_adjusted_prior_rate': _trace_number(trace, 'defense_adjusted_prior_rate', i),
                    'prior2_weight': _trace_number(trace, 'prior2_weight', i),
                    'prior_rate': _trace_number(trace, 'prior_rate', i),
                    'prior_source': _trace_value(trace, 'prior_source', i, 'position fallback'),
                    'role_scale': _trace_number(trace, 'role_scale', i, 1.0),
                    'expected_snap_share': _trace_number(trace, 'expected_snap_share', i),
                    'prior_snap_share': _trace_number(trace, 'prior_snap_share', i),
                    'pre_absence_snap_share': _trace_number(trace, 'pre_absence_snap_share', i),
                    'interrupted_season': bool(_trace_value(trace, 'interrupted_season', i, False)),
                    'rb_allocator_applied': bool(_trace_value(trace, 'rb_allocator_applied', i, False)),
                    'rb_core': bool(_trace_value(trace, 'rb_core', i, False)),
                    'rb_carry_allocation': _trace_number(trace, 'rb_carry_allocation', i),
                    'rb_target_allocation': _trace_number(trace, 'rb_target_allocation', i),
                    'rb_snap_capacity': _trace_number(trace, 'rb_snap_capacity', i),
                    'rb_carry_capacity': _trace_number(trace, 'rb_carry_capacity', i),
                    'rb_target_capacity': _trace_number(trace, 'rb_target_capacity', i),
                    'rb_other_snap': _trace_number(trace, 'rb_other_snap', i),
                    'rb_other_carries': _trace_number(trace, 'rb_other_carries', i),
                    'rb_other_targets': _trace_number(trace, 'rb_other_targets', i),
                    'rb_allocation_source': _trace_value(trace, 'rb_allocation_source', i, ''),
                    'rb_allocation_eligibility_reason': _trace_value(
                        trace, 'rb_allocation_eligibility_reason', i, ''),
                    'rb_established_incumbent_backstop': bool(_trace_value(
                        trace, 'rb_established_incumbent_backstop', i, False)),
                    'rb_segment_status': _trace_value(trace, 'rb_segment_status', i, ''),
                    'rb_segment_pre_absence_snap_share': _trace_number(
                        trace, 'rb_segment_pre_absence_snap_share', i),
                    'rb_segment_gap_games': _trace_number(trace, 'rb_segment_gap_games', i),
                    'rb_segment_return_snap_share': _trace_number(
                        trace, 'rb_segment_return_snap_share', i),
                    'rb_interrupted_incumbent_credit': _trace_number(
                        trace, 'rb_interrupted_incumbent_credit', i, 0.0),
                    'rb_shared_healthy_lead_score': _trace_number(
                        trace, 'rb_shared_healthy_lead_score', i, 0.0),
                    'rb_replacement_only_downweight': _trace_number(
                        trace, 'rb_replacement_only_downweight', i, 0.0),
                    'qb1_workload_override': bool(_trace_value(
                        trace, 'qb1_workload_override', i, False)),
                    'qb1_workload_source': _trace_value(trace, 'qb1_workload_source', i),
                    'qb1_selection_required': bool(_trace_value(
                        trace, 'qb1_selection_required', i, False)),
                    'qb_projected_starter': bool(_trace_value(
                        trace, 'qb_projected_starter', i, True)),
                    'qb_nonstarter_volume_factor': _trace_number(
                        trace, 'qb_nonstarter_volume_factor', i, 1.0),
                    'qb1_blend_applied': bool(_trace_value(trace, 'qb1_blend_applied', i, False)),
                    'qb1_blend_personal_dropbacks': _trace_number(trace, 'qb1_blend_personal_dropbacks', i),
                    'qb1_blend_personal_rush_share': _trace_number(trace, 'qb1_blend_personal_rush_share', i),
                    'qb1_blend_league_rush_share': _trace_number(trace, 'qb1_blend_league_rush_share', i),
                    'qb1_blend_evidence_weight': _trace_number(trace, 'qb1_blend_evidence_weight', i),
                    'qb1_blend_team_dropback_capacity': _trace_number(
                        trace, 'qb1_blend_team_dropback_capacity', i),
                    'qb1_blend_prior2_weight': _trace_number(trace, 'qb1_blend_prior2_weight', i),
                    'qb1_blend_personal_dropbacks_2024': _trace_number(
                        trace, 'qb1_blend_personal_dropbacks_2024', i),
                    'returning_role_restored': bool(_trace_value(
                        trace, 'returning_role_restored', i, False)),
                    'returning_role_reason': _trace_value(trace, 'returning_role_reason', i, 'none'),
                    'preseason_role_source': _trace_value(trace, 'preseason_role_source', i),
                    'ourlads_role_floor_applied': bool(_trace_value(
                        trace, 'ourlads_role_floor_applied', i, False)),
                    'ourlads_role_available_rank': _trace_number(
                        trace, 'ourlads_role_available_rank', i),
                    'ourlads_role_floor': _trace_number(trace, 'ourlads_role_floor', i),
                    'ourlads_role_position_label': _trace_value(
                        trace, 'ourlads_role_position_label', i, ''),
                    'ourlads_source_status': _trace_value(trace, 'ourlads_source_status', i, ''),
                    'ourlads_source_status_warning': _trace_value(
                        trace, 'ourlads_source_status_warning', i, ''),
                    'ourlads_identity_match_method': _trace_value(
                        trace, 'ourlads_identity_match_method', i, ''),
                    'ourlads_identity_match_confidence': _trace_value(
                        trace, 'ourlads_identity_match_confidence', i, ''),
                    'ourlads_identity_match_warning': _trace_value(
                        trace, 'ourlads_identity_match_warning', i, ''),
                    'ourlads_source_name': _trace_value(trace, 'ourlads_source_name', i, ''),
                    'ourlads_matched_identity_key': _trace_value(
                        trace, 'ourlads_matched_identity_key', i, ''),
                    'current_games': _trace_number(trace, 'current_games', i, 0.0),
                    'current_history_excluded_games': _trace_number(
                        trace, 'current_history_excluded_games', i, 0.0),
                    'current_history_exclusion_reasons': _trace_value(
                        trace, 'current_history_exclusion_reasons', i, ''),
                    'prior_history_excluded_games': _trace_number(
                        trace, 'prior_history_excluded_games', i, 0.0),
                    'prior_history_exclusion_reasons': _trace_value(
                        trace, 'prior_history_exclusion_reasons', i, ''),
                    'role_confidence': _trace_number(trace, 'role_confidence', i),
                    'role_change_confidence': _trace_number(trace, 'role_change_confidence', i),
                    'current_weight': _trace_number(trace, 'current_weight', i),
                    'blended_rate': _trace_number(trace, 'blended_rate', i),
                    'matchup_multiplier': _trace_number(trace, 'matchup_multiplier', i, 1.0),
                    'alignment_residual_multiplier': _trace_number(
                        trace, 'alignment_residual_multiplier', i, 1.0),
                    'alignment_residual_available': bool(_trace_value(
                        trace, 'alignment_residual_available', i, False)),
                    'defense_profile': _trace_value(trace, 'defense_profile', i, 'current season'),
                    'defense_estimator': _trace_value(trace, 'defense_estimator', i),
                    'defense_current_games': _trace_number(trace, 'defense_current_games', i, 0.0),
                    'defense_prior_games': _trace_number(trace, 'defense_prior_games', i, 0.0),
                    'role_overlay': _trace_value(trace, 'role_overlay', i),
                    'target_margin': _trace_number(trace, 'target_margin', i),
                    'script_multiplier': _trace_number(trace, 'script_multiplier', i, 1.0),
                    'script_status': _trace_value(trace, 'script_status', i, 'not modeled'),
                    'pace_multiplier': _trace_number(trace, 'pace_multiplier', i, 1.0),
                    'opponent_defensive_pace': _trace_number(trace, 'opponent_defensive_pace', i),
                    'league_pace': _trace_number(trace, 'league_pace', i),
                    'availability_multiplier': _trace_number(trace, 'availability_multiplier', i, 1.0),
                    'environment_multiplier': _trace_number(trace, 'environment_multiplier', i, 1.0),
                    'environment_status': _trace_value(trace, 'environment_status', i, 'feature disabled'),
                    'efficiency_denominator': _trace_value(trace, 'efficiency_denominator', i),
                    'efficiency_rate': _trace_number(trace, 'efficiency_rate', i),
                    'efficiency_evidence': _trace_number(trace, 'efficiency_evidence', i),
                    'efficiency_matchup_multiplier': _trace_number(
                        trace, 'efficiency_matchup_multiplier', i),
                    'efficiency_defense_estimator': _trace_value(
                        trace, 'efficiency_defense_estimator', i),
                    'two_year_td_prior': bool(_trace_value(trace, 'two_year_td_prior', i, False)),
                    'pre_vacancy_projection': _trace_number(trace, 'pre_vacancy_projection', i, 0.0),
                    # Updated to the post-vacancy board value below.
                    'vacancy_delta': 0.0,
                    'final_projection': round(float(row.get(stat, 0.0)), 3),
                    'projection': round(float(row.get(stat, 0.0)), 3),
                }
            _pgl_cur, _odl_cur = player_game_log_current[i], opponent_defense_log_current[i]
            _pgl_prior, _odl_prior = player_game_log_prior[i], opponent_defense_log_prior[i]
            _pgl_prior2, _odl_prior2 = player_game_log_prior2[i], opponent_defense_log_prior2[i]

            def _records(frame):
                return frame.to_dict('records') if isinstance(frame, pd.DataFrame) else []

            explanations[(player, pos, row['Team'])] = {
                'player': player, 'position': pos, 'team': row['Team'], 'opponent': row['Opponent'],
                'target_week': int(week), 'as_of_week': int(as_of_week), 'season_year': int(year),
                'scoring_mode': scoring_mode,
                'defense_matchup': defense_matchup_by_opponent.get(str(row['Opponent'])),
                'distribution': player_distribution(
                    pos, position_rank.iloc[i], float(row['Calibrated Model Proj Pts']),
                    role_confidence=float(row['Role Confidence']),
                    own_game_points=_eligible_fantasy_points(_pgl_cur, _pgl_prior)),
                'raw_points': float(row['Raw Model Proj Pts']),
                'calibrated_points': float(row['Calibrated Model Proj Pts']),
                'stat_line': {stat: float(row.get(stat, 0.0)) for stat in stats if stat in out.columns},
                # Deep Dive tab source data, keyed by season year so the UI
                # can offer a season selector instead of only ever showing
                # whichever season this branch happened to run as - per-game
                # log for this player (raw + defense-adjusted value per stat,
                # excluded games flagged) and, for that season's upcoming/
                # most-recent opponent read, the per-week allowed-to-every-
                # offense log the matchup multiplier was itself built from.
                # list-of-record-dicts, same plain-data contract as 'vacancy'
                # below (rebuilt into a DataFrame in the UI layer, never a
                # raw DataFrame stored here). The CURRENT season entry is
                # honestly empty at cold start (no games played yet); the
                # PRIOR season entry is populated in every branch.
                'game_log_by_season': {
                    int(year): _records(_pgl_cur),
                    int(year) - 1: _records(_pgl_prior),
                    int(year) - 2: _records(_pgl_prior2),
                },
                'defense_weekly_log_by_season': {
                    int(year): _records(_odl_cur),
                    int(year) - 1: _records(_odl_prior),
                    int(year) - 2: _records(_odl_prior2),
                },
                # Audit-only PFF slot/wide/inline + man/zone evidence for the
                # Context tab. Every *_candidate_multiplier here is a preview
                # only - see pff_alignment.py's module docstring - nothing in
                # this dict has ever been multiplied into a scored stat.
                'alignment_scheme_evidence': {
                    'player_slot_rate': row.get('_profile_slot_alignment_rate'),
                    'player_wide_rate': row.get('_profile_wide_alignment_rate'),
                    'player_inline_rate': row.get('_profile_inline_alignment_rate'),
                    'player_alignment_available': bool(row.get('_profile_alignment_available', False)),
                    'player_alignment_sample_weight': row.get('_profile_alignment_sample_weight'),
                    'defense_slot_candidate_multiplier': row.get(
                        '_profile_alignment_defense_targets_candidate_multiplier'),
                    'defense_alignment_candidate_available': bool(
                        row.get('_profile_alignment_defense_candidate_available', False)),
                    'defense_alignment_reason': row.get('_profile_alignment_defense_reason', ''),
                    'player_man_route_share': row.get('_scheme_profile_man_route_share'),
                    'player_zone_route_share': row.get('_scheme_profile_zone_route_share'),
                    'player_scheme_available': bool(row.get('_scheme_profile_scheme_available', False)),
                    'player_scheme_sample_weight': row.get('_scheme_profile_scheme_sample_weight'),
                    'defense_man_candidate_multiplier': row.get(
                        '_scheme_profile_scheme_defense_targets_candidate_multiplier'),
                    'defense_scheme_candidate_available': bool(
                        row.get('_scheme_profile_scheme_defense_candidate_available', False)),
                    'defense_scheme_reason': row.get('_scheme_profile_scheme_defense_reason', ''),
                },
                'role': role | {
                    'role_confidence': float(row['Role Confidence']),
                    'expected_snap_share': float(row['Expected Snap Share']),
                    'role_change_confidence': float(row['Role Change Confidence']),
                    # What _role_confidence actually read to produce the
                    # number above - see that function's docstring. NaN
                    # recent_snap_pct/route_rate means that ingredient
                    # wasn't available for this player (fell through to the
                    # 0.5 default, or to snap share alone).
                    'role_confidence_recent_snap_pct': (
                        None if pd.isna(row.get('_role_confidence_recent_snap_pct'))
                        else float(row['_role_confidence_recent_snap_pct'])),
                    'role_confidence_games_sampled': (
                        None if pd.isna(row.get('_role_confidence_games_sampled'))
                        else int(row['_role_confidence_games_sampled'])),
                    'role_confidence_route_rate': (
                        None if pd.isna(row.get('_role_confidence_route_rate'))
                        else float(row['_role_confidence_route_rate'])),
                    'role_confidence_method': row.get('_role_confidence_method') or 'unknown',
                    'qb1_workload_override': bool(row['QB1 Workload Override']),
                    'qb1_selection_required': bool(row['QB1 Selection Required']),
                    'qb_projected_starter': bool(row['QB Projected Starter']),
                    'starter_source': row['QB1 Workload Source'],
                    'partial_game_exclusions': int(row['Partial-Game Exclusions']),
                    'returning_role_restored': bool(returning_role_restored[i]),
                    'returning_role_reason': returning_role_reason[i],
                    'pre_absence_snap_share': (
                        None if not np.isfinite(player_pre_absence_share[i]) else float(player_pre_absence_share[i])),
                    'interrupted_season': bool(player_interrupted_season[i]),
                    'preseason_role_source': preseason_role_source[i],
                    'ourlads_role_floor_applied': bool(ourlads_role_floor_applied[i]),
                    'ourlads_role_available_rank': (
                        None if not np.isfinite(ourlads_role_rank[i]) else float(ourlads_role_rank[i])),
                    'ourlads_role_floor': (
                        None if not np.isfinite(ourlads_role_floor[i]) else float(ourlads_role_floor[i])),
                    'ourlads_role_position_label': ourlads_role_label[i],
                    # Source styling is deliberately not an availability
                    # decision.  It remains visible so a user can verify the
                    # chart snapshot, while the separate availability object
                    # records the only source allowed to zero a player.
                    'ourlads_source_name': ourlads_source_name[i],
                    'ourlads_source_status': ourlads_source_status[i],
                    'ourlads_source_status_warning': ourlads_source_status_warning[i],
                    'identity_match_method': ourlads_identity_match_method[i],
                    'identity_match_confidence': ourlads_identity_match_confidence[i],
                    'identity_match_warning': ourlads_identity_match_warning[i],
                    'matched_identity_key': ourlads_matched_identity_key[i],
                    'rb_allocator_applied': bool(rb_allocator_applied[i]),
                    'core_rb': bool(rb_core[i]),
                    'rb_snap_capacity': (
                        None if not np.isfinite(rb_snap_capacity[i]) else float(rb_snap_capacity[i])),
                    'rb_carry_capacity': (
                        None if not np.isfinite(rb_carry_capacity[i]) else float(rb_carry_capacity[i])),
                    'rb_target_capacity': (
                        None if not np.isfinite(rb_target_capacity[i]) else float(rb_target_capacity[i])),
                    'allocated_carries': (
                        None if not np.isfinite(rb_carry_allocation[i]) else float(rb_carry_allocation[i])),
                    'allocated_targets': (
                        None if not np.isfinite(rb_target_allocation[i]) else float(rb_target_allocation[i])),
                    'other_rb_snap_remainder': (
                        None if not np.isfinite(rb_other_snap[i]) else float(rb_other_snap[i])),
                    'other_rb_carry_remainder': (
                        None if not np.isfinite(rb_other_carries[i]) else float(rb_other_carries[i])),
                    'other_rb_target_remainder': (
                        None if not np.isfinite(rb_other_targets[i]) else float(rb_other_targets[i])),
                    'rb_allocation_source': rb_allocation_source[i],
                    'rb_allocation_eligibility_reason': rb_allocation_eligibility_reason[i],
                    'rb_established_incumbent_backstop': bool(rb_established_incumbent_backstop[i]),
                    'rb_role_segment_status': rb_segment_status[i],
                    'rb_segment_pre_absence_snap_share': (
                        None if not np.isfinite(rb_segment_pre_absence_snap_share[i])
                        else float(rb_segment_pre_absence_snap_share[i])),
                    'rb_segment_gap_games': (
                        None if not np.isfinite(rb_segment_gap_games[i]) else float(rb_segment_gap_games[i])),
                    'rb_segment_return_snap_share': (
                        None if not np.isfinite(rb_segment_return_snap_share[i])
                        else float(rb_segment_return_snap_share[i])),
                    'rb_interrupted_incumbent_credit': float(rb_interrupted_incumbent_credit[i]),
                    'rb_shared_healthy_lead_score': float(rb_shared_healthy_lead_score[i]),
                    'rb_replacement_only_downweight': float(rb_replacement_only_downweight[i]),
                    'alignment_note': (
                        f"{role.get('alignment_semantics', 'slot/non-slot alignment')} from "
                        f"{role.get('source_kind', 'local PFF archive')} "
                        f"(weeks {role.get('source_weeks', '—')}); defensive residual is neutral pending backtest."
                        if bool(role.get('alignment_available')) else
                        f"{pff_alignment_contract['status']}; alignment matchup is neutral."
                    ),
                },
                'availability': {
                    'status': row['Injury Status'] or 'No current designation',
                    'plays_probability': float(row['Availability']),
                    'workload_if_active': float(row['Workload If Active']),
                    'source': row.get('_availability_source', 'not recorded'),
                    'match_method': row.get('_availability_match_method', 'not recorded'),
                    'note': row.get('_availability_note', ''),
                },
                'stats': stat_detail,
                'calibration': {
                    'enabled': 'calibration' in feats and pos in WEEKLY_CALIBRATION,
                    'slope': float(WEEKLY_CALIBRATION.get(pos, (1.0, 0.0))[0]),
                    'intercept': float(WEEKLY_CALIBRATION.get(pos, (1.0, 0.0))[1]),
                    'raw_points': float(row['Raw Model Proj Pts']),
                    'displayed_points': float(row['Calibrated Model Proj Pts']),
                    'delta': float(row['Calibrated Model Proj Pts'] - row['Raw Model Proj Pts']),
                },
                'data_contract': source_contract.copy(),
                'features': sorted(feats),
            }
        all_rows.append(out)

    if not all_rows:
        return pd.DataFrame(), {'reason': f'No projectable players found for week {week}.'}
    result = pd.concat(all_rows, ignore_index=True).sort_values('Model Proj Pts', ascending=False).reset_index(drop=True)

    # Team-level target conservation, BEFORE vacancy: fits every team's
    # RB/WR/TE targets (and receptions/yards/TDs, rescaled by the same
    # factor) to that team's own projected pass attempts. Runs on the
    # ASSEMBLED board because a team's QB attempts and its receivers are
    # computed in separate position passes above - see
    # data/pass_capacity_allocator.py for why this specific defect (targets
    # running 1.19x-1.56x attempts, entirely in the tail beyond a team's
    # real 6-8 pass catchers) never shows up in a star-player spot check.
    # Deliberately BEFORE the vacancy pass below: vacancy should redistribute
    # a departing player's share of an already-realistic team total, not add
    # on top of one that still needs fitting to reality.
    pass_capacity_ledger, pass_capacity_adjusted, pass_capacity_room = [], False, []
    if 'v2_pass_capacity' in feats:
        result, pass_capacity_ledger_df = apply_pass_capacity_conservation(
            result, prior_history=prior_stats, team_col=prior_team_col)
        pass_capacity_ledger = pass_capacity_ledger_df.to_dict('records')
        pass_capacity_adjusted = bool(
            not pass_capacity_ledger_df.empty
            and (pass_capacity_ledger_df['capacity_source'] != 'no capacity signal').any())
        # Per-player room detail (see apply_pass_capacity_conservation's own
        # comment for why this rides on .attrs instead of a return value) -
        # who else is in this player's team+group and what the same
        # conservation pass did to each of them, not just the team totals.
        _room_detail_df = pass_capacity_ledger_df.attrs.get('player_detail')
        if _room_detail_df is not None and not _room_detail_df.empty:
            pass_capacity_room = _room_detail_df.to_dict('records')

    # Snapshot the board right here, AFTER pass-capacity conservation but
    # BEFORE the vacancy pass below, keyed by (Player, Pos, Team) rather than
    # positional index - both this pass and the vacancy functions below can
    # precede a sort/reset_index further down, so an index-aligned snapshot
    # would silently misattribute rows once that happens. Without this, the
    # 'vacancy_delta' reported in the popup below conflated TWO unrelated
    # mechanisms into one number: apply_pass_capacity_conservation shrinking
    # a team's tail pass-catchers toward a realistic team target budget (runs
    # on every board with 'v2_pass_capacity' on, whether or not anyone is
    # hurt) and the actual OUT-teammate vacancy redistribution below. A
    # low-usage receiving back (e.g. a bruiser RB whose team's WR/TE corps
    # already claims the trusted tier) could get his already-small target
    # share visibly squeezed by capacity conservation alone and have the UI
    # blame "vacancy" for it - a real mislabeling bug, confirmed 2026-08-24.
    _capacity_snapshot_stats = ('targets', 'receptions', 'receiving_yards', 'receiving_tds')
    post_capacity_snapshot = {}
    if any(c in result.columns for c in _capacity_snapshot_stats):
        for _, _snap_row in result.iterrows():
            post_capacity_snapshot[(_snap_row['Player'], _snap_row['Pos'], _snap_row['Team'])] = {
                stat: float(_snap_row[stat]) for stat in _capacity_snapshot_stats
                if stat in result.columns and pd.notna(_snap_row[stat])
            }

    vacancy_adjusted, vacancy_ledger = 0, []
    if 'v2_vacancy' in feats and injury_profiles:
        if 'v2_preseason_rb_allocator' in feats:
            # The V2 allocator owns all RB carry/target vacancy, and owns
            # WR/TE target vacancy so a departed pass catcher cannot leak
            # into a running back.  Keep the old helper only for the QB
            # handoff, where a single named replacement remains the right
            # football rule.
            injury_provenance = {
                player: {
                    'year': profile.get('source_year'),
                    'source': profile.get('source', 'target-season injury report'),
                }
                for player, profile in injury_profiles.items()
            }
            result, rb_vacancy_frame = redistribute_rb_vacancy_with_allocator(
                result, injury_profiles, as_of_year=year,
                injury_provenance=injury_provenance,
            )
            rb_vacancy_ledger = (rb_vacancy_frame.to_dict('records')
                                 if rb_vacancy_frame is not None and not rb_vacancy_frame.empty else [])
            result, qb_vacancy_adjusted, qb_vacancy_ledger = redistribute_v2_vacated_usage(
                result, injury_profiles, skip_rb=True, skip_receivers=True)
            vacancy_ledger = rb_vacancy_ledger + qb_vacancy_ledger
            vacancy_adjusted = int(any(float(entry.get('allocated', 0.0) or 0.0) > 0
                                        for entry in rb_vacancy_ledger)) + int(qb_vacancy_adjusted)
        else:
            result, vacancy_adjusted, vacancy_ledger = redistribute_v2_vacated_usage(result, injury_profiles)
    elif 'teammate_vacancy' in feats and injury_mult:
        # V1 behavior remains available for its existing measured baseline.
        result, vacancy_adjusted = redistribute_vacated_usage(result, injury_mult)
    result = result.drop(columns=[c for c in result.columns if c.startswith('_full_')])
    if 'v2_pass_capacity' in feats:
        result = clamp_dependent_stats(result)
    # Re-score whenever EITHER pass moved a stat line, not vacancy alone:
    # pass_capacity_conservation runs on every board (not just one with an
    # OUT player) and, unlike vacancy, was never covered by the existing
    # trigger here - a team with nobody injured that week would otherwise
    # keep pre-conservation points sitting next to a post-conservation stat
    # line, the same displayed-vs-real mismatch class of bug HANDOFF gotcha
    # #40 (the cross-position NaN fix) already documents one instance of.
    if vacancy_adjusted or pass_capacity_adjusted:
            # .fillna(0.0) IS THE LOAD-BEARING PART OF THIS LINE. Re-scoring
            # happens on the ASSEMBLED frame, whose columns are the union of
            # four positions' stat lists - so a receiver's row carries NaN
            # for every passing stat and a quarterback's carries NaN for
            # every receiving one. score_projected_stats reads its inputs
            # with `proj.get(stat, 0)`, which returns the NaN for a key that
            # EXISTS and is NaN, so the whole sum goes NaN - and
            # `max(0.0, nan)` is 0.0, not nan, so it doesn't even look like
            # an error downstream. Confirmed live before this fix: every
            # RB, WR and TE on a real 2026 week-1 board projected exactly
            # 0.00 points while carrying a perfectly sensible stat line, and
            # the position-rank column duly labelled a 0.0-point player
            # "RB1". The per-position scoring above is unaffected because it
            # only ever passes that position's own stat list.
            stat_cols = [c for c in _ALL_PROJECTION_STATS if c in result.columns]
            recomputed = [max(0.0, score_projected_stats(d, scoring_mode))
                          for d in result[stat_cols].fillna(0.0).to_dict('records')]
            result['Raw Model Proj Pts'] = np.round(recomputed, 2)
            if 'calibration' in feats:
                slopes = result['Pos'].map(lambda p: WEEKLY_CALIBRATION.get(p, (1.0, 0.0)))
                recomputed = [min(v, sl * v + ic) for v, (sl, ic) in zip(recomputed, slopes)]
            result['Model Proj Pts'] = np.round(np.clip(recomputed, 0.0, None), 2)
            result['Calibrated Model Proj Pts'] = result['Model Proj Pts']
            result = result.sort_values('Model Proj Pts', ascending=False).reset_index(drop=True)

    # Raw per-week slot/wide/inline defense-allowed evidence, grouped for
    # per-player lookup by (defense_team, position) - the pre-shrinkage
    # ingredients behind alignment_scheme_evidence's season-aggregate
    # candidate multipliers (see pff_alignment_defense_team_games's own
    # comment above). WR/TE only - team_games is empty for every other
    # position family since ALIGNMENT_DEFENSE_SUPPORTED_POSITIONS is WR/TE.
    #
    # Split by season (year/year-1/year-2) as of 2026-08-26, so the Deep
    # Dive tab's existing season radio also drives this table - it
    # previously only ever showed whichever single season happened to be
    # loaded for SCORING above (year-1 at cold start, year in-season),
    # silently ignoring a 2024 selection. This is a DISPLAY-only load,
    # independent of the scoring load above (which is untouched) - a
    # completed prior season may be loaded twice (once for scoring at cold
    # start, once here, since neither load is cached), acceptable for a
    # once-per-board cost. Each season's frame also gets the same leave-
    # one-out expected value (`_expected_value` - "that offense's own
    # average") and a recency weight (`_recency_weight`, decaying from that
    # season's own last included week) attached for the table below.
    defense_alignment_log_by_team_pos_by_season: dict[int, dict[tuple[str, str], list[dict]]] = {}
    # pff_alignment_defense_team_games (loaded above for SCORING) already
    # IS year-1's data at cold start (the season-prior fallback), or year's
    # own in-season data otherwise - reuse it under whichever year it
    # actually represents rather than reloading, but never relabel it as a
    # season it isn't: at cold start there is no real 'year' (e.g. 2026)
    # archive yet, so that slot is loaded fresh below (correctly empty,
    # same as every other cold-start 2026 lookup in this file) instead of
    # silently showing year-1's numbers under the current season's tab.
    _alignment_team_games_by_year = ({int(year) - 1: pff_alignment_defense_team_games} if cold_start
                                     else {int(year): pff_alignment_defense_team_games})
    for _display_year in (int(year), int(year) - 1, int(year) - 2):
        if _display_year in _alignment_team_games_by_year:
            continue
        try:
            _display_result = load_weekly_alignment_defense_profiles(
                _display_year, (as_of_week if _display_year == int(year) else PFF_ALIGNMENT_DEFENSE_COLD_START_AS_OF_WEEK),
                load_schedule(_display_year, include_postseason=(_display_year != int(year))),
                include_postseason=(_display_year != int(year)))
            _alignment_team_games_by_year[_display_year] = _display_result.team_games
        except Exception:
            _alignment_team_games_by_year[_display_year] = pd.DataFrame()
    for _display_year, _adtg in _alignment_team_games_by_year.items():
        if _adtg is None or _adtg.empty:
            defense_alignment_log_by_team_pos_by_season[_display_year] = {}
            continue
        _adtg = _attach_offense_leave_one_out_baselines(_adtg)
        _weeks_numeric = pd.to_numeric(_adtg['source_week'], errors='coerce')
        _as_of_for_weight = (
            as_of_week if _display_year == int(year) and not cold_start
            else int(_weeks_numeric.max()) + 1
        )
        _adtg['_recency_weight'] = defense_recency_weights(_weeks_numeric, _as_of_for_weight)
        by_team_pos: dict[tuple[str, str], list[dict]] = {}
        for (d_team, d_pos), group in _adtg.groupby(['defense_team', 'position'], observed=True):
            by_team_pos[(str(d_team), str(d_pos))] = (
                group.sort_values(['source_week', 'alignment', 'stat']).to_dict('records'))
        defense_alignment_log_by_team_pos_by_season[_display_year] = by_team_pos

    # Preserve the team-level preseason allocation while the internal RB
    # fields are still available.  Each player's popup can therefore show
    # both the capacity ledger and every teammate allocation without making
    # the main ranking table wider.
    rb_team_allocations = {}
    if '_rb_core' in result.columns:
        allocation_columns = [
            'Player', '_functional_position', 'Expected Snap Share', '_rb_core',
            '_rb_carry_allocation_share', '_rb_target_allocation_share',
            'rushing_attempts', 'targets',
        ]
        present = [column for column in allocation_columns if column in result.columns]
        rb_rows = result[result['Pos'].eq('RB')].copy()
        for team, group in rb_rows.groupby('Team', observed=True):
            display = group[present].rename(columns={
                '_functional_position': 'Functional role',
                '_rb_core': 'Core RB',
                '_rb_carry_allocation_share': 'Preseason carry share',
                '_rb_target_allocation_share': 'Preseason target share',
                'rushing_attempts': 'Projected carries',
                'targets': 'Projected targets',
            })
            rb_team_allocations[str(team)] = display.to_dict('records')

    # A dialog must report the final board value, including an availability
    # or vacancy adjustment, without recalculating in the UI. The boom/bust
    # band was originally built earlier in the per-position loop (line ~6100
    # above) off the PRE-conservation/PRE-vacancy 'Calibrated Model Proj Pts'
    # - for a player whose points moved materially in either pass, the band
    # would center on a number that no longer matches what the board (and
    # this same dict's own 'calibrated_points' below) actually display.
    # Recomputed here, once, against the FINAL board so the two always agree.
    final_pos_rank = result.groupby('Pos')['Model Proj Pts'].rank(ascending=False, method='min')
    for idx, row in result.iterrows():
        key = (row['Player'], row['Pos'], row['Team'])
        detail = explanations.get(key)
        if detail is None:
            continue
        detail['raw_points'] = float(row.get('Raw Model Proj Pts', row['Model Proj Pts']))
        detail['calibrated_points'] = float(row['Model Proj Pts'])
        detail['distribution'] = player_distribution(
            row['Pos'], final_pos_rank.loc[idx], detail['calibrated_points'],
            role_confidence=float(row.get('Role Confidence', np.nan)),
            own_game_points=_eligible_fantasy_points(*detail.get('game_log_by_season', {}).values()))
        # The vacancy pass mutates the assembled ranking frame after the
        # per-position explanation was created.  Refresh every displayed stat
        # from that final frame so the popup is a literal decomposition of the
        # row the user selected, not a pre-vacancy approximation.
        final_stat_line = {}
        row_capacity_snapshot = post_capacity_snapshot.get(key, {})
        for stat, values in detail.get('stats', {}).items():
            final_value = pd.to_numeric(pd.Series([row.get(stat, 0.0)]), errors='coerce').fillna(0.0).iloc[0]
            final_value = float(final_value)
            pre_vacancy = values.get('pre_vacancy_projection')
            pre_vacancy = float(pre_vacancy) if pre_vacancy is not None else final_value
            # Split at the post-capacity-conservation snapshot above so each
            # mechanism's own effect is reported separately instead of one
            # number blaming "vacancy" for both. Falls back to the old
            # combined behavior (all of it attributed to vacancy_delta) for a
            # stat the snapshot didn't cover (passing stats - conservation
            # only touches RB/WR/TE receiving volume).
            if stat in row_capacity_snapshot:
                post_capacity = row_capacity_snapshot[stat]
                values['pass_capacity_delta'] = round(post_capacity - pre_vacancy, 3)
                values['vacancy_delta'] = round(final_value - post_capacity, 3)
            else:
                values['pass_capacity_delta'] = 0.0
                values['vacancy_delta'] = round(final_value - pre_vacancy, 3)
            values['final_projection'] = round(final_value, 3)
            # Retain the original field for consumers of the V2 explanation
            # payload while making it agree with ``final_projection``.
            values['projection'] = round(final_value, 3)
            final_stat_line[stat] = final_value
        if final_stat_line:
            detail['stat_line'] = final_stat_line
        calibration = detail.get('calibration', {})
        calibration['raw_points'] = detail['raw_points']
        calibration['displayed_points'] = detail['calibrated_points']
        calibration['delta'] = detail['calibrated_points'] - detail['raw_points']
        detail['calibration'] = calibration
        detail['vacancy'] = [entry for entry in vacancy_ledger if entry['team'] == str(row['Team'])]
        detail['vacancy_adjusted'] = bool(vacancy_adjusted and any(
            entry['team'] == str(row['Team']) and entry['allocated'] > 0 for entry in vacancy_ledger))
        detail['rb_team_allocation'] = rb_team_allocations.get(str(row['Team']), [])
        detail['rb_capacity_ledger'] = [
            entry for entry in rb_allocation_ledger if entry.get('team') == str(row['Team'])
        ]
        detail['pass_capacity_ledger'] = [
            entry for entry in pass_capacity_ledger if entry.get('team') == str(row['Team'])
        ]
        _own_group = 'WR/TE' if row['Pos'] in ('WR', 'TE') else row['Pos']
        detail['pass_capacity_room'] = sorted(
            (entry for entry in pass_capacity_room
             if entry.get('team') == str(row['Team']) and entry.get('position_group') == _own_group),
            key=lambda entry: entry.get('targets_before', 0.0), reverse=True,
        )
        detail['defense_alignment_weekly_log_by_season'] = {
            _yr: _by_team_pos.get((str(row['Opponent']), str(row['Pos'])), [])
            for _yr, _by_team_pos in defense_alignment_log_by_team_pos_by_season.items()
        }
    # A fixed x-axis per position for the "Range of outcomes" chart -
    # explicit request, 2026-08-24: every player's curve was being auto-fit
    # to the full chart width (0 to that player's own P90 + margin), so a
    # deep-bench player's narrow, low band and a workhorse's wide, high one
    # rendered as the same-looking hump - correct in proportion, but visually
    # indistinguishable, since matplotlib rescaled each one's own axis to
    # match. Anchoring every player at this position's ACTUAL widest band
    # (this week's real max P90 across the position, not a guessed constant)
    # instead lets true differences in width/location show up as a shorter,
    # further-left curve rather than being silently rescaled away.
    position_axis_max: dict[str, float] = {}
    for detail in explanations.values():
        distribution = detail.get('distribution')
        if not distribution:
            continue
        widest = max(max(distribution['points'].values()),
                    max(distribution['position_points'].values()))
        position_axis_max[detail['position']] = max(
            position_axis_max.get(detail['position'], 0.0), widest)
    for detail in explanations.values():
        distribution = detail.get('distribution')
        if distribution:
            distribution['axis_max'] = position_axis_max.get(detail['position'], 0.0) + 1.0
    # Allocation fields are implementation detail, not ranking columns.
    result = result.drop(columns=[
        column for column in result.columns
        if column.startswith('_rb_') or column == '_functional_position'
    ], errors='ignore')
    meta = {'reason': None, 'year': year, 'week': week, 'as_of_week': as_of_week,
            'players': int(len(result)), 'scoring': scoring_mode, 'cold_start': cold_start,
            'features': sorted(feats), 'vacancy_adjusted': vacancy_adjusted,
            'source_contract': source_contract, 'explanations': explanations,
            'vacancy_ledger': vacancy_ledger,
            'rb_allocation_ledger': rb_allocation_ledger,
            'pass_capacity_ledger': pass_capacity_ledger}
    return result, meta
