"""Conservative, team-constrained preseason running-back role allocation.

This module intentionally contains no Streamlit or network code.  It takes a
small, auditable candidate table and returns a reconciled allocation for core
running backs plus an explicit ``other RB`` remainder.  That makes it possible
to test role arithmetic separately from the projection model and, crucially,
prevents an unknown reserve from receiving a league-median role simply because
he is listed as an RB in a roster feed.

The allocator is a preseason/V2 component.  It is not a claim that a depth
chart alone knows future usage: historic active role remains its main input,
while Ourlads order, continuity, draft capital, and interrupted-season evidence
are bounded tie-breakers.  Carries and targets are allocated independently.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from data.player_aliases import stable_roster_identity_keys


INELIGIBLE_ROSTER_STATUSES = frozenset({"RET", "CUT", "RES", "FA", "SUS", "NFI", "PUP"})
DEFAULT_CORE_RB_SNAP_CAPACITY = 1.00
DEFAULT_RB_CARRY_CAPACITY = 21.0
DEFAULT_RB_TARGET_CAPACITY = 5.0
MAX_INDIVIDUAL_CORE_RB_SNAP_SHARE = 0.86
# Week 1 ONLY (see `week` param below) tighter individual snap-share cap.
# A real back can and does clear 86%+ in-season when volume is genuinely
# concentrated for a few weeks - that is not this constant's business. Week 1
# is different: it is a full-season PROJECTION built off limited, often
# thin evidence, and 2026-09-04 live review found it landing two backs
# (Gibbs, McCaffrey) above any real 2025 workhorse's actual full-season
# snap share on the strength of that thin evidence alone. Same redistribution
# machinery as MAX_INDIVIDUAL_CORE_RB_SNAP_SHARE (_bounded_allocation) - the
# trimmed share moves to teammates, never vanishes.
RB_WEEK1_MAX_INDIVIDUAL_SNAP_SHARE = 0.82
# How far a WEEK-1, no-same-team-backstop player's carry/target share may
# sit from his own snap share before it gets pulled back (see
# _bound_share_toward_snap and the cross-team-evidence note in the per-team
# loop). Sized to sit ABOVE a real role-split's normal range - Aaron Jones
# (established, ungated) legitimately runs -6.6/+4.8 points off his own
# snap share in the receiving-back direction - so a genuine, still-thin
# same-team split signal is not flattened; it only stops the specific
# failure this exists for, an unrelated OLD team's rate silently costing a
# new arrival ~20 points on BOTH axes at once (Travis Etienne, 2026-09-04).
RB_WEEK1_CROSS_TEAM_SHARE_TOLERANCE = 0.08
VACANCY_SURVIVAL = 0.80

# Season-scoped RB eligibility, redone 2026-08-24 at explicit request after a
# real miscall: the old bar ("4+ games at >=10% base, ANYWHERE in a player's
# career, in EITHER season") let a player whose 2025 role was tiny and
# irrelevant on his CURRENT team still count as a credible core-RB candidate
# and pull real share away from an unambiguous starter (measured live -
# Dameon Pierce, 5 games at an 8.4% active-game share for a DIFFERENT team
# in 2025, was still splitting PHI's backfield with Saquon Barkley). The new
# rule is a two-step season check instead: does 2025 alone clear a real bar,
# and if not, was 2024 a genuine starter's season interrupted since (the
# Malik-Nabers-shaped case, not "any four games")? This governs `strong_
# evidence` below - it is deliberately SEPARATE from `base` (used for the
# actual SCORE magnitude), which stays the caller's already-blended, best
# point-estimate share; this only gates who is credible to compete at all.
RB_ELIGIBILITY_MIN_GAMES_2025 = 5
RB_ELIGIBILITY_MIN_SHARE_2025 = 0.15
RB_ELIGIBILITY_PRIOR2_MIN_GAMES = 4
RB_ELIGIBILITY_PRIOR2_STARTER_SHARE = 0.40

# Minimum prior combined carry+target involvement for weak_fb_evidence's
# "has real evidence" side - see that guard's own comment. A true fullback
# can clear both prior_games>0 and a real SNAP share (his job is blocking)
# while touching the ball almost never, so neither of those alone tells
# blocking from ball-carrying apart. Sized off Patrick Ricard's real 2025
# line (5 games, 45.6% snap share, 0.2 carries + 0.4 targets/game = 0.6
# combined) found 2026-09-04 clearing the games/snap-share check and
# drawing real core-RB capacity; a genuine complementary back clears this
# by a wide margin even in a thin committee role.
RB_MIN_PRIOR_TOUCH_RATE_FOR_FB_GUARD = 1.0     # combined carries+targets per game
RB_MIN_PRIOR_TOUCH_SHARE_FOR_FB_GUARD = 0.05   # combined carry+target SHARE, when that's what's on hand

# Without an imported Ourlads chart, every "eligible" candidate on a team
# used to compete on raw score alone - no penalty at all for being the
# team's 3rd (or 4th) back, the exact discount a real chart rank already
# applies below. Reuses THAT SAME discount schedule (1.08/0.98/0.88), just
# keyed off score-derived rank instead of a chart rank, so a team with no
# chart still down-weights a genuine third option instead of letting the
# 1.45 concentration exponent be its only defense.
SCORE_RANK_DISCOUNT = {1: 1.08, 2: 0.98, 3: 0.88}
SCORE_RANK_DISCOUNT_DEFAULT = 0.80
VACANCY_MAX_GROWTH = 2.00

# How many extra chart slots an injured-top-three backfield can promote into
# the "credible core RB" set. 1 => one absence reaches the chart RB4, never the
# RB5. See the `rank_ceiling` block in allocate_preseason_rb_roles.
RB_CHART_VACANCY_EXTENSION_MAX = 1

# WR/TE vacancy "pecking order" reshape (v2_receiver_vacancy_pecking_order).
# The default receiver-vacancy split weights recipients by their CURRENT
# projected targets, which routes most of a departed complementary pass
# catcher's work to the team's alpha (WR1/TE1) - a player who has usually
# never sustained that target share (measured live 2026-08-30: HOU's Nico
# Collins projected ~12 targets with Jayden Higgins out). Real "next man up"
# behaviour gives the alpha a small bump, the bulk to the one or two players
# immediately behind the injured man, a real but smaller piece to the clear
# reserve, and a fast-decaying trickle past that - nothing to the deep bench.
# Recipients are ranked by current projected volume (a pecking-order proxy):
#   - RANK_DECAY (<1): geometric falloff applied from the top backup (rank 2)
#     downward, so rank 3 gets DECAY x rank 2, rank 4 gets DECAY^2, ...
#   - LEAD_SHARE: the current lead's weight, as a fraction of the rank-2
#     weight (which is 1.0) - this is what holds the alpha to a ~10-15% bump.
#   - CROSS_POS_WEIGHT (<1): a departed WR's targets favour other WRs over
#     TEs (and vice-versa); a cross-position recipient's weight is scaled by
#     this.
#   - PARTICIPATION_RANKS: recipients ranked deeper than this get nothing.
# ABS_GROWTH_FLOOR lets a low-projected backup actually step into a vacated
# role: the multiplicative VACANCY_MAX_GROWTH cap alone limits a 2-target WR
# to +2 no matter how much room opened up.
RECEIVER_VACANCY_RANK_DECAY = 0.62
RECEIVER_VACANCY_LEAD_SHARE = 0.24
RECEIVER_VACANCY_CROSS_POS_WEIGHT = 0.30
RECEIVER_VACANCY_PARTICIPATION_RANKS = 8
RECEIVER_VACANCY_ABS_GROWTH_FLOOR = 2.0

# Explicit depth-chart-order nudge, added 2026-08-24 per request: even after
# SCORE_RANK_DISCOUNT above, a real preseason-committee RB3 can still clear
# 20%+ of team snaps once concentration and evidence scoring finish - well
# past what an actual NFL RB3 sees outside an injury or a blowout script
# ("generally an RB3 is not going to get more than 10% of snaps"). This is a
# SECOND, later pass: a bounded, PARTIAL pull of whatever share a rank>=3
# back holds above RB_DEPTH_RANK_SNAP_TARGET_RANK3 toward the team's rank-1
# back, conserving the team's total core-RB snap allocation exactly (a
# transfer, not new capacity). Deliberately partial (RB_DEPTH_RANK_SNAP_PULL
# < 1) and capped small per team (RB_DEPTH_RANK_SNAP_NUDGE_CAP) rather than a
# hard ceiling at the target - a real committee back with strong standalone
# evidence keeps most of his share; "a few outliers is fine to leave" was
# explicit. RB2 is deliberately untouched - only rank 1 (receiver) and
# rank>=3 (donor) participate, matching "mild uptick to RB1 and downtick to
# RB3" exactly as asked, not a general re-flattening of the whole backfield.
RB_DEPTH_RANK_SNAP_TARGET_RANK3 = 0.10
RB_DEPTH_RANK_SNAP_PULL = 0.5
RB_DEPTH_RANK_SNAP_NUDGE_CAP = 0.05

# A second, sibling correction, added 2026-08-24 per explicit request: the
# nudge above only guards RB1-vs-RB3+; it left a real case where a chart
# rank>=3 back's OWN evidence score was strong enough to outrun the chart
# rank-2 back next to him (measured live - Kimani Vidal, chart RB3, real
# 2024/2025 game evidence, ended up at 20.7% team share vs. Keaton Mitchell,
# chart RB2, at 9.4%). "The RB3 listed should not be so much higher than the
# RB2 listed" is the same shape of ask as the RB1 case, so this reuses the
# identical bounded/partial/capped/conserved mechanism, just keyed to RB2 as
# the receiver instead of RB1 - a real committee outlier still keeps most of
# an earned lead; only the excess above what RB2 himself holds is pulled.
RB_DEPTH_RANK2_ORDER_PULL = 0.5
RB_DEPTH_RANK2_ORDER_NUDGE_CAP = 0.05

# When a chart rank<=3 teammate is unavailable (a real target-week 'out'),
# the next chart slot is a genuine next-man-up, not a phantom that should
# stay at a hard, unrealistic zero - added 2026-08-24 after a real miscall
# (Seattle with Zach Charbonnet out left only Jadarian Price and George
# Holani eligible at all, so Price alone climbed to a true 74% workhorse
# share with no possible relief valve). Bounded to exactly the count of
# unavailable top-three slots, so a fully healthy backfield is unaffected.
# The newly admitted slot gets a small explicit score floor - real snap
# evidence for a true 4th/5th-string reserve is usually ~0, which would
# make him uncompetitive for a share even once he is technically eligible.
RB_VACANCY_EXTENSION_BASE_FLOOR = 0.03

# A team's core-RB snap shares are meant to describe its WHOLE backfield,
# not just the confidently-projected slice of it - added 2026-08-24 after a
# real miscall (a clean, fully-charted Bears room still summed to ~92% of
# team snaps, ~8 points short of "a full backfield", because both the
# `other RB` residual and the gap between snap_capacity and 1.0 were left
# unassigned to anyone). Rather than inventing a role for an unlisted
# reserve, the unclaimed remainder is redistributed only to already-
# projected core RBs, proportional to the share each already holds - see
# the final rescale call below, which reuses `_bounded_allocation` exactly
# as intended (it only ever distributes across `scores.gt(0)`).
RB_TEAM_SNAP_SHARE_TARGET = 1.00

# ...but the rescale target must not ride a genuinely high measured 2-RB
# committee capacity arbitrarily far above a full backfield. A team that runs
# two backs on the field a lot (ATL's 2025 Bijan + Allgeier 21-personnel
# usage measured ~1.09) should read as slightly over 1.0, not ~1.1 - added
# 2026-08-30 after the user flagged ATL summing to ~1.093 team RB snap share.
# `snap_share_target` is clamped to [RB_TEAM_SNAP_SHARE_TARGET,
# RB_TEAM_SNAP_SHARE_MAX]; carries/targets are unaffected (they reconcile
# against their own real per-game capacities).
RB_TEAM_SNAP_SHARE_MAX = 1.05

# --- v2_rb_snap_anchored_volume --------------------------------------------
# When on, the carry/target split starts from the depth-aware snap allocation
# and applies only a BOUNDED per-snap usage tilt (how many carries / targets
# this back earns per snap he plays, relative to his backfield's RB rate),
# instead of the older `0.62*snap_fraction + 0.38*(prior_per_GAME_rate /
# capacity)` blend. Per-game rate carries the player's OLD team's backfield
# split; a lead back who changed teams (Etienne to NO) was landing near the
# incumbent he is charted ahead of. The tilt is wider up-side for targets
# because genuine third-down backs really do earn >1x their snap share there.
RB_VOL_TILT_CARRY = (0.85, 1.20)
RB_VOL_TILT_TARGET = (0.70, 1.60)
# A cross-team player's stale per-snap rate is only partly trusted; a
# same-team player's is trusted in full. Thin histories fade toward
# snap-proportional too (games / (games + K)).
RB_VOL_TILT_CROSS_TEAM_TRUST = 0.35
RB_VOL_TILT_GAMES_K = 6.0
# The carry/target split also gets the SAME depth-rank discount the snap
# split already uses, so a charted RB2 with a big old per-game rate can no
# longer out-earn the charted RB1 on volume.
RB_VOL_RANK_DISCOUNT = {1: 1.0, 2: 0.92, 3: 0.80}
RB_VOL_RANK_DISCOUNT_DEFAULT = 0.72

# v2_rb_snap_anchored_volume also concentrates the SNAP allocation when a
# charted top-3 RB is OUT. Effective-rank compression (dense-ranking over only
# the still-eligible slots) is right for SCORING - it lets a chart RB2 read as
# "the guy" once RB1 is out - but it also silently promotes a chart RB3 to
# effective rank 2, which lets him skip the RB1<-RB3+ depth nudge entirely
# (measured live: NO 2026 wk1, Kamara out -> Devin Neal, chart RB3, ~34% team
# snaps while Etienne, the clear lead, sat at ~47%). Under the flag the depth
# nudge's DONOR side keys off the LITERAL chart rank (a chart RB3 stays a
# rank>=3 donor even when he is effective-rank 2 via the vacancy), and its
# per-team transfer cap is widened so more of that excess actually reaches the
# lead back. The rank-1 RECEIVER still uses effective rank, so an RB1-out /
# RB2-healthy backfield still promotes RB2 correctly.
RB_VOL_SNAP_NUDGE_CAP = 0.12
# A chart RB4+ let into the pool only because a top-3 slot is OUT (a
# "vacancy extension") gets a smaller phantom score floor than the plain
# RB_VACANCY_EXTENSION_BASE_FLOOR: he is genuine 4th-string, not the
# next-man-up the extension exists for. Keeps a Kendre-Miller-type RB4 near
# zero instead of ~8% once Kamara is ruled out.
RB_VOL_VACANCY_EXTENSION_FLOOR = 0.012

# --- v2_rb_snap_anchored_volume strength knobs ----------------------------
# Two independent 0..1 dials that let the flag's effect be tuned rather than
# taken whole. Added 2026-08-30 after the user judged the binary on/off
# ablation (START-RB dMAE -0.288 on 2023-25 wk1, n~3 correlated) too thin to
# retire the mechanism they see helping team-changer lead backs on the 2025
# wk1 board. Both default to 1.0, so `snap_anchored_volume=True` at the
# defaults reproduces the pre-knob behaviour exactly. Both can be overridden
# per process via the matching env var, so a strength sweep is just repeated
# `eval_weekly_model.py` runs (see scripts/sweep_rb_snap_anchored.py) rather
# than an edit-and-rerun of this file.
#
#   RB_VOL_TILT_STRENGTH - convex-blends the carry/target ALLOCATION between
#       the legacy `0.62*snap + 0.38*per-game-rate` path (0.0) and the full
#       per-snap usage tilt (1.0). This is the "portable volume" piece: a
#       lead back who changed teams keeps his real per-snap workload instead
#       of inheriting his old committee's split. Also scales the widened
#       downstream rate-scale clip in weekly_projections proportionally.
#   RB_VOL_VACANCY_STRENGTH - lerps the OUT-charted-top-3 snap-concentration
#       levers between legacy and snap-anchored: the per-team nudge transfer
#       cap (RB_DEPTH_RANK_SNAP_NUDGE_CAP <-> RB_VOL_SNAP_NUDGE_CAP) and the
#       vacancy-extension RB4 phantom floor (RB_VACANCY_EXTENSION_BASE_FLOOR
#       <-> RB_VOL_VACANCY_EXTENSION_FLOOR). The literal-chart-rank donor
#       test engages only when this is > 0. Much of this now overlaps the
#       standing RB chart hard-stop (RB_CHART_VACANCY_EXTENSION_MAX).
def _env_strength(name: str, default: float = 1.0) -> float:
    try:
        return float(np.clip(float(os.environ[name]), 0.0, 1.0))
    except (KeyError, TypeError, ValueError):
        return default


RB_VOL_TILT_STRENGTH = _env_strength("RB_VOL_TILT_STRENGTH")
RB_VOL_VACANCY_STRENGTH = _env_strength("RB_VOL_VACANCY_STRENGTH")

# How much of an incumbent's documented pre-injury role (``pre_gap``) is
# credited back to him when ``interrupted_incumbent_role_credit`` is at its
# max (1.0) - i.e. clear internal evidence he was starter-caliber before a
# mid-season absence, not just a same-role stat-line average. Tuned
# 2026-08-24 (0.30 -> 0.60) after the user flagged Cam Skattebo (2025 NYG
# rookie, injury-shortened season, credit=0.79) sitting well below Tyrone
# Tracy Jr. (fuller, uninterrupted 2025 role) and asked for the weight to be
# swept and judged on its own, not hand-picked to fix one pairing. Checked
# 0.30/0.40/0.50/0.60/0.70 against the full 2026 Week 1 board: the effect
# stays narrow and well-behaved at every step (6 backfields move at 0.50, 13
# at 0.70, no runaway share anywhere), and only 0.60 actually closes the
# ordering the user was questioning - Skattebo's snap share (0.390 -> 0.435)
# passes Tracy's (0.452 -> 0.418) instead of merely narrowing the gap, while
# every other affected player (Irving/Hampton/Stevenson/Dobbins - all
# similarly credited, real, interrupted 2025 seasons) moves by a comparable,
# modest amount. 0.70 was rejected as pushing the same handful of players
# further with no added ordering benefit - i.e. tuned toward the general
# mechanism holding at that step, not toward this one pairing specifically.
RB_INCUMBENT_CREDIT_WEIGHT = 0.60

# ``weekly_snap_pct`` can contain a one- or two-snap appearance.  That is
# real participation, but it is not enough on its own to establish a role or
# to make a player look as if he returned from a multi-week absence.  These
# guards are intentionally high precision: a role segment is only emitted
# when a meaningful pre-gap role, a real internal team-game gap, and a later
# return are all visible in actual snap data.
RB_SEGMENT_MIN_MEANINGFUL_SNAP_SHARE = 0.05
RB_SEGMENT_MIN_PRE_GAP_GAMES = 4
RB_SEGMENT_MIN_GAP_TEAM_GAMES = 3
RB_SEGMENT_PRE_GAP_WINDOW = 4


RB_ROLE_SEGMENT_COLUMNS = (
    "rb_segment_identity_key", "rb_segment_player", "rb_segment_team", "rb_segment_season",
    "rb_segment_status", "rb_segment_calendar_source", "rb_segment_has_snap_data",
    "whole_season_team_games", "whole_season_observed_snap_games", "whole_season_active_games",
    "whole_season_snap_share", "whole_season_active_snap_share",
    "whole_season_carries_per_game", "whole_season_targets_per_game",
    "interrupted_season", "pre_absence_games", "pre_absence_window_games",
    "pre_absence_start_week", "pre_absence_end_week", "pre_absence_snap_share",
    "pre_absence_carries_per_game", "pre_absence_targets_per_game",
    "absence_start_week", "absence_end_week", "absence_team_games",
    "absence_replacement_observed_games", "absence_replacement_top_rb_snap_share",
    "absence_replacement_core_rb_snap_share", "return_recovery_games",
    "return_recovery_start_week", "return_recovery_end_week", "return_recovery_snap_share",
    "return_recovery_carries_per_game", "return_recovery_targets_per_game",
    "interrupted_incumbent_role_credit", "pre_window_teammate_vacancy_downweight",
)

RB_TEAMMATE_CONTEXT_COLUMNS = (
    "rb_segment_team", "rb_segment_season", "incumbent_identity_key", "incumbent_player",
    "teammate_identity_key", "teammate_player", "shared_healthy_games",
    "incumbent_shared_healthy_snap_share", "teammate_shared_healthy_snap_share",
    "shared_healthy_lead_score", "teammate_pre_absence_snap_share",
    "absence_replacement_games", "teammate_absence_replacement_snap_share",
    "teammate_return_recovery_snap_share", "replacement_only_era_downweight",
    "context_status",
)


def _column(frame: pd.DataFrame, *names: str) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _numeric(frame: pd.DataFrame, *names: str, default: float = np.nan) -> pd.Series:
    column = _column(frame, *names)
    if column is None:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _text(frame: pd.DataFrame, *names: str, default: str = "") -> pd.Series:
    column = _column(frame, *names)
    if column is None:
        return pd.Series(default, index=frame.index, dtype=object)
    # pandas categorical columns reject a new fill value unless that value
    # is first added as a category.  Converting to object/string before the
    # fill keeps the allocator compatible with roster-derived categoricals.
    values = frame[column].astype(object)
    return values.where(values.notna(), "").astype(str).str.strip()


def classify_functional_position(frame: pd.DataFrame) -> pd.Series:
    """Return the projection subposition with fullback precedence.

    Roster feeds often expose a coarse ``position=RB`` while correctly
    identifying ``depth_chart_position=FB``.  A fullback is not a fantasy-RB
    workload candidate, so those more specific fields must win before any
    generic RB fallback is considered.  The same rule handles the locally
    imported Ourlads position and a historical/PFF FB label when the current
    roster lacks a depth-chart code.
    """
    if frame is None or frame.empty:
        return pd.Series(dtype=object)
    broad = _text(frame, "projection_position", "position", "Pos").str.upper()
    broad = broad.replace({"HB": "RB", "TB": "RB"})
    # This is precedence, not a union.  A current roster's explicit RB code
    # is stronger evidence than an old PFF/Ourlads label from a prior role.
    # Conversely, an explicit current FB code immediately overrides the
    # roster's broad fantasy grouping.
    resolved = broad.copy()
    sources = (
        _text(frame, "prior_pff_position", "prior_position", "pff_position").str.upper(),
        _text(frame, "ourlads_position").str.upper(),
        _text(frame, "depth_chart_position").str.upper(),
    )
    for source in sources:
        usable = source.isin({"RB", "HB", "TB", "FB"})
        normalized = source.replace({"HB": "RB", "TB": "RB"})
        resolved = resolved.where(~usable, normalized)
    return resolved.rename("functional_position")


def derive_preseason_rb_capacities(history: pd.DataFrame, team_col: str = "team") -> pd.DataFrame:
    """Estimate prior-team core-RB snap/carry/target capacities.

    Each capacity is an average of *team games*, not a sum of player season
    averages.  ``game_team`` is preferred to the roster-merged team field so
    a trade cannot rewrite the team capacity of an old game.  The result is
    deliberately a role prior; opponent, pace, and game environment remain in
    the normal weekly projection path.
    """
    columns = ["team", "core_rb_snap_capacity", "rb_carry_capacity", "rb_target_capacity",
               "capacity_games", "capacity_source"]
    if history is None or history.empty or "week" not in history.columns:
        return pd.DataFrame(columns=columns)
    frame = history.copy()
    game_team_col = _column(frame, "game_team", team_col, "Team")
    if game_team_col is None:
        return pd.DataFrame(columns=columns)
    frame["_team"] = _text(frame, game_team_col).str.upper()
    frame["_week"] = _numeric(frame, "week")
    frame["_functional_position"] = classify_functional_position(frame)
    frame = frame[(frame["_team"] != "") & frame["_week"].notna()]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    all_games = frame[["_team", "_week"]].drop_duplicates()
    core = frame[frame["_functional_position"].eq("RB")].copy()
    if core.empty:
        return pd.DataFrame(columns=columns)
    snaps = (_numeric(core, "weekly_snap_pct", default=np.nan) / 100.0).clip(0.0, 1.0)
    core["_snap"] = snaps
    core["_carries"] = _numeric(core, "rushing_attempts", default=0.0).fillna(0.0).clip(lower=0.0)
    core["_targets"] = _numeric(core, "targets", default=0.0).fillna(0.0).clip(lower=0.0)
    totals = core.groupby(["_team", "_week"], observed=True).agg(
        _snap=("_snap", "sum"), _carries=("_carries", "sum"), _targets=("_targets", "sum"),
    )
    game = all_games.merge(totals, left_on=["_team", "_week"], right_index=True, how="left")
    game[["_snap", "_carries", "_targets"]] = game[["_snap", "_carries", "_targets"]].fillna(0.0)
    capacity = game.groupby("_team", observed=True).agg(
        core_rb_snap_capacity=("_snap", "mean"),
        rb_carry_capacity=("_carries", "mean"),
        rb_target_capacity=("_targets", "mean"),
        capacity_games=("_week", "nunique"),
    ).reset_index().rename(columns={"_team": "team"})
    # Missing local weekly snap rows are not evidence that a team used no RBs.
    # Replace only an implausible zero with a league median, while carry and
    # target box scores remain direct evidence when they are present.
    snap_median = float(capacity.loc[capacity["core_rb_snap_capacity"].gt(0),
                                     "core_rb_snap_capacity"].median())
    carry_median = float(capacity.loc[capacity["rb_carry_capacity"].gt(0),
                                      "rb_carry_capacity"].median())
    target_median = float(capacity.loc[capacity["rb_target_capacity"].gt(0),
                                       "rb_target_capacity"].median())
    snap_default = snap_median if np.isfinite(snap_median) else DEFAULT_CORE_RB_SNAP_CAPACITY
    carry_default = carry_median if np.isfinite(carry_median) else DEFAULT_RB_CARRY_CAPACITY
    target_default = target_median if np.isfinite(target_median) else DEFAULT_RB_TARGET_CAPACITY
    capacity["core_rb_snap_capacity"] = capacity["core_rb_snap_capacity"].where(
        capacity["core_rb_snap_capacity"].gt(0), snap_default).clip(0.55, 1.45)
    capacity["rb_carry_capacity"] = capacity["rb_carry_capacity"].where(
        capacity["rb_carry_capacity"].gt(0), carry_default).clip(8.0, 35.0)
    capacity["rb_target_capacity"] = capacity["rb_target_capacity"].where(
        capacity["rb_target_capacity"].gt(0), target_default).clip(1.0, 13.0)
    capacity["capacity_source"] = "prior-season core-RB team games"
    return capacity.reindex(columns=columns)


def _capacity_value(group: pd.DataFrame, column: str, default: float) -> float:
    values = pd.to_numeric(group.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
    if values.empty:
        return float(default)
    return float(np.clip(values.iloc[0], 0.0, np.inf))


def _other_fraction(charted_players: int) -> float:
    # The residual represents uncertainty/lesser personnel, not a hidden
    # league-average workload for every unlisted reserve.  A supplied top-3
    # chart permits only a small residual; without a chart the residual is
    # deliberately larger.
    return {0: 0.25, 1: 0.18, 2: 0.10}.get(int(charted_players), 0.05)


def _bound_share_toward_snap(alloc: pd.Series, cap: float, snap_fraction: pd.Series,
                             bound_mask: pd.Series, tolerance: float) -> pd.Series:
    """Pull a GATED player's implied share (``alloc / cap``) to within
    ``tolerance`` of his own snap_fraction, redistributing the delta to his
    UNGATED teammates proportional to their own already-allocated share -
    same conservation rule as ``_bounded_allocation`` (the total never
    changes, nothing vanishes or is fabricated).

    Why this has to be a SEPARATE, later pass rather than just feeding
    ``_bounded_allocation`` an evidence term equal to snap_fraction (which
    is what the caller does first, see the cross-team-evidence note above):
    ``_bounded_allocation`` splits a team's TOTAL capacity by each player's
    score RELATIVE TO HIS TEAMMATES' scores. Making one player's own score
    internally consistent with his own snap_fraction does not, by itself,
    make his resulting SHARE track that snap_fraction - his teammates'
    (untouched, real) evidence can still be relatively over- or under-
    weighted and pull the split away from him regardless. This pass fixes
    the actual output number the caller asked to bound, not an upstream
    ingredient that only usually gets there.
    """
    if cap <= 0 or not bound_mask.any():
        return alloc
    share = (alloc / cap).clip(lower=0.0)
    target = snap_fraction.clip(lower=0.0)
    desired_share = share.clip(lower=(target - tolerance).clip(lower=0.0), upper=target + tolerance)
    desired_share = desired_share.where(bound_mask, share)
    desired = desired_share * cap
    delta = float((alloc - desired).where(bound_mask, 0.0).sum())
    out = alloc.where(~bound_mask, desired)
    if abs(delta) > 1e-9:
        donors = out.index[~bound_mask.to_numpy() & (out.to_numpy() > 0)]
        if len(donors):
            weights = out.loc[donors]
            out.loc[donors] = out.loc[donors] + delta * (weights / weights.sum())
    return out.clip(lower=0.0)


def _bounded_allocation(scores: pd.Series, total: float, max_each: float | None = None) -> pd.Series:
    """Allocate ``total`` by nonnegative scores with an optional hard cap."""
    result = pd.Series(0.0, index=scores.index, dtype=float)
    active = scores[scores.gt(0)].copy()
    remaining = float(max(total, 0.0))
    while not active.empty and remaining > 1e-9:
        proposed = active / active.sum() * remaining
        if max_each is None:
            result.loc[active.index] += proposed
            break
        room = (float(max_each) - result.loc[active.index]).clip(lower=0.0)
        capped = proposed > room + 1e-9
        if not capped.any():
            result.loc[active.index] += proposed
            break
        result.loc[active.index[capped]] += room.loc[capped]
        remaining = float(total - result.sum())
        active = active.loc[~capped]
    return result.clip(lower=0.0)


def _apply_depth_rank_snap_nudge(snap_alloc: pd.Series, effective_rank: pd.Series,
                                 donor_rank: pd.Series | None = None,
                                 cap: float = RB_DEPTH_RANK_SNAP_NUDGE_CAP) -> pd.Series:
    """Bounded, conserved RB1-up / RB3-down snap-share transfer for one team.

    See RB_DEPTH_RANK_SNAP_* module constants for the rationale. ``snap_alloc``
    is this team's already-computed core-RB snap shares; ``effective_rank`` is
    a chart rank (1/2/3/...) when the team has an imported Ourlads chart, or a
    score-derived rank (same convention) when it does not - the caller decides
    which. Only rank 1 (receiver) and rank>=3 (donor) participate; a two-man
    backfield (no rank>=3) or a team with no clear rank-1 is left untouched.
    ``donor_rank`` optionally supplies a SEPARATE rank series for the rank>=3
    donor test (v2_rb_snap_anchored_volume passes the literal chart rank here
    so a vacancy-promoted chart RB3 still donates); the rank-1 receiver always
    comes from ``effective_rank``. ``cap`` overrides the per-team transfer cap.
    """
    donor_rank = effective_rank if donor_rank is None else donor_rank
    rank1_idx = effective_rank.index[effective_rank.eq(1)]
    donor_idx = donor_rank.index[donor_rank.ge(3)]
    if not len(rank1_idx) or not len(donor_idx):
        return snap_alloc
    donor_share = snap_alloc.loc[donor_idx]
    excess = (donor_share - RB_DEPTH_RANK_SNAP_TARGET_RANK3).clip(lower=0.0)
    total_excess = float(excess.sum())
    if total_excess <= 1e-9:
        return snap_alloc
    transfer = min(RB_DEPTH_RANK_SNAP_PULL * total_excess, cap)
    nudged = snap_alloc.copy()
    nudged.loc[donor_idx] -= (excess / total_excess) * transfer
    nudged.loc[rank1_idx] += transfer / len(rank1_idx)
    return nudged


def _apply_depth_rank2_order_nudge(snap_alloc: pd.Series, effective_rank: pd.Series) -> pd.Series:
    """Bounded, conserved correction when a listed RB3+ outshares the listed RB2.

    See RB_DEPTH_RANK2_ORDER_* above.  Structurally identical to
    ``_apply_depth_rank_snap_nudge``, except the receiver is rank 2 (instead
    of rank 1) and the pull target is RB2's OWN current share (instead of a
    fixed ceiling) - only the excess a rank>=3 back holds ABOVE what RB2
    himself has is eligible to move, so this only ever restores order, never
    inverts it.
    """
    rank2_idx = effective_rank.index[effective_rank.eq(2)]
    donor_idx = effective_rank.index[effective_rank.ge(3)]
    if not len(rank2_idx) or not len(donor_idx):
        return snap_alloc
    rank2_share = float(snap_alloc.loc[rank2_idx].sum()) / len(rank2_idx)
    donor_share = snap_alloc.loc[donor_idx]
    excess = (donor_share - rank2_share).clip(lower=0.0)
    total_excess = float(excess.sum())
    if total_excess <= 1e-9:
        return snap_alloc
    transfer = min(RB_DEPTH_RANK2_ORDER_PULL * total_excess, RB_DEPTH_RANK2_ORDER_NUDGE_CAP)
    nudged = snap_alloc.copy()
    nudged.loc[donor_idx] -= (excess / total_excess) * transfer
    nudged.loc[rank2_idx] += transfer / len(rank2_idx)
    return nudged


def _role_base(group: pd.DataFrame) -> pd.Series:
    base = _numeric(group, "base_snap_share", "expected_snap_share", "Expected Snap Share")
    active = _numeric(group, "prior_active_snap_share", "active_snap_share")
    whole = _numeric(group, "prior_whole_snap_share", "whole_snap_share")
    pre_absence = _numeric(group, "pre_absence_snap_share")
    fallback = pd.concat([active, whole, pre_absence], axis=1).mean(axis=1, skipna=True)
    base = base.where(base.notna(), fallback).fillna(0.0).clip(0.0, 1.0)
    # A credible interrupted season can inform a returner without treating a
    # late missed stretch as permanent loss of role.  This is bounded by the
    # already-computed continuous base rather than replacing it outright.
    # A player the current chart lists only as a continuation / second-unit
    # row (``chart_deprioritized``, v2_rb_snap_anchored_volume) gets NO
    # pre-injury-role credit: the chart's live "deep reserve" placement
    # outranks a stale pre-gap role for a Week-1 projection.
    interrupted = _text(group, "interrupted_season", default="").str.lower().isin({"1", "true", "yes"})
    chart_deprioritized = (group["chart_deprioritized"].fillna(False).astype(bool)
                           if "chart_deprioritized" in group.columns
                           else pd.Series(False, index=group.index))
    pre_gap = (pre_absence - base).clip(lower=0.0).fillna(0.0)
    pre_gap = pre_gap.where(~chart_deprioritized.to_numpy(), 0.0)
    role = base + np.where(interrupted, 0.45 * pre_gap, 0.20 * pre_gap)
    # A clear internal absence + return is stronger evidence than an ordinary
    # late-season missed stretch.  Keep the added credit bounded and only
    # move toward the player's documented *pre-gap* role; it cannot invent a
    # bell-cow role from a small depth-chart sample.
    incumbent_credit = _numeric(group, "interrupted_incumbent_role_credit", default=0.0).fillna(0.0).clip(0.0, 1.0)
    role += RB_INCUMBENT_CREDIT_WEIGHT * incumbent_credit * pre_gap
    # Shared-healthy teammate data distinguishes an incumbent's role before
    # injury from a replacement's temporary absence-era workload.  A two-game
    # shared sample is not a verdict, so these are modest, capped score
    # adjustments—visible separately in the allocator input and popup.
    shared_lead = _numeric(group, "shared_healthy_lead_score", default=0.0).fillna(0.0).clip(-1.0, 1.0)
    replacement_downweight = _numeric(group, "replacement_only_era_downweight", default=0.0).fillna(0.0).clip(0.0, 0.55)
    teammate_multiplier = np.clip(1.0 + 0.10 * shared_lead - 0.35 * replacement_downweight, 0.75, 1.15)
    return (role * teammate_multiplier).clip(0.0, 0.95)


def allocate_preseason_rb_roles(candidates: pd.DataFrame,
                                snap_anchored_volume: bool = False,
                                week: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconcile core-RB Week-1 roles to independent team capacities.

    ``week`` (2026-09-04): the caller's literal target week, used ONLY to
    gate two Week-1-specific guards - see RB_WEEK1_MAX_INDIVIDUAL_SNAP_SHARE
    and the cross-team carry/target evidence note below. ``None`` (every
    existing caller/test) preserves prior behavior exactly; this function
    otherwise has no notion of which week it is being asked about.

    Required inputs are intentionally modest.  The public names documented in
    the V2 handoff are accepted in either lower-case or UI-style title case;
    omitted evidence falls back to a neutral/explicit residual rather than a
    fabricated 15.7% reserve share.

    ``snap_anchored_volume`` ('v2_rb_snap_anchored_volume'):
    - derive the carry/target split from the depth-aware SNAP allocation plus
      a bounded per-snap usage tilt, instead of blending snap fraction with a
      raw prior-season per-GAME rate (the per-game rate bakes in the player's
      OLD team's backfield split, so an every-down back who changed teams -
      Travis Etienne to NO - was landing near the aging incumbent he is
      charted ahead of); and
    - concentrate the SNAP allocation when a charted top-3 RB is OUT: the
      RB1<-RB3+ depth nudge's donor test keys off the LITERAL chart rank
      (so a chart RB3 promoted to effective-rank 2 by the vacancy still
      donates his excess to the lead) with a wider transfer cap, and a chart
      RB4+ let in only as a vacancy extension gets a smaller phantom floor; and
    - drop the interrupted-season / incumbent pre-injury-role credit in
      ``_role_base`` for a player the caller flags ``chart_deprioritized``
      (charted only as a continuation / second-unit row) - the chart's live
      "deep reserve" placement outranks a stale pre-gap role at Week 1.
    See RB_VOL_* constants.
    """
    ledger_columns = ["team", "resource", "capacity", "allocated", "unallocated",
                      "candidate_count", "other_fraction", "reason"]
    if candidates is None or candidates.empty:
        return pd.DataFrame() if candidates is None else candidates.copy(), pd.DataFrame(columns=ledger_columns)
    out = candidates.copy()
    team_col = _column(out, "team", "Team")
    player_col = _column(out, "Player", "player", "name")
    if team_col is None or player_col is None:
        raise ValueError("RB allocator requires a team and player column.")
    active_flag = _text(out, "is_active", default="true").str.lower().isin({"1", "true", "yes", "y"})
    hard_status = _text(out, "status", "roster_status").str.upper()
    # Explicitly retired/cut rows do not belong to an upcoming candidate pool
    # at all.  Fullbacks remain as a zero-core-role audit row so users can see
    # why they did not inherit RB volume.
    out = out.loc[active_flag & ~hard_status.isin(INELIGIBLE_ROSTER_STATUSES)].copy()
    if out.empty:
        return out, pd.DataFrame(columns=ledger_columns)
    out["_allocation_index"] = out.index
    out["_rb_team"] = _text(out, team_col).str.upper()
    out["functional_position"] = classify_functional_position(out)
    status = _text(out, "status", "roster_status").str.upper()
    availability = _numeric(out, "availability", "Availability", default=1.0).fillna(1.0).clip(0.0, 1.0)
    rank = _numeric(out, "ourlads_depth_rank", "ourlads_role_rank", "source_depth_rank")
    base = _role_base(out)
    prior_games = _numeric(out, "prior_games", "Games", default=0.0).fillna(0.0)
    observed_prior_column = _column(out, "has_observed_prior_role", "observed_prior_role")
    if observed_prior_column is None:
        # Direct callers that predate the explicit flag can still make a
        # conservative history-based allocation.  A synthetic median base
        # alone is never evidence of a real role.
        has_observed_prior_role = prior_games.gt(0)
    else:
        has_observed_prior_role = _text(out, observed_prior_column).str.lower().isin(
            {"1", "true", "yes", "y"})
    # Touch-volume evidence, for weak_fb_evidence's guard ONLY (see its own
    # comment) - deliberately not merged into has_observed_prior_role above,
    # which other eligibility logic (season_2025_qualified) already relies
    # on meaning "any recorded role at all," games or snap share included.
    # Checks whichever representation the caller supplied - a SHARE
    # (0-1, the allocator's own documented schema) or a per-game RATE
    # (what weekly_projections.py's live caller actually populates today).
    _prior_touch_share = (_numeric(out, "prior_active_carry_share").fillna(0.0)
                          + _numeric(out, "prior_active_target_share").fillna(0.0))
    _prior_touch_rate = (
        _numeric(out, "prior_carries_per_game", "prior_carry_rate", "prior_carries").fillna(0.0)
        + _numeric(out, "prior_targets_per_game", "prior_target_rate", "prior_targets").fillna(0.0))
    has_observed_prior_touches = (
        _prior_touch_share.gt(RB_MIN_PRIOR_TOUCH_SHARE_FOR_FB_GUARD)
        | _prior_touch_rate.gt(RB_MIN_PRIOR_TOUCH_RATE_FOR_FB_GUARD))
    draft_capital = _numeric(out, "draft_capital", "draft_number")
    is_rookie = _text(out, "is_rookie", "is_rookie_flag", default="").str.lower().isin({"1", "true", "yes"})
    core = out["functional_position"].eq("RB") & ~status.isin(INELIGIBLE_ROSTER_STATUSES)
    # A current Ourlads primary-row rank is evidence about *who belongs in
    # this particular Week-1 backfield*.  Once it exists for a team, do not
    # let every old rostered RB with four historical games join the core pool
    # and flatten the listed lead back.  Teams without a local chart retain a
    # conservative history/draft fallback so the 30-page preseason snapshot
    # does not make DET/PIT (or an unavailable source) unusable.
    listed = rank.notna() & rank.ge(1)
    team_has_chart = listed.groupby(out["_rb_team"], observed=True).transform("any")
    # Season-scoped evidence, redone 2026-08-24 - see RB_ELIGIBILITY_* above
    # for why. ``season_active_2025``/``prior2_games``/``prior2_active_share``
    # come from the caller (weekly_projections.py's cold-start blend); a
    # direct caller that predates them (e.g. an older test fixture) gets an
    # all-NaN column here, which correctly fails every ``.ge()`` check below
    # rather than raising.
    # Falls back to the older ``prior_active_snap_share`` column when the
    # caller doesn't supply the season-scoped one (an older/direct caller,
    # e.g. this module's own test fixtures) - that column is exactly this
    # value for such a caller anyway, since only the real production caller
    # in weekly_projections.py blends ``prior_active_snap_share`` itself
    # with 2024 evidence and therefore needs a separate, unblended field.
    season_active_2025 = _numeric(out, "season_active_snap_share_2025", "prior_active_snap_share")
    # A flat whole-season average treats a role that grew meaningfully late
    # in the year the same as one that never did - measured on real 2025
    # data, this alone was the difference between correctly keeping Tank
    # Bigsby (a real 2nd back with a strong finish) eligible and wrongly
    # excluding him alongside a genuine 3rd-string committee back whose
    # season average happened to land within two points of his. Takes
    # whichever of the two reads is more favorable, same "max of two role
    # signals" convention _blend_prior2/restore_cold_start_returning_role_
    # share already use elsewhere in this pipeline - a real recent role
    # should never be capped BY a thinner whole-season history.
    recent8_2025 = _numeric(out, "season_recent8_snap_share_2025")
    season_active_2025 = season_active_2025.combine(recent8_2025, lambda a, b: np.nanmax([a, b])
                                                     if pd.notna(a) or pd.notna(b) else np.nan)
    prior2_games = _numeric(out, "prior2_games", default=0.0).fillna(0.0)
    prior2_active_share = _numeric(out, "prior2_active_snap_share")
    season_2025_qualified = (
        has_observed_prior_role
        & prior_games.ge(RB_ELIGIBILITY_MIN_GAMES_2025)
        & season_active_2025.ge(RB_ELIGIBILITY_MIN_SHARE_2025)
    )
    # "Was he a starter who got hurt" - a real 2024 lead/near-lead role,
    # checked only when 2025 alone did not already clear the bar. This is a
    # binary OR-fallback, deliberately not blended with 2025 the way the
    # caller's own point-estimate share is - a player either has a credible
    # starter season on record somewhere recent enough to matter, or he does
    # not; there is no partial credit for eligibility itself.
    prior2_starter_fallback = (
        ~season_2025_qualified
        & prior2_games.ge(RB_ELIGIBILITY_PRIOR2_MIN_GAMES)
        & prior2_active_share.ge(RB_ELIGIBILITY_PRIOR2_STARTER_SHARE)
    )
    strong_evidence = season_2025_qualified | prior2_starter_fallback
    # Draft capital is a real signal ONLY for a player with essentially no
    # NFL role evidence yet - the case the comment below has always
    # described. It used to apply to any historically-drafted veteran
    # regardless of how much (or how irrelevant) his career evidence since
    # has been - confirmed live: a 2022 mid-round RB with a 5-game, 8.4%
    # active-game share for a DIFFERENT team in 2025 was still counted
    # credible on his new team purely off a 2022 draft slot. Scoped to
    # ``is_rookie`` now, matching what the comment already claimed it did.
    fallback_credible = strong_evidence | (is_rookie & draft_capital.ge(1) & draft_capital.le(150))
    # Identity/source failures must never erase a real same-team incumbent
    # who had a strong proven role.  This is an eligibility safety net—not a
    # free workload: the normal finite-capacity scoring still decides his
    # share and deep/unknown reserves cannot meet the evidence guard.
    incumbent_backstop = _text(out, "established_incumbent_backstop", default="").str.lower().isin(
        {"1", "true", "yes", "y"})
    # Vacancy-aware credibility ceiling - see RB_VACANCY_EXTENSION_BASE_FLOOR
    # above.  A team's literal rank<=3 gate assumes those three slots are all
    # actually available; when one is not, the next chart slot is a genuine
    # next-man-up, not a phantom.  The extension is CAPPED
    # (RB_CHART_VACANCY_EXTENSION_MAX): one injury to a top-three back promotes
    # the chart RB4 into the credible set but NOT the RB5 - a genuine
    # 5th-string back does not earn a projected role off a single absence
    # (measured live 2026-08-30: Kendre Miller, NO, ~8% projected snaps with
    # Kamara out). The cap also overrides the incumbent backstop for a deep
    # chart reserve, so an identity-match quirk cannot reinstate an RB5.
    #
    # A `chart_deprioritized` candidate is one the chart lists ONLY in a
    # `position_occurrence >= 1` continuation column (Ourlads' two-column
    # backfield overflow - Miller is nominally "RB4" there but that is a
    # layout artifact; he is really a 5th-stringer). Treat him as a deep
    # reserve regardless of the nominal rank.
    chart_deprioritized = (out["chart_deprioritized"].fillna(False).astype(bool)
                           if "chart_deprioritized" in out.columns
                           else pd.Series(False, index=out.index))
    unavailable_top3 = (rank.le(3) & availability.le(0.01)).groupby(
        out["_rb_team"], observed=True).transform("sum").fillna(0)
    rank_ceiling = 3 + np.minimum(unavailable_top3, RB_CHART_VACANCY_EXTENSION_MAX)
    deep_chart_reserve = team_has_chart & (
        rank.gt(rank_ceiling).fillna(False) | chart_deprioritized)
    credible = np.where(
        team_has_chart,
        np.where(deep_chart_reserve, False, rank.le(rank_ceiling) | incumbent_backstop),
        fallback_credible,
    )
    # A literal Ourlads FB listing for THIS player is direct, current,
    # curated evidence he is a fullback, even when `functional_position`
    # resolved to RB - added 2026-08-24 after two real miscalls (D.J.
    # Herman, MIA FB2; Max Bredeson, MIN FB, both climbing to a real core-RB
    # snap share). `classify_functional_position` intentionally still lets a
    # current roster's own depth_chart_position win a genuine conflict (see
    # its own docstring and the "Core Back" test) - a converted FB who is
    # now a real, evidenced RB must stay eligible. This guard is deliberately
    # narrower and only fires for a player with no observed prior role and
    # no established-incumbent backstop, i.e. exactly the "unproven depth
    # fullback the roster feed happens to broadly tag RB" case, not a real
    # RB with a track record.
    #
    # "No observed prior role" originally meant only prior_games>0 / any
    # recorded snap share - which a career blocking fullback clears just as
    # easily as a real committee back, since his job is legitimate real
    # snaps with almost no touches. Extended 2026-09-04 (Patrick Ricard: 5
    # games, 45.6% snap share, 0.6 combined carries+targets/game in 2025 -
    # cleared the original check and drew real core-RB capacity) to also
    # require some real TOUCH volume specifically - has_observed_prior_
    # touches above. Still additive only: a player can lose eligibility here
    # that the games/snap check alone would have kept, never gain it back.
    ourlads_fb_signal = _text(out, "ourlads_position", default="").str.upper().eq("FB")
    weak_fb_evidence = (ourlads_fb_signal & ~incumbent_backstop
                        & (~has_observed_prior_role | ~has_observed_prior_touches))
    eligible = core & credible & availability.gt(0.01) & ~weak_fb_evidence
    out["core_rb"] = core
    out["eligible_core_rb"] = eligible
    out["established_incumbent_backstop"] = incumbent_backstop
    out["pre_allocation_snap_share"] = base
    out["expected_snap_share"] = 0.0
    out["carry_share"] = 0.0
    out["target_share"] = 0.0
    out["allocated_carries"] = 0.0
    out["allocated_targets"] = 0.0
    out["allocation_source"] = np.where(
        out["functional_position"].eq("FB"), "functional fullback excluded from core-RB allocator",
        np.where(~core, "not a running-back candidate", "not a credible core-RB candidate"),
    )
    out["allocation_eligibility_reason"] = np.where(
        out["functional_position"].eq("FB"), "functional fullback excluded",
        np.where(weak_fb_evidence, "literal Ourlads fullback listing, no offsetting RB evidence",
                 np.where(~core, "not a functional RB",
                          np.where(rank.le(3), "literal Ourlads top-three role",
                                   np.where(incumbent_backstop,
                                            "established same-team incumbent safety backstop",
                                            np.where(fallback_credible, "observed role/draft fallback",
                                                     "no credible role evidence"))))))
    ledger: list[dict[str, Any]] = []
    is_week1 = week == 1

    for team, group in out.groupby("_rb_team", sort=False, observed=True):
        if not team:
            continue
        snap_capacity = _capacity_value(group, "core_rb_snap_capacity", DEFAULT_CORE_RB_SNAP_CAPACITY)
        carry_capacity = _capacity_value(group, "rb_carry_capacity", DEFAULT_RB_CARRY_CAPACITY)
        target_capacity = _capacity_value(group, "rb_target_capacity", DEFAULT_RB_TARGET_CAPACITY)
        indexes = group.index[group["eligible_core_rb"]]
        if not len(indexes):
            for metric, capacity in (("core_rb_snaps", snap_capacity), ("rb_carries", carry_capacity),
                                     ("rb_targets", target_capacity)):
                ledger.append({"team": team, "resource": metric, "capacity": capacity, "allocated": 0.0,
                               "unallocated": capacity, "candidate_count": 0, "other_fraction": 1.0,
                               "reason": "No credible active core-RB candidate."})
            continue
        candidates_team = out.loc[indexes]
        source_rank = rank.loc[indexes]
        has_chart = bool(team_has_chart.loc[indexes].any())
        # Effective rank re-ranks the literal chart order among only the
        # CURRENTLY ELIGIBLE candidates (dense, so no gaps survive an
        # unavailable teammate) - this is what lets a chart RB2 correctly
        # read as "the guy" once chart RB1 is out, instead of still
        # competing for a stale rank-2 discount against a rank-1 slot that
        # no one currently fills. A team with no chart already computes an
        # equivalent rank restricted to `indexes` below, so it needs no
        # separate vacancy handling.
        effective_rank = source_rank.rank(method='dense') if has_chart else pd.Series(dtype=float)
        # A candidate admitted only via the vacancy-extension ceiling above
        # (literal chart rank>3) rarely has any real snap evidence of his
        # own; without a small floor he would be mathematically uncompetitive
        # for a share even though he is now technically eligible.
        base_for_score = base.loc[indexes].copy()
        vacancy_extension_idx = indexes[source_rank.gt(3).fillna(False)] if has_chart else indexes[0:0]
        if len(vacancy_extension_idx):
            # RB_VOL_VACANCY_STRENGTH lerps the phantom floor between the plain
            # next-man-up value and the lower snap-anchored one.
            _sv = float(np.clip(RB_VOL_VACANCY_STRENGTH, 0.0, 1.0)) if snap_anchored_volume else 0.0
            _ext_floor = (RB_VACANCY_EXTENSION_BASE_FLOOR
                          + _sv * (RB_VOL_VACANCY_EXTENSION_FLOOR - RB_VACANCY_EXTENSION_BASE_FLOOR))
            base_for_score.loc[vacancy_extension_idx] = base_for_score.loc[vacancy_extension_idx].clip(
                lower=_ext_floor)
        # Preserve an established active-game role much more strongly than a
        # plain linear split.  A 78% lead-back role should not be averaged
        # down to 34% merely because a chart also lists two reserves; the
        # exponent is a bounded concentration, not a hard depth-chart lock.
        score = base_for_score.clip(lower=0.01).pow(1.45)
        low_evidence = score.lt(0.30)
        if has_chart:
            score *= np.where(effective_rank.eq(1) & low_evidence, 1.25,
                              np.where(effective_rank.eq(1), 1.08,
                                       np.where(effective_rank.eq(2), 0.98,
                                                np.where(effective_rank.eq(3), 0.88, 1.0))))
        else:
            # No imported chart for this team, added 2026-08-24: the block
            # just above is a no-op here (source_rank is all-NaN without a
            # chart), so a genuine 3rd/4th eligible back previously competed
            # on raw score alone with no depth penalty at all. Apply the
            # SAME discount schedule, keyed by SCORE-derived rank instead -
            # see SCORE_RANK_DISCOUNT's own comment.
            score_rank = base.loc[indexes].rank(ascending=False, method='first')
            score *= score_rank.map(SCORE_RANK_DISCOUNT).fillna(SCORE_RANK_DISCOUNT_DEFAULT)
        draft_bonus = (0.20 * (1.0 - (draft_capital.loc[indexes].fillna(999.0) - 1.0) / 149.0)
                       .clip(0.0, 1.0))
        # A vacancy-extension admit (see RB_VACANCY_EXTENSION_BASE_FLOOR) does
        # not get the low-evidence draft bonus below - it exists for a real
        # top-three-caliber rookie/veteran competing for a starting-caliber
        # role, not for a longshot 4th/5th-string arm let in only because a
        # teammate is out (measured live - Velus Jones Jr., a 2022 3rd-round
        # WR pick admitted at SEA's vacancy-extended rank 4, jumped to a real
        # ~15% team share purely off his old WR draft slot, which has nothing
        # to do with his current RB role).
        is_vacancy_extension = indexes.isin(vacancy_extension_idx)
        score += np.where((is_rookie.loc[indexes] | low_evidence) & ~is_vacancy_extension, draft_bonus, 0.0)
        same_team = _text(candidates_team, "same_team", default="").str.lower().isin({"1", "true", "yes"})
        score += np.where(same_team, 0.02, 0.0)
        score *= availability.loc[indexes]
        charted = int(source_rank.le(3).sum())
        # A team with no imported chart (``charted == 0``) is not
        # automatically an unsettled backfield: treat a player with a real,
        # measured prior-season role the same as a literal charted slot for
        # sizing the "other RB" residual, capped at the same three-slot
        # ceiling a chart would carry. This is what stops an established
        # bell-cow (real 2025 evidence, but no reimported 2026 Ourlads
        # snapshot) from having a quarter of his own backfield's capacity
        # walled off as "unknown" every single week - see HANDOFF.md's RB
        # snap-share note for the measured Henry/Hampton/Chase Brown cases
        # this fixes. Draft-capital-only credibility (a rookie with zero
        # career games) does NOT count here; that is real uncertainty, not
        # a documented role, so it keeps the harsher default residual.
        evidenced = int(strong_evidence.loc[indexes].sum())
        other_fraction = _other_fraction(max(charted, min(evidenced, 3)))
        player_snap_total = snap_capacity * (1.0 - other_fraction)
        snap_alloc = _bounded_allocation(score, player_snap_total, MAX_INDIVIDUAL_CORE_RB_SNAP_SHARE)
        nudge_rank = effective_rank if has_chart else base.loc[indexes].rank(ascending=False, method='first')
        # v2_rb_snap_anchored_volume: donor test on the LITERAL chart rank (so a
        # vacancy-promoted chart RB3 still donates) with a wider transfer cap;
        # the rank-1 receiver stays on effective rank so an RB1-out backfield
        # still promotes its RB2. Plain path is unchanged.
        # RB_VOL_VACANCY_STRENGTH: the literal-chart-rank donor test engages
        # only when > 0; the per-team transfer cap lerps from the plain value
        # to the wider snap-anchored one.
        _sv = float(np.clip(RB_VOL_VACANCY_STRENGTH, 0.0, 1.0)) if snap_anchored_volume else 0.0
        _donor_rank = source_rank if (_sv > 0.0 and has_chart) else None
        _nudge_cap = (RB_DEPTH_RANK_SNAP_NUDGE_CAP
                      + _sv * (RB_VOL_SNAP_NUDGE_CAP - RB_DEPTH_RANK_SNAP_NUDGE_CAP))
        snap_alloc = _apply_depth_rank_snap_nudge(snap_alloc, nudge_rank,
                                                 donor_rank=_donor_rank, cap=_nudge_cap)
        snap_alloc = _apply_depth_rank2_order_nudge(snap_alloc, nudge_rank)
        # Redistribute any unclaimed team-snap remainder back to the
        # already-projected core RBs, proportional to their current share -
        # see RB_TEAM_SNAP_SHARE_TARGET above. `_bounded_allocation` only
        # ever distributes across players who already hold a positive share,
        # so a zero-share bench reserve is untouched. The target is never
        # LOWER than the team's own real snap_capacity - a team whose
        # measured capacity already exceeds 100% (a real 2-RB-personnel
        # committee, where two backs share a single snap) must not be
        # artificially compressed back down to a flat 100% - but it is capped
        # at RB_TEAM_SNAP_SHARE_MAX so a high measured committee capacity
        # cannot ride the whole-backfield target arbitrarily far past 1.0.
        snap_share_target = min(max(snap_capacity, RB_TEAM_SNAP_SHARE_TARGET),
                                RB_TEAM_SNAP_SHARE_MAX)
        _snap_cap = RB_WEEK1_MAX_INDIVIDUAL_SNAP_SHARE if is_week1 else MAX_INDIVIDUAL_CORE_RB_SNAP_SHARE
        snap_alloc = _bounded_allocation(snap_alloc, snap_share_target, _snap_cap)

        prior_carry_rate = _numeric(candidates_team, "prior_carries_per_game", "prior_carry_rate", "prior_carries")
        prior_target_rate = _numeric(candidates_team, "prior_targets_per_game", "prior_target_rate", "prior_targets")
        snap_fraction = (snap_alloc / max(snap_capacity, 0.01)).clip(0.0, 1.0)
        # Legacy carry/target evidence blend - the v2_rb_snap_anchored_volume
        # OFF path, and also the anchor the tilt is blended toward when
        # RB_VOL_TILT_STRENGTH < 1.
        carry_evidence = (prior_carry_rate / max(carry_capacity, 0.01)).clip(lower=0.0).fillna(snap_fraction)
        target_evidence = (prior_target_rate / max(target_capacity, 0.01)).clip(lower=0.0).fillna(snap_fraction)
        _no_backstop_new_team = pd.Series(False, index=indexes)
        # WEEK 1 ONLY (2026-09-04): for a player with no same-team incumbent
        # backstop who also changed teams, prior_carry_rate/prior_target_rate
        # are his OLD team's per-game rates - dividing them by THIS team's
        # carry_capacity/target_capacity is not a share of anything real, it
        # is two different offenses' volumes collided through a mismatched
        # denominator. Confirmed live: Travis Etienne (JAX->NO) landed ~20
        # points of carry share AND target share below his own snap share -
        # both axes moving together, not the real carry/receiving trade-off
        # an established committee (Aaron Jones/Jordan Mason-shaped) shows,
        # because the SAME contaminated evidence term drags both down at
        # once. This is the identical cross-team distrust
        # v2_rb_snap_anchored_volume's tilt path already applies via `trust`
        # above - deliberately NOT that flag: turning it on rescores every
        # player's concentration/rank-discount too and measured negative on
        # START-RB broadly (.sweeps/ablate_rb_snap_anchored_wk1.txt,
        # "the etienne case is a true outlier" - the user's own verdict on
        # that broad version). This instead neutralizes ONLY the
        # cross-team-contaminated evidence term, ONLY for the narrow subset
        # with no offsetting evidence, so both scores fall back to pure
        # snap_fraction (carry share = target share = snap share) rather
        # than the mismatched ratio - an established player's real,
        # differentiated evidence is untouched either way.
        if is_week1:
            _no_backstop_new_team = pd.Series(
                (~same_team.to_numpy()) & (~incumbent_backstop.loc[indexes].to_numpy()), index=indexes)
            carry_evidence = carry_evidence.where(~_no_backstop_new_team, snap_fraction)
            target_evidence = target_evidence.where(~_no_backstop_new_team, snap_fraction)
        carry_score_legacy = (0.62 * snap_fraction + 0.38 * carry_evidence).clip(lower=0.005) * availability.loc[indexes]
        target_score_legacy = (0.50 * snap_fraction + 0.50 * target_evidence).clip(lower=0.005) * availability.loc[indexes]
        _carry_cap = carry_capacity * (1.0 - other_fraction)
        _target_cap = target_capacity * (1.0 - other_fraction)
        _st = float(np.clip(RB_VOL_TILT_STRENGTH, 0.0, 1.0)) if snap_anchored_volume else 0.0
        if _st > 0.0:
            # Per-SNAP usage tilt (portable across a team change), relative to
            # this backfield's own RB carry/target-per-snap rate.
            prior_snap = _numeric(candidates_team, "prior_whole_snap_share", "prior_active_snap_share")
            prior_gm = _numeric(candidates_team, "prior_games").fillna(0.0)
            team_carry_per_snap = carry_capacity / max(snap_capacity, 0.01)
            team_target_per_snap = target_capacity / max(snap_capacity, 0.01)
            valid_snap = prior_snap > 0.05
            player_carry_ps = (prior_carry_rate / prior_snap.where(valid_snap))
            player_target_ps = (prior_target_rate / prior_snap.where(valid_snap))
            carry_tilt = (player_carry_ps / max(team_carry_per_snap, 0.01)).clip(*RB_VOL_TILT_CARRY)
            target_tilt = (player_target_ps / max(team_target_per_snap, 0.01)).clip(*RB_VOL_TILT_TARGET)
            carry_tilt = carry_tilt.where(np.isfinite(carry_tilt), 1.0)
            target_tilt = target_tilt.where(np.isfinite(target_tilt), 1.0)
            # Cross-team + thin-history players fade toward snap-proportional.
            trust = np.where(same_team, 1.0, RB_VOL_TILT_CROSS_TEAM_TRUST) * (
                (prior_gm / (prior_gm + RB_VOL_TILT_GAMES_K)).clip(0.0, 1.0).to_numpy())
            carry_tilt = 1.0 + trust * (carry_tilt.to_numpy() - 1.0)
            target_tilt = 1.0 + trust * (target_tilt.to_numpy() - 1.0)
            rank_for_disc = (effective_rank if has_chart
                             else base.loc[indexes].rank(ascending=False, method="first"))
            rank_disc = rank_for_disc.map(RB_VOL_RANK_DISCOUNT).fillna(RB_VOL_RANK_DISCOUNT_DEFAULT).to_numpy()
            carry_score_tilt = pd.Series(
                np.clip(snap_alloc.to_numpy() * carry_tilt * rank_disc, 0.005, None)
                * availability.loc[indexes].to_numpy(), index=indexes)
            target_score_tilt = pd.Series(
                np.clip(snap_alloc.to_numpy() * target_tilt * rank_disc, 0.005, None)
                * availability.loc[indexes].to_numpy(), index=indexes)
            # Blend the reconciled ALLOCATIONS (each already conserved to the
            # same capacity) so RB_VOL_TILT_STRENGTH is a clean interpolation
            # between the two carry/target splits - at 1.0 this is byte-for-byte
            # the pre-knob tilt path.
            carry_alloc = ((1.0 - _st) * _bounded_allocation(carry_score_legacy, _carry_cap)
                           + _st * _bounded_allocation(carry_score_tilt, _carry_cap))
            target_alloc = ((1.0 - _st) * _bounded_allocation(target_score_legacy, _target_cap)
                            + _st * _bounded_allocation(target_score_tilt, _target_cap))
        else:
            # Receiving specialists can retain target evidence without inheriting
            # early-down carries; the two vectors intentionally remain separate.
            carry_alloc = _bounded_allocation(carry_score_legacy, _carry_cap)
            target_alloc = _bounded_allocation(target_score_legacy, _target_cap)

        if _no_backstop_new_team.any():
            # Neutralizing the evidence term above (carry_evidence/target_
            # evidence) is not sufficient on its own: _bounded_allocation
            # splits the team's capacity by each player's score RELATIVE TO
            # HIS TEAMMATES, so an untouched teammate's real evidence can
            # still pull the split away from this player's own honest
            # snap-proportional score. This second pass bounds the actual
            # OUTPUT share directly - see _bound_share_toward_snap.
            # The share/target comparison must use the TRUE team capacity
            # (matching snap_fraction's own denominator and what the board
            # displays as carry/target share) - NOT _carry_cap/_target_cap,
            # which are already reduced by other_fraction and would compare
            # two different-sized pies. _bound_share_toward_snap's actual
            # redistribution is an absolute delta, so this is safe either way.
            carry_alloc = _bound_share_toward_snap(
                carry_alloc, carry_capacity, snap_fraction, _no_backstop_new_team,
                RB_WEEK1_CROSS_TEAM_SHARE_TOLERANCE)
            target_alloc = _bound_share_toward_snap(
                target_alloc, target_capacity, snap_fraction, _no_backstop_new_team,
                RB_WEEK1_CROSS_TEAM_SHARE_TOLERANCE)

        out.loc[indexes, "expected_snap_share"] = snap_alloc
        # Keep the allocation in the same unit as the capacity ledger.  The
        # weekly model divides by the team capacity when it needs a fraction
        # for a per-player rate multiplier; keeping this raw here makes
        # conservation visible without an implicit conversion.
        out.loc[indexes, "carry_share"] = carry_alloc
        out.loc[indexes, "target_share"] = target_alloc
        out.loc[indexes, "allocated_carries"] = carry_alloc
        out.loc[indexes, "allocated_targets"] = target_alloc
        out.loc[indexes, "allocation_source"] = np.where(
            incumbent_backstop.loc[indexes] & ~source_rank.loc[indexes].le(3),
            "team-constrained preseason core-RB allocator (incumbent safety backstop)",
            "team-constrained preseason core-RB allocator",
        )
        # core_rb_snaps' ledger capacity is the post-rescale team-share
        # target, not the raw historical snap_capacity - the final rescale
        # above deliberately redistributes the gap between the two (plus the
        # `other RB` residual) back to the projected core RBs, so the ledger
        # must reconcile against the number they were actually rescaled to.
        # Carries/targets are untouched by that rescale and keep reconciling
        # against their own real capacities.
        for metric, capacity, allocation in (("core_rb_snaps", snap_share_target, snap_alloc),
                                             ("rb_carries", carry_capacity, carry_alloc),
                                             ("rb_targets", target_capacity, target_alloc)):
            allocated = float(allocation.sum())
            ledger.append({"team": team, "resource": metric, "capacity": float(capacity),
                           "allocated": allocated, "unallocated": max(0.0, float(capacity) - allocated),
                           "candidate_count": int(len(indexes)), "other_fraction": float(other_fraction),
                           "reason": "Core-RB capacity reconciled; residual held as explicit other RB."})
    return out.drop(columns=["_rb_team"], errors="ignore"), pd.DataFrame(ledger, columns=ledger_columns)


def redistribute_rb_vacancy_with_allocator(result: pd.DataFrame, injury_profiles: dict[str, dict[str, Any]],
                                           as_of_year: int | None = None,
                                           injury_provenance: dict[str, Any] | None = None,
                                           receiver_pecking_order: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Redistribute an out core RB's carries and targets with allocator shares.

    ``injury_provenance`` can state the source season.  A prior-season injury
    feed is never allowed to mutate a new-season Week-1 role; in that case a
    ledger row explains why the pass was skipped instead of silently using
    stale data.

    ``receiver_pecking_order`` (v2_receiver_vacancy_pecking_order) reshapes the
    WR/TE branch so a departed pass catcher's targets flow mostly to the
    players just behind him on the depth chart rather than to the team's
    alpha - see RECEIVER_VACANCY_PECKING_TAPER / _LEAD_DAMPEN / _ABS_GROWTH_FLOOR.
    """
    columns = ["team", "volume", "source_player", "functional_source_role", "injury_provenance",
               "vacated", "allocated", "unallocated", "recipients", "reason"]
    if result is None or result.empty or not injury_profiles:
        return result.copy() if result is not None else pd.DataFrame(), pd.DataFrame(columns=columns)
    provenance = injury_provenance or {}
    required = {"Player", "Team", "Pos"}
    if not required.issubset(result.columns):
        return result.copy(), pd.DataFrame(columns=columns)
    out = result.copy()
    availability = out["Player"].map(lambda player: float(injury_profiles.get(player, {}).get("plays_probability", 1.0)))
    functional = _text(out, "functional_position", "_functional_position", default="").str.upper()
    fallback_core = out["Pos"].eq("RB") & ~functional.eq("FB")
    core = out.get("_rb_core", out.get("Core RB", fallback_core))
    core = pd.Series(core, index=out.index).fillna(False).astype(bool) & ~functional.eq("FB")
    sidelined = availability.le(0.01)
    if not sidelined.any():
        return out, pd.DataFrame(columns=columns)
    ledgers: list[dict[str, Any]] = []
    dependent = {
        "rushing_attempts": ("rushing_yards", "rushing_tds"),
        "targets": ("receptions", "receiving_yards", "receiving_tds"),
    }
    share_columns = {"rushing_attempts": "_rb_carry_allocation_share", "targets": "_rb_target_allocation_share"}
    for source_index in out.index[sidelined]:
        source = out.loc[source_index]
        team = str(source["Team"])
        source_profile = injury_profiles.get(source["Player"], {})
        source_meta = provenance.get(source["Player"], {}) if isinstance(provenance, dict) else {}
        source_year = source_meta.get("year", source_profile.get("source_year")) if isinstance(source_meta, dict) else None
        source_provenance = str((source_meta.get("source") if isinstance(source_meta, dict) else None)
                                or source_profile.get("source", "live target-season report"))
        source_role = functional.loc[source_index] or str(source["Pos"])
        # Preserve the football subposition for allocation logic, while
        # making the audit ledger explicit that this was a *core* RB source
        # rather than an FB or a generic broad roster-RB row.
        source_role_label = (
            "core RB" if source_role == "RB" and bool(core.loc[source_index])
            else source_role
        )
        if as_of_year is not None and source_year is not None and int(source_year) != int(as_of_year):
            ledgers.append({"team": team, "volume": "", "source_player": source["Player"],
                            "functional_source_role": source_role_label,
                            "injury_provenance": f"stale {source_provenance} season {source_year}",
                            "vacated": 0.0, "allocated": 0.0, "unallocated": 0.0, "recipients": [],
                            "reason": "Skipped stale injury source; it does not apply to the target season."})
            continue
        # A fullback's absence is useful roster information but must not be
        # converted into lead-RB carries/targets.  Surface it in the ledger.
        if source_role == "FB":
            ledgers.append({"team": team, "volume": "", "source_player": source["Player"],
                            "functional_source_role": "FB", "injury_provenance": source_provenance,
                            "vacated": 0.0, "allocated": 0.0, "unallocated": 0.0, "recipients": [],
                            "reason": "Fullback excluded from fantasy-RB vacancy pools."})
            continue
        if source_role == "RB" and not core.loc[source_index]:
            continue
        # A WR/TE absence reassigns targets within its own pass-catcher pool;
        # it never feeds a generic RB carry or target boost.
        source_volumes = (("rushing_attempts",) if source_role == "RB" else tuple()) + (("targets",) if source_role in {"RB", "WR", "TE"} else tuple())
        for volume_col in source_volumes:
            dep_cols = dependent[volume_col]
            if volume_col not in out.columns:
                continue
            full_col = f"_full_{volume_col}"
            pre_injury = float(pd.to_numeric(pd.Series([source.get(full_col, source.get(volume_col, 0.0))]),
                                              errors="coerce").fillna(0.0).iloc[0])
            reusable = pre_injury * VACANCY_SURVIVAL
            if source_role == "RB":
                mask = (out["Team"].astype(str).eq(team) & out["Pos"].eq("RB") & core
                        & ~sidelined & availability.gt(0.01))
            else:
                mask = (out["Team"].astype(str).eq(team) & out["Pos"].isin(["WR", "TE"])
                        & ~sidelined & availability.gt(0.01))
            candidates = out.index[mask]
            if reusable <= 0 or not len(candidates):
                ledgers.append({"team": team, "volume": volume_col, "source_player": source["Player"],
                                "functional_source_role": source_role_label, "injury_provenance": source_provenance,
                                "vacated": pre_injury, "allocated": 0.0, "unallocated": reusable,
                                "recipients": [], "reason": "No active, allocator-eligible core-RB recipient."})
                continue
            shares = (_numeric(out.loc[candidates], share_columns[volume_col], default=np.nan).fillna(0.0)
                      if source_role == "RB" else pd.to_numeric(out.loc[candidates, volume_col], errors="coerce").fillna(0.0))
            if shares.sum() <= 0:
                shares = pd.to_numeric(out.loc[candidates, "Expected Snap Share"], errors="coerce").fillna(0.0)
            if shares.sum() <= 0:
                ledgers.append({"team": team, "volume": volume_col, "source_player": source["Player"],
                                "functional_source_role": source_role_label, "injury_provenance": source_provenance,
                                "vacated": pre_injury, "allocated": 0.0, "unallocated": reusable,
                                "recipients": [], "reason": "Eligible recipients had no role allocation evidence."})
                continue
            # "Next man up": rank the healthy recipients by current volume
            # (pecking-order proxy), weight by a geometric rank decay from the
            # top backup down, hold the current lead to a fixed small share,
            # favour the source's own position, and exclude the deep bench
            # outright - not a raw-target split that hands the alpha the
            # plurality.  RB carry/target vacancy keeps its allocator-share
            # split (this branch is WR/TE only).
            pecking = receiver_pecking_order and source_role in {"WR", "TE"} and len(shares) > 1
            if pecking:
                order = shares.rank(method="first", ascending=False)
                weights = pd.Series(
                    np.power(RECEIVER_VACANCY_RANK_DECAY, (order - 2.0).clip(lower=0.0)),
                    index=order.index)
                weights = weights.where(order > 1.0, RECEIVER_VACANCY_LEAD_SHARE)
                weights = weights.where(order <= RECEIVER_VACANCY_PARTICIPATION_RANKS, 0.0)
                cross_pos = out.loc[candidates, "Pos"].astype(str).ne(source_role).to_numpy()
                weights = weights.where(~cross_pos, weights * RECEIVER_VACANCY_CROSS_POS_WEIGHT)
                if weights.sum() > 0:
                    shares = weights
            volume = pd.to_numeric(out.loc[candidates, volume_col], errors="coerce").fillna(0.0)
            requested = shares / shares.sum() * reusable
            allowed = (volume * VACANCY_MAX_GROWTH - volume).clip(lower=0.0)
            if pecking:
                allowed = allowed.clip(lower=RECEIVER_VACANCY_ABS_GROWTH_FLOOR)
            gain = np.minimum(requested, allowed)
            factor = (volume + gain) / volume.replace(0.0, np.nan)
            factor = factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            out.loc[candidates, volume_col] = (volume + gain).round(2)
            for dep_col in dep_cols:
                if dep_col in out.columns:
                    base = pd.to_numeric(out.loc[candidates, dep_col], errors="coerce").fillna(0.0)
                    out.loc[candidates, dep_col] = (base * factor).round(2)
            # team_rank: 1-based standing of each fill-in among the active,
            # role-compatible recipient pool by post-redistribution volume.
            # Display-only metadata for the Deep Dive vacancy table (it trims
            # rows below the trusted tier) - nothing downstream reads it.
            _rank = (pd.to_numeric(out.loc[candidates, volume_col], errors="coerce")
                     .fillna(0.0).rank(ascending=False, method="min"))
            recipients = [{"player": str(out.at[index, "Player"]),
                           "allocated": round(float(gain.loc[index]), 3),
                           "team_rank": int(_rank.loc[index])}
                          for index in candidates if gain.loc[index] > 0]
            allocated = float(gain.sum())
            ledgers.append({"team": team, "volume": volume_col, "source_player": source["Player"],
                            "functional_source_role": source_role_label, "injury_provenance": source_provenance,
                            "vacated": pre_injury, "allocated": allocated,
                            "unallocated": max(0.0, reusable - allocated), "recipients": recipients,
                            "reason": "Allocator-weighted core-RB injury redistribution."})
    return out, pd.DataFrame(ledgers, columns=columns)


# ---------------------------------------------------------------------------
# Historical role segments (additive evidence helpers; no allocator behavior)
# ---------------------------------------------------------------------------


def _segment_bool(values: pd.Series) -> pd.Series:
    """Return an index-preserving boolean series for common feed encodings."""
    raw = pd.Series(values, index=values.index if isinstance(values, pd.Series) else None)
    if pd.api.types.is_bool_dtype(raw):
        return raw.fillna(False).astype(bool)
    numeric = pd.to_numeric(raw, errors="coerce")
    text = raw.astype(object).where(raw.notna(), "").astype(str).str.strip().str.lower().isin(
        {"1", "true", "t", "yes", "y"})
    return numeric.gt(0.0).where(numeric.notna(), text)


def _segment_team_keys(values: pd.Series) -> pd.Series:
    """Normalize NFL team labels without using a current-roster field as truth."""
    team = pd.Series(values).astype(object)
    team = team.where(team.notna(), "").astype(str).str.strip().str.upper()
    team = team.replace({"OAK": "LV", "SD": "LAC", "STL": "LA", "ARZ": "ARI", "JAC": "JAX"})
    return team.where(~team.isin({"", "NAN", "NONE", "<NA>"}), "")


def _segment_identity_keys(frame: pd.DataFrame, player_col: str) -> pd.Series:
    """Build stable, source-independent player keys for historical segments.

    The result deliberately mirrors the app's identity convention: a real
    ID wins over a display name, while the name fallback is canonicalized via
    the small reviewed alias layer.  This helper is not a fuzzy matcher and
    never uses last-name-only logic.
    """
    # Reuse the exact same authoritative hierarchy as the cold-start pool,
    # Ourlads resolver, and cross-season rate joins.  Segment analysis must
    # not quietly create a second identity convention just because its input
    # happens to be a historical weekly frame.
    return stable_roster_identity_keys(frame, player_col).astype(str)


def _immutable_segment_game_team(frame: pd.DataFrame, fallback_team_col: str | None) -> pd.Series:
    """Prefer the offense that played the historical game, never roster hindsight.

    ``game_team`` is the durable field created by the data loader before a
    current-roster merge.  For older caches that lack it, a standard nflverse
    game id plus the recorded opponent can still reconstruct the offense.  A
    visible current team is only the final fallback.
    """
    if frame.empty:
        return pd.Series(dtype=object)
    explicit = (_segment_team_keys(frame["game_team"]) if "game_team" in frame.columns
                else pd.Series("", index=frame.index, dtype=object))
    fallback = (_segment_team_keys(frame[fallback_team_col]) if fallback_team_col in frame.columns
                else pd.Series("", index=frame.index, dtype=object))
    inferred = pd.Series("", index=frame.index, dtype=object)
    opponent_col = _column(frame, "game_opponent", "opponent_team")
    if "game_id" in frame.columns and opponent_col is not None:
        parts = frame["game_id"].astype(str).str.rsplit("_", n=2, expand=True)
        if parts.shape[1] >= 3:
            side_a = _segment_team_keys(parts.iloc[:, -2]).set_axis(frame.index)
            side_b = _segment_team_keys(parts.iloc[:, -1]).set_axis(frame.index)
            opponent = _segment_team_keys(frame[opponent_col])
            inferred = pd.Series(np.where(opponent.eq(side_a), side_b,
                                           np.where(opponent.eq(side_b), side_a, "")),
                                 index=frame.index)
    return explicit.where(explicit.ne(""), inferred.where(inferred.ne(""), fallback))


def _segment_seasons(frame: pd.DataFrame) -> pd.Series:
    season_col = _column(frame, "season", "year", "Season", "Year")
    if season_col is None:
        return pd.Series(0, index=frame.index, dtype=int)
    return pd.to_numeric(frame[season_col], errors="coerce").fillna(0).astype(int)


def _segment_snap_share(frame: pd.DataFrame) -> pd.Series:
    """Read either the app's percentage field or an already fractional fixture."""
    raw = _numeric(frame, "weekly_snap_pct", "snap_pct", "snap_share", default=np.nan)
    finite = raw.dropna().abs()
    # The production loader emits 0-100.  Supporting a fractional input is
    # useful for narrow tests and makes the helper usable by a PFF-derived
    # source without an arbitrary conversion outside this module.
    if not finite.empty and float(finite.max()) <= 1.0:
        return raw.clip(0.0, 1.0)
    return (raw / 100.0).clip(0.0, 1.0)


def _segment_functional_position(frame: pd.DataFrame) -> pd.Series:
    """Classify RB/HB/TB versus FB while accepting sparse historical frames."""
    position = classify_functional_position(frame)
    fallback = _text(frame, "position_group", "Position", "pos").str.upper()
    fallback = fallback.replace({"HB": "RB", "TB": "RB"})
    return position.where(position.ne(""), fallback)


def _segment_calendar(
        history: pd.DataFrame,
        prepared_history: pd.DataFrame,
        team_game_calendar: pd.DataFrame | None,
) -> tuple[pd.DataFrame, str]:
    """Return the actual team-week universe used to measure an absence.

    A caller may pass a schedule/team-game table to make the denominator
    explicit.  Otherwise the full (not RB-filtered) historical table provides
    the universe.  In either case no numeric week arithmetic is used, so a
    bye does not become a fictitious missed game.
    """
    if team_game_calendar is None or team_game_calendar.empty:
        calendar = prepared_history[["_rb_segment_team", "_rb_segment_season", "_rb_segment_week"]].copy()
        source = "history team-week universe"
    else:
        calendar = team_game_calendar.copy()
        fallback_team_col = _column(calendar, "team", "Team")
        calendar["_rb_segment_team"] = _immutable_segment_game_team(calendar, fallback_team_col)
        calendar["_rb_segment_season"] = _segment_seasons(calendar)
        calendar["_rb_segment_week"] = _numeric(calendar, "week", "Week")
        calendar = calendar[["_rb_segment_team", "_rb_segment_season", "_rb_segment_week"]]
        source = "provided team-game calendar"
    calendar = calendar[(calendar["_rb_segment_team"] != "") & calendar["_rb_segment_week"].notna()].copy()
    if calendar.empty:
        return calendar, source
    calendar["_rb_segment_week"] = pd.to_numeric(calendar["_rb_segment_week"], errors="coerce")
    return calendar.drop_duplicates().sort_values(
        ["_rb_segment_season", "_rb_segment_team", "_rb_segment_week"], kind="stable"), source


def _segment_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def _empty_role_segments() -> pd.DataFrame:
    return pd.DataFrame(columns=RB_ROLE_SEGMENT_COLUMNS)


def _empty_teammate_context() -> pd.DataFrame:
    return pd.DataFrame(columns=RB_TEAMMATE_CONTEXT_COLUMNS)


def _select_internal_role_gap(player_weeks: pd.DataFrame, calendar_weeks: list[float]) -> dict[str, Any] | None:
    """Find one high-confidence internal absence followed by a real return.

    Only meaningful snap appearances anchor the pre-gap and return endpoints.
    Crucially, *any* positive observed snap in the candidate gap disqualifies
    it, so an ordinary rotation, a brief benching, or a staged return cannot
    be mislabeled as a clean injury/absence segment.
    """
    meaningful = player_weeks.loc[player_weeks["_meaningful"]].sort_values("_rb_segment_week", kind="stable")
    if len(meaningful) < RB_SEGMENT_MIN_PRE_GAP_GAMES + 1 or not calendar_weeks:
        return None
    calendar_index = {week: i for i, week in enumerate(calendar_weeks)}
    candidates: list[dict[str, Any]] = []
    for later_idx in range(1, len(meaningful)):
        before = meaningful.iloc[later_idx - 1]
        returned = meaningful.iloc[later_idx]
        start_index = calendar_index.get(before["_rb_segment_week"])
        end_index = calendar_index.get(returned["_rb_segment_week"])
        if start_index is None or end_index is None or end_index <= start_index + 1:
            continue
        gap_weeks = calendar_weeks[start_index + 1:end_index]
        if len(gap_weeks) < RB_SEGMENT_MIN_GAP_TEAM_GAMES:
            continue
        pre = meaningful.iloc[:later_idx]
        if len(pre) < RB_SEGMENT_MIN_PRE_GAP_GAMES:
            continue
        # A player who actually played between the anchors is not absent,
        # even if that appearance was too small to establish a normal role.
        active_in_gap = player_weeks.loc[
            player_weeks["_rb_segment_week"].isin(gap_weeks) & player_weeks["_active"]]
        if not active_in_gap.empty:
            continue
        candidates.append({
            "pre": pre,
            "return_start_week": float(returned["_rb_segment_week"]),
            "gap_weeks": gap_weeks,
            "gap_start_week": float(gap_weeks[0]),
            "gap_end_week": float(gap_weeks[-1]),
        })
    if candidates:
        # Multiple clear gaps are rare.  Use the longest one, then prefer the
        # one supported by more meaningful pre-gap games; both tie-breakers
        # are auditable in the returned raw fields.
        candidates.sort(key=lambda item: (len(item["gap_weeks"]), len(item["pre"]), item["return_start_week"]),
                        reverse=True)
        return candidates[0]
    # No return-based gap found. A season-ending absence - the player's
    # season simply stopped, with no later meaningful OR active game to
    # anchor a "return" - is not weaker evidence of a real interruption;
    # requiring an observed return excluded exactly the case it should have
    # covered best (added 2026-08-24, a rookie clearly winning a starting
    # job before a season-ending injury, Cam Skattebo-shaped: he got ZERO
    # incumbent credit purely because he never played again that season).
    last = meaningful.iloc[-1]
    start_index = calendar_index.get(last["_rb_segment_week"])
    if start_index is None or start_index + 1 >= len(calendar_weeks):
        return None
    gap_weeks = calendar_weeks[start_index + 1:]
    if len(gap_weeks) < RB_SEGMENT_MIN_GAP_TEAM_GAMES:
        return None
    active_in_gap = player_weeks.loc[
        player_weeks["_rb_segment_week"].isin(gap_weeks) & player_weeks["_active"]]
    if not active_in_gap.empty:
        return None
    return {
        "pre": meaningful,
        "return_start_week": None,
        "gap_weeks": gap_weeks,
        "gap_start_week": float(gap_weeks[0]),
        "gap_end_week": float(gap_weeks[-1]),
        "season_ending": True,
    }


RB_VACANCY_FULL_STRENGTH_SNAP_DROP = 0.50


def _pre_window_teammate_vacancy_downweight(team_weeks: pd.DataFrame, incumbent_key: str,
                                            incumbent_rows: pd.DataFrame,
                                            pre_weeks: list[float], calendar_weeks: list[float]) -> float:
    """Was the incumbent's own pre-gap stretch just a teammate's injury
    vacancy rather than an earned role?

    ADDED 2026-08-24 per explicit request (the Devin Neal / Travis Etienne
    Saints backfield case): the existing shared_healthy_lead_score /
    replacement_only_era_downweight pair only looks at what happened during
    the INCUMBENT's gap - it is blind to a clean handoff where the two
    players were simply never active in the same week, which is exactly
    what a real injury-driven vacancy looks like (Neal's real 2025 log:
    zero weeks of a meaningful Neal snap share while Kamara was still
    playing a normal workload - Kamara went 73-86% snaps weeks 1-7, then
    collapsed to 14%/out weeks 12-15, which is precisely when Neal's own
    "pre-gap" 65% stretch happened; nowhere in the log did the two share a
    healthy week with Neal leading, so the existing overlap-based check
    never fires and Neal's own late-season role reads as an earned,
    restorable incumbency instead of what it actually was).

    This checks the other direction instead: for each teammate, compare
    their own snap share in the team-calendar weeks BEFORE the incumbent's
    pre-window even started against their snap share DURING that same
    pre-window. A teammate who was clearly the more established back before
    the window (ahead of the incumbent's own level at that time) and then
    collapsed right as the incumbent's stretch began is the actual
    signature of a vacancy-driven promotion - scaled by how big that
    collapse was, capped at 1.0 (full discount) once it reaches a
    RB_VACANCY_FULL_STRENGTH_SNAP_DROP-point drop (a clear starter falling
    to a non-factor). A teammate who was only modestly ahead, or who stayed
    active throughout, does not trigger this - see Cam Skattebo/Tyrone
    Tracy's real log for the negative case this must NOT fire on: Tracy was
    already roughly co-equal with Skattebo early (44% vs. 41%), not clearly
    the incumbent, so Skattebo's real separation reads as earned, not a
    vacancy fill.
    """
    if not pre_weeks:
        return 0.0
    earlier_weeks = [w for w in calendar_weeks if w < min(pre_weeks)]
    if not earlier_weeks:
        return 0.0

    def _avg_snap(frame: pd.DataFrame, weeks: list[float]) -> float:
        indexed = frame.set_index("_rb_segment_week", drop=False) if "_rb_segment_week" in frame.columns else frame
        values = []
        for week in weeks:
            if week not in indexed.index:
                continue
            value = indexed.loc[week, "_snap"]
            if isinstance(value, pd.Series):
                value = pd.to_numeric(value, errors="coerce").max()
            value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.notna(value):
                values.append(float(value))
        return float(np.mean(values)) if values else np.nan

    incumbent_earlier_level = _avg_snap(incumbent_rows, earlier_weeks)
    others = team_weeks.loc[team_weeks["rb_segment_identity_key"].ne(incumbent_key)]
    best_drop = 0.0
    for _, teammate_weeks in others.groupby("rb_segment_identity_key", observed=True):
        teammate_earlier_level = _avg_snap(teammate_weeks, earlier_weeks)
        teammate_during_level = _avg_snap(teammate_weeks, pre_weeks)
        if not (np.isfinite(teammate_earlier_level) and np.isfinite(teammate_during_level)):
            continue
        # The teammate must have clearly been the more established back
        # before the handoff - otherwise a drop from an already-marginal
        # committee role does not establish that the incumbent's rise was a
        # vacancy fill rather than a real change in the pecking order.
        if np.isfinite(incumbent_earlier_level) and teammate_earlier_level <= incumbent_earlier_level:
            continue
        drop = max(0.0, teammate_earlier_level - teammate_during_level)
        best_drop = max(best_drop, drop)
    return float(np.clip(best_drop / RB_VACANCY_FULL_STRENGTH_SNAP_DROP, 0.0, 1.0))


def _absence_replacement_metrics(team_weeks: pd.DataFrame, incumbent_key: str,
                                 gap_weeks: list[float]) -> tuple[int, float, float]:
    """Summarize the RB workload other backs recorded during an incumbent gap."""
    others = team_weeks.loc[team_weeks["rb_segment_identity_key"].ne(incumbent_key)].copy()
    top_values: list[float] = []
    total_values: list[float] = []
    observed_games = 0
    for week in gap_weeks:
        week_rows = others.loc[(others["_rb_segment_week"] == week) & others["_observed"]]
        if not week_rows.empty:
            observed_games += 1
        snap = pd.to_numeric(week_rows.get("_snap", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        top_values.append(float(snap.max()) if not snap.empty else 0.0)
        total_values.append(float(snap.sum()) if not snap.empty else 0.0)
    return (observed_games,
            float(np.mean(top_values)) if top_values else np.nan,
            float(np.mean(total_values)) if total_values else np.nan)


def _teammate_context_for_gap(team_weeks: pd.DataFrame, incumbent: pd.Series,
                              pre_weeks: list[float], gap_weeks: list[float],
                              return_weeks: list[float], pre_snap_share: float,
                              incumbent_credit: float) -> list[dict[str, Any]]:
    """Build directed incumbent -> teammate evidence for one clear absence."""
    incumbent_key = str(incumbent["rb_segment_identity_key"])
    incumbent_name = str(incumbent["rb_segment_player"])
    team = str(incumbent["_rb_segment_team"])
    season = int(incumbent["_rb_segment_season"])
    team_rows = team_weeks.copy()
    incumbent_weeks = team_rows.loc[team_rows["rb_segment_identity_key"].eq(incumbent_key)].set_index(
        "_rb_segment_week", drop=False)
    records: list[dict[str, Any]] = []
    for teammate_key, teammate_weeks in team_rows.loc[
            team_rows["rb_segment_identity_key"].ne(incumbent_key)].groupby("rb_segment_identity_key", observed=True):
        teammate_weeks = teammate_weeks.set_index("_rb_segment_week", drop=False)
        teammate_name = str(teammate_weeks["rb_segment_player"].iloc[-1])
        shared_weeks = [
            week for week in pre_weeks
            if week in incumbent_weeks.index and week in teammate_weeks.index
            and bool(incumbent_weeks.loc[week, "_active"])
            and bool(teammate_weeks.loc[week, "_active"])
        ]
        # In the unlikely event a duplicate survived an upstream source, take
        # the largest snap observation for the game rather than forcing a
        # fragile scalar conversion.
        def snap_at(frame: pd.DataFrame, week: float) -> float:
            if week not in frame.index:
                return 0.0
            value = frame.loc[week, "_snap"]
            if isinstance(value, pd.Series):
                value = pd.to_numeric(value, errors="coerce").max()
            return float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0.0).iloc[0])

        incumbent_shared = float(np.mean([snap_at(incumbent_weeks, week) for week in shared_weeks])) \
            if shared_weeks else np.nan
        teammate_shared = float(np.mean([snap_at(teammate_weeks, week) for week in shared_weeks])) \
            if shared_weeks else np.nan
        denominator = max((0.0 if not np.isfinite(incumbent_shared) else incumbent_shared)
                          + (0.0 if not np.isfinite(teammate_shared) else teammate_shared), 0.15)
        lead_score = (float(incumbent_shared) - float(teammate_shared)) / denominator \
            if shared_weeks else 0.0
        lead_score = float(np.clip(lead_score, -1.0, 1.0))

        teammate_pre = float(np.mean([snap_at(teammate_weeks, week) for week in pre_weeks])) \
            if pre_weeks else 0.0
        teammate_gap = float(np.mean([snap_at(teammate_weeks, week) for week in gap_weeks])) \
            if gap_weeks else 0.0
        teammate_return = float(np.mean([snap_at(teammate_weeks, week) for week in return_weeks])) \
            if return_weeks else np.nan
        # A replacement-only penalty requires positive healthy-era evidence
        # that the returning incumbent led.  It is intentionally capped and
        # exposed rather than applied here; the projection layer decides its
        # final weight and can backtest that decision independently.
        excess_gap_role = max(0.0, teammate_gap - teammate_pre)
        shared_confidence = min(1.0, len(shared_weeks) / 2.0)
        gap_confidence = min(1.0, len(gap_weeks) / 4.0)
        downweight = 0.55 * min(1.0, excess_gap_role / 0.50) * max(0.0, lead_score) \
            * shared_confidence * gap_confidence * float(np.clip(incumbent_credit, 0.0, 1.0))
        records.append({
            "rb_segment_team": team,
            "rb_segment_season": season,
            "incumbent_identity_key": incumbent_key,
            "incumbent_player": incumbent_name,
            "teammate_identity_key": str(teammate_key),
            "teammate_player": teammate_name,
            "shared_healthy_games": int(len(shared_weeks)),
            "incumbent_shared_healthy_snap_share": incumbent_shared,
            "teammate_shared_healthy_snap_share": teammate_shared,
            "shared_healthy_lead_score": lead_score,
            "teammate_pre_absence_snap_share": teammate_pre,
            "absence_replacement_games": int(len(gap_weeks)),
            "teammate_absence_replacement_snap_share": teammate_gap,
            "teammate_return_recovery_snap_share": teammate_return,
            "replacement_only_era_downweight": float(np.clip(downweight, 0.0, 0.55)),
            "context_status": (
                "replacement-only era evidence" if downweight > 0 else "shared healthy context only"
            ),
        })
    return records


def analyze_rb_role_segments(
        history: pd.DataFrame,
        *,
        team_game_calendar: pd.DataFrame | None = None,
        player_col: str | None = None,
        team_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract high-confidence RB absence/return and teammate-role evidence.

    Parameters
    ----------
    history:
        Historical player-week data.  ``game_team`` (when available) is
        authoritative; ``team`` is only a compatibility fallback.  Real snap
        observations are required when a ``has_snap_match`` column exists.
    team_game_calendar:
        Optional regular-season team/week schedule.  Passing it is preferred
        when a caller has one.  Without it, the full history's immutable
        game-team/week rows form the calendar.  Bye weeks therefore never
        count as absences.

    Returns
    -------
    (player_segments, teammate_context)
        ``player_segments`` has one row per player/team/season and exposes
        whole-season, pre-gap, gap/replacement, and recovery evidence.  A
        clear segment requires at least four meaningful pre-gap snap games,
        a gap of at least three *team* games, and a later meaningful return.
        ``teammate_context`` is directed incumbent -> teammate evidence; it
        lets the projection layer distinguish a healthy shared backfield from
        a backup's replacement-only workload during the incumbent's absence.

    This is deliberately evidence extraction only.  It does not change an
    allocation, eligibility decision, or projection on its own.
    """
    if history is None or history.empty:
        return _empty_role_segments(), _empty_teammate_context()
    frame = history.copy()
    player_col = player_col or _column(frame, "player_display_name", "Player", "player", "name", "full_name")
    fallback_team_col = team_col or _column(frame, "team", "Team")
    if player_col is None:
        return _empty_role_segments(), _empty_teammate_context()
    frame["rb_segment_identity_key"] = _segment_identity_keys(frame, player_col)
    frame["rb_segment_player"] = _text(frame, player_col)
    frame["_rb_segment_team"] = _immutable_segment_game_team(frame, fallback_team_col)
    frame["_rb_segment_season"] = _segment_seasons(frame)
    frame["_rb_segment_week"] = _numeric(frame, "week", "Week")
    frame["_snap"] = _segment_snap_share(frame)
    if _column(frame, "has_snap_match") is None:
        frame["_observed"] = frame["_snap"].notna()
    else:
        frame["_observed"] = _segment_bool(frame["has_snap_match"]) & frame["_snap"].notna()
    frame["_active"] = frame["_observed"] & frame["_snap"].gt(0.0)
    frame["_meaningful"] = frame["_observed"] & frame["_snap"].ge(RB_SEGMENT_MIN_MEANINGFUL_SNAP_SHARE)
    frame["_carries"] = _numeric(frame, "rushing_attempts", "carries", "rush_attempts", default=0.0).fillna(0.0)
    frame["_targets"] = _numeric(frame, "targets", "receiving_targets", default=0.0).fillna(0.0)
    calendar, calendar_source = _segment_calendar(history, frame, team_game_calendar)

    frame["_functional_position"] = _segment_functional_position(frame)
    rb = frame.loc[
        frame["_functional_position"].eq("RB")
        & frame["_rb_segment_team"].ne("")
        & frame["_rb_segment_week"].notna()
        & frame["rb_segment_identity_key"].ne("name:")
    ].copy()
    if rb.empty:
        return _empty_role_segments(), _empty_teammate_context()
    # Do not double-count a player-game if two locally merged sources happen
    # to retain it.  The maximum snap/count is conservative for duplicate
    # copies and avoids summing the same game twice.
    group_keys = ["rb_segment_identity_key", "_rb_segment_team", "_rb_segment_season", "_rb_segment_week"]
    player_weeks = rb.groupby(group_keys, as_index=False, observed=True).agg(
        rb_segment_player=("rb_segment_player", "last"),
        _snap=("_snap", "max"), _observed=("_observed", "max"), _active=("_active", "max"),
        _meaningful=("_meaningful", "max"), _carries=("_carries", "max"), _targets=("_targets", "max"),
    )
    player_weeks["_observed"] = player_weeks["_observed"].astype(bool)
    player_weeks["_active"] = player_weeks["_active"].astype(bool)
    player_weeks["_meaningful"] = player_weeks["_meaningful"].astype(bool)
    calendar_by_team = {
        (int(season), str(team)): sorted(group["_rb_segment_week"].astype(float).unique().tolist())
        for (season, team), group in calendar.groupby(["_rb_segment_season", "_rb_segment_team"], observed=True)
    }

    records: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    for (identity, team, season), player_rows in player_weeks.groupby(
            ["rb_segment_identity_key", "_rb_segment_team", "_rb_segment_season"], observed=True):
        player_rows = player_rows.sort_values("_rb_segment_week", kind="stable")
        player_name = str(player_rows["rb_segment_player"].iloc[-1])
        calendar_weeks = calendar_by_team.get((int(season), str(team)), [])
        observed_rows = player_rows.loc[player_rows["_observed"]]
        active_rows = player_rows.loc[player_rows["_active"]]
        team_game_count = len(calendar_weeks)
        record: dict[str, Any] = {
            "rb_segment_identity_key": str(identity),
            "rb_segment_player": player_name,
            "rb_segment_team": str(team),
            "rb_segment_season": int(season),
            "rb_segment_status": "no_snap_data" if observed_rows.empty else "no_clear_internal_gap",
            "rb_segment_calendar_source": calendar_source,
            "rb_segment_has_snap_data": bool(not observed_rows.empty),
            "whole_season_team_games": int(team_game_count),
            "whole_season_observed_snap_games": int(len(observed_rows)),
            "whole_season_active_games": int(len(active_rows)),
            "whole_season_snap_share": (
                float(observed_rows["_snap"].sum() / team_game_count) if team_game_count else np.nan
            ),
            "whole_season_active_snap_share": _segment_mean(active_rows, "_snap"),
            "whole_season_carries_per_game": (
                float(observed_rows["_carries"].sum() / team_game_count) if team_game_count else np.nan
            ),
            "whole_season_targets_per_game": (
                float(observed_rows["_targets"].sum() / team_game_count) if team_game_count else np.nan
            ),
            "interrupted_season": False,
            "pre_absence_games": 0,
            "pre_absence_window_games": 0,
            "pre_absence_start_week": np.nan,
            "pre_absence_end_week": np.nan,
            "pre_absence_snap_share": np.nan,
            "pre_absence_carries_per_game": np.nan,
            "pre_absence_targets_per_game": np.nan,
            "absence_start_week": np.nan,
            "absence_end_week": np.nan,
            "absence_team_games": 0,
            "absence_replacement_observed_games": 0,
            "absence_replacement_top_rb_snap_share": np.nan,
            "absence_replacement_core_rb_snap_share": np.nan,
            "return_recovery_games": 0,
            "return_recovery_start_week": np.nan,
            "return_recovery_end_week": np.nan,
            "return_recovery_snap_share": np.nan,
            "return_recovery_carries_per_game": np.nan,
            "return_recovery_targets_per_game": np.nan,
            "interrupted_incumbent_role_credit": 0.0,
            "pre_window_teammate_vacancy_downweight": 0.0,
        }
        if not observed_rows.empty:
            gap = _select_internal_role_gap(player_rows, calendar_weeks)
            if gap is not None:
                season_ending = bool(gap.get("season_ending"))
                pre_all = gap["pre"]
                pre_window = pre_all.tail(RB_SEGMENT_PRE_GAP_WINDOW)
                gap_weeks = list(gap["gap_weeks"])
                if season_ending:
                    # No later meaningful game exists to anchor a "return" -
                    # the player's season simply stopped. Not weaker evidence
                    # of a real interruption than a mid-season gap-and-return
                    # (see _select_internal_role_gap's own comment): if
                    # anything, a rotation/benching implies SOME later role,
                    # which by definition did not happen here.
                    return_rows = player_rows.iloc[0:0]
                else:
                    return_rows = player_rows.loc[
                        player_rows["_meaningful"]
                        & player_rows["_rb_segment_week"].ge(gap["return_start_week"])
                    ].sort_values("_rb_segment_week", kind="stable")
                pre_snap = _segment_mean(pre_window, "_snap")
                # Evidence strength, not an automatic workload.  The later
                # caller decides how much of this 0..1 credit to use. The
                # return-games confidence factor is dropped (not zeroed) for
                # a season-ending gap - there being no return to measure is
                # the expected shape of this case, not missing evidence.
                return_confidence = 1.0 if season_ending else min(1.0, len(return_rows) / 2.0)
                credit = min(1.0, len(pre_window) / RB_SEGMENT_MIN_PRE_GAP_GAMES) \
                    * min(1.0, len(gap_weeks) / 4.0) * return_confidence \
                    * float(np.clip((pre_snap - 0.20) / 0.45, 0.0, 1.0))
                team_rows = player_weeks.loc[
                    player_weeks["_rb_segment_team"].eq(team)
                    & player_weeks["_rb_segment_season"].eq(season)
                ]
                # Was this pre-window itself just a teammate's injury vacancy
                # rather than an earned role? See the function's own docstring
                # (the real Devin Neal/Alvin Kamara 2025 case this fixes, and
                # the real Cam Skattebo/Tyrone Tracy case it must not fire on).
                # Applied directly to credit - a vacancy-fill stretch should
                # not restore a role that was never really the player's own.
                vacancy_downweight = _pre_window_teammate_vacancy_downweight(
                    team_rows, str(identity), player_rows,
                    pre_window["_rb_segment_week"].astype(float).tolist(), calendar_weeks)
                credit *= (1.0 - vacancy_downweight)
                replacement_observed, replacement_top, replacement_total = _absence_replacement_metrics(
                    team_rows, str(identity), gap_weeks)
                record.update({
                    "rb_segment_status": ("clear_internal_absence_season_ended" if season_ending
                                          else "clear_internal_absence_return"),
                    "interrupted_season": True,
                    "pre_absence_games": int(len(pre_all)),
                    "pre_absence_window_games": int(len(pre_window)),
                    "pre_absence_start_week": float(pre_window["_rb_segment_week"].iloc[0]),
                    "pre_absence_end_week": float(pre_window["_rb_segment_week"].iloc[-1]),
                    "pre_absence_snap_share": pre_snap,
                    "pre_absence_carries_per_game": _segment_mean(pre_window, "_carries"),
                    "pre_absence_targets_per_game": _segment_mean(pre_window, "_targets"),
                    "absence_start_week": gap["gap_start_week"],
                    "absence_end_week": gap["gap_end_week"],
                    "absence_team_games": int(len(gap_weeks)),
                    "absence_replacement_observed_games": int(replacement_observed),
                    "absence_replacement_top_rb_snap_share": replacement_top,
                    "absence_replacement_core_rb_snap_share": replacement_total,
                    "return_recovery_games": int(len(return_rows)),
                    "return_recovery_start_week": (
                        float(return_rows["_rb_segment_week"].iloc[0]) if not return_rows.empty else np.nan),
                    "return_recovery_end_week": (
                        float(return_rows["_rb_segment_week"].iloc[-1]) if not return_rows.empty else np.nan),
                    "return_recovery_snap_share": _segment_mean(return_rows, "_snap"),
                    "return_recovery_carries_per_game": _segment_mean(return_rows, "_carries"),
                    "return_recovery_targets_per_game": _segment_mean(return_rows, "_targets"),
                    "interrupted_incumbent_role_credit": float(np.clip(credit, 0.0, 1.0)),
                    "pre_window_teammate_vacancy_downweight": vacancy_downweight,
                })
                pre_weeks = pre_all["_rb_segment_week"].astype(float).tolist()
                return_weeks = return_rows["_rb_segment_week"].astype(float).tolist()
                contexts.extend(_teammate_context_for_gap(
                    team_rows, pd.Series({
                        "rb_segment_identity_key": identity,
                        "rb_segment_player": player_name,
                        "_rb_segment_team": team,
                        "_rb_segment_season": season,
                    }), pre_weeks, gap_weeks, return_weeks, pre_snap, float(credit)))
        records.append(record)
    segments = pd.DataFrame(records).reindex(columns=RB_ROLE_SEGMENT_COLUMNS)
    context = pd.DataFrame(contexts).reindex(columns=RB_TEAMMATE_CONTEXT_COLUMNS)
    return segments, context


def derive_rb_allocator_segment_fields(
        candidates: pd.DataFrame,
        player_segments: pd.DataFrame | None,
        teammate_context: pd.DataFrame | None,
        *,
        player_col: str | None = None,
        team_col: str | None = None,
) -> pd.DataFrame:
    """Attach role-segment evidence to allocator candidates without changing roles.

    This is the narrow bridge for ``weekly_projections``.  It intentionally
    does not call ``allocate_preseason_rb_roles`` or alter its eligibility.
    The projection layer can choose a backtested coefficient for the three
    direct inputs it needs:

    * ``interrupted_incumbent_role_credit`` (0..1),
    * ``shared_healthy_lead_score`` (-1..1), and
    * ``replacement_only_era_downweight`` (0..0.55).

    All raw source evidence is retained under the ``rb_segment_*`` prefix so
    the UI can explain any later projection adjustment.
    """
    if candidates is None:
        return pd.DataFrame()
    out = candidates.copy()
    if out.empty:
        for column in ("interrupted_incumbent_role_credit", "shared_healthy_lead_score",
                       "replacement_only_era_downweight"):
            if column not in out.columns:
                out[column] = pd.Series(dtype=float)
        return out
    player_col = player_col or _column(out, "Player", "player", "name", "full_name", "player_display_name")
    team_col = team_col or _column(out, "team", "Team")
    if player_col is None or team_col is None:
        raise ValueError("RB role-segment attachment requires a player and team column.")
    out["rb_segment_identity_key"] = _segment_identity_keys(out, player_col)
    out["_rb_segment_candidate_team"] = _segment_team_keys(out[team_col])
    out["_rb_segment_row_order"] = np.arange(len(out))

    segments = player_segments.copy() if isinstance(player_segments, pd.DataFrame) else _empty_role_segments()
    required_segment_keys = {"rb_segment_identity_key", "rb_segment_team"}
    if not required_segment_keys.issubset(segments.columns):
        segments = _empty_role_segments()
    if not segments.empty:
        segments = segments.copy()
        segments["rb_segment_team"] = _segment_team_keys(segments["rb_segment_team"])
        segments["_rb_segment_sort_season"] = pd.to_numeric(
            segments.get("rb_segment_season", pd.Series(0, index=segments.index)), errors="coerce").fillna(0)
        # A player can have several historical seasons.  Same-team continuity
        # receives only the most recent source segment, never a blend that
        # could revive a long-gone replacement role.
        segments = segments.sort_values("_rb_segment_sort_season", kind="stable").drop_duplicates(
            ["rb_segment_identity_key", "rb_segment_team"], keep="last")
        raw_columns = [column for column in segments.columns if column not in {
            "rb_segment_identity_key", "rb_segment_team", "_rb_segment_sort_season"
        }]
        rename = {
            column: (column if column.startswith("rb_segment_") else f"rb_segment_{column}")
            for column in raw_columns
        }
        merged = segments[["rb_segment_identity_key", "rb_segment_team"] + raw_columns].rename(columns=rename)
        # Make the helper idempotent if a caller refreshes a cached result.
        out = out.drop(columns=[column for column in merged.columns if column in out.columns
                                and column not in {"rb_segment_identity_key"}], errors="ignore")
        out = out.merge(merged, how="left", left_on=["rb_segment_identity_key", "_rb_segment_candidate_team"],
                        right_on=["rb_segment_identity_key", "rb_segment_team"], sort=False)
    else:
        out["rb_segment_team"] = pd.Series(np.nan, index=out.index)

    out["rb_segment_match_found"] = out.get("rb_segment_team", pd.Series(np.nan, index=out.index)).notna()
    credit = pd.to_numeric(out.get("rb_segment_interrupted_incumbent_role_credit",
                                   pd.Series(np.nan, index=out.index)), errors="coerce")

    context = teammate_context.copy() if isinstance(teammate_context, pd.DataFrame) else _empty_teammate_context()
    score = pd.Series(np.nan, index=out.index, dtype=float)
    downweight = pd.Series(np.nan, index=out.index, dtype=float)
    if not context.empty and {"rb_segment_team", "incumbent_identity_key", "teammate_identity_key"}.issubset(context.columns):
        context["rb_segment_team"] = _segment_team_keys(context["rb_segment_team"])
        shared_weight = pd.to_numeric(context.get("shared_healthy_games", 0), errors="coerce").fillna(0.0).clip(lower=1.0)
        lead = pd.to_numeric(context.get("shared_healthy_lead_score", 0), errors="coerce").fillna(0.0)
        incumbent_rows = pd.DataFrame({
            "rb_segment_identity_key": context["incumbent_identity_key"].astype(str),
            "_rb_segment_candidate_team": context["rb_segment_team"],
            "_score": lead,
            "_weight": shared_weight,
            "_downweight": 0.0,
        })
        teammate_rows = pd.DataFrame({
            "rb_segment_identity_key": context["teammate_identity_key"].astype(str),
            "_rb_segment_candidate_team": context["rb_segment_team"],
            "_score": -lead,
            "_weight": shared_weight,
            "_downweight": pd.to_numeric(context.get("replacement_only_era_downweight", 0),
                                            errors="coerce").fillna(0.0),
        })
        contributions = pd.concat([incumbent_rows, teammate_rows], ignore_index=True)
        contributions["_weighted_score"] = contributions["_score"] * contributions["_weight"]
        aggregate = contributions.groupby(["rb_segment_identity_key", "_rb_segment_candidate_team"], observed=True).agg(
            _weighted_score=("_weighted_score", "sum"), _weight=("_weight", "sum"),
            _downweight=("_downweight", "max"),
        ).reset_index()
        aggregate["_rb_segment_shared_score"] = (aggregate["_weighted_score"] / aggregate["_weight"])
        out = out.merge(aggregate[["rb_segment_identity_key", "_rb_segment_candidate_team",
                                   "_rb_segment_shared_score", "_downweight"]],
                        how="left", on=["rb_segment_identity_key", "_rb_segment_candidate_team"], sort=False)
        score = pd.to_numeric(out.get("_rb_segment_shared_score", pd.Series(np.nan, index=out.index)),
                              errors="coerce")
        downweight = pd.to_numeric(out.get("_downweight", pd.Series(np.nan, index=out.index)), errors="coerce")

    def retained_numeric(column: str, derived: pd.Series, lower: float, upper: float) -> pd.Series:
        existing = pd.to_numeric(out[column], errors="coerce") if column in out.columns else pd.Series(
            np.nan, index=out.index, dtype=float)
        return derived.where(derived.notna(), existing).fillna(0.0).clip(lower, upper)

    out["interrupted_incumbent_role_credit"] = retained_numeric(
        "interrupted_incumbent_role_credit", credit, 0.0, 1.0)
    out["shared_healthy_lead_score"] = retained_numeric(
        "shared_healthy_lead_score", score, -1.0, 1.0)
    out["replacement_only_era_downweight"] = retained_numeric(
        "replacement_only_era_downweight", downweight, 0.0, 0.55)
    out = out.sort_values("_rb_segment_row_order", kind="stable")
    original_index = candidates.index
    out.index = original_index
    return out.drop(columns=["_rb_segment_candidate_team", "_rb_segment_row_order", "_downweight",
                             "_rb_segment_shared_score"], errors="ignore")
