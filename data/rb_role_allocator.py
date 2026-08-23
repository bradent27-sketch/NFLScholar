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

from typing import Any

import numpy as np
import pandas as pd

from data.player_aliases import stable_roster_identity_keys


INELIGIBLE_ROSTER_STATUSES = frozenset({"RET", "CUT", "RES", "FA", "SUS", "NFI", "PUP"})
DEFAULT_CORE_RB_SNAP_CAPACITY = 1.00
DEFAULT_RB_CARRY_CAPACITY = 21.0
DEFAULT_RB_TARGET_CAPACITY = 5.0
MAX_INDIVIDUAL_CORE_RB_SNAP_SHARE = 0.86
VACANCY_SURVIVAL = 0.80
VACANCY_MAX_GROWTH = 2.00

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
    "interrupted_incumbent_role_credit",
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
    interrupted = _text(group, "interrupted_season", default="").str.lower().isin({"1", "true", "yes"})
    pre_gap = (pre_absence - base).clip(lower=0.0).fillna(0.0)
    role = base + np.where(interrupted, 0.45 * pre_gap, 0.20 * pre_gap)
    # A clear internal absence + return is stronger evidence than an ordinary
    # late-season missed stretch.  Keep the added credit bounded and only
    # move toward the player's documented *pre-gap* role; it cannot invent a
    # bell-cow role from a small depth-chart sample.
    incumbent_credit = _numeric(group, "interrupted_incumbent_role_credit", default=0.0).fillna(0.0).clip(0.0, 1.0)
    role += 0.30 * incumbent_credit * pre_gap
    # Shared-healthy teammate data distinguishes an incumbent's role before
    # injury from a replacement's temporary absence-era workload.  A two-game
    # shared sample is not a verdict, so these are modest, capped score
    # adjustments—visible separately in the allocator input and popup.
    shared_lead = _numeric(group, "shared_healthy_lead_score", default=0.0).fillna(0.0).clip(-1.0, 1.0)
    replacement_downweight = _numeric(group, "replacement_only_era_downweight", default=0.0).fillna(0.0).clip(0.0, 0.55)
    teammate_multiplier = np.clip(1.0 + 0.10 * shared_lead - 0.35 * replacement_downweight, 0.75, 1.15)
    return (role * teammate_multiplier).clip(0.0, 0.95)


def allocate_preseason_rb_roles(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconcile core-RB Week-1 roles to independent team capacities.

    Required inputs are intentionally modest.  The public names documented in
    the V2 handoff are accepted in either lower-case or UI-style title case;
    omitted evidence falls back to a neutral/explicit residual rather than a
    fabricated 15.7% reserve share.
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
    fallback_credible = (
        # ``base`` can be the position-median fallback for a rostered player
        # with no real prior sample.  Require a real observed role before it
        # can admit an uncharted veteran; otherwise an unknown DET/PIT
        # reserve gets the same synthetic 16%-ish role as a proven player.
        (has_observed_prior_role & prior_games.ge(4) & base.ge(0.10))
        | (draft_capital.ge(1) & draft_capital.le(150))
    )
    # Identity/source failures must never erase a real same-team incumbent
    # who had a strong proven role.  This is an eligibility safety net—not a
    # free workload: the normal finite-capacity scoring still decides his
    # share and deep/unknown reserves cannot meet the evidence guard.
    incumbent_backstop = _text(out, "established_incumbent_backstop", default="").str.lower().isin(
        {"1", "true", "yes", "y"})
    credible = np.where(team_has_chart, rank.le(3) | incumbent_backstop, fallback_credible)
    eligible = core & credible & availability.gt(0.01)
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
        np.where(~core, "not a functional RB",
                 np.where(rank.le(3), "literal Ourlads top-three role",
                          np.where(incumbent_backstop,
                                   "established same-team incumbent safety backstop",
                                   np.where(fallback_credible, "observed role/draft fallback",
                                            "no credible role evidence")))))
    ledger: list[dict[str, Any]] = []

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
        # Preserve an established active-game role much more strongly than a
        # plain linear split.  A 78% lead-back role should not be averaged
        # down to 34% merely because a chart also lists two reserves; the
        # exponent is a bounded concentration, not a hard depth-chart lock.
        score = base.loc[indexes].clip(lower=0.01).pow(1.45)
        low_evidence = score.lt(0.30)
        score *= np.where(source_rank.eq(1) & low_evidence, 1.25,
                          np.where(source_rank.eq(1), 1.08,
                                   np.where(source_rank.eq(2), 0.98,
                                            np.where(source_rank.eq(3), 0.88, 1.0))))
        draft_bonus = (0.20 * (1.0 - (draft_capital.loc[indexes].fillna(999.0) - 1.0) / 149.0)
                       .clip(0.0, 1.0))
        score += np.where(is_rookie.loc[indexes] | low_evidence, draft_bonus, 0.0)
        same_team = _text(candidates_team, "same_team", default="").str.lower().isin({"1", "true", "yes"})
        score += np.where(same_team, 0.02, 0.0)
        score *= availability.loc[indexes]
        charted = int(source_rank.le(3).sum())
        other_fraction = _other_fraction(charted)
        player_snap_total = snap_capacity * (1.0 - other_fraction)
        snap_alloc = _bounded_allocation(score, player_snap_total, MAX_INDIVIDUAL_CORE_RB_SNAP_SHARE)

        prior_carry_rate = _numeric(candidates_team, "prior_carries_per_game", "prior_carry_rate", "prior_carries")
        prior_target_rate = _numeric(candidates_team, "prior_targets_per_game", "prior_target_rate", "prior_targets")
        snap_fraction = (snap_alloc / max(snap_capacity, 0.01)).clip(0.0, 1.0)
        carry_evidence = (prior_carry_rate / max(carry_capacity, 0.01)).clip(lower=0.0).fillna(snap_fraction)
        target_evidence = (prior_target_rate / max(target_capacity, 0.01)).clip(lower=0.0).fillna(snap_fraction)
        carry_score = (0.62 * snap_fraction + 0.38 * carry_evidence).clip(lower=0.005) * availability.loc[indexes]
        target_score = (0.50 * snap_fraction + 0.50 * target_evidence).clip(lower=0.005) * availability.loc[indexes]
        # Receiving specialists can retain target evidence without inheriting
        # early-down carries; the two vectors intentionally remain separate.
        carry_alloc = _bounded_allocation(carry_score, carry_capacity * (1.0 - other_fraction))
        target_alloc = _bounded_allocation(target_score, target_capacity * (1.0 - other_fraction))

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
        for metric, capacity, allocation in (("core_rb_snaps", snap_capacity, snap_alloc),
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
                                           injury_provenance: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Redistribute an out core RB's carries and targets with allocator shares.

    ``injury_provenance`` can state the source season.  A prior-season injury
    feed is never allowed to mutate a new-season Week-1 role; in that case a
    ledger row explains why the pass was skipped instead of silently using
    stale data.
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
            volume = pd.to_numeric(out.loc[candidates, volume_col], errors="coerce").fillna(0.0)
            requested = shares / shares.sum() * reusable
            allowed = (volume * VACANCY_MAX_GROWTH - volume).clip(lower=0.0)
            gain = np.minimum(requested, allowed)
            factor = (volume + gain) / volume.replace(0.0, np.nan)
            factor = factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            out.loc[candidates, volume_col] = (volume + gain).round(2)
            for dep_col in dep_cols:
                if dep_col in out.columns:
                    base = pd.to_numeric(out.loc[candidates, dep_col], errors="coerce").fillna(0.0)
                    out.loc[candidates, dep_col] = (base * factor).round(2)
            recipients = [{"player": str(out.at[index, "Player"]), "allocated": round(float(gain.loc[index]), 3)}
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
    if not candidates:
        return None
    # Multiple clear gaps are rare.  Use the longest one, then prefer the
    # one supported by more meaningful pre-gap games; both tie-breakers are
    # auditable in the returned raw fields.
    candidates.sort(key=lambda item: (len(item["gap_weeks"]), len(item["pre"]), item["return_start_week"]),
                    reverse=True)
    return candidates[0]


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
        }
        if not observed_rows.empty:
            gap = _select_internal_role_gap(player_rows, calendar_weeks)
            if gap is not None:
                pre_all = gap["pre"]
                pre_window = pre_all.tail(RB_SEGMENT_PRE_GAP_WINDOW)
                gap_weeks = list(gap["gap_weeks"])
                return_rows = player_rows.loc[
                    player_rows["_meaningful"]
                    & player_rows["_rb_segment_week"].ge(gap["return_start_week"])
                ].sort_values("_rb_segment_week", kind="stable")
                pre_snap = _segment_mean(pre_window, "_snap")
                # Evidence strength, not an automatic workload.  The later
                # caller decides how much of this 0..1 credit to use.
                credit = min(1.0, len(pre_window) / RB_SEGMENT_MIN_PRE_GAP_GAMES) \
                    * min(1.0, len(gap_weeks) / 4.0) * min(1.0, len(return_rows) / 2.0) \
                    * float(np.clip((pre_snap - 0.20) / 0.45, 0.0, 1.0))
                team_rows = player_weeks.loc[
                    player_weeks["_rb_segment_team"].eq(team)
                    & player_weeks["_rb_segment_season"].eq(season)
                ]
                replacement_observed, replacement_top, replacement_total = _absence_replacement_metrics(
                    team_rows, str(identity), gap_weeks)
                record.update({
                    "rb_segment_status": "clear_internal_absence_return",
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
                    "return_recovery_start_week": float(return_rows["_rb_segment_week"].iloc[0]),
                    "return_recovery_end_week": float(return_rows["_rb_segment_week"].iloc[-1]),
                    "return_recovery_snap_share": _segment_mean(return_rows, "_snap"),
                    "return_recovery_carries_per_game": _segment_mean(return_rows, "_carries"),
                    "return_recovery_targets_per_game": _segment_mean(return_rows, "_targets"),
                    "interrupted_incumbent_role_credit": float(np.clip(credit, 0.0, 1.0)),
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
