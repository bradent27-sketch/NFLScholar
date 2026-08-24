"""Time-safe local PFF receiving-alignment archive helpers.

This module deliberately does *not* fetch, scrape, or automate PFF.  It
only reads manually exported local CSVs from the following reviewable layout::

    pff_imports/{year}/weekly/{week}/receiving_summary.csv
    pff_imports/{year}/weekly/{week}/receiving_concept.csv
    pff_imports/{year}/weekly/manifest.csv        # optional, recommended

The weekly reports cover WR, TE, HB/RB, and FB in one export.  A report is
eligible for a target week only when ``report_week < as_of_week``; this is a
hard guard against using future information in a projection or a backtest.

There are two intentionally separate source types:

* ``weekly_archive`` is the time-valid source for in-season alignment work.
* ``season_prior`` is allowed only when a caller explicitly confirms that a
  prior-season export is regular-season-only.  Existing PFF season totals can
  include playoffs, so the loader never silently treats an unmanifested total
  as a clean Week 1 prior.

The primary output is a *player role profile*, not a matchup multiplier.  A
separate, still-neutral defense foundation can also aggregate the same manual
weekly **offensive** reports by schedule-mapped defense.  It never reads PFF's
defender-level ``slot_coverage`` report and always returns an applied
multiplier of 1.0.  In particular, a TE's non-slot rate remains labelled
``non_slot`` rather than being asserted to be inline, and RB alignment is
marked experimental.  This keeps the foundation useful without
double-counting the broad defense model before a separately tested
defense-alignment residual exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import re

import pandas as pd


PFF_ALIGNMENT_VERSION = "v1_player_role_foundation"
# This is deliberately separate from the player-role profile version above.
# The defense side is built from *offensive* weekly reports mapped through the
# schedule, never from PFF's defender-level ``slot_coverage`` report.  It is
# an auditable research foundation only: no caller in this module applies its
# candidate residual to a projection.
PFF_ALIGNMENT_DEFENSE_VERSION = "v1_offensive_weekly_defense_foundation"
DEFAULT_PFF_ROOT = Path("pff_imports")
ALIGNMENT_CONFIDENCE_HALF_WEIGHT = 60.0

# A short archive can produce dramatic slot/non-slot splits.  These constants
# keep the stored defense evidence bounded and make its uncertainty explicit.
# They are intentionally conservative starting defaults, not fitted claims of
# accuracy; a future backtest may tune them behind a named experiment.
ALIGNMENT_DEFENSE_SHRINKAGE_GAMES = 4.0
ALIGNMENT_DEFENSE_RAW_RATIO_CLIP = (0.50, 1.75)
ALIGNMENT_DEFENSE_RESIDUAL_CLIP = (0.90, 1.10)
ALIGNMENT_DEFENSE_EVENT_HALF_WEIGHT = 20.0
ALIGNMENT_DEFENSE_STATS = ("targets", "receptions", "yards", "touchdowns")
ALIGNMENT_DEFENSE_SUPPORTED_POSITIONS = {"WR", "TE"}

# The profile table is intentionally stable even when no local files are
# available.  That lets callers use an empty frame / neutral lookup instead of
# treating a missing source as a zero-rate player alignment.
PROFILE_COLUMNS = [
    "player_id",
    "player",
    "position",
    "team",
    "identity_key",
    "identity_quality",
    "slot_alignment_rate",
    "non_slot_alignment_rate",
    "wide_alignment_rate",
    "inline_alignment_rate",
    "alignment_sample_weight",
    "alignment_confidence",
    "slot_snaps",
    "wide_snaps",
    "inline_snaps",
    "routes",
    "slot_routes",
    "slot_targets",
    "slot_receptions",
    "slot_yards",
    "slot_touchdowns",
    "non_slot_targets",
    "non_slot_receptions",
    "non_slot_yards",
    "non_slot_touchdowns",
    "slot_catch_rate",
    "slot_yards_per_target",
    "non_slot_catch_rate",
    "non_slot_yards_per_target",
    "alignment_available",
    "alignment_experimental",
    "alignment_effect_weight",
    "alignment_matchup_multiplier",
    "alignment_semantics",
    "source_kind",
    "source_year",
    "source_weeks",
    "source_week_count",
    "source_regular_season",
    "source_time_valid",
    "source_confidence",
    "source_paths",
    "source_notes",
    "profile_version",
]

ARCHIVE_COLUMNS = [
    "year",
    "week",
    "summary_path",
    "concept_path",
    "scheme_path",
    "scheme_present",
    "pair_complete",
    "manifest_present",
    "regular_season",
    "regular_season_status",
    "manifest_schema_valid",
    "manifest_source_confidence",
    "manifest_export_date",
    "eligible_as_of",
    "schema_valid",
    "schema_issues",
]

# One row per offense-versus-defense game, receiving position, alignment, and
# receiving event.  ``observed_value`` is derived from the week's offensive
# PFF report: total receiving event volume minus the report's real slot event
# volume produces the explicitly-labelled ``non_slot`` remainder.  It is not
# called "wide" because a TE's non-slot work can contain inline and wide work.
ALIGNMENT_TEAM_GAME_COLUMNS = [
    "source_year",
    "source_week",
    "offense_team",
    "defense_team",
    "position",
    "alignment",
    "stat",
    "observed_value",
    "event_available",
    "event_complete",
    "player_rows",
    "valid_player_rows",
    "alignment_snaps",
    "alignment_exposure_source",
    "source_regular_season",
    "source_time_valid",
    "source_confidence",
    "summary_path",
    "concept_path",
    "source_paths",
    "source_notes",
]

# One current-season, time-valid defensive profile for a position/alignment
# event.  These profiles describe evidence and a *candidate* residual only;
# ``scoring_active`` is always false in this foundation.
ALIGNMENT_DEFENSE_PROFILE_COLUMNS = [
    "defense_team",
    "position",
    "alignment",
    "stat",
    "profile_available",
    "candidate_available",
    "scoring_active",
    "observed_total",
    "expected_total",
    "raw_allowed_ratio",
    "clipped_allowed_ratio",
    "shrunk_allowed_ratio",
    "sample_games",
    "comparison_games",
    "uncompared_games",
    "sample_offenses",
    "event_sample_weight",
    "shrinkage_games",
    "shrinkage_weight",
    "alignment_confidence",
    "position_normal_slot_rate",
    "normal_alignment_source",
    "baseline_source",
    "fallback_reason",
    "source_kind",
    "source_method",
    "source_year",
    "source_weeks",
    "source_week_count",
    "source_regular_season",
    "source_time_valid",
    "source_confidence",
    "source_paths",
    "source_notes",
    "profile_version",
]

_SUMMARY_ALIASES = {
    "player": ("player", "player_name", "name"),
    "player_id": ("player_id", "pff_player_id", "pff_id"),
    "position": ("position", "pos"),
    "team_name": ("team_name", "team"),
    "slot_rate": ("slot_rate", "slot_pct", "slot_percent"),
    "wide_rate": ("wide_rate", "wide_pct", "wide_percent"),
    "inline_rate": ("inline_rate", "inline_pct", "inline_percent"),
    "slot_snaps": ("slot_snaps",),
    "wide_snaps": ("wide_snaps",),
    "inline_snaps": ("inline_snaps",),
    "routes": ("routes", "route_count"),
    "player_game_count": ("player_game_count", "games", "games_played"),
}

_CONCEPT_ALIASES = {
    "player": ("player", "player_name", "name"),
    "player_id": ("player_id", "pff_player_id", "pff_id"),
    "position": ("position", "pos"),
    "team_name": ("team_name", "team"),
    "slot_routes": ("slot_routes",),
    "slot_targets": ("slot_targets",),
    "slot_receptions": ("slot_receptions",),
    "slot_yards": ("slot_yards",),
    "slot_touchdowns": ("slot_touchdowns", "slot_tds"),
}

_MANIFEST_ALIASES = {
    "week": ("week", "week_number"),
    "regular_season": ("regular_season", "is_regular_season", "season_type", "game_type"),
    "schema_valid": ("schema_valid", "schema_validated", "validated"),
    "source_confidence": ("source_confidence", "confidence"),
    "export_date": ("export_date", "exported_at", "export_timestamp"),
    "time_valid": ("time_valid", "as_of_valid", "historically_time_valid"),
}

_SEASON_METADATA_ALIASES = {
    **_MANIFEST_ALIASES,
    "regular_season": (
        "regular_season",
        "regular_season_only",
        "is_regular_season",
        "season_type",
        "game_type",
    ),
}

_PFF_POSITION_MAP = {"HB": "RB", "RB": "RB", "WR": "WR", "TE": "TE", "FB": "FB"}
_SUPPORTED_RECEIVING_POSITIONS = set(_PFF_POSITION_MAP.values())
_SLOT_EVENT_COLUMNS = (
    "slot_routes",
    "slot_targets",
    "slot_receptions",
    "slot_yards",
    "slot_touchdowns",
)

# ---------------------------------------------------------------------------
# Weekly man/zone receiving-scheme tendency (player side only)
# ---------------------------------------------------------------------------
# receiving_scheme.csv is self-contained: man_* and zone_* already partition
# every route/target PFF assigned a coverage shell to, so unlike the slot
# alignment profile above there is no second "total" report to merge in -
# one file, one week, one validated table. This is deliberately a separate
# schema from PROFILE_COLUMNS (not a set of extra columns bolted onto it):
# alignment (slot/wide/inline) and scheme (man/zone) are two independent
# dimensions of the same route, not alternate labels for one dimension - a
# player has both a slot rate AND a man rate at once.

PFF_SCHEME_VERSION = "v1_player_scheme_tendency_foundation"

SCHEME_PROFILE_COLUMNS = [
    "player_id",
    "player",
    "position",
    "team",
    "identity_key",
    "identity_quality",
    "man_route_share",
    "zone_route_share",
    "scheme_sample_weight",
    "scheme_confidence",
    "man_routes",
    "man_targets",
    "man_receptions",
    "man_yards",
    "man_touchdowns",
    "man_catch_rate",
    "man_yards_per_target",
    "man_yprr",
    "zone_routes",
    "zone_targets",
    "zone_receptions",
    "zone_yards",
    "zone_touchdowns",
    "zone_catch_rate",
    "zone_yards_per_target",
    "zone_yprr",
    "scheme_available",
    "scheme_semantics",
    "source_kind",
    "source_year",
    "source_weeks",
    "source_week_count",
    "source_regular_season",
    "source_time_valid",
    "source_confidence",
    "source_paths",
    "source_notes",
    "profile_version",
]

_SCHEME_ALIASES = {
    "player": ("player", "player_name", "name"),
    "player_id": ("player_id", "pff_player_id", "pff_id"),
    "position": ("position", "pos"),
    "team_name": ("team_name", "team"),
    "man_routes": ("man_routes",),
    "man_targets": ("man_targets",),
    "man_receptions": ("man_receptions",),
    "man_yards": ("man_yards",),
    "man_touchdowns": ("man_touchdowns", "man_tds"),
    "man_yprr": ("man_yprr",),
    "zone_routes": ("zone_routes",),
    "zone_targets": ("zone_targets",),
    "zone_receptions": ("zone_receptions",),
    "zone_yards": ("zone_yards",),
    "zone_touchdowns": ("zone_touchdowns", "zone_tds"),
    "zone_yprr": ("zone_yprr",),
    "player_game_count": ("player_game_count", "games", "games_played"),
}

_SCHEME_EVENT_COLUMNS = (
    "man_routes", "man_targets", "man_receptions", "man_yards", "man_touchdowns", "man_yprr",
    "zone_routes", "zone_targets", "zone_receptions", "zone_yards", "zone_touchdowns", "zone_yprr",
)


@dataclass
class AlignmentLoadResult:
    """A non-throwing local-load result with source/audit metadata.

    ``profiles`` always has :data:`PROFILE_COLUMNS`; ``archives`` always has
    :data:`ARCHIVE_COLUMNS` for weekly loads (or one comparable season-prior
    record).  ``issues`` is intended for an app caption or decomposition
    trace, not just a log file that a user never sees.
    """

    profiles: pd.DataFrame
    archives: pd.DataFrame
    issues: list[str]
    metadata: dict[str, Any]

    @property
    def available(self) -> bool:
        return bool(self.metadata.get("available", False))


@dataclass
class AlignmentDefenseLoadResult:
    """Time-valid, schedule-mapped alignment-defense research output.

    ``team_games`` is intentionally retained alongside the aggregated
    ``profiles`` so a later UI/decomposition can show the actual offensive
    team-games that created a defense signal.  This avoids turning a small
    sample into an unexplained matchup multiplier.

    The result never means that scoring has been changed.  Its metadata and
    every profile record carry ``scoring_active=False``; callers may surface
    candidate evidence in an audit panel only until an out-of-sample release
    gate explicitly authorizes an experiment.
    """

    profiles: pd.DataFrame
    team_games: pd.DataFrame
    archives: pd.DataFrame
    issues: list[str]
    metadata: dict[str, Any]

    @property
    def available(self) -> bool:
        return bool(self.metadata.get("available", False))


def empty_alignment_profiles() -> pd.DataFrame:
    """Return the schema-stable, empty representation of no usable source."""
    return pd.DataFrame(columns=PROFILE_COLUMNS)


def empty_scheme_profiles() -> pd.DataFrame:
    """Return the schema-stable, empty representation of no usable man/zone source."""
    return pd.DataFrame(columns=SCHEME_PROFILE_COLUMNS)


def empty_alignment_team_games() -> pd.DataFrame:
    """Return an empty but schema-stable offense-versus-defense evidence table."""
    return pd.DataFrame(columns=ALIGNMENT_TEAM_GAME_COLUMNS)


def empty_alignment_defense_profiles() -> pd.DataFrame:
    """Return an empty but schema-stable alignment-defense profile table."""
    return pd.DataFrame(columns=ALIGNMENT_DEFENSE_PROFILE_COLUMNS)


def neutral_alignment_profile(reason: str = "No time-valid PFF alignment source") -> dict[str, Any]:
    """A safe no-op profile for an unmatched player or an absent archive.

    A missing profile must never be interpreted as a 0% slot player or a
    punitive matchup.  The future matchup layer can multiply by
    ``alignment_matchup_multiplier`` without special casing this object.
    """
    return {
        "player_id": "",
        "player": "",
        "position": "",
        "team": "",
        "identity_key": "",
        "identity_quality": "missing",
        "slot_alignment_rate": None,
        "non_slot_alignment_rate": None,
        "wide_alignment_rate": None,
        "inline_alignment_rate": None,
        "alignment_sample_weight": 0.0,
        "alignment_confidence": 0.0,
        "slot_snaps": 0.0,
        "wide_snaps": 0.0,
        "inline_snaps": 0.0,
        "routes": 0.0,
        "slot_routes": None,
        "slot_targets": None,
        "slot_receptions": None,
        "slot_yards": None,
        "slot_touchdowns": None,
        "non_slot_targets": None,
        "non_slot_receptions": None,
        "non_slot_yards": None,
        "non_slot_touchdowns": None,
        "slot_catch_rate": None,
        "slot_yards_per_target": None,
        "non_slot_catch_rate": None,
        "non_slot_yards_per_target": None,
        "alignment_available": False,
        "alignment_experimental": False,
        "alignment_effect_weight": 0.0,
        "alignment_matchup_multiplier": 1.0,
        "alignment_semantics": "missing / neutral",
        "source_kind": "missing",
        "source_year": None,
        "source_weeks": "",
        "source_week_count": 0,
        "source_regular_season": None,
        "source_time_valid": False,
        "source_confidence": "none",
        "source_paths": "",
        "source_notes": reason,
        "profile_version": PFF_ALIGNMENT_VERSION,
    }


def neutral_scheme_profile(reason: str = "No time-valid PFF scheme source") -> dict[str, Any]:
    """A safe no-op man/zone tendency profile, the SCHEME_PROFILE_COLUMNS
    twin of neutral_alignment_profile(). A missing profile is never a claim
    that a player is 0% man-coverage; nothing multiplies by a scheme rate
    from this yet (SCHEME_PROFILE_COLUMNS carries no scoring multiplier at
    all - man/zone is audit-only, unlike alignment_matchup_multiplier's
    always-1.0 placeholder)."""
    return {
        "player_id": "",
        "player": "",
        "position": "",
        "team": "",
        "identity_key": "",
        "identity_quality": "missing",
        "man_route_share": None,
        "zone_route_share": None,
        "scheme_sample_weight": 0.0,
        "scheme_confidence": 0.0,
        "man_routes": None,
        "man_targets": None,
        "man_receptions": None,
        "man_yards": None,
        "man_touchdowns": None,
        "man_catch_rate": None,
        "man_yards_per_target": None,
        "man_yprr": None,
        "zone_routes": None,
        "zone_targets": None,
        "zone_receptions": None,
        "zone_yards": None,
        "zone_touchdowns": None,
        "zone_catch_rate": None,
        "zone_yards_per_target": None,
        "zone_yprr": None,
        "scheme_available": False,
        "scheme_semantics": "missing / neutral",
        "source_kind": "missing",
        "source_year": None,
        "source_weeks": "",
        "source_week_count": 0,
        "source_regular_season": None,
        "source_time_valid": False,
        "source_confidence": "none",
        "source_paths": "",
        "source_notes": reason,
        "profile_version": PFF_SCHEME_VERSION,
    }


def neutral_alignment_defense_profile(
    reason: str = "No time-valid, schedule-mapped PFF alignment-defense evidence",
    *,
    defense_team: Any = None,
    position: Any = None,
    alignment: Any = None,
    stat: Any = None,
) -> dict[str, Any]:
    """Return a no-op defense-profile record for a missing or ambiguous match.

    Missing evidence is never a claim that a defense is neutral in reality;
    it simply means this optional alignment residual has no valid input and
    must leave the broad matchup model untouched.
    """
    return {
        "defense_team": _canonical_team_key(defense_team),
        "position": _normalise_position(position),
        "alignment": _normalise_alignment(alignment),
        "stat": _normalise_alignment_stat(stat),
        "profile_available": False,
        "candidate_available": False,
        "scoring_active": False,
        "observed_total": None,
        "expected_total": None,
        "raw_allowed_ratio": None,
        "clipped_allowed_ratio": None,
        "shrunk_allowed_ratio": 1.0,
        "sample_games": 0,
        "comparison_games": 0,
        "uncompared_games": 0,
        "sample_offenses": 0,
        "event_sample_weight": 0.0,
        "shrinkage_games": ALIGNMENT_DEFENSE_SHRINKAGE_GAMES,
        "shrinkage_weight": 0.0,
        "alignment_confidence": 0.0,
        "position_normal_slot_rate": None,
        "normal_alignment_source": "unavailable",
        "baseline_source": "neutral_fallback",
        "fallback_reason": reason,
        "source_kind": "missing",
        "source_method": "offensive_weekly_reports_mapped_via_schedule",
        "source_year": None,
        "source_weeks": "",
        "source_week_count": 0,
        "source_regular_season": None,
        "source_time_valid": False,
        "source_confidence": "none",
        "source_paths": "",
        "source_notes": reason,
        "profile_version": PFF_ALIGNMENT_DEFENSE_VERSION,
    }


def _normalise_column_names(frame: pd.DataFrame, aliases: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Rename known PFF/browser-export aliases into one canonical schema."""
    out = frame.copy()
    # PFF headers are normally clean, but browser exports can preserve BOMs or
    # stray spaces.  Do this before aliases while avoiding a broad lower-case
    # transformation that could silently merge genuinely distinct columns.
    out.columns = [str(column).replace("\ufeff", "").strip() for column in out.columns]
    lower_to_actual = {str(column).lower(): str(column) for column in out.columns}
    rename: dict[str, str] = {}
    existing = set(out.columns)
    for canonical, candidates in aliases.items():
        if canonical in existing:
            continue
        for candidate in candidates:
            actual = lower_to_actual.get(candidate.lower())
            if actual is not None:
                rename[actual] = canonical
                existing.add(canonical)
                break
    if rename:
        out = out.rename(columns=rename)
    return out


def _normalise_metadata_mapping(metadata: Mapping[str, Any], aliases: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    """Apply the same aliases to explicit metadata that CSV manifests use."""
    out = dict(metadata)
    lower_to_actual = {str(key).strip().lower(): key for key in out}
    for canonical, candidates in aliases.items():
        if canonical in out:
            continue
        for candidate in candidates:
            actual = lower_to_actual.get(candidate.lower())
            if actual is not None:
                out[canonical] = out[actual]
                break
    return out


def _coerce_bool(value: Any) -> bool | None:
    """Interpret common manifest values without guessing on unknown strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "t", "yes", "y", "reg", "regular", "regular season", "regular-season"}:
        return True
    if text in {"0", "0.0", "false", "f", "no", "n", "post", "postseason", "playoff", "playoffs", "pre"}:
        return False
    return None


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _clean_id(value: Any) -> str:
    text = _clean_text(value)
    # A nullable integer column can become "100632.0" after a CSV round trip.
    return re.sub(r"\.0$", "", text)


def _name_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", _clean_text(value).lower())


def _team_key(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", _clean_text(value).upper())


# The PFF receiving exports usually use the nflverse spellings already, but
# the handful of historical PFF codes below do occur elsewhere in this repo.
# Normalize only for schedule joins; raw player-role output remains exactly as
# supplied by the reviewed local export.
_TEAM_CODE_ALIASES = {
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "JAC": "JAX",
    "LAR": "LA",
    "WSH": "WAS",
}


def _canonical_team_key(value: Any) -> str:
    return _TEAM_CODE_ALIASES.get(_team_key(value), _team_key(value))


def _normalise_alignment(value: Any) -> str:
    raw = _clean_text(value).lower().replace("-", "_").replace(" ", "_")
    if raw in {"slot", "inside"}:
        return "slot"
    if raw in {"non_slot", "nonslot", "wide", "outside", "inline", "other"}:
        # The archive can only truthfully promise the complement of the real
        # PFF slot events.  Do not re-label it as literal wide or inline.
        return "non_slot"
    return raw


def _normalise_alignment_stat(value: Any) -> str:
    raw = _clean_text(value).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "target": "targets",
        "receiving_targets": "targets",
        "reception": "receptions",
        "receiving_receptions": "receptions",
        "receiving_yards": "yards",
        "yard": "yards",
        "receiving_touchdowns": "touchdowns",
        "receiving_tds": "touchdowns",
        "td": "touchdowns",
        "tds": "touchdowns",
        "touchdown": "touchdowns",
    }
    return aliases.get(raw, raw)


def _normalise_position(value: Any) -> str:
    raw = _clean_text(value).upper()
    return _PFF_POSITION_MAP.get(raw, raw)


def _normalise_rate(series: pd.Series) -> pd.Series:
    """Convert PFF's usual percentage values into bounded 0..1 rates."""
    values = pd.to_numeric(series, errors="coerce")
    # PFF provides 23.9 for 23.9%, while a hand-created reviewed export may
    # reasonably use .239.  Supporting both is deterministic per value.
    values = values.where(values.abs().le(1.0), values / 100.0)
    return values.clip(lower=0.0, upper=1.0)


def _numeric_column(frame: pd.DataFrame, column: str, default: float | None = None) -> pd.Series:
    if column not in frame.columns:
        if default is None:
            return pd.Series(float("nan"), index=frame.index, dtype=float)
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _stable_identity_key(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Use PFF IDs first; fall back only to a conservative name/team/position key."""
    ids = frame.get("player_id", pd.Series("", index=frame.index)).map(_clean_id)
    players = frame.get("player", pd.Series("", index=frame.index)).map(_name_key)
    teams = frame.get("team_name", pd.Series("", index=frame.index)).map(_team_key)
    positions = frame.get("position", pd.Series("", index=frame.index)).map(_normalise_position)
    has_id = ids.ne("") & ~ids.str.lower().isin({"nan", "none", "<na>"})
    fallback = "name:" + players + "|team:" + teams + "|pos:" + positions
    keys = pd.Series("pff:" + ids, index=frame.index).where(has_id, fallback)
    quality = pd.Series("pff_player_id", index=frame.index).where(has_id, "name_team_position_fallback")
    return keys.astype(str), quality.astype(str)


def _parse_week_directory_name(name: str) -> int | None:
    """Accept the documented ``1`` format and benign ``Week 1`` variants."""
    match = re.fullmatch(r"(?:week[ _-]*)?(\d{1,2})", str(name).strip().lower())
    if not match:
        return None
    value = int(match.group(1))
    return value if value > 0 else None


def _validate_week_number(as_of_week: int | None) -> int | None:
    if as_of_week is None:
        return None
    try:
        value = int(as_of_week)
    except (TypeError, ValueError) as exc:
        raise ValueError("as_of_week must be a positive integer") from exc
    if value <= 0:
        raise ValueError("as_of_week must be a positive integer")
    return value


def _load_manifest(weekly_dir: Path, issues: list[str]) -> tuple[dict[int, dict[str, Any]], bool]:
    """Read optional weekly metadata, retaining unknown fields only as metadata."""
    path = weekly_dir / "manifest.csv"
    if not path.exists():
        return {}, False
    try:
        manifest = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # local malformed export should not crash rankings
        issues.append(f"Could not read weekly PFF manifest {path}: {type(exc).__name__}: {exc}")
        return {}, True
    manifest = _normalise_column_names(manifest, _MANIFEST_ALIASES)
    if "week" not in manifest.columns:
        issues.append(f"Weekly PFF manifest {path} has no week column; treating report status as unverified.")
        return {}, True

    by_week: dict[int, dict[str, Any]] = {}
    for _, row in manifest.iterrows():
        try:
            week = int(pd.to_numeric(row.get("week"), errors="raise"))
        except Exception:
            issues.append(f"Weekly PFF manifest {path} has an invalid week value; row ignored.")
            continue
        if week <= 0:
            issues.append(f"Weekly PFF manifest {path} has non-positive Week {week}; row ignored.")
            continue
        if week in by_week:
            issues.append(f"Weekly PFF manifest {path} has duplicate Week {week}; using the last row.")
        regular_value = row.get("regular_season")
        schema_value = row.get("schema_valid")
        by_week[week] = {
            "regular_season": _coerce_bool(regular_value),
            "schema_valid": _coerce_bool(schema_value),
            "source_confidence": _clean_text(row.get("source_confidence")),
            "export_date": _clean_text(row.get("export_date")),
            "time_valid": _coerce_bool(row.get("time_valid")),
        }
    return by_week, True


def discover_weekly_alignment_exports(
    year: int,
    as_of_week: int | None = None,
    pff_root: str | Path = DEFAULT_PFF_ROOT,
) -> AlignmentLoadResult:
    """Discover local weekly PFF export pairs and mark their as-of eligibility.

    Discovery is read-only and returns all observed week folders.  The
    ``eligible_as_of`` flag is true only for complete, non-rejected pairs
    strictly before ``as_of_week``.  Unknown regular-season status remains
    visibly *unverified* rather than being mislabeled; it is allowed for a
    manually archived weekly report because the time cutoff is still exact.
    A known POST/preseason source is always rejected.
    """
    target_week = _validate_week_number(as_of_week)
    root = Path(pff_root)
    weekly_dir = root / str(int(year)) / "weekly"
    issues: list[str] = []
    rows: list[dict[str, Any]] = []
    if not weekly_dir.is_dir():
        metadata = {
            "available": False,
            "source_kind": "weekly_archive",
            "source_year": int(year),
            "as_of_week": target_week,
            "reason": f"No weekly PFF archive directory at {weekly_dir}",
        }
        return AlignmentLoadResult(empty_alignment_profiles(), pd.DataFrame(columns=ARCHIVE_COLUMNS), issues, metadata)

    manifest, manifest_present = _load_manifest(weekly_dir, issues)
    week_dirs: list[tuple[int, Path]] = []
    for child in weekly_dir.iterdir():
        if not child.is_dir():
            continue
        week = _parse_week_directory_name(child.name)
        if week is None:
            issues.append(f"Ignored PFF weekly directory {child.name!r}; expected a positive week number.")
            continue
        week_dirs.append((week, child))

    for week, week_dir in sorted(week_dirs, key=lambda item: item[0]):
        summary = week_dir / "receiving_summary.csv"
        concept = week_dir / "receiving_concept.csv"
        scheme = week_dir / "receiving_scheme.csv"
        meta = manifest.get(week, {})
        regular = meta.get("regular_season")
        regular_status = "confirmed" if regular is True else ("rejected" if regular is False else "unverified")
        schema_status = meta.get("schema_valid")
        complete = summary.is_file() and concept.is_file()
        eligible = bool(
            complete
            and target_week is not None
            and week < target_week
            and regular is not False
            and schema_status is not False
        )
        row = {
            "year": int(year),
            "week": week,
            "summary_path": str(summary) if summary.is_file() else "",
            "concept_path": str(concept) if concept.is_file() else "",
            "scheme_path": str(scheme) if scheme.is_file() else "",
            "scheme_present": scheme.is_file(),
            "pair_complete": complete,
            "manifest_present": manifest_present,
            "regular_season": regular,
            "regular_season_status": regular_status,
            "manifest_schema_valid": schema_status,
            "manifest_source_confidence": meta.get("source_confidence", ""),
            "manifest_export_date": meta.get("export_date", ""),
            "eligible_as_of": eligible,
            "schema_valid": None,
            "schema_issues": "",
        }
        rows.append(row)
        if not complete:
            missing = []
            if not summary.is_file():
                missing.append("receiving_summary.csv")
            if not concept.is_file():
                missing.append("receiving_concept.csv")
            issues.append(f"PFF Week {week} archive is incomplete (missing {', '.join(missing)}); ignored.")
        if target_week is None:
            issues.append(f"PFF Week {week} is not loaded because no as_of_week was supplied.")
        elif week >= target_week:
            issues.append(
                f"PFF Week {week} is not eligible for a Week {target_week} projection "
                "(the archive rule is strictly week < as_of_week)."
            )
        if regular is False:
            issues.append(f"PFF Week {week} is marked non-regular-season in the manifest and is ignored.")
        if schema_status is False:
            issues.append(f"PFF Week {week} is marked schema-invalid in the manifest and is ignored.")

    archives = pd.DataFrame(rows, columns=ARCHIVE_COLUMNS)
    metadata = {
        "available": bool(not archives.empty and archives["eligible_as_of"].any()),
        "source_kind": "weekly_archive",
        "source_year": int(year),
        "as_of_week": target_week,
        "manifest_path": str(weekly_dir / "manifest.csv") if manifest_present else "",
        "manifest_present": manifest_present,
        "discovered_weeks": tuple(archives["week"].tolist()) if not archives.empty else (),
    }
    return AlignmentLoadResult(
        profiles=pd.DataFrame(columns=PROFILE_COLUMNS),
        archives=archives,
        issues=_unique_issues(issues),
        metadata=metadata,
    )


def _safe_read_csv(path: str | Path, label: str, issues: list[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        issues.append(f"Could not read {label} at {path}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def _validate_summary(frame: pd.DataFrame, label: str, issues: list[str]) -> tuple[pd.DataFrame, bool]:
    out = _normalise_column_names(frame, _SUMMARY_ALIASES)
    if "slot_rate" not in out.columns:
        # A manually transformed export can still be valid if it gives all
        # three alignment snap counts.  Derive a rate rather than invent one.
        snap_cols = {"slot_snaps", "wide_snaps", "inline_snaps"}
        if snap_cols.issubset(out.columns):
            total = sum((_numeric_column(out, column, 0.0) for column in snap_cols))
            out["slot_rate"] = _numeric_column(out, "slot_snaps", 0.0) / total.where(total.gt(0.0))
        else:
            issues.append(f"{label} is missing slot_rate (and cannot derive it from alignment snaps).")
    required = ("player", "position", "team_name", "slot_rate")
    missing = [column for column in required if column not in out.columns]
    if missing:
        issues.append(f"{label} failed summary schema validation; missing {', '.join(missing)}.")
        return pd.DataFrame(), False
    if "player_id" not in out.columns:
        issues.append(f"{label} has no player_id; using conservative name/team/position joins.")
        out["player_id"] = ""
    return out, True


def _validate_concept(frame: pd.DataFrame, label: str, issues: list[str]) -> tuple[pd.DataFrame, bool]:
    out = _normalise_column_names(frame, _CONCEPT_ALIASES)
    required = ("player", "position", "team_name", "slot_routes")
    missing = [column for column in required if column not in out.columns]
    if missing:
        issues.append(f"{label} failed concept schema validation; missing {', '.join(missing)}.")
        return pd.DataFrame(), False
    if "player_id" not in out.columns:
        issues.append(f"{label} has no player_id; using conservative name/team/position joins.")
        out["player_id"] = ""
    for column in _SLOT_EVENT_COLUMNS:
        if column not in out.columns:
            # slot_routes is required.  Other event fields are useful for a
            # later defense profile but do not make a player-alignment prior
            # unusable; retain explicit missing values rather than zeros.
            out[column] = pd.NA
            issues.append(f"{label} has no {column}; this event field remains unavailable.")
    return out, True


def _validate_scheme(frame: pd.DataFrame, label: str, issues: list[str]) -> tuple[pd.DataFrame, bool]:
    out = _normalise_column_names(frame, _SCHEME_ALIASES)
    required = ("player", "position", "team_name", "man_targets", "zone_targets")
    missing = [column for column in required if column not in out.columns]
    if missing:
        issues.append(f"{label} failed scheme schema validation; missing {', '.join(missing)}.")
        return pd.DataFrame(), False
    if "player_id" not in out.columns:
        issues.append(f"{label} has no player_id; using conservative name/team/position joins.")
        out["player_id"] = ""
    for column in _SCHEME_EVENT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
            issues.append(f"{label} has no {column}; this event field remains unavailable.")
    return out, True


def save_weekly_alignment_export(
    summary_file: Any,
    concept_file: Any,
    year: int,
    week: int,
    *,
    pff_root: str | Path = DEFAULT_PFF_ROOT,
) -> tuple[bool, list[str]]:
    """Write one week's uploaded receiving_summary/receiving_concept pair to
    the local archive layout this module reads (pff_imports/{year}/weekly/
    {week}/*.csv), and upsert that week's manifest.csv row.

    Runs the SAME _validate_summary/_validate_concept schema check used at
    load time before writing anything, so a bad upload is rejected with a
    clear reason instead of silently landing as a file load_weekly_alignment_
    profiles will later reject anyway - and the manifest's schema_valid flag
    this writes is therefore true validation, not an optimistic guess.

    Marks regular_season=True: this is an in-season upload of a week that was
    just played, not a season-total export, so the ambiguity
    load_season_alignment_prior guards against (playoff-contaminated totals)
    does not apply here.
    """
    issues: list[str] = []
    if summary_file is None or concept_file is None:
        return False, ["Both the receiving_summary and receiving_concept files are required."]
    try:
        summary_df = pd.read_csv(summary_file, low_memory=False)
    except Exception as exc:
        return False, [f"Could not read the receiving_summary file: {type(exc).__name__}: {exc}"]
    try:
        concept_df = pd.read_csv(concept_file, low_memory=False)
    except Exception as exc:
        return False, [f"Could not read the receiving_concept file: {type(exc).__name__}: {exc}"]

    _summary_validated, summary_ok = _validate_summary(summary_df, "receiving_summary.csv", issues)
    _concept_validated, concept_ok = _validate_concept(concept_df, "receiving_concept.csv", issues)
    schema_valid = bool(summary_ok and concept_ok)
    if not schema_valid:
        return False, issues

    week_dir = Path(pff_root) / str(int(year)) / "weekly" / str(int(week))
    week_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(week_dir / "receiving_summary.csv", index=False)
    concept_df.to_csv(week_dir / "receiving_concept.csv", index=False)

    weekly_dir = Path(pff_root) / str(int(year)) / "weekly"
    manifest_path = weekly_dir / "manifest.csv"
    manifest = pd.DataFrame(columns=["week", "regular_season", "schema_valid", "source_confidence", "export_date"])
    if manifest_path.is_file():
        try:
            manifest = pd.read_csv(manifest_path, low_memory=False)
        except Exception:
            manifest = manifest.iloc[0:0]
    for column in ("week", "regular_season", "schema_valid", "source_confidence", "export_date"):
        if column not in manifest.columns:
            manifest[column] = pd.NA
    manifest["week"] = pd.to_numeric(manifest["week"], errors="coerce")
    manifest = manifest[manifest["week"].ne(int(week))]
    new_row = pd.DataFrame([{
        "week": int(week), "regular_season": True, "schema_valid": True,
        "source_confidence": "in_app_upload_regular_season_export",
        "export_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
    }])
    pd.concat([manifest, new_row], ignore_index=True).to_csv(manifest_path, index=False)
    return True, issues


def _prepare_summary_rows(
    summary: pd.DataFrame,
    *,
    week: int,
    path: str,
    regular_season: bool | None,
    source_confidence: str,
) -> pd.DataFrame:
    out = summary.copy()
    out["player"] = out["player"].map(_clean_text)
    out["player_id"] = out["player_id"].map(_clean_id)
    out["team_name"] = out["team_name"].map(_team_key)
    out["position"] = out["position"].map(_normalise_position)
    out = out[out["player"].ne("") & out["position"].isin(_SUPPORTED_RECEIVING_POSITIONS)].copy()
    keys, quality = _stable_identity_key(out)
    out["identity_key"] = keys
    out["identity_quality"] = quality
    out["_slot_rate"] = _normalise_rate(out["slot_rate"])
    out["_wide_rate"] = _normalise_rate(_numeric_column(out, "wide_rate"))
    out["_inline_rate"] = _normalise_rate(_numeric_column(out, "inline_rate"))
    for column in ("slot_snaps", "wide_snaps", "inline_snaps", "routes", "player_game_count"):
        out[column] = _numeric_column(out, column, 0.0).fillna(0.0).clip(lower=0.0)
    align_snaps = out[["slot_snaps", "wide_snaps", "inline_snaps"]].sum(axis=1)
    # Alignment snaps are the best direct sample size.  Routes and game count
    # are transparent fallbacks for valid PFF shapes that omit snap counts.
    out["_alignment_weight"] = align_snaps.where(align_snaps.gt(0.0), out["routes"])
    out["_alignment_weight"] = out["_alignment_weight"].where(
        out["_alignment_weight"].gt(0.0), out["player_game_count"]
    ).where(lambda values: values.gt(0.0), 1.0)
    out["source_week"] = int(week)
    out["source_path"] = str(path)
    out["source_regular_season"] = regular_season
    out["source_confidence"] = source_confidence
    return out


def _prepare_concept_rows(concept: pd.DataFrame, *, week: int) -> pd.DataFrame:
    out = concept.copy()
    out["player"] = out["player"].map(_clean_text)
    out["player_id"] = out["player_id"].map(_clean_id)
    out["team_name"] = out["team_name"].map(_team_key)
    out["position"] = out["position"].map(_normalise_position)
    out = out[out["player"].ne("") & out["position"].isin(_SUPPORTED_RECEIVING_POSITIONS)].copy()
    keys, _quality = _stable_identity_key(out)
    out["identity_key"] = keys
    out["source_week"] = int(week)
    for column in _SLOT_EVENT_COLUMNS:
        out[column] = _numeric_column(out, column)
    # A player can have duplicate export rows only in unusual source shapes;
    # sum them once before joining so a duplicate cannot inflate a role.
    return out.groupby(["identity_key", "source_week"], as_index=False, observed=True)[list(_SLOT_EVENT_COLUMNS)].sum(min_count=1)


def _prepare_scheme_rows(
    scheme: pd.DataFrame,
    *,
    week: int,
    path: str,
    regular_season: bool | None,
    source_confidence: str,
) -> pd.DataFrame:
    out = scheme.copy()
    out["player"] = out["player"].map(_clean_text)
    out["player_id"] = out["player_id"].map(_clean_id)
    out["team_name"] = out["team_name"].map(_team_key)
    out["position"] = out["position"].map(_normalise_position)
    out = out[out["player"].ne("") & out["position"].isin(_SUPPORTED_RECEIVING_POSITIONS)].copy()
    keys, quality = _stable_identity_key(out)
    out["identity_key"] = keys
    out["identity_quality"] = quality
    for column in _SCHEME_EVENT_COLUMNS:
        out[column] = _numeric_column(out, column)
    # NOTE: PFF's own man_route_rate/zone_route_rate columns are NOT a
    # slot_rate-style share of this player's routes - each is routes run
    # DIVIDED BY that coverage type's own pass_plays (route participation
    # conditional on the defense's call), so the two do not sum to 1.0 and
    # cannot answer "what share of this player's work came against man vs
    # zone."  That share is derived directly from route counts here instead,
    # the same way the slot alignment profile above derives its rate from
    # raw snap counts rather than trusting a differently-defined PFF percent
    # column.
    routes = out[["man_routes", "zone_routes"]].sum(axis=1, min_count=1)
    out["_man_route_share"] = (out["man_routes"] / routes).where(routes.gt(0.0))
    out["_zone_route_share"] = (out["zone_routes"] / routes).where(routes.gt(0.0))
    out["_scheme_weight"] = routes.where(routes.gt(0.0), _numeric_column(out, "player_game_count", 0.0).fillna(0.0))
    out["source_week"] = int(week)
    out["source_path"] = str(path)
    out["source_regular_season"] = regular_season
    out["source_confidence"] = source_confidence
    return out


def _merge_week_pair(summary_rows: pd.DataFrame, concept_rows: pd.DataFrame) -> pd.DataFrame:
    if summary_rows.empty:
        return summary_rows
    if concept_rows.empty:
        for column in _SLOT_EVENT_COLUMNS:
            summary_rows[column] = pd.NA
        return summary_rows
    return summary_rows.merge(
        concept_rows,
        how="left",
        on=["identity_key", "source_week"],
        validate="many_to_one",
    )


def _weighted_average(values: pd.Series, weights: pd.Series) -> float | None:
    valid = values.notna() & weights.notna() & weights.gt(0.0)
    if not valid.any():
        return None
    denom = float(weights[valid].sum())
    return float((values[valid] * weights[valid]).sum() / denom) if denom > 0.0 else None


_MIN_ALIGNMENT_TARGETS_FOR_EFFICIENCY = 5.0


def _alignment_efficiency(targets: float | None, receptions: float | None,
                          yards: float | None) -> tuple[float | None, float | None]:
    """(catch_rate, yards_per_target) for one alignment split, or (None, None)
    below a minimum target count. Deliberately NOT a rate/snap-share style
    weighted blend like the rest of this module - a single-digit weekly
    target count makes a per-target yardage ratio noise, not signal, so this
    withholds a number entirely rather than shrinking a bad one toward a
    league mean the way the rate fields do. Touchdown rate is deliberately
    excluded here: this app already has a dedicated, evidence-weighted TD
    estimator (see v2_td_two_year_prior in data.weekly_projections), and a
    raw slot-split TD rate off a handful of weekly targets would be far
    noisier than that without adding real information."""
    if targets is None or targets < _MIN_ALIGNMENT_TARGETS_FOR_EFFICIENCY:
        return None, None
    catch_rate = (receptions / targets) if receptions is not None else None
    yards_per_target = (yards / targets) if yards is not None else None
    return catch_rate, yards_per_target


def _alignment_semantics(position: str) -> tuple[str, bool]:
    if position == "WR":
        return "slot / non-slot (non-slot is a wide-or-other proxy)", False
    if position == "TE":
        return "slot / non-slot (non-slot is not asserted inline)", False
    if position == "RB":
        return "RB receiving alignment; experimental", True
    return "FB receiving alignment; experimental", True


def _aggregate_profiles(
    rows: pd.DataFrame,
    *,
    source_kind: str,
    source_year: int,
    source_time_valid: bool,
    source_notes: str,
) -> pd.DataFrame:
    if rows.empty:
        return empty_alignment_profiles()
    records: list[dict[str, Any]] = []
    sorted_rows = rows.sort_values(["identity_key", "source_week", "source_path"], kind="stable")
    for identity_key, group in sorted_rows.groupby("identity_key", sort=False, observed=True):
        group = group.copy()
        last = group.iloc[-1]
        weights = pd.to_numeric(group["_alignment_weight"], errors="coerce").fillna(0.0)
        weight = float(weights.sum())
        slot_rate = _weighted_average(group["_slot_rate"], weights)
        wide_rate = _weighted_average(group["_wide_rate"], weights)
        inline_rate = _weighted_average(group["_inline_rate"], weights)
        position = _normalise_position(last["position"])
        semantics, experimental = _alignment_semantics(position)
        event_totals = {
            column: float(pd.to_numeric(group[column], errors="coerce").sum(min_count=1))
            if pd.to_numeric(group[column], errors="coerce").notna().any()
            else None
            for column in _SLOT_EVENT_COLUMNS
        }
        # Non-slot = the summary report's OWN total minus the concept report's
        # slot-only total, both already time-safe (same as-of week filter, same
        # group) - not a second data source, just the other half of the split
        # the summary total already implies. None (not 0) whenever either side
        # is missing, so a real absence of data never reads as "zero targets".
        non_slot_totals: dict[str, float | None] = {}
        for total_col, slot_col, out_col in (
            ("targets", "slot_targets", "non_slot_targets"),
            ("receptions", "slot_receptions", "non_slot_receptions"),
            ("yards", "slot_yards", "non_slot_yards"),
            ("touchdowns", "slot_touchdowns", "non_slot_touchdowns"),
        ):
            slot_value = event_totals.get(slot_col)
            total_value = (
                float(pd.to_numeric(group[total_col], errors="coerce").sum(min_count=1))
                if total_col in group.columns and pd.to_numeric(group[total_col], errors="coerce").notna().any()
                else None
            )
            non_slot_totals[out_col] = (
                max(0.0, total_value - slot_value) if total_value is not None and slot_value is not None else None
            )
        slot_catch_rate, slot_ypt = _alignment_efficiency(
            event_totals.get("slot_targets"), event_totals.get("slot_receptions"), event_totals.get("slot_yards"))
        non_slot_catch_rate, non_slot_ypt = _alignment_efficiency(
            non_slot_totals["non_slot_targets"], non_slot_totals["non_slot_receptions"], non_slot_totals["non_slot_yards"])
        source_weeks = tuple(sorted({int(value) for value in group["source_week"].tolist() if int(value) > 0}))
        regular_values = [value for value in group["source_regular_season"].tolist() if value is not None and not pd.isna(value)]
        regular = True if regular_values and all(bool(value) for value in regular_values) else None
        confidence_values = [str(value) for value in group["source_confidence"].tolist() if _clean_text(value)]
        record = {
            "player_id": _clean_id(last.get("player_id")),
            "player": _clean_text(last.get("player")),
            "position": position,
            "team": _team_key(last.get("team_name")),
            "identity_key": str(identity_key),
            "identity_quality": _clean_text(last.get("identity_quality")) or "name_team_position_fallback",
            "slot_alignment_rate": slot_rate,
            "non_slot_alignment_rate": (1.0 - slot_rate) if slot_rate is not None else None,
            "wide_alignment_rate": wide_rate,
            "inline_alignment_rate": inline_rate,
            "alignment_sample_weight": weight,
            "alignment_confidence": weight / (weight + ALIGNMENT_CONFIDENCE_HALF_WEIGHT) if weight > 0 else 0.0,
            "slot_snaps": float(group["slot_snaps"].sum()),
            "wide_snaps": float(group["wide_snaps"].sum()),
            "inline_snaps": float(group["inline_snaps"].sum()),
            "routes": float(group["routes"].sum()),
            **event_totals,
            **non_slot_totals,
            "slot_catch_rate": slot_catch_rate,
            "slot_yards_per_target": slot_ypt,
            "non_slot_catch_rate": non_slot_catch_rate,
            "non_slot_yards_per_target": non_slot_ypt,
            "alignment_available": slot_rate is not None,
            "alignment_experimental": experimental,
            # This module is intentionally a role-source foundation.  A
            # tested defense residual may opt in later; until then it is a
            # literal neutral multiplier rather than an implicit hidden one.
            "alignment_effect_weight": 0.0,
            "alignment_matchup_multiplier": 1.0,
            "alignment_semantics": semantics,
            "source_kind": source_kind,
            "source_year": int(source_year),
            "source_weeks": ",".join(str(value) for value in source_weeks),
            "source_week_count": len(source_weeks),
            "source_regular_season": regular,
            "source_time_valid": bool(source_time_valid),
            "source_confidence": "; ".join(dict.fromkeys(confidence_values)) or "schema_validated_local_export",
            "source_paths": "; ".join(dict.fromkeys(str(value) for value in group["source_path"].tolist())),
            "source_notes": source_notes,
            "profile_version": PFF_ALIGNMENT_VERSION,
        }
        records.append(record)
    return pd.DataFrame(records, columns=PROFILE_COLUMNS)


def _aggregate_scheme_profiles(
    rows: pd.DataFrame,
    *,
    source_kind: str,
    source_year: int,
    source_time_valid: bool,
    source_notes: str,
) -> pd.DataFrame:
    if rows.empty:
        return empty_scheme_profiles()
    records: list[dict[str, Any]] = []
    sorted_rows = rows.sort_values(["identity_key", "source_week", "source_path"], kind="stable")
    for identity_key, group in sorted_rows.groupby("identity_key", sort=False, observed=True):
        group = group.copy()
        last = group.iloc[-1]
        weights = pd.to_numeric(group["_scheme_weight"], errors="coerce").fillna(0.0)
        weight = float(weights.sum())
        man_share = _weighted_average(group["_man_route_share"], weights)
        zone_share = _weighted_average(group["_zone_route_share"], weights)
        position = _normalise_position(last["position"])
        event_totals = {
            column: float(pd.to_numeric(group[column], errors="coerce").sum(min_count=1))
            if pd.to_numeric(group[column], errors="coerce").notna().any()
            else None
            for column in _SCHEME_EVENT_COLUMNS
        }
        man_catch_rate, man_ypt = _alignment_efficiency(
            event_totals.get("man_targets"), event_totals.get("man_receptions"), event_totals.get("man_yards"))
        zone_catch_rate, zone_ypt = _alignment_efficiency(
            event_totals.get("zone_targets"), event_totals.get("zone_receptions"), event_totals.get("zone_yards"))
        man_yprr = _weighted_average(pd.to_numeric(group["man_yprr"], errors="coerce"), weights)
        zone_yprr = _weighted_average(pd.to_numeric(group["zone_yprr"], errors="coerce"), weights)
        source_weeks = tuple(sorted({int(value) for value in group["source_week"].tolist() if int(value) > 0}))
        regular_values = [value for value in group["source_regular_season"].tolist() if value is not None and not pd.isna(value)]
        regular = True if regular_values and all(bool(value) for value in regular_values) else None
        confidence_values = [str(value) for value in group["source_confidence"].tolist() if _clean_text(value)]
        record = {
            "player_id": _clean_id(last.get("player_id")),
            "player": _clean_text(last.get("player")),
            "position": position,
            "team": _team_key(last.get("team_name")),
            "identity_key": str(identity_key),
            "identity_quality": _clean_text(last.get("identity_quality")) or "name_team_position_fallback",
            "man_route_share": man_share,
            "zone_route_share": zone_share if zone_share is not None else ((1.0 - man_share) if man_share is not None else None),
            "scheme_sample_weight": weight,
            "scheme_confidence": weight / (weight + ALIGNMENT_CONFIDENCE_HALF_WEIGHT) if weight > 0 else 0.0,
            "man_routes": event_totals.get("man_routes"),
            "man_targets": event_totals.get("man_targets"),
            "man_receptions": event_totals.get("man_receptions"),
            "man_yards": event_totals.get("man_yards"),
            "man_touchdowns": event_totals.get("man_touchdowns"),
            "man_catch_rate": man_catch_rate,
            "man_yards_per_target": man_ypt,
            "man_yprr": man_yprr,
            "zone_routes": event_totals.get("zone_routes"),
            "zone_targets": event_totals.get("zone_targets"),
            "zone_receptions": event_totals.get("zone_receptions"),
            "zone_yards": event_totals.get("zone_yards"),
            "zone_touchdowns": event_totals.get("zone_touchdowns"),
            "zone_catch_rate": zone_catch_rate,
            "zone_yards_per_target": zone_ypt,
            "zone_yprr": zone_yprr,
            "scheme_available": man_share is not None,
            "scheme_semantics": "man route-share / zone route-share (this player's own routes split by coverage faced)",
            "source_kind": source_kind,
            "source_year": int(source_year),
            "source_weeks": ",".join(str(value) for value in source_weeks),
            "source_week_count": len(source_weeks),
            "source_regular_season": regular,
            "source_time_valid": bool(source_time_valid),
            "source_confidence": "; ".join(dict.fromkeys(confidence_values)) or "schema_validated_local_export",
            "source_paths": "; ".join(dict.fromkeys(str(value) for value in group["source_path"].tolist())),
            "source_notes": source_notes,
            "profile_version": PFF_SCHEME_VERSION,
        }
        records.append(record)
    return pd.DataFrame(records, columns=SCHEME_PROFILE_COLUMNS)


def _unique_issues(issues: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(issue for issue in issues if issue))


def load_weekly_alignment_profiles(
    year: int,
    as_of_week: int,
    pff_root: str | Path = DEFAULT_PFF_ROOT,
) -> AlignmentLoadResult:
    """Load only time-valid weekly PFF player alignment profiles.

    ``as_of_week`` is required here (unlike discovery) so consumers cannot
    accidentally build a historical profile from a completed season.  The
    loader accepts an absent manifest but marks its regular-season status as
    unverified in both profile and result metadata; known non-regular reports
    are excluded.
    """
    target_week = _validate_week_number(as_of_week)
    discovered = discover_weekly_alignment_exports(year, target_week, pff_root)
    archives = discovered.archives.copy()
    issues = list(discovered.issues)
    records: list[pd.DataFrame] = []

    if archives.empty:
        metadata = {
            **discovered.metadata,
            "available": False,
            "included_weeks": (),
            "excluded_weeks": (),
            "alignment_adjustment_mode": "neutral_only_foundation",
        }
        return AlignmentLoadResult(empty_alignment_profiles(), archives, _unique_issues(issues), metadata)

    for index, archive in archives.loc[archives["eligible_as_of"].astype(bool)].iterrows():
        week = int(archive["week"])
        local_issues: list[str] = []
        summary = _safe_read_csv(archive["summary_path"], f"PFF Week {week} receiving_summary", local_issues)
        concept = _safe_read_csv(archive["concept_path"], f"PFF Week {week} receiving_concept", local_issues)
        summary, summary_valid = _validate_summary(summary, f"PFF Week {week} receiving_summary", local_issues)
        concept, concept_valid = _validate_concept(concept, f"PFF Week {week} receiving_concept", local_issues)
        valid = bool(summary_valid and concept_valid)
        archives.at[index, "schema_valid"] = valid
        archives.at[index, "schema_issues"] = " | ".join(local_issues)
        issues.extend(local_issues)
        if not valid:
            continue
        regular_season = _coerce_bool(archive.get("regular_season"))
        confidence = _clean_text(archive.get("manifest_source_confidence"))
        if not confidence:
            confidence = (
                "weekly_manifest_regular_season" if regular_season is True
                else "weekly_schema_valid_regular_season_unverified"
            )
        summary_rows = _prepare_summary_rows(
            summary,
            week=week,
            path=archive["summary_path"],
            regular_season=regular_season,
            source_confidence=confidence,
        )
        concept_rows = _prepare_concept_rows(concept, week=week)
        records.append(_merge_week_pair(summary_rows, concept_rows))

    combined = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    included_weeks = tuple(sorted({int(value) for value in combined.get("source_week", pd.Series(dtype=int)).tolist()}))
    profiles = _aggregate_profiles(
        combined,
        source_kind="weekly_archive",
        source_year=int(year),
        source_time_valid=True,
        source_notes=(
            f"Local manually exported weekly PFF archive through Week {max(included_weeks)} "
            f"for a Week {target_week} projection."
            if included_weeks else "No schema-valid, time-valid weekly PFF reports were available."
        ),
    )
    excluded_weeks = tuple(
        int(row.week) for row in archives.itertuples(index=False)
        if not bool(row.eligible_as_of) or row.schema_valid is False
    )
    metadata = {
        **discovered.metadata,
        "available": not profiles.empty,
        "included_weeks": included_weeks,
        "excluded_weeks": excluded_weeks,
        "schema_valid_weeks": tuple(
            int(row.week) for row in archives.itertuples(index=False) if row.schema_valid is True
        ),
        "alignment_adjustment_mode": "neutral_only_foundation",
    }
    return AlignmentLoadResult(profiles, archives, _unique_issues(issues), metadata)


def load_weekly_scheme_profiles(
    year: int,
    as_of_week: int,
    pff_root: str | Path = DEFAULT_PFF_ROOT,
) -> AlignmentLoadResult:
    """Load only time-valid weekly PFF player man/zone scheme profiles.

    Mirrors :func:`load_weekly_alignment_profiles`'s discovery/eligibility
    rules exactly - the same weekly/{week}/ archive, the same
    ``week < as_of_week`` cutoff, the same manifest handling - but reads
    ``receiving_scheme.csv`` instead of the receiving_summary/receiving_concept
    pair.  receiving_scheme.csv is self-contained, so there is no second file
    to merge here.  A week that has the summary/concept pair but not yet a
    receiving_scheme.csv is skipped for this profile only (not an error);
    :data:`ARCHIVE_COLUMNS`'s ``scheme_present`` flag makes that visible.
    """
    target_week = _validate_week_number(as_of_week)
    discovered = discover_weekly_alignment_exports(year, target_week, pff_root)
    archives = discovered.archives.copy()
    issues = list(discovered.issues)
    records: list[pd.DataFrame] = []

    if archives.empty:
        metadata = {
            **discovered.metadata,
            "available": False,
            "included_weeks": (),
            "excluded_weeks": (),
        }
        return AlignmentLoadResult(empty_scheme_profiles(), archives, _unique_issues(issues), metadata)

    for index, archive in archives.loc[archives["eligible_as_of"].astype(bool)].iterrows():
        week = int(archive["week"])
        scheme_path = archive.get("scheme_path", "")
        if not scheme_path:
            issues.append(f"PFF Week {week} has no receiving_scheme.csv; scheme profile skipped for that week.")
            continue
        local_issues: list[str] = []
        scheme = _safe_read_csv(scheme_path, f"PFF Week {week} receiving_scheme", local_issues)
        scheme, scheme_valid = _validate_scheme(scheme, f"PFF Week {week} receiving_scheme", local_issues)
        issues.extend(local_issues)
        if not scheme_valid:
            continue
        regular_season = _coerce_bool(archive.get("regular_season"))
        confidence = _clean_text(archive.get("manifest_source_confidence"))
        if not confidence:
            confidence = (
                "weekly_manifest_regular_season" if regular_season is True
                else "weekly_schema_valid_regular_season_unverified"
            )
        records.append(_prepare_scheme_rows(
            scheme, week=week, path=scheme_path,
            regular_season=regular_season, source_confidence=confidence,
        ))

    combined = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    included_weeks = tuple(sorted({int(value) for value in combined.get("source_week", pd.Series(dtype=int)).tolist()}))
    profiles = _aggregate_scheme_profiles(
        combined,
        source_kind="weekly_archive",
        source_year=int(year),
        source_time_valid=True,
        source_notes=(
            f"Local manually exported weekly PFF receiving_scheme archive through Week {max(included_weeks)} "
            f"for a Week {target_week} projection."
            if included_weeks else "No schema-valid, time-valid weekly PFF receiving_scheme reports were available."
        ),
    )
    metadata = {
        **discovered.metadata,
        "available": not profiles.empty,
        "included_weeks": included_weeks,
        "excluded_weeks": tuple(
            int(row.week) for row in archives.itertuples(index=False) if not bool(row.eligible_as_of)
        ),
    }
    return AlignmentLoadResult(profiles, archives, _unique_issues(issues), metadata)


# ---------------------------------------------------------------------------
# Weekly offense-derived alignment-defense foundation
# ---------------------------------------------------------------------------

_SCHEDULE_ALIASES = {
    "week": ("week", "week_number", "week_num"),
    "home_team": ("home_team", "home", "home_abbr", "home_team_abbr"),
    "away_team": ("away_team", "away", "away_abbr", "away_team_abbr"),
    "season_type": ("season_type", "game_type", "season"),
}


def _schedule_opponent_map(
    schedule_df: pd.DataFrame | None,
    issues: list[str],
) -> dict[tuple[int, str], str]:
    """Build a conservative ``(week, offense) -> defense`` map.

    The weekly PFF reports identify the offense only.  This map is the
    required bridge to the defense it faced.  If a source team/week maps to
    two opponents, it is omitted instead of guessing.  Schedule rows marked
    preseason or postseason are not permitted to create a regular-season
    alignment profile.
    """
    if schedule_df is None or schedule_df.empty:
        issues.append(
            "Alignment-defense archive needs a non-empty schedule_df with week, home_team, and away_team; "
            "no defense profiles were constructed."
        )
        return {}
    schedule = _normalise_column_names(schedule_df, _SCHEDULE_ALIASES)
    required = {"week", "home_team", "away_team"}
    missing = sorted(required.difference(schedule.columns))
    if missing:
        issues.append(
            "Alignment-defense schedule mapping is unavailable; schedule_df is missing "
            + ", ".join(missing)
            + "."
        )
        return {}
    schedule = schedule.copy()
    weeks = pd.to_numeric(schedule["week"], errors="coerce")
    schedule = schedule.loc[weeks.notna() & weeks.gt(0)].copy()
    schedule["week"] = pd.to_numeric(schedule["week"], errors="coerce").astype(int)
    if "season_type" in schedule.columns:
        # Unknown text is retained; explicit PRE/POST sources are not.  This
        # matches the archive policy: do not invent a status, but never mix a
        # known non-regular game into a regular-season profile.
        regular = schedule["season_type"].map(_coerce_bool)
        schedule = schedule.loc[regular.ne(False)].copy()
    if schedule.empty:
        issues.append("Alignment-defense schedule contains no usable regular-season team-game rows.")
        return {}

    candidates: dict[tuple[int, str], set[str]] = {}
    for row in schedule.itertuples(index=False):
        week = int(getattr(row, "week"))
        home = _canonical_team_key(getattr(row, "home_team"))
        away = _canonical_team_key(getattr(row, "away_team"))
        if not home or not away or home == away:
            issues.append(
                f"Alignment-defense schedule has an invalid Week {week} team pairing; row ignored."
            )
            continue
        candidates.setdefault((week, home), set()).add(away)
        candidates.setdefault((week, away), set()).add(home)

    opponents: dict[tuple[int, str], str] = {}
    for key, values in sorted(candidates.items()):
        if len(values) == 1:
            opponents[key] = next(iter(values))
        else:
            issues.append(
                f"Alignment-defense schedule maps {key[1]} to multiple Week {key[0]} opponents; that team-game is ignored."
            )
    return opponents


def _validate_alignment_defense_report_pair(
    summary: pd.DataFrame,
    concept: pd.DataFrame,
    *,
    label: str,
    issues: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Validate the additional event fields needed for a defense profile.

    The player-role archive only needs a slot route/alignment profile.  A
    defense matchup requires the corresponding *total* and *slot* targets,
    receptions, and yards so non-slot production can be measured as a true
    complement.  Touchdowns are retained when present for audit only and are
    never an active alignment residual input.
    """
    summary_normalized = _normalise_column_names(summary, _SUMMARY_ALIASES)
    concept_normalized = _normalise_column_names(concept, _CONCEPT_ALIASES)
    summary_columns = set(summary_normalized.columns)
    concept_columns = set(concept_normalized.columns)
    checked_summary, summary_valid = _validate_summary(summary_normalized, label + " receiving_summary", issues)
    checked_concept, concept_valid = _validate_concept(concept_normalized, label + " receiving_concept", issues)
    if not (summary_valid and concept_valid):
        return checked_summary, checked_concept, False

    required_total = ("targets", "receptions", "yards")
    required_slot = ("slot_targets", "slot_receptions", "slot_yards")
    missing_total = [column for column in required_total if column not in summary_columns]
    missing_slot = [column for column in required_slot if column not in concept_columns]
    if missing_total:
        issues.append(
            f"{label} cannot build alignment-defense events; receiving_summary is missing "
            + ", ".join(missing_total)
            + "."
        )
    if missing_slot:
        issues.append(
            f"{label} cannot build alignment-defense events; receiving_concept is missing "
            + ", ".join(missing_slot)
            + "."
        )
    return checked_summary, checked_concept, not missing_total and not missing_slot


def _nonnegative_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    values = _numeric_column(frame, column)
    return values.where(values.ge(0.0))


def _group_regular_season_status(values: pd.Series) -> bool | None:
    parsed = [_coerce_bool(value) for value in values.tolist()]
    parsed = [value for value in parsed if value is not None]
    return True if parsed and all(parsed) else None


def _prepare_alignment_team_games(
    combined: pd.DataFrame,
    *,
    source_year: int,
    opponent_map: Mapping[tuple[int, str], str],
    issues: list[str],
) -> pd.DataFrame:
    """Convert player-level weekly PFF rows into offense-defense event rows.

    This is purposefully a team-game aggregation before any defense estimate.
    A backup receiver's tiny personal ratio therefore cannot dominate a
    defense's apparent slot weakness.  The source is the weekly *offensive*
    report; no defender coverage export is read or inferred here.
    """
    if combined.empty:
        return empty_alignment_team_games()
    players = combined.copy()
    players["team_name"] = players["team_name"].map(_canonical_team_key)
    players["position"] = players["position"].map(_normalise_position)
    players = players.loc[players["team_name"].ne("") & players["position"].isin(_SUPPORTED_RECEIVING_POSITIONS)].copy()
    if players.empty:
        return empty_alignment_team_games()

    # Preserve a complete/non-complete indicator by metric.  A partial PFF
    # export should not quietly understate a team game just because one row
    # lacks an event field.
    for stat in ALIGNMENT_DEFENSE_STATS:
        total = _nonnegative_numeric(players, stat)
        slot = _nonnegative_numeric(players, f"slot_{stat}")
        valid = total.notna() & slot.notna() & slot.le(total + 1e-8)
        impossible = total.notna() & slot.notna() & slot.gt(total + 1e-8)
        if impossible.any():
            issues.append(
                f"PFF Week {int(players['source_week'].iloc[0])} has {int(impossible.sum())} player {stat} rows "
                "with slot production greater than total production; those event rows are unavailable."
            )
        players[f"_{stat}_slot"] = slot.where(valid)
        players[f"_{stat}_non_slot"] = (total - slot).where(valid).clip(lower=0.0)
        players[f"_{stat}_valid"] = valid

    slot_snaps = _nonnegative_numeric(players, "slot_snaps").fillna(0.0)
    wide_snaps = _nonnegative_numeric(players, "wide_snaps").fillna(0.0)
    inline_snaps = _nonnegative_numeric(players, "inline_snaps").fillna(0.0)
    reported_total = slot_snaps + wide_snaps + inline_snaps
    fallback_weight = _nonnegative_numeric(players, "_alignment_weight").fillna(0.0)
    slot_rate = pd.to_numeric(players.get("_slot_rate", pd.Series(float("nan"), index=players.index)), errors="coerce")
    fallback_slot = fallback_weight * slot_rate
    fallback_non_slot = fallback_weight * (1.0 - slot_rate)
    players["_slot_exposure"] = slot_snaps.where(reported_total.gt(0.0), fallback_slot)
    players["_non_slot_exposure"] = (wide_snaps + inline_snaps).where(reported_total.gt(0.0), fallback_non_slot)
    players["_derived_alignment_exposure"] = reported_total.le(0.0)

    records: list[dict[str, Any]] = []
    unmapped: set[tuple[int, str]] = set()
    group_columns = ["source_week", "team_name", "position"]
    for (week_raw, offense_team, position), group in players.groupby(group_columns, sort=True, observed=True):
        week = int(week_raw)
        defense_team = opponent_map.get((week, offense_team))
        if not defense_team:
            unmapped.add((week, offense_team))
            continue
        player_rows = int(len(group))
        regular = _group_regular_season_status(group["source_regular_season"])
        confidence_values = [_clean_text(value) for value in group["source_confidence"].tolist()]
        confidence = "; ".join(dict.fromkeys(value for value in confidence_values if value))
        summary_paths = tuple(dict.fromkeys(_clean_text(value) for value in group["source_path"].tolist() if _clean_text(value)))
        concept_paths = tuple(dict.fromkeys(_clean_text(value) for value in group.get("concept_path", pd.Series("", index=group.index)).tolist() if _clean_text(value)))
        source_paths = "; ".join((*summary_paths, *concept_paths))
        derived_count = int(group["_derived_alignment_exposure"].sum())
        exposure_source = (
            "reported_alignment_snaps" if derived_count == 0 else
            ("alignment_rate_weight_fallback" if derived_count == player_rows else "mixed_reported_and_rate_weight")
        )
        for alignment, exposure_col in (("slot", "_slot_exposure"), ("non_slot", "_non_slot_exposure")):
            exposure = pd.to_numeric(group[exposure_col], errors="coerce")
            alignment_snaps = float(exposure.sum(min_count=1)) if exposure.notna().any() else 0.0
            for stat in ALIGNMENT_DEFENSE_STATS:
                value_col = f"_{stat}_{alignment}"
                values = pd.to_numeric(group[value_col], errors="coerce")
                valid_rows = int(values.notna().sum())
                complete = bool(player_rows > 0 and valid_rows == player_rows)
                records.append({
                    "source_year": int(source_year),
                    "source_week": week,
                    "offense_team": offense_team,
                    "defense_team": defense_team,
                    "position": position,
                    "alignment": alignment,
                    "stat": stat,
                    "observed_value": float(values.sum()) if complete else None,
                    "event_available": complete,
                    "event_complete": complete,
                    "player_rows": player_rows,
                    "valid_player_rows": valid_rows,
                    "alignment_snaps": alignment_snaps,
                    "alignment_exposure_source": exposure_source,
                    "source_regular_season": regular,
                    "source_time_valid": True,
                    "source_confidence": confidence or "weekly_schema_valid_local_export",
                    "summary_path": "; ".join(summary_paths),
                    "concept_path": "; ".join(concept_paths),
                    "source_paths": source_paths,
                    "source_notes": (
                        "Team-game event from the manual weekly offensive PFF receiving_summary and "
                        "receiving_concept reports; non_slot is total minus real slot, not literal wide/inline."
                    ),
                })
    for week, offense_team in sorted(unmapped):
        issues.append(
            f"No unambiguous schedule opponent for PFF Week {week} offense {offense_team}; that team-game is excluded."
        )
    return pd.DataFrame(records, columns=ALIGNMENT_TEAM_GAME_COLUMNS) if records else empty_alignment_team_games()


def _attach_offense_leave_one_out_baselines(team_games: pd.DataFrame) -> pd.DataFrame:
    """Attach a quality baseline from an offense's other eligible games.

    For a defense to look soft because it faced a good slot offense is exactly
    the bias this layer needs to avoid.  Each observed team-game is compared
    only with the same offense/position/alignment/stat in *other* archived
    weeks.  If no comparison exists, it is deliberately neutral instead of
    borrowing a completed-season value or a defender-level report.
    """
    out = team_games.copy()
    out["_expected_value"] = float("nan")
    out["_baseline_games"] = 0
    out["_baseline_source"] = "neutral_no_offense_comparison"
    if out.empty:
        return out
    available = out["event_available"].fillna(False).astype(bool)
    observed = pd.to_numeric(out["observed_value"], errors="coerce")
    valid = available & observed.notna() & observed.ge(0.0)
    group_columns = ["offense_team", "position", "alignment", "stat"]
    for _key, group in out.loc[valid].groupby(group_columns, sort=False, observed=True):
        group = group.sort_values(["source_week", "defense_team"], kind="stable")
        for index, row in group.iterrows():
            # Excluding the current source week protects against an accidental
            # duplicated row and prevents an offense's one game from being its
            # own quality comparison.
            others = group.loc[group["source_week"].ne(row["source_week"])]
            values = pd.to_numeric(others["observed_value"], errors="coerce")
            values = values[values.notna() & values.ge(0.0)]
            if values.empty:
                continue
            out.at[index, "_expected_value"] = float(values.mean())
            out.at[index, "_baseline_games"] = int(others["source_week"].nunique())
            out.at[index, "_baseline_source"] = "offense_leave_one_out_time_valid"
    return out


def _position_normal_slot_rates(team_games: pd.DataFrame) -> dict[str, float | None]:
    """Return the observed position-normal slot mix from alignment exposure.

    This is not a defense stat.  It supplies the neutral reference mix used
    by the later residual preview, so a defense's broad WR matchup is not
    counted a second time merely because one player is more slot-heavy.
    """
    if team_games is None or team_games.empty:
        return {}
    base_columns = ["source_year", "source_week", "offense_team", "defense_team", "position", "alignment"]
    existing = [column for column in base_columns if column in team_games.columns]
    unique = team_games.drop_duplicates(subset=existing).copy() if existing else team_games.copy()
    exposure = pd.to_numeric(unique.get("alignment_snaps", pd.Series(float("nan"), index=unique.index)), errors="coerce")
    unique["_alignment_exposure"] = exposure.where(exposure.ge(0.0), 0.0)
    rates: dict[str, float | None] = {}
    for position, group in unique.groupby("position", sort=False, observed=True):
        slot = float(group.loc[group["alignment"].eq("slot"), "_alignment_exposure"].sum())
        non_slot = float(group.loc[group["alignment"].eq("non_slot"), "_alignment_exposure"].sum())
        total = slot + non_slot
        rates[_normalise_position(position)] = (slot / total) if total > 0.0 else None
    return rates


def aggregate_alignment_defense_profiles(
    team_games: pd.DataFrame | None,
    *,
    shrinkage_games: float = ALIGNMENT_DEFENSE_SHRINKAGE_GAMES,
) -> pd.DataFrame:
    """Build bounded, quality-adjusted defense evidence from team-game rows.

    ``team_games`` is expected to come from :func:`load_weekly_alignment_defense_profiles`,
    but the function is public for deterministic tests and future prior-year
    bridges.  It does not create a projection multiplier.  It only stores a
    shrunk observed/expected ratio plus provenance and a candidate-eligibility
    flag for a later, separately backtested residual experiment.
    """
    if team_games is None or team_games.empty:
        return empty_alignment_defense_profiles()
    try:
        shrinkage_games = float(shrinkage_games)
    except (TypeError, ValueError):
        shrinkage_games = ALIGNMENT_DEFENSE_SHRINKAGE_GAMES
    if shrinkage_games < 0.0:
        raise ValueError("shrinkage_games must be non-negative")

    required = {"offense_team", "defense_team", "position", "alignment", "stat", "source_week"}
    if not required.issubset(team_games.columns):
        return empty_alignment_defense_profiles()
    rows = team_games.copy()
    rows["defense_team"] = rows["defense_team"].map(_canonical_team_key)
    rows["offense_team"] = rows["offense_team"].map(_canonical_team_key)
    rows["position"] = rows["position"].map(_normalise_position)
    rows["alignment"] = rows["alignment"].map(_normalise_alignment)
    rows["stat"] = rows["stat"].map(_normalise_alignment_stat)
    rows = rows.loc[
        rows["defense_team"].ne("")
        & rows["offense_team"].ne("")
        & rows["alignment"].isin({"slot", "non_slot"})
        & rows["stat"].isin(ALIGNMENT_DEFENSE_STATS)
    ].copy()
    if rows.empty:
        return empty_alignment_defense_profiles()

    rows = _attach_offense_leave_one_out_baselines(rows)
    normal_slot_rates = _position_normal_slot_rates(rows)
    records: list[dict[str, Any]] = []
    group_columns = ["defense_team", "position", "alignment", "stat"]
    for (defense_team, position, alignment, stat), group in rows.groupby(group_columns, sort=True, observed=True):
        group = group.copy()
        observed = pd.to_numeric(group.get("observed_value"), errors="coerce")
        available = group.get("event_available", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        expected = pd.to_numeric(group.get("_expected_value"), errors="coerce")
        comparable = available & observed.notna() & observed.ge(0.0) & expected.notna() & expected.gt(0.0)
        valid = available & observed.notna() & observed.ge(0.0)
        sample_games = int(group.loc[valid, "source_week"].nunique())
        comparison_games = int(group.loc[comparable, "source_week"].nunique())
        uncompared_games = max(sample_games - comparison_games, 0)
        sample_offenses = int(group.loc[valid, "offense_team"].nunique())
        observed_total = float(observed[comparable].sum()) if comparable.any() else None
        expected_total = float(expected[comparable].sum()) if comparable.any() else None
        raw_ratio = (
            observed_total / expected_total
            if observed_total is not None and expected_total is not None and expected_total > 0.0
            else None
        )
        clipped_ratio = (
            float(min(max(raw_ratio, ALIGNMENT_DEFENSE_RAW_RATIO_CLIP[0]), ALIGNMENT_DEFENSE_RAW_RATIO_CLIP[1]))
            if raw_ratio is not None else None
        )
        shrinkage_weight = (
            comparison_games / (comparison_games + shrinkage_games)
            if comparison_games > 0 else 0.0
        )
        shrunk_ratio = (
            1.0 + shrinkage_weight * (clipped_ratio - 1.0)
            if clipped_ratio is not None else 1.0
        )
        event_weight = (
            expected_total / (expected_total + ALIGNMENT_DEFENSE_EVENT_HALF_WEIGHT)
            if expected_total is not None and expected_total > 0.0 else 0.0
        )
        confidence = float(shrinkage_weight * event_weight)
        normal_slot_rate = normal_slot_rates.get(position)
        regular = _group_regular_season_status(group.get("source_regular_season", pd.Series(dtype=object)))
        source_weeks = tuple(sorted({int(value) for value in group["source_week"].tolist()}))
        source_year_series = group.get("source_year", pd.Series(float("nan"), index=group.index))
        source_years = tuple(sorted({
            int(value) for value in pd.to_numeric(source_year_series, errors="coerce").dropna().tolist()
        }))
        source_paths = "; ".join(dict.fromkeys(
            _clean_text(value) for value in group.get("source_paths", pd.Series("", index=group.index)).tolist() if _clean_text(value)
        ))
        confidence_values = "; ".join(dict.fromkeys(
            _clean_text(value) for value in group.get("source_confidence", pd.Series("", index=group.index)).tolist() if _clean_text(value)
        ))
        baseline_sources = tuple(dict.fromkeys(
            _clean_text(value) for value in group.loc[comparable, "_baseline_source"].tolist() if _clean_text(value)
        ))
        profile_available = bool(comparison_games > 0 and raw_ratio is not None)
        candidate_available = bool(
            profile_available
            and position in ALIGNMENT_DEFENSE_SUPPORTED_POSITIONS
            and stat != "touchdowns"
            and normal_slot_rate is not None
        )
        if stat == "touchdowns":
            fallback_reason = "Touchdown alignment residuals are deliberately neutral pending a dedicated backtest."
        elif not profile_available:
            fallback_reason = (
                "No other time-valid game for this offense/position/alignment/stat; "
                "the candidate residual remains neutral."
            )
        elif normal_slot_rate is None:
            fallback_reason = "Position-normal alignment exposure is unavailable; the candidate residual remains neutral."
        elif position not in ALIGNMENT_DEFENSE_SUPPORTED_POSITIONS:
            fallback_reason = (
                f"{position} alignment is retained for audit only; WR/TE are the first candidate experiment."
            )
        else:
            fallback_reason = ""
        records.append({
            "defense_team": defense_team,
            "position": position,
            "alignment": alignment,
            "stat": stat,
            "profile_available": profile_available,
            "candidate_available": candidate_available,
            "scoring_active": False,
            "observed_total": observed_total,
            "expected_total": expected_total,
            "raw_allowed_ratio": raw_ratio,
            "clipped_allowed_ratio": clipped_ratio,
            "shrunk_allowed_ratio": float(shrunk_ratio),
            "sample_games": sample_games,
            "comparison_games": comparison_games,
            "uncompared_games": uncompared_games,
            "sample_offenses": sample_offenses,
            "event_sample_weight": float(expected_total or 0.0),
            "shrinkage_games": float(shrinkage_games),
            "shrinkage_weight": float(shrinkage_weight),
            "alignment_confidence": confidence,
            "position_normal_slot_rate": normal_slot_rate,
            "normal_alignment_source": "league_alignment_snaps" if normal_slot_rate is not None else "unavailable",
            "baseline_source": "; ".join(baseline_sources) if baseline_sources else "neutral_no_offense_comparison",
            "fallback_reason": fallback_reason,
            "source_kind": "weekly_offensive_alignment_archive",
            "source_method": "offensive_weekly_reports_mapped_via_schedule",
            "source_year": source_years[0] if len(source_years) == 1 else ",".join(str(value) for value in source_years),
            "source_weeks": ",".join(str(value) for value in source_weeks),
            "source_week_count": len(source_weeks),
            "source_regular_season": regular,
            "source_time_valid": bool(group.get("source_time_valid", pd.Series(False, index=group.index)).fillna(False).all()),
            "source_confidence": confidence_values or "weekly_schema_valid_local_export",
            "source_paths": source_paths,
            "source_notes": (
                "Observed/expected is quality-adjusted against the same offense's other time-valid games. "
                "It is bounded and shrunk to 1.0; no scoring multiplier is active."
            ),
            "profile_version": PFF_ALIGNMENT_DEFENSE_VERSION,
        })
    return pd.DataFrame(records, columns=ALIGNMENT_DEFENSE_PROFILE_COLUMNS) if records else empty_alignment_defense_profiles()


def load_weekly_alignment_defense_profiles(
    year: int,
    as_of_week: int,
    schedule_df: pd.DataFrame | None,
    pff_root: str | Path = DEFAULT_PFF_ROOT,
    *,
    shrinkage_games: float = ALIGNMENT_DEFENSE_SHRINKAGE_GAMES,
) -> AlignmentDefenseLoadResult:
    """Load a neutral-by-default alignment-defense evidence set.

    Only archive folders strictly before ``as_of_week`` are read.  Each
    weekly report must pass both the player-role schema and the additional
    total-vs-slot event schema, then map through the provided schedule before
    it can affect a defense record.  An absent schedule, incomplete report,
    non-regular manifest row, or ambiguous team mapping produces a visible
    neutral fallback rather than guessed matchup data.

    This function intentionally has no dependency on ``data.loaders`` or
    PFF's defender coverage exports.  The caller supplies the schedule it
    already has, which keeps the input boundary explicit and makes historical
    cutoff tests straightforward.
    """
    target_week = _validate_week_number(as_of_week)
    source_year = int(year)
    discovered = discover_weekly_alignment_exports(source_year, target_week, pff_root)
    archives = discovered.archives.copy()
    issues = list(discovered.issues)
    opponent_map = _schedule_opponent_map(schedule_df, issues)
    records: list[pd.DataFrame] = []

    if not archives.empty and opponent_map:
        for index, archive in archives.loc[archives["eligible_as_of"].astype(bool)].iterrows():
            week = int(archive["week"])
            local_issues: list[str] = []
            summary = _safe_read_csv(archive["summary_path"], f"PFF Week {week} receiving_summary", local_issues)
            concept = _safe_read_csv(archive["concept_path"], f"PFF Week {week} receiving_concept", local_issues)
            summary, concept, valid = _validate_alignment_defense_report_pair(
                summary,
                concept,
                label=f"PFF Week {week}",
                issues=local_issues,
            )
            archives.at[index, "schema_valid"] = bool(valid)
            archives.at[index, "schema_issues"] = " | ".join(local_issues)
            issues.extend(local_issues)
            if not valid:
                continue
            regular_season = _coerce_bool(archive.get("regular_season"))
            confidence = _clean_text(archive.get("manifest_source_confidence"))
            if not confidence:
                confidence = (
                    "weekly_manifest_regular_season" if regular_season is True
                    else "weekly_schema_valid_regular_season_unverified"
                )
            summary_rows = _prepare_summary_rows(
                summary,
                week=week,
                path=archive["summary_path"],
                regular_season=regular_season,
                source_confidence=confidence,
            )
            concept_rows = _prepare_concept_rows(concept, week=week)
            merged = _merge_week_pair(summary_rows, concept_rows)
            merged["concept_path"] = str(archive["concept_path"])
            records.append(merged)
    elif not archives.empty and not opponent_map:
        # The detail is already in ``issues``.  Mark discovered files as not
        # usable here rather than letting a caller mistake them for profiles.
        archives.loc[archives["eligible_as_of"].astype(bool), "schema_issues"] = (
            "Schedule mapping unavailable; no alignment-defense team-games constructed."
        )

    combined = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    team_games = _prepare_alignment_team_games(
        combined,
        source_year=source_year,
        opponent_map=opponent_map,
        issues=issues,
    ) if not combined.empty and opponent_map else empty_alignment_team_games()
    profiles = aggregate_alignment_defense_profiles(team_games, shrinkage_games=shrinkage_games)
    included_weeks = tuple(sorted({int(value) for value in team_games.get("source_week", pd.Series(dtype=int)).tolist()}))
    profile_weeks = tuple(sorted({int(value) for value in profiles.get("source_weeks", pd.Series(dtype=str)).astype(str).str.split(",").explode().tolist() if str(value).isdigit()}))
    excluded_weeks = tuple(
        int(row.week) for row in archives.itertuples(index=False)
        if not bool(row.eligible_as_of) or row.schema_valid is False
    )
    available = bool(not profiles.empty and profiles["profile_available"].astype(bool).any())
    metadata = {
        **discovered.metadata,
        "available": available,
        "source_kind": "weekly_offensive_alignment_archive",
        "source_method": "offensive_weekly_reports_mapped_via_schedule",
        "as_of_week": target_week,
        "included_weeks": included_weeks,
        "profile_weeks": profile_weeks,
        "excluded_weeks": excluded_weeks,
        "team_game_rows": int(len(team_games)),
        "profile_rows": int(len(profiles)),
        "schedule_mapping_available": bool(opponent_map),
        "alignment_adjustment_mode": "neutral_only_defense_foundation",
        "scoring_active": False,
        "requires_backtest_before_activation": True,
        "defender_slot_coverage_used": False,
    }
    return AlignmentDefenseLoadResult(
        profiles=profiles,
        team_games=team_games,
        archives=archives,
        issues=_unique_issues(issues),
        metadata=metadata,
    )


def lookup_alignment_defense_profile(
    profiles: pd.DataFrame | None,
    *,
    defense_team: Any,
    position: Any,
    alignment: Any,
    stat: Any,
) -> dict[str, Any]:
    """Find exactly one defense evidence record or return a neutral fallback."""
    wanted = {
        "defense_team": _canonical_team_key(defense_team),
        "position": _normalise_position(position),
        "alignment": _normalise_alignment(alignment),
        "stat": _normalise_alignment_stat(stat),
    }
    if profiles is None or profiles.empty:
        return neutral_alignment_defense_profile(**wanted)
    required = set(wanted)
    if not required.issubset(profiles.columns):
        return neutral_alignment_defense_profile("Alignment-defense profile schema is unavailable", **wanted)
    matches = profiles.copy()
    for column, value in wanted.items():
        normalizer = {
            "defense_team": _canonical_team_key,
            "position": _normalise_position,
            "alignment": _normalise_alignment,
            "stat": _normalise_alignment_stat,
        }[column]
        matches = matches[matches[column].map(normalizer).eq(value)]
    if len(matches) == 1:
        return matches.iloc[0].to_dict()
    reason = "No matching alignment-defense profile" if matches.empty else "Ambiguous alignment-defense profile"
    return neutral_alignment_defense_profile(reason, **wanted)


def alignment_defense_residual_multiplier(
    profiles: pd.DataFrame | None,
    *,
    defense_team: Any,
    position: Any,
    player_slot_rate: Any,
    stat: Any,
) -> dict[str, Any]:
    """Preview a bounded alignment residual while always returning 1.0 to score.

    The broad weekly defense model already owns the overall positional
    matchup.  A future experiment may use this helper's
    ``candidate_multiplier`` as the *incremental* alignment effect:

    ``player weighted slot/non-slot factor / position-normal factor``.

    For now, ``multiplier`` is deliberately and unconditionally ``1.0``.
    This gives the UI/decomposition a transparent preview without allowing an
    unvalidated component to alter rankings.  Touchdown statistics and
    unavailable/experimental data remain neutral even in the preview.
    """
    normalized_position = _normalise_position(position)
    normalized_stat = _normalise_alignment_stat(stat)
    result: dict[str, Any] = {
        "multiplier": 1.0,
        "candidate_multiplier": 1.0,
        "applied": False,
        "scoring_active": False,
        "candidate_available": False,
        "reason": "",
        "defense_team": _canonical_team_key(defense_team),
        "position": normalized_position,
        "stat": normalized_stat,
        "player_slot_rate": None,
        "position_normal_slot_rate": None,
        "slot_profile": None,
        "non_slot_profile": None,
        "raw_residual": None,
        "effect_weight": 0.0,
        "residual_clip": ALIGNMENT_DEFENSE_RESIDUAL_CLIP,
    }
    if normalized_stat == "touchdowns":
        result["reason"] = "Touchdown alignment residuals are intentionally neutral."
        return result
    if normalized_position not in ALIGNMENT_DEFENSE_SUPPORTED_POSITIONS:
        result["reason"] = f"{normalized_position} alignment remains audit-only until a position-specific experiment is validated."
        return result
    try:
        slot_rate = float(player_slot_rate)
    except (TypeError, ValueError):
        result["reason"] = "Player slot rate is unavailable."
        return result
    if pd.isna(slot_rate) or slot_rate < 0.0 or slot_rate > 1.0:
        result["reason"] = "Player slot rate must be a bounded 0..1 value."
        return result
    result["player_slot_rate"] = slot_rate
    slot_profile = lookup_alignment_defense_profile(
        profiles,
        defense_team=defense_team,
        position=normalized_position,
        alignment="slot",
        stat=normalized_stat,
    )
    non_slot_profile = lookup_alignment_defense_profile(
        profiles,
        defense_team=defense_team,
        position=normalized_position,
        alignment="non_slot",
        stat=normalized_stat,
    )
    result["slot_profile"] = slot_profile
    result["non_slot_profile"] = non_slot_profile
    if not (slot_profile.get("candidate_available") and non_slot_profile.get("candidate_available")):
        result["reason"] = "Slot/non-slot comparison evidence is unavailable; alignment residual remains neutral."
        return result
    normal_slot = slot_profile.get("position_normal_slot_rate")
    if normal_slot is None:
        normal_slot = non_slot_profile.get("position_normal_slot_rate")
    try:
        normal_slot = float(normal_slot)
        slot_factor = float(slot_profile.get("shrunk_allowed_ratio"))
        non_slot_factor = float(non_slot_profile.get("shrunk_allowed_ratio"))
    except (TypeError, ValueError):
        result["reason"] = "Alignment-defense evidence is incomplete; residual remains neutral."
        return result
    if any(pd.isna(value) for value in (normal_slot, slot_factor, non_slot_factor)) or not 0.0 <= normal_slot <= 1.0:
        result["reason"] = "Alignment-defense evidence is invalid; residual remains neutral."
        return result
    player_factor = slot_rate * slot_factor + (1.0 - slot_rate) * non_slot_factor
    normal_factor = normal_slot * slot_factor + (1.0 - normal_slot) * non_slot_factor
    if normal_factor <= 0.0:
        result["reason"] = "Position-normal alignment factor is non-positive; residual remains neutral."
        return result
    raw_residual = player_factor / normal_factor
    slot_confidence_value = pd.to_numeric(
        pd.Series([slot_profile.get("alignment_confidence")]), errors="coerce"
    ).iloc[0]
    non_slot_confidence_value = pd.to_numeric(
        pd.Series([non_slot_profile.get("alignment_confidence")]), errors="coerce"
    ).iloc[0]
    slot_confidence = 0.0 if pd.isna(slot_confidence_value) else float(slot_confidence_value)
    non_slot_confidence = 0.0 if pd.isna(non_slot_confidence_value) else float(non_slot_confidence_value)
    effect_weight = min(max(slot_confidence, 0.0), max(non_slot_confidence, 0.0), 1.0)
    candidate = 1.0 + effect_weight * (raw_residual - 1.0)
    candidate = float(min(max(candidate, ALIGNMENT_DEFENSE_RESIDUAL_CLIP[0]), ALIGNMENT_DEFENSE_RESIDUAL_CLIP[1]))
    result.update({
        "candidate_multiplier": candidate,
        "candidate_available": True,
        "reason": "Candidate alignment residual preview only; no scoring multiplier is active.",
        "position_normal_slot_rate": normal_slot,
        "raw_residual": float(raw_residual),
        "effect_weight": effect_weight,
    })
    return result


# ---------------------------------------------------------------------------
# Weekly offense-derived man/zone scheme-defense foundation
# ---------------------------------------------------------------------------
# Deliberately a PARALLEL set of functions, not a generalization of the
# alignment-defense pipeline above, even though the math (leave-one-out
# offense baseline, shrinkage, clipping, confidence) is identical in shape.
# Two real blockers make in-place generalization the riskier choice:
# aggregate_alignment_defense_profiles hard-filters to
# alignment.isin({"slot","non_slot"}), and _position_normal_slot_rates reads
# the literal "alignment" column name and 'slot'/'non_slot' values.
# Retrofitting both to a configurable split dimension would touch already-
# tested, carefully reasoned code for a currently-dormant, audit-only
# feature; a clean parallel copy is the safer choice. receiving_scheme.csv is
# also structurally simpler than the slot pair: man_*/zone_* already
# partition every target, so there is no "total minus slot" derivation and no
# impossible-value guard needed here.

PFF_SCHEME_DEFENSE_VERSION = "v1_offensive_weekly_scheme_defense_foundation"
SCHEME_DEFENSE_STATS = ("targets", "receptions", "yards", "touchdowns")
SCHEME_DEFENSE_SUPPORTED_POSITIONS = {"WR", "TE"}

SCHEME_TEAM_GAME_COLUMNS = [
    "source_year",
    "source_week",
    "offense_team",
    "defense_team",
    "position",
    "scheme",
    "stat",
    "observed_value",
    "event_available",
    "event_complete",
    "player_rows",
    "valid_player_rows",
    "scheme_routes",
    "source_regular_season",
    "source_time_valid",
    "source_confidence",
    "source_paths",
    "source_notes",
]

SCHEME_DEFENSE_PROFILE_COLUMNS = [
    "defense_team",
    "position",
    "scheme",
    "stat",
    "profile_available",
    "candidate_available",
    "scoring_active",
    "observed_total",
    "expected_total",
    "raw_allowed_ratio",
    "clipped_allowed_ratio",
    "shrunk_allowed_ratio",
    "sample_games",
    "comparison_games",
    "uncompared_games",
    "sample_offenses",
    "event_sample_weight",
    "shrinkage_games",
    "shrinkage_weight",
    "scheme_confidence",
    "position_normal_man_rate",
    "normal_scheme_source",
    "baseline_source",
    "fallback_reason",
    "source_kind",
    "source_method",
    "source_year",
    "source_weeks",
    "source_week_count",
    "source_regular_season",
    "source_time_valid",
    "source_confidence",
    "source_paths",
    "source_notes",
    "profile_version",
]


def empty_scheme_team_games() -> pd.DataFrame:
    """Return an empty but schema-stable offense-versus-defense scheme evidence table."""
    return pd.DataFrame(columns=SCHEME_TEAM_GAME_COLUMNS)


def empty_scheme_defense_profiles() -> pd.DataFrame:
    """Return an empty but schema-stable scheme-defense profile table."""
    return pd.DataFrame(columns=SCHEME_DEFENSE_PROFILE_COLUMNS)


def neutral_scheme_defense_profile(
    reason: str = "No time-valid, schedule-mapped PFF scheme-defense evidence",
    *,
    defense_team: Any = None,
    position: Any = None,
    scheme: Any = None,
    stat: Any = None,
) -> dict[str, Any]:
    """Return a no-op scheme-defense record for a missing or ambiguous match.

    Mirrors neutral_alignment_defense_profile's contract exactly - a missing
    profile is never a claim that a defense is neutral in reality.
    """
    return {
        "defense_team": _canonical_team_key(defense_team),
        "position": _normalise_position(position),
        "scheme": _clean_text(scheme).lower(),
        "stat": _normalise_alignment_stat(stat),
        "profile_available": False,
        "candidate_available": False,
        "scoring_active": False,
        "observed_total": None,
        "expected_total": None,
        "raw_allowed_ratio": None,
        "clipped_allowed_ratio": None,
        "shrunk_allowed_ratio": 1.0,
        "sample_games": 0,
        "comparison_games": 0,
        "uncompared_games": 0,
        "sample_offenses": 0,
        "event_sample_weight": 0.0,
        "shrinkage_games": ALIGNMENT_DEFENSE_SHRINKAGE_GAMES,
        "shrinkage_weight": 0.0,
        "scheme_confidence": 0.0,
        "position_normal_man_rate": None,
        "normal_scheme_source": "unavailable",
        "baseline_source": "neutral_fallback",
        "fallback_reason": reason,
        "source_kind": "missing",
        "source_method": "offensive_weekly_reports_mapped_via_schedule",
        "source_year": None,
        "source_weeks": "",
        "source_week_count": 0,
        "source_regular_season": None,
        "source_time_valid": False,
        "source_confidence": "none",
        "source_paths": "",
        "source_notes": reason,
        "profile_version": PFF_SCHEME_DEFENSE_VERSION,
    }


def _prepare_scheme_team_games(
    combined: pd.DataFrame,
    *,
    source_year: int,
    opponent_map: Mapping[tuple[int, str], str],
    issues: list[str],
) -> pd.DataFrame:
    """Convert player-level weekly scheme rows into offense-defense event rows.

    Same team-game-not-player-row aggregation as
    _prepare_alignment_team_games (one backup's relief-appearance row must
    not outvote an entire offense's game), but no complement derivation:
    man_*/zone_* already partition every target.
    """
    if combined.empty:
        return empty_scheme_team_games()
    players = combined.copy()
    players["team_name"] = players["team_name"].map(_canonical_team_key)
    players["position"] = players["position"].map(_normalise_position)
    players = players.loc[
        players["team_name"].ne("") & players["position"].isin(_SUPPORTED_RECEIVING_POSITIONS)
    ].copy()
    if players.empty:
        return empty_scheme_team_games()

    value_cols = [f"{prefix}_{stat}" for prefix in ("man", "zone") for stat in SCHEME_DEFENSE_STATS]
    for column in (*value_cols, "man_routes", "zone_routes"):
        if column not in players.columns:
            players[column] = np.nan
        players[column] = pd.to_numeric(players[column], errors="coerce")
    players["_scheme_routes_total"] = players["man_routes"].fillna(0.0) + players["zone_routes"].fillna(0.0)

    records: list[dict[str, Any]] = []
    unmapped: set[tuple[int, str]] = set()
    group_columns = ["source_week", "team_name", "position"]
    for (week_raw, offense_team, position), group in players.groupby(group_columns, sort=True, observed=True):
        week = int(week_raw)
        defense_team = opponent_map.get((week, offense_team))
        if not defense_team:
            unmapped.add((week, offense_team))
            continue
        player_rows = int(len(group))
        regular = _group_regular_season_status(group["source_regular_season"])
        confidence_values = [_clean_text(value) for value in group["source_confidence"].tolist()]
        confidence = "; ".join(dict.fromkeys(value for value in confidence_values if value))
        source_paths = tuple(dict.fromkeys(
            _clean_text(value) for value in group["source_path"].tolist() if _clean_text(value)))
        scheme_routes = float(group["_scheme_routes_total"].sum())

        for scheme in ("man", "zone"):
            for stat in SCHEME_DEFENSE_STATS:
                values = pd.to_numeric(group[f"{scheme}_{stat}"], errors="coerce")
                valid_rows = int(values.notna().sum())
                complete = bool(player_rows > 0 and valid_rows == player_rows)
                records.append({
                    "source_year": int(source_year),
                    "source_week": week,
                    "offense_team": offense_team,
                    "defense_team": defense_team,
                    "position": position,
                    "scheme": scheme,
                    "stat": stat,
                    "observed_value": float(values.sum()) if complete else None,
                    "event_available": complete,
                    "event_complete": complete,
                    "player_rows": player_rows,
                    "valid_player_rows": valid_rows,
                    "scheme_routes": scheme_routes,
                    "source_regular_season": regular,
                    "source_time_valid": True,
                    "source_confidence": confidence or "weekly_schema_valid_local_export",
                    "source_paths": "; ".join(source_paths),
                    "source_notes": (
                        "Team-game event from the manual weekly offensive PFF receiving_scheme report; "
                        "man/zone already partition every target, no derived complement."
                    ),
                })
    for week, offense_team in sorted(unmapped):
        issues.append(
            f"No unambiguous schedule opponent for PFF Week {week} offense {offense_team}; that scheme team-game is excluded."
        )
    return pd.DataFrame(records, columns=SCHEME_TEAM_GAME_COLUMNS) if records else empty_scheme_team_games()


def _attach_offense_leave_one_out_baselines_scheme(team_games: pd.DataFrame) -> pd.DataFrame:
    """Scheme-column twin of _attach_offense_leave_one_out_baselines.

    Identical reasoning (a defense must not look soft merely because it
    faced a good man- or zone-beating offense); duplicated only because the
    alignment version groups by the literal column name "alignment".
    """
    out = team_games.copy()
    out["_expected_value"] = float("nan")
    out["_baseline_games"] = 0
    out["_baseline_source"] = "neutral_no_offense_comparison"
    if out.empty:
        return out
    available = out["event_available"].fillna(False).astype(bool)
    observed = pd.to_numeric(out["observed_value"], errors="coerce")
    valid = available & observed.notna() & observed.ge(0.0)
    group_columns = ["offense_team", "position", "scheme", "stat"]
    for _key, group in out.loc[valid].groupby(group_columns, sort=False, observed=True):
        group = group.sort_values(["source_week", "defense_team"], kind="stable")
        for index, row in group.iterrows():
            others = group.loc[group["source_week"].ne(row["source_week"])]
            values = pd.to_numeric(others["observed_value"], errors="coerce")
            values = values[values.notna() & values.ge(0.0)]
            if values.empty:
                continue
            out.at[index, "_expected_value"] = float(values.mean())
            out.at[index, "_baseline_games"] = int(others["source_week"].nunique())
            out.at[index, "_baseline_source"] = "offense_leave_one_out_time_valid"
    return out


def _position_normal_man_rates(team_games: pd.DataFrame) -> dict[str, float | None]:
    """Observed position-normal man/zone route mix - the neutral reference
    mix used by scheme_defense_residual_multiplier, exactly analogous to
    _position_normal_slot_rates above."""
    if team_games is None or team_games.empty:
        return {}
    base_columns = ["source_year", "source_week", "offense_team", "defense_team", "position", "scheme"]
    existing = [column for column in base_columns if column in team_games.columns]
    unique = team_games.drop_duplicates(subset=existing).copy() if existing else team_games.copy()
    exposure = pd.to_numeric(unique.get("scheme_routes", pd.Series(float("nan"), index=unique.index)), errors="coerce")
    unique["_scheme_exposure"] = exposure.where(exposure.ge(0.0), 0.0)
    rates: dict[str, float | None] = {}
    for position, group in unique.groupby("position", sort=False, observed=True):
        man = float(group.loc[group["scheme"].eq("man"), "_scheme_exposure"].sum())
        zone = float(group.loc[group["scheme"].eq("zone"), "_scheme_exposure"].sum())
        total = man + zone
        rates[_normalise_position(position)] = (man / total) if total > 0.0 else None
    return rates


def aggregate_scheme_defense_profiles(
    team_games: pd.DataFrame | None,
    *,
    shrinkage_games: float = ALIGNMENT_DEFENSE_SHRINKAGE_GAMES,
) -> pd.DataFrame:
    """Build bounded, quality-adjusted man/zone defense evidence from team-game rows.

    Mirrors aggregate_alignment_defense_profiles's math and contract exactly
    (see that function's docstring) with "scheme" (man/zone) standing in for
    "alignment" (slot/non_slot). Never creates a projection multiplier: every
    record carries scoring_active=False.
    """
    if team_games is None or team_games.empty:
        return empty_scheme_defense_profiles()
    try:
        shrinkage_games = float(shrinkage_games)
    except (TypeError, ValueError):
        shrinkage_games = ALIGNMENT_DEFENSE_SHRINKAGE_GAMES
    if shrinkage_games < 0.0:
        raise ValueError("shrinkage_games must be non-negative")

    required = {"offense_team", "defense_team", "position", "scheme", "stat", "source_week"}
    if not required.issubset(team_games.columns):
        return empty_scheme_defense_profiles()
    rows = team_games.copy()
    rows["defense_team"] = rows["defense_team"].map(_canonical_team_key)
    rows["offense_team"] = rows["offense_team"].map(_canonical_team_key)
    rows["position"] = rows["position"].map(_normalise_position)
    rows["scheme"] = rows["scheme"].map(lambda value: _clean_text(value).lower())
    rows["stat"] = rows["stat"].map(_normalise_alignment_stat)
    rows = rows.loc[
        rows["defense_team"].ne("")
        & rows["offense_team"].ne("")
        & rows["scheme"].isin({"man", "zone"})
        & rows["stat"].isin(SCHEME_DEFENSE_STATS)
    ].copy()
    if rows.empty:
        return empty_scheme_defense_profiles()

    rows = _attach_offense_leave_one_out_baselines_scheme(rows)
    normal_man_rates = _position_normal_man_rates(rows)
    records: list[dict[str, Any]] = []
    group_columns = ["defense_team", "position", "scheme", "stat"]
    for (defense_team, position, scheme, stat), group in rows.groupby(group_columns, sort=True, observed=True):
        group = group.copy()
        observed = pd.to_numeric(group.get("observed_value"), errors="coerce")
        available = group.get("event_available", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        expected = pd.to_numeric(group.get("_expected_value"), errors="coerce")
        comparable = available & observed.notna() & observed.ge(0.0) & expected.notna() & expected.gt(0.0)
        valid = available & observed.notna() & observed.ge(0.0)
        sample_games = int(group.loc[valid, "source_week"].nunique())
        comparison_games = int(group.loc[comparable, "source_week"].nunique())
        uncompared_games = max(sample_games - comparison_games, 0)
        sample_offenses = int(group.loc[valid, "offense_team"].nunique())
        observed_total = float(observed[comparable].sum()) if comparable.any() else None
        expected_total = float(expected[comparable].sum()) if comparable.any() else None
        raw_ratio = (
            observed_total / expected_total
            if observed_total is not None and expected_total is not None and expected_total > 0.0
            else None
        )
        clipped_ratio = (
            float(min(max(raw_ratio, ALIGNMENT_DEFENSE_RAW_RATIO_CLIP[0]), ALIGNMENT_DEFENSE_RAW_RATIO_CLIP[1]))
            if raw_ratio is not None else None
        )
        shrinkage_weight = (
            comparison_games / (comparison_games + shrinkage_games)
            if comparison_games > 0 else 0.0
        )
        shrunk_ratio = (
            1.0 + shrinkage_weight * (clipped_ratio - 1.0)
            if clipped_ratio is not None else 1.0
        )
        event_weight = (
            expected_total / (expected_total + ALIGNMENT_DEFENSE_EVENT_HALF_WEIGHT)
            if expected_total is not None and expected_total > 0.0 else 0.0
        )
        confidence = float(shrinkage_weight * event_weight)
        normal_man_rate = normal_man_rates.get(position)
        regular = _group_regular_season_status(group.get("source_regular_season", pd.Series(dtype=object)))
        source_weeks = tuple(sorted({int(value) for value in group["source_week"].tolist()}))
        source_year_series = group.get("source_year", pd.Series(float("nan"), index=group.index))
        source_years = tuple(sorted({
            int(value) for value in pd.to_numeric(source_year_series, errors="coerce").dropna().tolist()
        }))
        source_paths = "; ".join(dict.fromkeys(
            _clean_text(value) for value in group.get("source_paths", pd.Series("", index=group.index)).tolist() if _clean_text(value)
        ))
        confidence_values = "; ".join(dict.fromkeys(
            _clean_text(value) for value in group.get("source_confidence", pd.Series("", index=group.index)).tolist() if _clean_text(value)
        ))
        baseline_sources = tuple(dict.fromkeys(
            _clean_text(value) for value in group.loc[comparable, "_baseline_source"].tolist() if _clean_text(value)
        ))
        profile_available = bool(comparison_games > 0 and raw_ratio is not None)
        candidate_available = bool(
            profile_available
            and position in SCHEME_DEFENSE_SUPPORTED_POSITIONS
            and stat != "touchdowns"
            and normal_man_rate is not None
        )
        if stat == "touchdowns":
            fallback_reason = "Touchdown scheme residuals are deliberately neutral pending a dedicated backtest."
        elif not profile_available:
            fallback_reason = (
                "No other time-valid game for this offense/position/scheme/stat; "
                "the candidate residual remains neutral."
            )
        elif normal_man_rate is None:
            fallback_reason = "Position-normal scheme exposure is unavailable; the candidate residual remains neutral."
        elif position not in SCHEME_DEFENSE_SUPPORTED_POSITIONS:
            fallback_reason = (
                f"{position} scheme is retained for audit only; WR/TE are the first candidate experiment."
            )
        else:
            fallback_reason = ""
        records.append({
            "defense_team": defense_team,
            "position": position,
            "scheme": scheme,
            "stat": stat,
            "profile_available": profile_available,
            "candidate_available": candidate_available,
            "scoring_active": False,
            "observed_total": observed_total,
            "expected_total": expected_total,
            "raw_allowed_ratio": raw_ratio,
            "clipped_allowed_ratio": clipped_ratio,
            "shrunk_allowed_ratio": float(shrunk_ratio),
            "sample_games": sample_games,
            "comparison_games": comparison_games,
            "uncompared_games": uncompared_games,
            "sample_offenses": sample_offenses,
            "event_sample_weight": float(expected_total or 0.0),
            "shrinkage_games": float(shrinkage_games),
            "shrinkage_weight": float(shrinkage_weight),
            "scheme_confidence": confidence,
            "position_normal_man_rate": normal_man_rate,
            "normal_scheme_source": "league_scheme_routes" if normal_man_rate is not None else "unavailable",
            "baseline_source": "; ".join(baseline_sources) if baseline_sources else "neutral_no_offense_comparison",
            "fallback_reason": fallback_reason,
            "source_kind": "weekly_offensive_scheme_archive",
            "source_method": "offensive_weekly_reports_mapped_via_schedule",
            "source_year": source_years[0] if len(source_years) == 1 else ",".join(str(value) for value in source_years),
            "source_weeks": ",".join(str(value) for value in source_weeks),
            "source_week_count": len(source_weeks),
            "source_regular_season": regular,
            "source_time_valid": bool(group.get("source_time_valid", pd.Series(False, index=group.index)).fillna(False).all()),
            "source_confidence": confidence_values or "weekly_schema_valid_local_export",
            "source_paths": source_paths,
            "source_notes": (
                "Observed/expected is quality-adjusted against the same offense's other time-valid games. "
                "It is bounded and shrunk to 1.0; no scoring multiplier is active."
            ),
            "profile_version": PFF_SCHEME_DEFENSE_VERSION,
        })
    return pd.DataFrame(records, columns=SCHEME_DEFENSE_PROFILE_COLUMNS) if records else empty_scheme_defense_profiles()


def load_weekly_scheme_defense_profiles(
    year: int,
    as_of_week: int,
    schedule_df: pd.DataFrame | None,
    pff_root: str | Path = DEFAULT_PFF_ROOT,
    *,
    shrinkage_games: float = ALIGNMENT_DEFENSE_SHRINKAGE_GAMES,
) -> AlignmentDefenseLoadResult:
    """Load a neutral-by-default man/zone scheme-defense evidence set.

    Mirrors load_weekly_alignment_defense_profiles exactly (same as-of
    cutoff, same schedule-mapping requirement, same neutral-on-missing
    contract) but reads receiving_scheme.csv - self-contained, so unlike the
    alignment version there is no receiving_summary/receiving_concept pair or
    total-vs-slot schema check to run first.
    """
    target_week = _validate_week_number(as_of_week)
    source_year = int(year)
    discovered = discover_weekly_alignment_exports(source_year, target_week, pff_root)
    archives = discovered.archives.copy()
    issues = list(discovered.issues)
    opponent_map = _schedule_opponent_map(schedule_df, issues)
    records: list[pd.DataFrame] = []

    if not archives.empty and opponent_map:
        for index, archive in archives.loc[archives["eligible_as_of"].astype(bool)].iterrows():
            week = int(archive["week"])
            scheme_path = archive.get("scheme_path", "")
            if not scheme_path:
                issues.append(f"PFF Week {week} has no receiving_scheme.csv; scheme defense profile skipped for that week.")
                continue
            local_issues: list[str] = []
            scheme = _safe_read_csv(scheme_path, f"PFF Week {week} receiving_scheme", local_issues)
            scheme, scheme_valid = _validate_scheme(scheme, f"PFF Week {week} receiving_scheme", local_issues)
            issues.extend(local_issues)
            if not scheme_valid:
                continue
            regular_season = _coerce_bool(archive.get("regular_season"))
            confidence = _clean_text(archive.get("manifest_source_confidence"))
            if not confidence:
                confidence = (
                    "weekly_manifest_regular_season" if regular_season is True
                    else "weekly_schema_valid_regular_season_unverified"
                )
            records.append(_prepare_scheme_rows(
                scheme, week=week, path=scheme_path,
                regular_season=regular_season, source_confidence=confidence,
            ))
    elif not archives.empty and not opponent_map:
        archives.loc[archives["eligible_as_of"].astype(bool), "schema_issues"] = (
            "Schedule mapping unavailable; no scheme-defense team-games constructed."
        )

    combined = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    team_games = _prepare_scheme_team_games(
        combined,
        source_year=source_year,
        opponent_map=opponent_map,
        issues=issues,
    ) if not combined.empty and opponent_map else empty_scheme_team_games()
    profiles = aggregate_scheme_defense_profiles(team_games, shrinkage_games=shrinkage_games)
    included_weeks = tuple(sorted({int(value) for value in team_games.get("source_week", pd.Series(dtype=int)).tolist()}))
    excluded_weeks = tuple(
        int(row.week) for row in archives.itertuples(index=False)
        if not bool(row.eligible_as_of) or row.schema_valid is False
    )
    metadata = {
        "available": bool(not profiles.empty),
        "source_kind": "weekly_offensive_scheme_archive",
        "source_year": source_year,
        "as_of_week": target_week,
        "included_weeks": included_weeks,
        "excluded_weeks": excluded_weeks,
        "team_game_rows": int(len(team_games)),
        "profile_rows": int(len(profiles)),
        "scoring_active": False,
        "requires_backtest_before_activation": True,
    }
    return AlignmentDefenseLoadResult(profiles, team_games, archives, _unique_issues(issues), metadata)


def lookup_scheme_defense_profile(
    profiles: pd.DataFrame | None,
    *,
    defense_team: Any,
    position: Any,
    scheme: Any,
    stat: Any,
) -> dict[str, Any]:
    """Find exactly one scheme-defense evidence record or return a neutral fallback."""
    wanted = {
        "defense_team": _canonical_team_key(defense_team),
        "position": _normalise_position(position),
        "scheme": _clean_text(scheme).lower(),
        "stat": _normalise_alignment_stat(stat),
    }
    if profiles is None or profiles.empty:
        return neutral_scheme_defense_profile(**wanted)
    required = set(wanted)
    if not required.issubset(profiles.columns):
        return neutral_scheme_defense_profile("Scheme-defense profile schema is unavailable", **wanted)
    matches = profiles.copy()
    for column, value in wanted.items():
        normalizer = {
            "defense_team": _canonical_team_key,
            "position": _normalise_position,
            "scheme": lambda v: _clean_text(v).lower(),
            "stat": _normalise_alignment_stat,
        }[column]
        matches = matches[matches[column].map(normalizer).eq(value)]
    if len(matches) == 1:
        return matches.iloc[0].to_dict()
    reason = "No matching scheme-defense profile" if matches.empty else "Ambiguous scheme-defense profile"
    return neutral_scheme_defense_profile(reason, **wanted)


def scheme_defense_residual_multiplier(
    profiles: pd.DataFrame | None,
    *,
    defense_team: Any,
    position: Any,
    player_man_rate: Any,
    stat: Any,
) -> dict[str, Any]:
    """Preview a bounded man/zone residual while always returning 1.0 to score.

    Exact structural mirror of alignment_defense_residual_multiplier: the
    broad weekly defense model already owns the overall positional matchup;
    ``multiplier`` stays unconditionally 1.0, ``candidate_multiplier`` is a
    transparent preview only.
    """
    normalized_position = _normalise_position(position)
    normalized_stat = _normalise_alignment_stat(stat)
    result: dict[str, Any] = {
        "multiplier": 1.0,
        "candidate_multiplier": 1.0,
        "applied": False,
        "scoring_active": False,
        "candidate_available": False,
        "reason": "",
        "defense_team": _canonical_team_key(defense_team),
        "position": normalized_position,
        "stat": normalized_stat,
        "player_man_rate": None,
        "position_normal_man_rate": None,
        "man_profile": None,
        "zone_profile": None,
        "raw_residual": None,
        "effect_weight": 0.0,
        "residual_clip": ALIGNMENT_DEFENSE_RESIDUAL_CLIP,
    }
    if normalized_stat == "touchdowns":
        result["reason"] = "Touchdown scheme residuals are intentionally neutral."
        return result
    if normalized_position not in SCHEME_DEFENSE_SUPPORTED_POSITIONS:
        result["reason"] = f"{normalized_position} scheme remains audit-only until a position-specific experiment is validated."
        return result
    try:
        man_rate = float(player_man_rate)
    except (TypeError, ValueError):
        result["reason"] = "Player man-coverage route share is unavailable."
        return result
    if pd.isna(man_rate) or man_rate < 0.0 or man_rate > 1.0:
        result["reason"] = "Player man-coverage route share must be a bounded 0..1 value."
        return result
    result["player_man_rate"] = man_rate
    man_profile = lookup_scheme_defense_profile(
        profiles, defense_team=defense_team, position=normalized_position, scheme="man", stat=normalized_stat)
    zone_profile = lookup_scheme_defense_profile(
        profiles, defense_team=defense_team, position=normalized_position, scheme="zone", stat=normalized_stat)
    result["man_profile"] = man_profile
    result["zone_profile"] = zone_profile
    if not (man_profile.get("candidate_available") and zone_profile.get("candidate_available")):
        result["reason"] = "Man/zone comparison evidence is unavailable; scheme residual remains neutral."
        return result
    normal_man = man_profile.get("position_normal_man_rate")
    if normal_man is None:
        normal_man = zone_profile.get("position_normal_man_rate")
    try:
        normal_man = float(normal_man)
        man_factor = float(man_profile.get("shrunk_allowed_ratio"))
        zone_factor = float(zone_profile.get("shrunk_allowed_ratio"))
    except (TypeError, ValueError):
        result["reason"] = "Scheme-defense evidence is incomplete; residual remains neutral."
        return result
    if any(pd.isna(value) for value in (normal_man, man_factor, zone_factor)) or not 0.0 <= normal_man <= 1.0:
        result["reason"] = "Scheme-defense evidence is invalid; residual remains neutral."
        return result
    player_factor = man_rate * man_factor + (1.0 - man_rate) * zone_factor
    normal_factor = normal_man * man_factor + (1.0 - normal_man) * zone_factor
    if normal_factor <= 0.0:
        result["reason"] = "Position-normal scheme factor is non-positive; residual remains neutral."
        return result
    raw_residual = player_factor / normal_factor
    man_confidence_value = pd.to_numeric(pd.Series([man_profile.get("scheme_confidence")]), errors="coerce").iloc[0]
    zone_confidence_value = pd.to_numeric(pd.Series([zone_profile.get("scheme_confidence")]), errors="coerce").iloc[0]
    man_confidence = 0.0 if pd.isna(man_confidence_value) else float(man_confidence_value)
    zone_confidence = 0.0 if pd.isna(zone_confidence_value) else float(zone_confidence_value)
    effect_weight = min(max(man_confidence, 0.0), max(zone_confidence, 0.0), 1.0)
    candidate = 1.0 + effect_weight * (raw_residual - 1.0)
    candidate = float(min(max(candidate, ALIGNMENT_DEFENSE_RESIDUAL_CLIP[0]), ALIGNMENT_DEFENSE_RESIDUAL_CLIP[1]))
    result.update({
        "candidate_multiplier": candidate,
        "candidate_available": True,
        "reason": "Candidate scheme residual preview only; no scoring multiplier is active.",
        "position_normal_man_rate": normal_man,
        "raw_residual": float(raw_residual),
        "effect_weight": effect_weight,
    })
    return result


def _read_season_metadata(year_dir: Path, explicit: Mapping[str, Any] | None, issues: list[str]) -> dict[str, Any]:
    """Load an optional audited season sidecar, with explicit args taking priority."""
    metadata: dict[str, Any] = {}
    path = year_dir / "season_manifest.csv"
    if path.exists():
        try:
            frame = pd.read_csv(path, low_memory=False)
            if not frame.empty:
                normalised = _normalise_column_names(frame, _SEASON_METADATA_ALIASES)
                metadata.update(normalised.iloc[-1].to_dict())
            else:
                issues.append(f"Season PFF manifest {path} is empty.")
        except Exception as exc:
            issues.append(f"Could not read season PFF manifest {path}: {type(exc).__name__}: {exc}")
    if explicit:
        metadata.update(dict(explicit))
    metadata = _normalise_metadata_mapping(metadata, _SEASON_METADATA_ALIASES)
    metadata["regular_season"] = _coerce_bool(metadata.get("regular_season"))
    metadata["time_valid"] = _coerce_bool(metadata.get("time_valid"))
    metadata["schema_valid"] = _coerce_bool(metadata.get("schema_valid"))
    metadata["source_confidence"] = _clean_text(metadata.get("source_confidence"))
    metadata["export_date"] = _clean_text(metadata.get("export_date"))
    return metadata


def _same_file_content(paths: Sequence[Path]) -> bool:
    if len(paths) < 2:
        return True
    digests = []
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digests.append(digest.hexdigest())
    return len(set(digests)) == 1


def _find_season_export(year_dir: Path, stem: str, issues: list[str]) -> Path | None:
    """Locate a single, reviewable season export without arbitrary glob choice."""
    exact = year_dir / f"{stem}.csv"
    if exact.is_file():
        return exact
    candidates = sorted(path for path in year_dir.glob(f"{stem}*.csv") if path.is_file())
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1 and _same_file_content(candidates):
        issues.append(f"Season PFF export {stem} has identical duplicate files; using {candidates[0].name} once.")
        return candidates[0]
    if len(candidates) > 1:
        issues.append(f"Season PFF export {stem} has multiple non-identical candidates; source is not auto-selected.")
    else:
        issues.append(f"Season PFF export {stem} is missing from {year_dir}.")
    return None


def load_season_alignment_prior(
    source_year: int,
    target_year: int,
    target_week: int,
    pff_root: str | Path = DEFAULT_PFF_ROOT,
    source_metadata: Mapping[str, Any] | None = None,
    *,
    historical_backtest: bool = False,
) -> AlignmentLoadResult:
    """Load an explicitly audited prior-season Week 1 player-role prior.

    This intentionally refuses an unmanifested season total.  To authorize a
    source, pass ``source_metadata={"regular_season": True}`` or create
    ``pff_imports/{source_year}/season_manifest.csv`` with that field.  A
    historical backtest additionally requires ``time_valid: true`` because a
    later downloaded total cannot be assumed available at the historical run.
    """
    target_week = _validate_week_number(target_week)
    source_year, target_year = int(source_year), int(target_year)
    issues: list[str] = []
    root = Path(pff_root)
    year_dir = root / str(source_year)
    archives = pd.DataFrame(columns=ARCHIVE_COLUMNS)
    base_metadata = {
        "available": False,
        "source_kind": "season_prior",
        "source_year": source_year,
        "target_year": target_year,
        "target_week": target_week,
        "historical_backtest": bool(historical_backtest),
        "alignment_adjustment_mode": "neutral_only_foundation",
    }
    if source_year >= target_year:
        issues.append("A season alignment prior must come from a completed prior year, not the target year or future.")
        return AlignmentLoadResult(empty_alignment_profiles(), archives, issues, base_metadata)
    if not year_dir.is_dir():
        issues.append(f"No PFF season directory exists at {year_dir}.")
        return AlignmentLoadResult(empty_alignment_profiles(), archives, issues, base_metadata)

    metadata = _read_season_metadata(year_dir, source_metadata, issues)
    if metadata.get("regular_season") is not True:
        issues.append(
            f"Season {source_year} PFF alignment prior is not loaded because regular-season-only status is unverified. "
            "Add season_manifest.csv or pass reviewed source_metadata with regular_season=True."
        )
        return AlignmentLoadResult(empty_alignment_profiles(), archives, _unique_issues(issues), base_metadata)
    if metadata.get("schema_valid") is False:
        issues.append(f"Season {source_year} PFF alignment prior is marked schema-invalid and is not loaded.")
        return AlignmentLoadResult(empty_alignment_profiles(), archives, _unique_issues(issues), base_metadata)
    if historical_backtest and metadata.get("time_valid") is not True:
        issues.append(
            f"Season {source_year} PFF prior is unavailable in a historical backtest without explicit time_valid=True metadata."
        )
        return AlignmentLoadResult(empty_alignment_profiles(), archives, _unique_issues(issues), base_metadata)

    summary_path = _find_season_export(year_dir, "receiving_summary", issues)
    concept_path = _find_season_export(year_dir, "receiving_concept", issues)
    if summary_path is None or concept_path is None:
        return AlignmentLoadResult(empty_alignment_profiles(), archives, _unique_issues(issues), base_metadata)
    summary = _safe_read_csv(summary_path, f"PFF season {source_year} receiving_summary", issues)
    concept = _safe_read_csv(concept_path, f"PFF season {source_year} receiving_concept", issues)
    summary, summary_valid = _validate_summary(summary, f"PFF season {source_year} receiving_summary", issues)
    concept, concept_valid = _validate_concept(concept, f"PFF season {source_year} receiving_concept", issues)
    if not (summary_valid and concept_valid):
        return AlignmentLoadResult(empty_alignment_profiles(), archives, _unique_issues(issues), base_metadata)

    confidence = metadata.get("source_confidence") or "season_manifest_regular_season"
    summary_rows = _prepare_summary_rows(
        summary,
        week=0,
        path=str(summary_path),
        regular_season=True,
        source_confidence=confidence,
    )
    concept_rows = _prepare_concept_rows(concept, week=0)
    combined = _merge_week_pair(summary_rows, concept_rows)
    profiles = _aggregate_profiles(
        combined,
        source_kind="season_prior",
        source_year=source_year,
        source_time_valid=True,
        source_notes=(
            f"Audited regular-season-only {source_year} PFF player-alignment prior for "
            f"Week {target_week} of {target_year}."
        ),
    )
    season_record = {
        "year": source_year,
        "week": 0,
        "summary_path": str(summary_path),
        "concept_path": str(concept_path),
        "pair_complete": True,
        "manifest_present": (year_dir / "season_manifest.csv").is_file() or bool(source_metadata),
        "regular_season": True,
        "regular_season_status": "confirmed",
        "manifest_schema_valid": metadata.get("schema_valid"),
        "manifest_source_confidence": confidence,
        "manifest_export_date": metadata.get("export_date", ""),
        "eligible_as_of": True,
        "schema_valid": True,
        "schema_issues": "",
    }
    archives = pd.DataFrame([season_record], columns=ARCHIVE_COLUMNS)
    final_metadata = {
        **base_metadata,
        "available": not profiles.empty,
        "regular_season_confirmed": True,
        "time_valid": True,
        "source_confidence": confidence,
        "source_paths": (str(summary_path), str(concept_path)),
    }
    return AlignmentLoadResult(profiles, archives, _unique_issues(issues), final_metadata)


def lookup_alignment_profile(
    profiles: pd.DataFrame | None,
    *,
    player_id: Any = None,
    player: str | None = None,
    team: str | None = None,
    position: str | None = None,
) -> dict[str, Any]:
    """Find a profile by stable PFF ID first, otherwise a conservative fallback.

    Ambiguous fallback names intentionally return a neutral object.  A future
    projection integration may use a crosswalk before this lookup, but this
    helper never guesses which of two same-name players should receive an
    alignment effect.
    """
    if profiles is None or profiles.empty:
        return neutral_alignment_profile()
    frame = profiles.copy()
    if player_id is not None and "player_id" in frame.columns:
        desired_id = _clean_id(player_id)
        matches = frame[frame["player_id"].map(_clean_id).eq(desired_id)] if desired_id else frame.iloc[0:0]
        if len(matches) == 1:
            return matches.iloc[0].to_dict()
        if len(matches) > 1:
            return neutral_alignment_profile("Ambiguous PFF player_id alignment profile")
    if not player:
        return neutral_alignment_profile()
    matches = frame[frame.get("player", pd.Series("", index=frame.index)).map(_name_key).eq(_name_key(player))]
    if team:
        matches = matches[matches.get("team", pd.Series("", index=matches.index)).map(_team_key).eq(_team_key(team))]
    if position:
        matches = matches[matches.get("position", pd.Series("", index=matches.index)).map(_normalise_position).eq(_normalise_position(position))]
    if len(matches) == 1:
        return matches.iloc[0].to_dict()
    return neutral_alignment_profile(
        "No unique local PFF alignment profile" if matches.empty else "Ambiguous name-based PFF alignment profile"
    )


def lookup_scheme_profile(
    profiles: pd.DataFrame | None,
    *,
    player_id: Any = None,
    player: str | None = None,
    team: str | None = None,
    position: str | None = None,
) -> dict[str, Any]:
    """Player-level man/zone tendency lookup - exact structural mirror of
    lookup_alignment_profile, PFF ID first then a conservative name/team/
    position fallback, neutral_scheme_profile() on any ambiguity or miss."""
    if profiles is None or profiles.empty:
        return neutral_scheme_profile()
    frame = profiles.copy()
    if player_id is not None and "player_id" in frame.columns:
        desired_id = _clean_id(player_id)
        matches = frame[frame["player_id"].map(_clean_id).eq(desired_id)] if desired_id else frame.iloc[0:0]
        if len(matches) == 1:
            return matches.iloc[0].to_dict()
        if len(matches) > 1:
            return neutral_scheme_profile("Ambiguous PFF player_id scheme profile")
    if not player:
        return neutral_scheme_profile()
    matches = frame[frame.get("player", pd.Series("", index=frame.index)).map(_name_key).eq(_name_key(player))]
    if team:
        matches = matches[matches.get("team", pd.Series("", index=matches.index)).map(_team_key).eq(_team_key(team))]
    if position:
        matches = matches[matches.get("position", pd.Series("", index=matches.index)).map(_normalise_position).eq(_normalise_position(position))]
    if len(matches) == 1:
        return matches.iloc[0].to_dict()
    return neutral_scheme_profile(
        "No unique local PFF scheme profile" if matches.empty else "Ambiguous name-based PFF scheme profile"
    )
