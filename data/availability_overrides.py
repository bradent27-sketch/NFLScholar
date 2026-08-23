"""Reviewable target-week availability overrides and safe roster resolution.

The weekly model deliberately keeps availability separate from a depth-chart
role.  A manually maintained CSV can therefore correct a current report
without using an Ourlads display color as a proxy for whether a player will
actually play.  Every external profile is resolved conservatively onto the
current roster and retains its match method for the projection decomposition.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.player_aliases import canonical_player_key
from data.utils import clean_name_exact, clean_name_for_merge


AVAILABILITY_OVERRIDE_PATH = Path(__file__).with_name("availability_overrides.csv")
AVAILABILITY_OVERRIDE_COLUMNS = (
    "year", "week", "team", "player", "status",
    "plays_probability", "workload_if_active", "note",
)

_OUT_STATUSES = frozenset({"out", "ir", "suspended", "inactive", "nfi", "pup"})
_STATUS_PROBABILITY = {"doubtful": 0.25, "questionable": 0.85}


def _text(values: Any) -> pd.Series:
    series = pd.Series(values, copy=False).astype(object)
    return series.where(series.notna(), "").astype(str).str.strip()


def _team_key(values: Any) -> pd.Series:
    series = _text(values).str.upper()
    return series.replace({"OAK": "LV", "SD": "LAC", "STL": "LA"})


def _bool(value: Any) -> bool:
    if value is None or value is pd.NA:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def _id_key(value: Any) -> str:
    text = _text([value]).iloc[0]
    if text.lower() in {"", "nan", "none", "<na>", "0"}:
        return ""
    return f"id:{text.removesuffix('.0')}"


def _probability(status: Any, supplied: Any = np.nan) -> float:
    explicit = pd.to_numeric(pd.Series([supplied]), errors="coerce").iloc[0]
    if pd.notna(explicit):
        return float(np.clip(explicit, 0.0, 1.0))
    label = _text([status]).str.lower().iloc[0]
    if label in _OUT_STATUSES:
        return 0.0
    return float(_STATUS_PROBABILITY.get(label, 1.0))


def _workload(supplied: Any) -> float:
    value = pd.to_numeric(pd.Series([supplied]), errors="coerce").iloc[0]
    return float(np.clip(value, 0.0, 1.0)) if pd.notna(value) else 1.0


def _empty_override_table() -> pd.DataFrame:
    return pd.DataFrame(columns=AVAILABILITY_OVERRIDE_COLUMNS)


def load_availability_overrides(year: int, week: int,
                                path: str | Path = AVAILABILITY_OVERRIDE_PATH) -> tuple[pd.DataFrame, str | None]:
    """Load manual target-week availability choices without guessing.

    The optional CSV schema is ``year,week,team,player,status,plays_probability,
    workload_if_active,note``.  A row has to identify one team/player at the
    requested year/week; malformed or ambiguous rows surface as warnings to
    the caller instead of quietly changing projections.
    """
    target = Path(path)
    if not target.is_file():
        return _empty_override_table(), None
    try:
        frame = pd.read_csv(target, dtype=str, keep_default_na=False)
    except Exception as exc:
        return _empty_override_table(), f"Could not read {target.name}: {exc}"
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    missing = [column for column in ("year", "week", "team", "player", "status")
               if column not in normalized]
    if missing:
        return _empty_override_table(), (
            f"{target.name} is missing required column(s): {', '.join(missing)}.")
    frame = frame.rename(columns={source: key for key, source in normalized.items()
                                  if key in AVAILABILITY_OVERRIDE_COLUMNS})
    for column in AVAILABILITY_OVERRIDE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame.loc[:, list(AVAILABILITY_OVERRIDE_COLUMNS)].copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame["team"] = _team_key(frame["team"])
    frame["player"] = _text(frame["player"])
    selected = frame[
        frame["year"].eq(int(year)) & frame["week"].eq(int(week))
        & frame["team"].ne("") & frame["player"].ne("")
    ].copy()
    if selected.empty:
        return _empty_override_table(), None
    duplicate = selected.groupby(["team", "player"], observed=True).size()
    duplicated = duplicate[duplicate.gt(1)]
    if not duplicated.empty:
        return _empty_override_table(), (
            f"{target.name} has duplicate manual availability choices for "
            + ", ".join(f"{team} {player}" for team, player in duplicated.index) + ".")
    return selected.reset_index(drop=True), None


def _roster_identity_pool(roster: pd.DataFrame, name_col: str, team_col: str) -> pd.DataFrame:
    if roster is None or roster.empty or name_col not in roster.columns or team_col not in roster.columns:
        return pd.DataFrame()
    columns = [name_col, team_col]
    columns.extend(column for column in ("player_id", "gsis_id", "pff_id", "position", "depth_chart_position")
                   if column in roster.columns)
    pool = roster.loc[:, list(dict.fromkeys(columns))].dropna(subset=[name_col]).copy()
    pool["_roster_name"] = _text(pool[name_col])
    pool["_team"] = _team_key(pool[team_col])
    pool["_exact"] = clean_name_exact(pool["_roster_name"])
    pool["_alias"] = canonical_player_key(pool["_roster_name"])
    pool["_loose"] = clean_name_for_merge(pool["_roster_name"])
    pool["_id"] = ""
    for column in ("player_id", "gsis_id", "pff_id"):
        if column not in pool.columns:
            continue
        candidate = pool[column].map(_id_key)
        pool["_id"] = pool["_id"].where(pool["_id"].ne(""), candidate)
    # Exact same player can appear in an overlaid/current source twice.  Keep
    # one deterministic roster identity, not two possible matches.
    return pool.drop_duplicates(subset=["_team", "_id", "_exact"], keep="first")


def _match_one(pool: pd.DataFrame, player: Any, *, team: Any = "", player_id: Any = "") -> tuple[pd.Series | None, str]:
    """Resolve one source row safely; never falls back to a last-name join."""
    if pool.empty:
        return None, "no current roster"
    candidates = pool.copy()
    source_team = _team_key([team]).iloc[0]
    if source_team:
        candidates = candidates[candidates["_team"].eq(source_team)]
    source_id = _id_key(player_id)
    if source_id:
        matches = candidates[candidates["_id"].eq(source_id)]
        if len(matches) == 1:
            return matches.iloc[0], "stable player ID"
        if len(matches) > 1:
            return None, "ambiguous stable player ID"
    source_exact = clean_name_exact(pd.Series([player])).iloc[0]
    matches = candidates[candidates["_exact"].eq(source_exact)]
    if len(matches) == 1:
        return matches.iloc[0], "exact normalized full name"
    if len(matches) > 1:
        return None, "ambiguous exact full name"
    source_alias = canonical_player_key(pd.Series([player])).iloc[0]
    matches = candidates[candidates["_alias"].eq(source_alias)]
    if len(matches) == 1:
        return matches.iloc[0], "reviewed alias"
    if len(matches) > 1:
        return None, "ambiguous reviewed alias"
    source_loose = clean_name_for_merge(pd.Series([player])).iloc[0]
    matches = candidates[candidates["_loose"].eq(source_loose)]
    if len(matches) == 1:
        return matches.iloc[0], "unique suffix-stripped full name"
    if len(matches) > 1:
        return None, "ambiguous suffix-stripped full name"
    return None, "not on current roster"


def resolve_target_week_availability(raw_profiles: dict[str, dict[str, Any]] | None,
                                     overrides: pd.DataFrame | None,
                                     roster: pd.DataFrame, name_col: str, team_col: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Resolve current report + manual override profiles onto roster display names.

    ``raw_profiles`` represents a target-season injury feed.  Manual entries
    win on the same player; source chart status is deliberately not accepted
    here, so a red Ourlads row can only produce a warning in the UI, never a
    zeroed workload by itself.
    """
    pool = _roster_identity_pool(roster, name_col, team_col)
    profiles: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for source_name, source_profile in (raw_profiles or {}).items():
        profile = dict(source_profile or {})
        row, method = _match_one(
            pool, source_name,
            team=profile.get("team", ""),
            player_id=profile.get("player_id", profile.get("gsis_id", "")),
        )
        if row is None:
            warnings.append(f"Availability source '{source_name}' was {method}; ignored.")
            continue
        roster_name = str(row["_roster_name"])
        status = str(profile.get("status", "")).strip().lower() or "unknown"
        profiles[roster_name] = {
            **profile,
            "status": status,
            "plays_probability": _probability(status, profile.get("plays_probability")),
            "workload_if_active": _workload(profile.get("workload_if_active")),
            "source": profile.get("source", "target-season injury report"),
            "source_name": str(source_name),
            "match_method": method,
            "manual_override": False,
        }

    if overrides is None or overrides.empty:
        return profiles, warnings
    for _, item in overrides.iterrows():
        row, method = _match_one(
            pool, item.get("player", ""), team=item.get("team", ""),
            player_id=item.get("player_id", ""),
        )
        if row is None:
            warnings.append(
                f"Manual availability override {item.get('team', '')} {item.get('player', '')} was {method}; ignored.")
            continue
        roster_name = str(row["_roster_name"])
        status = str(item.get("status", "")).strip().lower() or "unknown"
        profiles[roster_name] = {
            "status": status,
            "plays_probability": _probability(status, item.get("plays_probability")),
            "workload_if_active": _workload(item.get("workload_if_active")),
            "source_year": int(pd.to_numeric(item.get("year"), errors="coerce")),
            "source": "manual target-week availability override",
            "source_name": str(item.get("player", "")),
            "match_method": method,
            "manual_override": True,
            "note": str(item.get("note", "")).strip(),
        }
    return profiles, warnings
