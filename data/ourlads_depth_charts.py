"""Local Ourlads depth-chart imports for preseason role evidence.

This module deliberately never contacts Ourlads.  It reads the printer-friendly
pages a user has already saved locally, normalizes their offensive depth-chart
rows, and persists only a small, derived local snapshot.  That makes the
weekly model reproducible without turning an app refresh into a scraper or
requiring a browser session.

The source is *not* a snap-count projection.  Its first-listed QB can resolve
an otherwise ambiguous Week 1 QB1, while QB/RB/WR/TE order is exposed to the
weekly model as conservative preseason role evidence.  Fullbacks are retained
as their own functional role rather than silently folded into RB.  In
particular, LWR, RWR, and SWR remain distinct source labels even though all
map to fantasy WR: three listed receiving starters do not imply three players
will play every offensive snap.
"""
from __future__ import annotations

import re
import shutil
from datetime import datetime
from email import policy
from email.parser import BytesParser
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from config import PFF_TEAM_CODE_TO_ABBR, TEAM_CONFIG
from data.player_aliases import (
    STABLE_PLAYER_ID_COLUMNS,
    canonical_player_key,
    normalize_player_identifier,
    stable_roster_identity_keys,
)
from data.utils import clean_name_exact, clean_name_for_merge


PARSER_VERSION = "2026.1"
_REPO_ROOT = Path(__file__).resolve().parents[1]
OURLADS_IMPORT_PATH = _REPO_ROOT / "external_data" / "ourlads_depth_charts.csv"
# Frozen pre-Week-1 archive, one row per team per past season, built by
# scripts/import_ourlads_historical.py. load_ourlads_snapshot(year) falls back
# to this when the live snapshot has no rows for a past year - so a Week-1..3
# cold-start backtest can see the chart the model would have had at kickoff.
OURLADS_HISTORY_PATH = _REPO_ROOT / "external_data" / "ourlads_depth_charts_history.csv"
OURLADS_KEY_PATH = _REPO_ROOT / "external_data" / "ourlads_depth_chart_key.txt"

# History-CSV `slot` -> the model's coarse source position. Anything not here
# (O-line, defense, special teams) falls through and is dropped by the
# _SOURCE_POSITIONS filter, exactly like the live loader.
_HISTORY_SLOT_TO_POSITION = {
    "QB": "QB", "RB": "RB", "FB": "FB", "TE": "TE",
    "WR": "WR", "LWR": "WR", "RWR": "WR", "SWR": "WR",
}
_HISTORY_UNIT_TO_LABEL = {
    "OFF": "Offense", "DEF": "Defense", "ST": "Special Teams",
    "PS": "Practice Squad", "RES": "Reserves",
}
# Drop fresh saved team pages here; the Depth Charts tab's "Import from
# ourlads_inbox/" button and scripts/import_ourlads.py both read this folder.
OURLADS_INBOX_DIR = _REPO_ROOT / "external_data" / "ourlads_inbox"
# Every import copies the OUTGOING snapshot csv and the raw pages it consumed
# here first, under a timestamped subfolder, so a past-week analysis can pin
# the exact chart the model saw.
OURLADS_ARCHIVE_DIR = _REPO_ROOT / "external_data" / "ourlads_archive"
_OURLADS_PAGE_SUFFIXES = (".mhtml", ".mht", ".html")


def _current_nfl_season() -> int:
    """Season in progress or most recently completed (NFL runs Sep-Feb, so
    before August the current season is still last calendar year's)."""
    today = datetime.now()
    return today.year if today.month >= 8 else today.year - 1

OURLADS_COLUMNS = (
    "year", "team", "unit", "position_label", "position", "depth_rank",
    "source_row", "source_slot", "position_occurrence", "is_listed_starter", "is_inactive", "status_class",
    "source_player_id", "player", "player_key", "raw_player", "source_updated_at",
    "source_file", "source_url", "parser_version",
)

# Raw Ourlads pages do not expose GSIS/PFF ids.  Keep these optional rather
# than making them a parser requirement so a locally reviewed enrichment can
# supply a stable bridge without breaking existing saved snapshots.
_OPTIONAL_SOURCE_ID_COLUMNS = (
    "source_gsis_id", "source_nflverse_id", "source_player_identity_id",
    "source_pff_id", "source_pff_player_id", "source_espn_id",
    "source_pfr_id", "source_sportradar_id", "gsis_id", "nflverse_id",
    "player_id", "pff_id", "pff_player_id", "espn_id", "pfr_id",
    "sportradar_id",
)

# Ourlads commonly uses its own / historical abbreviations in the saved URL.
# Keep this local to the importer: downstream code should always see the
# app's nflverse-style team keys from TEAM_CONFIG.
_OURLADS_TEAM_ALIASES = {
    "ARZ": "ARI", "ARI": "ARI",
    "BLT": "BAL", "BAL": "BAL",
    "CLV": "CLE", "CLE": "CLE",
    "HST": "HOU", "HOU": "HOU",
    "JAC": "JAX", "JAX": "JAX",
    "KAN": "KC", "KC": "KC",
    "LVR": "LV", "LV": "LV",
    "LAR": "LA", "LA": "LA",
    "NOR": "NO", "NO": "NO",
    "NWE": "NE", "NE": "NE",
    "SFO": "SF", "SF": "SF",
    "TAM": "TB", "TB": "TB",
}

# Preserve formation labels in the snapshot, while rolling them into the
# fantasy position groups used by weekly_projections.py.  FB deliberately
# stays separate: a current roster frequently stores a fullback in the broad
# RB group, but that must not make the player eligible for a core-RB workload.
_POSITION_GROUPS = {
    "QB": "QB",
    "RB": "RB", "HB": "RB", "TB": "RB",
    "FB": "FB",
    "WR": "WR", "LWR": "WR", "RWR": "WR", "SWR": "WR",
    "XWR": "WR", "ZWR": "WR", "FL": "WR",
    "TE": "TE",
}

_SOURCE_POSITIONS = frozenset({"QB", "RB", "FB", "WR", "TE"})
_LEGACY_SKILL_POSITIONS = frozenset({"RB", "WR", "TE"})


def _source_bool(value: Any) -> bool:
    """Read bool-ish imported values without treating the string ``False`` as true."""
    if value is None or value is pd.NA:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _source_int(value: Any, default: int = 0) -> int:
    """Coerce an imported source number safely, including a manual sparse frame."""
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return int(number) if pd.notna(number) else int(default)


def _column_or_default(frame: pd.DataFrame, column: str, default: Any = "") -> pd.Series:
    """Return an index-aligned column even for a sparse manually supplied frame."""
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def _source_status(is_inactive: Any, status_class: Any = "") -> str:
    """Return a stable *source-status* label without resolving availability.

    Ourlads' red link class is documented as injured/inactive, but its saved
    depth chart has no target-week confirmation or timestamp guarantee.  The
    label remains visible for audit; it is not a workload/eligibility input.
    """
    if _source_bool(is_inactive):
        return "inactive"
    return "available"


def _source_entry_id(row: pd.Series | dict[str, Any]) -> str:
    """Stable identity for one literal Ourlads chart cell.

    `source_slot` is the displayed Player N column.  Keeping it in the key
    means a missing Player 2 can never cause a matched Player 3 to be
    renumbered when the source and current roster are compared.
    """
    get = row.get if hasattr(row, "get") else lambda _key, default="": default
    pieces = (
        _clean_text(get("team", "")).upper(),
        _clean_text(get("position_label", "")).upper(),
        str(_source_int(get("source_row", 0))),
        str(_source_int(get("source_slot", get("depth_rank", 0)))),
        _clean_text(get("player_key", "")).lower(),
    )
    return "|".join(pieces)


def _position_from_value(value: Any) -> str:
    """Map a roster/source label to the narrow role vocabulary when known."""
    return _position_group(_clean_text(value))


def _resolved_functional_position(match: pd.Series, source_position: str) -> str:
    """Classify a matched player with roster depth role taking precedence.

    The current roster's `position` field is intentionally broad in nflverse
    (for example, Alec Ingold is RB).  If its `depth_chart_position` says FB,
    that is more specific than the broad group.  If the roster lacks that
    field, an Ourlads FB row is still sufficient evidence to keep the player
    out of core RB allocation.

    An explicit Ourlads FB listing wins outright, fixed 2026-08-24 after a
    real miscall: D.J. Herman is Ourlads' literal MIA FB2 (``source_position
    == "FB"``), but the current roster's own ``depth_chart_position`` field
    still carried a stale/generic "RB" for him, and the old precedence order
    trusted that roster field unconditionally whenever it held ANY of the
    five source positions - discarding Ourlads' more specific, more current
    signal and letting a backup fullback compete for real core-RB snaps. A
    literal Ourlads FB row is a narrow, deliberate source label (unlike a
    roster's broad/possibly-stale depth field), so it is trusted first; the
    original precedence (roster depth role, then source position, then
    roster's broad position) is unchanged for every other case.
    """
    if source_position == "FB":
        return "FB"
    # Nullable pandas strings use ``pd.NA``, whose boolean value is
    # intentionally ambiguous.  Normalize before testing the current
    # roster's more-specific depth role.
    depth_role = _position_from_value(match.get("_depth_chart_position", ""))
    roster_role = _position_from_value(match.get("_roster_position", ""))
    if depth_role in _SOURCE_POSITIONS:
        return depth_role
    # The page's literal football position is more specific than nflverse's
    # broad `position` field when a current roster lacks depth-role detail.
    if source_position in _SOURCE_POSITIONS:
        return source_position
    return roster_role if roster_role in _SOURCE_POSITIONS else ""


class _HTMLTableParser(HTMLParser):
    """Tiny dependency-free HTML table reader suitable for saved MHTML pages."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[list[str]]]] = []
        self._table_stack: list[list[list[list[str]]]] = []
        self._row_stack: list[list[list[str]]] = []
        self._cell_stack: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        tag = tag.lower()
        if tag == "table":
            self._table_stack.append([])
        elif tag == "tr" and self._table_stack:
            row: list[list[str]] = []
            self._table_stack[-1].append(row)
            self._row_stack.append(row)
        elif tag in {"td", "th"} and self._row_stack:
            cell: list[str] = []
            self._row_stack[-1].append(cell)
            self._cell_stack.append(cell)
        elif tag == "br" and self._cell_stack:
            self._cell_stack[-1].append(" ")

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag in {"td", "th"} and self._cell_stack:
            self._cell_stack.pop()
        elif tag == "tr" and self._row_stack:
            self._row_stack.pop()
        elif tag == "table" and self._table_stack:
            table = self._table_stack.pop()
            if table:
                self.tables.append(table)

    def handle_data(self, data: str):
        if self._cell_stack:
            self._cell_stack[-1].append(data)


def _clean_text(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        # ``pd.isna`` can be array-like for a non-scalar.  This helper is
        # scalar-facing, so preserve a printable representation in that rare
        # case instead of asking pandas to coerce its truth value.
        pass
    return re.sub(r"\s+", " ", unescape(str(value)).replace("\xa0", " ")).strip()


def _extract_html(blob: bytes | str) -> str:
    """Return the primary HTML body from an MHTML export or an HTML file."""
    raw = blob.encode("utf-8") if isinstance(blob, str) else bytes(blob)
    try:
        message = BytesParser(policy=policy.default).parsebytes(raw)
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type().lower() != "text/html":
                    continue
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                charset = part.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="replace")
        elif message.get_content_type().lower() == "text/html":
            payload = message.get_payload(decode=True)
            if payload:
                return payload.decode(message.get_content_charset() or "utf-8", errors="replace")
    except Exception:
        # Browser "save as HTML" exports are not MIME messages.  The plain
        # decode fallback below still handles them.
        pass
    return raw.decode("utf-8", errors="replace")


def _normalize_team(value: str) -> str:
    key = _clean_text(value).upper()
    if key in TEAM_CONFIG:
        return key
    if key in _OURLADS_TEAM_ALIASES:
        return _OURLADS_TEAM_ALIASES[key]
    if key in PFF_TEAM_CODE_TO_ABBR:
        return PFF_TEAM_CODE_TO_ABBR[key]
    for abbr, config in TEAM_CONFIG.items():
        if key == str(config.get("name", "")).upper():
            return abbr
    return ""


def _extract_team(html: str) -> tuple[str, str]:
    source_url = ""
    url_match = re.search(r"https?://[^\s\"'>]*pfdepthchart/([A-Za-z]{2,4})", html, flags=re.I)
    if url_match:
        source_url = url_match.group(0)
        team = _normalize_team(url_match.group(1))
        if team:
            return team, source_url
    title_match = re.search(r"<title[^>]*>\s*([^<]+?)\s+Depth\s+Chart", html, flags=re.I)
    if title_match:
        team = _normalize_team(title_match.group(1))
        if team:
            return team, source_url
    # The visible page heading is a useful fallback when a saved browser
    # page has rewritten/removed the Content-Location MIME header.
    heading_match = re.search(r">\s*([^<]+?)\s+Depth\s+Chart\s*<", html, flags=re.I)
    if heading_match:
        return _normalize_team(heading_match.group(1)), source_url
    return "", source_url


def _extract_updated_at(html: str) -> str:
    match = re.search(r"Updated\s*:\s*([^<\r\n]+)", html, flags=re.I)
    return _clean_text(match.group(1)) if match else ""


def _table_rows(html: str) -> list[list[str]]:
    parser = _HTMLTableParser()
    parser.feed(html)
    parser.close()
    best: list[list[list[str]]] | None = None
    best_score = -1
    for table in parser.tables:
        for row in table[:5]:
            values = [_clean_text("".join(cell)) for cell in row]
            normalized = [value.lower() for value in values]
            player_count = sum(value.startswith("player") for value in normalized)
            score = (20 if any(value == "pos" for value in normalized) else 0) + player_count * 5 + len(table)
            if player_count and score > best_score:
                best, best_score = table, score
    if best is None:
        return []
    return [[_clean_text("".join(cell)) for cell in row] for row in best]


def _player_column_indexes(header: list[str]) -> list[tuple[int, int]]:
    columns = []
    for index, value in enumerate(header):
        match = re.fullmatch(r"player\s*(\d+)", _clean_text(value).lower())
        if match:
            columns.append((index, int(match.group(1))))
    return columns


def _strip_player_metadata(raw_player: str) -> str:
    """Remove Ourlads' draft/acquisition suffix while retaining name suffixes."""
    value = _clean_text(raw_player)
    # Examples from printer-friendly pages: ``Harrison Jr., Marvin 24/1``,
    # ``BRISSETT, JACOBY U/NE``, ``Weaver, Xavier CF24``.  Do not interpret
    # that suffix as a football role; it is provenance/transaction metadata.
    value = re.sub(
        r"\s+(?:\d{2}/\d+|(?:U|W|P|T|CC)/[A-Za-z]{2,4}|(?:CF|SF)\d{2}\*?)\s*$",
        "", value, flags=re.I,
    )
    value = value.strip(" *")
    if "," in value:
        last, first = value.split(",", 1)
        value = f"{first.strip()} {last.strip()}"
    return _clean_text(value)


def _position_group(label: str) -> str:
    normalized = re.sub(r"[^A-Z]", "", _clean_text(label).upper())
    return _POSITION_GROUPS.get(normalized, "")


def _attribute(attrs: str, name: str) -> str:
    match = re.search(
        rf"\b{re.escape(name)}\s*=\s*(['\"])(.*?)\1", attrs,
        flags=re.I | re.S,
    )
    return _clean_text(match.group(2)) if match else ""


def _anchor_metadata(html: str) -> dict[str, list[dict[str, str]]]:
    """Map displayed Ourlads player text to link id / availability class.

    The printer-friendly table puts ``lc_red`` on the player ``<a>`` tag,
    not the cell or row.  The Ourlads legend describes it as injured/inactive,
    so treating it as an ordinary first-string role would be demonstrably
    wrong.  Other color classes are retained as provenance but deliberately
    do not alter projected usage.
    """
    metadata: dict[str, list[dict[str, str]]] = {}
    for match in re.finditer(r"<a\b([^>]*)>(.*?)</a\s*>", html, flags=re.I | re.S):
        attrs, body = match.groups()
        href = _attribute(attrs, "href")
        player_id = re.search(r"/player/(\d+)", href)
        if not player_id:
            continue
        raw = _clean_text(re.sub(r"<[^>]+>", " ", body))
        if not raw:
            continue
        metadata.setdefault(raw, []).append({
            "status_class": _attribute(attrs, "class").lower(),
            "source_player_id": player_id.group(1),
        })
    return metadata


def parse_ourlads_depth_chart(blob: bytes | str, source_file: str = "") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse one saved printer-friendly team page into normalized offense rows.

    The returned diagnostics are intentionally specific: an unrecognized
    file should be visible to the importer UI rather than silently yielding a
    partial league snapshot.
    """
    html = _extract_html(blob)
    team, source_url = _extract_team(html)
    diagnostics: dict[str, Any] = {
        "source_file": str(source_file or "(unnamed file)"),
        "team": team,
        "error": "",
        "rows": 0,
    }
    if not team:
        diagnostics["error"] = "could not identify an NFL team from the saved page"
        return pd.DataFrame(columns=OURLADS_COLUMNS), diagnostics

    rows = _table_rows(html)
    if not rows:
        diagnostics["error"] = "could not find the printer-friendly depth-chart table"
        return pd.DataFrame(columns=OURLADS_COLUMNS), diagnostics

    header_index = None
    player_columns: list[tuple[int, int]] = []
    for index, row in enumerate(rows[:8]):
        player_columns = _player_column_indexes(row)
        if player_columns and any(_clean_text(value).lower() == "pos" for value in row):
            header_index = index
            break
    if header_index is None:
        diagnostics["error"] = "table did not include Pos / Player columns"
        return pd.DataFrame(columns=OURLADS_COLUMNS), diagnostics

    # The printer-friendly page exposes offense and defense as separate
    # tables, so the selected Pos/Player table often has no literal
    # "Offense" separator row.  Start in offense and still honor a marker if
    # an export includes one.
    section = "offense"
    output = []
    updated_at = _extract_updated_at(html)
    anchors = _anchor_metadata(html)
    position_occurrences: dict[str, int] = {}
    for source_row, raw_row in enumerate(rows[header_index + 1:], start=1):
        if not raw_row:
            continue
        first = _clean_text(raw_row[0] if raw_row else "")
        first_key = first.lower()
        if first_key in {"offense", "defense", "special teams", "reserves"}:
            section = first_key
            continue
        position_label = re.sub(r"\s+", "", first.upper())
        position = _position_group(position_label)
        if section != "offense" or not position:
            continue
        position_occurrence = position_occurrences.get(position_label, 0)
        position_occurrences[position_label] = position_occurrence + 1
        for col_index, depth_rank in player_columns:
            raw_player = _clean_text(raw_row[col_index] if col_index < len(raw_row) else "")
            player = _strip_player_metadata(raw_player)
            if not player or not re.search(r"[A-Za-z]", player):
                continue
            link_meta = (anchors.get(raw_player) or [{}])[0]
            status_class = _clean_text(link_meta.get("status_class"))
            output.append({
                "year": pd.NA,
                "team": team,
                "unit": "Offense",
                "position_label": position_label,
                "position": position,
                "depth_rank": int(depth_rank),
                "source_row": int(source_row),
                "source_slot": int(depth_rank),
                "position_occurrence": int(position_occurrence),
                # A second rendered RB/TE row is overflow, not a second
                # starter at the same position.  LWR/RWR/SWR are separate
                # formation labels, so each first row still carries its own
                # starter signal.
                "is_listed_starter": bool(depth_rank == 1 and position_occurrence == 0),
                "is_inactive": "lc_red" in status_class.split(),
                "status_class": status_class,
                "source_player_id": _clean_text(link_meta.get("source_player_id")),
                "player": player,
                "player_key": clean_name_exact(pd.Series([player])).iloc[0],
                "raw_player": raw_player,
                "source_updated_at": updated_at,
                "source_file": _clean_text(source_file),
                "source_url": source_url,
                "parser_version": PARSER_VERSION,
            })
    frame = pd.DataFrame(output, columns=OURLADS_COLUMNS)
    if frame.empty:
        diagnostics["error"] = "page contained no QB/RB/FB/WR/TE offense rows"
    diagnostics["rows"] = len(frame)
    return frame, diagnostics


def _file_blob(upload: Any) -> tuple[str, bytes]:
    """Accept Streamlit uploads, (name, bytes) fixtures, or a path-like input."""
    if isinstance(upload, tuple) and len(upload) == 2:
        return str(upload[0]), bytes(upload[1])
    if isinstance(upload, (str, Path)):
        path = Path(upload)
        return path.name, path.read_bytes()
    name = str(getattr(upload, "name", "(unnamed file)"))
    if hasattr(upload, "getvalue"):
        return name, bytes(upload.getvalue())
    if hasattr(upload, "read"):
        try:
            upload.seek(0)
        except Exception:
            pass
        return name, bytes(upload.read())
    raise TypeError("Expected a saved depth-chart file, upload, or (name, bytes) pair.")


def build_ourlads_snapshot(files: Iterable[Any], year: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize a collection of locally saved Ourlads team pages.

    Duplicate team pages are resolved deterministically by their displayed
    source-update timestamp and then filename.  The report keeps the duplicate
    visible so a stale browser export cannot quietly win.
    """
    frames, diagnostics = [], []
    for item in files or []:
        try:
            source_file, blob = _file_blob(item)
            frame, diagnostic = parse_ourlads_depth_chart(blob, source_file)
        except Exception as exc:
            frame = pd.DataFrame(columns=OURLADS_COLUMNS)
            diagnostic = {"source_file": str(getattr(item, "name", item)), "team": "", "rows": 0,
                          "error": f"could not read file: {exc}"}
        diagnostics.append(diagnostic)
        if not frame.empty:
            frame["year"] = int(year)
            frames.append(frame)
    snapshot = (pd.concat(frames, ignore_index=True) if frames
                else pd.DataFrame(columns=OURLADS_COLUMNS))
    duplicate_teams: list[str] = []
    if not snapshot.empty:
        page_order = (
            snapshot[["team", "source_file", "source_updated_at"]].drop_duplicates()
            .assign(_updated=lambda x: pd.to_datetime(
                x["source_updated_at"].astype(str).str.replace(r"\s+ET$", "", regex=True),
                format="%m/%d/%Y %I:%M%p", errors="coerce"))
            .sort_values(["team", "_updated", "source_file"], kind="stable")
        )
        keep_files = page_order.groupby("team", observed=True).tail(1)[["team", "source_file"]]
        duplicate_teams = (page_order[page_order.duplicated("team", keep=False)]["team"]
                           .drop_duplicates().sort_values().tolist())
        snapshot = snapshot.merge(keep_files, on=["team", "source_file"], how="inner")
        snapshot = snapshot.drop_duplicates(
            subset=["team", "position_label", "source_row", "source_slot", "player", "source_file"], keep="first")
        snapshot = snapshot.sort_values(
            ["team", "position", "position_label", "source_row", "source_slot", "player"],
            kind="stable").reset_index(drop=True)
    parsed_teams = sorted(snapshot["team"].dropna().astype(str).unique().tolist()) if not snapshot.empty else []
    report = {
        "diagnostics": diagnostics,
        "files_received": len(diagnostics),
        "files_parsed": int(sum(1 for item in diagnostics if not item.get("error"))),
        "teams": parsed_teams,
        "team_count": len(parsed_teams),
        "missing_teams": sorted(set(TEAM_CONFIG) - set(parsed_teams)),
        "duplicate_teams": duplicate_teams,
        "unreadable_files": [item for item in diagnostics if item.get("error")],
    }
    return snapshot.reindex(columns=OURLADS_COLUMNS), report


def _archive_ourlads_import(files: Iterable[Any], existing_csv: Path,
                            archive_root: Path | None = None) -> dict[str, Any]:
    """Copy the OUTGOING snapshot csv and the raw pages being consumed into a
    timestamped archive folder, so a past-week analysis can pin the exact
    chart the model saw. Best-effort: a failure here never blocks the import.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out: dict[str, Any] = {"archived_csv": None, "archived_pages": 0, "archive_dir": None}
    try:
        root = OURLADS_ARCHIVE_DIR if archive_root is None else archive_root
        batch_dir = Path(root) / f"import_{stamp}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        out["archive_dir"] = str(batch_dir)
        existing_csv = Path(existing_csv)
        if existing_csv.is_file() and existing_csv.stat().st_size > 0:
            prev_stamp = datetime.fromtimestamp(
                existing_csv.stat().st_mtime).strftime("%Y%m%d-%H%M%S")
            dest = batch_dir / f"ourlads_depth_charts_{prev_stamp}.csv"
            shutil.copy2(existing_csv, dest)
            out["archived_csv"] = str(dest)
        pages_dir = batch_dir / "pages"
        for item in files or []:
            try:
                name, blob = _file_blob(item)
            except Exception:
                continue
            safe = Path(str(name)).name or f"page_{out['archived_pages'] + 1}.mhtml"
            pages_dir.mkdir(parents=True, exist_ok=True)
            (pages_dir / safe).write_bytes(blob)
            out["archived_pages"] += 1
        # A batch with nothing to keep leaves no empty folder behind.
        if out["archived_csv"] is None and out["archived_pages"] == 0:
            try:
                batch_dir.rmdir()
                out["archive_dir"] = None
            except OSError:
                pass
    except Exception as exc:  # pragma: no cover - archiving is never fatal
        out["error"] = f"could not archive prior Ourlads import: {exc}"
    return out


def save_ourlads_snapshot(files: Iterable[Any], year: int,
                          path: str | Path = OURLADS_IMPORT_PATH,
                          archive: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build and persist a local normalized snapshot.

    ``archive`` (default on): before overwriting the snapshot csv, copy the
    current csv and the raw pages being imported into a timestamped folder
    under ``OURLADS_ARCHIVE_DIR`` (see ``_archive_ourlads_import``). The raw
    pages themselves are still never written to the live import path.
    """
    snapshot, report = build_ourlads_snapshot(files, year)
    if snapshot.empty:
        report["error"] = "No readable QB/RB/FB/WR/TE Ourlads team chart was imported."
        return snapshot, report
    target = Path(path)
    if archive:
        report["archive"] = _archive_ourlads_import(files, target)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot.to_csv(target, index=False)
    except Exception as exc:
        report["error"] = f"Parsed the depth charts but could not save the local snapshot: {exc}"
        return snapshot, report
    report["error"] = ""
    report["path"] = str(target)
    return snapshot, report


def save_ourlads_snapshot_from_dir(directory: str | Path = OURLADS_INBOX_DIR,
                                   year: int | None = None,
                                   path: str | Path = OURLADS_IMPORT_PATH,
                                   archive: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Import every saved Ourlads page in ``directory`` (default the inbox).

    Returns the same ``(snapshot, report)`` as ``save_ourlads_snapshot``; the
    report also carries ``source_dir`` and the sorted ``source_files`` list.
    ``year`` defaults to the current NFL season.
    """
    directory = Path(directory)
    if year is None:
        year = _current_nfl_season()
    pages = sorted(
        p for p in directory.glob("*")
        if p.is_file() and p.suffix.lower() in _OURLADS_PAGE_SUFFIXES)
    if not pages:
        empty = pd.DataFrame(columns=OURLADS_COLUMNS)
        return empty, {"error": f"No .mhtml/.mht/.html pages found in {directory}.",
                       "source_dir": str(directory), "source_files": []}
    snapshot, report = save_ourlads_snapshot(pages, int(year), path=path, archive=archive)
    report["source_dir"] = str(directory)
    report["source_files"] = [p.name for p in pages]
    return snapshot, report


def _finalize_snapshot_frame(frame: pd.DataFrame, year: int | None) -> pd.DataFrame:
    """Shared normalization for a raw OURLADS_COLUMNS frame (live or historical)."""
    optional_ids = [column for column in _OPTIONAL_SOURCE_ID_COLUMNS if column in frame.columns]
    frame = frame.loc[:, list(OURLADS_COLUMNS) + optional_ids].copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    if year is not None:
        frame = frame[frame["year"].eq(int(year))].copy()
    frame["team"] = frame["team"].map(_normalize_team)
    frame["position"] = frame["position"].astype(str).str.upper()
    frame["depth_rank"] = pd.to_numeric(frame["depth_rank"], errors="coerce")
    frame["is_listed_starter"] = frame["is_listed_starter"].astype(str).str.lower().isin(
        {"1", "true", "t", "yes", "y"})
    frame["is_inactive"] = frame["is_inactive"].astype(str).str.lower().isin(
        {"1", "true", "t", "yes", "y"})
    frame["status_class"] = frame["status_class"].fillna("").astype(str).str.strip().str.lower()
    frame["source_player_id"] = frame["source_player_id"].fillna("").astype(str).str.strip()
    frame["source_row"] = pd.to_numeric(frame["source_row"], errors="coerce").fillna(0).astype(int)
    frame["source_slot"] = pd.to_numeric(frame["source_slot"], errors="coerce").fillna(
        frame["depth_rank"]).fillna(0).astype(int)
    frame["position_occurrence"] = pd.to_numeric(
        frame["position_occurrence"], errors="coerce").fillna(0).astype(int)
    frame["player"] = frame["player"].astype(str).str.strip()
    frame["player_key"] = clean_name_exact(frame["player"])
    frame = frame[(frame["team"] != "") & frame["position"].isin(_SOURCE_POSITIONS)
                  & frame["player"].ne("")].copy()
    return frame.reset_index(drop=True)


def _historical_snapshot(year: int, path: str | Path = OURLADS_HISTORY_PATH) -> tuple[pd.DataFrame, str | None]:
    """Adapt the frozen pre-Week-1 archive CSV to an OURLADS_COLUMNS snapshot.

    Archive rows carry no injury colour and no continuation ("second unit")
    listings, so ``is_inactive``/``status_class`` are blank and
    ``position_occurrence`` is 0 for every row - a known, documented gap of
    this source relative to the live printer-friendly import.
    """
    target = Path(path)
    empty = pd.DataFrame(columns=OURLADS_COLUMNS)
    if not target.exists():
        return empty, None
    try:
        hist = pd.read_csv(target, dtype=str)
    except Exception as exc:
        return empty, f"Could not read {target.name}: {exc}"
    hist = hist[pd.to_numeric(hist["year"], errors="coerce").eq(int(year))].copy()
    if hist.empty:
        return empty, None
    slot = hist["slot"].astype(str).str.upper().str.strip()
    out = pd.DataFrame(index=hist.index)
    out["year"] = pd.to_numeric(hist["year"], errors="coerce")
    out["team"] = hist["team"].astype(str)
    out["unit"] = hist["unit"].map(_HISTORY_UNIT_TO_LABEL).fillna(hist["unit"])
    out["position_label"] = slot
    out["position"] = slot.map(_HISTORY_SLOT_TO_POSITION).fillna(slot)
    out["depth_rank"] = pd.to_numeric(hist["slot_rank"], errors="coerce")
    out["source_row"] = 0
    out["source_slot"] = pd.to_numeric(hist["slot_rank"], errors="coerce")
    out["position_occurrence"] = 0
    out["is_listed_starter"] = pd.to_numeric(hist["slot_rank"], errors="coerce").eq(1)
    out["is_inactive"] = False
    out["status_class"] = ""
    out["source_player_id"] = ""
    out["player"] = (hist["player_first"].fillna("").astype(str).str.strip() + " "
                     + hist["player_last"].fillna("").astype(str).str.strip()).str.strip()
    out["player_key"] = clean_name_exact(out["player"])
    out["raw_player"] = hist.get("player_raw", out["player"])
    out["source_updated_at"] = hist.get("last_updated", "")
    out["source_file"] = target.name
    out["source_url"] = ""
    out["parser_version"] = "historical-archive"
    return _finalize_snapshot_frame(out, int(year)), None


def load_ourlads_snapshot(year: int | None = None,
                          path: str | Path = OURLADS_IMPORT_PATH,
                          allow_historical: bool = False) -> tuple[pd.DataFrame, str | None]:
    """Load an already-normalized local snapshot and validate its schema.

    With ``allow_historical=True`` (the weekly model passes this only under
    ``v2_historical_ourlads``), a past ``year`` with no rows in the live
    snapshot falls back to the frozen pre-Week-1 archive
    (``OURLADS_HISTORY_PATH``) - a time-valid ~09/01 chart, so a cold-start
    backtest can see what the model would have had at kickoff. Default off
    keeps every existing backtest and the live path untouched.
    """
    empty = pd.DataFrame(columns=OURLADS_COLUMNS)

    def _fallback():
        if allow_historical and year is not None:
            return _historical_snapshot(year)
        return empty, None

    target = Path(path)
    if not target.exists():
        return _fallback()
    try:
        frame = pd.read_csv(target, dtype={"team": str, "player": str, "position": str,
                                            "position_label": str, "source_file": str})
    except Exception as exc:
        return empty, f"Could not read {target.name}: {exc}"
    missing = [column for column in OURLADS_COLUMNS if column not in frame.columns]
    if missing:
        return empty, f"{target.name} is missing required column(s): {', '.join(missing)}."
    finalized = _finalize_snapshot_frame(frame, year)
    if finalized.empty and year is not None:
        return _fallback()
    return finalized, None


_IDENTITY_MATCH_SPECS = (
    # Ourlads' own ``source_player_id`` is deliberately absent.  It is a web
    # page id, not a GSIS/PFF/player identity, and treating it as one could
    # silently attach an unrelated player with a coincidentally similar id.
    ("GSIS ID", ("gsis_id", "source_gsis_id", "nflverse_id", "source_nflverse_id"),
     ("gsis_id", "nflverse_id", "player_id")),
    ("player ID", ("player_id", "source_player_identity_id"),
     ("player_id", "gsis_id")),
    ("PFF ID", ("pff_id", "source_pff_id", "pff_player_id", "source_pff_player_id"),
     ("pff_id", "pff_player_id")),
    ("ESPN ID", ("espn_id", "source_espn_id"), ("espn_id",)),
    ("PFR ID", ("pfr_id", "source_pfr_id"), ("pfr_id",)),
    ("Sportradar ID", ("sportradar_id", "source_sportradar_id"), ("sportradar_id",)),
)


def _frame_column(frame: pd.DataFrame, column: str, default: Any = "") -> pd.Series:
    """Read a possibly duplicated CSV column as one deterministic Series."""
    if column not in frame.columns:
        return pd.Series(default, index=frame.index)
    value = frame.loc[:, frame.columns == column]
    return value.iloc[:, 0] if isinstance(value, pd.DataFrame) else value


def _strict_name_key(values) -> pd.Series:
    """Normalize a complete display name without applying any alias table.

    ``data.utils.clean_name_exact`` intentionally includes the app-wide
    reviewed aliases.  That is useful for ordinary joins, but this resolver
    needs to report the hierarchy honestly: raw full-name equality first,
    reviewed aliases second, then a unique suffix-stripped bridge.
    """
    series = pd.Series(values)
    series = series.astype(object).where(series.notna(), "").astype(str)
    series = series.map(_clean_text).str.normalize("NFKD")
    series = series.str.encode("ascii", "ignore").str.decode("ascii")
    return series.str.lower().str.replace(r"[^a-z]", "", regex=True)


def _row_identifier(row: pd.Series, columns: tuple[str, ...]) -> str:
    """Return the first real identifier in a declared source-id family."""
    for column in columns:
        if column not in row.index:
            continue
        value = normalize_player_identifier(pd.Series([row[column]])).iloc[0]
        if value:
            return str(value)
    return ""


def _build_roster_identity_pool(roster: pd.DataFrame, name_col: str,
                                team_col: str) -> pd.DataFrame:
    """Prepare one identity-preserving pool for every Ourlads join path."""
    # Use a private positional locator rather than relying on an index label
    # to be unique; a literal ``index`` column is valid user data.
    pool = roster.copy()
    pool["_roster_row"] = range(len(pool))
    pool = pool.dropna(subset=[name_col]).reset_index(drop=True)
    pool["_team"] = _frame_column(pool, team_col).map(_normalize_team)
    pool["_roster_position"] = _frame_column(pool, "position").map(_position_from_value)
    pool["_depth_chart_position"] = _frame_column(pool, "depth_chart_position").map(_position_from_value)
    pool["_player"] = _frame_column(pool, name_col).astype(str).str.strip()
    pool["_exact"] = _strict_name_key(pool["_player"])
    pool["_canonical_key"] = clean_name_exact(pool["_player"])
    pool["_alias_exact"] = canonical_player_key(pool["_player"])
    pool["_loose"] = clean_name_for_merge(pool["_player"])
    pool["_identity_key"] = stable_roster_identity_keys(pool, name_col)
    for column in STABLE_PLAYER_ID_COLUMNS:
        pool[f"_id_{column}"] = normalize_player_identifier(_frame_column(pool, column))
    # Keep a position-compatible row even if its current team text is stale or
    # nonstandard: a true GSIS/PFF id can still repair that assignment.  Name
    # tiers below continue to require a normalized same-team candidate.
    pool = pool[pool["_roster_position"].isin(_SOURCE_POSITIONS)].copy()
    # Multiple feed rows with the same GSIS/PFF identity are representations
    # of one player, not competing identities.  Keep a deterministic one;
    # different stable identities remain separate and therefore ambiguous.
    pool = pool.drop_duplicates(subset=["_identity_key"], keep="first")
    return pool.reset_index(drop=True)


def _position_compatible(candidates: pd.DataFrame, source_position: str) -> pd.Series:
    """Return the compatible football-role mask, including RB/FB bridging."""
    compatible = (candidates["_roster_position"].eq(source_position)
                  | candidates["_depth_chart_position"].eq(source_position))
    if source_position in {"RB", "FB"}:
        compatible = compatible | candidates["_roster_position"].isin({"RB", "FB"})
    return compatible


# Single-slot, object-identity cache (see _alignment_profile_index in
# pff_alignment.py for the same pattern, including why identity + a strong
# reference beats a raw id()-keyed cache). _unique_match's very first line
# is _position_compatible(pool, source_position) - a full-pool boolean scan,
# run once per Ourlads depth-chart row, every row, always against the SAME
# `pool` object (built once per resolve_ourlads_roster_identities call) and
# one of only 5 possible values (_SOURCE_POSITIONS). Deliberately NOT
# applied inside _position_compatible itself: _unique_match's SECOND call
# (`_position_compatible(id_pool, source_position)`) passes a fresh, tiny,
# already-ID-filtered frame every row, so a cache keyed only on
# `(pool_identity, source_position)` at that call site would thrash on
# every row instead of ever hitting - this wrapper is used only at the
# full-pool call site, where the object genuinely repeats.
_POSITION_COMPATIBLE_CACHE: dict[str, Any] = {"pool": None, "results": {}}


def _position_compatible_for_pool(pool: pd.DataFrame, source_position: str) -> pd.Series:
    cache = _POSITION_COMPATIBLE_CACHE
    if cache["pool"] is not pool:
        cache["pool"] = pool
        cache["results"] = {}
    results = cache["results"]
    if source_position not in results:
        results[source_position] = _position_compatible(pool, source_position)
    return results[source_position]


def _distinct_identity_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate rows for one identity without collapsing people."""
    if candidates.empty:
        return candidates
    return candidates.drop_duplicates(subset=["_identity_key"], keep="first")


def _source_stable_id_values(chart_row: pd.Series) -> list[tuple[str, str, tuple[str, ...]]]:
    values: list[tuple[str, str, tuple[str, ...]]] = []
    for method, source_columns, roster_columns in _IDENTITY_MATCH_SPECS:
        value = _row_identifier(chart_row, source_columns)
        if value:
            values.append((method, value, roster_columns))
    return values


def _id_candidates(pool: pd.DataFrame, value: str,
                   roster_columns: tuple[str, ...]) -> pd.DataFrame:
    masks = []
    for column in roster_columns:
        pool_column = f"_id_{column}"
        if pool_column in pool.columns:
            masks.append(pool[pool_column].eq(value))
    if not masks:
        return pool.iloc[0:0]
    mask = masks[0].copy()
    for item in masks[1:]:
        mask = mask | item
    return pool[mask]


def _unique_match(pool: pd.DataFrame, chart_row: pd.Series,
                  *, allow_cross_team_name: bool = False) -> tuple[pd.Series | None, str, str, str]:
    """Resolve one source row through the authoritative safe match hierarchy.

    Order is intentionally fixed: stable ids, exact full name within the
    source team/position, reviewed alias, then an *unambiguous* suffix-stripped
    full name.  There is no last-name-only path.  Stable IDs may bridge a
    stale team assignment, but still require a compatible football position.
    """
    source_position = _position_from_value(chart_row.get("position", ""))
    compatible = _position_compatible_for_pool(pool, source_position)

    id_matches: list[tuple[str, pd.DataFrame]] = []
    for method, value, roster_columns in _source_stable_id_values(chart_row):
        id_pool = _id_candidates(pool, value, roster_columns)
        candidates = _distinct_identity_candidates(
            id_pool[_position_compatible(id_pool, source_position)])
        if len(candidates) == 1:
            id_matches.append((method, candidates))
        elif len(candidates) > 1:
            return None, "ambiguous stable ID", "none", (
                f"{method} '{value}' maps to multiple compatible roster identities")
    if id_matches:
        # Two supplied source IDs that resolve to different people are an
        # explicit conflict, not a cue to choose whichever check ran first.
        ids = {str(rows.iloc[0]["_identity_key"]) for _, rows in id_matches}
        if len(ids) != 1:
            return None, "conflicting stable IDs", "none", (
                "source stable identifiers resolve to different roster identities")
        method, candidates = id_matches[0]
        return candidates.iloc[0], method, "high", ""

    if allow_cross_team_name:
        candidates = pool[compatible].copy()
    else:
        candidates = pool[pool["_team"].eq(_normalize_team(chart_row.get("team", ""))) & compatible].copy()
    candidates = _distinct_identity_candidates(candidates)
    if candidates.empty:
        return None, "unmatched", "none", "no compatible roster player"

    source_name = _clean_text(chart_row.get("player", ""))
    # Prefer resolve_ourlads_roster_identities' precomputed columns (see its
    # own comment) - fall back to computing it here directly so this
    # function still works correctly if ever called with a bare chart_row
    # that doesn't carry them (e.g. a future direct/standalone caller).
    if "_source_exact_key" in chart_row.index:
        source_exact = chart_row["_source_exact_key"]
    else:
        source_exact = _strict_name_key(pd.Series([source_name])).iloc[0]
    exact = _distinct_identity_candidates(candidates[candidates["_exact"].eq(source_exact)])
    if len(exact) == 1:
        return exact.iloc[0], "exact name", "high", ""
    if len(exact) > 1:
        return None, "ambiguous exact name", "none", (
            "exact normalized name maps to multiple compatible roster identities")

    # A reviewable alias is deliberately narrower than the suffix fallback.
    # It bridges source-only spelling changes without declaring similar names
    # interchangeable players.
    if "_source_alias_key" in chart_row.index:
        alias_key = chart_row["_source_alias_key"]
    else:
        alias_key = canonical_player_key(pd.Series([source_name])).iloc[0]
    alias = _distinct_identity_candidates(candidates[candidates["_alias_exact"].eq(alias_key)])
    if len(alias) == 1:
        return alias.iloc[0], "reviewed alias", "reviewed", ""
    if len(alias) > 1:
        return None, "ambiguous reviewed alias", "none", (
            "reviewed alias maps to multiple compatible roster identities")

    if "_source_loose_key" in chart_row.index:
        loose_key = chart_row["_source_loose_key"]
    else:
        loose_key = clean_name_for_merge(pd.Series([source_name])).iloc[0]
    loose = _distinct_identity_candidates(candidates[candidates["_loose"].eq(loose_key)])
    if len(loose) == 1:
        return loose.iloc[0], "suffix-stripped name", "medium", ""
    if len(loose) > 1:
        return None, "ambiguous suffix-stripped name", "none", (
            "suffix-stripped full name maps to multiple compatible roster identities")
    return None, "unmatched", "none", "no unique same-team, same-position identity match"


_MATCH_COLUMNS = [
    "team", "position", "functional_position", "source_position", "roster_position",
    "roster_depth_chart_position", "position_label", "depth_rank", "source_depth_rank",
    "source_rank", "source_row", "source_slot", "position_occurrence",
    "source_position_occurrence", "is_listed_starter", "source_is_listed_starter",
    "is_inactive", "source_is_inactive", "status_class", "source_status_class",
    "source_status", "source_status_warning", "source_player_id", "source_gsis_id",
    "source_pff_id", "source_entry_id", "player", "player_key", "matched_player",
    "matched_player_key", "matched_player_id", "matched_gsis_id", "matched_pff_id",
    "matched_identity_key", "matched_roster_row", "match_method", "match_confidence",
    "match_warning",
]

_IDENTITY_AUDIT_COLUMNS = [
    "source_entry_id", "source_name", "source_player_key", "source_team", "source_position",
    "source_player_id", "source_gsis_id", "source_pff_id", "roster_name",
    "roster_player_id", "roster_gsis_id", "roster_pff_id", "roster_identity_key",
    "match_method", "match_confidence", "match_status", "match_warning",
]


def _match_audit_row(chart_row: pd.Series, match: pd.Series | None,
                     method: str, confidence: str, warning: str) -> dict[str, str]:
    status = "matched" if match is not None else (
        "ambiguous" if method.startswith("ambiguous") or method.startswith("conflicting") else "unmatched")
    return {
        "source_entry_id": _source_entry_id(chart_row),
        "source_name": _clean_text(chart_row.get("player", "")),
        "source_player_key": _clean_text(chart_row.get("player_key", "")),
        "source_team": _normalize_team(chart_row.get("team", "")),
        "source_position": _position_from_value(chart_row.get("position", "")),
        "source_player_id": _clean_text(chart_row.get("source_player_id", "")),
        "source_gsis_id": _row_identifier(chart_row, ("gsis_id", "source_gsis_id", "nflverse_id", "source_nflverse_id")),
        "source_pff_id": _row_identifier(chart_row, ("pff_id", "source_pff_id", "pff_player_id", "source_pff_player_id")),
        "roster_name": _clean_text(match.get("_player", "")) if match is not None else "",
        "roster_player_id": _clean_text(match.get("_id_player_id", "")) if match is not None else "",
        "roster_gsis_id": _clean_text(match.get("_id_gsis_id", "")) if match is not None else "",
        "roster_pff_id": _clean_text(match.get("_id_pff_id", "")) if match is not None else "",
        "roster_identity_key": _clean_text(match.get("_identity_key", "")) if match is not None else "",
        "match_method": method,
        "match_confidence": confidence,
        "match_status": status,
        "match_warning": warning,
    }


def resolve_ourlads_roster_identities(snapshot: pd.DataFrame, roster: pd.DataFrame,
                                      name_col: str, team_col: str, *,
                                      allow_cross_team_name: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Resolve literal Ourlads rows to roster identities and a full audit ledger.

    This is the single authoritative source-to-roster resolver.  Both the
    cold-start overlay and downstream Ourlads role signal use it, preventing
    one path from accepting an exact spelling while another appends a suffix
    variant of the same active player.  ``allow_cross_team_name`` is only for
    verifying a truly missing preseason starter against historical evidence;
    it still requires a unique full-name/position match and never uses a last
    name alone.
    """
    if (snapshot is None or snapshot.empty or roster is None or roster.empty
            or not {name_col, team_col, "position"}.issubset(roster.columns)):
        return (pd.DataFrame(columns=_MATCH_COLUMNS),
                pd.DataFrame(columns=_IDENTITY_AUDIT_COLUMNS), [])
    pool = _build_roster_identity_pool(roster, name_col, team_col)
    # Precompute the three name-tier keys ONCE, vectorized over the whole
    # snapshot column - _unique_match picks these up straight off chart_row
    # below (see its own fallback) instead of recomputing them by calling
    # _strict_name_key/canonical_player_key/clean_name_for_merge on a
    # freshly-built ONE-ROW pd.Series every single source row. Those three
    # are vectorized string functions (.str.normalize/.encode/.decode/...);
    # their per-call overhead is roughly constant regardless of row count,
    # so calling them once per Ourlads row instead of once for the whole
    # column was paying that same fixed cost hundreds of times over -
    # profiled at ~19s of Ourlads identity resolution just for
    # _strict_name_key's share. Exact same normalization, same order
    # (_clean_text first, then each key function) as the per-row path.
    snapshot = snapshot.copy()
    _source_names = snapshot.get("player", pd.Series("", index=snapshot.index)).map(_clean_text)
    snapshot["_source_exact_key"] = _strict_name_key(_source_names)
    snapshot["_source_alias_key"] = canonical_player_key(_source_names)
    snapshot["_source_loose_key"] = clean_name_for_merge(_source_names)
    resolved, audit, warnings = [], [], []
    for _, chart_row in snapshot.iterrows():
        source_position = _position_from_value(chart_row.get("position", ""))
        if source_position not in _SOURCE_POSITIONS:
            continue
        match, method, confidence, warning = _unique_match(
            pool, chart_row, allow_cross_team_name=allow_cross_team_name)
        audit_row = _match_audit_row(chart_row, match, method, confidence, warning)
        audit.append(audit_row)
        inactive = _source_bool(chart_row.get("is_inactive", False))
        status_warning = (
            "Ourlads lc_red status is an unconfirmed source flag; current availability must confirm it."
            if inactive else "")
        if status_warning:
            warnings.append(
                f"{_normalize_team(chart_row.get('team', ''))} {chart_row.get('position_label', '')} "
                f"'{chart_row.get('player', '')}' has an Ourlads lc_red status flag; role remains conditional pending current availability.")
        if match is None:
            if audit_row["match_status"] == "ambiguous":
                warnings.append(
                    f"{chart_row.get('team', '')} {chart_row.get('position_label', '')} "
                    f"'{chart_row.get('player', '')}' identity is ambiguous: {warning}.")
            elif _source_bool(chart_row.get("is_listed_starter", False)):
                warnings.append(
                    f"{chart_row.get('team', '')} {chart_row.get('position_label', '')} starter "
                    f"'{chart_row.get('player', '')}' could not be uniquely matched to the current roster.")
            continue
        functional_position = _resolved_functional_position(match, source_position)
        depth_rank = _source_int(chart_row.get("depth_rank", 0))
        source_row = _source_int(chart_row.get("source_row", 0))
        source_slot = _source_int(chart_row.get("source_slot", depth_rank))
        occurrence = _source_int(chart_row.get("position_occurrence", 0))
        starter = _source_bool(chart_row.get("is_listed_starter", False))
        status_class = _clean_text(chart_row.get("status_class", ""))
        resolved.append({
            "team": _normalize_team(chart_row.get("team", "")),
            # Position is the functional role consumed by the model; source
            # position/rank/status remain independently visible below.
            "position": functional_position,
            "functional_position": functional_position,
            "source_position": source_position,
            "roster_position": _clean_text(match.get("_roster_position", "")),
            "roster_depth_chart_position": _clean_text(match.get("_depth_chart_position", "")),
            "position_label": _clean_text(chart_row.get("position_label", "")),
            "depth_rank": depth_rank,
            "source_depth_rank": depth_rank,
            "source_rank": depth_rank,
            "source_row": source_row,
            "source_slot": source_slot,
            "position_occurrence": occurrence,
            "source_position_occurrence": occurrence,
            "is_listed_starter": starter,
            "source_is_listed_starter": starter,
            "is_inactive": inactive,
            "source_is_inactive": inactive,
            "status_class": status_class,
            "source_status_class": status_class,
            "source_status": _source_status(inactive, status_class),
            "source_status_warning": status_warning,
            "source_player_id": _clean_text(chart_row.get("source_player_id", "")),
            "source_gsis_id": audit_row["source_gsis_id"],
            "source_pff_id": audit_row["source_pff_id"],
            "source_entry_id": _source_entry_id(chart_row),
            "player": _clean_text(chart_row.get("player", "")),
            "player_key": _clean_text(chart_row.get("player_key", "")),
            "matched_player": match["_player"],
            "matched_player_key": match["_canonical_key"],
            "matched_player_id": _clean_text(match.get("_id_player_id", "")),
            "matched_gsis_id": _clean_text(match.get("_id_gsis_id", "")),
            "matched_pff_id": _clean_text(match.get("_id_pff_id", "")),
            "matched_identity_key": _clean_text(match.get("_identity_key", "")),
            "matched_roster_row": int(match["_roster_row"]),
            "match_method": method,
            "match_confidence": confidence,
            "match_warning": warning,
        })
    matches = pd.DataFrame(resolved, columns=_MATCH_COLUMNS)
    if not matches.empty:
        matches = matches.drop_duplicates(
            subset=["team", "position_label", "source_row", "source_slot", "matched_identity_key"],
            keep="first").reset_index(drop=True)
    audit_frame = pd.DataFrame(audit, columns=_IDENTITY_AUDIT_COLUMNS)
    return matches, audit_frame, warnings


def resolve_ourlads_roster_matches(snapshot: pd.DataFrame, roster: pd.DataFrame,
                                   name_col: str, team_col: str) -> tuple[pd.DataFrame, list[str]]:
    """Backward-compatible matched-row view of :func:`resolve_ourlads_roster_identities`.

    The complete audit is attached to the returned frame and is also exposed
    directly by ``build_ourlads_projection_signal`` for UI/audit consumers.
    """
    matches, audit, warnings = resolve_ourlads_roster_identities(
        snapshot, roster, name_col, team_col)
    matches.attrs["identity_audit"] = audit
    return matches, warnings


_SOURCE_ROLE_COLUMNS = [
    "source_entry_id", "team", "source_position", "source_position_label",
    "source_depth_rank", "source_rank", "source_row", "source_slot",
    "source_position_occurrence", "source_is_listed_starter", "source_is_inactive",
    "source_status", "source_status_warning", "source_status_class", "source_player_id", "source_player",
    "source_player_key", "match_status", "matched_player", "matched_player_key",
    "resolved_functional_position", "roster_position", "roster_depth_chart_position",
    "matched_player_id", "matched_gsis_id", "matched_pff_id", "matched_identity_key",
    "match_method", "match_confidence", "match_warning",
]


def _source_role_rows(snapshot: pd.DataFrame, matches: pd.DataFrame,
                      identity_audit: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return every source role cell, including unmatched/inactive entries.

    This is deliberately not an ``available rank`` table.  It is a complete
    audit ledger keyed to the literal page cells, so a downstream allocator
    can keep an unmatched Player 2 from promoting Player 3 to RB2.
    """
    if snapshot is None or snapshot.empty:
        return pd.DataFrame(columns=_SOURCE_ROLE_COLUMNS)
    source = snapshot.copy()
    source["_source_position"] = _column_or_default(source, "position").map(_position_from_value)
    source = source[source["_source_position"].isin(_SOURCE_POSITIONS)].copy()
    if source.empty:
        return pd.DataFrame(columns=_SOURCE_ROLE_COLUMNS)
    source["source_entry_id"] = source.apply(_source_entry_id, axis=1)
    source["source_depth_rank"] = pd.to_numeric(
        _column_or_default(source, "depth_rank", 0), errors="coerce").fillna(0).astype(int)
    source["source_rank"] = source["source_depth_rank"]
    source["source_row"] = pd.to_numeric(
        _column_or_default(source, "source_row", 0), errors="coerce").fillna(0).astype(int)
    source["source_slot"] = pd.to_numeric(
        _column_or_default(source, "source_slot"), errors="coerce").fillna(
        source["source_depth_rank"]).astype(int)
    source["source_position_occurrence"] = pd.to_numeric(
        _column_or_default(source, "position_occurrence", 0), errors="coerce").fillna(0).astype(int)
    source["source_is_listed_starter"] = _column_or_default(
        source, "is_listed_starter", False).map(_source_bool)
    source["source_is_inactive"] = _column_or_default(
        source, "is_inactive", False).map(_source_bool)
    source["source_status_class"] = _column_or_default(
        source, "status_class").fillna("").astype(str)
    source["source_status"] = [
        _source_status(inactive, status)
        for inactive, status in zip(source["source_is_inactive"], source["source_status_class"])
    ]
    source["source_status_warning"] = source["source_is_inactive"].map(
        lambda inactive: "Ourlads lc_red status is an unconfirmed source flag; current availability must confirm it."
        if inactive else "")
    source["source_position"] = source["_source_position"]
    source["source_position_label"] = _column_or_default(
        source, "position_label").fillna("").astype(str)
    source["source_player_id"] = _column_or_default(
        source, "source_player_id").fillna("").astype(str)
    source["source_player"] = _column_or_default(source, "player").fillna("").astype(str)
    source["source_player_key"] = _column_or_default(
        source, "player_key").fillna("").astype(str)

    resolution_columns = [
        "source_entry_id", "matched_player", "matched_player_key", "functional_position",
        "roster_position", "roster_depth_chart_position", "matched_player_id",
        "matched_gsis_id", "matched_pff_id", "matched_identity_key", "match_method",
        "match_confidence", "match_warning",
    ]
    if matches is not None and not matches.empty:
        resolution = matches.reindex(columns=resolution_columns).drop_duplicates(
            subset=["source_entry_id"], keep="first")
        source = source.merge(resolution, on="source_entry_id", how="left", validate="one_to_one")
    else:
        for column in resolution_columns:
            if column not in source.columns:
                source[column] = ""

    if identity_audit is None and matches is not None:
        identity_audit = matches.attrs.get("identity_audit")
    if identity_audit is not None and not identity_audit.empty:
        audit_columns = ["source_entry_id", "match_status", "match_method", "match_confidence", "match_warning"]
        audit = identity_audit.reindex(columns=audit_columns).drop_duplicates(
            subset=["source_entry_id"], keep="first")
        # Match details are already available from the resolver's matched-row
        # view.  The audit merge fills the status/reason for unmatched and
        # ambiguous source rows without guessing a roster identity.
        source = source.merge(audit, on="source_entry_id", how="left", suffixes=("", "_audit"))
        for column in ("match_method", "match_confidence", "match_warning"):
            audit_column = f"{column}_audit"
            if audit_column in source.columns:
                source[column] = source[audit_column].where(
                    source[audit_column].notna() & source[audit_column].ne(""), source[column])
                source = source.drop(columns=[audit_column])
    if "match_status" not in source.columns:
        source["match_status"] = ""
    source["match_status"] = source["match_status"].fillna("")
    source.loc[source["match_status"].eq(""), "match_status"] = source["matched_player_key"].notna().map(
        {True: "matched", False: "unmatched"})
    source["resolved_functional_position"] = source["functional_position"].fillna("").astype(str)
    for column in _SOURCE_ROLE_COLUMNS:
        if column not in source.columns:
            source[column] = ""
    return source.loc[:, _SOURCE_ROLE_COLUMNS].sort_values(
        ["team", "source_position_label", "source_row", "source_slot"], kind="stable").reset_index(drop=True)


def build_ourlads_projection_signal(snapshot: pd.DataFrame, roster: pd.DataFrame,
                                    name_col: str, team_col: str) -> dict[str, Any]:
    """Return matched QB1/role rows and a complete literal-source audit ledger.

    A page has to provide exactly one matched first-listed QB for a team before
    the model can use it as a QB1 signal.  Ourlads color/status styling remains
    a visible source flag only: current availability is resolved elsewhere by
    the manual/current-injury layer and is never inferred from ``lc_red``.
    """
    matches, identity_audit, warnings = resolve_ourlads_roster_identities(
        snapshot, roster, name_col, team_col)
    source_roles = _source_role_rows(snapshot, matches, identity_audit)
    if matches.empty:
        return {
            "matches": matches,
            "qb_starters": matches.copy(),
            "skill_roles": matches.copy(),
            "fullback_roles": matches.copy(),
            "available_roles": matches.copy(),
            "source_roles": source_roles,
            "identity_audit": identity_audit,
            "warnings": warnings,
            "matched_teams": [],
        }
    qb_candidates = matches[(matches["position"] == "QB")
                            & matches["position_occurrence"].eq(0)].copy()
    valid_qbs = []
    for team, group in qb_candidates.groupby("team", observed=True):
        ordered = group.sort_values(["source_row", "source_slot"], kind="stable")
        first = ordered.iloc[0]
        # Multiple copies of the same player are harmless (the same QB can be
        # shown in more than one page fragment); competing first-listed
        # players at the same source position are not.
        tied = ordered[(ordered["source_row"] == first["source_row"])
                      & (ordered["source_slot"] == first["source_slot"])]
        if tied["matched_player_key"].nunique() == 1:
            valid_qbs.append(first)
        else:
            warnings.append(f"{team}: imported Ourlads chart has competing first-listed QBs; ignored for QB1.")
    qb_starters = (pd.DataFrame(valid_qbs).reset_index(drop=True)
                   if valid_qbs else pd.DataFrame(columns=matches.columns))
    # This historical name is retained for backwards compatibility.  It is
    # an Ourlads *role* collection, not an availability verdict; red/source
    # flagged rows intentionally remain in it until a current source says Out.
    available_roles = matches.copy()
    # Keep the legacy V1-facing collection FB-free.  V2 allocation can use
    # `fullback_roles`/`available_roles` to explicitly route FBs away from
    # core-RB capacity rather than accidentally applying an RB role floor.
    skill_roles = available_roles[available_roles["position"].isin(_LEGACY_SKILL_POSITIONS)].copy()
    fullback_roles = available_roles[available_roles["position"].eq("FB")].copy()
    return {
        "matches": matches,
        "qb_starters": qb_starters,
        "skill_roles": skill_roles,
        "fullback_roles": fullback_roles,
        "available_roles": available_roles,
        "source_roles": source_roles,
        "identity_audit": identity_audit,
        "warnings": warnings,
        "matched_teams": sorted(matches["team"].drop_duplicates().tolist()),
    }


def _effective_source_starters(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Starter-only source rows, retaining source status as non-binding audit data."""
    if snapshot is None or snapshot.empty:
        return pd.DataFrame(columns=OURLADS_COLUMNS)
    source = snapshot[snapshot["position"].isin({"QB", "RB", "WR", "TE"})].copy()
    skill = source[(source["position"] != "QB") & source["is_listed_starter"]]
    qbs = source[(source["position"] == "QB") & source["position_occurrence"].eq(0)].copy()
    if not qbs.empty:
        qbs = (qbs.sort_values(["team", "source_row", "source_slot"], kind="stable")
                .groupby("team", observed=True).head(1))
    return pd.concat([skill, qbs], ignore_index=True).drop_duplicates(
        subset=["team", "position", "player_key"], keep="first")


def apply_ourlads_starter_roster_overlay(snapshot: pd.DataFrame, roster: pd.DataFrame,
                                         name_col: str, team_col: str,
                                         prior_history: pd.DataFrame | None = None,
                                         prior_name_col: str | None = None,
                                         prior_team_col: str | None = None) -> tuple[pd.DataFrame, list[dict[str, str]], list[str]]:
    """Correct only stale/missing *source-listed starters* in a cold-start pool.

    Current roster snapshots can lag a finalized preseason chart.  Rather than
    adding every chart reserve (which would create duplicates and unknown
    player projections), this overlay is intentionally strict:

    * Existing current-roster player: resolve through the same stable-id /
      exact-name / reviewed-alias / unique-suffix hierarchy as role evidence.
    * Missing player: append only a listed starter with exactly one matching
      historical identity at the same position, preserving any stable ids.
    * No last-name-only guesses, no reserve additions, and no duplicate
      suffix variants of an already-active roster player.

    The caller uses this only for an upcoming preseason cold start.  It is a
    roster identity correction, not an historical-data rewrite.
    """
    if roster is None or roster.empty or not {name_col, team_col, "position"}.issubset(roster.columns):
        return roster.copy() if roster is not None else pd.DataFrame(), [], []
    output = roster.copy()
    starters = _effective_source_starters(snapshot)
    if starters.empty:
        return output, [], []

    current_matches, current_audit, current_warnings = resolve_ourlads_roster_identities(
        starters, output, name_col, team_col)
    # A missing current match can still be verified from historical evidence
    # below.  Keep source-status/ambiguity warnings, but do not falsely warn
    # about a missing roster identity until that final verification fails.
    warnings = [warning for warning in current_warnings
                if "could not be uniquely matched to the current roster" not in warning]
    current_by_entry = current_matches.drop_duplicates(
        subset=["source_entry_id"], keep="first").set_index("source_entry_id", drop=False)
    current_audit_by_entry = current_audit.drop_duplicates(
        subset=["source_entry_id"], keep="first").set_index("source_entry_id", drop=False)

    prior = pd.DataFrame()
    if (prior_history is not None and not prior_history.empty and prior_name_col and prior_team_col
            and {prior_name_col, prior_team_col, "position"}.issubset(prior_history.columns)):
        prior = prior_history.copy()
        # The resolver accepts the caller's current column names.  Add these
        # bridge columns without discarding identity fields such as gsis_id.
        if prior_name_col != name_col:
            prior[name_col] = prior[prior_name_col]
        if prior_team_col != team_col:
            prior[team_col] = prior[prior_team_col]

    changes: list[dict[str, str]] = []
    for _, source in starters.iterrows():
        entry_id = _source_entry_id(source)
        team = _normalize_team(source.get("team", ""))
        pos = _position_from_value(source.get("position", ""))
        if entry_id in current_by_entry.index:
            match = current_by_entry.loc[entry_id]
            row_number = int(match["matched_roster_row"])
            old_team = str(output.iloc[row_number][team_col])
            if _normalize_team(old_team) != team:
                output.iat[row_number, output.columns.get_loc(team_col)] = team
                changes.append({
                    "player": str(output.iloc[row_number][name_col]), "position": pos,
                    "old_team": old_team, "new_team": team,
                    "action": "reassigned stale roster team",
                    "match_method": str(match.get("match_method", "")),
                })
            continue

        audit = current_audit_by_entry.loc[entry_id] if entry_id in current_audit_by_entry.index else pd.Series()
        if str(audit.get("match_status", "")) == "ambiguous":
            # The authoritative resolver has already emitted a visible
            # warning.  Never work around ambiguity by adding another row.
            continue
        if prior.empty:
            warnings.append(
                f"{team} {pos} starter '{source['player']}' is absent from the current roster and has no prior-year identity to verify.")
            continue

        historical_matches, historical_audit, _ = resolve_ourlads_roster_identities(
            pd.DataFrame([source]), prior, name_col, team_col, allow_cross_team_name=True)
        if historical_matches.empty:
            warnings.append(
                f"{team} {pos} starter '{source['player']}' is absent from current roster and could not be uniquely verified from prior history.")
            continue

        verified_match = historical_matches.iloc[0]
        verified_row = int(verified_match["matched_roster_row"])
        verified = prior.iloc[verified_row]
        verified_identity = str(verified_match.get("matched_identity_key", ""))
        # A historic verification can reveal that a live roster row already
        # carries this stable identity under a different team/name spelling.
        # Do not append a duplicate in that situation; require an explicit
        # current-team source or manual correction instead of guessing.
        current_identity_keys = stable_roster_identity_keys(output, name_col)
        if verified_identity and verified_identity in set(current_identity_keys):
            warnings.append(
                f"{team} {pos} starter '{source['player']}' has a matching active roster identity under a different team/name; not duplicated automatically.")
            continue

        added = {column: pd.NA for column in output.columns}
        for column in output.columns:
            if column in verified.index:
                added[column] = verified[column]
        added[name_col] = verified[name_col]
        added[team_col] = team
        added["position"] = pos
        output = pd.concat([output, pd.DataFrame([added])], ignore_index=True)
        changes.append({
            "player": str(verified[name_col]), "position": pos,
            "old_team": "", "new_team": team, "action": "added verified missing starter",
            "match_method": str(verified_match.get("match_method", "")),
        })
    return output, changes, warnings


def save_ourlads_key(upload: Any, path: str | Path = OURLADS_KEY_PATH) -> tuple[str, str | None]:
    """Persist a plain-text version of the optional Ourlads legend locally."""
    try:
        _, blob = _file_blob(upload)
        html = _extract_html(blob)
        text = _clean_text(re.sub(r"<[^>]+>", " ", html))
    except Exception as exc:
        return "", f"Could not read Ourlads depth-chart key: {exc}"
    if not text:
        return "", "The Ourlads depth-chart key did not contain readable text."
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    except Exception as exc:
        return text, f"Parsed the Ourlads key but could not save it: {exc}"
    return text, None


def load_ourlads_key(path: str | Path = OURLADS_KEY_PATH) -> tuple[str, str | None]:
    target = Path(path)
    if not target.exists():
        return "", None
    try:
        return target.read_text(encoding="utf-8"), None
    except Exception as exc:
        return "", f"Could not read {target.name}: {exc}"
