"""FantasyPros-sourced injury/availability signal (V2 experiment).

WHY THIS EXISTS. The existing live injury source (``_injury_profiles`` /
``_injury_multipliers`` in ``data.weekly_projections``) reads nflverse's
``nflreadpy.load_injuries`` feed, which always returns each player's MOST
RECENT designation for the season. That is correct for "who's hurt right
now" but is a documented, measured source of real harm elsewhere in this
model (see ``build_weekly_projections``'s own docstring: on its own it
quietly zeroed or discounted roughly 1,000 of ~2,000 skill-position players
in the 2025 backtest before that was isolated) - and, per the user, it is
simply not the source they want driving this model: "injury data should be
sourcing from fantasy pros API loads and the model should assume that
players are healthy until that is uploaded."

THIS APP DOES NOT SCRAPE. It never reads a web page - every source it uses
is either a file the user hands over once (see ``data/fantasypros_import.py``'s
own explicit "this is deliberately not a scraper" policy) or FantasyPros'
own official API, called with the user's own key. This module supports
both for injury status: a saved FantasyPros export, or a live pull from
FantasyPros' dedicated ``GET /nfl/injuries`` endpoint (see
``fetch_fantasypros_injury_report`` below - NOT ``/nfl/players``, which
carries no injury field at all, confirmed against a real response).

THE DEFAULT: no file uploaded yet -> every player is healthy. Not "unknown",
not a stale prior-season carry-over, not nflverse's last-report-of-the-
season - healthy. A report that was never uploaded is not evidence of an
injury, and this module's empty-input behavior enforces that literally: an
empty/missing file returns an empty profile dict, and every downstream
availability computation already treats "no profile for this player" as
``plays_probability=1.0`` (see ``data.availability_overrides``).

SCHEMA TOLERANCE. FantasyPros does not ship one fixed column layout across
their products, so this reads by NORMALIZED column name (case/whitespace-
insensitive), looking for a player-name column and a status/injury column -
team and a return-date/note column are optional. A file that cannot be
resolved this way produces a clear error rather than a guessed mapping,
matching ``data/qb1_overrides.csv``'s own tolerant-header reader.

Storage and resolution reuse ``data.availability_overrides`` rather than a
second implementation: this module persists to its own CSV
(``fantasypros_injury_report.csv``, gitignored like every other external
import) in that module's exact column schema, then reads it back through
``load_availability_overrides``. The existing manual
``availability_overrides.csv`` layer remains available on top of this as the
reviewable human-correction path it already was - a manual entry still wins
over an imported FantasyPros row for the same player, unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests

from data.availability_overrides import AVAILABILITY_OVERRIDE_COLUMNS, load_availability_overrides
from data.draft_sources import FANTASYPROS_API_BASE, _REQUEST_HEADERS, _record_fantasypros_api_call


FANTASYPROS_INJURY_PATH = Path(__file__).with_name('fantasypros_injury_report.csv')

_NAME_COLUMNS = ('player', 'player name', 'name')
_TEAM_COLUMNS = ('team', 'tm')
_STATUS_COLUMNS = ('status', 'injury status', 'inj', 'injury', 'designation')
_NOTE_COLUMNS = ('note', 'est. return', 'return', 'comment', 'notes')

_STATUS_ALIASES = {
    'ir': 'out', 'injured reserve': 'out', 'out': 'out', 'o': 'out',
    'suspended': 'out', 'sus': 'out', 'pup': 'out', 'nfi': 'out', 'cov-ir': 'out',
    # FantasyPros' own enum for this field (confirmed against their OpenAPI
    # spec) is COV-IR/Doubtful/IR/Not Starting/OUT/PUP/Questionable/Suspended.
    # "Not Starting" isn't necessarily an injury (could be a healthy
    # benching), but it appears on the INJURY report, so treat it as
    # doubtful rather than either silently dropping it or zeroing the player
    # outright the way 'out' would.
    'not starting': 'doubtful',
    'doubtful': 'doubtful', 'd': 'doubtful',
    'questionable': 'questionable', 'q': 'questionable', 'probable': 'questionable', 'p': 'questionable',
    'active': 'healthy', 'healthy': 'healthy', '': 'healthy', 'nan': 'healthy',
}


def canonical_status(raw) -> str:
    """Any raw availability status string -> exactly one of healthy/
    questionable/doubtful/out, for display (the decomposition dialog's
    header, in place of a bare plays-probability percentage).

    Reuses _STATUS_ALIASES so this always agrees with what
    parse_fantasypros_injury_export already does. Missing/empty input reads
    as 'healthy' (no report = presumed fine - the same fallback
    _STATUS_ALIASES itself uses for '' and 'nan'). This includes 'No
    current designation' - data.weekly_projections' own
    `row['Injury Status'] or 'No current designation'` fallback string for
    a player with nothing on the report, not a real designation, so it has
    to map the same way plain emptiness does or every player with a clean
    bill of health would misread as flagged. A NON-EMPTY, genuinely
    unrecognized status - possible from the older nflverse-sourced V1
    availability path, which does not route through _STATUS_ALIASES
    upstream - reads as 'questionable' rather than 'healthy': a real but
    unmapped designation should never silently present as full health.
    """
    if raw is None:
        return 'healthy'
    key = str(raw).strip().lower()
    if key in ('', 'nan', 'no current designation'):
        return 'healthy'
    mapped = _STATUS_ALIASES.get(key, 'questionable')
    return mapped if mapped in ('healthy', 'questionable', 'doubtful', 'out') else 'questionable'


# The FantasyPros injury export's probability-of-playing column, if present
# (only the live API supplies this - a saved CSV export generally won't).
_PROB_COLUMNS = ('plays_probability', 'probability_of_playing', 'probability')


def _normalized_columns(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column).strip().lower(): column for column in frame.columns}


def _find_column(normalized: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    return next((normalized[c] for c in candidates if c in normalized), None)


def parse_fantasypros_injury_export(raw: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """A saved FantasyPros injury/status export -> the availability_overrides schema.

    Only rows carrying an actual non-healthy status are kept - a clean
    report simply has no row for a healthy player, which is exactly the
    "absence means healthy" contract this module exists to enforce.
    """
    empty = pd.DataFrame(columns=AVAILABILITY_OVERRIDE_COLUMNS)
    if raw is None or raw.empty:
        return empty, 'that file has no rows.'
    normalized = _normalized_columns(raw)
    name_col = _find_column(normalized, _NAME_COLUMNS)
    status_col = _find_column(normalized, _STATUS_COLUMNS)
    if name_col is None or status_col is None:
        return empty, (
            "couldn't find a player-name and a status/injury column in that file "
            f'(saw: {", ".join(str(c) for c in raw.columns)}).')
    team_col = _find_column(normalized, _TEAM_COLUMNS)
    note_col = _find_column(normalized, _NOTE_COLUMNS)
    prob_col = _find_column(normalized, _PROB_COLUMNS)

    out = pd.DataFrame({
        'player': raw[name_col].astype(str).str.strip(),
        'team': (raw[team_col].astype(str).str.strip().str.upper() if team_col else ''),
        'status': raw[status_col].astype(str).str.strip().str.lower(),
        'note': (raw[note_col].astype(str).str.strip() if note_col else ''),
        'plays_probability': (pd.to_numeric(raw[prob_col], errors='coerce') if prob_col else ''),
    })
    out = out[out['player'].ne('') & ~out['player'].str.lower().isin({'nan', 'none'})]
    out['status'] = out['status'].map(lambda s: _STATUS_ALIASES.get(s, s if s and s != 'nan' else 'healthy'))
    out = out[out['status'].ne('healthy')].copy()
    out['year'] = ''
    out['week'] = ''
    out['workload_if_active'] = ''
    return out.reindex(columns=AVAILABILITY_OVERRIDE_COLUMNS), None


def _persist_fantasypros_injury_rows(parsed: pd.DataFrame, year: int, week: int,
                                     path: str | Path) -> int:
    """Shared write path for both the upload and live-API sources below.

    Replaces only THIS year+week's previously saved rows, so a season's
    worth of weekly imports accumulates in one file the same way
    ``availability_overrides.csv`` already does - regardless of whether any
    one week's rows came from an uploaded file or a live API pull.
    """
    parsed = parsed.copy()
    parsed['year'] = int(year)
    parsed['week'] = int(week)

    target = Path(path)
    existing = pd.DataFrame(columns=AVAILABILITY_OVERRIDE_COLUMNS)
    if target.is_file():
        try:
            existing = pd.read_csv(target, dtype=str, keep_default_na=False)
        except Exception:
            existing = pd.DataFrame(columns=AVAILABILITY_OVERRIDE_COLUMNS)
    for column in AVAILABILITY_OVERRIDE_COLUMNS:
        if column not in existing.columns:
            existing[column] = ''
    existing['year'] = pd.to_numeric(existing['year'], errors='coerce')
    existing['week'] = pd.to_numeric(existing['week'], errors='coerce')
    kept = existing[~(existing['year'].eq(int(year)) & existing['week'].eq(int(week)))]
    combined = pd.concat([kept, parsed], ignore_index=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    combined.reindex(columns=AVAILABILITY_OVERRIDE_COLUMNS).to_csv(target, index=False)
    return len(parsed)


def save_fantasypros_injury_upload(uploaded_file, year: int, week: int,
                                   path: str | Path = FANTASYPROS_INJURY_PATH) -> tuple[int, str | None]:
    """Persist one week's FantasyPros injury export, stamped with year/week."""
    if uploaded_file is None:
        return 0, None
    try:
        uploaded_file.seek(0)
        raw = pd.read_csv(uploaded_file)
    except Exception as exc:
        return 0, f"couldn't read that file: {exc}"
    parsed, error = parse_fantasypros_injury_export(raw)
    if error:
        return 0, error
    return _persist_fantasypros_injury_rows(parsed, year, week, path), None


# Real field names from FantasyPros' own OpenAPI spec (Injury schema, GET
# /{sport}/injuries) - not a guess. The /nfl/players endpoint this module
# originally tried has NO injury field of any kind; a live response dump
# confirmed it carries only roster/ranking metadata (age, birthdate,
# draft_class, filename, first_name, last_name, player_id, player_name,
# position_id, positions, rank_adp, rank_adp_ppr, rank_ecr, rank_ecr_half,
# rank_ecr_pos, rank_ecr_ppr, reverse_name, rookie, short_name,
# sportsdata_player_id, team_id). Injuries live on their own dedicated
# endpoint instead, keyed by year+week like everything else in this app.
_API_NAME_FIELD = 'name'
_API_TEAM_FIELD = 'team_id'
_API_STATUS_FIELD = 'status'
_API_NOTE_FIELD = 'comment'
_API_PROBABILITY_FIELD = 'probability_of_playing'


def fetch_fantasypros_injury_report(api_key: str, year: int, week: int) -> tuple[pd.DataFrame, str | None]:
    """Live FantasyPros injury/status pull from GET /nfl/injuries, source-shaped
    for parse_fantasypros_injury_export.

    A clean week can legitimately return zero injuries - that's success, not
    an error, and comes back as an empty DataFrame with no error message so
    the caller doesn't show a false failure.
    """
    if not api_key:
        return pd.DataFrame(), 'no API key set'
    _record_fantasypros_api_call()
    try:
        resp = requests.get(
            f"{FANTASYPROS_API_BASE}/nfl/injuries",
            params={'year': int(year), 'week': int(week), 'include_probabilities': 'true'},
            headers={**_REQUEST_HEADERS, 'x-api-key': api_key},
            timeout=25,
        )
    except Exception as exc:
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}"
    if resp.status_code == 401:
        return pd.DataFrame(), 'API key rejected (401).'
    if resp.status_code == 403:
        return pd.DataFrame(), "API key valid but not authorized for this endpoint (403)."
    if resp.status_code == 429:
        return pd.DataFrame(), 'rate limited (429).'
    if resp.status_code != 200:
        return pd.DataFrame(), f'HTTP {resp.status_code}: {resp.text[:200]}'
    try:
        injuries = resp.json().get('injuries', [])
    except Exception as exc:
        return pd.DataFrame(), f'parse failed: {exc}'
    if not injuries:
        return pd.DataFrame(), None

    raw = pd.DataFrame({
        'player': [i.get(_API_NAME_FIELD, '') for i in injuries],
        'team': [i.get(_API_TEAM_FIELD, '') or '' for i in injuries],
        'status': [i.get(_API_STATUS_FIELD, '') for i in injuries],
        'note': [i.get(_API_NOTE_FIELD, '') or '' for i in injuries],
        'plays_probability': [i.get(_API_PROBABILITY_FIELD) for i in injuries],
    })
    return raw, None


def save_fantasypros_injury_api_pull(api_key: str, year: int, week: int,
                                     path: str | Path = FANTASYPROS_INJURY_PATH) -> tuple[int, str | None]:
    """Live-API counterpart to save_fantasypros_injury_upload - same output file,
    same year/week replace-in-place semantics, same downstream contract, so
    load_fantasypros_availability needs no changes regardless of which source
    filled the file for a given week (the user's own plan: upload until the
    API is live, switch the button, nothing else in the model changes)."""
    raw, error = fetch_fantasypros_injury_report(api_key, year, week)
    if error:
        return 0, error
    if raw.empty:
        return _persist_fantasypros_injury_rows(
            pd.DataFrame(columns=AVAILABILITY_OVERRIDE_COLUMNS), year, week, path), None
    parsed, parse_error = parse_fantasypros_injury_export(raw)
    if parse_error:
        return 0, parse_error
    return _persist_fantasypros_injury_rows(parsed, year, week, path), None


def load_fantasypros_availability(year: int, week: int,
                                  path: str | Path = FANTASYPROS_INJURY_PATH) -> tuple[dict[str, dict[str, Any]], str | None]:
    """This week's FantasyPros-sourced availability, as {player: profile}.

    Same return shape as ``data.weekly_projections._injury_profiles`` so it
    drops into the same ``raw_injury_profiles`` seam. No file uploaded yet,
    or nothing for this year/week, returns an empty dict - which
    ``resolve_target_week_availability`` and every downstream consumer
    already read as "no report, assume healthy", not as an error.
    """
    rows, error = load_availability_overrides(year, week, path=path)
    if rows.empty:
        return {}, error
    profiles: dict[str, dict[str, Any]] = {}
    for _, row in rows.iterrows():
        player = str(row.get('player', '')).strip()
        if not player:
            continue
        profiles[player] = {
            'status': str(row.get('status', '')).strip().lower(),
            'team': str(row.get('team', '')).strip(),
            'plays_probability': row.get('plays_probability'),
            'source': 'FantasyPros injury report',
        }
    return profiles, error


# ---------------------------------------------------------------------------
# PLAYER NEWS (added 2026-08-27, explicit request)
#
# Real field names below are from FantasyPros' own OpenAPI spec
# (api.fantasypros.com/public/v2/docs, GET /{sport}/news), read live via a
# browser 2026-08-27 - not a guess, same discipline as the injuries endpoint
# above. Full response shape confirmed from that spec's own example:
#
#   {"sport": "NFL", "title": "...", "description": "...", "count": 25,
#    "items": [{"id": 51970, "created": "2025-05-12 07:29:02",
#               "created_formated": "Mon, May 12th 7:29am UTC",
#               "author": "...", "player_id": 6880, "team_id": "IND",
#               "title": "...", "sport_id": "NFL", "categories": [],
#               "link": "https://...", "desc": "...", "impact": "..."}]}
#
# (note FantasyPros' own field is "created_formated", one t - not a typo on
# this side, kept exactly as the API spells it so a raw response maps 1:1.)
#
# WHY A PER-PLAYER CALL, NOT ONE BULK PULL: /nfl/news has no way to ask for
# "each player's own last 6" in one request - `limit` caps at 100 items
# total, league-wide, sorted by recency. A single generic pull would surface
# whichever handful of players are in the news RIGHT NOW and nothing at all
# for most others, not 6 items for whoever's actually being viewed. `fpid=`
# is the only way to guarantee a SPECIFIC player's own recent items, so this
# is one call per player. That was a hard constraint on FantasyPros' free
# tier (50/month, ~1.6/day - see data.draft_sources' own module comment) but
# this account is confirmed non-free at 500/day (user, 2026-08-27), which
# comfortably covers loading news as a player is actually viewed - see
# _render_player_news in ui/tabs/rankings.py for where this auto-fires
# (still cached per player per session, so revisiting the same player in one
# sitting costs nothing further).
FANTASYPROS_NEWS_DEFAULT_LIMIT = 6


def fetch_fantasypros_player_news(api_key: str, fantasypros_player_id, limit: int = FANTASYPROS_NEWS_DEFAULT_LIMIT):
    """Live pull from GET /nfl/news?fpid=<id>&limit=<n>&order_by=created.

    Returns (news, error) where news is a list of dicts (newest first,
    sorted defensively even though order_by=created should already do this -
    see the module comment above for why trusting an unverified default
    silently isn't this app's style) with keys: id, created, created_formatted,
    author, title, desc, impact, link, categories. A player with zero recent
    news is success, not an error - empty list, no error message, same
    convention as fetch_fantasypros_injury_report's "a clean report is not a
    failure" rule.
    """
    if not api_key:
        return [], 'no API key set'
    try:
        fpid = int(fantasypros_player_id)
    except (TypeError, ValueError):
        return [], 'no FantasyPros player ID for this player'
    _record_fantasypros_api_call()
    try:
        resp = requests.get(
            f"{FANTASYPROS_API_BASE}/nfl/news",
            params={'fpid': fpid, 'limit': int(limit), 'order_by': 'created'},
            headers={**_REQUEST_HEADERS, 'x-api-key': api_key},
            timeout=25,
        )
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    if resp.status_code == 401:
        return [], 'API key rejected (401).'
    if resp.status_code == 403:
        return [], "API key valid but not authorized for this endpoint (403)."
    if resp.status_code == 429:
        return [], 'rate limited (429).'
    if resp.status_code != 200:
        return [], f'HTTP {resp.status_code}: {resp.text[:200]}'
    try:
        items = resp.json().get('items', [])
    except Exception as exc:
        return [], f'parse failed: {exc}'
    news = [{
        'id': it.get('id'),
        'created': it.get('created', '') or '',
        'created_formatted': it.get('created_formated', '') or '',
        'author': it.get('author', '') or '',
        'title': it.get('title', '') or '',
        'desc': it.get('desc', '') or '',
        'impact': it.get('impact', '') or '',
        'link': it.get('link', '') or '',
        'categories': it.get('categories') or [],
    } for it in items]
    news.sort(key=lambda n: n['created'], reverse=True)
    return news[:limit], None


def resolve_fantasypros_player_id(player_name: str, id_map_df):
    """Match this app's player display name to a FantasyPros numeric
    player_id via the DynastyProcess crosswalk
    (data.draft_sources.load_player_id_map, its own 'fantasypros_id'
    column) - same two-tier exact-then-loose matching as every other
    cross-source name join in this app (data.utils.clean_name_exact /
    clean_name_for_merge), so a suffix mismatch ("Michael Pittman" vs
    "Michael Pittman Jr.") falls back the same way it does everywhere else
    rather than silently failing to match. Returns None (not a guess) when
    no confident match exists.
    """
    if id_map_df is None or id_map_df.empty or 'fantasypros_id' not in id_map_df.columns:
        return None
    name_col = 'name' if 'name' in id_map_df.columns else None
    if name_col is None or not player_name:
        return None
    from data.utils import clean_name_exact, clean_name_for_merge

    key = clean_name_exact(pd.Series([player_name])).iloc[0]
    exact_keys = clean_name_exact(id_map_df[name_col])
    match = id_map_df[exact_keys == key]
    if match.empty:
        loose_key = clean_name_for_merge(pd.Series([player_name])).iloc[0]
        loose_keys = clean_name_for_merge(id_map_df[name_col])
        match = id_map_df[loose_keys == loose_key]
    if match.empty:
        return None
    fpid = pd.to_numeric(match['fantasypros_id'].iloc[0], errors='coerce')
    return int(fpid) if pd.notna(fpid) else None
