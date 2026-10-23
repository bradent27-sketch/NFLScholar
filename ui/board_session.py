"""
Weekly boards built this session (2026-10-21), kept on disk and brought up to date by team (2026-10-23, ui/board_pin.py).

A board build is the expensive call in the app (minutes for a live week), and build_weekly_projections' own cache expires
after an hour, after which any full page rerun silently rebuilt the whole league. The board a user built is kept here
instead, keyed by (season, week, scoring), and saved to disk so a new browser session or an app restart shows it again
(ui/board_pin.BOARD_KEEP_HOURS). A change to what it depends on - the week's injury/availability overrides and FantasyPros
injury pull (availability_fingerprint) and the season's QB1 choices - re-projects only the teams whose rows changed and swaps
them in; Build board rebuilds the whole board. Weekly Rankings and Player Search share these boards, so both always show the
same numbers.
"""
import datetime
import hashlib

import streamlit as st

from data.availability_overrides import availability_fingerprint, AVAILABILITY_OVERRIDE_PATH
from data.fantasypros_availability import FANTASYPROS_INJURY_PATH
from data.weekly_projections import LeagueContextMissing, build_weekly_projections, load_qb1_overrides
from ui import board_pin

# Each kept board holds the board and its per-player explanations (a few hundred MB in memory), so only the last two
# selections are kept in the session (ui/board_pin keeps a few more on disk).
MAX_KEPT_BOARDS = 2
_STATE_KEY = 'weekly_rank_boards'


def qb1_override_signature(year):
    """Changes whenever a QB1 choice for this season is saved or cleared."""
    try:
        table, problem = load_qb1_overrides(year)
    except Exception:
        return ''
    text = (table.to_csv(index=False) if table is not None and not table.empty else '') + (str(problem) if problem else '')
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:12]


def board_inputs(year, week):
    """What a kept board depends on beyond its key: this week's availability overrides and the season's QB1 choices."""
    avail_fp = availability_fingerprint(year, week, AVAILABILITY_OVERRIDE_PATH, FANTASYPROS_INJURY_PATH)
    return avail_fp, qb1_override_signature(year)


def _boards():
    return st.session_state.setdefault(_STATE_KEY, {})


def _keep(key, entry):
    boards = _boards()
    boards.pop(key, None)
    boards[key] = entry
    while len(boards) > MAX_KEPT_BOARDS:
        boards.pop(next(iter(boards)))


def _session_or_saved(key):
    """This session's board for the key, else the one saved to disk (then kept in the session)."""
    entry = _boards().get(key)
    if entry is None:
        entry = board_pin.load(key)
        if entry is not None:
            _keep(key, entry)
    return entry


def has_board(year, week, scoring):
    return _session_or_saved((year, week, scoring)) is not None


def kept_board(year, week, scoring):
    """The kept board for these selectors, up to date with the current rulings, else None (the caller builds in full).

    When a ruling changed since it was built, the teams whose ruling rows changed are re-projected and swapped in
    (ui/board_pin.merge_team_board); the entry returned by THAT call carries 'just_reprojected'. None when nothing is
    kept, or when the change cannot be put down to teams (or the board predates the team-only build)."""
    key = (year, week, scoring)
    entry = _session_or_saved(key)
    if entry is None:
        return None
    inputs = board_inputs(year, week)
    if entry['inputs'] == inputs:
        return entry
    rulings = board_pin.ruling_snapshot(year, week)
    teams = board_pin.changed_teams(entry.get('rulings'), rulings)
    if not teams:
        return None
    try:
        with st.spinner(f"Re-projecting {', '.join(sorted(teams))} for the changed ruling..."):
            part, part_meta = build_weekly_projections(
                year, week, scoring, availability_fingerprint=inputs[0], only_teams=frozenset(teams),
                league_context=(entry['meta'] or {}).get('league_context'))
    except LeagueContextMissing:
        return None
    df, meta = board_pin.merge_team_board(entry['df'], entry['meta'], part, part_meta, teams)
    fresh = {'inputs': inputs, 'df': df, 'meta': meta, 'built_at': entry['built_at'], 'rulings': rulings,
             'reprojected': sorted(set(entry.get('reprojected') or []) | set(teams)),
             'reprojected_at': datetime.datetime.now()}
    _keep(key, fresh)
    board_pin.save(key, fresh)
    return {**fresh, 'just_reprojected': True}


def build_board(year, week, scoring):
    """Build (or read from build_weekly_projections' cache), keep and save the board; returns its entry."""
    inputs = board_inputs(year, week)
    rulings = board_pin.ruling_snapshot(year, week)
    model_df, model_meta = build_weekly_projections(year, week, scoring, availability_fingerprint=inputs[0])
    entry = {'inputs': inputs, 'df': model_df, 'meta': model_meta, 'built_at': datetime.datetime.now(),
             'rulings': rulings}
    _keep((year, week, scoring), entry)
    if model_df is not None and not model_df.empty:
        board_pin.save((year, week, scoring), entry)
    return entry
