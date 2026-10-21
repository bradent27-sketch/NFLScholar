"""
Weekly boards built this session (2026-10-21).

A board build is the expensive call in the app (minutes for a live week), and build_weekly_projections' own cache expires
after an hour, after which any full page rerun silently rebuilt the whole league. The board a user built is kept here
instead, keyed by (season, week, scoring), and reused for as long as the inputs it depends on are unchanged: the week's
injury/availability overrides (availability_fingerprint) and the season's QB1 choices. Build board (or a changed override)
replaces it. Weekly Rankings and Player Search share these boards, so both always show the same numbers.
"""
import datetime
import hashlib

import streamlit as st

from data.availability_overrides import availability_fingerprint, AVAILABILITY_OVERRIDE_PATH
from data.fantasypros_availability import FANTASYPROS_INJURY_PATH
from data.weekly_projections import build_weekly_projections, load_qb1_overrides

# Each kept board holds the board and its per-player explanations (a few hundred MB in memory), so only the last two
# selections are kept.
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


def kept_board(year, week, scoring):
    """The kept board entry for these selectors if its inputs are still current, else None."""
    entry = _boards().get((year, week, scoring))
    if entry is None or entry['inputs'] != board_inputs(year, week):
        return None
    return entry


def has_board(year, week, scoring):
    return (year, week, scoring) in _boards()


def build_board(year, week, scoring):
    """Build (or read from build_weekly_projections' cache) and keep the board; returns its entry."""
    inputs = board_inputs(year, week)
    model_df, model_meta = build_weekly_projections(year, week, scoring, availability_fingerprint=inputs[0])
    entry = {'inputs': inputs, 'df': model_df, 'meta': model_meta, 'built_at': datetime.datetime.now()}
    boards = _boards()
    boards.pop((year, week, scoring), None)
    boards[(year, week, scoring)] = entry
    while len(boards) > MAX_KEPT_BOARDS:
        boards.pop(next(iter(boards)))
    return entry
