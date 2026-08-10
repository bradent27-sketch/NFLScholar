"""
Weekly player props: when to pull them, and how to keep them.

WHY THIS IS A SEPARATE MODULE FROM odds_sources. That one knows how to talk
to each book. This one knows the only two things that make a weekly board
different from a season board:

  1. A weekly line has a shelf life. Books post the coming weekend's numbers
     on Tuesday or Wednesday and move them all week on news. A season line
     posted in June is still roughly the June line in August; a Thursday
     number is not the Tuesday number.

  2. So a weekly pull needs a NOTION OF WHEN IT WENT STALE, and that notion
     is a day of the week rather than a duration. A snapshot taken last
     Thursday is stale on Wednesday not because it is six days old but
     because a new slate has been posted since.

WHAT "AUTOMATIC WEEKLY LOAD" MEANS HERE. Streamlit has no scheduler - there
is no process running on Tuesday morning to go and fetch. What there is
instead is a snapshot on disk with the time it was taken, and a rule: if the
snapshot predates the most recent posting anchor, the next time the app is
opened it refetches on its own. Open the app any time after Tuesday morning
and the lines are current without having asked. That is the same observable
behaviour as a scheduled job for a tool that is only ever used while someone
is looking at it, and it cannot silently stop running.

Manual refresh sits on top and always wins, because "I am starting a real
session of analysis and I want the current numbers" is a judgement no
staleness rule should be allowed to overrule.
"""
import datetime
import json
import os

import pandas as pd

from data.odds_sources import (
    _empty_props, combine_props, fetch_prizepicks_lines, fetch_underdog_lines,
    fetch_draftkings_weekly_lines, load_saved_book_payload,
    parse_prizepicks_payload, parse_underdog_payload, parse_draftkings_payloads,
)

# Where a saved payload for each weekly book comes from, when there is one.
# A saved file WINS over the live fetch, for the same reason it does on the
# season side: it is a deliberate act by someone who has just looked at that
# board, and it is the path that keeps working when a book refuses us.
#
# PrizePicks is the one that needs this most. Cloudflare refuses this app
# whether it asks as a script or through a browser, so their weekly board is
# only ever going to arrive as a file - and it has to be a DIFFERENT file
# from the season one, because their weekly product is a separate league and
# saving one over the other would swap season totals for single-game props.
#
# Underdog has no separate entry: one endpoint returns its game and season
# lines together, so the season file already carries the weekly board and a
# second copy would only be a second thing to keep current.
WEEKLY_SAVED_SOURCES = {
    'PrizePicks': ('PrizePicks Weekly', parse_prizepicks_payload),
    'Underdog': ('Underdog', parse_underdog_payload),
    'DraftKings': ('DraftKings Weekly', parse_draftkings_payloads),
}

# The books that need no key and no quota. All three are pulled together on
# a weekly load; The Odds API is deliberately NOT in this list - it costs
# credits and is only ever spent on an explicit request.
WEEKLY_BOOKS = ('PrizePicks', 'Underdog', 'DraftKings')

WEEKLY_SNAPSHOT_PATH = os.path.join('external_data', 'weekly_props_snapshot.json')

# Tuesday, 10:00 America/New_York. Books put the coming weekend up on Tuesday
# or Wednesday; anchoring on the earlier of the two means the first Wednesday
# open is already current rather than a day behind. Stored as a UTC hour to
# avoid a timezone dependency - 10:00 ET is 14:00 UTC in summer and 15:00 in
# winter, and 15:00 is used because being an hour late in July is harmless
# while being an hour early in December would anchor to a slate that is not
# posted yet.
POST_WEEKDAY = 1        # Monday is 0
POST_HOUR_UTC = 15


def posting_anchor(now=None):
    """The most recent moment a new weekly slate is expected to be up."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    anchor = now.replace(hour=POST_HOUR_UTC, minute=0, second=0, microsecond=0)
    # Walk back to the most recent posting weekday at or before `now`.
    days_since = (now.weekday() - POST_WEEKDAY) % 7
    anchor -= datetime.timedelta(days=days_since)
    if anchor > now:
        anchor -= datetime.timedelta(days=7)
    return anchor


def is_stale(fetched_at, now=None):
    """Has a new slate been posted since this snapshot was taken?"""
    if fetched_at is None:
        return True
    return fetched_at < posting_anchor(now)


def _parse_time(value):
    try:
        stamp = datetime.datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=datetime.timezone.utc)


def load_snapshot(path=WEEKLY_SNAPSHOT_PATH):
    """
    The last weekly pull, as (props, fetched_at, status). Never raises.

    A corrupt or half-written file reads as "no snapshot", which makes the
    next call refetch - the correct behaviour, and better than an exception
    on a tab that is trying to show odds.
    """
    try:
        with open(path, encoding='utf-8') as handle:
            blob = json.load(handle)
        props = pd.DataFrame(blob.get('rows') or [])
        if not props.empty:
            props['line'] = pd.to_numeric(props['line'], errors='coerce')
        return props, _parse_time(blob.get('fetched_at')), blob.get('status') or {}
    except Exception:
        return _empty_props(), None, {}


def save_snapshot(props, status, fetched_at=None, path=WEEKLY_SNAPSHOT_PATH):
    """Persist a weekly pull. Returns an error string, or None."""
    fetched_at = fetched_at or datetime.datetime.now(datetime.timezone.utc)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump({'fetched_at': fetched_at.isoformat(),
                       'status': status,
                       'rows': props.to_dict('records')}, handle)
    except Exception as exc:
        return f"Pulled the lines but couldn't save them: {exc}"
    return None


def fetch_weekly_props(books=WEEKLY_BOOKS):
    """
    One pull of every free book's weekly board. Returns (props, status).

    A book that fails is reported and skipped rather than failing the pull.
    They are independent sources and two of three is a working slate.
    """
    status, frames = {}, []
    for name in books:
        props, err, source = _empty_props(), None, ''

        saved_key, parser = WEEKLY_SAVED_SOURCES.get(name, (None, None))
        payload = load_saved_book_payload(saved_key) if saved_key else None
        if payload is not None:
            props, err = parser(payload)
            source = 'your saved payload'

        # Only go to the network when there is no file to read. A saved
        # payload is there precisely because the live path did not work.
        if props.empty and not source:
            if name == 'PrizePicks':
                # season_first=False is the whole difference: their weekly
                # board is league 9 and the season board is a different
                # league, so asking for season first returns the wrong
                # product entirely.
                props, err = fetch_prizepicks_lines(season_first=False)
            elif name == 'Underdog':
                props, err = fetch_underdog_lines()
            elif name == 'DraftKings':
                props, err = fetch_draftkings_weekly_lines()
            else:
                continue
            source = 'live'

        if not props.empty:
            props = props[props['period'] == 'game']
        status[name] = {'rows': int(len(props)), 'error': err, 'source': source,
                        'players': int(props['player'].nunique()) if not props.empty else 0}
        if not props.empty:
            frames.append(props)

    return combine_props(*frames), status


def weekly_props(force=False, books=WEEKLY_BOOKS, path=WEEKLY_SNAPSHOT_PATH, now=None):
    """
    The current weekly board, refetching only when it needs to.

    Returns (props, meta) where meta carries fetched_at, whether this call
    went to the network, and the per-book status. Refetches when forced, when
    there is no snapshot, or when a new slate has been posted since the last
    one - and otherwise serves the file, which is what keeps opening the tab
    from costing three requests every time.
    """
    saved, fetched_at, status = load_snapshot(path)
    if not force and not saved.empty and not is_stale(fetched_at, now):
        return saved, {'fetched_at': fetched_at, 'from_network': False,
                       'status': status, 'stale': False}

    props, status = fetch_weekly_props(books)
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    if props.empty:
        # Nothing came back. A stale snapshot beats a blank tab, as long as
        # it is labelled stale rather than passed off as current.
        if not saved.empty:
            return saved, {'fetched_at': fetched_at, 'from_network': True,
                           'status': status, 'stale': True}
        return props, {'fetched_at': None, 'from_network': True,
                       'status': status, 'stale': True}

    save_error = save_snapshot(props, status, now_utc, path)
    return props, {'fetched_at': now_utc, 'from_network': True, 'status': status,
                   'stale': False, 'save_error': save_error}


def weekly_summary(props):
    """One row per book: how many lines, players and distinct markets."""
    if props is None or props.empty:
        return pd.DataFrame(columns=['Book', 'Lines', 'Players', 'Markets'])
    scorable = props[props['scorable'].astype(bool)] if 'scorable' in props else props
    rows = []
    for book, chunk in scorable.groupby('provider'):
        rows.append({'Book': book, 'Lines': len(chunk),
                     'Players': chunk['player'].nunique(),
                     'Markets': chunk['market'].nunique()})
    return pd.DataFrame(rows).sort_values('Lines', ascending=False).reset_index(drop=True)


def weekly_consensus(props):
    """
    One row per player and stat, with every book's line side by side.

    This is the shape a person actually reads: "what does each book think
    Mahomes throws for", not a thousand rows of provider/market/line. The
    median across books is the consensus, and the spread is the disagreement
    - which is the interesting column, because a stat every book agrees on
    is settled and one they don't is where an edge would live.
    """
    if props is None or props.empty:
        return pd.DataFrame()
    scorable = props[props['scorable'].astype(bool)].copy()
    if scorable.empty:
        return pd.DataFrame()
    wide = scorable.pivot_table(index=['player', 'team', 'market'],
                                columns='provider', values='line', aggfunc='median')
    books = list(wide.columns)
    wide['Consensus'] = wide[books].median(axis=1)
    wide['Books'] = wide[books].notna().sum(axis=1)
    wide['Spread'] = (wide[books].max(axis=1) - wide[books].min(axis=1)).round(2)
    out = wide.reset_index().rename(columns={'player': 'Player', 'team': 'Team',
                                             'market': 'Market'})
    ordered = ['Player', 'Team', 'Market', 'Consensus', 'Books', 'Spread'] + books
    return out[ordered].sort_values(['Player', 'Market']).reset_index(drop=True)
