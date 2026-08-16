"""
One registry of every manual import in Draft HQ, and what state each one is
actually in.

The problem this solves: the uploads were spread across three separate
corners of the settings panel (rankings and ADP under "Board & market", the
FFA export and cheat sheet under one heading, five sportsbook payloads under
another), and none of them said whether the file you dropped in last August
was still the one being used. "Did that import take?" was answerable only by
reading the board and guessing.

Every import here reports the same five things, so one table can show all of
them: whether anything is loaded, which file it came from, when it landed,
how big it is, and what was actually parsed out of it.

ON "THE DATE ON THE UPLOADED FILE": a browser file input does not send the
file's own modification time, and Streamlit's UploadedFile has no field for
it - so the original file's date genuinely cannot be recovered. What IS
knowable is when it was saved here, which is what `uploaded` reports, and
several sources also carry a date INSIDE the payload (a book's own
timestamp), which is strictly better evidence of freshness than either. That
one is reported as `data_date` where the source publishes it.

Session-held imports (rankings, ADP) are a special case: they are parsed and
kept in session state rather than written to disk, on purpose - see the call
sites - so they have no file and no mtime, and their upload time is recorded
alongside them instead.
"""
import datetime
import json
import os

import pandas as pd

# Session-state keys where an import's upload timestamp is recorded, for the
# two imports that live in memory rather than on disk.
UPLOAD_TIME_KEYS = {
    'ecr': '_import_time_ecr',
    'adp': '_import_time_adp',
}

# group -> ordered list of (key, label, what it's for). The UI renders one
# section per group, in this order, and the status table follows the same
# order, so a row in the table and its uploader are never far apart.
IMPORT_GROUPS = [
    ('Rankings & ADP', [
        ('ecr', 'FantasyPros rankings', 'Consensus rank + expert spread. Overrides the live mirror.'),
        ('adp', 'ADP export', 'Average draft position. Overrides every live source.'),
    ]),
    ('Projections & notes', [
        ('ffa', 'Fantasy Football Advice', "Their projected STAT LINE, re-scored under your league, plus notes."),
        ('cheatsheet', 'FantasyPros cheat sheet', 'Auction dollar values, their tiers, and a note per player.'),
        ('fp_players', 'FantasyPros player array', 'Lets the cheat sheet resolve ~100% of players instead of ~80%.'),
    ]),
    ('Sportsbooks (season-long)', [
        ('Underdog', 'Underdog', 'over_under_lines JSON'),
        ('PrizePicks', 'PrizePicks', 'projections JSON'),
        ('DraftKings', 'DraftKings', 'player-futures JSON (one file per stat)'),
        ('FanDuel', 'FanDuel', 'NFL page JSON'),
        ('Pinnacle', 'Pinnacle', 'matchups JSON'),
    ]),
]

ALL_IMPORT_KEYS = [key for _group, entries in IMPORT_GROUPS for key, _l, _d in entries]


def _file_status(path):
    """Presence, save time and size for a saved import, or a blank record."""
    if not path or not os.path.exists(path):
        return {'loaded': False, 'path': path, 'uploaded': None, 'bytes': None}
    try:
        stat = os.stat(path)
    except OSError:
        return {'loaded': False, 'path': path, 'uploaded': None, 'bytes': None}
    return {
        'loaded': True, 'path': path, 'bytes': int(stat.st_size),
        'uploaded': datetime.datetime.fromtimestamp(stat.st_mtime),
    }


def _book_status(provider, session=None):
    from data.odds_sources import SAVED_PAYLOADS, load_saved_book_payload

    path = SAVED_PAYLOADS.get(provider, (None, None))[0]
    status = _file_status(path)
    status['label'] = provider
    if not status['loaded']:
        # A saved file isn't the only way a book can be feeding the board -
        # the "Fetch season-long sportsbook lines" toggle pulls live and
        # never touches disk. draft_hq._record_live_market_status mirrors
        # that fetch's own result into session state; check it here so this
        # table doesn't say "-" for a book that's actually loaded live.
        live = ((session or {}).get('dhq_live_book_status') or {}).get(provider)
        if live:
            status['loaded'] = True
            status['uploaded'] = live.get('fetched_at')
            status['live_fetch'] = True
            # _file_status always fills 'path' with where a save WOULD land,
            # loaded or not - clear it here so the File column doesn't claim
            # an on-disk file that was never actually written.
            status['path'] = None
            if live.get('error'):
                status['detail'] = f"Live fetch: {live['error']}"
                status['error'] = live['error']
            else:
                rows, season = live.get('rows', 0), live.get('season', 0)
                status['rows'] = rows
                status['season_lines'] = season
                status['detail'] = f"{rows:,} lines · {season:,} season-long (live fetch, not saved)"
                if season == 0 and rows:
                    status['detail'] += (
                        " — no SEASON-LONG lines, so Book Proj gets nothing from this")
            return status
        status['detail'] = 'Not imported — the live fetch is used if enabled.'
        return status

    # Parsed, not just counted. A right-shaped-but-wrong file (a WEEKLY
    # board saved over a season one, which parses cleanly and is wrong by
    # two orders of magnitude) is invisible to a byte count and obvious the
    # moment you look at how many season-long lines came out.
    try:
        payload = load_saved_book_payload(provider)
        from ui.tabs.draft_hq import _MARKET_PARSERS
        props, error = _MARKET_PARSERS[provider](payload)
    except Exception as exc:
        status['detail'] = f'Saved, but could not be parsed: {exc}'
        status['error'] = str(exc)
        return status

    if error and (props is None or props.empty):
        status['detail'] = f'Saved, but: {error}'
        status['error'] = error
        return status
    season = int((props['period'] == 'season').sum()) if not props.empty else 0
    status['rows'] = int(len(props))
    status['season_lines'] = season
    status['detail'] = f"{len(props):,} lines · {season:,} season-long"
    if error:
        status['detail'] += f" · {error}"
    if season == 0 and len(props):
        # Loud, because this is the failure that looks like success.
        status['detail'] += " — no SEASON-LONG lines, so Book Proj gets nothing from this file"
    return status


def _ffa_status():
    from data.ffa_import import FFA_IMPORT_PATH, load_ffa_import

    status = _file_status(FFA_IMPORT_PATH)
    status['label'] = 'Fantasy Football Advice'
    if not status['loaded']:
        status['detail'] = 'Not imported.'
        return status
    frame, error = load_ffa_import()
    status['rows'] = int(len(frame))
    status['detail'] = f"{len(frame):,} players" if not frame.empty else (error or 'No players found.')
    if error:
        status['error'] = error
    return status


def _cheatsheet_status(which):
    from data.fantasypros_import import CHEATSHEET_PATH, PLAYER_ARRAY_PATH

    path = CHEATSHEET_PATH if which == 'cheatsheet' else PLAYER_ARRAY_PATH
    status = _file_status(path)
    status['label'] = 'FantasyPros cheat sheet' if which == 'cheatsheet' else 'FantasyPros player array'
    if not status['loaded']:
        status['detail'] = 'Not imported.'
        return status
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        status['rows'] = _rough_count(payload)
        status['detail'] = f"{status['rows']:,} entries" if status['rows'] else 'Saved.'
    except Exception as exc:
        status['detail'] = f'Saved, but could not be read: {exc}'
        status['error'] = str(exc)
    return status


def _rough_count(payload):
    """Entry count for a saved JSON blob, whatever shape it arrived in."""
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list) and value:
                return len(value)
    return 0


def _session_status(key, session):
    """An import held in memory rather than on disk (rankings, ADP)."""
    from ui.tabs.draft_hq import ADP_UPLOAD_KEY, ECR_UPLOAD_KEY

    state_key = ECR_UPLOAD_KEY if key == 'ecr' else ADP_UPLOAD_KEY
    frame = session.get(state_key)
    status = {
        'label': 'FantasyPros rankings' if key == 'ecr' else 'ADP export',
        'loaded': frame is not None and not getattr(frame, 'empty', True),
        'path': None, 'bytes': None,
        'uploaded': session.get(UPLOAD_TIME_KEYS[key]),
        'session_only': True,
    }
    if not status['loaded']:
        status['detail'] = 'Not imported — the live source is used.'
        return status
    status['rows'] = int(len(frame))
    detail = f"{len(frame):,} players"
    if key == 'ecr' and 'ECR SD' in frame.columns:
        detail += ', with expert spread' if frame['ECR SD'].notna().any() else ', no expert spread'
    # Both a pasted CSV and a FantasyPros API pull land in the same session
    # slot (see ui.tabs.draft_hq._render_fantasypros_api_import) - this is
    # the one place that still distinguishes them, since the API pull spends
    # a real call against a 50/month budget and is worth being able to spot.
    source = session.get(f'_import_source_{key}')
    if source:
        detail += f' — via {source}'
    status['detail'] = detail
    return status


def import_status(key, session=None):
    """One import's full state. `session` is st.session_state (or a dict)."""
    session = session if session is not None else {}
    if key in ('ecr', 'adp'):
        return _session_status(key, session)
    if key == 'ffa':
        return _ffa_status()
    if key in ('cheatsheet', 'fp_players'):
        return _cheatsheet_status(key)
    return _book_status(key, session)


def all_import_status(session=None):
    """Every import, keyed the same way IMPORT_GROUPS lists them."""
    out = {}
    for key in ALL_IMPORT_KEYS:
        try:
            out[key] = import_status(key, session)
        except Exception as exc:
            # A broken import must never take the panel - and therefore the
            # whole tab - down with it. That is the exact failure this panel
            # exists to make visible.
            out[key] = {'label': key, 'loaded': False, 'path': None, 'uploaded': None,
                        'bytes': None, 'detail': f'Could not be checked: {exc}',
                        'error': str(exc)}
    return out


def format_age(when):
    """'3 days ago' next to a timestamp - the part that answers 'is this
    stale', which an absolute date makes you compute yourself."""
    if not when:
        return ''
    delta = datetime.datetime.now() - when
    seconds = delta.total_seconds()
    if seconds < 90:
        return 'just now'
    if seconds < 5400:
        return f"{int(seconds // 60)} min ago"
    if seconds < 172800:
        return f"{int(seconds // 3600)} hours ago"
    return f"{int(seconds // 86400)} days ago"


def format_size(n_bytes):
    if not n_bytes:
        return ''
    if n_bytes < 1024:
        return f"{n_bytes} B"
    if n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.0f} KB"
    return f"{n_bytes / 1024 ** 2:.1f} MB"


def status_table(session=None):
    """The whole registry as one frame, ready to render."""
    statuses = all_import_status(session)
    rows = []
    for group, entries in IMPORT_GROUPS:
        for key, label, purpose in entries:
            status = statuses.get(key, {})
            uploaded = status.get('uploaded')
            rows.append({
                'Group': group,
                'Source': label,
                'Loaded': '✅' if status.get('loaded') else '—',
                'Imported': uploaded.strftime('%Y-%m-%d %H:%M') if uploaded else '',
                'Age': format_age(uploaded),
                'Size': format_size(status.get('bytes')),
                'What came out of it': status.get('detail', ''),
                'File': status.get('path') or (
                    'held in this session only' if status.get('session_only')
                    else 'live fetch, not saved' if status.get('live_fetch')
                    else ''),
                'Purpose': purpose,
            })
    return pd.DataFrame(rows)
