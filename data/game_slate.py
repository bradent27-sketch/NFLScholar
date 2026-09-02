"""
The week's games, normalized into one frame the card renderer can index.

WHAT THE SOURCE SURVEY FOUND (§0.1 of the porting notes says do this before
choosing, and it changed the design twice):

nflverse's `games.csv` carries the whole schedule - past and future - in one
free, keyless file the app already downloads for strength-of-schedule and
Vegas lines. Measured on the real file:

  7,548 rows across every season; 272 rows for 2026, EVERY one of them with
  a kickoff time and a null score, running 2026-09-09 to 2027-01-10.

That measurement decides the architecture. The porting notes' §3 two-source
ladder exists because a published college file is a rebuild of games that
already happened, so future games need a live endpoint. The NFL schedule is
published in May and the file carries it, so THERE IS NO LIVE PATH HERE and
§3 is deleted rather than ported. One download serves every week of the
season.

TWO MORE THINGS THE FILE DECIDES, both of which delete a class of bug:

  `gameday` IS ALREADY THE LOCAL GAME DATE. The notes' §8.1 - the worst bug
  of the CBB port, where a night game stamped in UTC rendered on the wrong
  calendar day - cannot happen here, and that is measured rather than
  assumed: `gameday`'s actual weekday matches the file's own `weekday`
  column on 100.00% of 7,548 rows. Nothing in this module slices a UTC
  timestamp, so nothing needs to un-slice one.

  AN UNPLAYED GAME'S SCORE IS NULL, NOT ZERO. §8.3 says never gate score
  rendering on presence, because ESPN sends "0" before kickoff and 0-0 for a
  postponement. This file does neither: all 272 future 2026 games carry NaN.
  So presence IS the honest gate here - but the reason is a measurement, not
  an assumption, and `Status` is still carried so the renderer keeps the same
  shape as the other two sports.
"""
import datetime
import html
import re

import pandas as pd
import streamlit as st

from config import TEAM_CONFIG

# Eastern, because the source is Eastern and the NFL schedules in it. The
# porting notes moved CBB to Central and made the timezone a knob; here the
# knob's honest default is the one the data already uses, which means a
# kickoff needs no conversion at all and cannot drift by an hour.
SLATE_DISPLAY_TZ = 'America/New_York'

SLATE_CSV = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'

# ESPN keys NFL marks by LOWERCASE ABBREVIATION, not by the numeric id the
# college leagues use - confirmed against a real URL found in live data
# (`.../i/teamlogos/nfl/500/cin.png`) rather than inferred from the college
# pattern, which the porting notes specifically warn does not carry over.
#
# The standard mark is used rather than ESPN's `500-dark` variant. This app
# has no light mode, so dark would be the better fit - but the dark path
# could not be verified from the environment this was written in, and an
# unverified URL renders as a broken image rather than degrading. The CSS
# drop-shadow lifts dark marks off the dark surface instead. To switch,
# change one string here and check the images actually load.
LOGO_URL = 'https://a.espncdn.com/i/teamlogos/nfl/500/{abbr}.png'

# Playoff rounds, for the card headline. Regular-season games get none.
ROUND_LABELS = {
    'WC': 'Wild Card', 'DIV': 'Divisional', 'CON': 'Conference Championship',
    'SB': 'Super Bowl', 'PRE': 'Preseason',
}

SLATE_COLUMNS = [
    # 'Game Id' is nflverse's own schedule id ("2025_01_DAL_PHI"), carried
    # so a card can open a box score and so any per-game stat elsewhere in
    # the app can resolve back to the game it came from. It is the SAME id
    # stats_player_week_{season}.csv uses - measured at 285/285 games in
    # both directions on the real 2025 files. See data/box_score.py.
    'Game Id',
    'Away', 'Home', 'Away Pts', 'Home Pts', 'Winner',
    'Date', 'Date Display', 'Kickoff', 'Kickoff Long', 'Time TBD',
    'Status', 'Status Detail', 'Played', 'Live',
    'Away Color', 'Home Color', 'Away Logo', 'Home Logo',
    'Venue', 'Neutral Site', 'Headline', 'Matchup',
    'Week', 'Season Type', '_source',
    # nflverse carries these on `games.csv` for future weeks too - positive
    # `spread_line` means the HOME team is favored (see data/odds_market.py).
    # NaN until a line is posted; the card renders nothing in that case.
    'Spread Line', 'Total Line',
]


def _col(df, name, default=None):
    """
    A column, or a column-shaped default.

    nflverse's schedule frame has gained and lost columns across seasons -
    `ftn`, `pff`, `stadium_id` and the betting fields are not all present in
    every era - and indexing one directly hard-crashes the tab on an
    otherwise perfectly good season.
    """
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index)


def _hex_color(value):
    """
    A validated 6-digit hex colour, or None.

    These are interpolated straight into a <style> block and an inline style
    attribute, and the `var(--x, fallback)` in the stylesheet only rescues a
    property that is ABSENT - not one that is present and malformed. A bad
    value would silently break the whole card's style block.
    """
    text = str(value or '').strip().lstrip('#')
    if not re.fullmatch(r'[0-9a-fA-F]{6}', text):
        return None
    return f'#{text.lower()}'


def team_color(abbr):
    entry = TEAM_CONFIG.get(str(abbr or '').upper())
    return _hex_color(entry.get('color')) if entry else None


def team_logo(abbr):
    """
    A team's mark, or None.

    OMITTED RATHER THAN None-MAPPED at the call site: every consumer does a
    truthiness check and renders nothing on a miss. Relocated franchises
    (OAK, SD, STL appear in seasons before 2020 and in no map this app has)
    land here and render as text, which is the correct outcome rather than a
    broken image.
    """
    key = str(abbr or '').upper()
    if key not in TEAM_CONFIG:
        return None
    return LOGO_URL.format(abbr=key.lower())


def _zone_label(day, clock):
    """
    EDT or EST for this kickoff, derived rather than hardcoded.

    An NFL season straddles the DST boundary in the direction opposite to
    college basketball's: September is EDT, January is EST. One fixed label
    would be wrong for the entire postseason.

    Falls back to saying UTC rather than lying. On Windows there is no
    system tz database, so ZoneInfo raises without the `tzdata` package -
    and a silent fallback that still printed "ET" would render every kickoff
    correctly labelled and hours wrong, which is the worst possible failure
    for a schedule because nothing about it looks broken.
    """
    try:
        from zoneinfo import ZoneInfo
        stamp = datetime.datetime.combine(day, clock, tzinfo=ZoneInfo(SLATE_DISPLAY_TZ))
        return stamp.strftime('%Z') or 'ET'
    except Exception:
        return 'ET'


def format_kickoff(day, clock, tbd=False, long=False):
    """
    "Sun 1:00 PM EDT" / "Sunday at 1:00 PM EDT".

    A TBD game takes its weekday from the DATE, never from a time - the
    porting notes' trap is a placeholder timestamp converting backward
    across midnight, and the defence is to never route a placeholder through
    a time conversion at all.
    """
    if day is None:
        return 'TBD'
    if tbd or clock is None:
        return f"{day.strftime('%A')} · time TBD" if long else 'TBD'
    label = _zone_label(day, clock)
    stamp = datetime.datetime.combine(day, clock)
    pattern = '%A at %I:%M %p' if long else '%a %I:%M %p'
    # Strip the leading zero on the HOUR before appending the zone label, so
    # a label that happens to contain " 0" can't be damaged by the replace.
    return stamp.strftime(pattern).replace(' 0', ' ') + f' {label}'


def _parse_clock(value):
    text = str(value or '').strip()
    if not text or text.lower() in ('nan', 'none'):
        return None
    for fmt in ('%H:%M', '%H:%M:%S', '%I:%M %p'):
        try:
            return datetime.datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def _raw_slate():
    """The whole published file. Daily - a schedule must not be a week stale."""
    try:
        return pd.read_csv(SLATE_CSV, low_memory=False), None
    except Exception as exc:
        return pd.DataFrame(), f"Couldn't download the schedule: {exc}"


@st.cache_data(ttl=86400, show_spinner=False)
def season_slate(season):
    """
    One season, normalized once and sliced per view.

    Normalizing the whole season rather than per week is what keeps the week
    list and the week filter keyed off the SAME derived values - the porting
    notes' §3.1 point, and the thing that stops a game appearing in one
    week's count and another week's cards.
    """
    raw, err = _raw_slate()
    if raw is None or raw.empty:
        return pd.DataFrame(columns=SLATE_COLUMNS), err or 'The schedule file was empty.'

    df = raw[_col(raw, 'season') == int(season)].copy()
    if df.empty:
        return pd.DataFrame(columns=SLATE_COLUMNS), f'No {season} games in the schedule file.'

    days = pd.to_datetime(_col(df, 'gameday'), errors='coerce')
    clocks = [_parse_clock(v) for v in _col(df, 'gametime')]
    away_pts = pd.to_numeric(_col(df, 'away_score'), errors='coerce')
    home_pts = pd.to_numeric(_col(df, 'home_score'), errors='coerce')
    away = _col(df, 'away_team').astype(str)
    home = _col(df, 'home_team').astype(str)
    neutral = _col(df, 'location').astype(str).str.lower().eq('neutral')
    kinds = _col(df, 'game_type').astype(str)
    spread_line = pd.to_numeric(_col(df, 'spread_line'), errors='coerce')
    total_line = pd.to_numeric(_col(df, 'total_line'), errors='coerce')

    rows = []
    for i, (_, src) in enumerate(df.iterrows()):
        day = days.iloc[i]
        day = day.date() if pd.notna(day) else None
        clock = clocks[i]
        a_pts, h_pts = away_pts.iloc[i], home_pts.iloc[i]
        played = bool(pd.notna(a_pts) and pd.notna(h_pts))

        winner = None
        if played and a_pts != h_pts:
            winner = away.iloc[i] if a_pts > h_pts else home.iloc[i]

        venue = str(src.get('stadium') or '').strip() or None
        is_neutral = bool(neutral.iloc[i])
        kind = kinds.iloc[i]
        detail = 'Final'
        if played and str(_col(df, 'overtime').iloc[i]) in ('1', '1.0', 'True'):
            detail = 'Final/OT'

        rows.append({
            'Game Id': str(src.get('game_id') or ''),
            'Away': away.iloc[i], 'Home': home.iloc[i],
            'Away Pts': a_pts, 'Home Pts': h_pts, 'Winner': winner,
            'Date': day.isoformat() if day else '',
            'Date Display': day.strftime('%m/%d/%Y') if day else '',
            'Kickoff': format_kickoff(day, clock),
            'Kickoff Long': format_kickoff(day, clock, long=True),
            'Time TBD': clock is None,
            # No live path means no 'in' state can ever be observed here.
            # Carried anyway so the renderer matches the other two sports.
            'Status': 'post' if played else 'pre',
            'Status Detail': detail if played else 'Scheduled',
            'Played': played, 'Live': False,
            'Away Color': team_color(away.iloc[i]),
            'Home Color': team_color(home.iloc[i]),
            'Away Logo': team_logo(away.iloc[i]),
            'Home Logo': team_logo(home.iloc[i]),
            'Venue': venue, 'Neutral Site': is_neutral,
            'Headline': ROUND_LABELS.get(kind),
            'Matchup': f"{away.iloc[i]} {'vs' if is_neutral else '@'} {home.iloc[i]}",
            'Week': int(src.get('week')) if pd.notna(src.get('week')) else 0,
            'Season Type': kind,
            '_source': 'nflverse schedule',
            'Spread Line': spread_line.iloc[i],
            'Total Line': total_line.iloc[i],
        })

    out = pd.DataFrame(rows, columns=SLATE_COLUMNS)
    # Sort on the real date and time, never on a display string: "Sun 9:00
    # AM" sorts after "Sun 10:00 AM" as text because '9' > '1'.
    order = pd.to_datetime(
        [f"{r['Date']} {c.strftime('%H:%M') if c else '00:00'}"
         for (_, r), c in zip(out.iterrows(), clocks)],
        errors='coerce')
    out = (out.assign(_order=order)
              .sort_values(['Week', '_order'], kind='stable')
              .drop(columns='_order')
              .reset_index(drop=True))
    return out, None


def slate_weeks(season):
    """
    Every week that has games, ordered as played, with counts.

    `completed` drives the default selection. There is no second level of
    fallback here - the porting notes' two-level version exists for a sport
    where a week can be full of games none of which are analyzable, and NFL
    has no such thing. Written as one deliberate level rather than dropped
    by accident: a season with no completed week at all still has to land
    somewhere, which is what the `default=` covers.
    """
    games, err = season_slate(season)
    if games.empty:
        return [], err
    weeks = []
    for week, chunk in games.groupby('Week', sort=True):
        kind = chunk['Season Type'].iloc[0]
        label = ROUND_LABELS.get(kind) if kind != 'REG' else f'Week {week}'
        weeks.append({
            'week': int(week),
            'season_type': kind,
            'label': label or f'Week {week}',
            'games': int(len(chunk)),
            'completed': bool(chunk['Played'].all()),
            'any_played': bool(chunk['Played'].any()),
        })
    return weeks, None


def default_week_index(weeks):
    """
    The week to open on: the last one with a played game, else the first.

    Guarded deliberately. A brand-new season has no completed week, and
    opening on an empty screen is the failure the porting notes call out.
    """
    if not weeks:
        return 0
    return max((i for i, w in enumerate(weeks) if w['any_played']), default=0)


def load_slate(season, week):
    """One week's games. Returns (frame, error)."""
    games, err = season_slate(season)
    if games.empty:
        return games, err
    return games[games['Week'] == int(week)].reset_index(drop=True), None


def find_slate_game(season, game_id):
    """
    One game, resolved against the WHOLE SEASON rather than whatever week
    the tab is currently showing. Returns the row, or None.

    This is deliberate and load-bearing: the box-score panel is the
    destination for every cross-tab link in the app, and requiring the game
    to also survive the slate's current week selector would make links fail
    for a reason that has nothing to do with the game they name - click a
    week 3 game from a player's log while the slate sits on week 12 and
    nothing would open.
    """
    if not game_id:
        return None
    games, _err = season_slate(season)
    if games.empty or 'Game Id' not in games.columns:
        return None
    match = games[games['Game Id'].astype(str) == str(game_id)]
    return match.iloc[0] if not match.empty else None


def slate_source(season, week=None):
    """Which source served this view, for the footer."""
    games, _err = season_slate(season)
    if games.empty:
        return None
    return games['_source'].iloc[0]


def refresh_slate():
    """
    Clear EVERY cached layer behind the slate.

    Clearing only the download leaves every view still served from the
    normalized copy, which is a refresh button that visibly does nothing.
    """
    _raw_slate.clear()
    season_slate.clear()


def slate_team_bridge(games):
    """
    {slate code: this app's team code}, for seeding another tab.

    Measured: for 2020 onward the two namespaces are IDENTICAL - every code
    in the schedule file is a TEAM_CONFIG key and vice versa - so this is a
    pass-through for any season anyone will actually open. It exists for the
    seasons where they are not: OAK, SD and STL appear in older files and in
    no map here.

    A code that doesn't resolve is OMITTED, never mapped to None. Every call
    site does .get() and disables the button on a miss, and a None VALUE
    would sail straight through that check and seed a null.
    """
    if games is None or games.empty:
        return {}
    codes = set(games['Away'].astype(str)) | set(games['Home'].astype(str))
    return {code: code for code in codes if code in TEAM_CONFIG}


def team_abbrev_label(abbr):
    """Short label for a button. Falls back to the code itself."""
    return str(abbr or '').upper() or '?'


def team_display_name(abbr):
    entry = TEAM_CONFIG.get(str(abbr or '').upper())
    return entry.get('name') if entry else str(abbr or '')


def escape(value):
    return html.escape(str(value)) if value is not None else ''


def escape_attr(value):
    return html.escape(str(value), quote=True) if value is not None else ''
