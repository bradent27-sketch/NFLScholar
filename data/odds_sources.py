"""
Sportsbook and DFS prop lines, normalized into one schema.

WHAT THIS IS FOR. The market prices players too, and it prices them with
money rather than with opinion. A season-long over/under on a receiver's
yards is a projection published by people who lose money when it's wrong,
which makes it the single best external check on this app's own projection
model - and, read the other way, the place to look for spots where the model
disagrees with the market strongly enough to bet on.

WHICH SOURCE ANSWERS WHICH QUESTION. This matters more than it looks, and it
is the whole reason this module exists alongside the existing Odds API
integration rather than replacing it:

  The Odds API (data/loaders.py, already integrated) is a licensed
  aggregator covering many books. Its NFL player props are PER-EVENT - you
  ask for one game and get that game's lines. That is the right tool for
  in-season weekly work and for FanDuel/DraftKings pricing, and the wrong
  tool for a draft board, because a draft board needs SEASON-LONG numbers
  and those aren't in the per-event feed.

  Underdog posts SEASON-LONG player over/unders through the preseason -
  "1,275.5 receiving yards", for the whole year. That is exactly the shape a
  draft projection needs, and it is why this module exists.

  PrizePicks posts season-long lines too, in a different format.

WHAT IS DELIBERATELY NOT HERE. No bot-detection evasion. No
undetected_chromedriver, no stealth browser patched to defeat a Cloudflare
interstitial, no rotating residential proxies to outrun a Datadome IP ban.
Those techniques exist to defeat an access control the operator put up on
purpose, and a personal fantasy tool is not a reason to break one. Every
adapter here makes ordinary HTTPS requests, identifies itself honestly, and
treats a block as a "no" rather than as a puzzle:

  Underdog serves this data from a public JSON endpoint their own web app
  calls. Plain requests work.

  PrizePicks sits behind Cloudflare and may refuse. When it does, the
  adapter says so and stops. It does not retry in a costume.

  FanDuel sits behind Datadome and is not scraped here at all. It doesn't
  need to be: The Odds API carries FanDuel's lines under license, which is
  the same data through a door that's open. See odds_api_bookmakers().

RATE LIMITS AND CACHING. Every fetch is cached for 30 minutes. Season-long
lines move slowly - they are not a live in-game feed - and hammering someone
else's endpoint for data that changes twice a week is both rude and
pointless.
"""
import json
import re

import pandas as pd
import requests
import streamlit as st

from data.utils import clean_name_exact

# Half an hour. Season-long lines move on the timescale of news, not seconds.
FETCH_TTL = 1800
FETCH_TIMEOUT = 20

# An honest User-Agent. This is a personal analytics tool making a handful of
# requests an hour, and it says so rather than impersonating a browser -
# if an operator wants to refuse this traffic, they are entitled to, and the
# adapter's job is to accept that answer.
USER_AGENT = 'NFLScholar/1.0 (personal fantasy analytics; non-commercial)'
DEFAULT_HEADERS = {
    'User-Agent': USER_AGENT,
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
}

UNDERDOG_LINES_URL = 'https://api.underdogfantasy.com/beta/v5/over_under_lines'
PRIZEPICKS_PROJECTIONS_URL = 'https://api.prizepicks.com/projections'
# PrizePicks' league id for the NFL. Their /leagues endpoint returns the
# mapping; 9 is NFL and is passed as a query param to avoid pulling every
# sport's board when only football is wanted.
PRIZEPICKS_NFL_LEAGUE_ID = 9


# ---------------------------------------------------------------------------
# Team standardization
# ---------------------------------------------------------------------------

# Sportsbooks write team names every way there is: "KC", "Kansas City",
# "Chiefs", "Kansas City Chiefs", and the three legacy abbreviations that
# nflverse spells differently ("JAX" vs "JAC", "LA" vs "LAR", "WSH" vs
# "WAS"). Everything here resolves to config.TEAM_CONFIG's key, which is what
# every other table in this app is keyed on - a prop that can't be joined to
# a player's row is worth nothing.
#
# Built from TEAM_CONFIG rather than typed out, so a future abbreviation
# change is a one-line edit there instead of a silent join failure here.
_TEAM_ALIAS_OVERRIDES = {
    'JAC': 'JAX', 'JAG': 'JAX', 'JAGUARS': 'JAX',
    'LAR': 'LA', 'RAMS': 'LA', 'STL': 'LA',
    'LACHARGERS': 'LAC', 'SD': 'LAC', 'CHARGERS': 'LAC',
    'WSH': 'WAS', 'WFT': 'WAS', 'COMMANDERS': 'WAS', 'REDSKINS': 'WAS',
    'OAK': 'LV', 'RAIDERS': 'LV', 'LASVEGAS': 'LV',
    'NWE': 'NE', 'NOR': 'NO', 'TAM': 'TB', 'SFO': 'SF',
    'KAN': 'KC', 'GNB': 'GB', 'ARZ': 'ARI', 'BLT': 'BAL',
    'CLV': 'CLE', 'HST': 'HOU',
}


def _build_team_lookup():
    """
    Every unambiguous spelling of a team -> its abbreviation.

    Cities are registered too ("Kansas City"), because books really do write
    them that way - but only where the city names exactly one team. "New
    York" and "Los Angeles" name two each, so they resolve to nothing rather
    than to whichever happened to be inserted last. That is the whole
    difference between a prop that visibly fails to join and one that
    silently joins to the wrong player.
    """
    from config import TEAM_CONFIG
    lookup = {}
    cities = {}
    for abbr, meta in TEAM_CONFIG.items():
        full = meta['name']
        nickname = full.split()[-1]
        city = ' '.join(full.split()[:-1])
        for form in (abbr, full, nickname):
            lookup[_team_key(form)] = abbr
        cities.setdefault(_team_key(city), set()).add(abbr)
    for city_key, abbrs in cities.items():
        if len(abbrs) == 1 and city_key not in lookup:
            lookup[city_key] = next(iter(abbrs))
    for alias, abbr in _TEAM_ALIAS_OVERRIDES.items():
        lookup[_team_key(alias)] = abbr
    return lookup


def _team_key(value):
    return re.sub(r'[^a-z]', '', str(value).lower())


_TEAM_LOOKUP = None


def standardize_team(value):
    """
    Any spelling of a team -> this app's abbreviation, or '' if unresolvable.

    Returns '' rather than guessing. "New York" is genuinely ambiguous
    between the Giants and the Jets, and a prop joined to the wrong team is
    worse than a prop that didn't join at all - the second one is visibly
    missing, the first one is silently wrong.
    """
    global _TEAM_LOOKUP
    if _TEAM_LOOKUP is None:
        _TEAM_LOOKUP = _build_team_lookup()
    if value is None:
        return ''
    return _TEAM_LOOKUP.get(_team_key(value), '')


# ---------------------------------------------------------------------------
# Stat normalization
# ---------------------------------------------------------------------------

# Provider stat label -> this app's stat column (data.draft_projections'
# PROJECTED_STATS vocabulary), so a line can be scored under the user's own
# league settings by the same score_stats() every other number on the board
# goes through.
#
# Keyed on a squashed lowercase form so "Passing Yards", "passing_yards" and
# "pass_yds" all land on the same entry without three dictionary rows each.
STAT_ALIASES = {
    'passingyards': 'passing_yards', 'passyds': 'passing_yards', 'passyards': 'passing_yards',
    'passingtds': 'passing_tds', 'passingtouchdowns': 'passing_tds', 'passtds': 'passing_tds',
    'passingattempts': 'attempts', 'passattempts': 'attempts',
    'interceptions': 'passing_interceptions', 'passinginterceptions': 'passing_interceptions',
    'intsthrown': 'passing_interceptions',

    'rushingyards': 'rushing_yards', 'rushyds': 'rushing_yards', 'rushyards': 'rushing_yards',
    'rushingtds': 'rushing_tds', 'rushingtouchdowns': 'rushing_tds', 'rushtds': 'rushing_tds',
    'rushingattempts': 'carries', 'carries': 'carries', 'rushattempts': 'carries',

    'receivingyards': 'receiving_yards', 'recyds': 'receiving_yards', 'recyards': 'receiving_yards',
    'receivingtds': 'receiving_tds', 'receivingtouchdowns': 'receiving_tds', 'rectds': 'receiving_tds',
    'receptions': 'receptions', 'recs': 'receptions', 'catches': 'receptions',
    'targets': 'targets',
}

# Lines whose stat is a COMBINATION the projection model doesn't carry as a
# single column. Kept out of the scoring path entirely rather than
# approximated: "rush+rec yards" is a real market and a useful thing to show,
# but silently mapping it onto rushing_yards would corrupt the projection it
# feeds. Recognized so they can be labelled and displayed, never scored.
COMBO_STATS = {
    'rushingreceivingyards': 'rush_rec_yards',
    'rushrecyds': 'rush_rec_yards',
    'passingrushingyards': 'pass_rush_yards',
    'fantasypoints': 'fantasy_points',
    'fantasyscore': 'fantasy_points',
    'pointsplusreboundsplusassists': None,
}


def _stat_key(value):
    return re.sub(r'[^a-z]', '', str(value).lower())


def normalize_stat(label):
    """
    Provider stat label -> (our column name, is_scorable).

    is_scorable is False for combination markets and anything unrecognized,
    which is what keeps an unknown market out of the fantasy-point math
    instead of quietly landing on the wrong column.
    """
    key = _stat_key(label)
    if key in STAT_ALIASES:
        return STAT_ALIASES[key], True
    if key in COMBO_STATS:
        return COMBO_STATS[key], False
    return None, False


# ---------------------------------------------------------------------------
# The normalized record
# ---------------------------------------------------------------------------

# Every adapter returns a frame with exactly these columns, so downstream
# code never branches on which book a line came from.
PROP_COLUMNS = [
    'provider',     # 'Underdog' / 'PrizePicks' / 'The Odds API'
    'player',       # display name as the provider wrote it
    'player_key',   # clean_name_exact form, for joining to the board
    'team',         # standardized abbreviation, or ''
    'position',
    'market',       # our stat column name, or the provider's label if unmapped
    'market_raw',   # always the provider's own label, for debugging
    'scorable',     # True when 'market' is a real PROJECTED_STATS column
    'line',         # the over/under number
    'over_payout',  # payout multiplier / decimal odds where published
    'under_payout',
    'period',       # 'season' or 'game'
    'source_id',
]


def _empty_props():
    return pd.DataFrame({c: pd.Series(dtype='object') for c in PROP_COLUMNS})


def _finalize(rows):
    """Rows of dicts -> the canonical frame, with keys and types settled."""
    if not rows:
        return _empty_props()
    df = pd.DataFrame(rows)
    for col in PROP_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df['player'] = df['player'].astype(str).str.strip()
    df = df[df['player'].ne('') & df['player'].str.lower().ne('nan')]
    if df.empty:
        return _empty_props()
    df['player_key'] = clean_name_exact(df['player'])
    df['line'] = pd.to_numeric(df['line'], errors='coerce')
    df = df.dropna(subset=['line'])
    df['scorable'] = df['scorable'].fillna(False).astype(bool)
    return df[PROP_COLUMNS].reset_index(drop=True)


def _get_json(url, params=None, headers=None):
    """
    One HTTPS GET, returning (payload, error). Never raises.

    A blocked or rate-limited endpoint is an ordinary outcome here, not an
    exception - the caller's job is to show a message and carry on with the
    sources that did load, exactly as the ADP and ECR fetches already do.
    """
    try:
        resp = requests.get(url, params=params or {},
                            headers={**DEFAULT_HEADERS, **(headers or {})},
                            timeout=FETCH_TIMEOUT)
    except Exception as exc:
        return None, f"Couldn't reach {url}: {exc}"
    if resp.status_code == 200:
        try:
            return resp.json(), None
        except ValueError:
            return None, f"{url} returned 200 but not JSON (got {resp.text[:120]!r})"
    if resp.status_code in (403, 401):
        return None, (f"{url} refused the request ({resp.status_code}). This source "
                      "blocks automated access; nothing further is attempted.")
    if resp.status_code == 429:
        return None, f"{url} rate-limited the request (429). Try again later."
    return None, f"{url} returned {resp.status_code}: {resp.text[:200]}"


# ---------------------------------------------------------------------------
# Underdog
# ---------------------------------------------------------------------------

def parse_underdog_payload(payload):
    """
    Underdog's over/under payload -> normalized props.

    Their response is RELATIONAL, not a flat list, and the join is the part
    worth getting right:

        players[]           id, first_name, last_name, position_id, team_id
        appearances[]       id, player_id, match_id, match_type
        over_under_lines[]  stat_value (the line), options[], and
                            over_under.appearance_stat.{appearance_id, stat}

    So a line names an APPEARANCE, an appearance names a PLAYER, and only the
    player carries the name. Two hops, and the middle one is easy to miss
    because the line object superficially looks self-contained.

    Pure function on purpose: this is where every realistic bug lives (a
    renamed key, a missing nesting level), and keeping the HTTP call out of
    it means it can be tested against a recorded fixture without a network.
    """
    if not isinstance(payload, dict):
        return _empty_props(), "Underdog returned an unexpected payload shape."

    players = {str(p.get('id')): p for p in payload.get('players') or []}
    appearances = {str(a.get('id')): a for a in payload.get('appearances') or []}
    lines = payload.get('over_under_lines') or []
    if not players or not lines:
        return _empty_props(), "Underdog returned no player lines."

    # match_id -> match_type, so a season-long line can be told apart from a
    # single-game one. Season-long is the whole point for a draft board.
    rows = []
    for line in lines:
        if str(line.get('status', '')).lower() == 'suspended':
            continue
        over_under = line.get('over_under') or {}
        stat_ref = over_under.get('appearance_stat') or {}
        appearance_id = str(stat_ref.get('appearance_id') or '')
        appearance = appearances.get(appearance_id)
        if not appearance:
            continue
        player = players.get(str(appearance.get('player_id') or ''))
        if not player:
            continue

        market, scorable = normalize_stat(stat_ref.get('stat'))
        payouts = {}
        for option in line.get('options') or []:
            choice = str(option.get('choice', '')).lower()
            side = {'higher': 'over', 'lower': 'under'}.get(choice, choice)
            payouts[side] = pd.to_numeric(option.get('payout_multiplier'), errors='coerce')

        name = ' '.join(str(player.get(k) or '').strip()
                        for k in ('first_name', 'last_name')).strip()
        rows.append({
            'provider': 'Underdog',
            'player': name,
            'team': standardize_team(player.get('team_id') or appearance.get('team_id')),
            'position': str(player.get('position_id') or '').upper(),
            'market': market or str(stat_ref.get('stat') or ''),
            'market_raw': str(stat_ref.get('stat') or ''),
            'scorable': bool(scorable),
            'line': line.get('stat_value'),
            'over_payout': payouts.get('over'),
            'under_payout': payouts.get('under'),
            'period': _underdog_period(appearance),
            'source_id': str(line.get('id') or ''),
        })
    props = _finalize(rows)
    if props.empty:
        return props, "Underdog returned lines but none could be joined to a player."
    return props, None


def _underdog_period(appearance):
    """
    'season' for a season-long line, 'game' for a single-game one.

    Underdog labels this on the appearance's match_type. The exact string has
    changed before, so this matches on a substring rather than an equality
    test - a new label like 'season_long_v2' should still read as a season.
    """
    match_type = str(appearance.get('match_type') or '').lower()
    return 'season' if 'season' in match_type else 'game'


@st.cache_data(ttl=FETCH_TTL, show_spinner=False)
def fetch_underdog_lines():
    """
    Underdog's current over/under board.

    NOT LIVE-TESTED FROM THE DEV ENVIRONMENT: the sandbox this was written in
    blocks every sportsbook host at the network layer, so the request path
    here is verified only against a recorded payload. The parse layer, which
    is where the bugs actually are, is fully tested (see
    tests/test_odds_sources.py). Run scripts/check_odds_sources.py on a
    normal network to confirm the live shape still matches.
    """
    payload, err = _get_json(UNDERDOG_LINES_URL)
    if err:
        return _empty_props(), err
    return parse_underdog_payload(payload)


# ---------------------------------------------------------------------------
# PrizePicks
# ---------------------------------------------------------------------------

def parse_prizepicks_payload(payload):
    """
    PrizePicks' projections payload -> normalized props.

    Theirs is JSON:API, which splits the answer in two: `data` holds the
    lines, each carrying only RELATIONSHIP POINTERS, and `included` holds the
    referenced objects (the players, the leagues) in a flat heterogeneous
    list keyed by (type, id). So the player's name is never on the line
    itself - it has to be looked up.
    """
    if not isinstance(payload, dict):
        return _empty_props(), "PrizePicks returned an unexpected payload shape."

    included = {}
    for item in payload.get('included') or []:
        included[(str(item.get('type')), str(item.get('id')))] = item.get('attributes') or {}

    rows = []
    for item in payload.get('data') or []:
        attrs = item.get('attributes') or {}
        rel = item.get('relationships') or {}
        player_ref = ((rel.get('new_player') or rel.get('player') or {}).get('data')) or {}
        player = included.get((str(player_ref.get('type')), str(player_ref.get('id')))) or {}

        market, scorable = normalize_stat(attrs.get('stat_type'))
        name = str(player.get('display_name') or player.get('name') or '').strip()
        rows.append({
            'provider': 'PrizePicks',
            'player': name,
            'team': standardize_team(player.get('team') or attrs.get('team')),
            'position': str(player.get('position') or '').upper(),
            'market': market or str(attrs.get('stat_type') or ''),
            'market_raw': str(attrs.get('stat_type') or ''),
            'scorable': bool(scorable),
            'line': attrs.get('line_score'),
            # PrizePicks is a pick'em product: both sides pay the same, so
            # there is no per-side price to record. Left blank rather than
            # filled with a fake 1.0, which would read as a real quote.
            'over_payout': None,
            'under_payout': None,
            'period': _prizepicks_period(attrs),
            'source_id': str(item.get('id') or ''),
        })
    props = _finalize(rows)
    if props.empty:
        return props, "PrizePicks returned no usable player lines."
    return props, None


def _prizepicks_period(attrs):
    """Season-long lines carry a season-ish flag on the projection itself."""
    blob = ' '.join(str(attrs.get(k) or '') for k in
                    ('odds_type', 'stat_type', 'description', 'period')).lower()
    return 'season' if 'season' in blob else 'game'


@st.cache_data(ttl=FETCH_TTL, show_spinner=False)
def fetch_prizepicks_lines(league_id=PRIZEPICKS_NFL_LEAGUE_ID):
    """
    PrizePicks' current projections, via one ordinary HTTPS request.

    THIS WILL SOMETIMES BE REFUSED, AND THAT IS THE DESIGN. PrizePicks sits
    behind Cloudflare. When the request comes back 403 the adapter reports it
    and stops - it does not retry behind a patched headless browser, spoof a
    browser fingerprint, or rotate IPs to get around the block. Those are
    techniques for defeating an access control the operator chose to put up,
    and "I want the data for my fantasy draft" is not a good enough reason to
    defeat one.

    If this source is blocked for you, Underdog carries the same kind of
    season-long lines and serves them from an open endpoint.
    """
    payload, err = _get_json(PRIZEPICKS_PROJECTIONS_URL,
                             params={'league_id': league_id, 'per_page': 1000})
    if err:
        return _empty_props(), err
    return parse_prizepicks_payload(payload)


# ---------------------------------------------------------------------------
# The Odds API - what's actually in it
# ---------------------------------------------------------------------------

def odds_api_bookmakers(payload):
    """
    Which books a fetched Odds API payload actually contained.

    This answers "what books does my key see?" from the user's own data
    rather than from a docs page, which matters because coverage varies by
    region, by plan tier and by how close the game is - the marketing list
    and what a given key returns at a given moment are not the same list.

    Returns a frame of (key, title, events, markets) sorted by coverage.
    """
    if not payload:
        return pd.DataFrame(columns=['key', 'title', 'events', 'markets'])
    seen = {}
    for event in payload if isinstance(payload, list) else [payload]:
        for book in (event or {}).get('bookmakers') or []:
            key = str(book.get('key') or '')
            entry = seen.setdefault(key, {'key': key, 'title': book.get('title') or key,
                                          'events': 0, 'markets': set()})
            entry['events'] += 1
            for market in book.get('markets') or []:
                entry['markets'].add(str(market.get('key') or ''))
    rows = [{'key': v['key'], 'title': v['title'], 'events': v['events'],
             'markets': ', '.join(sorted(v['markets']))} for v in seen.values()]
    if not rows:
        return pd.DataFrame(columns=['key', 'title', 'events', 'markets'])
    return pd.DataFrame(rows).sort_values('events', ascending=False).reset_index(drop=True)


def parse_odds_api_props(payload, provider='The Odds API', bookmaker=None):
    """
    A per-event Odds API player-prop response -> the same normalized frame.

    Included so FanDuel and DraftKings lines can sit in the same table as the
    DFS books without being scraped: The Odds API carries them under license.
    `bookmaker` filters to one book's key (e.g. 'fanduel').

    These are PER-GAME lines. They are labelled period='game' and are not
    used for season projections; the comparison layer only ever scores
    season-long lines.
    """
    if not payload:
        return _empty_props(), "No player-prop payload."
    events = payload if isinstance(payload, list) else [payload]
    rows = []
    for event in events:
        for book in (event or {}).get('bookmakers') or []:
            key = str(book.get('key') or '')
            if bookmaker and key != bookmaker:
                continue
            for market in book.get('markets') or []:
                market_name, scorable = normalize_stat(
                    str(market.get('key') or '').replace('player_', ''))
                for outcome in market.get('outcomes') or []:
                    rows.append({
                        'provider': f"{provider}: {book.get('title') or key}",
                        'player': outcome.get('description') or outcome.get('name'),
                        'team': '',
                        'position': '',
                        'market': market_name or str(market.get('key') or ''),
                        'market_raw': str(market.get('key') or ''),
                        'scorable': bool(scorable),
                        'line': outcome.get('point'),
                        'over_payout': outcome.get('price') if str(
                            outcome.get('name', '')).lower() == 'over' else None,
                        'under_payout': outcome.get('price') if str(
                            outcome.get('name', '')).lower() == 'under' else None,
                        'period': 'game',
                        'source_id': str(event.get('id') or ''),
                    })
    props = _finalize(rows)
    return props, (None if not props.empty else "No player props in that payload.")


# ---------------------------------------------------------------------------
# Combining
# ---------------------------------------------------------------------------

def combine_props(*frames):
    """
    Stack several providers' normalized frames into one.

    Deliberately does NOT average across books into a consensus line. Two
    books' numbers for the same player are the interesting comparison, and
    collapsing them to a mean throws away the disagreement that a bet is
    made out of.
    """
    usable = [f for f in frames if f is not None and not f.empty]
    if not usable:
        return _empty_props()
    return pd.concat(usable, ignore_index=True)[PROP_COLUMNS]


def load_props_fixture(path):
    """Read a recorded payload off disk - the offline path for tests."""
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)
