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
import os
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

# Underdog versions this endpoint, and the version moves. The one that was
# current when this was written came from a reference implementation whose
# last commit was September 2024, so treating any single version as correct
# was going to break on a schedule nobody here controls.
#
# So it's a ladder rather than a constant: newest first, stopping at the
# first version that answers. A wrong guess costs one 404, and a 404 is much
# cheaper than an adapter that reports "no lines" forever because the number
# in a URL went up by one. The payload SHAPE has been stable across versions
# (the players/appearances/over_under_lines split), which is what makes this
# safe - if a future version changes the shape, parse_underdog_payload
# returns its "couldn't join" error rather than silently mis-parsing.
UNDERDOG_LINE_ENDPOINTS = [
    'https://api.underdogfantasy.com/beta/v7/over_under_lines',
    'https://api.underdogfantasy.com/beta/v6/over_under_lines',
    'https://api.underdogfantasy.com/beta/v5/over_under_lines',
    'https://api.underdogfantasy.com/beta/v4/over_under_lines',
    'https://api.underdogfantasy.com/beta/v3/over_under_lines',
]
UNDERDOG_LINES_URL = UNDERDOG_LINE_ENDPOINTS[2]
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
# Every key below was READ OFF A REAL PAYLOAD, not guessed. The first version
# of this table was guessed and it mapped 8 of 5,314 lines, because Underdog's
# season-long markets are prefixed `season_` and use `yds`/`rec` rather than
# the spelled-out forms convention suggested.
STAT_ALIASES = {
    # Season-long markets - the ones a draft board actually wants. These are
    # Underdog's real `stat` values as of the 2026 preseason.
    'seasonpassyards': 'passing_yards', 'seasonpasstds': 'passing_tds',
    'seasonrushyards': 'rushing_yards', 'seasonrushtds': 'rushing_tds',
    'seasonreceivingyards': 'receiving_yards', 'seasonrectds': 'receiving_tds',
    'seasonreceptions': 'receptions', 'seasonrecs': 'receptions',
    'seasonpassints': 'passing_interceptions', 'seasoninterceptions': 'passing_interceptions',
    'seasonrushattempts': 'carries', 'seasoncarries': 'carries',
    'seasontargets': 'targets',

    'passingyards': 'passing_yards', 'passyds': 'passing_yards', 'passyards': 'passing_yards',
    'passingtds': 'passing_tds', 'passingtouchdowns': 'passing_tds', 'passtds': 'passing_tds',
    'passingattempts': 'attempts', 'passattempts': 'attempts',
    'interceptions': 'passing_interceptions', 'passinginterceptions': 'passing_interceptions',
    'intsthrown': 'passing_interceptions', 'passingints': 'passing_interceptions',

    'rushingyards': 'rushing_yards', 'rushyds': 'rushing_yards', 'rushyards': 'rushing_yards',
    'rushingyds': 'rushing_yards',
    'rushingtds': 'rushing_tds', 'rushingtouchdowns': 'rushing_tds', 'rushtds': 'rushing_tds',
    'rushingattempts': 'carries', 'carries': 'carries', 'rushattempts': 'carries',

    # PrizePicks' NFL spellings, read off a real payload: "Rush Yards",
    # "Pass Yards", "Rec Yards", "INT", "Pass TDs".
    'rushyds': 'rushing_yards', 'int': 'passing_interceptions',
    'ints': 'passing_interceptions',

    'receivingyards': 'receiving_yards', 'recyds': 'receiving_yards', 'recyards': 'receiving_yards',
    'receivingyds': 'receiving_yards',
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
    # Real Underdog NFL markets that must never reach the scoring path.
    # "Rush + Rec TDs" is their single most common NFL market by volume (349
    # lines in the payload this was built from) and it is a SUM - mapping it
    # onto rushing_tds would inflate every back on the board.
    'rushrectds': 'rush_rec_tds',
    'seasonrushrectds': 'rush_rec_tds',
    'rushrecyds': 'rush_rec_yards',
    'periodfirsttouchdownscored': 'first_td',
    # Defensive markets. Real, and irrelevant to an offensive fantasy board -
    # named here so they're classified rather than reported as "unmapped"
    # noise every time the check script runs.
    'sacks': 'sacks', 'seasonsacks': 'sacks',
    'regularseasongamesstarted': 'games_started',
}

# Markets scoped to part of a game (a quarter, a half). Underdog posts a lot
# of these and they are meaningless for a season projection, so they are
# dropped by prefix rather than enumerated - `period_1_receiving_yds`,
# `period_1_2_rush_rec_tds` and every future sibling.
PARTIAL_GAME_PREFIXES = ('period_', 'first_half_', 'second_half_', '1h_', '1q_')


def _stat_key(value):
    return re.sub(r'[^a-z]', '', str(value).lower())


def normalize_stat(label):
    """
    Provider stat label -> (our column name, is_scorable).

    is_scorable is False for combination markets, part-of-game markets and
    anything unrecognized, which is what keeps an unknown market out of the
    fantasy-point math instead of quietly landing on the wrong column.
    """
    raw = str(label).lower().strip()
    if raw.startswith(PARTIAL_GAME_PREFIXES):
        return raw, False
    key = _stat_key(label)
    if key in STAT_ALIASES:
        return STAT_ALIASES[key], True
    if key in COMBO_STATS:
        return COMBO_STATS[key], False
    return None, False


# Underdog labels a player's position in words. Mapped to this app's codes so
# coverage can be judged per position; their `position_name` field already
# carries the short form, and this is the fallback when it doesn't.
UNDERDOG_POSITIONS = {
    'quarterback': 'QB', 'running back': 'RB', 'full back': 'FB',
    'wide receiver': 'WR', 'tight end': 'TE', 'kicker': 'K',
}


def _underdog_team_map(payload):
    """
    Team UUID -> abbreviation, decoded from the games array.

    UNDERDOG IDENTIFIES TEAMS BY UUID, NOWHERE SPELLED OUT. A player carries
    `team_id: "5ce78c37-b02c-..."` and nothing else, so the first version of
    this adapter resolved zero teams out of 5,314 lines - it passed a UUID to
    standardize_team, which correctly said it had never heard of it.

    The abbreviation only exists on the GAMES, in `abbreviated_title` ("NE @
    SEA") alongside `home_team_id` and `away_team_id`. Splitting that title
    is the only place the two representations meet.
    """
    mapping = {}
    for game in payload.get('games') or []:
        title = str(game.get('abbreviated_title') or '')
        if '@' not in title:
            continue
        away, _, home = title.partition('@')
        if game.get('away_team_id'):
            mapping[str(game['away_team_id'])] = away.strip()
        if game.get('home_team_id'):
            mapping[str(game['home_team_id'])] = home.strip()
    return mapping


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

def parse_underdog_payload(payload, sport='NFL'):
    """
    Underdog's over/under payload -> normalized props.

    Their response is RELATIONAL, not a flat list, and the joins are where
    every real bug lives. Verified against a live 12.6 MB payload (5,314
    lines, 1,436 players, 88 games):

        players[]           id, first_name, last_name, sport_id, team_id (UUID),
                            position_name, position_display_name
        appearances[]       id, player_id, match_id, match_type
        games[]             home_team_id, away_team_id, abbreviated_title
        over_under_lines[]  stat_value (the line), options[], status, and
                            over_under.appearance_stat.{appearance_id, stat}

    A line names an APPEARANCE, an appearance names a PLAYER, and only the
    game knows what a team UUID is called. Three lookups, and the first
    version of this function got two of them wrong.

    THE PAYLOAD IS EVERY SPORT AT ONCE - NFL, MLB, CS, LOL, tennis, golf,
    racing. `sport` filters it; pass None to keep everything.

    Pure function on purpose, so it can be tested against a recorded payload
    with no network.
    """
    if not isinstance(payload, dict):
        return _empty_props(), "Underdog returned an unexpected payload shape."

    players = {str(p.get('id')): p for p in payload.get('players') or []}
    appearances = {str(a.get('id')): a for a in payload.get('appearances') or []}
    lines = payload.get('over_under_lines') or []
    if not players or not lines:
        return _empty_props(), "Underdog returned no player lines."

    teams = _underdog_team_map(payload)
    wanted = str(sport).upper() if sport else None

    rows = []
    for line in lines:
        if str(line.get('status', '')).lower() == 'suspended':
            continue
        over_under = line.get('over_under') or {}
        stat_ref = over_under.get('appearance_stat') or {}
        appearance = appearances.get(str(stat_ref.get('appearance_id') or ''))
        if not appearance:
            continue
        player = players.get(str(appearance.get('player_id') or ''))
        if not player:
            continue
        if wanted and str(player.get('sport_id') or '').upper() != wanted:
            continue

        market, scorable = normalize_stat(stat_ref.get('stat'))
        payouts = {}
        for option in line.get('options') or []:
            choice = str(option.get('choice', '')).lower()
            side = {'higher': 'over', 'lower': 'under'}.get(choice, choice)
            payouts[side] = pd.to_numeric(option.get('payout_multiplier'), errors='coerce')

        team_uuid = str(player.get('team_id') or appearance.get('team_id') or '')
        position = str(player.get('position_name') or '').upper()
        if not position:
            position = UNDERDOG_POSITIONS.get(
                str(player.get('position_display_name') or '').lower(), '')

        name = ' '.join(str(player.get(k) or '').strip()
                        for k in ('first_name', 'last_name')).strip()
        rows.append({
            'provider': 'Underdog',
            'player': name,
            'team': standardize_team(teams.get(team_uuid, '')),
            'position': position,
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
        return props, (f"Underdog returned lines but none were {wanted}."
                       if wanted else "Underdog lines couldn't be joined to a player.")
    return props, None


# Underdog's `match_type` values on a real payload: 'Game' (a single game),
# 'Series' (SEASON-LONG - their season-long pick'em product), 'SoloGame' (a
# head-to-head event like tennis or golf).
#
# 'Series' IS THE ONE THAT MATTERS and nothing about the word says so. The
# first version of this looked for the substring "season" in match_type,
# which matched none of the three, so every season-long line on the board was
# labelled 'game' and the season projection silently had nothing to work
# with. Matched exactly rather than by substring now, because guessing at
# this field is precisely what went wrong.
UNDERDOG_SEASON_MATCH_TYPES = {'series', 'season', 'seasonlong', 'season_long'}


def _underdog_period(appearance):
    """'season' for a season-long line, 'game' for a single-game one."""
    match_type = str(appearance.get('match_type') or '').lower().replace(' ', '')
    return 'season' if match_type in UNDERDOG_SEASON_MATCH_TYPES else 'game'


def fetch_underdog_payload(endpoints=None):
    """
    Underdog's raw payload, trying each endpoint version until one answers.

    Returns (payload, url, error). The url comes back so the caller can say
    which version actually worked - useful when the ladder has moved on and
    the constant in this file wants updating.
    """
    attempts = []
    for url in (endpoints or UNDERDOG_LINE_ENDPOINTS):
        payload, err = _get_json(url)
        if err is None:
            return payload, url, None
        attempts.append(f"{url.rsplit('/', 2)[-2]}: {err}")
        # A refusal or a rate limit is about the CLIENT, not the version -
        # walking further down the ladder would just repeat it four more
        # times against an endpoint that has already said no.
        if 'refused the request' in err or 'rate-limited' in err:
            return None, url, err
    return None, None, (
        "No Underdog endpoint version answered. They version this path and it "
        "moves; if their web app is working, find the current version in your "
        "browser's network tab and add it to UNDERDOG_LINE_ENDPOINTS in "
        "data/odds_sources.py. Tried:\n  " + "\n  ".join(attempts))


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
    payload, _url, err = fetch_underdog_payload()
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

    durations = {key[1]: (value or {}).get('name')
                 for key, value in included.items() if key[0] == 'duration'}

    rows = []
    for item in payload.get('data') or []:
        attrs = item.get('attributes') or {}
        rel = item.get('relationships') or {}
        player_ref = ((rel.get('new_player') or rel.get('player') or {}).get('data')) or {}
        player = included.get((str(player_ref.get('type')), str(player_ref.get('id')))) or {}

        # DEMON AND GOBLIN LINES ARE DELIBERATELY SHADED and must never reach
        # a projection. They are PrizePicks' altered-line products: a demon
        # sits ABOVE the true median and pays more, a goblin sits below and
        # pays less. Only 'standard' is the book's honest read of the middle,
        # and scoring a demon as though it were would bias a player upward by
        # design. Kept in the frame - a drafter may want to see them - but
        # marked unscorable and labelled in the provider name.
        odds_type = str(attrs.get('odds_type') or 'standard').lower()
        market, scorable = normalize_stat(attrs.get('stat_type'))
        if odds_type not in ('standard', ''):
            scorable = False

        # display_name comes back as an EMPTY STRING on real payloads, not
        # absent. `or` handles it only because '' is falsy - do not "fix"
        # this into a .get(key, default) chain.
        name = str(player.get('display_name') or player.get('name') or '').strip()

        # Combined-player props ("Tristan Jarry + Cam Talbot") can never
        # match a board row, and a name with two people in it is worse than
        # no row at all.
        if player.get('combo') or ' + ' in name:
            continue

        rows.append({
            'provider': 'PrizePicks' + ('' if odds_type == 'standard' else f' ({odds_type})'),
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
            'period': _prizepicks_period(attrs, rel, durations),
            'source_id': str(item.get('id') or ''),
        })
    props = _finalize(rows)
    if props.empty:
        return props, "PrizePicks returned no usable player lines."
    return props, None


def _prizepicks_period(attrs, relationships=None, durations=None):
    """
    'season' for a season-long line, 'game' otherwise.

    NOTHING IN A PRIZEPICKS PAYLOAD SAYS "SEASON" DIRECTLY, and the first
    version of this guessed that odds_type would. It doesn't - odds_type is
    'standard' / 'demon' / 'goblin', which describes how a line is SHADED,
    not what it covers. Their NFL board as of the 2026 preseason is week-one
    game props: Pass Yards median 229, Receiving Yards median 52.5, every
    row kicking off 2026-09-09.

    So this reads the two fields that could actually say it - the stat label
    ("Season Pass Yards", the shape Underdog uses) and the duration
    relationship, whose name is 'Full' for a whole game. Everything else is
    a game line, which is the safe default: mistaking a per-game line for a
    season one would put a 52-yard receiving projection on a draft board.
    """
    label = str(attrs.get('stat_type') or '').lower()
    if 'season' in label:
        return 'season'
    duration_ref = ((relationships or {}).get('duration') or {}).get('data') or {}
    duration = str((durations or {}).get(str(duration_ref.get('id')), '')).lower()
    return 'season' if 'season' in duration else 'game'


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


UNDERDOG_SAVED_PATH = os.path.join('external_data', 'underdog_over_under_lines.json')
PRIZEPICKS_SAVED_PATH = os.path.join('external_data', 'prizepicks_projections.json')

# Saved payloads, as provider -> (path, a key that must be present, parser).
# Both books go through the same save/load path because the reason is the
# same for both: these endpoints are undocumented, sometimes blocked, and a
# file the user saved from their own browser is the one input that keeps
# working regardless.
SAVED_PAYLOADS = {
    'Underdog': (UNDERDOG_SAVED_PATH, 'over_under_lines'),
    'PrizePicks': (PRIZEPICKS_SAVED_PATH, 'data'),
}


def save_book_payload(raw_bytes, provider):
    """
    Persist an uploaded payload so it survives a restart.

    Written to disk rather than kept in session state for the same reason
    save_ffa_import is: re-uploading a 12 MB file before every session is
    exactly the friction that gets a feature abandoned two days before a
    draft. Gitignored - it is someone else's data, not ours to redistribute.
    """
    if provider not in SAVED_PAYLOADS:
        return None, f"Unknown provider {provider!r}."
    path, required = SAVED_PAYLOADS[provider]
    try:
        payload = json.loads(raw_bytes)
    except Exception as exc:
        return None, f"Couldn't read that file as JSON: {exc}"
    if not isinstance(payload, dict) or required not in payload:
        return None, (f"That doesn't look like a {provider} response - expected a JSON "
                      f"object with a {required!r} key.")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle)
    except Exception as exc:
        return payload, f"Parsed it, but couldn't save it for next time: {exc}"
    return payload, None


def load_saved_book_payload(provider):
    """The last saved payload for one provider, or None. Never raises."""
    if provider not in SAVED_PAYLOADS:
        return None
    path, _ = SAVED_PAYLOADS[provider]
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as handle:
                return json.load(handle)
    except Exception:
        pass
    return None


def save_underdog_payload(raw_bytes, path=UNDERDOG_SAVED_PATH):
    """Back-compat shim for the Underdog-only save."""
    return save_book_payload(raw_bytes, 'Underdog')


def load_saved_underdog_payload(path=UNDERDOG_SAVED_PATH):
    return load_saved_book_payload('Underdog')
