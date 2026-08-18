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
import contextlib
import json
import os
import re

import pandas as pd
import requests
import streamlit as st
from urllib.parse import urlencode

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
PRIZEPICKS_LEAGUES_URL = 'https://api.prizepicks.com/leagues'
# PrizePicks' league id for the weekly NFL board. Confirmed against a real
# payload: league 9 is "NFL", and what it returns is week-by-week game props.
PRIZEPICKS_NFL_LEAGUE_ID = 9

# SEASON-LONG LIVES IN A DIFFERENT LEAGUE, NOT BEHIND A FILTER. PrizePicks
# runs its season-long product as its own league - "NFLSZN" - so pulling
# league 9 and hoping for season markets returns week-one game props forever.
#
# The id is NOT hardcoded, because it is undocumented and there is no reason
# to think it is stable. discover_prizepicks_league() reads /leagues and
# matches on NAME, which is the part users actually see and the part least
# likely to change.
PRIZEPICKS_SEASON_LEAGUE_NAMES = ('nflszn', 'nflseason', 'nflseasonlong')


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
    'rushyds': 'rushing_yards',
    'passints': 'passing_interceptions',

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
    'tacklesast': 'tackles', 'tackles': 'tackles',
    'puntsinside': 'punts_inside_20', 'puntsinside20': 'punts_inside_20',
    # Milestone COUNTS ("how many 100-yard games"), not totals. Real markets,
    # and meaningless to a projection that works in season sums.
    'recyardgames': 'milestone_games', 'rushyardgames': 'milestone_games',
    'passyardgames': 'milestone_games',
    # Kicking. FG Made is a single total, but this app scores field goals in
    # DISTANCE BUCKETS (fg_made_0_19 ... fg_made_60_), so a lump sum cannot be
    # placed without inventing a distribution across them.
    'fgmade': 'fg_made_total', 'yardfgmade': 'fg_made_long',
    'rushrecyds': 'rush_rec_yards', 'passrushtds': 'pass_rush_tds',
    'passrushyds': 'pass_rush_yards',
}

# Markets whose meaning depends on WHO the line is about, mapped to the
# position that makes them the stat named. Everyone else gets nothing.
#
# "INT" IS THE ONE THAT MATTERS AND IT POINTED THE WRONG WAY. On PrizePicks'
# season board it is DEFENSIVE interceptions - the 27 lines belong to DBs,
# linebackers and safeties - while the quarterback's thrown interceptions are
# a separate market called "Pass INTs". Mapping the bare label to
# passing_interceptions gave a cornerback with 1.5 picks a -3 point penalty
# for throwing them, and dropped the real QB stat entirely.
POSITION_SPECIFIC_STATS = {
    'int': ({'QB'}, 'passing_interceptions'),
    'ints': ({'QB'}, 'passing_interceptions'),
    'interceptions': ({'QB'}, 'passing_interceptions'),
}

# Markets scoped to part of a game (a quarter, a half). Underdog posts a lot
# of these and they are meaningless for a season projection, so they are
# dropped by prefix rather than enumerated - `period_1_receiving_yds`,
# `period_1_2_rush_rec_tds` and every future sibling.
PARTIAL_GAME_PREFIXES = ('period_', 'first_half_', 'second_half_', '1h_', '1q_')


def _stat_key(value):
    return re.sub(r'[^a-z]', '', str(value).lower())


def normalize_stat_for(label, position):
    """
    normalize_stat, but able to resolve labels whose meaning depends on the
    player's position. Falls back to the position-blind mapping otherwise.
    """
    key = _stat_key(label)
    if key in POSITION_SPECIFIC_STATS:
        positions, stat = POSITION_SPECIFIC_STATS[key]
        if str(position or '').upper() in positions:
            return stat, True
        return f'{label} (not a {"/".join(sorted(positions))} stat here)', False
    return normalize_stat(label)


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
    'provider',     # 'Underdog' / 'PrizePicks' / 'FanDuel' / 'Pinnacle' / 'The Odds API'
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
    # Probability of the over with the book's margin divided out, or None
    # where the source publishes no prices. 0.5 means the posted line IS the
    # median; anything else means the book's real middle sits off it. Null
    # for the pick'em books by nature - both sides pay the same there, which
    # is the same statement as 0.5 but should not be asserted as a measured
    # number.
    'p_over',
    'period',       # 'season' or 'game'
    'source_id',
]


def _empty_props():
    return pd.DataFrame({c: pd.Series(dtype='object') for c in PROP_COLUMNS})


def unmapped_markets(props):
    """
    Markets this app has no stat column for - the ones worth adding aliases
    for.

    Deliberately NOT "everything unscorable". A demon or goblin line has a
    perfectly good market behind it and is excluded for a different reason
    (deliberate shading), and lumping the two together made the diagnostic
    report 121 'Rec TDs' as unmapped when every one of them was mapped fine.
    """
    if props is None or props.empty:
        return props
    from data.draft_projections import PROJECTED_STATS
    return props[~props['market'].isin(PROJECTED_STATS)]


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


# Set while a shared_browser() block is open, so a run of calls to the same
# refusing book pays for one browser start rather than one each.
_BROWSER_GET = None

# Set to 0 to turn the fallback off entirely and go back to accepting a 403
# as the final answer.
BROWSER_FALLBACK_ENV = 'NFLSCHOLAR_BROWSER_FALLBACK'


def browser_fallback_enabled():
    return os.environ.get(BROWSER_FALLBACK_ENV, '1').strip() not in ('0', 'false', 'no')


@contextlib.contextmanager
def shared_browser():
    """
    Keep one browser open across a run of fetches.

    DraftKings needs eight calls and PrizePicks one; starting a browser eight
    times would turn a two-second refresh into a fifteen-second one. Nested
    use is a no-op so a caller can ask for it without knowing whether an
    outer one already did.
    """
    global _BROWSER_GET
    if _BROWSER_GET is not None or not browser_fallback_enabled():
        yield
        return
    try:
        from data.odds_browser import browser_session
    except ImportError:
        yield
        return
    with browser_session() as get:
        _BROWSER_GET = get
        try:
            yield
        finally:
            _BROWSER_GET = None


def _browser_retry(url, params=None):
    """One refused URL, retried through a real browser. Returns (payload, error)."""
    if not browser_fallback_enabled():
        return None, "The browser fallback is switched off."
    full = url
    if params:
        full = f"{url}{'&' if '?' in url else '?'}{urlencode(params)}"
    if _BROWSER_GET is not None:
        return _BROWSER_GET(full)
    try:
        from data.odds_browser import fetch_json_via_browser
    except ImportError:
        return None, "Playwright isn't installed, so no browser retry was possible."
    return fetch_json_via_browser(full)


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
        # Refused a script. Some of these books serve the same URL to a
        # browser perfectly well - DraftKings answered this exact endpoint
        # earlier the same day, and PrizePicks has always been browser-only -
        # so the next attempt is an actual browser rather than a script
        # claiming to be one. See data/odds_browser for where that line sits.
        payload, browser_err = _browser_retry(url, params)
        if payload is not None:
            return payload, None
        hint = ''
        if browser_err and 'to a browser too' in browser_err:
            # A headless browser was refused. The next thing that legitimately
            # changes the answer is a visible one - still the user's own
            # browser, just not hidden. After that the honest options run out
            # and the manual save is the path.
            hint = (" A headless browser was refused as well. Try "
                    "NFLSCHOLAR_BROWSER_HEADED=1 before launching, which runs a "
                    "visible browser; if that is also refused, save the JSON "
                    "from your own browser and upload it in Draft HQ → League "
                    "settings → Market lines.")
        return None, (f"{url} refused the request ({resp.status_code}). "
                      + (browser_err or "Nothing further is attempted.") + hint)
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

        position = str(player.get('position_name') or '').upper()
        if not position:
            position = UNDERDOG_POSITIONS.get(
                str(player.get('position_display_name') or '').lower(), '')
        market, scorable = normalize_stat_for(stat_ref.get('stat'), position)
        payouts = {}
        for option in line.get('options') or []:
            choice = str(option.get('choice', '')).lower()
            side = {'higher': 'over', 'lower': 'under'}.get(choice, choice)
            payouts[side] = pd.to_numeric(option.get('payout_multiplier'), errors='coerce')

        team_uuid = str(player.get('team_id') or appearance.get('team_id') or '')

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

def _league_key(value):
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


def prizepicks_leagues(payload):
    """{league id: name} from a /leagues response or an embedded `included`."""
    leagues = {}
    items = payload.get('data') if isinstance(payload, dict) else None
    for item in items or []:
        if str(item.get('type')) in ('league', 'leagues'):
            leagues[str(item.get('id'))] = str((item.get('attributes') or {}).get('name') or '')
    for item in (payload.get('included') or []) if isinstance(payload, dict) else []:
        if str(item.get('type')) == 'league':
            leagues[str(item.get('id'))] = str((item.get('attributes') or {}).get('name') or '')
    return leagues


@st.cache_data(ttl=FETCH_TTL, show_spinner=False)
def discover_prizepicks_league(names=PRIZEPICKS_SEASON_LEAGUE_NAMES):
    """
    Find a league id by NAME rather than hardcoding a number.

    Returns (id, name, error). Used to locate NFLSZN - PrizePicks' season-long
    product, which is a separate league rather than a filter on the weekly
    one. Matching on the name means a changed id costs nothing; hardcoding
    would fail silently by returning the weekly board instead.
    """
    payload, err = _get_json(PRIZEPICKS_LEAGUES_URL)
    if err:
        return None, None, err
    leagues = prizepicks_leagues(payload)
    if not leagues:
        return None, None, "PrizePicks /leagues returned nothing recognizable."
    wanted = {_league_key(n) for n in names}
    for league_id, name in leagues.items():
        if _league_key(name) in wanted:
            return league_id, name, None
    football = {i: n for i, n in leagues.items() if 'nfl' in _league_key(n)}
    return None, None, ("No season-long league found. Football leagues visible: "
                        + (', '.join(f'{n} (id {i})' for i, n in football.items()) or 'none'))


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
    leagues = {key[1]: (value or {}).get('name')
               for key, value in included.items() if key[0] == 'league'}

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
        position = str(player.get('position') or '').upper()
        market, scorable = normalize_stat_for(attrs.get('stat_type'), position)
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
            'position': position,
            'market': market or str(attrs.get('stat_type') or ''),
            'market_raw': str(attrs.get('stat_type') or ''),
            'scorable': bool(scorable),
            'line': attrs.get('line_score'),
            # PrizePicks is a pick'em product: both sides pay the same, so
            # there is no per-side price to record. Left blank rather than
            # filled with a fake 1.0, which would read as a real quote.
            'over_payout': None,
            'under_payout': None,
            'period': _prizepicks_period(attrs, rel, durations, leagues,
                                         player.get('league')),
            'source_id': str(item.get('id') or ''),
        })
    props = _finalize(rows)
    if props.empty:
        return props, "PrizePicks returned no usable player lines."
    return props, None


# Roughly the largest a single-GAME line can plausibly be, per stat. Used
# only as a cross-check on the period classification, never to set it.
#
# It earns its place: period detection on this payload has now been wrong
# twice for two different reasons, and both times it failed silently. A
# season receiving-yards line is 1,200 and a game line is 60 - two orders of
# magnitude apart - so a misclassification is trivially detectable even
# though it is not trivially preventable.
GAME_LINE_CEILING = {
    'passing_yards': 600, 'rushing_yards': 350, 'receiving_yards': 350,
    'receptions': 25, 'passing_tds': 8, 'rushing_tds': 6, 'receiving_tds': 6,
    'carries': 45, 'targets': 30, 'passing_interceptions': 6,
}


def implausible_period_rows(props):
    """
    Rows whose line is far too large for the period they claim.

    Returns the offending subset. A 1,200-yard "game" line means the season
    flag was missed; nothing here fixes it automatically, because guessing
    twice is how this went wrong in the first place - it is surfaced so a
    human can look.
    """
    if props is None or props.empty:
        return props
    ceilings = props['market'].map(GAME_LINE_CEILING)
    lines = pd.to_numeric(props['line'], errors='coerce')
    return props[(props['period'] == 'game') & ceilings.notna() & (lines > ceilings)]


def _prizepicks_period(attrs, relationships=None, durations=None,
                       leagues=None, player_league=None):
    """
    'season' for a season-long line, 'game' otherwise.

    THE LEAGUE IS THE ANSWER, and it took two wrong guesses to find it.
    PrizePicks runs season-long as a SEPARATE LEAGUE called NFLSZN, not as a
    flag on a projection and not as a filter on the weekly board. So inside
    an NFLSZN payload the stat labels read perfectly ordinary - "Receiving
    Yards", not "Season Receiving Yards" - and every earlier heuristic
    classified the whole thing as per-game.

    The two dead ends, kept so they aren't retried:
      - odds_type. It is 'standard' / 'demon' / 'goblin', which describes how
        a line is SHADED, not what it covers.
      - the stat label alone. Right for Underdog, blank for PrizePicks.

    Checked in order of reliability: the league name, then the stat label
    (for a book that does prefix it), then the duration relationship.
    """
    league_names = [str(player_league or '')]
    league_ref = ((relationships or {}).get('league') or {}).get('data') or {}
    league_names.append(str((leagues or {}).get(str(league_ref.get('id')), '')))
    if leagues and len(leagues) == 1:
        # A single-league payload - which is what /projections?league_id=N
        # returns - names itself in `included` even when a projection carries
        # no league relationship.
        league_names.extend(str(n) for n in leagues.values())
    for name in league_names:
        if _league_key(name) in {_league_key(n) for n in PRIZEPICKS_SEASON_LEAGUE_NAMES}:
            return 'season'

    if 'season' in str(attrs.get('stat_type') or '').lower():
        return 'season'
    duration_ref = ((relationships or {}).get('duration') or {}).get('data') or {}
    duration = str((durations or {}).get(str(duration_ref.get('id')), '')).lower()
    return 'season' if 'season' in duration else 'game'


@st.cache_data(ttl=FETCH_TTL, show_spinner=False)
def fetch_prizepicks_lines(league_id=None, season_first=True):
    """
    PrizePicks' current projections, via one ordinary HTTPS request.

    SEASON-LONG FIRST. Their season-long product (NFLSZN) is a separate
    league, so the weekly NFL board is the wrong place to look for it. This
    discovers the season league by name and asks for that; only if there is
    no such league does it fall back to the weekly board, which is still
    worth having for per-game work.

    THIS WILL SOMETIMES BE REFUSED, AND THAT IS THE DESIGN. PrizePicks sits
    behind Cloudflare. When the request comes back 403 the adapter reports it
    and stops - it does not retry behind a patched headless browser, spoof a
    browser fingerprint, or rotate IPs to get around the block. Those are
    techniques for defeating an access control the operator chose to put up,
    and "I want the data for my fantasy draft" is not a good enough reason to
    defeat one. Save the JSON from your browser instead; that path is
    supported and is the one that always works.
    """
    tried = []
    if league_id is None and season_first:
        season_id, season_name, err = discover_prizepicks_league()
        if season_id:
            payload, fetch_err = _get_json(
                PRIZEPICKS_PROJECTIONS_URL,
                params={'league_id': season_id, 'per_page': 1000})
            if fetch_err is None:
                props, parse_err = parse_prizepicks_payload(payload)
                if not props.empty:
                    return props, parse_err
                tried.append(f"{season_name} (id {season_id}): no usable lines")
            else:
                tried.append(f"{season_name} (id {season_id}): {fetch_err}")
        elif err:
            tried.append(err)

    payload, err = _get_json(
        PRIZEPICKS_PROJECTIONS_URL,
        params={'league_id': league_id or PRIZEPICKS_NFL_LEAGUE_ID, 'per_page': 1000})
    if err:
        return _empty_props(), '; '.join(tried + [err])
    props, parse_err = parse_prizepicks_payload(payload)
    if tried and props.empty:
        return props, '; '.join(tried + ([parse_err] if parse_err else []))
    return props, parse_err


# ---------------------------------------------------------------------------
# Two-way pricing
# ---------------------------------------------------------------------------

def _american(price):
    """
    American odds as a number, tolerating how books actually write them.

    DraftKings publishes "−115" using U+2212 MINUS SIGN rather than an
    ASCII hyphen - it is the typographically correct character and float()
    rejects it outright. Every price on their board is negative or positive
    with these, so a parser that misses it does not degrade, it dies.
    """
    if price is None:
        return None
    text = str(price).strip().replace('−', '-').replace('–', '-').replace('—', '-')
    try:
        return float(text)
    except ValueError:
        return None


def american_to_probability(price):
    """American odds -> the probability they IMPLY, vig included."""
    value = _american(price)
    if not value:
        return None
    return (-value) / (-value + 100.0) if value < 0 else 100.0 / (value + 100.0)


def american_to_decimal(price):
    """American odds -> decimal odds, the form over_payout/under_payout hold."""
    value = _american(price)
    if not value:
        return None
    return 1.0 + (100.0 / -value if value < 0 else value / 100.0)


def decimal_to_probability(price):
    """Decimal odds -> implied probability, vig included."""
    try:
        value = float(price)
    except (TypeError, ValueError):
        return None
    return 1.0 / value if value > 0 else None


def devig_two_way(over_price, under_price):
    """
    The true probability of the over, with the book's margin divided out.

    THIS IS THE STEP THAT MAKES A SPORTSBOOK LINE COMPARABLE TO A PICK'EM.
    Underdog and PrizePicks are even-money products: both sides pay the same,
    so the posted number IS their median estimate and can be read straight
    off. A sportsbook takes a cut, and when it takes that cut UNEVENLY the
    posted number stops being the middle. FanDuel's season props run about
    -114/-114 most of the time, which is symmetric and harmless - but 42 of
    146 in the first real payload were not, out to -148/+112. Read naively
    that line is the median; devigged it is a number the book thinks the
    player clears 56% of the time, which is a different claim.

    Proportional (multiplicative) devig: both implied probabilities are
    scaled so they sum to 1. It is the standard choice and the right default
    for a roughly balanced two-way market. It does assume the margin is
    spread proportionally across the two sides, which is known to understate
    the favourite slightly on lopsided prices - not a concern at the near
    50/50 prices season totals trade at.

    Returns None when either side is missing, which is the honest answer for
    Pinnacle's matchup feed - it carries lines but no prices at all.
    """
    return _devig(american_to_probability(over_price),
                  american_to_probability(under_price))


def devig_two_way_decimal(over_price, under_price):
    """devig_two_way for books that publish decimal odds instead."""
    return _devig(decimal_to_probability(over_price),
                  decimal_to_probability(under_price))


def _devig(p_over, p_under):
    if p_over is None or p_under is None:
        return None
    total = p_over + p_under
    if total <= 0:
        return None
    return p_over / total


# ---------------------------------------------------------------------------
# FanDuel
# ---------------------------------------------------------------------------

# One URL carries the entire NFL page, season-long player props included -
# no auth header, no per-market call, no category walk. That was a surprise;
# FanDuel was predicted to be the hardest of the majors and turned out to be
# the easiest, because the _ak in the query string is all the page itself
# sends. The state subdomain is interchangeable for this content.
FANDUEL_NFL_URL = ('https://sbapi.oh.sportsbook.fanduel.com/api/content-managed-page'
                   '?page=CUSTOM&customPageId=nfl&_ak=FhMFpcPWXMeyZxOx'
                   '&timezone=America%2FNew_York')

# "<Player> Regular Season <Stat> 2026-27". The trailing season stamp is what
# separates a season market from a game one, so it is required rather than
# optional - a game prop is named differently and must not match.
_FD_MARKET = re.compile(r'^(?P<player>.+?) Regular Season (?P<stat>.+?) 20\d\d-\d\d$')
# "<Player> Over 3050.5". THE LINE LIVES IN THE RUNNER NAME, NOT IN THE
# `handicap` FIELD - handicap is present and is 0.0 on every one of these
# markets. Reading it would have produced a board full of zero-yard
# projections that still looked structurally valid.
_FD_RUNNER = re.compile(r'^(?P<player>.+?)\s+(?P<side>Over|Under)\s+(?P<line>-?[\d.]+)\s*$')


def parse_fanduel_payload(payload):
    """
    FanDuel's NFL content page -> normalized props.

    Markets hang off `attachments.markets`, keyed by market id, each with its
    own `runners`. Season-long player props are the ones whose name carries
    the "Regular Season ... 2026-27" stamp.

    TWO MARKETS MATCH THAT NAME PATTERN AND ARE NOT OVER/UNDERS, and both
    would have parsed into convincing nonsense:

      "AP NFL Regular Season MVP 2026-27" splits into player "AP NFL" and
      stat "MVP", with one runner per candidate quarterback.

      "Most Regular Season Rookie Receiving Yards 2026-27" splits into player
      "Most" and stat "Rookie Receiving Yards", with nineteen runners.

    Both are winner markets wearing a totals market's name. The guard is
    structural rather than a name blacklist: a real over/under has EXACTLY
    two runners and BOTH must parse as a side plus a number. A leader market
    fails on runner count, an award market fails on runner shape, and any
    future market of either kind fails the same way without needing to be
    listed here.
    """
    if not isinstance(payload, dict):
        return _empty_props(), "FanDuel returned an unexpected payload shape."
    markets = ((payload.get('attachments') or {}).get('markets')) or {}
    if not markets:
        return _empty_props(), ("FanDuel's payload has no attachments.markets - this is "
                                "probably the wrong page, or one that didn't finish loading.")

    rows = []
    for market in markets.values():
        if not isinstance(market, dict):
            continue
        named = _FD_MARKET.match(str(market.get('marketName') or ''))
        if not named:
            continue
        runners = market.get('runners') or []
        if len(runners) != 2:
            continue
        parsed = [_FD_RUNNER.match(str(r.get('runnerName') or '')) for r in runners]
        if not all(parsed):
            continue

        sides = {}
        for runner, hit in zip(runners, parsed):
            odds = ((runner.get('winRunnerOdds') or {}).get('americanDisplayOdds') or {})
            sides[hit.group('side').lower()] = {
                'line': hit.group('line'),
                'price': odds.get('americanOdds'),
            }
        if 'over' not in sides or 'under' not in sides:
            continue

        market_name, scorable = normalize_stat(named.group('stat'))
        rows.append({
            'provider': 'FanDuel',
            'player': named.group('player').strip(),
            # FanDuel's payload names no team for these markets, and
            # marketType only groups them into QUARTERBACKS / RUNNING_BACKS /
            # WIDE_RECEIVERS - a bucket, not a position. Tight ends sit in
            # the receiver bucket, so filling `position` from it would assert
            # something false about every TE. The board supplies the real
            # position on the name join; blank is the honest value here.
            'team': '',
            'position': '',
            'market': market_name or named.group('stat'),
            'market_raw': named.group('stat'),
            'scorable': bool(scorable),
            'line': sides['over']['line'],
            'over_payout': american_to_decimal(sides['over']['price']),
            'under_payout': american_to_decimal(sides['under']['price']),
            'p_over': devig_two_way(sides['over']['price'], sides['under']['price']),
            'period': 'season',
            'source_id': str(market.get('marketId') or ''),
        })

    props = _finalize(rows)
    if props.empty:
        return props, "FanDuel returned no season-long player lines."
    return props, None


@st.cache_data(ttl=FETCH_TTL, show_spinner=False)
def fetch_fanduel_lines():
    """Live FanDuel season props. Returns (props, error)."""
    payload, err = _get_json(FANDUEL_NFL_URL)
    if err:
        return _empty_props(), err
    return parse_fanduel_payload(payload)


# ---------------------------------------------------------------------------
# DraftKings
# ---------------------------------------------------------------------------

DK_LEAGUE_NFL = 88808
DK_PLAYER_FUTURES_CATEGORY = 1759
DK_NASH_BASE = 'https://sportsbook-nash.draftkings.com/api/sportscontent/dkusoh/v1'

# EIGHT CALLS, NOT ONE, and the reason is worth keeping. The category route
# without a subcategory answers 200 with 25 markets - exactly the Passing
# Yards count, i.e. one default subcategory rather than the union. Nothing in
# that response says it is partial, so a one-call adapter would look like it
# worked while seeing a seventh of the board.
#
# DraftKings is the only book here that prices RECEPTIONS and RECEIVING TDs
# season-long, which is the whole reason it was worth chasing: in a PPR
# league receptions is the largest single scoring input, and before this it
# rested on PrizePicks alone with nothing to check it against.
DK_SUBCATEGORIES = {
    'Passing Yards': 17147, 'Passing TDs': 17148,
    'Rushing Yards': 17223, 'Rushing TDs': 17224,
    'Receiving Yards': 17314, 'Receiving TDs': 17315,
    'Receptions': 20168, 'Sacks': 17316,
}

# "Over 62.5" - the line is ONLY here. Futures selections carry no `points`
# field at all, unlike the game-lines feed from the same API where `points`
# is populated and is the natural place to look.
#
# The optional leading group is for weekly markets, where books commonly put
# the man's name in front of the side ("Patrick Mahomes Over 275.5") because
# one market covers a whole game rather than one player. Optional, so a
# season label still matches exactly as before.
_DK_LABEL = re.compile(r'^(?:(?P<player>.+?)\s+)?(?P<side>Over|Under)\s+(?P<line>-?[\d.]+)\s*$')


def dk_subcategory_url(subcategory_id, league=DK_LEAGUE_NFL,
                       category=DK_PLAYER_FUTURES_CATEGORY, base=DK_NASH_BASE):
    return f'{base}/leagues/{league}/categories/{category}/subcategories/{subcategory_id}'


def _dk_player_team_period(event, market, pair):
    """
    Who the line is about, which club he plays for, and whether it is a
    season or a single-game market. Returns (player, team, period).

    DraftKings files a season prop under a PLAYER EVENT and a weekly prop
    under a GAME EVENT, and that difference is structural rather than a
    matter of wording:

      season: participants are [the club, the player] - exactly one carries
              metadata.rosettaTeamName, and the other is the man.
      game:   participants are the two clubs - BOTH carry it, and the player
              is named somewhere else entirely.

    Counting how many participants are teams therefore classifies the market
    without reading a single English word, which is the property worth having
    given how many times a label has been the thing that moved.

    For a game market the player then has to come from the market name
    ("Patrick Mahomes Passing Yards") or from the selection label
    ("Patrick Mahomes Over 275.5"). Both shapes are handled because both are
    plausible and this has not been seen live - a weekly board is not posted
    in August. Neither is guessed at silently: if no player can be recovered
    the row is dropped rather than filed under a wrong name.
    """
    participants = event.get('participants') or []
    teams, others = [], []
    for part in participants:
        meta = part.get('metadata') or {}
        if meta.get('rosettaTeamName'):
            teams.append(standardize_team(meta.get('shortName') or part.get('name')))
        else:
            others.append(str(part.get('name') or '').strip())

    if len(teams) == 1 and others:
        return others[0], teams[0], 'season'

    # A game event. Two clubs, and the player named elsewhere.
    stat_label = str(((market.get('marketType') or {}).get('name')) or '')
    if stat_label.endswith(' OU'):
        stat_label = stat_label[:-len(' OU')]
    name = str(market.get('name') or '')
    for suffix in (stat_label, stat_label.replace('Regular Season ', '')):
        if suffix and name.endswith(suffix):
            candidate = name[:-len(suffix)].strip(' -–—')
            # "NFL 2026/27 - " style prefixes, when present.
            candidate = re.sub(r'^NFL\s+\d{4}/\d{2,4}\s*[-–—]\s*', '', candidate).strip()
            if candidate:
                return candidate, '', 'game'

    # Last resort: the selection labels carry the man's name in front of the
    # side, which is how FanDuel writes them.
    for side in ('over', 'under'):
        raw = (pair.get(side) or {}).get('raw_label') or ''
        hit = re.match(r'^(?P<player>.+?)\s+(?:Over|Under)\s+-?[\d.]+\s*$', str(raw))
        if hit and hit.group('player').strip():
            return hit.group('player').strip(), '', 'game'
    return '', '', 'game'


def parse_draftkings_payload(payload):
    """
    One DraftKings season-long subcategory -> normalized props.

    Three flat arrays that have to be joined: `selections` point at
    `markets` by marketId, and markets point at `events` by eventId. The
    market carries the stat, the selections carry the side and the line, and
    the event carries the player and his team.

    THE PLAYER COMES FROM THE EVENT, NOT THE MARKET NAME. Parsing
    "NFL 2026/27 - Mike Evans Regular Season Receptions" looks like the
    obvious route and quietly loses players: the separator is an ASCII
    hyphen on most rows, an EN DASH on Travis Kelce's, and a
    double-spaced hyphen on Romeo Doubs'. Three of 319 - small enough to
    never notice, and exactly the kind of loss that shows up later as "why
    is Kelce missing". The event's participant list has no such problem:
    exactly one of its two entries carries metadata.rosettaTeamName and is
    the team, the other is the player. Both are typed "Team" and the order
    varies, so the metadata is the key, not the position.

    Likewise the stat comes from marketType.name ("Regular Season Receptions
    OU"), which is uniform across all 319, rather than from the market name,
    where the same stat appears as both "Receiving TDs" and "Receiving
    Touchdowns".
    """
    if not isinstance(payload, dict):
        return _empty_props(), "DraftKings returned an unexpected payload shape."
    markets = payload.get('markets') or []
    selections = payload.get('selections') or []
    events = payload.get('events') or []
    if not markets or not selections:
        return _empty_props(), ("DraftKings returned no markets - check the category and "
                                "subcategory ids in the URL.")

    by_event = {str(e.get('id')): e for e in events if isinstance(e, dict)}
    sides = {}
    for sel in selections:
        if not isinstance(sel, dict):
            continue
        hit = _DK_LABEL.match(str(sel.get('label') or ''))
        if not hit:
            continue
        sides.setdefault(str(sel.get('marketId')), {})[hit.group('side').lower()] = {
            'line': hit.group('line'),
            # trueOdds is the unrounded decimal. displayOdds.decimal is
            # rounded to 2dp and displayOdds.american uses a Unicode minus,
            # so this is both the most precise field and the least
            # booby-trapped one.
            'decimal': sel.get('trueOdds'),
            'raw_label': sel.get('label'),
        }

    rows = []
    for market in markets:
        if not isinstance(market, dict):
            continue
        pair = sides.get(str(market.get('id')))
        if not pair or 'over' not in pair or 'under' not in pair:
            continue

        label = str(((market.get('marketType') or {}).get('name')) or '')
        if not label.endswith(' OU'):
            continue
        stat = label[:-len(' OU')]
        season = stat.startswith('Regular Season ')
        if season:
            stat = stat[len('Regular Season '):]

        event = by_event.get(str(market.get('eventId'))) or {}
        player, team, period = _dk_player_team_period(event, market, pair)
        if not player:
            continue
        # The market label is the authority when it says "Regular Season";
        # otherwise the event's own shape decides, which is structural rather
        # than a guess about wording.
        if season:
            period = 'season'

        market_name, scorable = normalize_stat(stat)
        rows.append({
            'provider': 'DraftKings',
            'player': player,
            'team': team,
            'position': '',
            'market': market_name or stat,
            'market_raw': stat,
            'scorable': bool(scorable),
            'line': pair['over']['line'],
            'over_payout': pair['over']['decimal'],
            'under_payout': pair['under']['decimal'],
            'p_over': devig_two_way_decimal(pair['over']['decimal'],
                                            pair['under']['decimal']),
            'period': period,
            'source_id': str(market.get('id') or ''),
        })

    props = _finalize(rows)
    if props.empty:
        return props, ("DraftKings returned no player over/under lines in that board - "
                       "it is real, it just isn't a totals market.")
    return props, None


def parse_draftkings_payloads(payloads):
    """
    Several subcategory payloads -> one frame.

    A saved DraftKings import is a LIST of subcategory responses, because
    one call only ever returns one stat. Accepts a single response too, so
    the saved-file path and the live path share a parser.
    """
    if isinstance(payloads, dict):
        if 'markets' in payloads:
            return parse_draftkings_payload(payloads)
        payloads = list(payloads.values())
    if not isinstance(payloads, list):
        return _empty_props(), "DraftKings returned an unexpected payload shape."

    frames, errors = [], []
    for payload in payloads:
        props, err = parse_draftkings_payload(payload)
        if not props.empty:
            frames.append(props)
        elif err:
            errors.append(err)
    combined = combine_props(*frames)
    if combined.empty:
        return combined, '; '.join(dict.fromkeys(errors)) or "DraftKings returned no lines."
    return combined, None


# Subcategory names on DraftKings' weekly board that are per-player counting
# stats. Matched case-insensitively against whatever the league feed reports,
# so the ids never have to be hardcoded - which matters because the weekly
# ids are not knowable in the preseason, when no weekly board is posted.
DK_WEEKLY_STAT_NAMES = (
    'pass yards', 'passing yards', 'pass tds', 'passing tds',
    'pass completions', 'pass attempts', 'interceptions',
    'rush yards', 'rushing yards', 'rush tds', 'rushing tds', 'rush attempts',
    'rec yards', 'receiving yards', 'rec tds', 'receiving tds', 'receptions',
)

# Categories whose contents are season-long or otherwise not a weekly player
# total, matched by name so a renamed id does not break the exclusion.
#
# "Player Matchups" is the one that has to be here and looks like it doesn't.
# It carries subcategories named exactly "Receiving Yards" and "Receiving
# TDs", so a name filter picks it up - but those are head-to-head markets
# ("does X out-gain Y"), not over/unders. Discovery matched them, fetched
# them, found no OU markets, and reported two confusing per-stat errors on a
# board that simply had no weekly props yet.
DK_NON_WEEKLY_CATEGORIES = (
    'player futures', 'stat leaders', 'milestones', 'season high totals',
    'rookie watch', 'awards', 'futures', 'fast futures', 'wins',
    'division specials', 'season specials', 'playoffs', 'team specials',
    'next player to record', 'player matchups',
)


@st.cache_data(ttl=FETCH_TTL, show_spinner=False)
def dk_weekly_subcategories():
    """
    Which subcategories on DraftKings' NFL board are weekly player totals.

    DISCOVERED AT RUNTIME, NOT HARDCODED. The league feed ships its own
    category and subcategory catalogue, so the right ids can be read off it
    rather than guessed - which is the only workable approach here, because
    the weekly ids cannot be looked up in August. No weekly board is posted
    in the preseason, so a constant written now would be a constant written
    blind.

    Returns (list of (category_name, subcategory_name, category_id,
    subcategory_id), error).
    """
    payload, err = _get_json(f'{DK_NASH_BASE}/leagues/{DK_LEAGUE_NFL}')
    if err:
        return [], err
    if not isinstance(payload, dict):
        return [], "DraftKings' league feed returned an unexpected shape."

    categories = {c.get('id'): str(c.get('name') or '')
                  for c in payload.get('categories') or [] if isinstance(c, dict)}
    found = []
    for sub in payload.get('subcategories') or []:
        if not isinstance(sub, dict):
            continue
        cat_name = categories.get(sub.get('categoryId'), '')
        if cat_name.lower() in DK_NON_WEEKLY_CATEGORIES:
            continue
        name = str(sub.get('name') or '')
        if name.lower() in DK_WEEKLY_STAT_NAMES:
            found.append((cat_name, name, sub.get('categoryId'), sub.get('id')))
    if not found:
        return [], ("DraftKings is not posting weekly player props right now - no "
                    "per-player stat subcategory is in their NFL board. That is "
                    "expected outside the season; the weekly slate goes up on "
                    "Tuesday or Wednesday.")
    return found, None


@st.cache_data(ttl=FETCH_TTL, show_spinner=False)
def fetch_draftkings_weekly_lines():
    """
    DraftKings' weekly player props - discover the subcategories, then pull
    one board per stat, the same route the season props use.

    NOT LIVE-VERIFIED. Everything below the discovery step is the route that
    is known to work for season props with a different subcategory id, and
    the parser classifies season vs game from the event's own shape rather
    than from wording. But a weekly board is not posted in August, so this
    path has never returned a real row. Run
    `scripts/probe_season_odds.py --weekly` in season to confirm the shape
    before trusting it - and note that the discovery step failing is a
    perfectly ordinary preseason answer, not a bug.
    """
    with shared_browser():
        subs, err = dk_weekly_subcategories()
        if err:
            return _empty_props(), err

        frames, errors = [], []
        for cat_name, name, cat_id, sub_id in subs:
            payload, fetch_err = _get_json(dk_subcategory_url(sub_id, category=cat_id))
            if fetch_err:
                errors.append(f'{name}: {fetch_err}')
                continue
            props, parse_err = parse_draftkings_payload(payload)
            if not props.empty:
                frames.append(props)
            elif parse_err:
                errors.append(f'{name}: {parse_err}')

    combined = combine_props(*frames)
    if combined.empty:
        # Every discovered board came back without a single over/under. That
        # is what a preseason board looks like, and repeating a per-stat
        # error for each one reads as several failures rather than one
        # ordinary state of the world.
        looked = ', '.join(name for _cat, name, _c, _s in subs) or 'nothing'
        return combined, (
            "DraftKings has no weekly player over/unders posted right now "
            f"(looked at: {looked}). Weekly boards go up on Tuesday or "
            "Wednesday in season.")
    return combined, ('; '.join(errors) if errors else None)


@st.cache_data(ttl=FETCH_TTL, show_spinner=False)
def fetch_draftkings_lines():
    """
    Live DraftKings season props - one call per stat, whatever answers.

    A subcategory that fails is skipped rather than failing the book: they
    are independent boards and eight-sevenths of an answer beats none.
    """
    frames, errors = [], []
    with shared_browser():
        for name, sub in DK_SUBCATEGORIES.items():
            payload, err = _get_json(dk_subcategory_url(sub))
            if err:
                errors.append(f'{name}: {err}')
                continue
            props, parse_err = parse_draftkings_payload(payload)
            if not props.empty:
                frames.append(props)
            elif parse_err:
                errors.append(f'{name}: {parse_err}')
    combined = combine_props(*frames)
    if combined.empty:
        return combined, '; '.join(errors) or "DraftKings returned no lines."
    return combined, ('; '.join(errors) if errors else None)


# ---------------------------------------------------------------------------
# Pinnacle
# ---------------------------------------------------------------------------

# Pinnacle splits its answer in two: /matchups describes WHAT is priced and
# /markets carries the PRICES. The lines themselves are in the matchup
# participant names, so the matchup feed alone is enough to read a median off
# - which is most of what a draft board wants. Without the markets call there
# are no prices, so nothing can be devigged; that is recorded as a null
# p_over rather than guessed at.
PINNACLE_NFL_LEAGUE_ID = 889
PINNACLE_MATCHUPS_URL = f'https://guest.api.arcadia.pinnacle.com/0.1/leagues/{PINNACLE_NFL_LEAGUE_ID}/matchups'

# "NFL 2026/2027 - Zay Flowers Regular Season Receiving Yards"
_PIN_SPECIAL = re.compile(r'^NFL \d{4}/\d{4} - (?P<player>.+?) Regular Season (?P<stat>.+)$')
# "Over 974.5 yards" - the unit word is optional and varies by stat.
_PIN_SIDE = re.compile(r'^(?P<side>Over|Under)\s+(?P<line>-?[\d.]+)\s*\w*\s*$')


def parse_pinnacle_payload(payload):
    """
    Pinnacle's guest matchup feed -> normalized props.

    Season player props arrive as `type: "special"` matchups whose
    special.description carries the player and stat, and whose two
    participants carry the side and the number.

    The team-facing specials in the same feed ("Pittsburgh Steelers Total
    Regular Season Wins", "New York Giants To Make the Playoffs", "NFC North
    Winner") do not match the description pattern, so they drop out without
    needing to be enumerated.
    """
    if not isinstance(payload, list):
        return _empty_props(), "Pinnacle returned an unexpected payload shape."

    rows = []
    for matchup in payload:
        if not isinstance(matchup, dict) or matchup.get('type') != 'special':
            continue
        described = _PIN_SPECIAL.match(
            str(((matchup.get('special') or {}).get('description')) or ''))
        if not described:
            continue
        participants = matchup.get('participants') or []
        if len(participants) != 2:
            continue
        sides = {}
        for part in participants:
            hit = _PIN_SIDE.match(str(part.get('name') or ''))
            if hit:
                sides[hit.group('side').lower()] = hit.group('line')
        if 'over' not in sides:
            continue

        market_name, scorable = normalize_stat(described.group('stat'))
        rows.append({
            'provider': 'Pinnacle',
            'player': described.group('player').strip(),
            'team': '',
            'position': '',
            'market': market_name or described.group('stat'),
            'market_raw': described.group('stat'),
            'scorable': bool(scorable),
            'line': sides['over'],
            # No prices in the matchup feed. Blank rather than a placeholder,
            # for the same reason PrizePicks leaves them blank: an invented
            # 1.0 reads downstream as a real even-money quote.
            'over_payout': None,
            'under_payout': None,
            'p_over': None,
            'period': 'season',
            'source_id': str(matchup.get('id') or ''),
        })

    props = _finalize(rows)
    if props.empty:
        return props, "Pinnacle returned no season-long player lines."
    return props, None


@st.cache_data(ttl=FETCH_TTL, show_spinner=False)
def fetch_pinnacle_lines():
    """Live Pinnacle season props. Returns (props, error)."""
    payload, err = _get_json(PINNACLE_MATCHUPS_URL)
    if err:
        return _empty_props(), err
    return parse_pinnacle_payload(payload)


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
# WEEKLY payloads live in their own files. PrizePicks' weekly board is a
# DIFFERENT league from its season board (9 vs NFLSZN), so saving one over
# the other would silently swap the season-long lines Draft HQ's Book Proj
# is built from for a set of single-game props - which would parse cleanly,
# look plausible, and be wrong by two orders of magnitude.
PRIZEPICKS_WEEKLY_SAVED_PATH = os.path.join('external_data',
                                            'prizepicks_weekly_projections.json')
DRAFTKINGS_WEEKLY_SAVED_PATH = os.path.join('external_data',
                                            'draftkings_weekly_props.json')
FANDUEL_SAVED_PATH = os.path.join('external_data', 'fanduel_nfl_page.json')
PINNACLE_SAVED_PATH = os.path.join('external_data', 'pinnacle_nfl_matchups.json')
DRAFTKINGS_SAVED_PATH = os.path.join('external_data', 'draftkings_player_futures.json')

# Saved payloads, as provider -> (path, a key that must be present, parser).
# Both books go through the same save/load path because the reason is the
# same for both: these endpoints are undocumented, sometimes blocked, and a
# file the user saved from their own browser is the one input that keeps
# working regardless.
SAVED_PAYLOADS = {
    'Underdog': (UNDERDOG_SAVED_PATH, 'over_under_lines'),
    'PrizePicks': (PRIZEPICKS_SAVED_PATH, 'data'),
    'FanDuel': (FANDUEL_SAVED_PATH, 'attachments'),
    # Pinnacle's matchup feed is a BARE LIST, not an object, so there is no
    # key to require. None means "any JSON of the right outer shape"; the
    # parser is what rejects a wrong file, and it does so by finding no
    # season specials rather than by guessing from a key name.
    'Pinnacle': (PINNACLE_SAVED_PATH, None),
    'DraftKings': (DRAFTKINGS_SAVED_PATH, None),
    # Weekly, kept apart from the season files above on purpose.
    'PrizePicks Weekly': (PRIZEPICKS_WEEKLY_SAVED_PATH, 'data'),
    'DraftKings Weekly': (DRAFTKINGS_WEEKLY_SAVED_PATH, None),
}

# Every book the board reads, in the order it reads them. One list so a new
# adapter reaches the loader, the uploader and the status panel together -
# the previous shape repeated the pair ('Underdog', 'PrizePicks') in four
# places, which is three chances to add a book that silently never loads.
BOOKS = ('Underdog', 'PrizePicks', 'DraftKings', 'FanDuel', 'Pinnacle')

# USER-SET RELIABILITY WEIGHTS for blending several books' lines on the same
# stat into one consensus number (data.odds_projections.market_stat_lines).
# Real sportsbooks (DraftKings, FanDuel, Pinnacle) price a line with their own
# money behind it and are generally the sharper number; the DFS pick'em
# products (PrizePicks, Underdog) are included at real but lower weight so the
# consensus isn't just the three-book average with two data points ignored.
# Pinnacle carries the most weight of any single book - it is a professional/
# sharp-market book with historically the lowest margins in the industry, the
# reference point the rest of this weighting scheme is built around.
#
# These are RELATIVE, not a strict 0-100 split (they sum to 100 anyway, which
# is a coincidence of these five numbers, not a requirement) - a player priced
# by only two of the five still gets a sensible answer because the weights
# used are renormalized over whichever books actually posted a line for that
# stat. See BOOK_WEIGHT_FALLBACK below for what happens when NONE of the
# providers for a given (player, stat) is in this table.
BOOK_WEIGHTS = {
    'DraftKings': 20,
    'FanDuel': 20,
    'Pinnacle': 25,
    'PrizePicks': 20,
    'Underdog': 15,
}

# A provider string outside BOOK_WEIGHTS (e.g. a future book added to BOOKS
# before its weight is set, or a raw 'The Odds API: <bookmaker>' label that
# slipped through somehow) gets this weight instead of being silently
# excluded - a plain equal-weight vote, so an unweighted book still counts
# rather than vanishing from the consensus entirely.
BOOK_WEIGHT_FALLBACK = 10

# DraftKings serves one stat per call, so its saved import is a LIST of
# responses rather than one. Uploading them one at a time has to accumulate
# instead of overwrite, or dropping in Receptions would silently delete
# Receiving Yards.
MULTI_FILE_BOOKS = ('DraftKings', 'DraftKings Weekly')


def _dk_subcategory_of(payload):
    """Which stat board a DraftKings response is, or None if it isn't one."""
    if not isinstance(payload, dict):
        return None
    for market in payload.get('markets') or []:
        if isinstance(market, dict) and market.get('subcategoryId') is not None:
            return market['subcategoryId']
    return None


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
    blobs = raw_bytes if isinstance(raw_bytes, (list, tuple)) else [raw_bytes]
    try:
        parsed = [json.loads(blob) for blob in blobs]
    except Exception as exc:
        return None, f"Couldn't read that file as JSON: {exc}"

    if provider in MULTI_FILE_BOOKS:
        # Accumulate rather than replace, keyed on the subcategory the
        # response is for, so dropping in one stat at a time builds the board
        # up instead of resetting it to whichever file went in last.
        existing = load_saved_book_payload(provider)
        merged = list(existing) if isinstance(existing, list) else []
        for payload in parsed:
            for one in (payload if isinstance(payload, list) else [payload]):
                sub = _dk_subcategory_of(one)
                merged = [m for m in merged if _dk_subcategory_of(m) != sub or sub is None]
                merged.append(one)
        payload = merged
    else:
        payload = parsed[0]
    if required is None:
        if not isinstance(payload, (dict, list)):
            return None, f"That doesn't look like a {provider} response - expected JSON."
    elif not isinstance(payload, dict) or required not in payload:
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
