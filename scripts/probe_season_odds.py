"""
Probe: which sources can actually give us SEASON-LONG player lines?

Run this on your own machine. It answers three questions that cannot be
answered from a docs page or from inside a sandboxed network:

  1. Does YOUR Odds API key return season-long player totals (e.g. Josh
     Allen passing yards for the whole year), or only per-game props and
     winner-style futures?
  2. How broad are The Odds API's in-season GAME player props, per market -
     how many books post each one, how many players it covers, and what a
     full slate actually costs in credits?
  3. Which sportsbooks answer a plain, honestly-identified HTTPS request at
     all - and of those, which ones expose season-long player props?

    python scripts/probe_season_odds.py --odds-api-key KEY
    python scripts/probe_season_odds.py --odds-api-key KEY --spend
    python scripts/probe_season_odds.py --books-only
    python scripts/probe_season_odds.py --odds-api-key KEY --save-dir /tmp/probe

CREDIT SAFETY. Discovery calls (/sports, /events) are free on The Odds API
and always run. Anything that costs credits is behind --spend, and the exact
cost is read back out of the response headers and printed, so you can see
what each answer cost. Without --spend this script cannot burn your quota.

WHAT THIS IS NOT. Every book request below is one ordinary GET with our real
User-Agent. No browser fingerprint spoofing, no proxy rotation, no
Cloudflare/Datadome bypass, no retry wearing a different hat. A 403 is
recorded as "this book declines automated access" and that is the end of it -
for those, the browser-save path already used for Underdog and PrizePicks is
the honest route, because it is your own logged-in session, not a bot
pretending to be you.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402

# Runs from inside the project OR from a bare folder with nothing but
# `requests` installed. A diagnostic that needs the whole app importable
# before it will tell you why the app can't see any odds is a diagnostic
# that fails exactly when you need it, so the two constants it borrows have
# a local fallback. In-repo they come from the real definitions and stay in
# step; standalone they are close enough to answer the question.
try:
    from config import ODDS_API_PLAYER_PROP_MARKETS  # noqa: E402
    from data.odds_sources import DEFAULT_HEADERS  # noqa: E402
except Exception:
    ODDS_API_PLAYER_PROP_MARKETS = [
        'player_pass_yds', 'player_pass_tds', 'player_pass_completions', 'player_pass_attempts',
        'player_pass_interceptions', 'player_pass_longest_completion',
        'player_rush_yds', 'player_rush_attempts', 'player_rush_longest',
        'player_reception_yds', 'player_receptions', 'player_reception_longest',
        'player_anytime_td', 'player_1st_td', 'player_last_td',
        'player_kicking_points', 'player_field_goals',
        'player_tackles_assists', 'player_sacks', 'player_solo_tackles',
    ]
    DEFAULT_HEADERS = {
        'User-Agent': 'NFLScholar/1.0 (personal fantasy analytics; non-commercial)',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
    }

ODDS_API = 'https://api.the-odds-api.com/v4'
TIMEOUT = 20

# Market keys a season-long player total WOULD plausibly be filed under if
# The Odds API carried one. None of these are documented; that is the point.
# An unknown key comes back as a 422 naming it, which is a clean "no" - and
# a 422 is not billed, so this list is safe to try in bulk.
CANDIDATE_SEASON_MARKETS = [
    'player_pass_yds_season', 'player_rush_yds_season', 'player_reception_yds_season',
    'player_pass_tds_season', 'player_receptions_season',
    'season_player_pass_yds', 'player_season_pass_yds',
    'player_pass_yds_ou_season', 'player_totals_season',
]

# One plain GET each. Where a book has moved its endpoint over time, the
# candidates are ordered newest-first and we stop at the first that answers -
# same ladder idea as the Underdog version list in data/odds_sources.py.
#
# WHAT A HAND CHECK FROM AN ORDINARY CONNECTION SHOWED (2026-08-09, iOS
# Safari, so treat as directional not definitive):
#
# MEASURED FROM A REAL CONNECTION, 2026-08-09. Four predictions in the
# original version of this file were wrong, and the ranking they produced was
# backwards, so the results are recorded here rather than in a chat log:
#
#   FanDuel      200, 1.15 MB of JSON, to a plain honestly-identified GET.
#                Labels seen include "season long" plus passing yards,
#                passing TDs, rushing yards, rushing TDs, receiving yards and
#                receptions - the broadest stat coverage of anything probed.
#                Predicted to need a lifted auth token and a Cloudflare
#                fight. It needed neither; the _ak in the query string was
#                enough.
#   DraftKings   200, 367 KB, on the nash host. Season-ish and stat labels
#                both present. The legacy sportsbook.draftkings.com host is
#                403 Access Denied, and nash /categories is 404 - so the
#                league feed is the entry point, not the category index.
#   Pinnacle     200, 356 KB, guest API, no auth. "season long" plus passing,
#                rushing and receiving yards. Predicted to have essentially
#                no season-long player props. It has them.
#   Bovada       Reachable and friendly - and EMPTY where it counts. Every
#                season-prop leaf returns "[]" (2 bytes) even though the nav
#                tree advertises 21 events under them. The game coupon
#                returns 44 KB perfectly well, so this is not a block, a bad
#                path or a filter - the season content simply is not served
#                through the coupon endpoint to us. Also rate-limits to 429
#                partway through a second run. Predicted to be the #1 target
#                on the strength of answering a request; answering turned out
#                not to be the same as having the data.
#   Caesars      403. A refusal. Left alone. Predicted "least defended".
#   BetMGM       400 "Access id missing" - needs the accessid from a page
#                load, as expected.
#
# The lesson worth keeping: "the host answered" and "the host has what we
# need" are different tests, and only the second one matters. The summary
# table below reports both, which is why it has a season-props column.
#
# Bovada's nav map is still recorded below, since it cost something to find
# and the paths are correct even though the coupons are empty:
#
#     /football/nfl-season-player-props   21   (empty via coupon)
#     /football/nfl-regular-season-wins   32
#     /football/nfl-futures               15
#     /football/nfl-awards                 8
#     /football/nfl-season-props           6
#     /football/nfl                       20   (games - this one works)
BOVADA_COUPON = 'https://www.bovada.lv/services/sports/event/coupon/events/A/description/'

# DraftKings. The league feed answers and carries only game lines - but it
# ships the whole category catalogue alongside them, which is how these IDs
# were found after the /categories endpoint itself returned 404. Category
# 1759 "Player Futures" is the season-long player board, and its subcategory
# list is the widest of any book probed: DK is the only one of the four that
# prices RECEPTIONS and RECEIVING TDs season-long, which are exactly the two
# markets FanDuel and Pinnacle leave us guessing at.
DK_NASH = 'https://sportsbook-nash.draftkings.com/api/sportscontent/dkusoh/v1'
DK_PLAYER_FUTURES = 1759
DK_PLAYER_FUTURE_SUBCATEGORIES = [
    ('Passing Yards', 17147), ('Passing TDs', 17148),
    ('Rushing Yards', 17223), ('Rushing TDs', 17224),
    ('Receiving Yards', 17314), ('Receiving TDs', 17315),
    ('Receptions', 20168), ('Sacks', 17316),
]

# THE THING THAT MAKES DK WORTH FIGHTING FOR: receptions and receiving TDs.
# Of the four books already wired in, only PrizePicks prices receptions
# season-long, so in a PPR league the single largest scoring input rests on
# one source with nothing to check it against.
DK_RECEPTIONS = 20168

# Game Lines / "Game". Found in the league feed's own subscriptionPartials,
# and it is the whole reason the bare league call looks like it has no
# futures: that call is not unfiltered, it is filtered TO THIS.
DK_GAME_SUBCATEGORY = 4518

# The league feed ships its own subscription definitions, and they give up
# the query grammar:
#
#   "league-events-88808": {
#     "entity": "events",
#     "query": "$filter=leagueId eq '88808' and
#               clientMetadata/Subcategories/any(s: s/Id eq '4518')",
#     "includeMarkets": "$filter=tags/all(t: t ne 'SportcastBetBuilder') and
#                        clientMetadata/subCategoryId eq '4518'"
#   }
#
# That is OData, and it explained why the league feed looks game-only. It was
# NOT the way in, though: passing those filters back is accepted and silently
# IGNORED - the response comes back byte-identical to the unfiltered one, 225
# game markets and all. Right about the diagnosis, wrong about the cure.
#
# WHAT ACTUALLY WORKS (measured 2026-08-10), and it is the boring one:
#
#     /leagues/88808/categories/1759/subcategories/{sub}
#
# 200, and every market in it is season-long. The earlier 404 on /categories
# was a route with no id, which is a different thing from the route being
# unavailable.
#
#   Receiving Yards  73   Receiving TDs  51   Receptions   43   Rushing Yards 41
#   Sacks            34   Rushing TDs    27   Pass Yards   25   Passing TDs   25
#
# EIGHT CALLS, NOT ONE. The category route without a subcategory returns 25
# markets - exactly the Passing Yards count, i.e. one default subcategory
# rather than the union. Nothing about the response says it is partial, so
# a single category call would look like a working adapter that quietly saw
# a seventh of the board.
#
# Legacy v5 is closed for good: 403 on the generic host and on the state
# host, so the Akamai refusal is the API's answer and not that hostname's.
DK_STATE_HOSTS = ['dkusoh', 'dkusnj', 'dkusmi', 'dkusva', 'dkusdc']

# The coupon endpoint only serves LEAF paths. /football/nfl-season-player-props
# is a branch, so asking it for a coupon returns an empty body - the same
# blank-page symptom as a wrong path, which is why the nav tree has to be
# walked all the way down rather than one level. Its children, with the event
# counts the nav reported:
#
#   regular-season-stat-leaders   9   CATEGORY     (league leaders, winner-style)
#   quarterbacks                  4
#   running-backs                 2
#   tight-ends                    2
#   wide-receivers                2
#   defensive-players-sacks       1
#   regular-season-milestones     1   COMPETITION  (X+ style milestones)
#
# Fantasy-relevant scoring lives in the four position leaves. The stat-leader
# and milestone groups are a different shape (winner markets and X+ ladders)
# and are pulled here only so their structure is on record.
BOVADA_SEASON_PROP_LEAVES = [
    'quarterbacks', 'running-backs', 'wide-receivers', 'tight-ends',
    'regular-season-stat-leaders', 'regular-season-milestones',
    'defensive-players-sacks',
]
BOOK_PROBES = [
    ('Bovada - nav tree (lists the valid NFL paths)', [
        'https://www.bovada.lv/services/sports/event/v2/nav/A/description/football',
    ]),
    ('Bovada - nav under season player props (per-stat paths)', [
        'https://www.bovada.lv/services/sports/event/v2/nav/A/description/'
        'football/nfl-season-player-props',
    ]),
] + [
    (f'Bovada - season player props: {leaf}', [
        BOVADA_COUPON + f'football/nfl-season-player-props/{leaf}?marketFilterId=def&lang=en',
    ])
    for leaf in BOVADA_SEASON_PROP_LEAVES
] + [
    ('Bovada - NFL season team props', [
        BOVADA_COUPON + 'football/nfl-season-props?marketFilterId=def&lang=en',
    ]),
    ('Bovada - NFL regular season wins', [
        BOVADA_COUPON + 'football/nfl-regular-season-wins?marketFilterId=def&lang=en',
    ]),
    ('Bovada - NFL game coupon', [
        BOVADA_COUPON + 'football/nfl?marketFilterId=def&preMatchOnly=true&lang=en',
    ]),
    ('DraftKings - NFL league feed', [
        'https://sportsbook-nash.draftkings.com/api/sportscontent/dkusoh/v1/leagues/88808',
        'https://sportsbook-nash.draftkings.com/api/sportscontent/dkusdc/v1/leagues/88808',
        'https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/88808?format=json',
    ]),
    ('DraftKings - NFL categories (futures live here)', [
        'https://sportsbook-nash.draftkings.com/api/sportscontent/dkusoh/v1/leagues/88808/categories',
        'https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/88808/categories?format=json',
    ]),
] + [
    (f'DraftKings - Player Futures: {name}', [
        f'{DK_NASH}/leagues/88808/categories/{DK_PLAYER_FUTURES}/subcategories/{sub}',
    ])
    for name, sub in DK_PLAYER_FUTURE_SUBCATEGORIES
] + [
    ('Caesars - NFL schedule', [
        'https://api.americanwagering.com/regions/us/locations/oh/brands/czr/sb/v3/sports/'
        'americanfootball/events/schedule?competitionId=007d7c61-07a7-4e18-bb40-15104b25eaf8',
        'https://api.americanwagering.com/regions/us/locations/nj/brands/czr/sb/v3/sports/'
        'americanfootball/events/schedule',
    ]),
    ('BetMGM - NFL fixtures', [
        'https://sports.oh.betmgm.com/cds-api/bettingoffer/fixtures'
        '?x-bwin-accessid=&lang=en-us&country=US&userCountry=US&sportIds=11&fixtureTypes=Standard',
    ]),
    ('FanDuel - NFL page', [
        'https://sbapi.oh.sportsbook.fanduel.com/api/content-managed-page'
        '?page=CUSTOM&customPageId=nfl&_ak=FhMFpcPWXMeyZxOx&timezone=America%2FNew_York',
    ]),
    ('Pinnacle - guest NFL matchups', [
        'https://guest.api.arcadia.pinnacle.com/0.1/leagues/889/matchups',
    ]),
]

SEASON_HINTS = ('season long', 'season-long', 'regular season', 'season total',
                'nflszn', 'season props', 'to lead the league', 'futures')
STAT_HINTS = ('passing yards', 'pass yards', 'rushing yards', 'receiving yards',
              'receptions', 'passing touchdowns', 'rushing touchdowns')


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get(url, params=None):
    """One GET. Returns (status, headers, payload_or_none, text, error)."""
    try:
        resp = requests.get(url, params=params or {}, headers=DEFAULT_HEADERS, timeout=TIMEOUT)
    except Exception as exc:
        return None, {}, None, '', str(exc)
    text = resp.text or ''
    payload = None
    if resp.status_code == 200:
        try:
            payload = resp.json()
        except ValueError:
            pass
    return resp.status_code, dict(resp.headers), payload, text, None


def _quota(headers):
    used = headers.get('x-requests-used')
    left = headers.get('x-requests-remaining')
    if used is None and left is None:
        return ''
    return f"  [credits used {used}, remaining {left}]"


def _save(save_dir, name, payload):
    if not save_dir or payload is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{name}.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=1)
    print(f"    saved -> {path}")


def _hits(text, needles):
    low = text.lower()
    return sorted({n for n in needles if n in low})


def _slug(label):
    """Filename-safe slug of the FULL probe label.

    The first version keyed the filename on the book name alone, so all
    twelve Bovada probes wrote to book_bovada_www.json and each overwrote
    the last - the eleven interesting results were destroyed by the twelfth
    before anyone could look at them. The whole label goes in the name now.
    """
    keep = [c.lower() if c.isalnum() else '_' for c in label]
    return ''.join(keep).strip('_').replace('__', '_')


# ---------------------------------------------------------------------------
# Part A - The Odds API
# ---------------------------------------------------------------------------

def probe_odds_api(key, spend, save_dir):
    print("\n" + "=" * 72)
    print("PART A - THE ODDS API")
    print("=" * 72)

    # 1. Sports list. Free (0 credits), and it is the whole answer to "what
    #    NFL products does my key see", including every futures/outright key.
    status, headers, payload, text, err = _get(f'{ODDS_API}/sports/',
                                               {'apiKey': key, 'all': 'true'})
    if err or status != 200:
        print(f"  /sports failed: {err or f'{status}: {text[:200]}'}")
        return
    print(f"  /sports OK - free call.{_quota(headers)}")
    _save(save_dir, 'oddsapi_sports', payload)

    nfl = [s for s in payload if 'americanfootball_nfl' in str(s.get('key', ''))]
    print(f"\n  NFL-related sport keys your plan can see ({len(nfl)}):")
    for s in nfl:
        flags = []
        if s.get('has_outrights'):
            flags.append('OUTRIGHTS/FUTURES')
        if s.get('active'):
            flags.append('active')
        print(f"    {s['key']:<46} {s.get('title', '')}  {' '.join(flags)}")

    outright_keys = [s['key'] for s in nfl if s.get('has_outrights')]
    if not outright_keys:
        print("\n  No NFL outright/futures sport key is visible to this key.")

    # 2. Futures shape. The decisive question is not "are there futures" but
    #    "does a futures outcome carry a POINT". A season passing-yards
    #    over/under needs a number attached to the selection; a Super Bowl
    #    winner market does not. If every outcome comes back without a point,
    #    the format itself cannot express a season-long player total.
    if outright_keys and spend:
        for sk in outright_keys:
            status, headers, payload, text, err = _get(
                f'{ODDS_API}/sports/{sk}/odds',
                {'apiKey': key, 'regions': 'us', 'markets': 'outrights', 'oddsFormat': 'american'})
            print(f"\n  {sk} outrights -> {status}{_quota(headers)}")
            if status != 200:
                print(f"    {text[:200]}")
                continue
            _save(save_dir, f'oddsapi_outrights_{sk}', payload)
            with_point, total, market_keys = 0, 0, set()
            for ev in payload or []:
                for book in ev.get('bookmakers') or []:
                    for m in book.get('markets') or []:
                        market_keys.add(m.get('key'))
                        for o in m.get('outcomes') or []:
                            total += 1
                            if o.get('point') is not None:
                                with_point += 1
            print(f"    {len(payload or [])} events, markets={sorted(market_keys)}")
            print(f"    {total} outcomes, {with_point} of them carry a numeric point")
            if total and not with_point:
                print("    -> winner-style futures only. No over/under number is present,")
                print("       so this cannot express a season-long player stat total.")
    elif outright_keys:
        print("\n  (futures shape not checked - re-run with --spend; costs 1 credit each)")

    # 3. Candidate season market keys. An unknown market key is rejected with
    #    a 4xx that names it, which is a clean "no such market". The credit
    #    headers are printed either way, so whatever this actually costs is
    #    visible rather than assumed.
    status, headers, events, text, err = _get(f'{ODDS_API}/sports/americanfootball_nfl/events',
                                              {'apiKey': key})
    if status != 200 or not events:
        print(f"\n  /events returned {status} - no NFL events to test markets against.")
        print(f"    {text[:200]}")
        return
    print(f"\n  /events OK - free call. {len(events)} upcoming NFL events.{_quota(headers)}")
    # The SOONEST event, not events[0]. Books post player props a few days
    # out, so measuring breadth against whatever the list happened to return
    # first can sample a game months away and report "nothing is posted" as
    # though it were a fact about the API rather than about the calendar.
    # Days-to-kickoff is printed for the same reason: without it the market
    # table is uninterpretable.
    def _kick(ev):
        try:
            return datetime.fromisoformat(str(ev.get('commence_time', '')).replace('Z', '+00:00'))
        except Exception:
            return datetime.max.replace(tzinfo=timezone.utc)
    soonest = min(events, key=_kick)
    ev_id = soonest['id']
    days = (_kick(soonest) - datetime.now(timezone.utc)).days
    print(f"  Testing markets against {soonest.get('away_team')} @ {soonest.get('home_team')}"
          f" - kickoff in {days} days ({soonest.get('commence_time')})")
    if days > 3:
        print("  NOTE: that is far enough out that most player props will not be")
        print("        posted yet. Breadth measured here is a floor, not the ceiling.")

    print("\n  Do season-long player market keys exist?")
    status, headers, payload, text, err = _get(
        f'{ODDS_API}/sports/americanfootball_nfl/events/{ev_id}/odds',
        {'apiKey': key, 'regions': 'us', 'oddsFormat': 'american',
         'markets': ','.join(CANDIDATE_SEASON_MARKETS)})
    if status == 422:
        print(f"    422 as expected - the API names what it rejected:")
        print(f"    {text[:400]}")
        print("    Any key NOT named in that message is a real market. Read it carefully.")
    elif status == 200:
        got = {m.get('key') for b in (payload or {}).get('bookmakers') or []
               for m in b.get('markets') or []}
        print(f"    200 - accepted. Markets returned: {sorted(got) or 'none (accepted but empty)'}")
        _save(save_dir, 'oddsapi_season_candidates', payload)
    else:
        print(f"    {status}: {text[:300]}")

    # 4. Breadth of in-season GAME props, which is the separate question of
    #    what the key is actually good for. One call, all documented markets,
    #    and the credit cost printed - cost is [markets] x [regions].
    if spend:
        print(f"\n  In-season GAME player props - all {len(ODDS_API_PLAYER_PROP_MARKETS)} "
              f"documented markets, one event, one region.")
        used_before = headers.get('x-requests-used')
        status, headers, payload, text, err = _get(
            f'{ODDS_API}/sports/americanfootball_nfl/events/{ev_id}/odds',
            {'apiKey': key, 'regions': 'us', 'oddsFormat': 'american',
             'markets': ','.join(ODDS_API_PLAYER_PROP_MARKETS)})
        print(f"    -> {status}{_quota(headers)}")
        if status == 200:
            _save(save_dir, 'oddsapi_game_props', payload)
            per = {}
            for book in (payload or {}).get('bookmakers') or []:
                for m in book.get('markets') or []:
                    e = per.setdefault(m.get('key'), {'books': set(), 'players': set()})
                    e['books'].add(book.get('key'))
                    for o in m.get('outcomes') or []:
                        if o.get('description'):
                            e['players'].add(o['description'])
            print(f"    {len((payload or {}).get('bookmakers') or [])} books answered.")
            print(f"    {'market':<34} {'books':>6} {'players':>8}")
            for mk in ODDS_API_PLAYER_PROP_MARKETS:
                e = per.get(mk)
                if e:
                    print(f"    {mk:<34} {len(e['books']):>6} {len(e['players']):>8}")
                else:
                    print(f"    {mk:<34} {'-':>6} {'-':>8}   not posted")
            missing = [m for m in ODDS_API_PLAYER_PROP_MARKETS if m not in per]
            print(f"\n    {len(per)}/{len(ODDS_API_PLAYER_PROP_MARKETS)} markets posted for this game.")
            if missing:
                print(f"    absent: {', '.join(missing)}")
            # COST IS BILLED ON MARKETS RETURNED, NOT MARKETS REQUESTED.
            # Measured 2026-08-09: a request naming all 20 markets, of which
            # only 2 were posted, moved the counter by 2 - not 20. That is a
            # much friendlier model than the documented "[markets] x
            # [regions]" reads, and it means asking for everything is close
            # to free when a book has posted little. Computed here rather
            # than assumed, because it is the number that decides whether a
            # weekly pull fits in the plan.
            try:
                spent = int(headers.get('x-requests-used')) - int(used_before)
            except (TypeError, ValueError):
                spent = None
            if spent is not None:
                print(f"\n    COST: that call named {len(ODDS_API_PLAYER_PROP_MARKETS)} markets, "
                      f"got {len(per)} back, and cost {spent} credits.")
                print(f"    Billing follows markets RETURNED, not requested.")
                if len(per):
                    print(f"    A 16-game slate at this posting level: ~{16 * spent} credits.")
                    print(f"    In-season, with most markets up, expect ~{16 * 12}-{16 * 18}.")
                left = headers.get('x-requests-remaining')
                if left:
                    print(f"    You have {left} left this period - budget accordingly.")
        else:
            print(f"    {text[:300]}")
    else:
        print("\n  (game-prop breadth not measured - re-run with --spend; costs "
              f"{len(ODDS_API_PLAYER_PROP_MARKETS)} credits)")


# ---------------------------------------------------------------------------
# Part B - sportsbooks, one honest request each
# ---------------------------------------------------------------------------

def _dk_candidates(sub=DK_RECEPTIONS, cat=DK_PLAYER_FUTURES, site='dkusoh'):
    """
    Every shape worth trying for one DraftKings subcategory, ordered by how
    much evidence there is for it. Each entry is (label, url, params).

    Tier 1 is the conventional REST nesting. Tier 2 is OData, using the exact
    grammar DraftKings' own subscriptionPartials publishes - the strongest
    lead here, because the bare league call is already a filtered call and
    the only thing that appears to change is the subcategory id. Tier 3 is
    the legacy v5 API on a STATE host rather than the generic one; the
    generic host answered 403 from Akamai, which is a property of that
    hostname's WAF and not necessarily of the API behind it.
    """
    nash = f'https://sportsbook-nash.draftkings.com/api/sportscontent/{site}/v1'
    odata_events = (f"leagueId eq '88808' and "
                    f"clientMetadata/Subcategories/any(s: s/Id eq '{sub}')")
    odata_markets = f"clientMetadata/subCategoryId eq '{sub}'"
    return [
        # --- tier 1: REST nesting
        ('nested subcategory', f'{nash}/leagues/88808/categories/{cat}/subcategories/{sub}', None),
        ('category only', f'{nash}/leagues/88808/categories/{cat}', None),
        ('league subcategory', f'{nash}/leagues/88808/subcategories/{sub}', None),
        ('bare category', f'{nash}/categories/{cat}', None),
        ('bare subcategory', f'{nash}/subcategories/{sub}', None),
        # --- tier 2: OData, grammar taken from DK's own subscriptionPartials
        ('odata on league', f'{nash}/leagues/88808',
         {'$filter': f"clientMetadata/Subcategories/any(s: s/Id eq '{sub}')"}),
        ('odata on league + markets', f'{nash}/leagues/88808',
         {'$filter': f"clientMetadata/Subcategories/any(s: s/Id eq '{sub}')",
          'includeMarkets': f'$filter={odata_markets}'}),
        ('odata on events entity', f'{nash}/events',
         {'$filter': odata_events, 'includeMarkets': f'$filter={odata_markets}'}),
        ('subcategoryId query param', f'{nash}/leagues/88808', {'subcategoryId': str(sub)}),
        # --- tier 3: legacy v5 on a state host, not the generic one
        ('legacy v5 state host',
         f'https://sportsbook-us-oh.draftkings.com/sites/US-OH-SB/api/v5/eventgroups/88808'
         f'/categories/{cat}/subcategories/{sub}', {'format': 'json'}),
        ('legacy v5 state host, league only',
         'https://sportsbook-us-oh.draftkings.com/sites/US-OH-SB/api/v5/eventgroups/88808',
         {'format': 'json'}),
    ]


def _dk_verdict(payload, text):
    """Did this response actually contain season-long player lines?"""
    if not isinstance(payload, dict):
        return 'not JSON object', 0, 0
    sel = payload.get('selections') or []
    mkts = payload.get('markets') or []
    names = {str(m.get('name') or '') for m in mkts if isinstance(m, dict)}
    season = {n for n in names if 'Regular Season' in n or 'Season' in n}
    return (f"{len(mkts)} markets / {len(sel)} selections", len(season), len(names))


def probe_draftkings(save_dir):
    """
    A dedicated, harder run at DraftKings alone.

    Worth its own pass because DK is the only book probed that prices
    RECEPTIONS and RECEIVING TDs season-long, and receptions is the largest
    single scoring input in a PPR league - currently resting on PrizePicks
    with nothing to cross-check it.
    """
    print("\n" + "=" * 72)
    print("DRAFTKINGS - going through every shape, one subcategory (Receptions)")
    print("=" * 72)
    print("  Anything that answers 200 with markets whose names carry 'Regular")
    print("  Season' is the endpoint we want.\n")

    winners = []
    for label, url, params in _dk_candidates():
        status, headers, payload, text, err = _get(url, params)
        if err:
            print(f"  {label:<30} network error: {err[:60]}")
            continue
        shape, season, total = _dk_verdict(payload, text)
        flag = ''
        if status == 200 and season:
            flag = f'  <<< {season} season markets'
            winners.append((label, url, params))
        elif status == 200:
            flag = f'  ({total} market names, none season-long)'
        print(f"  {label:<30} {status}  {len(text):>9,}b  {shape}{flag}")
        if status == 200 and payload is not None:
            _save(save_dir, 'dk_' + _slug(label), payload)
        time.sleep(0.6)

    if not winners:
        print("\n  No shape returned season-long markets.")
        print("  Next thing to try, and it needs a browser rather than this script:")
        print("    open sportsbook.draftkings.com/leagues/football/nfl?category=player-futures")
        print("    with devtools Network open, filter to 'sportscontent', and copy the")
        print("    request the page itself makes. That is the ground truth, and it is")
        print("    your own browser session rather than anything pretending to be one.")
        return

    print(f"\n  {len(winners)} shape(s) worked. Pulling every stat through the first one.")
    label, url, params = winners[0]
    for name, sub in DK_PLAYER_FUTURE_SUBCATEGORIES:
        _, u2, p2 = next(c for c in _dk_candidates(sub=sub) if c[0] == label)
        status, headers, payload, text, err = _get(u2, p2)
        shape, season, total = _dk_verdict(payload, text) if not err else ('error', 0, 0)
        print(f"    {name:<18} {status}  {shape}  season-markets={season}")
        if status == 200 and payload is not None:
            _save(save_dir, f'dk_futures_{name.lower().replace(" ", "_")}', payload)
        time.sleep(0.6)


def probe_books(save_dir):
    print("\n" + "=" * 72)
    print("PART B - SPORTSBOOKS (one plain GET each, honest User-Agent)")
    print("=" * 72)
    results = []
    for i, (label, urls) in enumerate(BOOK_PROBES):
        # Bovada started returning 429 partway through the second run of the
        # day. A probe that trips a rate limit reports "declined" for sources
        # that would have answered, which is a false negative in the one
        # place we can least afford one. One second between requests costs
        # nothing here and keeps the result honest.
        if i:
            time.sleep(1.0)
        print(f"\n  {label}")
        answered = False
        unreachable = False
        for url in urls:
            status, headers, payload, text, err = _get(url)
            host = url.split('/')[2]
            if err:
                print(f"    {host:<44} network error: {err[:80]}")
                unreachable = True
                continue
            size = len(text)
            if status == 200 and payload is not None:
                season = _hits(text, SEASON_HINTS)
                stats = _hits(text, STAT_HINTS)
                print(f"    {host:<44} 200  {size:>9,} bytes  JSON")
                print(f"      season-ish labels seen: {season or 'none'}")
                print(f"      stat labels seen:       {stats or 'none'}")
                _save(save_dir, 'book_' + _slug(label), payload)
                results.append((label, 'JSON', bool(season and stats)))
                answered = True
                break
            if status == 200:
                print(f"    {host:<44} 200  {size:>9,} bytes  but not JSON "
                      f"(HTML shell or challenge page)")
                results.append((label, '200 non-JSON', False))
                answered = True
                break
            print(f"    {host:<44} {status}  {text[:90].strip()!r}")
        if not answered:
            # "unreachable" is a property of the network this ran on, not of
            # the book - say so, rather than recording a sandbox egress rule
            # as the sportsbook refusing us.
            results.append((label, 'unreachable' if unreachable else 'declined', False))

    print("\n  " + "-" * 68)
    print(f"  {'source':<48} {'result':<14} season props?")
    for label, res, season in results:
        print(f"  {label:<48} {res:<14} {'yes' if season else 'no/unclear'}")
    print("\n  'unreachable' means this machine could not open the connection at all")
    print("  (blocked network, VPN, or geo-fence) - retry from a normal connection.")
    print("  A 403 or a challenge page is a 'no' and is left alone. For those,")
    print("  the browser-save path is the honest route: open the URL in your own")
    print("  logged-in browser, save the JSON, and load it the same way the")
    print("  Underdog and PrizePicks files already load.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--odds-api-key', default=os.environ.get('ODDS_API_KEY', ''))
    ap.add_argument('--spend', action='store_true',
                    help='allow the calls that cost Odds API credits')
    ap.add_argument('--books-only', action='store_true')
    ap.add_argument('--draftkings', action='store_true',
                    help='only the deep DraftKings hunt, nothing else')
    ap.add_argument('--save-dir', default='',
                    help='write every payload here for offline inspection')
    args = ap.parse_args()

    print(f"probe_season_odds  {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")

    if args.draftkings:
        probe_draftkings(args.save_dir)
        print("\ndone.")
        return

    if not args.books_only:
        key = args.odds_api_key.strip()
        if not key:
            try:
                from data.loaders import load_saved_odds_api_key
                key = load_saved_odds_api_key()
            except Exception:
                key = ''
        if key:
            probe_odds_api(key, args.spend, args.save_dir)
        else:
            print("\n  No Odds API key given (--odds-api-key, $ODDS_API_KEY, or the one")
            print("  saved by the Live Odds tab). Skipping Part A.")

    probe_books(args.save_dir)
    print("\ndone.")


if __name__ == '__main__':
    main()
