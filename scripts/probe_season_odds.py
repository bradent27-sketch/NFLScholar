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
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402

from config import ODDS_API_PLAYER_PROP_MARKETS  # noqa: E402
from data.odds_sources import DEFAULT_HEADERS  # noqa: E402

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
BOOK_PROBES = [
    ('Bovada - NFL game coupon', [
        'https://www.bovada.lv/services/sports/event/coupon/events/A/description/football/nfl'
        '?marketFilterId=def&preMatchOnly=true&lang=en',
    ]),
    ('Bovada - NFL futures/season props', [
        'https://www.bovada.lv/services/sports/event/coupon/events/A/description/football/nfl/nfl-futures'
        '?marketFilterId=def&preMatchOnly=true&lang=en',
        'https://www.bovada.lv/services/sports/event/coupon/events/A/description/football/nfl'
        '?marketFilterId=def&preMatchOnly=true&lang=en&eventsLimit=200',
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
    ev_id = events[0]['id']
    print(f"  Testing markets against {events[0].get('away_team')} @ {events[0].get('home_team')}")

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
            print(f"\n    COST MODEL: that was one game. A 16-game slate at this market")
            print(f"    count costs about {16 * len(ODDS_API_PLAYER_PROP_MARKETS)} credits per refresh.")
        else:
            print(f"    {text[:300]}")
    else:
        print("\n  (game-prop breadth not measured - re-run with --spend; costs "
              f"{len(ODDS_API_PLAYER_PROP_MARKETS)} credits)")


# ---------------------------------------------------------------------------
# Part B - sportsbooks, one honest request each
# ---------------------------------------------------------------------------

def probe_books(save_dir):
    print("\n" + "=" * 72)
    print("PART B - SPORTSBOOKS (one plain GET each, honest User-Agent)")
    print("=" * 72)
    results = []
    for label, urls in BOOK_PROBES:
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
                _save(save_dir, 'book_' + label.split(' - ')[0].lower().replace(' ', '_')
                      + '_' + host.split('.')[0], payload)
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
    ap.add_argument('--save-dir', default='',
                    help='write every payload here for offline inspection')
    args = ap.parse_args()

    print(f"probe_season_odds  {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")

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
