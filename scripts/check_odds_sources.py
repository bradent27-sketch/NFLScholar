"""
Live check for the odds adapters. Run this on a normal network.

WHY IT EXISTS SEPARATELY FROM THE TESTS. tests/test_odds_sources.py proves
the PARSING is right against recorded payloads and needs no network. This
proves the LIVE SHAPE still matches what those payloads describe - a
different question, and the one that goes stale, because these are private
endpoints that can be renamed or restructured without notice.

Run it when something looks wrong, or once before a draft:

    python scripts/check_odds_sources.py
    python scripts/check_odds_sources.py --save-fixtures   # refresh the recordings
    python scripts/check_odds_sources.py --odds-api-key KEY  # list your books

It reports rather than raises. A source being blocked is an ordinary result
and is printed as such.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
pd.options.mode.string_storage = "python"

from data.odds_sources import (  # noqa: E402
    unmapped_markets, PRIZEPICKS_PROJECTIONS_URL, PRIZEPICKS_NFL_LEAGUE_ID, PRIZEPICKS_LEAGUES_URL,
    _get_json, fetch_underdog_payload, parse_underdog_payload,
    parse_prizepicks_payload, odds_api_bookmakers, prizepicks_leagues,
    discover_prizepicks_league, implausible_period_rows,
)

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'tests', 'fixtures')


def _report(label, props, err):
    print(f"\n=== {label}")
    if err:
        print(f"  no data: {err}")
        return
    print(f"  {len(props)} lines, {props['player'].nunique()} players")
    seasons = props[props['period'] == 'season']
    print(f"  season-long: {len(seasons)} lines across {seasons['player'].nunique()} players")
    if seasons.empty:
        print("  NOTE: no season-long lines. These are posted in the preseason and")
        print("        pulled once the season starts, so this is expected in-season.")
    odd = implausible_period_rows(props)
    if odd is not None and len(odd):
        print(f"  WARNING: {len(odd)} lines are far too large for the 'game' period they")
        print("           claim - the season flag is probably being missed. Sample:")
        for _, r in odd.head(4).iterrows():
            print(f"      {r['player']} {r['market_raw']} = {r['line']}")
    shaded = int(props['provider'].str.contains(r'\(', regex=True).sum())
    if shaded:
        print(f"  {shaded} altered lines (demon/goblin) kept but not scored - they are")
        print("     deliberately shaded off the true median.")
    unmapped = unmapped_markets(props)['market_raw'].value_counts()
    if len(unmapped):
        print(f"  markets not mapped to a stat column ({len(unmapped)} kinds):")
        for market, count in unmapped.head(12).items():
            print(f"      {market!r}  x{count}")
        print("      ^ add real ones to STAT_ALIASES in data/odds_sources.py;")
        print("        combination markets belong in COMBO_STATS instead.")
    blank_team = int((props['team'] == '').sum())
    if blank_team:
        print(f"  {blank_team} lines had a team this app couldn't resolve "
              f"(sample: {sorted(set(props[props['team'] == '']['player']))[:5]})")
    print("  sample:")
    cols = ['player', 'team', 'position', 'market_raw', 'market', 'line', 'period']
    print(props[cols].head(8).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save-fixtures', action='store_true',
                    help="overwrite tests/fixtures/*.json with the live payloads")
    ap.add_argument('--odds-api-key', default=os.environ.get('ODDS_API_KEY'),
                    help="list which bookmakers your Odds API key actually returns")
    args = ap.parse_args()

    payload, url, err = fetch_underdog_payload()
    if err:
        print(f"\n=== Underdog\n  no data: {err}")
    else:
        props, perr = parse_underdog_payload(payload)
        _report('Underdog', props, perr)
        print(f"  endpoint that answered: {url}")
        if args.save_fixtures and not perr:
            path = os.path.join(FIXTURES, 'underdog_live.json')
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(payload, fh, indent=2)
            print(f"  saved live payload -> {path}")

    # Every league they run, so the season-long one can be found by NAME
    # rather than by a hardcoded id. This is the single most useful thing
    # this script prints: season-long lives in its own league (NFLSZN), and
    # asking the weekly board for it returns week-one props forever.
    print("\n=== PrizePicks leagues")
    leagues_payload, leagues_err = _get_json(PRIZEPICKS_LEAGUES_URL)
    league_id = PRIZEPICKS_NFL_LEAGUE_ID
    if leagues_err:
        print(f"  couldn't list leagues: {leagues_err}")
    else:
        found = prizepicks_leagues(leagues_payload)
        football = {i: n for i, n in found.items() if 'nfl' in n.lower().replace(' ', '')}
        print(f"  {len(found)} leagues; football ones: "
              + (', '.join(f'{n} (id {i})' for i, n in football.items()) or 'none'))
        season_id, season_name, derr = discover_prizepicks_league()
        if season_id:
            print(f"  SEASON-LONG league: {season_name} (id {season_id}) <- using this")
            league_id = season_id
        else:
            print(f"  no season-long league matched: {derr}")

    payload, err = _get_json(PRIZEPICKS_PROJECTIONS_URL,
                             params={'league_id': league_id, 'per_page': 1000})
    if err:
        print(f"\n=== PrizePicks\n  no data: {err}")
        print("  PrizePicks sits behind Cloudflare and refuses automated requests")
        print("  much of the time. That is their call and nothing here tries to")
        print("  get around it - Underdog carries the same season-long lines.")
    else:
        props, perr = parse_prizepicks_payload(payload)
        _report('PrizePicks', props, perr)
        if args.save_fixtures and not perr:
            path = os.path.join(FIXTURES, 'prizepicks_live.json')
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(payload, fh, indent=2)
            print(f"  saved live payload -> {path}")

    print("\n=== The Odds API")
    if not args.odds_api_key:
        print("  skipped - pass --odds-api-key or set ODDS_API_KEY to see exactly")
        print("  which books your key returns (coverage varies by plan and region).")
    else:
        from data.loaders import fetch_nfl_odds
        data, oerr, remaining = fetch_nfl_odds(args.odds_api_key)
        if oerr:
            print(f"  no data: {oerr}")
        else:
            books = odds_api_bookmakers(data)
            print(f"  {len(data)} events, {len(books)} bookmakers, "
                  f"{remaining} requests left this period")
            print(books.to_string(index=False))
            for name in ('fanduel', 'draftkings', 'prizepicks', 'underdog'):
                mark = 'YES' if name in set(books['key']) else 'no '
                print(f"    {mark}  {name}")

    print("\nParsing is covered offline by tests/test_odds_sources.py "
          "(python tests/test_odds_sources.py).")


if __name__ == '__main__':
    main()
