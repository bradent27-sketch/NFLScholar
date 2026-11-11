"""
The Odds API event props for line shopping (2026-11-11, audit B4), spent conservatively at the user's request ("it shouldn't
waste pulls, be conservative so that we don't run out").

  - only the stats the bet tiers use (receiving yards, receptions, rushing yards: data/prop_tiers.py), one region (us);
  - the cost is estimated before any paid call (markets x regions x games, The Odds API's own rule) from the free events
    call, which also reports the credits left; nothing is spent unless `reserve` credits would remain afterwards;
  - at most one pull per MIN_HOURS_BETWEEN hours, whatever calls it.

Rows go to their own folder (external_data/odds_api_archive), not the props archive: data.odds_weekly.players_with_lines reads
every archived line for v2_ps_line_routing, and extra books there would move projections. Providers are named
'OddsAPI <book>'. Read with load_odds_api_archive; data/line_shopping.py uses them for the best price per side.
"""
from __future__ import annotations

import datetime
import glob
import json
import os

import pandas as pd
import requests

ARCHIVE_DIR = os.path.join('external_data', 'odds_api_archive')
STATE_PATH = os.path.join('.sweeps', 'scheduled', 'odds_api_state.json')
BASE = 'https://api.the-odds-api.com/v4/sports/americanfootball_nfl'
MARKETS = {'player_reception_yds': 'receiving_yards', 'player_receptions': 'receptions', 'player_rush_yds': 'rushing_yards'}
REGIONS = 'us'
DEFAULT_RESERVE = 100
MIN_HOURS_BETWEEN = 20
WEEK_DAYS = 7


def estimate_cost(n_events, markets=MARKETS, regions=REGIONS):
    """Credits an event-odds pull of n_events games costs: markets x regions per game."""
    return int(n_events) * len(markets) * len(str(regions).split(','))


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def _last_pull():
    try:
        with open(STATE_PATH, encoding='utf-8') as fh:
            return datetime.datetime.fromisoformat(json.load(fh)['last_pull'])
    except (OSError, ValueError, KeyError):
        return None


def _save_state(**fields):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, 'w', encoding='utf-8') as fh:
        json.dump(fields, fh)


def week_events(api_key, now=None, session=requests):
    """([event dicts] of the games starting within WEEK_DAYS, credits remaining or None, error). The events call is free."""
    now = now or _now()
    try:
        resp = session.get(f'{BASE}/events', params={'apiKey': api_key}, timeout=15)
    except Exception as exc:
        return [], None, f'{type(exc).__name__}: {exc}'
    remaining = resp.headers.get('x-requests-remaining')
    remaining = int(float(remaining)) if remaining not in (None, '') else None
    if resp.status_code != 200:
        return [], remaining, f'HTTP {resp.status_code}: {resp.text[:200]}'
    events = []
    for ev in resp.json() or []:
        start = pd.Timestamp(ev.get('commence_time'))
        if start.tzinfo is None:
            start = start.tz_localize('UTC')
        if pd.Timestamp(now) <= start <= pd.Timestamp(now) + pd.Timedelta(days=WEEK_DAYS):
            events.append(ev)
    return events, remaining, None


def parse_event_odds(payload):
    """One event-odds payload -> prop rows (the props archive's columns), provider 'OddsAPI <book>'."""
    from data.odds_sources import devig_two_way_decimal
    rows = []
    for book in (payload or {}).get('bookmakers') or []:
        title = book.get('title') or book.get('key')
        for market in book.get('markets') or []:
            stat = MARKETS.get(market.get('key'))
            if not stat:
                continue
            pairs = {}
            for o in market.get('outcomes') or []:
                side = str(o.get('name') or '').lower()
                if side in ('over', 'under') and o.get('description') and o.get('point') is not None:
                    pairs.setdefault((o['description'], float(o['point'])), {})[side] = float(o['price'])
            for (player, line), sides in pairs.items():
                if set(sides) != {'over', 'under'}:
                    continue
                rows.append({'provider': f'OddsAPI {title}', 'player': player, 'team': '', 'position': '', 'market': stat,
                             'market_raw': market.get('key'), 'scorable': True, 'line': line,
                             'over_payout': sides['over'], 'under_payout': sides['under'],
                             'p_over': devig_two_way_decimal(sides['over'], sides['under']), 'period': 'game',
                             'source_id': str(payload.get('id') or '')})
    return rows


def pull_and_archive(api_key=None, reserve=DEFAULT_RESERVE, now=None, session=requests, archive_dir=ARCHIVE_DIR, force=False):
    """One conservative pull. Returns a short text saying what happened (and never raises)."""
    try:
        now = now or _now()
        if api_key is None:
            from data.loaders import load_saved_odds_api_key
            api_key = load_saved_odds_api_key()
        if not api_key:
            return 'skipped: no Odds API key saved (Live Odds tab)'
        last = _last_pull()
        if not force and last is not None and now - last < datetime.timedelta(hours=MIN_HOURS_BETWEEN):
            return f'skipped: last pull {(now - last).total_seconds() / 3600:.1f}h ago (minimum {MIN_HOURS_BETWEEN}h)'
        events, remaining, err = week_events(api_key, now, session)
        if err:
            return f'skipped: events call failed ({err})'
        cost = estimate_cost(len(events))
        if not events:
            return 'skipped: no games in the next week'
        if remaining is None or remaining - cost < reserve:
            return f'skipped: {cost} credits needed, {remaining} left, reserve {reserve}'
        rows, spent_left = [], remaining
        for ev in events:
            resp = session.get(f"{BASE}/events/{ev['id']}/odds", timeout=15,
                               params={'apiKey': api_key, 'regions': REGIONS, 'markets': ','.join(MARKETS),
                                       'oddsFormat': 'decimal'})
            left = resp.headers.get('x-requests-remaining')
            spent_left = int(float(left)) if left not in (None, '') else spent_left
            if resp.status_code == 200:
                rows += parse_event_odds(resp.json())
        _save_state(last_pull=now.isoformat(), credits_left=spent_left, estimated_cost=cost, games=len(events))
        if not rows:
            return f'pulled {len(events)} games, no prop rows; credits left {spent_left}'
        os.makedirs(archive_dir, exist_ok=True)
        path = os.path.join(archive_dir, f"oddsapi_{now.strftime('%Y%m%dT%H%M%SZ')}.json")
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump({'fetched_at': now.isoformat(), 'credits_left': spent_left, 'rows': rows}, fh)
        os.replace(tmp, path)
        return f'pulled {len(events)} games, {len(rows)} lines, ~{cost} credits; credits left {spent_left}'
    except Exception as exc:
        return f'failed: {type(exc).__name__}: {exc}'


def load_odds_api_archive(archive_dir=ARCHIVE_DIR):
    """Every archived Odds API row with its fetched_at (UTC)."""
    frames = []
    for path in sorted(glob.glob(os.path.join(archive_dir, 'oddsapi_*.json'))):
        try:
            with open(path, encoding='utf-8') as fh:
                snap = json.load(fh)
        except (OSError, ValueError):
            continue
        rows = pd.DataFrame(snap.get('rows') or [])
        if not rows.empty:
            rows['fetched_at'] = pd.Timestamp(snap['fetched_at'])
            frames.append(rows)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
