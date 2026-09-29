"""
Fit the constants tables `v2_xtd` (the expected-TD model,
docs/model_improvement_plan_2026-09-23.md item 6) reads at build time:

  1. LEAGUE TD RATE per opportunity, by (play type, position, zone) - how
     often a rush/target from this zone actually becomes a touchdown,
     measured on 2016-2021 play-by-play. This is the rate xTD multiplies a
     player's own projected zone opportunities by.
  2. TEAM ZONE-VOLUME ELASTICITY, by (play type, zone) - how much a team's
     OWN zone opportunity count (not who gets them - that's the player
     share, fit separately at build time from as-of-week usage) moves with
     its market-implied point total, the same "team runs more red-zone
     plays when it's expected to score more" relationship
     GAME_TOTAL_ELASTICITY_BY_STAT already captures for whole-game volume,
     fit here specifically for the zone counts data.transforms.
     build_redzone_usage's REDZONE_RUSH_ZONES/REDZONE_TARGET_ZONES define.

METHOD.
  (1) is a plain ratio: sum(zone TDs) / sum(zone opportunities), pooled by
  (type, position, zone) across the fit years - no regression needed, a
  TD-per-opportunity rate IS the quantity being measured directly.

  (2) is fit on BUCKETED team-week data, not a per-team-week log-log OLS:
  a team's own zone-opportunity COUNT per game is a small integer (often 0),
  so an individual team-week's ratio-to-own-season-mean is either undefined
  (0/0) or a noisy multiple of a tiny denominator - the same problem
  scripts/fit_game_total_elasticity_perstat.py's own "binned WLS
  cross-check" exists to route around for the same reason. Team-weeks are
  bucketed into quintiles of that week's implied-total ratio (to the
  league average that week), and the fit is one OLS line through the FIVE
  bucket means: log(bucket mean zone count / team's own season mean) =
  beta * log(bucket mean implied ratio). Team's own season mean is
  computed on the SAME fit-year data (not held out) since this is a
  population constant, not a per-team backtest target.

Usage:
    python scripts/fit_xtd_rates.py
    python scripts/fit_xtd_rates.py --years 2016-2021
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.loaders import load_pbp  # noqa: E402
from data.transforms import REDZONE_RUSH_ZONES, REDZONE_TARGET_ZONES, load_and_merge_data  # noqa: E402
from data.weekly_projections import game_environment, load_schedule  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(ROOT, 'data', 'xtd_rates.json')
POSITIONS = ('QB', 'RB', 'WR', 'TE')
N_BUCKETS = 5


def _zone_mask(pbp, lo, hi):
    hi_val = pbp['yardline_100'].max() if hi is None else hi
    return (pbp['yardline_100'] > lo) & (pbp['yardline_100'] <= hi_val)


def _position_by_gsis(year):
    """{gsis_id: most-common position that season} - this nflfastR pbp
    export carries no rusher_position/receiver_position column at all
    (confirmed empty on a real 2019 pull), so position has to come from
    the app's own weekly stats table instead. `player_id` there is the
    SAME gsis id format as pbp's rusher_player_id/receiver_player_id -
    data.transforms.build_redzone_usage's own docstring documents this
    exact id-format match and reuses it the same way."""
    stats_df, _, _, _ = load_and_merge_data(year, 'Full PPR')
    if stats_df.empty or 'player_id' not in stats_df.columns or 'position' not in stats_df.columns:
        return {}
    sub = stats_df[['player_id', 'position']].dropna()
    sub = sub[sub['player_id'].astype(str) != '']
    if sub.empty:
        return {}
    mode = sub.groupby('player_id', observed=True)['position'].agg(
        lambda s: s.astype(str).str.upper().value_counts().idxmax())
    return mode.to_dict()


def collect_td_rates(years):
    """{(type, position, zone): {'td': n, 'opp': n}} pooled across years."""
    totals = {}

    def add(play_type, position, zone, td, opp):
        key = (play_type, position, zone)
        entry = totals.setdefault(key, {'td': 0, 'opp': 0})
        entry['td'] += int(td)
        entry['opp'] += int(opp)

    for year in years:
        pbp = load_pbp(year)
        if pbp.empty:
            print(f"{year}: no play-by-play, skipped")
            continue
        needed = {'yardline_100', 'play_type', 'rusher_player_id', 'receiver_player_id',
                  'rush_touchdown', 'pass_touchdown', 'season_type'}
        if not needed.issubset(pbp.columns):
            print(f"{year}: missing pbp columns, skipped")
            continue
        pbp = pbp[pbp['season_type'] == 'REG']

        position_by_gsis = _position_by_gsis(year)
        if not position_by_gsis:
            print(f"{year}: no position lookup (weekly stats unavailable), skipped")
            continue

        rushes = pbp[(pbp['play_type'] == 'run') & pbp['rusher_player_id'].notna()].copy()
        rushes['_pos'] = rushes['rusher_player_id'].map(position_by_gsis).fillna('UNK')
        for zone_name, lo, hi in REDZONE_RUSH_ZONES:
            zone_rows = rushes[_zone_mask(rushes, lo, hi)]
            for pos, group in zone_rows.groupby('_pos', observed=True):
                add('rush', pos, zone_name, group['rush_touchdown'].sum(), len(group))

        targets = pbp[(pbp['play_type'] == 'pass') & pbp['receiver_player_id'].notna()].copy()
        targets['_pos'] = targets['receiver_player_id'].map(position_by_gsis).fillna('UNK')
        for zone_name, lo, hi in REDZONE_TARGET_ZONES:
            zone_rows = targets[_zone_mask(targets, lo, hi)]
            for pos, group in zone_rows.groupby('_pos', observed=True):
                add('target', pos, zone_name, group['pass_touchdown'].sum(), len(group))

    rates = {}
    for (play_type, pos, zone), entry in totals.items():
        if pos not in POSITIONS or entry['opp'] < 30:
            continue  # too thin a sample to trust a position/zone TD rate
        rates.setdefault(play_type, {}).setdefault(pos, {})[zone] = {
            'rate': round(entry['td'] / entry['opp'], 5), 'n_opportunities': entry['opp'],
        }
    return rates


def _team_zone_counts_by_week(pbp, play_type, zone_name, lo, hi):
    id_col = 'rusher_player_id' if play_type == 'rush' else 'receiver_player_id'
    team_col = 'posteam'
    rows = pbp[(pbp['play_type'] == ('run' if play_type == 'rush' else 'pass')) & pbp[id_col].notna()]
    zone_rows = rows[_zone_mask(rows, lo, hi)]
    return zone_rows.groupby([team_col, 'week'], observed=True)['play_id'].count()


def collect_team_zone_elasticity(years):
    """{(type, zone): beta} from bucketed team-week zone counts vs.
    implied-total ratio - see module docstring."""
    panel = []
    for year in years:
        pbp = load_pbp(year)
        if pbp.empty or not {'posteam', 'week', 'play_type', 'yardline_100', 'play_id'}.issubset(pbp.columns):
            continue
        pbp = pbp[pbp.get('season_type', 'REG') == 'REG'] if 'season_type' in pbp.columns else pbp
        schedule_df = load_schedule(year)
        weeks = sorted(pd.to_numeric(pbp['week'], errors='coerce').dropna().unique())
        implied_ratio = {}
        for wk in weeks:
            env = game_environment(schedule_df, int(wk))
            vals = [e['implied'] for e in env.values() if e.get('implied')]
            if not vals:
                continue
            league = float(np.mean(vals))
            if league <= 0:
                continue
            for team, e in env.items():
                imp = e.get('implied')
                if imp and imp > 0:
                    implied_ratio[(str(team).strip().upper(), float(wk))] = imp / league

        for play_type, zones in (('rush', REDZONE_RUSH_ZONES), ('target', REDZONE_TARGET_ZONES)):
            for zone_name, lo, hi in zones:
                counts = _team_zone_counts_by_week(pbp, play_type, zone_name, lo, hi)
                for (team, wk), n in counts.items():
                    ratio = implied_ratio.get((str(team).strip().upper(), float(wk)))
                    if ratio is None:
                        continue
                    panel.append({'type': play_type, 'zone': zone_name, 'team': team,
                                  'implied_ratio': ratio, 'count': int(n)})
    if not panel:
        return {}
    df = pd.DataFrame(panel)

    betas = {}
    for (play_type, zone), group in df.groupby(['type', 'zone'], observed=True):
        team_mean = group.groupby('team', observed=True)['count'].transform('mean').replace(0, np.nan)
        rel = (group['count'] / team_mean).dropna()
        g = group.loc[rel.index].copy()
        g['rel_count'] = rel
        try:
            g['_bucket'] = pd.qcut(g['implied_ratio'], N_BUCKETS, labels=False, duplicates='drop')
        except ValueError:
            continue
        bucketed = g.groupby('_bucket', observed=True).agg(
            implied_ratio=('implied_ratio', 'mean'), rel_count=('rel_count', 'mean')).dropna()
        bucketed = bucketed[(bucketed['implied_ratio'] > 0) & (bucketed['rel_count'] > 0)]
        if len(bucketed) < 3:
            continue
        x = np.log(bucketed['implied_ratio'].to_numpy())
        y = np.log(bucketed['rel_count'].to_numpy())
        beta = float(np.sum(x * y) / np.sum(x * x)) if np.sum(x * x) > 0 else 0.0
        betas.setdefault(play_type, {})[zone] = round(beta, 4)
    return betas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2016-2021')
    args = ap.parse_args()
    if '-' in args.years:
        lo, hi = args.years.split('-')
        years = list(range(int(lo), int(hi) + 1))
    else:
        years = [int(y) for y in args.years.split(',')]

    print(f"fit window: {years[0]}-{years[-1]}\n")
    print("== league TD rate per opportunity ==")
    td_rates = collect_td_rates(years)
    for play_type, positions in td_rates.items():
        for pos, zones in positions.items():
            for zone, entry in zones.items():
                print(f"{play_type:<7} {pos:<3} {zone:<5} rate={entry['rate']:.4f}  n={entry['n_opportunities']}")

    print("\n== team zone-volume elasticity (vs. implied team total) ==")
    elasticity = collect_team_zone_elasticity(years)
    for play_type, zones in elasticity.items():
        for zone, beta in zones.items():
            print(f"{play_type:<7} {zone:<5} beta={beta:.4f}")

    payload = {
        'fit_window': f"{years[0]}-{years[-1]}",
        'fitted_at': datetime.now(timezone.utc).isoformat(),
        'td_rate': td_rates,
        'zone_elasticity': elasticity,
    }
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"\nwrote {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
