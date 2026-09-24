"""
Measure how far a sportsbook yardage LINE (a median) sits below the true
weekly MEAN, per (position, stat), bucketed by how big the line is.

WHY THIS EXISTS. data/market_devig.py::implied_mean_from_line treats yardage
as a symmetric Normal, so an evenly-priced line is returned unchanged as the
"implied mean." Real weekly yardage is right-skewed (a few huge games pull
the mean above the median), and that skew shrinks as a player's usual volume
grows - a workhorse's yardage is more consistent than a committee back's, so
his mean/median ratio is closer to 1. A single flat multiplier per position
(the old MEDIAN_TO_MEAN behaviour) misses that: a bell-cow at 90 yds/gm and a
change-of-pace back at 15 yds/gm get the same correction, when the low-volume
player needs a much bigger one.

METHOD. Pool every player-season (>=8 games in that season, that position)
from stats_player_week_{2019..2025}.csv. For each player-season compute his
own weekly mean and weekly median. Bucket player-seasons into terciles of
their own median (separately per position/stat, since a QB's median passing
yards and a TE's median receiving yards live on totally different scales -
hardcoding one set of yard cutoffs for both would put every QB in the top
bucket and every backup TE in the bottom one). Within each bucket, pool
(games-weighted) mean and median across players and report ratio = mean/median.

This produces the bucketed table `data/market_devig.py::YARD_MEDIAN_TO_MEAN`
consumes; rerun this script and paste the new dict in whenever a fresh season
of data should update the fit (the fit window is printed at the top of the
output so a stale run is obvious).

Usage:
    python scripts/fit_yard_skew.py
    python scripts/fit_yard_skew.py --years 2021-2025 --min-games 10
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (position, stat-in-this-file) -> stat name build_weekly_projections uses.
# Only stats fed through the Normal-symmetric path in market_devig's
# _YARD_STATS are worth fitting here.
TARGETS = [
    ('QB', 'passing_yards'),
    ('QB', 'rushing_yards'),
    ('RB', 'rushing_yards'),
    ('RB', 'receiving_yards'),
    ('WR', 'receiving_yards'),
    ('TE', 'receiving_yards'),
]

N_BUCKETS = 3
BUCKET_LABELS = ['low', 'mid', 'high']


def _load_years(years):
    frames = []
    for y in years:
        path = os.path.join(ROOT, f'stats_player_week_{y}.csv')
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path, low_memory=False)
        d = d[d['season_type'] == 'REG']
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fit(years, min_games):
    df = _load_years(years)
    if df.empty:
        raise SystemExit(f"no stats_player_week_*.csv found for {years} under {ROOT}")

    table = {}
    rows_out = []
    for pos, stat in TARGETS:
        sub = df[(df['position'] == pos) & df[stat].notna()]
        if sub.empty:
            continue
        g = sub.groupby(['player_id', 'season'])[stat]
        n_games = g.transform('size')
        seasons = sub.assign(_n=n_games)
        seasons = seasons[seasons['_n'] >= min_games]
        if seasons.empty:
            continue
        per_player = seasons.groupby(['player_id', 'season']).agg(
            mean=(stat, 'mean'), median=(stat, 'median'), games=(stat, 'size'))
        per_player = per_player[per_player['median'] > 0]  # a 0 median has no ratio
        if len(per_player) < 15:
            continue

        # Terciles of each player-season's OWN median - separately per
        # (pos, stat), so the buckets track that stat's real scale.
        try:
            per_player['_bucket'] = pd.qcut(per_player['median'], N_BUCKETS,
                                            labels=BUCKET_LABELS, duplicates='drop')
        except ValueError:
            per_player['_bucket'] = BUCKET_LABELS[len(BUCKET_LABELS) // 2]

        buckets = []
        for label in BUCKET_LABELS:
            b = per_player[per_player['_bucket'] == label]
            if b.empty:
                continue
            w = b['games'].to_numpy(dtype=float)
            pooled_mean = float(np.average(b['mean'].to_numpy(), weights=w))
            pooled_median = float(np.average(b['median'].to_numpy(), weights=w))
            ratio = pooled_mean / pooled_median if pooled_median > 0 else 1.0
            upper = float(b['median'].max())
            buckets.append((upper, round(ratio, 3)))
            rows_out.append(dict(pos=pos, stat=stat, bucket=label, n_player_seasons=len(b),
                                 median_line_upto=round(upper, 1),
                                 pooled_mean=round(pooled_mean, 2), pooled_median=round(pooled_median, 2),
                                 ratio=round(ratio, 3)))
        buckets.sort(key=lambda t: t[0])
        if buckets:
            buckets[-1] = (float('inf'), buckets[-1][1])  # last bucket covers everything above
            table[(pos, stat)] = buckets

    return table, pd.DataFrame(rows_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2019-2025')
    ap.add_argument('--min-games', type=int, default=8)
    args = ap.parse_args()
    if '-' in args.years:
        lo, hi = args.years.split('-')
        years = list(range(int(lo), int(hi) + 1))
    else:
        years = [int(y) for y in args.years.split(',')]

    table, report = fit(years, args.min_games)
    print(f"fit window: {years[0]}-{years[-1]}, min {args.min_games} games/player-season\n")
    print(report.to_string(index=False))
    print("\nYARD_MEDIAN_TO_MEAN = {")
    for (pos, stat), buckets in table.items():
        parts = []
        for u, r in buckets:
            bound = "float('inf')" if u == float('inf') else str(round(u, 1))
            parts.append(f"({bound}, {r})")
        print(f"    ('{pos}', '{stat}'): [{', '.join(parts)}],")
    print("}")


if __name__ == '__main__':
    main()
