"""
Fit league-wide "how does a player's own volume move with game script" curves
- docs/model_improvement_plan_2026-09-23.md item 5, Step B.

WHY THIS EXISTS. `_vectorized_game_script_multiplier` (the existing,
already-shipped game-script correction) reads a PLAYER'S OWN history bucketed
by margin - real signal once he has a real sample, but noisy with only a
handful of games and completely inert for `passing_attempts` until
`v2_offense_script_pool_blend` extended it. This script instead fits ONE
LEAGUE-WIDE curve per (position, stat): how much a player's volume moves,
relative to his own season average, as a function of (a) how much his team
was actually leading/trailing that game (REALIZED script, from play-by-play)
and (b) how much his team was expected to lead/trail going in (pregame
spread). `data.weekly_projections`'s `v2_script_neutral_volume` uses (a) to
divide a PAST game's raw volume back to what it would have looked like in a
neutral script BEFORE averaging it into a player's rate, and (b) to scale the
resulting rate back up for the UPCOMING week's own expected script - so a
thin-sample player gets a real, evidence-based correction from week 1, the
same population-based idea `v2_offense_script_pool_blend` used, but built
from the actual mechanism (game state) instead of a pooled bucket-average.

METHOD.
  REALIZED script per team-game: data.loaders.realized_script_by_team_week's
  mean score_differential over that team's own run/pass plays, capped +/-21
  (nflverse PBP; not available before ~2000, plenty of coverage for the
  2016-2021 fit window). Falls back to the game's final schedule margin for
  any (team, week) missing from the PBP-derived table (a bye week, or a
  season PBP fetch that failed) - a noisier version of the same quantity,
  not a different one.

  For each (position, stat) in TARGETS: pool every player-season with >= 8
  games that season, that position. For each of his games, ratio =
  stat_value / his own season mean for that stat (excluding players whose
  season mean is 0 - nothing to be a ratio of). Fit, by ordinary least
  squares with NO intercept (script/spread = 0 must mean ratio = 1 by
  construction, log(1) = 0):

    f_real: log(ratio) = beta_lead * max(realized_script, 0)
                        + beta_trail * min(realized_script, 0)
    (piecewise-linear with a knot at 0 - leading and trailing are allowed
    different slopes, since a big lead means fewer, run-heavy plays while a
    big deficit means MORE, pass-heavy plays; these are not the same
    magnitude of effect.)

    f_exp: log(ratio) = beta_exp * (-own_pregame_spread)
    (a single slope against the PREGAME spread, sign-flipped so a positive
    x means "expected to need the ball more" - i.e. an underdog - the same
    direction a negative realized script already points. Not piecewise:
    a spread rarely reaches the extremes a real in-game blowout does, so a
    single slope is both simpler and the sample-appropriate choice; this is
    "f_exp is flatter than f_real" made structural, not just an empirical
    finding.)

  `log(ratio)` is clipped to +/-log(5) (ratio in [0.2, 5]) before fitting
  only - a handful of monster/token games would otherwise dominate an
  unweighted least-squares fit out of proportion to what they say about the
  league-wide relationship. The saved coefficients are unaffected by this
  clip in the normal case; it only matters for the small tail of games it
  touches.

Usage:
    python scripts/fit_script_curves.py
    python scripts/fit_script_curves.py --years 2016-2021 --min-games 8
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.transforms import load_and_merge_data  # noqa: E402
from data.loaders import load_schedule, realized_script_by_team_week  # noqa: E402
from data.weekly_projections import _team_week_margins  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(ROOT, 'data', 'script_curves.json')

# (position, stat) pairs to fit - the exact set the plan specifies. Every
# other stat this app projects derives from one of these at USE time (see
# 'v2_script_neutral_volume' in data/weekly_projections.py), not from its
# own fitted curve.
TARGETS = [
    ('QB', 'passing_attempts'),
    ('QB', 'passing_yards'),
    ('RB', 'rushing_attempts'),
    ('RB', 'targets'),
    ('WR', 'targets'),
    ('TE', 'targets'),
]

LOG_RATIO_CLIP = np.log(5.0)  # ratio in [0.2, 5] - fitting robustness only


def _own_spread_by_team_week(schedule_df):
    """{(team, week): own pregame spread, positive = favored} - same sign
    convention data.weekly_projections._historical_target_margins uses
    (home spread_line is the home team's own margin; the away team's is its
    negative)."""
    if schedule_df is None or schedule_df.empty:
        return {}
    needed = {'week', 'home_team', 'away_team', 'spread_line'}
    if not needed.issubset(schedule_df.columns):
        return {}
    out = {}
    weeks = pd.to_numeric(schedule_df['week'], errors='coerce')
    spread = pd.to_numeric(schedule_df['spread_line'], errors='coerce')
    for home, away, wk, sp in zip(schedule_df['home_team'], schedule_df['away_team'], weeks, spread):
        if pd.isna(wk) or pd.isna(sp):
            continue
        out[(str(home).strip().upper(), float(wk))] = float(sp)
        out[(str(away).strip().upper(), float(wk))] = float(-sp)
    return out


def _final_margin_by_team_week(schedule_df):
    margins = _team_week_margins(schedule_df)
    if margins.empty:
        return {}
    out = {}
    for team, week, margin in zip(margins['Team'], margins['week'], margins['margin']):
        if pd.isna(week) or pd.isna(margin):
            continue
        out[(str(team).strip().upper(), float(week))] = float(margin)
    return out


def collect(years, min_games):
    """One row per (position, stat, player-game) with `ratio`,
    `realized_script`, `own_spread` - pooled across every fit year."""
    rows = []
    for year in years:
        stats_df, team_col, name_col, _ = load_and_merge_data(year, 'Full PPR')
        if stats_df.empty or 'week' not in stats_df.columns:
            print(f"{year}: no weekly data, skipped")
            continue
        stats_df = stats_df[stats_df.get('season_type', 'REG').astype(str).str.upper().eq('REG')].copy()
        schedule_df = load_schedule(year)
        realized = realized_script_by_team_week(year)
        final_margin = _final_margin_by_team_week(schedule_df)
        own_spread = _own_spread_by_team_week(schedule_df)

        stats_df['_team'] = stats_df[team_col].astype(str).str.strip().str.upper()
        stats_df['_week'] = pd.to_numeric(stats_df['week'], errors='coerce')
        stats_df = stats_df.dropna(subset=['_week'])
        keys = list(zip(stats_df['_team'], stats_df['_week']))
        stats_df['_realized_script'] = [realized.get(k, final_margin.get(k, np.nan)) for k in keys]
        stats_df['_own_spread'] = [own_spread.get(k, np.nan) for k in keys]

        for pos, stat in TARGETS:
            if stat not in stats_df.columns:
                continue
            pos_rows = stats_df[stats_df['position'].astype(str).str.upper().eq(pos)].copy()
            if pos_rows.empty:
                continue
            pos_rows[stat] = pd.to_numeric(pos_rows[stat], errors='coerce').fillna(0.0)
            season_games = pos_rows.groupby(name_col, observed=True)['_week'].transform('nunique')
            season_mean = pos_rows.groupby(name_col, observed=True)[stat].transform('mean')
            eligible = (season_games >= min_games) & (season_mean > 0) \
                & pos_rows['_realized_script'].notna() & pos_rows['_own_spread'].notna()
            elig_rows = pos_rows[eligible]
            if elig_rows.empty:
                continue
            ratio = elig_rows[stat] / season_mean[eligible]
            rows.append(pd.DataFrame({
                'year': year, 'position': pos, 'stat': stat,
                'ratio': ratio.to_numpy(),
                'realized_script': elig_rows['_realized_script'].to_numpy(dtype=float),
                'own_spread': elig_rows['_own_spread'].to_numpy(dtype=float),
            }))
    if not rows:
        return pd.DataFrame(columns=['year', 'position', 'stat', 'ratio', 'realized_script', 'own_spread'])
    return pd.concat(rows, ignore_index=True)


def fit(pooled):
    """{position: {stat: {beta_lead, beta_trail, beta_exp, n}}}."""
    curves = {}
    report = []
    for (pos, stat), group in pooled.groupby(['position', 'stat'], observed=True):
        # A player-game with 0 of a stat he otherwise gets some of (hurt
        # early, benched, a QB split) gives ratio=0 -> log(0)=-inf, a real
        # value the clip below handles correctly - just silence the
        # (harmless) RuntimeWarning numpy raises getting there.
        with np.errstate(divide='ignore'):
            log_ratio = np.log(group['ratio'].to_numpy(dtype=float))
        log_ratio = np.clip(log_ratio, -LOG_RATIO_CLIP, LOG_RATIO_CLIP)
        script = group['realized_script'].to_numpy(dtype=float)
        spread = group['own_spread'].to_numpy(dtype=float)

        x_real = np.column_stack([np.clip(script, 0, None), np.clip(script, None, 0)])
        beta_real, *_ = np.linalg.lstsq(x_real, log_ratio, rcond=None)
        beta_lead, beta_trail = (float(beta_real[0]), float(beta_real[1]))

        x_exp = (-spread).reshape(-1, 1)
        beta_exp = float(np.sum(x_exp[:, 0] * log_ratio) / np.sum(x_exp[:, 0] ** 2)) if np.sum(x_exp[:, 0] ** 2) > 0 else 0.0

        n = len(group)
        curves.setdefault(pos, {})[stat] = {
            'beta_lead': round(beta_lead, 5), 'beta_trail': round(beta_trail, 5),
            'beta_exp': round(beta_exp, 5), 'n': int(n),
        }
        report.append((pos, stat, n, beta_lead, beta_trail, beta_exp))
    return curves, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2016-2021')
    ap.add_argument('--min-games', type=int, default=8)
    args = ap.parse_args()
    if '-' in args.years:
        lo, hi = args.years.split('-')
        years = list(range(int(lo), int(hi) + 1))
    else:
        years = [int(y) for y in args.years.split(',')]

    pooled = collect(years, args.min_games)
    if pooled.empty:
        raise SystemExit("no eligible player-games found")
    curves, report = fit(pooled)

    print(f"fit window: {years[0]}-{years[-1]}, min {args.min_games} games/player-season\n")
    print(f"{'pos':<4} {'stat':<20} {'n':>7} {'beta_lead':>10} {'beta_trail':>11} {'beta_exp':>9}")
    for pos, stat, n, beta_lead, beta_trail, beta_exp in report:
        print(f"{pos:<4} {stat:<20} {n:>7} {beta_lead:>10.4f} {beta_trail:>11.4f} {beta_exp:>9.4f}")

    payload = {
        'fit_window': f"{years[0]}-{years[-1]}",
        'min_games': args.min_games,
        'fitted_at': datetime.now(timezone.utc).isoformat(),
        'curves': curves,
    }
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"\nwrote {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
