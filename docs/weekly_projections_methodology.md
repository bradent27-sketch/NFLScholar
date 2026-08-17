# Weekly Projections — methodology

`data/weekly_projections.py`. Full derivation reasoning lives in that
module's own docstrings (and each helper function's docstring) — this doc
is the orientation plus the validation results, same split
`docs/draft_hq_methodology.md` uses for the draft engine (that file for
"how is this number actually made", this one for the same on the weekly
side, plus the backtest).

## What this is, and how it's different from what already existed

Player Search already has a next-game projection
(`data.transforms.build_player_projection`) for one player at a time —
recent-form/season blend × opponent-allowed × pace × alignment. This module
projects the **whole** skill-position pool for one week at once, for the
Weekly Rankings tab, and adds inputs that single-player model doesn't have:
prior-season blending (weighted down as the current season's own sample
grows), snap-share/route-share "role confidence" that changes how fast a
player's own small sample gets trusted, and a game-script read against the
Vegas-implied spread for the target week.

It is built by **composition**, not from scratch — every external signal
reuses a primitive already in production elsewhere in this app:

| Signal | Reused from |
|---|---|
| Opponent-allowed rates | `data.transforms.build_stat_allowed_matrix` (same one `build_player_projection` uses) |
| Pace | `data.loaders.load_team_pace` |
| Game script | Same bucket edges as `data.matchup_signals.game_script_sensitivity_curve`, vectorized across the whole pool instead of one player at a time |
| Injuries | `data.draft_sources.fetch_injury_report` |
| Scoring | `data.transforms.score_projected_stats` (same function `build_player_projection` scores with) |

## The one shrinkage mechanism

Every per-game rate (targets, carries, yards, TDs, ...) goes through the
same weighted blend:

```
in_season_rate = 0.6 * trailing-4-game average + 0.4 * season-to-date average
w_current       = games_this_season / (games_this_season + K[stat])
blended_rate    = w_current * in_season_rate + (1 - w_current) * prior
```

`prior` is the player's own prior-season rate when he has one, else the
CURRENT season's games-weighted position average (a rookie or new-role
player lands on the position baseline, not on nothing). `K` is bigger for
touchdowns/interceptions than for volume stats, so lumpy scoring stats get
pulled toward the prior harder on a small sample. Role confidence (recent
snap share + PFF season route rate) scales `K` down for a confirmed
every-down role and up for a thin one, per `K_EFFECTIVE_RANGE`.

This is the literal mechanism behind "the current season should outweigh
the past as the sample grows": `w_current` climbs from 0 toward 1
automatically as `games_this_season` increases — there's no separate
schedule to hand-tune, one formula produces the behavior for every stat.

## Why Vegas is used the way it is, not the way it failed before

`data/odds_market.py` already ran a real backtest (748 player-seasons,
2023–2025) on scaling a player's SEASON projection by his team's Vegas
implied points, and it made the projection worse at every strength tested —
see that module's own docstring. This module deliberately does **not**
repeat that: no flat "multiply by implied team total" anywhere.

What it does instead: for each player, bucket his own real games by that
game's actual final margin (same 4 buckets `game_script_sensitivity_curve`
uses — Trailed big / Lost close / Won close / Won big), then read his
projection off that curve at the **market's implied margin** for the
target week (`implied_points − implied_allowed` from
`data.odds_market.implied_team_points`), the same "interpolate this
player's own measured curve at the target value" pattern
`data.matchup_signals.efficiency_elasticity_curve` already uses for
opponent softness. This is a personalized read of how THIS player's role
has actually shifted with game state, not a league-wide scale-by-offense-
quality — a materially different technique, not the same one applied
again. Capped at ±15%, applied only to volume stats (targets, receptions,
receiving yards, rushing attempts, rushing yards) — never touchdowns (too
sparse per player-game to bucket reliably) and never passing volume (a
QB's own dropbacks are far stickier to game state at the team level than
an individual skill player's role is).

## A real bug found building this: the injury discount contaminated the backtest

`fetch_injury_report(year)` always returns each player's **most recent**
designation — correct for a live, current-week projection, but wrong for
validating a PAST week months or years later, since by then it's reporting
that player's last designation of the entire season (often 'Out' from
season-ending IR, or an unrelated 'Questionable' from some other week),
applied uniformly regardless of which week is actually being tested.

Measured: this alone was discounting or zeroing out roughly **1,000 of the
~2,000** skill-position player-weeks in the 2025 backtest below, and was
the dominant source of what first looked like a broad, systematic
under-projection across every position (bias around −2 points/player
before this was isolated — see the git history of this file's own drafting
process for the full before/after chase, not reproduced here). Fixed by
adding `apply_injury=False` to `build_weekly_projections` for backtesting
only — the live app (and the Weekly Rankings tab) always calls with
`apply_injury=True`, where "most recent designation" is exactly right.

## Backtest — one honest pass, not an iterated one

`scripts/validate_weekly_projections.py`. Every week in 2025 weeks 5–17,
using `as_of_week=<that week>` so no result the model is trying to predict
ever leaks into its own inputs (see `build_weekly_projections`'s own
docstring on this). Compared against a naive baseline (each player's own
trailing-4-game actual-points average) over the **same player pool** the
model produced that week — not the whole league (that inflates a naive
baseline's apparent accuracy by padding it with hundreds of near-zero
bench/DST/K rows that are trivially easy to predict; see the script's own
comment on this for the numbers before the fix).

The model's constants (`STAT_K`, the clip ranges) were set from the
reasoning above, run through this backtest ONCE, and left alone — tuning
them against this exact script's own output would be fitting to the test
set, not validating against it.

```
                    n      MAE      bias     rank_corr
Model (all pos)   4093    4.651    +0.280      0.655
Naive baseline    4093    4.582    −0.080      0.663

Model by position:
  QB   n=444   MAE=7.015   bias=+0.705   rank_corr=0.403
  RB   n=1067  MAE=4.663   bias=+0.209   rank_corr=0.706
  WR   n=1702  MAE=4.497   bias=+0.448   rank_corr=0.623
  TE   n=880   MAE=3.743   bias=−0.174   rank_corr=0.601
```

**Read honestly**: the model is essentially at parity with a simple
trailing-average baseline on this one-season backtest (MAE within 1.5%,
rank correlation within 1.2%), not a decisive improvement on either metric.
It is not overclaimed as one. What it adds over the naive baseline isn't
visible in these two numbers: a real prior-season floor for players with a
thin current-season sample (a naive trailing average has nothing to fall
back on for a player's first few games — this model does), explicit
matchup/pace/game-script context a bare average has none of, and a
touchdown-rate that's shrunk rather than taken at face value (naive treats
a fluky 3-TD game exactly like a normal one going forward). QB is the
weakest position (rank_corr 0.403) — a genuinely harder position to
project week-to-week because a single QB's role rarely shifts, so most of
the swing in his score is pure game-to-game variance in a small number of
big, discrete plays (a long TD run, a garbage-time INT) that no usage-rate
model captures.

## Known limitations

- **Week 1 can't be projected at all** — the player pool itself is read
  off the current season's own games. See `build_weekly_projections`'s
  docstring. FantasyPros' own weekly projection
  (`data.draft_sources.fetch_fantasypros_weekly_projections`, wired into
  the Weekly Rankings tab) is the right tool for the season opener instead.
- **Pace uses the full season's team stats, not an as-of-week-filtered
  cut** — `data.loaders.load_team_pace` isn't parameterized for a cutoff.
  A minor, accepted leak for the backtest (pace is a slow-moving signal
  relative to opponent-allowed rates and game script); not a leak at all
  for live use, where "the season so far" and "as of today" are the same
  thing.
- **Injury status has no historical week granularity** — see above. Live
  use is correct; backtesting needs `apply_injury=False`.
- **Vegas lines aren't posted for every future week** — `data/odds_market.py`
  only carries lines as far out as books have posted them. A missing line
  for the target week just means the game-script read sits out for that
  player (multiplier 1.0), same degrade-gracefully convention as every
  other best-effort signal in this app.
