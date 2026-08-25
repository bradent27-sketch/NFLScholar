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

## August 2026 pass — the components, and what each one measured

Every change in this pass is a named, individually switchable component
(`data.weekly_projections.MODEL_FEATURES`) evaluated with
`scripts/eval_weekly_model.py` on **2024 and 2025, weeks 5-17, 8,107 paired
player-weeks**. "Paired" is load-bearing: per week, every variant is scored
on the intersection of the player pools all variants produced, so a
component that merely drops a few hard-to-project players can't look like an
improvement. Injuries are off in every run (`apply_injury=False`) for the
reason in the section above.

Three of the six shipped. The other three are still in the code, still
switchable, and are documented here with the numbers that kept them off —
same discipline `docs/draft_hq_methodology.md` applies to the three changes
it built, measured and rejected.

### Defense-profile reliability correction — not yet an accuracy claim

The original quality-adjusted defense estimator divided each individual
player-game by that player's own season average, then averaged those ratios
by defense. That is vulnerable to a low-volume replacement: a 75-yard relief
appearance against a 35-yard personal baseline looks like a 2.1× defensive
failure even if the offense's full QB passing game was ordinary. A live
Houston audit exposed this exact problem, but the flaw was not Houston-only.

Weekly Rankings now aggregates every projected position's rows into one
offense-versus-defense game before estimating the defense. It compares each
position-team total to that offense's own positional baseline and computes a
recency-weighted **pooled observed / expected** factor, with four neutral
league-average games as a transparent sparse-stat prior. Real zero-output
position games are retained; player-row count is never treated as evidence.
QB rushing, RB rushing, RB receiving, WR receiving, and TE receiving are
separate inputs. QB team rushing yards are floored at zero before forming a
baseline because box-score kneels can make net QB rushing negative.

The raw weekly offense (`game_team`) is preserved before the current-roster
merge, so a player's later team after a trade cannot rewrite the historic
offense-vs-defense assignment. Role-conditioned profiles use the same
team-game grain; QB passing deliberately bypasses role overlays, including
when the optional volume×efficiency rebuild is enabled.

This is a **data-integrity correction**, not a fitted performance claim. It
has deterministic all-stat partition-invariance tests and a no-leakage 2025
smoke backtest, but it must complete the project's locked multi-season
evaluation before becoming evidence of an accuracy improvement or a reason
to retune calibration/clip settings.

### Partial-game player evidence and expected-QB gate — data integrity, not an injury forecast

The weekly box-score feed contains a player’s recorded snap percentage, but
not a trustworthy historical injury timestamp or the clock time at which he
left a game. A quiet box score is therefore never enough to label a player
injured or to discard a normal lower-volume outing.

For a player’s own full-game production history only, Weekly Rankings removes
a game when the measured participation makes it an unusually clear
interruption:

- two QBs split the game’s offensive snaps in a normal relief range;
- an established player falls to at most 50% of snaps and at most 60% of his
  prior established role;
- a previously fringe same-position teammate takes a partial role in that
  exact game after the established player’s exit; or
- an established player is sharply reduced in a **28+ point win**, the narrow
  final-score pattern consistent with late-game rest.

The filter requires a real matched weekly snap source. Missing or zero-filled
snap data is explicitly left alone, and mild workload changes remain evidence.
An excluded row is removed from player rate averages, current-season evidence
counts, role trends, and expected snap shares; it remains in the raw
offense-team game used for defense profiles. The player popup records the
count and reason so a user can audit it. This fixes a denominator/data-shape
problem; it has not been presented as a measured accuracy improvement.

Quarterbacks have an additional participation rule: exactly one expected QB1
per team receives normal passing and rushing volume. A manual QB1 selection
is strongest. In a cold start, a lone 65% prior-season incumbent can be
automatic; in season, the candidate must be recently active for that team and
show a clear full-snap most-recent eligible game. All nonstarters receive zero
normal QB volume rather than inheriting a relief-game per-appearance rate. An
ambiguous room is held at zero until a visible manual selection resolves it.
This prevents a backup QB from appearing as a plausible 5–10 point weekly
option merely because he threw during a replacement or garbage-time drive.

### Local preseason depth-chart evidence

The model can use a user-imported snapshot of printer-friendly Ourlads depth
charts. This is a local-file import only: the app does not fetch, scrape,
automate a signed-in session, or contact Ourlads at projection time. The
raw pages and derived local snapshot stay outside Git. The parser preserves
the source's QB/RB/TE and `LWR`/`RWR`/`SWR` labels, source order, timestamp,
and availability class. In the experimental V2 path, an `lc_red` player is
an **unconfirmed chart warning**, not a medical determination: the resolved
player keeps the conditional depth-role signal unless a target-week injury
report or explicit manual availability override confirms that he is out. The
released V1 baseline keeps its prior red-row behavior as a control.

For a live preseason Week 1-style cold start, a uniquely matched first-listed
QB is a QB1 eligibility signal with this precedence: manual choice, imported
chart, clear prior-season incumbent, then an explicit unresolved room. A
current availability source can veto the chart; chart colour alone cannot. It
does not affect historical targets or in-season selection. For RB, WR, and
TE, imported chart order is a **low-evidence role floor**, not a workload
forecast: it can prevent a verified new starter from being treated as an
unknown, but it cannot reduce an established role or assert full snaps,
targets, carries, or equal workloads across the three listed WR formations.
For V2 RBs, a team-constrained allocator now distributes core-RB snaps,
carries, and targets separately among credible functional RBs, retaining an
explicit other-RB remainder. It uses stable player identity, literal source
rank, same-team and active-game evidence, draft signal, current availability,
and clear pre-absence/return segments. Fullbacks are excluded from the
core-RB allocator and retain only their own historic touch rate.

An optional local `data/availability_overrides.csv` supports an explicit
target-week decision when the public/current report needs correction. Its
columns are `year,week,team,player,status,plays_probability,workload_if_active,note`.
It is ignored by Git, resolves by stable ID/full name/reviewed alias/unique
suffix in that order, and refuses an ambiguous player rather than guessing.

### Headline: before this pass vs. what ships now

```
                        n      MAE     RMSE     bias   rank-corr   weeks won
before (Aug 2026)     8107    4.710    6.440   +0.176    0.654
AFTER                 8107    4.422    6.292   -0.640    0.689       26 / 26
naive trailing-4      8107    4.615    6.508   -0.126    0.668

by position (MAE / rank-corr):
  QB   6.759 -> 6.357    0.433 -> 0.476     22-4
  RB   4.687 -> 4.340    0.690 -> 0.727     26-0
  WR   4.691 -> 4.380    0.616 -> 0.643     26-0
  TE   3.717 -> 3.605    0.602 -> 0.622     24-2
```

The most important line is the third one. Before this pass the model was
**behind** a naive trailing-4-game average on both MAE and rank correlation
— the previous pass's own write-up says so, and flagged it as unexplained.
It is now ahead of that baseline on both, at every position, and it beat the
previous model in every single one of the 26 weeks tested.

### `role_volume` — SHIPPED, and it is essentially the whole improvement

The measured failure it fixes: of the 25 largest upgrades the old model made
over a trailing-average baseline, **sixteen were backup quarterbacks** — Joe
Milton, Joshua Dobbs, Kedon Slovis, Jalen Milroe, Tyson Bagent, Taylor
Heinicke — projected 12 to 17 points for a week they spent holding a
clipboard. The mechanism was subtle and entirely structural: a backup's
per-GAME rate is computed over the games he actually appeared in, which are
garbage-time drives where he really did throw the ball, so his rate looks
like a starter's on a small sample; every shrinkage path then pulls that
small sample toward the POSITION's average per-game production, which is a
starter's workload. Nothing in the model could tell the two apart.

Snap share tells them apart cleanly and without a judgment call. The
component re-denominates the position baseline from "per game played" to
"per FULL-SNAP game" and gives each player his own expected share of one
(`expected_snap_share`), and scales a prior-season per-game rate by the
change in his role since (`ROLE_VOLUME_CLIP`). It also catches the opposite
case for free, which is the one worth getting right: Tyler Shough went 4% ->
54% -> 90% -> 82% -> 95% -> 99% of snaps over six weeks of 2025, and a
four-appearance window reads him as a starter three weeks before a season
average would.

```
                        MAE      rank-corr    weeks won
role_volume vs base   -0.212      +0.035        26 / 26
```

**One real design decision inside it, settled by measurement.** The first
version averaged snap share over the player's TEAM's last four games,
scoring a week he missed as a zero. That reads a backup correctly and a
returning starter completely wrong — a back who missed two weeks came back
projected at a third of his role, and the startable-RB pool got materially
worse (MAE +0.28, rank-corr -0.09, losing 19 of 26 weeks). Averaging over
his last four APPEARANCES answers the question a start/sit call actually
asks — how big is his role when he is out there — and leaves "is he playing
at all" to the injury feed, which is the input that knows. Both readings
still separate the backups: over their last four appearances Joe Milton sits
at 21% of snaps and Joshua Dobbs at 18%, against 100% for a starter.

**Two bugs it exposed, both caught on real data rather than by inspection**,
both in the same place — what to do with a player who has no measured role
at all:

- `np.nan_to_num(share, nan=1.0)` treats "no snap data" as "every snap",
  which put three undrafted rookie running backs at the very top of a week-1
  board (Jacory Croskey-Merritt at 24.7 projected points). The position's
  own median share is the honest stand-in.
- In a cold start the share has to be read off the prior season, and reading
  it as "share when he appeared" hands a mop-up QB3 a starter's baseline off
  three blowouts — it put a third-string quarterback at QB5 overall on the
  2026 week-1 board. A cold start has no injury feed answer to "is he the
  starter", so it uses share of the whole team season instead (see
  `season_snap_share`'s two modes).

### `role_matchup` — SHIPPED, measured NEUTRAL, kept because it was the ask

This is the requested mechanism, made concrete: "a defense soft to a
possession receiver and airtight deep", "a receiving back vs. a high-volume
runner", "a high-completion QB vs. a high-ADOT low-completion QB" are all
one question — how good is this defense against a player who does THIS for a
living — and one mechanism answers all of them. Every player gets a role
label from his own measured season-to-date profile (never a hand-assigned
list, which goes stale the week a role changes); the defense gets a separate
rating per role; a player is priced against the rating for players like him,
shrunk toward the defense's overall rating by how much role-specific
evidence exists. Labels are TERCILES of the qualifying pool, not fixed
thresholds, so "downfield relative to his peers" means the same thing in a
season whose league-wide ADOT has moved.

It assigns roles correctly on real data — Chase, Nacua, St. Brown and
Flowers land as short/possession receivers, Tyreek Hill as deep; McCaffrey,
Achane and Bijan Robinson as receiving backs against Henry and Taylor as
rushers; Goff and Herbert as quick passers against Mayfield and Mahomes as
downfield ones.

It does not measurably help.

```
                                        MAE     rank-corr   weeks won
role_matchup on top of role_volume    +0.000     -0.001       12 / 26
   (at ROLE_MATCHUP_K = 4, the first try)  +0.006  -0.002     12 / 26
```

Kept, at `ROLE_MATCHUP_K = 10`, on the same grounds `HISTORY_MATCHUP_CLIP`
was kept in the previous pass: it is exactly the mechanism that was asked
for, it is measurably harmless (RMSE is a hair better, 6.378 -> 6.369; WR
and RB MAE a hair better, QB and TE a hair worse), and it makes the model's
matchup read legible. It is **not** claimed as an improvement. The honest
read on why it doesn't move anything: a defense plays ~9 games, splitting
those three ways leaves 2-4 observations per bucket, and the shrinkage that
keeps that from being noise also keeps it from being signal.

### `calibration` — SHIPPED

A projection should be a conditional expectation: among every player
projected for 20 points, the average one should score 20. This model's
wasn't. The top 15% of each position came in **+2.6 (QB), +2.0 (RB), +2.3
(WR), +0.5 (TE)** above what they actually scored, and regressing actual on
projected gave a slope well under 1 at every position — over-dispersion, not
bias. That is what selection always produces: the players a noisy projection
ranks highest are disproportionately the ones its own noise pushed up.

The correction is a per-position line, `actual ~ a + b * projected`,
**fitted on 2021-2023** — deliberately outside the 2024-2025 evaluation
window, so it is a measurement rather than a curve fitted to its own test
(`scripts/fit_weekly_calibration.py`).

Two things about how it is applied were settled by measurement, not taste:

- **Fitted on the whole pool, not the startable pool.** Fitting only on the
  top 40 per position produced slopes of 0.58-0.65 with intercepts of
  3.4-6.1 — a fine description of the top of a position and a transform that
  turns a 2-point bench receiver into a 5-point one.
- **Applied one-sided**, `min(projection, line(projection))`. The whole-pool
  line crosses the identity around 13/10/8/7 points (QB/RB/WR/TE), so it
  shrinks above that and *inflates* below — and the bulk does not need
  inflating. The two-sided version bought every startable gain and cost
  +0.116 whole-pool MAE, winning 1 week of 26, entirely from lifting several
  hundred near-zero bench rows. Clipping it to the shrink half keeps the
  correction where the defect is.

```
                                       MAE     RMSE   rank-corr   weeks won
calibration on top of the above       -0.076   -0.077   +0.000      24 / 26
startable-pool MAE:  QB -0.069   RB -0.202   WR -0.289   TE -0.109
startable bias:      +0.80 -> -0.81 (QB), +0.81 -> -0.09 (RB), +0.87 -> -0.38 (WR)
```

It is a monotone transform inside a position, so it **cannot** change who is
ranked above whom and does not pretend to. What it changes is the level —
which is what a projected point total is read for when it sits next to
FantasyPros' and the market's numbers on the same row.

### `teammate_vacancy` — SHIPPED, but unmeasured, and flagged as such

When a team's WR1 is ruled out his targets do not evaporate; they go to the
other receivers, and a model built on games he played in cannot know that.
This redistributes a sidelined player's projected targets and carries onto
his healthy teammates in proportion to their own volume, then scales the
stats that ride on that volume, capped (`VACANCY_ABSORB = 0.75`,
`VACANCY_MAX_GROWTH = 1.40`).

**It is not measured and cannot be by this harness.** It fires only off the
live injury feed, and the backtest runs with injuries off, so it is inert in
every number above — the shipping model's measured results are identical
with it on or off. It ships anyway as a judgment call: a receiver's targets
demonstrably do not disappear when he is inactive, so ignoring it is
knowably wrong rather than merely unmeasured. The constants are conservative
for exactly that reason.

A real bug in it was caught by a unit test rather than by inspection: the
first version recovered a sidelined player's vacated volume as
`projection / injury_multiplier`, and a player ruled Out has a multiplier of
exactly 0.0 — so his projection is 0 and there is nothing to divide back
out. It silently redistributed zero for every Out player, which is the only
case that matters. The pre-injury volume is now stashed by the position loop.

### `volume_efficiency` — BUILT, MEASURED, REJECTED

The industry-standard layering: project OPPORTUNITY first (attempts,
carries, targets), then apply a per-opportunity efficiency to it, with
efficiency evidence counted in opportunities rather than games and shrunk on
published stabilization ranges. The motivation was real and measured — the
old model over-projected the top 15% of every position on every counting
stat at once, WR targets +11% / receptions +16% / receiving yards +18%,
which is the signature of yardage being modelled as its own independent
per-game rate rather than as opportunities × efficiency.

```
                                       MAE     rank-corr   weeks won
volume_efficiency on top of role_volume  +0.051   -0.005     5 / 26
```

Rejected. The diagnosis was right and the fix did not follow from it: the
shrinkage target for a top player's efficiency is his own prior-season
efficiency, which is also high, so the layer didn't pull down the players
that were over-projected. `calibration` addresses the same defect directly
and does work. The code stays, off, with this note.

### `game_env` — BUILT, MEASURED, REJECTED

Market-implied team total plus venue, with the elasticities measured on
**2019-2023** (21,330 player-games, outside the evaluation window): log-log
elasticity of a player's own game-to-season ratio against his team's implied
points is QB 0.416, TE 0.301, RB 0.168, WR 0.140, and indoor/outdoor is
QB 1.070, TE 1.052, WR 1.040, RB 1.001.

```
                                     MAE     rank-corr   weeks won
game_env at measured elasticity     +0.012    -0.000     11 / 26
game_env at half elasticity         +0.006    -0.000     10 / 26
```

Rejected at both strengths. QB rank-corr does improve (+0.013 at full
strength, and +0.034 on the startable-QB pool) which is not nothing given QB
is the model's weakest position — but it is one position moving inside noise
against a whole-pool cost, and that is not enough to ship on.

**The largest effect measured in that study is deliberately unused.** Wind,
in outdoor games at 15+ mph: QB 0.880 against 1.017 in calm air, TE 0.907,
WR 0.895, and RB unaffected at 0.965 — teams run more into a wind, which is
exactly the right shape for the effect to be real. nflverse populates `wind`
and `temp` AFTER a game is played, not when the schedule is published, so a
backtest would happily consume it and report an improvement the live model
could never reproduce: on the Thursday you actually set a lineup that column
is empty. Recorded so the next person to spot the wind column knows it was
measured, and why it was left out anyway. A real forecast feed would make
this the most valuable single addition available to this model.

### `v2_pff_alignment_matchup` (WR/TE slot/non-slot defense residual) — BUILT, MEASURED, REJECTED

The user's proposed design, built as specified: a WR/TE's own slot-rate /
non-slot-rate mix (`data.pff_alignment.load_weekly_alignment_profiles`, time-
valid weekly-grain PFF data, in-season only — 2025 is the only year with
weekly-grain files, see `pff_imports/`) run through the opponent's shrunk
slot vs. non-slot allowed rate (`aggregate_alignment_defense_profiles`),
blended as `player_factor = slot_rate * defense_slot_ratio + (1 - slot_rate)
* defense_non_slot_ratio`, expressed as an incremental residual against the
position-normal blend, double-shrunk toward 1.0 by both sides' confidence,
and clipped to `ALIGNMENT_DEFENSE_RESIDUAL_CLIP = (0.90, 1.10)`
(`alignment_defense_residual_multiplier` in `data/pff_alignment.py`). Wired
into the existing matchup step for targets/receptions/receiving_yards only —
touchdowns stay neutral, same reasoning as everywhere else in this model.

```
                                          MAE       rank-corr   weeks won
WR (whole pool)                        -0.003        -0.000      11 / 17
TE (whole pool)                        +0.001        +0.000       6 / 17
START-WR (startable, decision pool)    +0.022        -0.004       7 / 17
START-TE (startable, decision pool)    +0.082        -0.026       4 / 17
```

Paired A/B, `DEFAULT_FEATURES` vs `DEFAULT_FEATURES + v2_pff_alignment_matchup`,
2025 weeks 2-18 (week 1 excluded: cold start / season-prior fallback is a
separate code path). Rejected. The whole-pool WR number looks like a win but
is exactly the false signal `scripts/eval_weekly_model.py`'s own docstring
warns about — a pool dominated by bench players who are trivially easy to
rank. The population that actually drives a start/sit decision lost on both
metrics, worst on START-TE (4-13 losing weeks, +0.082 MAE — a larger loss
than either whole-pool WR gained). Most likely cause: 2025 is the only season
with weekly-grain PFF alignment data, so even with
`ALIGNMENT_DEFENSE_SHRINKAGE_GAMES=4` the defense-side slot/non-slot split is
built on a handful of games per team at this point in the season — thinner
than what `role_matchup`'s existing defense-vs-position table already has to
work with. Not a parameter to retune (the shrinkage/clip constants are
already conservative starting defaults, not fitted claims); more likely just
needs more weekly-grain seasons before this evidence is worth trusting. The
code stays in place, reachable by the explicit feature name, for a future
re-study.

**Update, 2026-08-24, same day:** re-enabled in `V2_EXPERIMENTAL_FEATURES`
(only — `DEFAULT_FEATURES` is unchanged) at the user's explicit request, to
inspect real per-player/week numbers on a live V2 board while looking for a
fixable upstream cause rather than accepting the rejection at face value.
The Weekly Rankings decomposition table shows an "Alignment residual" column
whenever it's active. This is a diagnostic convenience, not a reversal of
the result above — if a real cause is found and fixed, re-measure before
promoting it anywhere; if the re-look just confirms the rejection, pull it
back out of `V2_EXPERIMENTAL_FEATURES` too.

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

**Read honestly** (this paragraph describes the model as it stood BEFORE the
August 2026 component pass documented above — the table it refers to is the
pre-pass one; see the headline table in that section for where it stands
now): the model is essentially at parity with a simple trailing-average
baseline on this one-season backtest (MAE within 1.5%, rank correlation
within 1.2%), not a decisive improvement on either metric. It is not
overclaimed as one. What it adds over the naive baseline isn't
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

- **Week 1 is a cold start, not a blank** — it falls back entirely to
  prior-season rates and a prior-season defensive matchup matrix (see
  `build_weekly_projections`'s COLD START section). It is a real projection
  and a deliberately weak one: a whole offseason has happened since the
  numbers behind it. FantasyPros' own weekly projection
  (`data.draft_sources.fetch_fantasypros_weekly_projections`, wired into the
  Weekly Rankings tab) has season-opener-specific analyst input this app
  does not and remains the better source for the opener where it's available.
- **Residual calibration error at QB and TE.** The calibration line is
  fitted on 2021-2023 and transfers imperfectly: on 2024-2025 it takes the
  startable-QB bias from +0.80 to **-0.81** and startable TE from -0.05 to
  **-0.87** — i.e. it now under-projects the top of those two positions by
  about as much as it used to over-project them. RB (-0.09) and WR (-0.38)
  land close to centred. Not tuned away, because tuning it against
  2024-2025 is exactly the thing the out-of-sample fit exists to avoid;
  re-fitting on a wider window is the honest fix when more seasons are
  available.
- **Startable rank correlation is barely moved by any of this.** Whole-pool
  rank correlation is up meaningfully (0.654 -> 0.689), but inside the
  startable tier it is roughly flat and slightly down at QB (-0.021), WR
  (-0.011) and TE (-0.053). Everything shipped in this pass fixes LEVEL
  errors — who is a starter at all, how big the top of a position should
  read. None of it is new information about which of two comparable
  starters will out-score the other, which remains the hardest and least
  solved part of the problem.
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
