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

**Update, 2026-08-26:** two changes, both per explicit request, neither yet
re-measured against `scripts/validate_weekly_projections.py`.

First, `v2_pff_alignment_matchup` was folded into `DEFAULT_FEATURES` when the
separate V1/V2 toggle was retired (see that commit) — it now ships live, not
experimental, despite the measured loss above.

Second, the mechanism itself was redesigned the same day, for two stated
reasons: (a) it was "normalizing too aggressively back to league average"
for a model that should be confident in its reads, and (b) multiplying an
alignment residual on top of the broad role/defense matchup was two
independent opinions of the same WR/TE matchup stacked together —
redundant. Fix: `alignment_defense_residual_multiplier` (in
`data/pff_alignment.py`) no longer divides by a position-normal alignment
mix, and no longer applies a second confidence-based shrink toward 1.0 on
top of each alignment's own sample-size shrinkage — see that function's own
docstring. And in `data/weekly_projections.py`, its output now REPLACES the
broad matchup multiplier for a WR/TE stat with available alignment evidence,
rather than multiplying alongside it. Separately, the 2024 weekly-grain PFF
archive (`pff_imports/2024/weekly/`) — the "more seasons of evidence" fix
the original rejection's own "likely cause" pointed at — is now present and
loaded (previously excluded even at cold start; see
`ALIGNMENT_PRIOR2_MAX_WEIGHT`'s own comment).

The A/B numbers above describe the OLD incremental design measured against
2025-only evidence. They do not describe current behavior. Re-run the
backtest before drawing any conclusion about this component's accuracy —
in either direction.

**Update, 2026-08-26, same day, re-measured:** paired A/B via
`scripts/eval_weekly_model.py`, isolating exactly `v2_pff_alignment_matchup`
(`DEFAULT_FEATURES` with vs. without that one flag — every other component
held identical), 2025 weeks 2-18, same window as the original measurement
above.

```
                                          MAE       rank-corr   weeks won
WR (whole pool)                        -0.001        -0.001       8 / 17
TE (whole pool)                        -0.018        +0.000       8 / 17
START-WR (startable, decision pool)    -0.069        +0.024      11 / 17
START-TE (startable, decision pool)    +0.025        +0.002       8 / 17
```
(Sign convention here: negative MAE / positive rank-corr = alignment wins.
"weeks won" = weeks alignment's own MAE beat the without-alignment variant's.)

START-WR — the scope the original 2026-08-24 rejection failed worst on after
whole-pool WR's misleading near-zero "win" — reverses outright: -0.069 MAE,
+0.024 rank-corr, winning 11 of 17 weeks. START-TE moves the other way, a
small MAE loss (+0.025) though its rank-corr still edges in alignment's favor
(+0.002) — a much smaller and more ambiguous effect than the old design's
+0.082 MAE / 4-13 losing weeks on the same scope. Whole-pool WR/TE are
essentially a wash either way, consistent with those pools being bench-heavy
and easy to rank regardless of matchup detail. QB/RB are unaffected, as
expected (alignment only ever touches WR/TE targets/receptions/receiving_yards).

Kept ON in `DEFAULT_FEATURES` either way, per explicit instruction: "if it
doesn't pass don't just remove it, just report the statistics." This is not
a pass/fail gate — these are the current honest numbers, not a claim the
component is settled. Re-run this A/B again after any future change to
`alignment_defense_residual_multiplier` or its inputs.

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

## 2026-09-07 pass — practice-squad roster fix + cold-start receiver vacancy

Two changes, after a live 2026 Week-1 board review. The roster fix is a
reasoned correction; the receiver-room change was chosen from a backtest
(see item 2).

1. **Practice-squad players were entering the Week-1 pool.** A live season
   reads the nflverse roster feed (`data.loaders._load_feed_roster`), whose
   status vocabulary spells practice squad `DEV`. `_cold_start_pool`'s hard
   filter was a private `{'RET','CUT','RES','FA'}` set written against the
   frozen local snapshot's vocabulary and never updated for the feed, so
   ~540 practice-squad players/season (Cedric Tillman, Stone Smartt, Trey
   Palmer on 2026 NO, and the equivalent on every team) landed on the board
   with real snap shares and target volume. Fixed by pointing that filter at
   the shared `INELIGIBLE_ROSTER_STATUSES` set (`data.rb_role_allocator`) and
   adding `DEV`/`EXE` to it. A genuine Week-1 starter stuck on a lagging feed
   is still re-added by `apply_ourlads_starter_roster_overlay` (starters
   only) or a `data/availability_overrides.csv` row.

2. **`v2_receiver_cold_start_vacancy`** promoted to `DEFAULT_FEATURES`
   (cold start only). Symptom on the 2026 Week-1 board: a wideout left a
   team and his targets were not reclaimed by the remaining WRs, so the
   pass-capacity allocator's uniform WR/TE fit scaled the *tight end* up
   with them (LAC Gadsden after Keenan Allen, GB Kraft after Doubs+Wicks,
   TB Otton after Evans). This flag (`apply_cold_start_receiver_vacancy`)
   fixes it at the source: any WR/TE who held real role in last season's
   reference but is absent from this year's pool entirely has
   `RECEIVER_COLD_START_VACANCY_SURVIVAL` (0.70) of his prior share
   redistributed to the players who remain, weighted by each recipient's
   own current share — so the vacated targets flow to the other WRs before
   the capacity fit ever runs.

   Chosen over **`v2_wr_te_capacity_split`**, a same-goal lever that damped
   the WR/TE reconciliation instead of fixing the input. Three backtests
   (`.sweeps/wr_te_capacity_split_2022-2025.txt`,
   `receiver_cold_start_vacancy_wk1_2022-2025.txt`,
   `split_vs_vacancy_combo_2022-2025.txt`, all 2022-2025 wk1-2 / wk1, paired
   vs. the then-shipped stack):

   | config | ALL | WR | START-WR | START-TE |
   |---|---|---|---|---|
   | split ON (old default) | wash at every weight | +0.03…+0.13\* worse | +0.06…+0.22\* worse | −0.13…−0.35 better |
   | **vacancy ON, split OFF** | **−0.019\*** | **−0.098\*** (8-0) | **−0.275\*** | +0.117 (n.s.) |
   | both ON ("combo") | −0.008 (n.s.) | −0.077\* | −0.237\* | **+0.148\*** worse |
   | neither | −0.007 (n.s.) | −0.063\* | −0.101\* | +0.245 (n.s.) |

   `*` = bootstrap 95% CI excludes 0. Vacancy-alone is the only arm that
   moves ALL significantly in the right direction; it costs startable TEs
   ~+0.12 MAE (not significant, 2-6 by week) to route the vacancy where it
   belongs, on ~2.8× the startable population. Stacking both was strictly
   worse — the capacity split piles on after the vacancy layer already
   moved the volume, and it makes startable TEs significantly worse.
   `v2_wr_te_capacity_split` is retained as an opt-in flag only.

## 2026-09-07 — cold-start over-projection: deadband + season-phase calibration

A signed-bias check (`mean(pred - actual)`, wk1-2 2022-25) found the model
projects HIGH at cold start at **every** position — startable QB +1.07,
RB +1.41, WR +1.59, **TE +2.85** — and 8 of 8 weeks high at TE. This is the
opposite sign from the wk5-17 hold-out the single `WEEKLY_CALIBRATION` line
is fitted on, which is why every WR↔TE reallocation lever (`v2_wr_te_capacity_
split`, cold-start-vacancy cross-position bleed, the TE-2 dock) only ever
traded the error sideways: there is no under-projected room to move it to.

Two changes:

1. **`PASS_CAPACITY_DEADBAND` 1.0 → 0.5** (`data/pass_capacity_allocator.py`).
   The ±1 band left a re-shaped cold-start room (prior-year target shares
   summing past 100%) ~1 target/player over budget uncorrected. The per-stat
   bias is a volume problem (startable TE targets +0.9, receiving yards +9.7),
   so tightening the conservation fit is the on-target correction.

2. **Season-phase `WEEKLY_CALIBRATION` (v4)** — the single per-position line
   is refit on 2021-2025 and, for WR and TE only, split into two season
   phases: `WEEKLY_CALIBRATION_BY_BUCKET['cold']` for weeks 1-4 (the harder
   shrink), `['rest']` for weeks 5-18. QB and RB keep one line all year.
   `_weekly_calibration_for(pos, week)` does the lookup and is called at both
   calibration sites and the decomposition dialog.

   *How the scheme was chosen* (`scripts/fit_seasonal_calibration.py`:
   `--mode dump` builds every week 1-18 × 2021-2025 with
   `CALIBRATION_INPUT_FEATURES`, then `--mode analyze` runs a per-(pos, week)
   bias/slope grid and a bucketing bake-off, `--mode emit` prints the ship
   values). The per-week grid, startable pool, showed the cold-start *high*
   is a receiver-room effect that decays over ~4 weeks:

   | week | 1 | 2 | 3 | 4 | 5–18 mean |
   |---|---|---|---|---|---|
   | WR bias | +1.8 | −0.0 | +0.0 | +0.0 | −1.6 |
   | TE bias | +3.0 | +0.9 | +1.0 | −1.2 | −1.1 |
   | QB bias | +1.2 | −0.1 | +1.3 | +0.8 | +0.6 (noise, no phase) |
   | RB bias | +3.0 | +1.5 | +0.6 | −0.7 | −0.5 (wk1-2 only, too thin) |

   Bake-off (fit 2021-2023, scored held-out 2024-2025; re-checked fit
   2021-2024 / held-out 2025). Only WR and TE clear the bar and a 2-bucket
   split (`cold4_rest`) captures all of it — a third "late" bucket adds
   nothing; QB/RB bucketing is a wash-to-worse (d-START-MAE QB +0.010,
   RB +0.002), so they stay single.

   *A per-STAT calibration was checked and rejected.* The cold-start
   distortion is concentrated in `receiving_yards` for WR/TE (raw bias
   +10.6 / +11.6 in weeks 1-2), which the points-level cold bucket already
   absorbs (startable TE |bias| 1.07 → 1.01). An 11-stat × 2-bucket ×
   4-position layer bought no measured gain over the points bucket.

   *Held-out and end-to-end results.* Held-out 2025 startable MAE, v3 single
   line → v4: QB 6.47 → 6.47, RB 6.18 → 6.14, WR 6.10 → 6.03, TE 5.38 →
   5.25. Full-model A/B through `DEFAULT_FEATURES`, 2024-2025 wk1-17
   (`scripts/eval_weekly_model.py` logic, v3 globals vs v4 globals):

   | scope | START-MAE v3 → v4 | startable bias v3 → v4 |
   |---|---|---|
   | ALL | 4.554 → 4.518 | −0.42 → −0.56 |
   | START-QB | 6.251 → 6.247 | −0.12 → −0.22 |
   | START-RB | 6.250 → 6.215 | +0.12 → −0.04 |
   | START-WR | 6.478 → 6.390 | −0.41 → −0.71 |
   | START-TE | 5.560 → 5.404 | −0.51 → −0.78 |

   Startable MAE — the ranked metric — improves at all four positions, most
   where it was aimed (WR −0.088, TE −0.157); rank correlation is flat
   (±0.01), so lineup ordering is unchanged. **Known cost, accepted:**
   startable bias drifts ~0.15-0.30 more negative on QB/WR/TE, concentrated
   in the in-season weeks (not the cold weeks this targets) — the same
   MAE-for-bias trade the v3 two-sided line already made. Re-fit whenever
   `DEFAULT_FEATURES` or `PASS_CAPACITY_DEADBAND` changes: `--mode dump` then
   `--mode emit`.

## 2026-09-07 — implied-total elasticity moved to per-stat

`v2_game_total_elasticity` (one flat per-position exponent — QB 0.42 — applied
to *every* projected stat alike) was replaced in `DEFAULT_FEATURES` by
`v2_game_total_elasticity_perstat`, which carries a separately fitted exponent
per `(position, stat)`. Motivation, from a live 2026 Week-1 board review: an
underdog QB (ARI's Brissett, implied team total 18.75 vs a 22.7 league mean)
was getting his pass **attempts** scaled by `(18.75/22.72)^0.42 = 0.92`, on top
of an independent opponent-pace multiplier (0.93) that largely encodes the same
"this team will be behind" fact. A trailing team does not throw the ball 8%
less — pass volume is game-script and pace, not scoring environment.

The fitted QB exponents (`_GTE_PERSTAT_FITTED`, from
`scripts/fit_game_total_elasticity_perstat.py` on 2016-2023) bear that out:
`passing_attempts` **0.030**, `passing_completions` 0.083, `passing_yards`
0.138, `passing_tds` 0.266. The scoring-environment signal lands on TDs and,
softly, yardage — and barely touches attempts.

Held-out confirm, 2021-2023 **and** 2024-2025, weeks 3-18
(`scripts/gte_perstat_confirm.py`, `.sweeps/gte_perstat_confirm_*.txt`):

```
                              dMAE 2021-23    dMAE 2024-25 (held out)   CI excl 0
ALL (points)                    -0.001          -0.000
START-QB (points)               +0.040          +0.023                   no
START-TE (points)               -0.041          -0.059                   no
passing_attempts (startable)    -0.136          -0.196                   yes (2024-25)
passing_completions (startable) -0.066          -0.110                   yes (2024-25)
passing_yards (startable)       -0.580          -0.606                   no (close)
```

Points-level it is a wash (ALL ≈ 0.000), with a small, non-significant
startable-QB points cost and small WR/TE gains. The decisive line is the
volume breakdown: pass attempts and completions get materially and
significantly more accurate — which is what a Week-1 QB board, dominated by
pass-volume-driven projections, is actually read for. End-to-end on 2026
Week 1: Brissett 31.8→34.2 att / 214→226 yд; Herbert (−10 favourite)
37.6→34.3 att / 277→259 yд — the flat 0.42 had been symmetrically inflating
the favourite and deflating the underdog.

The two flags are mutually exclusive on the implied-total channel
(`_game_total_stat_multipliers`); venue is unaffected (its own flag).
`GAME_TOTAL_ELASTICITY_BY_STAT` values are now inlined; the fit script's JSON
override still layers on top if a re-fit writes one.

**Deferred:** `WEEKLY_CALIBRATION` was fitted against the old
`CALIBRATION_INPUT_FEATURES` (flat flag in the set). The ALL-scope move is
~0.000, so a re-fit is a queued follow-up, not a blocker — same call the
`v2_pff_defense_prior_blend` ship made. Re-fit: `fit_seasonal_calibration.py
--mode dump` then `--mode emit`.

## 2026-09-07 — cold-start receiver-room over-projection: two fixes tried, both rejected

Live 2026 Week-1 case: LAC's Oronde Gadsden II (a returning TE-2 who played
~62% of snaps in 2025, charted TE-2 again in 2026) projected for a **0.92**
expected snap share — the exact value of `RECEIVER_COLD_START_VACANCY_MAX_SHARE`.
Cause: three LAC tight ends who time-shared the *other* TE role in 2025 (Fisk
0.35, Dissly 0.32, Conklin 0.30) all left, `v2_receiver_cold_start_vacancy`
**summed** their shares (further inflated by the `_blend_with_prior2` step,
which pulls Conklin toward his 2024 NYJ starter role and Dissly toward his
2024 SEA role), and redistributed 70% of that ~1.2 pool onto the remaining
TEs weighted by current share — pinning the biggest incumbent at the cap.
The LAC TE room summed past 2.0; David Njoku (charted **TE-3**) sat at 0.34
while Gadsden sat at 0.92.

Snap share is genuinely not conserved to 100% — a team runs one, two, or
three TEs per play, so a room legitimately sums to ~1.2–1.5 for TE and
~2.1–2.4 for WR (measured 2025: LAC TE 0.89, ARI TE 1.52, PHI TE 1.21, league
WR ~2.2) — but it *is* bounded, and by a real team signal.

### `v2_cold_start_room_budget` — BUILT, BACKTESTED, REJECTED

Clamp the sum of a team's WR (or TE) room shares to `prior_room_snap_budget`
(that team's own prior-season Σ of whole-season participation shares) ×
`(1 + COLD_START_ROOM_BUDGET_TOLERANCE)`, water-filling the excess out of the
players the Ourlads chart lists as deep reserves first
(`apply_cold_start_room_snap_budget`), only scaling the protected top tier
if zeroing the whole unprotected tail is not enough. On the live board it did
exactly what was wanted: LAC TE room 2.07 → 1.15, Gadsden 0.92 → 0.59,
Njoku 0.34 → 0.03, Kolar 0.82 → 0.53.

```
scripts/sweep_cold_start_room_budget.py, wk1-2 2022-2025, paired vs the shipped stack
                     n   MAE base   MAE var     dMAE            95% CI     wk W-L
ALL               2358      4.882     4.902    +0.019  [+0.005,+0.034] *     0-4
WR                1016      5.306     5.353    +0.047  [+0.012,+0.086] *     0-4
START-WR           417      6.964     7.041    +0.077  [+0.013,+0.153] *     1-3
TE                 459      3.566     3.562    -0.005  [-0.040,+0.037]       3-1
START-TE           151      4.512     4.471    -0.042  [-0.128,+0.035]       2-2
startable recv-yд  568     30.416    30.725    +0.308  [+0.066,+0.583] *     0-4
```

Rejected. It reshapes ~115 rooms per Week 1, not the ~5 pathological ones,
and the depth-chart protection is not robust on the frozen historical
Ourlads charts — it water-filled genuine WR1s to the floor (2025 NE Stefon
Diggs 0.63 → 0.03, projected 8.3 → 1.7, actual 11.7). Every significant
move is a *regression*; the only improvements (whole-pool TE, startable TE)
are not significant. Kept as an opt-in `MODEL_FEATURES` flag, OFF.

### Dampening the vacancy pool itself — BUILT, BACKTESTED, REJECTED (not kept)

The narrower alternative: leave rooms alone, just stop the *vacancy* step
over-counting. Read each departed player's **unblended** prior-season share
(so `RECEIVER_COLD_START_VACANCY_MIN_SHARE` can drop a real 2025 bit-part),
then fold a room's departures instead of summing them — largest in full,
each additional at half, a hard `0.80` ceiling (`_effective_vacated_pool`).
On the board this took Gadsden 0.92 → 0.81.

```
scripts/sweep_vacancy_dampen.py, wk1 2022-2025, 3-arm (none / raw vacancy / guarded vacancy)
GUARDED vs RAW        n   MAE raw   MAE grd    dMAE            95% CI     wk W-L
ALL               1208     4.853     4.866    +0.013  [+0.002,+0.024] *    1-3
TE                 245     3.586     3.695    +0.109  [+0.082,+0.150] *    0-4
START-TE           77      4.253     4.563    +0.311  [+0.136,+0.563] *    0-4
START-WR          209      6.831     6.896    +0.065  [-0.075,+0.178]      1-3
```

Rejected and **not** kept in the tree. Same failure shape as the room
budget: the pool cap trims ~300 legitimate redistributions across the slate
to correct ~5 pathological ones, and it makes startable TE significantly
worse than the shipped raw vacancy. The `GUARDED-vs-NONE` and `RAW-vs-NONE`
arms both still show the original startable-WR win (−0.275 / −0.339), so the
vacancy feature itself stays exactly as it was.

**Net: the shipped `v2_receiver_cold_start_vacancy` is unchanged** by the two
room/pool-sizing attempts above. Both cost more accuracy across the slate
than they save on the handful of inflated rooms.

## 2026-09-08 — surgical vacancy-recipient guards (`v2_vacancy_bump_cap` SHIPPED)

Third pass, after the two failures above. Insight: don't touch the pool size
or any non-recipient — the thing to shape is *how far one recipient can be
moved*. Three opt-in flags, each guarding only the redistribution
(`apply_cold_start_receiver_vacancy`), each gated to recipients **with a real
prior role** so a no-prior rookie inheriting a departed starter's whole job
(Sam LaPorta, DET 2023, sole rostered TE after the Hockenson trade) keeps the
full inheritance — the earlier ungated version stranded him at 0.32:

- **`v2_vacancy_bump_cap`** — no one prior-role recipient gains more than
  `RECEIVER_COLD_START_VACANCY_MAX_BUMP` (0.15) from a team's vacancy;
  leftover pool dropped.
- **`v2_vacancy_growth_cap`** — post-vacancy ≤ pre-vacancy ×
  `RECEIVER_COLD_START_VACANCY_GROWTH_CAP` (1.35), with a `MIN_ABS_GAIN`
  (0.12) absolute floor. The cold-start analogue of the in-season
  `VACANCY_MAX_GROWTH`, which the cold-start path never had.
- **`v2_vacancy_chart_split`** — split the pool by Ourlads chart rank so a
  vacated role flows to the charted starter, not to a returning backup the
  chart still lists behind him. Pool total unchanged.

`scripts/sweep_vacancy_recipient_guards.py`, wk1 2022-2025, vs the shipped
stack:

```
arm            ALL       WR        TE        START-WR      START-TE      START-QB
bump_cap     -0.010    -0.002   -0.045*    +0.098 n.s.   +0.350 n.s.   -0.052
growth_cap   -0.007    +0.004   -0.042*    +0.084 n.s.   +0.333 n.s.   -0.125
bump+growth  -0.012*   -0.001   -0.055*    +0.147*       +0.353 n.s.   -0.109
all3         -0.015*   -0.008   -0.059     +0.128 n.s.   +0.206*       -0.109
```

(`*` = bootstrap 95% CI on the pooled weekly dMAE excludes zero. START-TE
n=77 — the small, noisy cold-start pool the calibration section already
flags; its CI spans zero and the by-week record is 1-3, not a rout.)

**`v2_vacancy_bump_cap` shipped** (`DEFAULT_FEATURES`, 2026-09-08). It is the
only arm with **no CI-excludes-0 regression on any pool**: `bump+growth`
makes START-WR significantly worse, `all3` (the chart-split stack) makes
START-TE significantly worse, and chart-split alone barely moves the target
case (it hands the charted TE-1 the pool instead). `bump_cap` improves
whole-pool TE significantly (−0.045) and ALL slightly (−0.010), for a
START-TE cost that stays inside noise. On the live 2026 board it takes **LAC
Gadsden 0.92 → 0.75** (and MHJ 0.92 → 0.84, Kolar 0.82 → 0.65 — both also
vacancy-inflated).

Known residuals, accepted at ship: it does not reach the ~0.62 a clean fix
would give — 0.75 is a returning TE-2 who genuinely absorbs *some* of a
departed rotation — and it clips a real incoming TE-1 whose own prior role
was also starter-level (Darren Waller, NYG 2023) by more than it should. A
second gate on the recipient's *own* prior ceiling would tighten that; not
built. `MAX_BUMP = 0.15` is a first-pass value, not swept. A
`WEEKLY_CALIBRATION` re-fit is the deferred follow-up (the ALL move is small,
same grounds as the per-stat-elasticity ship — re-fit both together:
`fit_seasonal_calibration.py --mode dump` then `--mode emit`).

`v2_vacancy_growth_cap` and `v2_vacancy_chart_split` stay switchable and OFF.

## 2026-09-08 — `v2_td_volume_shrink` (regress cold-start WR/TE TD rate to league) — BUILT, BACKTESTED, REJECTED

Observation: a projected WR1 off a TD-unlucky season (Justin Jefferson, 2
receiving TDs in 17 games on a bottom-tier 2025 offence) stays anchored near
that rate. `credibility_shrunk_td_prior` weights the player's own rate by his
OPPORTUNITY count - a ~550-target WR keeps ~91% of his rate though only
~15 TD *events* stand behind it - and it regresses toward a per-game position
mean diluted by every low-snap WR, which for a WR1 barely moves and can pull
down.

`v2_td_volume_shrink` re-does the shrink for WR/TE `receiving_tds` at cold
start: credibility from the 2-year TD COUNT (`n / (n + 12)`), regressed toward
`LEAGUE_TD_PER_TARGET[pos]` × the player's own projected target rate
(`td_volume_shrunk_prior`). On the live 2026 board it does the right thing
two-sided: Jefferson 0.25 → 0.32, Lamb 0.36 → 0.52, Burden 0.20 → 0.36 up;
St. Brown 0.62 → 0.58, Kittle 0.39 → 0.33, G. Wilson 0.54 → 0.48 down.

```
scripts/sweep_td_volume_shrink.py, wk1 2022-2025, vs the shipped stack
scope        n      MAE base   MAE var    dMAE             95% CI      wk W-L
ALL        1208      4.843     4.868    +0.024  [+0.006,+0.042] *      0-4
TE          245      3.541     3.639    +0.099  [+0.068,+0.134] *      0-4
START-WR    208      6.929     6.950    +0.020  [-0.064,+0.124]        2-2
START-TE     77      4.603     4.558    -0.044  [-0.272,+0.164]        2-2
receiving_tds MAE, startable WR/TE:  0.445 -> 0.475  +0.030 [+0.020,+0.043] *  0-4
```

Rejected. Every significant move is a regression, **including the target
stat** (startable WR/TE `receiving_tds` MAE +0.030, CI excludes 0). The
`LEAGUE_TD_PER_TARGET` anchors (WR 0.052, TE 0.058) applied to every player's
projected volume systematically lift the position - the TE ledger is ~all `+`
shifts toward a per-game rate a tight end does not hit in a single game -
because selection means the low-observed-rate players get pulled up while
credibility protects the high ones from coming down. Regressing an unlucky
rate UP adds more error than it removes on this window.

**Strength sweep** (`TD_VOLUME_SHRINK_STRENGTH` 0..1, two passes: `--strengths
0.25,0.4,0.55,0.7,1.0` then a softer `--strengths 0.1,0.175,0.25`, same
window): no dilution passes.

```
strength    ALL       WR       TE      recv_tds MAE   recv_tds MEDIAN
0.10      -0.010*  -0.021*   -0.007      +0.012*         +0.025
0.175     -0.007   -0.017    +0.003      +0.013*         +0.028
0.25      -0.004   -0.015*   +0.011      +0.015*         +0.030
0.40      +0.000   -0.013    +0.030*     +0.017*         +0.033
0.55      +0.005   -0.010    +0.047*     +0.021*         +0.039
1.00      +0.024*  +0.010    +0.099*     +0.029*         +0.049
```

`receiving_tds` MAE is significantly worse at **every** strength down to 0.10,
and its MEDIAN error never improves - so the TD projections themselves get
less accurate however lightly the shrink is applied. `START-TE` (a ship-gate
pool) is +0.129\*/+0.119\*/+0.127\* at strengths 0.10/0.175/0.25 - a
consistent, significant startable-TE regression at every soft strength,
because the shrink pulls elite TEs' durable red-zone TD rate (real role, not
luck) down toward a league mean. The whole-pool WR points improvement at
0.10-0.25 (-0.015\* to -0.021\*) is *not* coming from the TDs (which got
worse): it is measured on the full ~180-WR pool - mostly deep guys whose own
TD rate is noise - is non-monotonic across strength, and does not touch
targets/QB/RB (the shrink only edits the WR/TE `receiving_tds` line, verified
- no cross-position path). Kept as a switchable flag, OFF.

## 2026-09-08 — `v2_rookie_backup_wr_dampen` (dock a charted rookie backup WR) — BUILT, BACKTESTED, REJECTED

Live 2026 case: J. Michael Sturdivant (GB rookie, no NFL history) charted
`RWR-2` on Ourlads, projected **0.32 expected snaps / ~1.9 targets** behind a
GB WR room that lost Doubs. He gets the 0.16 rank-2 role floor AND a slice of
the cold-start vacancy pool. ~12 other rookie deep-WRs (Antonio Williams,
Caleb Douglas, Bryce Lance, ...) show the same shape.

`v2_rookie_backup_wr_dampen`: a no-prior WR the chart lists at slot rank >= 2
is pulled to `ROOKIE_BACKUP_WR_SHARE` (0.06) instead of the 0.16 floor, and
`apply_cold_start_receiver_vacancy` gives him zero weight (a departed WR1's
snaps flow to the proven WRs and the charted WR1, not a rookie 4th). Rank-1
no-prior WRs - a charted rookie *starter* - are untouched. On the board it
takes Sturdivant to ~0.05 snaps and correctly nukes Zavion Thomas, Chris
Bell, Barion Brown, Josh Cameron; the GB vacancy it removed from Sturdivant
lands on Bo Melton (a 3rd-year vet) instead.

```
scripts/sweep_rookie_backup_wr_dampen.py, wk1 2022-2025, vs the shipped stack
scope       n      MAE base   MAE var    dMAE             95% CI     wk W-L
ALL       1208      4.843     4.853    +0.009  [-0.006,+0.024]       1-2
WR         514      5.369     5.382    +0.013  [-0.018,+0.044]       1-2
TE         245      3.541     3.559    +0.018  [+0.002,+0.040] *     0-3
START-WR   208      6.929     6.952    +0.022  [-0.008,+0.058]       1-2
```

Rejected. Nothing improves; whole-pool TE is significantly worse (allocator
knock-on - cutting rookie WR targets re-spreads the team budget onto the TE
room against a noisy Week-1 actual). The deeper reason is in the ledger: a
charted rank-2 rookie WR is a genuine **boom/bust** population. The flag
correctly zeroes the camp bodies (Malik Heath, Jalin Hyatt, Xavier Gipson,
all 0 pts) but also docked **2023 Puka Nacua** (5th-round rookie, charted a
backup, behind injured Kupp Week 1 -> 21.9 pts) and 2022 Christian Watson /
Romeo Doubs.

**Strength sweep** (`ROOKIE_BACKUP_WR_DAMPEN_STRENGTH` 0..1, two passes:
`--strengths 0.3,0.5,0.7,1.0` then a softer `--strengths 0.1,0.2,0.3`), with a
direct test of the "the docked guys mostly produce nothing" hypothesis:

```
strength   ALL      WR       TE     START-WR   START-TE   START-WR MEDIAN-AE
0.10     -0.002   -0.005   +0.002     -0.004     +0.003        +0.000
0.20     -0.002   -0.005   +0.002     +0.014     +0.005        +0.000
0.30     -0.001   -0.005   +0.005*    +0.014     +0.009        +0.000
0.50     +0.004   +0.007   +0.007*    +0.016      --           +0.000
1.00     +0.009   +0.013   +0.018*    +0.022      --           +0.000

what the dock hits:   n   mean act  median act  % under 5   Σ|err| base->var
strength 0.10         3     5.30       4.80        67%         9.1 -> 9.3
strength 0.20        11     6.05       4.80        55%        50.9 -> 48.3
strength 0.30        17     4.46       1.90        65%        70.6 -> 66.9
strength 0.50        20     4.09       0.95        65%        80.9 -> 79.5
strength 1.00        25     5.27       4.80        52%       114.7 -> 114.6
```

No strength passes, including the ultra-soft 0.10-0.20. Nothing improves
significantly in the right direction at any strength (WR is a flat -0.005,
never starred). TE turns significantly worse by strength 0.30 and stays there
(allocator knock-on - cutting rookie WR targets re-spreads the team budget
onto the TE room against a noisy Week-1 actual). The startable-WR **median**
error is +0.000 at every strength (no improvement even on the outlier-robust
metric the eye-test appeals to). And - the point of the "what the dock hits"
table - the population is **not** the near-zero group that premise assumes: at
strengths 0.2-0.5 the *median* docked WR scored ~1-5 pts and only 52-65%
landed under 5; at 0.10 the dock fires on only 3 player-games in four seasons
(median actual 4.8) and total abs error goes the wrong way (9.1 -> 9.3). A
charted rank-2 rookie WR who ends up with a real 3-5 target role is common,
not rare - so docking the whole population trades "too low on the zeros" for
"too low on the many who play". Kept as a switchable flag, OFF.

## 2026-09-08 — `v2_rookie_backup_wr_dampen_narrow` (the same dock, gated on a proven trio) — BUILT, UNTESTABLE ON AVAILABLE DATA

The narrower gate the section above flagged as "the shape that might survive":
fire the same 0.16 -> `ROOKIE_BACKUP_WR_SHARE` (0.06) share dock **only** when
the rookie's team already has >= `ROOKIE_BACKUP_WR_NARROW_MIN_AHEAD` (3)
*established* WRs (finite prior-season role share >=
`ROOKIE_BACKUP_WR_NARROW_ESTAB_SHARE` = 0.10) charted AHEAD of him. A team that
lost a starter no longer has three proven rank-1 WRs, so the gate structurally
cannot catch a WR1-injury inheritor (2023 Puka Nacua) or a rookie who won a
real camp role. Share dock only - no vacancy-weight zeroing
(`count_established_receivers_ahead` + a `& (_estab_ahead >= 3)` on the
`rookie_charted_backup` mask). On the live 2026 board it fires correctly:
Sturdivant 1.91 -> 0.07 targets (his freed share -> Bo Melton, a GB vet), plus
Zavion Thomas (CHI), Josh Cameron (JAX) and ~5 small rookies.

```
scripts/sweep_rookie_backup_wr_dampen_narrow.py, wk1 2022-2025, strengths 0.2/0.35/0.5
every scope, every strength:  dMAE = +0.000   (narrow gate caught NOBODY)
```

Not rejected - **untestable**. The `v2_historical_ourlads` archive is exactly
2022-2025 and weeks 2+ are not cold starts, so this is the entire available
cold-start backtest, and zero historical Week-1 cases match "rank-2 no-prior WR
behind a fully-established trio". Every rank-2 rookie the broad flag docked in
that window was behind an *incomplete* room - precisely the case this gate is
built to skip. So it demonstrably works on the 2026 board and provably never
fired on four years of historical openers; shipping it is low-risk but
formally unvalidated. Kept as a switchable flag, OFF, pending a decision to
override the backtest-gated convention for the 2026 opener.

## 2026-09-08 — `v2_td_career_regress` (regress a TD-light season to the player's OWN career rate) — BUILT, BACKTESTED (gated + ungated), REJECTED

The successor idea to `v2_td_volume_shrink`: instead of a league/position
anchor, regress cold-start WR/TE `receiving_tds` toward the player's OWN
opportunity-weighted CAREER TD-per-game rate (sum tds / sum games across up to
`TD_PRIOR_CREDIBILITY_SEASONS` = 4 looked-back years), with a pull that GROWS
with how much career he has:

```
season_factor = clip((role_seasons - 1) / (TD_CAREER_REGRESS_SEASONS_FULL - 1), 0, 1)
vol_factor    = career_targets / (career_targets + TD_CAREER_REGRESS_OPP_K)   # K = 200
pull          = TD_CAREER_REGRESS_STRENGTH * season_factor * vol_factor
out           = last_rate + pull * (career_rate - last_rate)
```

Two-sided; never fires under `TD_CAREER_REGRESS_MIN_ROLE_SEASONS` (2) role
seasons or a non-finite career rate; alternative to `v2_td_volume_shrink` on
this channel (`career_regressed_td_prior`, wired off `_prior_rate_pre_cred`).
Eye-test on the live 2026 board (strength 1.0) is exactly the ask - UP: Lamb
+0.157, Kupp +0.105, Kelce +0.102, Jefferson +0.100, Diggs +0.087, Andrews
+0.060; DOWN: G.Wilson -0.172, D.London -0.127, Adams -0.073, Pickens -0.079;
untouched where career == recent: Kittle 0.389->0.389, A.J. Brown ~flat.

```
scripts/sweep_td_career_regress.py, wk1 2022-2025, strengths 0.3/0.6/1.0
strength      ALL       WR       TE      START-WR   START-TE   recv_tds  recvTD-medn
0.30       -0.003*   +0.003   -0.023     +0.036*     -0.015     +0.004*     +0.019
0.60       -0.004    +0.001   -0.024     +0.034      +0.019     +0.004*     +0.017
1.00       -0.004    +0.002   -0.025     +0.035      -0.064     +0.002      +0.017

what it moves:  n    n up   n down   mean|d|   Sum base|err| -> var|err| (recv_tds, touched set)
0.30           151    75      76      0.052     55.5 -> 52.6
0.60           198    99      99      0.051     76.9 -> 74.2
1.00           242   119     123     0.058     91.6 -> 89.4
```

Fails the bar: **START-WR +0.034..+0.036, CI-excludes-0 at strength 0.30** - a
real regression on a decision pool. `receiving_tds` MAE on the startable pool
is also marginally worse (+0.004\*, and its MEDIAN +0.017..+0.019). The
favourable side is genuine but sub-significant: whole-pool TE -0.023..-0.025
at every strength (never starred, n too small on 4 slates), ALL -0.003\*/-0.004,
and the touched set's own recv_td error always drops (Sum|err| 55.5->52.6 etc).
Direction is honestly two-sided (75 up / 76 down at 0.30).

First hypothesis for the START-WR cost was the DOWN arm on ascending WRs
(G.Wilson 0.53->0.36, D.London 0.48->0.36) being dragged toward a career
average still weighted by their weaker early seasons.

**v2 - role-comparability gate added and RE-SWEPT (2026-09-08): did not help,
made it worse.** The gate (recent opp/g within [0.60, 1.67] x career opp/g,
mirroring `blend_comparable_td_priors`) removed ~1/4 of the moved set, mostly
DOWN moves (n_down 76 -> 63 at strength 0.30):

```
scripts/sweep_td_career_regress.py (gated), wk1 2022-2025
strength      ALL       WR       TE      START-WR   recv_tds
0.30       -0.001    +0.010*  -0.024     +0.036*     +0.004*
0.60       -0.001    +0.009*  -0.024     +0.034      +0.004*
1.00       -0.001    +0.011   -0.026*    +0.035      +0.002
```

START-WR is still +0.036\* at strength 0.30 (unchanged), and whole-pool WR
went from flat (+0.003, v1) to a SIGNIFICANT regression (+0.009\*..+0.011\*),
while ALL lost its small v1 improvement. So the START-WR cost is NOT the
down-regression of ascending WRs - it is the UP arm: rescuing a TD-light
startable star (Jefferson 0.25 -> 0.35) adds projected TDs that mostly meet a
Week-1 zero, the exact mechanism that killed `v2_td_volume_shrink`. The anchor
(league mean vs own career mean) does not matter; regressing startable pass-
catcher TD rates UPward loses MAE against a mostly-zero Week-1 actual every
time.

**Direction closed.** Four sweeps now - `v2_td_volume_shrink` (league anchor,
soft + full) and `v2_td_career_regress` (career anchor, gated + ungated), every
strength from 0.10 to 1.0 - and each one trips a CI-excludes-0 regression in a
decision pool (START-WR or START-TE or whole-WR). The only recurring favourable
signal is whole-pool TE ~-0.02..-0.10, which never once clears significance on
the 4-slate cold-start window. Both flags kept OFF and switchable; the machinery
(`career_regressed_td_prior`, `td_volume_shrunk_prior`, the per-season TD-count
context) stays in place for a future idea with a different mechanism.

## 2026-09-09 — `v2_pass_capacity_matchup_flex` (let a matchup tilt survive the pass-capacity reconciliation) — SHIPPED

`apply_pass_capacity_conservation` is the LAST step on the target channel. It
fits each team's RB and WR/TE targets to a real budget (`QB projected attempts
x league target/attempt`), and that budget is split into an RB slice
(`capacity x prior-season RB catcher share`, ~14%) and a WR/TE slice. Each
player's own +-22% forward matchup multiplier is already baked into `targets`
one step earlier - so when a matchup should tilt volume *toward* the RBs (a
checkdown-friendly coverage look), the model applies it, then this pass, whose
RB sub-budget never moved, reconciles it straight back out with a uniform
`budget/claim < 1` factor. The raw board over-claims team targets league-wide,
so that factor is nearly always < 1: matchup UP-tilts on RBs get shaved
systematically.

Two bounded relaxations were built behind the flag (both dials env-overridable,
`data/pass_capacity_allocator.py`):

- **(1) RB share band.** `rb_capacity = capacity x clip(model_rb_share, prior
  +- BAND)` where `model_rb_share = rb_claim / (rb_claim + wr_te_claim)`. The
  split follows this week's projected mix, clamped to `+-BAND` of the prior.
- **(2) Multiplicative factor deadband.** A group is also left unadjusted when
  its fit factor is within `[1/(1+M), 1+M]` - the whole room only ~M off
  budget.

```
scripts/sweep_pass_capacity_matchup_flex.py, in-season 2022-2024 wk4-14 (33 slates)
band  dead   ALL    START-RB   WR      RB tgt MAE  RB rec MAE  RB recYd MAE  Sigma tgt/att
0.03  0.00  -0.004  -0.062*   -0.001    -0.060*     -0.046*     -0.334*       .962->.962
0.05  0.00  -0.004  -0.071*   -0.001    -0.081*     -0.068*     -0.471*       .962->.962
0.08  0.00  -0.005  -0.060    +0.001    -0.108*     -0.091*     -0.609*       .962->.963
0.00  0.10  +0.003  +0.004    +0.002    +0.007      +0.003      +0.045        .962->.965
0.00  0.15  +0.016* +0.022*   +0.019*   +0.013      +0.007      +0.125*       .962->.979
```

(`*` = bootstrap 95% CI excludes 0. RB *MAE on the startable/pass-catching RB
pool. Sigma tgt/att = mean team catcher-targets / QB attempts - the
conservation guarantee.)

**Arm (2) REJECTED.** `M=0.10` is neutral-to-slightly-worse; `M=0.15` is a
significant regression across ALL / START-RB / WR / TE and pushes the
catcher-target ratio to 0.979 - it just reopens the team-target over-claim this
pass exists to close. Shipped default `M = 0.0`.

**Arm (1) SHIPPED at band 0.08.** RB startable targets / receptions /
receiving_yards MAE all significantly better at every band; START-RB fantasy
points significantly better at 0.03 and 0.05 (`-0.062*`, `-0.071*`), directionally
better at 0.08 (`-0.060`, CI just spans 0 - a mild overcorrection). No
significant regression on any decision pool at any band; `START-WR +0.019` at
0.08 is the only negative and is not significant. Conservation is untouched
(0.962 -> 0.963). The "RBs the flag lifted" table settles the mechanism: ~860
RB-slates carried a **-0.65-point SIGNED under-projection** at base, moving to
`-0.04` at band 0.08 - the allocator *was* washing out a real RB receiving
matchup edge. Shipped at 0.08 per user (the non-significant WR / START-WR
downtick is worth the larger, significant RB receiving-line accuracy;
`-0.108*/-0.091*/-0.609*` vs `-0.081*/-0.068*/-0.471*` at 0.05).

ALL-scope move is `-0.005`; `WEEKLY_CALIBRATION` re-fit deferred per the
established small-move precedent.

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
