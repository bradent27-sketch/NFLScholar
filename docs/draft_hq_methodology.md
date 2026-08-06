# Draft HQ — how every number is made

This is the reference for the Draft HQ tab: every function, where each stat
comes from, and how each computed number is derived. It's written to be
audited — if a number on the board looks wrong, you should be able to find
the paragraph that made it.

Two categories run through the whole document:

- **Sourced** — fetched from somewhere else and displayed as-is. FantasyPros
  ECR, FFA Value, FFA ADP, injury designations, news. These are never
  recomputed or adjusted.
- **Computed** — derived here from raw play data. Projected points, VORP,
  VONA, tiers, availability, SOS, the value-add model. Everything in this
  category has its derivation written out below.

---

## 1. External sources

| Source | What it provides | Endpoint / origin | Refresh |
|---|---|---|---|
| **DynastyProcess** `db_fpecr_latest.csv` | FantasyPros Expert Consensus Rank — `ecr`, `sd`, `best`, `worst`, position, team, bye, per draft format | `raw.githubusercontent.com/dynastyprocess/data/master/files/` | 6h cache; source refreshes daily |
| **DynastyProcess** `db_playerids.csv` | Cross-platform ID crosswalk (sleeper / espn / yahoo / fantasypros / gsis / pff / mfl) plus **birthdate** and draft position | same | 6h |
| **DynastyProcess** `values-players.csv` | Dynasty trade values (`value_1qb` / `value_2qb`) | same | 6h |
| **nflverse / nflreadpy** weekly stats | Every play-derived stat: carries, targets, receptions, yards, TDs, attempts, INTs, fumbles, FG/PAT, opponent, week | local history via `data.loaders.load_weekly_stats_history` | on load |
| **nflverse** injury report | Current designations | `nflreadpy`, falls back one season at a time | short TTL — a Saturday IR move should invalidate a board within the hour |
| **ESPN site API** | Player news headlines | public JSON, several host candidates tried in order | 6h |
| **Fantasy Football Advice** (your upload) | `ffaValue`, `adpComposite`, `proj`, Elo / PPR Elo / Dynasty Elo, analyst stat lines, scouting notes, age | **file you supply** — `external_data/ffa_players.json`, gitignored, never fetched | on upload |
| **Fantasy Football Calculator** | ADP from mock drafts on their own site | `fantasyfootballcalculator.com/api/v1/adp` | last-resort fallback only |

**FantasyPros ECR is fetched per draft format**, not reordered from one
board: `Redraft 1QB`, `Redraft Superflex / 2QB`, `Best Ball`, `Dynasty 1QB`,
`Dynasty Superflex`. A superflex board is not the 1QB board with QBs shifted
up — the whole positional value structure changes when a second QB slot
exists.

**ADP preference order** (`fetch_adp`): uploaded CSV → FFA import → Fantasy
Football Calculator → ECR-as-estimate. FFC sits last deliberately: its ADP
comes from mock drafts run on a free public site and skews casual — tight
ends and quarterbacks consistently slide later there than in real drafts.
When no ADP source answers at all, consensus rank stands in so the
availability model keeps working; the source label says `ECR estimate` so
you know it isn't real ADP.

**The FFA import is upload-only by design.** Nothing in this repo talks to
their servers, polls their API, or automates a login. It reads a file you
already have, and their *stat line* is imported rather than their point
total — a point total is only correct for the scoring it was computed
under, and theirs is half-PPR. Re-scoring their stat line through this
app's own engine keeps your league's settings honest.

---

## 2. The projection pipeline

`data/draft_projections.py` → `build_projected_board()`

### 2.1 Positional volume curves — `build_volume_curves`

For each position, `curves[pos][stat][r]` = the season total of that stat
for the player who **finished** rank *r*, averaged over the last 5 seasons.
Pre-2021 (16-game) seasons are scaled by 17/16 so they're comparable; the
`games` entry is *not* scaled, because scaling a games count would invent a
game nobody played.

Ranking is done on **PPR points regardless of your scoring settings**. The
curve describes the typical usage at each rung of a position's pecking
order, and that ordering shouldn't shuffle every time you toggle a setting.
Your real scoring is applied later, to the projected stat line.

### 2.2 Player rates — `build_player_rates`

Each player's recency-weighted **per-game** rate for every stat, weighted by
season: `[1.0, 0.55, 0.30, 0.15, 0.08]` — last season carries 48% of the
total weight. Per-game rather than per-season because a player who missed
half a year still tells you what his role was when he played. `games_sample`
travels alongside so downstream can tell a 40-game sample from a 3-game one.

### 2.3 The blend — `project_stat_lines`

For each player, for each stat:

```
projected = w · (own per-game rate × 17 × age_factor)  +  (1 − w) · curve_total[rank]
w = STAT_SELF_WEIGHT[stat] × evidence
evidence = min(1, games_sample / 24) × role_change_damping
```

`STAT_SELF_WEIGHT` is per-stat **stickiness**, and this is the core of the
model. Usage persists, efficiency regresses, touchdowns regress hardest:

| Stat | Self-weight |
|---|---|
| carries, targets, attempts | 0.70 |
| receptions | 0.65 |
| passing yards | 0.60 |
| rushing / receiving yards | 0.55 |
| passing TDs | 0.40 |
| rushing / receiving TDs | 0.30 |

A player with no history lands entirely on the rank curve — which is exactly
right for a rookie.

**Role-change damping.** A player's history is only evidence about next
season if he's doing the same job. When consensus ranks someone far above
anything his usage has ever supported, the market is saying the job changed
— a backup QB who won a starting role, a back whose committee partner left.
Detected by comparing his own per-game workload against what the curve says
is normal at his consensus rank; below a ratio of 0.6 the self-weight slides
smoothly toward zero.

### 2.4 The games basis — *changed in this audit*

**What it does now:** the own-history side is projected across a full 17-game
season; the curve side is left exactly as measured.

**Why.** Two games figures exist and only one is a durability measure. The
curve carries "games played by whoever *finished* rank r," which slopes hard
downward — QB1 16.6, QB20 14.0, QB28 10.6 — but that slope is mostly
selection, not health: missing games is one of the main ways a player ends up
ranked 28th. Multiplying a player's own rates by it charged every late pick
for injuries he hadn't had, and left **every QB ~30 points under consensus**
(Jared Goff 247 here vs 302 at FFA) purely from the games assumption.

Two alternatives were built and measured before settling:

| Variant | ECR rank-corr | median &#124;bias&#124; | Projection bias vs FFA |
|---|---|---|---|
| Rank-curve games (old) | 0.931 | 9.5 | QB −30, RB −2, WR −4, TE +2 |
| Rescale curve up to flat 17 | 0.905 | 11.0 | QB +28, RB +19, WR +15, TE +17 |
| Player's own games/season | 0.924 | 11.0 | QB −25, RB −2, WR −3, TE +2 |
| **Own rates × 17, curve untouched** | **0.937** | **9.0** | **QB −7, RB +2, WR +3, TE +8** |

Rescaling the curve up invents a player who plays a full season *and* still
finishes 28th, which flattens every positional curve. Using each player's own
games-per-season sounds right and isn't — at this sample size it mostly
measures role changes and rookie seasons rather than health (a rookie who
played 10 games as a backup and 17 as a starter reads as "fragile" at 13.5),
and it marked down exactly the players analysts had already cleared.

A full season is also what every published projection assumes, which keeps
these totals on the same scale as every number you can compare them to.

`proj_games` still exists and is the **blended** basis —
`evidence × 17 + (1 − evidence) × curve_games` — used to turn a season total
back into a weekly mean for the milestone-bonus model. Exact at both ends: a
pure rookie gets the curve's games, a fully-established player gets 17.

### 2.5 Aging — *new in this audit*

Age correlated **−0.48** with the board's disagreement with consensus
(−0.81 at TE, −0.61 at QB), and the bias-by-age-band was a clean ladder:

```
age 20-23  median bias  +8      (I rank them too low)
age 23-25                +5
age 25-27                +3
age 27-29                −4
age 29-32               −12.5   (I rank them too high)
```

This was the single largest systematic gap in the model, and the model had no
age term at all.

**Measured, not assumed.** Off 1,644 contributor seasons (≥8 games, ≥4 ppg)
from 2014 on, matched to birthdates from the DynastyProcess crosswalk. The
quantity is the median year-over-year ratio of points per game within each
age band. Only the *shape* across bands is used — the level is meaningless,
because requiring a productive season N selects high, so every band sits
below 1.0.

```
RB  22:0.98  24:0.84  26:0.85  28:0.73  30:0.68
WR  22:1.01  24:0.91  26:0.85  28:0.78  30:0.76  32:0.67
TE  22:0.98  24:1.07  26:0.79  28:0.83  30:0.80  32:0.90
QB  22:1.02  24:0.96  26:0.93  28:0.92  30:0.93  32:0.90  36:0.91
```

Backs fall off a cliff, receivers slide steadily, tight ends and quarterbacks
essentially don't — the well-known shape, which is a good sign the
measurement isn't an artifact. Resulting rates, per year past the position's
peak:

| Position | Peak age | Decline / yr |
|---|---|---|
| RB | 25 | 6.5% |
| WR | 26 | 3.6% |
| TE | 27 | 2.0% |
| QB | 27 | 1.0% |

Applied to the **own-history side only**. The rank-curve side is indexed by
consensus rank, and consensus has already priced the player's age —
adjusting both halves would charge a 33-year-old twice. The decline is
integrated over the 1.92-season gap between where a player's history actually
sits (the weight-centroid of the recency weights, plus one season forward)
and the season being projected, counting only the part of that interval past
the peak, so the curve is continuous at the peak instead of stepping down on a
birthday.

**The symmetric young-player boost was tested and does nothing.** The raw
bands do show young players gaining (~4.5%/yr at RB/WR/TE), but adding it
left rank agreement identical to three decimals on both references and made
projection MAE marginally worse (22.0 → 22.2). In hindsight that's expected:
young players are exactly the ones with thin history, so `evidence` is
already low and the consensus curve is already carrying them. The upside half
is left to the market, which prices it better.

Age is shown as a board column and comes from **birthdate**, not from a
published `age` field — a birthdate is a fact, an age is a fact with a
timestamp on it. Coverage is 92% of the board.

### 2.6 Scoring the line — `score_projected_lines` / `score_stats`

Every stat is pulled by explicit name, never by a prefix filter. That's
deliberate: `receptions` doesn't start with `receiving_`, so a
`startswith(('passing_','rushing_','receiving_'))` selection silently drops
it and every curve gets built with PPR contributing exactly zero — a failure
that computes cleanly, renders cleanly, and quietly prices WRs like it's a
standard league. (This bug was real; WR1 read 265 instead of 380.)

Fully parameterized: 4- or 6-point passing TDs, points per carry, TE premium,
negative points for fumbles, per-tier FG values, and **per-game yardage
bonuses**.

**Yardage bonuses.** Rushing/receiving at 100/150/200/250 and passing at
300/400/500/600, either cumulative or highest-only. These need per-*game*
yardage but a projection is a season total, so each player's weekly yardage
is modelled as a **gamma distribution** with his projected per-game mean and
a position-typical coefficient of variation (rushing 0.62, receiving 0.78,
passing 0.35 — rushing is steadiest because carries arrive whether or not the
offense is working). The thresholds are then integrated over it via the
regularized upper incomplete gamma function, implemented directly so there's
no scipy dependency.

The shape matters, not just the mean: two backs with equal season yardage
earn very different bonus totals if one is a metronome and the other
alternates 30 and 170. Applying thresholds to the season total instead would
award a 100-yard bonus once for a 1,400-yard season.

`Proj Pts` is **floored at zero** — a season projection is an expectation and
no draftable player has a negative one. Before this audit, 20 deep-bench
players sat below zero, where a rank curve's thin share of fumbles and
interceptions outweighed a near-zero share of the production.

### 2.7 Outcome range — `add_outcome_range_from_projections`

`Ceiling` and `Floor` are the 85th and 15th percentile outcomes.

Derived by smearing each player across a distribution of *finish ranks* — how
far players actually land from where they were ranked — and reading the
resulting point distribution. The rank uncertainty is **measured**
(`calibrate_rank_uncertainty`), not a constant, because the spread isn't
remotely uniform: top-6 QBs and TEs land within about 10 slots of where they
were, top-6 RBs and WRs scatter more than twice as far. A board that treats
those as equally certain overrates early RB/WR reliability and underrates
elite QB/TE stability.

Honest caveat: the ideal measurement would regress final finish against
*preseason consensus rank* across many years, but no free source publishes an
archive of historical preseason ECR. What's available locally is each
player's finish in year N−1 and year N, so this measures year-over-year
finish dispersion and shrinks it by `CONSENSUS_SKILL_SHRINK = 0.75` to
account for consensus being better-informed than pure persistence.

The smearing weights are made **doubly stochastic via Sinkhorn
normalization**, so probability mass is conserved in both directions and the
positional point total isn't inflated or deflated by the smear. Percentiles
use the row-normalized (unbalanced) weights, and `Ceiling` is clamped at
`max(ceiling, projected)` — without it, the balanced weights could push the
top player's ceiling below his own expectation.

`Upside` = Ceiling − Proj. `Bust` = Proj − Floor. `Risk` = (Ceiling − Floor)
/ Proj × 100 — a normalized read on how much of the pick is a coin flip.

---

## 3. Valuation

`data/draft_board.py` → `build_draft_board()`

### 3.1 Starter demand — `compute_starter_demand`

How many players at each position actually get **started** league-wide, by
simulating every team's starting lineup being filled greedily from the top of
the board — not by the usual shortcut of splitting each FLEX slot evenly
across RB/WR/TE.

That shortcut is wrong in a way that matters: flex slots don't go to
positions in equal shares, they go to whoever is better at the margin. In
PPR they skew heavily WR; in TE-premium, TEs start taking them; in superflex,
QBs take essentially all of them. Splitting evenly hardcodes an answer that
changes with exactly the settings this board is supposed to be sensitive to.

### 3.2 Replacement level and VORP

- **`VORP`** = Proj Pts − the points of the first player at that position who
  would *not* be started anywhere in the league.
- **`VOLS`** = Proj Pts − the last player filling a *dedicated* starting slot
  at his own position.

Both are shown because they disagree informatively. VORP is the honest
surplus in a vacuum; VOLS is closer to how drafts actually behave, since
drafters fill their own dedicated slots before they think about flex. High
VOLS with low VORP means the room will draft him earlier than he's worth.

**Streaming baseline.** At QB/K/DST the replacement bar is raised to what
streaming the waiver wire actually returns, measured from history — but only
ever *raised*. If the measured streaming value comes out below the last
rostered player (which happens in superflex, where the free pool really is
barren), the standard baseline is already the harder test and is kept.

One bug worth recording: the first version defined "rostered" by *final*
season totals, so the free pool was systematically the worst players — a
hindsight leak. Fixed to use prior-season finish. The corrected measurement
came in below the last starter, which correctly *rejected* the streaming
hypothesis for that setting rather than confirming it.

### 3.3 Tiers — `assign_tiers`

1-D k-means on projected **points**, per position, deterministically seeded
at evenly spaced quantiles (random init would make tiers shuffle between
reruns and the board feel broken).

Clustering on points rather than rank puts the breaks where value actually
drops instead of every N players. Only players **at or above replacement**
are clustered — the board carries ~82 QBs of whom maybe 20 are ever drafted,
and clustering all 82 spends the entire budget separating grades of worthless
backup. Everyone below replacement is swept into one trailing tier.

### 3.4 ADP, availability and VONA

- **`ADP`** — sourced (see §1). **`Expected Pick`** is the ADP the
  availability model actually uses.
- **`Value vs ADP`** = ADP − Board Rank. Positive means the market drafts him
  later than this board rates him.
- **`Avail Next %`** — P(still on the board at your next pick). Each player's
  draft slot is modelled as normally distributed around his ADP with the
  spread from `effective_adp_sd`; the probability he lasts to pick N is the
  normal survival function of `(N − expected) / sd`. Computed with
  `math.erfc`, no scipy.
- **`VONA`** (Value Over Next Available) = his value minus the **expected**
  best player still available at your next pick. That expectation walks the
  position's board in value order, weighting each player by the chance he is
  both available *and* the best one available (everyone above him gone).

VONA is the number that resolves draft-room paralysis. A stud RB at VONA 40
and a WR at VONA 4 is not a close call even if the WR grades a few points
higher in isolation — the WR (or someone within four points of him) is coming
back to you and the RB isn't. Positional runs, tier cliffs and scarcity all
fall out of this one calculation.

### 3.5 Market blend — `apply_market_blend`

VBD and the market disagree most sharply at quarterback: with replacement set
at the last starting QB, VBD prices an elite QB as a top-15 overall pick in a
1QB league, and real drafts take him twenty-plus picks later. That gap is
genuine disagreement, not an error in either direction.

This board doesn't pretend to settle it. **`VORP` is never blended** — it
stays exactly what the model says. The blend only moves the *order* the board
is presented in, in rank space (projected points and ADP have no common unit,
but their orderings are directly comparable). At 0 you draft the model; at
100 you draft ADP. It falls back to blending toward ECR when ADP is
unavailable, because the blend is the board's main defence against
out-thinking the entire analyst industry on the strength of one model, and
switching it off whenever a feed is down is exactly backwards.

### 3.6 K/DST demotion — `demote_late_round_positions`

Kickers and defenses sort behind every offensive player.

This exists because of a real measured failure. An earlier version put **26
of the top 120 board slots** on K/DST. Two hypotheses were tested and ruled
out with data — predictability (K year-over-year correlation 0.23 vs WR 0.16,
so kickers are *not* less predictable) and streaming value. The actual cause
was **availability blindness**: VORP is a correct measure of surplus and a
useless measure of *when to spend a pick*, because a kicker with positive
VORP will still be there 150 picks later. The published VBD literature says
the same thing — the opportunity cost of the pick exceeds the surplus.

A VOND (value over next-drafted) metric was also tried and **abandoned**: it
rewarded deep sleepers in sparse market regions, ranking Joe Mixon 13th
overall.

### 3.7 Strength of schedule — `data/draft_sos.py`

Fantasy points allowed per game by every defense to each position group,
from the most recent completed season, converted to a 0–100 percentile where
100 is the softest matchup. Mapped per position: RB → rushing defense, QB /
WR / TE → passing defense, DST → overall, K → none (kicker SOS is noise).

Week ranges are selectable, which is the point — a full-season SOS and a
fantasy-playoffs (weeks 15–17) SOS are different questions.

---

## 4. The Draft HQ tab

`ui/tabs/draft_hq.py`

### 4.1 Structure

| Function | What it does |
|---|---|
| `render()` | Entry point; Draft Room and News sub-tabs |
| `_league_settings_ui()` | Teams, roster slots, scoring, bonuses, draft type, your slot |
| `_board_cache_key` / `_cached_board` / `_load_board` | Board build + cache. Cache key covers every setting that changes a number, so toggling PPR rebuilds and toggling a display option doesn't |
| `_render_source_status` | Which feeds answered, how stale ECR is, what the ADP source actually measures |
| `_draft_context` | Unifies live-draft and mock state behind one interface, so every panel below works identically in either mode |
| `_commit_pick` | Records a pick, advances the clock, triggers bot picks in mock mode |
| `_render_draft_room` | The main surface |
| `_render_position_filter` | All / QB / RB / WR / TE / FLEX / K / DST as **buttons** |
| `_render_single_recommendation` | **One** pick recommendation, scoped by the active position filter |
| `_render_positional_value_add` | The "+X% to your team" model — see §4.2 |
| `_render_selectable_board` | The main table; row click opens the player |
| `_render_roster_slots` / `_fill_roster_slots` | Your roster filling by slot as you draft |
| `_render_draft_board_grid` | Teams × rounds grid, your picks highlighted |
| `_render_recent_picks_strip` | The last dozen picks with position colors |
| `_render_pick_odds` | What's likely to be gone by your next pick |
| `_render_strategy_panel` | Which archetype your roster is tracking |
| `_render_run_pressure` | Positional run detection |
| `_render_player_detail` | Inline detail + jump to full Player Search profile |
| `_render_live_sync` / `_apply_synced_picks` / `_add_names_as_picks` | Import picks from a live draft |
| `_render_mock_tools` | Sim controls, autopick, draft grade |
| `_render_news` | Headlines and injury designations, keyed to board players |

### 4.2 The positional value-add model

`data/draft_intel.py` → `positional_value_add`

For each position:

```
value_add(pos) = marginal_lineup_gain(best available now)
               − marginal_lineup_gain(expected best at your next pick)

Team %       = value_add / projected_full_lineup_points × 100
```

**Why a difference and not a level.** The best receiver available is always
worth a lot in absolute terms, so ranking positions by "how good is the best
one" just re-sorts by raw scoring and tells you to take a quarterback every
time. Subtracting what you'd get by waiting removes exactly that and leaves
the scarcity — a tight end who won't be there next round scores high, a
quarterback with eight comparable ones behind him scores near zero.

Marginal gain is measured against your **actual** lineup, so a position
you've already filled contributes only bench value and correctly falls to the
bottom. Both terms use the same lineup calculation, so the difference is
apples to apples.

Validated output:

```
EMPTY ROSTER, pick 5, next 20:   RB +100.6 (6.7%)  WR +92.9 (6.2%)
                                 TE  +20.4 (1.4%)  QB  +6.0 (0.4%)  K/DST 0.0
AFTER RB/WR/RB, next pick 53:    RB +127.6 (6.4%)  WR +109.4 (5.5%)
                                 TE  +63.1 (3.2%)  QB +45.5 (2.3%)
```

FFA show a comparable per-position score, built as
`strategyFreq + boostPct/100 + valuePct` where `boostPct` comes from
server-side "dynamic position multipliers" that aren't in their client
bundle and can't be recovered from it. This is a different route to the same
question, with the advantage of being computable from your own board and
explainable line by line.

### 4.3 Mock draft simulator

`data/draft_sim.py`

Bots draft off **Board Rank** with a softmax exploration term, not off
marginal lineup gain. Two bugs are behind that choice: drafting off marginal
gain made bots take a QB in round 1 (on an empty lineup, Josh Allen's ~150
point edge dominates), and a deterministic argmax made the pick-odds panel
read 100% on a single position.

Roster construction rules (`_legal_positions`): dedicated starting slots fill
before any backup, and the backup ladder counts **dedicated slots only**.
Counting a single FLEX slot as a full slot for RB, WR *and* TE separately was
what produced rosters with 3–4 TEs and a QB2 in round 3.

`grade_draft` scores the finished roster on optimal-lineup points;
`run_many_drafts` and `pick_slot_comparison` run the sim across seeds and
draft slots.

---

## 5. Where this board still disagrees with consensus

Measured against FantasyPros ECR and FFA Value, top 120 by each, after the
changes above.

**Agreement:**

```
vs FantasyPros ECR    rank-corr = 0.937   median |bias| =  9.0
vs FFA Value          rank-corr = 0.896   median |bias| = 10.5

Projected points vs FFA analyst projections (both half-PPR, n=336):
  overall  r = 0.918   MAE = 22.0
  QB   r=0.855  MAE=49.8  bias= −7.1
  RB   r=0.946  MAE=18.7  bias= +1.7
  WR   r=0.919  MAE=17.2  bias= +2.7
  TE   r=0.883  MAE=15.7  bias= +8.0
```

**Sanity checks — all passing:** Ceiling ≥ Proj for every player, Floor ≤
Proj, no negative projections, Avail Next within 0–100, no K/DST in the top
100, tiers monotone with projected points, every drafted position has a
replacement level.

**Remaining known disagreements, and why:**

- **Veteran tight ends** (Kelce ranked 68 here vs ECR 98; LaPorta 59 vs 85;
  Kittle 58 vs 106). The aging markdown the data supports is a few percent
  and moves them two or three spots. The market's markdown is much larger and
  reflects situation-specific analyst judgment — scheme change, target
  competition, an explicit "he's done" read — that a statistical model can't
  derive from box scores. The market blend is the intended lever here; raise
  it if you want the board to defer more.
- **Young ascending receivers** (Odunze, Egbuka, McConkey, Watson ranked
  ~20–27 spots lower here). Their thin history pulls them toward the rank
  curve; analysts price expected role growth. Tested a symmetric young-player
  age boost to close this and it changed nothing measurable (§2.5).
- **QB projection MAE stays high (49.8)** even though the bias is now near
  zero. That's not a residual bug — it's the backup-QB problem. FFA project
  every named starter for a full workload, so their QB20 gets 4,288 passing
  yards; this model projects him off his own thin history plus the rank
  curve. Both are defensible and they disagree by a lot on individual
  players while agreeing on average.

---

## 6. What was changed by this audit

1. **Games basis** — own-history rates now project across a full season
   instead of a rank-derived games figure. Fixed a ~30-point systematic QB
   under-projection. Two alternatives measured and rejected (§2.4).
2. **Aging curve** — new, measured per position off 1,644 contributor
   seasons. Age was the largest systematic gap in the model and there was no
   age term at all (§2.5).
3. **Negative projections floored at zero** — 20 players were below zero.
4. **FFA import no longer blanks board columns** for the ~440 players its
   ~260-player export doesn't cover. Previously an import silently deleted
   Age for half the board.
5. **Age surfaced as a board column**, derived from birthdate.

Nothing sourced was touched: FFA Value, FFA ADP, FFA projections, FantasyPros
ECR, dynasty values, injury designations and news are all displayed exactly
as fetched.
