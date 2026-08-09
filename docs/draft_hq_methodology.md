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
| **FantasyPros ADP** (live) | Consensus ADP averaged across NFFC, Sleeper, ESPN, CBS and RTSports | `fantasypros.com/nfl/adp/{format}.php` | 6h |
| **FantasyPros ADP** (local) | The same consensus, recovered offline from `rankings/fantasypros_2026_*.csv` as `RK + "ECR VS. ADP"` | your own periodic export | on file change |
| **Fantasy Football Calculator** | ADP from mock drafts on their own site | `fantasyfootballcalculator.com/api/v1/adp` | manual selection only |

**FantasyPros ECR is fetched per draft format**, not reordered from one
board: `Redraft 1QB`, `Redraft Superflex / 2QB`, `Best Ball`, `Dynasty 1QB`,
`Dynasty Superflex`. A superflex board is not the 1QB board with QBs shifted
up — the whole positional value structure changes when a second QB slot
exists.

**ADP preference order** (`fetch_adp`): uploaded CSV → FFA import →
FantasyPros ADP (live) → FantasyPros ADP (local export) → ECR-as-estimate.

Fantasy Football Calculator is **no longer in the automatic chain at all** —
it is selectable by hand and nothing else. Its ADP is measured from free mock
drafts run on its own site, and while those are real humans, they are people
mock-drafting for free on a calculator site. The board it produces drifts hard
from where players actually go, sliding tight ends and quarterbacks a full
round or more, and everything downstream (`Value vs ADP`, `Avail Next %`,
`VONA`, the entire opponent model) inherits that error while still looking
authoritative. A visibly missing column is better than a confidently wrong one.

FantasyPros' consensus replaces it because it averages the ADP published by
the platforms real leagues draft on, high-stakes money drafts included. It
publishes one consensus per *format* rather than per *league size*, which is
strictly less settings-sensitive than FFC was — a deliberate trade, since a
generic number measured from real drafts beats a size-specific one measured
from mocks. The UI names which is in use rather than hiding the distinction.

The **local** FantasyPros path deserves its own note: the ranking CSVs already
in `rankings/` carry an `ECR VS. ADP` column, which is exactly
(ADP − consensus rank), so ADP is recoverable as `RK + that difference` with
no network at all. Draft night is the worst possible moment to discover a site
is unreachable, and the gap between a board with real ADP and one without is
the entire value/availability half of it. This is the floor under that.

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

### 2.5b Aging, corrected — *the curve was doing nothing past the peak*

The age term above was measured correctly and then applied in a way that
threw the measurement away. Because the markdown is integrated only over the
interval a player's history has to be carried forward
(`HISTORY_LAG_SEASONS`), a **constant** rate per position makes the resulting
multiplier constant for everyone past the peak:

```
OLD  RB   age 27:0.883  29:0.883  31:0.883  33:0.883  35:0.883  37:0.883
```

A 37-year-old back and a 27-year-old back were marked down identically. The
curve stopped discriminating at exactly the age it should have started, and a
draft audit found the board buying 30+ veterans — Kamara, Hill, Diggs,
Hockenson, Kittle — rounds ahead of the market in every mock.

**The fix is age-dependent rates** (`AGE_DECLINE_BANDS`), integrated across
the same interval so the curve stays continuous at every boundary:

```
NEW  RB   age 26:0.937  28:0.883  30:0.794  32:0.694
     WR   age 26:1.000  28:0.933  30:0.841  32:0.779
     TE   age 28:0.980  30:0.962  32:0.900
     QB   age 30:0.981  34:0.981  36:0.926
```

Re-measured from 1,072 year-over-year pairs, reading each band against its
position's peak band:

| | 26–28 | 28–30 | 30–32 | 32+ |
|---|---|---|---|---|
| RB | 0.86 | 0.73 | 0.64 | — |
| WR | 0.84 | 0.76 | 0.72 | 0.81 |
| TE | 0.94 | 0.77 | 0.99 | 0.87 |
| QB | 0.93 | 0.97 | 0.92 | 0.89 |

The sample is survivorship-biased **upward** — it can only contain players
still playing six games in both seasons — so the true decline is steeper than
this, not gentler. Rates are still set slightly inside the measurement
because the old bands are thin (RB 30–32 is n=14).

**Tight ends deliberately get a gentle curve.** Their bands are non-monotone
(0.77 then 0.99) on small samples, so the data does not support a TE cliff,
and inventing one to make Kelce and Kittle look right would be fitting the
model to two players. `AGE_ADJUST_MIN` also dropped 0.75 → 0.45; at 0.75 it
would have re-flattened the new curve for a different reason.

A steeper band past 34 was built, measured and **rejected**: neutral on every
metric, and resting on 12 RB / 26 WR / 23 TE seasons.

### 2.5c Injury history — RB and TE only

A season cut short predicts both more missed games *and* worse per-game
production the following year:

```
played 15-17   n=570   next-year games 13.7   per-game retained 0.92
played 12-14   n=300   next-year games 13.1   per-game retained 0.88
played  8-11   n=174   next-year games 12.3   per-game retained 0.79
played   1-7   n= 28   next-year games 13.1   per-game retained 0.83
```

Both halves are applied (`INJURY_HISTORY_BANDS`), to the own-history side of
the blend only — the rank curve already averages over players who missed
games, so marking it down too would charge the same injury twice. Surfaced on
the board as **Health** ("11/17"), so a marked-down veteran explains himself.

**Scoped to RB and TE by measurement, not taste.** Applied to quarterbacks it
made them worse (MAE 50.6 → 53.2, bias −10.0 → −17.2): a QB who played eight
games usually lost his *job*, not his health, and the rank curve has already
priced that. Receivers were borderline (14.1 → 14.4) and are excluded on the
same logic — a part-season for a WR is very often a rookie easing in.

### 2.5d Workload — measured and deliberately NOT modelled

"A back with 300 carries breaks down next year" is the most repeated claim in
fantasy football. Tested against the same pairs, age held under 28, it is
false — and backwards:

| RB touches last year | n | next-yr games | per-game retained |
|---|---|---|---|
| <150 | 71 | 13.1 | 0.77 |
| 150–225 | 72 | 13.5 | 0.87 |
| 225–300 | 69 | 14.1 | **0.94** |
| 300+ | 33 | 13.5 | 0.87 |

Heavy usage predicts *more* games and *better* retention at every band up to
300, and even 300+ sits above the low-usage cell. Touches proxy job security;
the players with few of them are committee backs whose *roles* are fragile. A
workload penalty would mark down the workhorses and promote replacement-level
players. Not implemented, and documented here so it isn't re-added on
intuition.

### 2.5e What all of it did

Measured on identical code, only the parameters swapped:

| | before | age only | +injury (all) | **+injury RB/TE** |
|---|---|---|---|---|
| ECR rank-corr | 0.957 | 0.956 | 0.952 | **0.962** |
| FFA rank-corr | 0.932 | 0.943 | 0.942 | **0.944** |
| projection MAE | 20.8 | 20.2 | 20.2 | **19.8** |
| projection bias | +2.4 | +1.1 | −1.5 | **+0.3** |
| QB MAE | 50.4 | 50.6 | 53.2 | **50.6** |
| RB MAE | 18.1 | 17.4 | 16.4 | **16.4** |
| WR MAE | 15.1 | 14.1 | 14.4 | **14.1** |
| TE MAE | 13.8 | 13.2 | 12.3 | **12.3** |

The shipped variant is best or tied-best on every metric. On the flagged
players the board now *agrees with FFA* where it previously didn't — Kittle
rank 90 vs their 89, Diggs 121 vs 121.5, Hockenson 139 vs 133 — which means
the remaining gap to ADP on those names is the market being the outlier, not
this model. Two genuine residuals remain: **Kelce** (36.9, we project 147.6
vs FFA's 109.0) and **Tyreek Hill** (32.5, 121.7 vs 69.9), both cases where
the market has priced information no statistical model can see.

Hill's half of that residual was explained and fixed in the next pass — see
2.5f. He is a free agent, and the board wasn't pricing it.

### 2.5f Free agency — `free_agent_adjustment`

A player with no NFL team is not a discounted starter. He is a bet that
someone signs him, and then a second bet about what role he gets. The board
priced neither: it read last season's usage rates off local history exactly
as it would for a starter. Every unsigned player sat well ahead of the
market, in the same direction — a model gap, not an edge.

| player | board rank (before) | ADP | ECR |
|---|---|---|---|
| Tyreek Hill | 151 | 224 | 244 |
| Joe Mixon | 211 | 327 | 303 |
| Zach Ertz | 234 | 313 | 337 |
| Austin Ekeler | 257 | 350 | 347 |
| Kareem Hunt | 287 | 383 | 311 |

**Measured** on 1,682 pairs — players who were real contributors in season
N−1 (6+ games, 4+ ppg), split by whether they were on an active week-1
roster in season N:

| | n | played at all | games | per-game kept |
|---|---|---|---|---|
| on an active roster | 1,361 | 99.9% | 12.5 | 0.926 |
| **not on one** | 321 | **53.6%** | **3.6** | **0.364** |
| …of those, if he played | — | — | 6.7 | 0.679 |

Both failure modes are real and large. Nearly half never play a down; those
who do play about half a season at roughly three-quarters of their old rate.
So it takes the same two-part shape as the injury markdown — a games
multiplier and a per-game multiplier — and like that one it is applied only
to the own-history side of the blend, since a free agent's consensus rank
has already been marked down by the market.

**The shipped rates (0.62 per-game, 0.55 games) are deliberately milder than
the measurement (0.39 / 0.29)**, on reference class rather than timidity: the
measurement's cutoff is week 1, and this board is built in August. A player
still unsigned on the eve of the season has been passed over by all 32 teams
with the deadline in sight — a much worse signal than being unsigned in early
August with a month of camp injuries still to come.

Swept against both market reads (the overall metrics barely move either way,
because 34 of 733 players sit mostly outside the top 120):

| variant | proj MAE | proj bias | FA vs ADP | FA vs ECR |
|---|---|---|---|---|
| off (1.00/1.00) | 19.8 | +0.3 | −46.4 | −72.1 |
| mild (0.80/0.75) | 19.6 | −0.1 | −11.2 | −39.3 |
| **shipped (0.62/0.55)** | **19.6** | **−0.4** | **+26.6** | **−9.1** |
| hard (0.50/0.40) | 19.7 | −0.5 | +50.8 | +12.1 |
| measured (0.39/0.29) | 19.7 | −0.6 | +66.8 | +26.0 |

The shipped setting sits closest to zero on the average of the two market
reads and ties for best projection MAE. Hill lands at board rank 219 against
an ADP of 224. ECR rank-correlation (0.962) and FFA rank-correlation (0.944)
are unchanged to three decimals in every variant, which is the point: this
fixes 34 players without touching the other 699.

**Detection.** `Team == 'FA'`, straight off the same FantasyPros export the
ranks come from, so no name matching is involved. A cross-check against
`roster_weekly_2026.csv` was built and **rejected**: marking anyone absent
from the real NFL roster file as a free agent looks more authoritative, but
it flagged 109 players against FantasyPros' 34 and the extras were Patrick
Mahomes, James Cook and Travis Etienne. The crosswalk carries "Patrick
Mahomes II" and the roster file carries "Patrick Mahomes", so
`clean_name_exact` produces two different keys. A free-agent penalty that
occasionally lands on the best quarterback in football is far worse than one
that misses a fringe tight end.

Surfaced in the **Health** column as `4/17 · FA`.

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

**Streaming baseline.** At QB/**TE**/K/DST the replacement bar is raised to
what streaming the waiver wire actually returns, measured from history — but
only ever *raised*. If the measured streaming value comes out below the last
rostered player (which happens in superflex, where the free pool really is
barren), the standard baseline is already the harder test and is kept.

**Tight end was added to that list and it mattered.** The original reasoning
for excluding it — "leagues roster 60+ of them, so the free pool really is
replacement level" — is true of running backs and receivers and simply false
of tight ends. A 12-team league starting one TE rosters roughly fifteen,
against 32 NFL starters, so ~17 startable tight ends sit free all season.
That is quarterback's structure, not a receiver's.

Measured, streaming a tight end returns **167** points against a standard
TE13 baseline of **156**. The eleven points in between were the whole
problem: every tight end from about TE8 to TE14 carried positive VORP,
floated 30–40 picks above his ADP, and the board went on recommending
another one deep into the draft.

Median `Value vs ADP` by round, before and after (positive = the board
thinks he's a bargain; a healthy board is near zero across positions):

| ADP band | QB | RB | WR | TE **before** | TE **after** |
|---|---|---|---|---|---|
| R1–2 | — | −2 | 0 | +3 | −1 |
| R3–5 | +16 | −1 | −5 | +6 | +2 |
| R6–8 | +14 | −17 | −10 | **+16** | +10 |
| R9–12 | −21 | −48 | −30 | **+18** | **+1** |
| R13+ | −75 | −80 | −41 | −12 | −19 |

Settings-sensitive without special-casing, because the measurement takes the
rostered count as input: with a 0.5 TE premium the bar moves to **201**, and
in a 2-TE league more get rostered, the free pool thins, and tight ends
correctly regain their surplus.

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
- **`Avail Next %`** — P(still on the board at your next pick, **given that
  he is still on the board right now**). Each player's draft slot is modelled
  as normally distributed around his ADP with the spread from
  `effective_adp_sd`, and the reported number is the ratio of two survival
  probabilities: `sf((next − expected)/sd) / sf((now − expected)/sd)`.
  Computed with `math.erfc`, no scipy; the far tail falls back to the
  `phi(z)/z` approximation, which stays finite exactly where the direct
  division becomes 0/0.

  The conditioning is what makes the column usable rather than decorative.
  The unconditional form asks "what are the odds this player goes after pick
  N", and for anyone whose ADP is already past N that is ~100% no matter how
  much of the draft has happened — so ten rounds in, the whole remaining
  board read 100% and the column said nothing. What a drafter is actually
  asking is "given he is *still here* at pick 109, does he last to my pick at
  121", and every pick he survives is evidence about where he'll go.

  The reference pick is your first pick **strictly after** the one on the
  clock. Strictly, because the question is "if I pass on him now, does he
  come back" — the *at-or-after* reading compares the board against the very
  pick it is already conditioned on, which makes every player read 100%.
  Both draft modes recompute this per render through
  `refresh_pick_context`, against the pool that mode has actually thinned;
  the cached board only carries the pre-draft baseline.
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
slot_value(p) = max( marginal_lineup_gain(p),          # as a starter
                     contingency_value(p) )            # as depth

value_add(pos) = slot_value(best available now)
               − wait_weight × slot_value(expected best at your next pick)

Team %        = value_add / projected_full_lineup_points × 100
```

**Depth is priced, not ignored.** `marginal_lineup_gain` is exactly zero once
every starting slot is full, so the panel used to read 0% for all six
positions through the entire back half of a draft. That is a modelling gap,
not a display one — a bench receiver and a third tight end are worth
visibly different amounts.

`contingency_value` fills it, and is built as an **option**, because that is
what a late pick is. You are not stuck with a bust: you drop him and stream
the position, so the downside is bounded near zero while the upside is a
starter you didn't pay for. It is `E[max(X − waiver, 0)]` with X normal
around the projection, spread taken from the board's own Ceiling — which is
why high-variance late running backs price above safe low-variance backups.

That option only pays in weeks he'd actually be in your lineup, which is
`expected_start_share` (§3.2b). It is the whole answer to *should I take
another tight end*.

**The wait weight.** "The cost of waiting" presumes you can wait. With one
pick left there is no next turn, so the value of spending it is the *level*,
not the drop-off. `wait_weight = (picks_left − 1) / picks_left` — 0.9 at ten
picks out, so early-round behaviour is unchanged, and 0 on your last pick,
which is what makes the panel keep saying something useful late instead of
collapsing to six zeros. Worked example, 12-team, from a real mock:

| | R1 | R5 | R9 | R12 | R14 |
|---|---|---|---|---|---|
| top card | RB 8.7% | RB 2.3% | K 1.3% | K 2.3% | K 4.1% |
| TE | 1.8% | 1.0% | 0.0% | 0.1% | 0.1% |

TE goes to zero and stays there the moment a second one is rostered.

### 3.2b Roster depth — `expected_start_share`

How much of the season the *n*-th player you own at a position spends in
your starting lineup. Within the slots a position occupies he starts
whenever he's available; past them he is contingency, entering only when
enough players above him are out — a binomial tail on a measured absence
rate (share of a season a *draftable* player misses: QB 26%, RB 21%, WR 20%,
TE 22%, plus a bye).

The absence rates are near-identical across positions, so **slot count does
all the work** — and it is the whole reason a second tight end and a third
receiver are not the same pick:

| | you own 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| **TE** (1.0 slots) | 1.00 | 0.27 | 0.00 | 0.00 |
| **RB** (2.0) | 1.00 | 1.00 | 0.47 | 0.07 |
| **WR** (3.0) | 1.00 | 1.00 | 1.00 | 0.60 |

A third receiver walks into a flex slot nearly every week. A second tight
end plays about three weeks a year and a third plays none.

Bench depth at a **streamable** position is further cut to 20%: the week
your quarterback is out there are twenty free ones, so you never needed to
roster the cover. Without that, the recommendation panel named the same
backup QB in six straight rounds of a 1QB league.

Feeds `positional_value_add` and `recommend_picks`. In the latter, a
position you effectively cannot play another of is dropped from suggestions
outright rather than merely scaled — VONA measures what you lose by
*waiting*, and you lose nothing by waiting on a player who would never enter
your lineup. Positions with a real starting need are exempt, and asking for
one explicitly via the position buttons still shows its best available.

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

Bots draft off a **market ranking** with an exponential exploration term, not
off marginal lineup gain. Two bugs are behind the latter choice: drafting off
marginal gain made bots take a QB in round 1 (on an empty lineup, Josh Allen's
~150 point edge dominates), and a deterministic argmax made the pick-odds
panel read 100% on a single position.

**The market ranking is a blend, not a single column**
(`build_opponent_ranking`). Three orderings are available — ADP, consensus
rank, and an imported FFA rank — and the default is a straight 50/50 average
of the first two, exposed as three sliders in League Settings. Pure ADP
simulates a room that has collectively memorised last week's market and never
has an opinion; pure ECR simulates a room of analysts who all read the same
rankings and ignore what everyone else is doing. Real drafters do some of
both, and neither extreme reproduces how a real room actually behaves.

Two implementation details are load-bearing:

- The blend happens in **rank space**. ADP is a pick number, ECR is a rank
  and FFA Rank is a rank over a different (smaller) player set; averaging
  those raw would let whichever column has the widest numeric spread dominate
  for reasons unrelated to what it says. Ranking each over this board first
  puts all three on one scale, so a weight of 0.5 genuinely means half.
- Weights are renormalized **per player**, over whichever components have a
  value for him. Otherwise a partial source acts as a penalty: a 260-player
  FFA export would push the other 440 down the board purely for being absent
  from it, handing you a mock where every unlisted player falls into your lap.

Players no component priced sort behind everyone who was priced, ordered by
this board's own value — the right guess for "the room hasn't priced him".

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
vs FantasyPros ECR    rank-corr = 0.941   median |bias| =  8.0
                      by position: QB −4, RB −5, WR +9, TE −8
vs FFA Value          rank-corr = 0.901   median |bias| = 10.0

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

- **Veteran tight ends** — **George Kittle is the largest single deviation on
  the board**: ranked 60 here against ECR 106, on a projection of 162.8 vs
  FFA's 129.7. Kelce 69 vs 98 and LaPorta 62 vs 85 are the same shape,
  smaller. The median TE bias improved from −19 to −8 vs ECR when TE entered
  the streaming baseline, so the *systematic* part of this is largely fixed;
  what's left is per-player. The aging markdown the data supports is a few
  percent and moves them two or three spots. The market's markdown is much
  larger and reflects situation-specific analyst judgment — scheme change,
  target competition, an explicit "he's done" read — that a statistical model
  can't derive from box scores. The market blend is the intended lever here;
  raise it if you want the board to defer more.

  Kittle specifically will not show up in a "top-100 ECR" deviation scan,
  because his ECR is 106 — outside the window. Any audit of where this board
  disagrees with consensus has to look at players the MODEL rates highly too,
  not just ones the market does, or it will systematically miss exactly the
  cases where the model is most out on a limb.
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

## 5b. The betting market — what it's worth, measured

Three market sources feed this app, and they answer different questions.

| source | cost | granularity | key needed |
|---|---|---|---|
| The Odds API | paid, 1,000 req/month | per-game props | yes |
| Underdog / PrizePicks | free | **season-long** player props | no |
| **nflverse game lines** | **free, uncapped** | per-game spread + total | **no** |

The third is the one this section is about, because it is the only one that
is free, uncapped, needs no key, and is reachable from a locked-down network:
nflverse mirrors the closing spread and total for every NFL game into a CSV
on GitHub, back to 1999.

### The implied-points identity

nflverse's `spread_line` is positive when the HOME team is favored, so:

```
home implied = total/2 + spread/2      away implied = total/2 - spread/2
```

Verified on a live row — NE at SEA, total 44.5, spread 3.5, Seattle the home
favorite at −192: SEA 24.0, NE 20.5. Two checks have to pass (the pair sums
to the total, and the favorite is higher) and only the correct sign
convention passes the second.

### Scaling projections by implied team scoring — measured and REJECTED

The obvious use is to scale a player's projection by his offense's implied
scoring. It is a reasonable idea and it does not survive a backtest. Project
season N from season N−1 per-game production, with and without multiplying by
`(team implied points / league average) ** alpha`; 748 player-seasons across
2023–25, players with 8+ games in both years:

| alpha | MAE (weeks 1–3 lines) | r | MAE (full-season lines) | r |
|---|---|---|---|---|
| **0.00 (off)** | **2.830** | 0.7850 | **2.830** | 0.7850 |
| 0.25 | 2.839 | 0.7874 | 2.819 | 0.7915 |
| 0.50 | 2.870 | 0.7874 | 2.826 | 0.7955 |
| 1.00 | 3.010 | 0.7807 | 2.925 | 0.7967 |

The preseason-available signal — weeks 1–3, the only lines posted in August —
makes the projection **worse at every strength**. Full-season lines need
hindsight and buy 0.011 points per game, which is nothing.

By position at alpha 0.5 on the preseason signal, only running backs improved
(MAE 3.290 → 3.220, n=187); QB went 3.202 → 3.326, TE 2.051 → 2.109, WR 2.900
→ 2.976. One position in four on a modest sample is what noise looks like.

**Why it fails, most likely:** a player's output is his SHARE of an offense
times that offense's output, and share moves far more between seasons than
the team total does. Scaling by team while holding share fixed corrects the
smaller term and adds variance to the larger one — and it double-counts,
since a player's own usage history already encodes the offense he plays in.

So the implied total is surfaced as a **Vegas PPG** column and nothing in the
projection path multiplies by it.

### What the market's early read is worth on its own

Preseason lines are a real but modest signal about team scoring:

| | r | MAE |
|---|---|---|
| weeks 1–3 implied vs the market's own full-season implied | 0.75–0.85 | 1.2–1.5 pts/g |
| weeks 1–3 implied vs ACTUAL season points per game | 0.50–0.59 | 2.9–3.3 |
| full-season implied vs ACTUAL | 0.86–0.88 | 1.8–2.1 |

The market's *complete* read is an excellent predictor of scoring (r ≈ 0.87).
Its *August* read explains about a quarter of the variance. Worth showing,
not worth multiplying by.

### Cross-check: does this board agree with the market about offenses?

Summing every board player on a team is confounded — a team with more ranked
players sums higher regardless of quality — so the fair test is a fixed
top-N:

| basis | r vs Vegas implied PPG |
|---|---|
| sum of ALL a team's players | +0.525 *(confounded)* |
| sum of top 5 | **+0.723** |
| sum of top 8 | +0.727 |

Against the FFA analyst projections the board sits at r = 0.932, MAE 19.6,
bias −0.4 on 322 shared players. Both this model and FFA correlate with the
market's team view at about the same strength (0.44 vs 0.41 on the all-player
basis), which is the reassuring answer: three independent methods that agree
about offenses without agreeing by construction.

Largest team-level disagreements, top-5 basis, as a z-score gap:
market higher on GB (+1.21), SEA (+1.11), CAR (+0.89); this board higher on
ARI (−1.90), ATL (−1.58), CIN (−1.57).

## 5c. The games basis — measured against the betting market

The board sat systematically above two independent sportsbooks. Chasing that
found two defects, one structural and one empirical.

### PACE_GAMES was structurally impossible

This app scores **weeks 1–17**, and since 2021 a team plays 17 games across
**eighteen** weeks. So weeks 1–17 hold at most **sixteen** games for a normal
player. Confirmed against every season in local history — the most any player
recorded inside weeks 1–17 is 16 in 2021–24; the 17s in 2019 and 2025 are
mid-season trades collecting two teams' bye weeks. The basis was 17.0, which
inflated every stat line ~6% for nobody's benefit. **Now 16.0.**

(The season lengthened in **2021**, not later: 2019–20 max out at week 17,
2021+ at week 18. Every measurement below uses 2022–25 only so the two eras
are never averaged together.)

### Starters miss more games than the model assumed

`proj_games` averaged **16.81 of 17** for the top 60 — 98.9% availability.

Cohort: last season's top-24 at a position, kept only if they held a starting
role the following season, judged by per-game usage in the games they played
(QB 20+ attempts, RB 10+ touches, WR 5+ targets, TE 3.5+ targets). That
filter is the whole point — an injured starter keeps starter usage in the
games he *did* play, a demoted one doesn't, so it separates "hurt" from "lost
the job."

| pos | n | mean | 95% CI | median | shipped |
|---|---|---|---|---|---|
| QB | 85 | 12.99 | [12.24, 13.75] | 15.0 | **16** *(set by hand)* |
| RB | 87 | 13.85 | [13.20, 14.47] | 15.0 | **14** |
| WR | 95 | 13.56 | [12.94, 14.16] | 15.0 | **14** |
| TE | 74 | 13.51 | [12.88, 14.11] | 14.0 | **14** |

**The median is 15 and the mean is 13.5.** The typical starter misses one
game; a minority who lose half a season drag the average. Drafting off the
mean would be drafting out of fear of injury. The rounded-up mean sits
between the two.

**Quarterback is set by hand at 16 and the data argues against it** —
role-holding QBs measured 12.99, *lower* than backs and receivers. The
measurement stays contaminated for QBs in a way the usage filter cannot fix:
a QB benched in week 9 still shows 20+ attempts per game in the games he
started, so he reads as an available starter who missed eight. 16 means
"assume he plays." It leaves QBs slightly rich against skill positions —
visible as passing yards still summing to 1.29× the league total while every
skill stat sits at ~1.04. One constant to change.

### What it did

| | before | after |
|---|---|---|
| gap vs the books | **+5.4%** [+3.8, +7.0] | **−2.5%** [−3.9, −1.0] |
| …TE specifically | **+10.7%** | **+0.2%** [−2.0, +0.9] |
| board ÷ actual NFL receiving yards | 1.144 | **1.041** |
| board ÷ actual NFL receptions | 1.165 | **1.045** |
| ECR rank-corr | 0.962 | 0.960 |
| FFA rank-corr | 0.944 | **0.950** |
| FFA projection bias | +0.3 | **−8.4** |

Rank correlation is unchanged, so **draft order is preserved** — only the
point levels moved. TE's outlier gap closed entirely, which says TE was never
a TE-specific problem; it was the games assumption.

The board now sits **below** the FFA analyst projections by 8.4 points. That
is expected rather than alarming: analyst projections are near-"if healthy"
numbers carrying the same full-season assumption just removed here, while the
books price actual expected outcomes. Moving away from one and toward the
other is the intended result, and the two cannot both be matched.

### The tail is inflated and it does not matter

Beyond a position's rank ~96, projections run 1.5–6.5× what those ranks
actually produce, driven by the own-history side giving backups a starter's
games. It was measured for decision impact and has none: replacement level
sits at QB12 / RB30 / WR45 / TE6, all far inside that band. **Deleting the
entire tail changes VORP by 0.00 for every top-200 player** (max rank move 6,
and only because one player leaves the pool). Left alone deliberately —
changing it is risk without benefit.

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
