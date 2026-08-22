# Weekly Rankings projection audit

**Date:** 2026-08-20  
**Scope:** Read-only audit of the Weekly Rankings projection pipeline. No application code was changed. This document is the only artifact added by the audit.

## Bottom line

The model has a sensible core: it starts with player usage, shrinks early-season evidence toward a prior, explicitly penalizes thin snap roles, and limits several noisy multipliers. That is meaningfully better than treating a one-game box score as a new baseline.

It is **not yet safe to present as a fully context-aware Week 1 production model**. The largest concern is that its Week 1 role mechanism equates *last year's availability* with *this week's expected role*. That creates severe under-projections for active players who missed part of last season, while several live-data and cache paths can compound the error. The model also has material transition gaps once the season starts: fresh data may remain cached, newly relevant players can disappear from the pool, and the prior-season defense/pace context is replaced abruptly rather than blended out.

The first fixes should be data correctness and availability/role separation, not a new scoring formula.

## What I inspected

- Projection implementation: `data/weekly_projections.py`
- Data loading/freshness: `data/loaders.py` and `data/draft_sources.py`
- Weekly Rankings rendering: `ui/tabs/rankings.py`
- Backtest/calibration tools and methodology
- Supplied benchmark: `E:\FantasyPros_2026_Week_1_OP_Rankings (1).csv`

The supplied FantasyPros file contains **292 overall Week 1 ranks**, positional ranks, matchup labels, opportunity, and efficiency fields. It does **not** contain FantasyPros projected point totals, so I used it as a rank/context sanity check rather than pretending it is a points-level validation set. A normalized name-and-position join matched 276 of the 292 rows. Rank correlation is encouraging for RB (.733), moderate for WR (.639) and TE (.586), and weak for QB (.391); this is diagnostic only, not a points-accuracy backtest.

I also ran the existing focused projection-helper test suite: **34/34 passed**. Those tests are valuable for helper arithmetic, but they intentionally do not exercise the full live data/build path, so they do not catch the live Week 1 issues below.

## How the current projection is calculated

For each stat (targets, carries, yards, touchdowns, passing volume, etc.), the model does the following.

1. Build a current-season rate from pre-target-week games, recency-weighted and adjusted for historical opponent quality/rematches.
2. Create a prior rate from the immediately previous season's total divided by games played.
3. Blend the two rates:

   ```text
   current weight = games_this_season / (games_this_season + effective_K)
   blended rate   = current weight × current rate
                  + (1 − current weight) × prior rate
   ```

   At a neutral role confidence, a volume stat with `K = 3` is 25% current after one game, 40% after two, 50% after three, 57% after four, and 73% after eight. Touchdowns have a larger K and move more slowly.

4. With `role_volume` enabled, scale a prior rate for a changed snap role.
5. Apply per-stat matchup, game-script, pace, injury, and optional vacancy effects.
6. Score the projected stat line for Full/Half/Standard PPR, then apply a one-sided positional calibration to the final point total.

The raw-rate blend is gradual. The *final* result is less gradual because role, matchup, pace, injury, and schedule inputs can change independently.

### Week 1 path

With zero current games, the rate blend correctly becomes 100% prior-season rate. The model also falls back to last season's defensive matchup matrix and, if current pace is empty, prior-year pace. This gives an actual board rather than an empty page, which is good. Its critical Week 1 role calculation is:

```text
expected share       = player snap share across the whole prior team season
prior active share   = player average snap share in games he appeared in
role scale           = expected share / prior active share
prior rate used      = prior per-game rate × role scale
```

That is good at recognizing a true backup. It is incorrect when a player was a starter whenever healthy but missed games last year.

## Week 1 FantasyPros sanity check

I replayed the 2026 Week 1 cold-start core using the current 2026 roster, 2025 player history, and the supplied FantasyPros opponent map. The offline replay intentionally held injury, live pace, and market-game-script signals neutral so that the underlying workload logic could be inspected directly. Therefore its exact point totals are not a claim about the fully connected live app; the workload/rank findings are still directly attributable to the model code.

### Plausible core lines

| Player | Core model result | FantasyPros reference | Logical read |
|---|---:|---:|---|
| Puka Nacua | WR1, 21.38 PPR; 10.88 targets, 8.61 receptions, 121.95 yards | Overall #6, WR2 | Aggressive, but directionally defensible: his 2025 production was elite and his projection is not a made-up low-volume spike. Treat the ceiling as matchup-sensitive. |
| Ashton Jeanty | RB12, 13.57 PPR; 14.44 carries, 3.57 targets | Overall #25, RB6 | Reasonable workload shape relative to his 2025 15.65 carries and 4.29 targets per game. The model is more conservative than ECR, not structurally nonsensical. |
| Jaxson Dart | QB8, 15.41 PPR; 20.54 attempts and 4.82 carries | Overall #17, QB9 | A plausible middle/high QB line from the available history; this is a useful example where the core prior-based logic agrees broadly with the external rank. |

### Clear context failures or warning cases

| Player | Core model result | FantasyPros reference | Why it is concerning |
|---|---:|---:|---|
| Joe Burrow | QB27, 9.50 PPR; 15.85 attempts, 123 yards | Overall #4, QB1 | He was active on the 2026 roster but had only eight 2025 games. The cold-start role scale converts last year's missed games into an approximately half-sized Week 1 role. A healthy starting QB projection near 16 attempts is not credible. |
| Jayden Daniels | QB50, 5.70 PPR; 11.69 attempts, 66 yards | Overall #12, QB6 | Same availability-vs-role failure. Seven prior-season games depress the projected role as though he remains a part-time player. |
| Kyler Murray | QB66, 4.90 PPR; 8.89 attempts | Overall #24, QB15 | Five prior-season games drive a roughly 28% whole-season snap role even though the current roster labels him active. |
| Brock Bowers | TE9, 9.54 PPR; 4.89 targets | Overall #49, TE1 | His prior 12-game availability lowers a starter-quality target profile into a fringe-TE workload. |
| CeeDee Lamb | WR24, 9.43 PPR; 6.26 targets | Overall #35, WR7 | Prior missed time remains embedded in the opener workload despite an active current roster status. |
| Malik Willis | QB88, 2.50 PPR; 1.68 attempts | Overall #28, QB18 | The model treats prior backup usage as the current role. This is exactly the sort of offseason depth-chart change it cannot independently identify. |
| Kyle Juszczyk | RB28, 9.60 PPR; 7.89 carries and 2.87 targets | Not ranked in supplied ECR | His current roster position is RB while his prior stat position is FB. The prior lookup misses, so he receives an RB baseline rather than his own FB history. This is an invalid workload, not merely a disagreement. |

The ECR comparison is a triage tool, not a declaration that every ECR difference is wrong. The repeating pattern above is the important evidence: large misses cluster around health/availability, depth-chart, and position-label context—not around the basic arithmetic of established full-season roles.

### Seeded stratified spot-check across the board

To avoid looking only at headline disagreements, I drew one matched player from each of six overall-rank bands with a fixed seed. These lower- and middle-range lines give a more balanced read of the workload logic.

| Player (FantasyPros overall rank) | 2025 evidence | Core Week 1 model line | Audit read |
|---|---|---|---|
| Jordan Love (#33) | 15 games, 29.27 attempts and 225.4 passing yards/game | 19.66 attempts, 140.87 yards, 9.60 PPR | Too conservative for an expected starter; confirms that the availability/role mechanism can matter even below the most obvious injury-return cases. |
| TreVeyon Henderson (#99) | 17 games, 10.59 carries and 2.47 targets/game | 7.94 carries, 2.94 targets, 9.76 PPR | Conservative but coherent committee-back workload. This is the type of reasonable low/mid projection the model should keep. |
| Isaiah Likely (#121) | 12 games, 3.00 targets, 2.25 catches, 25.6 yards/game | 2.11 targets, 1.60 catches, 19.15 yards, 3.90 PPR | Coherent downward regression from a modest prior role. |
| Rashod Bateman (#195) | 13 games, 2.92 targets, 1.46 catches, 17.2 yards/game | 2.35 targets, 1.24 catches, 16.11 yards, 3.50 PPR | Sensible low-volume projection relative to history. |
| MarShawn Lloyd (#229) | No 2025 NFL player-week row | 4.35 carries, 0.83 targets, 4.00 PPR | A conservative generic baseline; this is appropriate only if it is visibly labelled as a no-history estimate. |
| Savion Williams (#281) | 12 games, 0.83 targets and 0.83 catches/game | 0.42 targets, **0.47 catches**, 1.10 PPR | The level is plausibly low, but catches exceed targets. That is a physical-stat-line violation caused by separately projecting dependent stats. |

The mixed sample supports a nuanced conclusion: low-volume regressions are often believable; the major concern is not that every conservative line is bad, but that current role/availability and dependent-stat constraints are not reliably represented.

## Confirmed issues to prioritize

### P0 — injury multiplier can be assigned to the wrong player

`_injury_multipliers` builds `dict(zip(injuries['Player'], mult.dropna()))` (`data/weekly_projections.py:1209-1211`). Dropping only the multiplier values compresses the values while retaining every player name, shifting all later statuses up the list whenever the feed includes an unmapped status.

I verified it with a synthetic three-row feed:

```text
Input:  Unmapped=Active, Out player=Out, Questionable player=Questionable
Actual: {'Unmapped': 0.0, 'Out player': 0.85}
```

That can zero the wrong player, discount another, and trigger an incorrect teammate-vacancy redistribution. Filter the player/status rows together before creating the map; add feed-status coverage tests.

### P0 — a new-season board can use stale prior-season injuries

`fetch_injury_report` deliberately falls back from the requested year to prior years when a current injury feed does not exist (`data/draft_sources.py:1272-1334`). It records the source season in `DataFrame.attrs['season']`. `_injury_multipliers` ignores that provenance and applies the resulting statuses to the requested season (`data/weekly_projections.py:1188-1211`).

For a 2026 Week 1 board, it can take a **2025 Week 1** status and apply it to a 2026 player with the same name. I reproduced this when the provider was available: the 2026 request returned 79 fallback 2025 rows, discounted CMC from 24.42 to 22.24 for an old Questionable designation, and zeroed several others. This is especially harmful because `Out` zeroes the player and feeds the vacancy feature.

There is a related display defect: any multiplier below .9 is labelled `Out/Doubtful` in the output, so an old Questionable CMC appears as `Out/Doubtful` even though he was only multiplied by .85. Do not apply an injury feed whose source season is not the projected season; preserve exact status/source/freshness when current information exists.

### P0 — doubtful-player vacancy redistribution creates extra team volume

The injury multiplier leaves a Doubtful player at 40% (`INJURY_MULTIPLIER['doubtful'] = 0.4`), but vacancy logic considers every multiplier `<= 0.5` sidelined and redistributes 75% of that player's **full** pre-injury workload (`data/weekly_projections.py:1250-1295`).

Without a cap binding, that projects 40% of the player's original workload for him **plus** 75% for teammates: 115% of the original volume. Vacancy should redistribute only the lost portion (`full − discounted`) or apply only to fully unavailable players.

### P0 — Week 1 incorrectly treats missed games as a reduced current role

The cold-start path uses whole-team-season prior snap share as the expected current share (`data/weekly_projections.py:1493-1506`), then divides it by the player's share when active and scales every prior stat (`data/weekly_projections.py:1711-1722`). This is a valid backup detector but not a valid availability forecast.

The Burrow, Daniels, Murray, Bowers, and Lamb examples above are direct evidence. Add a separate availability/starting-status estimate rather than embedding last year's missed games in workload. At minimum, an active current roster status should prevent this role scale from lowering a historically starter-level player solely because of absence.

### P1 — independently projected counting stats can violate football constraints

The shipping model projects targets, receptions, and receiving touchdowns separately. In the offline Week 1 replay, 12 players had projected receptions greater than targets; examples include Savion Williams (0.47 receptions / 0.42 targets), Ian Thomas (0.83 / 0.71), and Adam Trautman (1.25 / 1.21). Devin Culp had 0.08 receiving touchdowns on 0.05 receptions.

Some differences are rounding-sized, but the broader issue is real: an expected number of catches cannot exceed expected targets, and receiving touchdowns cannot exceed catches. The disabled `volume_efficiency` experiment shows this dependency has already been considered. Preserve the demonstrated calibration gains while adding a constrained reconciliation step or opportunity-first stat-line layer so display lines remain physically valid.

### P0 — cached results can prevent new in-season data from reaching the board

`load_year_data`, `load_team_pace`, `load_schedule`, and `build_weekly_projections` use `@st.cache_data` without a TTL/version input (`data/loaders.py:177, 1052, 1091`; `data/weekly_projections.py:1366`). The Weekly Rankings tab calls the same projection arguments on each rerun and does not clear these caches (`ui/tabs/rankings.py:376-379`).

A new weekly CSV, roster refresh, or newly available network data can remain invisible for the life of the Streamlit process. Use source file modification times/data versions in cache keys, TTL live fetches, and provide a visible `Refresh model data` action.

### P1 — historical team labels are overwritten by a later roster snapshot

The year loader drops overlapping raw-stat fields, including `team`, then joins a latest roster snapshot per player (`data/loaders.py:347-401`). This can label every historical row for a player with his later team. In the 2025 data, 608 regular-season rows across 136 players differed from their raw game team; 180 rows covered 38 QB/RB/WR/TE players. Examples include Adam Thielen's raw MIN rows labelled PIT, Joe Flacco's CLE rows labelled CIN, and Rashid Shaheed's NO rows labelled SEA.

Opponent rates remain tied to the raw opponent field, but game-script team/week joins and team snap-share denominators can use the wrong game or no game. Preserve raw game team separately from current roster team, and use player IDs/current team only for the *upcoming* matchup.

## High-priority Week 1/context improvements

### Current roster status is not used to form the cold-start pool

`_cold_start_pool` selects only name/team/position (`data/weekly_projections.py:1319-1354`). It does not filter roster status. The current roster contains 20 non-ACT QB/RB/WR/TE entries, including retired, cut, exempt, and reserve players. The core replay gave positive points to several of them, e.g. reserve QBs Drew Allar (5.5), Fernando Mendoza (5.2), and Ty Simpson (5.2), plus retired Adam Thielen (2.4).

Use a canonical roster eligibility field; emit an explicit excluded-player count rather than silently projecting unavailable players.

### New team/offseason depth-chart context is not modeled

At Week 1, player rates and role shares come from last season. The current roster mostly supplies player/team/position and the upcoming opponent. New play-callers, QBs, target competition, free-agent moves, promotions, demotions, and rookie depth charts are not represented. This is a limitation, not an arithmetic bug—but it explains ECR disagreements such as Malik Willis and Jauan Jennings.

The next model layer should be a transparent Week 1 role prior: official depth chart/start designation, current roster status, recent preseason/coach signal if available, and a conservative rookie/new-team prior. It should adjust **opportunity**, not force an arbitrary points bump.

### Prior reliability is not modeled beyond “last year only”

The prior is only `year - 1`, using total divided by games (`data/weekly_projections.py:1430-1434, 1676-1679`). A one/two-game cameo can be trusted as a complete prior, while a veteran absent all last season falls to a position baseline even if multi-year history exists. In the current Week 1 pool, 376 of 915 skill rows had no exact 2025 history key; 81 of those have at least two years of experience. Keep a multi-year, games/opportunities-weighted prior with age/role decay and distinguish `no evidence` from `thin evidence`.

### Position label changes can create a false positional baseline

Kyle Juszczyk demonstrates this directly: current RB versus prior FB means no personal prior row is found inside the current position loop. The fallback produces an RB-level workload. Use stable player IDs and a role-family mapping (e.g. FB/RB) rather than name + exact historical position only.

## Does the model develop gracefully after Week 1?

**Partially, at the rate-blend level.** Current evidence is gated to `week < as_of_week`, so target-week box-score leakage is avoided for the main player rate. The blend gradually gives current-season evidence more weight, and recent snap shares respond faster than a full-season average.

**No, not consistently at the whole-projection level.** The following transition behavior needs work.

### A current roster player with no current stat row disappears after Week 1

Once any season game exists, `cold_start` is false globally and the pool is built from `_season_totals(hist)` only (`data/weekly_projections.py:1511-1522`). A returning IR player, newly signed/traded player, promoted backup, or player who logged snaps but no box-score stat can be omitted entirely—even if he has a prior-season history and is active on the current roster.

Build the pool from the current eligible roster every week, then left-join current stats and prior history. A player with zero current stats should get an explicit low-confidence prior-based projection, not disappear.

### A trade can retain the old team/opponent until a stat is recorded

The loader removes overlapping roster fields before its merge, including team (`data/loaders.py:381-383`), and in-season totals use the most recent stats-side team. A player who has moved but has not yet logged a stat can retain his former team, wrong opponent, bye, pace, and matchup. Use current roster team identity for the projection target, with a player ID join.

### Defense and pace change abruptly from prior year to tiny current samples

Week 1 uses prior-season defense and pace. Week 2 switches to current-season defense/pace rather than blending them. At Week 2, a defense's one-game player-baseline ratios are effectively neutral; at Weeks 3–4, the same unshrunk rating can move each stat by the full 0.75–1.30 matchup range. Pace similarly becomes a one-game current-season number.

Blend prior and current team context with an evidence weight, and report the effective defense/pace sample. This will make W1 → W2 → W3 behavior continuous instead of a regime switch.

### Schedule failure collapses the full board instead of falling back neutrally

If `load_schedule` returns empty, `_week_opponents` is empty and every player is dropped for missing `Opponent` (`data/weekly_projections.py:1175-1185, 1540-1542`). The app then reports no projectable players rather than showing a neutral-opponent board with a source-health warning. Preserve a usable baseline board when schedule context is unavailable.

## Calibration, backtest, and presentation findings

### The displayed point total does not always equal the displayed stat line

Calibration is intentionally applied only after the stat line is scored (`data/weekly_projections.py:1844-1857`). The UI displays the calibrated `Model Proj Pts` next to the uncalibrated stat line. This is mathematically intentional, but it is not self-explanatory: a user cannot sum the shown stats and reproduce the shown points.

Either show `Raw Stat-Line Pts` and `Calibration Adjustment`, or rescale the display line transparently. Do not imply the stat line alone is the point calculation.

### One calibration table is applied to Full, Half, and Standard PPR

The UI supports all three scoring formats, but `WEEKLY_CALIBRATION` is one positional table (`data/weekly_projections.py:285-290`) and is applied regardless of scoring mode. The fitting script defaults to Full PPR and the repository contains no separate saved coefficient set by format (`scripts/fit_weekly_calibration.py:53-64`). Because receptions have different point values, the same final-points regression is not logically portable without validation.

Fit and store format-specific calibration parameters, or disable the calibration outside its validated format until those fits exist.

### The published validation does not cover the highest-risk period

The primary evaluation defaults to Weeks 5–17, and the methodology's headline evaluation is Weeks 5–17. That excludes Week 1 and the W1→W2→W4 handoff. It also admits that pace uses a full-season, not as-of-week, cut in historical runs (`docs/weekly_projections_methodology.md`, Known limitations).

Add a rolling, walk-forward evaluation explicitly reporting Weeks 1–4 by position and by player archetype: returning starter, new starter, rookie, new team, backup, and injury return. For each, compare against a simple prior-season baseline and an external weekly rank benchmark.

### PFF route rate can be stale or leak future information into a historical test

`_role_confidence` blends recent snap share with PFF `route_rate`, but the receiving summary is season-level and has no `as_of_week` filter (`data/weekly_projections.py:925-955`). With no current-year PFF export, the loader deliberately falls back to the prior year (`data/loaders.py:869-898`). The 2025 Puka Nacua summary, for example, lists 19 games, demonstrating that it contains full-season/postseason data.

This affects K/shrinkage rather than an outright workload multiplier, but it breaks the claimed strict as-of-week discipline in historical tests and can make early current-season role confidence stale. Gate PFF input by availability date and target week, retain source-year metadata, or omit it from historical backtests.

### Hidden context makes the board hard to audit in the app

The model returns `cold_start`, `Games This Season`, `Role Confidence`, features, and vacancy-adjustment metadata, but Weekly Rankings only uses metadata when the board is empty and omits Games/Role Confidence from display (`ui/tabs/rankings.py:376-379, 521-526`).

For Weeks 1–4, show a compact banner and per-player explanation: data source year, current/prior blend weight, prior-role scale, matchup/pace/script/injury multipliers, calibration delta, and source freshness. This is more useful than an opaque point total.

## Lower-priority curiosities worth resolving

- `MODEL_FEATURES` lists `redzone_tds`, `role_trend`, and `volume_faced`, but they have no implementation references beyond the declaration. The evaluator accepts them as variants, producing a no-op experiment. Remove or mark them as planned.
- Pace and personalized game script are active but are not individually represented in the feature registry, so the A/B harness cannot isolate them. Game script also turns on only after a player has sufficient history, creating a step change.
- The methodology still describes the old `0.6 × trailing 4 + 0.4 × season-to-date` current rate in one section, while code uses recency/matchup/rematch weighting. Update prose so the model can be audited accurately.
- `RECENCY_DECAY` is described as per game, but implementation uses calendar week distance. A bye/injury absence therefore adds decay even without a player game. This may be intentional, but should be named/tested as calendar-time decay.
- Overall defense matchup ratings have no explicit evidence shrinkage toward 1.0; early samples are protected mainly by clipping. Add a sample-size-aware prior.

## Recommended improvement order

1. **Fix correctness blockers:** paired injury filtering, current-year injury provenance guard, and partial-injury vacancy conservation.
2. **Make data fresh:** version/TTL caches, visible refresh, source timestamps, and source-health warnings.
3. **Repair player identity/pool handling:** player IDs, current roster team/status, FB/RB role family, eligible roster as every-week base pool.
4. **Redesign Week 1 role logic:** separate availability, starter probability, and role share; do not use prior missed games as current workload by default.
5. **Smooth early-season context:** evidence-weighted prior/current defense and pace; retain neutral fallback rather than dropping boards.
6. **Make the result auditable:** cold-start banner, inputs/multipliers/calibration explanation, and format-specific calibration.
7. **Validate the actual risk window:** walk-forward W1–W4 backtests with no future PFF/pace leakage and separate archetype reporting.

## Overall assessment

For established, healthy, same-team players with a full prior season, the workload and output logic is generally believable and often agrees directionally with the supplied FantasyPros Week 1 rankings. The model's most useful current insight is role-aware regression: it prevents a small-sample backup from being projected like a starter.

The same mechanism currently overreaches in Week 1 by using last-season availability as a proxy for current role. Until that is separated and the live freshness/injury issues are corrected, I would treat the Week 1 board as a **historical prior with useful stat-line detail**, not as an autonomous start/sit ranking. As the season develops, the rate blend is designed to improve, but the cache, player-pool, and early-context transition issues must be fixed for it to do so reliably.
