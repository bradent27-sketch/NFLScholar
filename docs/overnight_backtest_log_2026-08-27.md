# Overnight backtest log — started 2026-08-27

Autonomous session, no user check-ins expected until morning. Every entry
below is a real backtest result or a real, verified finding — nothing here
is guessed or extrapolated. Nothing gets shipped to `DEFAULT_FEATURES`
without explicit sign-off; everything is a candidate flag in `MODEL_FEATURES`
until told otherwise, following this project's own established discipline.

Format per entry: **what was tested**, **result**, **read**.

---

## Carried over from earlier tonight (context, not new)

- `v2_role_change_by_stat` — SHIPPED to `DEFAULT_FEATURES` (real RB win,
  calibration re-fit to match).
- `v2_scheme_matchup` (man/zone, scheme-wins-outright) — real TE win
  (START-TE dMAE -0.052, directionally strong, borderline CI), WR wash.
  Not shipped.
- `v2_scheme_alignment_blend` (evidence-weighted ~50/50) — best result of
  any variant for WR (START-WR receiving_yards dMAE -0.297, CI excludes 0).
  For TE, real but weaker than scheme-alone (dilutes it). Not shipped.
- Two real bugs fixed and verified: negative-yards-caused gaps in the
  alignment defense-allowed table (`data/pff_alignment.py`), and a severe
  pre-existing Week 1 crash (`Categorical.fillna` in `_cold_start_pool`).
- `v2_cold_start_regression` built (25% pull-to-neutral at true cold start
  for the defense/context multipliers) — NOT YET BACKTESTED as of this log's
  start. High priority tonight given the user's Week 1 ask below.

## Dead placeholder flags (confirmed, zero gating code — free knowledge, no backtest needed)

`v2_channel_matchups`, `redzone_tds` (found earlier tonight), and now also
**`role_trend`** and **`volume_faced`** — all four are bare names in
`MODEL_FEATURES` from the original pre-V2 feature set with literally zero
`if 'flag' in feats` anywhere in the file. Confirmed via direct diff (zero
nonzero rows toggling each on) AND a grep for the gating string. Not bugs,
just never-built ideas sitting in the candidate list. If any of these turn
out to have real merit, they'd need to be built from scratch, not backtested
as-is.

## Week 1 investigation (user's explicit priority tonight)

### v2_cold_start_regression backtest — 5 seasons of Week 1 (2021-2025), the
full available sample (only 5 Week-1s exist; this is inherently thin)

Whole-pool: no effect anywhere (CI includes 0 for ALL/QB/RB/WR/TE).
START-QB at the default 0.25 strength: dMAE -0.227, 95% CI [-0.431, -0.019]
— excludes zero. Looked like a real effect size on its own.

**UPDATE after the strength sweep (0.10/0.40/0.60 vs the 0.25 default) - this
does NOT replicate as a stable trend, walking back the optimistic read
above.** START-QB dMAE by strength: 0.10 -> -0.468 (CI includes 0), 0.25 ->
-0.227 (CI excluded 0), 0.40 -> -0.108 (CI includes 0), 0.60 -> -0.779 (CI
includes 0, barely). That is NOT a monotonic dose-response - 0.40 produced
the SMALLEST effect and 0.60 the LARGEST, with 0.25 sitting in between and
being the only one that happened to clear significance. That pattern is
consistent with pure noise at n=5 weeks (a couple of players' idiosyncratic
Week 1 misses dominating a tiny sample), not a real, trustworthy effect.
WR showed a similar one-off: significant at 0.40 (dMAE -0.010, CI excludes
0) but not at 0.10 or 0.60. Whole-pool QB/RB/WR/TE show no effect at any
strength tested.

**Honest conclusion: `v2_cold_start_regression` does NOT have robust
evidence behind it from this Week 1 backtest.** The 5-week sample is
genuinely too thin to trust any single strength's result in isolation, and
this is exactly why the sweep was worth running before treating the first
number as a finding - it would have been a mistake to report the original
0.25 result as a clean win. Not recommending this for shipment based on
tonight's data. Note also that this mechanism structurally CANNOT fix the
QB1-selection 0.0-projection bug below (a multiplier applied to zero is
still zero), so even a real cold-start-regression effect wouldn't address
that specific failure mode.

### A real, separate, well-diagnosed bug found while reading the outlier
ledger from that run - NOT something v2_cold_start_regression can fix

Two Week 1 starting QBs projected at **exactly 0.0**: Jayden Daniels (2024,
WAS) and Dak Prescott (2021, DAL). Traced both: `QB1 Selection Required =
True`, `QB Projected Starter = False` for the team's ONLY QB in the pool -
the cold-start QB1 heuristic requires a minimum prior-season workload share
to confidently confirm an incumbent starter (`QB1_AUTO_INCUMBENT_MIN_SHARE`,
data/weekly_projections.py) before trusting him, and BOTH cases fail that
bar for different but related reasons:
  - Daniels: a true rookie, zero prior-NFL-season data to measure a share
    from at all.
  - Prescott: not a rookie, but missed 11 of 16 games in 2020 to a severe
    injury - his own healthy-game workload share that season doesn't clear
    the threshold either, even though he was unambiguously Dallas's real
    2021 Week 1 starter.

A 0.0 projection can't be partially fixed by ANY multiplier-dampening
mechanism (cold_start_regression included) - zero times anything is still
zero. This needs its own fix: either a smarter incumbent-confirmation rule
(e.g. a rookie who was clearly drafted/named the Week 1 starter, or an
established veteran whose LAST FULLY-HEALTHY season clears the bar even if
his most recent season doesn't), or leaning on the existing
`qb1_overrides.csv` manual-override mechanism (built for exactly this kind
of case, per data/draft_sources.py) - worth checking whether that file's
scope already covers backtest-relevant historical seasons or only the live
current season. LEAD FOR TOMORROW - a real fix here needs a design decision
about which of those two approaches (or both) makes sense, not a blind
guess tonight.

## Component/blend sweeps in progress

- TE fixed-weight blend sweep (0.6/0.7/0.75/0.8/0.9) -
  `scripts/sweep_scheme_blend_weight.py` - finding the right scheme:alignment
  ratio for TE (user wants to KEEP alignment context, just weighted less
  than the ~50/50 evidence-weighted default landed near).
- WR fixed-weight blend sweep (0.3/0.4/0.6/0.7), confirmatory - the
  evidence-weighted default already looked like the best variant tested for
  WR; checking whether a deliberate fixed ratio beats it further.

## Leads for tomorrow (real, scoped, but NOT attempted tonight - each needs
## a design call or touches high-blast-radius code I don't want to guess at
## without you able to review before it ships)

- **Red-zone player-side usage genuinely exists already, just not wired to
  the projection model or as-of-week-safe.** `data.transforms.
  build_redzone_usage` computes real per-player red-zone target/carry share
  and TDs from play-by-play - this is what would make `redzone_tds`
  (confirmed dead earlier tonight) buildable for real, not from scratch as
  I'd assumed. Two real gaps before it's usable in-season: (1) it pulls a
  FULL SEASON of pbp with no week filter - using it directly mid-season
  would leak future weeks into an earlier projection, so it needs an
  `as_of_week` param filtering `pbp['week'] < as_of_week` first; (2) it's
  currently only consumed by `data.draft_big_plays` (a draft-tool feature)
  and a display tab (`ui/tabs/defensive_yield.py`), never
  `weekly_projections.py`. TD modeling is central enough (every position,
  drives real fantasy points directly) that I'd rather you weigh in on HOW
  it plugs in (replace vs. blend with the existing two-year TD-rate prior)
  than guess overnight.

- **O-line PFF grades (`offense_blocking`/`offense_pass_blocking`/
  `offense_run_blockng`) are season-total only, no weekly archive** - same
  leakage problem as red-zone usage but with no fix available (there's no
  weekly file to filter down to). Usable as a Week-1-ONLY cold-start prior
  (last year's grade, adjusted for known personnel change) without leakage
  risk, which ties directly into tonight's Week 1 priority, but building a
  RB/QB matchup signal off it from scratch is real new work.

- **The QB1 cold-start selection bug** (Dak Prescott / Jayden Daniels, both
  projected exactly 0.0 in their real Week 1 starts) needs a real fix
  decision - loosen `QB1_AUTO_INCUMBENT_MIN_SHARE` for an injury-shortened
  prior season, add a 2-years-back fallback read (mirroring the prior2
  pattern already used elsewhere), or lean on the existing
  `qb1_overrides.csv` manual mechanism. Didn't want to touch QB1 selection
  logic - which affects every QB projection, not just Week 1 - without your
  eyes on it first.

## Operational note: confirmed OOM, switching to strictly sequential jobs

Running 5 heavy backtest jobs at once (each holding full-season DataFrames
across 34 weeks) oversubscribed this machine's memory - CONFIRMED via the
actual traceback on a re-run with stderr visible:
`numpy._core._exceptions._ArrayMemoryError: Unable to allocate 1.70 MiB for
an array with shape (19, 11699)` inside `build_weekly_projections` (a real
memory exhaustion, not a code bug - even a 1.7 MiB allocation failed). Three
of the five original jobs died this way (TE weight sweep, WR weight sweep,
game_env unbundling); DEFENSE_PRIOR_GAMES sweep may have survived (still
checking). Free memory recovered to ~4.85GB once the crashed processes
released theirs.

**Change for the rest of tonight: no more parallel heavy jobs. One at a
time, sequentially**, even though each still runs via run_in_background
individually (so a notification arrives on completion) - the fix is
capping concurrency at 1, not avoiding the background mechanism itself. All
three killed jobs will be re-queued once the current survivor finishes.

## New candidates built tonight, backtests queued next

- **`v2_game_total_elasticity`** / **`v2_venue_mult`** - unbundled
  `game_env` (rejected as a bundle: +0.012 MAE) into its two real
  components (implied-total scaling, indoor/outdoor venue). A bundle that
  fails as a whole can hide one real piece diluted by a non-working one -
  exactly what happened with the original (pre-redesign) alignment
  mechanism. Built, tests pass, not yet backtested standalone.

- **`DEFENSE_PRIOR_GAMES` sweep hook** (`v2_defense_prior_games_override`) -
  RESULT IN: this is the strongest, most credible finding of the whole
  night. Clean, MONOTONIC dose-response, not noise - unlike the cold-start
  strength sweep above, every tested value moves the same direction in
  sequence: 2.0 (+0.003 MAE, worse) -> 3.0 (+0.002, worse) -> [4.0 shipped
  baseline] -> 6.0 (-0.002, CI excludes 0, better) -> 8.0 (-0.004, CI
  excludes 0, better still - the best value tested so far). WR shows this
  MOST cleanly: every single value tested (2/3/6/8) reaches significance
  (CI excludes 0), flipping sign exactly at the shipped default - below 4.0
  hurts, above 4.0 helps, monotonically. TE: real at 8.0 (CI excludes 0),
  borderline at 6.0. QB/RB: no effect either direction (makes sense - this
  constant governs defense-matchup shrinkage, which mainly matters for
  pass-catchers). START-scopes mostly underpowered (smaller n) rather than
  contradicting the whole-pool trend.

  **8.0 (double the shipped 4.0) is still the best value tested, meaning
  the real optimum probably sits even higher - extending the sweep to
  10/12/16 next, now running solo per the concurrency fix above.** This
  looks like a genuine case where the original constant was simply left
  too conservative and nobody had gone back to test it since.

Also re-mined the enrichment dataset from earlier tonight (adot, carry_share,
target_earner_score) for anything missed - nothing new; confirms that
analysis was already reasonably thorough.

---
