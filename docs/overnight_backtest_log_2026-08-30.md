# Overnight session log — started 2026-08-30 ~00:15

Autonomous session. User signed off for the night with these instructions:

> run the ablation, fix the team capacity, and re-run the recalibration. Once
> you are done feel free to try to run more backtests for potentially new model
> parameters, or dig into the data better. Maybe see if the amount we are
> considering strong/weak defenses is accurate (too strong or weak?), look into
> game script sliders for different stats, we have an empty vegas environment
> multiplier ... check if whatever the hypothesis there is could be useful in
> projections, further check for games where the starter was injured ... back
> test if we considered that player out (while we don't have previous injury
> data we can in hindsight set a player as out) and see if your injury/vacancy
> distribution works, look into other parameters that we have in our data (pff
> grades ... several values in those data sets to try?).

Discipline (unchanged): nothing ships to `DEFAULT_FEATURES` without explicit
sign-off; candidates are flags in `MODEL_FEATURES` or documented findings;
**one heavy backtest at a time** (the box OOMs otherwise).

Format per entry: **what**, **result**, **read**.

---

## 0. Housekeeping done first

- **Team capacity fix (per the user's mid-session correction).** The 2026-08-29
  "anchor waterfall" (protect top-2 WR/TE, ranks 3+ absorb the whole over-budget
  amount) was **reverted**. An over-budget WR/TE room is now scaled by ONE
  uniform proportional factor — every player takes the same % move, symmetric
  with the upward-scale case. The `±1` deadband and symmetric upscaling stay.
  `v2_pass_capacity_anchor` flag removed. `data/pass_capacity_allocator.py`
  `_fit_group` simplified to: deadband → leave alone; else → `factor =
  budget/claim` applied to all. 397 tests green.
- **`DEFENSE_PRIOR_GAMES = 12.0`** (was 4.0) — from the 2026-08-29 sweep
  (plateau knee; see `docs/overnight_backtest_log_2026-08-27.md`).
- **Ablation matrix relaunched** (12 flags, 2025 wk 4–17) against the fixed
  stack. `.sweeps/ablation_matrix_2025.txt`.
- **Calibration:** stale. Fresh refit queued for right after the ablation, then
  paste the half-strength-damped result.

---

## 1. Ablation matrix — DONE (~8h, 2025 wk 4-17, post-revert stack)

Ablate = remove the flag. **+dMAE = removing it HURTS = the flag helps.**
`.sweeps/ablation_matrix_2025.txt`, outliers `.sweeps/ablation_outliers_2025.csv`.

| flag | ALL | RB | WR | TE | verdict |
|---|---|---|---|---|---|
| **role_volume** | **+0.129** | +0.152 | +0.175 | +0.078 | huge; the single most important flag. base ALL MAE 4.384 → 4.513 without it |
| **v2_defense_prior** | **+0.033** | +0.031 | +0.038 | +0.030 | solid keeper — confirms DPG=12 nets positive; TE won only 1/13 weeks without it |
| **v2_pass_capacity** | **+0.046** | +0.003 | **+0.105** | +0.006 | real WR win even after the uniform-dock revert (CI [+0.023,+0.181]); 5-9 weeks though — a few big weeks |
| v2_role_change_by_stat | +0.001 | +0.003 | 0 | 0 | detectable but practically nil (RB-only, as designed) |
| v2_qb_volume_blend | +0.008 | −0.003 | +0.013 | +0.011 | marginal, every CI includes 0 |
| v2_preseason_rb_allocator | +0.003 | +0.010 | 0 | 0 | mostly inert mid-season (preseason tool); CI includes 0 |
| v2_td_two_year_prior | +0.003 | +0.004 | +0.002 | +0.006 | no measurable effect this window (TD variance) |
| v2_continuous_roles | +0.002 | +0.012 | −0.003 | +0.001 | no clear effect; RB borderline |
| v2_adaptive_volume | +0.001 | 0 | +0.001 | +0.001 | **dead weight in this window** |
| **v2_pff_alignment_matchup** | **−0.004** | 0 | 0 | **−0.019** | slightly BETTER without it (esp. TE, CI [−0.042,+0.002]) — **removal candidate**, wants its own backtest |
| v2_vacancy | 0.000 | 0.000 | 0.000 | 0.000 | **untestable here** — never fires (historical backtest has no injuries; as-of guard disables it). Use validate_injury_vacancy.py |
| v2_fantasypros_availability | 0.000 | 0.000 | 0.000 | 0.000 | **untestable here** — no FantasyPros injury feed for past weeks |

**Reads:**
- Three flags carry essentially all the value: `role_volume` (massive),
  `v2_defense_prior`, `v2_pass_capacity` (WR).
- `v2_pff_alignment_matchup` showed a mild net negative on ALL and TE.
  **USER DECISION 2026-08-30: keep it** — "not bad enough to get rid of and it
  gives some nice context." No change.
- The "nothing here" cluster (`v2_adaptive_volume`, `v2_td_two_year_prior`,
  `v2_continuous_roles`, `v2_qb_volume_blend`, `v2_preseason_rb_allocator`,
  `v2_role_change_by_stat`) mostly have situational rationale (RB-only,
  preseason-only, TD-only) a whole-pool mid-season MAE can't see — not prune
  candidates, just not moving this needle.
- Outlier ledger is almost entirely **booms** (Puka Nacua 15.4→46.5, Brock
  Bowers 11.3→43.3): the model's big misses in 2025 are under-projections of
  ceiling games, not over-projections — consistent with the calibration's
  one-sided top-shrink being the right shape.

---

## 2. Recalibration — QUEUED (runs right after the ablation)

`scripts/fit_weekly_calibration.py --years 2021,2022,2023`, then half-strength
damp (`b_slope = 0.5 + 0.5*slope`, `b_int = 0.5*intercept`) and paste into
`WEEKLY_CALIBRATION`, updating the comment. This is against the FINAL stack
(DPG=12 + deadband + symmetric upscale + uniform over-budget dock).

---

## 3f. Buried-veteran dock — BUG FOUND + FIXED (2026-08-30)

User checked the live 2026 board: Marquise Brown / Troy Franklin still at full
workload. Traced it: **Ourlads ranks WR/TE per ALIGNMENT SLOT** (LWR / RWR /
SWR each numbered 1,2,3...). Brown is charted RWR-2, Franklin SWR-2 — slot rank
**2**, i.e. "primary backup". The dock (written 2026-08-29) checked
`_chart_rank >= 4` as if the number were team-wide WR depth, so it never fired
for an actual backup. Name bridge (`Hollywood Brown` → `Marquise Brown`) and
the `_proven` guard both worked; only the threshold was wrong.

**Fix:** slot rank 2 → keep 50%; rank 3+ → hard cap. Logic extracted to a pure
`apply_buried_veteran_dock(...)` helper. Live 2026 wk1 after: Brown 0.471→0.236
snap / 4.15→2.33 tgt; Franklin 0.562→0.281 / 4.81→2.86 — now both below
Lemon/Wicks. Starters (Smith/Sutton/Waddle) untouched.

**TE runs one slot deeper (user request 2026-08-30):** a charted TE-2 is very
often the RECEIVING tight end while TE-1 is the inline blocker (Mike Gesicki
listed CIN TE-2 but out-targets TE-1). So for TE the dock only fires at slot
rank **3+** (`RECEIVER_BURIED_VET_BACKUP_SLOT_RANK_TE = 3`): TE-2 untouched,
TE-3 → 50%, TE-4+ → hard cap. Verified: Gesicki stays at 3.23 tgt; Conklin /
Njoku / Musgrave / Ruckert (all charted TE-3) docked. 3 unit tests, 400 pass.
**Watch:** Ourlads has a probably-stale TE order for LAC (Njoku charted TE-3)
and GB (Musgrave TE-3) — the dock acts on it; fades over 4 weeks.

## 3c-result. Game-env isolation (`--add`, 2025 wk 4-17) — DONE

`--add` mode: variant = DEFAULT + flag; **−dMAE = adding it helps.**

| flag | ALL | QB | RB | WR | TE |
|---|---|---|---|---|---|
| **v2_game_total_elasticity** | −0.011 | **−0.044** | −0.019 | −0.008 | +0.012 |
| v2_venue_mult | +0.003 | +0.004 | +0.001 | +0.004 | +0.001 |

- **`v2_game_total_elasticity`**: consistent directional lean toward helping —
  ALL won 10 of 14 weeks, QB −0.044 (won 10-4), RB −0.019 (won 11-3, sign-test
  p=0.06). **No scope clears the bootstrap CI on one 14-week season**, but the
  QB/RB signal is where the elasticity is highest (QB 0.42) and it's coherent.
  Verdict: **promising, underpowered — re-run on 2024+2025 (34 wk) before any
  ship decision.**
- **`v2_venue_mult`**: nothing anywhere (all +0.001..+0.004, every CI includes
  0, ~7-7 weeks). Effect sizes are ±2-3% and wash out. **Drop it.**

## 3g. Game-total elasticity POWER sweep — QUEUED (user request 2026-08-30)

`scripts/sweep_game_total_elasticity.py --k 0.5,1.0,1.5,2.0,3.0 --years 2025
--weeks 4-17`. Scales `GAME_TOTAL_ELASTICITY` {QB .42, RB .17, WR .14, TE .30}
by k and reports per-position dMAE (flag OFF vs flag ON at k·shipped) with
paired CI, widening `GAME_TOTAL_CLIP` in step with k. Answers "how strong
should the exponent be, per position." NOTE: the game total itself is already
Vegas — `game_environment()` reads `total_line`/`spread_line` off the schedule
feed, which is posted for future weeks; team-implied = total/2 ± spread/2. This
sweep is only about the exponent strength, not the source. Follow-up: 2-season
confirm at the best k.

## 3h. FINAL calibration re-run — QUEUED AT THE END (user request 2026-08-30)

After the exploration items settle, re-run `scripts/fit_weekly_calibration.py
--years 2021,2022,2023`, damp, and paste — so `WEEKLY_CALIBRATION` describes the
final shipped feature set (DPG=12 + uniform dock + whatever exploration flags,
if any, get greenlit).

**Caveat the user flagged:** the buried-veteran dock may amplify the TOP of the
WR distribution (docking backups shrinks the team's WR/TE claim, so the uniform
capacity factor rises toward 1.0 and the WR1/WR2 keep more of their raw
projection). A straight 2021-2023 refit will NOT capture this — the dock only
fires in cold start / weeks 1-4 with an Ourlads chart, and there are no Ourlads
snapshots pre-2026. So the final refit covers DPG/dock-method changes but the
cold-start top-WR amplification needs a separate check: eyeball the 2026 wk1
top-WR projections against FantasyPros / market lines, and consider a Week-1-only
calibration cut if it looks hot.

## 3i / 3j. Two user case-study fixes — BUILT 2026-08-30 (behind flags, tested)

**Case 1 — `v2_rb_snap_anchored_volume`** (`data/rb_role_allocator.py`). Travis
Etienne (charted NO RB1, cross-team) was landing near Alvin Kamara (charted
RB2) on carries/targets and *below his own snap share*, because the carry/target
split blended `snap_fraction` with a raw prior-season per-GAME rate — which
bakes in the player's OLD team's backfield split (Etienne's diminished 2025
JAX committee role) and hands the aging incumbent his old lead-back per-game
volume. Fix: the split now starts from the depth-aware SNAP allocation and
applies only a **bounded per-snap usage tilt** (carries/targets per snap vs the
backfield's RB rate; carry tilt 0.85–1.20, target tilt 0.70–1.60), faded toward
snap-proportional for cross-team / thin-history players, plus the same
depth-rank discount the snap split already uses. Widened the
`rb_carry_rate_scale` clip to (0.15, 2.75) on this path. Unit test:
`test_snap_anchored_volume_keeps_a_team_changed_lead_back_ahead_of_the_incumbent`.

**Case 2 — `v2_td_prior_credibility`** (`data/weekly_projections.py`,
`credibility_shrunk_td_prior`). A cold-start TD-rate projection is 100% the
prior season's rate with no regression of a THIN one-season sample toward the
mean — so RJ Harvey's ~120-carry rookie half-season hot TD rate carries the
same weight as Derrick Henry's genuinely sticky multi-year rate. Fix: shrink
the prior TD rate toward the position mean by opportunity volume (beta-binomial
credibility, K = 220 carries / 90 targets / 340 pass att), with:
- 1 role season → credibility capped at 0.60 (hot small sample mostly regressed);
- 2 role seasons → uncapped (2024+2025 of a real role is already sticky — **not**
  penalised for not doing it a third year ago);
- 3–4 role seasons → small extra multiplier (+2.5 %/season, max +5 %) — Henry-style
  longevity bump.
Looks back up to `TD_PRIOR_CREDIBILITY_SEASONS = 4` prior years (loads year-3 /
year-4 season totals, gated + cold-start only). Smoke: Harvey 0.055→~0.039
(cred 0.35), Henry 0.045→~0.045 (cred 0.83, ×1.05), 2-yr solid 0.040→~0.037.
Unit test: `test_td_prior_credibility_regresses_a_thin_one_season_rate_but_spares_a_veteran`.

Both **off by default**. 404 tests pass. Live-board sanity checks (Etienne/Kamara,
Harvey/Henry) pending a free job slot. **Backtests queued below, ahead of the
final calibration** (per the user: calibration stays dead last until model
changes stop).

## 3k. Vacancy table display trims — DONE 2026-08-30 (UI only, no model change)

Deep Dive → "Role, audit & data sources" → "How vacated volume is redistributed"
([`_render_vacancy_redistribution`](../ui/tabs/rankings.py)). Per explicit
request:
- An OUT player who was vacating **< 1.5 targets/game** (or **< 3 carries/game**)
  collapses to a single caption — no recipient distribution table. Exception:
  kept in full if THIS player is one of the fill-ins. QB pass attempts have no
  bar (a missing QB is never minor).
- Recipients **below the trusted tier** are rolled into one remainder line
  ("N more fill-ins below the trusted tier absorbed +X.XX combined") instead of
  cluttering the grid. Tier = `PASS_CAPACITY_TRUSTED_TIER` (8) for a receiver's
  vacated targets, `PASS_CAPACITY_TRUSTED_TIER_RB` (2) for carries and an RB's
  vacated targets. THIS player's own row is always kept regardless of rank.
- Decision logic is a pure helper `_summarize_vacancy_entry` (unit-tested,
  `tests/test_vacancy_table_display.py`). The two ledger producers
  (`redistribute_v2_vacated_usage`, `redistribute_rb_vacancy_with_allocator`)
  now attach a display-only `team_rank` to each recipient dict — nothing
  downstream reads it, no projected value changes. Scope was UI-only; the
  `validate_injury_vacancy.py` diagnostic keeps its full unfiltered report.

414 tests pass.

## 4. Backtest queue — LIVE STATUS (sequential, one heavy job at a time)

1. ~~GTE power sweep~~ — **DONE** (§3g / `.sweeps/gte_power_2025.txt`). Per-position:
   RB wants ~1.5–2× more elasticity, TE wants less, QB/WR ~optimal. → per-position
   tune, confirmed on 2 seasons at step 4.
2. ~~MATCHUP_CLIP scan~~ — **DONE**, shipped (0.82, 1.22) (§3a-result).
3. ~~SCRIPT_CLIP scan~~ — **DONE 2026-08-30, NO CHANGE** (§3b-result). Clamp is
   dormant; leave (0.85, 1.15).
4. **Injury/vacancy mechanism check** (3d) — `python scripts/validate_injury_vacancy.py
   --year 2025 --weeks 5-17`. Retro-marks in-hindsight absences OUT, runs the
   shipped redistribution, reports vacated/allocated/unfilled + whether the
   recipients' projections moved toward their real box score.
5. **GTE 2-season confirm** — per-position elasticity (RB 0.17→~0.28, TE 0.30→~0.22,
   QB/WR keep), `GAME_TOTAL_CLIP` widened in step, 2024 + 2025.
6. **Case 2 backtest** — `python scripts/backtest_component.py --add v2_td_prior_credibility --years 2025 --weeks 4-17`.
7. **Case 1 backtest** — `python scripts/backtest_component.py --add v2_rb_snap_anchored_volume --years 2025 --weeks 4-17`.
8. **FINAL calibration re-run** — `python scripts/fit_weekly_calibration.py --years 2021,2022,2023`,
   half-strength damp, paste into `WEEKLY_CALIBRATION`. STAYS LAST — only once no
   more model changes.

> **PAUSE POINT 2026-08-30:** after step 3, stopping to commit + push this version
> so the user can pull it on a second machine and continue there. Steps 4–8 resume
> after that.

3e (PFF `yprr` as a WR/TE efficiency input) is documented in §3e but NOT built —
it needs a new candidate flag + weekly-archive plumbing, better done with the
user in the loop.

---

## 3. Exploration backlog (user's list) — prep notes below, backtests sequential

### 3a-result. MATCHUP_CLIP scan (2025 wk 4-17, absolute MAE, no CI) — DONE

| clip | ALL | QB | RB | WR | TE | START-RB |
|---|---|---|---|---|---|---|
| (0.75, 1.30) shipped | 4.3843 | 7.1248 | 4.3666 | 4.1385 | 3.5634 | 6.3452 |
| (0.82, 1.30) | 4.3840 | 7.1254 | 4.3672 | 4.1376 | 3.5628 | 6.3482 |
| (0.88, 1.30) | 4.3828 | 7.1275 | 4.3684 | 4.1354 | 3.5588 | 6.3746 |
| **(0.82, 1.22)** | **4.3811** | 7.1257 | **4.3602** | 4.1379 | **3.5569** | **6.3298** |

**Read:** total spread on ALL is 0.003 MAE (~0.07 %) — inside single-season noise.
- Raising just the LOW floor (0.75→0.88): WR/TE improve a hair, RB/QB worsen a
  hair. **Net wash — the user's "0.75 floor is too aggressive" hypothesis does
  NOT pan out on its own.**
- **Narrowing the whole band to ~(0.82, 1.22)** is marginally the best config
  tested — best ALL / RB / TE / START-RB, WR flat, nothing hurt. Trusting the
  defense-matchup signal a little less at BOTH extremes. Effect is small
  (~0.1 % MAE, RB-driven).
- **Verdict:** (0.82, 1.22) is a defensible narrow-the-band tweak. Effect is
  small and this had no paired CI, but it's low-risk and nothing regressed.
  **USER DECISION 2026-08-30: implement it.** `MATCHUP_CLIP = (0.82, 1.22)`
  shipped; the queued final calibration re-run will pick it up.

### 3b-result. SCRIPT_CLIP scan (2025 wk 4-17, absolute MAE, no CI) — DONE, NO CHANGE

`.sweeps/scriptclip_scan_2025.txt`. Swept (0.85,1.15) shipped / (0.80,1.20) /
(0.75,1.25) / (0.70,1.30) by monkeypatching `wp.SCRIPT_CLIP`.

**Read: every metric is byte-identical to 4 decimals across all four widths.**
ALL MAE 4.3811, RB 4.3602, WR 4.1379, TE 3.5569 — unchanged whether the game-
script multiplier is clamped at ±15 % or ±30 %. The clamp is **dormant**: the
underlying `projected / season_avg` game-script read never reaches ±15 % on real
2025 data, so loosening the bound is a pure no-op. (Monkeypatch mechanism is the
same one `sweep_matchup_clip.py` used to produce *varying* numbers, so this is a
real finding, not a broken patch.)

**Verdict:** leave `SCRIPT_CLIP = (0.85, 1.15)`. If blowout rushing is under-
served (the sweep's hypothesis), the lever is **upstream** — how
`_vectorized_game_script_multiplier` builds the per-stat curve — not the clamp.
A per-stat script elasticity (rushing more script-elastic than receiving) would
need its own flag + `backtest_component.py`; not pursued now.

### 3a. Strong/weak defense scaling — `MATCHUP_CLIP = (0.75, 1.30)`
The forward-looking defense matchup multiplier is clamped to **[0.75, 1.30]** —
asymmetric (−25% / +30%). `_role_adjusted_multiplier` /
`_continuous_role_adjusted_multiplier` / `_efficiency_matchup` all clip to this.
The user's earlier NO-defense-vs-RB note ("13th toughest yet every multiplier at
the 0.75 floor") points at the LOWER bound being too aggressive. **Plan:** sweep
`MATCHUP_CLIP` by monkeypatching `wp.MATCHUP_CLIP` (plain module tuple, looked up
at call time — no flag needed for a sweep) across e.g. (0.75,1.30) baseline vs
(0.80,1.25), (0.82,1.22), (0.85,1.20), (0.85,1.15). Same harness as the DPG sweep.

### 3b. Game-script — `SCRIPT_CLIP = (0.85, 1.15)` global, per-stat curve
`_vectorized_game_script_multiplier(..., stat)` already builds a **per-player,
per-stat** empirical curve (that player's own history bucketed by game margin,
`SCRIPT_BUCKETS` mids ±3.75/±12.5), read at the market-implied margin, then
clipped to a SINGLE global `SCRIPT_CLIP = ±15%`. `SCRIPT_ELIGIBLE_STATS` =
{targets, receptions, receiving_yards, rushing_attempts, rushing_yards} (TDs and
QB stats excluded). **Hypothesis to test:** rushing volume is more script-elastic
than receiving (a workhorse in a blowout genuinely gets +25–30% carries), so a
per-stat clip — wider for rushing_attempts/rushing_yards, tighter for receiving —
should help. Sweep by monkeypatching a per-stat clip dict.

### 3c. Vegas environment — READY TO BACKTEST, no build needed
`game_environment()` reads `total_line`/`spread_line`/`roof` from the schedule
feed, which **IS posted for future weeks** (unlike wind/temp). `_game_env_multiplier`
applies `GAME_TOTAL_ELASTICITY` {QB .42, TE .30, RB .17, WR .14} and `VENUE_MULT`
(indoor/outdoor, normalised to 1.0). Both fully wired behind flags
`v2_game_total_elasticity` / `v2_venue_mult`, NEITHER in `DEFAULT_FEATURES` — so
the `environment_multiplier` in every projection is currently a hardcoded 1.0
("empty"). The bundle (`game_env`) was rejected (+0.012 MAE) but the two pieces
were never isolated. **Plan:** `backtest_component.py --add v2_game_total_elasticity`
and `--add v2_venue_mult` (and both together). Elasticities measured on 2019-2023
(out of sample).

### 3d. Injury/vacancy validation — needs a script (building now)
No historical injury feed, but nflverse `weekly_snap_pct` / inactive status lets
us find, for a past week, a player who WAS a heavy-snap starter in the weeks
before and then played 0 snaps that week (real in-hindsight absence). Retro-set
him OUT via an availability override, rebuild, and check: (1) the vacancy/RB
allocator moved his carries/targets to plausible teammates, (2) those teammates'
projections for that week got closer to their actual box score. `scripts/
validate_injury_vacancy.py` — see §4 below.

### 3e. PFF grades — weekly archive is RECEIVING-ONLY
`pff_imports/2025/weekly/{1..18}/` has only `receiving_summary` / `receiving_scheme`
/ `receiving_concept`. **No weekly rushing / passing / defense / O-line**, and the
`2024/weekly/` dir is empty. Season-total files exist for everything but using
one mid-2025-season = future-week leakage. Currently only PFF `route_rate` is
used (→ `_role_confidence`, WR/TE). Unused and promising: **`yprr`** (yards per
route run — strong stable WR/TE efficiency signal, in the weekly archive so
leak-safe), `grades_pass_route`, and — Week-1-cold-start-only, no leakage —
`grades_run_defense`/`stop_percent` (RB rush matchup), O-line `grades_pass_block`
(QB), `elusive_rating`/`yco_attempt` (RB rushing prior). **Plan:** a candidate
flag blending weekly `yprr` into the WR/TE receiving-yards-per-target efficiency;
build + `--add` backtest. Lower priority (most build effort).
