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

---

## 3d-result. Injury/vacancy mechanism check — DONE 2026-08-31 (queue #4)

**What.** `scripts/validate_injury_vacancy.py --year 2025 --weeks 5-17`. Retro-marks
in-hindsight absences OUT (heavy pre-window snap share → 0 snaps that week), runs
the shipped redistribution, reports vacated/allocated/unfilled + whether recipient
projections moved toward the real box score. **Two harness bugs found and fixed
before the number meant anything:**
1. `build_weekly_projections` drops its `_full_<stat>` pre-vacancy volume columns
   before returning; the redistributors size the vacated volume from exactly those.
   The script called redistribution on the returned board, so `vacated` computed as
   0 for all 213 cases (first run: "0% re-placed"). Fix: script re-attaches
   `_full_*` from the board's own injury=False volume before zeroing the OUT row.
2. Redistribution moves raw volume columns but not the points column;
   `build_weekly_projections` re-scores internally afterward, the script didn't. So
   `before == after` for every recipient → closer/worse check was a silent no-op
   ("0 closer / 0 worse"). Fix: `_rescore_model_points()` mirrors the model's own
   post-vacancy re-score block.

**Result (213 in-hindsight absences, 2025 wk 5-17):**

| pos | cases | % vacated vol re-placed | recip proj: closer / worse |
|---|---|---|---|
| WR | 99 | 80% | 293 / 246 (54%) |
| TE | 46 | 80% | 128 / 119 (52%) |
| RB | 12 | 68% | 17 / 10 (63%) |
| QB | 56 | n/a (QB1-gated backup is not a valid recipient) | 0 / 0 |
| **all** | 213 | **77%** | **438 / 375 (54% closer)** |

**Read.** Mechanism **works and is marginally net-positive.** 77% of vacated volume
lands on plausible teammates (the ~20-23% unfilled is the deliberate
`V2_VACANCY_SURVIVAL` haircut, not a leak). Recipient projections move toward
reality 54% of the time — above coin-flip but weak; RB is best (63%, n=12). This
is an in-hindsight test, not leakage-free, so treat it as "not harmful, slight
help," not a fitted win. No ship decision needed — `v2_vacancy` already ships.
Verdict: **passed-but-weak.**

---

## 5. Depth-chart source comparison — nflverse vs Ourlads (DET 2024/2025), for the cold-start-depth-chart plan

**Context.** The plan (backlog §4-area / this session) is to backtest cold-start
accuracy with vs without a depth-chart signal, using nflverse historical depth
charts so the user doesn't have to hand-save 96+ Ourlads pages. User asked: do
nflverse charts line up with Ourlads well enough to train the model's
Ourlads-handling on them, and are they hindsight-contaminated? Compared the two
hand-saved Ourlads **archive** pages (`E:\Detroit Lions Depth Chart Archive _
Ourlads2024.html` = snapshot 09/02/2024, `...2025.html` = 09/01/2025 — both
genuine pre-Week-1 snapshots) against nflverse `load_depth_charts`.

**nflverse has TWO incompatible depth-chart products:**

| | 2001–2024 ("Format A") | 2025+ ("Format B") |
|---|---|---|
| schema | `season, club_code, week, game_type, depth_team, gsis_id, position, depth_position, full_name` | `dt, team, player_name, espn_id, gsis_id, pos_grp, pos_abb, pos_slot, pos_rank` |
| grain | weekly (`week` 1-22, REG/POST), `depth_team` = 1/2/3 | **`dt`-timestamped snapshots** (2025: 219 snapshots 2025-08-03 → 2026-03-14; 2026: 161 snapshots 2026-03-22 → **2026-08-30, i.e. now**) |
| flavor | ESPN/team-official, politically massaged, slow to update | alignment-slot structured (`pos_slot`/`pos_rank`, e.g. WR slot 1/2/8), updates continuously |
| hindsight? | contemporaneous (`week==1` is the pre-Wk1 chart) but low quality | **none if you slice by `dt`** — but the "season" tag is loose (2025's last snapshot is dated 2026-03-14 and shows the 2026 offseason roster), so you MUST filter on `dt`, never take max |

**Answering the user's questions:**
- **Are 2026 depth charts available now?** YES. Format B has snapshots through
  2026-08-30. The latest snapshot IS the current preseason chart — same semantics
  as an Ourlads pre-week save, timestamped, no hindsight.
- **Is nflverse hindsight vs Ourlads' pre-week snapshot?** Format B: no (slice by
  `dt`). Format A: no (contemporaneous weekly), but it's the weak, official-flavor
  source.

**DET line-up, skill positions (the ~90% that agrees is not the story):**

| spot | Ourlads (pre-Wk1) | nflverse | agree? |
|---|---|---|---|
| 2024 QB | Goff / Hooker | Goff / (gap) / Hooker | ~ |
| 2024 **RB** | **Gibbs 1**, Montgomery 2, Reynolds 3 | **Montgomery 1**, Gibbs 2, Reynolds 3 | **NO — flipped at RB1** |
| 2024 WR | Raymond / J.Williams / St.Brown (by alignment) + I.Williams | same top 3 (all listed "1") + I.Williams | yes |
| 2024 TE | LaPorta / Wright / Hesse | LaPorta / Wright / Hesse | yes |
| 2025 RB | Gibbs 1, Montgomery 2, Reynolds 3, Vaki 4 | Gibbs 1, Montgomery 2, Reynolds 3, Vaki 4 | yes |
| 2025 TE | LaPorta / Wright / Zylstra | LaPorta / Wright / Zylstra | yes |
| 2025 WR | TeSlaa(LWR) / J.Williams(RWR, Raymond behind) / St.Brown(SWR) | St.Brown + J.Williams locked, TeSlaa/Raymond/Patrick/Lovett mixed | ~ (both: SB+JW locked, rookie TeSlaa pushing perimeter) |

**Read.** For DET the two sources agree on ~90% of skill spots across two years.
The **one disagreement is 2024 RB1 — Gibbs vs Montgomery — and it's the single
highest-leverage call on the chart** (lead back in a committee). nflverse's
official Format A chart had Montgomery over Gibbs; Ourlads had Gibbs on top; 2024
actual had Gibbs as clearly the more valuable back. This is a textbook
official-chart-is-political failure, at exactly the position (`v2_preseason_rb_allocator`,
`v2_rb_snap_anchored_volume`, the NO/Etienne case) the model most needs right.

**Consequences for the plan (n=1 team, but the failure mode is well-known):**
- **2025+ (Format B):** genuinely Ourlads-comparable — `dt`-snapshotted (no
  hindsight), alignment-slot structured, available now for 2026. Use it.
- **2023–2024 (Format A only):** the weak source, and demonstrably wrong at RB1
  for DET 2024. Options: (a) restrict the cold-start backtest to 2025+2026 and
  skip 2023–2024; (b) use 2023–2024 Format A but exclude/down-weight RB for those
  years; (c) targeted Ourlads archive saves for 2023–2024 committee-backfield
  teams only (~20 pages, not 96 — the archive page has a date dropdown, so a
  dated pre-Wk1 snapshot is one page load per team).
- The Ourlads archive page also carries **HC / OC / DC** per archive date
  (DET 2024 DC Aaron Glenn → 2025 DC Kelvin Sheppard, plus OC Ben Johnson left) —
  which independently solves the coaching-staff-history data need for the
  DEFENSE_PRIOR regime-change idea if the user pulls archive pages.
- Format B early snapshots have each row triplicated — dedup on load.

**Follow-up 2026-08-31 — user pulled the Ourlads archive pages.** 32 teams ×
2022-2025 "Depth Chart Archive" HTML pages, ~09/01 pre-Week-1 snapshots, saved to
`external_data/OurLads Historical Depth Charts - Week Before Season 0901/`.
`scripts/import_ourlads_historical.py` → `ourlads_depth_charts_history.csv` (10,542
rows) + `ourlads_coaching_staff_by_season.csv` (128 rows). Validation: all 128 files
filename-year == in-file archive-year, 32/32 per season, no dupes. One file (PIT
2024) was initially the 08/01 pre-cuts snapshot, re-pulled at 09/02 and clean.
2026+ will be dropped in as HTML (same archive format — the `.mhtml` route was
arbitrary) and re-run folds them into the same CSVs; the volatile in-season
`ourlads_depth_charts.csv` pipeline is untouched. Archive HTML carries no injury
colour classes, so 2022-2025 rows have no `is_inactive`/`status_class` equivalent
(RES-section membership is the proxy).

**Fallback wired 2026-08-31.** `load_ourlads_snapshot(year, allow_historical=True)`
adapts the archive CSV to the live `OURLADS_COLUMNS` schema (`_historical_snapshot`);
`build_weekly_projections` passes `allow_historical` and lifts the
`historical_target` guard only under the new off-by-default `MODEL_FEATURES` flag
**`v2_historical_ourlads`**. Verified: 2024 wk1 NO backfield with the flag =
Kamara RB1 63.8% / Jamaal Williams RB2 36.2%, matching the 09/02/2024 chart.
Archive gaps carry through: no `position_occurrence>=1` continuation rows, so
`chart_deprioritized` never fires for a 2022-2025 team. `eval_weekly_model.py`
gained `default` / `default+flag` variant tokens (and a UTF-8 stdout reconfigure —
the redirected run first died on `cp1252` encoding the `ρ` header).

**Cold-start A/B result — DONE 2026-08-31 (queue #9).** `--years 2023,2024,2025
--weeks 1 --variants default,default+v2_historical_ourlads`. One-directional win;
adding the frozen ~09/01 chart improves **every** scope, 3-0 across seasons almost
everywhere:

| scope | default MAE | +chart MAE | dMAE | dρ | wk W-L |
|---|---|---|---|---|---|
| ALL | 5.378 | 4.727 | **−0.650** | +0.099 | 3-0 |
| QB | 6.863 | 5.328 | **−1.535** | +0.036 | 3-0 |
| RB | 4.654 | 4.534 | −0.120 | +0.006 | 3-0 |
| WR | 6.059 | 5.247 | **−0.812** | +0.071 | 3-0 |
| TE | 4.149 | 3.610 | **−0.539** | +0.002 | 3-0 |
| START-QB | 6.460 | 6.243 | −0.217 | **−0.198** | 2-1 |
| START-RB | 6.016 | 5.714 | −0.302 | +0.009 | 3-0 |
| START-WR | 8.028 | 6.774 | **−1.254** | −0.039 | 3-0 |
| START-TE | 6.403 | 5.057 | **−1.347** | +0.030 | 3-0 |

The default cold start massively over-projects (QB bias −3.23→+0.96, WR bias
+1.70→+0.03, START-WR bias +4.87→+2.53) — the chart is fixing *who is actually on
the field* Week 1. Only soft spot is START-QB rank-corr (−0.198, 2-1). n is small
(3 seasons × wk1 ≈ 3 correlated samples) but the sign is unambiguous.
**Conclusion:** the chart-consuming Week-1 machinery (RB allocator, role floor,
QB1 gate) is validated — keep it. `v2_historical_ourlads` stays a **backtest-only**
flag: the live board's cold-start path already reads the live CSV (`historical_target`
is false there), so the flag has zero live-board effect.

Window is **Week 1 only** — `cold_start` is false from Week 2 (observed snaps take over).

---

## 6. WR/TE vacancy pecking-order reshape — BUILT 2026-08-30 (user request, in DEFAULT_FEATURES)

**Problem (live, 2026 wk1).** A departed WR/TE's targets are split among the healthy
WR/TE room **weighted by each recipient's current projected targets** (the WR/TE
branch of `redistribute_rb_vacancy_with_allocator`). That routes most of the vacated
work to the team's alpha, who has usually never sustained that share — HOU's Nico
Collins projected ~12 targets with Jayden Higgins (torn ACL) OUT. The alpha deserves
a bump; the players just behind the injured man should get the larger share.

**Fix — `v2_receiver_vacancy_pecking_order`** (`data/rb_role_allocator.py`), only
touches the WR/TE branch, RB carry/target vacancy unchanged. First cut used a
compress-toward-equal taper, which inflated the buried tail (deep bench WRs each
absorbing ~2 targets via the growth floor). **Reworked 2026-08-30 per follow-up**
to a rank-decay curve with a hard cutoff + same-position affinity:
- Rank the healthy recipients by current projected volume (pecking-order proxy).
- `RECEIVER_VACANCY_RANK_DECAY = 0.62` — geometric falloff from the top backup
  (rank 2) down: rank 3 gets 0.62× rank 2, rank 4 gets 0.62², …
- `RECEIVER_VACANCY_LEAD_SHARE = 0.24` — the current lead's weight as a fraction
  of the rank-2 weight; this is what holds the alpha to a ~10-15% bump.
- `RECEIVER_VACANCY_CROSS_POS_WEIGHT = 0.30` — a departed WR's targets favour
  other WRs over TEs (and vice-versa); cross-position weight ×0.30.
- `RECEIVER_VACANCY_PARTICIPATION_RANKS = 8` — recipients ranked deeper get 0.
- `RECEIVER_VACANCY_ABS_GROWTH_FLOOR = 2.0` — unchanged; lets a low-projected
  backup actually step up past the multiplicative `VACANCY_MAX_GROWTH` cap.

HOU live trace (Higgins OUT, 3.14 targets redistributed), share of the vacated
volume — first cut → reworked:

| recipient | raw-target split | first cut | reworked |
|---|---|---|---|
| Collins (alpha WR) | 43% | 13% | **13%** |
| Schultz (TE1) | 27% | 21% | 17% |
| Noel (WR3) | 12% | 14% | **34%** |
| Hutchinson (WR4) | 9% | 12% | **21%** |
| Moreau (TE2) | 5% | 9% | 4% |
| Jha'Quan Jackson (WR5) | 0% | ~5% | 5% |
| Josh Kelly (WR6) | 0% | ~5% | 3% |
| buried tail | ~0 | ~31% (spread) | ~2% |

Collins total 11.71 → 10.70 targets (proj 21.1 → 19.3); Noel 3.16 → 3.94;
Hutchinson 2.41 → 2.84. Targets conserved team-wide. NOTE most of Collins'
elevation over his healthy 8.16-target baseline is the **upstream cold-start
role-share reallocation** over the healthy pool, not this vacancy step (its
contribution to Collins fell ~1.35 → ~0.41) — a separate, broader mechanism this
flag does not touch.

Regression test `test_receiver_vacancy_pecking_order_shifts_share_off_the_alpha`
(HOU-shaped room w/ TE1 + buried tail): alpha held to 8-20% of the vacated volume
and no longer top recipient; WR3+WR4 take >45%; WR3 out-gains the TE1 despite the
TE1 out-targeting both pre-injury; WR5 > WR6 > 0; ranks 7-8 get <5% combined.
Added to `DEFAULT_FEATURES` per explicit request (live + inspectable now).

**Backtest (queue #10) — DONE 2026-08-30.** `eval_weekly_model.py` runs
`apply_injury=False` so it cannot exercise this flag (no OUT players) — same
limitation as #6/#7. Ran `scripts/validate_injury_vacancy.py --year 2025 --weeks
5-17` with vs without `--no-pecking` (213 in-hindsight absence cases). Metric =
did each recipient's post-redistribution projected points move CLOSER to their
real box score.

| | pecking ON | pecking OFF (raw-target split) |
|---|---|---|
| overall closer / worse | 423 / 371 (53.3%) | 438 / 375 (53.9%) |
| WR | 287 / 240 (54.5%) | 293 / 246 (54.4%) |
| TE | 119 / 121 (49.6%) | 128 / 119 (51.8%) |
| RB (unaffected) | 17 / 10 | 17 / 10 |
| % vacated placed | 77% | 77% |
| recipients / case (WR/TE) | 8 (participation cap) | 9-15 |

**Read: neutral.** −0.6pp overall / −2.2pp TE on ~800 recipient judgements is ~5
flips — inside the noise, and no directional week pattern (unlike the two msg-1
flags below). The metric is also weak for this question: it scores a 0.1-target
sliver the same as a well-sized WR3 bump. The flag demonstrably fixes the
pathology it was built for (HOU Collins 43%→13% of the vacated targets) without a
measurable broad cost. **Recommendation: keep in DEFAULT_FEATURES**; the
`RECEIVER_VACANCY_*` constants stay eyeball-tunable on the live board.

Side effect confirmed harmless: the `validate_injury_vacancy.py` WR/TE-source
path fix (route through the allocator receiver branch, not the full v2 helper)
reproduced the original item-#4 numbers almost exactly on the pecking-OFF run
(438/375, 54%) — the two paths weight recipients identically (current target
volume), so it was a fidelity fix, not a measurement change.

---

## 7. Pre-calibration ablations of the untested DEFAULT_FEATURES flags — 2026-08-30

User: "if we have new default features without a backtest let's run those first
before the calibration." Three flags shipped to `DEFAULT_FEATURES` without a
standalone backtest: `v2_td_prior_credibility` + `v2_rb_snap_anchored_volume`
(added 2026-08-30, msg 1) and `v2_receiver_vacancy_pecking_order` (this session).
The first two are cold-start-only, so the only window that exercises them is
Week 1 — ablated inside the `default+v2_historical_ourlads` config so there is a
real chart to read. **`dMAE` here is (flag OFF) − (flag ON): negative = removing
the flag improved the projection = the flag is a drag.** n ≈ 3 correlated samples
(2023-2025 × wk1) — directional, not a verdict.

### `v2_rb_snap_anchored_volume` — `.sweeps/ablate_rb_snap_anchored_wk1.txt`

| scope | MAE (ON) | MAE (OFF) | dMAE | dρ | wk W-L |
|---|---|---|---|---|---|
| ALL | 4.833 | 4.816 | −0.017 | +0.003 | 3-0 |
| RB | 4.580 | 4.514 | **−0.065** | +0.007 | 3-0 |
| START-RB | 5.880 | 5.592 | **−0.288** | +0.078 | 3-0 |
| QB / WR / TE whole | — | — | 0.000 | 0.000 | (untouched, as expected) |
| START-QB | 6.185 | 6.281 | +0.097 | +0.018 | 1-2 |
| START-TE | 5.440 | 5.505 | +0.065 | +0.064 | 0-3 |

Removing the flag helps RB, clearly on START-RB (−0.288, 3-0). The tiny
START-QB/TE moves the other way are knock-on noise (the flag doesn't touch those
positions; n≈57-71). **Caveat:** this flag also carries the msg-12/13 work
(Kamara-out snap concentration, Kendre-Miller `chart_deprioritized`, RB4 phantom
floor) — none of which this backtest can see: the historical archive has no
`position_occurrence>=1` continuation rows and Kamara-OUT is a 2026 manual
override. So the measurable Week-1 slice is a small RB drag; the part the user
asked for is invisible here.

### `v2_td_prior_credibility` — `.sweeps/ablate_td_prior_cred_wk1.txt`

| scope | MAE (ON) | MAE (OFF) | dMAE | dρ | wk W-L |
|---|---|---|---|---|---|
| ALL | 4.833 | 4.814 | −0.019 | +0.007 | 3-0 |
| RB | 4.580 | 4.497 | **−0.082** | +0.013 | 2-1 |
| START-RB | 5.880 | 5.739 | −0.142 | +0.048 | 2-1 |
| START-QB | 6.185 | 6.301 | +0.117 | −0.023 | 0-3 |
| START-WR | 7.128 | 7.191 | +0.063 | +0.006 | 0-3 |
| START-TE | 5.440 | 5.496 | +0.056 | −0.059 | 1-2 |

Nets slightly negative everywhere it has signal, RB drag again (−0.082 whole,
−0.142 startable), only a marginal startable-pass upside. Combined with #6
(in-season ablation = exact 0.000, inert), the case for keeping it in
`DEFAULT_FEATURES` is weak.

### Resolution 2026-08-30 (user)
- **`v2_receiver_vacancy_pecking_order`** → KEEP (neutral backtest, fixes the
  flagged pathology).
- **`v2_rb_snap_anchored_volume`** → **REMOVED from `DEFAULT_FEATURES`** (stays a
  switchable `MODEL_FEATURES` flag). User: "the etienne case is a true outlier
  that is tough to predict with data so lets remove it."
- **`v2_td_prior_credibility`** → **KEPT in `DEFAULT_FEATURES`, then SOFTENED**
  (see below). Principled regularizer; the user asked to make it hurt less
  rather than pull it.

### 7a. RB "chart hard stop" — replaces the removed flag's Miller lever (2026-08-30)

Removing `v2_rb_snap_anchored_volume` reverted the msg-12/13 NO backfield: Miller
back to ~8.2% snaps with Kamara out. User: implement a **standing** Ourlads rule
— "hard stop at RB5 after one injury; RB4 involved, RB5 not — so Kendre Miller
isn't projected a relevant role." Two independent, always-on changes (no new
flag; both refine the already-shipped `v2_preseason_rb_allocator`):

1. **`RB_CHART_VACANCY_EXTENSION_MAX = 1`** (`data/rb_role_allocator.py`). The
   vacancy-aware credibility ceiling was `3 + (count of unavailable top-3
   backs)`; now `3 + min(that, 1)`. One out top-three back promotes the chart
   RB4 into the credible set, never the RB5. The cap also overrides the
   incumbent backstop for a deep reserve.
2. **`continuation_only` computed unconditionally** (was gated on the removed
   flag). Ourlads lists an overflow backfield in a SECOND column, so a genuine
   5th-stringer can carry a nominal `depth_rank` of 4 — Miller is "RB4" in NO's
   `source_slot 13` column, `position_occurrence = 1`, `lc_red`. A
   `chart_deprioritized` candidate is now a deep-reserve credibility stop in the
   allocator regardless of the nominal rank, and still zeroes the interrupted
   pre-injury role credit in `_role_base`.

Live 2026 wk1 NO backfield: Etienne 51.5% / Neal 36.9% / Estimé 11.6% / **Miller
0.0%** (was 8.2%). Regression tests:
`test_one_injury_reaches_the_chart_rb4_but_not_the_rb5`,
`test_continuation_only_chart_listing_is_a_deep_reserve_regardless_of_nominal_rank`.

### 7b. `v2_td_prior_credibility` softened (2026-08-30)

The cold-start ablation's RB cost traced to over-regression of PROVEN
multi-season starters: `credibility = opp/(opp+K)` with rushing K=220 left a
2-year ~540-carry back at only ~71% of his real (goal-line-role) TD rate. The
anti-"RJ Harvey" protection actually lives in the one-season credibility CAP,
which K does not touch. Changes:
- `TD_PRIOR_CREDIBILITY_K`: rushing 220→**130**, targets 90→**55**, passing
  340→**200** (K = opportunity total for 50% credibility).
- `TD_PRIOR_ONE_SEASON_CREDIBILITY_CAP`: 0.60→**0.70** — only bites a single
  season with real bellcow volume; a thin hot half-season is already below it.

Effect: Harvey (~120 carries, 1 season) credibility 0.35→0.48 (still ~half his
hot rate regressed, cap irrelevant); 2-year ~540-carry starter ~0.71→~0.81
retained; Henry-type essentially unchanged.

**Wide re-backtest — `.sweeps/ablate_td_prior_cred_wk1_wide.txt`, 2020-2025 wk1
(6 samples, double the earlier 3), softened K.** Ablation, dMAE = (flag OFF) −
(flag ON):

| scope | narrow / OLD K | wide / NEW K | wks (wide) |
|---|---|---|---|
| ALL | −0.019 | −0.024 | 6-0 |
| RB | −0.082 | −0.059 | 5-1 |
| **START-RB** | **−0.142** | **−0.009** | 4-2 |
| WR | +0.010 | −0.021 | 4-2 |
| START-WR | +0.063 | −0.023 | 3-3 |
| START-TE | +0.056 | +0.066 | 3-3 |
| START-QB | +0.117 | −0.019 | 2-4 |

The softening did its job: **START-RB drag −0.142 → −0.009** (essentially gone),
RB whole-pool −0.082 → −0.059. The flag still nets a hair negative on ALL
(−0.024, 6-0) and RB whole-pool (−0.059), but the whole pool is deep-bench-heavy
("trivially easy to rank" per the eval docstring) and the *startable* pools —
the lineup-decision population — are now a wash-to-slightly-helps everywhere.

**Verdict: KEEP `v2_td_prior_credibility` in DEFAULT_FEATURES, softened.** It is
neutral where decisions are made, principled (anti-"RJ Harvey" small-sample
regularization), and the residual whole-pool RB cost is small and low-stakes. No
further tuning before #8.

### 7c. FINAL calibration re-fit (#8) — DONE 2026-08-30

`scripts/fit_weekly_calibration.py --years 2021,2022,2023` (wk5-17, n=11,761),
run after the whole pre-calibration stack settled. Raw out-of-sample fit:

| pos | raw (this fit) | raw (prior 2026-08-30) | half-strength damped (stored) |
|---|---|---|---|
| QB | (0.472, 8.129) | (0.472, 8.132) | (0.736, 4.064) |
| RB | (0.856, 1.952) | (0.855, 1.963) | (0.928, 0.976) |
| WR | (0.934, 1.924) | (0.933, 1.928) | (0.967, 0.962) |
| TE | (0.942, 1.493) | (0.941, 1.495) | (0.971, 0.747) |

Within 0.001 slope / ~0.01 intercept of the prior fit at every position — a
near-no-op, and the expected result: all four DEFAULT_FEATURES changes this
session (pecking-order, softened TD-credibility, snap-anchored removal, RB chart
hard stop) are cold-start- or injury-only and barely touch the in-season wk5-17
population the line is fit on. Stored values moved ≤0.006 (RB intercept).
**Half-strength damp kept** (`b_slope = 1 + 0.5·(slope−1)`, `b_intercept =
0.5·intercept`, one-sided `min(proj, line)`): don't change the damping method in
the same step as an input refit; the legibility + noisy-slope-hedge rationale is
unchanged. 419 tests pass. **The pre-calibration backtest queue is now closed.**
