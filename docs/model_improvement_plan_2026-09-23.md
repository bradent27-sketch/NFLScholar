# Weekly model — improvement plan (audit of 2026-09-23)

Written as an execution spec for an implementing agent. Every item below says
**why** (the evidence, with numbers), **what** to build, **where** (files and
functions), **how** (steps), **how to test it**, and **when it ships**. The
design decisions have already been made here; the implementer should not
need to invent mechanisms, only build, measure and report. Where a real
choice is left open it is marked **USER DECISION**; stop and ask.

Read first: `docs/weekly_projections_methodology.md` (what has been
built, shipped and rejected, and why) and `docs/weekly_rankings_backlog.md`.
House rules that still apply: every model change is a named flag in
`MODEL_FEATURES`; nothing moves into `DEFAULT_FEATURES` without the user's
sign-off; heavy backtests run **one at a time** (the box OOMs running two);
each ship or reject gets a dated section in the methodology doc, with its
numbers.

---

## 0. Ranked summary

| # | Item | Type | Effort | Expected value | Depends on |
|---|------|------|--------|----------------|------------|
| 1 | Harness v2: paired startable pools, mean-consistent metrics, multiplicity rule, re-ablate everything shipped | Working against us | S–M | Foundational: every later decision depends on it | – |
| 2 | Prediction ledger + props archive + market yardage skew fix | Missing / bug | S | Time-sensitive: every unlogged week is lost for good | – |
| 3 | Historical injury replay in backtests; validate and tune the injury/vacancy layer | Unvalidated | M | High: this layer fires every live week and has never been measured time-validly | 1 |
| 4 | Per-(position, stat) early-season credibility (`STAT_K`) for weeks 2–6 | Working against us | S | Medium, strong prior evidence | 1 (and 5 for QB volume) |
| 5 | Script-neutral volume (QB first) + historical game lines in backtests | Missing / unvalidated | M–L | High for QB, the weakest position | 1 |
| 6 | Opportunity/red-zone expected-TD model (`v2_xtd`) | Missing / working against us | M–L | High: TDs are the largest unexplained variance | 1 |
| 7 | Model + market (+ FantasyPros) blend | Missing | S once 2 has data | Probably the largest single accuracy gain for the displayed number | 2 |

Suggested order: **1 and 2 together first** (2 is cheap and every week of
delay loses data), then **3**, then **4**, then **5**, then **6**. **7** runs
whenever the ledger holds 4+ scored weeks.

What the audit found to be **fine** (no action): WR/TE target and reception
blend speed already matches the empirical optimum (see §4 table); after the
skew correction in §2, the model's projected *level* for receiving yards,
rushing yards, carries and receptions sits within ~1–3% of the de-vigged
sportsbook consensus for Week 3 2026, with correlations of 0.82–0.94.

---

## Audit evidence (collected 2026-09-23)

Scripts and outputs live in the session scratchpad; re-create them from the
descriptions if you need them.

**E1 — Model vs. market, 2026 Week 3** (live board vs
`external_data/weekly_props_snapshot.json`, scorable DraftKings/Pinnacle/
standard PrizePicks/standard Underdog lines; demon/goblin/shaded alt lines
excluded; players with availability > 0.5).

| stat | n | corr | model/market (raw) | model/market (skew-corrected) | tiers low/mid/high (corrected) |
|---|---|---|---|---|---|
| receiving_yards | 124 | 0.88 | 1.106 | 0.987 | 1.06 / 1.03 / 0.99 |
| rushing_yards | 60 | 0.94 | 1.098 | 1.009 | 1.10 / 1.05 / 1.00 |
| receptions | 62 | 0.82 | 0.950 | – | 1.05 / 0.96 / 0.98 |
| rushing_attempts | 38 | 0.93 | 1.002 | – | 1.03 / 1.02 / 1.01 |
| passing_attempts | 23 | **0.57** | 0.982 | – | 1.00 / 1.03 / 0.98 |
| passing_yards | 28 | **0.75** | 1.032 | 1.032 | 1.00 / 1.04 / **1.06** |
| passing_tds | 14 | 0.70 | 1.009 | – | – |
| anytime TD P(≥1), Underdog-priced | 43 | 0.69 | model mean 0.235 vs market 0.279 | – | – |

Largest QB gaps: C.J. Stroud 41.9 attempts / 291 yds vs market 32.5 / 226
(his weeks 1–2: 38 then 55 attempts, the 55 in a game he was chasing);
Mahomes 35.7 / 281 vs 228 (27 then 47); Lamar 277 yds vs 238.
Largest skill gaps: Jonnu Smith 33.6 rec yds vs 8.5, Josh Oliver 27.3 vs 9.5,
Rhamondre Stevenson 35.8 vs 14.5, Dontayvion Wicks 69.7 vs 40.4 (model high);
Zay Flowers 49 vs 74 (81 skew-corrected), Kyle Pitts 23.6 vs 36.5, Terry
McLaurin 28.6 vs 46.0, DK Metcalf 37.4 vs 53.3 (model low). Pattern: role
players with one or two hot games are high, established alphas coming off a
quiet game are low.

**E2 — Yardage skew.** 2021–2025 player-seasons (≥10 games), pooled weekly
mean / weekly median: WR receiving 1.094, TE receiving 1.120, RB receiving
1.249, RB rushing 1.087, QB rushing 1.118, QB passing 0.993. A sportsbook
yardage line is a median; `data/market_devig.py` treats yards as symmetric
Normal, so the app's market mean is too low by those factors.

**E3 — Points skew.** 2021–2025 startable-level player-seasons (≥8 games):
weekly mean − median = +0.98 (RB), +0.98 (WR), +0.95 (TE), −0.02 (QB).
87% of RB/WR/TE player-games have 0 receiving TDs and 94% have 0 rushing TDs.

**E4 — Empirical current-vs-prior weight.** Raw nflverse weekly stats,
2020–2025, weeks 2–8, players with ≥4 prior-season games. For G games played
this season, best w in `w·current_mean + (1−w)·last_season_mean` vs the
model's implied `G/(G+K)` (role confidence 0.5):

| pos / stat | w_opt G=1 / 2 / 3 | model w | implied K |
|---|---|---|---|
| RB carries | 0.47 / 0.59 / 0.78 | 0.25 / 0.40 / 0.50 | ~1.1 (model 3) |
| RB rushing_yards | 0.27 / 0.50 / 0.70 | 0.25 / 0.40 / 0.50 | ~2.0 |
| RB targets | 0.22 / 0.66 / 0.59 | 0.25 / 0.40 / 0.50 | ~2.2 |
| QB attempts | 0.51 / 0.59 / 0.61 | 0.25 / 0.40 / 0.50 | ~1.4 (see item 5 first) |
| QB passing_yards | 0.43 / 0.63 / 0.66 | 0.25 / 0.40 / 0.50 | ~1.35 |
| QB carries | 0.16 / 0.15 / 0.52 | 0.25 / 0.40 / 0.50 | ~5 |
| QB rushing_yards | 0.29 / 0.21 / 0.40 | 0.25 / 0.40 / 0.50 | ~5 |
| WR targets | 0.26 / 0.44 / 0.52 | 0.25 / 0.40 / 0.50 | ~2.75 (≈ current 3, fine) |
| WR receptions | 0.25 / 0.43 / 0.50 | same | ~2.9 (fine) |
| WR receiving_yards | 0.17 / 0.30 / 0.35 | 0.25 / 0.40 / 0.50 | ~5 |
| WR receiving_tds | 0.06 / 0.13 / 0.16 | 0.14 / 0.25 / 0.33 | ~15 (model 6) |
| TE targets | 0.37 / 0.38 / 0.58 | 0.25 / 0.40 / 0.50 | ~2.4 |
| TE receiving_yards | 0.19 / 0.27 / 0.56 | same | ~4 |
| TE receiving_tds | 0.14 / 0.22 / 0.28 | 0.14 / 0.25 / 0.33 | ~7 |
| RB rushing_tds | 0.10 / 0.34 / 0.32 | 0.14 / 0.25 / 0.33 | ~6.5 (≈ current) |
| QB passing_tds | 0.25 / 0.23 / 0.41 | 0.17 / 0.29 / 0.38 | ~5 (≈ current) |

With an intercept, `a + b·cur + c·prior` gives b + c ≈ 0.55–0.95 (TDs
0.3–0.7): both means need extra regression toward the population. RB carries
at G=1 goes from MAE 3.39 (model weight) to 3.23 (optimal). `STAT_K` has only
ever been swept at weeks 4–15, and only as one uniform scale factor
(`.sweeps/const_stat_k.txt`).

**E5 — Harness.** `scripts/backtest_component.py::_scope_df` picks each
variant's START pool by **that variant's own** `nlargest(Model Proj Pts)`, so
base and variant are scored on different players in every START scope (the
whole-pool scopes are paired correctly). All ship gates use MAE, which is
minimized by the conditional median (see E3). Nine scopes are tested per run
and a single CI that excludes 0 has been treated as decisive.

**E6 — Backtest ≠ live.** In any historical run,
`build_weekly_projections` sets `target_margins = {}` (so the per-player
game-script multiplier is inert in every backtest ever run, but live every
week) and skips the whole availability path (so vacancy redistribution,
pecking order, RB vacancy, Questionable handling and the new in-season
returning-player restoration are all unmeasured). Both are avoidable:
nflverse `schedules` carry closing `spread_line`/`total_line`, and
`nflreadpy.load_injuries(season)` carries the official week-by-week report
(`report_status` Out/Doubtful/Questionable, `gsis_id`, `date_modified`),
2019 onward. The methodology doc's "injury status has no historical week
granularity" is true of `fetch_injury_report`, not of this source.

**E7 — TDs in season.** `credibility_shrunk_td_prior` is cold-start only;
from Week 2 a TD rate is the player's own per-game rate blended at K=5–6.
Live Week 3: Alvin Kamara 0.000 receiving TDs on 2.3 targets and 0.08
rushing TDs on 6.8 carries; Dalton Schultz 0.10 receiving TDs on 8.2 targets.
Every TD-regression experiment so far (`v2_td_volume_shrink`,
`v2_td_career_regress`) was rejected on MAE, which a mostly-zero count
rewards for staying low (E3).

**E8 — QB dispersion.** The raw calibration fit for QB is slope 0.472
(`.sweeps/calibration_fit_2026-08-30.txt`): about half of the model's QB
spread is noise. The shipped half-strength line (0.738, 4.154) is a patch.
E1 shows the same over-dispersion at the source, in pass volume.

---

## 1. Harness v2 — fix the measuring stick

**Why.** E5, E3. Any variant that simply compresses top-end projections
"wins" START-scope MAE regardless of ranking skill, and anything that raises
a skewed component (TDs, boom WRs) toward its true mean "loses". Many
decisions in the methodology doc were made on exactly these scopes and
metric.

**What to build.**

1. `scripts/harness_v2.py`, a shared module the existing scripts import
   instead of `_scope_df` / `_scope_metrics` / `_metrics`:
   - `paired_start_pool(base_df, var_df, pos, n)`: the union of the top-n by
     base projection and the top-n by variant projection, restricted to
     players present in both boards **and** in actuals. Both models are
     scored on this identical set.
   - `metrics(pred, actual)` returns n, MAE, RMSE, bias, Spearman, and
     `pairwise_acc`: over all pairs in the pool with `actual_i != actual_j`,
     the share where `sign(pred_i − pred_j) == sign(actual_i − actual_j)`.
     Prediction ties count as 0.5.
   - `td_metrics(mu, y)`: Poisson deviance
     `2·Σ[y·ln(y/μ) − (y − μ)]` (with μ floored at 1e-3 and `y·ln` taken as 0
     when y = 0), and the Brier score of `P(TD≥1) = 1 − e^(−μ)` against
     `y ≥ 1`.
   - `stat_metrics(board, actual_stats, stats)`: RMSE per stat on the paired
     START pool, for every projected stat column (targets, receptions,
     receiving_yards, rushing_attempts, rushing_yards, passing_attempts,
     passing_yards, TDs).
   - Week-cluster bootstrap (resample weeks, 3,000 draws, fixed seed) for
     every delta, reusing `_bootstrap_ci`'s approach.
2. Scopes: keep the existing nine and add **START-ALL** (the union of the four
   position START pools).
3. **Primary decision rule** (write it into the script's output):
   - SHIP-ELIGIBLE if START-ALL RMSE **or** START-ALL pairwise accuracy
     improves with a CI that excludes 0, **and** no START-position scope is
     significantly worse after Holm correction across the four, **and**
     |START-ALL bias| doesn't grow by more than 0.3.
   - REJECT if START-ALL is significantly worse on either primary metric.
   - Otherwise INCONCLUSIVE. Report it as such; don't ship on direction alone.
   - MAE is still printed, as a secondary column.
4. Port `backtest_component.py`, `sweep_model_constant.py` and
   `eval_weekly_model.py` onto it behind `--harness v2` (default v2; keep
   `--harness v1` so old numbers can be reproduced). Every `sweep_*.py` that
   imports `_scope_df`/`_scope_metrics` picks it up automatically if the v1
   names are kept as thin wrappers around a module-level switch.
5. Write per-week metric rows to `.sweeps/<run>_weekly.csv` so reruns can be
   re-analyzed without rebuilding boards.

**Validate the harness itself (unit tests, `tests/test_harness_v2.py`).**
- A/A: base vs. base returns exactly 0 on every delta.
- Shrinkage artifact: on synthetic data where actual = true mean + skewed
  noise, a variant equal to `0.9 × base` for the top half: the v1 START-MAE
  shows a "win"; v2 pairwise accuracy shows exactly 0 change. This test is
  the proof that the fix matters, so keep it.
- The pool is identical for both arms, and its size is ≥ n.
- Poisson deviance hand-checked on three small cases.

**Then re-ablate everything shipped.** Run
`backtest_component.py --harness v2 --flags <each DEFAULT flag>` on
**2023–2025, weeks 3–17** (in-season), plus a Week-1 run on 2022–2025 with
`v2_historical_ourlads` for the cold-start-only flags. Priority flags,
because each shipped on judgement, a single scope, or an MAE result that E3/E5
could have produced: `v2_pff_alignment_matchup`, `v2_defense_blowout_discount`
(QB/TE part), `v2_offense_prior_blend`, `v2_vacancy_bump_cap`,
`v2_td_prior_credibility`, `v2_pass_capacity_matchup_flex`,
`v2_receiver_cold_start_vacancy`, `WEEKLY_CALIBRATION_ONE_SIDED = False`,
`DEFENSE_PRIOR_GAMES = 12`, `MATCHUP_CLIP (0.82, 1.22)`. Also re-score the
two rejected TD flags (`v2_td_volume_shrink`, `v2_td_career_regress`) with
`td_metrics`: they were rejected on MAE.

Output one table (flag × primary metrics × verdict) into a new methodology
section. **USER DECISION:** revert any shipped flag that comes back REJECT.

**Finally, re-fit calibration.** Six refits have been deferred as "small
moves". Once the flag set is settled:
`fit_seasonal_calibration.py --mode dump --years 2021-2025 --weeks 1-18`,
then `--mode analyze`, then `--mode emit`. The fit is OLS
(mean-consistent), so keep it. Report the QB raw slope before and after
item 5; it should rise toward 1 if item 5 works.

**Pitfalls.** Don't change what the model computes in this item. The
harness change will move historical numbers; that is intended, and the doc
must say so. Runtime: each board build is tens of seconds; a 3-season ×
15-week ablation of ~20 flags is an overnight job. Queue it sequentially.

---

## 2. Prediction ledger, props archive, market skew fix

### 2a. Market yardage skew (bug; ~1 hour)

`data/market_devig.py::implied_mean_from_line` treats `_YARD_STATS` as
Normal and symmetric. Weekly yardage is right-skewed (E2), so the implied
mean is too low and **Market Proj Pts** in Weekly Rankings reads ~10% low on
yardage for RB/WR/TE. That is part of why the model "looks high" next to
books.

- Add `YARD_MEDIAN_TO_MEAN[(position, stat)]`. Measure it with a small script
  (`scripts/fit_yard_skew.py`) over 2019–2025 player-seasons with ≥8 games,
  **bucketed by the player's weekly median** (for example receiving yards
  <20, 20–40, 40–60, 60+), because skew shrinks as volume grows. Start from
  E2's pooled values: WR 1.094, TE 1.120, RB-rec 1.249, RB-rush 1.087,
  QB-rush 1.118, QB-pass 1.0.
- Apply it for `period == 'game'` only: `mean = (line + Φ⁻¹(p_over)·σ) ×
  factor(pos, stat, line)`, and on the no-p_over fallback path too. Season
  lines are sums of 17 games, close to symmetric: **don't** apply it there.
- When `position` is missing, use the board's position map the way
  `weekly_market_projection` already does. If it's still missing, use 1.0.
- Tests: an even WR 50.5 line returns ≈ 55; a season line is unchanged; an
  unknown position returns the line unchanged.
- Re-run the E1 comparison afterward and paste the table into the
  methodology doc.

### 2b. Props archive (~30 minutes)

`external_data/weekly_props_snapshot.json` is overwritten every week, so the
2026 Week 1–2 lines are already gone. In `data/odds_weekly.py::save_snapshot`
also write `external_data/props_archive/{season}_wk{week}_{fetched_at}.json`
(week from `posting_anchor` and the schedule). Keep every file; they're a
few MB each. Add the folder to `.gitignore` unless the user says otherwise.

### 2c. Prediction ledger

- New `data/prediction_ledger.py`:
  `record_board(year, week, merged_board, market_lines, fp_lines, feats)`
  writes `data/ledger/{year}_wk{week:02d}_{build_ts}.parquet`. Columns:
  player, `player_id`/gsis, Pos, Team, Opponent, every model stat column,
  Raw Model Proj Pts, Model Proj Pts, Availability, Injury Status, market
  per-stat de-vigged mean and book count, Market Proj Pts, Market Coverage,
  FP Proj Pts plus FP stat lines when loaded, a feature-set hash, and
  `git rev-parse HEAD`.
- Hook it where Weekly Rankings has the merged model/market/FP frame
  (`ui/tabs/rankings.py`, near where `merged_model['Market Proj Pts']` is
  built, around line 3415). Write only on an explicit board build, not every
  rerun, and at most one file per (year, week, feature hash) per hour.
- `scripts/score_ledger.py --year 2026`: for each ledger file whose week is
  complete, join actuals (`load_and_merge_data`) and report model vs. market
  vs. FP on the paired set, per position and stat, with the §1 metrics. Use
  the **latest pre-kickoff** build for each week.
- Backfill: rebuild 2026 weeks 1–3 as-of (`as_of_week=w`, live flags) into
  the ledger; the model side has no leakage. FP weeks 1–2 can be re-fetched
  (8 API calls; **USER DECISION**, it spends budget). Market lines for weeks
  1–2 are lost.

---

## 3. Historical injury replay — validate the injury/vacancy layer

**Why.** E6. Every backtest scores teammates of an absent star with no
knowledge that he's out, which adds error no feature can remove, and
leaves the entire availability layer unmeasured. That layer includes
`VACANCY_ABSORB = 0.75`, `VACANCY_MAX_GROWTH = 1.40`, the pecking-order
constants, the RB allocator vacancy handoff, and the in-season
returning-player restoration added 2026-09-23 (uncommitted at the time of
writing; see "Housekeeping" below).

**What to build.**

1. `data/historical_availability.py`:
   - `load_injury_reports(season)`: `nflreadpy.load_injuries([season])`,
     cached to `data/cache/injuries_{season}.parquet`, keep
     `game_type == 'REG'`.
   - `historical_injury_profiles(season, week, schedule_df)` returns the
     same `{player_name: {'plays_probability', 'workload_if_active', ...}}`
     shape `_injury_profiles` returns live, **mirroring the live policy
     exactly**. Read `data/availability_overrides.py::_probability` and the
     FantasyPros path (`data/fantasypros_availability.py`): Out/Doubtful → 0,
     Questionable → the same value the live path uses. Don't invent a new
     policy here.
   - Drop any row whose `date_modified` is later than that team's kickoff
     (schedule `gameday` + `gametime`; nflverse `gametime` is Eastern).
   - Identity by `gsis_id` → `player_id`, through the existing
     `resolve_target_week_availability` (it already resolves stable ID first).
2. In `build_weekly_projections`, add a backtest-only flag
   `v2_historical_injury_replay` (MODEL_FEATURES only, like
   `v2_historical_ourlads`). When `historical_target` and the flag is set,
   use `historical_injury_profiles` as `raw_injury_profiles` and run the
   availability resolver and vacancy path. The gate is currently
   `apply_injury and not (use_v2_guard and historical_target)` in three
   places (around lines 6490, 6891 and 6928); extend each one. Keep
   `fetch_injury_report` off for historical runs: it's still the leaky
   "most recent designation" source.
3. Harness addition: report a **RECIPIENT** scope, meaning players whose team
   had a player with ≥0.5 prior snap share ruled Out that week.

**Experiments, in order, all on harness v2, 2022–2025, weeks 3–17:**

1. `default + v2_historical_injury_replay` vs. `default`. It should improve
   RECIPIENT a lot and ALL a little. If it doesn't, stop and debug before
   going further.
2. With replay **on in both arms**, ablate `v2_vacancy`,
   `v2_receiver_vacancy_pecking_order`, the RB allocator vacancy path, and
   `v2_fantasypros_availability`'s Questionable handling.
3. `sweep_model_constant.py` (with replay on in both arms):
   `VACANCY_ABSORB` {0.5, 0.6, 0.75, 0.9, 1.0}, `VACANCY_MAX_GROWTH`
   {1.2, 1.4, 1.6, 2.0}, `RECEIVER_VACANCY_RANK_DECAY` {0.45, 0.62, 0.8}.
   The first two live in `data/weekly_projections.py`; the third lives in
   `data/rb_role_allocator.py`, and `sweep_model_constant.py` only
   monkeypatches `data.weekly_projections`. Add a `--module` argument first.
4. In-season returning-player restoration (added 2026-09-23 in
   `data/weekly_projections.py`: `_add_missing_returning_player_rows` plus the
   `zero_games_mask` branch). Subset: players with 0 current-season games
   and a prior-season role who played that week. Compare off vs. on, then
   sweep a multiplier on the restoration `alpha` {0.5, 1.0, 1.5} and the
   room-conservation trim on/off. The live evidence says it's too timid:
   Zay Flowers restored to 49 receiving yards against a skew-corrected
   market of 81.

**Ship criteria:** §1 rule. Replay itself is backtest-only and "ships" as
harness infrastructure. The constants ship on their own results.

---

## 4. Per-(position, stat) early-season credibility

**Why.** E4. The blend speed is right for WR/TE volume, too slow for RB
volume and QB pass volume, too fast for WR/TE yardage, and far too fast for
WR receiving TDs. Weeks 2–4 have never been part of a `STAT_K` sweep.

**What to build.**

1. Add `STAT_K_BY_POS = {pos: {stat: K}}` and fall back to `STAT_K` for
   anything unlisted. `_current_blend_weight` gains a `pos` argument; every
   caller passes the loop's `pos`. Find them all with
   `grep -n "_current_blend_weight\|_blended_rate(" data/weekly_projections.py`.
   Gate it behind flag `v2_stat_k_by_pos` so the default is unchanged.
2. Seed values from E4's implied K. The model's `cur` is recency-weighted
   and its `prior` is role-scaled, so these are starting points for a sweep,
   not ship values:
   - RB: rushing_attempts 1.2, rushing_yards 2.0, targets 2.2, receptions
     2.2, receiving_yards 3.0, rushing_tds 6, receiving_tds 8.
   - QB: passing_attempts 3 (unchanged until item 5 lands, then sweep
     toward 1.5), passing_yards 3 (same), rushing_attempts 5,
     rushing_yards 5, passing_tds 5, passing_interceptions 8.
   - WR: targets 3, receptions 3, receiving_yards 5, receiving_tds 15.
   - TE: targets 2.4, receptions 3, receiving_yards 4, receiving_tds 8.
3. `sweep_model_constant.py` gains `--mode scale_keys --keys RB:rushing_attempts,RB:rushing_yards`
   (scales only those leaves of `STAT_K_BY_POS`).

**Protocol.** Harness v2, **weeks 2–6**, fit years 2019–2022, confirm years
2023–2025. Sweep one group at a time, holding the rest at seed:
- RB volume (rushing_attempts, rushing_yards, targets, receptions):
  scales {0.33, 0.5, 0.75, 1, 1.5}
- WR/TE yardage: {0.75, 1, 1.5, 2}
- WR/TE TDs: {1, 1.5, 2, 3}
- QB rushing: {1, 1.5, 2}
- QB passing volume: only after item 5.

Report stat-level RMSE for the swept stats as well as points. Pick each
group's best value on the fit years, then run one confirm of the combined
setting on 2023–2025 against DEFAULT. Also run weeks 7–17 to make sure the
change doesn't hurt mid-season; K matters less there but isn't zero.

**Optional follow-up (only if the confirm passes):** E4 shows the RB
rushing-yard implied K falling with G (2.7 → 2.0 → 1.3), so the prior's
relevance decays faster than `G/(G+K)` allows. Test `K_eff = K · d^(G−1)`
with d ∈ {0.8, 0.9}.

**Ship:** §1 rule on the confirm run; `v2_stat_k_by_pos` into DEFAULT.

---

## 5. Script-neutral volume (QB first)

**Why.** E1, E6, E8. Volume stats are averaged raw, so a 55-attempt
comeback game (Stroud) or a 47-attempt shootout (Mahomes) feeds straight into
next week. The existing corrective, `_vectorized_game_script_multiplier`:
(a) needs ≥4 **current-season** games per player, so it's inert in weeks
2–4; (b) builds a curve per player from ~4 margin buckets, which is noisy;
(c) excludes passing volume by design; and (d) has never run in a backtest
(E6). The market prices a neutral script plus the spread, and it is tighter
on QB volume than the model is.

**Step A — make game lines available historically.** New backtest flag
`v2_historical_game_lines`: when `historical_target`, build `target_margins`
from the nflverse schedule's `spread_line`/`total_line` for that week (the
same arithmetic as `implied_team_points` in `data/odds_market.py`). Closing
lines are pre-kickoff, so this is time-valid. Then measure the **existing**
game-script multiplier: `default + v2_historical_game_lines` vs. `default`,
weeks 5–17, 2022–2025. It is live today and has never been measured. If it
comes back REJECT, that's a finding on its own.

**Step B — fit league script curves** (`scripts/fit_script_curves.py`,
fit on **2016–2021**; evaluation windows start at 2022):
- Realized script per team-game: the mean of `score_differential` over the
  team's offensive plays (`posteam` view) from `load_pbp`, capped at ±21.
  (`data/loaders.py::_progressive_blowout_team_weeks` already loads pbp;
  reuse its caching.) Fall back to the final margin when pbp is missing.
- For each (position, stat) in {QB passing_attempts, QB passing_yards,
  RB rushing_attempts, RB targets, WR targets, TE targets}: take each
  player-game's `stat / player season mean` (players with ≥8 games), fit
  `log(ratio) = β_real · script` (piecewise-linear with a knot at 0; allow
  different slopes when leading and trailing). This is `f_real`.
- Separately fit `log(ratio) = β_exp · (−spread from team POV)` using the
  **pregame** spread. This is `f_exp`; it's flatter than `f_real` because
  expected script is less extreme than realized script.
- Save the coefficients to `data/script_curves.json` with the fit window
  recorded.

**Step C — use them** (flag `v2_script_neutral_volume`):
- Before any per-game averaging of those stats (current-season
  `_weighted_player_rates` / `_in_season_rate` inputs **and** the
  prior-season rates), divide each game's value by `f_real(script_g)`.
- After blending, multiply by `f_exp(spread for the target week)` when a
  line exists; otherwise 1.0.
- When this flag is on, skip `_vectorized_game_script_multiplier` for the
  stats it covers, so script isn't counted twice.
- Receptions and yards follow their volume stat: scale `receptions` and
  `receiving_yards` by the targets factor, and `passing_completions` and
  `passing_yards` by the attempts factor. Rates per opportunity aren't
  script-adjusted in this version.
- Watch the knock-on through `apply_pass_capacity_conservation`: the team
  target budget comes from projected QB attempts, so it follows
  automatically. Check `scripts/check_volume_conservation.py` still reads
  ≈0.95 targets per attempt.

**Protocol.** Harness v2 with `v2_historical_game_lines` on in **both**
arms. Weeks 2–8 and 5–17, 2022–2025. Primary: START-QB RMSE and pairwise,
passing_attempts and passing_yards stat RMSE; also START-WR/TE. Also report
the QB raw calibration slope (E8) before and after. On the live ledger,
QB-attempt correlation with the market (0.57 in E1) should rise.

**Then** sweep QB passing-volume K (item 4, deferred group) on top.

**Possible Step D** (only if C ships): de-script team plays in pace too
(`load_team_pace` counts script-inflated plays).

---

## 6. Expected-TD model with red-zone opportunity

**Why.** E3, E4 (TD rows), E7. TD points are the largest share of weekly
variance the model doesn't explain, the in-season TD rate is an unregressed
own-rate that can be exactly zero, and past TD experiments were judged on a
metric that rewards zero.

**What to build** (flag `v2_xtd`, cold start and in season):

1. As-of-week red-zone usage: give `data/transforms.py::build_redzone_usage`
   an `as_of_week` parameter (filter `pbp['week'] < as_of_week`; it
   currently reads the whole season, which leaks) and split it into zones:
   - rushes: inside-5, 6–10, 11–20 yardline
   - targets: inside-10, 11–20, outside-20

   Per player-season-week, keep counts, not just shares. Join on gsis id,
   as it does today.
2. League TD rate per opportunity by (position, zone, type), measured on
   2016–2021 pbp. Store it as a constants table.
3. Player zone shares: the player's share of his team's zone opportunities,
   current season (as-of) blended with the prior season by the same
   `G/(G+K)` shape (K around 4 team-games), falling back to his overall
   carry/target share × a league zone-concentration ratio when he has no
   zone history.
4. Team zone opportunities per game: prior-season team rate, scaled by the
   target week's implied team total relative to league average (elasticity
   fitted in the same 2016–2021 script). The same implied-total source
   `v2_game_total_elasticity_perstat` uses.
5. `xTD = Σ_zones team_zone_opps × player_zone_share × league_td_rate`.
6. Final rate: `c · own_rate + (1 − c) · xTD`, where
   `c = td_events / (td_events + K_td)` counts TD events over current and
   prior seasons (prior weighted 0.7). Seed K_td at 8 for receiving and 6
   for rushing, then sweep {4, 8, 12, 20}. This replaces the in-season
   per-game TD blend for `receiving_tds`/`rushing_tds` when the flag is on;
   leave `passing_tds` alone in this version.
7. It can never produce exactly 0 for a player with projected
   opportunities > 0: the xTD part keeps it above zero.

**Protocol.** Harness v2 **with `td_metrics`** (Poisson deviance and Brier
are primary for this item; points RMSE and START pairwise are secondary).
Weeks 1 (with `v2_historical_ourlads`), 2–6, and 7–17; 2022–2025. Also
score the two old rejected TD flags under the same metrics as comparators.
The live check is the ledger's anytime-TD comparison (E1: model 0.235 vs.
market 0.279 mean P(TD≥1)).

**Pitfalls.** pbp is large: cache the per-season zone table once, and never
reload pbp inside the per-position loop. Don't use MAE on TD stats anywhere
in this item.

---

## 7. Model + market (+ FantasyPros) blend

**Why.** After §2a the market and model agree on level but diverge
player-by-player (E1), and a consensus of sharp books is the strongest
public benchmark. Averaging two partly independent good forecasts
typically beats either.

**Step 0 (one API call, USER DECISION):** check whether the FantasyPros
API returns past weeks:
`fetch_fantasypros_weekly_projections(key, 2024, 5, positions=('WR',))`.
If it does, a historical FP blend can be backtested right away across
2022–2025 (4 positions × ~17 weeks × 4 seasons ≈ 270 calls; cache
everything to `external_data/fp_weekly/`). That's budget, so ask first.

**Build** (flag-free: it's a display column, not a model change):
- `data/projection_blend.py::blend(model_stats, market_means, fp_stats, weights)`
  blends at the **stat** level per (position, stat), then scores. Where a
  source is missing, renormalize over the ones present.
- Weights come from `scripts/fit_blend_weights.py` on the scored ledger
  (item 2) or on historical FP. Per (position, stat) least squares on the
  paired set, shrunk toward equal weights with a pseudo-count of 200
  player-weeks, clipped to [0.2, 0.8].
- Weekly Rankings gets a **Blended Proj Pts** column next to Model / Market /
  FP. The Model column is unchanged.

**Ship:** once ≥4 scored ledger weeks (or the historical FP backtest) show
the blend beating the model on START-ALL RMSE and pairwise accuracy under
§1's rule. **USER DECISION:** whether Blended becomes the default sort.

---

## Housekeeping

- The in-season returning-player restoration (2026-09-23) is **uncommitted**
  in the working tree at the time of writing (`data/weekly_projections.py`,
  `tests/test_weekly_projections.py`). It fixes players who were missing
  from the board entirely, so commit it now and tune it in item 3.
- `v2_output_contract` / `v2_alignment_contract` are in `DEFAULT_FEATURES`
  but nothing reads them (backlog §4). Delete them or wire them up; they
  add noise to ablation lists.
- Each item's final numbers go in a dated section of
  `docs/weekly_projections_methodology.md`, same format as the existing
  entries.
