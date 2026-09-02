# Weekly Rankings — open backlog

Everything touching the weekly ranking / projection model that is proposed,
partially built, backtested-but-unshipped, blocked, or explicitly parked.
Compiled 2026-08-29, updated 2026-08-30. Sources: this codebase,
`docs/overnight_backtest_log_2026-08-27.md`, the 2026-08-29/30 session,
`HANDOFF.md` §7, and the `memory/` notes.

Live checkout is `C:\NFLScholar` (see `memory/codebase-location-and-model-status.md`).
The weekly model is a single standard model — the V1/V2 toggle was retired
2026-08-26. Backtest discipline: nothing ships to `DEFAULT_FEATURES` without an
explicit sign-off; candidates live as flags in `MODEL_FEATURES`; heavy backtests
run **strictly sequentially** (the ~16.7 GB box OOMs otherwise).

---

## 0. NEXT PUSH — action required: PFF weekly archive tracking

**User instruction (2026-09-01):** the PFF weekly archive must be committed to
the repo so the scheme/alignment backtests stay reproducible across checkouts
and model versions, "no matter what" for now. It can be dropped once the app is
finished and calibrated.

`pff_imports/2024/weekly/` was added 2026-09-01 (weeks 1-18 + postseason 19-22,
3 reports each + `manifest.csv`) and verified loading time-valid. It is
currently **untracked** — `.gitignore:87` (`pff_imports/*/weekly/`) excludes it,
so it needs `git add -f` to stage.

**DECIDED 2026-09-01 — user elected to proceed knowingly.** Track the archive;
purpose is running the local Streamlit app reproducibly, not redistribution.

Context that decision was made against, recorded so it stays reviewable:
remote `origin` is a **PUBLIC** GitHub repo (`bradent27-sketch/NFLScholar`),
so anything committed is world-readable, cloneable and search-indexed
regardless of intent. `.gitignore:87` exists for this case — its comment reads
"never stage the raw subscription reports by accident when preparing a public
push." These are licensed PFF subscription exports. `pff_imports/2025/weekly/`
(55 files) is already tracked and therefore already public.

**Cheap reversal if the intent is "local use only":** make the repo private —
that gives full reproducibility across checkouts with none of the exposure,
and needs no change to layout or code. Worth one minute before the next push.

TO DO AT NEXT PUSH:
```
git add -f pff_imports/2024/weekly/
```
(`.gitignore:87` blocks it otherwise; same force-add 2025 needed.) Verify with
`git status --short pff_imports/2024/weekly | wc -l` -> expect ~67 files
(22 weeks x 3 reports + manifest.csv).

Drop the whole archive from the repo once the model is finished and calibrated
— it is a testing input, not a runtime dependency of the app.

---

## 1. 2026-08-29 session — status of each item

| Item | Status | Notes |
|---|---|---|
| #1 Alignment table: drop Final-proj cols, fold context mult into `Combined` | shipped | `ui/tabs/rankings.py` `_render_alignment_mix` |
| #2 "Build board" button, upload-first flow (no auto-build on tab open) | shipped, **unverified** | No AppTest for this tab; needs a manual browser check |
| #3 Vacancy-redistribution table (Role/audit tab) | shipped | Extended `redistribute_v2_vacated_usage` ledger with per-recipient detail; `_render_vacancy_redistribution` |
| #4 Symmetric upward capacity scaling | shipped | Folded into the A/B/D bundle below |
| #5 FantasyPros injuries: only Out/Doubtful → 0%, probability shown for reference only | shipped | `data/availability_overrides.py`, both feed paths |
| **#6 New Orleans Kamara-out RB redistribution** | **done 2026-08-31** (behind `v2_rb_snap_anchored_volume`) | Kamara manually OUT (`availability_overrides.csv`). Root cause: with Kamara removed, chart RB3 Neal became effective-rank 2 and skipped the RB1←RB3+ depth nudge → ~34% snaps; and Miller's 8% was `_role_base` crediting back his pre-2024-injury role. Fixes under the flag: (a) nudge donor test keys off the **literal** chart rank + wider cap (`RB_VOL_SNAP_NUDGE_CAP=0.12`); (b) chart RB4+ vacancy-extension gets a smaller floor; (c) `ourlads_player_audit_arrays` now exposes `source_occurrence`, and a player charted **only** as a continuation/2nd-unit row (`position_occurrence≥1` — Miller) is shrunk toward ~1.5% AND has his interrupted-season pre-gap credit dropped in `_role_base` (`chart_deprioritized`). Result NO wk1: **Etienne 47→63%, Neal 34→24%, Estimé held 11%, Miller 8.2→2.1%**. 2 regression tests added, 24/24 allocator + 87/87 weekly-proj green. Defense-vs-RB side not touched. |
| A ±1 target capacity deadband | shipped | `PASS_CAPACITY_DEADBAND = 1.0`, both directions |
| B Buried-vet depth-chart dock (Brown/Franklin) | shipped + **fixed 2026-08-30** + tested | Was broken: checked Ourlads rank ≥4 as if team-wide, but Ourlads ranks per alignment slot, so a backup is rank **2**. Fixed: WR slot rank 2 → keep 50%, rank 3+ → hard cap. **TE runs one deeper** (`_TE = 3`) — a charted TE-2 is often the receiving TE (Gesicki). `apply_buried_veteran_dock` helper + 3 unit tests. Verified live: Brown 4.15→2.33 tgt, Franklin 4.81→2.86, Gesicki untouched at 3.23. Watch: stale Ourlads TE order for LAC/GB. |
| C Verify WR/TE budget uses the **post-defense/pace** QB attempt number | **static confirmed** | Trace confirms `apply_pass_capacity_conservation` reads `passing_attempts` after defense × script × pace × availability × environment. Empirical board-trace check (a few high-volume offenses) still NOT done. **No speculative attempt-raise, per user.** |
| D Over-budget WR/TE dock method | **reverted 2026-08-30 → uniform** | First shipped as an 8-tier trusted/tail split, then a top-2 "anchor waterfall". Both reverted at the user's request: an over-budget room is now scaled by ONE uniform proportional factor — every player (WR1 and deep reserve alike) takes the same % move, symmetric with the upward-scale case. Reducing a deep reserve's role is now entirely the role model's job (depth caps + B). `v2_pass_capacity_anchor` flag removed. |

**Verification still owed on what shipped this session:**
- ~~`scripts/check_volume_conservation.py` against A / D~~ — done 2026-08-30, clean (see §2).
- A backtest of the A + B + D bundle — the running ablation matrix (§2b) covers `v2_pass_capacity` + `v2_pass_capacity_anchor`; B is inert in that window (no Ourlads history pre-2025), so B still needs its own eyeball on the live 2026 board + a synthetic test.
- #2 browser smoke test
- A dedicated regression test for B (needs a full build fixture with an Ourlads chart burying a vet)
- Eyeball the real 2026 Week 1 board: Brown, Franklin, Lemon, Wicks, Josh Oliver, Hooper, Zaccheaus, Vele

**Never formally delivered:** the original "which backtested pieces to implement /
plan of next steps" writeup against the correct codebase. Overtaken by the edit
list; most content is in this doc + the overnight log.

---

## 2. `DEFENSE_PRIOR_GAMES` — SET to 12.0 (2026-08-30); calibration refit HELD

Constant is now **12.0** (`data/weekly_projections.py`, was 4.0). Governs
defense-matchup shrinkage — higher = trust the season's defensive sample less,
regress toward prior.

Overnight sweep (2026-08-27) found a clean monotonic dose-response: 2/3 worse,
6/8 better, WR significant at every value with the sign flip exactly at 4.0.

Extended sweep (2026-08-29, values 8/12/16/20, 2025 wk 4–17, vs 4.0 base):

| value | ALL dMAE | WR dMAE | TE dMAE | QB / RB |
|---|---|---|---|---|
| 8.0  | −0.012 (CI excl. 0) | −0.015 (excl. 0) | −0.012 (excl. 0) | no effect |
| 12.0 | −0.019 (excl. 0) | −0.024 (excl. 0) | −0.019 (excl. 0) | no effect |
| 16.0 | −0.023 (excl. 0) | −0.029 (excl. 0) | −0.026 (excl. 0) | no effect |
| 20.0 | −0.025 (excl. 0) | −0.033 (excl. 0) | −0.025 (excl. 0) | no effect |

**Read:** still monotonic on ALL/WR, but plainly **plateauing** (marginal gain per
step: ALL −.007 → −.004 → −.002). TE turns over at 20. Absolute effect small
(~0.6% MAE whole-pool at best). START scopes mostly underpowered; START-RB shows a
real −0.07 to −0.09 at 8/12/16 (noisy). **12.0 chosen** as the plateau knee.

**Calibration refit (2026-08-30):** first attempt `fit_weekly_calibration.py --years
2021,2022,2023` was run against DPG=12 + the *anchor-waterfall* version of the
pass-capacity pass and gave raw QB (0.472,8.132) RB (0.841,2.037) WR (0.758,2.721)
TE (0.795,2.386) — WR/TE slopes dropped hard, mostly *because of* the anchor
waterfall. **That waterfall has since been reverted to a uniform dock**, so those
numbers are obsolete. A fresh refit against the final (DPG=12 + deadband +
symmetric upscale + uniform dock) code is queued to run right after the ablation
matrix, then paste the half-strength-damped result. `WEEKLY_CALIBRATION` in code is
stale until then; dated note in its comment block.

**Volume-conservation guardrail (2026-08-30):** `scripts/check_volume_conservation.py`
run against the current model (DPG=12 + A/B/D) at 2026 wk1 / 2025 wk4 / 2025 wk10 —
targets/attempts 0.952–0.956 vs real-NFL 0.952, **0/32 teams over 1.0** at every
week. The audit's headline defect stays fixed. Also added as a fast synthetic
pytest: `test_team_volume_conservation_holds_across_pass_volume_levels`.

**Still owed:** confirm run on 2024 / multi-year (12.0 was chosen on 2025 wk4-17
alone); the calibration paste; re-run the startable-tier holdout check in
`docs/weekly_projections_methodology.md` against the eventual refit. Sweep hook:
`DEFENSE_PRIOR_GAMES_OVERRIDE` + gated flag `v2_defense_prior_games_override`;
`scripts/sweep_defense_prior_games.py`.

---

## 2c. Shipped 2026-08-30 (pending final calibration re-fit)

- `DEFENSE_PRIOR_GAMES` 4.0 → **12.0**
- Pass-capacity over-budget dock → **uniform proportional** (anchor waterfall reverted)
- `MATCHUP_CLIP` (0.75, 1.30) → **(0.82, 1.22)** — narrow-the-band, from the 2026-08-30 scan
- Buried-vet dock **fixed** (Ourlads per-slot rank; WR slot-2 dock, TE slot-3)
- `v2_td_prior_credibility` + `v2_rb_snap_anchored_volume` → **added to DEFAULT_FEATURES**
  at the user's explicit request (live/inspectable now; standalone ablation backtests
  still queued at §8 #6/#7, either may be reverted if it doesn't pass)

The final calibration re-run (queued, §3h) covers all four.

## 2b. V2 ablation matrix — DONE (2026-08-30)

`scripts/backtest_component.py` ablating 12 DEFAULT_FEATURES flags one at a time,
2025 wk 4–17, against the current stack (DPG=12 + deadband + symmetric upscale +
uniform over-budget dock). Flags: `role_volume`, `v2_adaptive_volume`,
`v2_td_two_year_prior`, `v2_defense_prior`, `v2_continuous_roles`,
`v2_pff_alignment_matchup`, `v2_vacancy`, `v2_preseason_rb_allocator`,
`v2_pass_capacity`, `v2_qb_volume_blend`, `v2_role_change_by_stat`,
`v2_fantasypros_availability`. Output: `.sweeps/ablation_matrix_2025.txt`,
outlier ledger `.sweeps/ablation_outliers_2025.csv`.

**Result (ablate = remove; +dMAE = the flag helps):** three flags carry the
value — `role_volume` (ALL +0.129, huge), `v2_defense_prior` (+0.033, confirms
DPG=12 nets positive), `v2_pass_capacity` (WR +0.105). `v2_pff_alignment_matchup`
was mildly negative (ALL −0.004, TE −0.019) — **user chose to keep it**. Six
flags showed ~nothing this window but have situational rationale (RB/preseason/
TD-only). `v2_vacancy` and `v2_fantasypros_availability` are untestable this way
(historical backtests have no injury feed). Full table in
`docs/overnight_backtest_log_2026-08-30.md` §1.

---

## 3. Model pieces built + backtested, NOT shipped (candidate flags)

- **`v2_scheme_matchup`** — man/zone, scheme wins outright. Real TE win (START-TE
  receiving_yards dMAE −0.052, directionally strong, borderline CI); WR wash. Off.
- **`v2_scheme_alignment_blend`** — evidence-weighted ~50/50. **Best WR result of any
  variant tested** (START-WR receiving_yards dMAE −0.297, CI excludes 0); for TE it
  dilutes scheme-alone. Off.
- **TE scheme:alignment fixed-weight blend sweep** (0.6/0.7/0.75/0.8/0.9) — was running
  overnight to find the ratio that keeps alignment context but weights it below 50/50.
  **Resolution not in the log — unknown.** `scripts/sweep_scheme_blend_weight.py`.
- **WR fixed-weight blend sweep** (0.3/0.4/0.6/0.7) — confirmatory; evidence-weighted
  default already looked best for WR. **Resolution unknown.**
- **`v2_game_total_elasticity`** and **`v2_venue_mult`** — the rejected `game_env`
  bundle (+0.012 MAE) unbundled into its two real parts (implied-total scaling,
  indoor/outdoor venue). Built, tests pass, **never backtested standalone.**
- **`v2_cold_start_regression`** — 25% pull-to-neutral on defense/context multipliers
  at true cold start. Built, backtested, **not robust** (n=5 Week-1s only;
  strength sweep 0.10/0.25/0.40/0.60 non-monotonic = noise). Not recommended;
  code retained.

**Rejected, code retained:** `game_env` bundle (+0.012 MAE), `volume_efficiency`
(+0.051 MAE). Also built-but-off and not standalone-backtested: `v2_scheme_matchup`
kin above.

---

## 4. Model-parameter parking lot — placeholder flags + ideas to brainstorm

This is the spot for new weekly-model parameter ideas. The entries below are names
that already sit in `MODEL_FEATURES` with **zero gating code** (confirmed by diff +
grep — not bugs, never built); each needs building from scratch. Add new ideas here
rather than scattering TODOs. `MODEL_FEATURES` in `data/weekly_projections.py`
carries a comment pointing here.

- **`redzone_tds`** — most promising. `data.transforms.build_redzone_usage` already
  computes real per-player red-zone target/carry share + TDs from pbp. Gaps: (1)
  needs an `as_of_week` param filtering `pbp['week'] < as_of_week` (currently
  full-season = leakage); (2) never imported by `weekly_projections.py` (only
  `data.draft_big_plays` and `ui/tabs/defensive_yield.py`). **Design call:** replace
  vs blend with the existing two-year TD-rate prior.
- **`role_trend`** — never built. (Idea: a step-change read on snap share instead of
  a decayed average — see `credibility_shrunk_td_prior`'s role-season logic for a
  related "how many seasons of this role" signal already in code.)
- **`volume_faced`** — never built (also measured neutral/negative in the study).
- **`v2_channel_matchups`** — never built.
- **`v2_output_contract`, `v2_alignment_contract`** — in both `MODEL_FEATURES` and
  `DEFAULT_FEATURES`, but nothing reads them (`'x' in feats` count = 0). Vestigial
  contract markers. Decide: wire up as real validation gates, or delete.

**Ideas not yet flagged (from prior sessions / the overnight log):**
- **Per-stat `SCRIPT_CLIP` / game-script elasticity** — the 2026-08-30 scan showed
  the single global clamp `(0.85, 1.15)` is dormant (never binding on real data);
  loosening it is inert. If blowout rushing is under-served, the lever is *how*
  `_vectorized_game_script_multiplier` builds the per-stat curve (rushing more
  script-elastic than receiving), not the clamp. Needs its own flag + `backtest_component.py`.
- **Per-position `GAME_TOTAL_ELASTICITY`** — the GTE power sweep (§ overnight log 3g)
  found RB wants ~1.5–2× the shipped elasticity, TE wants less, QB/WR ~optimal. A
  per-position tune is queued (§8 step 5) but the mechanism could also become a
  first-class flag.

---

## 5. Known bugs — no fix decided (touch high-blast-radius code)

- **QB1 cold-start 0.0 projection.** A real Week-1 starter projects at *exactly* 0.0
  when `QB1_AUTO_INCUMBENT_MIN_SHARE` isn't cleared — true rookie named starter
  (2024 Jayden Daniels), or a vet off an injury-shortened prior season (2021 Dak
  Prescott). No multiplier can fix a zero. **Design decision needed:** loosen the
  threshold for injury-shortened priors / add a 2-years-back fallback read (mirror
  the `prior2` pattern) / extend `qb1_overrides.csv` to cover historical seasons.
  Parked because QB1 selection affects every QB projection, not just Week 1.
- **Team pass/receive volume conservation** (2026-08-22 V2 audit). The WR/TE side is
  handled by `data/pass_capacity_allocator.py`, but the audit's recommended
  guardrail — a regression test asserting team targets ÷ team pass attempts stays
  ≈ 1.0 at three different week values — was never added.

---

## 6. Blocked on missing data feeds

- **wind / temp** — the single biggest measured effect in the study, deliberately
  unused: nflverse populates `wind`/`temp` **after** kickoff. Blocked on a real
  pregame forecast feed. Do NOT wire the nflverse column in.
- **O-line PFF grades** (`offense_blocking` / `offense_pass_blocking` /
  `offense_run_blockng`) — season-total only, no weekly archive → same leakage
  problem, no fix available. Usable as a **Week-1-only cold-start prior** (last
  year's grade adjusted for known personnel change) without leakage — unbuilt.
- **Red-zone usage** — the as-of-week-safe version doesn't exist yet (see §4
  `redzone_tds`).

---

## 7. UI / surfacing

- **#2 build-board flow** — shipped, unverified in a browser.
- **Vacancy-fill table (#3)** — shipped this session.
- **Injury reports** are surfaced only in Draft HQ's News sub-tab
  (`nflreadpy.load_injuries`, 2019+) — not on the weekly rankings tab or the other
  tabs.
- **Name-match fragility feeding the weekly model** — depth-chart PFF lookups still
  go through two-tier name matching (`clean_name_exact` → `clean_name_for_merge` +
  `match_abbreviated_name`). A missed/ambiguous match silently mislabels a player
  ROOKIE or drops their PFF receiving metrics. Clean fix: ID joins via
  `load_player_id_crosswalk` (pff_id ↔ gsis_id, verified 100%) — "not yet wired in",
  ~15 call sites (`HANDOFF.md` §7).

---

## 8. LIVE backtest queue — the next agent picks up here

Sequential (one heavy job at a time; the box OOMs otherwise). Status as of
**2026-08-30**, mid-session, just before the `weeklymodel-2026-08-30` branch push.
Detailed evidence for each item is in `docs/overnight_backtest_log_2026-08-30.md`
(§ numbers below point there).

| # | Item | Command / note | Status |
|---|---|---|---|
| 1 | GTE power sweep (§3g) | `.sweeps/gte_power_2025.txt` | **DONE** — per-position: RB wants ~1.5–2× more elasticity, TE less, QB/WR ~optimal. → per-position tune, confirm on 2 seasons at #5. |
| 2 | MATCHUP_CLIP scan (§3a) | — | **DONE** — shipped `(0.82, 1.22)`. |
| 3 | SCRIPT_CLIP scan (§3b) | — | **DONE — NO CHANGE.** Clamp is dormant; `(0.85, 1.15)` stays. |
| 4 | Injury/vacancy mechanism check (§3d) | `python scripts/validate_injury_vacancy.py --year 2025 --weeks 5-17` | **DONE 2026-08-31** — 2 harness bugs fixed; 77% vacated vol re-placed, 54% recip proj closer (RB 63%). Passed-but-weak. Log §3d-result. |
| 5 | GTE 2-season confirm | elasticity RB 0.28 / TE 0.22 / QB 0.42 / WR 0.14, clip (0.82, 1.24); `--add v2_game_total_elasticity --years 2024,2025 --weeks 4-17` → `.sweeps/gte_perpos_confirm_2024-2025.txt` | **DONE 2026-08-31.** ALL −0.010 (19-9 wks). **RB −0.024, CI excl 0** (confirms RB↑). QB −0.055 (21-7, p=0.01, clip-widen spillover). WR flat (keep 0.14). **TE 0.22 REJECTED — START-TE +0.102, CI excl 0** (0.30→0.22 hurt startable TE; power-sweep's "TE wants less" was inferred from up-only scaling). → **revert TE to 0.30**, keep RB 0.28 + clip widen, pending ship decision on the flag itself. |
| 6 | Case 2 backtest | ablation: `backtest_component.py --flags v2_td_prior_credibility --years 2025 --weeks 4-17` → `.sweeps/case2_td_prior_credibility_2025.txt` | **RESOLVED 2026-08-30 — KEPT (softened) in DEFAULT_FEATURES.** In-season ablation exact 0.000 (cold-start-only). First cold-start ablation (old K): −0.08 RB / **−0.14 START-RB** from removing it. Traced to over-regressing proven multi-season starters → **softened**: `TD_PRIOR_CREDIBILITY_K` 220/90/340 → 130/55/200, one-season cap 0.60→0.70. **Wide re-backtest** (`.sweeps/ablate_td_prior_cred_wk1_wide.txt`, 2020-2025 wk1): START-RB drag **−0.142 → −0.009** (gone); startable pools now wash-to-helps; ALL still −0.024 (6-0) but deep-bench-weighted. Neutral where decisions are made + principled ⇒ keep. Log §7/§7b. |
| 7 | Case 1 backtest | **RESOLVED 2026-08-30 — `v2_rb_snap_anchored_volume` REMOVED from DEFAULT_FEATURES** (stays a switchable MODEL_FEATURES flag). Cold-start ablation (`.sweeps/ablate_rb_snap_anchored_wk1.txt`, wk1 2023-25): removing it helps RB — RB −0.065, **START-RB −0.288 (3-0 wks)**. User: "the etienne case is a true outlier that is tough to predict with data." **Miller replaced with a standing chart rule** (not the removed flag): (a) `RB_CHART_VACANCY_EXTENSION_MAX=1` — one out top-3 back reaches the chart RB4, never the RB5; (b) `continuation_only` (Ourlads `position_occurrence>=1` second-column listing) is now computed unconditionally and makes a candidate a deep-reserve credibility stop in the allocator. Live result: Miller 8.2%→**0%** snaps (Etienne 51.5%, Neal 36.9%, Estimé 11.6%). Tests added. Log §7. |
| 8 | **FINAL calibration re-run** | `python scripts/fit_weekly_calibration.py --years 2021,2022,2023`, half-strength damp, paste into `WEEKLY_CALIBRATION` | **DONE 2026-08-30** (`.sweeps/calibration_fit_2026-08-30.txt`, n=11,761). Raw fit QB (0.472, 8.129) / RB (0.856, 1.952) / WR (0.934, 1.924) / TE (0.942, 1.493) — within 0.001 slope / ~0.01 intercept of the prior fit; a near-no-op, as expected (all four DEFAULT_FEATURES changes are cold-start/injury-only, don't touch the wk5-17 fit population). Half-strength damped values stored: QB (0.736, 4.064) / RB (0.928, 0.976) / WR (0.967, 0.962) / TE (0.971, 0.747) — largest move RB intercept −0.006. Half-strength kept (see §8-note). 419 tests pass. |
| 10 | **WR/TE vacancy pecking-order A/B** | `validate_injury_vacancy.py --year 2025 --weeks 5-17` ± `--no-pecking` | **DONE 2026-08-30 — NEUTRAL, keep.** Flag `v2_receiver_vacancy_pecking_order` (in DEFAULT_FEATURES). Rank-decay curve + hard cutoff + same-position affinity (`RECEIVER_VACANCY_RANK_DECAY=0.62`, `_LEAD_SHARE=0.24`, `_CROSS_POS_WEIGHT=0.30`, `_PARTICIPATION_RANKS=8`, `_ABS_GROWTH_FLOOR=2.0`). 213 cases: recipient-closer 53.3% ON vs 53.9% OFF (−0.6pp = ~5 flips, noise; no week pattern). Fixes the pathology it targets — HOU Collins 43%→13% of vacated targets, Noel 12%→34%, Hutchinson 9%→21%, WR5/6 the rest, tail ~0 — at no measurable broad cost. `validate_injury_vacancy.py` also fixed to route WR/TE sources through the production path (allocator receiver branch); reproduced the old item-#4 numbers on the OFF run. Log §6 + §7. |
| 9 | **Cold-start Week-1 A/B** (the real test for the RB-allocator / role-floor / QB1 chart machinery) | `python scripts/eval_weekly_model.py --years 2023,2024,2025 --weeks 1 --variants default,default+v2_historical_ourlads` → `.sweeps/coldstart_ourlads_wk1_2023-2025.txt` | **DONE 2026-08-31 — CLEAR WIN, one-directional.** Every scope improves with the frozen ~09/01 chart, 3-0 weeks almost everywhere. ALL dMAE **−0.650** (ρ +0.099); QB **−1.535** (bias −3.23→+0.96 — cold-start QB1 identity); WR **−0.812** (ρ +0.071, bias +1.70→+0.03); TE **−0.539**; RB −0.120. Startable: START-WR −1.254, START-TE −1.347, START-RB −0.302, all 3-0. Only soft spot: **START-QB** MAE −0.217 but **ρ −0.198 (2-1)**. Huge default cold-start over-projection (START-WR bias +4.87→+2.53) is the chart fixing who's actually on the field. n is small (3 seasons × wk1 ≈ 3 correlated samples) but the sign is unambiguous. **Validates the chart-consuming Week-1 machinery — keep it.** `v2_historical_ourlads` stays a backtest-only flag (zero live-board effect; the live path already reads the live CSV). |

> **PAUSE POINT 2026-08-30:** stopped after #3 to commit + push `weeklymodel-2026-08-30`
> so the work can continue on a second machine. Resume at #4.

### Also still owed (not blocking the queue)

- **#6 New Orleans** — Kamara-out RB redistribution **done 2026-08-31** (see §1 row #6).
  Still open only: the NO **defense-vs-RB** side (does NO's D allow enough RB
  rushing/receiving volume) was not audited; and Miller at 8.2% is off real
  prior-season evidence — revisit only if a chart-order-dominance lever is wanted.
- **B (buried-vet dock) live eyeball** on the 2026 Week 1 board: Brown / Franklin /
  Lemon / Wicks / Oliver / Hooper / Zaccheaus / Vele. Plus a dedicated regression
  test (needs a build fixture with an Ourlads chart burying a vet).
- **C:** empirical QB-attempt board traces for a few high-volume offenses (confirm
  the pass-capacity input value — do **not** raise it, per user).
- **#2 build-board flow** browser smoke test (no AppTest for that tab).
- ~~**Confirm run** for `DEFENSE_PRIOR_GAMES = 12` on 2024 / multi-year~~ **DONE
  2026-08-31** — `sweep_defense_prior_games.py` values 8/10/12/16/20 on 2022-2023
  wk4-14 (`.sweeps` task bc6ixh376). Flat basin: whole-pool ALL spans 0.004
  across pg 8-20; >12 hurts TE (CI-excl-0). **Keep 12.** Faint pg=8-10 startable
  TE/WR hint logged, not worth a calibration re-fit. See
  `docs/overnight_backtest_log_2026-08-31.md`.
- **Lost result:** the scheme-blend fixed-weight sweeps (`sweep_scheme_blend_weight.py`,
  TE 0.6–0.9 / WR 0.3–0.7) ran in a prior session and the output was never recorded
  (§3 "Resolution unknown"). Re-run if `v2_scheme_*` is revisited.
- **`v2_rb_snap_anchored_volume` / `v2_td_prior_credibility` live-board sanity checks**
  (Etienne/Kamara, Harvey/Henry) — deferred pending a free job slot.
- **Draft projection — ECR dependence reduced 2026-08-31 (untuned).** `data/draft_projections.py`:
  `STAT_SELF_WEIGHT` usage stats 0.70→0.80–0.82, receptions 0.65→0.76, yards +0.05,
  TDs +0.02 (kept low — genuine regression); `FULL_TRUST_GAMES` 24→20;
  `ROLE_CHANGE_EVIDENCE_FLOOR = 0.30` (new — a promoted player keeps ≥30% of his own
  line instead of collapsing onto the rank curve). Board rebuilds clean, no neg/NaN,
  402 players on "own history". **Owed:** (1) no draft-projection backtest harness
  exists — build one that scores `build_projected_board` (year N-1 data + ECR) against
  year N actuals, then sweep the self-weights + the floor; (2) the cleaner role-change
  fix — project *volume* from the curve at the new rank but *efficiency* (yds/touch,
  TD/touch) from the player's own history, instead of one blended `evidence`.

### 2026-08-31 overnight — two new flags built, BOTH negative, nothing shipped
- **`v2_coaching_aware_defense_prior`** (MODEL_FEATURES only). `data/coaching_changes.py`
  (HC nflverse 1999-2026, DC Ourlads 2023-25). Analysis: HC+DC change resets a
  defense (persistence corr +0.02), DC-only-under-retained-HC is *more* stable
  (+0.33). Model backtest 2023-25 wk4-17: whole-pool dead flat, START-RB +0.029
  (small real drag). **Not shipping.** Live use would need curated current-season
  DC data anyway.
- **`v2_weather_adjustment`** (MODEL_FEATURES only) + **`data/weather.py`**
  (provider-agnostic; Open-Meteo keyless, verified pulling live 2026 forecasts;
  Visual Crossing adapter ready for a key). Wind/cold effect measured 2015-25
  (QB −14% @ 15-20mph, WR −11%, RB none). Backtests: QB effect flips sign
  across windows; WR unexploitable (boom variance); only a tiny RB volume
  knock-on survives (−0.005). **Not shipping the adjustment; keeping the infra.**
- Full detail + tables: `docs/overnight_backtest_log_2026-08-31.md`.
