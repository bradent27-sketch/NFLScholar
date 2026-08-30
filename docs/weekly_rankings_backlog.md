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

## 1. 2026-08-29 session — status of each item

| Item | Status | Notes |
|---|---|---|
| #1 Alignment table: drop Final-proj cols, fold context mult into `Combined` | shipped | `ui/tabs/rankings.py` `_render_alignment_mix` |
| #2 "Build board" button, upload-first flow (no auto-build on tab open) | shipped, **unverified** | No AppTest for this tab; needs a manual browser check |
| #3 Vacancy-redistribution table (Role/audit tab) | shipped | Extended `redistribute_v2_vacated_usage` ledger with per-recipient detail; `_render_vacancy_redistribution` |
| #4 Symmetric upward capacity scaling | shipped | Folded into the A/B/D bundle below |
| #5 FantasyPros injuries: only Out/Doubtful → 0%, probability shown for reference only | shipped | `data/availability_overrides.py`, both feed paths |
| **#6 New Orleans defense-vs-RB audit + Kamara-out RB redistribution** | **not started** | Etienne under-projected on snap/rush/reception share; Neal (RB3) ~35% too high; Kendre Miller (RB5) 10% should be 0. Needs a board build. |
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
| 4 | Injury/vacancy mechanism check (§3d) | `python scripts/validate_injury_vacancy.py --year 2025 --weeks 5-17` | **NEXT** |
| 5 | GTE 2-season confirm | per-position elasticity (RB 0.17→~0.28, TE 0.30→~0.22, QB/WR keep), `GAME_TOTAL_CLIP` widened in step, 2024 + 2025 | queued |
| 6 | Case 2 backtest | `python scripts/backtest_component.py --add v2_td_prior_credibility --years 2025 --weeks 4-17` | queued |
| 7 | Case 1 backtest | `python scripts/backtest_component.py --add v2_rb_snap_anchored_volume --years 2025 --weeks 4-17` | queued |
| 8 | **FINAL calibration re-run** | `python scripts/fit_weekly_calibration.py --years 2021,2022,2023`, half-strength damp, paste into `WEEKLY_CALIBRATION` | **STAYS LAST** — only once no more model changes are pending |

> **PAUSE POINT 2026-08-30:** stopped after #3 to commit + push `weeklymodel-2026-08-30`
> so the work can continue on a second machine. Resume at #4.

### Also still owed (not blocking the queue)

- **#6 New Orleans D-vs-RB audit + Kamara-out RB redistribution** (§1 table row #6) —
  needs a 2026 Week 1 board build. Etienne under-projected on snap/rush/reception
  share; Neal (RB3) ~35% too high; Kendre Miller (RB5) 10% should be 0. Related to
  the Case 1 `v2_rb_snap_anchored_volume` work.
- **B (buried-vet dock) live eyeball** on the 2026 Week 1 board: Brown / Franklin /
  Lemon / Wicks / Oliver / Hooper / Zaccheaus / Vele. Plus a dedicated regression
  test (needs a build fixture with an Ourlads chart burying a vet).
- **C:** empirical QB-attempt board traces for a few high-volume offenses (confirm
  the pass-capacity input value — do **not** raise it, per user).
- **#2 build-board flow** browser smoke test (no AppTest for that tab).
- **Confirm run** for `DEFENSE_PRIOR_GAMES = 12` on 2024 / multi-year (value was
  chosen on 2025 wk4-17 alone).
- **Lost result:** the scheme-blend fixed-weight sweeps (`sweep_scheme_blend_weight.py`,
  TE 0.6–0.9 / WR 0.3–0.7) ran in a prior session and the output was never recorded
  (§3 "Resolution unknown"). Re-run if `v2_scheme_*` is revisited.
- **`v2_rb_snap_anchored_volume` / `v2_td_prior_credibility` live-board sanity checks**
  (Etienne/Kamara, Harvey/Henry) — deferred pending a free job slot.
