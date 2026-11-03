# Mechanism audit (2026-11-02)

User: "continue working through the model and essentially audit the projection mechanism ... where is it lacking, where is it
not properly tuned, where is it most inaccurate, where does it have redundant mechanisms, where can it be simplified".

**What earlier audits already settled (not repeated here):**
- error budget, residual scan and market gap (`docs/model_audit_2026-10-21.md`);
- the family ablations and the mechanism-by-mechanism run on the 65-flag model (methodology 2026-10-23/24,
  `docs/mechanism_review_2026-10-24.md`). Ten of twelve flag families earned their keep; the dead and harmful pieces were
  removed;
- the stat overshoot and prop shapes (`docs/deep_audit_2026-10-28.md`);
- the data bugs (`docs/model_sense_check_2026-10-26.md`).

**What this audit adds:**
1. **Sizing.** Is each stage right-sized, not just worth keeping? Measured as a beta per stage (below).
2. **The 16 flags added since 10-23.** Most of them are a growing stack of points corrections, plus the new stat recal.
3. **One structural question.** Can the points total come straight from the displayed stat line?
4. **An error map of today's 81-flag model.**
5. **Code that serves flags that are off.**

## 1. The points chain is redundant with the stat recal (the main finding)

The model now has two separate systems that pull a projection toward the player's own level:
- **On the points total:**
  - the points line (with the low-end line, the start-row refit, the WR/TE recency lines and the zero-row rule);
  - the season anchor (with its s refit);
  - expected-points luck (with the QB and TE refits).

  That is about 11 flags.
- **On the stat columns:** `v2_stat_recal_own_history`, `_edges` and `_game_script` (shipped 2026-11-01).

Neither feeds the other. Points are scored from the stat line before the recal, then corrected on their own, so the
displayed stats do not add up to the displayed points.

**Test:** score the displayed (recal'd) stat line straight as fantasy points, with no line, anchor or luck. Deep-audit panel,
played rows, weeks 3-17.

| START pool RMSE | shipped points chain | displayed stat line scored | seasons better |
|---|---|---|---|
| 2021-2025, recal refit leaving each season out (chain in-sample) | 7.518 | **7.511** | 2/5 |
| QB | 7.564 | **7.523** | 4/5 |
| RB | 7.573 | **7.561** | 4/5 |
| WR | 7.694 | 7.706 | 1/5 |
| TE | 6.829 | **6.822** | 1/5 |
| 2026 weeks 3-7 (out of sample for both) | 7.349 | **7.341** | |
| every played row, 2021-2025 | 5.794 | 5.791 | |

The stat line alone ties the chain, even though the chain's lines were fit in-sample. QB is a little better, WR a little worse
(2021-2025; 2026 reverses WR and TE). The chain's job is already done by the recal.
- **Candidate:** `v2_points_from_stat_line` (off). Weeks 3-17, the total = the displayed stat line scored, times the
  Questionable cut.
- **Harness (2026-11-02, 2022-2025 weeks 3-17): passes outright.**
  - START-ALL 7.434 -> 7.415 (-0.019, CI [-0.036, -0.001]). Every START position is better (QB -0.072 with its CI below 0).
  - EVERY-START-ALL -0.027 and EVERY-ALL -0.021, both with CIs below 0; pairwise +0.003.
  - Bias: START-ALL +0.001 -> -0.173, almost all WR (traced in the methodology entry: WR rushing and receiving yards, plus the
    Questionable cut; it swings by season).
- **What it retires.** With pfs on, the season anchor and luck (+ their three refits, five flags) only touch the recal's
  3-point gate. A `lean` arm (pfs minus those five) is queued. The points line and the opening-week calibration still set
  weeks 1-2. They could go too if the stat line also sets weeks 1-2 (the edges recal is there; not yet tested).

## 2. Code that serves flags that are off

`data/weekly_projections.py` is ~14,950 lines. MODEL_FEATURES lists 132 flags: 81 on, 51 off (with this audit's two
candidates). All but a handful of the off flags still have branches in the code (1-9 references each):
- rejected candidates (e.g. `v2_share_prior_shrink`, `v2_partial_game_role_carry`, `v2_backup_te_activity`,
  `v2_points_scale_v6`, `v2_rb_participation_060`, `v2_te_fill_in_role`);
- superseded versions (e.g. `v2_game_total_elasticity` beside `_perstat`, `v2_rb_game_script` beside `_rush`,
  `v2_defense_blowout_discount_progressive`);
- sweep hooks.

The off-by-default switches the harness needs (`v2_historical_injury_replay`, `v2_historical_reserve_replay`) stay.

Removing the rejected and superseded branches changes no projection. Verify with identical default boards before and after, plus
the test suite. It would cut a large share of the branching. The methodology doc keeps each rejected flag's record.

**The list** (2026-11-02; MODEL_FEATURES 136, DEFAULT_FEATURES 81, 55 off):

| group | flags | proposal |
|---|---|---|
| harness switches | `v2_historical_injury_replay`, `v2_historical_reserve_replay`, `v2_historical_ourlads` | keep |
| this audit's candidates (pending) | `v2_points_from_stat_line`, `_edges`, `v2_vacancy_source_order`, `v2_te_capacity_relief`, `v2_wr1_capacity_relief`, `v2_xtd_missing_prior_share` | ship or reject per section 7 |
| rejected / not shipped | `v2_td_volume_shrink`, `v2_td_career_regress`, `v2_rb_snap_anchored_volume`, `v2_scheme_alignment_blend`, `v2_cold_start_regression`, `v2_coaching_aware_defense_prior`, `v2_new_team_starter_restoration`, `v2_cold_start_room_budget`, `v2_rookie_backup_wr_dampen`, `v2_vacancy_growth_cap`, `v2_vacancy_chart_split`, `v2_offense_blowout_discount`, `v2_rb_carry_budget_lead`, `v2_qb_passing_td_k`, `v2_own_tempo_regression`, `v2_vacancy_before_capacity`, `v2_qb_style_blend_k80`, `v2_te_fill_in_role`, `v2_backup_te_activity`, `v2_backup_te_bands_silent`, `v2_backup_te_bands_pieces`, `v2_rb_participation_060`, `v2_keep_fully_excluded_history`, `v2_points_scale_v6`, `v2_rb_game_script`, `v2_share_prior_shrink`, `v2_partial_game_role_carry`, `v2_share_recency`, `v2_wr_te_capacity_split` | remove branches |
| superseded | `v2_game_total_elasticity`, `v2_venue_mult` (the bundled versions ship), `v2_defense_blowout_discount_progressive` | remove branches |
| ablation arms (remove a shipped piece) | `v2_no_history_exclusions`, `v2_no_td_matchup`, `v2_td_matchup_audit`, `v2_td_matchup_pass_rec_off`, `v2_no_pace` | remove, or keep as audit tools |
| sweep hook / never built | `v2_defense_prior_games_override`, `v2_channel_matchups` | remove |
| legacy names | `volume_efficiency`, `role_matchup`, `redzone_tds`, `role_trend`, `volume_faced`, `game_env`, `teammate_vacancy` | check app/UI use first |

Removing the rejected, superseded and hook groups is 34 flags. With the `lean` ship (5 more), MODEL_FEATURES would fall from
136 to about 97.

## 3. Checked and fine

- **Vacancy growth compounds across out players.** Each out player's targets are handed out against volume the previous one
  already grew, so a 3-target WR can more than double with two receivers out (repro in `tests/test_vacancy_source_order.py`'s
  board). On history (panel 2021-2025, WR/TE projected 3+):
  - target bias by the out teammates' projected points (none / <8 / 8-16 / 16-24 / 24+): +0.22 / +0.17 / +0.07 / +0.24 /
    +0.10;
  - no over-projection where more receivers are out.

  A design oddity, not a cost.
- **The receiver vacancy row-order bug:** fixed behind `v2_vacancy_source_order` (harness queued). See the methodology doc.
- **Touchdowns are spread too wide, but a fix does not pay.** RB/WR/TE TDs are not in the stat recal. START slope of actual
  on projected: WR receiving TDs 0.58, RB rushing TDs 0.69, TE 0.41 (yards and receptions sit at 0.96-1.03). A held-out TD
  recal trims TD MSE 0.3-0.9% but leaves START points flat (7.511 -> 7.513): shrinking TDs lowers the top players.
- **The WR Questionable cut** (0.83) still fits on the stat line: actual / line 0.85 on the 219 cut START rows.

## 4. Live gap (2026 weeks 3-8, pre-kickoff boards, paired with the market)

| START | n | model | market | info slope |
|---|---|---|---|---|
| QB | 126 | 6.59 | 6.52 | 0.61 |
| RB | 183 | 7.26 | 7.05 | 0.84 |
| WR | 303 | 7.71 | 7.28 | 1.00 |
| TE | 116 | 7.18 | 7.09 | 0.68 |
| all | 728 | 7.33 | 7.07 | 0.88 |

The gap is still WR receptions and yards (swapping in the market's number helps 0.23 points each), and QB passing volume runs
low (attempts -1.6, yards -6.5: backlog W1). These boards are mixed model vintages.

## 5. Stage sizing (beta per stage)

**How it was measured.** A capture of today's 81-flag model on 2021-2025 weeks 3-17 (`.sweeps/stage_capture.parquet`,
`scripts/audit_stage_attribution.py`, full output `.sweeps/stage_audit_2021-2025.txt`). For each stage and stat, on played
games:
- c = the stage's contribution to the final number;
- beta = how much of c came true (actual = final - c + beta x c);
- the held-out MSE change from removing the stage (rm) or rescaling it to beta (scale), fit on the other seasons.

Beta near 1 means sized right, 0 means it adds nothing, and below 1 means too strong.

| stage | verdict | evidence |
|---|---|---|
| matchup multiplier | sized right (not TDs) | beta 0.97-1.31 on volume and yards; removing it costs 0.7-2.1% MSE, 4-5 of 5 seasons |
| carry budget (RB) | sized right | beta 0.95-0.96; removing it costs 9.5% (carries) / 5.1% (yards) |
| vacancy | right to a little weak | beta 1.06-1.26 alone, 0.95-1.12 fit jointly with capacity (the two overlap); worth 2.4-14% MSE; no points gain from rescaling |
| **pass capacity, TE** | **too strong** | beta 0.58-0.63 (0.51-0.67 by season); team total right after the pass. Candidate `v2_te_capacity_relief` |
| pass capacity, RB / WR | a little strong | RB receiving 0.64-0.69, WR targets 0.83; rescaling does nothing for RB/WR points |
| pass capacity, 2026 weeks 3-7 (out of sample) | too strong for everyone | WR targets / receptions / yards 0.59 / 0.53 / 0.76, RB 0.40-0.52, TE 0.72 / 0.78 / 1.18. In 2026 the whole cut runs strong: the team pass volume is low (backlog W1), which is also the live WR receptions / yards gap to the market (section 4). Input for the W1 decision |
| stat recal (own history) | sized right | beta 0.89-1.17 (fit in-sample; TE 1.15-1.17) |
| top-end yardage | sized right | beta 0.97-1.07 |
| environment / weather (QB) | sized right to weak | QB passing TDs, rushing yards and passing volume beta 1.05-1.5; removing costs 0.7-3.3% |
| pace | negligible on stats | moves a stat by ~1%; at most 0.33% MSE. The 2026-10-24 harness kept it (START-RB p 0.027): not re-opened |
| TD matchup, passing / receiving TDs | adds nothing at half strength | beta -0.10 to 0.17. Settled 2026-10-24 (half strength won its run): not re-opened |
| participation | low on played games, by design | beta 0-0.4 on box-score games: it is an every-game device (silent games, judged on EVERY scopes 2026-10-25) |
| WR Questionable cut | sized right | beta 1.04-1.11 |

**The points chain** (relevant only if pfs is not shipped):
- the line is sized right overall (beta 1.03; RB 1.44 every game, WR 0.54, TE 1.68);
- RB season anchor beta 0.6, removing it costs 0.03%;
- QB luck beta 0.4, removing it costs 0.02%;
- TE luck beta 0.35, and removing it helps (-0.32%, 4 of 5 seasons).

**A line on top of the stat line** (forward-chained) helps only rows outside the START pool (backups who came off the bench:
QB non-START rows actual 9.0 vs projected 6.0). In the START pool the plain stat line is as good as any line (QB 7.413 vs
7.425, TE 6.677 vs 6.668). pfs needs no line.

## 6. Error map of today's model

START pool, played, weeks 3-17, 2021-2025, residual = actual - projection. Points as pfs would make them (the stat line
scored); the chain's numbers are in `.sweeps/stage_audit_2021-2025.txt`.

| finding | size | consistency | status |
|---|---|---|---|
| **Rookie RBs under-projected early** | +1.00 overall; weeks 3-5 +2.11, 6-9 +1.61, 10-13 +0.88, 14-17 +0.09; carries +0.5, rushing yards +1.7 | 5 of 5 seasons | lead: in-season trust of a rookie's rushing role is slow (backlog) |
| **Rookie WRs under-projected late** | +0.95 overall; weeks 10-13 +1.42, 14-17 +1.93 | 4 of 5 (star rookies: 2021, 2023, 2024) | lead, hard to fix without hindsight |
| **WR1s under, WR3/4 over** | WR1 +0.71, WR3 -0.17 (chain: WR1 +0.27, WR3 -0.83, WR4 -0.73) | 5 of 5 | the target split is still flat at the top (backlog L2); the 10-05 within-room tilt was inconclusive |
| **TEs under** | +0.40; +0.58 where the capacity pass cut 2+ targets | 5 of 5 | `v2_te_capacity_relief` takes it to about +0.1 |
| RB game-script residual | favored by 3-7.5 +0.45, underdog by 3-7.5 -0.36 | 4-5 of 5 | small; `v2_rb_game_script_rush` already ships the main part |
| QB over-projected with teammates out | 8-16 points out -0.89 | fades: 2024 +0.48, 2025 +0.08; not monotonic | not a lead |
| WR stat line low overall | +0.33 | swings by season | traced in section 1; not a fix target |

Accuracy by position under the shipped chain / pfs (START, this panel): QB 7.568 / 7.506, RB 7.545 / 7.531, WR 7.639 / 7.643,
TE 6.727 / 6.797. TE is the one position pfs loses on this panel. The harness has it better (6.576 -> 6.556) because its START pool
and every-game rows differ. The TE capacity relief targets exactly that gap.

## 7. What to do (the user's calls)

1. **Ship `v2_points_from_stat_line`** (passed outright) **and take out the season anchor, luck and their three refits.** The
   `lean` arm (pfs minus those five) ties pfs: START RMSE identical at every position, every-game ALL +0.0001. The five flags
   only nudge 358 rows below START across the recal gate. DEFAULT_FEATURES 81 -> 77, and the points equal the stats.
   SHIPPED 2026-11-02 with item 2 (DEFAULT_FEATURES = 79 after the vso ship).
2. **`v2_points_from_stat_line_edges`** SHIPPED 2026-11-02 for RB/WR only (TE kept the points line). Then the points line survives only for QBs in weeks
   1-2.
3. **`v2_vacancy_source_order`** SHIPPED 2026-11-02 (DEFAULT_FEATURES = 82). A correctness fix (the split no longer depends on row order). It passed: START-ALL -0.001,
   no position worse. All 584 moved rows trace to team-weeks with 2+ vacancy sources, except one row that crossed the recal
   gate.
4. **`v2_te_capacity_relief`** missed its bar (EVERY-START-TE +0.010; in the chain the TE bias overshoots to 0.16 over). In
   the pfs world it reads START-TE -0.014, inside the noise. Follow-up: the chain overshoots because it scores the stat line
   before the recal. With pfs it lands right on 2022-2025, but 2026 weeks 3-7 get worse. The cut that fails to come true is
   the backup TEs' (beta 0.32, 2026 ~0); the TE1 cut mostly holds (0.67; 2022 and 2026 about 1). A backup-only relief is
   worth ~0 points. Stays off; revisit with more 2026 weeks.
5. **Dead code**: remove the rejected and superseded flags' branches (section 2), verified by identical default boards.
   DONE 2026-11-03 for the 34 flags of the rejected / superseded / hook groups (`docs/retired_flags.md`: what each tried, the
   evidence, and how to restore it). The 5 retired-by-pfs flags, the 5 ablation arms and the 7 legacy names remain.
6. **Leads for the backlog:** rookie RB early-season rushing (M18); WR1 concentration; the stat recal's 3.0-point gate is a
   cliff (M19: targets jump 35-50% across it, below START only). Each needs its own study.
7. **`v2_wr1_capacity_relief`** passed its bar (START-WR -0.010); in the pfs world START-WR -0.011, 4 of 4 seasons. The
   SHIPPED 2026-11-02 (DEFAULT_FEATURES = 81).
8. **xTD missing prior share**: both versions passed. `v2_xtd_missing_prior_overall` is the pick (START-ALL -0.007,
   START-RB -0.017; pfs world RB -0.016, 4 of 4). Rookie RB TDs per carry 0.018 -> 0.026 (actual 0.028). Weeks 1-2: the
   overall version passes (START-ALL -0.003), the plain one fails. SHIPPED 2026-11-02 with item 7 (DEFAULT_FEATURES = 81).
