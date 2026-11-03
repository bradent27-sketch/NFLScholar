# Retired model flags

On 2026-11-03, with the user's go-ahead ("work down the 34 flags... don't totally get rid of them... it's good to know what we've done before so we don't waste time"), 34 flags that were OFF in `DEFAULT_FEATURES` had their code removed from `data/weekly_projections.py` and its helper modules (`MODEL_FEATURES` 139 -> 105; the 84 default flags and every default projection are unchanged). This page is the record: what each flag tried, what the evidence said, and how to get the code back.

## Read this before re-testing an idea

"Retired" is not "ruled out". Of the 34: **23 failed**, **7 inconclusive**, **2 hook**, **1 untested**, **1 superseded**.

- **failed** - tested in the harness (or against its pre-set offline bar) and missed it or lost. Do not re-run without a new reason.
- **inconclusive** - tested, no usable effect either way (CI spans 0, or a no-op). Only worth a re-test if the inputs change.
- **untested** - built but never given a recorded backtest. **`v2_new_team_starter_restoration`** is the one: its week-1 run was queued 2026-09-07 and no result was written down (`docs/open_items_2026-10-07.md`, section C). It needs a real test before the 2027 opener; restore it from patch 05.
- **superseded** / **hook** - replaced by a shipped version, or a sweep hook / name that never had gating code.

## How to get one back

- Each family below is one commit. `git revert <sha>` restores every flag in it (conflicts are likely if the surrounding code has moved - the patch shows exactly what was removed).
- `archive/retired_flags/patches/` holds the same eleven commits as patch files (`git apply` / read them; each includes the removed tests and sweep scripts).
- Tag `retire-flags-base` is the last commit with all 34 live: `git show retire-flags-base:data/weekly_projections.py`, or check it out to re-run an old sweep exactly.
- The harness validates `--variant` names against `MODEL_FEATURES`, so a retired name is rejected until restored. Old launchers in `.sweeps/` reference them and will fail the same way.
- `data/coaching_changes.py` (cohort labels and the per-team prior-games function) and `blend_defense_prior`'s per-team Series support were kept: they are general machinery.

## The 34 flags

| Flag | Evidence | What it did | Why it was retired | Removed in |
|---|---|---|---|---|
| `v2_td_career_regress` | failed | WR/TE receiving TDs at cold start: regress the recent rate toward the player's own opportunity-weighted career TD rate, pull growing with career length and volume, behind a role-comparability gate. | START-WR +0.034 to +0.036* (CI excludes 0 at strength 0.30), gated and ungated. The cost is the up arm: rescuing a TD-light startable star adds TDs that meet a week-1 zero. | `f4bab1f` |
| `v2_td_volume_shrink` | failed | WR/TE receiving TDs at cold start: replace the opportunity-count credibility shrink with a TD-count credibility regressed toward a league TD-per-target rate on the player's own projected targets. | Week 1 2022-2025: ALL +0.024*, TE +0.099*, startable WR/TE receiving_tds MAE +0.030* (all CIs exclude 0, worse). No strength from 0.10 to 1.0 helped: regressing an unlucky rate up adds more error than it removes. | `f4bab1f` |
| `v2_partial_game_role_carry` | failed | An established player's abrupt low-snap game (normally set aside as an injury exit) counts in his role: counting stats scaled by the share with that game over the share without it, floored at 0.40. | Part 2 of its bar failed: part 1 passed (EVERY-ALL -0.006 [-0.010, -0.003], EVERY-WR -0.011) but the bias moved away from zero (-0.21 -> -0.27). | `20ed1ea` |
| `v2_share_prior_shrink` | failed | In-season expected snap share pulled toward last season's share: (n x observed + k x prior) / (n + k), k RB 0.25, WR/TE 0.75, n = appearances in the 4-game window. | Harness bar missed: START-ALL -0.001 [-0.004, +0.001] against -0.003, EVERY-ALL +0.001. Offline share RMSE improved (WR -4.4%, TE -2.8%, RB -1.3%) but points did not follow. | `20ed1ea` |
| `v2_share_recency` | failed | In-season expected snap share weights the latest of the last four appearances twice (SHARE_RECENCY_LAST_WEIGHT 2.0). | START-ALL -0.001 [-0.004, +0.003]; START-TE +0.003 against a limit of +0.002. Offline share RMSE improved 1.9-2.9% in every season; points did not follow. | `20ed1ea` |
| `v2_backup_te_activity` | failed | A TE below the room leader is scaled by how often backups with his last snap share really play (BACKUP_TE_ACTIVITY_BANDS); the capacity budget hands the difference to the TE1. | The transfer to the TE1 was the part that hurt the start pool; the discount alone shipped as v2_backup_te_discount (EVERY-TE -0.023, CI below 0). The bands remain, read by that shipped flag. | `d3efa22` |
| `v2_backup_te_bands_pieces` | failed | Only the bands that still discount on the silent-games model, the rest set to 1. | Same silent-layers run: made EVERY-TE worse. | `d3efa22` |
| `v2_backup_te_bands_silent` | failed | The backup-TE bands re-read on the silent-games model (0.93 / 0.39 / 0.87 / 1.07 / 1.08 / 1.21). | Silent-layers run 2026-10-25: none of the layers passed, each made EVERY-TE / EVERY-RB worse; the layers as shipped win on every-game scoring. | `d3efa22` |
| `v2_rb_participation_060` | failed | The RB participation model at 0.6 strength with silent games in the history (RB_PARTICIPATION_SILENT_STRENGTH). | Same silent-layers run: made EVERY-RB worse. | `d3efa22` |
| `v2_te_fill_in_role` | failed | A backup TE whose games this season came while a TE now ahead of him was out is scaled to his normal-game snap share (or 0.61 if every game was a fill-in); the TE1 gets the difference. | The user's fill-in hypothesis was confirmed in the data (fill-in games ran 42.5% of snaps against 28.8% in normal games) but the factor fixed the level, not the error: missed its bar. Only the backup discount (v2_backup_te_discount) shipped. | `d3efa22` |
| `v2_cold_start_room_budget` | failed | Cold start: clamp the sum of a team's WR / TE room snap shares to what the team ran at the position last season (+15%), water-filling the cut out of chart deep reserves. | Weeks 1-2 2022-2025: ALL +0.019*, WR +0.047*, START-WR +0.077*, for a non-significant START-TE -0.042. It water-filled real WR1s (2025 NE Diggs) to the floor. | `42723f0` |
| `v2_rookie_backup_wr_dampen` | failed | Cold start: a no-prior-role WR the Ourlads chart lists at slot rank 2 is pulled to a 0.06 share and gets zero vacancy weight (the broad version of the shipped _narrow flag). | Week 1 2022-2025: ALL +0.009, WR +0.013, TE +0.018*; no strength helped. It docked a real boom/modest-role pool (2023 Puka Nacua). The narrow version ships. | `42723f0` |
| `v2_vacancy_chart_split` | failed | Cold-start vacancy: split the pool by Ourlads chart rank (weights 0.55 / 0.25 / 0.70) so a departed role flows to the charted starter. | Weak alone; stacked with the bump and growth caps it made START-TE +0.206* (significant). Do not combine. | `42723f0` |
| `v2_vacancy_growth_cap` | inconclusive | Cold-start vacancy: post-vacancy share capped at 1.35 x the pre-vacancy share (with a 0.12 absolute-gain floor), the analogue of the in-season VACANCY_MAX_GROWTH. | A milder sibling of v2_vacancy_bump_cap with a similar backtest (ALL -0.007, TE -0.042*, startables CI spans 0); only the bump cap shipped. | `42723f0` |
| `v2_cold_start_regression` | inconclusive | Cold start: pull every matchup / script / pace / availability / environment / weather multiplier 25% of the way back to 1.0. | Five week-1s (2021-2025): no whole-pool effect; START-QB -0.227 at 0.25 but not monotonic across strengths (0.10 -0.468, 0.40 -0.108, 0.60 -0.779), consistent with noise at n = 5. Not recommended for shipment (docs/overnight_backtest_log_2026-08-27.md). | `15e4930` |
| `v2_new_team_starter_restoration` | untested | Cold start: let restore_cold_start_returning_role_share recover a team-changer's role when the new team's chart lists him rank 1 and his own active role was starter-caliber (capped at 0.80, lower pull than a same-team recovery). Built for Waddle MIA->DEN and Evans TB->SF. | Never given a recorded backtest: it was queued as 'L3' on 2026-09-07 and no result was written down (docs/open_items_2026-10-07.md, section C: run it for 2022-2025 before the next preseason). Retired as dead weight, NOT as ruled out - the patch restores it, and the 2027 opener is when to test it. | `15e4930` |
| `v2_coaching_aware_defense_prior` | failed | Per-team DEFENSE_PRIOR_GAMES by coaching-change cohort: HC+DC reset -> 8, coordinator-only change under a retained HC -> 18, no signal -> unchanged. | The weeks 2-9 gain (START-QB -0.016, START-TE -0.039) was 2 significant cells out of 63; it flipped sign on weeks 10-17 (START-WR +0.020*) and vanished fit-on-odd / scored-on-even. See docs/archive/coaching_defense_prior_RETIRED.md. data/coaching_changes.py (cohort labels) remains. | `7937ce7` |
| `v2_defense_prior_games_override` | hook | A sweep hook: DEFENSE_PRIOR_GAMES_OVERRIDE replaced DEFENSE_PRIOR_GAMES when this flag was on. | Never a model change; used only by scripts/sweep_defense_prior_games.py. | `7937ce7` |
| `v2_scheme_alignment_blend` | failed | Evidence-weighted blend of the alignment and scheme defense matchups (optionally a fixed scheme share per position). | Cross-season sweep (2024+2025 weeks 2-18): scheme alone is a clean win for TE (shipped as v2_scheme_matchup, TE only); no startable WR win at any blend weight (START-WR +0.039 / +0.149 at 0.70). The 'START-WR receiving_yards -0.297' lead was one season: -0.073 [-0.345, +0.203] across two. | `7937ce7` |
| `v2_vacancy_before_capacity` | failed | Run vacancy redistribution before the pass-capacity pass instead of after (the Dowdle / Warren double-count). | INCONCLUSIVE with bias growth +0.359 against a 0.3 cap (START-ALL bias -0.337 -> -0.696) despite real WR / START-WR RMSE gains; the capacity pass's symmetric top-up corrects under-projection when it runs last. The double count was fixed instead by the shipped v2_pass_capacity_injury_neutral_claim. | `10050ba` |
| `v2_wr_te_capacity_split` | failed | Weeks 1-2: fit the WR and TE sub-rooms against separate budgets that split the off-budget delta by TE_MARGINAL_TARGET_WEIGHT (0.20), so a departed or added wideout barely moves the tight end. | Shipped on 2026-09-07 by request, pulled the same day once backtested: a wash on ALL at every weight with a significant WR / START-WR cost, and strictly worse than v2_receiver_cold_start_vacancy alone when stacked on it (START-TE significantly worse). | `10050ba` |
| `v2_rb_snap_anchored_volume` | failed | Cold-start RB: carry/target split from the depth-aware snap allocation plus a bounded per-snap usage tilt; literal-chart-rank donor test and a wider transfer cap when a charted top-3 RB is out; a smaller phantom floor for a chart RB4+. Two 0..1 strength dials. | Built for the team-changer lead back (Etienne JAX->NO). The binary ablation on week 1 2023-2025 hurt RB (START-RB dMAE -0.288, 3-0 weeks): at full strength it is a broad 'concentrate toward the RB1' tilt on every backfield. The user's verdict: the Etienne case is a true outlier. The dials were never re-backtested into a pass. | `11daf09` |
| `v2_defense_blowout_discount_progressive` | failed | The defense-side blowout discount with a quarter-checkpoint 'decided' test (21 at half, 17 after three, 24 final) instead of a single final-margin cutoff. | 26-week confirm: every scope's CI spanned 0, including START-RB - it washed out the real RB win along with the WR loss instead of separating them. The shipped v2_defense_blowout_discount keeps the final-margin test with a position gate. | `ffb176d` |
| `v2_offense_blowout_discount` | inconclusive | Discount a player's own team's 28+ point blowout games (either direction) in his recency-weighted rate. | Checked live against the case it was built for (C.J. Stroud's 6-20 loss, a 14-point margin) and measured exactly inert. The graded, sub-28-point script effect is reached by the shipped v2_offense_script_pool_blend. | `ffb176d` |
| `v2_game_total_elasticity` | superseded | The implied game-total scaling alone, unbundled from game_env (one flat per-position exponent, QB 0.42). | Superseded by the shipped per-stat version, v2_game_total_elasticity_perstat. game_env (the legacy bundle) still gates the flat form. | `dd9b691` |
| `v2_own_tempo_regression` | inconclusive | Scale passing / receiving volume by 1 - kappa(G) x (team plays per game / league - 1), kappa = 0.65 x max(0, 1 - G/10), capped at +-10%. | Harness INCONCLUSIVE, no effect (START-ALL RMSE +0.002), although the offline signal looked real (fast quintile over-projected ~6% on pass attempts, t = -3.6). | `dd9b691` |
| `v2_rb_game_script` | failed | RB rushing TDs / yards and receiving yards scaled with the game's expected margin (RB_GAME_SCRIPT, all three stats). | The rushing part passed and shipped as v2_rb_game_script_rush; the receiving-yards term made RB receiving yards worse (held out 5 of 5 seasons, harness 3 of 4). | `dd9b691` |
| `v2_venue_mult` | inconclusive | Indoor / outdoor venue scaling alone, unbundled from game_env. | Harness 2026-10-20: INCONCLUSIVE, START-ALL 7.594 -> 7.594 (0.000, CI [-0.008, +0.008], 31-29 weeks). | `dd9b691` |
| `v2_channel_matchups` | hook | A per-route-channel matchup read. | Never built (no gating code); the mechanism-by-mechanism run 2026-10-24 measured exactly 0.000 on every player; removed from DEFAULT then. | `011dbe0` |
| `v2_keep_fully_excluded_history` | failed | A RB/WR/TE whose every game this season the partial-game filter excluded keeps those games as his history (user: Ja'Kobi Lane missing from the week 7 board). | Harness 2026-10-25: START-ALL +0.002 against a bar of +0.001; veterans whose only game was cut short lost their prior-season line (Worthy 2025: 7.8 -> 1.0 pts, he scored 17.1). The revision v2_keep_fully_excluded_no_prior shipped. | `011dbe0` |
| `v2_points_scale_v6` | inconclusive | A whole points-scale re-fit at once (v4 lines, buckets, low end, season anchor s) by scripts/fit_points_scale.py from a calibration-lab dump. | Nothing beat the shipped scale held out; POINTS_SCALE_V6 was never filled, so the flag read the shipped tables. | `011dbe0` |
| `v2_qb_passing_td_k` | inconclusive | K = 15 (shared 5) for QB passing_tds: a slower blend of the QB's own current-season TD rate. | Harness v2, 2026-10-06: START-QB -0.010 with a CI spanning 0, START-ALL -0.002, better in 25 of 60 weeks. QB_PASSING_TD_K stays because the shipped v2_qb_td_blend uses it. | `011dbe0` |
| `v2_qb_style_blend_k80` | failed | QB1's own pass/rush split shrinks toward the league mean with K = 80 dropbacks instead of 10 (QB_STYLE_BLEND_K_CANDIDATE). | Harness v2 2022-2025 weeks 3-17: INCONCLUSIVE, START-QB 7.504 -> 7.503 (CI [-0.004, +0.003], 29-31 weeks), START-ALL +0.001: fails its pre-set bar. The offline QB-only re-score (-0.0017, 5 of 5 seasons) did not carry: a QB1's split also sets his team's passing volume, which receivers and backs read through pass-capacity conservation, and those moves cancel the gain. Lesson: score QB-blend changes on full builds. | `011dbe0` |
| `v2_rb_carry_budget_lead` | failed | A modifier of the shipped v2_rb_carry_budget: split the room's carry change by carries^2 so the lead back absorbs most of it. | Offline, uniform beat lead-weighted on every player measure, so only uniform went to the harness (methodology doc, RB carry budget section; backlog M8). | `011dbe0` |

## The eleven commits

- `f4bab1f` - Cold-start WR/TE touchdown-rate regressions: `v2_td_career_regress`, `v2_td_volume_shrink`. Patch: `archive/retired_flags/patches/01-f4bab1f-cold-start-wr-te-touchdown-rate-regressions.patch`.
- `20ed1ea` - In-season snap-share refinements: `v2_partial_game_role_carry`, `v2_share_prior_shrink`, `v2_share_recency`. Patch: `archive/retired_flags/patches/02-20ed1ea-in-season-snap-share-refinements.patch`.
- `d3efa22` - Backup tight-end and RB participation variants: `v2_backup_te_activity`, `v2_backup_te_bands_pieces`, `v2_backup_te_bands_silent`, `v2_rb_participation_060`, `v2_te_fill_in_role`. Patch: `archive/retired_flags/patches/03-d3efa22-backup-tight-end-and-rb-participation-variants.patch`.
- `42723f0` - Cold-start receiver vacancy and room guards: `v2_cold_start_room_budget`, `v2_rookie_backup_wr_dampen`, `v2_vacancy_chart_split`, `v2_vacancy_growth_cap`. Patch: `archive/retired_flags/patches/04-42723f0-cold-start-receiver-vacancy-and-room-guards.patch`.
- `15e4930` - New-team starter restoration and cold-start multiplier regression: `v2_cold_start_regression`, `v2_new_team_starter_restoration`. Patch: `archive/retired_flags/patches/05-15e4930-new-team-starter-restoration-and-cold-start-mult.patch`.
- `7937ce7` - Scheme/alignment blend, coaching-aware defense prior, prior-games hook: `v2_coaching_aware_defense_prior`, `v2_defense_prior_games_override`, `v2_scheme_alignment_blend`. Patch: `archive/retired_flags/patches/06-7937ce7-scheme-alignment-blend-coaching-aware-defense-pr.patch`.
- `10050ba` - WR/TE capacity split and vacancy-before-capacity ordering: `v2_vacancy_before_capacity`, `v2_wr_te_capacity_split`. Patch: `archive/retired_flags/patches/07-10050ba-wr-te-capacity-split-and-vacancy-before-capacity.patch`.
- `11daf09` - Snap-anchored RB volume: `v2_rb_snap_anchored_volume`. Patch: `archive/retired_flags/patches/08-11daf09-snap-anchored-rb-volume.patch`.
- `ffb176d` - Offense blowout discount and progressive defense blowout: `v2_defense_blowout_discount_progressive`, `v2_offense_blowout_discount`. Patch: `archive/retired_flags/patches/09-ffb176d-offense-blowout-discount-and-progressive-defense.patch`.
- `dd9b691` - Own-tempo regression, standalone environment switches, all-stats RB game script: `v2_game_total_elasticity`, `v2_own_tempo_regression`, `v2_rb_game_script`, `v2_venue_mult`. Patch: `archive/retired_flags/patches/10-dd9b691-own-tempo-regression-standalone-environment-swit.patch`.
- `011dbe0` - Carry-budget lead split, QB TD K, QB style K80, keep-excluded, points scale v6, channel matchups: `v2_channel_matchups`, `v2_keep_fully_excluded_history`, `v2_points_scale_v6`, `v2_qb_passing_td_k`, `v2_qb_style_blend_k80`, `v2_rb_carry_budget_lead`. Patch: `archive/retired_flags/patches/11-011dbe0-carry-budget-lead-split-qb-td-k-qb-style-k80-kee.patch`.

## Where each flag is recorded elsewhere

The methodology doc and the dated audit docs keep the full numbers. Section headings that name each flag:

- `v2_td_career_regress`: `docs/mechanism_audit_2026-11-02.md` line 82 ("2. Code that serves flags that are off"); `docs/model_improvement_plan_2026-09-23.md` line 139 ("Audit evidence (collected 2026-09-23)"); `docs/model_improvement_plan_2026-09-23.md` line 216 ("1. Harness v2 — fix the measuring stick"); `docs/open_items_2026-10-07.md` line 213 ("E. Tested and closed (do not re-run without new evidence)"); `docs/weekly_projections_methodology.md` line 1097 ("2026-09-08 — `v2_td_career_regress` (regress a TD-light season to the player's OWN career "); `docs/weekly_projections_methodology.md` line 2003 ("2026-09-23/24 — harness v2 re-ablation of every priority shipped/rejected flag")
- `v2_td_volume_shrink`: `docs/mechanism_audit_2026-11-02.md` line 82 ("2. Code that serves flags that are off"); `docs/model_improvement_plan_2026-09-23.md` line 138 ("Audit evidence (collected 2026-09-23)"); `docs/model_improvement_plan_2026-09-23.md` line 216 ("1. Harness v2 — fix the measuring stick"); `docs/open_items_2026-10-07.md` line 213 ("E. Tested and closed (do not re-run without new evidence)"); `docs/weekly_projections_methodology.md` line 924 ("2026-09-08 — `v2_td_volume_shrink` (regress cold-start WR/TE TD rate to league) — BUILT, B"); `docs/weekly_projections_methodology.md` line 1099 ("2026-09-08 — `v2_td_career_regress` (regress a TD-light season to the player's OWN career ")
- `v2_partial_game_role_carry`: `docs/mechanism_audit_2026-11-02.md` line 65 ("2. Code that serves flags that are off"); `docs/model_sense_check_2026-10-26.md` line 17 ("Findings"); `docs/model_sense_check_2026-10-26.md` line 70 ("Runs"); `docs/weekly_projections_methodology.md` line 6613 ("2026-10-26 - Sense-check audit: an excluded partial game still shows the role (CANDIDATE),"); `docs/weekly_projections_methodology.md` line 6658 ("2026-10-26 - Sense-check audit harness: shrink, partial-game carry and recency all miss th")
- `v2_share_prior_shrink`: `docs/mechanism_audit_2026-11-02.md` line 65 ("2. Code that serves flags that are off"); `docs/model_sense_check_2026-10-26.md` line 18 ("Findings"); `docs/model_sense_check_2026-10-26.md` line 69 ("Runs"); `docs/weekly_projections_methodology.md` line 6582 ("2026-10-26 - Model sense-check audit: the in-season share pulled toward last season's (CAN"); `docs/weekly_projections_methodology.md` line 6620 ("2026-10-26 - Sense-check audit: an excluded partial game still shows the role (CANDIDATE),"); `docs/weekly_projections_methodology.md` line 6626 ("2026-10-26 - Sense-check audit: the latest game counts double in the share (CANDIDATE, bac")
- `v2_share_recency`: `docs/improvement_backlog.md` line 133 ("L. Audit leads (2026-10-21, error-first audit; `docs/model_audit_2026-10-21.md`)"); `docs/mechanism_audit_2026-11-02.md` line 82 ("2. Code that serves flags that are off"); `docs/model_sense_check_2026-10-26.md` line 19 ("Findings"); `docs/model_sense_check_2026-10-26.md` line 71 ("Runs"); `docs/weekly_projections_methodology.md` line 6632 ("2026-10-26 - Sense-check audit: the latest game counts double in the share (CANDIDATE, bac"); `docs/weekly_projections_methodology.md` line 6657 ("2026-10-26 - Sense-check audit harness: shrink, partial-game carry and recency all miss th")
- `v2_backup_te_activity`: `docs/improvement_backlog.md` line 128 ("L. Audit leads (2026-10-21, error-first audit; `docs/model_audit_2026-10-21.md`)"); `docs/improvement_backlog.md` line 148 ("W. Watch items (nothing to build yet)"); `docs/mechanism_audit_2026-11-02.md` line 65 ("2. Code that serves flags that are off"); `docs/open_items_2026-10-07.md` line 13 ("State at the sweep"); `docs/weekly_projections_methodology.md` line 5875 ("2026-10-22 - Audit lead L2 (TE part): `v2_backup_te_activity` (CANDIDATE), pre-set bar"); `docs/weekly_projections_methodology.md` line 5917 ("2026-10-22 - The user's fill-in hypothesis: `v2_te_fill_in_role` (CANDIDATE), pre-set bars")
- `v2_backup_te_bands_pieces`: `docs/mechanism_audit_2026-11-02.md` line 82 ("2. Code that serves flags that are off"); `docs/weekly_projections_methodology.md` line 6381 ("2026-10-25 - The RB layer and the backup-TE discount on the silent-games model (CANDIDATES")
- `v2_backup_te_bands_silent`: `docs/mechanism_audit_2026-11-02.md` line 82 ("2. Code that serves flags that are off"); `docs/weekly_projections_methodology.md` line 6382 ("2026-10-25 - The RB layer and the backup-TE discount on the silent-games model (CANDIDATES")
- `v2_rb_participation_060`: `docs/mechanism_audit_2026-11-02.md` line 66 ("2. Code that serves flags that are off"); `docs/weekly_projections_methodology.md` line 6383 ("2026-10-25 - The RB layer and the backup-TE discount on the silent-games model (CANDIDATES")
- `v2_te_fill_in_role`: `docs/improvement_backlog.md` line 128 ("L. Audit leads (2026-10-21, error-first audit; `docs/model_audit_2026-10-21.md`)"); `docs/improvement_backlog.md` line 148 ("W. Watch items (nothing to build yet)"); `docs/mechanism_audit_2026-11-02.md` line 66 ("2. Code that serves flags that are off"); `docs/open_items_2026-10-07.md` line 13 ("State at the sweep"); `docs/weekly_projections_methodology.md` line 5902 ("2026-10-22 - The user's fill-in hypothesis: `v2_te_fill_in_role` (CANDIDATE), pre-set bars"); `docs/weekly_projections_methodology.md` line 5947 ("2026-10-22 - Backup-TE candidates: none meets its pre-set bar (not shipped)")
- `v2_cold_start_room_budget`: `docs/mechanism_audit_2026-11-02.md` line 82 ("2. Code that serves flags that are off"); `docs/open_items_2026-10-07.md` line 214 ("E. Tested and closed (do not re-run without new evidence)"); `docs/weekly_projections_methodology.md` line 808 ("`v2_cold_start_room_budget` — BUILT, BACKTESTED, REJECTED")
- `v2_rookie_backup_wr_dampen`: `docs/mechanism_audit_2026-11-02.md` line 82 ("2. Code that serves flags that are off"); `docs/weekly_projections_methodology.md` line 987 ("2026-09-08 — `v2_rookie_backup_wr_dampen` (dock a charted rookie backup WR) — BUILT, BACKT")
- `v2_vacancy_chart_split`: `docs/mechanism_audit_2026-11-02.md` line 82 ("2. Code that serves flags that are off"); `docs/weekly_projections_methodology.md` line 883 ("2026-09-08 — surgical vacancy-recipient guards (`v2_vacancy_bump_cap` SHIPPED)")
- `v2_vacancy_growth_cap`: `docs/mechanism_audit_2026-11-02.md` line 82 ("2. Code that serves flags that are off"); `docs/weekly_projections_methodology.md` line 879 ("2026-09-08 — surgical vacancy-recipient guards (`v2_vacancy_bump_cap` SHIPPED)")
- `v2_cold_start_regression`: `docs/mechanism_audit_2026-11-02.md` line 82 ("2. Code that serves flags that are off"); `docs/open_items_2026-10-07.md` line 184 ("C. Product, UI and data items"); `docs/overnight_backtest_log_2026-08-27.md` line 26 ("Carried over from earlier tonight (context, not new)"); `docs/overnight_backtest_log_2026-08-27.md` line 43 ("v2_cold_start_regression backtest — 5 seasons of Week 1 (2021-2025), the"); `docs/overnight_backtest_log_2026-08-27.md` line 75 ("A real, separate, well-diagnosed bug found while reading the outlier"); `docs/weekly_rankings_backlog.md` line 385 ("Rejected on evidence, code retained")
- `v2_new_team_starter_restoration`: `docs/mechanism_audit_2026-11-02.md` line 82 ("2. Code that serves flags that are off"); `docs/open_items_2026-10-07.md` line 182 ("C. Product, UI and data items"); `docs/open_items_2026-10-07.md` line 195 ("D. Housekeeping"); `docs/weekly_rankings_backlog.md` line 106 ("SHIPPED 2026-09-05 — `v2_pff_defense_prior_blend`")
- `v2_coaching_aware_defense_prior`: `docs/mechanism_audit_2026-11-02.md` line 82 ("2. Code that serves flags that are off"); `docs/open_items_2026-10-07.md` line 215 ("E. Tested and closed (do not re-run without new evidence)"); `docs/overnight_backtest_log_2026-08-31.md` line 32 ("1. Coaching change × defense prior — `v2_coaching_aware_defense_prior`"); `docs/overnight_backtest_log_2026-08-31.md` line 89 ("Implementation"); `docs/overnight_backtest_log_2026-08-31.md` line 150 ("Coaching — FINAL verdict (2026-08-31)"); `docs/overnight_backtest_log_2026-08-31.md` line 169 ("Model backtest — `v2_coaching_aware_defense_prior` (v1 config) → NEUTRAL, DO NOT SHIP")
- `v2_defense_prior_games_override`: `docs/mechanism_audit_2026-11-02.md` line 85 ("2. Code that serves flags that are off"); `docs/open_items_2026-10-07.md` line 193 ("D. Housekeeping"); `docs/overnight_backtest_log_2026-08-27.md` line 179 ("New candidates built tonight, backtests queued next"); `docs/weekly_rankings_backlog.md` line 315 ("2. `DEFENSE_PRIOR_GAMES` — SET to 12.0 (2026-08-30); calibration refit HELD"); `docs/weekly_rankings_backlog.md` line 394 ("Correctly off — not candidates, nothing to test")
- `v2_scheme_alignment_blend`: `docs/mechanism_audit_2026-11-02.md` line 82 ("2. Code that serves flags that are off"); `docs/overnight_backtest_log_2026-08-27.md` line 20 ("Carried over from earlier tonight (context, not new)"); `docs/weekly_rankings_backlog.md` line 366 ("Genuinely open — built and worth testing")
- `v2_vacancy_before_capacity`: `docs/mechanism_audit_2026-11-02.md` line 82 ("2. Code that serves flags that are off"); `docs/open_items_2026-10-07.md` line 214 ("E. Tested and closed (do not re-run without new evidence)"); `docs/weekly_projections_methodology.md` line 2345 ("2026-09-24 — three live-model bug reports: a rookie's own role capped by a same-team share"); `docs/weekly_projections_methodology.md` line 2469 ("2026-09-25 — `RECEIVER_VACANCY_RANK_DECAY` dose-response sweep: the shipped 0.62 is measur"); `docs/weekly_projections_methodology.md` line 2471 ("2026-09-25 — `v2_vacancy_before_capacity` backtested: the double-count is real, but this f"); `docs/weekly_projections_methodology.md` line 2567 ("2026-09-25 — `v2_pass_capacity_injury_neutral_claim` backtested: the narrower fix hits the")
- `v2_wr_te_capacity_split`: `docs/mechanism_audit_2026-11-02.md` line 82 ("2. Code that serves flags that are off"); `docs/weekly_projections_methodology.md` line 643 ("2026-09-07 pass — practice-squad roster fix + cold-start receiver vacancy"); `docs/weekly_rankings_backlog.md` line 138 ("SHIPPED 2026-09-05 — `v2_pff_defense_prior_blend`")
- `v2_rb_snap_anchored_volume`: `docs/mechanism_audit_2026-11-02.md` line 82 ("2. Code that serves flags that are off"); `docs/open_items_2026-10-07.md` line 214 ("E. Tested and closed (do not re-run without new evidence)"); `docs/overnight_backtest_log_2026-08-30.md` line 164 ("3i / 3j. Two user case-study fixes — BUILT 2026-08-30 (behind flags, tested)"); `docs/overnight_backtest_log_2026-08-30.md` line 238 ("4. Backtest queue — LIVE STATUS (sequential, one heavy job at a time)"); `docs/overnight_backtest_log_2026-08-30.md` line 437 ("5. Depth-chart source comparison — nflverse vs Ourlads (DET 2024/2025), for the cold-start"); `docs/overnight_backtest_log_2026-08-30.md` line 596 ("7. Pre-calibration ablations of the untested DEFAULT_FEATURES flags — 2026-08-30")
- `v2_defense_blowout_discount_progressive`: `docs/mechanism_audit_2026-11-02.md` line 68 ("2. Code that serves flags that are off"); `docs/weekly_projections_methodology.md` line 1676 ("2026-09-16 — `v2_defense_blowout_discount`: confirming the split result with a proper CI, ")
- `v2_offense_blowout_discount`: `docs/mechanism_audit_2026-11-02.md` line 82 ("2. Code that serves flags that are off"); `docs/weekly_projections_methodology.md` line 2830 ("(2)/(3) CJ Stroud / team-plays: a harness blind spot, then a real (if modest, unconfirmed)")
- `v2_game_total_elasticity`: `docs/mechanism_audit_2026-11-02.md` line 67 ("2. Code that serves flags that are off"); `docs/overnight_backtest_log_2026-08-27.md` line 172 ("New candidates built tonight, backtests queued next"); `docs/overnight_backtest_log_2026-08-30.md` line 121 ("3c-result. Game-env isolation (`--add`, 2025 wk 4-17) — DONE"); `docs/overnight_backtest_log_2026-08-30.md` line 322 ("3c. Vegas environment — READY TO BACKTEST, no build needed"); `docs/weekly_projections_methodology.md` line 740 ("2026-09-07 — implied-total elasticity moved to per-stat"); `docs/weekly_rankings_backlog.md` line 204 ("0b. OPEN QUEUE as of 2026-09-02 (post-sweep)")
- `v2_own_tempo_regression`: `docs/improvement_backlog.md` line 163 ("Closed (do not re-run without new evidence)"); `docs/mechanism_audit_2026-11-02.md` line 82 ("2. Code that serves flags that are off"); `docs/open_items_2026-10-07.md` line 210 ("E. Tested and closed (do not re-run without new evidence)"); `docs/weekly_projections_methodology.md` line 4113 ("2026-10-06 - Historical pace proxy credited traded players' games to the wrong club (fixed")
- `v2_rb_game_script`: `docs/improvement_backlog.md` line 132 ("L. Audit leads (2026-10-21, error-first audit; `docs/model_audit_2026-10-21.md`)"); `docs/mechanism_audit_2026-11-02.md` line 67 ("2. Code that serves flags that are off"); `docs/weekly_projections_methodology.md` line 7373 ("2026-11-01 - RB game script by spread: both arms pass the harness gate (CANDIDATE `v2_rb_g"); `docs/weekly_projections_methodology.md` line 7420 ("2026-11-01 - Shipped: `v2_rb_game_script_rush` (DEFAULT_FEATURES = 80); the receiving term")
- `v2_venue_mult`: `docs/improvement_backlog.md` line 65 ("M. Model leads"); `docs/mechanism_audit_2026-11-02.md` line 83 ("2. Code that serves flags that are off"); `docs/open_items_2026-10-07.md` line 132 ("B. Model leads, ranked by (expected value / effort)"); `docs/open_items_2026-10-07.md` line 194 ("D. Housekeeping"); `docs/overnight_backtest_log_2026-08-27.md` line 172 ("New candidates built tonight, backtests queued next"); `docs/overnight_backtest_log_2026-08-30.md` line 122 ("3c-result. Game-env isolation (`--add`, 2025 wk 4-17) — DONE")
- `v2_channel_matchups`: `docs/cfb_comparison_2026-10-23.md` line 105 ("5. Are NFL's parameters redundant? (leanness check)"); `docs/mechanism_audit_2026-11-02.md` line 85 ("2. Code that serves flags that are off"); `docs/mechanism_review_2026-10-24.md` line 23 ("Summary"); `docs/mechanism_review_2026-10-24.md` line 34 ("v2_channel_matchups - dead flag"); `docs/mechanism_review_2026-10-24.md` line 102 ("What this means for "leaner""); `docs/open_items_2026-10-07.md` line 191 ("D. Housekeeping")
- `v2_keep_fully_excluded_history`: `docs/mechanism_audit_2026-11-02.md` line 82 ("2. Code that serves flags that are off"); `docs/weekly_projections_methodology.md` line 6400 ("2026-10-25 - Players with every game excluded stay on the board (CANDIDATE), pre-set bar")
- `v2_points_scale_v6`: `docs/mechanism_audit_2026-11-02.md` line 66 ("2. Code that serves flags that are off"); `docs/progress_summary_2026-10-26.md` line 26 ("2. What changed this weekend (2026-10-25)"); `docs/weekly_projections_methodology.md` line 6323 ("2026-10-25 - Points-scale re-fit, pre-set bar (written before the dump and the fit)"); `docs/weekly_projections_methodology.md` line 6368 ("2026-10-25 - Points-scale re-fit result: no variant beats the shipped scale; it stays"); `docs/weekly_projections_methodology.md` line 6747 ("2026-10-27 - Shipped: `v2_recency_calibration_wrte` (DEFAULT_FEATURES = 72)")
- `v2_qb_passing_td_k`: `docs/mechanism_audit_2026-11-02.md` line 82 ("2. Code that serves flags that are off"); `docs/open_items_2026-10-07.md` line 207 ("E. Tested and closed (do not re-run without new evidence)"); `docs/weekly_projections_methodology.md` line 4347 ("2026-10-06 - Week-5 pre-slate check: model vs the posted lines, bug scan, one fix, one can"); `docs/weekly_projections_methodology.md` line 4367 ("2026-10-06 (evening) - QB passing-TD blend: how much is there, and the best setting (offli")
- `v2_qb_style_blend_k80`: `docs/improvement_backlog.md` line 107 ("C. Calibration audit (2026-10-20): factors and weights never tuned on the current harness"); `docs/mechanism_audit_2026-11-02.md` line 82 ("2. Code that serves flags that are off"); `docs/open_items_2026-10-07.md` line 13 ("State at the sweep"); `docs/weekly_projections_methodology.md` line 5647 ("2026-10-21 - C13 follow-up: QB pass/rush split K (QB_STYLE_BLEND_K 10): candidate K = 80, "); `docs/weekly_projections_methodology.md` line 5668 ("2026-10-21 - C13 follow-up result: `v2_qb_style_blend_k80` fails its pre-set bar, not ship"); `docs/weekly_projections_methodology.md` line 5750 ("2026-10-21 - Fast screen validated (audit step 5): good for large effects and harms, too c")
- `v2_rb_carry_budget_lead`: `docs/improvement_backlog.md` line 66 ("M. Model leads"); `docs/mechanism_audit_2026-11-02.md` line 82 ("2. Code that serves flags that are off"); `docs/open_items_2026-10-07.md` line 136 ("B. Model leads, ranked by (expected value / effort)"); `docs/weekly_projections_methodology.md` line 3563 ("2026-09-30 - `v2_rb_carry_budget` (CANDIDATE): in-season team RB carry budget")

## The original registry notes, verbatim

Copied from the `MODEL_FEATURES` entry of each flag in `retire-flags-base` (comments are the code's own words at the time).

### `v2_td_career_regress`

```text
'v2_td_career_regress',  # WR/TE receiving_tds, cold start: regress the
# recent rate toward the player's OWN opportunity-
# weighted CAREER TD-per-game rate instead of a
# league/position anchor, pull growing with career
# length (role_seasons) + career opportunity volume.
# Two-sided; fires only for role_seasons >= 2 +
# finite career rate. Alternative to
# v2_td_volume_shrink. TD_CAREER_REGRESS_STRENGTH is
# the 0..1 dial. BUILT, BACKTESTED (gated +
# ungated), REJECTED - stays OFF. scripts/sweep_td_
# career_regress.py wk1 2022-2025: START-WR
# +0.034..+0.036 (CI excludes 0 at 0.30) in BOTH
# versions; the comparability gate did not fix it
# and turned whole-pool WR into +0.009*..+0.011*.
# The cost is the UP arm (rescuing a TD-light
# startable star adds TDs that meet a Week-1 zero),
# same failure as v2_td_volume_shrink - regressing
# startable TD rates UPward always loses on this
# window regardless of the anchor. Kept for the
# machinery. See career_regressed_td_prior and the
# dated section in docs/weekly_projections_methodology.md.
```

### `v2_td_volume_shrink`

```text
'v2_td_volume_shrink',   # WR/TE receiving_tds, cold start: REPLACE the
# above with a TD-COUNT credibility regressed
# toward a league TD-per-target rate on the
# player's own projected target volume - so a
# TD-unlucky WR1 (Jefferson 0.25 -> 0.32) pulls UP
# toward a league rate on his role. Eye-test does
# the right thing two-sided (Jefferson/Lamb/Burden
# up, St. Brown/Kittle down) but BUILT, BACKTESTED,
# REJECTED - stays OFF. scripts/sweep_td_volume_
# shrink.py, wk1 2022-2025: ALL +0.024* / TE
# +0.099* / startable-WR/TE receiving_tds MAE
# +0.030* - all bootstrap-CI-excludes-0 WORSE. It
# lifts nearly every TE toward a 0.058/target rate
# they don't hit in a single game; regressing an
# unlucky rate UP adds more error than it removes.
# TD_VOLUME_SHRINK_STRENGTH sweeps {0.25..1.0} and
# {0.10,0.175,0.25} confirm: recv_tds MAE
# +0.012*..+0.029* and its MEDIAN error never
# improves at ANY strength, and START-TE is
# +0.12*..+0.13* at every soft strength. See
# td_volume_shrunk_prior and
# docs/weekly_projections_methodology.md.
```

### `v2_partial_game_role_carry`

```text
'v2_partial_game_role_carry',  # CANDIDATE (2026-10-26, model sense-check audit). An established
# player's sudden low-snap game ("abrupt partial role") is
# left out of his history, so the next week he is projected
# at his old role. It is half real: he plays the next week at
# 0.47 of snaps from a 0.77 role (0.57 back from a missed
# game). Count it in the share - his volume is scaled by the
# share with that game over the share without it
# (partial_game_role_multiplier). Harness boards 2022-2025:
# those rows run +1.49 pts over (+0.91 after a missed game),
# the rest -0.25; share 0.68 projected vs 0.48 played.
```

### `v2_share_prior_shrink`

```text
'v2_share_prior_shrink',  # CANDIDATE (2026-10-26, model sense-check audit). In season, a RB/WR/TE's
# expected snap share is the plain mean of his last <=4
# appearances, so one odd game (a kick-return-only week,
# a blowout) is his whole role. Pull it toward his prior
# season's share as if that were SHARE_PRIOR_SHRINK_K[pos]
# more games: (n x observed + k x prior) / (n + k). Offline
# 2022-2025 (next-week share, n 1-4): RMSE WR -4.4%,
# TE -2.8%, RB -1.3%; held out by season 4/4, 4/4, 3/4.
```

### `v2_share_recency`

```text
'v2_share_recency',  # CANDIDATE (2026-10-26, model sense-check audit; backlog L6). The in-season
# expected snap share weights his latest appearance
# SHARE_RECENCY_LAST_WEIGHT (2) times the other three.
# Offline 2022-2025 (next-week share, every player-week):
# RMSE WR -2.9%, TE -2.4%, RB -1.9%, every season; stacks
# with v2_share_prior_shrink.
```

### `v2_backup_te_activity`

```text
'v2_backup_te_activity',  # CANDIDATE (2026-10-22, model audit lead L2, TE part). A TE below the room
# leader is scaled by how often backups with his last
# snap share really play and see targets
# (BACKUP_TE_ACTIVITY_BANDS, from the snap-count table,
# stat line or not); the capacity budget hands the
# difference back to the TE1.
```

### `v2_backup_te_bands_pieces`

```text
'v2_backup_te_bands_pieces',  # CANDIDATE (2026-10-25). Only the bands that still discount on the silent-games
# model (BACKUP_TE_ACTIVITY_BANDS_PIECES), the rest 1.
```

### `v2_backup_te_bands_silent`

```text
'v2_backup_te_bands_silent',  # CANDIDATE (2026-10-25). The backup-TE bands re-read on the silent-games
# model (BACKUP_TE_ACTIVITY_BANDS_SILENT).
```

### `v2_rb_participation_060`

```text
'v2_rb_participation_060',  # CANDIDATE (2026-10-25). The RB participation model at 0.6 strength with silent
# games in the history (RB_PARTICIPATION_SILENT_STRENGTH).
```

### `v2_te_fill_in_role`

```text
'v2_te_fill_in_role',  # CANDIDATE (2026-10-22, the user's fill-in hypothesis). A backup TE whose games
# this season came while a TE now ahead of him was out is
# scaled to his normal-game snap share (or by
# FILL_IN_ONLY_FACTOR if every game was a fill-in); the
# TE1 gets the difference (transfer_te_backup_volume).
```

### `v2_cold_start_room_budget`

```text
'v2_cold_start_room_budget',  # cold start only: clamp the SUM of a team's
# WR / TE room snap shares to what that team
# actually ran at the position last season
# (prior_room_snap_budget) + a tolerance, water-
# filling the excess out of the players the
# Ourlads chart lists as deep reserves first
# (apply_cold_start_room_snap_budget). Built to
# bound the room sum the restoration / Ourlads-
# floor / vacancy stack otherwise leaves unbounded
# (LAC 2026 TE room summed past 2.0). BUILT,
# BACKTESTED, REJECTED - stays OFF. scripts/
# sweep_cold_start_room_budget.py, wk1-2 2022-2025
# (.sweeps/cold_start_room_budget_2022-2025.txt):
# ALL +0.019* / WR +0.047* / START-WR +0.077*
# (all bootstrap-CI-excludes-0 WORSE), for a
# non-significant START-TE -0.042. The clamp
# reshapes ~115 rooms/week and the depth-chart
# protection is not robust enough on the frozen
# historical charts (it water-filled real WR1s
# like 2025 NE Diggs to the floor). Kept
# switchable per repo convention; see
# docs/weekly_projections_methodology.md.
# 2026-09-08 surgical vacancy-recipient guards (scripts/
# sweep_vacancy_recipient_guards.py, wk1 2022-2025, vs the shipped stack;
# all gated to recipients WITH a real prior role so a rookie inheriting a
# departed starter - LaPorta DET 2023 - keeps the full inheritance).
```

### `v2_rookie_backup_wr_dampen`

```text
'v2_rookie_backup_wr_dampen',  # cold start: a NO-PRIOR-ROLE WR the Ourlads
# chart lists at slot rank 2 (a per-alignment
# backup) is pulled to ROOKIE_BACKUP_WR_SHARE
# (~0.06) instead of the 0.16 rank-2 role floor,
# AND gets zero cold-start vacancy weight. Targets
# J.Michael Sturdivant GB 2026 (0.32 snaps / ~1.9
# targets) and ~12 other rookie deep-WRs projected
# as 3-4 target rotation players. BUILT,
# BACKTESTED, REJECTED - stays OFF. scripts/
# sweep_rookie_backup_wr_dampen.py, wk1 2022-2025:
# ALL +0.009 (n.s.), WR +0.013 (n.s.), TE +0.018*
# (CI excludes 0 - allocator knock-on), nothing
# better. ROOKIE_BACKUP_WR_DAMPEN_STRENGTH sweeps
# {0.3..1.0} and {0.1,0.2,0.3}: WR is a flat -0.005
# (never significant), TE turns significantly worse
# by 0.30, startable-WR MEDIAN error is +0.000 at
# every strength, and the docked pool's median
# actual is ~2-5 pts (only 52-65% under 5) - NOT
# the near-zero group the eye-test assumes. Charted
# rank-2 rookie WRs are a real boom/modest-role
# pool (2023 Puka Nacua was one). See
# docs/weekly_projections_methodology.md.
```

### `v2_vacancy_chart_split`

```text
'v2_vacancy_chart_split',  # split the vacancy pool by Ourlads chart rank
# (RANK2/RANK3/UNRANKED weights) so a departed role
# flows to the charted starter. Pool total
# unchanged. Weak alone (Gadsden barely moves,
# pushes the charted TE1 up instead); STACKED with
# bump+growth it made START-TE +0.206* (a
# significant knock) - do NOT combine. OFF.
```

### `v2_vacancy_growth_cap`

```text
'v2_vacancy_growth_cap',  # post-vacancy share <= pre-vacancy share x
# RECEIVER_COLD_START_VACANCY_GROWTH_CAP (1.35),
# MIN_ABS_GAIN floor - the cold-start analogue of
# in-season VACANCY_MAX_GROWTH. Milder than
# bump_cap (Gadsden -> 0.81); similar backtest
# (ALL -0.007, TE -0.042*, startables CI spans 0).
```

### `v2_cold_start_regression`

```text
'v2_cold_start_regression',  # pull every matchup/context multiplier 25%
# of the way back toward neutral 1.0, but ONLY
# at true cold start (Week 1) - see
# COLD_START_MULTIPLIER_REGRESSION's own note.
# Not yet backtested.
```

### `v2_new_team_starter_restoration`

```text
'v2_new_team_starter_restoration',  # let restore_cold_start_returning_role_
# share recover a TEAM-CHANGER's role, not just a
# same-team returner - only when the new team's own
# chart lists him rank 1 and his own active role was
# clearly starter-caliber. Capped lower
# (NEW_TEAM_STARTER_CAP) than a same-team recovery.
# OFF by default; backtest queued 2026-09-04
# (Waddle MIA->DEN, Evans TB->SF cases).
```

### `v2_coaching_aware_defense_prior`

```text
'v2_coaching_aware_defense_prior',  # scale DEFENSE_PRIOR_GAMES per team by
# its off-season coaching-change cohort: HC+DC (or
# lone new HC) reset -> shorter prior leash (~8);
# coordinator-only promotion under a retained HC ->
# longer leash (~18); no signal -> unchanged. See
# data/coaching_changes.py + scripts/analyze_
# coaching_defense_prior.py. Backtest-only; live
# use needs current-season coordinator data.
```

### `v2_defense_prior_games_override`

```text
'v2_defense_prior_games_override',  # sweep hook for DEFENSE_PRIOR_GAMES
# (currently 12.0) - see that constant's own note
# and scripts/sweep_defense_prior_games.py. A no-op
# unless DEFENSE_PRIOR_GAMES_OVERRIDE is also set.
```

### `v2_scheme_alignment_blend`

```text
'v2_scheme_alignment_blend',  # evidence-weighted blend of alignment and
# scheme instead of either replacing the broad
# matchup alone or one outright replacing the
# other - see the note above
# 'v2_scheme_alignment_blend' in this file's
# per-stat loop for the full reasoning. Mutually
# exclusive with 'v2_scheme_matchup' in practice
# (both test different combination strategies);
# if both are set, the blend wins.
```

### `v2_vacancy_before_capacity`

```text
'v2_vacancy_before_capacity',  # run vacancy redistribution BEFORE pass-
# capacity conservation instead of after (see
# the ordering comment above _run_pass_capacity/
# _run_vacancy in build_weekly_projections).
# Built 2026-09-24 on a real reported case (Rico
# Dowdle OUT, PIT, Jaylen Warren double-counted:
# +2 targets from capacity conservation reading
# the RB room as under-budget once Dowdle's own
# row zeroed, ANOTHER +2.2 from vacancy
# separately reassigning Dowdle's own vacated
# targets). BACKTESTED 2026-09-25, NOT SHIPPED:
# INCONCLUSIVE, bias growth +0.359 trips the 0.3
# cap (START-ALL bias -0.337 -> -0.696) despite
# real WR/START-WR RMSE gains (CI excludes 0
# both). Capacity conservation's SYMMETRIC
# budget top-up (pass_capacity_allocator.py) is
# apparently doing real work correcting this
# model's general under-projection tendency
# whenever it runs LAST - running it after an
# already-vacancy-adjusted board loses that. The
# double-count this flag fixes is real and
# confirmed, but this specific fix trades it for
# a worse one. See the dated methodology-doc
# entry.
```

### `v2_wr_te_capacity_split`

```text
'v2_wr_te_capacity_split',  # cold start / weeks 1-2 only: the pass-
# capacity allocator fits the WR and TE sub-rooms
# against separate budgets that split the signed
# off-budget delta by TE_MARGINAL_TARGET_WEIGHT
# (~0.20), instead of one uniform WR/TE factor -
# so a departed/added wideout barely moves the
# tight end (LAC Gadsden, GB Kraft, TB Otton, IND
# Warren). Shipped ON 2026-09-07 by request, then
# PULLED from DEFAULT_FEATURES 2026-09-07 once
# backtested: the full TE_MARGINAL_TARGET_WEIGHT
# sweep (.sweeps/wr_te_capacity_split_2022-2025.txt)
# is a wash on ALL at every weight with a
# significant WR / START-WR cost, and stacking it
# ON TOP of v2_receiver_cold_start_vacancy (the
# "combo" arm) was strictly worse than vacancy
# alone - lost ALL significance and made START-TE
# significantly worse. Kept as an opt-in flag.
```

### `v2_rb_snap_anchored_volume`

```text
'v2_rb_snap_anchored_volume',  # cold-start RB: (1) carry/target split =
# snap allocation + bounded per-snap tilt, not a
# raw prior per-GAME rate; (2) when a charted
# top-3 RB is OUT, concentrate the snap allocation
# toward the lead (vacancy-promoted chart RB3 no
# longer skips the depth nudge; chart RB4+ gets a
# smaller phantom floor). See RB_VOL_* in
# data/rb_role_allocator.py. NOT in
# DEFAULT_FEATURES (see below). Strength of each
# half is a separate 0..1 dial - RB_VOL_TILT_STRENGTH
# / RB_VOL_VACANCY_STRENGTH, env-overridable - swept
# by scripts/sweep_rb_snap_anchored.py.
```

### `v2_defense_blowout_discount_progressive`

```text
'v2_defense_blowout_discount_progressive',  # same discount mechanism and
# magnitude (DEFENSE_BLOWOUT_WEIGHT_DISCOUNT) as
# the flag above, but a richer "was this game
# decided" test: ANY of PROGRESSIVE_BLOWOUT_
# CHECKPOINTS' quarter-end/margin thresholds
# (built 2026-09-15 at the user's suggestion,
# from play-by-play score progression - see
# data.loaders._progressive_blowout_team_weeks),
# not just a single FINAL-margin cutoff. Built to
# test whether a final-margin-only definition was
# missing/mislabeling the specific games driving
# WR's regression above (e.g. a game decided by
# half whose final margin closed under 28 from
# garbage-time scoring). REJECTED 2026-09-16: the
# full 26-week confirm showed EVERY scope's CI
# spanning 0, including START-RB - it washed out
# the real RB win along with the WR loss instead
# of separating them. Mutually exclusive with
# 'v2_defense_blowout_discount' (this one wins if
# both are set - see _blowout_weeks_for_matchup).
# NOT in DEFAULT_FEATURES; stays OFF.
```

### `v2_offense_blowout_discount`

```text
'v2_offense_blowout_discount',  # CANDIDATE - the same DEFENSE_BLOWOUT_
# MARGIN/WEIGHT_DISCOUNT test as the flags above,
# applied to a player's OWN team's blowout games
# instead of his opponent's, inside
# _weighted_player_rates - see
# OFFENSE_BLOWOUT_DISCOUNT_POSITIONS' own comment
# for the full rationale. CHECKED LIVE 2026-09-28
# against the CJ Stroud case it was built for -
# measured EXACTLY INERT: his week-2 game (6-20
# loss to CIN) was only a 14-point margin, under
# DEFENSE_BLOWOUT_MARGIN's 28-point bar (tuned
# for a different, more extreme purpose on the
# defense side), so nothing fired. Kept - not
# wrong, just insufficient on its own for the
# graded, sub-28-point script effect that
# actually drives everyday pass-volume swings;
# see 'v2_offense_script_pool_blend' below for
# the mechanism that actually reaches this case.
# NOT in DEFAULT_FEATURES - backtest before
# shipping, if ever (a real 28+ point game is
# still worth discounting on its own merits).
```

### `v2_game_total_elasticity`

```text
'v2_game_total_elasticity',  # implied game-total scaling ALONE, unbundled
# from 'game_env' (rejected as a bundle, +0.012
# MAE) - see _game_env_multiplier's own note.
# Not yet backtested standalone.
```

### `v2_own_tempo_regression`

```text
'v2_own_tempo_regression',  # UNSHIPPED 2026-10-06 (harness INCONCLUSIVE, no effect: START-ALL RMSE +0.002). Passing and receiving volume
# (not rushing) is scaled by 1 - kappa(G) * (the team's
# own plays per game / league - 1), kappa = 0.65 * max(0,
# 1 - G/10): a team's own tempo regresses to the mean
# (~0.37 persistence) but the per-game rates carry it in
# full, so fast teams were over-projected (~6% on pass
# attempts, t = -3.6) and slow ones under. See
# own_tempo_multiplier and TEMPO_*.
```

### `v2_rb_game_script`

```text
'v2_rb_game_script',  # NOT SHIPPED (2026-11-01, backlog L5): v2_rb_game_script_rush plus RB receiving
# yards scaled down with the margin (RB_GAME_SCRIPT). Passed the
# harness, but the receiving term made RB receiving yards worse
# (held out 5 of 5 seasons; harness 3 of 4), so it stays off.
```

### `v2_venue_mult`

```text
'v2_venue_mult',        # indoor/outdoor venue scaling ALONE, same
# unbundling as above. Not yet backtested
# standalone.
```

### `v2_channel_matchups`

```text
'v2_channel_matchups', # per-route-channel matchup read - never built (no gating code); out of DEFAULT_FEATURES 2026-10-24
```

### `v2_keep_fully_excluded_history`

```text
'v2_keep_fully_excluded_history',  # CANDIDATE (2026-10-25, user: Ja'Kobi Lane missing from the week 7
# board). A RB/WR/TE whose every game this season the
# partial-game filter excluded keeps those games as his
# history instead of dropping out of the in-season pool
# (_keep_fully_excluded_players). HARNESS 2026-10-25: misses
# its bar (START-ALL +0.002 vs +0.001; veterans whose only
# game was cut short lose their prior-season line).
```

### `v2_points_scale_v6`

```text
'v2_points_scale_v6',  # CANDIDATE (2026-10-25, user: "Refit the points scale"). The points lines, the low
# end and (if re-fit) the season anchor's s from one re-fit
# on the current model (POINTS_SCALE_V6 / points_scale).
```

### `v2_qb_passing_td_k`

```text
'v2_qb_passing_td_k',  # TESTED 2026-10-06, INCONCLUSIVE, left UNSHIPPED (harness v2: START-QB -0.010 CI spans 0,
# START-ALL -0.002, better in 25 of 60 weeks). K=15 (shared 5) for QB passing_tds: a
# slower blend of the QB's own current-season TD rate. See QB_PASSING_TD_K.
```

### `v2_qb_style_blend_k80`

```text
'v2_qb_style_blend_k80',  # CANDIDATE (2026-10-21, calibration audit C13 follow-up). A QB1's
# own pass/rush split shrinks toward the league mean with
# K = 80 dropbacks instead of 10 (see
# data/qb_volume_blend.QB_STYLE_BLEND_K_CANDIDATE).
```

### `v2_rb_carry_budget_lead`

```text
'v2_rb_carry_budget_lead',  # CANDIDATE modifier of v2_rb_carry_budget:
# split the room's change in proportion to
# carries^2 so the lead back absorbs most of it
# (the leftover room excess is ~all RB1). No
# effect without v2_rb_carry_budget.
```
