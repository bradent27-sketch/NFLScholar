# Improvement backlog (started 2026-10-19, updated 2026-10-19 after the first reads)

One list of everything worth doing to make the weekly model more accurate, more trustworthy or more useful. Each item has an
ID, the evidence that put it here, the data it needs, a rough size, and a status. Update the status and add the methodology
entry date when an item moves. This file replaces the "Model leads" section of `docs/open_items_2026-10-07.md` as the place to
look (that file still holds the history and the closed-item table).

**Status key:** TODO, IN PROGRESS, SHIPPED, REJECTED (tested, not worth it), PARKED (waiting on something).
**Rules that apply to every item:** one named flag per model change; harness v2 (played-only, 2022-2025 weeks 3-17) is the
gate; a pre-set bar is written before the run; shipping into `DEFAULT_FEATURES` needs the user's sign-off; no model+market
blend; every ship or reject gets a dated entry in `docs/weekly_projections_methodology.md`.

## Baseline (2026 live, weeks 3-6, each player's last pre-kickoff board, same players for both sides, startable pools)

| | model | market | FantasyPros |
|---|---|---|---|
| all positions, RMSE | 7.39 | 7.12 | 7.21 |
| WR | 7.82 | 7.34 | 7.37 |
| TE | 7.46 | 7.26 | 7.30 |
| RB | 7.05 | 6.96 | 7.20 |
| QB | 6.70 | 6.67 | 6.76 |

Where it comes from: the receivers. WR ordering is right 62% of the time vs the market's 66%. Re-score with the Track Record page
(Weekly Fantasy > Track Record) or `data/ledger_scorecard.py`; never with the raw `scripts/score_ledger.py` output.

---

## A. Data bugs and hygiene (cheap, do first)

| ID | Item | Evidence | Size | Status |
|---|---|---|---|---|
| D1 | **Injury-report cache frozen.** `load_injury_reports` cached the live season forever. | 2026 file stopped at week 4 (341 h old) while the feed had week 6; the QB returning-starter rule reads it for weeks before the target week. | small | SHIPPED 2026-10-19 (6 h refresh for a partial season, same as the roster cache; 3 tests) |
| D2 | **Refresh the injury feed after inactives** (about 90 min before kickoff) before the last board build. | Week 5: Chase and McLaurin were Out on the last pre-kickoff board and played (19.7 and 12.2 pts). | process | SHIPPED 2026-10-19: a note under the Build board button (ui/freshness.py): injury file age against the next kickoff, a warning inside 5 h when the file predates the inactives window; 6 tests |
| D3 | **PFF weekly manifest mislabel.** Row says week 6 (2026-10-13), folder is `weekly/5`. | The loader goes by folder name, so the model reads it correctly; the manifest row is wrong. | tiny | HARMLESS, needs the user to OK a one-cell edit of pff_imports/2026/weekly/manifest.csv (row 6 should say 5; the 10-13 export is week 5). Today week 5 reads as unverified-but-allowed. |
| D4 | **Name-match fragility.** `load_player_id_crosswalk` (pff_id <-> gsis_id, 100% verified) is wired in at none of ~15 call sites. | Measured 2026-10-19: with the app's two-tier name match every PFF receiver (20+ routes) matches nflverse, 300 of 300 in 2025 and 199 of 199 in 2026 (the exact tier alone misses 8 and 5, all suffix cases such as Jr. and III, which the second tier fixes). Insurance only. | medium | LOW PRIORITY (do it with N1 if that is ever built; nothing is failing today) |
| D5 | **Suspended player with no board row.** | Rashee Rice 2025 (suspended weeks 1-6) has no board row in weeks 7-8, his first games back. | small | CHECKED 2026-10-19: live path is fine (Jeshaun Jones got a row the week his suspension ended); looks like a backtest-only artifact of D6a (end-of-season roster placeholder). Fold into D6. |
| D6 | **Backtest-only leaks.** (a) the default roster placeholder reads a player's later-week roster status; (b) other call sites group by the merged traded-player `team` column (the pace proxy is already fixed). | Changes harness base arms, never live boards. Quantify before touching. nflverse `load_trades` has the dates for (b). | medium | TODO |
| D7 | **Retractable roofs with no posted roof status** are treated as outdoors. | Week 5: TB @ DAL read 79.6 F, dry. Retractable stadiums were closed in 143 of 167 home games 2021-25 (68 of 68 in 2024-25). | small | SHIPPED 2026-10-19: a blank roof at ARI/ATL/DAL/HOU counts as closed (data/weather.py game_is_outdoor; a posted roof always wins; 3 tests). Live path only. |
| D8 | **Floor/ceiling ("Range of outcomes") never checked live**: bands were fit on 2021-2023 under an older model and the ledger does not save them. | Used for start/sit. | small | CHECKED OFFLINE 2026-10-19 on the lab dump: Starter-tier bands are calibrated, held-out 2024-25 included (below P10 8-12%, P25 22-25%, P75 75-77%, P90 90-92%). Only the RB bench lower tail is optimistic (19-22% below P10). No refit. Saving bands in the ledger is optional. |

## N. New or underused data (free, already loaded or one call away)

| ID | Item | Why | Source | Size | Status |
|---|---|---|---|---|---|
| N1 | **Expected stats (`ff_opportunity`): usage-vs-results correction.** | Luck (actual minus expected points, earlier games this season) predicts the board's miss: WR -0.18 per point (5/5 seasons), TE -0.16 (4/5), QB -0.16 (held-out version worse), RB none. Re-confirmed 2026-10-20 on the current-model board. Held out by season, WR/TE only, no games shrink: START-ALL -0.0055 [-0.013, +0.001], START-WR/TE -0.010, ALL -0.0045 [-0.009, -0.000], 4/4 seasons. Caveat: nflverse's expectation model is in-sample for past seasons; 2026 is the clean check. | `nflreadpy.load_ff_opportunity` | medium | SHIPPED 2026-10-20 (user: "turn it on"), DEFAULT_FEATURES = 59. Harness: START-ALL -0.007 (CI [-0.015, +0.000]), START-WR -0.013, START-TE -0.010, QB/RB 0.000, bias growth +0.032, verdict INCONCLUSIVE on a met bar. |
| N2 | **Routes run and targets per route run (WR/TE).** | Our PFF exports already have `routes`, `route_rate`, `yprr`, but they only feed role confidence. **First read 2026-10-19: no signal.** On 4,145 WR/TE rows (PFF weekly 2024-2025) a hot targets-per-route rate is not over-trusted (WR slope +0.09, i.e. slightly under-trusted; TE -0.10 n.s.; points nothing). Only 2 seasons of weekly routes exist locally. | PFF exports (live); nflverse `load_participation` for more history | - | REJECTED for now (`scripts/eval_routes_targets.py`); revisit only if more PFF weekly history is added |
| N3 | **Weekly depth charts.** | 141 backtest team-weeks have no resolved QB1; backtests only have a real depth chart in week 1; could cut manual QB1 overrides live. **Tested 2026-10-19: no.** The depth chart lists injured starters as QB1: right 87% (2025) and 90% (2026, weeks 2+) against 88% and 91% for simply taking the last game's starter; on the 76 team-weeks the last-starter rule misses it is right on 31 and wrong on 38 of the 587 it gets. | `nflreadpy.load_depth_charts` (official weekly through 2024, ESPN daily from 2025) | - | REJECTED as a QB1 source |
| N4 | **Trade dates.** | Fixes the traded-player team leak (D6b) cleanly. | `nflreadpy.load_trades` | small | TODO |
| N6 | **Pass rate over expected (PROE)** from play-by-play `xpass`. | First read 2026-10-19: no signal. Team PROE to date predicts the board's team pass-attempt miss at 0.015 attempts per point (corr 0.01); prior-season PROE n.s. | `load_pbp` (already loaded for big plays) | - | REJECTED (`scripts/eval_proe.py`) |
| N7 | **Days of rest / short weeks** from the schedule. | First read 2026-10-19: nothing consistent; post-bye WR -0.6 pts (about 2.4 sigma, one of 16 cells). | schedule `home_rest`/`away_rest` | - | REJECTED |
| N5 | **FTN charting** (drops, catchable balls, play action) and **Next Gen** (separation, cushion). | Interesting but second order; FTN only from 2022. | `load_ftn_charting`, `load_nextgen_stats` | medium | PARKED (skip for now) |

## M. Model leads

| ID | Item | Evidence | Size | Status |
|---|---|---|---|---|
| M1 | **QB passing volume tracks the league environment.** | 2026 teams throw 33.7 passes a game vs 32.1-32.7 in 2024-25; starting QBs were projected 31.5 attempts and threw 33-36 (weeks 3-6; FantasyPros misses the same way). **Tested 2026-10-19: history does not support it.** The model's QB attempt miss does not follow the league shift (correlation 0.12), league weekly passing level has no persistence (0.13), and the model is 2-3% HIGH on QB attempts in every past season. 2026's weeks 3-6 are a surge (34.9 per team vs 31.5 in weeks 1-2). | existing box scores | - | REJECTED, with a WATCH (W1) |
| M2 | **Current-season weights, per stat.** | Swept 2026-10-20 (capture of every stat's blend inputs, 36,865 player-weeks, exact re-score). The WR-targets hint is not there; faster current-season weights hurt receptions and yards in 0 of 5 seasons; slower weights help single groups (TE yards x2 -0.011, 5/5 seasons) but a held-out joint choice is worse than no change (+0.007 START-ALL), so it is selection noise. | medium | REJECTED as a broad lever. Spin-off parked: TE receiving_yards K x2 (at most -0.002 START-ALL, not run). TDs are M3. |
| M3 | **TD levels on the final defaults.** | Re-measured 2026-10-19 on the lab dump (START pool, played, 2021-25, from the shipped stack): levels are right on average, WR +4.8%, TE -4.2%, RB 0.0%, QB +2.3%, and flat across weeks 3-17. By season they swing with luck (WR 2021-22 +15/+21%, 2024 -9.5%; TE 2025 -23%: 0.380 scored vs 0.293 projected). What is off is dispersion: top-quintile TD projections run 20-43% high (WR +25%, RB +20%, TE +43%), bottom quintile 30-41% low; slope of actual on projected 0.33-0.64. TD compression is already closed for points. | small | REJECTED for levels (nothing to ship); the dispersion goes to P1. W3 watches TE TDs. |
| M4 | **QB passing yards at the top still run high.** | Measured 2026-10-19 (1,918 starter-weeks): projected 275-300 runs +13.5 yards (+5%), 300+ runs +36 (+13%, only 24% of those QBs beat the projection), under 200 runs -12 (-6%). Slope of actual on projected 0.72. A top-end shrink (knot 225, share 0.3) improves yards RMSE by 0.75 of 76 (1%) in 5 of 5 held-out seasons. QB points are untouched by design (the points line already absorbs it); this only matters for passing-yard props. | small | MERGED into P1 (display-only calibrated stat lines), waiting on the user's call about the prop view |
| M5 | **Season debuts after an injury.** | A zero-game player gets a partial role (Flowers wk4 52% snaps vs 86% last season). By reason: suspensions under-projected (shipped), PUP/NFI debuts ran 0.44x, IR debuts 1.32x, too mixed to fit on 34 rows. | small | PARKED (needs more rows) |
| M6 | **Ranked fallback for teams with no resolved QB1.** | Candidate flag `v2_qb1_probable_starter_fallback` (sole available QB 1.0; several weighted 0.62/0.35/0.03 by games played; team stays flagged as selection-required). Live case: PHI week 6 (Hurts Out, Dalton 0.0). Harness 2022-25 wk3-17: START-QB -0.126 (CI [-0.255, -0.023]), START-ALL -0.028 (CI [-0.051, -0.010]), START-WR -0.009, no position worse, bias growth -0.007; SHIP-ELIGIBLE, pre-set bar met in full. The backtest has no overrides, so the live gain is smaller. | medium | SHIPPED 2026-10-20 (user: "Turn on QB fallback"), DEFAULT_FEATURES = 58. |
| M7 | **`v2_venue_mult` never backtested.** | Harness 2026-10-20: START-ALL 0.000 (CI [-0.008, +0.008]), START-QB +0.014, START-TE +0.011, START-WR -0.006; bar (START-ALL <= -0.003) not met. | one harness run | REJECTED (inconclusive, stays unshipped). |
| M8 | **RB2 rushing yards** still about -2.8 per game. | `v2_rb_carry_budget_lead` tested worse than uniform. | small | PARKED |
| M9 | **Per-player boom/bust.** | First read 2026-10-19 on the Range-of-outcomes bands (Starter tier, 7,479 rows, weeks 5-17, 2021-25): the plain tier bands are calibrated (below P10 10.9%, P90 91.3%). The per-player width scaling (own variance vs the tier's) is WORSE: mean pinball loss 2.097 vs 2.076, below-P10 15.4%, P90 88.0%. Own variance has only a weak real signal (correlation 0.03 with the size of the miss; realised spread of actual/shown rises 14% from the lowest to the highest own-variance quintile) and the scale averages 0.873 with sd 0.23, so bands run about 13% too narrow. Recentred to mean 1 with a quarter of its spread the loss is 2.075, a tie with the tier bands that fixes the narrowing. | small | TODO, needs the user's OK (it changes the displayed floor and ceiling; points untouched): recentre and shrink `player_width_scale`. A real per-player spread model is still open but the data says own variance alone will not carry it. |

## P. Prop and presentation leads (new 2026-10-19; these need the user's call before any build)

| ID | Item | Evidence | Size | Status |
|---|---|---|---|---|
| P1 | **Calibrated stat lines for props (display only).** The points are calibrated by the points line; the stat lines a prop bettor reads (yards, receptions, TDs) are not, and they are over-dispersed. | Slope of actual on projected, START pool 2021-25 (re-measured 2026-10-20 on the current model): QB passing yards 0.71, QB attempts 0.73, QB rush yards 0.80, RB rush yards 0.95 and WR yards 0.93 (top-end yardage already fixed most of those), RB/TE receiving yards 0.70/0.76, receptions 0.76/0.82/0.75, TDs 0.33-0.72. A fitted line per stat (leave-one-season-out) cuts stat RMSE by 0.3-1.1% (QB pass yards -0.85 of 74, 5/5 seasons; QB rush yards -0.29 of 17.9, 5/5; RB/WR/TE yards 4/5 to 5/5 but about 0 MAE gain). Small for accuracy, larger for edge-hunting: the tails (300+ passing yards, top-quintile TDs) are where projection minus line is biggest and least real. | medium | PARKED until the user wants the prop view (deferred idea 2): include calibrated lines and over/under probabilities there. Points and ranking stay untouched. |
| P2 | **User-deferred ideas:** 2 prop view, 3 Sleeper "my team", 4 "why" column. | - | - | WAITING on the user |

## C. Calibration audit (2026-10-20): factors and weights never tuned on the current harness

Read from the code comments, the sweep outputs in `.sweeps/` and the methodology doc. "Old sweep" = tested once in Aug-Sep 2026 on the
pre-v2 harness (MAE, narrow windows, an older model): not comparable to today's gate. Nothing here has been started.

| ID | Factor (value) | What it decides | Status |
|---|---|---|---|
| C1 | **PFF alignment WR/TE matchup**: shrinkage 4 games, no recency weighting, raw clip 0.50-1.75, final clip 0.75-1.30, event half-weight 20, confidence half-weight 60 (`data/pff_alignment.py`) | The WR/TE targets/receptions/yards matchup wherever PFF weekly data exists (2024-25 backtests, every 2026 live board) | CLOSED 2026-10-20 (WR): no robust gain on 2024-25 (best -0.0021, CI spans 0; cross-season -0.0012 / +0.0011). Revisit with more PFF seasons. |
| C1b | **PFF scheme (man/zone) TE matchup**: shrinkage, prior-season weight 0.40, clip | The TE receiving matchup wherever scheme evidence exists | NOT TESTED: the alignment capture showed TE uses the scheme multiplier; same hook approach would cover it (2024-25 only). |
| C2 | **DEFENSE_PRIOR_GAMES 12** used twice | (a) neutral league-average games mixed into every defense profile, (b) how fast this season's defense replaces last season's | CLOSED 2026-10-20: verified on the current model (defense-variant capture); no setting beats shipped beyond noise (methodology 2026-10-20). |
| C3 | **Defense recency per game** (reuses RECENCY_DECAY 0.85) | How much recent defensive games outweigh older ones | CLOSED 2026-10-20: verified on the current model (defense-variant capture); no setting beats shipped beyond noise (methodology 2026-10-20). |
| C4 | **PRIOR_SEASON_DEFENSE_RECENCY_FLOOR 0.80** | How much last season's late games outweigh its early games | CLOSED 2026-10-20: verified on the current model (defense-variant capture); no setting beats shipped beyond noise (methodology 2026-10-20). |
| C5 | **Matchup strength** (multiplier applied at full strength) and **MATCHUP_CLIP 0.82-1.22** | How much of a defense rating reaches the projection | CLOSED 2026-10-20: verified on the current model (defense-variant capture); no setting beats shipped beyond noise (methodology 2026-10-20). |
| C6 | **Role matchups**: ROLE_MATCHUP_K 10, ROLE_MIN_EVENTS, tercile buckets | Possession vs deep receiver, receiving vs rushing back | K old sweep, no signal; the rest never tested |
| C7 | **Blowout discount** (28 points, x0.5) and **OFFENSE_PRIOR_GAMES 6** (RB only) | Decided games in a defense's history; the early-season offense baseline | Old sweeps |
| C8 | **Pace**: PACE_PRIOR_GAMES 12, PACE_CLIP 0.85-1.15 | Opponent pace effect | Prior copied from the defense value, clip never tested (the 0.15 cap is measured) |
| C9 | **Within-season recency RECENCY_DECAY 0.85** | How much a player's recent games outweigh earlier ones | Old sweep, no signal |
| C10 | **K_EFFECTIVE_RANGE 0.7-1.3** | Role confidence speeding up / slowing trust in this season's rate | SHIPPED 2026-10-20 (user: "yes ship it"), DEFAULT_FEATURES = 60, as `v2_role_confidence_reversed_wrte` (WR/TE receiving volume mirrored). Offline -0.0038 [-0.0066, -0.0011] 4/5; harness SHIP-ELIGIBLE, START-ALL -0.004 [-0.007, -0.001], WR -0.008, TE -0.007, QB/RB 0.000, bias +0.002. |
| C11 | **ROLE_CHANGE_K_REDUCTION 0.5** | How much a detected role change speeds trust in new volume | CLOSED 2026-10-20: no effect (WR/TE target and QB carry RMSE flat at every reduction 0-1; 2.8% of rows touched). |
| C12 | **Two-seasons-back blend** (base 0.20, cap 0.55, damping 0.35, 8-game fade; the QB blend's own copy) | How much a season two years back counts | Never tested |
| C13 | **QB volume/efficiency blends** (volume K 60, pass efficiency 100, rush efficiency 5, rush TD 40, style 10, starter floor 20/game) | Team vs personal QB volume and efficiency | No sweep on record |
| C14 | **TD priors**: two-year TD weight 0.30 and band 0.60-1.67, credibility K (130/55/200), one-season cap 0.70, longevity bonus; xTD event K 6/8, prior TD weight 0.7, share K 4 | Touchdown rates | Credibility K adjusted once after an ablation; the rest seeded or hand-set (xTD scoring rates themselves are fitted) |
| C15 | **Partial-game and blowout exclusions** (0.65 / 0.50 / 0.60 snap shares; 28 points, 0.65 / 0.75) | Which past games count in a player's rate | Hand-set |
| C16 | **Injured-teammate redistribution** (survival 0.85 / 0.80 / 0.80, growth cap 2.0) and cold-start vacancy weights | Where an injured player's volume goes | Mostly hand-set (some guards swept) |

Already calibrated (no action): points calibration lines v4, season anchor, Questionable cut, opening-week lines, top-end yardage, wind,
per-stat game-total elasticity, WR/TE and RB participation, RB carry budget, QB TD blend, rematch weight (measured twice), pace cap,
per-stat current-season weights (STAT_K, M2).

Plan when approved: (1) a defense-profile capture (one ~80 min run) so C2-C5 and C8 re-score offline in minutes each, winner to the
harness; (2) C1 the same way on 2024-25; (3) C10-C11 from the existing M2 capture; (4) C12 and C14 with a small extra capture;
(5) C15-C16 as on/off checks.

## W. Watch items (nothing to build yet)

| ID | Item | Trigger |
|---|---|---|
| W1 | **2026 passing surge.** Starting QBs out-threw their projections by about 8% in weeks 3-6 (league attempts per team: weeks 1-2 31.5, weeks 3-6 34.9). History says league weekly passing does not persist. | Re-check on the Track Record page through week 10; if starters still beat their projections by over 5%, test in-season recency weights for volume (M2). |
| W3 | **TE touchdowns** ran 10% (2024) and 23% (2025) above projection after being near even in 2021-23 (2025: 0.380 per game scored, 0.293 projected). | If the 2026 live weeks (Track Record page, stat columns) show TEs scoring over projection again, re-check the TE TD level. |
| W2 | **Post-bye WRs** were 0.6 points over-projected (2021-2025, about 2.4 sigma, one of 16 cells). | Only if a second cut (a different window or the live weeks) shows it again. |

**Housekeeping (2026-10-19):** the lab dump `.sweeps/calibration_lab_v7.csv` (built 10-08) no longer matches the current model; new analyses should read
`.sweeps/stat_blend_traces.parquet.board.csv` (written by the M2 capture, same columns). TODO when the capture finishes: re-run the M3, M4/P1, D8 and
M9 checks on it and correct their numbers here if they move.

## Closed (do not re-run without new evidence)

Model+market blend (rejected permanently); first-game-back factor for skill players (2026-10-13, too rare); per-stat and segment
recalibration in season (22 schemes, 2026-10-08); yards-per-opportunity shrink (2026-10-06); TD compression; the old Questionable
play-probability haircut; `v2_own_tempo_regression`; `v2_pace_alpha_cap` ablation. Full table in
`docs/open_items_2026-10-07.md` section E.

## Shipped this fall (for reference)

2026-10-19 data fixes (not model flags): injury-report cache refresh (D1), retractable-roof default (D7), injury freshness note (D2).

`v2_role_confidence_reversed_wrte` (2026-10-20, `DEFAULT_FEATURES` = 60); `v2_expected_points_luck` (2026-10-20, `DEFAULT_FEATURES` = 59); `v2_qb1_probable_starter_fallback` (2026-10-20, 58);
`v2_top_end_yardage`, `v2_suspension_return_full_role`, `v2_qb1_latest_game_starter` (2026-10-13/14, `DEFAULT_FEATURES` = 57);
`v2_opening_week_stat_calibration`, `v2_questionable_workload` (10-08); `v2_rb_depleted_room_fill` (10-07);
`v2_season_anchor`, `v2_qb_td_blend` (10-06).
