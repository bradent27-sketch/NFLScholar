# Open items and loose ends (swept 2026-10-07, after the "Week 4 Update" push)

**Start here when picking the project back up.** This is the current list of what is unfinished, undecided or worth
doing next. It was built by going back through every message the user sent (2026-09-08 to 2026-10-07, 104 distinct
requests), the methodology doc, the backlog, `HANDOFF.md`, the code comments and the flag list, and checking each
item against the code. `docs/weekly_rankings_backlog.md` (Aug 2026) and the "Current direction" text in `HANDOFF.md`
are older and partly stale; where they disagree with this file, this file wins.

## State at the sweep

- Work since the sweep sits on branch `open-items-2026-10-07` (LOCAL commits, not pushed; `PreNFL2026_v.02` on origin is at
  `001ad1b`, `main` was never touched). 762 tests pass. Nothing is running.
- `DEFAULT_FEATURES` = 63 flags; `MODEL_FEATURES` = 96 (as of 2026-10-22: `v2_qb_implied_td_boost`, `v2_fallback_prior_fast_trust` and `v2_vacancy_lead_share_050` shipped from the error-first audit, docs/model_audit_2026-10-21.md; candidates `v2_no_history_exclusions` and `v2_qb_style_blend_k80` stay off). The model leads now live in `docs/improvement_backlog.md`. Calibration lines are **v4** (kept three times: 10-01, 10-06, 10-08).
- Newest ships: `v2_season_anchor` (10-06) and `v2_qb_td_blend` (10-06). Both shipped on a pre-set bar although the
  harness's own verdict was INCONCLUSIVE. Undo = remove the flag name from `DEFAULT_FEATURES`.
- Standing rules: every model change is a named flag; a `DEFAULT_FEATURES` change needs sign-off or a win on a bar written
  down first; heavy backtests one at a time; every ship or reject gets a dated methodology entry; harness v2 (played-only)
  is the gate; no model+market blend, ever (market and FantasyPros are benchmarks only); never delete anything in
  `data/ledger`; do not push to `main` unasked. Reading logs/results is pre-authorized.

---

## A. Time-sensitive (do these first)

1. **Week-5 ledger: RESOLVED at the time of this sweep, keep an eye on it.** `data/ledger/2026_wk05_20261005T234357970352Z.parquet`
   was built Monday evening and carries WEEK-4 market columns (identical for 342 of 342 players). A fresh board was
   built later (`2026_wk05_20261007T115209952873Z.parquet`, untracked like the rest of the ledger) and its market columns
   are week 5's (15 of 266 identical to week 4, i.e. coincidences). `scripts/score_ledger.py` scores the *latest* file
   per week, so week 5 now scores against the right market. Repeat for every new week: build the board AFTER the Tuesday
   15:00 UTC lines are up before the first kickoff (the ledger writes at most once an hour; the app blanks a market
   snapshot that predates the week via `data.odds_weekly.snapshot_is_for_week`). Never delete the old file.
   *Small hardening, not built:* have `score_ledger.py` warn or refuse when a file's `Mkt Market Pts` equals the previous
   week's.
2. **After week 5 is played:** refresh the 2026 data, then `python scripts/review_live_season.py` and
   `python scripts/score_ledger.py --year 2026`. Weeks 3-4 were model 6.30 / market 6.49 / FantasyPros 6.84 RMSE on 572
   player-weeks. Also look at the week-5 role-call gaps flagged on 10-06 and see who was right: model far BELOW the
   market on Kamara (5.5 vs 13.3), Love (9.8 vs 16.5), Braelon Allen (6.2 vs 11.8), Addison (8.4 vs 13.9); far ABOVE on
   Lamb (24.1 vs 17.5), Coleman (10.1 vs 4.8), Kyren Williams (21.1 vs 15.7). Several are returning-from-absence or
   small-sample QB cases; if the model is wrong in one direction the same way, that is a lead.
3. **Watch the two newest ships live.** `v2_qb_td_blend` (K=15 plus prior TD rate regressed 25% to 1.46): QB TD counts
   should now move slowly and be ~23% less spread out. `v2_season_anchor`: WR/RB/TE projections are pulled toward the
   player's own to-date average (mean change 0.2-0.5 points, QB untouched). If either looks wrong on real boards, the
   undo is one line; if kept, nothing to do.
4. **Decision for the user: the repo is PUBLIC** (`bradent27-sketch/NFLScholar`, checked 2026-10-07) and licensed PFF
   subscription exports are committed (`pff_imports/2025/weekly`, and `pff_imports/2024/weekly`, 67 tracked files, by the
   user's 2026-09-01 choice). The backlog said to spend "one minute" on making the repo private before pushes. Not changed;
   it is the user's call.

5. **`v2_rb_depleted_room_fill` is SHIPPED (2026-10-07, DEFAULT_FEATURES = 52).** Harness run 2: RB RMSE -0.014 (CI excludes 0),
   START-RB -0.019, START-ALL -0.006, bias growth -0.016, QB/WR/TE unchanged. Live PHI: Shipley 4.6 -> 12.8 carries, the three
   practice-squad backs 0.68 carries each ("Practice squad" in the Injury column). Undo = remove the flag. Details and the
   discarded first run are in the methodology doc, 2026-10-07 entry. The RB calibration re-check was done on 2026-10-08
   (see item 6): the v4 lines stay.

6. **Calibration rework (2026-10-08, methodology entries of that date): DONE, two flags SHIPPED** (user: "both sound good,
   commit them"; DEFAULT_FEATURES = 54).
   - `v2_opening_week_stat_calibration`: week 1 (RB/WR/TE) and week 2 (RB) get a calibration line per stat; the board ran +1.8
     pts high on startable players in week 1. Held out: week-1 START-ALL RMSE -0.223 (4/4 seasons), week 2 -0.058 (5/5). No
     effect from week 3 on, so nothing before 2027 week 1. Undo = remove the flag.
   - `v2_questionable_workload`: a Questionable WR who plays produces ~17% less than projected (0.83 of projection vs 0.99
     untagged; 2021-2025 final Friday designations, 4 of 5 seasons; RB/TE/QB not distinguishable from 1). WR-only x0.83 on the
     final points, last step, availability untouched, not a play-probability haircut. Harness SHIP-ELIGIBLE (START-ALL
     -0.011, START-WR -0.027, bias growth +0.066). Undo = remove the flag. **Open caveat: the history is the final
     (Friday) designation; an early-week tag is untested (one report row per player-week is kept). Worth re-checking on
     live 2026 data once a few more weeks of Questionable WRs have played** (compare Q-tagged WR actual/projection on the
     ledger boards: `scripts/diag_questionable_live.py`; only 14 WR-games so far).
   - In season (weeks 3-17) no recalibration beats the shipped v4 lines + season anchor, including per-stat lines (22
     schemes tested held out). Re-check with the lab dump (`scripts/calibration_lab.py`, `scripts/eval_calibration_schemes.py`)
     after future stat-pipeline ships instead of the old `fit_seasonal_calibration.py` dump. The harness base arm now
     includes the Questionable cut (see the methodology entry).

7. **2026-10-13 - top-end yardage, Track Record page, live record corrected** (methodology entries of that date).
   - `v2_top_end_yardage` (receiving / RB rushing yards above 60 count in part; QB untouched): harness **SHIP-ELIGIBLE**
     (START-ALL -0.006, CI [-0.012, -0.000]; START-WR -0.014, CI excludes 0), pre-set bar met in full. **SHIPPED 2026-10-13
     (user: "Yes, add in top-end yardage"), DEFAULT_FEATURES = 55.** Undo = remove the flag.
   - **Correction to item 2 above:** the "model 6.30 / market 6.49 / FantasyPros 6.84" live comparison was wrong twice over
     (`score_ledger.py` scored FantasyPros in standard scoring against PPR actuals - fixed; and each source on its own
     pool). Fairly scored (each player's last pre-kickoff board, paired pools, startable, weeks 3-5) the model trails:
     7.60 vs market 7.32, 7.78 vs FantasyPros 7.44, behind both in every week. Use Weekly Fantasy > Track Record
     (`data/ledger_scorecard.py`) for the live record from now on.
   - **Strongest live lead:** players back from missed games (Flowers, Bowers, Adams, Metcalf, Waddle) projected well
     under the market and scoring like the market said. Also: refresh the injury feed after inactives before the last
     build (week 5 Chase / McLaurin were Out on the last pre-kickoff board and played).
   - QB passing yards at the top still run high on the board (275+ about -7%); left out of the top-end flag because it
     pushed the already under-projected top QBs further down.

8. **2026-10-14 - returning players** (methodology entries 2026-10-13 night and 2026-10-14).
   - **`v2_qb1_latest_game_starter` SHIPPED 2026-10-14 (user: "yes add QB starter fix"), DEFAULT_FEATURES = 57.** Harness
     START-QB RMSE 7.910 -> 7.390 (-0.519, CI [-0.691, -0.356]), START-ALL -0.092 (CI [-0.122, -0.063]), 34-3 weeks; QB1 pick
     correct 83.4% -> 86.4% over 2021-2025. A starter back from injury was projected 0 in his second game back (picker tie
     with the fill-in). No live effect week 6 (every team already has a QB1). Undo = remove the flag.
   - **`v2_suspension_return_full_role` SHIPPED (user decision), DEFAULT_FEATURES = 56.** Verified end to end on the 3 return
     weeks in 2021-2025; no live effect this week. Roster cache staleness fixed (6 h refresh).
   - **`v2_qb1_probable_starter_fallback` SHIPPED 2026-10-20 (user: "Turn on QB fallback"), DEFAULT_FEATURES = 58.** An unresolved QB room
     with a QB not listed out gets a probability-weighted starter instead of zeros (PHI week 6). Harness START-QB -0.126, START-ALL -0.028,
     no position worse. Undo = remove the flag.
   - Rejected: a "first game back" factor for skill players (too rare, -0.001). Open, not built: season debuts after injury
     (PUP/NFI debuts ran 0.44x their projection, IR debuts 1.32x, too mixed to fit), and a suspended player who has no row on the
     board at all.

## B. Model leads, ranked by (expected value / effort)

Context for all of them: at the level the model is near its ceiling (RMSE within 0-2% of an oracle that knows each
player's season average), weekly points are mostly noise, and every ship this month was worth 0.1-0.3% of RMSE. Expect
small wins. The large unexplained variance is touchdowns and boom/bust games.

1. **Re-audit receiving and rushing TD levels on the final defaults.** The TE-room audit (10-05) found WR/TE receiving TDs
   short (TE 247 vs 300, WR 669 vs 749, short in all four years); `v2_pass_capacity_keep_tds` (shipped 10-06) and
   `v2_xtd_rush_outside_zone` were the fixes, but TD levels were never re-measured after the full stack plus
   `v2_season_anchor`. START-TE bias was still about -0.5. Re-run `scripts/diag_receiver_audit.py` and
   `scripts/analyze_te_room.py` (28 boards, 2022-2025). Effort: small. Also the only calibration line with any support
   is a TE two-bucket refit (RMSE 6.909 -> 6.851 on the old dump) that did not repeat across splits.
2. **Ranked fallback for the ~10% of team-weeks with no resolved QB1** (188 of 1,790 in 2022-2025; e.g. ATL and MIN had
   zero pass attempts in 2025 wk9). Live, the board shows these for a manual pick, so it is a lead and not a bug:
   a "most likely starter" guess probably beats projecting zero, in the backtest and on the live board. Related: the
   old QB1 cold-start 0.0 case (backlog sec. 5: rookie starter or a veteran off an injury-shortened season) is only
   partly covered by `v2_qb1_returning_starter`. Effort: medium (QB1 selection touches every QB).
3. **Reuse the capture-then-sweep tool for other blend constants.** `scripts/capture_qb_td_traces.py` (one ~80 min
   capture) plus `scripts/eval_qb_td_blend_sweep.py` re-scored any QB TD blend exactly and matched the 2h harness within
   0.004 RMSE. The same trace (current_rate, prior_rate, weight, blended_rate, final) would let `STAT_K` /
   `STAT_K_BY_POS` for passing yards, rushing, receptions, yards and targets be swept offline in seconds. Effort: medium
   once, then cheap forever.
4. **Traded-player "merged team column" leak, other call sites.** Fixed so far: the backtest pace proxy
   (`_as_of_team_game_plays` / `as_of_team_pace`, now uses `_historical_game_team`), which had credited a traded
   player's earlier games to his latest club. The methodology notes it as "the same class as the open traded-player
   end-of-season team item". A grep audit of every place that groups historical stats by the merged `team` column is
   not done. Backtest-only effect (live boards read nflreadpy), but it changes harness base arms.
5. **Backtest-only leak in the default roster placeholder read.** `_add_missing_returning_player_rows_core` takes each
   player's later-week roster status in a backtest (found 2026-10-07: removing it changed a 2024 wk3 board from 447 to 435
   rows). Live has no later weeks, so only backtests and calibration dumps are touched; every harness base arm contains it.
   Quantify before touching it (it would move base arms).
6. **`v2_venue_mult` has never been backtested standalone.** It is the last untested piece of the old `game_env` bundle
   (the sibling elasticity shipped as `v2_game_total_elasticity_perstat`). One arm, ~2h15m. Low expected value (dome vs
   outdoor), but it is the only built-and-never-measured flag.
7. **RB2 rushing yards.** After `v2_rb_rush_yards_script_neutral` the RB1 yards gap fell from -2.60 to -0.61/game, but RB2
   was still -2.83 (clean RB1/RB2 combined bias -3.11 -> -1.68). `v2_rb_carry_budget_lead` (lead-weighted carry trim)
   is an unshipped modifier that tested worse than uniform.
8. **Parked, three attempts failed:** the TE target split (TE1 -0.35, TE2 +0.23 targets/game; participation, WR+TE and a
   within-room tilt all left START-TE RMSE unchanged or worse). Weak tiers (WR rank 31-55, WR team #3+ with bias +1.1,
   TE rank 1-4) are small.
9. **Parked, by user decision or evidence:** team-level scoring. The market beats the model on team TDs and yards
   (model team-TD slope 0.53, i.e. ~2x too spread out) but the model matches posted player props (r = 0.90) and beat
   them live. Blending is rejected permanently. TD compression and yards-per-opportunity shrinkage did not help points.

## C. Product, UI and data items

- **Phone exports (2026-10-18).** Weekly Rankings > Export for phone (after Build board): a game-report PDF (every game in
  kickoff order: kickoff, venue, line / O/U, implied team totals on the team bars, weather, each team's QB / top-2 RB / top-3
  WR / top TE with projections and injury tags, three games a page) and a bare-bones board PDF (per position: rank, player,
  team, opponent with @ for away, projection, injury tag). Download buttons, or save both into a folder - defaults to
  `OneDrive\NFL Scholar`, which syncs to the phone. `ui/report_export.py` (matplotlib, no new dependency),
  `tests/test_report_export.py`. The breakdown header's matchup rank is now its own full-size metric tile ("27/32",
  green easiest / red toughest).

- **Box score inside the projection breakdown (shipped 10-06).** Verified end to end in a browser, but on a
  ledger-stubbed private copy and with a test-driver run that re-pulled the props cache. Worth one click-through in
  the real app on a week-4 or week-5 player. Known limits: one dialog at a time (so it swaps the breakdown for the box
  score with a Back button), the AVG row is not clickable, 2026 box scores come from live nflreadpy (there is no local
  `stats_player_week_2026.csv`), so confirm it stays quick offline or on a slow connection.
- **Props archive / ledger plumbing:** DraftKings weekly props can be absent from a snapshot (needs the Playwright
  fallback) and FanDuel weekly props need a browser capture; Odds API already carries FanDuel under licence.
- **Older backlog items, dated Aug 2026 and not re-checked:** build-board flow browser smoke test; live eyeball of the
  buried-veteran dock (Brown / Franklin / Lemon / Wicks / Oliver / Hooper / Zaccheaus / Vele) plus a regression test
  with an Ourlads chart fixture; New Orleans defense-vs-RB audit; a regression test that team targets / team pass
  attempts stays ~1.0 at several week values (recommended 08-22, never added).
- **Name-match fragility:** depth-chart/PFF lookups still use two-tier name matching; the ID crosswalk
  (`load_player_id_crosswalk`, pff_id <-> gsis_id, 100% verified) is wired in at none of ~15 call sites. A missed match
  silently mislabels a player ROOKIE or drops his PFF metrics.
- **Draft HQ / draft projections:** no backtest harness exists for `build_projected_board`; the self-weights
  (`STAT_SELF_WEIGHT`, `FULL_TRUST_GAMES`, `ROLE_CHANGE_EVIDENCE_FLOOR`) were moved untuned on 2026-08-31. The cleaner
  role-change fix (volume from the new rank's curve, efficiency from the player's own history) is not built.
- **Retractable roofs with no posted roof status:** for a FUTURE game the schedule's `roof` column is blank at ATL, DAL and
  ARI (and the domes are only marked once the game is played), and the weather code treats a blank roof as outdoors, so a
  hot or wet day gets an outdoor forecast that likely does not apply (week 5: TB @ DAL reads 79.6 F, dry). Small, but a
  closed-roof assumption below ~60 F or above ~85 F / in rain would be a better default. Not built.
- **Weather:** pregame forecasts come from `data/weather.py` (Open-Meteo). International games now use the real stadium
  (`VENUE_COORDS`; fixed 2026-10-07 after PHI @ JAX at Tottenham was read as Jacksonville). `v2_weather_adjustment` IS shipped, wind
  only: QB and RB carry a confirmed win, the WR slopes are zeroed (WR measured worse at every strength), TE is
  wired. Temperature and precipitation are shown as icons only and are not modelled.
- **Blocked on data:** O-line PFF grades have no weekly archive (leakage); injury reports are only surfaced on Draft
  HQ's News sub-tab.
- **Next offseason (2027 opener), cold-start work that cannot be tested in-season:** `v2_new_team_starter_restoration`
  (built 2026-09-05 for Waddle/Evans-type cases; its Week-1 backtest was queued as "L3" on 09-07 and no result was
  ever recorded in the docs, so run it for 2022-2025 before the next preseason), `v2_cold_start_regression` (not
  robust on n = 5 Week-1s), the Ourlads chart import, `data/qb1_overrides.csv`, and the PFF 2026 archive.

## D. Housekeeping

- **Vestigial flags:** `v2_output_contract` and `v2_alignment_contract` sit in `DEFAULT_FEATURES` but nothing reads them
  (decide: delete, or wire them up as real validation gates). `redzone_tds`, `role_trend`, `volume_faced` and
  `v2_channel_matchups` were never built (`redzone_tds` is effectively superseded by `v2_xtd`).
- **Built but not shipped (31 flags in `MODEL_FEATURES` and not in `DEFAULT_FEATURES`):** most are measured rejects,
  legacy aliases, or backtest-only switches (`v2_historical_*`, `v2_defense_prior_games_override`). Candidate cleanup:
  decide which to delete so the list stops growing. Built with no recorded result: `v2_venue_mult` (section B) and
  `v2_new_team_starter_restoration` (section C, next offseason).
- **Uncommitted scratch** (left deliberately): `.sweeps/receiver_audit/`, `.sweeps/receiver_audit_keeptds/`,
  `.sweeps/*.done`, `.sweeps/*.bat`. The audit results they back are already in the methodology doc.
- **Stale docs:** `HANDOFF.md` ("Current direction": branch `weeklymodel-2026-08-30`, working copy `C:\NFLScholar`) and
  `docs/weekly_rankings_backlog.md` (sec. 8 "live queue") describe August; pointers to this file were added to both.
- **This file is not committed.** It is a new untracked file; `git add docs/open_items_2026-10-07.md` when ready.

## E. Tested and closed (do not re-run without new evidence)

| Item | Result |
|---|---|
| Model + market blend | Rejected by the user, permanently |
| `v2_qb_passing_td_k` (K alone) | Inconclusive; superseded by `v2_qb_td_blend` (shipped) |
| Yards-per-opportunity shrink (`eval_efficiency_shrink.py`) | Rejected; not robust, lines already absorb it |
| TD compression (`eval_td_calibration.py`) | Not built; TE worse in every split |
| `v2_own_tempo_regression` | Inconclusive, no effect, unshipped |
| `v2_pace_alpha_cap` full-window ablation | Neutral; cap stays (direct play-count test) |
| Calibration v5 (all-rows) | Reverted to v4; v4 re-checked 10-06 on final defaults, kept |
| `v2_td_volume_shrink`, `v2_td_career_regress` | Rejected (WR/TE TD up-regression adds error) |
| `v2_cold_start_room_budget`, `v2_vacancy_before_capacity`, `v2_rb_snap_anchored_volume` | Rejected |
| `v2_coaching_aware_defense_prior`, `volume_efficiency`, `game_env` bundle | Rejected |
| `v2_weather_adjustment` | Shipped (wind; WR slopes zeroed) |
| TE outside-20 / WR-TE-QB outside-the-red-zone TDs (user asked 09-30) | Already covered: receiving xTD has an `oz20` zone; QB rushing uses the shared rush zone via `v2_xtd_rush_outside_zone` |
| Scheme/alignment matchup | Resolved: scheme shipped for TE only; WR blend not shipped |
| Per-position game-total elasticity | Resolved 08-31 (`v2_game_total_elasticity_perstat` shipped; TE kept at 0.30) |
| `script_status` per-character display bug | Fixed 10-06 |
| Stafford -6.2 rushing yards in market projections | Fixed (negative implied mean now falls back to the posted line) |

## F. How to pick up

```
python scripts/review_live_season.py                      # live record so far
python scripts/score_ledger.py --year 2026                # model vs market vs FantasyPros
python scripts/scan_board_integrity.py                    # bug scan on the current board
python scripts/eval_qb_td_blend_sweep.py                  # offline blend sweep (instant)
python scripts/backtest_component.py --add FLAG --add-features v2_historical_injury_replay,v2_historical_reserve_replay --years 2022,2023,2024,2025 --weeks 3-17
```

The harness takes ~2h15m for two arms and must be launched detached (see the long-runs note in memory). Newest
methodology entries are at the bottom of `docs/weekly_projections_methodology.md`.
