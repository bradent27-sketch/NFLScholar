# Model sense check (2026-10-26)

User: "dig through projections and try to find bugs in the model ... not necessarily things that are totally broken ... but things
that don't make sense in trying to accurately predict future games. Sometimes that may be misunderstanding the context of a game,
sometimes that is incorrectly valuing certain evidence over other evidence, sometimes that is incorrectly placing volume on the wrong
players, sometimes that is data that is understood or broken down incorrectly."

Method: start from the live cases where the model and the market disagreed most (2026 weeks 3-7, `.sweeps/audit_market_gap_2026wk3-7.txt`),
trace each one through the model (single-team rebuilds, scratchpad `trace_team.py`), and when a case points at a rule, measure the
rule on history (snap counts 2019-2025, the 2022-2025 harness boards) before building anything. Each fix is a named candidate flag
with a bar written before its harness run (`docs/weekly_projections_methodology.md`, 2026-10-26 entries).

## Findings

| # | Kind | Finding | Evidence | What was done |
|---|---|---|---|---|
| 1 | Evidence weighting / volume | **A partial game is set aside as an injury exit, and the old role is projected as if it stood.** An established player's sudden low-snap game (median of last 3 >= 65%, this one <= 50% and <= 0.6 x that) is dropped from his history, with no injury information behind the call. It misfires on committee changes too: NE week 3, Rhamondre Stevenson's week 2 (36% after 85%, the week TreVeyon Henderson came back) was read as an injury exit, so Stevenson kept his week 1 role (14.1 carries) next to Henderson (13.3): about 30 RB carries for a team the market had at 19 (NE ran 21). | Snap counts 2019-2025, 625 such games: the next share is 0.47 from a 0.77 role (played the next week) and 0.57 from 0.79 (back from a missed game). Leaving the game out is the worst reading in all 7 seasons, every position. Harness boards 2022-2025: those rows run **+1.49 pts over** (243) / +0.91 (92) against -0.25 for every other row; every season over. | CANDIDATE `v2_partial_game_role_carry`: the game stays out of the per-game rates but scales the role (share with it / share without it). In the harness run. |
| 2 | Evidence weighting | **One game is a whole role.** In season the expected snap share is the plain mean of the last <=4 appearances, and nothing pulls it toward last season's share (the prior per-game rate is rescaled by current share / prior share, so the share drives the prior side too). Dylan Sampson (CLE week 7): his one appearance was a kick-return week with 0 offensive snaps -> 0.0 projected against a 27% role in 2025; Raheim Sanders and Audric Estimé were out and he scored 10.7. | Offline 2022-2025 next-week share, n = 1-4 appearances: pulling toward the prior as if it were k more games (WR/TE 0.75, RB 0.25) cuts RMSE 4.4% / 2.8% / 1.3%; held out by season 4/4, 4/4, 3/4. | CANDIDATE `v2_share_prior_shrink`. In the harness run. |
| 3 | Evidence weighting | **The latest game is under-read** (backlog L6). The share window weights a player's last four appearances equally. | Offline, every player-week 2022-2025: counting the latest twice cuts next-week share RMSE WR 2.9%, TE 2.4%, RB 1.9%, every season at each position; stacks with #2. | CANDIDATE `v2_share_recency`. In the harness run. |
| 4 | Volume (narrow) | **A team-changer back from an early injury is projected at almost nothing.** A.J. Brown (NE week 6): his only NE game (week 1) was excluded as an injury exit, he missed weeks 2-5, and the "returning from a missed start" restoration treats any team change as disqualifying -> 0.01 snap share, 1.4 pts (market 12.4). He drew 10 targets. | The rule is right in general: team-changers who miss a season's start come back small (snap counts 2016-2025, 31 starter-level team-changers: first game 25% of snaps, next three 34%, against 53-59% for players on their old team). Brown's difference is that he had already started for NE before he was hurt. | Not built: a handful of cases a decade, and the general rule is right. The live answer is the user's availability/role override for that one player. |
| 5 | Data | **A stale injury pull ruled out two players who played.** Ja'Marr Chase and Terry McLaurin (week 5) were Out on the 13:11 UTC game-day board from the FantasyPros pull and Questionable on the 20:13 rebuild; both played (19.7 and 12.2). | data/ledger 2026_wk05 builds. Not a model rule. | Suggest: show the pull's age on the board and warn on game day when it is older than a few hours. |
| 6 | Volume (narrow) | The third back gets none of an out back's work when his own share is 0 (Sampson again: Sanders' 3.6 carries went to Judkins and McLaughlin; "not an in-season eligible core-RB recipient"). | One case. | Partly covered by #2 (a pulled-up share makes him a recipient). |
| 7 | Data | **A week whose snap counts are not posted yet reads as 0% snaps for every player.** The loader fills a box-score row with no snap match with 0, and `has_snap_match` is per season, so nothing tells the model the number is missing. On Monday evening (snap cache refreshed 18:16) the 2026 week 7 box scores were in and the snap counts were not (only the Thursday game). A week 8 board built then: 67% of the 128 RB/WR/TE with a 60%+ share cut 15%+ (Garrett Wilson 0.96 -> 0.73, Drake London 0.95 -> 0.72, Trey McBride 0.94 -> 0.71), every team but Chicago. The week 5 board was built on a Monday evening (10-05 23:43 UTC); whether it was hit cannot be told from the ledger (the model changed between builds). | Regular seasons 2021-2025 have snap counts for every team-week (only playoff weeks are missing), so history never saw it. | **FIXED (data guard, not a model flag):** `mask_unposted_snap_weeks` blanks the share of a team-week the snap source does not have yet, so the last known appearances stand; the build lists those team-weeks in its data contract (`snap_counts_not_posted`). No change wherever snap data is complete (CLE week 7 rebuild identical; NYJ week 8 now reads Garrett Wilson at 0.962). `tests/test_unposted_snap_weeks.py`. The first version crashed every build for about 1.5 hours (a missing team value broke a sort); caught by this audit's rebuild and fixed the same evening. |
| 8 | Data | **One player's snap share is always 0 from a name mismatch.** Bam Knight (ARI) is "Zonovan Knight" in the snap counts; the loader joins snaps by name, so his three games (18 carries) read 0% (the partial-game screen honours `has_snap_match`; the share read does not). | The only 2026 RB/WR/TE/QB with stats and no snap match. | **FIXED 2026-10-27 (user: "join Bam and Zonovan Knight's id"):** a row the name match misses is filled from the snap counts by PFR id and week (`snap_pct_by_pfr_id` / `fill_snap_pct_by_pfr_id`, data/loaders.py); rows the name match found are untouched. Bam Knight 2026 now 6 / 8 / 44 / 53%; the same fallback also finds Bisi Johnson, Benjamin Watson, Dee Eskridge, Joseph Fortson (4-22 rows a season, 2019-2024). |
| 9 | Data | **Two players with one name in the same week are averaged in the snap counts.** The snap table is pivoted by name alone (`aggfunc='mean'`). Justin Jefferson (MIN, 89%) and a CLE linebacker of the same name read 44.5% in 2026 week 5; the partial-game screen then set that game aside as an injury exit, his share fell 0.94 -> 0.79 and his week 7 projection was 11.7 (market 16.4). | Snap counts 2019-2026, same name + week + different player involving an offensive player: Michael Thomas 2019 (15 weeks), Lamar Jackson 2020 (11), A.J. Green 2021-2022 (24), DJ Turner 2023-2024 (20), Jefferson 2026 (1). Michael Thomas 2022 read 30/38/34% for 61/76/68%. | **FIXED (data, not a flag):** `snap_name_collisions` / `apply_snap_name_collisions` (data/loaders.py) re-read only those player-weeks by the game's team; every other row is untouched. `tests/test_snap_name_collisions.py`. |
| 10 | Data (backtests) | **Deduping the roster by name dropped real players from the backtests.** `load_year_data` kept one roster row per *name* before joining on gsis_id, so one of two same-named players lost his team and the model left him off the board: **Josh Allen was off every 2022 harness board** (a JAX LB shares the name), **Quinshon Judkins off every 2025 board** (a stray id-less GB row in week 18), and Michael Carter (2022-2023), Michael Thomas (2022-2023), DJ Turner (2023), Russell Gage (2025), Xavier Weaver (2024). Their teammates were projected without them. | Base arm of the 2022-2025 harness: 145 player-weeks came back (the 10-26 run below has them; earlier runs did not). 2026 has no skill player with a blank team, so live boards are unchanged by this one. | **FIXED (data, not a flag):** the roster is deduped by gsis_id (the join key); a row with no gsis_id is kept only if no identified player has its name. |
| 11 | Tooling | **The today-vs-market re-score rebuilt a week that was not final.** Week 7's Monday game had not been played, so the model built it as a *live* week: the injury replay is off, and the rebuild passes apply_injury=False, so nobody was out (every week 7 status blank) and the live-only sources were on. | Week 7 was the one week "today" beat the market (7.13 vs 7.17). Clean weeks 3-6, the scorecard's start-worthy pool: **today 7.31, live model 7.40, Market 7.13 (gap 2.6%)**; FantasyPros 7.19. | **FIXED:** `scripts/eval_today_vs_market.py` skips a week that is not final and says so. |
| 12 | Rebuild caveat | The re-score reads today's `data/qb1_overrides.csv` for past weeks, so three QB rows have the wrong starter (Penix week 4: the returning-starter rule replaced the Penix choice with Cooper Rush; Bagent week 4; Mayfield week 7: an open-ended Jalon Daniels row). The returning-starter rule cannot tell a fill-in from a permanent change; live, it warns and the user re-saves. | Without those 3 rows weeks 3-6 change by 0.02. | Noted, not changed. |
| 13 | Data / identity | **A defender who shares an offensive player's name takes over his history.** Two places read players by name across every position: (a) `_identity_by_name` gave the name "Justin Jefferson" the CLE rookie linebacker's id (the last row in a stable sort), so the receiver's 2026 totals carried an id with no 2025 season and his prior was the **WR position average** (4.2 targets, 33 yards a game for his own 8.4 / 67) from week 6 on; (b) `expected_snap_share` and the other name-grouped reads averaged the linebacker's 0%-offense week 5 into the receiver's window: **share 0.705 for his own 0.935**. | 2026 names with an offensive player and a different defender: Justin Jefferson, DeVonta Smith, Devin Neal, Brandon Johnson, Cam Miller; 1-6 such names in every season 2019-2025 (Lamar Jackson and a CB 2020-2024, A.J. Green 2020-2022, ...). Where both seasons pick the same wrong id the prior still matches (checked: Lamar Jackson 2025 has his own prior either way), so the live 2026 cases are the ones shown to bite. | **FIXED (data, not a flag):** `_identity_by_name` prefers the offensive player's id when a name is shared; `drop_shared_name_defenders` removes a defender's rows when an offensive player with a different id has his name (current and prior seasons; the build lists them in its data contract, `shared_name_defenders_dropped`). A two-way player (one id) is untouched. `tests/test_identity_same_name.py`. Week 8, Jefferson: share 0.705 -> 0.935, prior position average -> his own, **11.66 -> 13.07 pts**. |
| 14 | Volume (evidence weighting) | **2026 pass volume is projected 10% low, and the pass-capacity trim carries it to every receiver.** The receivers' own rates claim more targets than the QB's projected attempts allow, so the team's WR/TE are scaled down to fit (median team x0.94 on the week 7 board, a quarter of teams x0.85 or less; MIN week 8 x0.80, Jefferson -1.7 targets / -15 yards). That trim is right on history - the model projects 96-99% of actual team targets every season 2022-2025 - but in 2026 the QB attempts it is tied to run low. | Projected / actual, team-weeks: targets 0.986 / 0.971 / 0.956 / 0.979 (2022-2025) vs **0.900** (2026 weeks 3-6); QB attempts 0.965 / 0.953 / 0.957 / 0.956 vs **0.904**. This is where the live "WR receptions under the market, the gap is targets" comes from. | Not built. It is backlog W1 (2026 passing surge) with the mechanism now traced: a fix belongs in how fast the team pass volume follows the season's observed level (backlog M2), not in the receivers. Same measure on every season, so the 2026 drop (0.904 vs ~0.955; the history figure includes backups' attempts, which is why it sits under 1) is the signal; M1 (2026-10-19) found league passing levels do not persist on history, and week 7 cooled to 31.7 attempts a team. W1's trigger (5% beyond normal) is met through week 6: re-check with weeks 7-8 final, then test M2 with a pre-set bar. |

Questions checked that turned out fine:
- **WR receptions under the market** (live): not a catch-rate problem - projected catch rate matches actual at every target tier (WR
  0.62 / 0.64 / 0.65 / 0.68 against 0.60 / 0.65 / 0.64 / 0.68); the gap is targets, and #14 is why.
- **Players returning from an ordinary absence** (no partial game before it): no over-projection on the harness boards (2,494 rows,
  -0.12 pts; share 0.35 projected vs 0.36).
- **Top-end calibration on 2022-2025**: RB 17-24 within +-0.3 pts; WR 20+ over by 2.5 on 59 rows with no steady sign by season. The
  2026 live boards show RB/WR 17+ about 1.5-2.5 over where the market is right; whether today's model still does is the rebuild below.
- **Questionable WRs**: the 0.83 factor is flat whatever FantasyPros' probability (52% to 97%); it was fit on Friday-final official
  designations, which have no probability, so the probability cannot be tested on history. Left as is.
- **Chasing one big game** (live: Lamb 16.4 -> 23.9 after a 41-point week, JSN 153 projected yards after 35 points, Coleman 14.3 after
  23.6, the market well below each): not on history. On the 2022-2025 boards the miss does not grow with last week's surprise (slopes
  -0.08 to +0.02 outside one small TE bucket) and a projection moves 0.09-0.21 pts per point of surprise (Lamb's live move was ~0.3).
  Whether today's model still reacts that much on those weeks is in the rebuild.
- **2026 passing surge (W1)**: weeks 3-6 ran 34-37 attempts per team, week 7 (24 teams in) 31.7. Superseded by #14: today's model
  projects 90% of 2026 team pass attempts and targets against 95-99% on every season before.
- **QB rushing yards** (live, +1.8 over the market): concentrated in QBs projected 30+ yards (model 36.0, market 28.8, actual 28.0,
  24 rows, mixed by player). Too few to act on.

- **Rebuilt boards, 2026 weeks 3-6:** projected spread vs the market. Calibration slope of actual on projection, the paired START
  pool: WR 0.79 (market 0.93; 90% CI of the gap [-0.26, -0.02]), TE 0.76 (0.90), RB 0.95 (1.04), QB 0.68 (1.07, three of the rows
  are the wrong-starter rows in #12). On 2022-2025 the same slope is WR 0.99, RB 0.91, TE 0.85, QB 0.96, so a 2026 WR spread this
  wide is new or noise (205 rows); a watch item, nothing to act on. RB tiers: projected 20+ run 1.9 over (18 rows), under 11 run 1.6
  under (75), as on the live boards.
- "Nan (100%)" in the rebuilt boards' Injury Status: a practice-report row with no game status in the historical replay. Read as
  healthy, so no number moves; display only, backtests only.

## Runs

- Today's 71-flag model rebuilt as of 2026 weeks 3-7 (`.sweeps/eval_today_vs_market_71.txt`, boards in `.sweeps/today_boards_71`):
  start-worthy today 7.230 / live 7.317 / Market 7.087 over weeks 3-7, but week 7 was not a valid rebuild (#11). Weeks 3-6: 7.31 /
  7.40 / 7.13. The gap to the market is 2.6%; today's model is 0.09 better than the boards that actually ran those weeks.
- Harness 2022-2025 weeks 3-17 (`.sweeps/harness_audit_share_partial_2022-2025_wk3-17.log`; the first launch ran out of memory when
  four CFB week_review jobs started a minute after it and was re-run 21:29-23:04), each variant against the 71-flag default, both arms
  with the data fixes #9-#10. **None passes its pre-set bar:**

| variant | START-ALL | EVERY-ALL | START RB / WR / TE | changed rows (played) | bar |
|---|---|---|---|---|---|
| `v2_share_prior_shrink` | -0.001 [-0.004, +0.001] | +0.001 | +0.000 / -0.003 / -0.004 | RMSE -0.003, n 8,296 | FAIL (START-ALL needs <= -0.003; EVERY-ALL > 0) |
| `v2_partial_game_role_carry` | -0.001 [-0.008, +0.006] | **-0.006 [-0.010, -0.003]** (WR -0.011) | -0.003 / +0.004 / -0.007 | RMSE 6.310 -> 6.299, bias -0.21 -> -0.27 | FAIL (part 2: bias moves away from 0) |
| `v2_share_recency` | -0.001 [-0.004, +0.003] | -0.002 | -0.001 / -0.002 / +0.003 | RMSE -0.003, n 7,978 | FAIL (START-ALL; START-TE > +0.002) |

  Reading: the share studies were right about the share (offline next-week share RMSE fell 2-4%), but at points level the gain is
  ~0.001 because the per-game rates, the team budgets and the vacancy rules already absorb most of a share error. The partial-game
  carry is real on the games that follow a partial game when the player then sits (EVERY-WR -0.011): the partial game is often the
  first sign of an injury. Where he plays it cuts too far (WR rows it lowered: +0.28 over -> -0.34 under; TE -0.01 -> -0.48). All
  three stay off.

## Worked example: Justin Jefferson, 2026 week 8 (MIN, live build, 2026-10-27)

How one projection is put together, and what the same-name bugs (#9, #13) did to it. Targets; yards and receptions follow the same
steps.

| step | what it is | before the fixes | after |
|---|---|---|---|
| history | 2026 games: wk1 92% / 9 tgt, wk2 100% / 6, wk3 12% / 2 (left out: abrupt partial game), wk5 89% / 7, wk7 93% / 7 | wk5 read 44.5% (snap table averaged him with the linebacker); a CLE LB's 0% wk5 row in his window | his own rows |
| expected snap share | mean of the last <= 4 appearances | **0.705** | **0.935** |
| current rate | targets per eligible game, 2026 | 7.13 | 7.13 |
| prior rate | 2025 targets per game, defense-adjusted, role-scaled to this share | **4.18, WR position average** (no 2025 season found under the linebacker's id) | **8.39, his own** |
| blend | current weight G/(G+K): 0.76 with a fallback prior (it is trusted less), 0.52 with his own | 6.42 | 7.74 |
| matchup x pace | the defense's targets allowed by alignment (x1.084), opponent pace (x1.017), script (x1.004) | 7.10 | 8.56 |
| pass capacity | MIN's WR/TE claim 30.3 targets, the QB's 31 attempts leave 24.3: everyone x0.80 (#14) | -1.19 | -1.71 |
| targets | | 5.91 | **6.85** |
| points | stat line 12.9 raw -> calibration +0.59 -> season anchor +0.16 -> expected-points luck -0.58 (he has scored 3.2 a game over his expected points) | 11.66 | **13.07** |

What is still arguable after the fixes: the x0.80 capacity trim (#14 says the QB attempts it is tied to run ~7% low this season) and
the luck correction (-0.58 for a receiver whose touchdowns ran hot; on history that correction is right for WRs).
