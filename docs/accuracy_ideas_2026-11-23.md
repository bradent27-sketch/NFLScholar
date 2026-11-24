# Accuracy ideas (2026-11-23)

Ten ways to make the weekly model more accurate, each with what to add, how to test it and what to look at. They come from
public football-projection research (sources at the end) and from first reads on our own data run the same day. Nothing here
changes the model yet; each idea goes through the usual path: offline read, then one named flag, then the harness, and
`DEFAULT_FEATURES` only on the user's sign-off. No idea uses the market as an input to a projection.

Scripts and outputs: `.sweeps/accuracy_ideas_2026-11-23/`.

## Status ledger

Update a row when an idea moves: the status, the date and where the result is written up. The status key is the backlog's:
TODO, IN PROGRESS, SHIPPED, REJECTED (tested, not worth it), PARKED (waiting on something).

| # | Idea | Status | Updated | Notes / where |
|---|---|---|---|---|
| 1 | Route-based target estimate in the stat recal | SHIPPED (with 2) | 2026-11-23 | merged with 2: the first read's routes cancelled out (it was target share), so the route input is the trend; see row 2 |
| 2 | Route trend as a continuous signal, falls and backs included | SHIPPED | 2026-11-23 | `v2_recal_pass_snap_trend` (recal term, both directions, WR/TE/RB, replaces route rise). Harness PASSED the pre-set bar: 2022-2025 START-ALL -0.006, every position better, moved rows better 4/4 seasons; 2026 wk3-10 START-ALL +0.003 (at the bar). Shipped by the user in place of the route rises (DEFAULT_FEATURES = 88); live it waits on the PFF weekly exports for weeks 10+. Re-check on 2026 weeks 11+. `.sweeps/pass_snap_trend_bar.md` |
| 3 | A room share model, and share error in the scorecard | TODO | 2026-11-23 | |
| 4 | QB and pass-catcher passing yards made to agree | TODO | 2026-11-23 | re-measure on current boards first (the game-script recal overlaps) |
| 5 | A second source before an Out tag zeroes a player | TODO | 2026-11-23 | the props-existence check needs the user's OK |
| 6 | Full role for players back from injured reserve | FAILED | 2026-11-24 | re-measure: same-team starters' (0.70+ last season) true season debuts were projected 0.53 snaps, played 0.82 (n 9); PUP debuts ramp. Flag `v2_debut_starter_full_role` failed its bar: moved played rows +0.089 RMSE, better in 1 of 4 seasons; START-ALL +0.001. The mask's role (max of active and pre-absence share) also marked depth WRs off a late-season fill-in stretch, and on the true starters the points didn't follow (3 better, 5 worse). 2026 weeks 3-10 better (-0.010, Bowers week 3 only). Not shipped. `.sweeps/debut_starter_bar.md` |
| 7 | Depth-chart promotions as an early role signal | TODO | 2026-11-23 | |
| 8 | Past opponents taken out of a player's own rates | TODO | 2026-11-23 | |
| 9 | Man/zone matchups refit on six seasons of free data | TODO | 2026-11-23 | |
| 10 | Use the market gap to nominate features (never as an input) | TODO | 2026-11-23 | run every four weeks |

## Where the live gap to the market is

2026 weeks 3-11, last pre-kickoff board, START pool (each side's top-N per position), rows with a full market line.

| | model RMSE | market RMSE | model pairwise | market pairwise |
|---|---|---|---|---|
| all | 7.38 | 7.10 | 66.9% | 68.5% |
| QB | 7.66 | 7.30 | 62.0% | 64.2% |
| RB | 7.69 | 7.45 | 66.5% | 68.0% |
| WR | 7.30 | 7.00 | 64.8% | 66.9% |
| TE | 6.71 | 6.56 | 60.3% | 62.9% |

The market's squared error is about 7% lower. Broken down (`gap.py`, 937 START rows):

- **The gap is about who on a team gets the points, not how many the team scores.** With each team's total set to the
  actual, so only the split between teammates is left, the model's error is 45.3 against the market's 41.6: the whole gap
  (3.8). The model's own team totals combined with the market's split score 55.3, the same as the market (55.0). Ideas aimed at
  team volume or pace are aimed at the wrong place.
- **WRs are 55% of the gap**, RBs 28%, TEs 11%, QBs 6%.
- **Large disagreements carry it.** Rows where the two differ by 3+ points are 12% of the rows and 69% of the gap. The model
  lower by 3+ points (65 rows): model bias -4.1, market +0.5.
- **Four players listed Out on the last board then played** (Chase and McLaurin week 5, Golden week 8, Stevenson week 10).
  The three on START rows are 26% of the gap.
- By stat, the market is better on receiving yards (+0.89 points² per row), passing TDs (+0.82), passing yards (+0.63) and
  receptions (+0.45); receiving TDs are even.

## The ten ideas

"First read" is one linear term, fit on the other seasons and scored on the held-out one, against the deep-audit panel
(2021-2026, the 2026-10-28 model, before the game-script recal, capacity relief, the Questionable RB cut and role drift). Played
rows, weeks 3-17. MSE changes are of that stat's error.

| # | Idea | First read | Size |
|---|---|---|---|
| 1 | Route-based target estimate in the stat recal | targets: RB START -1.7% (5/5 seasons), WR START -1.0% (4/5) | medium |
| 2 | Route trend as a continuous signal, falls and backs included | targets, every played row: WR -1.2%, TE -0.8%, RB -0.5% (5/5 each) | small |
| 3 | A room share model, and share error in the scorecard | the whole live gap is share error | large |
| 4 | QB and pass-catcher passing yards made to agree | QB passing yards -0.2% to -1.5% (6/6), 2026 -3.4% | small |
| 5 | A second source before an Out tag zeroes a player | 26% of the live START gap from 3 rows | small |
| 6 | Full role for players back from injured reserve | season-debut WRs +1.1 points a game under (6/6 seasons) | small-medium |
| 7 | Depth-chart promotions as an early role signal | not read yet (2025-2026 data only) | medium |
| 8 | Past opponents taken out of a player's own rates | right sign 5/6 seasons, START-ALL -0.06% points MSE; WR 2024-2026 -0.3 to -0.5% | medium |
| 9 | Man/zone matchups refit on six seasons of free data | not read yet; retries C1/C1b, which closed for lack of seasons | medium |
| 10 | Use the market gap to nominate features (never as an input) | method | small, recurring |

Order: 5 and 4 first (small, clear), then 1 and 2 (the strongest new signal), then 6, 3, 8, 7, 9; run 10 every four weeks
alongside.

---

### 1. Route-based target estimate in the stat recal

- **Why.** A receiver's targets are his routes times his targets per route. The model reads target history and snap share, and
  routes only for role confidence and the route-rise trigger. First read (`passsnap.py`): an estimate of
  (his share of the team's pass-play snaps) x (team pass attempts) x (his own targets per pass-play snap this season), minus the
  board, against the targets residual. Held out:
  - RB START -1.66% (5 of 5 seasons); RB every played row -1.03% (4 of 5).
  - WR START -1.04% (4 of 5); WR every played row -0.68% (4 of 5).
  - TE about zero.
  - The fitted weight is 0.2-0.3, so it is one input among several, not a replacement.
  - Outside: PFF's route-level predicted target share adds about 7% R² over plain target share for next week's share.
    Sumer finds target share (~0.70 year to year) and yards per route (~0.60) the stickiest receiving numbers.
  - N2 (targets per route) was closed in October on two seasons of PFF weekly routes. nflverse participation
    (`load_participation`) lists the players on the field for every play from 2016 to 2025, so the history is now ten
    seasons.
- **Add.**
  - A participation loader cached like the snap counts: per player-game pass-play snaps and team dropbacks, 2016-2025.
  - Live 2026 input: participation stops at 2025, so the live season reads the PFF weekly receiving export (routes), which
    `data/route_participation.py` already loads for route rise.
  - A new input column in `STAT_RECAL_OWN_HISTORY` (`data/weekly_projections.py` line 3401) for targets. Receptions and yards
    then follow through their own inputs.
  - Flag `v2_route_target_input`.
- **Test.**
  1. Agreement check: on 2024-2025, compare PFF routes with participation pass-play snaps per WR / TE / RB game (correlation,
     bias by position). The fit uses one and the live board the other, so the two must agree, or be mapped to each other.
  2. Refit the recal with and without the input, held out by season (`scripts/eval_stat_recal_forms.py`); 2026 forward on PFF.
  3. Harness with the injury replay.
- **Bar.** Targets MSE -1% or better in 4 of 5 seasons; receptions and yards not worse; START-ALL points not worse.
- **Look at.** Which rows move: route share rising while target history lags, and backs with a passing-down role. Check
  whether the target gain reaches points: in the first read, volume gains did not carry to points for WRs.

### 2. Route trend as a continuous signal

- **Why.** Last game's pass-play snap share minus his season-to-date share predicts the targets residual on every played row:
  - WR -1.21% (5 of 5 seasons), TE -0.78% (5 of 5), RB -0.47% (5 of 5).
  - START rows are smaller: WR -0.14%, TE -0.27%, RB -0.19% (4 of 5).
  - The shipped route rise (2026-11-16) acts only on WR/TE rises of 0.15-0.20 or more. The panel predates it, so part of
    this is already captured.
- **Add.** Swap the thresholds for a continuous term in both directions, so a player losing routes comes down too, and add
  backs. Same PFF weekly input live.
- **Test.** On the role-drift harness boards (2024-2025 and 2026 weeks 3-10), continuous against the shipped thresholds;
  participation for 2021-2023 offline.
- **Bar.** Better than the shipped route rise on targets MSE in 4 of 5 seasons; START-WR and START-TE not worse.
- **Look at.** The falls are new: list the 2026 players whose routes dropped and what the board did with them.

### 3. A room share model, and share error in the scorecard

- **Why.** The live gap is all share error (see above). Audit L2 found the rooms too flat (TE1 and WR1 under, depth over),
  and the capacity-relief ships fixed part of it.
- **Add.**
  1. Measurement first. Add share error to `data/ledger_scorecard.py` and the harness report: error with team totals set to
     the actual, beside the team-total error. Every later idea can then be judged on the part of the error it targets.
  2. A share model. For each team-week, predict each player's share of the team's targets (and the backs' share of carries)
     from:
     - season-to-date, last-3, last-game and last-season shares;
     - route share (idea 1) and alignment;
     - teammates out and teammates returning;
     - draft capital.
  3. Use it as a final split of the board's own team totals (totals untouched). A multinomial logit or a constrained
     regression, fit on 2021-2025 and held out by season. Flag `v2_room_share_model`.
- **Test.** Share-only MSE held out (team totals set to the actual) against the board, then the harness.
- **Bar.** Share MSE -2% held out in 4 of 5 seasons, 2026 not worse; harness START-ALL <= -0.003, no position worse than +0.002.
- **Look at.** By room type (one dominant receiver, committee backfield, TE-heavy), and the 2026 rows where the model and the
  market split a room differently.

### 4. QB and pass-catcher passing yards made to agree

- **Why.** The QB's projected passing yards and the summed receiving yards of his pass-catchers (weighted by availability)
  estimate the same team number from two directions. First read (`qb_coherence.py`): moving the QB's line 40-45% of the way
  toward the rescaled receivers' sum cuts QB passing-yard MSE in every held-out season:
  - -0.25 / -1.42 / -0.76 / -1.46 / -0.19% (2021-2025), and -3.38% in 2026.
  - Attempts against targets (3 of 6) and TDs (mixed) do not hold.
  - The panel predates the shipped game-script recal for QB passing yards (2026-11-01), so re-measure first.
  - The market's edge on passing yards is +0.63 points² per QB row.
- **Add.** After the stat recal, reconcile QB passing yards toward w x the scaled receivers' sum. Optionally, a second step
  scales the receivers' yards toward the reconciled total (receiving yards are the largest single-stat gap, +0.89). Flag
  `v2_pass_yards_coherence`.
- **Test.** Re-run the read on today's harness base boards, both directions held out, then the harness.
- **Bar.** QB passing yards MSE -0.5% or better in 4 of 5 seasons and 2026; receiving yards not worse; START-QB not worse.
- **Look at.** Big favorites (the game-script recal's territory) and weeks with a QB change, where the two sides disagree most.

### 5. A second source before an Out tag zeroes a player

- **Why.** Four players were Out on the last pre-kickoff board and played: Chase and McLaurin (week 5), Golden (week 8),
  Stevenson (week 10). The three START rows are 26% of the live gap. The freshness note (D2) and the scheduled pulls (since
  11-11) have not stopped it: Stevenson's last board before kickoff was built Thursday 11-13.
- **Add.**
  - Before an Out or Doubtful zeroes a player, cross-check the official report (nflverse injuries: game status and its
    date) and the gameday inactives when they exist.
  - A FantasyPros Out that the official report does not carry, or that predates the final report, becomes a "status conflict":
    keep the projection with a badge, or force a re-pull, rather than zero.
  - Optional, the user's call because it reads market data: whether books still list the player's props at build time,
    as a third check (books pull props for players ruled out).
  - Questionable is untouched (always assumed to play).
- **Test.** Replay 2026 weeks 1-11 from `external_data/fantasypros_injury_archive` and the nflverse injury file: count wrong
  zeros caught against correct zeros lost.
- **Bar.** No player who sat gets projected; at least half of the wrong Outs caught.
- **Look at.** For each case, the board's build time against the final report time, and why Stevenson's last board was a
  Thursday one.

### 6. Full role for players back from injured reserve

- **Why.** Season debuts of established WRs run +1.08 points a game under the board on every played row, and positive in all
  six seasons (+0.66 / +0.94 / +0.54 / +0.46 / +0.34 / +1.11) (`returns.py`).
  - 2026 START debuts: +14 points on 4 rows. Flowers week 4: model 9.7, market 15.9, actual 25.8. Bowers week 3: 8.9, 12.0,
    27.6.
  - Backlog M5: IR debuts ran 1.32x, PUP/NFI 0.44x. Suspensions are already handled by `v2_suspension_return_full_role`.
  - Audit M17: returning players were the worst group among the model's high calls.
  - Returns after 1-2 missed games show no steady bias, so this is about debuts and long absences, not every return.
- **Add.**
  - Extend the suspension path (`suspension_return_mask` / `restore_cold_start_returning_role_share`,
    `data/weekly_projections.py` lines 6107-6150 and 11166) to injured-reserve returns, read from the weekly roster status
    (reserve to active), for players with a starter's role in their last healthy games.
  - Give them that last healthy snap share, possibly x0.9 in the first game back.
  - Flag `v2_ir_return_full_role`.
- **Test.** Cohort by reason (IR, PUP, NFI, none) for 2021-2026: snap share and points bias in the first and second games back.
  Then the flag, with a harness bar on the rows it moves (as for the Questionable RB cut).
- **Bar.** Moved rows' RMSE better in 4 of 4 harness seasons; START not worse.
- **Look at.** PUP/NFI against IR separately: M5 says they go opposite ways.

### 7. Depth-chart promotions as an early role signal

- **Why.** The market was ahead on promoted backups and rookies in 2026: Jordyn Tyson week 10 (model 3.8, market 10.9, actual
  19.0), Jonah Coleman, Konata Mumpfield, Dameon Pierce, Braelon Allen. The depth chart records a promotion before the snap
  history shows it. nflreadpy `load_depth_charts` is ESPN daily from 2025. N3 tested only QB1 and found injured starters still
  listed, so filter by injury status.
- **Add.**
  - For each player, the change in depth-chart rank from Tuesday to Saturday.
  - A flag that moves the expected snap share part of the way toward the new rank's typical share: the in-season version of
    the Ourlads preseason floor.
- **Test.** First check that the daily snapshots are point in time (a pull date on each row). Then a first read on the 2025
  harness boards and the 2026 ledger: snap share and target / carry residuals for promoted against unchanged players.
- **Bar.** Promoted rows' snap-share residual cut by half in both seasons; START not worse. Only two seasons, so keep it
  strict.
- **Look at.** Rookies and RB committees, where the 2026 misses were.

### 8. Past opponents taken out of a player's own rates

- **Why.** The model already computes defense-adjusted per-game values (`_defadj_*`) and the "Previous Adj Avg" column, but
  they are display only (`_defense_adjusted_prior_average`, line 4311). The rate it uses is the raw one, with this week's
  defense multiplier on top, so a player who faced soft defenses carries an inflated rate into every matchup. First read
  (`leads.py` R2; log of his opponents' allowed ratio x projection, weighted by this season's games):
  - right sign in 5 of 6 seasons, START-ALL -0.06% points MSE;
  - WR 2024 / 2025 / 2026 -0.30 / -0.43 / -0.52%; RB -0.13% (5 of 6);
  - QB and TE mixed.
  - PFF and Footballguys both adjust past production for opponents.
- **Add.** `v2_schedule_adjusted_rates`: the current- and prior-season rates built from the `_defadj` values at strength s.
- **Test.** One capture, then re-score s = 0.25 / 0.5 / 1 offline; the winner goes to the harness.
- **Bar.** START-ALL <= -0.002, no position worse than +0.002.
- **Look at.** Early season (few games, the largest schedule swings) against late. Expect a small gain.

### 9. Man/zone matchups refit on six seasons of free data

- **Why.** The PFF alignment and scheme tuning (C1, C1b) closed for lack of seasons (2024-2025 only), and D4 says to retry
  once there are three. nflverse participation carries man or zone coverage for every pass play from 2020 to 2025. That gives
  six seasons of each defense's man/zone rate and, joined to play-by-play, each receiver's targets and yards against man and
  against zone. The live 2026 season still reads PFF.
- **Add.** An offline tuning set from participation and play-by-play. Refit the TE scheme matchup's shrinkage and prior
  weight (C1b pointed to more of both), and try a WR scheme term.
- **Test.** The C1b sweep on 2021-2025 instead of 2024-2025.
- **Bar.** As C1b: START-TE <= -0.005 held out across seasons, in both directions.
- **Look at.** Matchups have been small everywhere (A2, C1-C7), so expect a small result. The value is closing a question left
  open for lack of data.

### 10. Use the market gap to nominate features

- **Why.** The market is better on room splits and on large disagreements. Instead of guessing what it knows, measure which
  pre-game facts explain where it differs from the model, and keep only those that also explain the actual result.
- **Add.** A script, run every four weeks:
  1. On ledger rows with a full market line, regress (market - model) on the pre-game features: role history, routes, depth
     chart, injury tags, returns, draft capital, team context.
  2. Rank the features.
  3. Run the usual held-out residual read against the actual on the top ten.
  The market is never an input to a projection; it only nominates features to test.
- **Bar.** A feature becomes a flag only if it passes the residual read: same sign in 4 of 5 seasons and 0.5%+ of a stat's
  MSE.
- **Look at.** By position and by gap size.

---

## Checked today and dropped

- **First-read target share** (FTN charting, 2022-2026): correlation 0.94 with target share. Targets -0.11% START (4 of 5),
  points +0.07%. It adds nothing that target share does not already carry.
- **Depth-of-target efficiency prior** (expected yards per target and catch rate from aDOT): -0.02% on yards and receptions.
- **Stacking** the projection with season-to-date, last-3 and last-season averages: +0.22% worse held out. The model already
  uses those averages well, so a simple ensemble has no room.
- **Pass-play snap share minus total snap share** for WRs and backs: no help (the two correlate 0.96-0.99). For TEs it helps
  START targets (-0.83%, 4 of 5); fold that into idea 1.
- **Players back after 1-2 missed games**: no steady bias by season.
- **Team volume and pace**: not where the gap is (the model's team totals with the market's split score the same as the
  market).

Other public ideas not read: stable defense indicators (pressure rate, EPA per dropback allowed) as priors for the matchup
multipliers, and Next Gen separation (weekly 2026 data exists; N5 parked).

## Caveats

- The first reads ran on the 2026-10-28 panel. Several ships since then (recal game script, capacity relief, Questionable RB,
  role drift) overlap with ideas 2 and 4, so re-measure on today's boards before building.
- The live gap read is 937 START rows over nine weeks, and a handful of rows dominate it.

## Sources

- PFF, actual against predicted target shares: https://www.pff.com/news/nfl-refining-coach-i-was-open-a-model-comparing-actual-vs-predicted-target-shares
- SumerSports, sticky stats: https://sumersports.com/the-zone/sticky-football-stats-predictive-nfl-metrics/
- SumerSports, yards per route run: https://sumersports.com/the-zone/revisiting-yards-per-route-run/
- 4for4, most predictable WR stats: https://www.4for4.com/2023/preseason/most-predictable-wide-receiver-stats
- Sharp Football, WR stats that matter: https://www.sharpfootballanalysis.com/fantasy/wide-receiver-stats-that-matter-fantasy-football-2024/
- PFF, aDOT and WR scoring: https://www.pff.com/news/fantasy-football-the-relationship-between-adot-and-wr-fantasy-scoring
- PFF, strength of opposition: https://www.pff.com/news/strength-of-opposition-and-fantasy-points-allowed
- Footballguys, strength of schedule for defenses: https://www.footballguys.com/article/2023-strength-schedule-defenses
- ffopportunity (expected fantasy points): https://github.com/ffverse/ffopportunity
- Princeton thesis, weekly model with an ensemble: https://theses-dissertations.princeton.edu/entities/publication/d6981488-f30c-47ce-8f06-b6c3e98f2578
- OpenFPL, ensemble forecasting (soccer): https://arxiv.org/pdf/2508.09992
- Bayesian hierarchical fantasy projections: https://srome.github.io/Bayesian-Hierarchical-Modeling-Applied-to-Fantasy-Football-Projections-for-Increased-Insight-and-Confidence
