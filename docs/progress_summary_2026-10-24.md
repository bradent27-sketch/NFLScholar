# Progress summary (2026-10-24)

User: "summarize our progress as a whole ... what has been added/removed, how much improvement has been made. How accurate are we
compared to the market ... are our leans sharp ... does the market shift towards our lines from opening to close. Also make a summary
of anything left that we were considering ... your recommendation on it, and if it is worth going ahead with or not."

## 1. How much better the model is

One harness run (2022-2025 weeks 3-17, 60 weeks, paired pools), today's 66-flag default against the flag sets of three earlier dates
(each earlier set = today's minus the flags shipped since). RMSE of fantasy points (lower is better):

| today vs | flags since | START-ALL (start-worthy) | every game | QB / RB / WR / TE starters | ranking (pairwise) | weeks today better |
|---|---|---|---|---|---|---|
| 2026-10-21 (start of this audit) | 7 | 7.598 -> 7.581 (-0.2%) [CI -0.026, -0.008] | 5.495 -> 5.453 (-0.8%) | -0.052 / -0.005 / -0.013 / -0.010 | +0.002 | 39 of 60 (every game 57) |
| 2026-10-01 | 22 | 7.702 -> 7.558 (-1.9%) [-0.186, -0.098] | 5.570 -> 5.452 (-2.1%) | -0.377 / -0.053 / -0.126 / -0.067 | +0.010 | 51 of 60 (54) |
| 2026-09-16 | 37 | 7.877 -> 7.480 (-5.0%) [-0.461, -0.330] | 5.805 -> 5.452 (-6.1%) | -0.374 / -0.268 / -0.551 / -0.287 | +0.027 | 58 of 60 (59) |

The over-projection is gone too: start-worthy bias +0.52 (Sep 16) -> -0.20; every game +1.12 -> +0.08. These are lower bounds: code
fixes that were not behind a flag (bug fixes, calibration re-fits) are in every arm.

## 2. What was added and removed

**This round (2026-10-21 to 10-24), model:**

| change | what it does | measured |
|---|---|---|
| + `v2_qb_implied_td_boost` | QB TDs and rushing scale more with the team's implied total | START-QB -0.048, START-ALL -0.008, 4/4 seasons |
| + `v2_fallback_prior_fast_trust` | rookies with no own history trust their in-season rate 3x faster | START-ALL -0.006, RB -0.009, TE -0.010 |
| + `v2_vacancy_lead_share_050` | an injured WR's targets go more to the next man up, less to the alpha | START-ALL -0.004, WR -0.009 |
| + `v2_backup_te_discount` | backup TEs discounted for the games they take no snap | every-game TE -0.031, 57 of 60 weeks |
| + `v2_low_end_points_line` | the points line for low projections re-fit counting every game | every game -0.023, bias +0.41 -> +0.08 |
| + `v2_no_qb_rush_script` | QB rushing game-script multiplier off | QB rushing attempts -2.4%, yards -1.0% error |
| + `v2_td_matchup_half` | the defense matchup on TD stats at half strength | START-ALL -0.0016, TD lines 0.3-0.7% sharper |
| - `v2_channel_matchups` | a flag with no code behind it | 0.000 (cleanup) |

**Earlier in October (15 flags):** expected-points luck, opening-week stat calibration, pace weight cap, pass-capacity TD keep, three
QB1 rules (latest starter, probable starter fallback, returning starter), QB TD blend, Questionable workload cut, depleted RB room
fill, reversed role confidence for WR/TE, season anchor, suspension return, top-end yardage, zero-row display fix.

**Not the model:** a Median view beside every projection (median per stat, calibrated so half of results land above it); the GAMES
view; boards kept on disk and team-only re-projection on a ruling; a lighter Weekly Rankings layout; harness every-game scopes,
per-week records, saved boards, group/variant ablations, and a build-cache fix that stopped full runs dying out of memory; disk caches
for every download; audits: error budget, strength audit, leanness check (10 of 12 flag families earn their keep), mechanism review.

## 3. Against the market (2026 live weeks 3-7)

**Fantasy points (start-worthy, same players):**

| | RMSE | weeks better than the market |
|---|---|---|
| live model (as each week ran) | 7.372 | 1 of 5 |
| today's model, rebuilt as of each week | 7.277 | 0 of 5 |
| market (books' lines re-scored) | 7.109 | - |
| FantasyPros | 7.145 | - |

The market is still about 2.4% better on fantasy points (it was 3.6% with the live boards). When the model sits 2+ points above the
market, it ends up closer only 36% of the time; its high calls are where it loses. By stat, the gap is largest on passing yards
(RMSE 77.6 vs 65.7; the model ran 15 yards low on QBs in 2026) and receptions.

**Leans (the model's side of each line; ~52.4% needed to beat -110):** with the median (the number a line is priced at):

| stat | leans | hit | stronger leans |
|---|---|---|---|
| rushing yards | 324 | 54.9% | 5+ yards gap: 56.8% |
| rushing attempts | 224 | 50.4% | 1+ gap 58.1%, 2+ gap 65.5% (n=55) |
| receiving yards | 718 | 52.2% | 5+ gap 54.3%, 10+ gap 56.1% |
| passing TDs | 114 | 57.0% | - |
| receptions | 648 | 48.1% | - |
| passing yards | 114 | 43.9% | (the 2026 passing under-projection) |
| passing attempts | 114 | 43.9% | 2+ gap 29% |

The median beats the mean as a lean on rushing yards (54.9% vs 48.5%), receiving yards and passing TDs, as expected. Five weeks is
a small sample: none of these is statistically above 52.4% yet (rushing attempts at a 2+ gap comes closest, CI 52-77%).

**Line movement - yes, the market moves toward our numbers.** Of the lines that moved between the earliest board built in a week and
the last board before kickoff, the share that moved toward the model's median: receiving yards 66.2% (526 lines, CI 62-70%), rushing
yards 63.6% (247), passing attempts 64%, passing yards 58%, receptions 57%; correlation of our lean with the move 0.3-0.4 for yards.
That is the classic sign of a model with real information (closing-line value). Caveat: the "open" is the earliest board built that
week (usually Tuesday-Wednesday), not the book's true opening number.

**Reading:** early in the week our numbers point where the market is heading; by kickoff the market has absorbed that and more, and
is the better number. The practical use is timing: act on leans early, before the line moves.

## 4. What is left, and whether it is worth doing

| item | what | evidence so far | recommendation |
|---|---|---|---|
| **W1 2026 passing surge** | QB starters threw 7.4% more and gained 4.6% more yards than projected (live weeks 3-7; the market also ran 4% low) | the watch trigger (over 5% through week 10) is effectively met; passing-yard and attempt leans lose | **Yes, next.** Test faster in-season trust for QB volume / a league pass-rate term on 2023-2026 with a pre-set bar |
| **Full calibration re-fit** | the v4 points lines and low-end knots were fit before 7+ ships | L4 (TE level, START-TE -0.023 offline) sits here too | **Yes.** One re-fit for the 66-flag set, harness-gated |
| **Silent games** (CFB) | count snap games with no stat in a player's history | backups' rates run high; the backup-TE and low-end patches treat the symptom | **Yes, worth testing.** Medium build; could let participation layers simplify |
| **Assumed out** (CFB) | a player with no injury listing who missed 2+ straight team games | 5.9% of projected points; only 21-30% of them play; a 0.4 factor fits | **Yes, cheap test** (partly absorbed by the low-end line already) |
| **Lean tracking / EDGES view** | the model's median vs each line, with the live lean record | rushing and receiving-yard leans 54-57% at bigger gaps; CLV positive | **Yes.** Build it now that medians exist; it also gathers the sample that tells us which leans are real |
| L5 RB game script by spread | favored teams' RBs beat the board | START-RB -0.008 offline, 5/5 | Worth a harness test (cheap) |
| L6 last game's snap share | a jump over the 3-game average is under-read | RB -0.010 (4/5), TE -0.007 offline | Worth a harness test (cheap) |
| Market availability (CFB) | a posted prop line = the player is active | availability information, not a projection blend | Your call; small, helps edge cases |
| M9 boom/bust band width | recentre and shrink the per-player width | display only, needs your OK | Low priority; do with the floor/ceiling question |
| Player-history floor/ceiling | steady vs boom-or-bust players | did not help the median; may help the band | Test only if you want floor/ceiling sharper |
| C12 two-seasons-back blend | never tested | - | Low value; skip unless idle |
| Opportunity x efficiency rework (CFB) | separate credibility for volume and rate | CFB measured efficiency wants ~5x more shrinkage | Large rework, uncertain; not now |
| Momentum, per-game rate bound, history levelling (CFB) | trends, outlier games, opponent-adjusted history | small or covered (rookie trust L3, matchup strength) | Not worth it now |
| Weather forecast shrink (CFB) | backtest reads recorded weather, live reads forecasts | a fidelity issue, small live effect | Low priority |
| D6 backtest leaks, N4 trade dates | traded-player team leak in backtests | hygiene | Do when convenient |
| UI: per-position default board, fewer comparison columns | items 4 and 5 of the UI review | you parked 5 | When you want them |
| Watches W3 (TE TDs), W4 (RB xTD), W5 (injury redistribution), W6 (backup TE) | lean right in recent seasons | need 2026 data | Re-check after the season |

Rule kept throughout: no model + market blend; the market and FantasyPros stay benchmarks.
