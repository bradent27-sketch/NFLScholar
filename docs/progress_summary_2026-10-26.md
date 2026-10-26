# Progress summary (2026-10-26)

User: "good stopping point? Summarize where we are at ... a good report of where we are at and next steps."

Follows `docs/progress_summary_2026-10-24.md` (which covers how far the model came since September, the market comparison and the
lean record). This one covers the weekend of 2026-10-25 and the list from here. Every change below has a dated entry in
`docs/weekly_projections_methodology.md`.

## 1. Where the model stands

- **DEFAULT_FEATURES = 71** (66 on 10-24). All committed and pushed on branch `open-items-2026-10-07` (`main` untouched).
- **Against the market** (last measured 10-24, 2026 live weeks 3-7, start-worthy, same players): today's model rebuilt as of each
  week 7.28 RMSE, the market 7.11, FantasyPros 7.15 - the market about 2.4% better. That run predates this weekend's five ships;
  re-scoring with the 71-flag model is the first item below.
- **Against the model of September 16** (harness 2022-2025, measured 10-24 on the 66-flag set): start-worthy RMSE -5.0%, every game
  -6.1%, over-projection gone. Silent games added -0.008 start-worthy / -0.016 every game on top.
- **Leans** (posted lines, 2026 weeks 3-7): receiving yards 52.9% (10+ yard gap 62%), rushing yards 53% (5+ gap 57%), passing yards
  and attempts lose (the 2026 passing surge, W1). No line-movement edge.

## 2. What changed this weekend (2026-10-25)

| change | what | measured | status |
|---|---|---|---|
| `v2_silent_games` + `v2_silent_participation` | a RB/WR/TE game with snaps but no stat line counts as a 0-point game in his history; the WR/TE participation curve reads P(any snap) | PLAYED ALL -0.010 [-0.015, -0.005], START-ALL -0.008 [-0.015, -0.001], EVERY-ALL -0.016, no position worse | SHIPPED |
| Harness scores silent games as played (0) | backtests no longer skip the games a backup played without a stat | measurement fix | DONE |
| Points-scale re-fit (`v2_points_scale_v6`) | one re-fit of the points lines, low end and season anchor on the silent-games model | no variant beats the shipped scale held out | NOT SHIPPED (the scale stays) |
| RB layer / backup-TE discount variants | the layers' double count on played games (TE bands off / pieces / re-fit, RB participation 0.6) | each makes EVERY-TE or EVERY-RB worse | NOT SHIPPED (the layers stay as shipped) |
| `v2_keep_fully_excluded_no_prior` | a player whose every game this season was excluded as a partial game, and who has no prior-season role, keeps those games (Ja'Kobi Lane / Elijah Sarratt were off the week 7 board) | START-ALL +0.0005, EVERY-ALL +0.0003 (bar: +0.001); restored players under, not over | SHIPPED (passed its bar; the first version, for everyone, failed: it took veterans like Xavier Worthy off their prior-season line) |
| `v2_rb_lone_lead_vacancy` | when a backup RB is out and the lead back is the only back with a role, he takes 25% of the backup's volume (was 80%); the rest goes to the team's other healthy backs (depth backs, practice-squad backs) | 2021-2025, 121 games: RB1s picked up 24% of an absent RB2's carries and ~22% of his points; harness missed its bar on 13 games (START-ALL +0.0019) | SHIPPED on the user's call |
| `v2_ps_line_routing` | of those other backs, the one(s) with a posted market line this week take the share; otherwise the even split | not testable (no archived lines before 2026 week 3); reads only that a line exists | SHIPPED on the user's call |
| App: kept boards and lines | a line posted after a board build re-projects that team (cache key and per-team rulings carry the week's RB lines) | - | DONE |
| Phone export | rebuilt as CFB Scholar's (`ui/export_view.py`) | - | DONE |
| Tools | `scripts/eval_restored_players.py` (players only a variant projects), `scripts/eval_changed_rows.py` (only the rows a variant moves) | - | DONE |

Week 7, default board: Bijan Robinson 24.0 -> 19.2 carries (books 18.5-19.5), Khalil Herbert 0 -> 4.8 (the practice-squad back with
lines), Tony Pollard 20.6 -> 17.4 (books 16.5), Chuba Hubbard 19.7 -> 17.2 (books 17.0), Ja'Kobi Lane on the board at 5.3 pts. The
books are a check here, not an input.

**Questions answered (no change):**
- *Median or average lines for over/unders?* Median. Means read as medians over-rate by 0.38-0.54; the median model gives 0.50-0.51.
- *Is the top-end yardage dock still needed with medians?* Yes: the median curves were fit on docked boards; without the dock the
  receiving-yard over-rate at 70-80 / 80+ falls to 0.42 / 0.34.
- *RB rushing medians at 70+ projected yards run high* (actual median 68 vs 72.6; 80.5 vs 89.8; the means are right) - user: leave
  it for now (backlog M10).

## 3. Next steps, in order

| # | item | why | size |
|---|---|---|---|
| 1 | **Rebuild the week 7 board** in the app (Build board) | the saved board is from the 66/68-flag model | - |
| 2 | **Week 7 scorecard** after Monday night (`data/ledger_scorecard.py`, paired, pre-kickoff) and the week 8 board | the live record; first week of the lone-lead rule | small |
| 3 | **Re-score 2026 weeks 3-7 against the market with the 71-flag model** | tells whether the 2.4% gap moved this weekend | one run (~30 min) |
| 4 | **W1 - 2026 passing surge** | QB starters out-threw their projections by ~7%; passing-yard and attempt leans lose; the watch trigger is met | medium; pre-set bar, 2023-2026 |
| 5 | **Lean tracking / EDGES view** | the model's median against each posted line with the live record; builds the sample that says which leans are real | medium (UI) |
| 6 | **Assumed out** (CFB idea) | a player with no injury listing who missed 2+ straight team games plays only 21-30% of the time | cheap test |
| 7 | **L5 RB game script by spread**, **L6 last game's snap share** | offline START-RB -0.008 (5/5) / RB -0.010 (4/5) | cheap harness tests (one run) |
| 8 | **M10 RB rushing median at 70+** | user parked it; matters for rushing-yard overs at the top | small |
| 9 | Hygiene: D6 / N4 backtest trade-date leak | changes harness base arms only | medium |

**Watch (in the weekly ledger):** the lone-lead rule (do lead backs still out-score their carries when the backup sits - what the 13
harness games showed and the 121-game study did not); the line routing (do the backs it picks play); W3-W6 after the season.

**Waiting on the user:** D3 (one-cell PFF manifest fix), M9 (boom/bust band width), P1/P2 (prop view, Sleeper team, "why" column);
the uncommitted live-data files (PFF imports, FantasyPros injury report, QB1 choices, `.claude/launch.json`) stay out of commits
unless asked.

Rules kept: one named flag per change, a pre-set bar before every harness run, `DEFAULT_FEATURES` changes on the user's sign-off,
no model + market blend.
