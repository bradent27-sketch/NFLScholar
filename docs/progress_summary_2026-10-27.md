# Progress summary (2026-10-27)

User: "give me a progress summary of where the model is at going forward, its most optimal leans, biggest weaknesses, next
steps, etc".

This follows `docs/progress_summary_2026-10-26.md` and covers 10-26 to 10-27: the sense-check audit
(`docs/model_sense_check_2026-10-26.md`) and the season-trend study (`docs/season_trend_2026-10-27.md`). It re-measures everything
with week 7 in (all but Monday's DAL-PHI game, which is not in the data yet). Every change has a dated entry in `docs/weekly_projections_methodology.md`.

## 1. Where the model stands

- **DEFAULT_FEATURES = 71**, unchanged since 10-25. The branch `open-items-2026-10-07` is pushed; `main` is untouched.
- **Since 10-26:**
  - Six data fixes from the sense check, none of them model flags:
    - four same-name joins (Justin Jefferson, Josh Allen 2022, Quinshon Judkins 2025 and others);
    - a guard for snap counts that are not posted yet;
    - the Bam Knight PFR-id snap join.
  - Three snap-share candidates failed their pre-set bars and stay off.
  - The season-trend study needed no change.
- **Live record, weeks 3-7.** These are the boards as they were built before each kickoff, on the paired start-worthy pool:

| | n | Model | Market | n | Model | FantasyPros |
|---|---|---|---|---|---|---|
| All | 713 | 7.32 | 7.10 | 747 | 7.33 | 7.09 |
| QB | 124 | 6.63 | 6.58 | 124 | 6.75 | 6.62 |
| RB | 179 | 7.26 | 7.05 | 209 | 7.30 | 7.17 |
| WR | 299 | 7.64 | 7.29 | 303 | 7.56 | 7.23 |
| TE | 111 | 7.26 | 7.19 | 111 | 7.36 | 7.06 |

  - Each pair is scored on its own paired pool, so the model's number differs between the two.
  - The market is 3.2% better, FantasyPros 3.3% better.
  - Week 7 was the closest week yet against the market: 7.04 vs 6.98.
  - Those boards were built by the model versions of their weeks, not today's.
- **Today's model, weeks 3-6.** This rebuilds each week with the current code and scores it on the same players. Start-worthy:
  - against the market: **7.31**, against 7.13 for the market (a 2.5% gap); the live boards scored 7.40 on the same weeks;
  - against FantasyPros: 7.36, against 7.19.
  - Today's model beats the live boards overall (7.31 vs 7.40), but it beats the market in none of the 4 weeks.
  - Week 7 cannot be re-scored yet: the data cache does not have Monday's DAL-PHI game. The week 7 rows above and in the
    leans below leave out those two teams.

## 2. Leans: the model's median against the posted line

Scored on weeks 3-7, using 1,600 player-weeks and the sharpest book's line. A lean needs about 52.4% to beat -110.

| | stat / slice | n | hit | 90% CI | note |
|---|---|---|---|---|---|
| **Best** | receiving yards, gap 10+ yards | 163 | **62.0%** | 54-69% | clear of 52.4% at the bottom of its interval on the largest sample; overs 66%, unders 58%; steady (62.2% on 143 through 10-25) |
| | passing TDs, gap 0.2+ | 80 | 63.7% | 53-73% | also clear of 52.4%, but small; all passing TDs 58.5% (118) |
| | rushing attempts, gap 2+ | 71 | 62.0% | 50-72% | small; the 1+ gap is only 53% |
| | rushing yards, unders | 214 | 57.0% | - | overs 44.8%: RB medians at the top run high (M10) |
| Break-even | receptions | 764 | 53.0% | 50-57% | 1+ gap 55% (94) |
| | receiving yards, all | 858 | 53.0% | 50-56% | gap 5+ 54.3% |
| | rushing yards, all | 368 | 51.9% | 47-57% | down from 53.4%; gap 10+ 48% |
| **Avoid** | passing yards | 138 | 44.9% | 37-53% | gap 10+ 34.5%; the unders lose (37%) |
| | passing attempts | 138 | 44.2% | 36-53% | gap 2+ 36% |

- **Timing is not an edge.** Lines move toward the model's number no more often than away: receiving yards 47.6% (189 lines),
  rushing yards 44.1% (93).
- Many slices were looked at, so read the best rows as promising, not proven.

## 3. Biggest weaknesses

1. **High calls.** When the model is 2+ points above the market, it ends up closer to the result only 34% of the time (122 rows);
   at 4+ points, 21%.
   - On those rows the model averaged 12.6, the market 9.6, the actual result 10.0.
   - This happens at every position (WR 30%, TE 35%, RB 41%, QB 29%) and in every week (27-42%).
   - Its low calls break even (50% at 2+ points below).
   - The spread is too wide at both ends. Projected 20+ came in at 22.4 projected vs 20.5 actual; under 5 came in at 3.4 projected
     vs 5.1 actual.
   - On history the early-season spread is also a little wide. The WR slope of actual on projected is 0.88 in weeks 3-7, against
     0.98-1.15 later; for QBs it is 0.90 early. In 2026 the live WR slope is 0.79, against 0.93 for the market.
   - This is the largest single place the market beats the model.
2. **2026 passing volume (W1).**
   - Passing yards is the worst stat against the market: RMSE 76.8 vs 65.7, bias -11.6 vs -5.8.
   - The passing-yard and attempt leans lose.
   - The pass-capacity trim carries the low QB volume into every WR/TE, which is part of the WR gap.
   - Week 7 flipped it: team attempts were projected at 1.046 of actual once passing cooled (26 teams so far). The surge may be
     over; week 8 decides.
3. **WR accuracy.** WR has the widest position gap (7.64 vs 7.29). Weaknesses 1 and 2 feed it, plus the WR target concentration
   that L2 only partly fixed.
4. **What the model cannot see.** Late injury news (Chase and McLaurin ruled Out on a stale pull), role changes the market prices
   first (A.J. Brown's return, NE's backfield in week 3), and fill-in vs permanent QB changes. These are process items: the user's
   overrides and a re-pull after inactives.
5. **Untested windows.** The backtests cover weeks 3-17. Weeks 1-2 rest on priors alone. Week 18 has the league's biggest shift
   (31.8 pass attempts, against about 33.6 in a normal week) and rested starters; the model has never been checked there.
6. **Smaller known biases.**
   - TE level: TEs run under and no fix has beaten held-out data (L4); TE1 targets drift under late in the season (W9).
   - RB rushing medians at 70+ projected yards run high (M10).
   - TE TDs ran over in 2024-2025 (W3).

What is solid:
- QB is within 1% of the market.
- Team totals are right on history.
- The season's pass-to-run lean is already followed.
- Over-projection is gone, and the board is -5% RMSE against September's.
- The data joins are clean after this week's fixes.

## 4. Next steps, in order

| # | item | why | size |
|---|---|---|---|
| 1 | **Week 8 board built on today's code** (Build board), with the injury pull refreshed after inactives | the week 8 board will be the first built with the same-name and snap fixes (Jefferson 11.7 -> 13.1) | - |
| 2 | **High-call audit** | weakness 1. For the rows where the model sits 2+ above the market, which input drives the gap (share, rate, matchup, vacancy, the luck or top-end steps)? Is it early-season noise that more prior weight early would fix? Error-first: decompose, then a named flag and a pre-set bar | medium (offline first) |
| 3 | **W1 decision with week 8 final** | under again: test M2 (in-season volume weights) on 2023-2026 with a pre-set bar. Over again like week 7: close W1 as a faded surge (the season's lean says passing cools from here) | small |
| 4 | **Lean tracking / EDGES view** | shows the receiving-yard 10+ gap edge live and builds the sample that proves or kills it; flags passing yards as "do not bet" until W1 clears | medium (UI) |
| 5 | **Week 18 check** before January | a weeks-18 harness run on 2022-2025: how bad is week 18, and does a "nothing to play for" rule help? | one harness run |
| 6 | Cheap tests: L5 RB game script by spread, "assumed out" (missed 2+ straight team games, no listing), M10 | offline leads already there | small each |
| 7 | Hygiene: D6 / N4 backtest trade-date leak | changes harness base arms only | medium |

**Watch:** lone-lead and line-routing rows in the weekly ledger (W7, W8); TE1 late-season targets (W9); W3-W6 after the season.

**Waiting on the user:**
- D3: a one-cell fix to the PFF manifest.
- M9: boom/bust band width.
- P1/P2: prop view, Sleeper team, "why" column.
- The stray untracked file `none` in the repo root.
- The live-data files (PFF imports, the FantasyPros injury report, QB1 choices, `.claude/launch.json`) stay out of commits unless
  asked.

Rules kept: one named flag per change; a pre-set bar before every harness run; DEFAULT_FEATURES changes only on the user's
sign-off; no model + market blend; heavy runs one at a time.
