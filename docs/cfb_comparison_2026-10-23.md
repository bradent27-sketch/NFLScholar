# NFL Scholar vs CFB Scholar: the weekly model, compared (2026-10-23)

User request: "compare really the whole app ... the biggest piece ... is the weekly model ... the data sources, how it's broken down
... how the model works, all of the different parameters ... Are some of them overlapping or redundant? ... see if potentially a more
minimal version is better."

CFB Scholar (`C:\New folder\CFBScholar`) was read, not changed. Its own docs were the main source: `docs/model_ledger.md` (one
index of every mechanism and its state), `HANDOFF.md`, `DATA_SOURCES.md`, the latest plans, and `data/weekly_model.py`.

## 1. At a glance

| | NFL Scholar | CFB Scholar |
|---|---|---|
| Model file | `data/weekly_projections.py`, 877 KB, 143 functions, 196 named constants | `data/weekly_model.py`, 485 KB, 73 functions, 118 constants |
| Flags | 100 defined, 65 on | 76 defined, 56 on |
| What a stat number is | the **mean** (expected fantasy value) | the **median** (the over/under number), with an **average** beside every stat (`<stat>_avg`, a fitted line floored at the median) |
| Main metric | RMSE and pairwise ranking on paired START pools, bootstrap CI, Holm; since 2026-10-22 also every-game scopes (no box score = 0) | MAE, rank correlation and over-rate (share of results above the number) on an "honest set" (recorded a stat in the previous game, projected a real role) |
| Gate | harness v2: full builds, 2022-2025 weeks 3-17 (60 weeks), pre-set bar | paired replay boards: 34 week-pairs (2023-2025 weeks 4-13, 2026 weeks 3-6), pre-registered criteria |
| Who it serves | fantasy lineups (points) first, props second (P1 parked) | props first (the user bets them), fantasy second |
| Availability | injury reports (historical replay), FantasyPros, user overrides, Questionable workload factor, participation curves | **full role or out**: out if missed 2+ straight team games (game log), out if unseen after 2 team games, conference availability reports, a posted prop line = active, a person's ruling wins |
| Stat structure | each stat blended as its own per-game rate (current vs prior, K per stat), volume rescaled by snap share | **opportunity x efficiency**, each half with its own opponent channel and its own credibility (efficiency shrunk ~4x harder) |
| Opponent model | defense profiles per stat (one pass, prior blend), PFF alignment/scheme residuals, role matchups | solved **two-way** ratings (offense and defense fit jointly), past games **levelled** by the opponent's rating, ratings applied at their measured strength (~half) |
| Calibration | one points line per position (v4), low end re-fit on every game, season anchor, luck step | a line per (position, stat) at the median, per-stat strength (`cal_strength`), a separate line fit on games played |
| Idea record | `docs/improvement_backlog.md` + 5,700-line methodology; 35 off flags still in the code | `docs/model_ledger.md`: every idea in one of five states; rejected code removed on 2026-09-30 (kept at a git tag) |

## 2. Data sources

Mostly sport-specific and not comparable: CFB reads the CollegeFootballData API, cfbfastR play-by-play and ESPN; NFL reads nflverse.
Shared tooling: PFF exports (weekly archive in both), Ourlads depth charts, The Odds API, Open-Meteo weather. Three real differences:

- **Snap counts for every snap-taker.** CFB rebuilds them from PFF's season-to-date blocking export (differenced week to week) and uses
  them to count a game a player was on the field for without a stat. NFL has them for free (nflverse snap counts list every
  snap-taker), but the model's stats table only keeps snap shares for games with a box-score line, so those games never reach a
  player's history. See 3.1.
- **Prop history.** CFB archives every line it polls (`data/prop_history.py`) and uses it twice: as a benchmark and as availability
  evidence (a lined player is active). NFL's ledger keeps market numbers as a benchmark only.
- **Weather in backtests.** CFB fits each weather term on archived FORECASTS and shrinks it by how well a forecast that many days out
  reads it (rain 0.6 -> 0.25, wind 0.45 -> 0.1 by lead time). NFL's backtest reads the recorded game weather (filled after the game),
  live reads a forecast - so the backtest credits wind with information the live board does not have.

## 3. Where CFB is ahead (ranked by expected value for NFL)

### 3.1 Games with no stat line (CFB `snap_appearances`, `silent_games_rate`)
A game the snap counts put a player on the field for, with no target, is a zero in his per-play rate. NFL drops those games, which
inflates every backup's rate (found from the other side on 2026-10-22: backup TEs run UNDER in box-score games and OVER once every
game counts, and 21-32% of a TE2/TE3's expected games are snaps without a stat). The two patches shipped yesterday
(`v2_backup_te_discount`, `v2_low_end_points_line`) correct the symptom on the final number; counting silent games in the history fixes
it at the source and could let some of the participation layers go (see 5).

### 3.2 Multiplier strengths are fitted, not assumed (CFB `scripts/week_review/factor_gamma.py`, `matchup_elasticity`, `qb_chain_cleanup`)
CFB fits the strength each multiplier should have. Run on NFL today (`scripts/audit_factor_gamma.py`, capture traces 2021-2025, every
game, RMSE; offline, so about half of a gain is expected to survive a full build):
- **Matchup on TD stats: best strength 0** at every position (WR/TE/RB receiving TDs, QB passing TDs; switching it off -0.17% to -0.44%,
  3-4 of 5 seasons; box-score rows agree). On yards and volume the best strength is about 0.5 for many stats - the same "applied about
  twice too hard" CFB measured.
- **QB rushing game script: best strength 0**, -1.2% when switched off, 4 of 5 seasons, both views.
- **Pace: inert** (the multiplier moves a projection by about 1%; switching it off costs at most 0.05%).
- **Game total (implied points) under-applied** on volume: best strength 2-3 for QB attempts/completions (5/5 seasons, every game), RB
  targets/receptions/carries and WR targets (3-4/5). CFB's QB chain clean-up landed in the same place (game total x3.5 on yards).
- **Vacancy hand-out to WRs half-strength** (best 0.5, 4/5), RBs full. **Pass-capacity cut on TEs and RB catches half-strength** (0.5,
  4/5, both views).
- Participation earns its keep (switching it off costs up to 10% every game).

### 3.3 Who plays (CFB `assumed_out`, `assume_out_unseen`, `market_availability`)
NFL has no rule for a player with no injury listing who keeps not playing. On the 2021-2025 board, players the model expected to play
after two straight team games with no snap: RB 2,127 rows (21% played; projected 2.0 points, scored 0.8), WR 2,304 (25%; 2.2 vs 0.9),
TE 907 (30%; 1.9 vs 0.7) - 5.9% of all projected points. Zeroing them (CFB's rule) is worse on NFL data (a quarter come back); a
factor of about 0.4 fits. Partly absorbed by yesterday's low-end points line. A posted prop line as evidence the player is active is
availability information, not a projection blend, but it needs the user's call.

### 3.4 A median and an average for every stat
For props the over/under number is the median; for lineups it is the mean. CFB carries both. NFL's parked P1 (calibrated stat lines for
props) would be the same design: keep the mean as is and add a median beside it, fitted per (position, stat).

### 3.5 Smaller ideas
- **Opportunity x efficiency with separate credibility**: CFB measured that every efficiency channel wants a ~5x larger shrinkage
  constant and volume channels slightly smaller. NFL blends each stat per game, which mixes the two (M2 hinted the same: TE yards K x2).
- **Momentum**: a trailing OLS trend multiplier (QB/RB/TE, strength 0.15). NFL has recency weighting only; the rookie under-projection
  fixed by L3 is partly a trend.
- **Per-game bound on a rate** (`game_bound`, in CFB's current plan): one 75-yard run does not set a back's yards per carry.
- **Opponent-adjusted history** (`history_leveling`): bigger in college (huge mismatches); in the NFL the matchup step already exists and
  the audit says it is too strong, so this is low priority.
- **Weather forecast shrink by lead time** (see 2).

## 4. Where NFL is ahead (worth porting to CFB)
- Harness v2: mean-consistent RMSE plus pairwise ranking, paired pools, weekly bootstrap CIs, Holm correction, pre-set bars; per-week
  output and saved boards; every-game scopes.
- The error-first audit (error budget, residual scan, market gap) and the fast screen's validation.
- Disk caches for every download (CFB lesson 9 is the same failure).
- QB1 resolution rules (latest full-snap start, returning starter, probable-starter fallback), expected-points luck, season anchor, xTD.

## 5. Are NFL's parameters redundant? (leanness check)

**The inventory.** 100 flags, 65 on. Of the 35 off, most are rejected candidates still in the code (CFB removed its rejected code on
2026-09-30 and kept it at a git tag). The 65 on group into families that stack several knobs on one quantity:

| family | flags on | what they all touch |
|---|---|---|
| trust in this season's rate | v2_adaptive_volume, v2_role_change_by_stat, v2_stat_k_by_pos, v2_role_confidence_reversed_wrte, v2_fallback_prior_fast_trust, v2_offense_prior_blend | the same blend constant K |
| does he play / how much | role_volume (core), v2_wrte_participation, v2_rb_participation, v2_rookie_backup_wr_dampen_narrow, v2_uncalibrate_zero_rows, v2_backup_te_discount | the snap-share / participation discount |
| game script | v2_offense_script_pool_blend, v2_script_neutral_volume, v2_script_neutral_level_fix, v2_rb_rush_yards_script_neutral (+ game total elasticity) | volume vs the projected margin |
| RB volume conservation | v2_rb_carry_budget (+keep_tds), v2_rb_in_season_share_conservation, v2_rb_depleted_room_fill, v2_preseason_rb_allocator | the backfield's carries |
| pass capacity | v2_pass_capacity + injury_neutral_claim, keep_tds, matchup_flex | the team's target budget |
| vacancy modifiers | v2_vacancy_absence_decay, v2_vacancy_bump_cap, v2_receiver_vacancy_pecking_order, v2_vacancy_lead_share_050 | an injured player's volume |
| defense extras | v2_channel_matchups, v2_continuous_roles, v2_pff_alignment_matchup, v2_scheme_matchup, v2_pff_defense_prior_blend, v2_defense_blowout_discount | the matchup multiplier |
| TD extras | v2_td_prior_credibility, v2_td_two_year_prior, v2_xtd, v2_xtd_rush_outside_zone | TD rates |
| late point steps | v2_season_anchor, v2_questionable_workload, v2_expected_points_luck, v2_top_end_yardage, v2_low_end_points_line (+ calibration) | the final points total |
| QB rates | v2_qb_volume_blend, v2_qb_td_blend, v2_qb_implied_td_boost | QB1 volume and TDs |
| environment | v2_game_total_elasticity_perstat, v2_weather_adjustment, v2_pace_alpha_cap | game-level multipliers |

**The test.** One full harness run removes each family as a whole (plus `v2_pff_alignment_matchup` alone, which the 2026-09-23
re-ablation flagged as possibly hurting and was never decided). Decision rule written before the run (methodology 2026-10-23).
Results: see section 6.

## 6. Family ablation results

(pending) The first launch (13:16) died out of memory at 13:47, before its first week finished: CFB Scholar's
`ablation_pairs.py` runs, two UI-comparison preview servers and two long-running Streamlit apps were on the machine with it. It is
queued to start again from `.sweeps/wait_then_run_leanness.ps1`: it starts once no other heavy Python run has been seen for 10
minutes and at least 12 GB of commit memory is free (progress in `.sweeps/leanness_waiter.log`). Same groups, same decision rule.

## 7. The weekly board UI: Weekly Rankings vs CFB's Weekly Model

Both tabs were built for 2025 (NFL week 7, CFB week 8) and read side by side. Ranked by how much each change helps a weekly decision
per unit of work (S/M/L = rough size).

1. **Default order = the model's own number (S, your call).** The first column, `Rank`, averages FantasyPros' and the model's
   positional ranks, so the order a reader scans first is a model + FantasyPros blend. That predates the rule that market and
   FantasyPros are benchmarks only. CFB sorts by its own headline stat and puts the market beside it. Proposal: default sort by
   `Model Proj Pts` (positional rank from the model), keep FantasyPros and market as comparison columns, and make the blended rank an
   opt-in sort.
2. **One scoring control (S).** The header's Full PPR / Half-PPR / Standard control drives only Player Search; the tab has its own
   `Scoring` selectbox. The two can show different settings on the same screen. Make the header control app-wide (the tab reads it)
   or take it out of the header.
3. **The board survives a restart, and a ruling re-projects one team (M).** NFL keeps a built board in session state, so a restart or
   a new browser session costs a full rebuild of several minutes. CFB pins the board to disk with a keep window, and an injury
   ruling re-projects only that team (`only_teams`), so moving around the tab never rebuilds the board.
4. **Per-position tables by default, about 15 columns each (M).** In the default SUPERFLEX view one mixed table carries every stat
   column (passing, rushing and receiving), most of them "—" for any one player. Unused stats are hidden only after you narrow the
   position. CFB shows one table per position, stats in prop order, sorted by that position's headline stat. SUPERFLEX/FLEX would
   become a compact points-only list.
5. **Fewer comparison columns (M).** Today up to 12 leading columns + the stat line + 9 trailing ones: three point projections,
   three source ranks, ECR, model vs ECR, market coverage, and wind, temperature and precipitation as three columns. Proposal: one
   `Proj` (model) column, one `vs FP` and one `vs Market` delta (the other sources' full numbers in a tooltip or a Compare view),
   weather as one chip column flagged only when it matters, season and last-5 snaps as one column, and one rank column with
   "Rank by: Model / FantasyPros / Market". A matchup column where higher = better for the player reads faster than `Def Rank`
   (1 = softest).
6. **Column tooltips instead of the long caption (S).** About 600 words of caption sit under the table. CFB keeps one summary
   line (players, teams, built from weeks before N, league pace, players priced) plus short per-view notes. Most of the NFL caption
   is per-column definitions, which `column_config(help=...)` shows on hover.
7. **A compact control row (S).** NFL: Season / Week / Scoring, then a "This app's weekly model" heading, the live data hub, two
   freshness captions, and the Build button on its own row. CFB: Season / Week / Run model in one row, then one line saying when
   the board was built and what a rebuild does. Fold the freshness lines into the data-hub label (e.g. "Live data · FantasyPros
   2 h ago · injuries 35 min ago").
8. **Move "Upload a weekly rankings export" into the live data hub (S).** It is a data input but renders below the board.
9. **A GAMES view inside the tab (M-L).** CFB's game card shows model vs market team totals, spread and total, weather, each
   team's projected volume (plays, pass attempts and yards, rush attempts and yards), and the availability flags with their
   reasons ("assumed out: no recorded stat in 4 straight team games ...", "missed last game"), then both rosters. NFL has every
   piece (Game Slate tab, the model's team volume, the new backup-TE and low-end steps), but not on one card next to the board.
   This makes a team-level mistake visible (a team projected for 45 pass attempts) without opening player dialogs one by one.
10. **An EDGES view (M, lower priority while props are parked).** CFB lists model vs line, largest disagreement first, plus a line
    shop. It only matters for props (P1).
11. **Split `ui/tabs/rankings.py` (261 KB) into view modules (L, maintenance).** CFB keeps the board, edges, games and dialogs in
    separate modules and computes team colors and matchup shading once per board. Measure rerun time before treating this as a speed
    fix.

**The user's calls (2026-10-23)** and what was built the same day:
- 1 (default order): no change.
- 2 (one scoring control): the header's Full/Half/Standard control is gone; it drove Player Search and Matchup Analyzer, which each
  carry it in their own control row now (`ui/components.scoring_control`, one shared choice).
- 3 (board kept, team-only reload): `ui/board_pin.py` saves a built board to `data/cache/boards/` (git-ignored, 12 hours, dropped
  when the model code or default features change) and a changed ruling re-projects only the teams whose rows changed
  (`build_weekly_projections(only_teams=...)`, exact against a full build; methodology 2026-10-23). A re-projected board is recorded
  in the ledger as a rebuild was.
- 4: explained (below), not built yet.
- 5: left for now.
- 6 (less text): one control row with Build board; no "This app's weekly model" heading; the FantasyPros and injury freshness ride in
  the Live data pulls label (a stale injury report near kickoff is still a warning); the rankings upload is a Live data pulls tab;
  the ~600-word caption is one summary line, its definitions are column hover help.
- 7 (GAMES view): `ui/games_view.py` (pure parts, a port of CFB's) and `ui/tabs/rankings_games.py` (widgets); BOARD / GAMES views
  over the same built board. Differences from CFB's, each because the board differs: a PTS column; health flags from this app's
  availability columns (injury report, FantasyPros, your rulings, the QB1 choice); the MODEL score is `data/points_fit.py`
  (display only); team nicknames in the scoreboard rows (full names overflowed it).

**Item 4, per-position tables, spelled out.** CFB's BOARD is four tabs - QB, RB, WR, TE - each its own table with only that
position's stats in prop order (a WR table: targets, receptions, yards, TDs, the line). NFL's BOARD opens on SUPERFLEX: one table
mixing every position, so it carries every stat column (passing, rushing and receiving, about 25 columns), and for any one player
most of them are blank. NFL does hide the unused stat columns once you pick a single position with the buttons above the table; the
difference is the default view and that the mixed view is the one you land on.
