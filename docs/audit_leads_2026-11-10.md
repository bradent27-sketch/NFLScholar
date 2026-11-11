# Audit leads for execution (2026-11-10)

User: "audit the model as a whole ... create new leads for model improvement as well as model display, UI, and betting
recommendations". The user approved every lead in the compact list except A11 (game simulation), B1 (historical line backfill),
B7 (staking) and C4 (board presets). This file is the work order for the agents that execute them.

Two leads were added during verification (D0, and the week-9 numbers under "State"). Nothing in the model changed in this audit.

## How to use this file

- Work the phases in order. Phase 0 unblocks the record; phase 1 builds the measurement that phases 2-3 are judged by.
- Each lead has: **Why** (evidence), **Where** (files and line anchors as of commit 752d1e0; re-find by name if they moved),
  **Do** (steps), **Bar** (written here so it is pre-set), **Size**.
- When a lead moves, update its status in `docs/improvement_backlog.md` and add a dated entry to
  `docs/weekly_projections_methodology.md`.

## Standing rules (they apply to every lead)

1. One named flag per model change, off by default. Shipping into `DEFAULT_FEATURES` needs the user's sign-off.
2. Error first: measure the bias offline before building. A lead whose step-1 read shows nothing is closed with its numbers.
3. A bar is written before any harness run (`.sweeps/<name>_bar.md`). The bars below are the starting text.
4. No model + market blend. The market may decide **which bets are shown**; it never moves a projection.
5. Week 18 is out of scope. Week 1 is in scope.
6. Heavy runs go one at a time through `.sweeps/wait_then_run.ps1`, launched detached (the full harness is about 2 h 15 min).
7. `data/ledger/` and `external_data/props_archive/` are permanent records: never delete, rename or rewrite a file in them.
8. Leans and closing-line value use posted lines, never the `Mkt <stat>` implied means.
9. Stat-column changes that leave points alone are judged held out by season on stat MSE (the M13 pattern), not by the points
   harness. Cross-stat consistency (receptions <= targets) is checked every time.
10. Before re-testing an idea, check `docs/retired_flags.md` and the Closed section of the backlog.
11. UI text stays minimal: long notes go in `quiet_note`, tables are centered with a bold header.
12. Anything that runs on a schedule or sends a notification needs the user's explicit OK first.

**What the exclusions mean.** Without B1 every betting rule rests on 2026 only (about 3,800 priced props so far), so rules must
be few, simple, written down before they are scored, and shrunk hard. Without B7 the app shows no stake; the record uses a flat
1 unit per bet.

## State on 2026-11-10

Prop record, pre-kickoff boards against priced lines (Pinnacle, then DraftKings), `scripts/prop_scorecard.py`:

| | weeks 3-8 | week 9 (first week on the new stat columns) |
|---|---|---|
| priced props | 3,231 | 586 |
| Brier, model / book | 0.261 / 0.248 | 0.259 / 0.248 |
| model's average P(over) / share that went over | 0.459 / 0.485 | 0.462 / 0.462 |
| 5%+ edge, receiving yards + receptions + rushing yards | 53.0% (1,555) | 51.6% (275) |
| 5%+ edge, passing props | 46.1% (334) | 40.8% (49) |
| 5%+ edge, rushing attempts | 49.4% (180) | 43.2% (44) |

Weeks 3-8, bets at a 3%+ edge unless stated:
- All stats at 5%+: 51.4% (2,074), about -1.8% at -110. The shrink toward 50% that minimises Brier is 0.20 (it was 0.34
  through week 7).
- Edge size carries no signal: 51.5% at 3%+, 51.4% at 12%+. On the three skill stats, gaps over 25 points hit about 48% (128).
- The model leans under on 64% of bets. Unders 52.7%, overs 49.4%. QB props 47.2% (610).
- Rushing yards: a blind under hits 57.4% (418 lines). The model's unders hit 59.4%, its overs 40.2%. Receptions blind under
  52.3%, receiving yards 50.7%.
- Three skill stats: the model on the same side as the book's price lean 56.0% (979); against it 49.9% (843).
- By book: against DraftKings lines 57.4% (176), against Pinnacle 51.0% (2,306).
- Week 9 is one week; read it as a first look only.

Read: the means are close to the market's, the probabilities carry about a fifth of the confidence they show, and no slice is a
proven edge at real prices yet. Receiving yards and receptions are the only stats where the model adds something over a blind
side.

---

## Phase 0. Fixes (do first; small)

### D0. Every board built since 2026-11-01 is dropped from the record

**DONE 2026-11-10** (steps 1-4 below). After the fix all 42 ledger files read as Full PPR, the record has week 9 (321
pre-kickoff player-weeks with a box score), and week 8 takes the 11-01 16:50 board for most players (256 of 299).

- **Why.** `data/ledger_scorecard.py` `infer_scoring` (line 45) decides a ledger file's scoring mode by checking that the stat
  line, scored, reproduces `Raw Model Proj Pts`. Since the stat recal shipped, the stat columns are re-predicted, so nothing
  fits (mean error 0.31 against the 0.05 limit) and the function returns None. `load_scored_ledger` (line 106) then drops the
  file. Result: the Track Record page and the prop scorecard stop at week 8, week 8 is scored from an older board, and weeks
  9-10 are missing. Scoring the `_pre_recal_<stat>` columns reproduces the raw points (error 0.011).
- **Where.** `data/ledger_scorecard.py` lines 45-60 and 106-132; `data/prediction_ledger.py` `build_ledger_frame` (line 98);
  `tests/test_ledger_scorecard.py`, `tests/test_prediction_ledger.py`.
- **Do.**
  1. `infer_scoring`: score from `_pre_recal_<stat>` where the column exists (fall back to the stat), then compare.
  2. `build_ledger_frame`: write a `scoring` column from now on; `infer_scoring` reads it first when present. The caller
     (`record_board`) must be handed the scoring mode; trace it from `ui/tabs/rankings.py`.
  3. `load_scored_ledger`: a file whose mode is None is kept under the majority mode and counted in `info['unknown_scoring']`,
     so a future break shows up as a number instead of missing weeks.
  4. Tests: a post-recal frame infers Full PPR; a frame with a `scoring` column skips inference.
- **Bar.** `load_scored_ledger(2026, ...)` returns week 9 with about 500 pre-kickoff rows; week 8 picks the 11-01 16:50 board for
  Sunday players; the Track Record page shows week 9. Existing ledger files are not touched.
- **Size.** Small.

### D2. A malformed props snapshot crashes the prop scorecard

**DONE 2026-11-10.** `load_archive` skips and names an unreadable snapshot (the scorecard header says how many); the archive
writer writes a temp file and moves it into place. The bad file stays in the archive. The `data/odds_weekly.py` readers already
caught it.

- **Why.** `external_data/props_archive/2026_wk09_20261103T001625Z.json` is not valid JSON (about 2 MB, breaks at character
  1,549,272; probably an interrupted write). `load_archive` has no guard, so `scripts/prop_scorecard.py` raises.
- **Where.** `scripts/eval_posted_line_record.py` `load_archive` (line 47); `data/odds_weekly.py` `_archive_snapshot` (line 191);
  any other reader of the archive (`data/odds_weekly.py` `_players_with_lines`, `_rb_lines_by_team`).
- **Do.** Readers: catch `json.JSONDecodeError`, skip the file, report the skipped count. Writer: dump to a temp file in the
  same folder, then `os.replace`. Leave the bad file where it is.
- **Bar.** The scorecard runs with no patching and prints "1 snapshot skipped"; a test writes a truncated file and reads past it.
- **Size.** Small.

---

## Phase 1. Measurement (everything in phases 2-3 is judged by this)

### B3. Score against the right baseline and the real price

**DONE 2026-11-11.** `data/prop_record.py`; the sharpest-book view reproduces the old record exactly; the new tables are in
`scripts/prop_scorecard.py` and on the Track Record page. Read: no lift over the blind under overall (-1.4% ROI); receiving
yards +3.4%; rushing-yards blind under +7.7%; DraftKings 52.2% vs Pinnacle 50.8% at 5%+ (methodology entry of the same date).

- **Why.** The record grades every bet at -110 against the sharpest book. That hides three things measured above: a blind under
  already wins on rushing yards, the model does best when it agrees with the price lean (where the price is worse than -110),
  and the softer book is where the hits are.
- **Where.** `scripts/prop_scorecard.py`: `priced_lines` (line 60) keeps only the sharpest book; `record_tables` (line 111).
  The archive rows carry `provider`, `line`, `over_payout`, `under_payout`, `p_over`.
- **Do.**
  1. Move the record logic into a module (`data/prop_record.py`) so the script and the Track Record page share it.
  2. Keep one row per (player, stat, book), with both prices.
  3. Add to the record: ROI at the actual price of the side taken; a per-book table; a per-side table; and for each stat the
     three blind baselines (always under, always over, always the book's favoured side) with the model's lift over the best one.
  4. Report the shrink per stat, not only overall.
- **Bar.** Output reproduces today's headline numbers on the sharpest-book view, and the new tables are present. No model change.
- **Size.** Small to medium.

### D1. The ledger saves what a bet decision needs

**PARTLY DONE 2026-11-11.** Medians, the posted line and book, the priced line with both prices and the book's P(over), the
model's raw P(over) and (B2) the tier are stored; a new file round-trips through `load_scored_ledger` (test). Still to add once B8 defines them: the flags.

- **Why.** The ledger keeps means only. Medians, posted lines, prices, P(over) and the tier are recomputed later with later code,
  so the record is not what was actually shown.
- **Where.** `data/prediction_ledger.py` `build_ledger_frame`; `data/weekly_median.py` `attach_medians`;
  `ui/games_view.py` `posted_lines`.
- **Do.** Add per stat: `Median <stat>`, the posted line with its book and both prices at build time, raw P(over), shrunk
  P(over), tier and flags (B2, B8). New columns only; old files stay readable.
- **Bar.** A new ledger file round-trips through `load_scored_ledger`; `tests/test_prediction_ledger.py` covers the columns.
- **Size.** Small. Do it after B2 defines the tier.

### B6. Closing-line value

- **Why.** Win rate needs seasons to separate 52% from 54%; closing-line value answers in weeks. The earlier read (lines moved
  toward the model 47.6% on receiving yards, 44.1% on rushing yards) says the leans have had no closing-line value so far. It
  must be measured per tier. Pulls are manual and irregular (17 to 53 snapshots a week), so "open" and "close" are whatever
  happened to be pulled.
- **Where.** `data/odds_weekly.py` `weekly_props` (line 339), `_archive_snapshot`; new `data/bet_log.py`;
  `data/prop_record.py` (B3).
- **Do.**
  1. A headless pull script (`scripts/pull_props.py`) that calls `weekly_props(force=True)`.
  2. Ask the user to approve a schedule (suggested: Tue 10:00, Thu 12:00, Fri 18:00, Sat 18:00, Sun 11:30 ET, and 60 minutes
     before each kickoff window). Do not create it without the OK.
  3. Bet log, append-only (`data/bet_log/<year>.parquet`): time, week, player, stat, side, line, price, book, raw and shrunk
     P(over), tier, feature hash. Written when a board is built, for every tiered row.
  4. After the week: closing line, closing fair probability, result. Two measures per bet: line value (close minus bet line, in
     the bet's direction) and price value (closing fair probability of the side minus the implied probability paid).
  5. Scorecard section: average of both by tier, stat and side, with the share of bets that beat the close.
- **Bar.** A tier counts as "beating the close" when its average price value is above 0 with a 90% interval clear of 0 on 200+
  bets. This is the primary weekly number once it exists.
- **Size.** Medium.

---

## Phase 2. Deciding when an edge is bettable

### B2. A bettable tier in place of gap %

**DONE (v0) 2026-11-11.** `data/prop_tiers.py`, bar in `.sweeps/prop_tiers_v0_bar.md`. Status counts boards from 2026 week 10
on only (the rules came from weeks 3-8); every tier reads tracking. TIER column on the game view's props table; tier record on
Track Record; ledger stores 'Tier <stat>'. Fair odds at the best available price wait on B4 (v0 uses the sharpest priced book).

- **Why.** Edge size does not predict the hit rate; side, stat, book and agreement with the price lean do. The app sorts and
  colours by gap %, the weakest signal.
- **Where.** New `data/prop_tiers.py`; inputs from `data/prop_probability.py` `prop_probabilities` and the sheet built by
  `scripts/prop_scorecard.py` `run_sheet` (line 182).
- **Do.**
  1. Shrunk probability: `p' = 0.5 + s_stat (p - 0.5)`, `s_stat` from the record (B3), refit weekly using earlier weeks only.
  2. Fair odds and EV at the best available price (B4) from `p'`.
  3. Tiers, version 0 (write these into `.sweeps/prop_tiers_v0_bar.md` before scoring them):
     - **No bet:** passing yards, attempts, TDs, interceptions and rushing attempts while their switch is off (B8); any gap
       between the model's and the book's probability above 0.25; any row with a B8 flag.
     - **A:** receiving yards, receptions, or a rushing-yards under, with the model on the same side as the book's price lean,
       gap 0.03 to 0.25.
     - **B:** the same stats with the model against the price lean.
     - **Track:** everything else. Shown, never recommended.
  4. Each tier carries its running record: bets, hit rate, interval, ROI at price, closing-line value. A cell's hit rate is
     shrunk toward 50% (prior weight about 200 bets) before it is shown.
  5. Score the tiers forward-chained: tiers for week w use weeks before w only.
- **Bar.** A tier is labelled "bettable" only when its ROI at the actual price has a 90% lower bound above 0 on 300+ forward
  bets. Until then every tier reads "tracking". Changing a tier's definition restarts its record.
- **Size.** Medium.

### B8. Automatic no-bet rules

- **Why.** The losing groups are known: passing props all season, gaps over 25 points, players back from missed games (high
  calls on returning players were right 24% of the time), Questionable players.
- **Where.** `data/prop_tiers.py`; board columns `Injury Status`, `Availability`, `Games This Season`;
  `resolve_inseason_qb1s` in `data/weekly_projections.py` for a QB change.
- **Do.** One flag per rule, each with its own record line:
  1. Stat switch: off when the stat's last 150 bets (or 4 weeks) at 3%+ are under 50%; back on above 52.4% on 150.
  2. First game back after 2+ missed team games.
  3. Listed Questionable or Doubtful.
  4. The team's QB1 differs from its last game.
  5. Gap above 0.25.
  6. Line posted at one book only, or the sharpest book has no line.
  7. "Market knows something": the line sits well below the player's own season average while the model sits at it (the M17
     label). Fit the threshold on the record; display only.
- **Bar.** A rule blocks a bet only if its flagged rows hit under 50% on 60+ bets in the forward record; otherwise it shows as a
  flag and does not block.
- **Size.** Small to medium.

### B4. Line shopping

- **Why.** The weekly pull covers PrizePicks, Underdog, DraftKings and Pinnacle (`WEEKLY_BOOKS`, `data/odds_weekly.py` line 71).
  The record is better against the softer book. A FanDuel parser already exists and is not in the weekly pull.
- **Where.** `data/odds_sources.py` `fetch_fanduel_lines` (line 1355); `data/loaders.py` `fetch_nfl_player_props` (line 1677) and
  `config.ODDS_API_PLAYER_PROP_MARKETS` for the Odds API books; `ui/games_view.py` `LINE_BOOKS` (line 421).
- **Do.**
  1. Add FanDuel to the weekly pull and archive.
  2. Optional, needs the user's OK because it spends Odds API quota: pull event props for the week's games into the same archive
     with both prices per book. Estimate the request count first and report it.
  3. Per (player, stat): best over and best under (price and book), and the spread of lines across books.
- **Bar.** The archive rows for added books have `scorable` true with a de-vigged `p_over`; `tests/test_odds_sources.py` covers
  the new path; the per-book record (B3) shows them.
- **Size.** Medium.

### B5. Top-down bets (the sharp price against soft and pick'em lines)

- **Why.** PrizePicks standard, demon and goblin rows and Underdog rows are archived every pull (several thousand a snapshot) and
  never scored. Their lines move slowly and pay fixed amounts, so a line that sits off the sharp price is an edge that needs no
  projection. This is bet selection only; no projection reads it.
- **Where.** The archive; `data/market_devig.py`; `data/prop_probability.py` (the empirical shape, to move a probability from the
  sharp book's line to the pick'em line).
- **Do.**
  1. Confirm with the user that market-based bet selection is acceptable alongside the no-blend rule, and get the current payout
     tables as config (do not hard-code them from memory).
  2. Fair probability at the pick'em line: take the stat's empirical shape, centre it so that P(over the sharp line) equals the
     sharp book's de-vigged price, then read it at the pick'em line.
  3. Score weeks 3 onward from the archive: hit rate of legs whose fair probability clears the per-leg break-even, by stat and
     product.
  4. Show the model's side as a second column (agree / disagree).
- **Bar.** Surface it only if flagged legs beat their break-even with a 90% lower bound above it on 300+ legs.
- **Size.** Medium.

### B9. Score the markets that are archived but never graded

- **Why.** Each snapshot holds anytime TD (about 640 rows), longest reception and rush (about 570), combos (pass + rush yards,
  rush + receiving yards), first-quarter and first-half stats, and fantasy-points lines. Thinner markets are where a model has
  the most room; they also carry more juice, so score before showing anything.
- **Where.** `data/odds_sources.py` (market names and the `scorable` flag); `data/prop_probability.py` `PROP_MARKETS`;
  `scripts/deep_audit.py --stage props2` (how engines are fit); `ui/games_view.py` `NO_LEAN_STATS`.
- **Do, in this order (stop at the first that fails its bar):**
  1. **Fantasy-points lines** against `Model Proj Pts`: no engine needed; add to the scorecard.
  2. **Anytime TD:** P = 1 - exp(-projected rushing + receiving TDs); the deep audit found that form calibrated (Brier
     0.177-0.190). Score at the posted price; many books post one side only, so report ROI at price, not a de-vigged Brier.
  3. **Combos:** fit the same ratio tables on the combined result over the combined mean (no explicit correlation model).
  4. **Longest reception / rush:** an empirical table of the longest play by projected receptions (or carries) and yards per
     play.
  5. First-quarter and first-half stats: only after 1-4.
- **Bar.** Per market: Brier within 0.005 of the book where a two-sided price exists, and the model's side at a 5%+ gap above
  53% with a 90% lower bound above 50% on 200+ bets, before it appears as a lean.
- **Size.** Small each; A8 is a prerequisite for showing combos next to their parts.

### B10. Inactives automation

- **Why.** Live boards beat rebuilt ones (52.3% against 50.9% at a 5%+ edge): the edge that exists is fresh injury and role
  information. Week 5 lost Chase and McLaurin to a stale pull.
- **Where.** `ui/board_session.py` `build_board` (line 106); `data/fantasypros_availability.py`; `data/odds_weekly.py`
  `weekly_props`; `data/prediction_ledger.py` `record_board`, `latest_ledger_paths`; `ui/freshness.py`.
- **Do.**
  1. `scripts/refresh_pre_kickoff.py`: pull injuries, pull props, rebuild the board without Streamlit, record it to the ledger,
     and diff against the previous board for the week.
  2. Diff output (`.sweeps/what_moved_wkNN.md`): players whose points moved 1+ or a stat 10%+, each with the reason (status
     change, teammate out, line routing, new line).
  3. Check the build runs outside a Streamlit session (the loaders use `st.cache_data`); the harness scripts already do this.
  4. Schedule (75 minutes before each kickoff window) only with the user's OK. The machine is shared with the CFB jobs: check
     free memory first and skip the run if a heavy job is active.
- **Bar.** A headless run produces the same board as the app's Build button on the same inputs (use
  `.sweeps/board_identity.py`), writes one ledger file and one diff file, and finishes in under 2 minutes.
- **Size.** Medium.

---

## Phase 3. Presentation

### C1. A Props view as the main betting surface

- **Why.** The stated purpose is prop betting, yet the prop output is a CSV from a script and a table sorted by gap %
  (`ui/games_view.py` `props_table`, line 494). P(over) is never shown in the app.
- **Where.** `ui/tabs/rankings.py` line 4259 (`st.tabs(['BOARD', 'GAMES'])`): add `PROPS`. New `ui/props_view.py` for the frame
  and styling (keep `rankings.py` from growing). Data from `data/prop_tiers.py`.
- **Do.**
  1. One row per priced prop: player, team and opponent, stat, line, best book and price, model median, shrunk P(over) for the
     side, fair odds, EV %, tier chip, the tier's record ("54% (120)"), line move since first pull, flags.
  2. Default sort: tier, then EV. Filters: game, stat group, tier, book, side. Track-tier rows hidden by default.
  3. Colour only the tier chip and the EV cell. No gap-% shading.
  4. A one-line header: bets shown, tiers' records, and the active no-bet switches.
  5. Change `props_table` in the GAMES view to the same order and columns (fewer of them).
- **Bar.** With lines loaded the view renders for a full week in under 2 seconds from the kept board; every number on a row
  matches the ledger columns written by D1; `tests/` has a frame-level test for the view's rows.
- **Size.** Medium to large. Needs B2; B4 and B6 fill columns as they land.

### C2. Bet card

- **Why.** One click should answer "why this side": today the decomposition dialog explains the mean, not the bet.
- **Where.** `ui/tabs/rankings.py` `_render_market_lines_tab` (line 2603), `_render_stat_deep_dive` (line 1505),
  `_render_distribution_chart` (line 304); `ui/charts.py`.
- **Do.** In the dialog's market tab, per priced stat: the outcome distribution from the prop engine's quantiles with the line
  drawn; raw and shrunk P(over); the last 10 games against this line (results and hit count); and three short "why" lines taken
  from the largest multipliers already in the decomposition (role, matchup, script, weather, teammate out).
- **Bar.** Opens from a Props row; no new model computation (reads the kept board and the engine); follows the quiet-text rules.
- **Size.** Medium. (This also delivers backlog P2's "why" column.)

### C3. Track Record as a betting scoreboard

- **Where.** `ui/tabs/track_record.py`; `data/prop_record.py` (B3); the bet log (B6).
- **Do.** Add a Props section above the points tables: cumulative units by week, ROI at price and closing-line value by tier,
  stat, side and book, a calibration chart (raw, shrunk, book), and the blind-side baselines. Keep the points sections as they
  are.
- **Bar.** Numbers equal the script's output for the same weeks. **Size.** Medium.

### C5. Status strip

- **Where.** `ui/freshness.py`; `ui/tabs/rankings_games.py` `market_freshness` (line 358); `data/prediction_ledger.py`
  `feature_hash`.
- **Do.** One dim line above the board: line age per book, injury pull age, whether the inactives window has passed since the
  last pull, the week the stats run through, the model's feature hash, and the stats currently switched off for betting.
- **Bar.** A stale input (lines older than 12 h inside 24 h of kickoff, injuries pulled before the inactives window) turns its
  item amber. **Size.** Small.

### C6. "Changed since last build"

- **Where.** `data/prediction_ledger.py` `latest_ledger_paths`; the diff from B10.
- **Do.** A collapsed table under the board: players whose points or a stat moved since the previous board for the week, with
  the reason. Alerts (a notification on a new A-tier line) only with the user's OK.
- **Bar.** Same rows as the B10 diff file. **Size.** Small after B10.

### C7. Phone export edges page

- **Where.** `ui/tabs/rankings_games.py` `export_edge_rows` (line 284); `ui/export_view.py` `edges_html` (line 889), which
  already renders RECORD and MOVE columns when rows carry `record_text` and `move`.
- **Do.** Feed it the tiered rows: tier order, the tier's record, the line move, the best price. Drop Track-tier rows.
- **Bar.** `tests/test_export_view.py` covers a row with a record and a move. **Size.** Small after B2.

---

## Phase 4. Model accuracy

The points residual has no headroom left in box-score history, so each lead here adds information the model does not read today.
Every one starts with an offline read on the deep-audit panel (`.sweeps/deep_audit/`, `scripts/deep_audit.py`); build a flag
only if the read passes.

### A1. Practice participation

- **Why.** The model reads the game status (Out, Doubtful, Questionable) and nothing about practice. The cached injury feed has
  `practice_status` (full, limited, did not participate) for every listed player, 2021-2026, so it is testable in the injury
  replay. The Questionable handling today is one factor for WRs.
- **Where.** `data/historical_availability.py` `load_injury_reports` (line 67), `historical_injury_profiles` (line 247);
  `data/weekly_projections.py` `QUESTIONABLE_WORKLOAD_FACTORS` (line 3183), `QUESTIONABLE_STAT_FACTORS` (line 3220),
  `_injury_profiles` (line 8657). The live board reads FantasyPros for status; practice status would come from the same
  nflverse feed for the live season (6-hour cache).
- **Do.**
  1. Offline: join status and practice status to the panel. For each position and cell (Questionable x full / limited / none;
     no game status x limited / none), measure the share who played, and for those who played the bias on points, snaps and the
     main stats, by season.
  2. If a cell is off in the same direction in 4 of 5 seasons, build `v2_practice_status_workload`: play probability and the
     workload factor keyed by the cell.
  3. Note the feed keeps the week's last filing only, so this is the final practice status, not the day-by-day pattern.
- **Bar.** Harness with the injury replay: START-ALL <= -0.003, no START position worse than +0.002, EVERY-ALL not worse. Stat
  columns: played-row stat MSE for flagged players better in 4 of 5 seasons.
- **Size.** Medium.

### A2. Defensive personnel, and A3. Offensive line state (one shared panel)

- **Why.** The matchup multiplier comes from what a defense has allowed. It does not know a starting corner, safety or edge
  rusher is out this week, or was traded. The same holds for a team's own line: PFF blocking and pass-rush files are loaded
  (`data/loaders.py` lines 992-1014) and used only in the Matchup Analyzer and player snapshot. The injury feed lists defenders
  and linemen (LB, CB, DT, T, DE, S, G, C are its most common positions).
- **Where.** `data/historical_availability.py`; snap counts in `data/cache/snap_counts_<year>.parquet` (defensive and offensive
  snaps); `data/weekly_projections.py` `blend_defense_prior` (line 4708), `MATCHUP_CLIP` (line 632); PFF season grades in
  `pff_imports/<year>/`.
- **Do.**
  1. `scripts/audit_personnel.py`: per team-week, the share of season snaps missing (Out, Doubtful, reserve lists) by unit:
     secondary, linebackers, defensive front, offensive line. Point in time: snaps from earlier weeks only.
  2. A2 read: residual of the opposing offense's stats (QB passing yards and TDs, WR/TE receiving yards, RB rushing yards)
     against the missing share of the relevant unit, held out by season.
  3. A3 read: residual of the team's own QB sacks, passing yards and RB yards per carry against the missing line share.
  4. Check whether the 245-feature residual scan already held these (it had `sacks_suffered` history; it did not have
     personnel availability).
  5. A unit that passes gets one flag (`v2_defense_personnel_adjust`, `v2_oline_availability`): a multiplier on the matching
     stats, sized by the held-out slope, clipped.
- **Bar to build:** a held-out slope with the same sign in 4 of 5 seasons and at least 0.5% of that stat's MSE. **Bar to ship:**
  harness START-ALL <= -0.002 with no position worse than +0.002, or for a stat-only version the stat MSE bar in rule 9.
- **Size.** Medium for the panel; small per flag.

### A4. Rookie and new-role prior

- **Why.** A player with no prior of his own lands on the position's fallback (the 15th-percentile snap share and the position
  average), whatever his draft slot. Draft capital is used only in the preseason RB allocator
  (`data/rb_role_allocator.py` lines 654-860). Week-1 rookies with 5+ targets were projected 2.0 (actual 7.0); rookie backs with
  10+ carries 3.7 (actual 12.9). This extends backlog M20.
- **Where.** `data/weekly_projections.py` fallback prior (around lines 9012 and 10830), `FALLBACK_PRIOR_K_SCALE` (line 5253),
  `apply_ourlads_preseason_role_floor` (line 6156); `draft_number`, `is_rookie_flag`, `years_exp` are already on the rows.
- **Do.**
  1. Offline cohort table, 2021-2025: rookies by position and draft bucket (round 1, 2, 3, day 3 / undrafted), per-game snap
     share, targets, carries and points in weeks 1-2, 3-6 and 7+, against what the board projected.
  2. If the buckets separate, build `v2_rookie_draft_prior`: the no-prior player's prior share and rates come from his bucket's
     cohort, not the position-wide fallback. Non-TD stats first (the xTD missing-prior fix already covers TDs).
  3. Combine with M20's open item: a backtest-only switch that opens the depth-chart floor under the archived charts, so weeks
     1-2 can be scored.
- **Bar.** Weeks 3-17 harness: START-RB and START-WR <= -0.003 each, START-ALL not worse. Weeks 1-2 harness (with
  `v2_historical_ourlads` on): rookie rows' volume bias cut by half, START-ALL not worse.
- **Size.** Medium. Target: before the 2027 opener; the in-season part helps now.

### A5. QB rushing sub-model

- **Why.** Under half of the model's moves away from a QB's own rushing level come true (attempts 0.43, yards 0.51; raises
  0.33). It is the one stat where the boosted residual search found something held out (7.4% and 5.1%, every season). Points are
  fine (0.96), so this is for the stat columns and QB rushing props.
- **Where.** `STAT_RECAL_OWN_HISTORY` and `STAT_RECAL_GAME_SCRIPT` (`data/weekly_projections.py` lines 3276 and 3348);
  `scripts/eval_stat_recal_forms.py`, `scripts/eval_stat_recal_game_script.py`; play-by-play is cached and has scramble and
  kneel flags.
- **Do.** Split each QB's rushing history into designed runs, scrambles and kneel-downs from play-by-play. Add to the QB rushing
  recal inputs: designed-run rate, scramble rate per dropback, the spread and implied total (kneels), and the opponent's
  pressure rate. Fit and score with the existing recal scripts, held out by season.
- **Bar.** QB rushing attempts and yards MSE at least 2% better than the shipped recal in 4 of 5 seasons, 2026 not worse; QB
  points unchanged.
- **Size.** Medium.

### A6. Per-stat level check on the new stat columns

- **Why.** In weeks 3-8 the model's average P(over) on receptions and receiving yards was 0.44-0.46 while 0.48-0.49 went over.
  In week 9, the first week on the recal and capacity-relief columns, the overall average matched (0.462 against 0.462). One
  week does not settle it.
- **Do.** Add to the scorecard: per stat, the model's average P(over) against the share that went over, and the share of
  results above the model's median, for weeks 9 on. No change before week 12 has results.
- **Bar.** Recentre a stat's median (in `data/weekly_median.py` `MEDIAN_CURVES_RECAL`) only if weeks 9-12 show a gap of 3+
  points on 400+ lines with the same sign as weeks 3-8.
- **Size.** Small.

### A7. Distribution shape by player type

- **Why.** The prop engine reads the spread from the projection's level and the position only (`data/prop_engines.json`, table
  `M1`). A deep threat and a slot receiver with the same mean do not share a median or a tail. The earlier boom/bust read (M9)
  tested a player's own points variance, not his type.
- **Where.** `data/prop_probability.py` `prob_over` (line 138); `data/weekly_median.py`; `scripts/deep_audit.py --stage props2`.
- **Do.** On the panel, within each mean bin, split receivers by depth of target (season-to-date air yards per target), backs by
  receiving share, and all by TD share of points. Compare the ratio quantiles and the share above the median. If they differ,
  fit type-specific tables and score them forward-chained 2023-2026.
- **Bar.** CRPS at least 0.3% better in 3 of 4 folds for a stat, and the share above the median within 2 points of 50% in every
  type.
- **Size.** Small to test, medium to ship.

### A8. One coherent receiving line

- **Why.** Targets, receptions and yards are recalibrated separately, with a guard to stop receptions passing targets. Separate
  fits can leave an implied catch rate or yards per catch that the player has never had, and combos (B9) need the parts to agree.
- **Where.** `apply_stat_recal_own_history` (`data/weekly_projections.py` line 3419); `clamp_dependent_stats` (line 8749).
- **Do.** Test a chained form: recal targets; receptions = targets x a catch rate pulled toward the player's own; yards =
  receptions x yards per catch pulled the same way. Same for carries and rushing yards.
- **Bar.** No stat's held-out MSE worse by more than 0.2%, zero consistency violations, medians still centred. Low accuracy
  upside: do it only when B9 reaches combos.
- **Size.** Medium.

### A9. Teams out of the race, weeks 15-17

- **Why.** Untested. Teams with nothing to play for give young players more and veterans less before week 18. Week 18 itself
  stays out of scope.
- **Do.** Offline only: from the schedule results, mark a team as out when it can no longer reach the last playoff seed's
  current win total. Residuals in weeks 15-17 for veterans (4+ years) against first- and second-year players on those teams, by
  season.
- **Bar.** Build a flag only if the split has the same sign in 4 of 5 seasons and is at least 0.5 points or 5% of volume.
  Otherwise close it with the numbers.
- **Size.** Small.

### A10. Re-earn ablation on 2026

- **Why.** About 140 flags have been judged on the same four harness seasons, many passing by 0.002-0.006. The 2026 season is
  the only data none of them saw.
- **Where.** `scripts/backtest_component.py` (one `--variant "a=-x;b=-y"` argument, not repeated flags), `--ablate-group`;
  the ship dates are in the `MODEL_FEATURES` comments.
- **Do.** After week 17 (a first pass on weeks 3-10 is fine now): rebuild 2026 with each family of flags shipped since
  2026-09-01 removed, injury replay on, and score against the default. Report each family's 2026 delta beside its harness delta.
- **Bar.** Nothing ships or retires from this alone. A family that was marginal in the harness and negative in 2026 goes on a
  watch list for the user.
- **Size.** Medium; several queued harness runs. Caveat for the write-up: rebuilt boards lack the live injury pulls.

---

## Phase 5. Over time

### D3. End-of-season routine (after week 17)

**SCRIPTED 2026-11-11:** `scripts/end_of_season.py` (step 1 runs with `--run`; steps 2-5 are printed as the hand list).

1. `scripts/deep_audit.py --stage all`, then `--stage engines`, then `scripts/prop_scorecard.py --install-engines`.
2. A10 on the full season.
3. Refit the tiers' shrink and switches (B2, B8) on the full season, then freeze them for 2027 weeks 1-4.
4. Week-1 work before the 2027 opener: M20, M21, A4, and store a pre-kickoff week-1 board.
5. Re-run the watch items W3-W6 and W9 with the sixth season.

### D4. More history before retrying closed ideas

**2026-11-11:** `scripts/pff_weekly_coverage.py` counts and verifies the seasons (2 of 3: 2024, 2025). 2026 folder 4 holds week 2,
folder 10 holds week 9 (upload default fixed; save now checks the week). Folders fixed 11-11 with the user's OK; week 4
re-exported and verified the same day.

Routes and targets per route (N2), charting and tracking data (N5) and the PFF alignment tuning (C1, C1b) were closed for lack
of seasons, not lack of signal. Keep archiving the weekly PFF exports; retry each once it has three seasons of weekly data.

### D5. Role-state model (research, low confidence)

**STOPPED 2026-11-11.** Held out on 2024-25 the change point does not beat the model's share (paired -0.04 pts MAE, SE 0.06) or a
last-three mean (`scripts/eval_role_changepoint.py`, methodology entry of the same date).

A change-point model on snap and route share to replace the fixed trust speed and the role-change rules. The residual scan
suggests little is left to find here, and three share-recency flags have already failed. Scope a two-day offline study at most:
does a detected change point predict the next game's share better than the shipped blend, held out? Stop there if not.

---

## Order and dependencies

| step | leads | needs |
|---|---|---|
| 1 | D0, D2 | - |
| 2 | B3, then B2 and B8, then D1 | step 1 |
| 3 | B6, B10 (the user's OK for schedules), B4 | step 2 |
| 4 | C1, C5, then C2, C3, C6, C7 | B2; B4 and B6 fill columns |
| 5 | A1, A2/A3 panel, A6 (offline reads, can run beside steps 2-4) | - |
| 6 | A4, A5, A7, A9 (offline reads), then flags that pass | step 5 for shared panel code |
| 7 | B5, B9 | B3; the user's OK for B5 |
| 8 | A8, A10, D3-D5 | B9 for A8; season end for A10, D3 |

## Questions for the user before work starts

1. B5: is market-based bet selection (no projection touched) acceptable under the no-blend rule?
2. B6 / B10: may scheduled pulls and rebuilds run on this machine, and at which times?
3. B4: may the weekly pull spend Odds API quota on event props?
4. C1: should PROPS open as the default view of Weekly Rankings?
