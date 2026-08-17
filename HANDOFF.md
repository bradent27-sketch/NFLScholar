# NFL Scholar — Handoff Doc

Personal Streamlit NFL fantasy-football analytics app. Single user (Yahoo/ESPN/Sleeper
leagues), runs locally via `streamlit run app.py` (or the `.claude/launch.json` config
named `gridiron-hub`). Last major work pass: July 2026 "pro polish" pass (perf fixes +
UI elevation toward a pro sports-site look - stat tile grid, hero stat band, team
banners, and the react-aria CSS-selector migration in gotcha #15, which had silently
killed the previous pass's tab/input styling). All 9 tabs at the time verified working
end-to-end (AppTest per-tab sweep + live browser click-through, zero exceptions). Still 9
now — the August 2026 pass below added Draft HQ and deleted the VORP Draft Sheet it
superseded.

Second July 2026 pass (same week): Player Search's fantasy-points chart converted from a
bar strip to an SVG line chart and moved to the very top of the player profile; the WR
percentile-bar chart's value labels no longer crowd/overlap near the 100th percentile;
the skill radar grid's spoke labels/titles got more breathing room; the Career Totals
table is now centered in a narrower column instead of stretched full-bleed; Defensive
Yield's coverage-radar title is a real styled/centered component
(`ui.components.render_matchup_title`); VORP Draft Sheet moved to the last tab (and was
deleted outright in the August pass); Player
Compare now overlays both players on ONE radar/bar chart in their own team colors instead
of two separate side-by-side charts (`render_percentile_radar_grid_compare` /
`render_percentile_bars_figure_compare` in `ui/player_snapshot.py`); and the snap-count
data gap behind both the O-line "N/A" display and the Depth Chart's OL ordering was fixed
at the loader level (see gotcha #16 for a real bug hit while building this pass, and the
Depth Charts / Snap-count coverage entries in section 3 for the data fix itself).

Same-week follow-up: the OL snap-share fix above surfaced a second, deeper bug in how the
Depth Chart's displayed "season share" was computed - see the Depth Charts / Snap-count
coverage entries in section 3 (`snap_games_played`) for the real fix. The Coverage Matchup
Radar's metric boxes (`ui.tabs.defensive_yield._render_coverage_radar`) now carry league-
relative percentile context (`st.metric(..., delta=...)`) alongside every raw number - the
percentiles themselves come from `data.coverage_radar.build_radar_data`'s expanded
`matchup_summary` dict and the already-computed `def_vals` array (see that function's
docstring for how "Blended Exp. YPRR" specifically gets ranked - it's not a source column,
so its percentile is computed by re-running the same blend for the whole receiver pool
against that same opponent). The fantasy-points-by-week line chart (Player Search) is
shorter (viewBox height 220->140) and curves through its weekly points via a Catmull-Rom-
to-Bezier spline (`ui.components._smooth_svg_path`) instead of straight polyline segments.

**August 2026 pass — DRAFT HQ (the current major work).** A full draft product built to
compete with the paid tools the user actually uses (thefantasyfootballadvice.com and
bdge.co): live market data, a settings-aware volume projection engine, value-based
valuation, a mock draft simulator, and one merged draft room. ~6,600 new lines across
seven new `data/` modules and one new tab. This is now the largest single system in the
app and it does NOT reuse `data/transforms.py`'s scoring or the old VORP sheet - see
section 3.5 for why the parallel implementation is deliberate.

Everything in that pass has its own dedicated reference: **`docs/draft_hq_methodology.md`**
documents every function in the tab, every computed stat's derivation, and every external
source with endpoints and refresh intervals. That file is the detail; section 3.5 here is
the orientation. Read this section for "what exists and what will bite you", read the
methodology doc for "how is this number actually made".

This doc exists so a fresh session (human or AI) can orient without re-deriving the
project's hard-won lessons. **The gotchas in section 5 are all real bugs that happened
here — several more than once.** Read that section before changing anything.

**August 2026 follow-up pass — Matchup Analyzer / Weekly Rankings / Draft HQ refinement.**
Note: by this pass the tab list had already grown past the "9 tabs" described above in an
earlier undocumented pass (Game Slate added first, Matchup Analyzer added after Player
Compare, and Risers/Rookie Watch/Weekly Rankings merged into one "Weekly Fantasy" tab with
sub-tabs) - section 1's tab list and tree above are stale against the current
`config.TAB_LABELS` and should be re-verified rather than trusted verbatim. This pass didn't
attempt a full doc reconciliation; it added the entries below for what it actually touched.

- **Matchup Analyzer** (`ui/tabs/matchup_analyzer.py`, `data/matchup_signals.py`) - the
  player/defense columns now render in matched ROW PAIRS (tendency profile next to
  positional vulnerability, route efficiency next to coverage, usage&role next to run
  defense, game log next to weekly detail) instead of two independently-stacked halves,
  per explicit request that the two sides read as one comparison. The old "Efficiency
  Elasticity" curve is now labeled "Defensive Tendency Elasticity" (soft to the player's
  OWN position); a new "Efficiency Elasticity" curve buckets by
  `data.matchup_signals.team_defensive_prowess` instead - PFF's snap-weighted overall team
  defense grade (from `run_defense_summary` + `defense_coverage_summary`'s `grades_defense`,
  the SAME number in both exports since it's not role-specific), not fantasy points allowed
  - deliberately a different axis than the positional curve, since a defense can be
  excellent overall and still be one player's softest positional matchup. The defense's
  week-by-week detail collapsed from a position picker that stacked up to five charts into
  ONE chart with By Position / By Stat tabs (`_render_defense_weekly_detail`), and its
  reference line is now the LEAGUE average allowed
  (`data.matchup_signals.league_average_allowed`), not that team's own season average.
  Yards-per-target-allowed moved from the Coverage panel into Positional Vulnerability as a
  smaller "YPRR"-labeled sub-bar directly under each position's row (per explicit naming
  request - it's really YPT, the closest measured equivalent this app has; the sub-bar's
  hover text says so) - `ui.charts.render_percentile_bar_list` gained a `'sub': True` entry
  flag for this (shorter row, indented, muted).
- **Weekly Rankings** (`ui/tabs/rankings.py`) - the FantasyPros pull now shows a "current
  to Week N, pulled at TIME" caption instead of a bare number; the model's projected STAT
  LINE is shown alongside Model Proj Pts (the columns were already being computed by
  `build_weekly_projections`, just never displayed); recent-form window is fixed at L5
  games (no more adjustable control); Games This Season / Role Confidence dropped from the
  display; a new live-player-prop-derived "Market Proj Pts" column
  (`data.odds_weekly.weekly_market_projection`, reusing the same free PrizePicks/Underdog/
  DraftKings weekly board the Live Odds tab pulls and the same market-scoring arithmetic
  Draft HQ uses for season-long book lines) sits beside the model and FantasyPros numbers,
  never blended into either; a new "Rank" column (e.g. "RB4") sits right after Opponent,
  shaded by tier (`data.draft_board.tier_by_position`, a generalized sibling of the draft
  board's own `assign_tiers`/`_kmeans_1d` clustering, and `ui.styling.style_plain_dataframe`
  gained a `tier_cols` param to shade an arbitrary column by an externally-computed tier
  rather than only a literal `'Tier'` column). Live Odds tab is untouched.
- **Draft HQ** (`ui/tabs/draft_hq.py`, `data/odds_market.py`) - League Settings dropped the
  Projection Uncertainty slider and Projection Baseline Through selector (values now fixed
  at their old defaults, no UI); the whole settings panel is regrouped into League Settings
  / League Scoring / Draft & Market Settings / Data imports headings (same widgets, just
  grouped - no behavior change); the Risk column is gone from the player table (the
  underlying `Risk` computation in `data.draft_board.add_outcome_range_from_projections` is
  untouched, just not displayed); "Market lines vs this board" gained a second table, a
  matchup-adjusted full-season scoring estimate
  (`data.odds_market.estimate_full_season_scoring`) that fills in every game WITHOUT a
  posted Vegas line using that team's own measured scoring level against its actual
  scheduled opponent's measured defense (games-count-shrunk baselines toward league
  average, clipped 0.75-1.3x - same matchup-multiplier shape as everywhere else in this
  app) - the existing posted-only table can be a badly biased read of an offense early in a
  season when only a handful of (non-representative) games have a line; still
  information-only, nothing feeds a projection, same as the table it sits beside. "Pick
  odds & positional scarcity", "Market lines vs this board", "Run many mocks / compare
  slots" (and live-sync in Live draft mode) moved from a full-width stack below the draft
  room into the right column under Your Roster, to make the page more compact.

---

## 1. Architecture

```
app.py              Thin entrypoint: page config, theme, sidebar, tab wiring ONLY.
config.py           THEME, TEAM_CONFIG (32 teams: id/name/color), tab-label constants,
                    AVAILABLE_SEASONS[_WITH_UPCOMING], position lists, PFF team-code maps,
                    Odds API market list.
data/
  utils.py          Pure helpers: percentiles, name cleaning (clean_name_exact /
                    clean_name_for_merge), abbreviated-name bridging
                    (build_last_name_index / match_abbreviated_name),
                    american_odds_to_prob. No Streamlit imports.
  loaders.py        All raw ingestion: local CSVs, nflreadpy pulls, PFF per-year folders,
                    Odds API, Sleeper API. Nearly everything @st.cache_data.
  transforms.py     Everything computed on top of loaders: fantasy scoring, percentiles,
                    risers/rookie/VORP boards, depth-chart synthesis, projection model,
                    red-zone usage, recent-form ranking.
  rankings.py       FantasyPros/custom rankings ingestion + comparison logic.
  coverage_radar.py Man/zone coverage correlator (radar chart) helpers.

  --- Draft HQ engine (August 2026, ~4,700 lines) --------------------------
  draft_sources.py  Live market data: FantasyPros ECR (per draft format), the
                    DynastyProcess ID crosswalk (+ birthdates), dynasty values,
                    ADP with a source-preference chain, injuries, ESPN news.
                    Also the scipy-free normal survival function every
                    availability calc uses.
  draft_projections.py  Volume-based projections. Positional usage curves,
                    per-player recency-weighted rates, the stickiness blend,
                    the measured aging curve, and gamma-distribution pricing
                    of per-game yardage bonuses.
  draft_board.py    Valuation. Parameterized scoring (score_stats), starter
                    demand, replacement level, VORP/VOLS/VONA, tiers,
                    availability, market blend, outcome ranges. The lowest
                    module in the draft stack - draft_projections imports FROM
                    it, never the reverse.
  draft_intel.py    Strategy archetype inference, pick-frequency simulation,
                    roster percentile, and the positional value-add model
                    ("+X% to your team").
  draft_sim.py      Mock draft simulator: bot logic, roster-construction
                    legality rules, autopick, lineup optimizer, draft grading.
  draft_sos.py      Positional strength of schedule over a selectable week
                    range, from fantasy points allowed by defense.
  ffa_import.py     Reads a Fantasy Football Advice player export the USER
                    supplies. No network calls, by design (section 8).
ui/
  styling.py        Theme CSS injection + every table Styler (percentile heatmap,
                    matchup colors, depth chart cells, sticky game log HTML).
  components.py     Shared widgets: switch_tab, player card, data-health sidebar,
                    position filter, drafted-players helpers, build_player_search_labels.
  player_snapshot.py Shared per-player positional stat snapshot (build_player_snapshot)
                    + the percentile-bar chart renderer - lives in ui/, not
                    data/transforms.py, because it needs ui.styling.get_pff_color and
                    produces display-ready output. Used by both Player Search and
                    Player Compare so the two never drift apart.
  tabs/             One module per tab, each exposing render().
tools/
  har_extract.py    Standalone (no project imports - the user runs it on their
                    own machine). Unpacks a browser HAR into api/ js/ css/
                    html/ other/ + an index.json manifest, stripping every
                    credential on the way out. How the FFA payload got here.
docs/
  draft_hq_methodology.md   Full derivation reference for Draft HQ.
```

**Tabs** (order = `config.TAB_LABELS`): Player Search, NFL Depth Charts,
Defensive Yield Schemes, Risers/Waiver Wire, Rookie Watch, Weekly Rankings, Live Odds,
Player Compare, **Draft HQ**.

Nine tabs, not ten. **VORP Draft Sheet was deleted** (August 2026) once Draft HQ made it
redundant - it computed replacement-level VORP off last season's per-game pace × 17,
which is a strict subset of what Draft HQ does. Deleted with it: `build_vorp_draft_sheet`,
`build_efficiency_volume_data` + `render_efficiency_volume_quadrants` (the EPA
efficiency-vs-volume quadrant chart, which lived only on that tab), `load_fantasypros_rankings`,
`_merge_ryoe` and `load_ngs_rushing`. Recover any of it from git history if wanted.

Draft HQ sits LAST despite being the most capable tab: it answers a question that's only
live for a week or two a year, and every other tab is used all season. `drafted_players`
in session state - which Player Search and Rookie Watch filter on via
`ui.components.get_drafted_players_clean_keys` - is now maintained by Draft HQ (it used
to be written by the VORP tab's Live Draft Tracker too).

**Tab wiring** (app.py): `st.tabs(TAB_LABELS, key="active_tab", on_change="rerun")` —
Streamlit ≥1.59 syntax. Each tab's body only runs when `tabN.open` is True (lazy
execution — this is the app's main performance mechanism; don't undo it). The active tab
lives in `st.session_state["active_tab"]`, which is what `ui.components.switch_tab()`
writes to for cross-tab navigation. Every render() call goes through
`app._render_guarded` — an exception in one tab degrades to an in-tab error message with
a collapsible traceback, never a full-page crash (verified via
streamlit.testing.v1.AppTest with a deliberately raising fake tab).

## 2. Data sources

### Local CSVs (project root)
- `stats_player_week_{2019..2025}.csv` — weekly player stats (nflverse schema).
  `stats_player_reg_*.csv` are SEASON TOTALS — never load them where weekly granularity
  is expected (load_year_data guards on the presence of a `week` column for this reason).
  **Every one of these files mixes REG and POST (playoff) rows** under one `season_type`
  column, with POST weeks numbered 19-21+ instead of continuing 18+ - confirmed on all
  seven years. `load_year_data` filters to `season_type == 'REG'` right after loading (any
  source: local file, fallback scan, or the nflreadpy fallback) - do not remove this filter
  or add a new stats source that bypasses it, or playoff games silently blend into every
  "season" figure in the app again for any team that made a run that year (confirmed real:
  2019 Mahomes showed weeks up to 21 and an inflated/distorted season average before this
  filter existed).
- `roster_weekly_{2002..2026}.csv` — weekly rosters (bio, team, position, draft capital).
- `snap_counts_2025.csv.csv` (double extension is real) — wide format: one row per player,
  `Wk N` / `Wk N pct` columns. Years WITHOUT a local file fall back to
  `nflreadpy.load_snap_counts()` (covers 2012–present), which loaders.py pivots into the
  same wide shape so downstream code has one schema.
- `qbr_season_level.csv` — ESPN QBR, one row per QB **per season per season_type**
  (2006–2025, Regular + Playoffs). Never take `.iloc[0]` off a bare name match here.
- `ftn_charting_*.csv`, `teams_colors_logos.csv`, `otc_players*.csv` — present, unused.

### PFF exports — `pff_imports/{year}/` (2019–2025)
One subfolder per season, ~16 files each. **Filenames are inconsistent across years**
(browser dedup suffixes like `receiving_summary (5).csv`, year suffixes like
`_2025`, or neither) — `data.loaders.load_pff_year(base_filename, year)` matches by
prefix glob, so never hardcode a full PFF filename. `load_all_pff_data(year)` returns
the dict of all frames + precomputed percentile columns; `year` is its cache key.
The `receiving_concept` duplicate pairs within a folder are byte-identical (verified) —
only one is loaded.

To add a new season: create `pff_imports/{year}/`, drop the same ~16 exports in, add the
year to `config.AVAILABLE_SEASONS*`. That's it.

### nflreadpy (live pulls, all cached)
nflreadpy is set to FILESYSTEM cache mode at import time in data/loaders.py
(24h expiry, OS user-cache dir) — restarts reuse yesterday's downloads and the app
tolerates being offline. NOTE: `update_config` lives in `nflreadpy.config`, NOT the
top-level package — a top-level call raises AttributeError, and the defensive
try/except around the config call would swallow exactly that mistake silently, so
re-verify `get_config().cache_mode` actually reports FILESYSTEM if touching this.
- `load_player_stats`, `load_rosters`, `load_schedules` — fallbacks when no local CSV.
- `load_snap_counts` (2012+), `load_nextgen_stats` (RYOE), `load_pfr_advstats`
  (pressure data), `load_team_stats` (pace, pass attempts faced).
- `load_pbp` — **raw nflfastR play-by-play** (~50k rows/season, 372 cols; EPA/CPOE/WP/
  yardline_100/goal_to_go). Currently feeds `build_redzone_usage` only. Player IDs in
  pbp are gsis_id (`00-00XXXXX`), names are abbreviated "F.Last" — always join by ID.
- `load_players` (via `load_player_id_crosswalk`) — master ID table. **Verified: PFF's
  `player_id` column == this table's `pff_id`, 100% match on a full export.** Loaded and
  cached but NOT yet wired into the PFF lookups (still name-based) — that refactor is
  known-possible and known-unstarted.
- Available but unused: `load_injuries` (2019+), `load_ff_opportunity`, `load_contracts`.

### External / scraped (2025-only, no historical archive — confirmed)
- `sharp_coverage_schemes_2025.csv` — team man/zone/MOF rates. MOF Closed/Open % has NO
  substitute source for other years (PFF doesn't carry it); man/zone for other years is
  derived from PFF's defense_coverage_scheme instead
  (`coverage_radar.build_team_man_zone_rates`).
- `external_data/sharp_positional_coverage_2025.csv` — YPT allowed by receiver type /
  alignment. Feeds the radar's defense side + the projection model's alignment
  multiplier. 2025-only.
- `external_data/sumersports_*_2025.csv` — personnel/formation tendency context panels.

### Rankings — `rankings/`
Three FantasyPros 2026 DRAFT rankings CSVs (ppr / half_ppr / standard). Schema:
`RK, TIERS, PLAYER NAME, TEAM, POS, BYE WEEK, ...`. User refreshes these by overwriting
the same filenames. Weekly rankings are uploaded per-session via the Weekly Rankings tab
(same schema, `parse_fantasypros_upload`), not stored.

### APIs
- **The Odds API** — game lines (bulk) + player props (per-event, costs more credits).
  Key persisted PLAINTEXT at `.streamlit/odds_api_key.txt` — the Live Odds tab warns the
  user to delete it before sharing the folder; **do not remove that warning**. Both
  fetches `@st.cache_data(ttl=900)`; props fetches are button-gated on purpose (quota).
- **Sleeper API** — public, keyless; draft-pick sync for the Live Draft Tracker.

### Draft HQ market data (`data/draft_sources.py`, all `@st.cache_data(ttl=6h)`)
- **DynastyProcess GitHub CSVs** — `raw.githubusercontent.com/dynastyprocess/data/
  master/files/{f}`. Two hosts are tried in order and **only raw.githubusercontent.com
  is reachable from a sandboxed/proxied environment** — keep it first.
  - `db_fpecr_latest.csv` → FantasyPros ECR with `ecr`/`sd`/`best`/`worst`. Fetched
    **per draft format** via the `page_type` column (`ECR_BOARDS`: Redraft 1QB, Redraft
    Superflex/2QB, Best Ball, Dynasty 1QB, Dynasty Superflex). A superflex board is NOT
    the 1QB board with QBs shifted up — the whole positional value structure changes.
    `ecr_age_days()` surfaces staleness in the UI.
  - `db_playerids.csv` → the cross-platform ID crosswalk AND **birthdates**, which is
    where the aging curve's ages come from (92% board coverage).
  - `values-players.csv` → dynasty trade values.
- **ADP has a preference chain** (`fetch_adp`): uploaded CSV → FFA import → FantasyPros
  consensus ADP (live) → FantasyPros export (local) → ECR-as-estimate.
  - **Fantasy Football Calculator is NOT in the automatic chain** — it's selectable by
    hand and nothing else. Its ADP comes from free mock drafts on its own site and drifts
    hard from real drafts (TEs and QBs slide a full round or more). The user tested it and
    called it "way off"; that was correct, and everything downstream (`Value vs ADP`,
    `Avail Next %`, `VONA`, the opponent model) inherits the error while still looking
    authoritative.
  - **The local FantasyPros path is the draft-night floor.** The ranking CSVs already in
    `rankings/` carry an `ECR VS. ADP` column, which is exactly (ADP − consensus rank), so
    ADP is recoverable offline as `RK + that difference`. Draft night is the worst
    possible moment to find a site unreachable.
  - **Sleeper ADP was removed entirely** — its `search_rank` is a popularity ordering, not
    ADP, and it put a QB at overall #2.
  - When nothing answers, ECR stands in so availability still works, and
    `ADP_SOURCE_NOTES` tells the user what the active source actually measures.
- **nflverse injuries** — short TTL (a Saturday IR move should invalidate a board within
  the hour). Falls back a season at a time: asking for the season you're DRAFTING raises
  "Season must be between 2009 and 2025" every single time during draft season, because
  nflverse has no rows for a season that hasn't kicked off. That's the normal case, not
  an edge case.
- **ESPN site API** — news headlines keyed to player names. Several host candidates tried
  in order (public JSON endpoints move around and some networks 403 `site.api`
  specifically). Every consumer treats an empty return as "no news column", never an
  error.

### Fantasy Football Advice import (`data/ffa_import.py`) — USER-SUPPLIED, no network
The user has an FFA subscription. Their payload carries things this app can't derive:
`ffaValue`, `adpComposite`, Elo ratings, analyst stat lines, and a hand-written scouting
note per player. **Nothing in this repo talks to their servers, polls their API, or
automates a login** — `save_ffa_import` reads a file the user hands over, once.

- Lives at `external_data/ffa_players.json`, **gitignored** (this repo is public — see
  section 8). Dropping the file there by hand works exactly as well as uploading it.
- **The STAT LINE is imported, not the point total.** Their `proj` is half-PPR, so
  reading it straight would be silently wrong in full-PPR/TE-premium and badly wrong once
  yardage bonuses are on. `merge_ffa_into_board` blends their stat line against this
  app's per stat, then re-scores through `score_stats` — a blended projection still has a
  carry count that matches its rushing yards.
- **`diagnose_ffa_payload` exists because of a real support incident.** The user uploaded
  `index.json` (the HAR manifest) instead of `api/players.json`; the importer correctly
  rejected it and said nothing useful, so it looked like the upload feature was broken.
  It now names the specific file you actually want.
- `tools/har_extract.py` is how the payload got here: the user captured a HAR in their
  browser, ran the script locally, and sent the unpacked `api/` folder. The script strips
  all headers/cookies, redacts credential query params, field-redacts POST bodies, and
  skips auth endpoints whole. **`*.har` is gitignored** — a raw HAR contains live session
  cookies.

## 3. Key computed systems

- **Scoring** — `apply_scoring_and_percentiles` (transforms.py). One formula, PPR
  multiplier parameterized, includes IDP defaults AND kicker defaults (3/4/5 pts for FG
  makes under 40 / 40-49 / 50+ yards via the `fg_made_X_Y` distance-bucket columns, 1 pt
  per PAT) - nflverse's own `fantasy_points`/`fantasy_points_ppr` columns are offense-only
  and never include kicking, so kickers scored a flat 0 every week app-wide before this,
  identical in kind to the IDP gap noted below. `score_projected_stats` is a second
  dict-of-scalars twin used by the projection model — **the two are kept in sync
  manually**; change one, check the other. (A third copy lived in
  `build_vorp_draft_sheet` until that tab was deleted.)
- **Projection model** (Player Search → "Next Game Projection") —
  `build_player_projection`: per-raw-stat blend of last-4-games form (60%) + season
  average (40%), × opponent-allowed multiplier (`build_stat_allowed_matrix`, clipped
  0.75–1.3), × opponent pace multiplier (`load_team_pace`, clipped 0.85–1.15), × WR/TE
  slot/outside alignment multiplier (`build_alignment_multiplier`). When sportsbook prop
  lines are loaded, the final number is 70% market / 30% model per stat
  (`build_market_projection`) — deliberate, user asked for market-dominant weighting.
  Bails out to `({}, 0.0)` the moment `'week' not in p_data.columns` (a season with no
  weekly stats file at all yet, e.g. 2026 pre-kickoff) - checked BEFORE the per-week
  filter, not after: `load_year_data`'s roster-only fallback still leaves exactly one
  placeholder row per player with every counting stat at 0, which used to survive every
  filter below (nothing to filter ON with no 'week' column) and produce a real-looking
  "0.0 projected fantasy points" instead of the "not enough data" message this exact
  situation shows everywhere else - confirmed live before this fix.
- **Red-zone usage** — `build_redzone_usage` (from pbp, yardline_100 ≤ 20): per-player
  RZ target share / carry share / TDs, joined by gsis_id then mapped back to the
  caller's name column. `_merge_redzone_share` attaches it to boards
  (WR/TE → Tgt share, RB → Rush share; replaced RYOE on Rookie Watch).
- **VORP** — now lives entirely in Draft HQ (`data/draft_board.py`), off real projections
  rather than last season's pace. The old `build_vorp_draft_sheet` was deleted with its
  tab.
- **Vegas game lines** — `data/odds_market.py`. **Free, uncapped, no API key**, from
  nflverse's games CSV on raw.githubusercontent.com (the nflverse-data *release assets*
  live on github.com proper, which some networks block — raw is the fallback that works,
  and it's the same data). Gives spread + total per game → implied points per team. In
  early August 2026 that was 52 of 272 games posted (weeks 1–4), ~3 per team, so
  `games_posted` travels with every row. Surfaced as the **Vegas PPG** column.
  **Deliberately NOT multiplied into any projection** — backtested on 748 player-seasons
  and it made projections worse at every strength (§5b of the methodology doc). The sign
  convention is the one thing that can silently invert everything: `spread_line` is
  positive when the HOME team is favored.
- **Sportsbook lines** — `data/odds_sources.py` (adapters) + `data/odds_projections.py`
  (scoring and comparison). Underdog's SEASON-LONG player over/unders, re-scored under the
  user's league settings into a market projection built with no reference to this app's
  model, then compared to it (Draft HQ → "Market lines vs this board"). Which source
  answers which question is the whole design: The Odds API is per-EVENT and is right for
  in-season/per-game work; Underdog posts season-long lines and is right for a draft board.
  **No bot-detection evasion anywhere** — no undetected_chromedriver, no stealth browser,
  no proxy rotation. PrizePicks is attempted with a plain request and a block is accepted
  as a "no"; FanDuel is not scraped at all because The Odds API carries it under licence.
  Adds no dependencies (requests/pandas/streamlit only). Parsing is tested offline against
  recorded-shape fixtures — `python tests/test_odds_sources.py`, 14 tests, no network;
  `python scripts/check_odds_sources.py` checks the LIVE shape on a normal network.
  Gotcha worth keeping: the market→board join is **two-tier** (`attach_board_player`) for
  the same reason `load_year_data` is — books write "Patrick Mahomes", the board says
  "Patrick Mahomes II", and the loose fallback is refused when the stripped key is
  ambiguous.
- **Recent form** — `build_recent_form_rank` (last N games avg FPTS, min 2 games) — the
  Weekly Rankings tab's internal baseline. Draft value and weekly form are deliberately
  separate comparisons on separate tabs.
- **Depth charts** — `fetch_intelligent_depth_chart`: median weekly snap % ranking
  (median, not mean — see docstring for the Josh Sweat case), draft-capital boost for
  rookies that fades as real snaps accrue. Dedups players by `exact_name` (the stable,
  full-name-derived key), NOT the abbreviated `player_name` column it displays - that
  abbreviation can itself change mid-season for the SAME real player (confirmed real:
  Michael Wilson, ARI 2025, is "Mi.Wilson" weeks 1-11 and "M.Wilson" weeks 12-18), which
  used to split him into two "different" depth-chart entries. Its `depth_score` is only
  as good as `snap_score` underneath it - see the snap-count supplement below for the
  real gap that used to leave O-line ordering running on draft-capital/experience alone.
- **Snap-count coverage** — `data.loaders._pivot_nflreadpy_snap_counts(year)` +
  `load_year_data`'s snap-loading block (July 2026 pass): `snap_counts_2025.csv.csv` (the
  only local snap file that exists - confirmed no other `snap_counts_{year}.csv.csv` is on
  disk) has ZERO offensive-line rows at all (verified: none of its 1,828 rows carry an OL
  `positionGroup`), so every O-lineman showed "N/A" snap % and the Depth Chart's OL
  ordering fell back to draft-capital/experience with no real playing-time signal - this is
  what looked like a wrong starter (e.g. a higher-drafted backup ranked above the real
  starter) when it was really "no snap data to rank on at all." Fixed by topping up
  whichever local file loaded with `nflreadpy.load_snap_counts(year)` rows for any player
  NOT already present locally (matched via `clean_name_for_merge`) - never overrides real
  local data, only fills genuine gaps. Separately, the nflreadpy fallback used ONLY
  `offense_pct` (correct for offense/O-line, but a hardcoded 0% for every DEFENSIVE player
  in any season relying on it - i.e. every year except 2025, since that's the only year
  with a local file) - it now reads `max(offense_pct, defense_pct)` since this export
  carries both columns per row (one real, one 0, depending which side of the ball that
  player was on that week). Verified end to end on real data: 2025 OL snap-match rate went
  from ~0% to 81.5%, 2024 (nflreadpy-only year) defense match rate is 93.2%, OL 82.0%.

  That fix alone wasn't enough, though - a follow-up same-week pass fixed a SECOND, deeper
  bug it exposed: `ui.tabs.depth_charts._build_snap_totals`'s displayed "season share" used
  to sum the per-row `weekly_snap_pct` column and divide by team weeks, but that column only
  has a real value on weeks `stats_player_week_{year}.csv` itself has a row for that player -
  and nflverse's weekly box score is STAT-triggered, not full-roster: confirmed real, Kingsley
  Suamataia (KC OT/G, 2025) has exactly 5 rows in that file all season (weeks 5/11/12/16/18)
  despite playing the large majority of KC's offensive snaps nearly every week. Every OL
  player was getting summed over only a handful of real rows out of ~17 team weeks, capping
  everyone's displayed share at roughly 20-35% regardless of real usage - invisible before the
  snap-source fix above (nearly all OL showed "N/A" instead), but a real, actively misleading
  number once "N/A" started resolving to real matches. Fixed by computing season share as
  `snap_pct_avg * snap_games_played / team_weeks` instead - both of those come straight from
  the snap source's own per-week columns (a new `data.loaders.load_year_data` output,
  `snap_games_played` - count of real weeks the snap source itself has for that player),
  completely bypassing the stats-row gate. Verified: Suamataia's displayed share went from
  ~26% to ~93%, Caliendo (backup guard, same team) to ~33% - both now plausible.
- **PFF grade fallback** — `data.loaders.load_pff_data_with_fallback(year)`: wraps
  `load_all_pff_data`, falling back to the most recent prior season with real data when
  the selected season's `pff_grades_map` is empty (e.g. 2026 pre-kickoff). Used by both
  Depth Charts and Player Search (both need "show last year's grade until this year's is
  uploaded"), returns `(pff_dict, source_year)` - callers show a caption when
  `source_year != year`. Swaps the WHOLE `pff` dict, not just the grades map, so slot-rate
  and other secondary PFF fields stay from the same season as the grades shown.
  Self-clears the moment real files land in `pff_imports/{year}/` — no manual flag to flip.
- **Rankings comparison** — `build_rankings_comparison(value_df, value_col, rank_label,
  ...)`: generalized; deltas re-ranked within the matched subset (pool-size artifact fix
  documented in its docstring). Unranked players get a max+1 sentinel, never NaN.
- **EPA efficiency** — `build_epa_efficiency` (transforms.py, from `load_pbp`, REG-only):
  per-QB `epa_per_dropback`/`success_rate_pass`/`cpoe_avg` (filtered on `qb_dropback == 1`,
  NOT `play_type == 'pass'` - that alone silently drops scrambles, ~4.7% of dropbacks,
  understating mobile-QB value; grouped via `passer_player_id.fillna(rusher_player_id)` so
  a scramble still attributes to the right QB), per-RB `epa_per_rush`/`success_rate_rush`,
  per-WR/TE `epa_per_target`/`success_rate_rec`. Volume floors
  (`MIN_DROPBACKS_FOR_EPA`=100, `MIN_CARRIES_FOR_EPA`=20, `MIN_TARGETS_FOR_EPA`=15) keep
  small samples from showing noise as signal. Merged into `precompute_league_percentiles`
  right before its percentile loop, so `epa_per_dropback_pct` etc. fall out for free.
- **Target location profile** — `build_target_location_profile` (transforms.py): buckets
  `air_yards` into Short(<10)/Intermediate(10-19)/Deep(20+) crossed with `pass_location`
  (left/middle/right) into a 3×3 share-of-targets grid per receiver. This is the honest,
  data-available stand-in for a route tree - **never call it a route tree** in code or UI
  copy, since a real one needs route-geometry data this app doesn't have locally
  (`ftn_charting_*.csv` only has play-level boolean flags like `is_play_action`, not route
  shapes - confirmed by direct inspection before building this instead).
- **Percentile-bar chart** — `ui.player_snapshot.render_percentile_bars_figure`: Baseball
  Savant-style horizontal bars, one per stat, length = percentile, color =
  `get_pff_color(pct)`. **`get_pff_color` returns a CSS `rgba(r, g, b, a)` string
  (0-255 ints) — matplotlib's color parser does not understand that syntax and raises
  `ValueError: Invalid RGBA argument` on it.** Every other `get_pff_color` call site feeds
  an HTML/CSS context (Styler `bg`, a card's inline style) where this is correct; this
  chart is the one consumer that hands the string to matplotlib directly, so
  `player_snapshot._mpl_color()` converts `rgba(...)` to a 0-1 float tuple first (hex
  fallback colors like `#313131` pass through unchanged). If a future chart colors bars/
  points via `get_pff_color`/`get_matchup_color`, it needs this same conversion — a plain
  `python -m py_compile` and even calling the data-producing function directly (it just
  returns strings; no matplotlib involved) both look completely fine. Only actually
  calling the chart-rendering function with a real player's data, or loading it live in
  the browser, raises the error — this bug fires for essentially any player with at
  least one non-fallback-gray stat, which in practice is almost everyone.
- **Player snapshot builder** — `ui.player_snapshot.build_player_snapshot`: the shared
  per-position stat lookup (QB / RB / [WR,TE] / OLINE_POSITIONS / DEFENSIVE_POSITIONS / K
  / [P,LS] branches) behind BOTH Player Search's matrix table and Player Compare's bio
  halves - returns `list[{'label','value_str','pct','color'}]` instead of two parallel
  dicts so the raw percentile survives for the bar chart (previously discarded once the
  color string was computed). `pct=None` means "no real percentile, skip from the bar
  chart" - it is NOT the same as `pct=0`, which is a real (if degenerate) percentile the
  original code always computed a color for. Player Compare's "Edge" column
  (`_render_comparison_table` in player_compare.py) has to check the DISPLAYED value
  string (`'--'`/`'N/A'`) rather than `pct is not None` for this exact reason - two blank
  cells both carrying `pct=0` would otherwise net a misleading `Edge = 0.0` ("dead even")
  instead of "not comparable" (`NaN`).

### 3.5 Draft HQ engine (August 2026)

Full derivations live in `docs/draft_hq_methodology.md`. This is the orientation: what
each piece is, and what will bite you.

**Why it doesn't reuse the existing scoring.** `apply_scoring_and_percentiles` implements
this app's three fixed presets and hardcodes the rest. A draft board has to honor whatever
a real league uses — 6-point passing TDs, .5/carry, TE premium, negative fumbles, per-game
yardage bonuses — so `draft_board.score_stats` is a fully parameterized fourth
implementation. It is NOT a fourth thing to keep in sync with the other three: it's
deliberately independent, takes a scoring dict, and is the only one the draft stack uses.

- **Projections are volume-based, not rank→points.** `build_projected_board` projects a
  full stat LINE (carries/targets/yards/TDs) and then scores it under your league's
  settings. Getting a point total from a rank curve directly would be wrong the moment you
  toggle PPR.
  - `build_volume_curves` — `curves[pos][stat][r]` = season total for the player who
    FINISHED rank r, averaged over 5 seasons. Ranked on PPR **regardless of your
    settings**: the curve describes usage at each rung of a pecking order and shouldn't
    reshuffle every time a setting changes.
  - `build_player_rates` — recency-weighted per-game rates, `[1.0, 0.55, 0.30, 0.15,
    0.08]` by season. Vectorized; the per-player loop version dominated the whole build.
  - `project_stat_lines` — blends `w·(own rate × 17 × age_factor) + (1−w)·curve_total`,
    where `w = STAT_SELF_WEIGHT[stat] × evidence`. **Per-stat stickiness is the core
    idea**: usage persists (0.70), efficiency regresses (0.55), TDs regress hardest
    (0.30). A player with no history lands entirely on the curve, which is right for a
    rookie. Role-change damping slides `w` toward zero when a player's own usage is far
    below what his consensus rank implies (a backup QB who just won a job).
- **The games basis is the subtlest thing in the module — read the note above
  `PACE_GAMES` before touching it.** Own-history rates project across a full 17-game
  season; the curve's totals are left exactly as measured. The curve's `games` entry is a
  SELECTION ARTIFACT, not durability (missing games is one of the main ways a player ends
  up ranked 28th), and using it as a per-player games figure left every QB ~30 points
  under consensus. Two alternatives were built and measured and both were worse — the
  comparison table is in the methodology doc. Do not "fix" this by rescaling the curve to
  17; that was variant B and it dropped ECR rank-corr from 0.937 to 0.905.
- **Aging curve** (`AGE_PEAK` / `AGE_DECLINE_PER_YEAR`) — measured, not assumed, off 1,644
  contributor seasons matched to birthdates. RB −6.5%/yr past 25, WR −3.6% past 26, TE
  −2.0% and QB −1.0% past 27. Applied to the **own-history side only** — the rank curve is
  indexed by a consensus that already priced age, so adjusting both halves charges a
  33-year-old twice. The symmetric young-player boost was tested and does literally
  nothing (rank agreement identical to three decimals); don't re-add it without new
  evidence. **Age matching needs a nickname tier**: the ID crosswalk carries LEGAL names
  and the consensus feed carries JERSEY names (Kenneth/Kenny Gainwell, Andres/Andy,
  Chigoziem/Chig), and no age means no aging markdown at all. Keyed on last name + first
  three letters + position — a first-INITIAL version recovered four real nicknames and
  four strangers, which is not a trade worth making for a silent multiplier.
- **Yardage bonuses** (100/150/200/250 rush+rec, 300/400/500/600 pass, cumulative or
  highest-only) need per-GAME yardage from a season total, so each player's weekly yardage
  is modelled as a **gamma distribution** with a position-typical CV (rushing 0.62,
  receiving 0.78, passing 0.35) and the thresholds integrated over it. `_gammq` implements
  the regularized upper incomplete gamma directly — **no scipy in this project.**
  Same for `normal_sf` / `vectorized_normal_sf`, which use `math.erfc`.
- **Outcome range** — `Ceiling`/`Floor` are 85th/15th percentiles, from smearing each
  player across a MEASURED finish-rank distribution (`calibrate_rank_uncertainty`), not a
  constant: top-6 QBs/TEs land within ~10 slots, top-6 RBs/WRs scatter more than twice as
  far. Weights are made doubly stochastic via **Sinkhorn normalization** so positional
  point totals are conserved. `Risk` = (Ceiling−Floor)/Proj.
- **Valuation** — `compute_starter_demand` simulates every team's lineup filling greedily
  rather than splitting FLEX evenly across RB/WR/TE (flex goes to whoever's better at the
  margin, which is exactly what changes with settings). `VORP` vs the first unstarted
  player; `VOLS` vs the last dedicated-slot starter; `VONA` vs the EXPECTED best player
  still there at your next pick. VONA is the one that resolves draft-room paralysis.
- **`STREAMABLE_POSITIONS` includes TE**, which is not obvious and was a real fix. The
  reasoning that excluded it ("leagues roster 60+ of them") is true of RB/WR and false of
  TE: a 12-team league starting one rosters ~15 against 32 NFL starters, so ~17 startable
  TEs sit free all year. That's quarterback's structure, not receiver's. Streaming returns
  167 against a TE13 baseline of 156, and those 11 points were the whole TE distortion —
  every TE from ~TE8 to TE14 carried positive VORP and floated 30-40 picks above ADP.
  Settings-sensitive without special-casing (a 0.5 TE premium moves the bar to 201).
- **Roster depth is priced** — `expected_start_share(pos, depth, slots, absence)`: how
  much of a season the nth player you own at a position actually spends in your lineup.
  Absence rates are near-identical across positions (QB 26%, RB 21%, WR 20%, TE 22%), so
  SLOT COUNT does all the work: a 3rd WR fills a flex nearly every week (1.00), a 2nd TE
  plays ~3 weeks (0.27), a 3rd plays none (0.00). Recommendations DROP a position you
  can't play another of rather than scaling it down — VONA measures what you lose by
  waiting, and you lose nothing waiting on a player who'd never enter your lineup.
  (Symptom before the fix: the same backup QB recommended six rounds running.)
- **`Avail Next %`** — normal survival function over ADP. This is the column that changes
  how you draft, and the user reported it broken twice before it was actually fixed
  (gotcha #22).
- **Market blend** (`apply_market_blend`) — blends the board's ORDER toward ADP in rank
  space. **`VORP` is never blended** — it stays exactly what the model says. Falls back to
  blending toward ECR when ADP is down, because the blend is the board's main defence
  against out-thinking the whole analyst industry on one model's say-so, and switching it
  off when a feed dies is backwards.
- **K/DST demotion** exists because of a measured failure: an earlier board put 26 of the
  top 120 slots on K/DST. Two hypotheses were tested and ruled out with data (kickers are
  NOT less predictable — K year-over-year corr 0.23 vs WR 0.16; streaming didn't explain
  it either). Root cause was availability blindness: VORP is a correct measure of surplus
  and a useless measure of when to spend a pick. A VOND metric was also tried and
  **abandoned** — it rewarded deep sleepers in sparse ADP regions (Joe Mixon at overall
  13).
- **Positional value-add** (`draft_intel.positional_value_add`) — the "+X% to your team"
  model. `marginal_lineup_gain(best available now) − marginal_lineup_gain(expected best at
  next pick)`, over `_projected_full_lineup`. It is a DIFFERENCE, not a level, and that's
  the whole point: ranking by "how good is the best one available" just re-sorts by raw
  scoring and says take a QB every time. Two corrections keep it from reading 0% once your
  lineup fills: bench depth carries an OPTION value (`E[max(X − waiver, 0)]` off the
  board's own Ceiling — a late pick is an option, you drop a bust and stream), and the
  wait-one-turn term is discounted by `(picks_left − 1) / picks_left`, because "the cost of
  waiting" presumes you CAN wait and on your last pick you can't. Early rounds are
  unchanged by construction (0.9 at ten picks out).
- **Mock simulator** (`draft_sim.py`) — bots draft off **Board Rank** with softmax
  exploration, NOT off marginal lineup gain (gotcha #26). `_legal_positions` enforces
  starters-before-backups and counts **dedicated slots only** for the backup ladder
  (gotcha #25).
- **Board caching** — `_board_cache_key` covers every setting that changes a number, so
  toggling PPR rebuilds and toggling a display option doesn't. Underscore-prefixed frames
  are passed unhashed (gotcha #2 applies here in full force).

**Model agreement, as of the August 2026 audit** (keep this honest if you change
anything): vs FantasyPros ECR rank-corr **0.941**, median positional |bias| 8.0
(QB −4, RB −5, WR +9, TE −8); vs FFA Value 0.901 / 10.0. Projected points vs FFA analyst
projections r=0.918, MAE 22.0, bias QB −7 / RB +2 / WR +3 / TE +8. Known remaining disagreements (veteran TEs ranked higher
here, young ascending WRs lower) are documented with reasons in §5 of the methodology doc.
There's an audit script pattern worth reusing — rebuild the board, join ECR and FFA Value,
report rank-corr + per-position bias + a sanity-check block.

**August 2026 follow-up** (methodology doc §7): the pure-model (pre-market-blend)
RB bias vs ADP was measured separately by draft depth and found concentrated almost
entirely in the backup/handcuff tier (+6 inside the realistically-drafted range,
+18 past the bench-stash line) rather than flat across the position. Two changes
shipped off that finding — a recency-evidence path in `project_stat_lines` (a thin
CAREER games sample no longer dilutes a player who just proved a full, uncontested
season) and a new RB-only `Handcuff Value` term in `add_value_over_replacement`'s
pipeline (`data.draft_board.add_handcuff_value`, reusing `contingency_value`'s
option-value math against the STARTER's own distribution). Median RB bias vs ADP:
pure model +17 → +4, blended board +8 → +2. Backtested against 62 real
team-seasons: a confirmed handcuff RB's PPG jumps a median +6.3 when the starter
misses a game (positive in 85.5% of cases) — the premise is real, not assumed.
Full derivation, the two real bugs hit building it (a `nan` truthiness bug and a
depth-chart over-spreading bug, both fixed), and the remaining book-confirmed gaps
(veteran QBs, some handcuffs) are in the methodology doc, not repeated here.

## 4. UI conventions

- **Stat tiles / hero tiles / team banner** (July 2026 polish pass) -
  `ui.components.render_stat_tiles` (Savant-style tile grid with percentile bubble +
  bottom meter; replaced Player Search's old single-row horizontal-scroll matrix table,
  same snapshot entries/colors/percentiles), `render_hero_tiles` (headline-number band:
  fantasy PPG / total points / position rank / games - suppressed for seasons with no
  weekly data, guarded on `'week' in p_data.columns` like everything else), and
  `render_team_banner` (Depth Charts team-color gradient + ESPN logo via
  `data.loaders.load_team_logos()`, which finally uses the previously-unused
  `teams_colors_logos.csv`). Their static CSS classes (`.stat-tile*`, `.hero-tile*`,
  `.team-banner`) live in `inject_theme`; per-tile color/width is inline (data-driven).
- **Perf pattern for per-rerun work**: `build_player_search_labels` is vectorized (no
  iterrows - it runs on every Player Search/Compare rerun), `compute_bye_weeks` and
  Depth Charts' `_build_snap_totals` are `@st.cache_data`-cached on year. If adding
  per-rerun logic over the full stats frame to a tab, cache or vectorize it - the
  felt per-click lag from exactly this class of code was real and measured.

- Widget sizing uses `width="stretch"` everywhere — `use_container_width` was fully
  migrated out (it's removed from Streamlit after 2025-12-31; reintroducing it will
  break on a future upgrade).
- Sidebar shows an in-season data-freshness banner
  (`ui.components.check_current_season_freshness`): compares the latest COMPLETED week
  on the real schedule against the max week in the current season's local weekly stats
  CSV — loud warning when the local file is stale, quiet caption when current. Its
  per-file cache is keyed on the CSV's mtime so dropping in a fresh file busts it
  automatically.
- Tables: pandas Styler + `st.dataframe` (NOT raw HTML — grid headers come from
  `.streamlit/config.toml` theme; page CSS can't reach them). `style_plain_dataframe`
  takes `numeric_pct_cols` (PFF-grade rainbow), `matchup_pct_cols` (muted green↔red for
  matchup tables), `diverging_cols` (signed deltas). Column hover-tooltips via
  `build_column_help_config` / `COLUMN_HELP` (styling.py) — raw HTML `title` attrs do
  NOT work on the canvas-rendered grid.
- Sticky game log (Player Search) is the one raw-HTML table — sticky AVG footer can't be
  done with st.dataframe. Per-position `log_cols` picks which raw stat columns show -
  K uses `fg_made`/`fg_att`/`fg_long`/`pat_made`/`pat_att` (all real per-week columns in
  the source file), not `weekly_snap_pct` like most other non-QB positions - kickers
  aren't in the offensive/defensive snap-count export at all (same gap the bio card's
  "N/A" already covers), so that column was always a flat, misleading 0.0% for them.
- Cross-tab navigation: `switch_tab(TAB_X, **context)` — **must be invoked as a widget
  callback** (`on_click=` / `on_select=` closure), never inline after checking a widget
  return value (see gotcha #1). Destination tabs consume context keys with
  `st.session_state.pop(...)`. Every jump also records where you jumped FROM in
  `st.session_state['nav_back_tab']`; `ui.components.render_back_button()` (called at
  the top of Player Search's `render()`) shows a "← Back to X" button when that's set.
  Clicking it calls `switch_tab` again, which naturally re-records the reverse
  direction — a two-way toggle, not a full history stack.
- Table row/cell → player jumps: `st.dataframe(on_select=..., selection_mode=
  "single-row"|"single-cell", key=...)`; the callback reads
  `st.session_state[key]['selection']`.
- Player Search: one autocomplete selectbox (no separate text input) + optional team
  filter; selected player persists across year changes via
  `st.session_state['player_sel_t1_name']` re-resolved into each year's label list.
  Uses `index=None, placeholder="..."` (a real empty/unselected widget state) — NOT a
  fake "— Select a player —" string stuffed into the options list at index 0. The
  latter renders as literal, editable text the moment the box gets a real selection,
  which is exactly why the old version required deleting the whole placeholder before
  typing a search; `index=None` shows genuine placeholder text that clears itself the
  instant the box gets focus/input.
- League-leader marker (top PFF grade at a position, for the selected season —
  `data.loaders._build_master_pff_grades`'s `league_gold_players` set): on the Depth
  Chart, a 🏆 prefixed onto the cell text in `style_depth_chart_table` - NOT a
  border/box-shadow, since glide-data-grid's canvas renderer silently drops those (see
  gotcha #11). On the Player Search bio card, the WHOLE card's border/glow recolors to
  gold (`render_player_card(..., is_gold=...)`) instead of the usual grade-color one,
  plus the same 🏆 suffix on the name - a real HTML page, so border/box-shadow do work
  there, but an outline layered onto just the headshot specifically fought with the
  card's own overflow:hidden and absolute-positioned badges and was barely visible in
  practice; recoloring the card's own already-working border was the reliable fix.
  Depth chart needs the abbreviated-name bridge (`match_abbreviated_name`) to check
  membership; Player Search doesn't (its n_col is already a full name in PFF's own
  format).
- Seasons with no weekly stats yet (e.g. 2026 pre-kickoff) show the schedule as the
  game log; detect via `'week' not in p_data.columns` — NOT via `log_df.empty` (gotcha
  #6).

### Draft HQ UI conventions (August 2026)
- **Position colors are global and mainstream** (`config.POSITION_COLORS`): QB orange
  `#f59e0b`, RB green `#22c55e`, WR blue `#60a5fa`, TE purple `#c084fc`, K pink, DST cyan.
  Chosen to match what every other fantasy site uses, so the board reads correctly to
  someone who's used one. `get_position_chip_bg()` gives the muted background for chips.
  **Every place a player is listed carries the position color** — user requirement, not a
  nice-to-have: board cells, roster slots, the draft grid, the recent-picks strip, the
  value-add cards. `ui.styling` has an `is_position` styling branch alongside `is_team`.
- **The board table's alignment bug is fixed and easy to reintroduce.**
  `div[data-testid="stDataFrame"]` needs `box-sizing: border-box !important` and
  `overflow: hidden !important` — the 6px padding + 1px border were being added OUTSIDE
  the content box, so the canvas rendered ~14px wider than its frame and visibly
  overhung its background. Verified via measured `elW`/`canvasW` in the browser.
- **Hover language is shared across every clickable surface**:
  `transform: translateY(-1px) scale(1.015)` + glow on hover, `scale(0.985)` on active.
  Applied to `.rp-card` (recent picks), `.pv-card` (value-add), `.db-cell` (draft grid)
  and buttons so the whole tab feels like one product.
- **Position filter is BUTTONS, not a dropdown** (All/QB/RB/WR/TE/FLEX/K/DST) — explicit
  user requirement. The active filter also scopes the single pick recommendation.
- **One recommendation, not a list.** `_render_single_recommendation` shows exactly one
  player, scoped by the position filter. It used to be a top-6 list and the user asked for
  it cut to one.
- **Live draft and mock share one surface.** `_draft_context` unifies both behind one
  interface so every panel below works identically; there is no separate mock tab and no
  separate draft-board tab (both were merged in, deliberately).
- **Player profile jump uses `on_click`, not an inline branch.** `switch_tab` writes to
  the keyed `active_tab` widget, so it is only legal from a callback (gotcha #1). Verified
  that jumping to Player Search and coming back preserves both draft state and the
  selected row.

## 5. Gotchas — every one of these was a real bug here

1. **Streamlit widget-state ordering.** `st.session_state[key]` for a keyed widget can
   only be assigned BEFORE that widget is instantiated in the current script pass, or
   from a widget callback. Assigning after raises `StreamlitAPIException`. All cross-tab
   navigation therefore goes through `on_click`/`on_select` callbacks.

2. **`@st.cache_data` cache keys.** Underscore-prefixed params are NOT hashed. A cached
   function whose only real inputs are underscore-prefixed shares ONE stale entry across
   all calls (year switches returned wrong-year data until `year` was added as a real
   param). Every cached transform here takes `year`; keep that pattern. Cheap groupbys
   over already-cached frames are deliberately left uncached instead.

3. **Name formats differ per source, in ~4 distinct ways:**
   - nflverse `player_display_name`: "Patrick Mahomes" (full)
   - roster_weekly `player_name`: "P.Mahomes" (abbreviated — depth chart uses this)
   - PFF `player`: full name; PFF `team_name` in coverage files holds team CODES
     (ARZ/BLT/CLV/HST), not names — map via `config.pff_team_to_abbr`.
   - Sharp Football `team`: bare nicknames ("Dolphins").
   Tools: `clean_name_exact` (first choice — preserves suffixes so Byron Murphy ≠ Byron
   Murphy II), `clean_name_for_merge` (suffix-stripped FALLBACK only — as a primary key
   it merged two real players), `match_abbreviated_name`/`build_last_name_index`
   (bridges "P.Mahomes"→"patrick mahomes"; skipping this made 100% of depth-chart PFF
   lookups miss and mislabeled everyone ROOKIE). Test tricky names: Byron Murphy II,
   A.J. Terrell, Marvin Harrison Jr., Amon-Ra St. Brown, Ray-Ray McCloud.
   The clean long-term fix is ID joins via `load_player_id_crosswalk` (pff_id↔gsis_id,
   verified 100%) — not yet wired in.

   `match_abbreviated_name` matches on the FULL prefix before the period, not just its
   first letter - nflverse itself lengthens the abbreviation (e.g. "Mi." vs "Ma.")
   specifically when two same-last-name, same-initial teammates would otherwise collide
   (Michael Wilson / Mack Wilson, both ARI 2025). Collapsing to one letter throws away
   exactly the disambiguating information nflverse added and can resolve to the WRONG
   player. Even fixed, a bare single-letter abbreviation is only unambiguous WITHIN one
   team - resolving a cross-tab jump league-wide also needs the player's team passed as
   context (`switch_tab(..., jump_to_team=...)`) so the candidate pool gets narrowed
   before the name match runs, not just the (initial, last-name) pair alone.

4. **Per-season data pretending to be per-player.** qbr_season_level.csv has ~15 rows
   per veteran QB; a bare name match + `.iloc[0]` froze every QB's QBR at his earliest
   season. Filter by `season` + `season_type == 'Regular'` and compute percentiles
   within-season. Same class of bug applies to any multi-season file.

5. **NaN sorting in st.dataframe.** glide-data-grid does not reliably push NaN to the
   bottom on sort — unranked/undrafted rows scattered above rank #1. Fix everywhere: a
   real sentinel one worse than the max real value, never NaN/None in sortable rank
   columns.

6. **Roster-only seasons have phantom stat columns.** For a year with no weekly stats
   file, `load_year_data` still emits `passing_yards` etc. as real 0-filled columns —
   only `week`/`opponent_team` are truly absent. `df.empty` checks pass when they
   shouldn't; test `'week' in columns` instead. Also: `weekly_snap_pct` is a uniform
   0.0 placeholder in those years, not NaN — never rank on it without checking
   `'week' in columns` first (depth chart does this via `has_real_weekly`).

7. **Silent empty-DataFrame fallthrough.** `load_pff` returns empty on a missing file
   and every downstream `if not df.empty` silently skips — a misnamed file once killed
   all WR/TE receiving metrics with nothing visibly wrong. The Data Health sidebar
   (`ui/components.check_data_health`) exists to surface this; keep it in sync with
   loader paths when they change (it went stale once already after the per-year
   foldering).

8. **nflreadpy schema ≠ local CSV schema.** load_snap_counts is long-format
   (`player`/`week`/`offense_pct` 0–1) vs the local wide `Wk N pct` (0–100) format;
   the unpivoted fallback silently produced no snap data for years without local files.
   Loaders normalize to the local shape — preserve that if touching snap loading.

9. **Column duplication in wide frames.** 16 sequential single-column inserts on a
   190-col frame triggers pandas' block-manager fragmentation (PerformanceWarnings in
   the logs are from this; mostly fixed via batch concat in load_year_data — remaining
   warnings there are known/cosmetic).

10. **Categorical dtypes vs Arrow.** Team/Pos columns are categorical upstream; after
    enough groupby/merge steps st.dataframe's Arrow conversion can refuse them — boards
    cast to `str` before display. Keep doing that on new boards.

11. **`st.dataframe`'s grid (glide-data-grid) ignores border/box-shadow/outline.** It
    draws cells on an HTML canvas, not real DOM nodes - `ui/elements/lib/
    pandas_styler_utils.py` forwards a Styler's CSS to the frontend verbatim, but the
    frontend cell renderer only actually reads background-color/color/font-weight out of
    it and silently drops everything else. A gold border on a depth-chart cell rendered
    correctly in `styler.to_html()` (a separate, unrelated static-HTML code path pandas
    Styler also supports) and never appeared in the real grid - confirmed by checking the
    user's own screenshot, not by trusting the HTML check. If a table needs a highlight
    the grid can't paint, use background color or a text marker (e.g. a prefixed emoji)
    instead, never border/box-shadow/outline. Real HTML tables (Player Search's bio card,
    sticky game log) don't have this limitation - border/outline work fine there.

12. **`match_abbreviated_name` must use the abbreviation's FULL prefix, not just its
    first letter.** nflverse itself lengthens an abbreviation (e.g. "Mi." vs "Ma.")
    specifically when two same-last-name, same-initial teammates would otherwise
    collide - confirmed real: Michael Wilson and Mack Wilson, both ARI 2025, are
    "Mi.Wilson" and "Ma.Wilson". An earlier version of this bridge collapsed both to a
    single first-letter "m" and silently resolved one player's cell to the other's PFF
    grade/rookie status/whatever. Also: a first-initial match is only unambiguous
    WITHIN one team - resolving a Depth Chart's abbreviated cell name into Player
    Search's league-wide dropdown needs the clicked player's team passed along as
    context too (`jump_to_team`), or two unrelated same-initial players elsewhere in the
    league can still collide.

13. **A player identity key must be stable across the season, not just within one
    week's rows.** `fetch_intelligent_depth_chart` used to group by roster_weekly's own
    abbreviated `player_name` column directly - fine most of the time, except nflverse
    itself sometimes changes that abbreviation for the SAME player mid-season (the
    Wilson example above: "Mi.Wilson" weeks 1-11, "M.Wilson" weeks 12-18), which split
    him into two "different" depth-chart entries each with half a season's stats.
    `exact_name` (derived from the full, always-consistent name column) is the actual
    stable identity key; the abbreviated column is display text only.

14. **`get_pff_color`'s CSS `rgba(...)` strings are not valid matplotlib colors.** Every
    existing call site fed an HTML/CSS context (Styler backgrounds, a card's inline
    style), so this never mattered until `render_percentile_bars_figure` passed a
    snapshot entry's color straight to `ax.barh(color=...)` — matplotlib's parser wants
    hex or 0-1 float tuples, not comma-separated 0-255 ints, and raised
    `ValueError: Invalid RGBA argument: 'rgba(244, 135, 18, 0.82)'` the moment a QB's real
    (non-fallback-gray) percentile color reached it. Direct-data testing of the
    data-producing function looked fine (it returns plain strings, no matplotlib
    involved) — only exercising the actual chart-rendering call in the browser surfaced
    it. `ui.player_snapshot._mpl_color()` now converts before handing color to
    matplotlib; any new chart that reuses `get_pff_color`/`get_matchup_color` output
    needs the same conversion, not a fresh reimplementation.

15. **Streamlit's widget DOM migrated from BaseWeb to react-aria - old CSS selectors
    die silently.** On the current Streamlit version, st.tabs renders as
    `[data-testid="stTab"]` divs inside a `[role="tablist"]` (aria-selected on the tab
    itself) and selectbox/multiselect/number fields render as react-aria groups
    (`.stSelectbox [role="group"]` etc, class `react-aria-ComboBox`) - NOT the old
    `button[data-baseweb="tab"]` / `div[data-baseweb="select"]` markup. Confirmed by
    inspecting the live DOM: the previous design pass's pill-tab and dark-input styling
    matched NOTHING and every tab/dropdown had been silently rendering with default
    Streamlit styling, with zero errors anywhere. `inject_theme` now carries BOTH
    selector generations side by side. If a styling rule ever seems to have no effect,
    check the real rendered DOM first - a dead selector fails with no signal at all.

16. **`inject_theme`'s whole CSS block is one big f-string - any literal `{`/`}` in a
    comment breaks the entire app, not just that comment.** Added a `/* ... "{player} VS
    {team}" ... */` descriptive comment while building the matchup-title component (July
    2026 pass) and it raised `NameError: name 'player' is not defined` on EVERY tab (the
    whole app 500'd, since `inject_theme()` runs before any tab body does) - Python's
    f-string parser doesn't care whether a `{...}` is inside a CSS comment vs a real rule,
    it tries to evaluate anything between single braces as an expression. Every literal
    brace in this file (including inside comments, not just CSS rules) must be doubled -
    `{{player}}` - exactly like the real CSS rules around it already do. `python -m
    py_compile` does NOT catch this (the f-string is syntactically valid Python, it just
    references an undefined name at *format* time) - only actually running the app (or an
    AppTest sweep) surfaces it, confirmed live this session.

17. **`data.utils.calculate_percentile` returns all-ZEROS (not NaN, not an error) for a
    column name that doesn't exist in the given DataFrame.** Found via a systematic July
    2026 audit sampling real players across every position and flagging stat labels that
    were ALWAYS blank: `data.loaders.load_all_pff_data` was computing `mtf_pct` from
    `'missed_tackles_forced'` (real column is `elu_rush_mtf` - rushing_summary.csv has no
    `missed_tackles_forced` column at all) and `yprr_slot_pct`/`yprr_screen_pct`/
    `grades_pass_route_slot_pct` from columns that don't exist in `pff['rec']`
    (receiving_summary.csv) either - the real slot/screen alignment split lives in a
    DIFFERENT PFF export, `pff['route_concept']` (receiving_concept.csv), under different
    names again (`slot_yprr`/`screen_yprr`/`slot_grades_pass_route`). Because
    `calculate_percentile` silently returns zeros instead of raising or returning NaN for a
    missing column, every one of these looked like ordinary "no percentile available" cells
    (a real, expected state elsewhere in this app) rather than a broken lookup - `MTF/G`
    showed "--" for every single RB, every season, and `ui.player_snapshot`'s WR/TE branch
    had TWO entries for the same real-world stat side by side in `SECONDARY_TILE_GROUPS`'s
    "Slot Production" group ('Y/RR Slot' reading the broken column, 'Slot YPRR' correctly
    reading `route_concept` two lines below it) - one silently dead, one working, neither
    erroring. Fixed by correcting the column names/source dataframe in both
    `data.loaders.load_all_pff_data` and `ui.player_snapshot.build_player_snapshot`, and
    deleting the dead duplicate entries rather than fixing them in place. **Lesson: don't
    trust `python -m py_compile` OR a percentile of exactly/near 0 as proof a PFF-sourced
    stat is wired correctly - verify the raw column name actually exists in
    `pff[...].columns` for the real loaded file, not just that the code runs.**

### Draft HQ gotchas (August 2026 — all real, all found on real data)

18. **A prefix filter over stat columns silently drops `receptions`.** The value curves
    selected scored stats with `startswith(('passing_', 'rushing_', 'receiving_', ...))`.
    `receptions` does not start with `receiving_`, so **PPR contributed exactly zero to
    every curve** — WR1 read 265 points instead of 380. It computed cleanly, rendered
    cleanly, and just quietly priced every receiver like it was a standard league.
    `score_stats` now names every stat explicitly via `_col`, which makes this whole class
    of bug impossible. **Never reintroduce a prefix filter over stat columns.**

19. **A rank curve's `games` figure is a selection artifact, not durability.** Games
    played by whoever FINISHED rank r slopes 16.6 (QB1) → 10.6 (QB28), but that slope is
    mostly "missing games is how you end up ranked 28th". Using it as a per-player games
    multiplier put every QB ~30 points under consensus. The tempting fix — rescale the
    curve UP to a flat 17 — is worse: it invents a player who plays a full season and
    still finishes 28th, flattening every positional curve (ECR rank-corr 0.937 → 0.905,
    all four positions 15-28 points ABOVE consensus). A third attempt using each player's
    own games-per-season also failed, because at this sample size that measures role
    changes and rookie seasons rather than health. See the `PACE_GAMES` note.

20. **Any "measure what the free pool returns" calculation needs a HINDSIGHT check.**
    `build_streaming_replacement` originally defined "rostered" by FINAL season totals, so
    the leftover free pool was systematically the worst players by construction — the
    measurement was guaranteed to make streaming look bad. Fixed to use PRIOR-season
    finish. (The corrected number still came in below the last starter, which correctly
    REJECTED the streaming hypothesis rather than confirming it — a fixed measurement that
    changes your conclusion is the point.)

21. **Sinkhorn-balanced weights can push a player's ceiling below his own projection.**
    Balancing the finish-probability matrix in both directions conserves positional point
    totals, which is what you want for the LEVEL — but reading percentiles off the
    balanced weights put the top player's 85th percentile under his own expectation.
    Percentiles use the row-normalized (unbalanced) weights, plus a
    `np.maximum(ceiling, projected)` clamp. There is a sanity check for exactly this.

22. **`next_pick_for` must index by how many picks YOU have made, not the total.** Using
    the total picks logged made `Avail Next %` freeze — the user reported it twice before
    it was actually fixed. Correct form is `my_picks[len(my_roster)]`. Verify by watching
    it advance 20 → 29 → 44 across three of your own picks, not by reading the code.

23. **A projected stat line can score NEGATIVE and it looks completely normal.** Twenty
    deep-bench players sat below zero, where a rank curve's thin share of fumbles and
    interceptions outweighed a near-zero share of production. `Proj Pts` is now floored at
    zero. A season projection is an expectation; no draftable player has a negative one.

24. **A partial-coverage import will blank a wider column if you assign instead of
    fill.** `merge_ffa_into_board` wrote `out[col] = [ffa_value_or_None for each row]`.
    FFA's export covers ~260 players and the board carries ~700, so importing silently
    DELETED `Age` for the ~440 players it doesn't list — a column with 92% coverage from
    birthdates, replaced by one with 42%. Passthrough columns now `combine_first` onto
    whatever's already there.

25. **A single FLEX slot is not a full slot for RB, WR AND TE.** `draft_sim`'s backup
    ladder counted it three times, which produced rosters with 3-4 tight ends and a QB2 in
    round 3. Fixed to dedicated slots only, plus a no-backups-until-starters-are-full
    rule. Roster construction is the thing users notice first when a sim is wrong.

26. **Don't draft bots off marginal lineup gain, and don't use a deterministic argmax.**
    On an empty lineup, marginal gain sees Josh Allen's ~150-point edge over every RB and
    takes a QB with the 1.01. Bots draft off Board Rank instead. Separately, a
    deterministic pick made the pick-odds panel read 100% on one position — bots use a
    softmax with `explore=0.6`.

27. **When a user says a feature "did nothing", check what file they actually gave it
    before debugging the parser.** The FFA upload was rejecting `index.json` (the HAR
    manifest) exactly as designed and reporting only "no recognisable player list", which
    reads identically to "this feature is broken". `diagnose_ffa_payload` now names the
    file they want (`api/players.json`). A correct rejection with a useless message is a
    bug.

28. **A same-URL POST repeated N times will overwrite itself if you key output files by
    URL alone.** `tools/har_extract.py` lost 15 of 16 captures of one endpoint. It now
    distinguishes two collision types: different URLs sharing a basename get a hash
    suffix, the same URL repeated gets an ordinal (`_002`, `_003`).

29. **A string-replace edit that lands inside a docstring can produce prose outside it.**
    Left orphaned text after a closing `"""` and got `SyntaxError: invalid decimal
    literal` — a genuinely confusing error for what was a documentation edit. Worth a
    `py_compile` after any docstring surgery.

30. **`if x:` is truthy for `float('nan')` — NaN is non-zero, so a NaN guard needs
    `is not None and np.isfinite(x)`, not a bare truthiness check.** Building the
    August 2026 recency-evidence path (methodology doc §7.1), `if games_last_season:`
    let every player with an undefined `games_last_season` (no local-CSV row last
    season — true of most deep backups) through as if the value were real, because
    `min(1.0, nan / 17.0)` silently returns `1.0` rather than raising or propagating
    the NaN. Confirmed real: Sam Ehlinger, Case Keenum, Bailey Zappe, Skylar Thompson
    and Hendon Hooker — QB4+ arms with tiny, noisy own-rate samples and no
    `games_last_season` at all — jumped to 100% own-history trust off garbage-time
    attempt rates several times what a player at their real draft slot carries.
    `python -m py_compile` and a plain read of the logic both look completely fine;
    only printing the actual `evidence` value for a real NaN case surfaces it.

31. **Don't let one option-value calculation fire once per row when it should fire
    once per GROUP.** The first version of the RB handcuff term (methodology doc
    §7.2) looped every backup on a team's depth chart against the same starter,
    which priced the same "starter gets hurt" event again for the 3rd- and
    4th-string bodies behind the real handcuff — a McCaffrey backup and San
    Francisco's fullback landed within 5 points of each other. Fixed by restricting
    to the single next-best player by projection, not the whole remaining group.

## 6. Verification workflow (what "done" means here)

1. `python -m py_compile <changed files>` — necessary, wildly insufficient.
2. Run the changed function directly in Python against REAL local data; eyeball actual
   values for plausibility (e.g. Mahomes 2019 PFF ≈ 90, CMC RZ carry share ≈ 77%).
   Suppress noise: pipe through `grep -v "No runtime found\|ScriptRunContext\|
   MemoryCacheStorageManager\|PerformanceWarning"`.
3. Start the dev server (launch config `gridiron-hub`; if port 8501 is stuck, a stale
   python.exe owns it — kill it), click through the affected tab, check
   `preview_logs level=error` and the browser console.
4. Year-sensitive changes: test 2025 (full local data), 2019 (nflreadpy fallbacks), and
   2026 (roster-only, no weekly stats) — they exercise three different code paths.

Known test-environment quirk: the browser tooling's synthetic clicks don't register on
glide-data-grid's canvas hit-testing, so table row-click navigation can only be verified
code-level + manually.

**For Draft HQ specifically, "done" also means a measured audit.** Model changes are not
verifiable by eyeballing a board — every change in the August 2026 pass was accepted or
rejected on numbers. The pattern: rebuild the board, join FantasyPros ECR and FFA Value,
and report (a) rank correlation and median positional bias against each, (b) projected
points vs FFA's analyst projections per position, (c) a block of sanity assertions
(Ceiling ≥ Proj, Floor ≤ Proj, no negative projections, Avail Next in 0-100, no K/DST in
the top 100, tiers monotone with points, every position has a replacement level).

Three changes in that pass were BUILT, MEASURED, AND REJECTED — a flat-17 curve rescale,
per-player durability games, and a symmetric young-player age boost. If a change doesn't
improve agreement, it doesn't ship, and the rejection gets written down next to the
constant so nobody tries it again.

Also worth knowing: in a sandboxed environment the ADP column reads empty, because
`external_data/ffa_players.json` is gitignored (so absent) and Fantasy Football
Calculator is network-blocked. That is expected, not a bug — ADP falls back to the ECR
estimate and the source label says so.

## 7. Deliberately NOT done / parked

- **ID-based PFF joins** via the crosswalk (section 2) — known-good approach, real
  refactor (~15 call sites), unstarted.
- **Injury reports** (`nflreadpy.load_injuries`, 2019+) — surfaced in Draft HQ's News
  sub-tab as of August 2026; still not surfaced on the other nine tabs.
- **WP (win probability) surfacing** — EPA/success rate/CPOE are now surfaced (section 3);
  WP itself is still an unused pbp column.
- **Literal route tree / spray chart** — needs route-geometry data this app doesn't have
  locally; `build_target_location_profile` (section 3) is the honest stand-in.
- **"Similar players" comp engine** (Baseball-Savant-style) — needs a defined multi-stat
  similarity metric; considered during the July 2026 pass, deliberately scoped out as a
  separate future feature rather than bundled in.
- **MOF Closed/Open % history** — impossible without a paid source; UI says so.
- **Yahoo OAuth draft sync** — blocked (dev app lacks Fantasy Sports permission);
  paste-parser + Sleeper sync are the substitutes.
- **SportsBlaze** — evaluated, rejected (strict subset of what nflreadpy provides).
- **Single-file .exe packaging** — evaluated, rejected: Streamlit is a web server, so
  an .exe still just opens a browser; PyInstaller-freezing Streamlit is fragile (asset/
  metadata introspection breaks, hooks break on upgrades) and this dependency stack
  would produce a 500MB+ brittle bundle. `run_app.bat` at the project root is the
  double-click launcher instead.
- `stats_player_reg_*.csv`, `ftn_charting_*.csv`, `otc_players*.csv`,
  `teams_colors_logos.csv` — on disk, unused.

### Draft HQ specifically
- **Real-time sync to a live draft league** — investigated and answered: FFA does NOT do
  this either. Their `DraftSitePicker` is an ADP-SOURCE selector, and `LiveHelper` is a
  premium in-simulator panel, not a league connection. Sleeper's public API does support
  draft-pick polling and `_render_live_sync` uses it; ESPN/Yahoo need OAuth (Yahoo is
  already blocked, see above). Paste-parser + Sleeper sync are the substitutes.
- **Reverse-engineering FFA's position score** — their draft assistant computes
  `score = strategyFreq + boostPct/100 + valuePct` where `boostPct = (dynamicMultiplier −
  1) × 100`, but the multipliers come from their SERVER and are not recoverable from the
  client bundle. `positional_value_add` is a different route to the same question and has
  the advantage of being explainable line by line. Don't go looking for the multipliers
  again — they aren't there.
- **VOND (value over next-drafted)** — built, measured, abandoned. Rewarded deep sleepers
  in sparse ADP regions (Joe Mixon at overall 13).
- **Symmetric young-player age boost** — built, measured, does nothing (see the
  `AGE_ADJUST_MIN` note in `draft_projections.py`). Don't re-add without new evidence.
- **Historical preseason ECR archive** — would make `calibrate_rank_uncertainty` properly
  correct (regress finish against PRESEASON consensus instead of year-over-year finish).
  No free source publishes one; `CONSENSUS_SKILL_SHRINK = 0.75` is the stand-in.
- **ID-based joins in the draft stack** — `db_playerids.csv` is already fetched and
  carries sleeper/espn/yahoo/fantasypros/gsis IDs. Live sync uses `sleeper_id` where the
  platform gives one; the board's own merges are still two-tier name matching
  (`clean_name_exact` → `clean_name_for_merge`). Same known-possible refactor as the PFF
  one above.

## 8. Constraints (user-set, don't violate)

- Don't restructure app.py's tab wiring or undo lazy tab execution.
- Don't rename files in `pff_imports/` or the root CSVs.
- Don't remove the plaintext-API-key warning in Live Odds.
- No new paid data sources or heavyweight dependencies. **No scipy** — the draft engine
  needs a gamma distribution and a normal survival function and implements both directly
  (`_gammq`, `math.erfc`) rather than pulling the dependency in.
- **This repo is PUBLIC and the user is fine with that.** The condition is that FFA's paid
  data is upload-only and never committed: `external_data/ffa_players.json` and `*.har`
  are gitignored and must stay that way. No code may fetch from FFA's servers.
- **Don't change the Player Search tab.** Direct user instruction. Draft HQ links INTO it
  (`switch_tab(TAB_PLAYER_SEARCH, jump_to_player=...)`) and consumes its existing context
  keys; it does not modify it.
- **Don't touch externally-sourced stats.** FFA Value, FFA ADP, FFA projections,
  FantasyPros ECR, dynasty values, injury designations and news are displayed exactly as
  fetched. Everything the app computes itself is fair game to improve — that distinction
  is the user's, and it's why `docs/draft_hq_methodology.md` splits its stats into
  "sourced" and "computed" up front.
- Historical depth standard is 2019 (`config.AVAILABLE_SEASONS`) — one-line change if
  the user ever pulls more PFF years.
- A separate trimmed distributable lives at `C:\FantasyF_NFLScholar_dad\` (+
  `C:\NFL_Scholar.zip`) — a copy for a relative; it does NOT auto-sync with this repo
  and excludes the odds key.
