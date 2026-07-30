# NFL Scholar — Handoff Doc

Personal Streamlit NFL fantasy-football analytics app. Single user (Yahoo/ESPN/Sleeper
leagues), runs locally via `streamlit run app.py` (or the `.claude/launch.json` config
named `gridiron-hub`). Last major work pass: July 2026 "pro polish" pass (perf fixes +
UI elevation toward a pro sports-site look - stat tile grid, hero stat band, team
banners, and the react-aria CSS-selector migration in gotcha #15, which had silently
killed the previous pass's tab/input styling). All 9 tabs verified working end-to-end
at that time (AppTest per-tab sweep + live browser click-through, zero exceptions).

Second July 2026 pass (same week): Player Search's fantasy-points chart converted from a
bar strip to an SVG line chart and moved to the very top of the player profile; the WR
percentile-bar chart's value labels no longer crowd/overlap near the 100th percentile;
the skill radar grid's spoke labels/titles got more breathing room; the Career Totals
table is now centered in a narrower column instead of stretched full-bleed; Defensive
Yield's coverage-radar title is a real styled/centered component
(`ui.components.render_matchup_title`); VORP Draft Sheet moved to the last tab; Player
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

**Season-readiness audit pass (July 30 2026, pre-kickoff):** full walkthrough of all 9
tabs (AppTest sweep across every tab × year × all 32 teams, zero exceptions, plus live
browser click-through) specifically checking 2026-27 readiness. Found and fixed real bugs,
none previously caught:
- **Risers/Waiver Wire and Weekly Rankings' Season dropdowns used the plain
  `AVAILABLE_SEASONS` list (excludes 2026), not `AVAILABLE_SEASONS_WITH_UPCOMING`** - the
  two tabs a user most needs mid-SEASON had no way to ever select the 2026 season at all,
  which would have made both effectively dead for the entire 26-27 season until someone
  manually edited `config.py`. Both already degrade cleanly on empty data ("not enough
  weekly data yet") so switching them to the WITH_UPCOMING list was a safe, one-line fix
  each (`ui/tabs/risers.py`, `ui/tabs/rankings.py`). VORP Draft Sheet's baseline-season
  selector deliberately stays on the season-only list - a draft baseline should be a
  completed season's pace, not the unplayed one.
- **`data.loaders.load_weekly_stats_history` (feeds Player Search's Career Totals table)
  never filtered to `season_type == 'REG'`**, unlike `load_year_data`'s own well-documented
  filter for the exact same class of bug (section 2) - playoff games were silently blended
  into every season row for any team that made a run. Confirmed real: Mahomes' 2023 row
  showed "20" games played (16 REG + 4 POST, Super Bowl LVIII) instead of 16, with
  Pass Yds/TDs/Fantasy Pts inflated to match. Fixed by applying the same filter a second
  time in this separate loader.
- **`ui.components._local_stats_max_week` (sidebar freshness caption) had the identical
  gap** - it read the raw `week` column unfiltered, so once a season's playoffs are in the
  same file it reported POST week numbers (19-22) as if they were real season weeks. The
  completed 2025 season showed "current through week 22" (not a real NFL week) instead of
  18. Same REG-only filter added here too.
- **A second, independent instance of gotcha #17's exact failure mode**: `adj_comp_pct`
  ("Adj Comp %") was computed from a PFF column name (`adjusted_completion_percent` /
  `adj_completion_percent`) that doesn't exist in the real export - the real column is
  `accuracy_percent`. `calculate_percentile` silently returns all-zeros for a missing
  column, so every QB league-wide showed "Adj Comp % 0.0%" - a plausible-looking wrong
  number, not an error. Separately (and independent of the column-name bug), the display
  code was reading the PERCENTILE column (`adj_comp_pct`) as if it were the raw stat value
  - every sibling stat (BTT%/TWP%/ADOT/P2S%) correctly displays its own raw source column
  and uses a separate `_pct` column only for bar color, but Adj Comp % had no raw-value
  column at all. Fixed both: `adj_comp_pct_raw` now carries the real `accuracy_percent`
  value for display; `adj_comp_pct` stays the percentile, color-only, same as its siblings.
- **`fetch_intelligent_depth_chart` always rendered the "LB (unclassified)" / "S / DB
  (unclassified)" catch-all rows even when empty** - most teams have zero players tagged
  with only the bare generic LB/S/DB code (everyone else gets sorted into their precise
  ILB/OLB/CB/SS/FS row first), so most Depth Charts showed a fully blank row wasting space.
  Now dropped when every slot in the row is empty (never drops "BREAK", the offense/defense
  divider `ui.tabs.depth_charts.render()` splits on).
- **Roster full_name dedup could pick a garbage/ID-less row over the real player sharing
  that name** - see new gotcha #18 below.
- **Skill Radar (Player Search) blew up to ~3x its intended size whenever fewer than 3 of
  the normal 3 stat-group panels had enough real data to render** - confirmed live on a
  2026 pre-kickoff QB (Efficiency/Situational both need per-game weekly/pbp rates that
  don't exist yet; only PFF-fallback-backed Ball Security survives), a real, common
  situation for the first weeks of any new season, not an edge case. `st.pyplot(...,
  width="stretch")` filled the SAME full-bleed container width regardless of the figure's
  actual panel count. Fixed in `ui.tabs.player_search.render()` by sizing the render
  column to the figure's own panel count (inferred from its width in inches) instead of
  always stretching full width.
- **VORP Draft Sheet's Efficiency-vs-Volume quadrant chart's player-name callouts used one
  fixed label offset**, which overlapped illegibly whenever 2+ of a panel's (up to 6)
  labeled points sat close together in data space - confirmed real on 2025 QBs (Goff/
  Prescott/Mahomes cluster tightly). Fixed in `ui.player_snapshot._draw_quadrant_panel` by
  fanning labels into their own horizontal lane (cumulative dx by x-rank) plus alternating
  above/below - a same-direction-only alternation wasn't enough on its own (two labels one
  x-rank apart, with an unrelated third point's label from a very different y-value sitting
  between them in sort order, still landed on the same side).
- **Player Compare's "Shared Stat Comparison" column was labeled bare "Edge"** but is a
  percentile-POINT gap between the two players (not a raw-value delta) - deliberate, one
  scale works across every stat regardless of native units, but reads as obviously wrong
  next to two close raw values (97.4% vs 93.0% snap share showing a "42.5" edge). Relabeled
  to "Edge (Pctl)" - no logic change, `ui/tabs/player_compare.py`.
- Defensive Yield's "Remaining games only" SOS checkbox already correctly fell back to the
  full season when a season is complete (`build_strength_of_schedule`'s existing fallback -
  not new), but gave no on-screen indication that happened, so the checkbox stayed checked
  and reading "remaining" while showing the whole season - added a one-line caption for the
  fallback case, matching every other fallback caption's convention in this app.

This doc exists so a fresh session (human or AI) can orient without re-deriving the
project's hard-won lessons. **The gotchas in section 5 are all real bugs that happened
here — several more than once.** Read that section before changing anything.

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
```

**Tabs** (order = `config.TAB_LABELS`): Player Search, NFL Depth Charts, Defensive Yield
Schemes, Risers/Waiver Wire, Rookie Watch, Weekly Rankings, VORP Draft Sheet, Live Odds,
Player Compare.

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

## 3. Key computed systems

- **Scoring** — `apply_scoring_and_percentiles` (transforms.py). One formula, PPR
  multiplier parameterized, includes IDP defaults AND kicker defaults (3/4/5 pts for FG
  makes under 40 / 40-49 / 50+ yards via the `fg_made_X_Y` distance-bucket columns, 1 pt
  per PAT) - nflverse's own `fantasy_points`/`fantasy_points_ppr` columns are offense-only
  and never include kicking, so kickers scored a flat 0 every week app-wide before this,
  identical in kind to the IDP gap noted below. `build_vorp_draft_sheet` has its OWN
  separate from-scratch scoring formula (does not reuse this column) and needed the same
  fix independently - see below. `score_projected_stats` is a third dict-of-scalars twin
  used by the projection model — **all three are kept in sync manually**; change one,
  check the others.
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
- **VORP** — `build_vorp_draft_sheet`: last season's per-game pace × 17, replacement
  level by league settings, 4-game minimum sample. Explicitly labeled a volume stand-in,
  not a projection. Rankings comparison (FantasyPros draft / custom upload) is merged
  into the same table on the VORP Draft Sheet tab. Includes K (League Settings has a "K
  starters" input, default 1) with the same hardcoded distance-tiered scoring as
  apply_scoring_and_percentiles above - this function builds `proj_points` from its own
  raw stat sums rather than reusing the shared `fantasy_points` column, so it needed the
  fix applied separately; every kicker projected to exactly 0 and sat at VORP 0 before.
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

18. **A `full_name` collision in `roster_weekly_*.csv` between the real player and a
    garbage/incomplete row can make `load_year_data`'s roster dedup pick the WRONG one -
    and it doesn't fail loudly, it just quietly loses the real player's bio data.**
    `rosters_clean = rosters.drop_duplicates(subset=['full_name'])` (after sorting
    newest-week-first) assumed the only reason two rows share a `full_name` is a real
    player appearing on multiple weeks. Confirmed real: `roster_weekly_2025.csv` also
    carries a second "Quinshon Judkins" row - team GB, position DL, status DEV, week 18
    only, no `gsis_id`/`birth_date`/`college` at all (reads as a garbage/placeholder entry,
    not a real second person) - alongside the actual CLE RB's rows. Sorting by week alone
    let that week-18 phantom row "win" the dedup purely for being more recent, which threw
    away the real player's own `gsis_id` from `rosters_clean` entirely - the later
    `player_id`/`gsis_id` merge then couldn't match ANY of his real stat rows to a bio row,
    so his Team/Draft Pick/etc. went blank or wrong (Rookie Watch showed Team "None", Draft
    Pick sentinel 256) on every single row he has all season, while his stats themselves
    kept loading fine (they don't depend on this merge) - exactly the kind of bug that looks
    like a data gap rather than a merge defect. Fixed by sorting on "has a real `gsis_id`"
    (True first) BEFORE the week sort, so a real player can never lose the dedup to an
    ID-less row sharing his name. **This is a different failure mode from gotcha #3's
    same-person-different-spelling problem** - this is two DIFFERENT roster rows, one of
    them garbage, sharing an EXACT name string. It does not fix - and isn't meant to fix -
    two REAL different players who happen to share a full name (confirmed still real and
    present: e.g. two real "Byron Young"s in 2025); that case still needs the parked
    ID-based crosswalk join (section 7) to resolve correctly, since both sides of a
    real/real collision have a legitimate `gsis_id` and this ordering doesn't disambiguate
    between them.

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

## 7. Deliberately NOT done / parked

- **ID-based PFF joins** via the crosswalk (section 2) — known-good approach, real
  refactor (~15 call sites), unstarted.
- **Injury reports** (`nflreadpy.load_injuries`, 2019+) — available, not surfaced.
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

## 8. Constraints (user-set, don't violate)

- Don't restructure app.py's tab wiring or undo lazy tab execution.
- Don't rename files in `pff_imports/` or the root CSVs.
- Don't remove the plaintext-API-key warning in Live Odds.
- No new paid data sources or heavyweight dependencies.
- Historical depth standard is 2019 (`config.AVAILABLE_SEASONS`) — one-line change if
  the user ever pulls more PFF years.
- A separate trimmed distributable lives at `C:\FantasyF_NFLScholar_dad\` (+
  `C:\NFL_Scholar.zip`) — a copy for a relative; it does NOT auto-sync with this repo
  and excludes the odds key.
