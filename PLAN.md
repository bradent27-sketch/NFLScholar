# Gridiron Intelligence Hub — UI Overhaul & Feature Implementation Plan

Status: COMPLETE — all 8 phases implemented and verified against a running `streamlit run app.py` session (all 8 tabs click-through clean, zero console/server errors).

## 0. Decisions (resolved)

1. **Rankings data source** — resolved. You provided three FantasyPros
   `FantasyPros_2026_Draft_ALL_Rankings*.csv` exports. Mapping auto-detected by comparing
   RB/WR ordering across the three files (a pure-rushing back like Derrick Henry ranks
   much lower under PPR than under Standard scoring, since he doesn't add receiving value —
   confirmed his rank moves from "outside the top 30" → #23 → #13 across the three files in
   exactly that order):
   - `FantasyPros_2026_Draft_ALL_Rankings.csv` (no suffix) → **Full PPR**
   - `FantasyPros_2026_Draft_ALL_Rankings (1).csv` → **Half-PPR**
   - `FantasyPros_2026_Draft_ALL_Rankings (2).csv` → **Standard**
   You said you'll periodically refresh these with new exports in the same format — the
   loader is keyed on schema (`RK, TIERS, PLAYER NAME, TEAM, POS, BYE WEEK, UPSIDE, BUST,
   SOS SEASON, ECR VS. ADP`), not on the specific filenames, so drop-in replacement works
   without code changes. Copied into the project at:
   - `rankings/fantasypros_2026_draft_rankings_ppr.csv`
   - `rankings/fantasypros_2026_draft_rankings_half_ppr.csv`
   - `rankings/fantasypros_2026_draft_rankings_standard.csv`
2. **Defensive Coverage Correlator data** — resolved. You linked 6 SumerSports team-stats
   pages (offense + defense × Overview / Personnel Tendency / Formation Tendency). I fetched
   all 6 directly and confirmed they contain EPA/personnel-grouping/formation-alignment
   tendency — **not** Cover-1/2/3/4/6 shell rates (that data isn't published in tabular form
   on SumerSports' free team pages). Saved as one-time local CSV pulls (not a standing
   scraper) in `external_data/`:
   - `sumersports_defensive_overview_2025.csv`, `sumersports_offensive_overview_2025.csv`
   - `sumersports_defensive_personnel_tendency_2025.csv`,
     `sumersports_offensive_personnel_tendency_2025.csv` (each team's "11 personnel" usage
     rate + EPA — the standard benchmark personnel grouping)
   - `sumersports_defensive_formation_tendency_2025.csv`,
     `sumersports_offensive_formation_tendency_2025.csv` (each team's "2X2" usage rate + EPA
     — the standard benchmark alignment)

   Final approach: build the radar's core correlation on local PFF man/zone data
   (`receiving_scheme_2025.csv` player-level man/zone YPRR + target share,
   `defense_coverage_scheme_2025.csv` team-level man/zone volume), with the SumerSports
   personnel/formation tendency data folded in as a secondary "scheme context" panel
   alongside it (e.g. "this defense faces 11 personnel at X% league rank, gives up Y EPA
   against it").
3. **Modularize now vs. later** — no explicit answer; proceeding with the stated default
   (modularize as part of this pass, see §2) since the Sleeper restyle touches nearly every
   line of the current inline-CSS/UI code and deferring would mean redoing that work twice.
   Flag before Phase 2 if you'd rather keep this pass single-file.

## 1. Design language (synthesized from the 3 reference systems)

Read `design.md/DESIGN.md` (Sleeper), `design.md/Dunks and Threes Design MD.md`, and
`design.md/Spotify Design MD.md`. Sleeper is the named target; the other two inform specific
gaps Sleeper's own spec doesn't cover in enough depth for this app (dense stat tables).

- **Base surface & accent (Sleeper):** deep navy canvas `#050921` → `#0a0f2a` →
  `#131b38` surface stack, cyan `#00fff9` primary accent, electric blue `#00baff`
  secondary, warm orange `#ffae58` tertiary/alerts. This replaces the current
  `#121212`-black/Spotify-green theme app-wide.
- **Typography (Sleeper):** Poppins for headlines/labels/section headers, Inter for body
  and table cells. Both are Google Fonts — loaded via `@import` in the injected CSS (no
  new Python dependency).
- **Table density & numeric legibility (Dunks & Threes):** compact cell padding, uppercase
  bold table headers, and — new — monospace numerals (JetBrains Mono, Dunks & Threes'
  convention) for stat columns specifically, since this app's tables are almost entirely
  numeric and misaligned proportional digits are a real readability cost at this density.
  Positive/negative stat coloring convention (green/red) borrowed for VORP and "Pct Jump"
  columns.
- **Elevation & card glassmorphism (Sleeper + Spotify):** `backdrop-filter: blur()` cards
  per Sleeper's spec for containers (player card, settings panel, radar panel); Spotify's
  more restrained shadow-only depth for simple list rows.
- **Shape:** full-radius pills for buttons/chips/tabs (all three systems agree on this),
  `4–10px` radius for cards/inputs (Sleeper's scale).
- Percentile heatmap cell coloring (the purple→red `get_pff_color` scale) is a load-bearing
  existing feature, not a Sleeper convention — kept as-is, just re-themed to sit correctly
  against the new navy background instead of the old near-black.

Concretely: a new `THEME` token dict (colors/fonts/radii/spacing) drives one injected CSS
block, replacing the current hardcoded `#121212`/`#1ed760` values scattered across
`style_*` functions.

## 2. Target module layout

```
app.py                      # thin entrypoint: page config, theme injection, sidebar, tab wiring
config.py                   # TEAM_CONFIG, MASTER_TEAMS_LIST, STAT_DECIMALS, THEME tokens
data/
  loaders.py                 # load_year_data, load_pff, load_weekly_stats_history,
                              # load_external_coverage_schemes, build_veteran_database,
                              # load_master_pff_grades, fetch_nfl_odds, fetch_nfl_player_props,
                              # fetch_sleeper_draft_picks
  transforms.py               # apply_scoring_and_percentiles, precompute_league_percentiles,
                              # build_player_historical_summary, build_risers_report,
                              # build_rookie_watch, build_vorp_draft_sheet,
                              # name-matching utils, parse_pasted_draft_picks, match_names_to_board,
                              # fetch_intelligent_depth_chart
  rankings.py                 # NEW: FantasyPros CSV ingestion, custom-ranking upload, blending
  coverage_radar.py           # NEW: man/zone WR-vs-scheme correlation + radar chart data prep
ui/
  styling.py                  # style_plain_dataframe, style_game_log_table,
                              # style_game_log_avg_row (→ sticky footer), style_depth_chart_table,
                              # df_auto_height, get_pff_color
  components.py               # player card HTML, team-agnostic search box, collapsible
                              # settings wrapper, sticky-footer container helper
  tabs/
    player_search.py          # tab1
    depth_charts.py           # tab2
    defensive_yield.py         # tab3 + new radar section
    risers.py                 # tab4
    rookie_watch.py           # tab5
    vorp_draft.py             # tab6, restyled + collapsible settings
    rankings.py                # NEW tab
    live_odds.py               # tab7
requirements.txt              # new: pin current deps (none new needed — matplotlib already
                              # installed and covers the radar chart; no plotly required)
```

Shared cross-tab state (`selected_player`, `t1_target_year`, `drafted_players`, etc.) moves
to `st.session_state`, keyed per-tab (e.g. `st.session_state.player_search.selected_player`)
so tab modules don't rely on bare script-level variables — required once tab bodies become
functions in separate files (flagged in the handoff as a prerequisite, not optional).

Two-tier name matching, dtype-downcast exclusions, rounding-before-cast, and the other
gotchas in `HANDOFF.md` carry over unchanged — this is a structural move, not a rewrite of
that logic.

## 3. Phases

**Phase 1 — Design system foundation**
New `config.py` THEME tokens + rewritten CSS injection block (Sleeper navy/cyan, Poppins/
Inter/JetBrains Mono, pill buttons, glass cards). Applied globally before any other change,
so every subsequent phase is built and screenshotted against the real target look rather
than the old dark-green theme.

**Phase 2 — Modularization**
Extract `config.py`, `data/loaders.py`, `data/transforms.py`, `ui/styling.py` out of
`app.py` with no logic changes (pure move + import wiring). Migrate shared state to
`st.session_state`. Verify the app runs identically (same tabs, same data) before touching
any tab's UI code.

**Phase 3 — Table refinement (sticky footer)**
Replace the current "always-visible AVG row rendered above the slider-filtered table"
pattern with a real CSS `position: sticky; bottom: 0` footer bar, so the Season Average row
stays pinned to the bottom of the viewport while the weekly rows scroll underneath it inside
their own scrollable container — matches the literal ask ("remain visible at the bottom of
the viewport while the weekly stats area above it remains scrollable"), not just "always
rendered."

**Phase 4 — Player Search refactor**
Remove the mandatory "Select Team" selectbox. Replace with a single team-agnostic
`st.selectbox` (searchable-by-typing, which Streamlit selectboxes already support) over
every player league-wide for the selected year, with team shown as metadata next to each
name in the dropdown rather than as a pre-filter. Team badge still shown on the result card.

**Phase 5 — VORP Draft Sheet restyle + collapsible settings**
Re-skin the draft sheet table/cards in the new palette (readable navy/slate, not
white-on-near-black). Wrap "League Settings" in a collapsed-by-default
`st.expander`/dropdown so the draft tracker is the primary visible surface, settings tucked
away until opened.

**Phase 6 — Rankings tab (new)**
- Ingest the three FantasyPros CSVs you provided (pending the PPR/Half/Standard mapping
  from §0.1) as the "professional rankings" source, selectable by scoring format.
- File-upload widget for a custom ranking CSV (any Player/Rank pair — parsed via the
  existing two-tier name matcher against the app's player universe, same as VORP's
  draft-pick matcher).
- Side-by-side view: your custom rank vs. selected FantasyPros rank vs. this app's own VORP
  rank, with a delta column highlighting where they disagree most.

**Phase 7 — Defensive Coverage Correlator (new "Radar" view)**
Per §0.2 default: for a selected WR, pull `receiving_scheme_2025.csv` (man YPRR/target-share
vs. zone YPRR/target-share for that player) and cross it against each opposing team's
man%/zone% tendency from `defense_coverage_scheme_2025.csv` (aggregated to team level) to
project which upcoming matchups favor that player's actual man/zone split. Rendered as a
matplotlib polar/radar chart (no new dependency — matplotlib is already installed).
Structured so a future CSV drop with real Cover-1/2/3/4/6 shell rates slots in without a
rewrite.

**Phase 8 — Polish pass**
`requirements.txt` committed. Full click-through of all 8 tabs in a running `streamlit run`
session to confirm no regressions (data still loads, VORP/draft tracker/odds still work),
self-review diff, fix anything that looks off against the design docs.

## 4. Definition of done (per your directive)

- [ ] PLAN.md created and approved (this file — pending your sign-off)
- [ ] UI components updated to match the Sleeper-inspired design language across all tabs
- [ ] Rankings tab functional: FantasyPros ingestion + custom upload + comparison view
- [ ] Defensive Coverage Radar functional against real local data
- [ ] Sticky Season-Average footer behaves as specified (pinned, weekly area scrolls)
- [ ] Player Search has no mandatory team filter
- [ ] VORP sheet restyled + League Settings collapsible
- [ ] All features integrated with existing stat data, no regressions vs. current behavior
