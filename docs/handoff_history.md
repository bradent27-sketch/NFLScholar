# NFL Scholar — Handoff History (archived narrative)

Pass-by-pass build narrative moved out of `HANDOFF.md` on 2026-08-30 so the
handoff doc stays an orientation document, not a changelog. Nothing here is
load-bearing — every durable fact (data contracts, gotchas, model decisions) is
also carried in `HANDOFF.md` §1–§8, `docs/weekly_projections_methodology.md`,
`docs/draft_hq_methodology.md`, `docs/weekly_rankings_backlog.md`, and the
`docs/overnight_backtest_log_*.md` series. Read this only for "why was it built
this way" context on an older pass.

Test counts, "N tabs verified", and "current major work" phrasing in the blocks
below are accurate **as of that pass**, not now.

---

## July 2026 — "pro polish" passes

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

---

## August 2026 passes

**August 2026 follow-up pass — Matchup Analyzer / Weekly Rankings / Draft HQ refinement.**
Note: the app remains at nine top-level tabs, but its structure changed after an earlier
undocumented pass: Game Slate was added first, Matchup Analyzer was added after Player
Compare, and Risers/Rookie Watch/Weekly Rankings merged into one "Weekly Fantasy" tab with
sub-tabs. Section 1 now reflects the current `config.TAB_LABELS`; keep it synchronized when
tab wiring changes.

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

**August 2026 pass — Matchup Analyzer bolstering (receiver alignment stats,
defense coverage overhaul, usage trend chart).** Scoped entirely to
`ui/tabs/matchup_analyzer.py` / `data/matchup_signals.py` per explicit
request; Player Search and its shared `ui/player_snapshot.py` builder were
left untouched to limit the scope of that pass. This is historical scope, not
a standing prohibition: section 8 now permits Player Search changes with
targeted regression coverage.

- **Calculated Wide YPRR** (`data.matchup_signals.build_wide_yprr_table` /
  `wide_yprr_entry`) - PFF's route-concept export breaks out Slot specifically
  (real `slot_routes`/`slot_yards`/`slot_yprr`) but never publishes Wide as
  its own concept. Computed as "the rest of his routes" - season total minus
  the real slot routes/yards, from `receiving_summary` + `receiving_concept`
  - per explicit request. For a WR this is clean (in-line snaps are
  negligible); for a TE the non-slot remainder is genuinely "wide + in-line
  combined" since no export in this app has in-line yardage broken out
  separately - `includes_inline` flags that on every row so callers can
  disclose it rather than overclaiming precision. Surfaces in two places:
  Route Efficiency (as a sub-bar under Wide rate, mirroring Slot YPRR's
  existing sub-bar under Slot rate) and the new receiver Tendency Profile.
- **Receiver Tendency Profile is now a dedicated builder**
  (`data.matchup_signals.receiver_tendency_entries`), a deliberate FORK of
  `ui.player_snapshot.build_player_snapshot`'s WR/TE branch, not an edit to
  it - that shared function also feeds Player Search's matrix table and
  Player Compare, both off-limits. New order: PFF Rec Grade (renamed from
  RecV Grade), PFF Blocking Grade (new -
  `data.matchup_signals.blocking_grade_entry`, from PFF's `offense_blocking`
  export, percentiled within the player's OWN position only since that file
  also carries O-line), Targets/G, Rec/G, RecYd/G (new), [TE only: Inline%],
  Slot% (+ Slot YPRR / Slot Rec Grade sub-rows), Wide% (+ calculated Wide
  YPRR sub-row), EPA/Target, Success Rate, ADOT, YPRR, Drop Rate. Slot Route
  %/Screen Route %/Screen YPRR are gone from this tab's profile per explicit
  request (still present, unchanged, on Player Search/Player Compare via the
  shared builder). QB/RB/every other position still goes through
  `build_player_snapshot` exactly as before.
- **Defense Coverage panel rebuilt** (`_render_coverage`) - Man vs Zone is
  now a 2-column split-bar table (`data.matchup_signals.man_zone_grade_rows`)
  laid out exactly like the player side's own vs-Man/vs-Zone panel (Rate,
  Coverage Grade, QB Rating Allowed rows, Man left/Zone right, league
  percentile driving bar length/color) instead of four flat hero tiles with
  no league context. MOF Closed/Open renamed to Single-High/Two-High (the
  correct shell-naming direction - MOF closed = one deep safety = single-high;
  MOF open = two deep safeties = two-high; the OLD hero-tile caption had this
  backwards) and given real league percentiles for the first time
  (`coverage_profile`'s scheme dict gained `zone_pct`/`mof_closed_pct`/
  `mof_open_pct` alongside the pre-existing `man_pct`). Outside/Slot yards-
  per-target (a single rate with no volume context) replaced by
  `data.matchup_signals.defense_alignment_allowed`: real Targets/Rec/Yards/
  TDs/Yds-per-Target, Slot vs "Wide". Slot is real, measured PFF data
  (`pff['slot_cov']`); PFF has no outside/wide coverage export, so Wide is
  computed the same "total minus real slot" way Wide YPRR is
  (`pff['cov_summary']` minus `pff['slot_cov']`, summed per team) - same
  caveat inherited (a safety in middle-of-field zone isn't "outside" either).
- **Usage & Role is now a week-by-week line chart** (`_render_usage_and_role`)
  instead of three static season-average tiles - one chart, a picker for
  Target share / Carry share / Opportunity share, same shape as the player's
  own Game By Game chart. Catch rate dropped from this section (still
  computed by `ms.usage_and_role`, just not shown here - it's an efficiency
  read, not a role one). `usage_and_role`'s weekly frame gained an `opponent`
  column to feed the chart's tooltips/bar labels.
- **Defense Week By Week Detail simplified** - the "By Stat" tab is gone (it
  was a strictly worse path to the same chart, position-second instead of
  position-first); Position and Stat pickers now sit side by side
  (`st.columns(2)`) instead of stacked, and a small percentile-colored
  indicator tile (`data.matchup_signals.defense_stat_rank`,
  `ui.components.render_percentile_metric_tiles`) sits above the chart
  showing the season per-game average, league rank, and league average
  before the trend line does.
- **Hover polish extended to `ui.charts.render_split_bars`** (now used by
  the new Man/Zone and Slot/Wide tables, on top of its pre-existing Route
  Efficiency/Scheme Fit callers) - same brighten-the-fill/bold-the-label
  language `.pbar-row:hover` already established, via new `.split-bar-row`/
  `.split-bar-fill`/`.split-bar-label`/`.split-bar-value` CSS classes.
  `.metric-tile` also picked up the same lift/border-glow card-hover token
  `.stat-tile`/`.hero-tile` already had (it never got it when those did).

---

**August 2026 pass — Draft HQ big-play bonuses/book weighting, weekly model
matchup+recency overhaul, 2026 season coverage.** A broad pass across three
areas per explicit request: user-configurable big-play scoring in Draft HQ,
a reliability-weighted multi-book consensus, and a materially deeper
Weekly Rankings model (matchup-quality-adjusted history on both sides of
the ball, plus a real week-1/cold-start fallback). Full detail below;
HANDOFF's own "measure, don't eyeball" discipline (section 6) was followed
throughout - `scripts/validate_weekly_projections.py` and the new
`scripts/audit_book_vs_model.py` back every claim made here.

- **Draft HQ: user-defined big-play bonuses** (`data/draft_big_plays.py`,
  new module; hooked into `data.draft_projections.score_projected_lines`
  via a new `latest_season` param) - "+2 points for any reception of 40+
  yards"-style rules, ADD/REMOVE ANY NUMBER of them live via a
  `st.data_editor(num_rows="dynamic")` grid in Draft HQ's League Scoring
  panel (`ui.tabs.draft_hq`'s "Big-play bonuses" section) - this codebase's
  first use of that widget anywhere (the existing per-game yardage bonuses,
  `data.draft_board.YARDAGE_BONUSES`, are a fixed hardcoded list of twelve
  thresholds and were NOT generalized into this, deliberately - a big play
  is an event on ONE PLAY, not a threshold on a game's summed yardage, and
  needs a genuinely different distributional model).
  - Priced from real per-play NFL play-by-play (`data.loaders.load_pbp`,
    the first consumer of it in the draft stack - previously only
    `data.transforms.build_redzone_usage`/`build_epa_efficiency` touched
    it), never touched by `PROJECTED_STATS` before this. Each player's own
    recency-weighted (`SEASON_RECENCY_WEIGHTS`, same ladder
    `build_player_rates` uses) big-play RATE is blended toward his
    position's rate with his own qualifying-touch count as evidence
    (`FULL_TRUST_TOUCHES=300`, `BIG_PLAY_SELF_WEIGHT=0.45` - same
    `weight = evidence * self_weight` shape as `STAT_SELF_WEIGHT`
    elsewhere), so a rookie or a thin sample doesn't read a single lucky
    play as an established skill. A name/team not found in the rate table
    (a rookie with no NFL history) falls back to the POSITION's baseline
    rate, never to zero.
  - **Real bug caught and fixed before shipping**: the first version
    counted a pass-completion rule's rate denominator as "completed passes
    only" while multiplying it against the board's PROJECTED ATTEMPTS
    (all attempts, complete or not) - silently inflating every QB's
    big-play bonus by roughly 1/completion-rate (measured: Josh Allen's
    three-rule total dropped from 12.6 to 8.6 points once fixed). Fixed by
    carrying a `complete` flag through the shared per-play frame so the
    denominator (all non-sack attempts) and the hit condition (completed
    AND long enough) are counted separately - see
    `data.draft_big_plays._big_play_history`'s docstring.
  - Adds a `Big Play Pts` column to the board (same "computed, not shown by
    default" convention the existing `Bonus Pts` column already uses -
    folded into `Proj Pts`, inspectable but not forced into `BOARD_COLUMNS`).
- **Draft HQ: ECR/ADP settings disclosure** (`ui.tabs.draft_hq._ecr_adp_
  disclosure_note`) - a caption next to the ADP source picker stating
  exactly what's actually pulled: ECR reflects ONLY the chosen draft format
  (Redraft 1QB/Superflex/Best Ball/Dynasty variants) and is NOT
  scoring-adjusted at all (FantasyPros' free feed has no PPR/Half-PPR/
  Standard split), unlike ADP, which DOES follow the PPR slider (bucketed
  Standard/Half-PPR/Full PPR) but collapses onto one Superflex page
  regardless of PPR when Superflex is on. Also added as `COLUMN_HELP`
  hover-tooltip entries (`ui/styling.py`) on the `ECR`/`ADP` board columns
  themselves. A LIVE per-week ECR pull was investigated (FantasyPros' API
  does have `/rankings`/`/consensus-rankings` endpoints - see
  `data/draft_sources.py`'s existing note on them) but NOT shipped - no
  verified real response to build a parser against in this environment, and
  a wrong guess would silently spend the 50-call/month budget on garbage.
  The reliable substitute (a real FantasyPros weekly CSV export) already
  works and now also feeds the Weekly Rankings comparison below.
- **Draft HQ: weighted book consensus** (`data.odds_sources.BOOK_WEIGHTS` -
  DraftKings 20 / FanDuel 20 / Pinnacle 25 / PrizePicks 20 / Underdog 15,
  user-set) replaces `data.odds_projections.market_stat_lines`'s old plain
  MEDIAN-across-providers with a weighted mean, vectorized (weight*line
  summed per (player, stat) group over the summed weight of whichever
  books actually posted a line - no per-group Python loop). A book that
  didn't price a stat is simply not in the group, so it can never drag the
  consensus down - confirmed on real data (Ja'Marr Chase's 5-book
  `receiving_yards` consensus computed to exactly the hand-verified
  weighted figure; a 2-book case, e.g. Mahomes with only DraftKings +
  PrizePicks, renormalizes over just those two). An unrecognized provider
  gets `BOOK_WEIGHT_FALLBACK=10` (equal-weight vote) rather than being
  silently excluded.
  - **`scripts/audit_book_vs_model.py`** (new, reusable) - builds the board
    at Half-PPR and Full PPR, joins the weighted book consensus, and
    reports rank-correlation/bias/a receptions-specific comparison plus the
    section-6 sanity block. Measured result (Aug 2026, 5 real book
    payloads): rank-corr 0.93-0.96 vs Book Proj at both PPR levels, median
    bias -3 to -5 points (model slightly under book), reception deltas
    mostly under 2 catches (biggest outlier a rookie WR with no own
    history, Emeka Egbuka at -7.4 - the "market knows something the model
    can't" case `data/odds_projections.py`'s own docstring already
    describes as the common one) - close enough that no model recalibration
    was made off this audit, just the weighting change itself.
- **Weekly model: matchup-quality-adjusted, recency-weighted history on
  BOTH sides of the ball** (`data/weekly_projections.py` -
  `build_team_game_quality_adjusted_matchup`, `_weighted_player_rates`, replacing an
  earlier flat 60%-trailing-4-game/40%-season-average split) - per explicit
  request, three stacked adjustments on every past game feeding a player's
  own rate:
  1. RECENCY - continuous per-game decay (`RECENCY_DECAY=0.85`), not a
     hard 4-game cutoff.
  2. MATCHUP STRENGTH - a big game against a bad defense is scaled DOWN
     before averaging (divided by that defense's rating), a quiet game
     against a good defense scaled UP - `HISTORY_MATCHUP_CLIP=(0.85,1.15)`,
     narrower than the forward-looking `MATCHUP_CLIP=(0.75,1.3)` used for
     the upcoming opponent, since a single retroactive game correction is a
     noisier estimate than the multiplier applied once to a whole
     projection (measured - a wider clip here tested net-neutral to
     slightly worse).
  3. REMATCH - a past game against the SAME team faced again this week
     gets extra weight (`REMATCH_WEIGHT_MULT=1.6`) on top of its ordinary
     recency weight.
  - The DEFENSE side's matchup rating is now an **offense-position team-game
    profile** (`build_team_game_quality_adjusted_matchup`), not an average of
    individual player/season ratios. For each QB, RB, WR, or TE channel, all
    same-position player rows are summed into one offense-versus-defense
    game, compared with that offense's own normal positional output, then
    estimated as a recency-weighted pooled observed/expected ratio. Four
    league-average neutral games shrink sparse channels (especially TD/INT
    counts) toward 1.0 before league re-centering. This means a replacement,
    injury fill-in, or one statless relief appearance cannot manufacture a
    2–3x defense signal from a tiny personal denominator or count as another
    independent game. QB rushing, RB rushing, RB receiving, WR receiving,
    and TE receiving remain independent channels.
  - Historical team identity is immutable for this calculation:
    `data.loaders.load_year_data` preserves raw weekly `game_team` before a
    roster merge replaces visible `team` with the player's latest team. Use
    `game_team` for historical team-game math; never substitute the current
    roster team after a trade. A `game_id` fallback exists only for older
    cached frames with no `game_team` field.
  - Role-conditioned tables use the same team-game grain for players whose
    measured role was present in that game, and remain shrunk toward the
    broad profile. QB passing bypasses role overlays entirely because its
    one-QB team-game profile is the relevant evidence. The obsolete
    player-row helper is retained only as a regression-test counterexample,
    never as a Weekly Rankings production input.
  - **Honest measured result, not a claimed win**: this whole reweighting
    is within noise of the flat split it replaced on
    `scripts/validate_weekly_projections.py` (2025 & 2024, weeks 5-17) -
    rank-corr 0.655/0.654 either way, MAE within ~0.02-0.1 across every
    variant tried. QB and TE improved a little, RB and WR moved a little
    the other way. Kept because it's exactly the mechanism asked for and
    measurably doesn't hurt, not because it moved the aggregate numbers -
    see the `HISTORY_MATCHUP_CLIP` constant's own comment for the full
    tuning note. Also notable and NOT investigated further (out of scope
    this pass, flagged for later): the model was already sitting within
    noise of - and on raw MAE/rank-corr, slightly behind - a naive
    trailing-4-game recent-form baseline BEFORE this pass, on both
    seasons tested.
  - Usage/efficiency emphasis (the other half of this request) falls out
    of the same change: a raw box-score total is never averaged on its own
    terms anymore, only after being leveled for opponent quality and
    recency - a talent/usage read rather than a highlight-reel average.
- **Weekly model: week-1/cold-start fallback to prior-season data**
  (`build_weekly_projections`'s `cold_start` branch, `_cold_start_pool`) -
  previously hard-bailed with "not enough data" the moment `hist` (this
  season's played weeks before the target week) was empty, which is ALWAYS
  true for week 1 and was the literal reported symptom ("weekly fantasy
  doesn't populate for week 1 of 2026"). Now: the player pool (who's on
  which team) comes from the target week's own roster rows when they exist
  or the roster-only fallback frame when they don't (never the whole
  unfiltered season, which would leak a later week's post-trade team
  backward into a week-1-only read); every rate falls through to
  `_blended_rate`'s existing prior-season branch (`cur_games=0` already
  meant `w_current=0` even before this pass - the ACTUAL gap was never
  having a player pool to run that math for at all); the defense-matchup
  matrix is rebuilt off PRIOR season's full year, anchored so its FINAL
  week reads as "most recent" (`prior_max_week + 1`); team pace falls back
  to the prior season too if the current one has none yet. `meta['cold_start']`
  flags the result so a caller can show "based on last season" - the
  Weekly Rankings tab doesn't consume that flag yet, a small follow-up if
  wanted. **Real bug caught and fixed before shipping**: a name-column
  mismatch (`year - 1`'s frame uses `player_display_name`, a roster-only
  current year uses `player_name`) raised a bare `KeyError` the moment this
  path was actually exercised against real 2026 data - existing code
  had never reached this line because of the old hard bail-out. Fixed by
  keying the cross-season rate lookup on `clean_name_exact`, not a shared
  raw column name. Verified end to end against real 2026 rosters (915
  players projected for week 1, Christian McCaffrey/Puka Nacua/Patrick
  Mahomes at the top, no NaN, no negative stat lines) and, as a second
  no-network-dependency check, by treating `as_of_week=1` of the completed
  2025 season as a synthetic cold start (354 players, using real 2024 rates).
- **Individual projected stat lines floored at zero** in both
  `data/draft_projections.py` (`project_stat_lines`) and
  `data/weekly_projections.py` (the final per-stat multiply, and
  `Model Proj Pts` itself) - gotcha #23 already floored the POINTS total;
  this pass's audit (`scripts/audit_book_vs_model.py`'s sanity block)
  found individual raw stat columns (`rushing_yards`, `receiving_yards`,
  `passing_yards`) still going negative for near-zero-volume players (34/15/2
  rows on a real board) and `weekly_projections.Model Proj Pts` going
  slightly negative for a near-zero-volume passer whose small INT-rate
  subtraction outweighed an otherwise-empty line. Same class of small-sample
  artifact, same fix, one level lower than before.
- **Weekly Rankings: a rank column per projection source** (`ui.tabs.
  rankings._positional_rank_col`) - Model Rank/Market Rank/FantasyPros Rank
  (all positional, "RB4"-style) sit side by side now, not just the model's
  own rank as before. Also, when a real FantasyPros weekly CSV export has
  been uploaded this session, the table gains **FantasyPros ECR** (their
  actual published consensus rank, via the existing `parse_fantasypros_
  upload` -> `build_rankings_comparison` path, re-ranked within the matched
  subset the same way the VORP-vs-FantasyPros comparison elsewhere in this
  app already works) and **Model vs FantasyPros ECR** (the delta) - "does
  our derived rank actually match FantasyPros' real rank," which is a
  different question than "does it match a rank WE derived from their own
  points projection," since a published ECR folds in analyst judgment a
  bare stat-line projection doesn't capture. Read via an early
  `st.session_state.get(...)` on the uploader's key so an upload from
  earlier in the session is picked up without moving the uploader widget
  itself (reading, not assigning, a keyed widget's session-state value
  before it's instantiated this run is always legal - see gotcha #1).
- **2026 selectable everywhere a season selector exists** - two remaining
  gaps found and fixed: `ui/tabs/matchup_analyzer.py` and
  `ui/tabs/risers.py` were still importing the season list WITHOUT the
  upcoming year (`config.AVAILABLE_SEASONS`, not `_WITH_UPCOMING`) and
  defaulted to index 0 on that shorter list - swapped to
  `AVAILABLE_SEASONS_WITH_UPCOMING` with `index=1` (not 0) to keep the same
  effective default (the most recent COMPLETED season), matching the
  convention every other season-gated tab already uses. Both tabs already
  had a graceful `if ...empty: st.info(...)` fallback for a season with no
  real data, so no new empty-state handling was needed - verified live
  (Matchup Analyzer resolves to "Pick a player above to build the
  matchup," Risers/Waiver Wire to "Not enough weekly data yet this season,"
  neither crashes) once 2026 became selectable. Every other season-gated
  tab (Depth Charts, Defensive Yield, Player Search, Player Compare, Draft
  HQ, Game Slate, Rookie Watch, Weekly Rankings) already used the
  `_WITH_UPCOMING` list; Live Odds has no season concept at all.

---

**August 2026 pass — full-app QA sweep (Draft HQ rerun cost, verification).**
Not a feature pass: worked through every tab checking calculations, UI/hover
consistency, rough/placeholder text, and crash safety, per an explicit
"make sure nothing crashes" request. Verification used every layer this repo's
own section 6 describes - `pytest tests/` (228 tests), a `streamlit.testing.v1
.AppTest` sweep of all 9 tabs (zero exceptions), and a real headless-Chromium
Playwright session against a live `streamlit run` server exercising an actual
mock draft (mode switch, `New mock`, four `Auto-pick` commits) with console/
page-error capture. Result: no rough/unfinished-looking UI text, no bare
`except:` or unguarded `.iloc[0]` patterns found outside already-guarded
call sites, hover language already consistent app-wide (one shared
`inject_theme()`), and zero crashes anywhere in the sweep.

- **Found and fixed the real cause of "Draft HQ feels slow to click around
  in, and making a pick has a long load"** - measured, not eyeballed.
  `ui.tabs.draft_hq.render()` calls `_load_board(settings)` on EVERY rerun
  (every button, every position-filter click, every pick commit - the
  comment above that call already says so). The big assembled board
  (`_cached_board`) really was cache-fast, but `_load_board` also calls
  `data.draft_sources.build_ecr_board(ecr_raw, board_format)` directly,
  UNCACHED, every single time - seven filter passes plus concats/groupbys
  over the stacked FantasyPros ECR table, measured at **~117ms per call**
  on real data, on top of `load_ecr_raw()` (which WAS already cached).
  Added `@st.cache_data` to `build_ecr_board` - safe because `ecr_raw` is a
  real, hashed parameter (not underscore-prefixed), so a genuinely different
  input still correctly busts the cache; verified against
  `tests/test_draft_sources.py` and the full suite. Measured effect:
  117ms -> 12-20ms per call, `_load_board`'s warm-path rerun cost down from
  125-180ms to 85-105ms. Also tried caching `data.odds_market
  .team_scoring_environment`/`estimate_full_season_scoring` the same way
  (11ms and 54-117ms respectively, same "redone every rerun for no reason"
  shape) and **reverted both** - `tests/test_odds_market.py` monkeypatches
  `fetch_game_lines` to exercise different scenarios under the same
  `season=2025`, and caching on `season` alone silently served one test's
  monkeypatched result to the next. Same class of bug as gotcha #2
  (a real dependency hidden from the cache key), just via a mocked global
  instead of an underscore-prefixed param - caught by the full test suite
  going 3 red before this was caught and reverted, not by inspection. See
  each function's own docstring for the specifics.
- **The "switching Live/Mock draft mode navigates away from the tab" scare
  was a false alarm from `streamlit.testing.v1.AppTest`, not a real bug** -
  worth recording since it cost real time to run down. AppTest's bare-script
  reruns showed the top-level active tab reverting to Game Slate after
  setting the mode radio's value; a real Playwright browser against a live
  server, clicking the actual rendered label rather than driving the widget
  through AppTest's API, stayed on Draft HQ every time. Trust the live
  browser over AppTest for anything involving nested `st.tabs()` state -
  HANDOFF's own section 6 already says as much for glide-data-grid clicks,
  and this is the same lesson for a different widget.
- **Also chased down what looked like a "pick didn't save to my roster"
  bug and it wasn't one.** Firing `Auto-pick` four times back-to-back in a
  paced ("Fast 0.25s") mock left the roster panel empty in one Playwright
  run - alarming, since that would mean picks silently not counting. Root
  cause was the test script, not the app: `_tick_mock_draft` correctly
  gates the bot ticker on `dc['on_clock_me']` and stops exactly on the
  user's turn (`ui.tabs.draft_hq` line ~2608), so clicking `Auto-pick`
  while it ISN'T actually your turn is a correct, silent no-op
  (`autopick_for_user` returns `None`) - my rapid clicks were mostly
  landing in those no-op windows. A slower, deliberate re-run (one click,
  wait for the room to resettle, screenshot) showed the pick landing in the
  roster correctly every time. Recorded so a future session doesn't
  re-chase the same false lead.

---

**August 2026 pass — WEEKLY RANKINGS TAB + WEEKLY MODEL COMPONENT OVERHAUL.**
Two halves, both scoped to the Weekly Rankings sub-tab and the model behind
it.

**UI half** (`ui/tabs/rankings.py`, plus small additions to `ui/charts.py`,
`ui/components.py`, `ui/styling.py`, `data/transforms.py`) - five reported
problems, each fixed at the level it actually lives at:

- **Rank columns are a sortable NUMBER carrying a "RB4" label**
  (`ui.tabs.rankings._woven_rank` + a new `label_cols` param on
  `ui.styling.style_plain_dataframe`). st.dataframe's grid sorts on the raw
  Arrow value and only DISPLAYS the Styler's formatted string - confirmed by
  reading the frontend bundle's own sort comparator, not guessed. A string
  rank column therefore sorted "QB10" between "QB1" and "QB2", sorted every
  QB above every RB above every WR, and sorted a MISSING rank to the TOP of
  an ascending sort (the grid reads a null cell as an empty string, and an
  empty string compares below every number - gotcha #5, now confirmed at
  source). All three complaints were one root cause. The encoding is
  `rank * 10 + position_slot`, so ascending order is WOVEN - QB1, RB1, WR1,
  TE1, QB2, ... - and unranked players get a sentinel worse than every real
  rank that renders as an em dash. The table now also OPENS in that order.
- **Position filter is a row of lineup-SLOT buttons** - QB/RB/WR/TE/FLEX/
  SUPERFLEX (`ui.components.position_group_buttons`), styled to read like
  this app's TABS rather than its pill buttons (a slot filter is a view
  switcher, and it sits directly under the real sub-tabs). CSS is scoped to
  the `st-key-posgrp_` prefix, the same containment trick the Game Slate
  cards use. Replaces the multiselect on all three tables on the tab.
- **Column order** per request: Rank, Player, Position, Team, Opponent, the
  three projections side by side, the model's stat line, Market Coverage,
  Injury Status, Last 5 Weeks, then the three source ranks as a comparison
  block. The leading Rank is FantasyPros' when their projection has been
  pulled, the model's otherwise. `L5 Avg FPTS` was dropped from this table
  (superseded by the sparkline; still on the upload-comparison table below).
- **Last 5 Weeks now populates**, and draws a dotted season-average
  reference line under the trend (`ui.charts.sparkline_data_uri`, an inline
  SVG in an `st.column_config.ImageColumn`). It showed nothing because the
  tab opens on the UPCOMING season, which has no weekly stat rows at all;
  the series is now gated to games before the selected week
  (`data.transforms.build_form_series`) and falls back to the prior season,
  captioned. **`st.column_config.LineChartColumn` cannot do this** - its
  whole cell payload is `{values, yAxis, color, graphKind}`, with no
  reference-line option anywhere (verified in the frontend bundle). An
  ImageColumn cell is drawn scaled to the ROW height with width from the
  aspect ratio, so a too-wide SVG silently overflows a "small" column and is
  clipped by the next one - `_SPARK_W/_SPARK_H` are set for that fit.
- Streamlit renders a null numeric cell as a grey literal **"None"** in this
  version - app-wide, including a bare `st.dataframe(df)`, nothing to do
  with the Styler (its display value really is "--"; the grid draws its own
  missing-value placeholder over it). Pre-existing, not from this pass, and
  it is why QB rows show "None" under Tgt/Rec.

**Model half** (`data/weekly_projections.py`, `scripts/eval_weekly_model.py`
and `scripts/fit_weekly_calibration.py`, both new) - full detail and every
number in **`docs/weekly_projections_methodology.md`**, which is the file to
read before touching this model. Orientation only here:

- The model is now a set of **named, individually switchable components**
  (`MODEL_FEATURES` / `DEFAULT_FEATURES`), and every one was accepted or
  rejected on a paired 2024+2025 backtest (8,107 player-weeks). Three
  shipped, three did not. **A component that can't be turned off can't be
  shown to help** - that is the whole point of the switch.
- **Headline, measured**: MAE 4.710 -> 4.422, rank-corr 0.654 -> 0.689,
  better in **26 of 26 weeks**, and it now beats the naive trailing-4-game
  baseline (4.615 / 0.668) that the previous pass honestly reported it was
  BEHIND. Every position improves on both metrics.
- **Essentially all of that is one component, `role_volume`**: the old model
  had no way to tell a backup from a starter, because a backup's per-GAME
  rate is computed over garbage-time appearances and then shrunk toward the
  POSITION's per-game average, which is a starter's workload. Sixteen of the
  25 biggest upgrades it made over a trailing average were backup QBs
  projected 12-17 points. Snap share separates them. Read
  `expected_snap_share`'s docstring before changing anything here - the
  choice between "share of team weeks" and "share when active" is a real
  decision that was settled by measurement, and the wrong one materially
  hurts returning starters.
- **Rejected, with numbers**: `volume_efficiency` (+0.051 MAE, 5 of 26
  weeks), `game_env` (+0.012 at the measured elasticity, +0.006 at half).
  `role_matchup` - the requested slot-vs-wide / receiving-back / QB-ADOT
  role-conditioned matchup - measured exactly NEUTRAL and ships anyway, on
  the same grounds `HISTORY_MATCHUP_CLIP` was kept: it is the mechanism that
  was asked for and it measurably does not hurt. It is not claimed as a win.
- **Wind is the biggest measured effect in the whole study and is
  deliberately unused** (QB 0.880 vs 1.017 at 15+ mph outdoors). nflverse
  populates `wind`/`temp` AFTER a game, not when the schedule publishes, so
  a backtest would consume information the live model can never have. Don't
  "fix" this by wiring the column in.
- `_vectorized_game_script_multiplier` was ~73% of the entire model build
  (3.2s of 4.4s) - it was rebuilt once per position AND per stat, and looped
  a pandas groupby per player. Hoisted and pivoted to numpy: warm build
  2.2s -> 0.9s, byte-identical output. That is what made iterating on the
  components affordable.

Verification: 241 tests pass (13 new), an AppTest sweep of all 9 tabs raises
nothing, cold start (2026 wk1, 915 players), synthetic cold start, in-season
and a 2019 nflreadpy-fallback season all produce sane boards with no
negative stats, and a real headless-Chromium session against a live server
confirmed the woven ordering, the button hover/selected states and both
sparkline lines rendering.

---
