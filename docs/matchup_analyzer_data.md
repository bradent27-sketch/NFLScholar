# Matchup Analyzer — data assessment

What the tab is built on, what it can't do and why, and where the NFL data
beats the CFB template it was ported from. Written against the real files in
this repo, not against what the sources claim to publish — every "verified"
below means a script read the actual file and counted rows.

---

## 1. What's available, and what each section uses

Everything except the two marked ✱ reads files already on disk. No API key,
no network, and it works on a fresh clone.

| Section | Source | Notes |
|---|---|---|
| Tendency Profile | `stats_player_week_{yr}.csv` + `pff_imports/{yr}/*` via `build_player_snapshot` | Reuses Player Search's own builder — the same percentile on every tab by construction |
| Snap Workload | `snap_counts_{yr}.csv.csv` + PFF `receiving_summary` (routes, route rate) | |
| Route Efficiency | PFF `receiving_summary` (slot/wide/YPRR/ADOT) + `receiving_scheme_2025` (man/zone splits) | **No CFB equivalent** |
| Usage & Role | weekly stats, per-week shares vs real team totals | |
| Game by Game | weekly stats | 8 stats for QB, 8 for RB, 6 for WR, 5 for TE |
| Matchup Curves | weekly stats + `load_schedule` | Projections, labelled as such |
| Positional Vulnerability | `build_points_allowed_matrix` | Existing function, reused |
| Coverage | `sharp_coverage_schemes_2025.csv`, `external_data/sharp_positional_coverage_2025.csv`, PFF `defense_coverage_scheme_2025` | Three sources, three key formats |
| Red Zone Defense ✱ | nflverse play-by-play (`load_pbp`) | Opt-in button — see §3 |
| Run Defense | PFF `run_defense_summary_2025` + SumerSports `def_overview` | |
| Allowed to {POS} | weekly stats grouped by `opponent_team` | Plus a last-6-games trend |
| Scheme Fit | PFF `rushing_summary` (gap/zone) or `receiving_summary` (slot/wide) | |
| Anytime TD | weekly stats | Poisson, opponent-adjusted |
| Compare Board | all of the above | Up to 3 players, one defense |
| Prop Analysis | weekly stats; live lines ✱ from The Odds API | Manual line entry needs no key |

Season coverage is 2019–2025 for weekly stats, rosters and snap counts;
PFF exports are foldered per year back to 2019; the external Sharp and
SumerSports CSVs are **2025 only** and are used unversioned, so Coverage and
Run Defense show 2025 numbers regardless of the season picker. That's a real
wart — see §4.

---

## 2. Where the NFL data is better than the CFB template

The port isn't a downgrade. Three sections here are genuinely richer than
the college build, all from files already committed:

1. **Man/zone receiving splits per player.** PFF's `receiving_scheme` gives
   YPRR, catch rate, ADOT and target share *separately against man and
   against zone*, for every qualifying receiver. The CFB build has no
   equivalent, so its "coverage tendency" section can only describe the
   defense. Here the two halves meet: a defense that plays 74.7% zone next
   to a receiver whose YPRR is 2.52 vs zone and 1.21 vs man is an actual
   read, not a hint.
2. **Alignment-level yards per target allowed.** Sharp publishes outside vs
   slot separately, which is what makes Scheme Fit a like-for-like
   comparison for WR/TE rather than a proxy — both sides are measured on
   alignment.
3. **Snap-weighted team coverage grades.** PFF grades every defender in
   coverage with their snap counts, so the team number can be weighted
   properly instead of averaged. A plain mean lets a 40-snap nickel corner
   count the same as a 600-snap starter.

Two NFL-specific additions worth having later that aren't wired in yet:
QB pressure data (`pass_rush_summary_2025` has PRP, pressure counts and
true-pass-set splits) and O-line protection (`load_pfr_pass_block`) — both
would sharpen a QB matchup specifically, which is currently the thinnest
position on this tab.

---

## 3. Gaps, stated plainly

**No defense-side gap/zone run split.** PFF publishes `gap_attempts` /
`zone_attempts` on the *rusher*, never on the defense faced. A defensive
number could be manufactured by weighting each defense's allowed production
by how gap-heavy its opponents happened to be, but that's a schedule
artefact wearing a scheme label and it would render as authoritatively as a
measured one. So the defense side reports what's real (grade, stop rate,
missed tackles, EPA allowed) and the rusher's own gap/zone split is shown on
the player side, where it *is* measured. Scheme Fit puts them next to each
other rather than fusing them into one score.

**Red zone needs a live pull.** "Did a drive that reached the 20 finish in
the end zone" needs field position per play, which no local export carries.
It's therefore nflverse play-by-play — a multi-megabyte download — and it's
behind a button so it can't make an otherwise sub-second tab slow. It's
cached per session and is already warm if Player Search's red-zone usage
section has been opened. **This section could not be verified against real
data during the build** (the sandbox had no route to nflverse); its logic is
tested against a fixture in `tests/test_matchup_signals.py`, and the first
real use should be treated as the spot-check.

**No joint alignment × coverage distribution.** Alignment and coverage
faced come from two independent PFF exports with no cross-tab, so there is
no "YPRR from the slot against zone" number. Multiplying the two margins
would invent one. The UI shows them as two panels and says so.

**Thin pools get no percentile at all.** `_percentile_of` returns `None`
below 10 samples rather than a number, because a percentile computed from
four samples renders identically to one computed from four hundred.

---

## 4. Known wart: the external CSVs are 2025-only

`sharp_coverage_schemes_2025.csv`, `external_data/sharp_positional_coverage_2025.csv`
and the six SumerSports files are single-season snapshots with the year in
the filename, and their loaders take no year argument. Pick 2023 and the
Coverage and Run Defense panels still show 2025 numbers.

This is pre-existing — Defensive Yield's Coverage Correlator has the same
behaviour — so it wasn't changed as part of this pass. The fix is to fold
these into a per-year folder the way `pff_imports/{year}/` already works and
give the loaders a `year` argument; that touches Defensive Yield too, which
is why it's flagged here rather than done quietly.

---

## 5. Performance

Measured under `streamlit.testing.v1.AppTest` against the real 2025 files:

| Action | Time |
|---|---|
| First render of the whole app | 4.5 s (shared season load, not this tab) |
| Selecting a player, all 16 sections | **0.4 – 1.5 s** |
| Switching defense | < 1 s |

Every heavy call is an existing `@st.cache_data` loader
(`load_and_merge_data`, `load_all_pff_data`, `build_points_allowed_matrix`,
`precompute_league_percentiles`), so the second interaction onward is cache
hits. Nothing new was added to the app's startup path — the tab only
computes when it's the open tab, per `app.py`'s existing `_tab.open` gate.

---

## 6. Testing

`tests/test_matchup_signals.py` — 21 cases against hand-built fixtures, no
network. The ones that earn their keep: a real zero is kept as a game;
postseason rows are excluded; duplicate week rows collapse; games faced
counts weeks not player rows; usage share is per-week not season-over-season;
softness percentiles point the softer-is-higher way; an unplayed game has no
margin rather than a 0; red zone counts drives not plays; an unreachable
play-by-play pull reports unavailable rather than a defense that allows
nothing; a thin pool gets no percentile.

One of these caught a real error during the build: the touchdown docstring
claimed Poisson gives a higher number than the empirical hit rate. It gives
a lower one (39.3% vs 41.7% on the fixture). The reasoning was rewritten and
the number pinned.
