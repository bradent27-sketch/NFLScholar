# Weekly PFF alignment archive

This optional local archive supplies time-safe alignment evidence for the experimental
weekly-projection model. It does not fetch PFF, scrape a signed-in account, reuse browser
credentials, or upload any subscription data.

## Per-week layout

Save one league-wide export of each report after the week's games:

```text
pff_imports/{year}/weekly/{week}/receiving_summary.csv
pff_imports/{year}/weekly/{week}/receiving_concept.csv
pff_imports/{year}/weekly/manifest.csv
```

For example:

```text
pff_imports/2026/weekly/1/receiving_summary.csv
pff_imports/2026/weekly/1/receiving_concept.csv
pff_imports/2026/weekly/manifest.csv
```

One pair is league-wide: it covers WR, TE, HB/RB, and FB. Do not create one export per
position or player. The raw weekly folder is intentionally Git-ignored so it cannot be
staged accidentally for a public push.

## Manifest

The manifest is optional for loading, but recommended for auditability. A minimal example:

```csv
week,regular_season,schema_valid,source_confidence,export_date
1,true,true,manual_pff_regular_season_export,2026-09-10
2,true,true,manual_pff_regular_season_export,2026-09-17
```

`regular_season=true` means the reports contain regular-season data for that listed week.
`schema_valid` records your export check; the loader also validates required columns itself.
The source confidence field is displayed in the projection data contract and may be any
plain-language label that identifies the manual export process.

## What the model uses

- A target Week *N* projection can read only archives from Weeks `< N`.
- `receiving_summary` supplies player slot/wide/inline alignment rates and snaps.
- `receiving_concept` supplies slot routes, targets, receptions, yards, and touchdowns.
- The app creates WR and TE player slot/wide/inline profiles from `receiving_summary`'s real
  per-player `slot_snaps`/`wide_snaps`/`inline_snaps`. RB/FB alignment remains audit-only.
- Defense profiles are built from the same weekly **offensive** reports, mapped through the
  schedule to the defense faced, then aggregated by defense, offensive position, alignment,
  and event. This avoids the incompatible defender `slot_coverage` measurement. Alongside the
  original slot/non-slot split, the defense side (added 2026-08-24) also carries real **wide**
  and **inline** buckets: `receiving_concept.csv` only ever reports a slot event split (no
  separate wide/inline target/reception/yard columns), so each player-row's real non-slot
  production is apportioned between wide and inline in proportion to that row's own real
  wide_snaps/inline_snaps - a labelled approximation, not a second independently measured PFF
  event. A player-row with no reported wide/inline snaps simply contributes no wide/inline row
  rather than guessing a split.

The UI can display a bounded candidate residual for WR/TE targets, receptions, and yards,
shown in each player's projection decomposition under "Alignment mix (slot / wide / inline)".
`alignment_defense_residual_multiplier` blends slot/wide/inline (a real 3-way blend) whenever a
player's own wide/inline rates and the opposing defense's wide/inline comparison evidence are
both available; it falls back to the original slot/non-slot 2-way blend otherwise (e.g. Week 1
of a season, before this app has any 2026 weekly archive of its own - see "Week 1 and season
files" below for the season-prior fallback this needs on BOTH the player and defense side).
This candidate multiplier is `v2_pff_alignment_matchup`-gated (V2 board only, currently rejected
from `DEFAULT_FEATURES` after a 2026-08-24 backtest - see `docs/weekly_projections_methodology.md`)
and, when that feature is active, IS multiplied into the WR/TE targets/receptions/receiving_yards
matchup step - "explanatory only" describes the underlying `pff_alignment.py` helper's own
`multiplier` field (always 1.0), not the `candidate_multiplier` this app's caller uses. Alignment
touchdown effects are always neutral.

## Week 1 and season files

At Week 1 there are no prior weekly reports for the new season. Existing season-level PFF
files can become a player-alignment prior only when a reviewed `season_manifest.csv` confirms
that they are regular-season-only and time-valid. This
repo's own `pff_imports/2025/season_manifest.csv` makes that call for the 2025 season total
(`regular_season=true`) - the same file `data.loaders.load_pff_data_with_fallback` already
trusts elsewhere in this app's default (non-alignment) prior-season blend.

The **defense** side has its own, separate Week 1 gap: `load_weekly_alignment_defense_profiles`
is always keyed to the CURRENT season's own weekly archive, which is empty until games are
actually played. `build_weekly_projections` works around this at cold start (added 2026-08-24)
by loading the full PRIOR season's weekly archive instead (every one of its weeks passes the
as-of-week eligibility check, since it's a year old already), mapped through that SAME prior
season's own schedule - never the new season's schedule, which would misattribute the
offense/defense matchups. This needs no manifest of its own beyond the existing per-week
`manifest.csv` rows the prior season's weekly archive already carries.

## Postseason weeks (added 2026-08-25)

A player's slot/wide/inline tendency is not "contaminated" by playoff opponent quality the way
a box-score production rate is - PFF's own WC/DIV/CONF/SB weekly exports are therefore usable
as SUPPLEMENTARY alignment evidence on top of the regular-season prior, unlike the box-score
season totals above (which stay regular-season-only, unchanged). This is opt-in and off by
default everywhere in `data/pff_alignment.py` - `discover_weekly_alignment_exports`,
`load_weekly_alignment_profiles`, and `load_weekly_alignment_defense_profiles` all take an
`include_postseason=False` kwarg, and `load_season_alignment_prior` takes
`include_postseason_weeks=False` - so every caller's behavior is unchanged unless it explicitly
asks for postseason weeks. Only the two cold-start call sites in `build_weekly_projections`
turn this on. A postseason week is still saved into `weekly/{week}/`, same layout as any other
week, using nflverse's own postseason week numbers (19 Wild Card, 20 Divisional, 21 Conference
Championship, 22 Super Bowl) - its manifest row is marked `regular_season=false` honestly
(a "known postseason source" is never silently relabeled as regular season); the loader itself
decides whether that honest label excludes it (the default) or the caller has opted in.
`load_schedule(year, include_postseason=True)` is the matching opt-in on the schedule side,
needed so the defense-side loader can map a playoff week's games to the actual opponent faced -
`nflreadpy`'s schedule already carries WC/DIV/CON/SB rows under those same week numbers, so no
new schedule source was needed, just an opt-in flag past the existing `game_type == 'REG'`
filter (which stays the default for every other schedule caller, including the SOS tables).

## Season-total fallback to the full weekly archive (added 2026-08-25)

`load_season_alignment_prior`'s season-total file is not always usable as a Week 1 cold-start
player prior - e.g. 2024's own season-total `receiving_summary.csv` was confirmed (by diffing
it against real regular-season-only box scores) to include real POSTSEASON production for every
2024 team that made the playoffs, so it is correctly left with no `season_manifest.csv` rather
than dishonestly marked `regular_season=true`. When the season-total path reports unavailable,
the player-side cold start in `build_weekly_projections` now falls back to aggregating that
prior year's own full WEEKLY archive instead (`load_weekly_alignment_profiles(year - 1, ...)`,
`include_postseason` following the same historical-backtest caution as everywhere else) - the
same source the defense side has always used, and, once a full season of weekly exports exists,
arguably a more precise source anyway (real per-game measurement, not one lump sum). A year with
BOTH a reviewed season-total file and a full weekly archive keeps using the season-total file;
this is a fallback, not a replacement.

## Two-years-back grounding for an in-season read (added 2026-08-25)

Mid-season (not cold start), a player's slot/wide/inline rate for the CURRENT season so far can
optionally be pulled slightly toward his OWN full season two years back (e.g. 2024 grounding a
2026 in-season read) - the same "ground a breakout or down year against last-known-good" request
that already exists for other stats via `prior2_blend_weight`/`QB_PRIOR2_*`, extended to
alignment. `data.pff_alignment.blend_alignment_profile_toward_prior2` does the blending; it is
deliberately SYMMETRIC (no directional dampening) unlike the QB/other-stat version, since a
higher or lower slot/wide/inline share carries no inherent "more fantasy value" direction to
stay bullish about. The pull is sample-size-scaled by `alignment_sample_weight` (real snaps, the
same evidence unit this module already scores confidence by): `ALIGNMENT_PRIOR2_BASE_WEIGHT`
(0.10) even at a full current-season sample, rising to `ALIGNMENT_PRIOR2_MAX_WEIGHT` (0.40) a
few games into the season. Gated behind the SAME `v2_td_two_year_prior` flag every other stat's
2024 grounding already uses (not a new flag), and behind `v2_pff_alignment_matchup` as always;
a prior-year sample under `ALIGNMENT_PRIOR2_MIN_SAMPLE_WEIGHT` (20 snaps) is treated as no read
at all. Visible in the decomposition's role data as `alignment_prior2_weight` /
`alignment_prior2_sample_weight` whenever it actually fired.

Before saving or sharing any export, confirm that your PFF agreement permits the intended
local use, AI-assisted development, and distribution. Do not push raw PFF files without
explicit redistribution rights.
