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
- The app creates WR and TE player slot/non-slot profiles. TE non-slot is never mislabeled
  inline; RB/FB alignment remains audit-only.
- Defense profiles are built from the same weekly **offensive** reports, mapped through the
  schedule to the defense faced, then aggregated by defense, offensive position, alignment,
  and event. This avoids the incompatible defender `slot_coverage` measurement.

The UI can display a bounded candidate residual for WR/TE targets, receptions, and yards.
It is explanatory only and is currently fixed at a neutral 1.0 scoring multiplier. Alignment
touchdown effects are always neutral. No alignment effect should be enabled until it clears a
predeclared out-of-sample backtest.

## Week 1 and season files

At Week 1 there are no prior weekly reports for the new season. Existing season-level PFF
files can become a player-alignment prior only when a reviewed `season_manifest.csv` confirms
that they are regular-season-only and time-valid. The existing postseason-contaminated totals
remain unavailable by design; they are not silently treated as a clean Week 1 source.

Before saving or sharing any export, confirm that your PFF agreement permits the intended
local use, AI-assisted development, and distribution. Do not push raw PFF files without
explicit redistribution rights.
