# Ourlads inbox

Drop a fresh **saved Ourlads printer-friendly page for each team** here
(`.mhtml` / `.mht` / `.html`), then import — this is a local-file workflow, the
app never contacts Ourlads.

## To update the depth charts

1. Save one page per team from Ourlads (File > Save Page As > "Web Page,
   Single File (.mhtml)"). Any filename is fine; the parser reads the team
   from the page. Put all 32 in this folder.
2. Import them one of two ways:
   - **In the app:** Depth Charts tab -> **"Import from ourlads_inbox/"**.
     Clears the weekly-projection cache and reloads automatically.
   - **CLI:** `python scripts/import_ourlads.py`
     (add `--year 2026` to force a season). Then hit "Clear cache" or
     restart the app so a running session picks up the change.
3. Rebuild the board (Weekly Rankings -> Build board). The import panel lists
   any teams still missing so you can confirm all 32 before Week 1.

## What gets kept

Every import first copies the OUTGOING `external_data/ourlads_depth_charts.csv`
and the raw pages it consumed into a timestamped folder under
`external_data/ourlads_archive/import_<date>/`, so a past-week analysis can pin
the exact chart the model saw. The CLI also moves the processed files out of
this inbox into that archive folder, leaving the inbox empty for next time.

The live snapshot the model reads is always `external_data/ourlads_depth_charts.csv`.
