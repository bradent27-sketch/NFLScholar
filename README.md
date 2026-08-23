# NFL Scholar

Fantasy football analytics and matchup intelligence, built with [Streamlit](https://streamlit.io). NFL Scholar is a local-first decision workspace for weekly lineup, waiver, matchup, and draft preparation. Its nine workflows combine sourced NFL and market data with transparent, settings-aware calculations rather than presenting unexplained recommendations.

## Tabs

- **Game Slate** — a schedule-first launchpad for the current slate, with jumps into the relevant player, matchup, and odds views.
- **Player Search** — bio card, league-percentile stat profile, weekly game log, career totals, and a next-game projection blending model output with live sportsbook lines.
- **NFL Depth Charts** — snap-share-driven depth chart synthesizer for all 32 teams, click any player to jump to their profile. It can also import locally saved Ourlads printer-friendly pages as conservative preseason QB1/role evidence; those pages do not replace modeled skill-player snap shares.
- **Defensive Yield Schemes** — fantasy-points-allowed matrix, man/zone coverage tendencies, and a coverage-matchup radar crossing a receiver's man/zone performance against an opposing defense's scheme mix.
- **Live Odds** — game lines and player props via [The Odds API](https://the-odds-api.com/), plus a readout of exactly which sportsbooks your key returns.
- **Player Compare** — two players, one overlaid percentile radar/bar chart, each in their own team color.
- **Matchup Analyzer** — player-versus-defense analysis that pairs role, efficiency, alignment, coverage, and positional-vulnerability signals.
- **Weekly Fantasy** — one workspace with Risers / Waiver Wire, Rookie Watch, and Weekly Rankings sub-tabs; rankings compare the app's model, market projections, and an optional FantasyPros weekly export.
- **Draft HQ** — a full draft board built on its own projection model (rank curves blended with each player's own usage rates, aged and injury-adjusted), valued by VORP/VONA against a league-settings replacement level, blended with market ADP/ECR. Runs live or mock drafts with a draft tracker (Sleeper sync or pasted picks) and simulates the resulting roster to a win total. Optionally pulls season-long sportsbook over/unders, re-scores them under your league settings, and ranks where the board disagrees with the money.

### On the odds adapters

`data/odds_sources.py` makes ordinary HTTPS requests and identifies itself honestly. There is no bot-detection evasion in this project — no patched headless browser, no fingerprint spoofing, no proxy rotation. Where a source declines automated access, the adapter reports it and stops. FanDuel and DraftKings aren't scraped at all; The Odds API already carries them under licence.

### FFA data

Fantasy Football Advice (FFA) data may be supplied through a reviewed, sanitized dataset in the repository or through a future authenticated source adapter. The current code supports a local JSON import; it does not yet fetch FFA data automatically. If an adapter is added, credentials must stay in local/hosted secrets and never be committed. Raw browser HAR files, cookies, tokens, and login material remain out of Git.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or double-click `run_app.bat` on Windows.

The app works out of the box using live data pulls ([nflverse](https://github.com/nflverse) via `nflreadpy`), and picks up local CSV exports automatically when present for faster loads and PFF-graded stats (see `HANDOFF.md` for the full data-source breakdown and every local filename it looks for).

Saved Ourlads printer-friendly `.mhtml`, `.mht`, or `.html` depth-chart pages can be imported from the **NFL Depth Charts** tab. This is a local import, not an automated download; the derived snapshot remains local and Git-ignored. A healthy imported QB can inform a preseason QB1 selection, while listed RB/WR/TE starters supply only a conservative floor when the model has little usable role evidence.

### Optional weekly PFF alignment archive

If your PFF agreement permits local use, save exactly two **league-wide** reports after each regular-season week—`receiving_summary.csv` and `receiving_concept.csv`—under:

```text
pff_imports/{year}/weekly/{week}/receiving_summary.csv
pff_imports/{year}/weekly/{week}/receiving_concept.csv
pff_imports/{year}/weekly/manifest.csv
```

The manifest is recommended and should record the week, regular-season status, export date, schema check, and source confidence. The app reads only reports from weeks before the projected week. It uses the reports for player slot/non-slot evidence and an audit-only, offense-derived defense profile; no PFF alignment residual changes rankings until it passes a separate backtest. See [the archive guide](docs/pff_weekly_alignment_archive.md) for the schema and guardrails. Do not scrape a signed-in PFF session or commit new raw PFF exports to the public repository.

## Deploying

Point [Streamlit Community Cloud](https://streamlit.io/cloud) at this repo with `app.py` as the entrypoint. To enable Live Odds, add your own [The Odds API](https://the-odds-api.com/) key via the tab itself (stored locally, never committed) or Streamlit Cloud's secrets manager.

## Notes on the bundled data

The app can use local PFF and FantasyPros exports alongside openly licensed nflverse
data. Treat subscription exports as licensed source material, not automatically
redistributable repository assets: before adding, sharing, or pushing any raw PFF export,
confirm that the applicable agreement permits the intended local use, AI-assisted
development, and distribution. Do not scrape signed-in pages, automate downloads, commit
cookies/tokens/browser captures, or push new raw subscription data to a public repository
without explicit rights. Curated FFA snapshots may be versioned only when their source and
redistribution terms have been reviewed; always record source and refresh date.

## More

`HANDOFF.md` is a detailed engineering handoff doc - architecture, every data source and its quirks, and a running list of real bugs hit and fixed along the way. It also defines the verification standard for model and Player Search changes. Start there for anything beyond a quick read of the code.
