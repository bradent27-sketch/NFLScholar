# NFL Scholar

Fantasy football analytics and matchup intelligence, built with [Streamlit](https://streamlit.io). Nine tabs covering player search, depth charts, defensive coverage matchups, waiver-wire risers, rookie tracking, weekly rankings, a VORP-based draft sheet, live sportsbook odds, and head-to-head player comparisons.

## Tabs

- **Player Search** — bio card, league-percentile stat profile, weekly game log, career totals, and a next-game projection blending model output with live sportsbook lines.
- **NFL Depth Charts** — snap-share-driven depth chart synthesizer for all 32 teams, click any player to jump to their profile.
- **Defensive Yield Schemes** — fantasy-points-allowed matrix, man/zone coverage tendencies, and a coverage-matchup radar crossing a receiver's man/zone performance against an opposing defense's scheme mix.
- **Risers / Waiver Wire** — biggest week-over-week percentile jumps.
- **Rookie Watch** — rookie performance leaderboard.
- **Weekly Rankings** — upload your own FantasyPros export and compare against this app's own recent-form ranking.
- **VORP Draft Sheet** — value-over-replacement draft board with a configurable scoring/roster model, plus a live draft tracker (Sleeper sync or pasted picks).
- **Live Odds** — game lines and player props via [The Odds API](https://the-odds-api.com/).
- **Player Compare** — two players, one overlaid percentile radar/bar chart, each in their own team color.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or double-click `run_app.bat` on Windows.

The app works out of the box using live data pulls ([nflverse](https://github.com/nflverse) via `nflreadpy`), and picks up local CSV exports automatically when present for faster loads and PFF-graded stats (see `HANDOFF.md` for the full data-source breakdown and every local filename it looks for).

## Deploying

Point [Streamlit Community Cloud](https://streamlit.io/cloud) at this repo with `app.py` as the entrypoint. To enable Live Odds, add your own [The Odds API](https://the-odds-api.com/) key via the tab itself (stored locally, never committed) or Streamlit Cloud's secrets manager.

## Notes on the bundled data

This repo includes local exports from PFF and FantasyPros alongside the openly-licensed nflverse data, so the app runs fully featured without any setup. Those first two are paid, licensed services - if you fork this for your own use, refresh them with your own subscription's exports rather than assuming redistribution rights.

## More

`HANDOFF.md` is a detailed engineering handoff doc - architecture, every data source and its quirks, and a running list of real bugs hit and fixed along the way. Start there for anything beyond a quick read of the code.
