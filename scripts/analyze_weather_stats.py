"""Per-STAT weather effect: what wind and temperature actually do to each
projected stat, per position, so a weather adjustment can REDISTRIBUTE
(pass down / rush up, arm down / legs up) instead of scaling fantasy points
flat - which is why the flat v1 came back negligible.

Method: each player-game's stat as a ratio to THAT player's own season mean
(players with >=6 games and a non-trivial season mean for the stat), outdoor
games only (roof in {outdoors, open}). Wind in mph buckets + an OLS slope per
10 mph; temperature in the user's buckets (<30 / 30-50 / 50-75 / >75 F).

    python scripts/analyze_weather_stats.py --years 2015-2025
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import nflreadpy as nfl  # noqa: E402

WIND_BUCKETS = [(-1, 8, "<=8"), (8, 12, "8-12"), (12, 15, "12-15"), (15, 20, "15-20"), (20, 99, "20+")]
TEMP_BUCKETS = [(-99, 30, "<30"), (30, 50, "30-50"), (50, 75, "50-75"), (75, 200, ">75")]

# position -> {stat column: min season-mean to keep a player}
STATS = {
    "QB": {"attempts": 12, "completions": 8, "passing_yards": 120, "passing_tds": 0.4,
           "passing_interceptions": 0.3, "sacks_suffered": 0.8, "passing_air_yards": 90,
           "carries": 1.5, "rushing_yards": 6},
    "RB": {"carries": 6, "rushing_yards": 25, "rushing_tds": 0.15, "targets": 1.5,
           "receptions": 1.0, "receiving_yards": 8},
    "WR": {"targets": 3, "receptions": 2, "receiving_yards": 22, "receiving_tds": 0.15,
           "receiving_air_yards": 30},
    "TE": {"targets": 2.5, "receptions": 1.8, "receiving_yards": 18, "receiving_tds": 0.12},
}
# derived efficiency ratios (numer, denom) - computed per game then ratio'd
EFF = {
    "QB": [("comp_pct", "completions", "attempts"), ("yds_per_att", "passing_yards", "attempts"),
           ("aDOT", "passing_air_yards", "attempts"), ("sack_rate", "sacks_suffered", "attempts"),
           ("pass_share", "attempts", "_team_plays")],
    "RB": [("yds_per_carry", "rushing_yards", "carries"), ("catch_rate", "receptions", "targets")],
    "WR": [("catch_rate", "receptions", "targets"), ("yds_per_tgt", "receiving_yards", "targets"),
           ("aDOT", "receiving_air_yards", "targets")],
    "TE": [("catch_rate", "receptions", "targets"), ("yds_per_tgt", "receiving_yards", "targets")],
}


def _load(years):
    ps = nfl.load_player_stats(seasons=years).to_pandas()
    ps = ps[ps["season_type"] == "REG"] if "season_type" in ps else ps
    keep = ["player_id", "player_display_name", "position", "season", "week", "team",
            "attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions",
            "sacks_suffered", "passing_air_yards", "carries", "rushing_yards", "rushing_tds",
            "targets", "receptions", "receiving_yards", "receiving_tds", "receiving_air_yards"]
    ps = ps[[c for c in keep if c in ps.columns]].copy()
    for c in ps.columns:
        if c not in ("player_id", "player_display_name", "position", "team"):
            ps[c] = pd.to_numeric(ps[c], errors="coerce")
    ps["position"] = ps["position"].astype(str).str.upper()

    # team plays this game (for pass_share) - sum attempts+carries by team-game
    tp = (ps.groupby(["season", "week", "team"])[["attempts", "carries"]].sum()
          .sum(axis=1).rename("_team_plays").reset_index())
    ps = ps.merge(tp, on=["season", "week", "team"], how="left")

    sch = nfl.load_schedules(seasons=years).to_pandas()
    sch = sch[sch["game_type"] == "REG"]
    wx = pd.concat([
        sch[["season", "week", "home_team", "roof", "temp", "wind"]].rename(columns={"home_team": "team"}),
        sch[["season", "week", "away_team", "roof", "temp", "wind"]].rename(columns={"away_team": "team"}),
    ], ignore_index=True)
    wx["outdoor"] = wx["roof"].astype(str).str.lower().isin(["outdoors", "open"])
    for c in ("temp", "wind"):
        wx[c] = pd.to_numeric(wx[c], errors="coerce")
    m = ps.merge(wx, on=["season", "week", "team"], how="left")
    return m[m["outdoor"] & m["wind"].notna() & m["temp"].notna()].copy()


def _ratio_frame(df, pos, stat, min_mean):
    d = df[df["position"] == pos][["player_id", "season", stat, "wind", "temp"]].dropna(subset=[stat])
    base = d.groupby(["player_id", "season"])[stat].agg(["mean", "count"])
    keep = base[(base["count"] >= 6) & (base["mean"] >= min_mean)].index
    d = d.set_index(["player_id", "season"]).loc[d.set_index(["player_id", "season"]).index.isin(keep)].reset_index()
    d = d.merge(base["mean"].rename("b"), left_on=["player_id", "season"], right_index=True)
    d["r"] = d[stat] / d["b"]
    return d


def _report(d, col, buckets, label):
    print(f"  {label:<16}", end="")
    for lo, hi, name in buckets:
        s = d[(d[col] > lo) & (d[col] <= hi)]["r"].dropna()
        cell = f"{s.mean():.3f}(n{len(s)})" if len(s) >= 30 else f"--(n{len(s)})"
        print(f"{cell:>15}", end="")
    # OLS slope of r on wind, per +10 mph
    dd = d.dropna(subset=[col, "r"])
    if len(dd) > 100 and col == "wind":
        b = np.polyfit(dd[col], dd["r"], 1)[0] * 10
        print(f"   slope/+10mph={b:+.3f}", end="")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2015-2025")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.years.split("-"))
    years = list(range(lo, hi + 1))
    df = _load(years)
    print(f"outdoor player-games with weather, {years[0]}-{years[-1]}: {len(df):,}\n")

    for pos in ("QB", "RB", "WR", "TE"):
        print(f"\n{'=' * 92}\n{pos}\n{'=' * 92}")
        # --- WIND ---
        print("WIND  (ratio to player's own season mean; * eff ratios computed per-game)")
        print(f"  {'stat':<16}" + "".join(f"{n:>15}" for _, _, n in WIND_BUCKETS))
        for stat, mm in STATS[pos].items():
            _report(_ratio_frame(df, pos, stat, mm), "wind", WIND_BUCKETS, stat)
        for name, num, den in EFF[pos]:
            e = df[df["position"] == pos].copy()
            e = e[(e[den].fillna(0) > 0)]
            e["_e"] = e[num] / e[den]
            eb = e.groupby(["player_id", "season"])["_e"].agg(["mean", "count"])
            kk = eb[(eb["count"] >= 6) & (eb["mean"].abs() > 1e-6)].index
            e = e.set_index(["player_id", "season"])
            e = e.loc[e.index.isin(kk)].reset_index().merge(
                eb["mean"].rename("b"), left_on=["player_id", "season"], right_index=True)
            e["r"] = e["_e"] / e["b"]
            _report(e, "wind", WIND_BUCKETS, name + " *")
        # --- TEMPERATURE ---
        print("\nTEMPERATURE")
        print(f"  {'stat':<16}" + "".join(f"{n:>15}" for _, _, n in TEMP_BUCKETS))
        for stat, mm in STATS[pos].items():
            _report(_ratio_frame(df, pos, stat, mm), "temp", TEMP_BUCKETS, stat)


if __name__ == "__main__":
    main()
