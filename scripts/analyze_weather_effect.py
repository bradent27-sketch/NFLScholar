"""Measure the wind / temperature effect on fantasy output, per position.

The module already documents a 2019-2023 wind study (QB 0.880 at 15+ mph vs
1.017 in calm air, TE 0.907, WR 0.895, RB ~unaffected) that is DELIBERATELY
UNUSED because nflverse only fills `wind`/`temp` after kickoff - a backtest
would consume future information the live model never has. The user wants a
real forecast feed wired in (data/weather.py, Open-Meteo), which makes the
effect usable; this script re-measures it on a longer window and splits out
temperature so the flag's multipliers are current, not copied from a comment.

Method (identical to the VENUE_MULT study): each player-game's fantasy points
as a ratio to THAT player's own season average, players with >=6 games and a
real scoring baseline only, so a bad team in a windy city doesn't masquerade
as weather. Outdoor games only (roof in {outdoors, open}); the dome games are
the clean-air control the VENUE_MULT already handles.

    python scripts/analyze_weather_effect.py --years 2015-2025
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
from data.transforms import load_and_merge_data  # noqa: E402

POS = ("QB", "RB", "WR", "TE")
WIND_BUCKETS = [(-1, 8, "calm <=8"), (8, 12, "8-12"), (12, 15, "12-15"),
                (15, 20, "15-20"), (20, 99, "20+")]
TEMP_BUCKETS = [(-99, 20, "<=20"), (20, 32, "20-32"), (32, 45, "32-45"),
                (45, 65, "45-65"), (65, 80, "65-80"), (80, 200, "80+")]


def _player_game_ratios(years):
    frames = []
    for y in years:
        df, _tc, ncol, _ = load_and_merge_data(y, "Full PPR")
        d = df[[ncol, "position", "week", "recent_team" if "recent_team" in df.columns else "team",
                "opponent_team", "fantasy_points_ppr"]].copy()
        d.columns = ["player", "position", "week", "team", "opp", "fp"]
        d["position"] = d["position"].astype(str).str.upper()
        d = d[d["position"].isin(POS)]
        d["week"] = pd.to_numeric(d["week"], errors="coerce")
        d["fp"] = pd.to_numeric(d["fp"], errors="coerce")
        d = d.dropna(subset=["week", "fp"])
        d = d[(d["week"] >= 1) & (d["week"] <= 18)]
        d["season"] = y
        # per-player season baseline (>=6 games, mean fp > 3 so a deep reserve
        # doesn't create 10x ratios off a 0.3-point average)
        g = d.groupby("player")["fp"].agg(["mean", "count"])
        keep = g[(g["count"] >= 6) & (g["mean"] > 3.0)].index
        d = d[d["player"].isin(keep)]
        d = d.merge(g["mean"].rename("base"), left_on="player", right_index=True)
        d["ratio"] = d["fp"] / d["base"]
        frames.append(d)
    allg = pd.concat(frames, ignore_index=True)

    sched = nfl.load_schedules(seasons=years).to_pandas()
    sched = sched[sched["game_type"] == "REG"]
    long = pd.concat([
        sched[["season", "week", "home_team", "roof", "temp", "wind"]].rename(columns={"home_team": "team"}),
        sched[["season", "week", "away_team", "roof", "temp", "wind"]].rename(columns={"away_team": "team"}),
    ], ignore_index=True)
    long["week"] = pd.to_numeric(long["week"], errors="coerce")
    m = allg.merge(long, on=["season", "week", "team"], how="left")
    m["outdoor"] = m["roof"].astype(str).str.lower().isin(["outdoors", "open"])
    return m


def _bucket_report(df, col, buckets, label):
    print(f"\n--- {label} (outdoor games, ratio to player's own season avg) ---")
    print(f"{'bucket':<12}{'':>4}" + "".join(f"{p:>16}" for p in POS))
    ref = {}
    for lo, hi, name in buckets:
        sub = df[(df[col] > lo) & (df[col] <= hi)]
        cells = []
        for p in POS:
            s = sub[sub["position"] == p]["ratio"].dropna()
            if len(s) < 40:
                cells.append(f"  n={len(s):<4}       ")
            else:
                cells.append(f" {s.mean():.3f} (n={len(s):>4})")
                ref.setdefault(p, {})[name] = (s.mean(), len(s))
        print(f"{name:<16}" + "".join(cells))
    return ref


def _dist_report(df, buckets):
    """Per-position ratio DISTRIBUTION by wind bucket - the mean alone hides
    that wind compresses QB from both ends (a real, exploitable shift) but
    barely dents the WR boom tail (so a mean penalty on WR lowers the target
    without improving MAE - the 2026-08-31 backtest finding)."""
    o = df[df["wind"].notna()]
    for pos in ("QB", "WR", "TE"):
        print(f"\n--- {pos} ratio distribution by wind ---")
        print(f"{'bucket':<10}{'n':>6}{'mean':>8}{'median':>8}{'p25':>7}{'p75':>7}"
              f"{'sd':>7}{'bust<.6':>9}{'boom>1.5':>10}")
        for lo, hi, name in buckets:
            s = o[(o.wind > lo) & (o.wind <= hi) & (o.position == pos)]["ratio"].dropna()
            if len(s) < 40:
                continue
            print(f"{name:<10}{len(s):>6}{s.mean():>8.3f}{s.median():>8.3f}"
                  f"{s.quantile(.25):>7.3f}{s.quantile(.75):>7.3f}{s.std():>7.3f}"
                  f"{(s < 0.6).mean() * 100:>8.1f}%{(s > 1.5).mean() * 100:>9.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2015-2025")
    args = ap.parse_args()
    if "-" in args.years:
        lo, hi = args.years.split("-")
        years = list(range(int(lo), int(hi) + 1))
    else:
        years = [int(y) for y in args.years.split(",")]

    m = _player_game_ratios(years)
    outdoor = m[m["outdoor"] & m["wind"].notna() & m["temp"].notna()]
    print(f"years {years[0]}-{years[-1]}: {len(m)} player-games, "
          f"{len(outdoor)} outdoor-with-weather, "
          f"{outdoor['week'].nunique()} weeks, {outdoor['season'].nunique()} seasons")

    wind_ref = _bucket_report(outdoor, "wind", WIND_BUCKETS, "WIND")
    temp_ref = _bucket_report(outdoor, "temp", TEMP_BUCKETS, "TEMPERATURE")
    _dist_report(outdoor, WIND_BUCKETS)

    # temp effect isolated to non-windy games so the two don't confound
    calm = outdoor[outdoor["wind"] <= 10]
    _bucket_report(calm, "temp", TEMP_BUCKETS, "TEMPERATURE (wind<=10 only)")
    # wind effect isolated to mild temps
    mild = outdoor[(outdoor["temp"] >= 40) & (outdoor["temp"] <= 75)]
    _bucket_report(mild, "wind", WIND_BUCKETS, "WIND (temp 40-75 only)")

    print("\n\n=== SUGGESTED MULTIPLIERS (relative to the calm/mild baseline) ===")
    for p in POS:
        base = wind_ref.get(p, {}).get("calm <=8", (1.0, 0))[0]
        if not base:
            continue
        parts = []
        for _, _, name in WIND_BUCKETS:
            v = wind_ref.get(p, {}).get(name)
            if v:
                parts.append(f"{name}={v[0] / base:.3f}")
        print(f"  {p} wind (vs calm): " + "  ".join(parts))
    for p in POS:
        base = temp_ref.get(p, {}).get("45-65", (1.0, 0))[0]
        if not base:
            continue
        parts = []
        for _, _, name in TEMP_BUCKETS:
            v = temp_ref.get(p, {}).get(name)
            if v:
                parts.append(f"{name}={v[0] / base:.3f}")
        print(f"  {p} temp (vs 45-65): " + "  ".join(parts))


if __name__ == "__main__":
    main()
