"""Wind-bucketed paired backtest for v2_weather_adjustment.

Same paired-pool discipline as scripts/backtest_component.py (DEFAULT_FEATURES
base vs DEFAULT_FEATURES + v2_weather_adjustment, scored on the intersection of
the player pools each week), but the weekly MAE delta is broken out by the
GAME'S WIND - because ~85% of outdoor games are under 12 mph and just dilute
the aggregate. A wind effect that is real at 16-20 mph is invisible in a
whole-slate mean.

Buckets (recorded schedule wind, via data.weather.recorded_game_weather):
    indoor/na | 0-8 | 8-12 | 12-16 | 16-20 | 20+ mph
reported per bucket, and per (bucket x position), with a bootstrap 95% CI on
the pooled per-week dMAE.

Knee/slope come from data.weekly_projections; sweep them with the env hooks
WEATHER_STRENGTH[_<POS>], WEATHER_WIND_KNEE_<POS>, WEATHER_KNEE_SHIFT.

    python scripts/backtest_weather.py --years 2016-2025 --weeks 1-18
    python scripts/backtest_weather.py --years 2019-2024 --weeks 8-18   # wind-heavy window
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from scripts.eval_weekly_model import _metrics, _weighted, _actual_points  # noqa: E402

WIND_BUCKETS = [("indoor/na", None), ("0-8", (0, 8)), ("8-12", (8, 12)),
                ("12-16", (12, 16)), ("16-20", (16, 20)), ("20+", (20, 999))]
POSITIONS = ("QB", "RB", "WR", "TE")
VARIANT = frozenset(DEFAULT_FEATURES | {"v2_weather_adjustment"})


def _wind_by_team(year, week):
    try:
        import nflreadpy as nfl
        from data.weather import recorded_game_weather
        sch = nfl.load_schedules(seasons=[year]).to_pandas()
        wx = recorded_game_weather(sch, week)
        out = {}
        for tm, gw in wx.items():
            if not gw.is_outdoor:
                out[tm] = None
            elif gw.wind_mph is None or not np.isfinite(gw.wind_mph):
                out[tm] = None
            else:
                out[tm] = float(gw.wind_mph)
        return out
    except Exception as e:
        print(f"  {year} w{week}: wind lookup failed ({e})")
        return {}


def _bucket(w):
    # None -> indoor/dome or wind not recorded. NaN -> team wasn't in the
    # week's wind lookup at all (bye, schedule miss): also "no wind", NOT a
    # 20+ mph game. Only a real number >= 20 is "20+".
    if w is None or (isinstance(w, float) and not np.isfinite(w)):
        return "indoor/na"
    for name, rng in WIND_BUCKETS:
        if rng and rng[0] <= w < rng[1]:
            return name
    return "20+" if w >= 20 else "indoor/na"


def _sub(df, pool, players):
    d = df[df["Player"].isin(pool)]
    return d[d["Player"].isin(players)] if players is not None else d


def _pair_metrics(base_df, var_df, actual, mask_players):
    b = base_df[base_df["Player"].isin(mask_players)]
    v = var_df[var_df["Player"].isin(mask_players)]
    if len(b) < 5 or len(v) < 5:
        return None
    mb = _metrics(pd.Series(b["Model Proj Pts"].to_numpy(), index=b["Player"]), actual)
    mv = _metrics(pd.Series(v["Model Proj Pts"].to_numpy(), index=v["Player"]), actual)
    return (mb, mv) if (mb and mv) else None


def _boot_ci(deltas, weights, n=3000, seed=0):
    if len(deltas) < 4:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    d, w = np.asarray(deltas, float), np.asarray(weights, float)
    idx = np.arange(len(d))
    m = np.array([np.average(d[s], weights=w[s])
                  for s in (rng.choice(idx, len(idx), replace=True) for _ in range(n))])
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2016-2025")
    ap.add_argument("--weeks", default="1-18")
    ap.add_argument("--scoring", default="Full PPR")
    args = ap.parse_args()
    y0, y1 = (int(x) for x in args.years.split("-"))
    years = list(range(y0, y1 + 1))
    w0, w1 = (int(x) for x in args.weeks.split("-"))
    weeks = list(range(w0, w1 + 1))
    scoring_col = "fantasy_points_ppr" if args.scoring != "Standard" else "fantasy_points"

    print(f"years={years[0]}-{years[-1]} weeks={weeks[0]}-{weeks[-1]}")
    perpos = " ".join(p + ":" + os.environ.get("WEATHER_STRENGTH_" + p, "-") for p in POSITIONS)
    print("WEATHER_STRENGTH=%s KNEE_SHIFT=%s per-pos[%s]"
          % (os.environ.get("WEATHER_STRENGTH", "1"),
             os.environ.get("WEATHER_KNEE_SHIFT", "0"), perpos))

    # rows[(bucket, scope)] = list of (mb, mv) weekly pairs ; scope in {'ALL','QB',...}
    rows = {}
    for year in years:
        stats_df, _tc, name_col, _ = load_and_merge_data(year, args.scoring)
        if "week" not in stats_df.columns:
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                continue
            base, bmeta = build_weekly_projections(year, week, args.scoring, as_of_week=week,
                                                   apply_injury=False, features=DEFAULT_FEATURES)
            var, vmeta = build_weekly_projections(year, week, args.scoring, as_of_week=week,
                                                  apply_injury=False, features=VARIANT)
            if base.empty or var.empty:
                continue
            pool = sorted(set(base["Player"]) & set(var["Player"]))
            if len(pool) < 20:
                continue
            base = base[base["Player"].isin(pool)].copy()
            var = var[var["Player"].isin(pool)].copy()
            wind = _wind_by_team(year, week)
            base["_wind"] = base["Team"].astype(str).map(wind)
            base["_bkt"] = base["_wind"].map(_bucket)
            bkt_of = dict(zip(base["Player"], base["_bkt"]))
            var["_bkt"] = var["Player"].map(bkt_of)

            for bname, _ in WIND_BUCKETS:
                bplayers = set(base.loc[base["_bkt"] == bname, "Player"])
                if len(bplayers) < 8:
                    continue
                pm = _pair_metrics(base, var, actual, bplayers & set(actual.index))
                if pm:
                    rows.setdefault((bname, "ALL"), []).append(pm)
                for pos in POSITIONS:
                    pp = bplayers & set(base.loc[base["Pos"] == pos, "Player"]) & set(actual.index)
                    pm = _pair_metrics(base, var, actual, pp)
                    if pm:
                        rows.setdefault((bname, pos), []).append(pm)

    print(f"\n{'bucket':<10}{'scope':<7}{'n':>7}{'MAE base':>10}{'MAE var':>10}"
          f"{'dMAE':>9}{'  wks w-l':>10}{'  boot95%CI':>20}")
    print("-" * 84)
    for bname, _ in WIND_BUCKETS:
        for scope in ("ALL",) + POSITIONS:
            pairs = rows.get((bname, scope))
            if not pairs:
                continue
            mb = [p[0] for p in pairs]
            mv = [p[1] for p in pairs]
            n = sum(m["n"] for m in mb)
            mae_b, mae_v = _weighted(mb, "mae"), _weighted(mv, "mae")
            deltas = [b["mae"] - a["mae"] for a, b in [(p[0], p[1]) for p in pairs]]
            wl = sum(1 for a, b in pairs if b["mae"] < a["mae"])
            ll = sum(1 for a, b in pairs if a["mae"] < b["mae"])
            lo, hi = _boot_ci(deltas, [m["n"] for m in mb])
            sig = " *" if (np.isfinite(lo) and (lo > 0 or hi < 0)) else ""
            print(f"{bname:<10}{scope:<7}{n:>7}{mae_b:>10.3f}{mae_v:>10.3f}"
                  f"{mae_v - mae_b:>+9.3f}{f'  {wl}-{ll}':>10}{f'  [{lo:+.3f},{hi:+.3f}]':>20}{sig}")
        print()


if __name__ == "__main__":
    main()
