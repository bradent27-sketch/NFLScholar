"""Is the shipped model systematically biased on STARTABLE tight ends?

Every indirect lever we have tried - coaching-aware defense prior, weather,
DEFENSE_PRIOR_GAMES pg 8-10 - nudges START-TE the same (helpful) direction,
which is the signature of a small standing bias in the base top-12 TE
projection that any downward pressure partially corrects. This measures it
directly, with no A/B: build the shipped DEFAULT_FEATURES board per week, take
the startable cut per position, and report mean (pred - actual).

    python scripts/analyze_startable_te_bias.py --years 2021-2025 --weeks 4-17
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

from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from scripts.eval_weekly_model import _actual_points, STARTABLE_N  # noqa: E402

POSITIONS = ("QB", "RB", "WR", "TE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2021-2025")
    ap.add_argument("--weeks", default="4-17")
    ap.add_argument("--scoring", default="Full PPR")
    args = ap.parse_args()
    y0, y1 = (int(x) for x in args.years.split("-"))
    w0, w1 = (int(x) for x in args.weeks.split("-"))
    years, weeks = list(range(y0, y1 + 1)), list(range(w0, w1 + 1))
    scoring_col = "fantasy_points_ppr" if args.scoring != "Standard" else "fantasy_points"

    # recs: one row per scored startable player-week
    recs = []
    for year in years:
        stats_df, _tc, name_col, _ = load_and_merge_data(year, args.scoring)
        if "week" not in stats_df.columns:
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                continue
            board, meta = build_weekly_projections(year, week, args.scoring, as_of_week=week,
                                                   apply_injury=False, features=DEFAULT_FEATURES)
            if board.empty:
                continue
            for pos in POSITIONS:
                cut = board[board["Pos"] == pos].nlargest(STARTABLE_N.get(pos, 30), "Model Proj Pts")
                cut = cut.reset_index(drop=True)
                for rank, r in cut.iterrows():
                    p = r["Player"]
                    if p not in actual.index:
                        continue
                    recs.append({"year": year, "week": week, "pos": pos, "rank": rank + 1,
                                 "pred": float(r["Model Proj Pts"]), "actual": float(actual[p])})
    d = pd.DataFrame(recs)
    if d.empty:
        print("no data")
        return
    d["err"] = d["pred"] - d["actual"]        # + = model OVER-projected

    print(f"shipped DEFAULT_FEATURES, startable cut, {years[0]}-{years[-1]} wk{weeks[0]}-{weeks[-1]}, "
          f"{len(d):,} player-weeks\n")
    print(f"{'pos':<5}{'n':>6}{'mean(pred-act)':>16}{'median':>10}{'MAE':>9}{'mean pred':>11}"
          f"{'mean actual':>13}{'bias %':>9}")
    print("-" * 82)
    for pos in POSITIONS:
        s = d[d["pos"] == pos]
        bias = s["err"].mean()
        print(f"{pos:<5}{len(s):>6}{bias:>+16.3f}{s['err'].median():>+10.3f}{s['err'].abs().mean():>9.3f}"
              f"{s['pred'].mean():>11.2f}{s['actual'].mean():>13.2f}{100 * bias / s['actual'].mean():>+8.1f}%")

    print("\n-- TE by projection rank (is it the elite TEs or the streamers?) --")
    te = d[d["pos"] == "TE"]
    for lo, hi, lab in [(1, 3, "TE1-3"), (4, 6, "TE4-6"), (7, 12, "TE7-12")]:
        s = te[(te["rank"] >= lo) & (te["rank"] <= hi)]
        if len(s):
            print(f"  {lab:<8} n={len(s):>4}  mean(pred-act) {s['err'].mean():>+7.3f}  "
                  f"MAE {s['err'].abs().mean():>6.3f}  mean pred {s['pred'].mean():>6.2f}  "
                  f"mean actual {s['actual'].mean():>6.2f}")

    print("\n-- TE bias by season (is it stable or one bad year?) --")
    for y in years:
        s = te[te["year"] == y]
        if len(s):
            print(f"  {y}  n={len(s):>4}  mean(pred-act) {s['err'].mean():>+7.3f}  "
                  f"MAE {s['err'].abs().mean():>6.3f}")

    print("\n-- TE bias by week bucket --")
    for lo, hi in [(w0, 6), (7, 10), (11, 14), (15, w1)]:
        s = te[(te["week"] >= lo) & (te["week"] <= hi)]
        if len(s):
            print(f"  wk{lo:>2}-{hi:<2} n={len(s):>4}  mean(pred-act) {s['err'].mean():>+7.3f}")


if __name__ == "__main__":
    main()
