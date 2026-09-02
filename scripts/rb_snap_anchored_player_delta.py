"""Per-player RB delta from v2_rb_snap_anchored_volume on one board.

Builds a single (year, week) board twice - flag OFF vs flag ON at the current
RB_VOL_TILT_STRENGTH / RB_VOL_VACANCY_STRENGTH (env-overridable) - and prints
every RB whose projection or opportunity moves, largest |delta| first. This is
the eyeball companion to scripts/sweep_rb_snap_anchored.py: the aggregate MAE
grid says whether the flag helps on average; this says WHICH players it moves
and by how much, so a human can spot the team-changer lead backs the flag was
built for.

The historical-Ourlads chart flag is added so a past-season week has a real
chart to read (matches the sweep's base config).

Usage:
    python scripts/rb_snap_anchored_player_delta.py --year 2025 --week 1
    RB_VOL_TILT_STRENGTH=0.5 python scripts/rb_snap_anchored_player_delta.py --year 2025 --week 1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402
from data.rb_role_allocator import RB_VOL_TILT_STRENGTH, RB_VOL_VACANCY_STRENGTH  # noqa: E402

FLAG = "v2_rb_snap_anchored_volume"
CHART = "v2_historical_ourlads"


def _board(year, week, feats):
    board, meta = build_weekly_projections(
        year, week, "Full PPR", as_of_week=week, apply_injury=True, features=frozenset(feats))
    return board, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--week", type=int, default=1)
    ap.add_argument("--min-delta", type=float, default=0.15,
                    help="only show RBs whose |proj delta| >= this")
    args = ap.parse_args()

    off_feats = (set(DEFAULT_FEATURES) | {CHART}) - {FLAG}
    on_feats = set(DEFAULT_FEATURES) | {CHART, FLAG}

    print(f"year={args.year} week={args.week}  "
          f"RB_VOL_TILT_STRENGTH={RB_VOL_TILT_STRENGTH}  "
          f"RB_VOL_VACANCY_STRENGTH={RB_VOL_VACANCY_STRENGTH}")

    off, off_meta = _board(args.year, args.week, off_feats)
    on, on_meta = _board(args.year, args.week, on_feats)
    print(f"cold_start off={off_meta.get('cold_start')} on={on_meta.get('cold_start')}")

    keep = ["Player", "Team", "Model Proj Pts", "Expected Snap Share",
            "rushing_attempts", "targets"]
    o = off[off["Pos"] == "RB"][keep].rename(columns={
        "Model Proj Pts": "proj_off", "Expected Snap Share": "snap_off",
        "rushing_attempts": "car_off", "targets": "tgt_off"})
    n = on[on["Pos"] == "RB"][keep].rename(columns={
        "Model Proj Pts": "proj_on", "Expected Snap Share": "snap_on",
        "rushing_attempts": "car_on", "targets": "tgt_on"})
    m = o.merge(n, on=["Player", "Team"], how="outer")
    for c in ("proj_off", "proj_on", "snap_off", "snap_on", "car_off", "car_on", "tgt_off", "tgt_on"):
        m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0.0)
    m["d_proj"] = m["proj_on"] - m["proj_off"]
    m["d_snap"] = m["snap_on"] - m["snap_off"]
    m["d_car"] = m["car_on"] - m["car_off"]
    m["d_tgt"] = m["tgt_on"] - m["tgt_off"]
    m = m[m["d_proj"].abs() >= args.min_delta].sort_values("d_proj", key=lambda s: s.abs(), ascending=False)

    if m.empty:
        print("\n(no RB moves above threshold)")
        return
    print(f"\n{'player':<24}{'tm':>4}{'proj_off':>10}{'proj_on':>9}{'d_proj':>8}"
          f"{'d_snap':>8}{'d_car':>7}{'d_tgt':>7}")
    for _, r in m.iterrows():
        print(f"{r['Player']:<24}{r['Team']:>4}{r['proj_off']:>10.2f}{r['proj_on']:>9.2f}"
              f"{r['d_proj']:>+8.2f}{r['d_snap']:>+8.3f}{r['d_car']:>+7.2f}{r['d_tgt']:>+7.2f}")
    print(f"\n{len(m)} RBs moved >= {args.min_delta} proj pts")


if __name__ == "__main__":
    main()
