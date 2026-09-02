"""Strength sweep for v2_rb_snap_anchored_volume's two knobs.

The binary on/off ablation of this flag (docs/overnight_backtest_log_2026-08-30.md
section on `.sweeps/ablate_rb_snap_anchored_wk1.txt`) landed START-RB dMAE
-0.288 on only 2023-25 wk1 (n~3 correlated), and the user judged that too
thin to retire the mechanism - a per-snap carry/target tilt that keeps a
team-changed lead back (Etienne -> NO) tracking his real workload instead of
inheriting his old committee's split. This script sweeps the two 0..1 dials
added 2026-08-30:

  RB_VOL_TILT_STRENGTH     - legacy blend (0) <-> full per-snap tilt (1)
  RB_VOL_VACANCY_STRENGTH  - legacy OUT-top-3 concentration (0) <-> full (1)

Each grid point is one full `eval_weekly_model.py` run (env vars set for the
subprocess) comparing:

    base    = default+v2_historical_ourlads                      (flag OFF)
    variant = default+v2_historical_ourlads+v2_rb_snap_anchored_volume (flag ON)

over wk1 of 2022-2025 - every season the frozen pre-Week-1 Ourlads archive
covers, so the flag always has a real chart to read. Runs are STRICTLY
sequential (the box OOMs on concurrent heavy builds).

Usage:
    python scripts/sweep_rb_snap_anchored.py
    python scripts/sweep_rb_snap_anchored.py --years 2022,2023,2024,2025 --out .sweeps/sweep_rb_snap_anchored_2026-08-30.txt
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

BASE_VARIANT = "default+v2_historical_ourlads"
FLAG_VARIANT = "default+v2_historical_ourlads+v2_rb_snap_anchored_volume"
SCOPES = ("ALL", "RB", "START-RB", "START-WR", "START-TE")

# (tilt, vacancy) grid: a tilt sweep at full vacancy, a vacancy sweep at full
# tilt, plus the all-off corner as a sanity anchor (should reproduce the flag
# being inert).
GRID = [
    (0.0, 0.0),
    (0.0, 1.0),
    (0.33, 1.0),
    (0.66, 1.0),
    (1.0, 1.0),
    (1.0, 0.0),
    (1.0, 0.5),
]


def _run_point(tilt, vac, years, weeks):
    env = dict(os.environ)
    env["RB_VOL_TILT_STRENGTH"] = str(tilt)
    env["RB_VOL_VACANCY_STRENGTH"] = str(vac)
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [
        sys.executable, os.path.join(HERE, "eval_weekly_model.py"),
        "--years", years, "--weeks", weeks,
        "--variants", f"{BASE_VARIANT},{FLAG_VARIANT}",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                          encoding="utf-8", errors="replace")
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def _parse(out):
    """scope -> dict(base_mae, var_mae, dmae, drho, wl) for the flag variant."""
    result = {}
    lines = out.splitlines()
    scope = None
    for line in lines:
        m = re.match(r"^--- (.+) ---\s*$", line)
        if m:
            scope = m.group(1).strip()
            continue
        if scope is None:
            continue
        toks = line.split()
        if not toks:
            continue
        if toks[0] == BASE_VARIANT and len(toks) >= 6:
            try:
                result.setdefault(scope, {})["base_mae"] = float(toks[2])
            except ValueError:
                pass
        elif toks[0] == FLAG_VARIANT and len(toks) >= 9:
            try:
                result.setdefault(scope, {}).update(
                    var_mae=float(toks[2]), dmae=float(toks[-3]),
                    drho=float(toks[-2]), wl=toks[-1])
            except ValueError:
                pass
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2022,2023,2024,2025")
    ap.add_argument("--weeks", default="1-1")
    ap.add_argument("--out", default=os.path.join(ROOT, ".sweeps", "sweep_rb_snap_anchored.txt"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    full_log = []
    grid_rows = []

    for i, (tilt, vac) in enumerate(GRID, 1):
        print(f"[{i}/{len(GRID)}] tilt={tilt} vacancy={vac} ...", flush=True)
        out = _run_point(tilt, vac, args.years, args.weeks)
        full_log.append(f"\n{'=' * 78}\nTILT={tilt}  VACANCY={vac}\n{'=' * 78}\n{out}")
        parsed = _parse(out)
        grid_rows.append((tilt, vac, parsed))

    # Grid table: dMAE (variant - base), negative => flag HELPS at this setting.
    header = f"{'tilt':>5}{'vac':>5}  " + "".join(f"{s:>22}" for s in SCOPES)
    table = [header, "-" * len(header)]
    for tilt, vac, parsed in grid_rows:
        cells = []
        for s in SCOPES:
            d = parsed.get(s, {})
            if "dmae" in d:
                cells.append(f"{d['dmae']:+.3f}/{d.get('wl', '?'):>5}")
            else:
                cells.append("        -")
        table.append(f"{tilt:>5}{vac:>5}  " + "".join(f"{c:>22}" for c in cells))
    table.append("")
    table.append("cell = dMAE(flagON - flagOFF) / week W-L for the flag variant; "
                 "negative dMAE => snap-anchored helps at that (tilt,vacancy).")
    base_line = grid_rows[-1][2].get("START-RB", {}).get("base_mae")
    if base_line is not None:
        table.append(f"reference: flag-OFF START-RB MAE = {base_line:.3f} "
                     f"(env-independent, identical across all rows)")
    grid_text = "\n".join(table)

    print("\n" + grid_text)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(grid_text + "\n\n" + "\n".join(full_log))
    print(f"\nfull logs + grid -> {args.out}")


if __name__ == "__main__":
    main()
