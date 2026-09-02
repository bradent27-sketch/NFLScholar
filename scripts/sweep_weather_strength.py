"""Strength sweep for v2_weather_adjustment.

Runs scripts/backtest_component.py once per WEATHER_STRENGTH grid point (env
var, read at import in data/weekly_projections.py), scaling the penalty depth
(0 = inert, 1 = as-measured 2015-2025, >1 = deeper). Sequential - the box
OOMs on concurrent heavy builds.

ABLATION, NOT ADDITION - and that changed on 2026-09-01, when
v2_weather_adjustment shipped into DEFAULT_FEATURES. While it was a candidate
this script used `--add`, which is now WRONG: with the flag already in
DEFAULT_FEATURES, `--add` makes the variant set identical to the base set and
every cell reports exactly 0.000. `--flags` (ablate) is the correct pairing
now: base = shipped model with weather ON at the grid strength, variant =
same model with weather removed. So a POSITIVE dMAE means "removing weather
hurts", i.e. that strength is earning its keep, and the best grid point is
the one with the LARGEST positive dMAE.

Use this only AFTER the base backtest shows v2_weather_adjustment has a real
effect worth tuning; if the base run is flat, the strength sweep will be flat
too.

    python scripts/sweep_weather_strength.py --years 2021,2022,2023,2024,2025 --weeks 3-17
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# (label, QB strength, WR strength, TE strength). The base backtest
# (2021-25 wk3-17) found: RB helped via the QB-volume knock-on, QB leaned
# helpful, WR leaned slightly WORSE. So the grid probes dropping / halving
# WR and boosting QB rather than a single uniform scale.
GRID = [
    ("shipped 1/1/1",  1.0, 1.0, 1.0),
    ("no-WR",          1.0, 0.0, 1.0),
    ("half-WR",        1.0, 0.5, 1.0),
    ("quarter-WR",     1.0, 0.25, 1.0),
    ("no-WR strongQB", 1.3, 0.0, 1.0),
    ("no-WR no-TE",    1.0, 0.0, 0.0),
]
SCOPES = ("ALL", "QB", "RB", "WR", "TE", "START-QB", "START-RB", "START-WR", "START-TE")


def _run(cfg, years, weeks):
    _lbl, q, w, t = cfg
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
               WEATHER_STRENGTH_QB=str(q), WEATHER_STRENGTH_WR=str(w), WEATHER_STRENGTH_TE=str(t))
    cmd = [sys.executable, os.path.join(HERE, "backtest_component.py"),
           "--flags", "v2_weather_adjustment", "--years", years, "--weeks", weeks]
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                       encoding="utf-8", errors="replace")
    return (p.stdout or "") + "\n" + (p.stderr or "")


def _parse(out):
    """scope -> (dMAE, CI-excludes-0?) for the v2_weather_adjustment row."""
    res = {}
    for line in out.splitlines():
        m = re.match(r"\s*([A-Z-]+)\s+n=\d+\s+MAE base [\d.]+ vs variant [\d.]+\s+"
                     r"dMAE\(var-base\)\s+([+-][\d.]+).*?(excludes 0|includes 0|too few)", line)
        if m:
            res[m.group(1)] = (float(m.group(2)), m.group(3) == "excludes 0")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2021,2022,2023,2024,2025")
    ap.add_argument("--weeks", default="3-17")
    ap.add_argument("--out", default=os.path.join(ROOT, ".sweeps", "sweep_weather_strength.txt"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows, full = [], []
    for i, cfg in enumerate(GRID, 1):
        print(f"[{i}/{len(GRID)}] {cfg[0]} QB={cfg[1]} WR={cfg[2]} TE={cfg[3]} ...", flush=True)
        out = _run(cfg, args.years, args.weeks)
        full.append(f"\n{'=' * 78}\n{cfg[0]}  QB={cfg[1]} WR={cfg[2]} TE={cfg[3]}\n{'=' * 78}\n{out}")
        rows.append((cfg, _parse(out)))

    hdr = f"{'config':>16}  " + "".join(f"{sc:>13}" for sc in SCOPES)
    tbl = [hdr, "-" * len(hdr)]
    for cfg, parsed in rows:
        cells = []
        for sc in SCOPES:
            if sc in parsed:
                d, sig = parsed[sc]
                cells.append(f"{d:+.3f}{'*' if sig else ' '}")
            else:
                cells.append("   -   ")
        tbl.append(f"{cfg[0]:>16}  " + "".join(f"{c:>13}" for c in cells))
    tbl += ["", "dMAE = variant-base (NEGATIVE = weather flag helps). * = boot 95% CI excludes 0."]
    text = "\n".join(tbl)
    print("\n" + text)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n\n" + "\n".join(full))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
