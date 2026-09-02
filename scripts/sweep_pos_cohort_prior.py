"""Model backtest sweep of the position x cohort DEFENSE_PRIOR_GAMES table for
v2_coaching_aware_defense_prior.

Each config sets POS_COHORT_PRIOR_GAMES in the env (read fresh by
data.coaching_changes._pos_cohort_prior_games) and runs one paired A/B via
scripts/backtest_component.py: DEFAULT_FEATURES vs DEFAULT_FEATURES +
v2_coaching_aware_defense_prior. Sequential - the box OOMs on concurrent
heavy builds.

The point is to bracket the "middle ground", not test one guess:
  * flat (control, every cell = 12) - must read ~0.000 everywhere.
  * a lambda spread (0.5 / 0.8 / 1.1) interpolating each cell between 12 and
    its fitted value (data/coaching_changes._POS_COHORT_DEFAULTS, capped 4-30),
    so the backtest can find where on that line the model MAE turns over.
  * shape splits: reset-only (move both / hc_only, hold none / dc_only at 12),
    stable-only (the reverse), and a reset-overshoot - to attribute any effect
    to the right side of the table.

Early weeks only (2-9): week 1 is cold-start and skips the defense-prior blend
entirely, and by week ~10 the prior is a minority input regardless of pg - so
this is the window where the table can actually matter.

    python scripts/sweep_pos_cohort_prior.py --years 2020,2021,2022,2023,2024,2025 --weeks 2-9
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from data.coaching_changes import _POS_COHORT_DEFAULTS  # noqa: E402

FLAT = 12.0
CLAMP = (4.0, 30.0)
COHORTS = ("none", "dc_only", "both", "hc_only")
_RESET = {"both", "hc_only"}
SCOPES = ("ALL", "QB", "RB", "WR", "TE", "START-QB", "START-RB", "START-WR", "START-TE")


def _scaled(lam, shape="all", overrides=None):
    """Full 16-cell POS_COHORT_PRIOR_GAMES spec at spread strength ``lam``."""
    cells = []
    for pos, d in _POS_COHORT_DEFAULTS.items():
        for coh in COHORTS:
            fitted = d.get(coh)
            move = (shape == "all" or (shape == "reset" and coh in _RESET)
                    or (shape == "stable" and coh not in _RESET))
            if overrides and (pos, coh) in overrides:
                pg = overrides[(pos, coh)]
            elif fitted is None or not move:
                pg = FLAT
            else:
                pg = float(np.clip(FLAT + lam * (fitted - FLAT), *CLAMP))
            cells.append(f"{pos}:{coh}={pg:g}")
    return ",".join(cells)


# reset-overshoot: push both/hc_only below the fitted floor, mild none bump
_OVERSHOOT = {(p, c): v for p in ("QB", "RB", "WR") for c, v in
              (("none", 15.0), ("dc_only", 12.0), ("both", 5.0), ("hc_only", 5.0))}
_OVERSHOOT.update({("TE", "none"): 12.0, ("TE", "dc_only"): 18.0,
                   ("TE", "both"): 16.0, ("TE", "hc_only"): 12.0})

GRID = [
    ("flat (control)",   _scaled(0.0)),
    ("lambda 0.5",       _scaled(0.5)),
    ("lambda 0.8",       _scaled(0.8)),
    ("lambda 1.1",       _scaled(1.1)),
    ("reset-only l1",    _scaled(1.0, "reset")),
    ("stable-only l1",   _scaled(1.0, "stable")),
    ("reset-overshoot",  _scaled(1.0, overrides=_OVERSHOOT)),
]


def _run(spec, years, weeks):
    env = dict(os.environ, POS_COHORT_PRIOR_GAMES=spec, PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, os.path.join(HERE, "backtest_component.py"),
           "--add", "v2_coaching_aware_defense_prior", "--years", years, "--weeks", weeks]
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                       encoding="utf-8", errors="replace")
    return (p.stdout or "") + "\n" + (p.stderr or "")


def _parse(out):
    res = {}
    for line in out.splitlines():
        m = re.match(r"\s*([A-Z-]+)\s+n=\d+\s+MAE base [\d.]+ vs variant [\d.]+\s+"
                     r"dMAE\(var-base\)\s+([+-][\d.]+).*?(excludes 0|includes 0|too few)", line)
        if m:
            res[m.group(1)] = (float(m.group(2)), m.group(3) == "excludes 0")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2020,2021,2022,2023,2024,2025")
    ap.add_argument("--weeks", default="2-9")
    ap.add_argument("--out", default=os.path.join(ROOT, ".sweeps", "sweep_pos_cohort_prior.txt"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows, full = [], []
    for i, (label, spec) in enumerate(GRID, 1):
        print(f"[{i}/{len(GRID)}] {label} ...", flush=True)
        out = _run(spec, args.years, args.weeks)
        full.append(f"\n{'=' * 88}\n{label}\nPOS_COHORT_PRIOR_GAMES={spec}\n{'=' * 88}\n{out}")
        rows.append((label, _parse(out)))

    hdr = f"{'config':>18}  " + "".join(f"{s:>13}" for s in SCOPES)
    tbl = [hdr, "-" * len(hdr)]
    for label, parsed in rows:
        cells = []
        for s in SCOPES:
            if s in parsed:
                d, sig = parsed[s]
                cells.append(f"{d:+.3f}{'*' if sig else ' '}")
            else:
                cells.append("   -   ")
        tbl.append(f"{label:>18}  " + "".join(f"{c:>13}" for c in cells))
    tbl += ["", "dMAE = variant-base (NEGATIVE = coaching-aware table helps). * = 95% CI excludes 0."]
    text = "\n".join(tbl)
    print("\n" + text)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n\n" + "\n".join(full))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
