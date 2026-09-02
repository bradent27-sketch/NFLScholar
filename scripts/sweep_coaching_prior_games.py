"""Sweep the per-cohort DEFENSE_PRIOR_GAMES for v2_coaching_aware_defense_prior.

Each config is one scripts/backtest_component.py run with COHORT_PRIOR_GAMES
set in the env (read by data/coaching_changes._cohort_prior_games), a paired
A/B of DEFAULT_FEATURES vs DEFAULT_FEATURES + v2_coaching_aware_defense_prior.
Sequential - the box OOMs on concurrent heavy builds.

The v1 flat-cohort config (dc_only=18, both=8) backtested NEUTRAL on 2023-25
(START-RB +0.029). This re-tests on the wider window the Wikipedia coordinator
backfill now supports (2019+), with a few cohort configs so the per-cohort
prior_games can actually be honed rather than guessed from the proxy.

    python scripts/sweep_coaching_prior_games.py --years 2020,2021,2022,2023,2024,2025 --weeks 4-14
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (label, COHORT_PRIOR_GAMES spec). "off" = every cohort at the scalar default
# => the flag is inert, a control that must read exactly 0.000 everywhere.
GRID = [
    ("off",            "dc_only=default,both=default,dc_to_hc=default,hc_only=default"),
    ("v1 18/8/8",      "dc_only=18,both=8,dc_to_hc=8,hc_only=8"),
    ("strong 26/6/4",  "dc_only=26,both=6,dc_to_hc=4,hc_only=6"),
    ("dc_only-only 20", "dc_only=20,both=default,dc_to_hc=default,hc_only=default"),
]
SCOPES = ("ALL", "QB", "RB", "WR", "TE", "START-QB", "START-RB", "START-WR", "START-TE")


def _run(spec, years, weeks):
    env = dict(os.environ, COHORT_PRIOR_GAMES=spec, PYTHONIOENCODING="utf-8")
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
    ap.add_argument("--weeks", default="4-14")
    ap.add_argument("--out", default=os.path.join(ROOT, ".sweeps", "sweep_coaching_prior_games.txt"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows, full = [], []
    for i, (label, spec) in enumerate(GRID, 1):
        print(f"[{i}/{len(GRID)}] {label}  ({spec}) ...", flush=True)
        out = _run(spec, args.years, args.weeks)
        full.append(f"\n{'=' * 78}\n{label}   {spec}\n{'=' * 78}\n{out}")
        rows.append((label, _parse(out)))

    hdr = f"{'config':>16}  " + "".join(f"{s:>13}" for s in SCOPES)
    tbl = [hdr, "-" * len(hdr)]
    for label, parsed in rows:
        cells = []
        for s in SCOPES:
            if s in parsed:
                d, sig = parsed[s]
                cells.append(f"{d:+.3f}{'*' if sig else ' '}")
            else:
                cells.append("   -   ")
        tbl.append(f"{label:>16}  " + "".join(f"{c:>13}" for c in cells))
    tbl += ["", "dMAE = variant-base (NEGATIVE = coaching-aware prior helps). * = CI excludes 0."]
    text = "\n".join(tbl)
    print("\n" + text)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n\n" + "\n".join(full))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
