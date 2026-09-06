#!/usr/bin/env bash
# Lane K - alignment/scheme deep dive on the 2022-2025 weekly PFF archive
# (2022/2023 backfilled + reorganised 2026-09-04).
#
# Phase A (transfer analysis) already ran inline - .sweeps/defense_split_
# transfer.csv. This lane does the model-build parts, in decision-priority
# order so the most important answers land first even if the tail is still
# running:
#   K1  are the SHIPPED alignment + scheme flags still right on 4 seasons?
#   K2  prior-season defense blend w0 sweep (early season)
#   K3  does WR scheme scoring help now (4 seasons vs the 2 it failed on)?
#   K4  full-season confirm of the prior blend at Phase-A's w0=0.40
#   K5  scheme/alignment fixed-weight blend sweep (the "lost result")
#
# Waits for Lane J. One heavy build lane at a time; ~3.5 GB peak.
cd /c/NFLScholar
export PYTHONIOENCODING=utf-8
log=.sweeps/laneK.log
echo "Lane K waiting for Lane J $(date)" > "$log"
while ! grep -q "LANE J DONE" .sweeps/laneJ.log 2>/dev/null; do sleep 180; done
echo "Lane J done; Lane K start $(date)" >> "$log"

mg() { for i in $(seq 1 40); do
  f=$(powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory" 2>/dev/null | tr -dc '0-9'); f=$(( ${f:-0}/1024 ))
  [ "$f" -ge 2600 ] && { echo "  mg ok ${f}MB $(date)" >> "$log"; return; }
  echo "  mg wait ${f}MB $i $(date)" >> "$log"; sleep 120; done; }

mg; echo "[K1] ablate v2_pff_alignment_matchup + v2_scheme_matchup  2022-2025 wk3-18 $(date)" >> "$log"
python scripts/reconfirm_alignment_scheme.py --mode ablate --years 2022,2023,2024,2025 --weeks 3-18 \
  > .sweeps/alignment_scheme_ablate_2022-2025.txt 2>&1
echo "  K1 done rc=$? $(date)" >> "$log"

mg; echo "[K2] defense-prior-blend w0 sweep  2023-2025 wk1-8 $(date)" >> "$log"
python scripts/sweep_defense_prior_blend.py --years 2023,2024,2025 --weeks 1-8 \
  > .sweeps/defense_prior_blend_sweep_wk1-8.txt 2>&1
echo "  K2 done rc=$? $(date)" >> "$log"

mg; echo "[K3] WR scheme scoring  2022-2025 wk3-18 $(date)" >> "$log"
python scripts/reconfirm_alignment_scheme.py --mode wr_scheme --years 2022,2023,2024,2025 --weeks 3-18 \
  > .sweeps/wr_scheme_scoring_2022-2025.txt 2>&1
echo "  K3 done rc=$? $(date)" >> "$log"

mg; echo "[K4] prior-blend full-season confirm scheme w0=0.40  2023-2025 wk3-18 $(date)" >> "$log"
python scripts/sweep_defense_prior_blend.py --years 2023,2024,2025 --weeks 3-18 --scheme-only 0.40 \
  > .sweeps/defense_prior_blend_confirm_wk3-18.txt 2>&1
echo "  K4 done rc=$? $(date)" >> "$log"

mg; echo "[K5] scheme/alignment fixed-weight blend sweep  2023-2025 wk4-14 $(date)" >> "$log"
python scripts/reconfirm_alignment_scheme.py --mode blend --years 2023,2024,2025 --weeks 4-14 \
  > .sweeps/scheme_alignment_blend_sweep_2023-2025.txt 2>&1
echo "  K5 done rc=$? $(date)" >> "$log"

echo "LANE K DONE $(date)" >> "$log"
