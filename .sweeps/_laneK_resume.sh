#!/usr/bin/env bash
# Lane K RESUME (2026-09-05) - K1/K2/K3 already completed cleanly (rc=0,
# verified in laneK.log). K4 was killed a few minutes into its run: it was
# about to confirm the ORIGINAL prediction (scheme w0=0.40, align w0=0),
# which K2's own held-out early-season result just contradicted - the best
# candidate there was scheme w0=0.40 / align w0=0.40, not align=0. Rather
# than hand-pick a single point again, this resumed K4 runs the SAME full
# 7-variant default grid as K2 (no --scheme-only), just on the full-season
# window, which is a strictly more informative confirm.
# Appends to the same laneK.log so the K1-K3 history stays intact, and Lane
# L's own wait-loop (greps laneK.log for "LANE K DONE") still works.
cd /c/NFLScholar
export PYTHONIOENCODING=utf-8
log=.sweeps/laneK.log
echo "Lane K RESUMED from K4 (K1-K3 already done) $(date)" >> "$log"

mg() { for i in $(seq 1 40); do
  f=$(powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory" 2>/dev/null | tr -dc '0-9'); f=$(( ${f:-0}/1024 ))
  [ "$f" -ge 2600 ] && { echo "  mg ok ${f}MB $(date)" >> "$log"; return; }
  echo "  mg wait ${f}MB $i $(date)" >> "$log"; sleep 120; done; }

mg; echo "[K4] prior-blend full-season confirm, FULL GRID (corrected)  2023-2025 wk3-18 $(date)" >> "$log"
python scripts/sweep_defense_prior_blend.py --years 2023,2024,2025 --weeks 3-18 \
  > .sweeps/defense_prior_blend_confirm_wk3-18.txt 2>&1
echo "  K4 done rc=$? $(date)" >> "$log"

mg; echo "[K5] scheme/alignment fixed-weight blend sweep  2023-2025 wk4-14 $(date)" >> "$log"
python scripts/reconfirm_alignment_scheme.py --mode blend --years 2023,2024,2025 --weeks 4-14 \
  > .sweeps/scheme_alignment_blend_sweep_2023-2025.txt 2>&1
echo "  K5 done rc=$? $(date)" >> "$log"

echo "LANE K DONE $(date)" >> "$log"
