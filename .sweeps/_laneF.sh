#!/usr/bin/env bash
# Lane F - scheme/alignment fixed-weight sweep, CROSS-SEASON (2024 + 2025).
#
# 2024's PFF weekly archive was backfilled 2026-09-01, so this is no longer
# a single-season exploratory run - each season is scored independently and a
# weight has to hold its sign in both to count. Slots into whichever of Lane
# D or Lane E frees first; never a third concurrent heavy job.
cd /c/NFLScholar
export PYTHONIOENCODING=utf-8
log=.sweeps/laneF.log
echo "Lane F waiting for a free lane (D or E to finish) $(date)" > "$log"

for i in $(seq 1 400); do
  grep -q "LANE D DONE" .sweeps/laneD.log 2>/dev/null && { echo "  lane D freed $(date)" >> "$log"; break; }
  grep -q "LANE E DONE" .sweeps/laneE.log 2>/dev/null && { echo "  lane E freed $(date)" >> "$log"; break; }
  sleep 180
done

mg() { for i in $(seq 1 60); do
  f=$(powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory" 2>/dev/null | tr -dc '0-9'); f=$(( ${f:-0}/1024 ))
  [ "$f" -ge 3500 ] && { echo "  mg ok ${f}MB $(date)" >> "$log"; return; }
  echo "  mg wait ${f}MB $i $(date)" >> "$log"; sleep 120; done; }

mg; echo "[F1] TE scheme weight sweep 0.6-1.0 (2024+2025 wk2-18)" >> "$log"
python scripts/sweep_scheme_blend_weight2.py --position TE \
  --weights 0.6,0.7,0.8,0.9,1.0 --years 2024,2025 --weeks 2-18 \
  > .sweeps/scheme_blend_TE_v2.txt 2>&1
echo "  done $(date)" >> "$log"

mg; echo "[F2] WR scheme weight sweep 0.3-0.7 (2024+2025 wk2-18)" >> "$log"
python scripts/sweep_scheme_blend_weight2.py --position WR \
  --weights 0.3,0.4,0.5,0.7 --years 2024,2025 --weeks 2-18 \
  > .sweeps/scheme_blend_WR_v2.txt 2>&1
echo "  done $(date)" >> "$log"

# scheme-alone reference on the identical weeks, so the blend weights are
# compared against a number measured on the same window rather than an older
# run with a different one.
mg; echo "[F3] scheme-alone reference (v2_scheme_matchup, 2024+2025 wk2-18)" >> "$log"
python scripts/backtest_stat_level.py --add v2_scheme_matchup \
  --years 2024,2025 --weeks 2-18 > .sweeps/scheme_alone_2024-2025.txt 2>&1
echo "  done $(date)" >> "$log"
echo "LANE F DONE $(date)" >> "$log"
