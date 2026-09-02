#!/usr/bin/env bash
# Lane E - the coaching final-verdict run.
#
# Waits for Lane D's D2 (the 7GB weather job) to clear before starting, so
# this runs alongside D3-D5 (the ~3.5GB constant sweeps) instead of racing
# the one job on the queue that alone eats half the box. 2 concurrent at
# ~3.5GB each is the pattern that has held all week; 3, or 1 alongside a
# 7GB job, is what OOMs.
cd /c/NFLScholar
export PYTHONIOENCODING=utf-8
log=.sweeps/laneE.log
echo "Lane E waiting for Lane D to reach D3 $(date)" > "$log"

# D3 starting is the signal that the two memory-heavy weather jobs are done.
for i in $(seq 1 300); do
  grep -q "\[D3\]" .sweeps/laneD.log 2>/dev/null && break
  grep -q "LANE D DONE" .sweeps/laneD.log 2>/dev/null && break
  sleep 180
done
echo "Lane D past the heavy jobs; Lane E arming $(date)" >> "$log"

mg() { for i in $(seq 1 60); do
  f=$(powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory" 2>/dev/null | tr -dc '0-9'); f=$(( ${f:-0}/1024 ))
  [ "$f" -ge 3500 ] && { echo "  mg ok ${f}MB $(date)" >> "$log"; return; }
  echo "  mg wait ${f}MB $i $(date)" >> "$log"; sleep 120; done; }

mg
echo "[E1] coaching final verdict 2016-2025 wk2-10 (7 configs + base)" >> "$log"
python scripts/coaching_final_verdict.py --years 2016-2025 --weeks 2-10 \
  > .sweeps/coaching_final_verdict.txt 2>&1
echo "  done $(date)" >> "$log"
echo "LANE E DONE $(date)" >> "$log"
