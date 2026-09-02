#!/usr/bin/env bash
# Lane H - the last untested built flag.
#   H1  v2_venue_mult standalone ablation-style add
# v2_venue_mult is built, gated (2 call sites) and has passing unit tests, but
# has never been backtested on its own. It is the other half of the unbundled
# `game_env`; its sibling v2_game_total_elasticity shipped 2026-08-31 on its
# own confirm. Waits for a free lane - 3 concurrent heavy builds OOM the box.
cd /c/NFLScholar
export PYTHONIOENCODING=utf-8
log=.sweeps/laneH.log
echo "Lane H waiting for a free lane (F or G) $(date)" > "$log"
for i in $(seq 1 480); do
  grep -q "LANE F DONE" .sweeps/laneF.log 2>/dev/null && { echo "  lane F freed $(date)" >> "$log"; break; }
  grep -q "LANE G DONE" .sweeps/laneG.log 2>/dev/null && { echo "  lane G freed $(date)" >> "$log"; break; }
  sleep 180
done
mg() { for i in $(seq 1 60); do
  f=$(powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory" 2>/dev/null | tr -dc '0-9'); f=$(( ${f:-0}/1024 ))
  [ "$f" -ge 3500 ] && { echo "  mg ok ${f}MB $(date)" >> "$log"; return; }
  echo "  mg wait ${f}MB $i $(date)" >> "$log"; sleep 120; done; }

mg; echo "[H1] v2_venue_mult standalone (2021-25 wk3-17)" >> "$log"
python scripts/backtest_component.py --add v2_venue_mult \
  --years 2021,2022,2023,2024,2025 --weeks 3-17 \
  > .sweeps/venue_mult_standalone.txt 2>&1
echo "  done $(date)" >> "$log"
echo "LANE H DONE $(date)" >> "$log"
