#!/usr/bin/env bash
cd /c/NFLScholar
export PYTHONIOENCODING=utf-8
log=.sweeps/laneB.log
echo "Lane B (v2) start $(date)" > "$log"

# free-RAM guard: proceed once >=2.6 GB physical RAM is free (headroom for one
# more ~3.5 GB peak alongside Lane A). Caps the wait at ~90 min then proceeds.
mem_guard() {
  for i in $(seq 1 45); do
    freekb=$(powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory" 2>/dev/null | tr -dc '0-9')
    freemb=$(( ${freekb:-0} / 1024 ))
    if [ "$freemb" -ge 2600 ]; then
      echo "  mem_guard ok: ${freemb} MB free $(date)" >> "$log"; return 0
    fi
    echo "  mem_guard: only ${freemb} MB free, wait $i $(date)" >> "$log"
    sleep 120
  done
  echo "  mem_guard: proceeding after max wait $(date)" >> "$log"
}

mem_guard
echo "[B1] weather Stage-2 wind-heavy (2019-24 wk8-18) $(date)" >> "$log"
python scripts/backtest_weather.py --years 2019-2024 --weeks 8-18 > .sweeps/weather_stage2_windheavy.txt 2>&1
echo "  done $(date)" >> "$log"

mem_guard
echo "[B2] weather Stage-2 full span (2016-25 wk1-18) $(date)" >> "$log"
python scripts/backtest_weather.py --years 2016-2025 --weeks 1-18 > .sweeps/weather_stage2_fullspan.txt 2>&1
echo "  done $(date)" >> "$log"

echo "[B3] Phase-1 defense-blend design sweep $(date)" >> "$log"
if [ -f scripts/sweep_defense_blend_design.py ]; then
  python scripts/sweep_defense_blend_design.py --years 2016-2025 > .sweeps/defense_blend_design.txt 2>&1
  echo "  done $(date)" >> "$log"
else
  echo "  SKIPPED - script not built yet" >> "$log"
fi
echo "LANE B DONE $(date)" >> "$log"
