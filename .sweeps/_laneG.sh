#!/usr/bin/env bash
# Lane G - follow-ups queued 2026-09-01 after shipping v2_weather_adjustment.
#   G1  WR wind-strength sweep (the known weak part of the shipped table)
#   G2  startable-calibration prediction dump (base builds only, no variants)
# Waits for TWO lanes to be free so this never makes a third heavy job.
cd /c/NFLScholar
export PYTHONIOENCODING=utf-8
log=.sweeps/laneG.log
echo "Lane G waiting for two lanes to finish $(date)" > "$log"

done_count() {
  n=0
  grep -q "LANE D DONE" .sweeps/laneD.log 2>/dev/null && n=$((n+1))
  grep -q "LANE E DONE" .sweeps/laneE.log 2>/dev/null && n=$((n+1))
  grep -q "LANE F DONE" .sweeps/laneF.log 2>/dev/null && n=$((n+1))
  echo $n
}
for i in $(seq 1 480); do
  [ "$(done_count)" -ge 2 ] && { echo "  two lanes free $(date)" >> "$log"; break; }
  sleep 180
done

mg() { for i in $(seq 1 60); do
  f=$(powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory" 2>/dev/null | tr -dc '0-9'); f=$(( ${f:-0}/1024 ))
  [ "$f" -ge 3500 ] && { echo "  mg ok ${f}MB $(date)" >> "$log"; return; }
  echo "  mg wait ${f}MB $i $(date)" >> "$log"; sleep 120; done; }

mg; echo "[G1] WR wind-strength sweep (ablation, 2021-25 wk3-17)" >> "$log"
python scripts/sweep_weather_strength.py --years 2021,2022,2023,2024,2025 --weeks 3-17 \
  --out .sweeps/sweep_weather_strength_WR.txt >> "$log" 2>&1
echo "  done $(date)" >> "$log"

mg; echo "[G2] startable-calibration prediction dump (2019-2025 wk4-17)" >> "$log"
python scripts/fit_startable_calibration.py --mode dump --years 2019-2025 --weeks 4-17 \
  > .sweeps/startable_calib_dump.txt 2>&1
echo "  done $(date)" >> "$log"
echo "LANE G DONE $(date)" >> "$log"
