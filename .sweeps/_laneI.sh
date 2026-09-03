#!/usr/bin/env bash
# Lane I - the v3 (final) calibration fit.
#   I1  uncalibrated whole-pool prediction dump, 2019-2025 wk5-17
# Two more seasons than any previous calibration fit has used. Nothing else
# is running, so no memory guard is needed - the box is free.
cd /c/NFLScholar
export PYTHONIOENCODING=utf-8
log=.sweeps/laneI.log
echo "[I1] v3 calibration dump 2019-2025 wk5-17 (91 builds) $(date)" > "$log"
python scripts/fit_weekly_calibration_v3.py --mode dump --years 2019-2025 --weeks 5-17 \
  > .sweeps/calibration_v3_dump.txt 2>&1
echo "  done $(date)" >> "$log"
echo "LANE I DONE $(date)" >> "$log"
