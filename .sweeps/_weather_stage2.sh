#!/usr/bin/env bash
cd /c/NFLScholar
export PYTHONIOENCODING=utf-8
log=.sweeps/weather_stage2.log
echo "waiting for constant sweeps to finish $(date)" > "$log"
while ! grep -q "ALL CONSTANT SWEEPS DONE" .sweeps/const_sweeps.log 2>/dev/null; do sleep 180; done
echo "constant sweeps done; starting weather Stage-2 $(date)" >> "$log"
# wind-bucketed paired backtest of the rewritten v2_weather_adjustment wind table
python scripts/backtest_weather.py --years 2019-2024 --weeks 8-18 > .sweeps/weather_stage2_windheavy.txt 2>&1
echo "  wind-heavy window (2019-24 wk8-18) done $(date)" >> "$log"
python scripts/backtest_weather.py --years 2016-2025 --weeks 1-18 > .sweeps/weather_stage2_fullspan.txt 2>&1
echo "  full span (2016-25 wk1-18) done $(date)" >> "$log"
echo "WEATHER STAGE-2 DONE $(date)" >> "$log"
