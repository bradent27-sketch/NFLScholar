#!/usr/bin/env bash
cd /c/NFLScholar
export PYTHONIOENCODING=utf-8
log=.sweeps/laneD.log
echo "Lane D waiting for Lane A (const sweeps) $(date)" > "$log"
while ! grep -q "ALL CONSTANT SWEEPS DONE" .sweeps/const_sweeps.log 2>/dev/null; do sleep 180; done
echo "Lane A done; Lane D start $(date)" >> "$log"
mg() { for i in $(seq 1 40); do
  f=$(powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory" 2>/dev/null | tr -dc '0-9'); f=$(( ${f:-0}/1024 ))
  [ "$f" -ge 2600 ] && { echo "  mg ok ${f}MB $(date)" >> "$log"; return; }
  echo "  mg wait ${f}MB $i $(date)" >> "$log"; sleep 120; done; }

mg; echo "[D1] weather whole-slate confirm (2019-25 wk1-18, no bucket)" >> "$log"
python scripts/backtest_component.py --add v2_weather_adjustment --years 2019,2020,2021,2022,2023,2024,2025 --weeks 1-18 > .sweeps/weather_wholeslate_confirm.txt 2>&1
echo "  done $(date)" >> "$log"
mg; echo "[D2] weather wind-heavy re-run (bucket bug fixed)" >> "$log"
python scripts/backtest_weather.py --years 2018-2024 --weeks 6-18 > .sweeps/weather_windheavy_v2.txt 2>&1
echo "  done $(date)" >> "$log"
mg; echo "[D3] REMATCH wide confirm 1.0/1.2/1.3 x 2022-25" >> "$log"
python scripts/sweep_model_constant.py --target REMATCH_WEIGHT_MULT --mode set --values 1.0,1.2,1.3 --years 2022,2023,2024,2025 --weeks 4-16 > .sweeps/const_rematch_wide.txt 2>&1
echo "  done $(date)" >> "$log"
mg; echo "[D4] PRIOR_SEASON_DEFENSE_RECENCY_FLOOR wide 0.90/0.95/1.00 x 2022-25" >> "$log"
python scripts/sweep_model_constant.py --target PRIOR_SEASON_DEFENSE_RECENCY_FLOOR --mode set --values 0.90,0.95,1.00 --years 2022,2023,2024,2025 --weeks 4-16 > .sweeps/const_prior_def_floor_wide.txt 2>&1
echo "  done $(date)" >> "$log"
mg; echo "[D5] RECENCY_DECAY fine 0.86/0.88/0.90/0.92 x 2022-25" >> "$log"
python scripts/sweep_model_constant.py --target RECENCY_DECAY --mode set --values 0.86,0.88,0.90,0.92 --years 2022,2023,2024,2025 --weeks 4-16 > .sweeps/const_recency_decay_fine.txt 2>&1
echo "  done $(date)" >> "$log"
echo "LANE D DONE $(date)" >> "$log"
