#!/usr/bin/env bash
# Lane J - game-total elasticity PER (position, stat).
#
# The shipped v2_game_total_elasticity uses one flat elasticity per position
# for every projected stat. This lane fits a separate elasticity per
# (pos, stat) on 2016-2023, then confirms it on held-out 2024-2025 (and
# 2021-2023 as a cross-season replication). Nothing in data/ changes tonight:
# the candidate rides in .sweeps/gte_perstat_candidate.json, which the
# OFF-by-default flag v2_game_total_elasticity_perstat loads at build time.
#
# Sequential (fit must finish before the confirm can read its JSON). One
# heavy build job at a time; ~3.5 GB peak; no other lane is running.
cd /c/NFLScholar
export PYTHONIOENCODING=utf-8
log=.sweeps/laneJ.log
echo "Lane J start $(date)" > "$log"

mg() { for i in $(seq 1 40); do
  f=$(powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory" 2>/dev/null | tr -dc '0-9'); f=$(( ${f:-0}/1024 ))
  [ "$f" -ge 2600 ] && { echo "  mg ok ${f}MB $(date)" >> "$log"; return; }
  echo "  mg wait ${f}MB $i $(date)" >> "$log"; sleep 120; done; }

mg
echo "[J1] Phase-1 fit  2016-2023 wk1-18 $(date)" >> "$log"
python scripts/fit_game_total_elasticity_perstat.py --mode fit --years 2016-2023 --weeks 1-18 \
  > .sweeps/gte_perstat_fit.txt 2>&1
echo "  J1 done rc=$? $(date)" >> "$log"

mg
echo "[J2] Phase-2 confirm  2024,2025 wk3-18 (GATE run) $(date)" >> "$log"
python scripts/gte_perstat_confirm.py --years 2024,2025 --weeks 3-18 \
  --outliers-csv .sweeps/gte_perstat_outliers.csv \
  > .sweeps/gte_perstat_confirm_2024-2025.txt 2>&1
echo "  J2 done rc=$? $(date)" >> "$log"

mg
echo "[J3] Phase-2 cross-season  2021,2022,2023 wk3-18 $(date)" >> "$log"
python scripts/gte_perstat_confirm.py --years 2021,2022,2023 --weeks 3-18 \
  > .sweeps/gte_perstat_confirm_2021-2023.txt 2>&1
echo "  J3 done rc=$? $(date)" >> "$log"

echo "LANE J DONE $(date)" >> "$log"
