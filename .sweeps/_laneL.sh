#!/usr/bin/env bash
# Lane L - GTE reruns + 3 new-team/vacancy cold-start fixes (2026-09-04).
# Waits for Lane K. One heavy build lane at a time.
#   L1  GTE pooled 5-year (2021-2025) MAE reconfirm + TD calibration
#   L2  TE2 buried-vet dock: does docking TE-2 like WR-2 help at Week 1?
#   L3  v2_new_team_starter_restoration (Waddle/Evans-shaped team-change fix)
#   L4  v2_receiver_cold_start_vacancy (Watson-shaped departed-teammate fix)
cd /c/NFLScholar
export PYTHONIOENCODING=utf-8
log=.sweeps/laneL.log
echo "Lane L waiting for Lane K $(date)" > "$log"
while ! grep -q "LANE K DONE" .sweeps/laneK.log 2>/dev/null; do sleep 180; done
echo "Lane K done; Lane L start $(date)" >> "$log"

mg() { for i in $(seq 1 40); do
  f=$(powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory" 2>/dev/null | tr -dc '0-9'); f=$(( ${f:-0}/1024 ))
  [ "$f" -ge 2600 ] && { echo "  mg ok ${f}MB $(date)" >> "$log"; return; }
  echo "  mg wait ${f}MB $i $(date)" >> "$log"; sleep 120; done; }

mg; echo "[L1] GTE pooled reconfirm + TD calibration  2021-2025 wk3-18 $(date)" >> "$log"
python scripts/gte_perstat_reconfirm.py --years 2021,2022,2023,2024,2025 --weeks 3-18 \
  --outliers-csv .sweeps/gte_perstat_reconfirm_outliers.csv \
  > .sweeps/gte_perstat_reconfirm_2021-2025.txt 2>&1
echo "  L1 done rc=$? $(date)" >> "$log"

mg; echo "[L2] TE2 buried-vet dock (slot rank 3->2)  wk1 2022-2025 $(date)" >> "$log"
python scripts/sweep_te_buried_vet_slot.py --years 2022,2023,2024,2025 --weeks 1-1 \
  > .sweeps/te_buried_vet_slot_wk1_2022-2025.txt 2>&1
echo "  L2 done rc=$? $(date)" >> "$log"

mg; echo "[L3] v2_new_team_starter_restoration  wk1 2022-2025 $(date)" >> "$log"
python scripts/sweep_new_team_starter_restoration.py --years 2022,2023,2024,2025 --weeks 1-1 \
  > .sweeps/new_team_starter_restoration_wk1_2022-2025.txt 2>&1
echo "  L3 done rc=$? $(date)" >> "$log"

mg; echo "[L4] v2_receiver_cold_start_vacancy  wk1 2022-2025 $(date)" >> "$log"
python scripts/sweep_receiver_cold_start_vacancy.py --years 2022,2023,2024,2025 --weeks 1-1 \
  > .sweeps/receiver_cold_start_vacancy_wk1_2022-2025.txt 2>&1
echo "  L4 done rc=$? $(date)" >> "$log"

echo "LANE L DONE $(date)" >> "$log"
