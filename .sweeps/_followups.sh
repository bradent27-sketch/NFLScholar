#!/usr/bin/env bash
cd /c/NFLScholar
export PYTHONIOENCODING=utf-8
log=.sweeps/followups.log
echo "waiting for coaching confirm (_confirm_coaching.py) to finish..." > $log
while tasklist //FI "IMAGENAME eq python.exe" //FO CSV 2>/dev/null | grep -q '"14432"'; do sleep 60; done
echo "coaching confirm done $(date). starting follow-ups." >> $log

echo "[1/4] GTE ship-config confirm (ablation)" >> $log
python scripts/backtest_component.py --flags v2_game_total_elasticity \
  --years 2022,2023,2024,2025 --weeks 4-17 > .sweeps/gte_ship_confirm.txt 2>&1
echo "  done $(date)" >> $log

echo "[2/4] startable-TE bias" >> $log
python scripts/analyze_startable_te_bias.py --years 2021-2025 --weeks 4-17 \
  > .sweeps/startable_te_bias.txt 2>&1
echo "  done $(date)" >> $log

echo "[3/4] scheme-blend sweep TE" >> $log
python scripts/sweep_scheme_blend_weight.py --position TE --weights 0.6,0.7,0.8,0.9,1.0 \
  --years 2023,2024,2025 --weeks 4-17 > .sweeps/scheme_blend_TE.txt 2>&1
echo "  done $(date)" >> $log

echo "[4/4] scheme-blend sweep WR" >> $log
python scripts/sweep_scheme_blend_weight.py --position WR --weights 0.3,0.5,0.7 \
  --years 2023,2024,2025 --weeks 4-17 > .sweeps/scheme_blend_WR.txt 2>&1
echo "  done $(date). ALL FOLLOW-UPS COMPLETE." >> $log
