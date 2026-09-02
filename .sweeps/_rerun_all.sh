#!/usr/bin/env bash
cd /c/NFLScholar
export PYTHONIOENCODING=utf-8
log=.sweeps/rerun_all.log
echo "start $(date)" > "$log"

echo "[1/5] coaching confirm (l0.8 + reset-only; wk10-17 no-harm + wk2-17 full)" >> "$log"
python .sweeps/_confirm_coaching.py >> "$log" 2>&1
echo "  done $(date)" >> "$log"

echo "[2/5] GTE ship-config confirm (ablation; v2_game_total_elasticity now in DEFAULT)" >> "$log"
python scripts/backtest_component.py --flags v2_game_total_elasticity \
  --years 2022,2023,2024,2025 --weeks 4-17 > .sweeps/gte_ship_confirm.txt 2>&1
echo "  done $(date)" >> "$log"

echo "[3/5] startable-TE bias measurement" >> "$log"
python scripts/analyze_startable_te_bias.py --years 2021-2025 --weeks 4-17 \
  > .sweeps/startable_te_bias.txt 2>&1
echo "  done $(date)" >> "$log"

echo "[4/5] scheme-blend sweep TE (0.6-1.0)" >> "$log"
python scripts/sweep_scheme_blend_weight.py --position TE --weights 0.6,0.7,0.8,0.9,1.0 \
  --years 2023,2024,2025 --weeks 4-17 > .sweeps/scheme_blend_TE.txt 2>&1
echo "  done $(date)" >> "$log"

echo "[5/5] scheme-blend sweep WR (0.3-0.7)" >> "$log"
python scripts/sweep_scheme_blend_weight.py --position WR --weights 0.3,0.5,0.7 \
  --years 2023,2024,2025 --weeks 4-17 > .sweeps/scheme_blend_WR.txt 2>&1
echo "  ALL DONE $(date)" >> "$log"
