#!/usr/bin/env bash
cd /c/NFLScholar
export PYTHONIOENCODING=utf-8
log=.sweeps/overnight_params.log
echo "waiting for batch-1 (_rerun_all.sh -> .sweeps/rerun_all.log) to finish $(date)" > "$log"
while ! grep -q "ALL DONE" .sweeps/rerun_all.log 2>/dev/null; do sleep 120; done
echo "batch-1 done. starting 5 constant sweeps $(date)" >> "$log"

Y="2024,2025"; W="4-15"
python scripts/sweep_model_constant.py --target RECENCY_DECAY --mode set \
  --values 0.75,0.80,0.90,0.95 --years $Y --weeks $W > .sweeps/const_recency_decay.txt 2>&1
echo "1/5 RECENCY_DECAY done $(date)" >> "$log"
python scripts/sweep_model_constant.py --target STAT_K --mode scale \
  --values 0.5,0.75,1.5,2.0 --years $Y --weeks $W > .sweeps/const_stat_k.txt 2>&1
echo "2/5 STAT_K done $(date)" >> "$log"
python scripts/sweep_model_constant.py --target REMATCH_WEIGHT_MULT --mode set \
  --values 1.0,1.3,2.0,2.5 --years $Y --weeks $W > .sweeps/const_rematch.txt 2>&1
echo "3/5 REMATCH_WEIGHT_MULT done $(date)" >> "$log"
python scripts/sweep_model_constant.py --target PRIOR_SEASON_DEFENSE_RECENCY_FLOOR --mode set \
  --values 0.60,0.70,0.90,1.00 --years $Y --weeks $W > .sweeps/const_prior_def_floor.txt 2>&1
echo "4/5 PRIOR_SEASON_DEFENSE_RECENCY_FLOOR done $(date)" >> "$log"
python scripts/sweep_model_constant.py --target ROLE_MATCHUP_K --mode set \
  --values 4,7,15,25 --years $Y --weeks $W > .sweeps/const_role_matchup_k.txt 2>&1
echo "5/5 ROLE_MATCHUP_K done $(date)" >> "$log"
echo "ALL CONSTANT SWEEPS DONE $(date)" >> "$log"
