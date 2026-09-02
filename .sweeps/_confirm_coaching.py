import os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from data.coaching_changes import _POS_COHORT_DEFAULTS
import numpy as np
FLAT=12.0; CLAMP=(4.0,30.0); COH=("none","dc_only","both","hc_only"); RESET={"both","hc_only"}
def scaled(lam, shape="all"):
    cells=[]
    for pos,d in _POS_COHORT_DEFAULTS.items():
        for c in COH:
            f=d.get(c); mv=(shape=="all" or (shape=="reset" and c in RESET))
            pg=FLAT if (f is None or not mv) else float(np.clip(FLAT+lam*(f-FLAT),*CLAMP))
            cells.append(f"{pos}:{c}={pg:g}")
    return ",".join(cells)
GRID=[("l0.8 wk10-17", scaled(0.8), "10-17"), ("reset-only wk10-17", scaled(1.0,"reset"), "10-17"),
      ("l0.8 wk2-17", scaled(0.8), "2-17"), ("reset-only wk2-17", scaled(1.0,"reset"), "2-17")]
SCOPES=("ALL","QB","RB","WR","TE","START-QB","START-RB","START-WR","START-TE")
def run(spec,weeks):
    env=dict(os.environ, POS_COHORT_PRIOR_GAMES=spec, PYTHONIOENCODING="utf-8")
    p=subprocess.run([sys.executable, os.path.join(ROOT,"scripts","backtest_component.py"),
        "--add","v2_coaching_aware_defense_prior","--years","2020,2021,2022,2023,2024,2025","--weeks",weeks],
        cwd=ROOT, env=env, capture_output=True, encoding="utf-8", errors="replace")
    return (p.stdout or "")+"\n"+(p.stderr or "")
def parse(o):
    r={}
    for ln in o.splitlines():
        m=re.match(r"\s*([A-Z-]+)\s+n=\d+\s+MAE base [\d.]+ vs variant [\d.]+\s+dMAE\(var-base\)\s+([+-][\d.]+).*?(excludes 0|includes 0|too few)",ln)
        if m: r[m.group(1)]=(float(m.group(2)), m.group(3)=="excludes 0")
    return r
rows=[]
for i,(lab,spec,wk) in enumerate(GRID,1):
    print(f"[{i}/{len(GRID)}] {lab}",flush=True); rows.append((lab,parse(run(spec,wk))))
hdr=f"{'config':>20}  "+"".join(f"{s:>12}" for s in SCOPES); print("\n"+hdr+"\n"+"-"*len(hdr))
for lab,pr in rows:
    print(f"{lab:>20}  "+"".join((f"{pr[s][0]:+.3f}{'*' if pr[s][1] else ' '}" if s in pr else "   -   ").rjust(12) for s in SCOPES))
