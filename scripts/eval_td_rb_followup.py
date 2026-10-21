"""
Calibration audit C14 follow-up (2026-10-20): the RB touchdown blend leaning harder on expected TDs, the one consistent hint in
scripts/eval_td_variants.py. Both RB groups (rushing and receiving TDs) moved together: xTD K x {2, 4, 8, inf} and last season's TD
weight {0.2, 0.4, 0.7}, with and without one scale per stat that keeps the shipped average RB TD rate (expected TDs run below
actual for startable RBs, so leaning on them also lowers the level). Per-season dRMSE and a leave-one-season-out choice.

    python scripts/eval_td_rb_followup.py   (needs .sweeps/td_capture.parquet from capture_stat_blend_traces.py --xtd-variants)
"""
import itertools, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scripts.eval_td_variants import load, out_under
from scripts.eval_blend_modifiers import board
from scripts.eval_stat_k_sweep import drmse, paired_pool
pd.set_option('display.width', 220)
d, t, j = load('.sweeps/td_capture.parquet')
base = board(d, t, t['final_projection'].to_numpy(float))
act = d['actual'].to_numpy(float); years = d['year'].to_numpy()
def final_level(xs, pw, keep_level):
    fn = t['final_projection'].to_numpy(float).copy()
    for stat in ('rushing_tds', 'receiving_tds'):
        m = (j['pos'] == 'RB').to_numpy() & (j['stat'] == stat).to_numpy()
        sub = j[m]
        new = out_under(sub, 1.0, xs, pw); old = sub['out'].to_numpy(float)
        if keep_level:  # one scale per stat so RB TD rates keep the shipped average
            new = new * old.sum() / new.sum()
        ratio = np.where(old > 1e-9, new / np.where(old > 1e-9, old, 1.0), 1.0)
        idx = sub['_i'].to_numpy()
        fn[idx] = fn[idx] + t['pre_vacancy_projection'].to_numpy(float)[idx] * (ratio - 1.0)
    return fn
rows, boards = [], {}
for xs, pw, kl in itertools.product((2.0, 4.0, 8.0, 1e6), (0.2, 0.4, 0.7), (False, True)):
    var = board(d, t, final_level(xs, pw, kl)); boards[(xs, pw, kl)] = var
    r = drmse(d, base, var, 'RB'); ra = drmse(d, base, var)
    m = paired_pool(d, base, var) & (d['pos'] == 'RB').to_numpy()
    yrs = []
    for y in range(2021, 2026):
        mm = paired_pool(d, base, var) & (years == y)
        yrs.append(np.sqrt(((var[mm]-act[mm])**2).mean()) - np.sqrt(((base[mm]-act[mm])**2).mean()))
    rows.append({'xTD K x': xs, 'prior w': pw, 'keep level': kl, 'START-RB': r['d'], 'ci': f"[{r['lo']:+.4f},{r['hi']:+.4f}]",
                 'START-ALL': ra['d'], 'ALLci': f"[{ra['lo']:+.4f},{ra['hi']:+.4f}]", 'RB bias': float((var[m]-act[m]).mean()),
                 'by season': ' '.join(f'{v:+.4f}' for v in yrs)})
print(pd.DataFrame(rows).round(4).to_string(index=False))
for kl in (False, True):
    out = []
    for y in range(2021, 2026):
        best, br = None, 0.0
        for key, var in boards.items():
            if key[2] != kl: continue
            mask = paired_pool(d, base, var) & (years != y)
            r_ = float(np.sqrt(((var[mask]-act[mask])**2).mean()) - np.sqrt(((base[mask]-act[mask])**2).mean()))
            if r_ < br: best, br = key, r_
        var = boards[best] if best else base
        mask = paired_pool(d, base, var) & (years == y)
        out.append(float(np.sqrt(((var[mask]-act[mask])**2).mean()) - np.sqrt(((base[mask]-act[mask])**2).mean())))
        print(f"  keep level {kl} held {y}: chosen {best} -> START-ALL {out[-1]:+.4f}")
    print(f"  mean {np.mean(out):+.4f}, better {sum(o < 0 for o in out)}/5")
