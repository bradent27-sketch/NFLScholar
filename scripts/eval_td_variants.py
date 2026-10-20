"""
Calibration audit C14 (2026-10-20): how the model sets rushing/receiving TD rates, re-scored offline from
scripts/capture_stat_blend_traces.py --xtd-variants (no builds).

    own   = w * this season's TD rate + (1 - w) * last season's,  w = games / (games + k_eff)   (the ordinary blend; K by stat
            and position: RB rushing/receiving 6, WR receiving 30, TE 16, QB rushing 6)
    out   = c * own + (1 - c) * xTD,  c = events / (events + K_xtd),  events = this season's TDs + 0.7 x last season's
            (K_xtd 6 rushing, 8 receiving; xTD = what a league-average player scores from his red-zone opportunities)

Recorded per player: own, the TD event counts and out, so xTD = (out - c * own) / (1 - c) is recovered exactly. Varied: the
own-rate K (x0.5 / x2 / x4), K_xtd (x0.5 / x2 / x4) and the 0.7 weight on last season's TDs (0.4 / 1.0), one group at a time
(RB rushing, RB receiving, WR receiving, TE receiving, QB rushing); the TD moves by out'/out and points follow through the shipped
late steps (paired START pools, weekly bootstrap CI). Then a leave-one-season-out joint choice across groups.

    python scripts/eval_td_variants.py [--traces .sweeps/td_capture.parquet]
"""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data.weekly_projections as wp  # noqa: E402
from scripts.eval_blend_modifiers import board  # noqa: E402
from scripts.eval_stat_k_sweep import drmse, load_joined, paired_pool  # noqa: E402

GROUPS = (('RB', 'rushing_tds'), ('RB', 'receiving_tds'), ('WR', 'receiving_tds'), ('TE', 'receiving_tds'), ('QB', 'rushing_tds'))
OWN_SCALE, XTD_SCALE, PRIOR_W = (0.5, 1.0, 2.0, 4.0), (0.5, 1.0, 2.0, 4.0), (0.4, 0.7, 1.0)
KEYS = ['year', 'week', 'player', 'pos', 'team', 'stat']


def load(trace_path):
    d, t = load_joined(trace_path)
    x = pd.read_parquet(trace_path + '.xtd.parquet').drop_duplicates(KEYS, keep=False)
    t = t.copy()
    t['_i'] = np.arange(len(t))
    j = t[['_i'] + KEYS + ['current_games', 'current_weight', 'current_rate', 'prior_rate']].merge(x, on=KEYS, how='inner')
    k = j['stat'].map(wp.XTD_TD_EVENT_K).fillna(8.0).to_numpy()
    ev = j['cur_td'].to_numpy() + wp.XTD_PRIOR_TD_WEIGHT * j['pri_td'].to_numpy()
    c = ev / (ev + k)
    own, out = j['own'].to_numpy(), j['out'].to_numpy()
    j['xtd'] = np.where(1 - c > 1e-9, (out - c * own) / np.where(1 - c > 1e-9, 1 - c, 1.0), own)
    g, w = j['current_games'].to_numpy(float), j['current_weight'].to_numpy(float)
    ok = (g > 0) & (w > 0) & (w < 1)
    j['k_own'] = np.where(ok, g * (1 - w) / np.where(ok, w, 1.0), np.nan)
    return d, t, j


def out_under(j, own_scale, xtd_scale, prior_w):
    g = j['current_games'].to_numpy(float)
    ke = j['k_own'].to_numpy(float)
    ok = np.isfinite(ke)
    cur_r, pri_r = np.nan_to_num(j['current_rate'].to_numpy(float)), np.nan_to_num(j['prior_rate'].to_numpy(float))
    w1 = j['current_weight'].to_numpy(float)
    w2 = np.where(ok, g / np.where(ok, g + own_scale * ke, 1.0), w1)
    # the recorded own rate is exact; the 3-decimal trace inputs only supply the CHANGE a different K makes
    own = j['own'].to_numpy(float) + np.where(ok, (w2 - w1) * (cur_r - pri_r), 0.0)
    k = j['stat'].map(wp.XTD_TD_EVENT_K).fillna(8.0).to_numpy() * xtd_scale
    ev = j['cur_td'].to_numpy() + prior_w * j['pri_td'].to_numpy()
    c = ev / (ev + k)
    return c * own + (1 - c) * j['xtd'].to_numpy()


def final_under(t, j, settings):
    """settings: {(pos, stat): (own_scale, xtd_scale, prior_w)}; groups not listed stay shipped."""
    fn = t['final_projection'].to_numpy(float).copy()
    for (pos, stat), (os_, xs, pw) in settings.items():
        m = (j['pos'] == pos).to_numpy() & (j['stat'] == stat).to_numpy()
        if not m.any():
            continue
        sub = j[m]
        new = out_under(sub, os_, xs, pw)
        old = sub['out'].to_numpy(float)
        ratio = np.where(old > 1e-9, new / np.where(old > 1e-9, old, 1.0), 1.0)
        idx = sub['_i'].to_numpy()
        fn[idx] = fn[idx] + t['pre_vacancy_projection'].to_numpy(float)[idx] * (ratio - 1.0)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', default='.sweeps/td_capture.parquet')
    a = ap.parse_args()
    pd.set_option('display.width', 220)
    d, t, j = load(a.traces)
    shipped = out_under(j, 1.0, 1.0, wp.XTD_PRIOR_TD_WEIGHT)
    print(f"TD rows joined {len(j):,}; replay of the shipped settings vs recorded: max gap {np.abs(shipped - j['out']).max():.2e}")
    base = board(d, t, t['final_projection'].to_numpy(float))
    rows, boards = [], {}
    for grp in GROUPS:
        for os_, xs, pw in itertools.product(OWN_SCALE, XTD_SCALE, PRIOR_W):
            if (os_, xs, pw) == (1.0, 1.0, 0.7):
                continue
            var = board(d, t, final_under(t, j, {grp: (os_, xs, pw)}))
            boards[(grp, os_, xs, pw)] = var
            r = drmse(d, base, var, grp[0])
            rows.append({'group': f'{grp[0]} {grp[1]}', 'own K x': os_, 'xTD K x': xs, 'prior TD w': pw,
                         'pos START dRMSE': r['d'], 'ci': f"[{r['lo']:+.4f},{r['hi']:+.4f}]", 'yrs better': f"{r['yrs_better']}/{r['yrs']}",
                         'START-ALL': drmse(d, base, var)['d']})
    res = pd.DataFrame(rows)
    print('\n=== one knob at a time per group (others shipped) ===')
    one = res[((res['xTD K x'] == 1.0) & (res['prior TD w'] == 0.7)) | ((res['own K x'] == 1.0) & (res['prior TD w'] == 0.7))
              | ((res['own K x'] == 1.0) & (res['xTD K x'] == 1.0))]
    print(one.round(4).to_string(index=False))
    print('\n=== best 3 settings per group ===')
    print(res.sort_values('START-ALL').groupby('group').head(3).sort_values(['group', 'START-ALL']).round(4).to_string(index=False))
    res.to_csv('.sweeps/td_variants_grid.csv', index=False)

    print('\n=== leave one season out: each group chosen on four seasons (per-row error), applied together to the fifth ===')
    act = d['actual'].to_numpy(float)
    years = d['year'].to_numpy()
    out = []
    for y in sorted(np.unique(years)):
        chosen = {}
        for grp in GROUPS:
            best, best_r = (1.0, 1.0, 0.7), 0.0
            for (g2, os_, xs, pw), var in boards.items():
                if g2 != grp:
                    continue
                mask = paired_pool(d, base, var) & (years != y) & (d['pos'] == grp[0]).to_numpy()
                r_ = float(np.sqrt(((var[mask] - act[mask]) ** 2).mean()) - np.sqrt(((base[mask] - act[mask]) ** 2).mean()))
                if r_ < best_r:
                    best, best_r = (os_, xs, pw), r_
            chosen[grp] = best
        var = board(d, t, final_under(t, j, chosen))
        mask = paired_pool(d, base, var) & (years == y)
        held = float(np.sqrt(((var[mask] - act[mask]) ** 2).mean()) - np.sqrt(((base[mask] - act[mask]) ** 2).mean()))
        out.append({'held-out': int(y), 'START-ALL': held,
                    'chosen': ' '.join(f"{g[0]}.{g[1][:4]}={v}" for g, v in chosen.items() if v != (1.0, 1.0, 0.7)) or 'all shipped'})
    loyo = pd.DataFrame(out)
    pd.set_option('display.max_colwidth', 200)
    print(loyo.round(4).to_string(index=False))
    print(f"mean {loyo['START-ALL'].mean():+.4f}, better in {(loyo['START-ALL'] < 0).sum()} of {len(loyo)}")


if __name__ == '__main__':
    main()
