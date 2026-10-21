"""
Calibration audit C13 (2026-10-20): the QB1 prior-season rate (data/qb_volume_blend.py: team dropbacks x the QB's own style and
efficiency, each shrunk by its own K), re-scored offline from scripts/capture_stat_blend_traces.py --qb-blend-variants
--xtd-variants (no builds).

Recorded per selected QB1 and stat: the blended prior rate as shipped and under
    QB_STYLE_BLEND_K {5, 20, 40} (shipped 10)            pass/rush split of his own dropbacks
    QB_VOLUME_BLEND_K {30, 120, 240} (60)                 his own dropbacks per game vs his team's
    QB_PASS_EFFICIENCY_BLEND_K {50, 200, 400} (100)       his own yards/TD/completion rates per attempt vs league
    QB_RUSH_EFFICIENCY_BLEND_K {2.5, 10, 20} (5)          yards per carry
    QB_RUSH_TD_BLEND_K {20, 80, 160} (40)                 rushing TDs per carry
    starter floor {15, 25} dropbacks per game (20)        below it his own volume gets no weight
The prior enters the in-season blend with weight (1 - w): blended' = blended + (1 - w) x d(prior), with passing TDs' prior first
pulled 25% to the league rate (x0.75) and rushing TDs passed through the expected-TD step (x c). Each row's stat moves by
blended' / blended; points follow through the shipped late steps (paired START pools, weekly bootstrap CI).

    python scripts/eval_qb_blend_variants.py [--traces .sweeps/qb_blend_capture.parquet]
"""
import argparse
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

KEYS = ['year', 'week', 'player', 'pos', 'team', 'stat']


def load(trace_path):
    d, t = load_joined(trace_path)
    t = t.copy()
    t['_i'] = np.arange(len(t))
    x = pd.read_parquet(trace_path + '.xtd.parquet').drop_duplicates(KEYS, keep=False)
    ev = x['cur_td'] + wp.XTD_PRIOR_TD_WEIGHT * x['pri_td']
    x['c'] = ev / (ev + x['stat'].map(wp.XTD_TD_EVENT_K).fillna(8.0))
    q = pd.read_parquet(trace_path + '.qbblend.parquet').drop_duplicates(KEYS, keep=False)
    j = t[['_i', 'current_weight', 'blended_rate'] + KEYS].merge(q, on=KEYS, how='inner')
    j = j.merge(x[KEYS + ['c']], on=KEYS, how='left')
    pass_td = j['stat'].eq('passing_tds').to_numpy()
    xtd = j['stat'].isin(('rushing_tds', 'receiving_tds')).to_numpy()
    j['slope'] = ((1.0 - j['current_weight'].to_numpy(float)) * np.where(pass_td, 1.0 - wp.QB_PASSING_TD_PRIOR_REGRESS, 1.0)
                  * np.where(xtd, j['c'].to_numpy(float), 1.0))
    j = j[np.isfinite(j['slope'])].reset_index(drop=True)
    return d, t, j


def final_under(t, j, col):
    delta = np.nan_to_num(j[col].to_numpy(float) - j['shipped'].to_numpy(float))
    blended = j['blended_rate'].to_numpy(float)
    new = np.clip(blended + j['slope'].to_numpy(float) * delta, 0.0, None)
    ratio = np.where(blended > 1e-9, new / np.where(blended > 1e-9, blended, 1.0), 1.0)
    fn = t['final_projection'].to_numpy(float).copy()
    idx = j['_i'].to_numpy()
    fn[idx] = fn[idx] + t['pre_vacancy_projection'].to_numpy(float)[idx] * (ratio - 1.0)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', default='.sweeps/qb_blend_capture.parquet')
    a = ap.parse_args()
    pd.set_option('display.width', 220)
    d, t, j = load(a.traces)
    print(f"QB1 blend rows {len(j):,} ({j.groupby(['year', 'week', 'player']).ngroups:,} QB1-weeks)")
    base = board(d, t, t['final_projection'].to_numpy(float))
    act = d['actual'].to_numpy(float)
    years = d['year'].to_numpy()
    rows = []
    for col in [c for c in j.columns if c.startswith('QB_')]:
        moved = int((np.abs(j[col] - j['shipped']) > 1e-6).sum())
        var = board(d, t, final_under(t, j, col))
        r, rq = drmse(d, base, var), drmse(d, base, var, 'QB')
        yrs = []
        for y in sorted(np.unique(years)):
            mask = paired_pool(d, base, var) & (years == y) & (d['pos'] == 'QB').to_numpy()
            yrs.append(np.sqrt(((var[mask] - act[mask]) ** 2).mean()) - np.sqrt(((base[mask] - act[mask]) ** 2).mean()))
        rows.append({'variant': col, 'rows moved': moved, 'START-QB': rq['d'], 'ci': f"[{rq['lo']:+.4f},{rq['hi']:+.4f}]",
                     'yrs better': f"{rq['yrs_better']}/{rq['yrs']}", 'START-ALL': r['d'],
                     'QB by season': ' '.join(f'{v:+.4f}' for v in yrs)})
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
