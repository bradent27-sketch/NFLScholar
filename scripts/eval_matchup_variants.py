"""
Calibration audit C6 + C7 (2026-10-20): the role overlay on the matchup and the defense estimator's blowout / RB offense-prior
settings, re-scored offline from scripts/capture_stat_blend_traces.py --matchup-variants (no builds).

Recorded per player call of the forward matchup (continuous role blend): the shipped multiplier and the multiplier under
    C6  ROLE_MATCHUP_K {2.5, 5, 20, 40, off}                 (shipped 10: role evidence / (evidence + K) of the role rating)
    C7  blowout discount {0, 0.25, 0.75, 1 = none}           (shipped 0.5, RB/QB/TE defenses' 28+ point games)
        blowout margin {21, 35}                              (shipped 28)
        RB offense prior games {3, 12}                       (shipped 6)
QB passing uses the overall rating only, so its variant RATINGS are recorded and clipped here. The matchup multiplies the
blended rate, so each row's stat moves by m' / m; points follow through the shipped late steps (paired START pools, weekly
bootstrap CI). A row moves only where the board used this multiplier (a WR/TE row whose PFF alignment or scheme multiplier
replaced it is left alone).

    python scripts/eval_matchup_variants.py [--traces .sweeps/matchup_capture.parquet]
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
LABELS = {'k_2.5': 'C6 role K 2.5', 'k_5': 'C6 role K 5', 'k_20': 'C6 role K 20', 'k_40': 'C6 role K 40',
          'k_1e+09': 'C6 role overlay off', 'ov_disc_0': 'C7 blowout games dropped', 'ov_disc_0.25': 'C7 discount 0.25',
          'ov_disc_0.75': 'C7 discount 0.75', 'ov_disc_1': 'C7 no blowout discount', 'ov_margin_21': 'C7 margin 21',
          'ov_margin_35': 'C7 margin 35', 'ov_oprior_3': 'C7 RB offense prior 3', 'ov_oprior_12': 'C7 RB offense prior 12'}


def load(trace_path):
    d, t = load_joined(trace_path)
    t = t.copy()
    t['_i'] = np.arange(len(t))
    t['opp'] = d['opp'].astype(str).to_numpy()[t['_row'].to_numpy()]
    m = pd.read_parquet(trace_path + '.matchup.parquet').drop_duplicates(KEYS, keep=False)
    j = t[['_i', 'matchup_multiplier'] + KEYS].merge(m, on=KEYS, how='inner')
    q = pd.read_parquet(trace_path + '.qbratings.parquet').drop_duplicates(['year', 'week', 'pos', 'defense', 'stat'], keep=False)
    qb = t[t['pos'].eq('QB') & t['stat'].isin(wp.QB_PASSING_MATCHUP_STATS)][['_i', 'matchup_multiplier', 'opp'] + KEYS]
    qb = qb.merge(q.rename(columns={'defense': 'opp'}), on=['year', 'week', 'pos', 'opp', 'stat'], how='inner')
    for c in [c for c in qb.columns if c == 'shipped' or c.startswith('ov_')]:
        qb[c] = np.clip(qb[c].fillna(1.0), *wp.MATCHUP_CLIP)
    j = pd.concat([j, qb.drop(columns='opp')], ignore_index=True)
    j = j[(j['matchup_multiplier'] - j['shipped']).abs() < 6e-4].reset_index(drop=True)
    return d, t, j


def final_under(t, j, col):
    m_old = j['shipped'].to_numpy(float)
    m_new = j[col].to_numpy(float) if col in j.columns else m_old
    ratio = np.where(np.isfinite(m_new) & (m_old > 0), m_new / np.where(m_old > 0, m_old, 1.0), 1.0)
    fn = t['final_projection'].to_numpy(float).copy()
    idx = j['_i'].to_numpy()
    fn[idx] = fn[idx] + t['pre_vacancy_projection'].to_numpy(float)[idx] * (ratio - 1.0)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', default='.sweeps/matchup_capture.parquet')
    a = ap.parse_args()
    pd.set_option('display.width', 220)
    d, t, j = load(a.traces)
    print(f"trace rows {len(t):,}; rows whose matchup is the recorded multiplier: {len(j):,} "
          f"(by position: {j.groupby('pos').size().to_dict()})")
    base = board(d, t, t['final_projection'].to_numpy(float))
    act = d['actual'].to_numpy(float)
    years = d['year'].to_numpy()
    rows = []
    for col, label in LABELS.items():
        if col not in j.columns:
            continue
        moved = int((np.abs(j[col].fillna(j['shipped']) - j['shipped']) > 1e-6).sum())
        var = board(d, t, final_under(t, j, col))
        r = drmse(d, base, var)
        row = {'variant': label, 'rows moved': moved, 'START-ALL': r['d'], 'ci': f"[{r['lo']:+.4f},{r['hi']:+.4f}]",
               'yrs better': f"{r['yrs_better']}/{r['yrs']}"}
        for pos in ('QB', 'RB', 'WR', 'TE'):
            row[pos] = drmse(d, base, var, pos)['d']
        yrs = []
        for y in sorted(np.unique(years)):
            mask = paired_pool(d, base, var) & (years == y)
            yrs.append(np.sqrt(((var[mask] - act[mask]) ** 2).mean()) - np.sqrt(((base[mask] - act[mask]) ** 2).mean()))
        row['by season'] = ' '.join(f'{v:+.4f}' for v in yrs)
        rows.append(row)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
