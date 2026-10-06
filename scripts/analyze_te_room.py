"""
TE room analysis on receiver-audit boards (scripts/diag_receiver_audit.py).

The 2021-2025 calibration dump showed league TE targets right in total but too
FLAT across TEs: TEs ranked 4-32 short 0.2-0.45 targets a game, TEs ranked 33+
over. This splits every team's TE room by within-team rank (projected
targets) and traces the target channel through each model stage.

    python scripts/analyze_te_room.py --boards .sweeps/receiver_audit
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def load(path):
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(path, 'rec_*.parquet')))],
                  ignore_index=True)
    d = d[d['Availability'] > 0.01].copy()
    keys = ['year', 'week', 'Team', 'Pos']
    d['room_rank'] = d.groupby(keys)['p_targets'].rank(ascending=False, method='first')
    d['rk'] = pd.cut(d['room_rank'], [0, 1, 2, 99], labels=['1', '2', '3+'])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boards', default='.sweeps/receiver_audit')
    a = ap.parse_args()
    pd.set_option('display.width', 220)
    d = load(a.boards)
    print(f"rows {len(d)} live WR/TE, weeks {sorted(d['week'].unique())}, years {sorted(d['year'].unique())}")
    for pos in ('TE', 'WR'):
        s = d[d['Pos'] == pos]
        clean = s[~s['room_has_out']]
        print(f"\n{'=' * 90}\n{pos} - by within-team rank (projected targets), clean rooms, ALL live rows (no box score = 0)\n{'=' * 90}")
        g = clean.groupby('rk', observed=True)
        out = pd.DataFrame({
            'n': g.size(), 'played': g['played'].mean(),
            'snap exp': g['snap_share'].mean(), 'snap act': g['act_snap'].mean() / 100,
            'tgt p': g['p_targets'].mean(), 'tgt a': g['act_targets'].mean(),
            'yds p': g['p_receiving_yards'].mean(), 'yds a': g['act_receiving_yards'].mean(),
            'td p': g['p_receiving_tds'].mean(), 'td a': g['act_receiving_tds'].mean(),
            'pts bias': g.apply(lambda x: (x['raw_pts'] - x['act_pts']).mean()),
        })
        print(out.round(3).to_string())
        rooms = clean.groupby(['year', 'week', 'Team']).agg(p=('p_targets', 'sum'), a=('act_targets', 'sum'))
        print(f"  {pos} room total targets per team-game: projected {rooms['p'].mean():.2f} actual {rooms['a'].mean():.2f}")

        print(f"\n  {pos} target channel by stage (carry-free: mean per row, clean rooms):")
        rows = []
        for lab, x in list(clean.groupby('rk', observed=True)) + [('all', clean)]:
            w = x['targets.blended_rate'].clip(lower=1e-6)
            rows.append({'rank': lab,
                         'blended': x['targets.blended_rate'].mean(),
                         'pre-vacancy': x['targets.pre_vacancy_projection'].mean(),
                         'capacity d': x['targets.pass_capacity_delta'].fillna(0).mean(),
                         'vacancy d': x['targets.vacancy_delta'].fillna(0).mean(),
                         'final': x['p_targets'].mean(), 'actual': x['act_targets'].mean(),
                         'matchup': np.average(x['targets.matchup_multiplier'].fillna(1), weights=w),
                         'descript': np.average(x['targets.script_neutral_multiplier'].fillna(1), weights=w),
                         'pace': np.average(x['targets.pace_multiplier'].fillna(1), weights=w),
                         'partic': np.average(x['targets.participation_multiplier'].fillna(1), weights=w)})
        print(pd.DataFrame(rows).set_index('rank').round(3).to_string())

        pl = clean[clean['played']]
        print(f"\n  {pos} efficiency, played rows: projected / actual")
        for lab, x in list(pl.groupby('rk', observed=True)) + [('all', pl)]:
            t, at = x['p_targets'].sum(), x['act_targets'].sum()
            print(f"    rank {lab:<4} n={len(x):4d}  tgt {t / len(x):.2f}/{at / len(x):.2f}  "
                  f"yds/tgt {x['p_receiving_yards'].sum() / t:.2f}/{x['act_receiving_yards'].sum() / at:.2f}  "
                  f"td/tgt {x['p_receiving_tds'].sum() / t:.4f}/{x['act_receiving_tds'].sum() / at:.4f}  "
                  f"catch {x['p_receptions'].sum() / t:.3f}/{x['act_receptions'].sum() / at:.3f}")

        if pos == 'TE':
            te1 = clean[clean['rk'] == '1'].copy()
            te1['sb'] = pd.qcut(te1['snap_share'].rank(method='first'), 5, labels=['q1', 'q2', 'q3', 'q4', 'q5'])
            print("\n  TE1 by expected snap share quintile (all rows):")
            print(te1.groupby('sb', observed=True).agg(snap=('snap_share', 'mean'), tgt_p=('p_targets', 'mean'),
                                                        tgt_a=('act_targets', 'mean'), td_p=('p_receiving_tds', 'mean'),
                                                        td_a=('act_receiving_tds', 'mean')).round(3).to_string())
            te1['tb'] = pd.qcut(te1['p_targets'].rank(method='first'), 5, labels=['q1', 'q2', 'q3', 'q4', 'q5'])
            print("\n  TE1 by projected targets quintile (all rows): proj -> actual")
            print(te1.groupby('tb', observed=True).agg(tgt_p=('p_targets', 'mean'), tgt_a=('act_targets', 'mean'),
                                                        blended=('targets.blended_rate', 'mean'),
                                                        cap=('targets.pass_capacity_delta', 'mean')).round(3).to_string())
            print("\n  TE target bias by year (all ranks, clean rooms, all rows):")
            print(clean.groupby(['year', 'rk'], observed=True).apply(
                lambda x: x['p_targets'].mean() - x['act_targets'].mean()).unstack().round(2).to_string())


if __name__ == '__main__':
    main()
