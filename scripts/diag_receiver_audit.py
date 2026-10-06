"""
WR/TE audit collector (2026-10-05): the RB level audit's collector
(scripts/diag_rb_level_audit.py) generalised to receivers, so TE target
allocation can be looked at by team room and by model stage. Writes one
parquet per (year, week) with team, availability, snap share, projected stat
line, per-stage trace and actuals (DNP = 0, `played` flag).

    python scripts/diag_receiver_audit.py --years 2022,2023,2024,2025 --weeks 4,6,8,10,12,14,16 --out .sweeps/receiver_audit
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.diag_rb_level_audit import REPLAY, RESERVE, STATS, ACT_COL, TRACE_FIELDS  # noqa: E402

POSITIONS = ('WR', 'TE')


def collect(years, weeks, out_dir, features_extra=(), drop=()):
    from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES
    from data.transforms import load_and_merge_data
    feats = frozenset((DEFAULT_FEATURES | {REPLAY, RESERVE} | set(features_extra)) - set(drop))
    os.makedirs(out_dir, exist_ok=True)
    for year in years:
        stats_df, team_col, name_col, _ = load_and_merge_data(year, 'Full PPR')
        s = stats_df.copy()
        s['_w'] = pd.to_numeric(s['week'], errors='coerce')
        for c in list(ACT_COL.values()) + ['fantasy_points_ppr', 'weekly_snap_pct']:
            s[c] = pd.to_numeric(s.get(c), errors='coerce')
        rb_stats = s[s['position'].astype(str).isin(POSITIONS)]
        for week in weeks:
            wk = rb_stats[rb_stats['_w'] == week]
            if wk.empty:
                continue
            act = wk.groupby(name_col).agg(**{f'act_{k}': (v, 'sum') for k, v in ACT_COL.items()},
                                           act_pts=('fantasy_points_ppr', 'sum'), act_snap=('weekly_snap_pct', 'max'))
            act['played'] = True
            proj, meta = build_weekly_projections(year, week, 'Full PPR', as_of_week=week, apply_injury=False, features=feats)
            if proj.empty:
                continue
            expl = meta.get('explanations') or {}
            rows = []
            for _, r in proj[proj['Pos'].isin(POSITIONS)].iterrows():
                st = ((expl.get((r['Player'], r['Pos'], r['Team'])) or {}).get('stats') or {})
                rec = {'year': year, 'week': week, 'Player': r['Player'], 'Pos': r['Pos'], 'Team': str(r['Team']).upper(),
                       'Availability': float(r.get('Availability', np.nan)),
                       'raw_pts': float(r.get('Raw Model Proj Pts', np.nan)), 'pts': float(r.get('Model Proj Pts', np.nan)),
                       'snap_share': float(r.get('Expected Snap Share', np.nan))}
                for stat in STATS:
                    rec[f'p_{stat}'] = float(r.get(stat, np.nan))
                    t = st.get(stat) or {}
                    for f in TRACE_FIELDS:
                        rec[f'{stat}.{f}'] = t.get(f, np.nan)
                rows.append(rec)
            fr = pd.DataFrame(rows).merge(act, left_on='Player', right_index=True, how='left')
            fr['played'] = fr['played'].fillna(False).astype(bool)
            for c in [c for c in fr.columns if c.startswith('act_') and c != 'act_snap']:
                fr[c] = fr[c].fillna(0.0)
            vac = {str(e.get('team', '')).upper() for e in (meta.get('vacancy_ledger') or []) if e.get('volume') == 'targets'}
            fr['room_has_out'] = fr['Team'].isin(vac)
            fr.to_parquet(os.path.join(out_dir, f'rec_{year}_w{week:02d}.parquet'), index=False)
            print(f"{year} w{week} done ({len(fr)} rows)", flush=True)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2022,2023,2024,2025')
    ap.add_argument('--weeks', default='4,6,8,10,12,14,16')
    ap.add_argument('--out', default='.sweeps/receiver_audit')
    ap.add_argument('--features', default='')
    a = ap.parse_args()
    collect([int(v) for v in a.years.split(',')], [int(v) for v in a.weeks.split(',')], a.out,
            [f for f in a.features.split(',') if f])


if __name__ == '__main__':
    main()
