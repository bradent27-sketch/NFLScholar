"""
Data-integrity scan of one built weekly board (2026-10-06): the checks a person would make by eye, run on every
row. Prints each failed check with a handful of example rows and exits 0 either way (it is a report, not a gate).

    python scripts/scan_board_integrity.py --year 2026 --week 5            # live availability sources
    python scripts/scan_board_integrity.py --year 2025 --week 9 --historical
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data.prediction_ledger as pl  # noqa: E402

pl.LEDGER_DIR = tempfile.mkdtemp(prefix='ledger_scan_')      # nothing here writes the ledger; belt and braces

import data.weekly_projections as wp  # noqa: E402
from data.loaders import load_schedule  # noqa: E402

STAT = ['passing_attempts', 'passing_completions', 'passing_yards', 'passing_tds', 'passing_interceptions',
        'rushing_attempts', 'rushing_yards', 'rushing_tds', 'targets', 'receptions', 'receiving_yards', 'receiving_tds']


def report(title, bad, cols, limit=6):
    print(f"\n[{'FAIL' if len(bad) else 'ok  '}] {title}: {len(bad)}")
    if len(bad):
        print(bad[cols].head(limit).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=2026)
    ap.add_argument('--week', type=int, default=5)
    ap.add_argument('--historical', action='store_true', help='as-of build with injury/reserve replay (a played week)')
    a = ap.parse_args()
    feats = wp.DEFAULT_FEATURES | ({'v2_historical_injury_replay', 'v2_historical_reserve_replay'} if a.historical else set())
    b, meta = wp.build_weekly_projections(a.year, a.week, 'Full PPR', as_of_week=a.week,
                                          apply_injury=not a.historical, features=feats)
    pd.set_option('display.width', 220)
    print(f"board {a.year} week {a.week}: {len(b)} rows, {b['Team'].nunique()} teams, positions {b['Pos'].value_counts().to_dict()}")
    for c in STAT:
        if c in b.columns:
            b[c] = pd.to_numeric(b[c], errors='coerce')
    ident = ['Player', 'Pos', 'Team']

    report('duplicate (Player, Pos, Team) rows', b[b.duplicated(['Player', 'Pos', 'Team'], keep=False)], ident)
    report('same player on more than one team', b[b.duplicated(['Player', 'Pos'], keep=False) & ~b.duplicated(['Player', 'Pos', 'Team'], keep=False)], ident)
    report('missing Opponent', b[b['Opponent'].isna() | (b['Opponent'].astype(str).str.strip() == '')], ident + ['Opponent'])
    sched = load_schedule(a.year)
    wk = sched[pd.to_numeric(sched['week'], errors='coerce') == a.week]
    playing = set(wk['home_team'].astype(str)) | set(wk['away_team'].astype(str))
    report('rows for a team that is not playing this week (bye / wrong team)', b[~b['Team'].astype(str).isin(playing)], ident + ['Opponent'])
    report('teams playing this week with no rows on the board', pd.DataFrame({'Team': sorted(playing - set(b['Team'].astype(str)))}), ['Team'])
    nonfinite = b[b[[c for c in STAT if c in b.columns]].apply(lambda s: ~np.isfinite(s.fillna(0))).any(axis=1)]
    report('non-finite stat values', nonfinite, ident)
    report('negative stat values', b[(b[[c for c in STAT if c in b.columns]].fillna(0) < -1e-9).any(axis=1)], ident)
    report('receptions > targets', b[b['receptions'] > b['targets'] + 1e-6], ident + ['targets', 'receptions'])
    report('receiving TDs > receptions', b[b['receiving_tds'] > b['receptions'] + 1e-6], ident + ['receptions', 'receiving_tds'])
    report('rushing TDs > rushing attempts', b[b['rushing_tds'] > b['rushing_attempts'] + 1e-6], ident + ['rushing_attempts', 'rushing_tds'])
    report('passing TDs > completions', b[b['passing_tds'] > b['passing_completions'] + 1e-6], ident + ['passing_completions', 'passing_tds'])
    t = b[b['targets'] > 1]
    report('receiving yards per target > 18 or < 3 (targets > 1)', t[(t['receiving_yards'] / t['targets'] > 18) | (t['receiving_yards'] / t['targets'] < 3)],
           ident + ['targets', 'receiving_yards'])
    r = b[b['rushing_attempts'] > 2]
    report('rushing yards per carry > 8 or < 2.0 (carries > 2)', r[(r['rushing_yards'] / r['rushing_attempts'] > 8) | (r['rushing_yards'] / r['rushing_attempts'] < 2.0)],
           ident + ['rushing_attempts', 'rushing_yards'])
    qa = b[b['Pos'] == 'QB'].groupby('Team')['passing_attempts'].apply(lambda s: (s > 5).sum())
    print(f"\n[{'FAIL' if ((qa != 1) & qa.index.isin(playing)).any() else 'ok  '}] teams with other than exactly one QB projected over 5 attempts:",
          qa[(qa != 1) & qa.index.isin(playing)].to_dict())
    team = b.groupby('Team').agg(pa=('passing_attempts', 'sum'), ra=('rushing_attempts', 'sum'), tg=('targets', 'sum'))
    team = team[team.index.isin(playing)]
    print(f"\nteam volume: pass attempts {team['pa'].min():.1f}-{team['pa'].max():.1f}, carries {team['ra'].min():.1f}-{team['ra'].max():.1f}, "
          f"targets {team['tg'].min():.1f}-{team['tg'].max():.1f}; targets/attempt {(team['tg'] / team['pa']).min():.2f}-{(team['tg'] / team['pa']).max():.2f}")
    odd = team[(team['pa'] < 22) | (team['pa'] > 45) | (team['ra'] < 18) | (team['ra'] > 36) | (team['tg'] / team['pa'] > 1.05) | (team['tg'] / team['pa'] < 0.8)]
    report('teams with implausible projected volume (pass 22-45, rush 18-36, targets/attempt 0.8-1.05)', odd.reset_index(), ['Team', 'pa', 'ra', 'tg'])
    if 'Availability' in b.columns:
        avail = pd.to_numeric(b['Availability'], errors='coerce')
        report('Availability <= 0.01 but still projected points > 0.5', b[(avail <= 0.01) & (pd.to_numeric(b['Model Proj Pts'], errors='coerce') > 0.5)], ident + ['Availability', 'Model Proj Pts'])
        report('Availability outside [0, 1]', b[(avail < -1e-9) | (avail > 1 + 1e-9)], ident + ['Availability'])
    pts = pd.to_numeric(b['Model Proj Pts'], errors='coerce')
    raw = pd.to_numeric(b['Raw Model Proj Pts'], errors='coerce')
    report('Model Proj Pts negative or NaN', b[pts.isna() | (pts < 0)], ident + ['Model Proj Pts'])
    report('Model Proj Pts > 3 pts but Raw is ~0 (calibration intercept lifting nothing)', b[(pts > 3) & (raw < 0.5)], ident + ['Raw Model Proj Pts', 'Model Proj Pts'])
    ss = pd.to_numeric(b.get('Expected Snap Share'), errors='coerce')
    report('Expected Snap Share outside [0, 1]', b[(ss < -1e-9) | (ss > 1 + 1e-9)], ident + ['Expected Snap Share'])
    top = b.sort_values('Model Proj Pts', ascending=False).groupby('Pos').head(3)[['Pos', 'Player', 'Team', 'Opponent', 'Model Proj Pts']]
    print('\ntop 3 per position (eyeball):'); print(top.sort_values(['Pos', 'Model Proj Pts'], ascending=[True, False]).to_string(index=False))
    # explanations present for every row, and chain consistent
    ex = meta.get('explanations') or {}
    missing = [(r.Player, r.Pos, r.Team) for r in b.itertuples() if (r.Player, r.Pos, r.Team) not in ex]
    print(f"\n[{'FAIL' if missing else 'ok  '}] rows without a decomposition: {len(missing)} {missing[:5]}")
    gap = []
    for key, d in ex.items():
        for stat, v in (d.get('stats') or {}).items():
            fin = v.get('final_projection')
            chain = (v.get('pre_vacancy_projection') or 0) + (v.get('pass_capacity_delta') or 0) + (v.get('carry_budget_delta') or 0) + (v.get('vacancy_delta') or 0)
            if fin is not None and abs(chain - fin) > 0.01:
                gap.append((key[0], stat, round(chain, 3), fin))
    print(f"[{'FAIL' if gap else 'ok  '}] decomposition stages not adding to the final value: {len(gap)} {gap[:5]}")
    for key, d in ex.items():
        sl = d.get('stat_line') or {}
        row = b[(b['Player'] == key[0]) & (b['Pos'] == key[1]) & (b['Team'] == key[2])]
        if len(row) == 1:
            sc = wp.score_projected_stats({k: v for k, v in sl.items()}, 'Full PPR')
            if abs(sc - float(row['Raw Model Proj Pts'].iloc[0])) > 0.15:
                print('   raw-points vs stat-line mismatch', key, round(sc, 2), float(row['Raw Model Proj Pts'].iloc[0]))
                break


if __name__ == '__main__':
    main()
