"""
A2 / A3 offline study (2026-11-11, docs/audit_leads_2026-11-10.md): does a defense missing its usual personnel (A2), or an
offence missing its usual line (A3), move results past what the model projects?

Panel, per team-week (2022-2025, weeks 3-17), point in time: each unit's share of its season-to-date snaps (snap counts from
earlier weeks only) belonging to players unavailable this week - listed Out or Doubtful on the week's injury report, on a
reserve / exempt / retired / practice-squad list, cut or traded, or no longer on the team's weekly roster. Gameday inactives
(INA) do not count: they are announced about 90 minutes before kickoff, after the board.

Units: secondary (CB, S, FS, SS, DB), linebackers (LB, ILB, OLB, MLB), front (DE, DT, NT, DL), offensive line (T, G, C).

Reads (residual = actual minus the base harness board's projection, played rows only):
  A2  opposing QB passing yards and TDs, WR/TE receiving yards against the defense's missing secondary share; RB rushing
      yards against the missing front + linebacker share;
  A3  the team's own QB passing yards and RB rushing yards against its missing line share.
Per season, the OLS slope of the residual on the share; held out by season (slope fit on the other seasons, applied to the
held-out one) the share of that stat's MSE removed. Bar to build a flag: the same sign in 4 of 4 seasons (the harness has
no 5th) and at least 0.5% of the stat's MSE held out.

    python scripts/audit_personnel.py [--out .sweeps/audit_personnel.txt]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.historical_availability import load_injury_reports, load_weekly_rosters  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402

BOARD = os.path.join('.sweeps', 'harness_boards', 'capacity_screen_2022-2025_wk3-17.parquet')
SEASONS = (2022, 2023, 2024, 2025)
UNITS = {'secondary': {'CB', 'S', 'FS', 'SS', 'DB'}, 'linebackers': {'LB', 'ILB', 'OLB', 'MLB'},
         'front': {'DE', 'DT', 'NT', 'DL'}, 'oline': {'T', 'G', 'C', 'OT', 'OG', 'OL'}}
PRESENT = {'ACT', 'INA'}
MSE_BAR = 0.005


def unit_missing(season):
    """One row per (team, week) with the missing share of each unit's season-to-date snaps."""
    snaps = pd.read_parquet(os.path.join('data', 'cache', f'snap_counts_{season}.parquet'))
    snaps = snaps[snaps['game_type'].astype(str).eq('REG')].copy()
    snaps['_key'] = clean_name_exact(snaps['player'].astype(str))
    snaps['unit'] = ''
    for unit, pos in UNITS.items():
        snaps.loc[snaps['position'].isin(pos), 'unit'] = unit
    snaps = snaps[snaps['unit'] != '']
    snaps['snaps'] = np.where(snaps['unit'].eq('oline'), snaps['offense_snaps'], snaps['defense_snaps'])
    rosters = load_weekly_rosters(season)
    rosters = rosters.assign(_key=clean_name_exact(rosters['full_name'].astype(str)),
                             _present=rosters['status'].astype(str).str.upper().isin(PRESENT))
    present = set(zip(rosters.loc[rosters['_present'], 'week'].astype(int), rosters.loc[rosters['_present'], 'team'],
                      rosters.loc[rosters['_present'], '_key']))
    roster_weeks = set(zip(rosters['week'].astype(int), rosters['team']))
    inj = load_injury_reports(season)
    inj = inj[inj['report_status'].isin(['Out', 'Doubtful'])]
    listed = set(zip(inj['week'].astype(int), inj['team'], clean_name_exact(inj['full_name'].astype(str))))
    out = []
    for (team, unit), g in snaps.groupby(['team', 'unit']):
        for week in range(3, 18):
            if (week, team) not in roster_weeks:
                continue
            prior = g[g['week'] < week].groupby('_key')['snaps'].sum()
            prior = prior[prior > 0]
            if prior.empty:
                continue
            missing = [k for k in prior.index if (week, team, k) in listed or (week, team, k) not in present]
            out.append({'season': season, 'team': team, 'week': week, 'unit': unit,
                        'missing': float(prior[missing].sum() / prior.sum())})
    t = pd.DataFrame(out)
    return t.pivot_table(index=['season', 'team', 'week'], columns='unit', values='missing').reset_index()


def residual_panel():
    board = pd.read_parquet(BOARD)
    board = board[board['variant'] == 'base'].copy()
    board['_key'] = clean_name_exact(board['Player'].astype(str))
    frames = []
    for season in SEASONS:
        b = board[board['year'] == season]
        stats, _tc, name_col, _ = load_and_merge_data(season, 'Full PPR')
        stats = stats[stats['season'] == season] if 'season' in stats.columns else stats
        cols = ['passing_yards', 'passing_tds', 'receiving_yards', 'rushing_yards']
        act = stats[['week', name_col] + cols].copy()
        act['_key'] = clean_name_exact(act[name_col].astype(str))
        act = act.drop(columns=[name_col]).drop_duplicates(['week', '_key']).rename(columns={c: f'act_{c}' for c in cols})
        b = b.merge(act, on=['week', '_key'], how='inner')          # played rows (a box score)
        miss = unit_missing(season)
        opp = miss.rename(columns={'team': 'Opponent', **{u: f'opp_{u}' for u in UNITS}}).drop(columns=['season'])
        own = miss.rename(columns={'team': 'Team', **{u: f'own_{u}' for u in UNITS}}).drop(columns=['season'])
        b['Opponent'] = b['Opponent'].astype(str).str.lstrip('@').str.strip()
        b = b.merge(opp, on=['Opponent', 'week'], how='left').merge(own, on=['Team', 'week'], how='left')
        b['opp_box'] = (b['opp_front'].fillna(0) + b['opp_linebackers'].fillna(0)) / 2
        frames.append(b)
    return pd.concat(frames, ignore_index=True)


READS = [
    ('A2', 'QB', 'passing_yards', 'opp_secondary'), ('A2', 'QB', 'passing_tds', 'opp_secondary'),
    ('A2', 'WR', 'receiving_yards', 'opp_secondary'), ('A2', 'TE', 'receiving_yards', 'opp_secondary'),
    ('A2', 'RB', 'rushing_yards', 'opp_box'), ('A2', 'QB', 'passing_yards', 'opp_front'),
    ('A3', 'QB', 'passing_yards', 'own_oline'), ('A3', 'QB', 'passing_tds', 'own_oline'),
    ('A3', 'RB', 'rushing_yards', 'own_oline'),
]


def read(panel, pos, stat, share):
    """(per-season slopes and n, the held-out MSE share removed, the share's mean) for one read."""
    p = panel[(panel['Pos'] == pos) & panel[share].notna()].copy()
    p['res'] = pd.to_numeric(p[f'act_{stat}'], errors='coerce') - pd.to_numeric(p[stat], errors='coerce')
    p = p[p['res'].notna()]
    x_mean = {s: p.loc[p['year'] == s, share].mean() for s in SEASONS}
    rec = {'mean share': p[share].mean(), 'rows': len(p)}
    sse0 = sse1 = 0.0
    for s in SEASONS:
        g, rest = p[p['year'] == s], p[p['year'] != s]
        x, y = g[share] - x_mean[s], g['res'] - g['res'].mean()
        rec[f'slope {s}'] = float((x * y).sum() / (x * x).sum()) if (x * x).sum() > 0 else np.nan
        xr = rest[share] - rest['year'].map(x_mean)
        yr = rest['res'] - rest.groupby('year')['res'].transform('mean')
        b = float((xr * yr).sum() / (xr * xr).sum()) if (xr * xr).sum() > 0 else 0.0
        y_held = g['res'] - g['res'].mean()
        sse0 += float((y_held ** 2).sum())
        sse1 += float(((y_held - b * x) ** 2).sum())
    slopes = np.array([rec[f'slope {s}'] for s in SEASONS])
    pooled = np.sign(np.nansum(slopes))
    rec['same sign'] = int((np.sign(slopes) == pooled).sum())
    rec['held-out MSE removed'] = 1 - sse1 / sse0 if sse0 else np.nan
    rec['passes build bar'] = rec['same sign'] == len(SEASONS) and rec['held-out MSE removed'] >= MSE_BAR
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join('.sweeps', 'audit_personnel.txt'))
    a = ap.parse_args()
    panel = residual_panel()
    rows = [{'lead': lead, 'Pos': pos, 'stat': stat, 'share': share, **read(panel, pos, stat, share)}
            for lead, pos, stat, share in READS]
    t = pd.DataFrame(rows)
    shares = panel.drop_duplicates(['year', 'Team', 'week'])[[c for c in panel.columns if c.startswith('own_')]].describe()
    text = '\n'.join([
        f"A2/A3: residual (actual minus the base harness projection, played rows) against the missing share of a unit's "
        f"season-to-date snaps, 2022-2025 wk3-17 ({len(panel):,} player-weeks). Slope = residual per 100% of the unit missing. "
        f"Build bar: same sign in 4 of 4 seasons and {MSE_BAR:.1%}+ of the stat's MSE removed held out by season.", '',
        t.to_string(index=False, float_format=lambda v: f'{v:.4f}'), '',
        'Missing share per team-week (own units):', shares.to_string(float_format=lambda v: f'{v:.3f}')])
    print(text)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
