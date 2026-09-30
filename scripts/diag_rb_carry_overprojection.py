"""
Diagnostic for RB carry over-projection (reported 2026-09-29): in rooms with
nobody OUT the board's RB carries run ~6 per team-week over actual (27.6 vs
21.4, 2024-2025 wk3-17), while the SHARES within the room are close to right.
This locates the stage that creates the level error, from the per-player
trace build_weekly_projections records for 'rushing_attempts':

  blended_rate            own rate blended with the prior (per game, pre-multiplier)
  x matchup/script/script-neutral/pace/env   the multipliers
  pre_vacancy_projection  after those
  final board value       after vacancy

and against three references: the room's actual RB carries, the team's actual
total rush attempts (all positions), and the team's own recent carries.

Also records RB rushing yards / TDs / points (raw and calibrated) so the
carry error can be checked against what the calibration line is hiding, and
projected QB/WR/TE carries so the whole team's rush budget can be compared.

Usage:
    python scripts/diag_rb_carry_overprojection.py --years 2024,2025 --weeks 3-17 --out .sweeps/diag_rb_carries
    python scripts/diag_rb_carry_overprojection.py --summarize --out .sweeps/diag_rb_carries
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPLAY = 'v2_historical_injury_replay'
RESERVE = 'v2_historical_reserve_replay'
TRACE = ('current_rate', 'raw_prior_rate', 'prior_rate', 'prior_source', 'current_weight', 'current_games',
         'role_scale', 'expected_snap_share', 'prior_snap_share', 'blended_rate', 'matchup_multiplier',
         'script_multiplier', 'script_neutral_multiplier', 'pace_multiplier', 'environment_multiplier',
         'availability_multiplier', 'pre_vacancy_projection', 'vacancy_delta', 'role_confidence')


def collect(years, weeks, out_dir, extra=(), drop=()):
    from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES
    from data.transforms import load_and_merge_data

    feats = frozenset((DEFAULT_FEATURES | {REPLAY, RESERVE} | set(extra)) - set(drop))
    players, teams = [], []
    for year in years:
        stats_df, team_col, name_col, _ = load_and_merge_data(year, 'Full PPR')
        s = stats_df.copy()
        s['_w'] = pd.to_numeric(s['week'], errors='coerce')
        s['_team'] = (s['game_team'] if 'game_team' in s.columns else s[team_col]).astype(str).str.upper()
        s['_pos'] = s['position'].astype(str)
        for c in ('rushing_attempts', 'rushing_yards', 'rushing_tds', 'fantasy_points_ppr', 'weekly_snap_pct'):
            s[c] = pd.to_numeric(s.get(c), errors='coerce')
        for week in weeks:
            wk = s[s['_w'] == week]
            if wk.empty:
                continue
            act = (wk.groupby(name_col).agg(act_carries=('rushing_attempts', 'sum'), act_rush_yds=('rushing_yards', 'sum'),
                                            act_rush_tds=('rushing_tds', 'sum'), act_pts=('fantasy_points_ppr', 'sum'),
                                            act_snap=('weekly_snap_pct', 'max')))
            act['played'] = True
            grp = wk.assign(g=np.where(wk['_pos'] == 'QB', 'qb', np.where(wk['_pos'] == 'RB', 'rb',
                                       np.where(wk['_pos'].isin(['WR', 'TE']), 'wrte', 'other'))))
            act_team = grp.pivot_table(index='_team', columns='g', values='rushing_attempts', aggfunc='sum',
                                       fill_value=0.0).add_prefix('act_rush_')
            act_team['act_rush_total'] = grp.groupby('_team')['rushing_attempts'].sum()

            proj, meta = build_weekly_projections(year, week, 'Full PPR', as_of_week=week, apply_injury=False, features=feats)
            if proj.empty:
                continue
            out_rb = set()
            for e in meta.get('vacancy_ledger') or []:
                if e.get('volume') == 'rushing_attempts':
                    out_rb.add(str(e.get('team', '')).upper())
            expl = meta.get('explanations') or {}
            proj = proj.copy()
            proj['Team'] = proj['Team'].astype(str).str.upper()
            tm = proj.groupby(['Team', 'Pos'])['rushing_attempts'].sum().unstack('Pos', fill_value=0.0)
            tm = tm.reindex(columns=['QB', 'RB', 'WR', 'TE'], fill_value=0.0)
            tm.columns = ['proj_rush_qb', 'proj_rush_rb', 'proj_rush_wr', 'proj_rush_te']
            tm['proj_qb_pass_att'] = proj[proj['Pos'] == 'QB'].groupby('Team')['passing_attempts'].sum()
            tm = tm.join(act_team, how='left')
            tm['year'], tm['week'] = year, week
            tm['room_has_out_rb'] = tm.index.isin(out_rb)
            teams.append(tm.reset_index().rename(columns={'index': 'Team'}))

            rows = []
            for _, r in proj[proj['Pos'] == 'RB'].iterrows():
                t = ((expl.get((r['Player'], 'RB', r['Team'])) or {}).get('stats') or {}).get('rushing_attempts') or {}
                rec = {'year': year, 'week': week, 'Player': r['Player'], 'Team': r['Team'],
                       'Availability': float(r.get('Availability', np.nan)),
                       'rushing_attempts': float(r.get('rushing_attempts', np.nan)),
                       'rushing_yards': float(r.get('rushing_yards', np.nan)),
                       'rushing_tds': float(r.get('rushing_tds', np.nan)),
                       'targets': float(r.get('targets', np.nan)),
                       'raw_pts': float(r.get('Raw Model Proj Pts', np.nan)),
                       'pts': float(r.get('Model Proj Pts', np.nan)),
                       'room_has_out': r['Team'] in out_rb}
                for f in TRACE:
                    rec[f] = t.get(f, np.nan)
                rows.append(rec)
            fr = pd.DataFrame(rows).merge(act, left_on='Player', right_index=True, how='left')
            fr['played'] = fr['played'].fillna(False).astype(bool)
            for c in ('act_carries', 'act_rush_yds', 'act_rush_tds'):
                fr[c] = fr[c].fillna(0.0)
            players.append(fr)
            print(f"{year} w{week} done ({len(fr)} RBs)", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    pl, tmdf = pd.concat(players, ignore_index=True), pd.concat(teams, ignore_index=True)
    pl.to_parquet(os.path.join(out_dir, 'rb_players.parquet'), index=False)
    tmdf.to_parquet(os.path.join(out_dir, 'rb_teams.parquet'), index=False)
    return pl, tmdf


def summarize(pl, tm):
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 40)
    keys = ['year', 'week', 'Team']
    live = pl[pl['Availability'] > 0.01].copy()
    for c in ('rushing_attempts', 'blended_rate', 'pre_vacancy_projection'):
        live[c] = pd.to_numeric(live[c], errors='coerce').fillna(0.0)
    clean = live[~live['room_has_out']].copy()
    print(f"clean RB rooms: {clean.groupby(keys).ngroups} team-weeks, {len(clean)} RB rows\n")

    print("=== STAGE DECOMPOSITION, clean rooms: room sums per team-week ===")
    stages = {
        'blended rate (per game, pre-multiplier)': 'blended_rate',
        'after multipliers (pre-vacancy)': 'pre_vacancy_projection',
        'final board': 'rushing_attempts',
    }
    rows = {name: clean.groupby(keys)[col].sum().mean() for name, col in stages.items()}
    rows['ACTUAL room carries (board RBs)'] = clean.groupby(keys)['act_carries'].sum().mean()
    t = tm.merge(clean[keys].drop_duplicates(), on=keys)
    rows['ACTUAL team RB carries (all RBs)'] = t['act_rush_rb'].mean()
    rows['ACTUAL team rush attempts (all positions)'] = t['act_rush_total'].mean()
    for k, v in rows.items():
        print(f"  {k:46s} {v:7.2f}")
    mult = (clean['pre_vacancy_projection'].sum() / clean['blended_rate'].sum())
    print(f"  carry-weighted multiplier blended->pre-vacancy: x{mult:.3f}")
    for m in ('matchup_multiplier', 'script_multiplier', 'script_neutral_multiplier', 'pace_multiplier',
              'environment_multiplier', 'availability_multiplier'):
        v = pd.to_numeric(clean[m], errors='coerce')
        w = clean['blended_rate']
        ok = v.notna() & (w > 0)
        print(f"     {m:28s} weighted mean {np.average(v[ok], weights=w[ok]):.3f}" if ok.any() else f"     {m}: n/a")

    print("\n=== TEAM budget, clean rooms: projected vs actual rush attempts by position group ===")
    t = tm.merge(clean[keys].drop_duplicates(), on=keys)
    print(pd.DataFrame({
        'proj': [t['proj_rush_qb'].mean(), t['proj_rush_rb'].mean(), t['proj_rush_wr'].mean() + t['proj_rush_te'].mean(),
                 t[['proj_rush_qb', 'proj_rush_rb', 'proj_rush_wr', 'proj_rush_te']].sum(axis=1).mean()],
        'actual': [t['act_rush_qb'].mean(), t['act_rush_rb'].mean(), t['act_rush_wrte'].mean(), t['act_rush_total'].mean()],
    }, index=['QB', 'RB', 'WR+TE', 'TEAM TOTAL']).assign(err=lambda d: d['proj'] - d['actual']).round(2).to_string())

    print("\n=== does the over-projection scale with roster depth or with the lead back? (clean rooms) ===")
    room = clean.groupby(keys).agg(proj=('rushing_attempts', 'sum'), act=('act_carries', 'sum'),
                                   n_rb=('Player', 'size'),
                                   lead=('rushing_attempts', 'max')).reset_index()
    room['err'] = room['proj'] - room['act']
    room['depth_mass'] = room['proj'] - room['lead']
    print(room.groupby(pd.cut(room['n_rb'], [0, 2, 3, 4, 5, 20]), observed=True).agg(
        n=('err', 'size'), proj=('proj', 'mean'), act=('act', 'mean'), err=('err', 'mean')).round(2).to_string())
    print(room.groupby(pd.qcut(room['lead'], 4, duplicates='drop'), observed=True).agg(
        n=('err', 'size'), lead_proj=('lead', 'mean'), proj=('proj', 'mean'), act=('act', 'mean'), err=('err', 'mean')).round(2).to_string())

    print("\n=== calibration: actual carries vs projected carries by projection bin (clean rooms, all RB rows) ===")
    clean['bin'] = pd.cut(clean['rushing_attempts'], [-0.01, 0.5, 2, 5, 9, 13, 17, 21, 40])
    g = clean.groupby('bin', observed=True).agg(n=('act_carries', 'size'), proj=('rushing_attempts', 'mean'),
                                                act=('act_carries', 'mean'), played=('played', 'mean'))
    g['ratio'] = g['act'] / g['proj']
    print(g.round(3).to_string())

    print("\n=== by season-week bucket (structural vs early/late?) ===")
    clean['wb'] = pd.cut(clean['week'], [0, 5, 9, 13, 18], labels=['3-5', '6-9', '10-13', '14-17'])
    per_room = clean.groupby(keys + ['wb'], observed=True).agg(
        proj=('rushing_attempts', 'sum'), act=('act_carries', 'sum')).reset_index()
    per_room['err'] = per_room['proj'] - per_room['act']
    print(per_room.groupby(['wb', 'year'], observed=True)['err'].mean().unstack('year').round(2).to_string())

    print("\n=== RB points: is the carry error hidden by the calibration line? (clean rooms, played) ===")
    d = clean[clean['played']].copy()
    for a, b, lab in (('rushing_attempts', 'act_carries', 'carries'), ('rushing_yards', 'act_rush_yds', 'rush yards'),
                      ('rushing_tds', 'act_rush_tds', 'rush TDs'), ('raw_pts', 'act_pts', 'RAW points'),
                      ('pts', 'act_pts', 'CALIBRATED points')):
        print(f"  {lab:18s} proj {d[a].mean():7.2f}  actual {d[b].mean():7.2f}  bias {d[a].mean() - d[b].mean():+.2f}")
    print(f"  yards per carry: projected {d['rushing_yards'].sum() / d['rushing_attempts'].sum():.2f}  "
          f"actual {d['act_rush_yds'].sum() / d['act_carries'].sum():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2024,2025')
    ap.add_argument('--weeks', default='3-17')
    ap.add_argument('--out', default='.sweeps/diag_rb_carries')
    ap.add_argument('--features', default='')
    ap.add_argument('--drop', default='', help='comma-separated flags to REMOVE from the arm')
    ap.add_argument('--summarize', action='store_true')
    args = ap.parse_args()
    if args.summarize:
        pl = pd.read_parquet(os.path.join(args.out, 'rb_players.parquet'))
        tm = pd.read_parquet(os.path.join(args.out, 'rb_teams.parquet'))
    else:
        years = [int(y) for y in args.years.split(',')]
        if '-' in args.weeks:
            lo, hi = (int(x) for x in args.weeks.split('-'))
            weeks = list(range(lo, hi + 1))
        else:
            weeks = [int(x) for x in args.weeks.split(',')]
        pl, tm = collect(years, weeks, args.out, [f for f in args.features.split(',') if f],
                         [f for f in args.drop.split(',') if f])
    summarize(pl, tm)


if __name__ == '__main__':
    main()
