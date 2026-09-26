"""
Diagnostic for the pass-capacity / vacancy double-count (2026-09-25).

Two fixes for the double-count (v2_vacancy_before_capacity and
v2_pass_capacity_injury_neutral_claim) each tripped the harness's bias-growth
cap by the same ~0.35, through completely different mechanisms. This script
answers WHERE that volume actually belongs, which the harness's aggregate
verdict cannot:

  * TEAM level: on a team-week with an OUT pass catcher, does the shipped
    (double-counted) board's projected RB / WR+TE target total land above or
    below what the team actually threw to those rooms? And on a team-week
    with nobody out? If the shipped board is right on OUT weeks only because
    the capacity budget is too tight EVERYWHERE, the double-count is masking
    a budget problem, and fixing the budget is the real fix.
  * PLAYER level: where the fantasy-point bias moves between the two arms -
    the direct vacancy recipient, other teammates in an OUT room, or clean
    rooms.

Arms (both with v2_historical_injury_replay, the precondition that makes
vacancy fire in a backtest at all):
    base = DEFAULT_FEATURES
    fix  = DEFAULT_FEATURES + v2_pass_capacity_injury_neutral_claim

Writes one parquet of player rows and one of team-room rows; the summary is
printed at the end and can be re-printed from the parquet with --summarize.

Usage:
    python scripts/diag_vacancy_double_count.py --years 2022,2023,2024,2025 --weeks 3-17 --out .sweeps/diag_dc
    python scripts/diag_vacancy_double_count.py --summarize --out .sweeps/diag_dc
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPLAY = 'v2_historical_injury_replay'
FIX = 'v2_pass_capacity_injury_neutral_claim'


def _room(pos):
    return 'RB' if pos == 'RB' else ('WR/TE' if pos in ('WR', 'TE') else None)


def _actual_frame(stats_df, name_col, team_col, week):
    rows = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week].copy()
    team = rows['game_team'] if 'game_team' in rows.columns else rows[team_col]
    rows['_team'] = team.astype(str).str.upper()
    pos_col = 'position' if 'position' in rows.columns else 'Pos'
    keep = {
        'Player': rows[name_col].astype(str),
        'act_team': rows['_team'],
        'act_pos': rows[pos_col].astype(str),
        'act_pts': pd.to_numeric(rows.get('fantasy_points_ppr'), errors='coerce'),
        'act_targets': pd.to_numeric(rows.get('targets'), errors='coerce'),
        'act_pass_att': pd.to_numeric(rows.get('attempts', rows.get('passing_attempts')), errors='coerce'),
    }
    return pd.DataFrame(keep)


def collect(years, weeks, out_dir):
    from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES
    from data.transforms import load_and_merge_data

    arms = {
        'base': frozenset(DEFAULT_FEATURES | {REPLAY}),
        'fix': frozenset(DEFAULT_FEATURES | {REPLAY, FIX}),
    }
    player_rows, room_rows = [], []
    for year in years:
        stats_df, team_col, name_col, _ = load_and_merge_data(year, 'Full PPR')
        for week in weeks:
            act = _actual_frame(stats_df, name_col, team_col, week)
            if act.empty:
                continue
            act_by_player = act.groupby('Player').agg(
                act_pts=('act_pts', 'sum'), act_targets=('act_targets', 'sum'))
            # Actual team-room target totals from EVERY player who caught a
            # target, not just the ones on the model's board - a team's real
            # throw count doesn't care who the model projected.
            act['room'] = act['act_pos'].map(_room)
            act_rooms = (act.dropna(subset=['room'])
                         .groupby(['act_team', 'room'])['act_targets'].sum())
            act_team_att = (act[act['act_pos'] == 'QB'].groupby('act_team')['act_pass_att'].sum())

            boards = {}
            for arm, feats in arms.items():
                proj, meta = build_weekly_projections(
                    year, week, 'Full PPR', as_of_week=week, apply_injury=False, features=feats)
                if proj.empty:
                    continue
                boards[arm] = (proj, meta)
            if len(boards) < 2:
                print(f"{year} w{week}: missing an arm, skipped", flush=True)
                continue

            base_proj, base_meta = boards['base']
            # OUT sources per team/room and the direct top vacancy recipient,
            # from the shipped arm's own vacancy ledger (identical sources in
            # both arms - the replay feed doesn't depend on the flag).
            out_by_room, vac_top, vac_gain = {}, {}, {}
            for entry in base_meta.get('vacancy_ledger') or []:
                if entry.get('volume') != 'targets':
                    continue
                team = str(entry.get('team', '')).upper()
                role = str(entry.get('functional_source_role', ''))
                room = 'RB' if 'RB' in role else ('WR/TE' if role in ('WR', 'TE') else None)
                if room is None:
                    continue
                key = (team, room)
                out_by_room[key] = out_by_room.get(key, 0.0) + float(entry.get('vacated') or 0.0)
                recips = entry.get('recipients') or []
                for r in recips:
                    vac_gain[(team, r['player'])] = vac_gain.get((team, r['player']), 0.0) + float(r['allocated'])
                if recips:
                    top = max(recips, key=lambda r: r['allocated'])
                    vac_top.setdefault(key, set()).add(top['player'])

            for arm, (proj, _meta) in boards.items():
                p = proj[['Player', 'Pos', 'Team', 'Model Proj Pts']].copy()
                for col in ('targets', 'passing_attempts', 'Availability', 'Raw Model Proj Pts'):
                    p[col] = pd.to_numeric(proj[col], errors='coerce') if col in proj.columns else np.nan
                p['Team'] = p['Team'].astype(str).str.upper()
                p['room'] = p['Pos'].map(_room)
                p['year'], p['week'], p['arm'] = year, week, arm
                p = p.join(act_by_player, on='Player')
                p['room_has_out'] = [bool(out_by_room.get((t, r), 0.0) > 0) for t, r in zip(p['Team'], p['room'])]
                p['room_vacated'] = [out_by_room.get((t, r), 0.0) for t, r in zip(p['Team'], p['room'])]
                p['is_top_recipient'] = [pl in vac_top.get((t, r), set()) for pl, t, r in
                                         zip(p['Player'], p['Team'], p['room'])]
                p['vacancy_gain_base'] = [vac_gain.get((t, pl), 0.0) for t, pl in zip(p['Team'], p['Player'])]
                player_rows.append(p)

                cap = {}
                for entry in _meta.get('pass_capacity_ledger') or []:
                    grp = entry.get('position_group')
                    if grp in ('RB', 'WR/TE', 'WR', 'TE') and entry.get('capacity') is not None:
                        room = 'RB' if grp == 'RB' else 'WR/TE'
                        k = (str(entry.get('team', '')).upper(), room)
                        cap[k] = cap.get(k, 0.0) + float(entry['capacity'])
                catchers = p[p['room'].notna()]
                proj_rooms = catchers.groupby(['Team', 'room'])['targets'].sum()
                qb_att = p[p['Pos'] == 'QB'].groupby('Team')['passing_attempts'].sum()
                for (team, room), proj_tgt in proj_rooms.items():
                    room_rows.append({
                        'year': year, 'week': week, 'arm': arm, 'team': team, 'room': room,
                        'proj_targets': float(proj_tgt),
                        'budget': cap.get((team, room), np.nan),
                        'act_targets': float(act_rooms.get((team, room), np.nan)),
                        'proj_qb_att': float(qb_att.get(team, np.nan)),
                        'act_qb_att': float(act_team_att.get(team, np.nan)),
                        'room_has_out': bool(out_by_room.get((team, room), 0.0) > 0),
                        'room_vacated': out_by_room.get((team, room), 0.0),
                    })
            print(f"{year} w{week} done", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    players = pd.concat(player_rows, ignore_index=True)
    rooms = pd.DataFrame(room_rows)
    players.to_parquet(os.path.join(out_dir, 'players.parquet'), index=False)
    rooms.to_parquet(os.path.join(out_dir, 'rooms.parquet'), index=False)
    return players, rooms


def summarize(players, rooms):
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 30)

    print("\n=== TEAM-ROOM TARGET TOTALS: projected vs actual (mean per team-week) ===")
    r = rooms.dropna(subset=['act_targets']).copy()
    r['err'] = r['proj_targets'] - r['act_targets']
    r['budget_err'] = r['budget'] - r['act_targets']
    g = r.groupby(['room', 'room_has_out', 'arm']).agg(
        n=('err', 'size'), proj=('proj_targets', 'mean'), budget=('budget', 'mean'),
        actual=('act_targets', 'mean'), proj_minus_actual=('err', 'mean'),
        budget_minus_actual=('budget_err', 'mean'), vacated=('room_vacated', 'mean'))
    print(g.round(2).to_string())

    print("\n=== QB pass attempts: projected vs actual (base arm, per team-week) ===")
    q = rooms[(rooms['arm'] == 'base')].drop_duplicates(['year', 'week', 'team']).dropna(
        subset=['proj_qb_att', 'act_qb_att'])
    q = q[q['proj_qb_att'] > 0]
    print(f"  n={len(q)}  proj {q['proj_qb_att'].mean():.2f}  actual {q['act_qb_att'].mean():.2f}  "
          f"proj-actual {(q['proj_qb_att'] - q['act_qb_att']).mean():+.2f}")

    print("\n=== PLAYER fantasy-point bias (pred - actual), players who played ===")
    p = players.dropna(subset=['act_pts']).copy()
    p = p[p['room'].notna()]
    p['err'] = p['Model Proj Pts'] - p['act_pts']
    p['tgt_err'] = p['targets'] - p['act_targets']
    p['cat'] = np.where(~p['room_has_out'], 'clean room',
                        np.where(p['is_top_recipient'], 'OUT room: top vacancy recipient',
                                 'OUT room: other teammate'))
    g = p.groupby(['cat', 'room', 'arm']).agg(
        n=('err', 'size'), pred=('Model Proj Pts', 'mean'), actual=('act_pts', 'mean'),
        bias=('err', 'mean'), rmse=('err', lambda e: float(np.sqrt((e ** 2).mean()))),
        tgt_pred=('targets', 'mean'), tgt_act=('act_targets', 'mean'), tgt_bias=('tgt_err', 'mean'))
    print(g.round(3).to_string())

    # Startable-ish slice: players the base arm projected >= 8 pts, the
    # population the harness's START pools are drawn from.
    keys = ['year', 'week', 'Player', 'Team']
    start_keys = p[(p['arm'] == 'base') & (p['Model Proj Pts'] >= 8.0)][keys]
    ps = p.merge(start_keys, on=keys)
    print("\n=== same, base-arm projection >= 8 pts ===")
    g = ps.groupby(['cat', 'room', 'arm']).agg(
        n=('err', 'size'), bias=('err', 'mean'),
        rmse=('err', lambda e: float(np.sqrt((e ** 2).mean()))),
        tgt_pred=('targets', 'mean'), tgt_act=('act_targets', 'mean'), tgt_bias=('tgt_err', 'mean'))
    print(g.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2022,2023,2024,2025')
    ap.add_argument('--weeks', default='3-17')
    ap.add_argument('--out', default='.sweeps/diag_dc')
    ap.add_argument('--summarize', action='store_true', help='re-print the summary from saved parquet')
    args = ap.parse_args()
    if args.summarize:
        players = pd.read_parquet(os.path.join(args.out, 'players.parquet'))
        rooms = pd.read_parquet(os.path.join(args.out, 'rooms.parquet'))
    else:
        years = [int(y) for y in args.years.split(',')]
        lo, hi = (int(x) for x in args.weeks.split('-'))
        players, rooms = collect(years, list(range(lo, hi + 1)), args.out)
    summarize(players, rooms)


if __name__ == '__main__':
    main()
