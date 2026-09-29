"""
How the vacancy allocator behaves once the backtest's injury replay is
complete (v2_historical_reserve_replay adds IR/PUP/NFI/suspended players,
who never appear on the weekly injury report), split by how long the OUT
player had already been gone.

Vacancy hands out the OUT player's FULL projected volume, which comes from
his last appearances. For a fresh absence that's the right amount. For a
player several weeks into IR, his teammates' own recency-weighted rates
already include the games without him, so handing them his full volume
again would double-count. This measures whether that happens.

Arms (both on DEFAULT_FEATURES + v2_historical_injury_replay):
    A = report-only replay (every replay-on backtest so far)
    B = A + v2_historical_reserve_replay

A room's absence class comes from its OUT sources: 'fresh' = every source
played his team's previous game, 'extended' = none did, 'mixed' otherwise.

Usage:
    python scripts/diag_vacancy_absence_length.py --years 2024,2025 --weeks 3-17 --out .sweeps/diag_absence
    python scripts/diag_vacancy_absence_length.py --summarize --out .sweeps/diag_absence
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPLAY = 'v2_historical_injury_replay'
RESERVE = 'v2_historical_reserve_replay'
ROOM_VOLUME = {'WR/TE': ('targets',), 'RB': ('rushing_attempts', 'targets')}


def _room(pos):
    return 'RB' if pos == 'RB' else ('WR/TE' if pos in ('WR', 'TE') else None)


def _games_missed(stats_df, name_col, week):
    """{player: team games missed in a row before `week`} - 0 if he played
    his team's most recent game, the whole season so far if he never
    appeared. Team games come from game_team so a trade can't blur them."""
    s = stats_df.copy()
    s['_w'] = pd.to_numeric(s['week'], errors='coerce')
    s = s[s['_w'] < week]
    team = (s['game_team'] if 'game_team' in s.columns else s['team']).astype(str).str.upper()
    team_weeks = s.assign(_t=team).groupby('_t')['_w'].apply(lambda w: sorted(set(w))).to_dict()
    last = s.assign(_t=team).sort_values('_w').groupby(name_col).agg(last_w=('_w', 'last'), t=('_t', 'last'))

    def missed(player, board_team):
        weeks = team_weeks.get(str(board_team).upper(), [])
        if player not in last.index:
            return len(weeks)
        lw = last.at[player, 'last_w']
        return int(sum(1 for w in weeks if w > lw))
    return missed


def collect(years, weeks, out_dir):
    from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES
    from data.transforms import load_and_merge_data

    arms = {'A': frozenset(DEFAULT_FEATURES | {REPLAY}),
            'B': frozenset(DEFAULT_FEATURES | {REPLAY, RESERVE})}
    player_rows, room_rows = [], []
    for year in years:
        stats_df, team_col, name_col, _ = load_and_merge_data(year, 'Full PPR')
        s = stats_df.copy()
        s['_w'] = pd.to_numeric(s['week'], errors='coerce')
        for c in ('targets', 'rushing_attempts', 'fantasy_points_ppr'):
            s[c] = pd.to_numeric(s.get(c), errors='coerce')
        s['_team'] = (s['game_team'] if 'game_team' in s.columns else s[team_col]).astype(str).str.upper()
        s['room'] = s['position'].astype(str).map(_room)
        for week in weeks:
            wk = s[s['_w'] == week]
            if wk.empty:
                continue
            act_player = wk.groupby(name_col).agg(act_pts=('fantasy_points_ppr', 'sum'),
                                                  act_targets=('targets', 'sum'),
                                                  act_carries=('rushing_attempts', 'sum'))
            act_room = wk.dropna(subset=['room']).groupby(['_team', 'room'])[['targets', 'rushing_attempts']].sum()
            missed = _games_missed(stats_df, name_col, week)
            for arm, feats in arms.items():
                proj, meta = build_weekly_projections(
                    year, week, 'Full PPR', as_of_week=week, apply_injury=False, features=feats)
                if proj.empty:
                    continue
                p = proj[['Player', 'Pos', 'Team', 'Model Proj Pts']].copy()
                for c in ('targets', 'rushing_attempts', 'Availability'):
                    p[c] = pd.to_numeric(proj[c], errors='coerce') if c in proj.columns else np.nan
                p['Team'] = p['Team'].astype(str).str.upper()
                p['room'] = p['Pos'].map(_room)
                p = p[p['room'].notna()]
                # OUT sources per room, with how long each had been gone.
                sources = p[p['Availability'] <= 0.01]
                src_missed = {}
                for _, r in sources.iterrows():
                    src_missed.setdefault((r['Team'], r['room']), []).append(missed(r['Player'], r['Team']))
                gain = {}
                for e in meta.get('vacancy_ledger') or []:
                    for rc in e.get('recipients') or []:
                        k = (str(e.get('team', '')).upper(), rc['player'])
                        gain[k] = gain.get(k, 0.0) + float(rc['allocated'])

                def cls(key):
                    m = src_missed.get(key)
                    if not m:
                        return 'clean'
                    if all(x == 0 for x in m):
                        return 'fresh'
                    if all(x >= 1 for x in m):
                        return 'extended'
                    return 'mixed'
                p['room_class'] = [cls((t, r)) for t, r in zip(p['Team'], p['room'])]
                p['src_max_missed'] = [max(src_missed.get((t, r), [0])) for t, r in zip(p['Team'], p['room'])]
                p['vacancy_gain'] = [gain.get((t, pl), 0.0) for t, pl in zip(p['Team'], p['Player'])]
                p = p.join(act_player, on='Player')
                p['year'], p['week'], p['arm'] = year, week, arm
                player_rows.append(p)
                live = p[p['Availability'] > 0.01]
                for (team, room), g in live.groupby(['Team', 'room']):
                    rec = {'year': year, 'week': week, 'arm': arm, 'team': team, 'room': room,
                           'room_class': cls((team, room)),
                           'src_max_missed': max(src_missed.get((team, room), [0])),
                           'n_sources': len(src_missed.get((team, room), []))}
                    for vol in ROOM_VOLUME[room]:
                        rec[f'proj_{vol}'] = float(g[vol].sum())
                        rec[f'act_{vol}'] = (float(act_room.loc[(team, room), vol])
                                            if (team, room) in act_room.index else np.nan)
                    room_rows.append(rec)
            print(f"{year} w{week} done", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    players = pd.concat(player_rows, ignore_index=True)
    rooms = pd.DataFrame(room_rows)
    players.to_parquet(os.path.join(out_dir, 'players.parquet'), index=False)
    rooms.to_parquet(os.path.join(out_dir, 'rooms.parquet'), index=False)
    return players, rooms


def summarize(players, rooms):
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 30)
    rooms = rooms.copy()
    rooms['gone'] = pd.cut(rooms['src_max_missed'], [-1, 0, 1, 3, 99], labels=['0', '1', '2-3', '4+'])

    print("=== ROOM volume, projected - actual, per team-week ===")
    for room, vols in ROOM_VOLUME.items():
        room_rows = rooms[rooms['room'] == room]
        for vol in vols:
            r = room_rows.assign(err=room_rows[f'proj_{vol}'] - room_rows[f'act_{vol}']).dropna(subset=['err'])
            g = r.groupby(['room_class', 'arm']).agg(n=('err', 'size'), proj=(f'proj_{vol}', 'mean'),
                                                     act=(f'act_{vol}', 'mean'), err=('err', 'mean'))
            print(f"\n{room} {vol}:")
            print(g.round(2).to_string())
            ext = r[(r['arm'] == 'B') & (r['room_class'] != 'clean')]
            if not ext.empty:
                print(f"  arm B OUT rooms by games the source had already missed:")
                print(ext.groupby('gone', observed=True).agg(n=('err', 'size'), err=('err', 'mean')).round(2).to_string())

    print("\n=== PLAYERS who played, fantasy-point bias / RMSE by room class ===")
    p = players[(players['Availability'] > 0.01) & players['act_pts'].notna()].copy()
    p['err'] = p['Model Proj Pts'] - p['act_pts']
    p['is_recipient'] = p['vacancy_gain'] > 0
    g = p.groupby(['room', 'room_class', 'is_recipient', 'arm']).agg(
        n=('err', 'size'), bias=('err', 'mean'), rmse=('err', lambda e: float(np.sqrt((e ** 2).mean()))),
        gain=('vacancy_gain', 'mean'))
    print(g.round(3).to_string())

    print("\n=== arm A vs B on the SAME scored player-weeks (players who played) ===")
    keys = ['year', 'week', 'Player', 'Team']
    a = p[p['arm'] == 'A'].set_index(keys)
    b = p[p['arm'] == 'B'].set_index(keys)
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    for room in ('RB', 'WR/TE'):
        m = a['room'] == room
        ea, eb = a.loc[m, 'err'], b.loc[m, 'err']
        changed = (a.loc[m, 'Model Proj Pts'] - b.loc[m, 'Model Proj Pts']).abs() > 0.05
        print(f"  {room}: n={int(m.sum())}  changed={int(changed.sum())}   "
              f"RMSE {np.sqrt((ea ** 2).mean()):.3f} -> {np.sqrt((eb ** 2).mean()):.3f}   "
              f"bias {ea.mean():+.3f} -> {eb.mean():+.3f}   | changed only: RMSE "
              f"{np.sqrt((ea[changed] ** 2).mean()):.3f} -> {np.sqrt((eb[changed] ** 2).mean()):.3f}  "
              f"bias {ea[changed].mean():+.3f} -> {eb[changed].mean():+.3f}")
    bb = b.copy()
    bb['gone'] = pd.cut(bb['src_max_missed'], [-1, 0, 1, 3, 99], labels=['0', '1', '2-3', '4+'])
    rec = bb[(bb['vacancy_gain'] > 0)]
    print("\n  arm B vacancy recipients by games the source had already missed:")
    print(rec.groupby(['room', 'gone'], observed=True).agg(
        n=('err', 'size'), bias=('err', 'mean'), gain=('vacancy_gain', 'mean')).round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2024,2025')
    ap.add_argument('--weeks', default='3-17')
    ap.add_argument('--out', default='.sweeps/diag_absence')
    ap.add_argument('--summarize', action='store_true')
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
