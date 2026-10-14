"""
Players back from missed games (2026-10-13). User: "start looking back into players coming back from missed games and how
we need to adjust for that." The live record (Track Record page, weeks 3-5) shows the model well under the market on
players with a short or empty 2026 log (Flowers, Bowers, Adams, Metcalf...), and scoring like the market said.

Uses the lab dump (the harness base-arm board, `scripts/calibration_lab.py`), re-applies the shipped late steps offline
(top-end yardage, points line, season anchor, Questionable WR cut) and splits the error by where each player stands in
his own season at projection time, from the box scores before that week:

    healthy     played his team's previous game (and has played 3+ games this season)
    back-1      missed exactly his team's previous game
    back-2-3    missed the previous 2-3 team games
    back-4+     missed the previous 4+ team games (not a season debut)
    debut       has not played yet this season (week 3+), i.e. missed every team game so far
    early       played every game but has fewer than 3 (weeks 3-4 for everyone)
    2nd-game    played the previous game, but it was his first game back after missing 2+

    python scripts/diag_returning_players.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.loaders import load_schedule  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402
from data.weekly_projections import (QUESTIONABLE_WORKLOAD_FACTORS, SEASON_ANCHOR_S, TOP_END_YARDAGE,  # noqa: E402
                                     SIDELINED_AVAILABILITY)
from scripts.eval_calibration_schemes import anchor, load_dump, shipped_line, top_n_mask  # noqa: E402

SEGMENTS = ('healthy', 'early', 'back-1', 'back-2-3', 'back-4+', 'debut', '2nd-game')


def shipped_board(d):
    """The dump's raw total with the shipped late steps re-applied: top-end yardage on the stat line (week 3+), the
    points line, the season anchor, the Questionable WR cut."""
    raw = d['raw'].to_numpy(float).copy()
    live = (d['avail'] > SIDELINED_AVAILABILITY).to_numpy() & (raw > 0.005) & (d['week'] >= 3).to_numpy()
    for stat, (positions, knot, share) in TOP_END_YARDAGE.items():
        p = d['p_' + stat].to_numpy(float)
        m = live & d['pos'].isin(positions).to_numpy() & (p > knot)
        new = np.where(m, p - share * (p - knot), p)
        raw += 0.1 * (new - p)
        d['p_' + stat] = new
    dd = d.copy()
    dd['raw'] = np.clip(raw, 0.0, None)
    shown = anchor(shipped_line(dd), dd, SEASON_ANCHOR_S)
    q = d['inj'].astype(str).str.lower().str.startswith('questionable').to_numpy()
    for pos, f in QUESTIONABLE_WORKLOAD_FACTORS.items():
        m = q & (d['pos'] == pos).to_numpy() & (d['avail'] > SIDELINED_AVAILABILITY).to_numpy()
        shown[m] = np.round(shown[m] * f, 2)
    return shown


def season_status(d, years):
    """Per dump row: team games before the week, games he played before it, consecutive missed team games right before
    it, and whether his previous game was itself a return from 2+ missed."""
    out = []
    for year in years:
        stats, team_col, name_col, _ = load_and_merge_data(int(year), 'Full PPR')
        wk = pd.to_numeric(stats['week'], errors='coerce')
        st = stats[(wk >= 1) & (wk <= 18)].copy()
        if 'season_type' in st.columns:
            st = st[st['season_type'].isna() | st['season_type'].astype(str).str.upper().eq('REG')]
        st['_key'] = clean_name_exact(st[name_col].astype(str))
        st['_week'] = pd.to_numeric(st['week'], errors='coerce').astype(int)
        played_weeks = st.groupby('_key')['_week'].apply(lambda s: set(s.tolist())).to_dict()
        sched = load_schedule(int(year))
        sched = sched[sched['game_type'] == 'REG'] if 'game_type' in sched.columns else sched
        team_weeks = {}
        for w, h, a in zip(sched['week'], sched['home_team'], sched['away_team']):
            team_weeks.setdefault(str(h), []).append(int(w))
            team_weeks.setdefault(str(a), []).append(int(w))
        team_weeks = {t: sorted(v) for t, v in team_weeks.items()}
        sub = d[d['year'] == year]
        keys = clean_name_exact(sub['player'].astype(str))
        for idx, key, team, week in zip(sub.index, keys, sub['team'].astype(str), sub['week']):
            tw = [w for w in team_weeks.get(team, []) if w < week]
            mine = played_weeks.get(key, set())
            played_before = [w for w in tw if w in mine]
            streak = 0
            for w in reversed(tw):
                if w in mine:
                    break
                streak += 1
            # was his most recent game a return from 2+ missed?
            prev_return = False
            if tw and tw[-1] in mine:
                before_last = tw[:-1]
                s2 = 0
                for w in reversed(before_last):
                    if w in mine:
                        break
                    s2 += 1
                prev_return = s2 >= 2 and len([w for w in before_last if w in mine]) > 0 or (
                    s2 >= 2 and s2 == len(before_last))
            out.append((idx, len(tw), len(played_before), streak, prev_return))
    s = pd.DataFrame(out, columns=['idx', 'team_games', 'games_before', 'missed_streak', 'prev_return']).set_index('idx')
    return s


def classify(row):
    if row.team_games == 0:
        return 'opening'
    if row.games_before == 0:
        return 'debut'
    if row.missed_streak >= 4:
        return 'back-4+'
    if row.missed_streak >= 2:
        return 'back-2-3'
    if row.missed_streak == 1:
        return 'back-1'
    if row.prev_return:
        return '2nd-game'
    if row.games_before < 3:
        return 'early'
    return 'healthy'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default='.sweeps/calibration_lab_v7.csv')
    ap.add_argument('--week1', default='.sweeps/calibration_lab_v7_wk1_ourlads.csv')
    ap.add_argument('--save', default='.sweeps/returning_players_rows.parquet')
    a = ap.parse_args()
    d = load_dump(a.dump, a.week1)
    d['shown'] = shipped_board(d)
    status = season_status(d, sorted(d['year'].unique()))
    d = d.join(status)
    d['segment'] = d.apply(classify, axis=1)
    d['start'] = top_n_mask(d, d['shown'].to_numpy())
    win = d['played'] & d['actual'].notna() & d['week'].between(3, 17)
    d.loc[win].to_parquet(a.save) if a.save else None

    pd.set_option('display.width', 230)
    for pool_name, pool in (('START pool', win & d['start']), ('all rows with a projection >= 5', win & (d['shown'] >= 5))):
        print(f'\n== {pool_name}, 2021-2025 weeks 3-17, played rows: shown - actual by segment ==')
        rows = []
        for pos in ('ALL', 'QB', 'RB', 'WR', 'TE'):
            for seg in SEGMENTS:
                m = pool & d['segment'].eq(seg) & (d['pos'].eq(pos) if pos != 'ALL' else True)
                s = d[m]
                if len(s) < 15:
                    continue
                e = s['shown'] - s['actual']
                by_year = s.assign(e=e).groupby('year')['e'].mean()
                rows.append({'pos': pos, 'segment': seg, 'n': len(s), 'shown': s['shown'].mean(), 'actual': s['actual'].mean(),
                             'bias': e.mean(), 't': e.mean() / (e.std() / np.sqrt(len(e))),
                             'ratio': s['actual'].sum() / s['shown'].sum(),
                             'years same sign': f"{int((np.sign(by_year) == np.sign(e.mean())).sum())}/{len(by_year)}"})
        print(pd.DataFrame(rows).round(2).to_string(index=False))

    print('\n== Where it comes from (START pool, actual / projected per stat) ==')
    pool = win & d['start']
    for pos, stats in (('WR', ('targets', 'receptions', 'receiving_yards', 'receiving_tds')),
                       ('TE', ('targets', 'receptions', 'receiving_yards', 'receiving_tds')),
                       ('RB', ('rushing_attempts', 'rushing_yards', 'targets', 'receiving_yards', 'rushing_tds'))):
        cells = []
        for seg in SEGMENTS:
            s = d[pool & d['segment'].eq(seg) & d['pos'].eq(pos)]
            if len(s) < 15:
                continue
            cells.append(f"{seg} (n={len(s)}): " + ', '.join(
                f"{st.split('_')[-1] if st != 'rushing_attempts' else 'carries'} {s['a_' + st].sum() / max(s['p_' + st].sum(), 1e-9):.2f}"
                for st in stats))
        print(f'  {pos}: ' + ' | '.join(cells))


if __name__ == '__main__':
    main()
