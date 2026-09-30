"""
How predictable is a team's RB carry total, and would a team carry budget
beat the model's own room sum?

Builds one row per team-game (weeks 3-17) with only information known before
kickoff:
    plays      offensive plays / game (attempts + carries + sacks), this season
               to date, shrunk toward last season
    rb_rate    RB carries / offensive plays, same shrinkage
    rb_cpg     RB carries / game, same shrinkage (plays x rb_rate, directly)
    spread     closing spread from the team's side (+ = favored)
    total      closing total
    opp_pace   opponent's defensive plays faced / game (shrunk)
    opp_rb     opponent's RB carries allowed / game (shrunk)
and fits linear budgets on 2019-2022, scoring 2023-2025 out of sample.

Then, on saved boards (diag_rb_carry_overprojection.py rb_teams.parquet), it
compares the board's RB room sum, the budget, and a cap rule
(room -> budget when |room - budget| > DEADBAND) against actual RB carries.

Usage:
    python scripts/diag_rb_carry_budget.py --boards .sweeps/diag_rb_carries_rbpart/rb_teams.parquet
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

K_SHRINK = 4.0          # prior-season weight, in games
TRAIN = (2019, 2020, 2021, 2022)
TEST = (2023, 2024, 2025)


def team_games(year):
    """(team, week) rows: plays, rb carries, opponent, spread/total from the team's side."""
    import nflreadpy
    from data.transforms import load_and_merge_data
    from data.loaders import load_schedule
    ts = nflreadpy.load_team_stats([year], summary_level='week').to_pandas()
    ts = ts[ts['season_type'] == 'REG'] if 'season_type' in ts.columns else ts
    ts['plays'] = ts[['attempts', 'carries', 'sacks_suffered']].sum(axis=1)
    g = ts[['team', 'opponent_team', 'week', 'plays', 'carries']].copy()
    s, tc, nc, _ = load_and_merge_data(year, 'Full PPR')
    s = s.copy()
    s['team_'] = (s['game_team'] if 'game_team' in s.columns else s[tc]).astype(str).str.upper()
    s['week'] = pd.to_numeric(s['week'], errors='coerce')
    rb = (s[s['position'].astype(str) == 'RB'].groupby(['team_', 'week'])['rushing_attempts'].sum()
          .rename('rb_carries').reset_index().rename(columns={'team_': 'team'}))
    g = g.merge(rb, on=['team', 'week'], how='left')
    g['rb_carries'] = g['rb_carries'].fillna(0.0)
    sc = load_schedule(year)
    sc = sc[['week', 'home_team', 'away_team', 'spread_line', 'total_line']].copy()
    home = sc.rename(columns={'home_team': 'team'}).assign(spread=sc['spread_line'])
    away = sc.rename(columns={'away_team': 'team'}).assign(spread=-sc['spread_line'])
    lines = pd.concat([home[['week', 'team', 'spread', 'total_line']], away[['week', 'team', 'spread', 'total_line']]])
    g = g.merge(lines, on=['team', 'week'], how='left').rename(columns={'total_line': 'total'})
    g['year'] = year
    return g


def season_to_date(g, prior):
    """Shrunk to-date rates for every (team, week) from games before that week."""
    rows = []
    pri_off = prior.groupby('team').agg(p_plays=('plays', 'mean'), p_rb=('rb_carries', 'mean'))
    pri_off['p_rate'] = pri_off['p_rb'] / pri_off['p_plays']
    pri_def = prior.groupby('opponent_team').agg(p_dplays=('plays', 'mean'), p_drb=('rb_carries', 'mean'))
    lg = prior[['plays', 'rb_carries']].mean()
    for (team, week), r in g.set_index(['team', 'week']).iterrows():
        before = g[(g['team'] == team) & (g['week'] < week)]
        opp = r['opponent_team']
        opp_def = g[(g['opponent_team'] == opp) & (g['week'] < week)]
        n, nd = len(before), len(opp_def)
        po = pri_off.loc[team] if team in pri_off.index else pd.Series({'p_plays': lg['plays'], 'p_rb': lg['rb_carries'],
                                                                          'p_rate': lg['rb_carries'] / lg['plays']})
        pdf = pri_def.loc[opp] if opp in pri_def.index else pd.Series({'p_dplays': lg['plays'], 'p_drb': lg['rb_carries']})
        plays = (before['plays'].sum() + K_SHRINK * po['p_plays']) / (n + K_SHRINK)
        rb_cpg = (before['rb_carries'].sum() + K_SHRINK * po['p_rb']) / (n + K_SHRINK)
        opp_pace = (opp_def['plays'].sum() + K_SHRINK * pdf['p_dplays']) / (nd + K_SHRINK)
        opp_rb = (opp_def['rb_carries'].sum() + K_SHRINK * pdf['p_drb']) / (nd + K_SHRINK)
        raw_cpg = before['rb_carries'].mean() if n else np.nan
        rows.append({'team': team, 'week': week, 'n': n, 'plays': plays, 'rb_cpg': rb_cpg,
                     'rb_rate': rb_cpg / plays, 'opp_pace': opp_pace, 'opp_rb': opp_rb, 'raw_cpg': raw_cpg,
                     'y': r['rb_carries'], 'spread': r['spread'], 'total': r['total']})
    out = pd.DataFrame(rows)
    out['lg_plays'], out['lg_rb'] = lg['plays'], lg['rb_carries']
    return out


def build(years):
    frames = []
    cache = {}
    for year in years:
        for y in (year - 1, year):
            if y not in cache:
                cache[y] = team_games(y)
        d = season_to_date(cache[year], cache[year - 1])
        d['year'] = year
        frames.append(d[(d['week'] >= 3) & (d['week'] <= 17)])
    return pd.concat(frames, ignore_index=True).dropna(subset=['spread', 'total'])


def design(d, kind):
    base = d['rb_cpg'].to_numpy()
    cols = {
        'trailing rb carries/game': [base],
        '+ spread': [base, d['spread']],
        '+ spread + opp pace': [base, d['spread'], d['opp_pace'] - d['lg_plays']],
        '+ spread + opp pace + opp rb allowed': [base, d['spread'], d['opp_pace'] - d['lg_plays'], d['opp_rb'] - d['lg_rb']],
        '+ all + total': [base, d['spread'], d['opp_pace'] - d['lg_plays'], d['opp_rb'] - d['lg_rb'], d['total'] - 44.0],
        '+ spread + opp rb allowed': [base, d['spread'], d['opp_rb'] - d['lg_rb']],
        '+ spread(clip 7) + opp rb allowed': [base, d['spread'].clip(-7, 7), d['opp_rb'] - d['lg_rb']],
        'plays + rb rate + spread(clip 7) + opp rb': [d['plays'] - d['lg_plays'], d['rb_rate'] - d['lg_rb'] / d['lg_plays'],
                                                      d['spread'].clip(-7, 7), d['opp_rb'] - d['lg_rb']],
    }[kind]
    return np.column_stack([np.ones(len(d))] + [np.asarray(c, float) for c in cols])


KINDS = ['trailing rb carries/game', '+ spread', '+ spread + opp pace', '+ spread + opp pace + opp rb allowed',
         '+ all + total', '+ spread + opp rb allowed', '+ spread(clip 7) + opp rb allowed',
         'plays + rb rate + spread(clip 7) + opp rb']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boards', default=None, help='rb_teams.parquet from diag_rb_carry_overprojection.py')
    ap.add_argument('--deadband', type=float, default=1.0)
    ap.add_argument('--k', type=float, default=K_SHRINK, help='prior-season weight in games')
    ap.add_argument('--kind', default='+ spread(clip 7) + opp rb allowed', help='budget form used for the board comparison')
    args = ap.parse_args()
    globals()['K_SHRINK'] = args.k
    pd.set_option('display.width', 200)
    tr, te = build(TRAIN), build(TEST)
    print(f"team-games: train {len(tr)}  test {len(te)}  | mean RB carries train {tr['y'].mean():.2f} test {te['y'].mean():.2f}")
    print(f"naive: raw season-to-date mean  OOS RMSE {np.sqrt(((te['raw_cpg'] - te['y']) ** 2).mean()):.3f}")
    fits = {}
    for kind in KINDS:
        X, Xt = design(tr, kind), design(te, kind)
        b, *_ = np.linalg.lstsq(X, tr['y'].to_numpy(float), rcond=None)
        p = Xt @ b
        fits[kind] = b
        e = p - te['y']
        print(f"{kind:40s} OOS RMSE {np.sqrt((e ** 2).mean()):.3f}  corr {np.corrcoef(p, te['y'])[0, 1]:.3f}  bias {e.mean():+.2f}  coef {np.round(b, 3)}")

    # Game-script shape: residual of the trailing-rate model by spread bucket.
    b0 = fits['trailing rb carries/game']
    both = pd.concat([tr, te], ignore_index=True)
    both['resid'] = both['y'] - design(both, 'trailing rb carries/game') @ b0
    both['sb'] = pd.cut(both['spread'], [-30, -9.5, -6.5, -3.5, -1, 1, 3.5, 6.5, 9.5, 30])
    print("\n--- RB carries vs spread (+ = favored), residual of trailing rate ---")
    print(both.groupby('sb', observed=True).agg(n=('y', 'size'), actual=('y', 'mean'), resid=('resid', 'mean')).round(2).to_string())
    both['ob'] = pd.qcut(both['opp_pace'], 5)
    print("\n--- by opponent defensive plays faced / game ---")
    print(both.groupby('ob', observed=True).agg(n=('y', 'size'), resid=('resid', 'mean')).round(2).to_string())

    if not args.boards:
        return
    bd = pd.read_parquet(args.boards)
    bd = bd.rename(columns={'Team': 'team'})
    bd['team'] = bd['team'].astype(str).str.upper()
    kind = args.kind
    j = bd.merge(te.assign(budget=design(te, kind) @ fits[kind]), on=['year', 'week', 'team'], how='inner')
    j = j[j['proj_rush_rb'] > 0]
    y = j['act_rush_rb'].to_numpy(float)
    room, bud = j['proj_rush_rb'].to_numpy(float), j['budget'].to_numpy(float)
    print(f"\n=== boards vs budget ({len(j)} team-weeks, all rooms) | actual RB carries {y.mean():.2f} ===")
    for lab, p in (('board room sum', room), ('budget', bud)):
        print(f"  {lab:28s} mean {p.mean():.2f}  RMSE {np.sqrt(((p - y) ** 2).mean()):.3f}  corr {np.corrcoef(p, y)[0, 1]:.3f}")
    for w in (0.25, 0.5, 0.75):
        p = w * bud + (1 - w) * room
        print(f"  blend {w:.2f} budget          mean {p.mean():.2f}  RMSE {np.sqrt(((p - y) ** 2).mean()):.3f}")
    for db in (0.0, 0.5, 1.0, 2.0):
        gap = room - bud
        p = np.where(np.abs(gap) > db, bud + np.sign(gap) * db, room)
        moved = (np.abs(gap) > db).mean()
        pc = np.where(gap > db, bud + db, room)
        print(f"  cap deadband {db:.1f} symmetric  mean {p.mean():.2f}  RMSE {np.sqrt(((p - y) ** 2).mean()):.3f}  moved {moved:.0%}"
              f"  | down-only mean {pc.mean():.2f} RMSE {np.sqrt(((pc - y) ** 2).mean()):.3f}")
    print(f"  room over budget: mean gap {np.mean(room - bud):+.2f}, share of rooms over by >1: {np.mean(room - bud > 1):.0%}")


if __name__ == '__main__':
    main()
