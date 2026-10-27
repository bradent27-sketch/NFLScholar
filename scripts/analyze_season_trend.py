"""Does the league lean from passing to running as the season goes on, how much of it is weather, and does the board
follow it? (2026-10-27, docs/season_trend_2026-10-27.md.)

Part 1 - league, from the play-by-play cache (REG, pass/run plays, no kneels/spikes), one row per team-game:
  pass attempts (no sacks), designed rushes (no scrambles), neutral pass rate (wp 0.2-0.8, quarters 1-3, downs 1-2,
  >2 min left in the half), mean pass_oe. By period, slope per week (season fixed effects, bootstrap over games) split
  outdoor / indoor (dome or closed roof), and outdoor games regressed on week + temperature + wind with team-season
  fixed effects.
Part 2 - the board (--boards, variant 'base'): projected minus actual by period and game temperature, team totals and
  per position (played rows, projected >= 5 points), with the week slope of each bias.
Part 3 - two fast screens, each fitted on two seasons and scored on the other two: an outdoor cold-game RB carries lift
  (x 1 + k*max(0, T0 - temp)) and a TE1 target lift that grows with the week (x 1 + c*max(0, week - w0)).

    python scripts/analyze_season_trend.py --boards .sweeps/harness_boards/audit_share_partial_2022-2025_wk3-17.parquet
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.loaders import load_year_data  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402

PBP_COLS = ['game_id', 'season_type', 'week', 'posteam', 'qtr', 'down', 'play_type', 'pass', 'rush', 'qb_scramble', 'sack',
            'pass_attempt', 'qb_kneel', 'qb_spike', 'wp', 'pass_oe', 'roof', 'temp', 'wind', 'half_seconds_remaining',
            'passing_yards']
STATS = ['fantasy_points_ppr', 'passing_attempts', 'passing_yards', 'targets', 'receiving_yards', 'rushing_attempts',
         'rushing_yards']


def team_games(years):
    out = []
    for y in years:
        p = pd.read_parquet(f'data/cache/pbp_{y}.parquet', columns=PBP_COLS)
        p = p[(p.season_type == 'REG') & p.posteam.notna() & p.play_type.isin(['pass', 'run'])
              & (p.qb_kneel != 1) & (p.qb_spike != 1)]
        neutral = (p.wp.between(0.2, 0.8) & p.qtr.isin([1, 2, 3]) & p.down.isin([1, 2])
                   & (p.half_seconds_remaining > 120))
        p = p.assign(is_pass=(p['pass'] == 1).astype(float),
                     att=((p.pass_attempt == 1) & (p.sack != 1)).astype(float),
                     rushes=((p.rush == 1) & (p.qb_scramble != 1)).astype(float))
        p['n_pass'] = p.is_pass.where(neutral)
        g = p.groupby(['game_id', 'posteam'], as_index=False).agg(
            week=('week', 'first'), dropbacks=('is_pass', 'sum'), att=('att', 'sum'), rushes=('rushes', 'sum'),
            neutral_pass=('n_pass', 'mean'), proe=('pass_oe', 'mean'), roof=('roof', 'first'), temp=('temp', 'first'),
            wind=('wind', 'first'), pass_yds=('passing_yards', 'sum'))
        out.append(g.assign(year=y))
    g = pd.concat(out, ignore_index=True)
    g['week'] = g.week.astype(int)
    g['indoor'] = g.roof.isin(['dome', 'closed'])
    g['Team'] = g.posteam.replace({'LA': 'LAR'})
    return g


def week_slope(df, col, x='week', group='year'):
    d = df[[col, x, group]].dropna()
    yv = d[col] - d.groupby(group)[col].transform('mean')
    xv = d[x] - d.groupby(group)[x].transform('mean')
    return float((xv * yv).sum() / (xv * xv).sum())


def boot(df, fn, cluster=None, n=300, seed=7):
    rng = np.random.default_rng(seed)
    if cluster is None:
        draws = [fn(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(n)]
    else:
        codes, uniq = pd.factorize(df[cluster])
        rows = pd.Series(np.arange(len(df))).groupby(codes).apply(np.asarray).tolist()
        draws = [fn(df.iloc[np.concatenate([rows[i] for i in rng.integers(0, len(uniq), len(uniq))])]) for _ in range(n)]
    return np.percentile(draws, [5, 95], axis=0)


def fe_ols(df, ycol, xcols, fe, cluster, n=300):
    d = df[[ycol, fe, cluster] + xcols].dropna().reset_index(drop=True)
    Y = (d[ycol] - d.groupby(fe)[ycol].transform('mean')).to_numpy()
    X = np.column_stack([d[c] - d.groupby(fe)[c].transform('mean') for c in xcols])
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    codes, uniq = pd.factorize(d[cluster])
    rows = pd.Series(np.arange(len(d))).groupby(codes).apply(np.asarray).tolist()
    rng = np.random.default_rng(3)
    bs = []
    for _ in range(n):
        r = np.concatenate([rows[i] for i in rng.integers(0, len(uniq), len(uniq))])
        bs.append(np.linalg.lstsq(X[r], Y[r], rcond=None)[0])
    se = np.std(bs, axis=0)
    return '  '.join(f'{c}={v:+.5f} (se {s:.5f})' for c, v, s in zip(xcols, b, se))


def part_league(G):
    H = G[G.year <= 2025].copy()
    H['period'] = pd.cut(H.week, [0, 4, 8, 12, 15, 18], labels=['1-4', '5-8', '9-12', '13-15', '16-18'])
    print(f'== Part 1: league, {H.year.min()}-{H.year.max()} REG, {len(H)} team-games')
    print(H.groupby('period', observed=True)[['att', 'rushes', 'neutral_pass', 'proe', 'temp']].mean().round(3).to_string())
    print('\nslope per week (season-demeaned), 90% bootstrap over games:')
    for venue, df in (('all', H), ('outdoor', H[~H.indoor]), ('indoor', H[H.indoor])):
        for col in ('neutral_pass', 'att', 'rushes', 'proe'):
            lo, hi = boot(df, lambda s, c=col: week_slope(s, c), cluster='game_id')
            print(f'  {venue:8} {col:13} {week_slope(df, col):+.5f}/wk [{lo:+.5f}, {hi:+.5f}]  n={len(df)}')
    O = H[~H.indoor & H.temp.notna()].assign(ts=lambda d: d.year.astype(str) + d.Team)
    O['ypd'] = O.pass_yds / O.dropbacks
    print(f'\noutdoor games with a temperature ({len(O)}), team-season fixed effects, bootstrap over games:')
    for y in ('neutral_pass', 'att', 'rushes', 'proe', 'ypd'):
        for xs in (['week'], ['week', 'temp', 'wind']):
            print(f'  {y:12} ~ {"+".join(xs):15} {fe_ols(O, y, xs, "ts", "game_id")}')
    if (G.year == 2026).any():
        G26 = G.assign(p=pd.cut(G.week, [0, 2, 4, 6, 8, 18], labels=['1-2', '3-4', '5-6', '7-8', '9+']))
        print('\npass attempts per team-game, by season and weeks:')
        print(G26.pivot_table(index='year', columns='p', values='att', aggfunc='mean', observed=True).round(2).to_string())


def actual_by_player(years):
    rows = []
    for y in years:
        st, _t, name_col, _ = load_and_merge_data(y, 'Full PPR')
        st = st.assign(_w=pd.to_numeric(st['week'], errors='coerce'))
        a = st.groupby([name_col, '_w'])[STATS].sum().reset_index().rename(columns={name_col: 'Player', '_w': 'week'})
        rows.append(a.assign(year=y))
    return pd.concat(rows).rename(columns={c: f'act_{c}' for c in STATS})


def actual_by_team(years):
    rows = []
    for y in years:
        st = load_year_data(y)[0]
        tc = 'game_team' if 'game_team' in st.columns else 'team'
        st = st[st['position'].astype(str).isin(['RB', 'WR', 'TE', 'FB', 'QB'])]
        g = st.groupby([tc, 'week']).agg(act_tgt=('targets', 'sum'), act_rush=('rushing_attempts', 'sum')).reset_index()
        qb = (st[st['position'].astype(str).eq('QB')].groupby([tc, 'week'])['passing_attempts'].sum()
              .rename('act_att').reset_index())
        rows.append(g.merge(qb, on=[tc, 'week'], how='left').rename(columns={tc: 'Team'}).assign(year=y))
    return pd.concat(rows)


def part_board(B, G):
    years = sorted(B.year.unique())
    cond = G[['year', 'Team', 'week', 'indoor', 'temp', 'wind']].assign(week=lambda d: d.week.astype(float))
    B = B.assign(week=B.week.astype(float))
    for c in ('passing_attempts', 'passing_yards', 'targets', 'receptions', 'receiving_yards', 'receiving_tds',
              'rushing_attempts', 'rushing_yards', 'rushing_tds', 'Model Proj Pts'):
        B[c] = pd.to_numeric(B[c], errors='coerce')
    labels = ['3-6', '7-10', '11-14', '15-17']
    print(f'\n== Part 2: board, {years[0]}-{years[-1]} weeks {int(B.week.min())}-{int(B.week.max())}')
    proj = B.groupby(['year', 'Team', 'week']).agg(proj_tgt=('targets', 'sum'), proj_rush=('rushing_attempts', 'sum')).reset_index()
    proj = proj.merge(B[B.Pos.eq('QB')].groupby(['year', 'Team', 'week'])['passing_attempts'].sum().rename('proj_att')
                      .reset_index(), on=['year', 'Team', 'week'], how='left')
    T = actual_by_team(years).assign(week=lambda d: d.week.astype(float))
    m = proj.merge(T, on=['year', 'Team', 'week']).merge(cond, on=['year', 'Team', 'week'], how='left')
    m = m[(m.act_tgt > 0) & (m.act_att > 0)]
    m['period'] = pd.cut(m.week, [2, 6, 10, 14, 17], labels=labels)
    for c in ('att', 'tgt', 'rush'):
        m[f'r_{c}'] = m[f'proj_{c}'] - m[f'act_{c}']
    print('team-game, projected minus actual:')
    print(m.groupby('period', observed=True)[['proj_att', 'act_att', 'r_att', 'r_tgt', 'proj_rush', 'act_rush', 'r_rush']]
          .mean().round(2).to_string())
    for col in ('r_att', 'r_tgt', 'r_rush'):
        lo, hi = boot(m, lambda s, c=col: week_slope(s, c))
        print(f'  {col} slope {week_slope(m, col):+.4f}/wk [{lo:+.4f}, {hi:+.4f}]')

    P = B.merge(actual_by_player(years), on=['year', 'week', 'Player']).merge(cond, on=['year', 'Team', 'week'], how='left')
    P['proj_pts'] = P['Model Proj Pts']
    P['period'] = pd.cut(P.week, [2, 6, 10, 14, 17], labels=labels)
    P['band'] = np.where(P.indoor == True, 'indoor',  # noqa: E712 - NaN (no conditions) falls to the temperature cut
                         pd.cut(P.temp, [-20, 32, 45, 60, 120], labels=['<=32', '33-45', '46-60', '>60']).astype(str))
    spec = {'QB': ['passing_attempts', 'passing_yards'], 'RB': ['rushing_attempts', 'rushing_yards'],
            'WR': ['targets', 'receiving_yards'], 'TE': ['targets', 'receiving_yards']}
    rel = P[P.proj_pts >= 5]
    print('\nplayers (played, projected >= 5): bias = projected - actual, (sum projected / sum actual)')
    for pos, stats in spec.items():
        d = rel[rel.Pos.eq(pos)]
        if pos == 'QB':
            d = d[d.passing_attempts >= 20]
        for s in stats + ['proj_pts']:
            a = 'act_fantasy_points_ppr' if s == 'proj_pts' else f'act_{s}'
            d = d.assign(_r=d[s] - d[a])
            def cell(g):
                return f'{g._r.mean():+.2f} ({g[s].sum() / g[a].sum():.3f})'
            per = d.groupby('period', observed=True).apply(cell, include_groups=False)
            tb = d.groupby('band').apply(cell, include_groups=False).reindex(['indoor', '>60', '46-60', '33-45', '<=32'])
            lo, hi = boot(d, lambda x: week_slope(x, '_r'))
            print(f'  {pos} {s:17} {"  ".join(f"{k}:{v}" for k, v in per.items())}')
            print(f'  {"":20} {"  ".join(f"{k}:{v}" for k, v in tb.items())}   slope {week_slope(d, "_r"):+.4f}/wk '
                  f'[{lo:+.4f}, {hi:+.4f}]')
    return P


def part_screens(P):
    print('\n== Part 3: fast screens (fit on two seasons, score the other two)')
    folds = [((2022, 2023), (2024, 2025)), ((2024, 2025), (2022, 2023)), ((2022, 2024), (2023, 2025)), ((2023, 2025), (2022, 2024))]
    rb = P[P.Pos.eq('RB') & (P.rushing_attempts > 0)].copy()
    rb['per'] = (rb.rushing_yards * 0.1 + rb.rushing_tds * 6) / rb.rushing_attempts
    rb['out'] = (rb.indoor == False) & rb.temp.notna()  # noqa: E712

    def rb_mse(d, k, t0):
        lift = np.where(d.out, k * np.clip(t0 - d.temp, 0, None), 0.0)
        return float(((d.proj_pts + d.rushing_attempts * lift * d.per - d.act_fantasy_points_ppr) ** 2).mean())

    te = P[P.Pos.eq('TE') & (P.targets > 0)].copy()
    te = te[te.groupby(['year', 'week', 'Team']).targets.rank(ascending=False, method='first').eq(1) & (te.proj_pts >= 5)]
    te['per'] = (te.receptions + te.receiving_yards * 0.1 + te.receiving_tds * 6) / te.targets

    def te_mse(d, c, w0):
        return float(((d.proj_pts + d.targets * c * np.clip(d.week - w0, 0, None) * d.per - d.act_fantasy_points_ppr) ** 2).mean())

    for name, d, fn, grid in (
            ('RB cold carries lift (k, T0)', rb, rb_mse, [(k, t) for k in (0, .001, .002, .003, .004, .006) for t in (40, 50, 60)]),
            ('TE1 week target lift (c, w0)', te, te_mse, [(c, w) for c in (0, .002, .004, .006, .008, .012) for w in (3, 6, 9)])):
        print(f'  {name}, n={len(d)}')
        for train, test in folds:
            best = min(grid, key=lambda g: fn(d[d.year.isin(train)], *g))
            ts = d[d.year.isin(test)]
            print(f'    train {train} best {best} -> test {test}: MSE {fn(ts, *grid[0]):.4f} -> {fn(ts, *best):.4f} '
                  f'({fn(ts, *best) - fn(ts, *grid[0]):+.4f})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2020-2026', help='play-by-play seasons for part 1 (2026 shown separately)')
    ap.add_argument('--boards', default='', help="harness boards parquet (variant 'base') for parts 2-3; '' skips them")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.years.split('-'))
    G = team_games(range(lo, hi + 1))
    part_league(G)
    if a.boards:
        B = pd.read_parquet(a.boards)
        P = part_board(B[B.variant.eq('base')].copy(), G)
        part_screens(P)


if __name__ == '__main__':
    main()
