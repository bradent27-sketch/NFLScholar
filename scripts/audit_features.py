"""
Pre-game feature table for the model audit (step 2, 2026-10-21): one row per board row of a capture's board dump, with what was
knowable before kickoff joined from data already on disk, plus the outcomes (prefixed a_ / out_) for scoring only.

    board        the board's own inputs and outputs (games, availability, injury status, role confidence, projected snap
                 share, target rank, raw and final points, projected stat line)
    to date      this season's earlier games (weekly stats file): targets, carries, attempts, target / air-yards share, WOPR,
                 aDOT, EPA per opportunity, PPR points (mean, last 3, last game, spread), weeks since the last game
    last season  games, points / targets / carries per game, target share, team change since
    snaps        offense snap share to date, last 3, last game (snap counts cache)
    roster       age, years in the league, draft pick, height, weight (weekly roster files, first row of the season)
    game         spread and total from the schedule (team side), implied team total, home, rest, roof, temperature, wind
    team         projected QB passing yards, a QB change from the team's previous board, points sidelined teammates scored
                 to date (all and same position), active same-position teammates
    opponent     PPR points the defense allowed to the position to date vs the league (earlier weeks only)
    expected     to-date expected PPR points per game and luck (actual minus expected; nflverse's model is in-sample for past
                 seasons)
    coaching     head coach / coordinator changed this season, own and opponent

Every to-date join is as-of the target week, strictly earlier weeks (merge_asof, allow_exact_matches=False). Roster status is
not used (INA is a game-day status).

    python scripts/audit_features.py [--board .sweeps/qb_style_capture.parquet.board.csv] [--out .sweeps/audit_features.parquet]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.utils import clean_name_for_merge  # noqa: E402
from scripts.eval_calibration_schemes import load_dump, top_n_mask  # noqa: E402

KEYS = ['year', 'week', 'player', 'pos', 'team']
POS_GROUP = {'FB': 'RB', 'HB': 'RB'}


def name_key(names, positions):
    pos = pd.Series(positions, index=names.index).astype(str).replace(POS_GROUP)
    return clean_name_for_merge(names.astype(str)) + '|' + pos


def load_board(path):
    d = load_dump(path)
    d = d[d['week'].between(3, 17)].drop_duplicates(KEYS, keep=False).reset_index(drop=True)
    d['wk_key'] = d['year'] * 100 + d['week']
    d['pool'] = top_n_mask(d, d['final'].to_numpy(float)) & d['played'].to_numpy() & d['actual'].notna().to_numpy()
    d['key'] = name_key(d['player'], d['pos'])
    return d


def asof(d, logs, cols, by='key'):
    """Each board row gets `cols` from its player's latest log row in an earlier week of the same season."""
    left = d[['_rid', 'year', 'week', by]].astype({'year': 'int64', 'week': 'int64'}).sort_values('week')
    right = logs[['year', 'week', by] + cols].dropna(subset=['week']).astype({'year': 'int64', 'week': 'int64'})
    right = right.rename(columns={'week': '_lw'}).sort_values('_lw')
    m = pd.merge_asof(left, right, left_on='week', right_on='_lw', by=['year', by], allow_exact_matches=False)
    return m.set_index('_rid')[cols + ['_lw']].reindex(d['_rid'])


def stats_logs(years):
    frames = []
    for y in years:
        s = pd.read_csv(f'stats_player_week_{y}.csv', low_memory=False)
        s = s[(s['season'] == y) & (s.get('season_type', 'REG') == 'REG') & s['player_id'].notna()]
        frames.append(s)
    s = pd.concat(frames, ignore_index=True).rename(columns={'season': 'year'}).copy()
    s['key'] = name_key(s['player_display_name'], s['position'])
    dup = s.groupby(['year', 'key'])['player_id'].nunique()
    s = s[~s.set_index(['year', 'key']).index.isin(dup[dup > 1].index)]
    return s.sort_values(['year', 'key', 'week']).reset_index(drop=True)


def to_date_features(s):
    """Cumulative and rolling per-player features at each logged game (to be read as-of the next week)."""
    out = s[['year', 'key', 'week', 'team']].copy()
    s = s.copy()
    s['pts'] = s['fantasy_points_ppr'].astype(float)
    s['adot_num'] = s['receiving_air_yards'].astype(float)
    for c in ('targets', 'carries', 'attempts', 'receptions', 'pts', 'target_share', 'air_yards_share', 'wopr',
              'receiving_epa', 'rushing_epa', 'passing_epa', 'receiving_yards', 'adot_num'):
        s[c] = pd.to_numeric(s[c], errors='coerce').fillna(0.0)
    g = s.groupby(['year', 'key'], sort=False)
    out['g_td'] = g.cumcount() + 1
    for c, name in (('targets', 'tgt'), ('carries', 'car'), ('attempts', 'att'), ('pts', 'pts'),
                    ('target_share', 'tshare'), ('air_yards_share', 'ayshare'), ('wopr', 'wopr')):
        out[f'{name}_td'] = g[c].cumsum() / out['g_td']
        out[f'{name}_l3'] = g[c].transform(lambda x: x.rolling(3, min_periods=1).mean())
        out[f'{name}_last'] = s[c]
    out['pts_sd_td'] = g['pts'].transform(lambda x: x.expanding().std())
    tg = g['targets'].cumsum()
    out['adot_td'] = g['adot_num'].cumsum() / tg.where(tg > 0)
    out['rec_epa_per_tgt_td'] = g['receiving_epa'].cumsum() / tg.where(tg > 0)
    out['ypt_td'] = g['receiving_yards'].cumsum() / tg.where(tg > 0)
    ca = g['carries'].cumsum()
    out['rush_epa_per_car_td'] = g['rushing_epa'].cumsum() / ca.where(ca > 0)
    at = g['attempts'].cumsum()
    out['pass_epa_per_att_td'] = g['passing_epa'].cumsum() / at.where(at > 0)
    return out


def last_season(s):
    agg = s.groupby(['year', 'key']).agg(games_ly=('week', 'nunique'), pts_pg_ly=('fantasy_points_ppr', 'mean'),
                                         tgt_pg_ly=('targets', 'mean'), car_pg_ly=('carries', 'mean'),
                                         tshare_ly=('target_share', 'mean'), team_ly=('team', 'last')).reset_index()
    agg['year'] += 1
    return agg


def snap_logs(years):
    frames = []
    for y in years:
        p = f'data/cache/snap_counts_{y}.parquet'
        if not os.path.exists(p):
            continue
        sn = pd.read_parquet(p)
        sn = sn[sn['game_type'] == 'REG'] if 'game_type' in sn else sn
        frames.append(sn.assign(year=y))
    sn = pd.concat(frames, ignore_index=True)
    sn['key'] = name_key(sn['player'], sn['position'])
    sn = sn.drop_duplicates(['year', 'week', 'key'], keep=False).sort_values(['year', 'key', 'week'])
    g = sn.groupby(['year', 'key'], sort=False)['offense_pct']
    sn['snap_last'] = sn['offense_pct']
    sn['snap_l3'] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
    sn['snap_td'] = g.transform(lambda x: x.expanding().mean())
    return sn


def roster_attrs(years):
    frames = []
    for y in years:
        r = pd.read_csv(f'roster_weekly_{y}.csv', low_memory=False,
                        usecols=lambda c: c in {'season', 'week', 'full_name', 'position', 'birth_date', 'years_exp',
                                                'draft_number', 'height', 'weight'})
        r = r.sort_values('week').drop_duplicates('full_name', keep='first') if 'week' in r else r
        frames.append(r.assign(year=y))
    r = pd.concat(frames, ignore_index=True)
    r['key'] = name_key(r['full_name'], r['position'])
    r = r.drop_duplicates(['year', 'key'], keep=False)
    bd = pd.to_datetime(r['birth_date'], errors='coerce')
    r['age'] = (pd.to_datetime(r['year'].astype(str) + '-09-01') - bd).dt.days / 365.25
    r['draft_pick'] = pd.to_numeric(r['draft_number'], errors='coerce').fillna(300.0)
    h = r['height'].astype(str)
    r['height_in'] = np.where(h.str.contains('-'), pd.to_numeric(h.str.split('-').str[0], errors='coerce') * 12
                              + pd.to_numeric(h.str.split('-').str[-1], errors='coerce'), pd.to_numeric(h, errors='coerce'))
    r['weight_lb'] = pd.to_numeric(r['weight'], errors='coerce')
    r['years_exp'] = pd.to_numeric(r['years_exp'], errors='coerce')
    return r[['year', 'key', 'age', 'years_exp', 'draft_pick', 'height_in', 'weight_lb']]


def schedule_sides(years):
    import nflreadpy
    sc = nflreadpy.load_schedules(list(years)).to_pandas()
    sc = sc[sc['game_type'] == 'REG']
    rows = []
    for side, opp in (('home', 'away'), ('away', 'home')):
        sign = 1.0 if side == 'home' else -1.0
        t = pd.DataFrame({'year': sc['season'], 'week': sc['week'], 'team': sc[f'{side}_team'], 'opp_sched': sc[f'{opp}_team'],
                          'spread': sign * sc['spread_line'], 'total': sc['total_line'], 'home': float(side == 'home'),
                          'rest': sc[f'{side}_rest'], 'opp_rest': sc[f'{opp}_rest'], 'div_game': sc['div_game'],
                          'dome': sc['roof'].isin(['dome', 'closed']).astype(float), 'temp': sc['temp'], 'wind': sc['wind']})
        rows.append(t)
    t = pd.concat(rows, ignore_index=True)
    t['implied'] = t['total'] / 2.0 + t['spread'] / 2.0
    return t


def opponent_allowed(s):
    """PPR points a defense allowed per game to each position, to date, relative to the league's to-date average."""
    s = s[s['position'].isin(['QB', 'RB', 'WR', 'TE', 'FB'])].copy()
    s['pos'] = s['position'].replace(POS_GROUP)
    per = s.groupby(['year', 'week', 'opponent_team', 'pos'])['fantasy_points_ppr'].sum().reset_index()
    per = per.sort_values(['year', 'opponent_team', 'pos', 'week'])
    g = per.groupby(['year', 'opponent_team', 'pos'])['fantasy_points_ppr']
    per['allow_td'] = g.transform(lambda x: x.expanding().mean())
    lg = per.groupby(['year', 'week', 'pos'])['fantasy_points_ppr'].mean().groupby(level=[0, 2]).transform(
        lambda x: x.expanding().mean()).rename('lg_td').reset_index()
    per = per.merge(lg, on=['year', 'week', 'pos'])
    per['opp_allow_ratio'] = per['allow_td'] / per['lg_td']
    per['dkey'] = per['opponent_team'] + '|' + per['pos']
    return per[['year', 'week', 'dkey', 'opp_allow_ratio']]


def ffopp_logs(years):
    frames = []
    for y in years:
        p = f'data/cache/ff_opportunity_{y}.parquet'
        if os.path.exists(p):
            frames.append(pd.read_parquet(p))
    f = pd.concat(frames, ignore_index=True).rename(columns={'season': 'year'})
    f['year'] = pd.to_numeric(f['year'], errors='coerce').astype(int)
    f['week'] = pd.to_numeric(f['week'], errors='coerce')
    f['key'] = name_key(f['full_name'], f['position'])
    f = f.drop_duplicates(['year', 'week', 'key'], keep=False).sort_values(['year', 'key', 'week'])
    g = f.groupby(['year', 'key'], sort=False)
    f['xfp_td'] = g['total_fantasy_points_exp'].transform(lambda x: x.expanding().mean())
    f['luck_td'] = g['total_fantasy_points'].transform(lambda x: x.expanding().mean()) - f['xfp_td']
    f['xfp_l3'] = g['total_fantasy_points_exp'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    return f


def team_context(d):
    out = pd.DataFrame(index=d.index)
    side = (d['avail'] <= 0.05)
    am = d['anchor_mean'].fillna(0.0).where(side, 0.0)
    out['vac_pts_team'] = am.groupby([d['year'], d['week'], d['team']]).transform('sum')
    out['vac_pts_pos'] = am.groupby([d['year'], d['week'], d['team'], d['pos']]).transform('sum') - am
    active = ((d['avail'] > 0.5) & (d['final'] >= 5.0)).astype(float)
    out['n_active_pos'] = active.groupby([d['year'], d['week'], d['team'], d['pos']]).transform('sum') - active
    qb = d[d['pos'] == 'QB']
    qb_top = qb.sort_values('p_passing_attempts', ascending=False).drop_duplicates(['year', 'week', 'team'])
    qb_top = qb_top[['year', 'week', 'team', 'player', 'p_passing_yards', 'games_season']].rename(
        columns={'player': '_qb', 'p_passing_yards': 'team_qb_pass_yds', 'games_season': 'team_qb_games'})
    qb_top = qb_top.sort_values(['year', 'team', 'week'])
    qb_top['_qb_prev'] = qb_top.groupby(['year', 'team'])['_qb'].shift(1)
    qb_top['qb_change'] = (qb_top['_qb_prev'].notna() & (qb_top['_qb'] != qb_top['_qb_prev'])).astype(float)
    m = d[['year', 'week', 'team']].merge(qb_top.drop(columns=['_qb', '_qb_prev']), on=['year', 'week', 'team'], how='left')
    for c in ('team_qb_pass_yds', 'team_qb_games', 'qb_change'):
        out[c] = m[c].to_numpy()
    tot = d.groupby(['year', 'week', 'team'])['p_targets'].transform('sum')
    out['proj_tshare'] = d['p_targets'] / tot.where(tot > 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--board', default='.sweeps/qb_style_capture.parquet.board.csv')
    ap.add_argument('--out', default='.sweeps/audit_features.parquet')
    a = ap.parse_args()
    d = load_board(a.board)
    d['_rid'] = np.arange(len(d))
    years = sorted(d['year'].unique())
    f = pd.DataFrame(index=d.index)

    # board
    for c in ('games_season', 'avail', 'workload', 'role_conf', 'snap', 'role_change_conf', 'target_rank', 'evidence_games',
              'raw', 'final', 'anchor_mean', 'anchor_games', 'week'):
        f[c] = pd.to_numeric(d[c], errors='coerce')
    f['qb_starter'] = d['qb_starter'].astype(str).str.lower().eq('true').astype(float)
    f['pff_weekly'] = d['source_kind'].eq('weekly_archive').astype(float)
    inj = d['inj'].astype(str).str.lower()
    f['inj_q'] = inj.str.startswith('questionable').astype(float)
    f['inj_d'] = inj.str.startswith('doubtful').astype(float)
    f['inj_out'] = (inj.str.startswith('out') | inj.str.startswith('reserve')).astype(float)
    f['line_move'] = d['final'] - d['raw']
    for c in [c for c in d.columns if c.startswith('p_')]:
        f[c] = pd.to_numeric(d[c], errors='coerce')

    # to date and last season
    s = stats_logs(sorted(set(years) | {years[0] - 1}))
    td = to_date_features(s)
    cols = [c for c in td.columns if c not in ('year', 'key', 'week', 'team')]
    j = asof(d, td, cols)
    for c in cols:
        f[c] = j[c].to_numpy()
    f['weeks_since_game'] = d['week'].to_numpy() - j['_lw'].to_numpy()
    f['g_td'] = f['g_td'].fillna(0.0)
    ly = last_season(s)
    m = d[['year', 'key', 'team']].merge(ly, on=['year', 'key'], how='left')
    for c in ('games_ly', 'pts_pg_ly', 'tgt_pg_ly', 'car_pg_ly', 'tshare_ly'):
        f[c] = m[c].to_numpy()
    f['games_ly'] = f['games_ly'].fillna(0.0)
    f['team_changed'] = (m['team_ly'].notna() & (m['team_ly'] != m['team'])).astype(float).to_numpy()
    f['dev_pts_td'] = f['final'] - f['pts_td']
    f['dev_pts_l3'] = f['final'] - f['pts_l3']
    f['dev_pts_ly'] = f['final'] - f['pts_pg_ly']
    f['dev_tgt_l3'] = f['p_targets'] - f['tgt_l3']
    f['dev_tgt_td'] = f['p_targets'] - f['tgt_td']
    f['dev_car_l3'] = f['p_rushing_attempts'] - f['car_l3']

    # snaps (this week's actual snap share kept as an outcome column only)
    sn = snap_logs(years)
    j = asof(d, sn, ['snap_last', 'snap_l3', 'snap_td'])
    for c in ('snap_last', 'snap_l3', 'snap_td'):
        f[c] = j[c].to_numpy()
    f['dev_snap_l3'] = f['snap'] - f['snap_l3']
    now = d[['year', 'week', 'key']].merge(sn[['year', 'week', 'key', 'offense_pct']], on=['year', 'week', 'key'], how='left')
    out_snap = now['offense_pct'].to_numpy()

    # roster attributes
    m = d[['year', 'key']].merge(roster_attrs(years), on=['year', 'key'], how='left')
    for c in ('age', 'years_exp', 'draft_pick', 'height_in', 'weight_lb'):
        f[c] = m[c].to_numpy()

    # game
    sides = schedule_sides(years)
    m = d[['year', 'week', 'team']].merge(sides, on=['year', 'week', 'team'], how='left')
    for c in ('spread', 'total', 'implied', 'home', 'rest', 'opp_rest', 'div_game', 'dome', 'temp', 'wind'):
        f[c] = pd.to_numeric(m[c], errors='coerce').to_numpy()
    f['rest_diff'] = f['rest'] - f['opp_rest']

    # team
    tc = team_context(d)
    for c in tc.columns:
        f[c] = tc[c].to_numpy()

    # opponent
    oa = opponent_allowed(s)
    d['dkey'] = d['opp'].astype(str) + '|' + d['pos'].astype(str)
    j = asof(d, oa, ['opp_allow_ratio'], by='dkey')
    f['opp_allow_ratio'] = j['opp_allow_ratio'].to_numpy()

    # expected points
    ff = ffopp_logs(years)
    j = asof(d, ff, ['xfp_td', 'luck_td', 'xfp_l3'])
    for c in ('xfp_td', 'luck_td', 'xfp_l3'):
        f[c] = j[c].to_numpy()

    # coaching
    try:
        from data.coaching_changes import coaching_change_table
        cc = coaching_change_table(years[0], years[-1]).rename(columns={'season': 'year'})
        m = d[['year', 'team', 'opp']].merge(cc[['year', 'team', 'hc_changed', 'oc_changed']], on=['year', 'team'], how='left')
        m = m.merge(cc[['year', 'team', 'dc_changed']].rename(columns={'team': 'opp', 'dc_changed': 'opp_dc_changed'}),
                    on=['year', 'opp'], how='left')
        for c in ('hc_changed', 'oc_changed', 'opp_dc_changed'):
            f[c] = pd.to_numeric(m[c], errors='coerce').to_numpy()
    except Exception as exc:  # optional source
        print('coaching table unavailable:', exc)

    out = pd.concat([d[KEYS + ['opp', 'wk_key', 'pool', 'played', 'actual']].reset_index(drop=True),
                     d[[c for c in d.columns if c.startswith('a_')]].reset_index(drop=True).astype(float),
                     f.reset_index(drop=True).add_prefix('f_')], axis=1)
    out['out_snap'] = out_snap
    out.to_parquet(a.out, index=False)
    feats = [c for c in out.columns if c.startswith('f_')]
    cov = out.loc[out['pool'], feats].notna().mean().sort_values()
    print(f'{len(out):,} rows, {len(feats)} features -> {a.out}')
    print('coverage on the START pool (lowest 15):')
    print(cov.head(15).round(3).to_string())
    print('this-week snap share found for', round(float(pd.notna(out.loc[out['pool'], 'out_snap']).mean()), 3), 'of the pool')


if __name__ == '__main__':
    main()
