"""
High-call audit, information-focused (2026-11-01; bar in .sweeps/high_call_audit_bar.md, set before this ran).

When the live model is 2+ points above the market it ends up closer to the result only about a third of the time. This tags each
of those rows with what was known before kickoff and asks which tags the misses concentrate in (2026 live ledger), whether the
board over-projects the same tag on history without any market (deep-audit panel, 2021-2025), and whether today's model (the
panel's 2026 replay) still makes the call. Diagnostic only: the market is the benchmark, never an input.

    python scripts/audit_high_calls.py [--max-week 7] [--out .sweeps/high_call_audit.txt]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data.ledger_scorecard as lsc  # noqa: E402
from data.loaders import load_schedule  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from scripts.deep_audit import PANEL  # noqa: E402

HIST = (2021, 2022, 2023, 2024, 2025)
GAP = 2.0
TD_POINTS = {'passing_tds': 4.0, 'rushing_tds': 6.0, 'receiving_tds': 6.0}
PANEL_COLS = ['year', 'week', 'key', 'pos', 'team', 'played', 'actual', 'p_points', 'Injury Status', 'Availability',
              'a_weekly_snap_pct', 'rookie_like', 'tm_out_pts_QB', 'tm_out_pts_RB', 'tm_out_pts_WR', 'tm_out_pts_TE',
              'p_rushing_tds', 'p_receiving_tds', 'p_passing_tds']


def injury_tag(s):
    s = s.fillna('').astype(str).str.lower()
    return np.select([s.str.startswith('questionable'), s.str.startswith('doubtful'), (s.str.len() > 0) & ~s.str.startswith('nan')],
                     ['Questionable', 'Doubtful', 'other listed'], 'none')


def history_tags(p, schedule_games):
    """Per panel row, the tags that need earlier weeks: last game's snap share minus the 3-game average, missed games among the
    team's last two, season-to-date points, and a same-position teammate who sat last week and is active now."""
    p = p.sort_values(['year', 'key', 'pos', 'week']).copy()
    played = p['played'].astype(bool)
    snap = p['a_weekly_snap_pct'].where(played)
    g = p.assign(_snap=snap, _pts=p['actual'].where(played)).groupby(['year', 'key', 'pos'], sort=False)
    p['snap_last'] = g['_snap'].transform(lambda x: x.ffill().shift(1))
    p['snap_l3'] = g['_snap'].transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
    p['role_trend'] = p['snap_last'] - p['snap_l3']
    p['s2d_pts'] = g['_pts'].transform(lambda x: x.shift(1).expanding().mean())
    p['games_s2d'] = g['_pts'].transform(lambda x: x.shift(1).notna().cumsum())
    # missed one of the team's last two games (the team played them: from the schedule)
    pw = set(zip(p.loc[played, 'year'], p.loc[played, 'key'], p.loc[played, 'pos'], p.loc[played, 'week']))
    missed = []
    for y, k, ps, t, w in zip(p['year'], p['key'], p['pos'], p['team'], p['week']):
        team_weeks = [x for x in schedule_games.get((y, t), ()) if x < w][-2:]
        missed.append(sum((y, k, ps, x) not in pw for x in team_weeks) if len(team_weeks) == 2 else 0)
    p['missed_last2'] = missed
    # returning teammate: same team and position, projected 5+ now with Availability > 0.5, did not play last week
    sat = set(zip(p.loc[~played, 'year'], p.loc[~played, 'team'], p.loc[~played, 'pos'], p.loc[~played, 'key'],
                  p.loc[~played, 'week']))
    cand = p[(p['p_points'] >= 5) & (pd.to_numeric(p['Availability'], errors='coerce').fillna(1) > 0.5)]
    back = {}
    for y, t, ps, k, w in zip(cand['year'], cand['team'], cand['pos'], cand['key'], cand['week']):
        team_weeks = [x for x in schedule_games.get((y, t), ()) if x < w]
        if team_weeks and (y, t, ps, k, team_weeks[-1]) in sat:
            back.setdefault((y, t, ps, w), set()).add(k)
    p['teammate_back'] = [bool(back.get((y, t, ps, w), set()) - {k})
                          for y, t, ps, w, k in zip(p['year'], p['team'], p['pos'], p['week'], p['key'])]
    return p


def schedule_games(years):
    out = {}
    for y in years:
        s = load_schedule(y)
        s = s[s['game_type'] == 'REG'] if 'game_type' in s.columns else s
        for w, h, a in zip(s['week'], s['home_team'], s['away_team']):
            out.setdefault((y, str(h)), []).append(int(w))
            out.setdefault((y, str(a)), []).append(int(w))
    return {k: sorted(v) for k, v in out.items()}


def add_tags(f, proj_col, td_cols):
    f = f.copy()
    f['inj'] = injury_tag(f['Injury Status'])
    f['returning'] = (f['missed_last2'] >= 1) | ((f['week'] >= 4) & (f['games_s2d'] <= 2))
    tm = np.select([f['pos'].eq(x) for x in ('QB', 'RB', 'WR', 'TE')],
                   [f[f'tm_out_pts_{x}'] for x in ('QB', 'RB', 'WR', 'TE')], 0.0)
    f['vacancy'] = pd.to_numeric(pd.Series(tm, index=f.index), errors='coerce').fillna(0) > 0
    f['role_down'] = f['role_trend'] <= -10
    f['role_up'] = f['role_trend'] >= 10
    f['rookie'] = f['rookie_like'].fillna(0).astype(float) > 0
    f['above_own'] = (f[proj_col] - f['s2d_pts']) >= 3
    td = sum(TD_POINTS[c.split('_', 1)[1] if c.startswith('p_') else c] * pd.to_numeric(f[c], errors='coerce').fillna(0)
             for c in td_cols)
    f['td_heavy'] = td >= 0.4 * f[proj_col]
    return f


TAGS = [('Questionable', lambda f: f['inj'].eq('Questionable')), ('Doubtful', lambda f: f['inj'].eq('Doubtful')),
        ('other listed', lambda f: f['inj'].eq('other listed')), ('returning player', lambda f: f['returning']),
        ('teammate back', lambda f: f['teammate_back']), ('vacancy beneficiary', lambda f: f['vacancy']),
        ('snaps down 10+', lambda f: f['role_down']), ('snaps up 10+', lambda f: f['role_up']),
        ('rookie', lambda f: f['rookie']), ('model 3+ over own s2d', lambda f: f['above_own']),
        ('TD-heavy', lambda f: f['td_heavy'])]


def start_pool(f, cols):
    keep = []
    for (w, pos), g in f.groupby(['week', 'pos']):
        n = lsc.STARTABLE_N.get(pos)
        if not n:
            continue
        idx = set()
        for c in cols:
            idx |= set(g[c].nlargest(n).index)
        keep += list(idx)
    return f.loc[sorted(keep)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-week', type=int, default=7)
    ap.add_argument('--out', default=os.path.join('.sweeps', 'high_call_audit.txt'))
    a = ap.parse_args()
    pd.set_option('display.width', 240)
    pd.set_option('display.max_columns', 40)
    lines = []

    def say(x=''):
        print(x)
        lines.append(str(x))

    stats_df, _tc, name_col, _ = load_and_merge_data(2026, 'Full PPR')
    led, info = lsc.load_scored_ledger(2026, stats_df, name_col, load_schedule(2026))
    led = led[led['week'].between(3, a.max_week) & led['pre_kickoff'] & led['model_pts'].notna() & led['market_pts'].notna()]
    p = pd.read_parquet(PANEL, columns=PANEL_COLS)
    p = history_tags(p, schedule_games(sorted(p['year'].unique())))
    # live rows: ledger projections, tags from the panel's same-week row (pre-kickoff facts only) and the ledger's own injury
    hcols = ['year', 'week', 'key', 'pos', 'role_trend', 's2d_pts', 'games_s2d', 'missed_last2', 'teammate_back', 'rookie_like',
             'tm_out_pts_QB', 'tm_out_pts_RB', 'tm_out_pts_WR', 'tm_out_pts_TE', 'p_points']
    live = led.merge(p[p['year'] == 2026][hcols].rename(columns={'key': '_key', 'pos': 'Pos', 'p_points': 'replay_pts'}),
                     on=['_key', 'week', 'Pos'], how='left').rename(columns={'Pos': 'pos'})
    live = add_tags(live, 'model_pts', [c for c in TD_POINTS if c in live.columns])
    live = start_pool(live, ['model_pts', 'market_pts'])
    live['gap'] = live['model_pts'] - live['market_pts']
    live['model_right'] = (live['model_pts'] - live['actual_pts']).abs() < (live['market_pts'] - live['actual_pts']).abs()
    hc = live[live['gap'] >= GAP]
    say(f'2026 live ledger, weeks 3-{a.max_week}, pre-kickoff START pool paired with the market: {len(live):,} rows; '
        f'high calls (model - market >= {GAP:g}): {len(hc)}, model right {hc["model_right"].mean():.0%}')
    say(f'  high calls: model {hc["model_pts"].mean():.2f}, market {hc["market_pts"].mean():.2f}, actual '
        f'{hc["actual_pts"].mean():.2f}; replay (today\'s model) {hc["replay_pts"].mean():.2f} on {hc["replay_pts"].notna().sum()} rows')
    fp = hc['fp_pts']
    say(f'  FantasyPros on them: nearer the model on {((fp - hc["model_pts"]).abs() < (fp - hc["market_pts"]).abs()).sum()} of '
        f'{fp.notna().sum()}, mean FP {fp.mean():.2f}')
    say(f'  by position: {hc.groupby("pos")["model_right"].agg(["size", "mean"]).round(2).to_dict("index")}')
    say(f'  by week: {hc.groupby("week")["model_right"].agg(["size", "mean"]).round(2).to_dict("index")}')

    # history: the board's residual per tag, START pool (top-N by projection among played), weeks 3-17
    h = p[p['played'].astype(bool) & p['week'].between(3, 17)].copy()
    h['rk'] = h.groupby(['year', 'week', 'pos'])['p_points'].rank(ascending=False, method='first')
    h = h[h['rk'] <= h['pos'].map(lsc.STARTABLE_N)]
    h = add_tags(h, 'p_points', ['p_rushing_tds', 'p_receiving_tds', 'p_passing_tds'])
    h['res'] = h['actual'] - h['p_points']
    rep = h[h['year'] == 2026]
    say('')
    say('Per tag: 2026 live high calls (n, model right, share of the high calls\' total miss), all live START rows with the tag '
        '(n, model - actual, market - actual), history 2021-2025 board residual (actual - board) pooled and per season, the 2026 '
        'replay\'s residual; then the bar.')
    rows = []
    miss_total = (hc['model_pts'] - hc['actual_pts']).sum()
    for name, fn in TAGS:
        t_hc, t_live, t_h, t_rep = fn(hc), fn(live), fn(h), fn(rep)
        per = h[t_h].groupby('year')['res'].mean().reindex(list(HIST))
        n_hc = int(t_hc.sum())
        right = float(hc.loc[t_hc, 'model_right'].mean()) if n_hc else np.nan
        pooled = float(h.loc[t_h & h['year'].isin(HIST), 'res'].mean())
        over_seasons = int((per <= -0.5).sum())
        rep_res = float(rep.loc[t_rep, 'res'].mean()) if t_rep.any() else np.nan
        a_ = n_hc >= 15 and right <= 0.25
        b_ = pooled <= -0.5 and over_seasons >= 4
        c_ = np.isfinite(rep_res) and rep_res < 0
        rows.append({'tag': name, 'hc n': n_hc, 'hc right': right,
                     'hc miss share': float((hc.loc[t_hc, 'model_pts'] - hc.loc[t_hc, 'actual_pts']).sum() / miss_total),
                     'live n': int(t_live.sum()),
                     'live model-act': float((live.loc[t_live, 'model_pts'] - live.loc[t_live, 'actual_pts']).mean()),
                     'live mkt-act': float((live.loc[t_live, 'market_pts'] - live.loc[t_live, 'actual_pts']).mean()),
                     'hist n': int((t_h & h['year'].isin(HIST)).sum()), 'hist res': pooled,
                     **{f'res {y}': per[y] for y in HIST}, 'replay res': rep_res,
                     '(a)': a_, '(b)': b_, '(c)': c_,
                     'verdict': 'MODEL LEAD' if a_ and b_ and c_ else ('information (process)' if a_ else '-')})
    r = pd.DataFrame(rows)
    say(r.round(2).to_string(index=False))
    say('')
    say('Largest high calls (model - market), with tags:')
    tagcols = [n for n, _ in TAGS]
    show = hc.assign(**{n: fn(hc) for n, fn in TAGS})
    show['tags'] = show[tagcols].apply(lambda x: ', '.join(c for c in tagcols if x[c]), axis=1)
    say(show.sort_values('gap', ascending=False).head(30)[['week', 'Player', 'pos', 'Team', 'model_pts', 'market_pts', 'fp_pts',
                                                           'actual_pts', 'replay_pts', 'tags']].round(1).to_string(index=False))
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    r.to_csv(a.out.replace('.txt', '.csv'), index=False)
    hc.to_csv(a.out.replace('.txt', '_rows.csv'), index=False)


if __name__ == '__main__':
    main()
