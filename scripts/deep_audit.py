"""Deep audit (2026-10-28, user: "build a memory intensive test to work through the next 4-5 hours ... whatever kind of testing you
can come up with that can help identify flaws, strengths, new factors we can't currently see etc. This should have the capability
of building prop specific accuracy for prop betting going forward").

Every stage reads and writes .sweeps/deep_audit/; each can be re-run on its own (capture resumes where it stopped):

  capture   every board of TODAY's model - DEFAULT_FEATURES plus the harness's injury/reserve replay (the harness v2 base arm) -
            2021-2025 weeks 1-18 and 2026 weeks 1-7, every column the build returns, with that week's actual box-score line
            (a silent game - snaps, no row - scores 0s, the harness's definition of played)
  assemble  one panel: boards + actuals + pre-game context - the game's posted spread/total, roof, weather, rest; the player's
            own earlier games this season and last season; the opponent's allowed-to-position rates so far; the team's
            volume so far; teammates out this week
  accuracy  per position x stat, the model against simple baselines (season-to-date, last 3, last season) on the same rows, and
            whether a leave-one-season-out stack of those baselines adds anything (information the model leaves on the table)
  segments  residual bias per position x stat x context bin: week-clustered errors, Benjamini-Hochberg false discovery
            control, sign consistency across seasons, the share of squared error a fix would remove
  discover  gradient-boosted models of the residual, leave-one-season-out (2026 held out from 2021-2025), against a shuffled-
            residual null; permutation importance and the shape of each top feature: factors the model cannot see
  props     the distribution of every prop stat given the projection, four ways (the Poisson/Normal the de-vig code assumes,
            empirical by projection level, fitted negative-binomial / zero-inflated gamma, quantile boosting), forward-chained
            2023-2026; CRPS, PIT, P(over) calibration at book-like lines, and on 2026's posted lines (Brier against the books'
            de-vigged prices, hit rate and ROI by edge). The fitted engines on every season are saved for the weekly prop tooling.
  report    REPORT.md from every stage's tables

Pass 2 (--stage pass2, on pass 1's panel; findings in docs/deep_audit_2026-10-28.md):
  windows   weeks 1, 2, 3-8, 9-17 and 18 side by side
  overshoot how much of the model's departure from a player's own level comes true (k), by group, and the held-out gain of
            shrinking by k
  props2    prop engines on recalibrated means (linear on the player's own history; boosted on every feature) and quantile
            boosting on every feature, scored like props; posted lines at the open and the close
  report2   REPORT_PASS2.md
  engines   (alone) the engines data/prop_probability.py reads: all seasons and 2021-2025; install with
            scripts/prop_scorecard.py --install-engines

    python scripts/deep_audit.py --stage all          (the overnight chain; each stage in its own process)
    python scripts/deep_audit.py --stage props        (one stage)
"""
import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
logging.disable(logging.WARNING)
import warnings  # noqa: E402
warnings.filterwarnings('ignore', category=Warning, module='pandas')
warnings.filterwarnings('ignore', message='.*fragmented.*')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get('DEEP_AUDIT_OUT') or os.path.join(ROOT, '.sweeps', 'deep_audit')
MS = float(os.environ.get('DEEP_AUDIT_MIN_SCALE', '1'))      # < 1 only for a smoke test on a few weeks
BOARDS = os.path.join(OUT, 'boards')
PANEL = os.path.join(OUT, 'panel.parquet')
PROGRESS = os.path.join(OUT, 'progress.log')
SCORING = 'Full PPR'
POS = ('QB', 'RB', 'WR', 'TE')
OFF_POS = ('QB', 'RB', 'WR', 'TE', 'FB')
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)
STATS = ('passing_attempts', 'passing_completions', 'passing_yards', 'passing_tds', 'passing_interceptions',
         'rushing_attempts', 'rushing_yards', 'rushing_tds', 'targets', 'receptions', 'receiving_yards', 'receiving_tds')
# nflverse spellings of the board's stat names, when the loader does not carry the board's own
ALIASES = {'passing_attempts': ('attempts',), 'passing_completions': ('completions',), 'rushing_attempts': ('carries',),
           'passing_interceptions': ('interceptions',)}
# usage / efficiency the player's history features carry beyond the board stats
HIST_EXTRA = ('fantasy_points_ppr', 'target_share', 'air_yards_share', 'wopr', 'receiving_air_yards', 'passing_air_yards',
              'sacks_suffered', 'passing_epa', 'rushing_epa', 'receiving_epa', 'passing_cpoe', 'receiving_first_downs',
              'rushing_first_downs', 'receiving_yards_after_catch')
TARGETS = {   # position -> the stats graded there (plus fantasy points)
    'QB': ('passing_attempts', 'passing_completions', 'passing_yards', 'passing_tds', 'passing_interceptions',
           'rushing_attempts', 'rushing_yards', 'points'),
    'RB': ('rushing_attempts', 'rushing_yards', 'rushing_tds', 'targets', 'receptions', 'receiving_yards', 'points'),
    'WR': ('targets', 'receptions', 'receiving_yards', 'receiving_tds', 'points'),
    'TE': ('targets', 'receptions', 'receiving_yards', 'receiving_tds', 'points'),
}
SEED = 20261028


# ------------------------------------------------------------------------------------------------------------- plumbing
def log(msg):
    os.makedirs(OUT, exist_ok=True)
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(PROGRESS, 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def memory_gb():
    """(free physical GB, free commit GB) on Windows; (inf, inf) elsewhere."""
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 2 ** 30, m.ullAvailPageFile / 2 ** 30
    except Exception:
        return float('inf'), float('inf')


def wait_for_memory(need_gb=2.0, max_wait_s=3600):
    """Hold a build until there is room (the machine is shared with the CFB session's jobs)."""
    t0 = time.time()
    while True:
        phys, commit = memory_gb()
        if min(phys, commit) >= need_gb or time.time() - t0 > max_wait_s:
            return
        log(f'  waiting for memory (free {phys:.1f} GB physical, {commit:.1f} GB commit)')
        time.sleep(60)


def _to_str(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    return str(v)


def num(s):
    return pd.to_numeric(s, errors='coerce')


def clustered_mean_se(x, cluster):
    """Mean of x and its standard error with clusters (weeks) as the unit."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float(np.mean(x)) if n else np.nan, np.nan
    m = x.mean()
    g = pd.Series(x - m).groupby(np.asarray(cluster)).sum().to_numpy()
    k = len(g)
    if k < 2:
        return m, np.nan
    return float(m), float(np.sqrt((g ** 2).sum() * k / (k - 1)) / n)


def bh_qvalues(p):
    p = np.asarray(p, dtype=float)
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    if not len(pv):
        return q
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(len(pv))
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def norm_sf(z):
    from scipy.stats import norm
    return norm.sf(z)


def fmt_table(df, floatfmt='{:.4f}'):
    """A plain markdown table."""
    if df is None or df.empty:
        return '(none)\n'
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(str(c) for c in cols) + ' |', '|' + '---|' * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append('' if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


# -------------------------------------------------------------------------------------------------------------- capture
def capture_week(year, week, stats_df, team_col, name_col, features):
    from data.weekly_projections import build_weekly_projections
    from data.silent_games import played_players
    from scripts.eval_weekly_model import _actual_points

    weeks = num(stats_df['week'])
    actual = _actual_points(stats_df, name_col, week, 'fantasy_points_ppr')
    rows = stats_df[weeks == week]
    if 'position' in rows.columns:
        rows = rows[rows['position'].isin(OFF_POS)]
    numeric = [c for c in rows.columns if c not in ('season', 'week') and pd.api.types.is_numeric_dtype(rows[c])]
    sums = rows.groupby(name_col, observed=True)[numeric].sum(min_count=1)
    silent_names = played_players(year, week, stats_df, name_col, team_col) - set(actual.index.astype(str))
    build_weekly_projections.clear()
    proj, _meta = build_weekly_projections(year, week, SCORING, as_of_week=week, apply_injury=False, features=features)
    build_weekly_projections.clear()
    if proj is None or proj.empty:
        return None
    d = proj.reset_index(drop=True).copy()
    players = d['Player'].astype(str)
    d['actual'] = players.map(actual)
    d['played_box'] = d['actual'].notna()
    d['silent'] = ~d['played_box'] & players.isin(silent_names)
    d.loc[d['silent'], 'actual'] = 0.0
    d['played'] = d['played_box'] | d['silent']
    a = sums.reindex(players.values)
    a.columns = ['a_' + c for c in a.columns]
    a.index = d.index
    a.loc[d['silent'].to_numpy()] = 0.0
    d = pd.concat([d, a], axis=1)
    d['year'], d['week'] = year, week
    for c in d.columns:
        if d[c].dtype == object or str(d[c].dtype) == 'category':
            d[c] = d[c].map(_to_str)
    return d


def stage_capture(years, weeks):
    from scripts.capture_stat_blend_traces import install_snap_guard
    install_snap_guard()
    from data.transforms import load_and_merge_data
    from scripts.calibration_lab import HARNESS_BASE_FEATURES

    os.makedirs(BOARDS, exist_ok=True)
    log(f'capture: {len(HARNESS_BASE_FEATURES)} flags (harness base arm), years {years}')
    for year in years:
        stats_df, team_col, name_col, _ = load_and_merge_data(year, SCORING)
        if stats_df is None or 'week' not in stats_df.columns:
            log(f'  {year}: no weekly data')
            continue
        max_week = int(num(stats_df['week']).max())
        for week in weeks:
            if week > max_week:
                continue
            path = os.path.join(BOARDS, f'{year}_w{week:02d}.parquet')
            if os.path.exists(path):
                continue
            wait_for_memory()
            t0 = time.time()
            try:
                d = capture_week(year, week, stats_df, team_col, name_col, HARNESS_BASE_FEATURES)
            except Exception:
                log(f'  {year} w{week} FAILED\n' + traceback.format_exc())
                continue
            if d is None:
                log(f'  {year} w{week} empty board')
                continue
            d.to_parquet(path, index=False)
            phys, commit = memory_gb()
            log(f'  {year} w{week:02d}: {len(d)} rows, {int(d["played"].sum())} played, {time.time() - t0:.0f}s '
                f'(free {phys:.1f}/{commit:.1f} GB)')


# ------------------------------------------------------------------------------------------------------------- assemble
def _stat_col(frame, stat):
    """The frame's column for a board stat (the board spelling, else the nflverse one), or None."""
    if stat in frame.columns:
        return stat
    for alt in ALIASES.get(stat, ()):
        if alt in frame.columns:
            return alt
    return None


def season_logs(year):
    """One row per offensive player-game of `year` (regular season): key, pos, team, opp, week and the stats."""
    from data.transforms import load_and_merge_data
    from data.utils import clean_name_exact
    stats_df, team_col, name_col, _ = load_and_merge_data(year, SCORING)
    s = stats_df.copy()
    s['week'] = num(s['week'])
    if 'season_type' in s.columns:
        s = s[s['season_type'].astype(str).eq('REG') | s['season_type'].isna()]
    s = s[s['week'].between(1, 18)]
    if 'position' in s.columns:
        s = s[s['position'].isin(OFF_POS)]
    out = pd.DataFrame({'key': clean_name_exact(s[name_col].astype(str)).to_numpy(),
                        'pos': s['position'].replace({'FB': 'RB'}).to_numpy() if 'position' in s.columns else None,
                        'team': s[team_col].astype(str).to_numpy(),
                        'opp': s['opponent_team'].astype(str).to_numpy() if 'opponent_team' in s.columns else None,
                        'week': s['week'].astype(int).to_numpy()})
    for stat in STATS:
        c = _stat_col(s, stat)
        out[stat] = num(s[c]).to_numpy() if c else np.nan
    for c in HIST_EXTRA:
        out[c] = num(s[c]).to_numpy() if c in s.columns else np.nan
    keys = ['key', 'pos', 'week']
    g = out.groupby(keys)
    first = g[['team', 'opp']].first()
    sums = g[list((*STATS, *HIST_EXTRA))].sum(min_count=1)
    return first.join(sums).reset_index()


HIST_STATS = (*STATS, *HIST_EXTRA)


def player_history(logs):
    """For each player-game: the season so far INCLUDING that game (mean, last 3, last, sd, games), ready for an as-of join on
    the next week (allow_exact_matches=False)."""
    logs = logs.sort_values(['key', 'pos', 'week']).reset_index(drop=True)
    g = logs.groupby(['key', 'pos'], sort=False)
    out = logs[['key', 'pos', 'week']].copy()
    out['h_games'] = g.cumcount() + 1
    for c in HIST_STATS:
        v = logs[c]
        out[f'h_s2d_{c}'] = g[c].cumsum() / out['h_games']
        out[f'h_l3_{c}'] = g[c].transform(lambda x: x.rolling(3, min_periods=1).mean())
        out[f'h_last_{c}'] = v
        if c in (*STATS, 'fantasy_points_ppr'):
            out[f'h_sd_{c}'] = g[c].transform(lambda x: x.expanding().std())
    return out


def defense_allowed(logs):
    """Per (defense, position, week): what that defense allowed to the position that week; then the season-to-date mean
    (including that week) for an as-of join."""
    cols = ('fantasy_points_ppr', 'targets', 'receptions', 'receiving_yards', 'rushing_attempts', 'rushing_yards',
            'passing_attempts', 'passing_yards', 'passing_tds', 'receiving_tds', 'rushing_tds')
    d = logs.groupby(['opp', 'pos', 'week'], as_index=False)[list(cols)].sum().sort_values(['opp', 'pos', 'week'])
    g = d.groupby(['opp', 'pos'], sort=False)
    out = d[['opp', 'pos', 'week']].copy()
    out['o_games'] = g.cumcount() + 1
    for c in cols:
        out[f'o_{c}'] = g[c].cumsum() / out['o_games']
    return out


def team_volume(logs):
    """Per (team, week): the team's offensive volume that week, then season-to-date means for an as-of join."""
    agg = logs.groupby(['team', 'week'], as_index=False).agg(
        t_pass_att=('passing_attempts', 'sum'), t_carries=('rushing_attempts', 'sum'), t_targets=('targets', 'sum'),
        t_points=('fantasy_points_ppr', 'sum')).sort_values(['team', 'week'])
    g = agg.groupby('team', sort=False)
    out = agg[['team', 'week']].copy()
    n = g.cumcount() + 1
    for c in ('t_pass_att', 't_carries', 't_targets', 't_points'):
        out[c] = g[c].cumsum() / n
        out[c + '_l3'] = g[c].transform(lambda x: x.rolling(3, min_periods=1).mean())
    out['t_pass_rate'] = out['t_pass_att'] / (out['t_pass_att'] + out['t_carries'])
    out['t_pass_rate_l3'] = out['t_pass_att_l3'] / (out['t_pass_att_l3'] + out['t_carries_l3'])
    return out


def asof_join(left, right, by, prefix_cols):
    """left (needs `by` + week) gets right's latest row with week < left's week, per `by`."""
    l = left[list(by) + ['week', '_rid']].copy()
    l['week'] = l['week'].astype('int64')
    r = right.copy()
    r['week'] = r['week'].astype('int64')
    l = l.sort_values('week')
    r = r.sort_values('week')
    m = pd.merge_asof(l, r[list(by) + ['week'] + list(prefix_cols)], on='week', by=list(by), allow_exact_matches=False,
                      direction='backward')
    return m.set_index('_rid')[list(prefix_cols)]


def schedule_context(year):
    from data.loaders import load_schedule
    s = load_schedule(year)
    if 'game_type' in s.columns:
        s = s[s['game_type'] == 'REG']
    rows = []
    for _, g in s.iterrows():
        spread, total = num(pd.Series([g.get('spread_line')]))[0], num(pd.Series([g.get('total_line')]))[0]
        hour = np.nan
        try:
            hour = int(str(g.get('gametime'))[:2])
        except (TypeError, ValueError):
            pass
        for side, other, home in (('home', 'away', 1), ('away', 'home', 0)):
            margin = spread if home else -spread
            rows.append({
                'year': year, 'week': int(g['week']), 'team': str(g[f'{side}_team']), 'g_opp': str(g[f'{other}_team']),
                'home': home if str(g.get('location')) != 'Neutral' else 0.5, 'spread': margin, 'total': total,
                'imp_team': total / 2 + margin / 2 if pd.notna(total) and pd.notna(margin) else np.nan,
                'imp_opp': total / 2 - margin / 2 if pd.notna(total) and pd.notna(margin) else np.nan,
                'roof': str(g.get('roof')), 'surface': str(g.get('surface')).strip().lower(),
                'temp': num(pd.Series([g.get('temp')]))[0], 'wind': num(pd.Series([g.get('wind')]))[0],
                'rest': num(pd.Series([g.get(f'{side}_rest')]))[0], 'opp_rest': num(pd.Series([g.get(f'{other}_rest')]))[0],
                'div_game': num(pd.Series([g.get('div_game')]))[0], 'weekday': str(g.get('weekday')),
                'kick_hour': hour, 'primetime': float(str(g.get('weekday')) != 'Sunday' or (hour == hour and hour >= 19)),
            })
    out = pd.DataFrame(rows)
    out['rest_diff'] = out['rest'] - out['opp_rest']
    out['dome'] = out['roof'].isin(('dome', 'closed')).astype(float)
    return out


def stage_assemble():
    from data.utils import clean_name_exact
    files = sorted(glob.glob(os.path.join(BOARDS, '*.parquet')))
    log(f'assemble: {len(files)} boards')
    frames = [pd.read_parquet(f) for f in files]
    p = pd.concat(frames, ignore_index=True)
    del frames
    p = p.rename(columns={'Player': 'player', 'Pos': 'pos', 'Team': 'team', 'Opponent': 'opp'})
    p = p[p['pos'].isin(POS)].reset_index(drop=True)
    p['key'] = clean_name_exact(p['player'].astype(str))
    p['_rid'] = np.arange(len(p))
    # model and actual
    for stat in STATS:
        p['p_' + stat] = num(p[stat]) if stat in p.columns else np.nan
        ac = 'a_' + stat if 'a_' + stat in p.columns else next(
            ('a_' + a for a in ALIASES.get(stat, ()) if 'a_' + a in p.columns), None)
        y = num(p[ac]) if ac else pd.Series(np.nan, index=p.index)
        p['y_' + stat] = np.where(p['played'].astype(bool), y.fillna(0.0), np.nan)
    p['p_points'] = num(p['Model Proj Pts'])
    p['p_raw_points'] = num(p['Raw Model Proj Pts'])
    p['y_points'] = np.where(p['played'].astype(bool), num(p['actual']), np.nan)
    p['p_anytime_td'] = p['p_rushing_tds'].fillna(0) + p['p_receiving_tds'].fillna(0)
    p['y_anytime_td'] = (p['y_rushing_tds'].fillna(0) + p['y_receiving_tds'].fillna(0) >= 1).astype(float).where(
        p['played'].astype(bool))
    # context, season by season
    hist_parts, opp_parts, team_parts, sched_parts, prior_parts = [], [], [], [], []
    for year in sorted(p['year'].unique()):
        year = int(year)
        sub = p[p['year'] == year]
        logs = season_logs(year)
        hist = player_history(logs)
        hcols = [c for c in hist.columns if c.startswith('h_')]
        hist_parts.append(asof_join(sub, hist, ('key', 'pos'), hcols))
        dfa = defense_allowed(logs)
        ocols = [c for c in dfa.columns if c.startswith('o_')]
        oj = asof_join(sub.assign(opp=sub['opp'].astype(str)), dfa, ('opp', 'pos'), ocols)
        opp_parts.append(oj)
        tv = team_volume(logs)
        tcols = [c for c in tv.columns if c.startswith('t_')]
        team_parts.append(asof_join(sub.assign(team=sub['team'].astype(str)), tv, ('team',), tcols))
        sched_parts.append(sub[['_rid', 'year', 'week', 'team']].merge(schedule_context(year), on=['year', 'week', 'team'],
                                                                        how='left').set_index('_rid'))
        try:
            prev = season_logs(year - 1)
            g = prev.groupby(['key', 'pos'])
            ps = g[list(HIST_STATS)].mean()
            ps.columns = [f'h_ps_{c}' for c in ps.columns]
            ps['h_ps_games'] = g.size()
            prior_parts.append(sub[['_rid', 'key', 'pos']].merge(ps.reset_index(), on=['key', 'pos'], how='left')
                               .set_index('_rid').drop(columns=['key', 'pos']))
        except Exception:
            log(f'  {year}: no prior season\n' + traceback.format_exc())
        log(f'  context {year}: {len(sub)} rows')
    p = p.set_index('_rid')
    p = p.join(pd.concat(hist_parts)).join(pd.concat(opp_parts)).join(pd.concat(team_parts))
    sched = pd.concat(sched_parts).drop(columns=['year', 'week', 'team'])
    p = p.join(sched)
    if prior_parts:
        p = p.join(pd.concat(prior_parts))
    p = p.reset_index()
    # league-relative defense: the opponent's allowed fantasy points to the position over the league's mean that week
    lg = p.groupby(['year', 'week', 'pos'])['o_fantasy_points_ppr'].transform('mean')
    p['o_pts_rel'] = p['o_fantasy_points_ppr'] / lg
    # teammates out: same team, Availability <= 0.25, with a season-to-date line worth something
    avail = num(p['Availability']).fillna(1.0)
    s2d_pts = p['h_s2d_fantasy_points_ppr'].fillna(p.get('h_ps_fantasy_points_ppr', pd.Series(0, index=p.index))).fillna(0)
    out_mask = (avail <= 0.25) & (s2d_pts >= 5)
    p['_out_pts'] = np.where(out_mask, s2d_pts, 0.0)
    p['_out_n'] = out_mask.astype(float)
    grp = p.groupby(['year', 'week', 'team'])
    p['tm_out_pts_total'] = grp['_out_pts'].transform('sum') - p['_out_pts']
    for pos in POS:
        m = p['pos'] == pos
        tot = p['_out_pts'].where(m, 0.0).groupby([p['year'], p['week'], p['team']]).transform('sum')
        p[f'tm_out_pts_{pos}'] = tot - np.where(m, p['_out_pts'], 0.0)
    p = p.drop(columns=['_out_pts', '_out_n'])
    # derived
    p = p.copy()
    extra = {}
    for stat in (*STATS, 'points'):
        hs = 'h_s2d_fantasy_points_ppr' if stat == 'points' else f'h_s2d_{stat}'
        hl = 'h_l3_fantasy_points_ppr' if stat == 'points' else f'h_l3_{stat}'
        hp = 'h_ps_fantasy_points_ppr' if stat == 'points' else f'h_ps_{stat}'
        nan = pd.Series(np.nan, index=p.index)
        extra[f'dev_s2d_{stat}'] = p[f'p_{stat}'] - p[hs] if hs in p.columns else nan
        extra[f'dev_ps_{stat}'] = p[f'p_{stat}'] - p[hp] if hp in p.columns else nan
        extra[f'trend_{stat}'] = p[hl] - p[hs] if hs in p.columns else nan
    extra['rookie_like'] = p['h_ps_games'].isna().astype(float) if 'h_ps_games' in p.columns else pd.Series(np.nan, index=p.index)
    p = pd.concat([p, pd.DataFrame(extra, index=p.index)], axis=1)
    p.columns = [str(c) for c in p.columns]
    p.to_parquet(PANEL, index=False)
    log(f'assemble: panel {p.shape[0]:,} rows x {p.shape[1]} columns -> {PANEL}')


def load_panel():
    p = pd.read_parquet(PANEL)
    p['played'] = p['played'].astype(bool)
    return p


def target_rows(p, pos, stat, weeks=(3, 17)):
    m = (p['pos'] == pos) & p['played'] & p['week'].between(*weeks) & p[f'p_{stat}'].notna() & p[f'y_{stat}'].notna()
    if pos == 'QB' and 'QB Projected Starter' in p.columns:
        m &= p['QB Projected Starter'].astype(str).str.lower().isin(('true', '1', '1.0'))
    return p[m]


# ------------------------------------------------------------------------------------------------------------- accuracy
def baseline_cols(stat):
    s = 'fantasy_points_ppr' if stat == 'points' else stat
    return {'season_to_date': f'h_s2d_{s}', 'last_3': f'h_l3_{s}', 'last_game': f'h_last_{s}', 'last_season': f'h_ps_{s}'}


def stage_accuracy():
    from sklearn.linear_model import LinearRegression
    p = load_panel()
    log('accuracy: model vs baselines, and the leave-one-season-out stack')
    rows, stack_rows = [], []
    for pos, stats in TARGETS.items():
        for stat in stats:
            d = target_rows(p, pos, stat)
            if len(d) < 300 * MS:
                continue
            y, m = d[f'y_{stat}'].to_numpy(float), d[f'p_{stat}'].to_numpy(float)
            cl = (d['year'] * 100 + d['week']).to_numpy()
            bl = baseline_cols(stat)
            for name, col in bl.items():
                if col not in d.columns:
                    continue
                b = d[col].to_numpy(float)
                ok = ~np.isnan(b)
                if name == 'last_season':
                    ok &= d['h_ps_games'].fillna(0).to_numpy() >= 4
                if ok.sum() < 200 * MS:
                    continue
                em, eb = (m - y)[ok], (b - y)[ok]
                diff, se = clustered_mean_se(em ** 2 - eb ** 2, cl[ok])
                rows.append({'pos': pos, 'stat': stat, 'baseline': name, 'n': int(ok.sum()),
                             'rmse_model': float(np.sqrt((em ** 2).mean())), 'rmse_base': float(np.sqrt((eb ** 2).mean())),
                             'skill': float(1 - (em ** 2).mean() / (eb ** 2).mean()), 'mse_diff': diff, 'mse_diff_se': se,
                             'bias_model': float(em.mean()), 'bias_base': float(eb.mean()),
                             'corr_model': float(np.corrcoef(m[ok], y[ok])[0, 1]), 'corr_base': float(np.corrcoef(b[ok], y[ok])[0, 1])})
            # the stack: does history the model already saw add anything out of sample?
            X = np.column_stack([m] + [np.where(np.isnan(d[c].to_numpy(float)), m, d[c].to_numpy(float))
                                       for c in bl.values() if c in d.columns]
                                + [np.isnan(d[c].to_numpy(float)).astype(float) for c in bl.values() if c in d.columns])
            yr = d['year'].to_numpy()
            pred_cal, pred_stack = np.full(len(d), np.nan), np.full(len(d), np.nan)
            for t in np.unique(yr):
                tr = ((yr != t) & (yr <= 2025)) if t <= 2025 else (yr <= 2025)
                te = yr == t
                if tr.sum() < 200 * MS or te.sum() < 30 * MS:
                    continue
                pred_cal[te] = LinearRegression().fit(X[tr, :1], y[tr]).predict(X[te, :1])
                pred_stack[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
            ok = ~np.isnan(pred_stack)
            coef = LinearRegression().fit(X[yr <= 2025], y[yr <= 2025]).coef_
            names = ['model'] + [n for n, c in bl.items() if c in d.columns]
            for scope, sel in (('2021-2025', ok & (yr <= 2025)), ('2026', ok & (yr == 2026))):
                if sel.sum() < 30 * MS:
                    continue
                r0, r1, r2 = (np.sqrt(np.mean((v[sel] - y[sel]) ** 2)) for v in (m, pred_cal, pred_stack))
                d01, se01 = clustered_mean_se((pred_stack[sel] - y[sel]) ** 2 - (pred_cal[sel] - y[sel]) ** 2, cl[sel])
                stack_rows.append({'pos': pos, 'stat': stat, 'scope': scope, 'n': int(sel.sum()), 'rmse_model': r0,
                                   'rmse_model_recal': r1, 'rmse_stack': r2, 'stack_gain_pct': 100 * (1 - r2 / r1),
                                   'z_stack_vs_recal': d01 / se01 if se01 else np.nan,
                                   **{f'coef_{n}': float(c) for n, c in zip(names, coef[:len(names)])}})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'accuracy_baselines.csv'), index=False)
    pd.DataFrame(stack_rows).to_csv(os.path.join(OUT, 'accuracy_stack.csv'), index=False)
    log(f'accuracy: {len(rows)} baseline rows, {len(stack_rows)} stack rows')


# ------------------------------------------------------------------------------------------------------------- segments
SEGMENT_FEATURES = (
    'p_level', 'week', 'Role Confidence', 'Expected Snap Share', 'Games This Season', 'Availability', 'Workload If Active',
    'Injury Status', '_profile_source_kind', '_profile_target_earner_rank', 'Role Change Confidence', 'home', 'spread', 'total',
    'imp_team', 'imp_opp', 'roof', 'surface', 'temp', 'wind', 'rest', 'rest_diff', 'div_game', 'primetime', 'weekday',
    'kick_hour', 'o_pts_rel', 'o_games', 't_pass_rate', 't_pass_rate_l3', 't_pass_att', 't_carries', 'tm_out_pts_total',
    'tm_out_pts_QB', 'tm_out_pts_RB', 'tm_out_pts_WR', 'tm_out_pts_TE', 'h_games', 'h_ps_games', 'rookie_like', 'trend',
    'dev_s2d', 'dev_ps', 'cv_hist', 'h_s2d_target_share', 'h_s2d_air_yards_share', 'h_s2d_wopr', 'h_l3_target_share',
    'h_s2d_passing_cpoe', 'h_s2d_sacks_suffered', 'h_s2d_receiving_epa', 'h_s2d_rushing_epa', 'h_s2d_passing_epa',
)


def segment_values(d, feat, stat):
    if feat == 'p_level':
        return d[f'p_{stat}']
    if feat in ('trend', 'dev_s2d', 'dev_ps'):
        return d[f'{feat}_{stat}']
    if feat == 'cv_hist':
        s = 'fantasy_points_ppr' if stat == 'points' else stat
        return d.get(f'h_sd_{s}', pd.Series(np.nan, index=d.index)) / d.get(f'h_s2d_{s}', pd.Series(np.nan, index=d.index)).replace(0, np.nan)
    return d[feat] if feat in d.columns else None


def binned(v, nbins=10):
    """Labels for a feature: its own values when few, else deciles; missing is its own bin."""
    if v.dtype == object or str(v.dtype) in ('category', 'bool'):
        lab = v.astype(str)
    else:
        x = num(v)
        if x.nunique(dropna=True) <= 12:
            lab = x.map(lambda t: 'NA' if pd.isna(t) else f'{t:g}')
        else:
            try:
                q = pd.qcut(x, nbins, duplicates='drop')
                lab = q.astype(str)
            except ValueError:
                lab = x.astype(str)
            lab = lab.where(x.notna(), 'NA')
    return lab.fillna('NA')


def stage_segments():
    p = load_panel()
    log('segments: residual bias by context bin')
    out = []
    for pos, stats in TARGETS.items():
        for stat in stats:
            d = target_rows(p, pos, stat)
            if len(d) < 500 * MS:
                continue
            r = (d[f'p_{stat}'] - d[f'y_{stat}']).to_numpy(float)          # projection - actual: + = projected too high
            cl = (d['year'] * 100 + d['week']).to_numpy()
            yr = d['year'].to_numpy()
            overall = r.mean()
            sse = float((r ** 2).sum())
            scale = float(np.nanmean(np.abs(d[f'y_{stat}'])))
            for feat in SEGMENT_FEATURES:
                v = segment_values(d, feat, stat)
                if v is None or v.notna().sum() < 300 * MS:
                    continue
                lab = binned(v).to_numpy()
                for b in pd.unique(lab):
                    sel = lab == b
                    n = int(sel.sum())
                    if n < 150 * MS:
                        continue
                    bias, se = clustered_mean_se(r[sel], cl[sel])
                    if not se or np.isnan(se):
                        continue
                    z_ex = (bias - overall) / se
                    seasons = [r[sel & (yr == t)].mean() for t in np.unique(yr) if (sel & (yr == t)).sum() >= 25]
                    same = np.mean([np.sign(s - overall) == np.sign(bias - overall) for s in seasons]) if seasons else np.nan
                    out.append({'pos': pos, 'stat': stat, 'feature': feat, 'bin': b, 'n': n, 'bias': bias, 'overall_bias': overall,
                                'excess': bias - overall, 'se': se, 'z_excess': z_ex, 'p': 2 * norm_sf(abs(z_ex)),
                                'excess_rel': (bias - overall) / scale if scale else np.nan,
                                'seasons': len(seasons), 'season_consistency': same,
                                'sse_share_removable': n * (bias - overall) ** 2 / sse})
    seg = pd.DataFrame(out)
    seg['q'] = bh_qvalues(seg['p'].to_numpy())
    seg = seg.sort_values('sse_share_removable', ascending=False)
    seg.to_csv(os.path.join(OUT, 'segments.csv'), index=False)
    log(f'segments: {len(seg):,} bins tested, {int((seg["q"] < 0.05).sum())} at q < 0.05')


# ------------------------------------------------------------------------------------------------------------- discover
EXCLUDE_PREFIX = ('y_', 'a_', '_rid')
EXCLUDE = {'actual', 'played', 'played_box', 'silent', 'year', 'key', 'player', 'team', 'opp', 'g_opp', 'pos'}
CATEGORICAL = ('Injury Status', '_profile_source_kind', '_role_confidence_method', 'roof', 'surface', 'weekday',
               'QB1 Workload Source', '_availability_source', 'QB Projected Starter', 'QB1 Selection Required')


def feature_matrix(d):
    cols, cats = [], []
    for c in d.columns:
        if c in EXCLUDE or c.startswith(EXCLUDE_PREFIX):
            continue
        if c in CATEGORICAL:
            cats.append(c)
            continue
        if pd.api.types.is_numeric_dtype(d[c]) or pd.api.types.is_bool_dtype(d[c]):
            if d[c].notna().mean() > 0.02 and d[c].nunique(dropna=True) > 1:
                cols.append(c)
    X = d[cols].astype(float)
    for c in cats:
        codes = pd.Categorical(d[c].astype(str)).codes.astype(float)
        X[c] = np.where(codes < 0, np.nan, codes)
    return X, cats


def hgb(**kw):
    from sklearn.ensemble import HistGradientBoostingRegressor
    base = dict(max_iter=250, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=1.0,
                random_state=SEED)
    base.update(kw)
    return HistGradientBoostingRegressor(**base)


def usable_cols(X):
    """Columns with at least two distinct non-missing values in X (a fold can lose a column another season has)."""
    return [c for c in X.columns if X[c].nunique(dropna=True) > 1]


def oos_residual_model(X, r, yr, cats, shuffle_groups=None, rng=None):
    pred = np.full(len(r), np.nan)
    models = {}
    for t in np.unique(yr):
        tr = (yr != t) & ((yr <= 2025) | (t == 2026)) if t <= 2025 else (yr <= 2025)
        te = yr == t
        if tr.sum() < 300 * MS or te.sum() < 50 * MS:
            continue
        rt = r[tr].copy()
        if shuffle_groups is not None:
            g = shuffle_groups[tr]
            for k in np.unique(g):
                idx = np.where(g == k)[0]
                rt[idx] = rt[rng.permutation(idx)]
        cols = usable_cols(X[tr])
        cat_mask = [c in cats for c in cols]
        m = hgb(categorical_features=cat_mask if any(cat_mask) else None).fit(X.loc[tr, cols], rt)
        pred[te] = m.predict(X.loc[te, cols])
        models[t] = (m, cols)
    return pred, models


DISCOVER_PRIORITY = [('WR', 'points'), ('RB', 'points'), ('TE', 'points'), ('QB', 'points'),
                     ('WR', 'receiving_yards'), ('WR', 'receptions'), ('RB', 'rushing_yards'), ('QB', 'passing_yards'),
                     ('TE', 'receiving_yards'), ('WR', 'targets'), ('RB', 'rushing_attempts'), ('QB', 'passing_attempts'),
                     ('TE', 'receptions'), ('RB', 'receptions'), ('RB', 'receiving_yards'), ('QB', 'passing_tds'),
                     ('QB', 'rushing_yards'), ('TE', 'targets'), ('RB', 'targets'), ('WR', 'receiving_tds'),
                     ('RB', 'rushing_tds'), ('TE', 'receiving_tds'), ('QB', 'passing_completions'),
                     ('QB', 'passing_interceptions'), ('QB', 'rushing_attempts')]


def stage_discover(n_null=2, n_repeats=3, top_k=12):
    p = load_panel()
    log('discover: boosted residual models, leave one season out')
    rng = np.random.default_rng(SEED)
    summary, imps, shapes = [], [], []
    budget = float(os.environ.get('DEEP_AUDIT_DISCOVER_MIN', '150')) * 60
    start = time.time()
    order = sorted(((pos, stat) for pos, stats in TARGETS.items() for stat in stats),
                   key=lambda ps: DISCOVER_PRIORITY.index(ps) if ps in DISCOVER_PRIORITY else 99)
    for pos, stat in order:
        if time.time() - start > budget:
            log(f'  {pos} {stat}: skipped, discover time budget ({budget / 60:.0f} min) used')
            continue
        d = target_rows(p, pos, stat)
        d = d[num(d['p_points']) >= 3.0].reset_index(drop=True)   # rows someone would start or bet
        if len(d) < 1000 * MS:
            continue
        t0 = time.time()
        X, cats = feature_matrix(d)
        r = (d[f'y_{stat}'] - d[f'p_{stat}']).to_numpy(float)
        yr = d['year'].to_numpy()
        groups = (d['year'] * 100 + d['week']).to_numpy()
        pred, models = oos_residual_model(X, r, yr, cats)
        ok = ~np.isnan(pred)
        base = np.mean(r[ok] ** 2)
        row = {'pos': pos, 'stat': stat, 'n': int(ok.sum()), 'features': X.shape[1], 'rmse_model': float(np.sqrt(base))}
        for s in (0.5, 1.0):
            row[f'gain_pct_s{s}'] = float(100 * (1 - np.mean((r[ok] - s * pred[ok]) ** 2) / base))
        for t in np.unique(yr[ok]):
            sel = ok & (yr == t)
            row[f'gain_pct_{t}'] = float(100 * (1 - np.mean((r[sel] - 0.5 * pred[sel]) ** 2) / np.mean(r[sel] ** 2)))
        nulls = []
        for _ in range(n_null):
            pn, _m = oos_residual_model(X, r, yr, cats, shuffle_groups=groups, rng=rng)
            okn = ~np.isnan(pn)
            nulls.append(float(100 * (1 - np.mean((r[okn] - 0.5 * pn[okn]) ** 2) / np.mean(r[okn] ** 2))))
        row['null_gain_max'] = max(nulls) if nulls else np.nan
        row['null_gains'] = ';'.join(f'{x:.3f}' for x in nulls)
        # permutation importance on each held-out season (loss = MSE of r - pred)
        # (screen every feature with one shuffle on up to 2,500 held-out rows, then n_repeats for the top 40)
        acc = {}
        for t, (m, cols) in models.items():
            te = np.where(yr == t)[0]
            if len(te) > 2500:
                te = np.sort(rng.choice(te, 2500, replace=False))
            Xt = X.iloc[te][cols].copy()
            rt = r[te]
            l0 = np.mean((rt - m.predict(Xt)) ** 2)

            def perm_inc(c):
                saved = Xt[c].to_numpy().copy()
                Xt[c] = rng.permutation(saved)
                v = np.mean((rt - m.predict(Xt)) ** 2) - l0
                Xt[c] = saved
                return v
            incs = {c: [perm_inc(c)] for c in cols}
            for c in sorted(cols, key=lambda k: -incs[k][0])[:40]:
                incs[c] += [perm_inc(c) for _ in range(n_repeats - 1)]
            for c, inc in incs.items():
                acc.setdefault(c, []).append((np.mean(inc), len(te)))
        imp = pd.DataFrame([{'pos': pos, 'stat': stat, 'feature': c,
                             'importance': sum(v * n for v, n in vals) / sum(n for _, n in vals),
                             'positive_folds': sum(v > 0 for v, _ in vals), 'folds': len(vals)} for c, vals in acc.items()])
        imp['importance_pct_of_mse'] = 100 * imp['importance'] / base
        imp = imp.sort_values('importance', ascending=False)
        imps.append(imp)
        # the shape of each top feature: held-out residual and prediction by feature decile, by season
        for c in imp['feature'].head(top_k):
            v = X[c]
            lab = binned(v)
            for b in pd.unique(lab):
                sel = (lab == b).to_numpy() & ok
                if sel.sum() < 60:
                    continue
                by = {int(t): float(r[sel & (yr == t)].mean()) for t in np.unique(yr) if (sel & (yr == t)).sum() >= 15}
                shapes.append({'pos': pos, 'stat': stat, 'feature': c, 'bin': b, 'n': int(sel.sum()),
                               'mean_resid': float(r[sel].mean()), 'mean_oos_pred': float(pred[sel].mean()),
                               'feature_mean': float(num(v[sel]).mean()) if pd.api.types.is_numeric_dtype(v) else np.nan,
                               'by_season': json.dumps(by)})
        summary.append(row)
        log(f'  {pos} {stat}: n {row["n"]}, gain {row["gain_pct_s0.5"]:+.2f}% (null max {row["null_gain_max"]:+.2f}%), '
            f'top {", ".join(imp["feature"].head(4))}  [{time.time() - t0:.0f}s]')
        pd.DataFrame(summary).to_csv(os.path.join(OUT, 'discover_summary.csv'), index=False)
        pd.concat(imps).to_csv(os.path.join(OUT, 'discover_importance.csv'), index=False)
        pd.DataFrame(shapes).to_csv(os.path.join(OUT, 'discover_shapes.csv'), index=False)


# ---------------------------------------------------------------------------------------------------------------- props
PROP_SPECS = {
    'passing_yards': dict(pos=('QB',), kind='yards', market='passing_yards', min_p=150.0, grid=600, offsets=(-40, -20, 0, 20, 40)),
    'passing_attempts': dict(pos=('QB',), kind='count', market='attempts', min_p=20.0, grid=75, offsets=(-4, -2, 0, 2, 4)),
    'passing_completions': dict(pos=('QB',), kind='count', market='passing_completions', min_p=12.0, grid=55,
                                offsets=(-3, -2, 0, 2, 3)),
    'passing_tds': dict(pos=('QB',), kind='count', market='passing_tds', min_p=0.8, grid=8, offsets=(-1, 0, 1)),
    'passing_interceptions': dict(pos=('QB',), kind='count', market='passing_interceptions', min_p=0.3, grid=7, offsets=(0, 1)),
    'rushing_attempts': dict(pos=('QB', 'RB'), kind='count', market='carries', min_p=4.0, grid=45, offsets=(-4, -2, 0, 2, 4)),
    'rushing_yards': dict(pos=('QB', 'RB'), kind='yards', market='rushing_yards', min_p=12.0, grid=260, offsets=(-20, -10, 0, 10, 20)),
    'targets': dict(pos=('RB', 'WR', 'TE'), kind='count', market='targets', min_p=2.0, grid=25, offsets=(-2, -1, 0, 1, 2)),
    'receptions': dict(pos=('RB', 'WR', 'TE'), kind='count', market='receptions', min_p=1.5, grid=20, offsets=(-2, -1, 0, 1, 2)),
    'receiving_yards': dict(pos=('RB', 'WR', 'TE'), kind='yards', market='receiving_yards', min_p=12.0, grid=260,
                            offsets=(-20, -10, 0, 10, 20)),
}
TAUS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                 0.98])
QFEATS = ('Role Confidence', 'Expected Snap Share', 'Games This Season', 'Availability', 'Workload If Active', 'imp_team',
          'spread', 'total', 'home', 'dome', 'wind', 'temp', 'o_pts_rel', 'tm_out_pts_total', 't_pass_rate_l3', 'week',
          'h_games', 'h_ps_games')
METHODS = ('M0_poisson_normal', 'M1_empirical', 'M2_fitted', 'M3_quantile_gb')


def prop_rows(p, stat):
    spec = PROP_SPECS[stat]
    m = p['pos'].isin(spec['pos']) & p['played'] & p['week'].between(3, 17) & (p[f'p_{stat}'] >= spec['min_p'])
    m &= num(p['Availability']).fillna(1.0) > 0.25
    if 'QB Projected Starter' in p.columns:
        m &= ~((p['pos'] == 'QB') & ~p['QB Projected Starter'].astype(str).str.lower().isin(('true', '1', '1.0')))
    return p[m].reset_index(drop=True)


def qfeatures(d, stat):
    s = stat
    X = pd.DataFrame({'p': d[f'p_{s}'].astype(float), 'logp': np.log1p(d[f'p_{s}'].astype(float)),
                      'pos': d['pos'].map({k: i for i, k in enumerate(POS)}).astype(float)})
    for c in QFEATS:
        X[c] = num(d[c]) if c in d.columns else np.nan
    for c in (f'h_s2d_{s}', f'h_l3_{s}', f'h_sd_{s}', f'h_ps_{s}'):
        X[c] = num(d[c]) if c in d.columns else np.nan
    X['cv'] = X[f'h_sd_{s}'] / X[f'h_s2d_{s}'].replace(0, np.nan)
    X['inj'] = pd.Categorical(d['Injury Status'].astype(str)).codes.astype(float) if 'Injury Status' in d.columns else np.nan
    return X


def cdf_m0(m, pos, spec, grid):
    from scipy.stats import norm, poisson
    from data.market_devig import _yard_sigma
    k = np.arange(grid + 1)
    if spec['kind'] == 'count':
        return poisson.cdf(k[None, :], np.maximum(m, 1e-6)[:, None])
    sig = np.array([_yard_sigma(ps, spec['market'], 'game') for ps in pos])
    return norm.cdf((k[None, :] + 0.5 - m[:, None]) / sig[:, None])


def fit_m1(m, y, nb_min=8, nb_max=40):
    """Projection-level bins (on log mean) with each bin's sorted ratio sample actual / mean."""
    nb = int(np.clip(len(m) / 400, nb_min, nb_max))
    lm = np.log(np.maximum(m, 1e-6))
    edges = np.unique(np.quantile(lm, np.linspace(0, 1, nb + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.clip(np.searchsorted(edges, lm, side='right') - 1, 0, len(edges) - 2)
    ratios = {int(i): np.sort(y[b == i] / np.maximum(m[b == i], 1e-6)) for i in np.unique(b)}
    return {'edges': edges, 'ratios': ratios}


def cdf_m1(fit, m, grid):
    lm = np.log(np.maximum(m, 1e-6))
    b = np.clip(np.searchsorted(fit['edges'], lm, side='right') - 1, 0, len(fit['edges']) - 2)
    k = np.arange(grid + 1)
    F = np.zeros((len(m), grid + 1))
    for i in np.unique(b):
        rows = np.where(b == i)[0]
        R = fit['ratios'].get(int(i))
        if R is None or not len(R):
            R = np.concatenate(list(fit['ratios'].values()))
            R.sort()
        thr = (k[None, :] + 0.5) / np.maximum(m[rows], 1e-6)[:, None]       # P(m * R < k + 0.5)
        F[rows] = np.searchsorted(R, thr.ravel(), side='left').reshape(thr.shape) / len(R)
    return F


def fit_m2(m, y, kind):
    """counts: a calibrated mean (y ~ a + b m) and a negative-binomial size r; yards: P(0) logistic in log m, the positive part
    a gamma with mean c * m^e and a coefficient of variation linear in log m."""
    from sklearn.linear_model import LinearRegression, LogisticRegression
    if kind == 'count':
        lr = LinearRegression().fit(m[:, None], y)
        mu = np.maximum(lr.predict(m[:, None]), 1e-3)
        den = ((y - mu) ** 2 - mu).sum()
        r = float((mu ** 2).sum() / den) if den > 0 else 1e6
        return {'a': float(lr.intercept_), 'b': float(lr.coef_[0]), 'r': min(r, 1e6)}
    lm = np.log(np.maximum(m, 1e-3))
    zero = (y <= 0).astype(int)
    lg = LogisticRegression(C=10.0).fit(lm[:, None], zero) if 0 < zero.sum() < len(zero) else None
    pos = y > 0
    e, c = np.polyfit(lm[pos], np.log(y[pos]), 1)
    # E[y | y > 0] is not exp(E[log y]): rescale so the positive mean matches in each projection band
    scale = float(y[pos].mean() / np.exp(c + e * lm[pos]).mean())
    nb = 12
    edges = np.quantile(lm[pos], np.linspace(0, 1, nb + 1))
    mids, cvs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = pos & (lm >= lo) & (lm <= hi)
        if sel.sum() >= 50:
            mids.append(lm[sel].mean())
            cvs.append(y[sel].std() / y[sel].mean())
    k1, k0 = np.polyfit(mids, cvs, 1) if len(mids) >= 3 else (0.0, float(y[pos].std() / y[pos].mean()))
    return {'logit_a': float(lg.intercept_[0]) if lg else -9.0, 'logit_b': float(lg.coef_[0][0]) if lg else 0.0,
            'e': float(e), 'c': float(c), 'scale': scale, 'cv0': float(k0), 'cv1': float(k1)}


def cdf_m2(fit, m, kind, grid):
    from scipy.stats import gamma, nbinom
    k = np.arange(grid + 1)
    if kind == 'count':
        mu = np.maximum(fit['a'] + fit['b'] * m, 1e-3)
        r = fit['r']
        return nbinom.cdf(k[None, :], r, (r / (r + mu))[:, None])
    lm = np.log(np.maximum(m, 1e-3))
    p0 = 1 / (1 + np.exp(-(fit['logit_a'] + fit['logit_b'] * lm)))
    mpos = fit['scale'] * np.exp(fit['c'] + fit['e'] * lm)
    cv = np.clip(fit['cv0'] + fit['cv1'] * lm, 0.15, 3.0)
    shape = 1 / cv ** 2
    G = gamma.cdf(k[None, :] + 0.5, shape[:, None], scale=(mpos / shape)[:, None])
    return p0[:, None] + (1 - p0[:, None]) * G


def fit_m3(X, y):
    cols = usable_cols(X)
    models = []
    for tau in TAUS:
        models.append(hgb(loss='quantile', quantile=float(tau), max_iter=200, learning_rate=0.05, min_samples_leaf=100)
                      .fit(X[cols], y))
    return cols, models


def cdf_m3(fit, X, grid):
    cols, models = fit
    Q = np.column_stack([m.predict(X[cols]) for m in models])
    Q = np.sort(Q, axis=1)
    k = np.arange(grid + 1) + 0.5
    F = np.empty((len(X), grid + 1))
    lo_y, hi_y = -0.5, grid + 1.0
    for i in range(len(X)):
        xs = np.concatenate([[lo_y], Q[i], [hi_y]])
        ys = np.concatenate([[0.0], TAUS, [1.0]])
        xs = np.maximum.accumulate(xs + np.arange(len(xs)) * 1e-9)
        F[i] = np.interp(k, xs, ys)
    return F


def score_cdf(F, y, rng):
    """CRPS (discrete ranked probability score on the integer grid), randomized PIT, interval coverage."""
    G = F.shape[1]
    yi = np.clip(np.round(np.maximum(y, 0)).astype(int), 0, G - 1)
    ind = (np.arange(G)[None, :] >= yi[:, None]).astype(float)
    crps = ((F - ind) ** 2).sum(axis=1)
    Fy = F[np.arange(len(y)), yi]
    Fy1 = np.where(yi > 0, F[np.arange(len(y)), np.maximum(yi - 1, 0)], 0.0)
    pit = Fy1 + rng.random(len(y)) * (Fy - Fy1)
    hist = np.histogram(pit, bins=10, range=(0, 1))[0] / len(pit)
    cov = {}
    for lo, hi in ((0.25, 0.75), (0.1, 0.9)):
        cov[f'cover_{int((hi - lo) * 100)}'] = float(np.mean((pit >= lo) & (pit <= hi)))
    return crps, pit, {'pit_max_dev': float(np.abs(hist - 0.1).max()), 'pit_hist': ';'.join(f'{h:.3f}' for h in hist),
                       'pit_mean': float(pit.mean()), **cov}


def p_over_at(F, line):
    """P(Y > line) for half-point lines (an integer line counts a push as neither: P(Y > line))."""
    idx = np.clip(np.floor(line).astype(int), 0, F.shape[1] - 1)
    return 1 - F[np.arange(len(line)), idx]


def reliability(prob, outcome, nb=10):
    df = pd.DataFrame({'p': prob, 'o': outcome})
    df['bin'] = np.clip((df['p'] * nb).astype(int), 0, nb - 1)
    g = df.groupby('bin').agg(n=('o', 'size'), p_mean=('p', 'mean'), o_mean=('o', 'mean'))
    ece = float((g['n'] * (g['p_mean'] - g['o_mean']).abs()).sum() / g['n'].sum())
    return ece, g


def posted_lines_2026():
    """{(week, key, market): (line, p_over, book)}: the last scorable line with a de-vigged price (Pinnacle first, then
    DraftKings) pulled before the player's team kicked off."""
    from data.ledger_scorecard import kickoffs_utc
    from data.loaders import load_schedule
    from scripts.eval_posted_line_record import load_archive
    arc = load_archive(2026)
    arc = arc[arc['p_over'].notna()]
    return arc, kickoffs_utc(load_schedule(2026))


def posted_join(dt, spec, arc, kick):
    """dt's rows (index 0..n-1) matched to the last priced line pulled before the player's team kicked off - the sharpest book
    first (Pinnacle, then DraftKings) - with the first line that book posted that week: i, week, book, line, p_book, open_line,
    open_p_over."""
    cols = ['i', 'week', 'book', 'line', 'p_book', 'open_line', 'open_p_over']
    mk = arc[arc['market'] == spec['market']]
    if not len(mk):
        return pd.DataFrame(columns=cols)
    when = {(int(w), k): kick.get((int(w), tm)) for w, k, tm in zip(dt['week'], dt['key'], dt['team'])}
    a = mk.copy()
    a['_ko'] = [when.get((w, k)) for w, k in zip(a['week'], a['_key'])]
    a = a[a['_ko'].notna()]
    a = a[a['fetched_at'] <= pd.to_datetime(a['_ko'], utc=True)]
    if not len(a):
        return pd.DataFrame(columns=cols)
    last = a.sort_values('fetched_at').groupby(['week', '_key', 'book']).last().reset_index()
    first = a.sort_values('fetched_at').groupby(['week', '_key', 'book']).first().reset_index()
    last = last.merge(first[['week', '_key', 'book', 'line', 'p_over']].rename(
        columns={'line': 'open_line', 'p_over': 'open_p_over'}), on=['week', '_key', 'book'], how='left')
    last = last.sort_values('_rank').groupby(['week', '_key']).first().reset_index()
    idx = {(int(w), k): i for i, (w, k) in enumerate(zip(dt['week'], dt['key']))}
    last['i'] = [idx.get((int(w), k)) for w, k in zip(last['week'], last['_key'])]
    last = last[last['i'].notna()].rename(columns={'p_over': 'p_book'})
    return last[cols].reset_index(drop=True)


def stage_props():
    from data.weekly_median import stat_median
    p = load_panel()
    rng = np.random.default_rng(SEED)
    log('props: distributions, forward-chained (test 2023-2026, trained on the seasons before)')
    scores, calib, rel_rows, posted_rows = [], [], [], []
    try:
        arc, kick = posted_lines_2026()
    except Exception:
        log('  no posted-line archive\n' + traceback.format_exc())
        arc, kick = None, None
    os.makedirs(os.path.join(OUT, 'prop_models'), exist_ok=True)
    for stat, spec in PROP_SPECS.items():
        t_stat = time.time()
        d = prop_rows(p, stat)
        if len(d) < 1000 * MS:
            continue
        y = d[f'y_{stat}'].to_numpy(float)
        m = d[f'p_{stat}'].to_numpy(float)
        yr = d['year'].to_numpy()
        X = qfeatures(d, stat)
        grid = spec['grid']
        med = np.array([stat_median(ps, stat, mm) for ps, mm in zip(d['pos'], m)], dtype=float)
        base_line = np.floor(np.where(np.isnan(med), m, med)) + 0.5
        for t in (2023, 2024, 2025, 2026):
            tr, te = yr < t, yr == t
            if te.sum() < 50 * MS or tr.sum() < 500 * MS:
                continue
            cdfs = {'M0_poisson_normal': cdf_m0(m[te], d['pos'].to_numpy()[te], spec, grid)}
            pos_te = d['pos'].to_numpy()[te]
            F1 = np.zeros((te.sum(), grid + 1))
            F2 = np.zeros((te.sum(), grid + 1))
            for ps in np.unique(pos_te):
                trp = tr & (d['pos'].to_numpy() == ps)
                tep = pos_te == ps
                if trp.sum() < 300 * MS:
                    trp = tr
                F1[tep] = cdf_m1(fit_m1(m[trp], y[trp]), m[te][tep], grid)
                F2[tep] = cdf_m2(fit_m2(m[trp], y[trp], spec['kind']), m[te][tep], spec['kind'], grid)
            cdfs['M1_empirical'] = F1
            cdfs['M2_fitted'] = F2
            cdfs['M3_quantile_gb'] = cdf_m3(fit_m3(X[tr], y[tr]), X[te], grid)
            yt = y[te]
            for meth, F in cdfs.items():
                crps, pit, info = score_cdf(F, yt, rng)
                scores.append({'stat': stat, 'test_season': t, 'method': meth, 'n': int(te.sum()), 'crps': float(crps.mean()),
                               **info})
                for off in spec['offsets']:
                    line = base_line[te] + off
                    line = np.maximum(line, 0.5)
                    po = p_over_at(F, line)
                    outc = (yt > line).astype(float)
                    ece, g = reliability(po, outc)
                    calib.append({'stat': stat, 'test_season': t, 'method': meth, 'offset': off, 'n': len(outc),
                                  'brier': float(np.mean((po - outc) ** 2)), 'mean_p': float(po.mean()),
                                  'hit_rate': float(outc.mean()), 'ece': ece})
                    if off == 0:
                        for b, rowg in g.iterrows():
                            rel_rows.append({'stat': stat, 'test_season': t, 'method': meth, 'bin': int(b), **rowg.to_dict()})
            # 2026: the books' posted lines
            if t == 2026 and arc is not None:
                dt = d[te].reset_index(drop=True)
                for _, L in posted_join(dt, spec, arc, kick).iterrows():
                    i = int(L['i'])
                    yi = yt[i]
                    line = float(L['line'])
                    if yi == line:
                        continue
                    rec = {'stat': stat, 'week': int(L['week']), 'player': dt.at[i, 'player'], 'pos': dt.at[i, 'pos'],
                           'book': L['book'], 'line': line, 'p_book': float(L['p_book']), 'open_line': L['open_line'],
                           'open_p_over': L['open_p_over'], 'model_mean': float(m[te][i]),
                           'model_median': float(med[te][i]), 'actual': float(yi), 'over': float(yi > line)}
                    for meth, F in cdfs.items():
                        rec[f'p_{meth}'] = float(p_over_at(F[i:i + 1], np.array([line]))[0])
                    posted_rows.append(rec)
        # the engines on every season, for the weekly tooling
        allr = np.ones(len(d), dtype=bool)
        eng = {'stat': stat, 'spec': {k: v for k, v in spec.items() if k != 'offsets'}, 'M1': {}, 'M2': {}}
        for ps in np.unique(d['pos']):
            sel = d['pos'].to_numpy() == ps
            f1 = fit_m1(m[sel], y[sel])
            eng['M1'][ps] = {'edges': [float(e) for e in f1['edges'][1:-1]],
                             'ratio_quantiles': {int(k): [float(x) for x in np.quantile(v, np.linspace(0, 1, 201))]
                                                 for k, v in f1['ratios'].items()}}
            eng['M2'][ps] = fit_m2(m[sel], y[sel], spec['kind'])
        with open(os.path.join(OUT, 'prop_models', f'{stat}.json'), 'w', encoding='utf-8') as fh:
            json.dump(eng, fh)
        try:
            import joblib
            cols, models = fit_m3(X[allr], y[allr])
            joblib.dump({'features': cols, 'taus': TAUS.tolist(), 'models': models},
                        os.path.join(OUT, 'prop_models', f'{stat}_m3.joblib'))
        except Exception:
            log('  m3 save failed\n' + traceback.format_exc())
        pd.DataFrame(scores).to_csv(os.path.join(OUT, 'props_scores.csv'), index=False)
        pd.DataFrame(calib).to_csv(os.path.join(OUT, 'props_calibration.csv'), index=False)
        pd.DataFrame(rel_rows).to_csv(os.path.join(OUT, 'props_reliability.csv'), index=False)
        pd.DataFrame(posted_rows).to_csv(os.path.join(OUT, 'props_posted_2026.csv'), index=False)
        log(f'  {stat}: {len(d):,} rows  [{time.time() - t_stat:.0f}s]')
    # anytime TD (a yes/no market): Poisson on the projected TDs vs a fitted logistic vs boosted classification
    try:
        props_anytime_td(p, rng)
    except Exception:
        log('  anytime TD failed\n' + traceback.format_exc())


def props_anytime_td(p, rng):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    d = p[p['pos'].isin(('RB', 'WR', 'TE')) & p['played'] & p['week'].between(3, 17) & (p['p_anytime_td'] >= 0.1)
          & (num(p['Availability']).fillna(1.0) > 0.25)].reset_index(drop=True)
    y = d['y_anytime_td'].to_numpy(float)
    lam = d['p_anytime_td'].to_numpy(float)
    X = qfeatures(d.assign(p_anytime_td=lam), 'anytime_td')
    X['p_rec_td'] = d['p_receiving_tds'].to_numpy(float)
    X['p_rush_td'] = d['p_rushing_tds'].to_numpy(float)
    yr = d['year'].to_numpy()
    out = []
    for t in (2023, 2024, 2025, 2026):
        tr, te = yr < t, yr == t
        if te.sum() < 100 * MS or tr.sum() < 100 * MS:
            continue
        probs = {'poisson': 1 - np.exp(-lam[te])}
        lg = LogisticRegression(C=10.0).fit(np.column_stack([np.log(lam[tr]), d['pos'].map({'RB': 0, 'WR': 1, 'TE': 2})[tr]]), y[tr])
        probs['logistic'] = lg.predict_proba(np.column_stack([np.log(lam[te]), d['pos'].map({'RB': 0, 'WR': 1, 'TE': 2})[te]]))[:, 1]
        cols = usable_cols(X[tr])
        gb = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=100,
                                            random_state=SEED).fit(X.loc[tr, cols], y[tr])
        probs['boosted'] = gb.predict_proba(X.loc[te, cols])[:, 1]
        for meth, pr in probs.items():
            pr = np.clip(pr, 1e-4, 1 - 1e-4)
            ece, _g = reliability(pr, y[te])
            out.append({'test_season': t, 'method': meth, 'n': int(te.sum()), 'brier': float(np.mean((pr - y[te]) ** 2)),
                        'logloss': float(-np.mean(y[te] * np.log(pr) + (1 - y[te]) * np.log(1 - pr))),
                        'mean_p': float(pr.mean()), 'rate': float(y[te].mean()), 'ece': ece})
    pd.DataFrame(out).to_csv(os.path.join(OUT, 'props_anytime_td.csv'), index=False)
    log(f'  anytime TD: {len(d):,} rows')


# --------------------------------------------------------------------------------------------------------------- report
def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def posted_summary(df):
    """Hit rate and ROI by method and edge size (bet the side the model's P(over) favours against the book's de-vigged price)."""
    rows = []
    meths = [c[2:] for c in df.columns if c.startswith('p_M')]
    for stat in ['ALL'] + sorted(df['stat'].unique()):
        s = df if stat == 'ALL' else df[df['stat'] == stat]
        for meth in ['book'] + meths:
            pr = s['p_book'] if meth == 'book' else s[f'p_{meth}']
            rows.append({'stat': stat, 'method': meth, 'n': len(s), 'brier': float(np.mean((pr - s['over']) ** 2)),
                         'mean_p': float(pr.mean()), 'over_rate': float(s['over'].mean())})
    brier = pd.DataFrame(rows)
    bets = []
    for stat in ['ALL'] + sorted(df['stat'].unique()):
        s = df if stat == 'ALL' else df[df['stat'] == stat]
        for meth in meths:
            edge = s[f'p_{meth}'] - s['p_book']
            for thr in (0.03, 0.05, 0.08, 0.12):
                sel = edge.abs() >= thr
                if sel.sum() < 15:
                    continue
                side_over = edge[sel] > 0
                win = np.where(side_over, s.loc[sel, 'over'] == 1, s.loc[sel, 'over'] == 0)
                k, n = int(win.sum()), int(sel.sum())
                lo, hi = wilson(k, n)
                fair_p = np.where(side_over, s.loc[sel, 'p_book'], 1 - s.loc[sel, 'p_book'])
                bets.append({'stat': stat, 'method': meth, 'edge_min': thr, 'bets': n, 'win_rate': k / n, 'ci_lo': lo,
                             'ci_hi': hi, 'roi_at_-110': float(np.mean(np.where(win, 100 / 110, -1.0))),
                             'roi_at_fair_price': float(np.mean(np.where(win, 1 / fair_p - 1, -1.0)))})
    return brier, pd.DataFrame(bets)


def stage_report():
    parts = ['# Deep audit, 2026-10-28\n', 'Today\'s model (DEFAULT_FEATURES + the harness injury/reserve replay), every board '
             '2021-2025 weeks 1-18 and 2026 weeks 1-7, rebuilt with today\'s code. Graded rows: played (a box score or a silent '
             'game), weeks 3-17 unless stated; QBs = the projected starter. Files in .sweeps/deep_audit/.\n']

    def read(name):
        path = os.path.join(OUT, name)
        return pd.read_csv(path) if os.path.exists(path) else None

    acc = read('accuracy_baselines.csv')
    if acc is not None:
        parts.append('\n## 1. The model against simple baselines (same rows; skill = 1 - MSE model / MSE baseline)\n')
        piv = acc.pivot_table(index=['pos', 'stat'], columns='baseline', values='skill').reset_index()
        rm = acc.groupby(['pos', 'stat'])['rmse_model'].first().reset_index()
        parts.append(fmt_table(piv.merge(rm, on=['pos', 'stat']), '{:.3f}'))
    st = read('accuracy_stack.csv')
    if st is not None:
        parts.append('\n## 2. Information left on the table: a leave-one-season-out stack of model + history vs the model '
                     're-calibrated alone (positive gain = history the model under-uses)\n')
        cols = ['pos', 'stat', 'scope', 'n', 'rmse_model', 'rmse_model_recal', 'rmse_stack', 'stack_gain_pct', 'z_stack_vs_recal'] + \
            [c for c in st.columns if c.startswith('coef_')]
        parts.append(fmt_table(st[cols], '{:.3f}'))
    seg = read('segments.csv')
    if seg is not None:
        strong = seg[(seg['q'] < 0.05) & (seg['season_consistency'] >= 0.8) & (seg['seasons'] >= 4)]
        parts.append(f'\n## 3. Segments with a real bias (q < 0.05, same sign in at least 80% of seasons): {len(strong)} of '
                     f'{len(seg):,} bins. Top 50 by the share of squared error a fix would remove\n')
        parts.append(fmt_table(strong.head(50)[['pos', 'stat', 'feature', 'bin', 'n', 'bias', 'excess', 'excess_rel', 'z_excess', 'q',
                                                'season_consistency', 'sse_share_removable']], '{:.4f}'))
    ds = read('discover_summary.csv')
    if ds is not None:
        parts.append('\n## 4. Factors the model cannot see: boosted models of the residual (actual - projection) on rows '
                     'projected 3+ points, held out by season (gain = % of MSE removed at half strength; the null shuffles '
                     'the residual within each week; targets past the time budget are skipped)\n')
        parts.append(fmt_table(ds, '{:.3f}'))
        imp = read('discover_importance.csv')
        if imp is not None:
            top = imp.sort_values('importance', ascending=False).groupby(['pos', 'stat']).head(6)
            parts.append('\nTop features per target (permutation importance on held-out seasons, % of the residual MSE):\n\n')
            parts.append(fmt_table(top[['pos', 'stat', 'feature', 'importance_pct_of_mse', 'positive_folds', 'folds']], '{:.3f}'))
    sc = read('props_scores.csv')
    if sc is not None:
        parts.append('\n## 5. Prop distributions (forward-chained; lower CRPS is better; PIT max deviation 0 = perfectly '
                     'calibrated shape; cover_50/80 should be 0.50/0.80)\n')
        agg = sc.groupby(['stat', 'method'])[['n', 'crps', 'pit_max_dev', 'cover_50', 'cover_80']].agg(
            {'n': 'sum', 'crps': 'mean', 'pit_max_dev': 'mean', 'cover_50': 'mean', 'cover_80': 'mean'}).reset_index()
        parts.append(fmt_table(agg, '{:.4f}'))
        cal = read('props_calibration.csv')
        if cal is not None:
            parts.append('\nP(over) at book-like lines (the median rounded to a half point, and offsets), Brier and calibration '
                         'error (ECE), all test seasons:\n\n')
            a2 = cal.groupby(['stat', 'method'])[['brier', 'ece']].mean().reset_index()
            parts.append(fmt_table(a2, '{:.4f}'))
    post = read('props_posted_2026.csv')
    if post is not None and len(post):
        brier, bets = posted_summary(post)
        brier.to_csv(os.path.join(OUT, 'props_posted_brier.csv'), index=False)
        bets.to_csv(os.path.join(OUT, 'props_posted_bets.csv'), index=False)
        parts.append(f'\n## 6. 2026 posted lines (weeks 3-7, the last priced line before kickoff; {len(post):,} lines): the '
                     'model\'s P(over) against the books\' de-vigged price\n')
        parts.append(fmt_table(brier[brier['stat'] == 'ALL'], '{:.4f}'))
        parts.append('\nBy stat:\n\n' + fmt_table(brier[brier['stat'] != 'ALL'], '{:.4f}'))
        parts.append('\nBetting the model\'s side when it disagrees with the book by at least the edge (ROI per unit):\n\n')
        parts.append(fmt_table(bets, '{:.3f}'))
    td = read('props_anytime_td.csv')
    if td is not None:
        parts.append('\n## 7. Anytime TD (RB/WR/TE)\n')
        parts.append(fmt_table(td, '{:.4f}'))
    with open(os.path.join(OUT, 'REPORT.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(parts))
    log('report: ' + os.path.join(OUT, 'REPORT.md'))


# ------------------------------------------------------------------------------------------------------------- pass 2
# Pass 2 (same day): pass 1 found the stat projections overshoot whenever they leave the player's own level (segments: the top
# dev_s2d / dev_ps bins run high in every season; discover: the projection itself is the residual's top feature), that the
# points layer absorbs this but the stats - what props are priced on - do not, and that the Poisson shape is too narrow for
# volume stats. Pass 2 measures the overshoot and where it comes from, checks the untested weeks (1, 2, 18), and tries prop
# engines on recalibrated means.
WINDOWS = (('wk1', 1, 1), ('wk2', 2, 2), ('wk3-8', 3, 8), ('wk9-17', 9, 17), ('wk18', 18, 18))


def own_baseline(d, stat):
    """The player's own level going in: his season-to-date average after 3+ games, else last season's after 4+ games, else NaN."""
    s = 'fantasy_points_ppr' if stat == 'points' else stat
    g, pg = d['h_games'].fillna(0).to_numpy(), d['h_ps_games'].fillna(0).to_numpy()
    b = np.full(len(d), np.nan)
    if f'h_ps_{s}' in d.columns:
        b = np.where(pg >= 4, d[f'h_ps_{s}'].to_numpy(float), b)
    if f'h_s2d_{s}' in d.columns:
        b = np.where(g >= 3, d[f'h_s2d_{s}'].to_numpy(float), b)
    return b


def stage_windows():
    p = load_panel()
    log('windows: weeks 1, 2, 3-8, 9-17 and 18 (rows projected 3+ points)')
    rows = []
    for pos, stats in TARGETS.items():
        for stat in stats:
            d = target_rows(p, pos, stat, weeks=(1, 18))
            d = d[num(d['p_points']) >= 3.0]
            y, m = d[f'y_{stat}'].to_numpy(float), d[f'p_{stat}'].to_numpy(float)
            b = own_baseline(d, stat)
            wk = d['week'].to_numpy()
            cl = (d['year'] * 100 + d['week']).to_numpy()
            for name, lo, hi in WINDOWS:
                sel = (wk >= lo) & (wk <= hi)
                if sel.sum() < 100 * MS:
                    continue
                e = m[sel] - y[sel]
                bias, se = clustered_mean_se(e, cl[sel])
                rmse = float(np.sqrt(np.mean(e ** 2)))
                row = {'pos': pos, 'stat': stat, 'window': name, 'n': int(sel.sum()), 'mean_actual': float(y[sel].mean()),
                       'rmse': rmse, 'rel_rmse': rmse / max(float(y[sel].mean()), 1e-6), 'bias': bias, 'bias_se': se,
                       'corr': float(np.corrcoef(m[sel], y[sel])[0, 1]),
                       'slope': float(np.polyfit(m[sel], y[sel], 1)[0]) if np.std(m[sel]) > 0 else np.nan}
                ok = sel & ~np.isnan(b)
                if ok.sum() >= 50 * MS:
                    row['n_own'] = int(ok.sum())
                    row['skill_vs_own'] = float(1 - np.mean((m[ok] - y[ok]) ** 2) / np.mean((b[ok] - y[ok]) ** 2))
                if stat == 'points':
                    big = sel & (m >= 10)
                    row['collapse_rate_10plus'] = float(np.mean(y[big] < 0.25 * m[big])) if big.sum() >= 30 else np.nan
                rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'windows.csv'), index=False)
    log(f'windows: {len(rows)} rows')


OVERSHOOT_STATS = {'QB': ('passing_attempts', 'passing_yards', 'passing_tds', 'passing_interceptions', 'rushing_attempts',
                          'rushing_yards', 'points'),
                   'RB': ('rushing_attempts', 'rushing_yards', 'targets', 'receptions', 'receiving_yards', 'points'),
                   'WR': ('targets', 'receptions', 'receiving_yards', 'points'),
                   'TE': ('targets', 'receptions', 'receiving_yards', 'points')}


def _listed(d):
    st = d['Injury Status'].astype(str) if 'Injury Status' in d.columns else pd.Series('', index=d.index)
    return st.str.startswith(('Questionable', 'Doubtful')).to_numpy()


OVERSHOOT_GROUPS = {
    'all': lambda d, x: np.ones(len(d), dtype=bool),
    'departure_up': lambda d, x: x > 0,
    'departure_down': lambda d, x: x < 0,
    'teammate_out': lambda d, x: d['tm_out_pts_total'].fillna(0).to_numpy() > 0,
    'no_teammate_out': lambda d, x: d['tm_out_pts_total'].fillna(0).to_numpy() <= 0,
    'weeks_3_6': lambda d, x: d['week'].to_numpy() <= 6,
    'weeks_7_17': lambda d, x: d['week'].to_numpy() >= 7,
    'baseline_last_season': lambda d, x: d['h_games'].fillna(0).to_numpy() < 3,
    'role_change': lambda d, x: num(d['Role Change Confidence']).fillna(0).to_numpy() > 0,
    'listed_q_or_d': lambda d, x: _listed(d),
    'favorite_3plus': lambda d, x: d['spread'].to_numpy(float) >= 3,
    'underdog_3plus': lambda d, x: d['spread'].to_numpy(float) <= -3,
    'soft_defense': lambda d, x: d['o_pts_rel'].to_numpy(float) >= 1.1,
    'hard_defense': lambda d, x: d['o_pts_rel'].to_numpy(float) <= 0.9,
}


def follow_through(x, z, cl):
    """k in z = k x (no intercept) and its week-clustered standard error: the share of the model's departure from the
    player's own level that came true (1 = all of it, 0 = none)."""
    sxx = float((x ** 2).sum())
    if sxx <= 0:
        return np.nan, np.nan
    k = float((x * z).sum() / sxx)
    u = z - k * x
    g = pd.Series(x * u).groupby(np.asarray(cl)).sum().to_numpy()
    n_g = len(g)
    se = float(np.sqrt((g ** 2).sum() * n_g / max(n_g - 1, 1)) / sxx)
    return k, se


def stage_overshoot():
    p = load_panel()
    log("overshoot: how much of the model's departure from a player's own level comes true, and where it does not")
    rows, gains = [], []
    for pos, stats in OVERSHOOT_STATS.items():
        for stat in stats:
            d = target_rows(p, pos, stat)
            d = d[num(d['p_points']) >= 3.0]
            b = own_baseline(d, stat)
            ok = ~np.isnan(b)
            d, b = d[ok].reset_index(drop=True), b[ok]
            y, m = d[f'y_{stat}'].to_numpy(float), d[f'p_{stat}'].to_numpy(float)
            x, z = m - b, y - b
            cl = (d['year'] * 100 + d['week']).to_numpy()
            yr = d['year'].to_numpy()
            big_cut = np.quantile(np.abs(x), 0.8) if len(x) else np.nan
            for gname, fn in OVERSHOOT_GROUPS.items():
                try:
                    sel = np.asarray(fn(d, x), dtype=bool)
                except KeyError:
                    continue
                if sel.sum() < 150 * MS:
                    continue
                k, se = follow_through(x[sel], z[sel], cl[sel])
                by = {int(t): round(follow_through(x[sel & (yr == t)], z[sel & (yr == t)], cl[sel & (yr == t)])[0], 3)
                      for t in np.unique(yr) if (sel & (yr == t)).sum() >= 60 * MS}
                big = sel & (np.abs(x) >= big_cut)
                up, down = big & (x > 0), big & (x < 0)
                rows.append({'pos': pos, 'stat': stat, 'group': gname, 'n': int(sel.sum()), 'k': k, 'k_se': se,
                             'z_k_vs_1': (k - 1) / se if se else np.nan, 'mean_abs_departure': float(np.abs(x[sel]).mean()),
                             'mean_departure': float(x[sel].mean()), 'n_big': int(big.sum()),
                             'bias_big_up': float((m - y)[up].mean()) if up.sum() >= 20 else np.nan,
                             'bias_big_down': float((m - y)[down].mean()) if down.sum() >= 20 else np.nan,
                             'k_by_season': json.dumps(by)})
            # held out by season: p' = own level + k (p - own level), k fit on the other seasons (2026 on 2021-2025)
            pred = np.full(len(d), np.nan)
            for t in np.unique(yr):
                tr = ((yr != t) & (yr <= 2025)) if t <= 2025 else (yr <= 2025)
                te = yr == t
                if tr.sum() < 300 * MS:
                    continue
                k_tr, _ = follow_through(x[tr], z[tr], cl[tr])
                pred[te] = b[te] + k_tr * x[te]
            for scope, sel in (('2021-2025', ~np.isnan(pred) & (yr <= 2025)), ('2026', ~np.isnan(pred) & (yr == 2026))):
                if sel.sum() < 50 * MS:
                    continue
                d01, se01 = clustered_mean_se((pred[sel] - y[sel]) ** 2 - (m[sel] - y[sel]) ** 2, cl[sel])
                mse_m, mse_s = np.mean((m[sel] - y[sel]) ** 2), np.mean((pred[sel] - y[sel]) ** 2)
                gains.append({'pos': pos, 'stat': stat, 'scope': scope, 'n': int(sel.sum()),
                              'rmse_model': float(np.sqrt(mse_m)), 'rmse_shrunk': float(np.sqrt(mse_s)),
                              'mse_gain_pct': float(100 * (1 - mse_s / mse_m)), 'z': float(d01 / se01) if se01 else np.nan,
                              'k_2021_2025': follow_through(x[yr <= 2025], z[yr <= 2025], cl[yr <= 2025])[0]})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'overshoot.csv'), index=False)
    pd.DataFrame(gains).to_csv(os.path.join(OUT, 'overshoot_gain.csv'), index=False)
    log(f'overshoot: {len(rows)} group rows')


def recal_design(d, stat, m):
    """The linear recalibration's inputs: the projection; the player's season to date, last 3 and last season (the projection
    where missing); and a flag for each missing one."""
    cols, flags = [m], []
    for c in (f'h_s2d_{stat}', f'h_l3_{stat}', f'h_ps_{stat}'):
        v = d[c].to_numpy(float) if c in d.columns else np.full(len(d), np.nan)
        miss = np.isnan(v)
        cols.append(np.where(miss, m, v))
        flags.append(miss.astype(float))
    return np.column_stack(cols + flags)


def lin_means(d, stat, m, y, tr, pos_arr, floor):
    """The recalibrated mean for every row, a ridge fit per position on the rows in tr (the raw projection where a position
    has too few)."""
    from sklearn.linear_model import Ridge
    Z = recal_design(d, stat, m)
    out = m.copy()
    models = {}
    for ps in np.unique(pos_arr):
        trp = tr & (pos_arr == ps)
        if trp.sum() < 200 * MS:
            continue
        mdl = Ridge(alpha=1.0).fit(Z[trp], y[trp])
        sel = pos_arr == ps
        out[sel] = np.maximum(mdl.predict(Z[sel]), floor)
        models[ps] = mdl
    return out, models


def gb_adjust(X, r, yr, tr, te, cats):
    """The boosted residual model's prediction: out of fold on the training rows (each training season predicted by the
    others) and fit on every training row for the test rows."""
    adj = np.full(len(r), np.nan)

    def fit_predict(a, b):
        cols = usable_cols(X[a])
        cm = [c in cats for c in cols]
        return hgb(categorical_features=cm if any(cm) else None).fit(X.loc[a, cols], r[a]).predict(X.loc[b, cols])
    for s_ in np.unique(yr[tr]):
        a, b = tr & (yr != s_), tr & (yr == s_)
        adj[b] = fit_predict(a, b) if a.sum() >= 300 * MS else 0.0
    adj[te] = fit_predict(tr, te)
    return adj


def dist_cdfs(mean_v, y, tr, te, pos_arr, spec, grid):
    """M1 (empirical) and M2 (fitted) on a given mean, per position, trained on tr and evaluated on te."""
    pos_te = pos_arr[te]
    F1 = np.zeros((te.sum(), grid + 1))
    F2 = np.zeros((te.sum(), grid + 1))
    for ps in np.unique(pos_te):
        trp = tr & (pos_arr == ps)
        if trp.sum() < 300 * MS:
            trp = tr
        tep = pos_te == ps
        F1[tep] = cdf_m1(fit_m1(mean_v[trp], y[trp]), mean_v[te][tep], grid)
        F2[tep] = cdf_m2(fit_m2(mean_v[trp], y[trp], spec['kind']), mean_v[te][tep], spec['kind'], grid)
    return F1, F2


def save_prop_engine(d, stat, spec, m, y, pos_arr, floor, rows, out_dir, recal=True):
    """The engine for the weekly tooling (data/prop_probability.py), fit on `rows`: the linear recalibration per position
    (recal=False: none, the projection as is), then the empirical ratio tables (M1) and the fitted shape (M2) on that mean."""
    m_lin, models = lin_means(d, stat, m, y, rows, pos_arr, floor) if recal else (m, {})
    eng = {'stat': stat, 'spec': {k: v for k, v in spec.items() if k != 'offsets'}, 'recal': {}, 'M1': {}, 'M2': {},
           'seasons': sorted(int(v) for v in np.unique(d['year'].to_numpy()[rows]))}
    for ps, mdl in models.items():
        eng['recal'][ps] = {'inputs': ['projection', f'h_s2d_{stat}', f'h_l3_{stat}', f'h_ps_{stat}',
                                       'missing_s2d', 'missing_l3', 'missing_ps'],
                            'coef': [float(c) for c in mdl.coef_], 'intercept': float(mdl.intercept_), 'floor': floor}
    for ps in np.unique(pos_arr[rows]):
        sel = rows & (pos_arr == ps)
        f1 = fit_m1(m_lin[sel], y[sel])
        eng['M1'][ps] = {'edges': [float(e) for e in f1['edges'][1:-1]],
                         'ratio_quantiles': {int(k): [float(v) for v in np.quantile(r_, np.linspace(0, 1, 201))]
                                             for k, r_ in f1['ratios'].items()}}
        eng['M2'][ps] = fit_m2(m_lin[sel], y[sel], spec['kind'])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'{stat}.json'), 'w', encoding='utf-8') as fh:
        json.dump(eng, fh)


# QB volume and interceptions: the recalibration lost forward-chained (CRPS +0.2 to +0.8%; QB passing departures from the
# player's own level come true in full, k ~1), and the empirical tables already absorb the level's slope bin by bin
ENGINE_RAW_MEAN = ('passing_attempts', 'passing_completions', 'passing_interceptions')


def stage_engines():
    """The prop engines alone: on every season (prop_models_v2) and on 2021-2025 (prop_models_v2_thru2025, for scoring 2026
    out of sample)."""
    p = load_panel()
    for stat, spec in PROP_SPECS.items():
        d = prop_rows(p, stat)
        if len(d) < 1000 * MS:
            continue
        y, m = d[f'y_{stat}'].to_numpy(float), d[f'p_{stat}'].to_numpy(float)
        pos_arr, yr = d['pos'].to_numpy(), d['year'].to_numpy()
        floor = 0.05 if spec['kind'] == 'count' else 1.0
        recal = stat not in ENGINE_RAW_MEAN
        save_prop_engine(d, stat, spec, m, y, pos_arr, floor, np.ones(len(d), dtype=bool), os.path.join(OUT, 'prop_models_v2'),
                         recal)
        save_prop_engine(d, stat, spec, m, y, pos_arr, floor, yr <= 2025, os.path.join(OUT, 'prop_models_v2_thru2025'), recal)
    log('engines: prop_models_v2 (2021-2026) and prop_models_v2_thru2025')


def stage_props2():
    from data.weekly_median import stat_median
    p = load_panel()
    rng = np.random.default_rng(SEED + 2)
    budget = float(os.environ.get('DEEP_AUDIT_M3FULL_MIN', '75')) * 60
    log("props2: prop engines on recalibrated means (linear on the player's own history; boosted on everything), "
        "forward-chained 2023-2026")
    try:
        arc, kick = posted_lines_2026()
    except Exception:
        log('  no posted-line archive\n' + traceback.format_exc())
        arc, kick = None, None
    scores, calib, posted_rows, mean_rows = [], [], [], []
    os.makedirs(os.path.join(OUT, 'prop_models_v2'), exist_ok=True)
    prepared = {}
    for stat, spec in PROP_SPECS.items():
        t_stat = time.time()
        d = prop_rows(p, stat)
        if len(d) < 1000 * MS:
            continue
        y = d[f'y_{stat}'].to_numpy(float)
        m = d[f'p_{stat}'].to_numpy(float)
        yr = d['year'].to_numpy()
        pos_arr = d['pos'].to_numpy()
        grid = spec['grid']
        floor = 0.05 if spec['kind'] == 'count' else 1.0
        X, cats = feature_matrix(d)
        med = np.array([stat_median(ps, stat, mm) for ps, mm in zip(pos_arr, m)], dtype=float)
        base_line = np.floor(np.where(np.isnan(med), m, med)) + 0.5
        prepared[stat] = (d, X, y, yr, base_line)
        for t in (2023, 2024, 2025, 2026):
            tr, te = yr < t, yr == t
            if te.sum() < 50 * MS or tr.sum() < 500 * MS:
                continue
            m_lin, _ = lin_means(d, stat, m, y, tr, pos_arr, floor)
            m_gb = np.maximum(m + 0.5 * gb_adjust(X, y - m, yr, tr, te, cats), floor)
            means = (('raw', m), ('lin', m_lin), ('gb', m_gb))
            cdfs = {}
            for vname, mv in means:
                e = mv[te] - y[te]
                mean_rows.append({'stat': stat, 'test_season': t, 'mean': vname, 'n': int(te.sum()),
                                  'rmse': float(np.sqrt(np.mean(e ** 2))), 'bias': float(e.mean())})
                cdfs[f'{vname}_M1'], cdfs[f'{vname}_M2'] = dist_cdfs(mv, y, tr, te, pos_arr, spec, grid)
            yt = y[te]
            for meth, F in cdfs.items():
                crps, pit, info = score_cdf(F, yt, rng)
                scores.append({'stat': stat, 'test_season': t, 'method': meth, 'n': int(te.sum()), 'crps': float(crps.mean()),
                               **info})
                for off in spec['offsets']:
                    line = np.maximum(base_line[te] + off, 0.5)
                    po = p_over_at(F, line)
                    outc = (yt > line).astype(float)
                    ece, _g = reliability(po, outc)
                    calib.append({'stat': stat, 'test_season': t, 'method': meth, 'offset': off, 'n': len(outc),
                                  'brier': float(np.mean((po - outc) ** 2)), 'ece': ece})
            if t == 2026 and arc is not None:
                dt = d[te].reset_index(drop=True)
                for _, L in posted_join(dt, spec, arc, kick).iterrows():
                    i = int(L['i'])
                    rec = {'stat': stat, 'week': int(L['week']), 'player': dt.at[i, 'player'], 'pos': dt.at[i, 'pos'],
                           'book': L['book'], 'line': float(L['line']), 'p_book': float(L['p_book']),
                           'open_line': float(L['open_line']), 'open_p_over': float(L['open_p_over']),
                           'model_mean': float(m[te][i]), 'lin_mean': float(m_lin[te][i]), 'gb_mean': float(m_gb[te][i]),
                           'actual': float(yt[i])}
                    for meth, F in cdfs.items():
                        rec[f'p_{meth}'] = float(p_over_at(F[i:i + 1], np.array([rec['line']]))[0])
                        rec[f'po_{meth}'] = float(p_over_at(F[i:i + 1], np.array([rec['open_line']]))[0])
                    posted_rows.append(rec)
        save_prop_engine(d, stat, spec, m, y, pos_arr, floor, np.ones(len(d), dtype=bool),
                         os.path.join(OUT, 'prop_models_v2'))
        pd.DataFrame(scores).to_csv(os.path.join(OUT, 'props2_scores.csv'), index=False)
        pd.DataFrame(calib).to_csv(os.path.join(OUT, 'props2_calibration.csv'), index=False)
        pd.DataFrame(posted_rows).to_csv(os.path.join(OUT, 'props2_posted_2026.csv'), index=False)
        pd.DataFrame(mean_rows).to_csv(os.path.join(OUT, 'props2_means.csv'), index=False)
        log(f'  {stat}: {len(d):,} rows  [{time.time() - t_stat:.0f}s]')
    # quantile boosting on every feature (the ceiling for a learned shape), as far as the time budget allows
    start = time.time()
    m3_scores, m3_posted = [], []
    for stat, (d, X, y, yr, base_line) in prepared.items():
        if time.time() - start > budget:
            log(f'  M3 full {stat}: skipped, time budget used')
            continue
        t_stat = time.time()
        spec = PROP_SPECS[stat]
        for t in (2023, 2024, 2025, 2026):
            tr, te = yr < t, yr == t
            if te.sum() < 50 * MS or tr.sum() < 500 * MS:
                continue
            F = cdf_m3(fit_m3(X[tr], y[tr]), X[te], spec['grid'])
            crps, pit, info = score_cdf(F, y[te], rng)
            briers = []
            for off in spec['offsets']:
                line = np.maximum(base_line[te] + off, 0.5)
                briers.append(np.mean((p_over_at(F, line) - (y[te] > line)) ** 2))
            m3_scores.append({'stat': stat, 'test_season': t, 'method': 'M3_full', 'n': int(te.sum()), 'crps': float(crps.mean()),
                              'brier_lines': float(np.mean(briers)), **info})
            if t == 2026 and arc is not None:
                dt = d[te].reset_index(drop=True)
                for _, L in posted_join(dt, spec, arc, kick).iterrows():
                    i = int(L['i'])
                    m3_posted.append({'stat': stat, 'week': int(L['week']), 'player': dt.at[i, 'player'],
                                      'p_M3_full': float(p_over_at(F[i:i + 1], np.array([float(L['line'])]))[0]),
                                      'po_M3_full': float(p_over_at(F[i:i + 1], np.array([float(L['open_line'])]))[0])})
        pd.DataFrame(m3_scores).to_csv(os.path.join(OUT, 'props2_m3full_scores.csv'), index=False)
        pd.DataFrame(m3_posted).to_csv(os.path.join(OUT, 'props2_m3full_posted.csv'), index=False)
        log(f'  M3 full {stat}  [{time.time() - t_stat:.0f}s]')


def shrink_fit(pr, out, wk):
    """The shrink toward 1/2 (p' = 1/2 + lam (p - 1/2)) that minimises Brier, and the Brier when each week's lam is fit on the
    other weeks."""
    grid = np.linspace(0, 1.2, 61)

    def best(a):
        return grid[np.argmin([np.mean((0.5 + g * (pr[a] - 0.5) - out[a]) ** 2) for g in grid])]
    lowo = np.empty(len(pr))
    for w in np.unique(wk):
        lowo[wk == w] = 0.5 + best(wk != w) * (pr[wk == w] - 0.5)
    return float(best(np.ones(len(pr), dtype=bool))), float(np.mean((lowo - out) ** 2))


def bets_table(df, pcols, line_col, book_col, label):
    """Hit rate and ROI betting the model's side where it disagrees with the book's de-vigged price by the edge (pushes out)."""
    rows = []
    for stat in ['ALL'] + sorted(df['stat'].unique()):
        s = df if stat == 'ALL' else df[df['stat'] == stat]
        s = s[s['actual'] != s[line_col]]
        over = (s['actual'] > s[line_col]).astype(float)
        for meth in pcols:
            if meth not in s.columns:
                continue
            edge = s[meth] - s[book_col]
            for thr in (0.0, 0.03, 0.05, 0.08, 0.12):
                sel = (edge.abs() >= thr) & (edge.abs() > 0)
                if sel.sum() < 15:
                    continue
                side_over = (edge[sel] > 0).to_numpy()
                win = np.where(side_over, over[sel] == 1, over[sel] == 0)
                k, n = int(win.sum()), int(sel.sum())
                lo, hi = wilson(k, n)
                fair_p = np.where(side_over, s.loc[sel, book_col], 1 - s.loc[sel, book_col])
                rows.append({'line': label, 'stat': stat, 'method': meth.split('_', 1)[1], 'edge_min': thr, 'bets': n,
                             'win_rate': k / n, 'ci_lo': lo, 'ci_hi': hi,
                             'roi_at_-110': float(np.mean(np.where(win, 100 / 110, -1.0))),
                             'roi_at_fair_price': float(np.mean(np.where(win, 1 / fair_p - 1, -1.0)))})
    return pd.DataFrame(rows)


def stage_report2():
    def read(name):
        path = os.path.join(OUT, name)
        return pd.read_csv(path) if os.path.exists(path) else None

    parts = ['# Deep audit pass 2, 2026-10-28\n', "The same panel as REPORT.md. Pass 1 found the stat projections overshoot when "
             "they leave a player's own level; pass 2 measures that, checks weeks 1, 2 and 18, and tries prop engines on "
             "recalibrated means.\n"]
    w = read('windows.csv')
    if w is not None:
        parts.append("\n## 1. Week windows (rows projected 3+ points; bias = projection - actual; slope of actual on projection, "
                     "1 = right spread; skill_vs_own = against the player's own level)\n")
        cols = [c for c in ['pos', 'stat', 'window', 'n', 'mean_actual', 'rmse', 'rel_rmse', 'bias', 'bias_se', 'slope', 'corr',
                            'skill_vs_own', 'collapse_rate_10plus'] if c in w.columns]
        parts.append(fmt_table(w[cols], '{:.3f}'))
    o = read('overshoot.csv')
    if o is not None:
        parts.append("\n## 2. Follow-through: k = the share of the model's departure from the player's own level that came true "
                     "(1 = all of it); weeks 3-17, rows projected 3+ points\n")
        parts.append(fmt_table(o[o['group'] == 'all'][['pos', 'stat', 'n', 'k', 'k_se', 'z_k_vs_1', 'mean_abs_departure',
                                                       'bias_big_up', 'bias_big_down', 'k_by_season']], '{:.3f}'))
        piv = o.pivot_table(index=['pos', 'stat'], columns='group', values='k').reset_index()
        parts.append('\nk by group:\n\n' + fmt_table(piv, '{:.2f}'))
        g = read('overshoot_gain.csv')
        if g is not None:
            parts.append("\nHeld out by season, p' = own level + k (p - own level) with k fit on the other seasons:\n\n"
                         + fmt_table(g, '{:.3f}'))
    mr = read('props2_means.csv')
    if mr is not None:
        parts.append("\n## 3. Prop means (RMSE): the raw projection vs a linear recalibration on the player's own history vs "
                     "a boosted residual model at half strength, forward-chained\n")
        parts.append(fmt_table(mr.pivot_table(index=['stat', 'test_season'], columns='mean', values='rmse').reset_index(), '{:.3f}'))
    sc = read('props2_scores.csv')
    if sc is not None:
        agg = sc.groupby(['stat', 'method'])[['crps', 'pit_max_dev', 'cover_50', 'cover_80']].mean().reset_index()
        cal = read('props2_calibration.csv')
        if cal is not None:
            agg = agg.merge(cal.groupby(['stat', 'method'])['brier'].mean().rename('brier_lines').reset_index(),
                            on=['stat', 'method'])
        m3 = read('props2_m3full_scores.csv')
        if m3 is not None and len(m3):
            agg = pd.concat([agg, m3.groupby(['stat', 'method'])[['crps', 'pit_max_dev', 'cover_50', 'cover_80', 'brier_lines']]
                             .mean().reset_index()])
        parts.append('\n## 4. Prop engines, all test seasons (lower CRPS / Brier is better)\n')
        parts.append(fmt_table(agg.sort_values(['stat', 'crps']), '{:.4f}'))
    post = read('props2_posted_2026.csv')
    if post is not None and len(post):
        m3p = read('props2_m3full_posted.csv')
        if m3p is not None and len(m3p):
            post = post.merge(m3p, on=['stat', 'week', 'player'], how='left')
        pcols = [c for c in post.columns if c.startswith('p_') and c != 'p_book']
        rows = []
        for stat in ['ALL'] + sorted(post['stat'].unique()):
            s = post if stat == 'ALL' else post[post['stat'] == stat]
            s = s[s['actual'] != s['line']]
            out = (s['actual'] > s['line']).to_numpy(float)
            wk = s['week'].to_numpy()
            for meth in ['p_book'] + pcols:
                pr = s[meth].to_numpy(float)
                ok = ~np.isnan(pr)
                lam, lowo = shrink_fit(pr[ok], out[ok], wk[ok]) if meth != 'p_book' and ok.sum() >= 50 else (np.nan, np.nan)
                rows.append({'stat': stat, 'method': meth[2:], 'n': int(ok.sum()),
                             'brier': float(np.mean((pr[ok] - out[ok]) ** 2)), 'best_shrink': lam, 'brier_shrunk_lowo': lowo})
        br = pd.DataFrame(rows)
        br.to_csv(os.path.join(OUT, 'props2_posted_brier.csv'), index=False)
        parts.append(f"\n## 5. 2026 posted lines, weeks 3-7 ({len(post):,} closing lines): Brier against the outcome; best_shrink "
                     "= the factor toward 1/2 that minimises Brier (1 = trust as is); brier_shrunk_lowo = each week's factor "
                     "fit on the other weeks\n")
        parts.append(fmt_table(br, '{:.4f}'))
        close = bets_table(post, pcols, 'line', 'p_book', 'close')
        opn = bets_table(post.dropna(subset=['open_line', 'open_p_over']), ['po_' + c[2:] for c in pcols],
                         'open_line', 'open_p_over', 'open')
        bets = pd.concat([close, opn])
        bets.to_csv(os.path.join(OUT, 'props2_posted_bets.csv'), index=False)
        parts.append("\nBetting the model's side by edge at the closing and the opening line (all stats; then per stat at a 5%+ "
                     "edge, the two best methods):\n\n")
        parts.append(fmt_table(bets[bets['stat'] == 'ALL'], '{:.3f}'))
        top = bets[(bets['stat'] != 'ALL') & (bets['edge_min'] == 0.05)].sort_values('roi_at_-110', ascending=False)
        parts.append('\n' + fmt_table(top.groupby(['line', 'stat']).head(2), '{:.3f}'))
    with open(os.path.join(OUT, 'REPORT_PASS2.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(parts))
    log('report2: ' + os.path.join(OUT, 'REPORT_PASS2.md'))


# ------------------------------------------------------------------------------------------------------------------ all
STAGES = ('capture', 'assemble', 'accuracy', 'segments', 'props', 'discover', 'report')   # props before the open-ended search
STAGES2 = ('windows', 'overshoot', 'props2', 'report2')   # pass 2, on pass 1's panel


def run_all(args, stages=STAGES):
    t0 = time.time()
    log(f'=== deep audit start (stages {", ".join(stages)})')
    for stage in stages:
        ts = time.time()
        cmd = [sys.executable, os.path.abspath(__file__), '--stage', stage, '--years', args.years, '--weeks', args.weeks]
        rc = subprocess.call(cmd, cwd=ROOT)
        log(f'=== stage {stage} exit {rc} after {(time.time() - ts) / 60:.1f} min')
        if stage == 'capture' and not glob.glob(os.path.join(BOARDS, '*.parquet')):
            log('no boards captured: stopping')
            break
        if stage == 'assemble' and not os.path.exists(PANEL):
            log('no panel: stopping')
            break
    log(f'=== deep audit done in {(time.time() - t0) / 60:.1f} min')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=STAGES + STAGES2 + ('all', 'pass2', 'engines'))
    ap.add_argument('--years', default='2021,2022,2023,2024,2025,2026')
    ap.add_argument('--weeks', default='1-18')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    years = [int(x) for x in a.years.split(',') if x]
    w0, w1 = (int(x) for x in a.weeks.split('-'))
    weeks = list(range(w0, w1 + 1))
    t0 = time.time()
    try:
        if a.stage == 'all':
            run_all(a)
        elif a.stage == 'pass2':
            run_all(a, STAGES2)
        elif a.stage == 'engines':
            stage_engines()
        elif a.stage in STAGES2:
            {'windows': stage_windows, 'overshoot': stage_overshoot, 'props2': stage_props2, 'report2': stage_report2}[a.stage]()
        elif a.stage == 'capture':
            stage_capture(years, weeks)
        elif a.stage == 'assemble':
            stage_assemble()
        elif a.stage == 'accuracy':
            stage_accuracy()
        elif a.stage == 'segments':
            stage_segments()
        elif a.stage == 'discover':
            stage_discover()
        elif a.stage == 'props':
            stage_props()
        elif a.stage == 'report':
            stage_report()
    except Exception:
        log(f'stage {a.stage} FAILED after {(time.time() - t0) / 60:.1f} min\n' + traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
