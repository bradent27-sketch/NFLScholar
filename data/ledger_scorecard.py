"""
The live track record: what the saved boards (data.prediction_ledger) said before each game, scored against what
happened - model vs market vs FantasyPros. Shared by the Track Record page (ui/tabs/track_record.py); the scoring rules
match scripts/score_ledger.py's metrics (harness_v2) with two differences that make it a fair live record:

- **Which board counts.** For each player, the last board saved BEFORE HIS OWN GAME kicked off (his team's kickoff on
  the schedule), not simply the week's last build: the week-5 ledger's last file was built on Monday night, after
  Sunday's games. A player-week with no pre-kickoff board (weeks 1-2 of 2026 were rebuilt on 09-24, after they were
  played) is kept but flagged ``pre_kickoff=False``; the page reports those weeks separately and keeps them out of the
  live totals.
- **Paired pools.** Every model-vs-benchmark comparison is on the same players: the rows where both have a number.
  (score_ledger.py scores each source on its own pool, so its model and market n differ.)

Nothing here changes a projection.
"""
import glob
import os

import numpy as np
import pandas as pd

from data.prediction_ledger import LEDGER_DIR, STAT_COLS
from data.transforms import score_projected_stats
from data.utils import clean_name_exact

POSITIONS = ('QB', 'RB', 'WR', 'TE')
# The harness's startable pool sizes (scripts/harness_v2.STARTABLE_N), kept here so the app does not import scripts/.
STARTABLE_N = {'QB': 24, 'RB': 40, 'WR': 55, 'TE': 20}
SCORING_MODES = ('Full PPR', 'Half-PPR', 'Standard')
SOURCES = {'Model': 'model_pts', 'Market': 'market_pts', 'FantasyPros': 'fp_pts'}
# market / FantasyPros column per projected stat in the ledger
BENCHMARK_STAT_COLS = {'Market': 'Mkt {}', 'FantasyPros': 'FP {}'}


def ledger_years(ledger_dir=LEDGER_DIR):
    years = set()
    for path in glob.glob(os.path.join(ledger_dir, '*_wk*_*.parquet')):
        try:
            years.add(int(os.path.basename(path).split('_', 1)[0]))
        except ValueError:
            continue
    return sorted(years, reverse=True)


def infer_scoring(frame):
    """The scoring mode a ledger file was built in (it is not stored): the mode whose score of each row's stat line
    reproduces 'Raw Model Proj Pts'. None when nothing fits within 0.05 pts on average."""
    cols = [c for c in STAT_COLS if c in frame.columns]
    if not cols or 'Raw Model Proj Pts' not in frame.columns or frame.empty:
        return None
    lines = frame[cols].fillna(0.0).to_dict('records')
    raw = pd.to_numeric(frame['Raw Model Proj Pts'], errors='coerce').to_numpy(dtype=float)
    best, best_err = None, np.inf
    for mode in SCORING_MODES:
        scored = np.clip([score_projected_stats(r, mode) for r in lines], 0.0, None)
        err = float(np.nanmean(np.abs(scored - raw)))
        if err < best_err:
            best, best_err = mode, err
    return best if best_err < 0.05 else None


def kickoffs_utc(schedule):
    """{(week, team): kickoff as a UTC Timestamp} for regular-season games (gameday + gametime are US Eastern)."""
    s = schedule
    if 'game_type' in s.columns:
        s = s[s['game_type'] == 'REG']
    s = s.dropna(subset=['gameday', 'gametime'])
    local = pd.to_datetime(s['gameday'].astype(str) + ' ' + s['gametime'].astype(str), errors='coerce')
    utc = local.dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')
    out = {}
    for week, home, away, ko in zip(s['week'], s['home_team'], s['away_team'], utc):
        if pd.isna(ko):
            continue
        out[(int(week), str(home))] = ko
        out[(int(week), str(away))] = ko
    return out


def select_pre_kickoff_rows(ledger, kickoffs):
    """One row per (week, player key): from the latest build at or before his team's kickoff; when there is none, from
    the latest build at all, with pre_kickoff False. ``ledger`` is every file's rows stacked, with 'week', 'Team',
    'build_ts' (UTC) and '_key'."""
    if ledger.empty:
        return ledger.assign(pre_kickoff=pd.Series(dtype=bool))
    lg = ledger.copy()
    ko = [kickoffs.get((int(w), str(t))) for w, t in zip(lg['week'], lg['Team'])]
    lg['kickoff'] = pd.to_datetime(pd.Series(ko, index=lg.index), utc=True)
    build = pd.to_datetime(lg['build_ts'], utc=True)
    lg['pre_kickoff'] = lg['kickoff'].notna() & (build <= lg['kickoff'])
    lg['_build'] = build
    lg = lg.sort_values(['pre_kickoff', '_build'])
    picked = lg.groupby(['week', '_key'], sort=False).tail(1)
    return picked.drop(columns=['_build']).reset_index(drop=True)


def _actual_points(stats_df, scoring):
    ppr = pd.to_numeric(stats_df.get('fantasy_points_ppr'), errors='coerce')
    std = pd.to_numeric(stats_df.get('fantasy_points'), errors='coerce')
    if scoring == 'Standard':
        return std
    if scoring == 'Half-PPR':
        return (ppr + std) / 2.0
    return ppr


def load_scored_ledger(year, stats_df, name_col, schedule, ledger_dir=LEDGER_DIR):
    """(frame, info) - one row per player-week with a box score: projections from the pre-kickoff board (see the module
    docstring) next to the actual points and stats. ``info`` = {'scoring', 'files', 'mixed_scoring'}."""
    paths = sorted(glob.glob(os.path.join(ledger_dir, f'{year}_wk*_*.parquet')))
    frames, modes = [], []
    for path in paths:
        try:
            f = pd.read_parquet(path)
        except Exception:
            continue
        if f.empty or 'week' not in f.columns:
            continue
        mode = infer_scoring(f)
        modes.append(mode)
        f['_scoring'] = mode
        frames.append(f)
    info = {'files': len(frames), 'scoring': None, 'mixed_scoring': False}
    if not frames:
        return pd.DataFrame(), info
    ledger = pd.concat(frames, ignore_index=True)
    known = [m for m in modes if m]
    scoring = max(set(known), key=known.count) if known else 'Full PPR'
    info['scoring'] = scoring
    info['mixed_scoring'] = len(set(known)) > 1
    ledger = ledger[ledger['_scoring'].eq(scoring)].copy()
    ledger['_key'] = clean_name_exact(ledger['Player'])
    rows = select_pre_kickoff_rows(ledger, kickoffs_utc(schedule))

    weeks = pd.to_numeric(stats_df['week'], errors='coerce')
    st = stats_df.assign(_pts=_actual_points(stats_df, scoring), week=weeks)
    if 'season_type' in st.columns:
        st = st[st['season_type'].isna() | st['season_type'].astype(str).str.upper().eq('REG')]
    stat_cols = [c for c in STAT_COLS if c in st.columns]
    actual = st.groupby([name_col, 'week'], observed=True)[['_pts'] + stat_cols].sum(min_count=1).reset_index()
    actual['_key'] = clean_name_exact(actual[name_col].astype(str))
    actual = actual.drop(columns=[name_col]).groupby(['_key', 'week'], as_index=False).sum(min_count=1)
    actual = actual.rename(columns={'_pts': 'actual_pts', **{c: f'actual_{c}' for c in stat_cols}})

    out = rows.merge(actual, on=['_key', 'week'], how='inner')
    out = out[out['actual_pts'].notna()].copy()
    out['model_pts'] = pd.to_numeric(out['Model Proj Pts'], errors='coerce')
    out['market_pts'] = pd.to_numeric(out.get('Mkt Market Pts'), errors='coerce') if 'Mkt Market Pts' in out else np.nan
    fp_col = {'Full PPR': 'FP Proj Pts PPR', 'Half-PPR': 'FP Proj Pts Half', 'Standard': 'FP Proj Pts'}.get(scoring)
    if fp_col in out.columns and out[fp_col].notna().any():
        out['fp_pts'] = pd.to_numeric(out[fp_col], errors='coerce')
    elif 'FP Proj Pts' in out.columns:
        out['fp_pts'] = pd.to_numeric(out['FP Proj Pts'], errors='coerce')
    else:
        out['fp_pts'] = np.nan
    # a market total that covers only part of a player's line is not comparable to a full projection
    if 'Mkt Coverage' in out.columns:
        cov = pd.to_numeric(out['Mkt Coverage'], errors='coerce')
        out.loc[cov < 0.9, 'market_pts'] = np.nan
    return out.reset_index(drop=True), info


def _pairwise_acc(pred, actual):
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    if len(pred) < 2:
        return np.nan
    i, j = np.triu_indices(len(pred), 1)
    dp, da = pred[i] - pred[j], actual[i] - actual[j]
    keep = (dp != 0) & (da != 0)
    return float((np.sign(dp[keep]) == np.sign(da[keep])).mean()) if keep.any() else np.nan


def _metrics(pred, actual):
    e = np.asarray(pred, float) - np.asarray(actual, float)
    return {'n': int(len(e)), 'rmse': float(np.sqrt((e ** 2).mean())), 'mae': float(np.abs(e).mean()),
            'bias': float(e.mean()), 'pairwise': _pairwise_acc(pred, actual)}


def _scope_rows(frame, pos, startable, other_col):
    """Rows of one week in scope, paired on model and ``other_col``: position filter, then (startable) the union of
    each source's top-N at that position among the paired rows - the harness's paired START pool."""
    sub = frame if pos is None else frame[frame['Pos'] == pos]
    sub = sub[sub['model_pts'].notna() & sub[other_col].notna()]
    if not startable or sub.empty:
        return sub
    keep = pd.Series(False, index=sub.index)
    for p, grp in sub.groupby('Pos'):
        n = STARTABLE_N.get(p, 30)
        keep.loc[grp['model_pts'].nlargest(n).index] = True
        keep.loc[grp[other_col].nlargest(n).index] = True
    return sub[keep]


def weekly_comparison(frame, benchmark, pos=None, startable=True):
    """Per week (pre-kickoff rows only) and pooled: model vs ``benchmark`` ('Market' or 'FantasyPros') on the paired
    pool. Columns: week, n, model_rmse, bench_rmse, model_bias, bench_bias, model_pairwise, bench_pairwise."""
    col = SOURCES[benchmark]
    live = frame[frame['pre_kickoff']] if 'pre_kickoff' in frame.columns else frame
    rows, pooled = [], []
    for week, wk in live.groupby('week'):
        sub = _scope_rows(wk, pos, startable, col)
        if len(sub) < 5:
            continue
        m, b = _metrics(sub['model_pts'], sub['actual_pts']), _metrics(sub[col], sub['actual_pts'])
        rows.append({'week': int(week), 'n': m['n'], 'model_rmse': m['rmse'], 'bench_rmse': b['rmse'],
                     'model_bias': m['bias'], 'bench_bias': b['bias'],
                     'model_pairwise': m['pairwise'], 'bench_pairwise': b['pairwise']})
        pooled.append(sub)
    out = pd.DataFrame(rows)
    if pooled:
        allp = pd.concat(pooled)
        m, b = _metrics(allp['model_pts'], allp['actual_pts']), _metrics(allp[col], allp['actual_pts'])
        # pairwise is a within-week measure: pool it as the mean over weeks
        out = pd.concat([out, pd.DataFrame([{
            'week': 'All', 'n': m['n'], 'model_rmse': m['rmse'], 'bench_rmse': b['rmse'], 'model_bias': m['bias'],
            'bench_bias': b['bias'], 'model_pairwise': out['model_pairwise'].mean(),
            'bench_pairwise': out['bench_pairwise'].mean()}])], ignore_index=True)
    return out


def disagreement_table(frame, gaps=(2.0, 4.0), positions=(None,) + POSITIONS):
    """When the model and the market disagree by at least ``gap`` points (pre-kickoff rows with a full market line):
    how often the model ended closer to the actual score, split by which side the model was on."""
    live = frame[frame['pre_kickoff']] if 'pre_kickoff' in frame.columns else frame
    sub = live[live['model_pts'].notna() & live['market_pts'].notna()].copy()
    sub['gap'] = sub['model_pts'] - sub['market_pts']
    sub['model_closer'] = (sub['model_pts'] - sub['actual_pts']).abs() < (sub['market_pts'] - sub['actual_pts']).abs()
    rows = []
    for pos in positions:
        p = sub if pos is None else sub[sub['Pos'] == pos]
        for gap in gaps:
            for side, mask in (('Model higher', p['gap'] >= gap), ('Model lower', p['gap'] <= -gap)):
                s = p[mask]
                rows.append({'Position': pos or 'All', 'Gap': f'{gap:g}+ pts', 'Side': side, 'n': int(len(s)),
                             'Model closer': float(s['model_closer'].mean()) if len(s) else np.nan,
                             'Model avg': float(s['model_pts'].mean()) if len(s) else np.nan,
                             'Market avg': float(s['market_pts'].mean()) if len(s) else np.nan,
                             'Actual avg': float(s['actual_pts'].mean()) if len(s) else np.nan})
    return pd.DataFrame(rows)


def stat_comparison(frame, benchmark):
    """Per projected stat, model vs ``benchmark`` on the paired rows (pre-kickoff, both have a number, played)."""
    live = frame[frame['pre_kickoff']] if 'pre_kickoff' in frame.columns else frame
    pattern = BENCHMARK_STAT_COLS[benchmark]
    rows = []
    for stat in STAT_COLS:
        bcol, acol = pattern.format(stat), f'actual_{stat}'
        if stat not in live.columns or bcol not in live.columns or acol not in live.columns:
            continue
        m = pd.to_numeric(live[stat], errors='coerce')
        b = pd.to_numeric(live[bcol], errors='coerce')
        a = pd.to_numeric(live[acol], errors='coerce')
        ok = m.notna() & b.notna() & a.notna()
        if ok.sum() < 10:
            continue
        rows.append({'stat': stat, 'n': int(ok.sum()),
                     'model_rmse': float(np.sqrt(((m[ok] - a[ok]) ** 2).mean())),
                     'bench_rmse': float(np.sqrt(((b[ok] - a[ok]) ** 2).mean())),
                     'model_bias': float((m[ok] - a[ok]).mean()), 'bench_bias': float((b[ok] - a[ok]).mean())})
    return pd.DataFrame(rows)


def line_calls(frame, stats=('receiving_yards', 'rushing_yards', 'passing_yards', 'receptions'),
               min_gaps=None):
    """The model's lean against the market's number for a stat (over when the model is higher, under when lower) and
    how often the actual result landed on the model's side of the market number. Exact ties with the market number and
    pushes are left out. ``min_gaps`` = {stat: smallest |model - market| counted}; defaults to every lean."""
    live = frame[frame['pre_kickoff']] if 'pre_kickoff' in frame.columns else frame
    min_gaps = min_gaps or {}
    rows = []
    for stat in stats:
        mcol, acol = f'Mkt {stat}', f'actual_{stat}'
        if stat not in live.columns or mcol not in live.columns or acol not in live.columns:
            continue
        m = pd.to_numeric(live[stat], errors='coerce')
        line = pd.to_numeric(live[mcol], errors='coerce')
        a = pd.to_numeric(live[acol], errors='coerce')
        for lo in (0.0, min_gaps.get(stat, 0.0)) if min_gaps.get(stat) else (0.0,):
            ok = m.notna() & line.notna() & a.notna() & ((m - line).abs() > max(lo, 1e-9)) & (a != line)
            if ok.sum() == 0:
                continue
            lean_over = (m[ok] > line[ok])
            hit = np.where(lean_over, a[ok] > line[ok], a[ok] < line[ok])
            rows.append({'stat': stat, 'min_gap': lo, 'n': int(ok.sum()),
                         'overs': int(lean_over.sum()), 'over_hit': float(hit[lean_over.to_numpy()].mean())
                         if lean_over.any() else np.nan,
                         'unders': int((~lean_over).sum()), 'under_hit': float(hit[~lean_over.to_numpy()].mean())
                         if (~lean_over).any() else np.nan,
                         'hit': float(hit.mean())})
    return pd.DataFrame(rows)
