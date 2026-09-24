"""
A permanent, per-week record of what the model, the market, and FantasyPros
each said for a real board - see docs/model_improvement_plan_2026-09-23.md
item 2c.

WHY THIS DOESN'T ALREADY EXIST. build_weekly_projections is cached IN
SESSION, and a live market/FantasyPros pull is never saved at all beyond
data.odds_weekly's own one-file cache (overwritten every week - see that
module's PROPS_ARCHIVE_DIR comment). The model side has no leakage risk - it
can always be rebuilt as-of any past week from the raw stats - but the
market and FantasyPros numbers for a week that has already happened are
GONE once the live pull moves on, so "was the model actually better than the
market three weeks ago" was a question this repo could never answer. This
writes one row per player, per real board build, to a durable file; nothing
here changes what the model or the UI compute.

Written ONLY on an explicit board build (the UI calls record_board right
after build_weekly_projections, not on every fragment rerun a click causes -
see ui/tabs/rankings.py), and throttled to at most one file per (year, week,
feature hash) per hour via _recent_ledger_path, so a page full of clicks in
one sitting doesn't spam the ledger directory.
"""
import glob
import hashlib
import os
import subprocess

import pandas as pd

from data.utils import clean_name_exact

LEDGER_DIR = os.path.join('data', 'ledger')

# Every stat build_weekly_projections might carry, across all four
# positions (data.weekly_projections.OFFENSE_PROJECTION_STATS) - a superset
# is fine, only the columns actually present get written.
STAT_COLS = ['passing_attempts', 'passing_completions', 'passing_yards', 'passing_tds',
            'passing_interceptions', 'rushing_attempts', 'rushing_yards', 'rushing_tds',
            'targets', 'receptions', 'receiving_yards', 'receiving_tds']

_META_COLS = ['Player', 'Pos', 'Team', 'Opponent', 'Games This Season',
             'Availability', 'Injury Status', 'Raw Model Proj Pts', 'Model Proj Pts']

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git_sha():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except Exception:
        return None


def feature_hash(feats):
    """A short, stable id for a feature set, so a re-build under a different
    MODEL_FEATURES combination doesn't silently overwrite/collide with an
    earlier one in the throttle check below. None (unspecified) hashes to a
    fixed sentinel rather than colliding with an accidental empty set."""
    if feats is None:
        return 'unspecified'
    key = ','.join(sorted(str(f) for f in feats))
    return hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]


def _recent_ledger_path(year, week, feat_hash, within_seconds=3600):
    """The most recent ledger file already covering this exact (year, week,
    feature hash) within the last hour, if any - so record_board is safe to
    call from a UI fragment that reruns on every click without writing a new
    file each time."""
    pattern = os.path.join(LEDGER_DIR, f'{year}_wk{week:02d}_*.parquet')
    for path in sorted(glob.glob(pattern), reverse=True):
        try:
            mtime_age = pd.Timestamp.now().timestamp() - os.path.getmtime(path)
            if mtime_age >= within_seconds:
                continue
            existing_hash = pd.read_parquet(path, columns=['feature_hash'])['feature_hash'].iloc[0]
        except Exception:
            continue
        if existing_hash == feat_hash:
            return path
    return None


def _market_columns(market_df):
    """(name_col, {source_col: ledger_col}) for the market frame -
    data.odds_weekly.weekly_market_projection's output. consensus_stats
    (e.g. 'receiving_yards') are read off market_df.attrs, the same
    contract ui/tabs/rankings.py already relies on."""
    name_col = 'Player' if 'Player' in market_df.columns else 'player'
    consensus_stats = [s for s in market_df.attrs.get('consensus_stats', []) if s in market_df.columns]
    rename = {'Market Pts': 'Mkt Market Pts', 'Coverage': 'Mkt Coverage'}
    rename.update({s: f'Mkt {s}' for s in consensus_stats})
    return name_col, rename


def build_ledger_frame(year, week, model_df, market_df=None, fp_weekly=None, feats=None):
    """The frame record_board writes, built (and returned) separately from
    the file write so tests can check its shape directly. Raises on a
    genuinely malformed input - callers that must never raise (the live UI)
    go through record_board instead."""
    out = model_df[[c for c in _META_COLS if c in model_df.columns]].copy()
    for stat in STAT_COLS:
        if stat in model_df.columns:
            out[stat] = model_df[stat]
    out['_key'] = clean_name_exact(out['Player'])

    if market_df is not None and not market_df.empty:
        name_col, rename = _market_columns(market_df)
        m = market_df[[name_col] + list(rename)].rename(columns={name_col: '_mkt_player', **rename})
        m['_key'] = clean_name_exact(m['_mkt_player'])
        m = m.drop(columns=['_mkt_player']).drop_duplicates(subset=['_key'])
        out = out.merge(m, on='_key', how='left')

    if fp_weekly is not None and not fp_weekly.empty and 'Player' in fp_weekly.columns:
        fp_pts_cols = [c for c in fp_weekly.columns if c.startswith('FP Proj Pts')]
        fp_stat_cols = [c for c in STAT_COLS if c in fp_weekly.columns]
        fp = fp_weekly[['Player'] + fp_pts_cols + fp_stat_cols].copy()
        fp = fp.rename(columns={c: f'FP {c}' for c in fp_stat_cols})
        fp['_key'] = clean_name_exact(fp['Player'])
        fp = fp.drop(columns=['Player']).drop_duplicates(subset=['_key'])
        out = out.merge(fp, on='_key', how='left')

    out = out.drop(columns=['_key'])
    out['year'] = year
    out['week'] = week
    out['feature_hash'] = feature_hash(feats)
    out['git_sha'] = _git_sha()
    out['build_ts'] = pd.Timestamp.now(tz='UTC')
    return out


def record_board(year, week, model_df, market_df=None, fp_weekly=None, feats=None):
    """Write one ledger row per model-board player. Never raises - a failed
    ledger write must not break the board a user is looking at. Returns the
    path written (a fresh file, or a recent-enough existing one reused
    per the throttle), or None if there was nothing to write."""
    try:
        if model_df is None or model_df.empty or 'Player' not in model_df.columns:
            return None
        feat_hash = feature_hash(feats)
        existing = _recent_ledger_path(year, week, feat_hash)
        if existing:
            return existing

        out = build_ledger_frame(year, week, model_df, market_df, fp_weekly, feats)
        os.makedirs(LEDGER_DIR, exist_ok=True)
        stamp = out['build_ts'].iloc[0].strftime('%Y%m%dT%H%M%S%fZ')
        path = os.path.join(LEDGER_DIR, f'{year}_wk{week:02d}_{stamp}.parquet')
        out.to_parquet(path, index=False)
        return path
    except Exception:
        return None


def latest_ledger_paths(year=None):
    """{(year, week): path of the latest build} across every ledger file on
    disk (optionally restricted to one season) - scripts/score_ledger.py's
    'use the latest pre-kickoff build for each week' rule."""
    pattern = os.path.join(LEDGER_DIR, f'{year if year else "*"}_wk*_*.parquet')
    latest = {}
    for path in glob.glob(pattern):
        base = os.path.basename(path)
        try:
            y_str, wk_str, _rest = base.split('_', 2)
            y, wk = int(y_str), int(wk_str[2:])
        except (ValueError, IndexError):
            continue
        key = (y, wk)
        if key not in latest or os.path.getmtime(path) > os.path.getmtime(latest[key]):
            latest[key] = path
    return latest
