"""
Time-valid historical injury data for backtesting the availability/vacancy
layer - docs/model_improvement_plan_2026-09-23.md item 3.

WHY THIS EXISTS. Every backtest to date runs with apply_injury=False,
because build_weekly_projections' own docstring (and this file's methodology
doc) says the injury feed "has no historical week granularity" - true of
data.draft_sources.fetch_injury_report, which only ever returns each
player's LATEST designation for the whole season regardless of which past
week is asked for. It is NOT true of nflreadpy.load_injuries, which carries
the official week-by-week report (report_status Out/Doubtful/Questionable/
blank, gsis_id, date_modified) back to 2019. This module reads that source
instead, so the whole injury/vacancy/pecking-order layer - which fires every
live week and has never been measured against a real outcome - can finally
be backtested.

LEAKAGE GUARD. A row filed (date_modified) after that TEAM's kickoff for the
target week is dropped: a Wednesday practice-report entry is fair game
projecting Sunday's game, a note filed mid-game or the following Monday is
not, even though nflverse files both under the same "week" number.

SAME POLICY AS THE LIVE PATH, ON PURPOSE. `_ASSUME_OUT_STATUSES` is the same
set data.weekly_projections._injury_profiles and
data.availability_overrides._probability already use (Out/Doubtful/IR/
Suspended/Inactive/NFI/PUP -> unavailable, Questionable and everything else
-> healthy) - this module does not invent a new policy, it only supplies a
time-valid version of the same one to a build that could not previously use
it in a backtest at all.
"""
import os

import pandas as pd

from data.availability_overrides import _ASSUME_OUT_STATUSES

CACHE_DIR = os.path.join('data', 'cache')


def load_injury_reports(season):
    """The official week-by-week injury report for one season, REG only,
    cached to data/cache/injuries_{season}.parquet. Never raises - an
    unreachable source returns an empty frame, same degrade-gracefully
    convention as every other historical loader in this app."""
    cache_path = os.path.join(CACHE_DIR, f'injuries_{season}.parquet')
    if os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass
    try:
        import nflreadpy
        df = nflreadpy.load_injuries([season]).to_pandas()
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if 'game_type' in df.columns:
        df = df[df['game_type'].astype(str).str.upper() == 'REG'].copy()
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    except Exception:
        pass
    return df


# Weekly-roster statuses that mean "not eligible to play this week", known
# before kickoff: the reserve lists (IR, IR-designated-to-return, PUP, NFI,
# suspension - every RES/R0x code), the exempt list, and retired. A player
# on one of these is OFF the 53 and never appears on the weekly injury
# report, which is why the report-only replay missed them (2024 wk8:
# Collins R01, Hockenson R04, Jameson Williams R40). Deliberately NOT INA:
# a gameday inactive is announced ~90 minutes before kickoff, later than the
# live board's injury feed, so replaying it would leak. Measured 2022-2025
# wk3-17: of ~2,000 board player-weeks carrying one of these statuses, zero
# recorded a stat that week (the status is a pre-game snapshot).
RESERVE_ROSTER_STATUSES = frozenset({'RES', 'EXE', 'RET'})


def load_weekly_rosters(season):
    """nflverse weekly rosters for one season (REG only, the columns the
    reserve replay needs), cached to data/cache/rosters_weekly_{season}.parquet.
    Never raises - an unreachable source returns an empty frame."""
    cache_path = os.path.join(CACHE_DIR, f'rosters_weekly_{season}.parquet')
    if os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass
    try:
        import nflreadpy
        df = nflreadpy.load_rosters_weekly([season]).to_pandas()
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if 'game_type' in df.columns:
        df = df[df['game_type'].astype(str).str.upper() == 'REG']
    keep = [c for c in ('season', 'week', 'team', 'full_name', 'gsis_id', 'position', 'status',
                        'status_description_abbr') if c in df.columns]
    df = df[keep].copy()
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    except Exception:
        pass
    return df


def historical_reserve_profiles(season, week):
    """{player_name: profile} for every player on a reserve/exempt/retired
    list in this (season, week)'s weekly roster - same shape as
    historical_injury_profiles, plays_probability 0.0. Empty on any failure."""
    try:
        rosters = load_weekly_rosters(season)
        if rosters is None or rosters.empty or 'week' not in rosters.columns:
            return {}
        wk = rosters[(pd.to_numeric(rosters['week'], errors='coerce') == int(week))
                     & rosters['status'].astype(str).str.upper().isin(RESERVE_ROSTER_STATUSES)]
        out = {}
        for _, row in wk.iterrows():
            name = row.get('full_name')
            if not name or pd.isna(name):
                continue
            code = str(row.get('status_description_abbr') or '').strip()
            out[str(name)] = {
                'status': f"reserve list ({str(row.get('status')).upper()}{'/' + code if code else ''})",
                'plays_probability': 0.0,
                'workload_if_active': 1.0,
                'source_year': int(season),
                'source': 'historical weekly roster (reserve list)',
                'gsis_id': row.get('gsis_id', ''),
                'team': row.get('team', ''),
            }
        return out
    except Exception:
        return {}


def merge_reserve_profiles(injury_profiles, reserve_profiles):
    """Injury-report profiles plus reserve-list players. A reserve-list
    status wins over a same-week report entry (a player can't be both
    Questionable and on IR); an existing OUT entry is left as is."""
    merged = dict(injury_profiles or {})
    for name, profile in (reserve_profiles or {}).items():
        existing = merged.get(name)
        if existing is None or float(existing.get('plays_probability', 1.0)) > 0.0:
            merged[name] = profile
    return merged


def _kickoff_by_team(week_schedule):
    """{TEAM -> kickoff timestamp (tz-aware, UTC)} for one week's games.
    nflverse `gametime` is documented as US/Eastern; a missing gametime
    falls back to the early-window default (13:00 ET) rather than dropping
    the game, since a late-arriving row is the rarer, more suspicious case
    to silently keep - see the caller's stale-row check."""
    if week_schedule is None or week_schedule.empty:
        return {}
    if not {'gameday', 'home_team', 'away_team'}.issubset(week_schedule.columns):
        return {}
    gameday = pd.to_datetime(week_schedule['gameday'], errors='coerce')
    gametime = (week_schedule['gametime'].astype(str) if 'gametime' in week_schedule.columns
               else pd.Series('13:00', index=week_schedule.index))
    gametime = gametime.where(gametime.str.match(r'^\d{1,2}:\d{2}$', na=False), '13:00')
    naive = pd.to_datetime(gameday.dt.strftime('%Y-%m-%d') + ' ' + gametime, errors='coerce')
    kickoff_utc = naive.dt.tz_localize('US/Eastern', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')

    out = {}
    for (_, row), k in zip(week_schedule.iterrows(), kickoff_utc):
        if pd.isna(k):
            continue
        for side in ('home_team', 'away_team'):
            team = row.get(side)
            if team:
                out[str(team).upper()] = k
    return out


def historical_injury_profiles(season, week, schedule_df):
    """{player_name: {...}} in EXACTLY the shape
    data.weekly_projections._injury_profiles returns live, so
    build_weekly_projections' existing resolve_target_week_availability path
    (data.availability_overrides) can consume it completely unchanged -
    including its own gsis_id-first matching, since each profile here also
    carries 'gsis_id' and 'team'. Empty dict on any failure or when there is
    nothing usable for this (season, week) - never raises.
    """
    try:
        reports = load_injury_reports(season)
        if reports is None or reports.empty or 'week' not in reports.columns:
            return {}
        wk = reports[pd.to_numeric(reports['week'], errors='coerce') == int(week)].copy()
        if wk.empty:
            return {}

        week_schedule = pd.DataFrame()
        if schedule_df is not None and not schedule_df.empty and 'week' in schedule_df.columns:
            week_schedule = schedule_df[pd.to_numeric(schedule_df['week'], errors='coerce') == int(week)]
        kickoff_by_team = _kickoff_by_team(week_schedule)
        if kickoff_by_team and 'date_modified' in wk.columns and 'team' in wk.columns:
            team_kickoff = pd.to_datetime(
                wk['team'].astype(str).str.upper().map(kickoff_by_team), errors='coerce', utc=True)
            modified = pd.to_datetime(wk['date_modified'], errors='coerce', utc=True)
            stale = modified.notna() & team_kickoff.notna() & (modified > team_kickoff)
            wk = wk[~stale]
        if wk.empty:
            return {}

        # A player can have more than one row in a week's report (separate
        # practice-day filings); the most recently modified one is that
        # week's final designation.
        if 'date_modified' in wk.columns and 'gsis_id' in wk.columns:
            wk = wk.sort_values('date_modified').drop_duplicates(subset=['gsis_id'], keep='last')

        out = {}
        for _, row in wk.iterrows():
            name = row.get('full_name')
            if not name or pd.isna(name):
                continue
            label = str(row.get('report_status') or '').strip().lower()
            availability = 0.0 if label in _ASSUME_OUT_STATUSES else 1.0
            out[str(name)] = {
                'status': label or 'unknown',
                'plays_probability': availability,
                'workload_if_active': 1.0,
                'source_year': int(season),
                'source': 'historical injury replay',
                'gsis_id': row.get('gsis_id', ''),
                'team': row.get('team', ''),
            }
        return out
    except Exception:
        return {}


def significant_out_players(season, week, schedule_df, stats_df, name_col='name',
                            lookback=4, threshold=0.5):
    """{player_name: team} for players ruled unavailable this week (per
    historical_injury_profiles) whose own recent CURRENT-SEASON snap share
    (the mean of up to his team's last `lookback` PLAYED weeks strictly
    before `week` - same convention as
    data.weekly_projections.expected_snap_share) was >= `threshold`: a real
    starter-level loss, not a healthy scratch or a rostered depth piece.
    Used to build the harness's RECIPIENT scope for scoring
    v2_historical_injury_replay - see
    docs/model_improvement_plan_2026-09-23.md item 3's protocol. Empty dict
    on any failure or missing snap-share data."""
    try:
        profiles = historical_injury_profiles(season, week, schedule_df)
        out_names = {name: p.get('team', '') for name, p in profiles.items()
                    if p.get('plays_probability', 1.0) <= 0.0}
        if not out_names or stats_df is None or stats_df.empty or 'weekly_snap_pct' not in stats_df.columns:
            return {}
        hist = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') < int(week)]
        if hist.empty:
            return {}
        recent = hist.sort_values('week').groupby(name_col, observed=True).tail(lookback)
        share = recent.groupby(name_col, observed=True)['weekly_snap_pct'].mean() / 100.0
        return {name: team for name, team in out_names.items() if share.get(name, 0.0) >= threshold}
    except Exception:
        return {}


def recipient_teammates(out_players_by_team, board_players_by_team):
    """Teammates (excluding the injured player himself) of every team in
    `out_players_by_team` (significant_out_players' output), restricted to
    players actually present on the board. `board_players_by_team` is
    {team: set(players on that team's board this week)}. Returns a set."""
    recipients = set()
    for injured, team in out_players_by_team.items():
        recipients |= {p for p in board_players_by_team.get(team, set()) if p != injured}
    return recipients
