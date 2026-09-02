"""
Scoring, percentile, and report-building transforms that run on top of the
raw data from data/loaders.py: fantasy scoring, league percentiles, historical
summaries, risers, rookie watch, VORP draft sheet, depth chart synthesis, and
the name-matching helpers used by the Live Draft Tracker's paste/sync parser.
"""
import re
import pandas as pd
import numpy as np
import streamlit as st

from data.utils import calculate_percentile, calculate_percentile_qualified, clean_name_exact, clean_name_for_merge
from data.loaders import load_pfr_pass_block, load_pfr_def_pressure, load_team_pass_attempts_faced, load_schedule, load_pbp


def build_redzone_usage(_stats_df, name_col, year):
    """
    Per-player red-zone (yardline_100 <= 20) target/carry share and TDs,
    built from nflreadpy's raw play-by-play (nflfastR) - no pre-aggregated
    nflverse export (player_stats, snap_counts, etc) carries this at all,
    so there was no season-long red-zone usage signal anywhere in this app
    before. Play-by-play identifies players by an abbreviated "F.Last"
    string (real collision risk - multiple real "J.Jefferson"-style
    players), not a full display name, so this joins by gsis_id
    (receiver_player_id/rusher_player_id) instead - the same id format
    already sitting in _stats_df['player_id'] from load_year_data - then
    maps back to _stats_df's own name_col for that id, so the result is
    keyed by whatever name format the caller's table already uses (no new
    third name format to reconcile against).

    `_stats_df` underscore-prefixed (not hashed, same convention as
    apply_scoring_and_percentiles) since load_pbp itself is already
    cached on `year` - re-running this groupby per call is cheap relative
    to hashing a wide multi-thousand-row stats frame on every rerun.
    """
    pbp = load_pbp(year)
    if pbp.empty or _stats_df.empty or 'player_id' not in _stats_df.columns or name_col not in _stats_df.columns:
        return pd.DataFrame()
    required = {'yardline_100', 'play_type', 'posteam', 'receiver_player_id', 'rusher_player_id', 'complete_pass', 'pass_touchdown', 'rush_touchdown', 'play_id'}
    if not required.issubset(pbp.columns):
        return pd.DataFrame()
    # Regular season only - load_pbp doesn't filter this itself, and this
    # was silently mixing playoff plays into "season" red-zone share for
    # any team that made a playoff run (same contamination class already
    # fixed for weekly stats in load_year_data - that fix was never
    # extended to pbp's consumers until now).
    if 'season_type' in pbp.columns:
        pbp = pbp[pbp['season_type'] == 'REG']

    rz = pbp[(pbp['yardline_100'] <= 20) & (pbp['play_type'].isin(['run', 'pass']))]
    if rz.empty:
        return pd.DataFrame()

    targets = rz[rz['receiver_player_id'].notna()].groupby(['posteam', 'receiver_player_id'], observed=True).agg(
        rz_targets=('play_id', 'count'), rz_receptions=('complete_pass', 'sum'), rz_rec_tds=('pass_touchdown', 'sum'),
    ).reset_index().rename(columns={'receiver_player_id': 'player_id'})
    carries = rz[rz['rusher_player_id'].notna()].groupby(['posteam', 'rusher_player_id'], observed=True).agg(
        rz_carries=('play_id', 'count'), rz_rush_tds=('rush_touchdown', 'sum'),
    ).reset_index().rename(columns={'rusher_player_id': 'player_id'})

    team_rz_targets = targets.groupby('posteam', observed=True)['rz_targets'].sum()
    team_rz_carries = carries.groupby('posteam', observed=True)['rz_carries'].sum()
    targets['rz_target_share'] = (targets['rz_targets'] / targets['posteam'].map(team_rz_targets).replace(0, 1) * 100).round(1)
    carries['rz_carry_share'] = (carries['rz_carries'] / carries['posteam'].map(team_rz_carries).replace(0, 1) * 100).round(1)

    usage = pd.merge(
        targets[['player_id', 'rz_targets', 'rz_target_share', 'rz_receptions', 'rz_rec_tds']],
        carries[['player_id', 'rz_carries', 'rz_carry_share', 'rz_rush_tds']],
        on='player_id', how='outer',
    )
    value_cols = [c for c in usage.columns if c != 'player_id']
    usage[value_cols] = usage[value_cols].fillna(0)

    id_to_name = _stats_df[['player_id', name_col]].dropna().drop_duplicates(subset=['player_id']).set_index('player_id')[name_col]
    usage[name_col] = usage['player_id'].map(id_to_name)
    return usage.dropna(subset=[name_col]).drop(columns=['player_id'])


# Volume floors for the EPA/success-rate stats below - a per-PLAY efficiency
# metric (unlike this app's existing game-count-based floors, e.g.
# build_recent_form_rank's 2-game minimum) needs an attempt-count
# floor instead: a QB's EPA/dropback over 8 plays is nearly meaningless
# noise, and letting a tiny sample land near the top of a percentile ranking
# would be exactly the kind of small-sample distortion this app already
# guards against elsewhere. 100 dropbacks/20 carries/15 targets roughly
# match standard "qualified leaderboard" conventions for a single season.
MIN_DROPBACKS_FOR_EPA = 100
MIN_CARRIES_FOR_EPA = 20
MIN_TARGETS_FOR_EPA = 15


def _pbp_reg_season_only(year):
    """
    Shared load_pbp(year) + regular-season filter for both functions below -
    see the identical fix (and its rationale) in build_redzone_usage just
    above. Returns an empty DataFrame (not an exception) if pbp itself is
    unavailable, so callers can use the same `if pbp.empty: return ...`
    short-circuit they already use elsewhere in this file.
    """
    pbp = load_pbp(year)
    if pbp.empty:
        return pbp
    if 'season_type' in pbp.columns:
        pbp = pbp[pbp['season_type'] == 'REG']
    return pbp


def build_epa_efficiency(_stats_df, name_col, year):
    """
    Per-player EPA (Expected Points Added) and success rate, split by role -
    the same raw signal both nflsavant.com and baseballsavant.mlb.com build
    their headline "how efficient is this player, not just how much volume
    do they get" visuals from. Nothing in this app surfaced EPA/success/CPOE
    before this, despite load_pbp already carrying all three - red-zone
    share (build_redzone_usage) was the only consumer of play-by-play at all.

    QB dropback EPA deliberately filters on the `qb_dropback` flag, NOT
    `play_type == 'pass'` - the two are NOT equivalent. Confirmed against
    real cached play-by-play: roughly 5% of real dropbacks are scrambles
    (`play_type == 'run'`, `passer_player_id` null, `rusher_player_id` =
    the QB), which `play_type == 'pass'` alone silently excludes - exactly
    the plays that make a mobile QB's efficiency look worse than it really
    is. `passer_player_id.fillna(rusher_player_id)` attributes both dropback
    types to the right QB. `cpoe` is only ever non-null on true pass
    attempts anyway, so it doesn't need a separate filter - `.mean()`
    already skips the scramble rows' NaNs.

    Same join pattern as build_redzone_usage: play-by-play identifies
    players by gsis_id (passer_player_id/rusher_player_id/receiver_player_id),
    not name, so this joins on id first and maps back to `_stats_df`'s own
    `name_col` afterward - no new name format to reconcile against.

    `_stats_df` underscore-prefixed (not hashed) - load_pbp(year) is already
    cached, and this groupby is cheap enough to redo per call, same
    reasoning as build_redzone_usage.
    """
    pbp = _pbp_reg_season_only(year)
    if pbp.empty or _stats_df.empty or 'player_id' not in _stats_df.columns or name_col not in _stats_df.columns:
        return pd.DataFrame()
    required = {'qb_dropback', 'play_type', 'passer_player_id', 'rusher_player_id', 'receiver_player_id', 'epa', 'success', 'cpoe', 'play_id'}
    if not required.issubset(pbp.columns):
        return pd.DataFrame()

    dropbacks = pbp[pbp['qb_dropback'] == 1].copy()
    dropbacks['epa_player_id'] = dropbacks['passer_player_id'].fillna(dropbacks['rusher_player_id'])
    passing = dropbacks[dropbacks['epa_player_id'].notna()].groupby('epa_player_id', observed=True).agg(
        epa_per_dropback=('epa', 'mean'), success_rate_pass=('success', 'mean'),
        cpoe_avg=('cpoe', 'mean'), dropbacks=('play_id', 'count'),
    ).reset_index().rename(columns={'epa_player_id': 'player_id'})
    passing = passing[passing['dropbacks'] >= MIN_DROPBACKS_FOR_EPA].drop(columns=['dropbacks'])

    rushes = pbp[(pbp['play_type'] == 'run') & pbp['rusher_player_id'].notna()]
    rushing = rushes.groupby('rusher_player_id', observed=True).agg(
        epa_per_rush=('epa', 'mean'), success_rate_rush=('success', 'mean'), carries=('play_id', 'count'),
    ).reset_index().rename(columns={'rusher_player_id': 'player_id'})
    rushing = rushing[rushing['carries'] >= MIN_CARRIES_FOR_EPA].drop(columns=['carries'])

    targeted = pbp[pbp['receiver_player_id'].notna()]
    receiving = targeted.groupby('receiver_player_id', observed=True).agg(
        epa_per_target=('epa', 'mean'), success_rate_rec=('success', 'mean'), targets=('play_id', 'count'),
    ).reset_index().rename(columns={'receiver_player_id': 'player_id'})
    receiving = receiving[receiving['targets'] >= MIN_TARGETS_FOR_EPA].drop(columns=['targets'])

    efficiency = pd.merge(passing, rushing, on='player_id', how='outer')
    efficiency = pd.merge(efficiency, receiving, on='player_id', how='outer')
    if efficiency.empty:
        return pd.DataFrame()

    id_to_name = _stats_df[['player_id', name_col]].dropna().drop_duplicates(subset=['player_id']).set_index('player_id')[name_col]
    efficiency[name_col] = efficiency['player_id'].map(id_to_name)
    return efficiency.dropna(subset=[name_col]).drop(columns=['player_id'])


def build_target_location_profile(_stats_df, name_col, year):
    """
    Per-receiver share of targets by depth x direction - the honest,
    data-available stand-in for nflsavant.com's "route tree" visual. A
    literal route tree needs per-route geometry/concept charting data this
    app doesn't have locally (the local ftn_charting_*.csv files only carry
    play-level boolean flags like is_play_action/is_screen_pass, not route
    shapes) - this uses only what's actually in play-by-play instead:
    `air_yards` (bucketed short/intermediate/deep) crossed with
    `pass_location` (left/middle/right), so it's a genuinely different,
    coarser thing. Deliberately not called a "route tree" anywhere in code
    or UI copy.

    Same gsis_id join pattern as build_redzone_usage/build_epa_efficiency.
    """
    pbp = _pbp_reg_season_only(year)
    if pbp.empty or _stats_df.empty or 'player_id' not in _stats_df.columns or name_col not in _stats_df.columns:
        return pd.DataFrame()
    required = {'receiver_player_id', 'air_yards', 'pass_location', 'play_id'}
    if not required.issubset(pbp.columns):
        return pd.DataFrame()

    targeted = pbp[pbp['receiver_player_id'].notna() & pbp['air_yards'].notna() & pbp['pass_location'].notna()].copy()
    targeted['depth_bucket'] = pd.cut(
        targeted['air_yards'], bins=[-100, 9, 19, 200], labels=['Short (<10yd)', 'Intermediate (10-19yd)', 'Deep (20+yd)'],
    )
    targeted['direction'] = targeted['pass_location'].str.title()

    totals = targeted.groupby('receiver_player_id', observed=True)['play_id'].count()
    qualified = totals[totals >= MIN_TARGETS_FOR_EPA].index
    targeted = targeted[targeted['receiver_player_id'].isin(qualified)]
    if targeted.empty:
        return pd.DataFrame()

    grid = targeted.groupby(['receiver_player_id', 'depth_bucket', 'direction'], observed=True)['play_id'].count().reset_index(name='targets')
    grid['share_pct'] = (grid['targets'] / grid.groupby('receiver_player_id')['targets'].transform('sum') * 100).round(1)

    id_to_name = _stats_df[['player_id', name_col]].dropna().drop_duplicates(subset=['player_id']).set_index('player_id')[name_col]
    grid[name_col] = grid['receiver_player_id'].map(id_to_name)
    return grid.dropna(subset=[name_col]).drop(columns=['receiver_player_id'])


def _merge_redzone_share(df, _stats_df, stats_name_col, year, out_name_col='Player', pos_col='Pos'):
    """
    Left-merges red-zone target share (WR/TE) and red-zone carry share (RB)
    onto a per-player table - the replacement for RYOE (see build_rookie_watch)
    on boards that mix positions, since RYOE is RB-only and reads as an
    always-blank column for the WRs/TEs that dominate a typical rookie
    board. Both share columns are position-conditional - blank outside
    their relevant position.

    Two separate name-column params on purpose: `stats_name_col` is the
    name column as it exists in `_stats_df` (e.g. 'player_display_name',
    from load_year_data) - build_redzone_usage needs that exact column to
    bridge its gsis_id-keyed play-by-play aggregation back to a name at
    all. `out_name_col` is whatever `df` (the caller's already-built,
    already-renamed board) calls that same column (e.g. 'Player', after
    build_rookie_watch's own rename_map) - used only for the final merge
    key here. Both refer to the same underlying player names, just under
    two different column labels at two different stages of the pipeline,
    so an exact-name match (not the looser suffix-stripped one) is safe.
    """
    df = df.copy()
    usage = build_redzone_usage(_stats_df, stats_name_col, year)
    if usage.empty or out_name_col not in df.columns:
        df['RZ Tgt Share %'] = np.nan
        df['RZ Rush Share %'] = np.nan
        return df
    df['_merge_key'] = clean_name_exact(df[out_name_col])
    usage = usage.copy()
    usage['_merge_key'] = clean_name_exact(usage[stats_name_col])
    usage = usage.drop(columns=[stats_name_col]).drop_duplicates(subset=['_merge_key'])
    df = pd.merge(df, usage[['_merge_key', 'rz_target_share', 'rz_carry_share']], on='_merge_key', how='left')
    df = df.drop(columns=['_merge_key']).rename(columns={'rz_target_share': 'RZ Tgt Share %', 'rz_carry_share': 'RZ Rush Share %'})
    if pos_col in df.columns:
        df.loc[~df[pos_col].isin(['WR', 'TE']), 'RZ Tgt Share %'] = np.nan
        df.loc[df[pos_col] != 'RB', 'RZ Rush Share %'] = np.nan
    return df


@st.cache_data
def build_points_allowed_matrix(_stats_df, year):
    """
    Fantasy points allowed per game by opponent team and position - the
    source matrix for the Defensive Yield heatmap AND the rest-of-season
    strength-of-schedule table below (which re-aggregates these same
    per-defense rows onto each team's set of upcoming opponents). Per-game
    average (not season sum) so bye weeks don't distort a team's rate.

    `_stats_df` underscore-prefixed (not hashed, same convention as
    apply_scoring_and_percentiles) - `year` is a cheap, real cache key.
    IMPORTANT: without a real (non-underscore) argument here, every call
    would share one cache entry regardless of which year's frame was
    actually passed in, since st.cache_data has nothing left to hash -
    caught by testing a year switch in the running app returning stale data.
    """
    if _stats_df.empty or 'opponent_team' not in _stats_df.columns:
        return pd.DataFrame()
    gp = _stats_df.groupby('opponent_team', observed=True)['week'].nunique().to_dict()
    def_pts = _stats_df.groupby(['opponent_team', 'position'], observed=True)['fantasy_points'].sum().reset_index()
    def_pts['games'] = def_pts['opponent_team'].map(gp).replace(0, 1)
    def_pts['pts_per_game'] = def_pts['fantasy_points'] / def_pts['games']
    matrix = def_pts.pivot(index='opponent_team', columns='position', values='pts_per_game').fillna(0)
    matrix = matrix[[c for c in ['QB', 'RB', 'WR', 'TE'] if c in matrix.columns]].reset_index()
    matrix.rename(columns={'opponent_team': 'Team'}, inplace=True)
    matrix['Team'] = matrix['Team'].astype(str)
    return matrix


def build_stat_allowed_matrix(_stats_df, position_filter=None):
    """
    Per-game average of each raw counting stat allowed by an opponent team,
    filtered to one position group (e.g. position_filter=['WR'] for "how
    many targets/yards/TDs does this defense allow to WRs") - the same
    per-game-average-by-opponent idea as build_points_allowed_matrix,
    generalized from fantasy_points down to the underlying raw stats the
    matchup projection model (build_player_projection below) blends
    against. Feeds that model's opponent-adjustment multiplier: this
    opponent's allowed rate vs the league-average allowed rate.

    Deliberately NOT @st.cache_data: a cheap groupby over an already-loaded,
    already-cached dataframe, and position_filter is a list (unhashable) -
    caching this would need a tuple-ified key and buys nothing real given
    how cheap the groupby itself is (see build_recent_trend's docstring for
    the same reasoning, and build_points_allowed_matrix's for what a
    missing/wrong cache key looks like when it goes unnoticed).
    """
    if _stats_df.empty or 'opponent_team' not in _stats_df.columns:
        return pd.DataFrame()
    df = _stats_df
    if position_filter:
        df = df[df['position'].isin(position_filter)]
    if df.empty:
        return pd.DataFrame()
    stat_cols = ['targets', 'receptions', 'receiving_yards', 'receiving_tds',
                 'rushing_attempts', 'rushing_yards', 'rushing_tds',
                 'passing_attempts', 'passing_completions', 'passing_yards',
                 'passing_tds', 'passing_interceptions']
    stat_cols = [c for c in stat_cols if c in df.columns]
    if not stat_cols:
        return pd.DataFrame()
    gp = df.groupby('opponent_team', observed=True)['week'].nunique().replace(0, 1)
    totals = df.groupby('opponent_team', observed=True)[stat_cols].sum()
    return totals.div(gp, axis=0)


def build_alignment_multiplier(_coverage_df, opponent_nickname, slot_rate, wide_rate):
    """
    WR/TE slot-vs-outside matchup refinement: blends how much yardage per
    target this opponent allows to SLOT-aligned vs OUTSIDE-aligned receivers
    (Sharp Football's positional coverage export, data.loaders.
    load_sharp_positional_coverage - team names are bare nicknames, e.g.
    "Dolphins", not abbreviations - see data.coverage_radar.team_nickname)
    weighted by this player's own slot_rate/wide_rate snap split (PFF
    receiving_summary) - a defense that's soft over the middle matters a lot
    more for a slot-heavy receiver than a boundary/X receiver, and vice
    versa. Falls back to a neutral 1.0 multiplier if either input is
    missing (no PFF slot/wide split for this player, or this opponent isn't
    in the - 2025-only - Sharp Football export).
    """
    if _coverage_df.empty or 'team' not in _coverage_df.columns:
        return 1.0
    row = _coverage_df[_coverage_df['team'] == opponent_nickname]
    if row.empty or 'ypt_allowed_slot' not in _coverage_df.columns or 'ypt_allowed_outside' not in _coverage_df.columns:
        return 1.0
    league_slot_avg = _coverage_df['ypt_allowed_slot'].mean()
    league_outside_avg = _coverage_df['ypt_allowed_outside'].mean()
    if league_slot_avg <= 0 or league_outside_avg <= 0:
        return 1.0
    slot_mult = float(row.iloc[0]['ypt_allowed_slot']) / league_slot_avg
    outside_mult = float(row.iloc[0]['ypt_allowed_outside']) / league_outside_avg
    slot_frac = (slot_rate or 0) / 100.0
    wide_frac = (wide_rate or 0) / 100.0
    remainder = max(0.0, 1.0 - slot_frac - wide_frac)
    blended = slot_frac * slot_mult + wide_frac * outside_mult + remainder * 1.0
    return float(np.clip(blended, 0.75, 1.3))


OFFENSE_PROJECTION_STATS = {
    'QB': ['passing_attempts', 'passing_completions', 'passing_yards', 'passing_tds',
           'passing_interceptions', 'rushing_attempts', 'rushing_yards', 'rushing_tds'],
    'RB': ['rushing_attempts', 'rushing_yards', 'rushing_tds', 'targets', 'receptions',
           'receiving_yards', 'receiving_tds'],
    'WR': ['targets', 'receptions', 'receiving_yards', 'receiving_tds'],
    'TE': ['targets', 'receptions', 'receiving_yards', 'receiving_tds'],
}


def score_projected_stats(proj, scoring_mode):
    """
    Same weights as apply_scoring_and_percentiles' offensive terms, applied
    to a dict of PROJECTED (not actual) raw stats instead of a stats
    DataFrame column - kept in sync manually since one operates on a dict of
    scalars per-player and the other on a whole-league DataFrame column; if
    the league scoring weights above ever change, update both.
    """
    rec_mult = 1.0 if "Full" in scoring_mode else (0.5 if "Half" in scoring_mode else 0.0)
    return round(
        proj.get('rushing_yards', 0) * 0.1 + proj.get('rushing_tds', 0) * 6 +
        proj.get('receiving_yards', 0) * 0.1 + proj.get('receiving_tds', 0) * 6 +
        proj.get('receptions', 0) * rec_mult +
        proj.get('passing_yards', 0) * 0.04 + proj.get('passing_tds', 0) * 4 -
        proj.get('passing_interceptions', 0) * 2
    , 1)


def build_player_projection(p_data, pos, opponent_abbr, scoring_mode, allowed_matrix, pace_df, alignment_mult=1.0):
    """
    Next-game per-stat projection for one player. Blends each raw counting
    stat's last-4-game form with their full-season average (60/40, weighted
    toward recent form), then adjusts for two matchup factors: how much of
    that stat this opponent allows to the player's position relative to
    league average (`allowed_matrix`, from build_stat_allowed_matrix), and
    the opponent's pace - plays faced per game (`pace_df`, from
    data.loaders.load_team_pace) relative to league average. Both
    multipliers are clipped so a small early-season sample (one unusually
    bad game allowed) can't swing a projection to an implausible extreme.
    `alignment_mult` (see build_alignment_multiplier) optionally folds a
    WR/TE's own slot/outside split into their receiving stats specifically.

    Deliberately projects each raw counting stat directly (recent/season
    blend x matchup x pace) rather than decomposing into yards-per-target x
    targets, TD rate x carries, etc. - simpler, and for a personal
    single-user tool the opponent-allowed signal is captured either way;
    a full per-play efficiency model would be a much larger undertaking for
    marginal accuracy gain here.

    Returns (stat_projections: dict[str, float], projected_fantasy_points: float).
    """
    stats = OFFENSE_PROJECTION_STATS.get(pos, [])
    if p_data.empty or not stats:
        return {}, 0.0

    df = p_data
    if 'week' not in df.columns:
        # No weekly-granularity stats file exists for this season at all yet
        # (e.g. 2026 before kickoff) - load_year_data's roster-only fallback
        # still emits ONE placeholder row per player with every counting
        # stat filled with 0 (see HANDOFF.md gotcha #6), which is "no data",
        # not "zero recent games of real data". Without bailing out here,
        # that lone placeholder row survives every filter below untouched
        # (there's no 'week' column to filter ON), producing a real-looking
        # "0.0 projected fantasy points" instead of the same "not enough
        # game log data yet" message this exact situation shows everywhere
        # else in the app (confirmed live: 2026 Mahomes showed a bogus 0.0
        # projection before this fix).
        return {}, 0.0
    df = df[pd.to_numeric(df['week'], errors='coerce').fillna(0) > 0].sort_values('week')
    if df.empty:
        return {}, 0.0
    recent = df.tail(4)

    proj = {}
    for stat in stats:
        if stat not in df.columns:
            continue
        season_avg = pd.to_numeric(df[stat], errors='coerce').fillna(0).mean()
        recent_avg = pd.to_numeric(recent[stat], errors='coerce').fillna(0).mean() if not recent.empty else season_avg
        base = 0.6 * recent_avg + 0.4 * season_avg

        matchup_mult = 1.0
        if allowed_matrix is not None and not allowed_matrix.empty and stat in allowed_matrix.columns:
            league_avg = allowed_matrix[stat].mean()
            if league_avg > 0:
                opp_allowed = allowed_matrix[stat].get(opponent_abbr, league_avg)
                matchup_mult = float(np.clip(opp_allowed / league_avg, 0.75, 1.3))

        pace_mult = 1.0
        if pace_df is not None and not pace_df.empty and 'def_pace' in pace_df.columns and opponent_abbr in pace_df.index:
            league_pace = pace_df['def_pace'].mean()
            opp_pace = pace_df.loc[opponent_abbr, 'def_pace']
            if league_pace > 0 and pd.notna(opp_pace):
                pace_mult = float(np.clip(opp_pace / league_pace, 0.85, 1.15))

        stat_alignment_mult = alignment_mult if stat in ('targets', 'receptions', 'receiving_yards', 'receiving_tds') else 1.0

        proj[stat] = round(base * matchup_mult * pace_mult * stat_alignment_mult, 2)

    return proj, score_projected_stats(proj, scoring_mode)


PROP_MARKET_TO_STAT = {
    'player_pass_yds': 'passing_yards',
    'player_pass_tds': 'passing_tds',
    'player_rush_yds': 'rushing_yards',
    'player_reception_yds': 'receiving_yards',
    'player_receptions': 'receptions',
}


def build_market_projection(props_data, player_name):
    """
    Extracts sportsbook player-prop lines as market-implied point estimates
    for whichever markets are actually posted (see fetch_nfl_player_props'
    docstring for why most won't be, far from kickoff): Over/Under markets
    (player_pass_yds etc.) contribute their posted line directly, averaged
    across books when more than one has it posted. player_anytime_td's
    "Yes" price is converted to an implied probability (American odds ->
    probability, data.utils.american_odds_to_prob) and surfaced separately
    as 'anytime_td_prob' - used directly as a TD-probability read rather
    than split between rushing/receiving TDs (this market doesn't
    distinguish which), and not converted through a Poisson expected-value
    transform (-ln(1-p)) - close enough at the probabilities anytime-TD
    markets typically trade at (mostly well under 50%) for this tool's
    purposes, and avoids modeling a full scoring distribution.
    """
    from data.utils import clean_name_for_merge, american_odds_to_prob
    if not props_data:
        return {}
    target_key = clean_name_for_merge(pd.Series([player_name])).iloc[0]
    lines = {}
    td_probs = []
    for book in props_data.get('bookmakers', []):
        for market in book.get('markets', []):
            stat = PROP_MARKET_TO_STAT.get(market.get('key'))
            is_td_market = market.get('key') == 'player_anytime_td'
            if not stat and not is_td_market:
                continue
            for o in market.get('outcomes', []):
                desc = o.get('description') or o.get('name') or ''
                if clean_name_for_merge(pd.Series([desc])).iloc[0] != target_key:
                    continue
                if stat and o.get('point') is not None:
                    lines.setdefault(stat, []).append(float(o['point']))
                elif is_td_market and str(o.get('name', '')).lower() == 'yes' and o.get('price') is not None:
                    td_probs.append(american_odds_to_prob(o['price']))
    market_stats = {stat: round(sum(vals) / len(vals), 1) for stat, vals in lines.items()}
    if td_probs:
        market_stats['anytime_td_prob'] = round(sum(td_probs) / len(td_probs), 3)
    return market_stats


@st.cache_data
def build_strength_of_schedule(_schedule_df, _def_matrix, year, remaining_only=True):
    """
    Rest-of-season strength of schedule: for each team, averages the
    fantasy-points-allowed-per-game (by position) of every opponent left on
    their schedule - the same def_matrix already built for the Defensive
    Yield heatmap, just re-aggregated from "defense's own row" to "every
    team's set of upcoming opponents' rows". Higher = softer remaining
    schedule (opponents give up more fantasy points at that position), so
    this reads the same direction as the heatmap it's built from.

    Falls back to the full season's games if remaining_only is requested but
    every game in _schedule_df is already played (e.g. viewing a completed
    prior season) - a 0-game remaining table would just be empty and useless.

    Both `_schedule_df`/`_def_matrix` are underscore-prefixed (not hashed,
    same convention as apply_scoring_and_percentiles) since `year` plus the
    small `remaining_only` bool are already a sufficient, cheap cache key.
    """
    if _schedule_df.empty or _def_matrix.empty:
        return pd.DataFrame()

    sched = _schedule_df.copy()
    if remaining_only:
        remaining = sched[sched['home_score'].isna()]
        if not remaining.empty:
            sched = remaining

    home_side = sched[['week', 'home_team', 'away_team']].rename(columns={'home_team': 'Team', 'away_team': 'Opponent'})
    away_side = sched[['week', 'home_team', 'away_team']].rename(columns={'away_team': 'Team', 'home_team': 'Opponent'})
    flat = pd.concat([home_side, away_side], ignore_index=True)

    pos_cols = [c for c in ['QB', 'RB', 'WR', 'TE'] if c in _def_matrix.columns]
    matrix = _def_matrix.set_index('Team')[pos_cols] if 'Team' in _def_matrix.columns else _def_matrix[pos_cols]

    flat = flat[flat['Opponent'].isin(matrix.index)]
    if flat.empty:
        return pd.DataFrame()

    joined = flat.join(matrix, on='Opponent')
    agg = joined.groupby('Team', observed=True).agg({**{c: 'mean' for c in pos_cols}, 'week': 'nunique'}).reset_index()
    agg = agg.rename(columns={'week': 'Games', **{c: f'{c} SOS' for c in pos_cols}})
    for c in pos_cols:
        agg[f'{c} SOS'] = agg[f'{c} SOS'].round(1)
    return agg.sort_values('Team')


@st.cache_data
def build_oline_protection_metrics(_pfr_pass_df, year):
    """
    Team pass-protection quality proxy: the team's primary starter (most
    pass attempts) PFR pressure_pct - there's no free per-lineman
    pressure-allowed export, but pressure allowed is overwhelmingly a
    function of the five guys in front of him, so this is a reasonable
    team-level stand-in. Lower pressure_pct = better protection.

    `year` is a real (non-underscore) cache key alongside the unhashed
    `_pfr_pass_df` - see build_points_allowed_matrix's docstring for why
    this matters (without it, every year would share one stale cache entry).
    """
    if _pfr_pass_df.empty or 'team' not in _pfr_pass_df.columns:
        return pd.DataFrame()
    df = _pfr_pass_df.sort_values('pass_attempts', ascending=False).drop_duplicates(subset=['team'])
    out = df[['team', 'player', 'pass_attempts', 'pressure_pct', 'times_pressured']].rename(columns={
        'team': 'Team', 'player': 'Starting QB', 'pass_attempts': 'Dropbacks', 'pressure_pct': 'Pressure Allowed %', 'times_pressured': 'Times Pressured',
    })
    return out.sort_values('Pressure Allowed %')


@st.cache_data
def build_def_pressure_metrics(_pfr_def_df, _pass_attempts_faced, year):
    """
    Defensive pressure rate generated: total pressures (PFR's `prss`, which
    already combines sacks/hurries/knockdowns into one count) summed by
    team, divided by that defense's total pass attempts faced (from
    nflverse team stats) - turns a raw counting stat into a rate so teams
    that played more snaps/games aren't overstated.

    `year` is a real (non-underscore) cache key - see
    build_points_allowed_matrix's docstring for why (both other args here
    are underscore-prefixed and would otherwise share one stale cache entry
    across every year).
    """
    if _pfr_def_df.empty or 'tm' not in _pfr_def_df.columns or _pass_attempts_faced.empty:
        return pd.DataFrame()
    agg = _pfr_def_df.groupby('tm', observed=True).agg(
        Pressures=('prss', 'sum'), Sacks=('sk', 'sum'), Hurries=('hrry', 'sum'), Knockdowns=('qbkd', 'sum'),
    ).reset_index().rename(columns={'tm': 'Team'})
    agg['Dropbacks Faced'] = agg['Team'].map(_pass_attempts_faced)
    agg = agg[agg['Dropbacks Faced'].notna() & (agg['Dropbacks Faced'] > 0)]
    agg['Pressure Rate %'] = (agg['Pressures'] / agg['Dropbacks Faced'] * 100).round(1)
    return agg[['Team', 'Pressures', 'Dropbacks Faced', 'Pressure Rate %', 'Sacks', 'Hurries', 'Knockdowns']].sort_values('Pressure Rate %', ascending=False)


@st.cache_data
def apply_scoring_and_percentiles(_stats, year, scoring_mode="Full PPR"):
    """
    Cheap, scoring-format-dependent step, split out of what used to be one
    fused load_and_merge_data(year, scoring_mode) function. Toggling the PPR
    dropdown now only re-runs this (measured ~0.09s on the real dataset)
    instead of the full load_year_data() pipeline it used to bust
    (measured ~1.5s / ~120MB peak) -- about 16x less work per toggle.
    Canonical column names (receiving_yards, receptions, etc.) already exist
    at this point courtesy of load_year_data(), so no alias fallback needed.

    IMPORTANT: the stats param is underscore-prefixed (`_stats`) so
    st.cache_data does NOT hash it - every tab calls this at least once per
    Streamlit rerun (Streamlit re-executes every tab's body on every widget
    interaction anywhere in the app, not just the visible tab), and hashing
    a ~20k-row / 190-col DataFrame on every single one of those calls was
    the dominant source of felt "choppiness" clicking around the app,
    independent of how fast the actual transform logic itself runs. `year` +
    `scoring_mode` (cheap primitives) are the real cache key instead -
    correct because this function's output is fully determined by which
    year's data was loaded and the scoring mode, never by object identity.
    """
    stats = _stats.copy()
    rec_mult = 1.0 if "Full" in scoring_mode else (0.5 if "Half" in scoring_mode else 0.0)

    # IDP (defensive) scoring - this was a pre-existing gap (confirmed in
    # the original uploaded file too): fantasy_points never included
    # tackles/sacks/INTs/forced fumbles, so every defensive player scored
    # 0 regardless of real production. Values below are a commonly-cited
    # "medium" IDP default (1/tackle, 2/sack, 4/INT, 2/forced fumble) -
    # scoring conventions vary a lot league to league, so treat these as a
    # starting point to check against your actual league settings, not a
    # verified match to them.
    #
    # Kicker scoring - the SAME class of gap, confirmed the same way: PFF/
    # nflverse's own fantasy_points/fantasy_points_ppr columns are offense-
    # only and never include kicking at all, so every kicker showed 0.0
    # fantasy points every single week regardless of real makes (confirmed
    # live: Harrison Butker, 2025, 3/3 FG + 0/1 PAT in Week 1 alone, still
    # showed 0.0). Standard distance-tiered make values (3/4/5 pts for
    # under 40 / 40-49 / 50+ yards, 1 pt per PAT) using the same distance-
    # bucketed fg_made_X_Y columns PFF's own kicker export breaks out
    # elsewhere in this app - not a verified match to any specific league's
    # actual kicker scoring either, same caveat as the IDP values above.
    fg_short = stats.get('fg_made_0_19', 0) + stats.get('fg_made_20_29', 0) + stats.get('fg_made_30_39', 0)
    fg_mid = stats.get('fg_made_40_49', 0)
    fg_long = stats.get('fg_made_50_59', 0) + stats.get('fg_made_60_', 0)
    stats['fantasy_points'] = (
        (stats['rushing_yards'] * 0.1) +
        (stats['rushing_tds'] * 6) +
        (stats['receiving_yards'] * 0.1) +
        (stats['receiving_tds'] * 6) +
        (stats['receptions'] * rec_mult) +
        (stats['passing_yards'] * 0.04) +
        (stats['passing_tds'] * 4) -
        (stats['passing_interceptions'] * 2) +
        (stats['tackles'] * 1.0) +
        (stats['sacks'] * 2.0) +
        (stats['def_ints'] * 4.0) +
        (stats['forced_fumbles'] * 2.0) +
        (fg_short * 3) + (fg_mid * 4) + (fg_long * 5) +
        (stats.get('pat_made', 0) * 1.0)
    )

    # Calculate Weekly Game Log Percentiles for Heatmapping - ranked against
    # a same-week, snap-qualified reference pool (calculate_percentile_
    # qualified) rather than every rostered name at the position that week,
    # so a limited-snap/garbage-time row doesn't drag the comparison pool
    # down for everyone else that week. See data.utils.
    # MIN_SNAP_PCT_FOR_PERCENTILE for the per-position bars.
    stat_cols_for_color = ['fantasy_points', 'passing_yards', 'passing_tds', 'passing_attempts', 'passing_completions',
                           'rushing_yards', 'rushing_tds', 'rushing_attempts', 'ypc',
                           'receptions', 'receiving_yards', 'receiving_tds', 'targets', 'y/c',
                           'tackles', 'sacks', 'def_ints', 'forced_fumbles', 'weekly_snap_pct']
    for c in stat_cols_for_color:
        if c in stats.columns:
            stats[f"{c}_wk_pct"] = calculate_percentile_qualified(
                stats, c, position_col='position', snap_col='weekly_snap_pct', ascending=True
            )

    return stats


def load_and_merge_data(year, scoring_mode="Full PPR"):
    """
    Thin wrapper kept so every existing call site is unchanged. Splits the
    expensive year-load (cached on year) from the cheap scoring-format math
    (cached on year+scoring_mode, but far cheaper to recompute on a miss) --
    see data/loaders.load_year_data() and apply_scoring_and_percentiles()
    above for the reasoning and numbers.
    """
    from data.loaders import load_year_data
    stats, team_col, name_col, global_rookie_names = load_year_data(year)
    stats = apply_scoring_and_percentiles(stats, year, scoring_mode)
    return stats, team_col, name_col, global_rookie_names


@st.cache_data
def precompute_league_percentiles(_df, name_field, year):
    """
    `_df` is underscore-prefixed (not hashed) - see apply_scoring_and_percentiles
    above for why; `year` (cheap) is the real cache key alongside name_field.
    """
    df_calc = _df.copy()
    if df_calc.empty or name_field not in df_calc.columns: return pd.DataFrame()

    # A year with no weekly stats file yet (e.g. an upcoming season before
    # games are played) has no 'week' column at all in the merged frame -
    # load_year_data's roster-only outer-join fallback has nothing to put
    # there. Default games_played to 1 for everyone rather than crashing;
    # the per-game rate columns below (targets_per_g etc.) still make sense
    # since every raw stat is 0 pre-season anyway.
    if 'week' in df_calc.columns:
        gp = df_calc.groupby([name_field, 'position'], observed=True)['week'].nunique().reset_index().rename(columns={'week': 'games_played'})
    else:
        gp = df_calc[[name_field, 'position']].drop_duplicates().copy()
        gp['games_played'] = 1

    summary = df_calc.groupby([name_field, 'position'], observed=True).agg({
        'fantasy_points': ['sum', 'mean'],
        'passing_yards': 'mean',
        'passing_tds': 'sum',
        'passing_interceptions': 'sum',
        'targets': 'sum',
        'receptions': 'sum',
        'receiving_yards': 'mean',
        'receiving_tds': 'sum',
        'rushing_yards': 'sum',
        'rushing_attempts': 'sum',
        'rushing_tds': 'sum',
        'tackles': 'sum',
        'sacks': 'sum',
        'def_ints': 'sum',
        'forced_fumbles': 'sum',
        'snap_pct_avg': 'max'
    })
    summary.columns = [f"{c[0]}_{c[1]}" for c in summary.columns]
    summary = summary.reset_index()

    summary = pd.merge(summary, gp, on=[name_field, 'position'], how='left')
    summary['games_played'] = summary['games_played'].replace(0, 1)

    summary['targets_per_g'] = summary['targets_sum'] / summary['games_played']
    summary['rec_per_g'] = summary['receptions_sum'] / summary['games_played']
    summary['rush_att_per_g'] = summary['rushing_attempts_sum'] / summary['games_played']
    summary['ypc'] = summary['rushing_yards_sum'] / summary['rushing_attempts_sum'].replace(0, 1)
    summary['tackles_per_g'] = summary['tackles_sum'] / summary['games_played']

    # EPA/success-rate efficiency (see build_epa_efficiency's docstring) -
    # merged in here, right before the percentile loop below, so its columns
    # get percentile-ranked for free by that same loop instead of needing
    # their own separate percentile computation. Plain equality join on
    # name_field is safe (not the fuzzy clean_name_exact/for_merge matching
    # used for cross-source joins elsewhere) since both sides already derive
    # from the identical _stats_df[name_col] strings within this one
    # pipeline stage - there's no name-format mismatch to bridge here.
    epa_df = build_epa_efficiency(df_calc, name_field, year)
    if not epa_df.empty:
        summary = pd.merge(summary, epa_df, on=name_field, how='left')

    # Ranked against a snap-qualified reference pool (calculate_percentile_
    # qualified), not every rostered name at the position - a raw groupby
    # rank() here let a pile of low-snap bench/committee rows drag a whole
    # position's comparison pool down, which is exactly what was inflating
    # a real starter's percentile (confirmed real: a clear WR1 reading as
    # 94th-percentile fantasy PPG purely because dozens of fringe receivers
    # who barely play were sitting in the same denominator). See
    # data.utils.MIN_SNAP_PCT_FOR_PERCENTILE for the per-position bars.
    metrics_to_rank = [c for c in summary.columns if c not in [name_field, 'position', 'games_played']]
    for m in metrics_to_rank:
        ascending = 'interceptions' not in m
        summary[f"{m}_pct"] = calculate_percentile_qualified(
            summary, m, position_col='position', snap_col='snap_pct_avg_max', ascending=ascending
        )
    return summary


def build_recent_trend(_stats_df, name_col, metric='fantasy_points', n_weeks=5):
    """
    Per-player list of their last n_weeks games' `metric` value, oldest to
    newest - feeds an inline sparkline column (st.column_config.
    LineChartColumn/BarChartColumn) so a trend is visible directly in a
    table row without opening the full game log. Not cache_data-decorated:
    this is a groupby over an already-loaded (and separately cached)
    dataframe, cheap enough on its own that adding a second cache layer
    would mostly just be a fresh set of cache-key bugs to get right for no
    real benefit - see build_points_allowed_matrix's docstring elsewhere in
    this file for what happens when that's gotten wrong before.
    """
    if _stats_df.empty or 'week' not in _stats_df.columns or metric not in _stats_df.columns:
        return {}
    df = _stats_df[pd.to_numeric(_stats_df['week'], errors='coerce').fillna(0) > 0].copy()
    df['week'] = pd.to_numeric(df['week'], errors='coerce')
    df = df.sort_values('week')
    trend = {}
    for name, g in df.groupby(name_col, observed=True):
        vals = pd.to_numeric(g[metric], errors='coerce').fillna(0).tail(n_weeks).tolist()
        if vals:
            trend[name] = vals
    return trend


def build_form_series(_stats_df, name_col, metric='fantasy_points', n_weeks=5, before_week=None):
    """
    Per player: (last n_weeks values oldest->newest, SEASON average over
    every game in scope, the week numbers behind those values).

    The richer sibling of build_recent_trend, for a sparkline that draws a
    reference line as well as a trend - the season average has to come from
    the same filtered population the trend does, or the dotted line sits at
    a level the visible points were never measured against.

    `before_week` gates BOTH halves to games strictly earlier than that week
    - the whole point of a "recent form" column on a WEEK N projection table
    is what was known going into week N. Without it, selecting an early week
    of a completed season shows that player's last five games of the SEASON
    (December form next to a week-4 projection), which is not a subtle
    inaccuracy: it is the wrong five games entirely.

    Returns {} for a frame with no weekly rows at all (a roster-only season,
    HANDOFF.md gotcha #6) rather than a dict of empty series, so a caller
    can tell "no weekly data exists" from "this player has none".
    """
    if _stats_df.empty or 'week' not in _stats_df.columns or metric not in _stats_df.columns:
        return {}
    weeks = pd.to_numeric(_stats_df['week'], errors='coerce')
    keep = weeks.fillna(0) > 0
    if before_week is not None:
        keep &= weeks < before_week
    df = _stats_df[keep].copy()
    if df.empty:
        return {}
    df['week'] = pd.to_numeric(df['week'], errors='coerce')
    df['_val'] = pd.to_numeric(df[metric], errors='coerce').fillna(0.0)
    df = df.sort_values('week')

    out = {}
    for name, g in df.groupby(name_col, observed=True):
        vals = g['_val'].tolist()
        if not vals:
            continue
        out[name] = (g['_val'].tail(n_weeks).tolist(),
                     float(sum(vals) / len(vals)),
                     g['week'].tail(n_weeks).tolist())
    return out


def build_recent_form_rank(_stats_df, name_col, team_col, n_weeks=4):
    """
    Recent-form fantasy output per player - average fantasy points over
    just their last n_weeks games played, not a season-long or
    17-game-extrapolated number. This is the internal baseline the Weekly
    Rankings tab compares an uploaded weekly ranking against, since a
    WEEKLY ranking answers "who's producing right now" - a season-long
    draft-value projection (Draft HQ's board) answers a different question
    and would be the wrong yardstick for a week-to-week comparison.

    Offense + kickers only, same as build_rookie_watch - a standard non-IDP
    league doesn't roster defenders or O-linemen.
    Requires at least 2 games played, a lighter bar than the 4 a
    season-long projection needs - this isn't extrapolating a rate to a
    full season, so a 2-game recent sample is a real, if noisy, signal
    rather than the kind
    of wild single-game outlier VORP's stricter guard exists to avoid.

    Not cache_data-decorated - see build_recent_trend's docstring for why
    a cheap groupby over an already-cached frame doesn't need its own cache
    layer.
    """
    if _stats_df.empty or 'week' not in _stats_df.columns:
        return pd.DataFrame()
    df = _stats_df[pd.to_numeric(_stats_df['week'], errors='coerce').fillna(0) > 0].copy()
    from config import OFFENSE_SKILL_POSITIONS
    if 'position' in df.columns:
        df = df[df['position'].isin(OFFENSE_SKILL_POSITIONS)]
    if df.empty:
        return pd.DataFrame()
    df['week'] = pd.to_numeric(df['week'], errors='coerce')
    df = df.sort_values('week')

    agg_spec = {
        'Recent Avg FPTS': ('fantasy_points', lambda s: round(s.tail(n_weeks).mean(), 1)),
        'Games': ('week', 'nunique'),
        'Pos': ('position', 'first'),
    }
    if team_col in df.columns:
        agg_spec['Team'] = (team_col, 'first')
    summary = df.groupby(name_col, observed=True).agg(**agg_spec).reset_index().rename(columns={name_col: 'Player'})

    MIN_GAMES = 2
    summary = summary[summary['Games'] >= MIN_GAMES]
    if summary.empty:
        return pd.DataFrame()
    for c in ['Player', 'Team', 'Pos']:
        if c in summary.columns:
            summary[c] = summary[c].astype(str)
    summary = summary[summary['Player'].str.strip().ne('') & summary['Player'].ne('0')]
    summary = summary.drop_duplicates(subset=['Player'])
    return summary.sort_values('Recent Avg FPTS', ascending=False).reset_index(drop=True)


@st.cache_data
def build_risers_report(_stats_df, name_col, team_col, year, min_weeks=2, top_n=25):
    """
    Waiver-wire "risers" - players whose most recent game was a big
    percentile jump over their season-to-date average. Built almost
    entirely from data load_year_data() already computes (the *_wk_pct
    columns from apply_scoring_and_percentiles), so this is a thin layer
    on top rather than a new stats pipeline.

    `_stats_df` underscore-prefixed (not hashed) - see
    apply_scoring_and_percentiles for why; `year` is the real cache key.
    """
    if _stats_df.empty or 'week' not in _stats_df.columns or 'fantasy_points_wk_pct' not in _stats_df.columns:
        return pd.DataFrame()

    df = _stats_df[pd.to_numeric(_stats_df['week'], errors='coerce').fillna(0) > 0].copy()
    # Offense + kickers only - a standard non-IDP waiver wire doesn't
    # include defensive players or O-linemen.
    from config import OFFENSE_SKILL_POSITIONS
    if 'position' in df.columns:
        df = df[df['position'].isin(OFFENSE_SKILL_POSITIONS)]
    if df.empty:
        return pd.DataFrame()

    latest_week = df['week'].max()
    prior = df[df['week'] < latest_week]
    latest = df[df['week'] == latest_week]
    if prior.empty or latest.empty:
        return pd.DataFrame()

    prior_agg = prior.groupby(name_col, observed=True).agg(
        prior_avg_pct=('fantasy_points_wk_pct', 'mean'),
        weeks_played=('week', 'nunique'),
        position=('position', 'first'),
    ).reset_index()
    prior_agg = prior_agg[prior_agg['weeks_played'] >= min_weeks]

    latest_slim = latest[[name_col, team_col, 'fantasy_points_wk_pct', 'fantasy_points', 'snap_pct_avg']].rename(
        columns={'fantasy_points_wk_pct': 'latest_pct', 'fantasy_points': 'latest_fp'})

    merged = pd.merge(latest_slim, prior_agg, on=name_col, how='inner')
    if merged.empty:
        return pd.DataFrame()

    merged['Pct Jump'] = (merged['latest_pct'] - merged['prior_avg_pct']).round(1)
    merged = merged.sort_values('Pct Jump', ascending=False).head(top_n)
    merged = merged.rename(columns={
        name_col: 'Player', team_col: 'Team', 'position': 'Pos', 'latest_fp': 'Latest FPTS',
        'snap_pct_avg': 'Season Snap %', 'weeks_played': 'Weeks Played',
    })
    merged['Latest FPTS'] = merged['Latest FPTS'].round(1)
    merged['Season Snap %'] = merged['Season Snap %'].round(1)
    merged['Latest Week'] = int(latest_week)
    out = merged[['Player', 'Team', 'Pos', 'Latest Week', 'Latest FPTS', 'Pct Jump', 'Season Snap %', 'Weeks Played']].copy()
    # Team/Pos come from groupby('first') on columns that are categorical
    # dtype upstream (kept categorical there deliberately - see
    # load_year_data). Cast to plain str here since this frame is meant for
    # st.dataframe display, and a categorical that's passed through this
    # many groupby/merge steps can end up in a state pyarrow's Arrow
    # conversion (which st.dataframe uses) refuses to serialize.
    for c in ['Player', 'Team', 'Pos']:
        out[c] = out[c].astype(str)
    out = out[out['Player'].str.strip().ne('') & out['Player'].ne('0')]
    # A table like this should never have two rows for the same player -
    # this is what actually broke st.dataframe (Styler requires a unique
    # index once you .set_index('Player')). Root-caused and fixed upstream
    # (see the two-tier name matching in load_year_data), but keeping this
    # as a safety net since any duplicate here is always a bug, never valid
    # data - drop_duplicates keeps the first (already sorted by Pct Jump).
    out = out.drop_duplicates(subset=['Player'])
    # NGS Rush Yards Over Expected deliberately NOT merged in here -
    # confirmed live (real 2025 data, real RB rows in this exact report)
    # that it rendered blank for every player: unlike almost every other
    # source in this app it had no local-file fallback, and its live
    # nflreadpy.load_nextgen_stats() pull is failing. Red-zone carry share
    # (_merge_redzone_share, off play-by-play) covers the same "is he
    # actually valuable per touch" question from a source that works.
    return out


@st.cache_data
def build_rookie_watch(_stats_df, name_col, team_col, global_rookie_names, year):
    """
    Rookie leaderboard using global_rookie_names (now that its underlying
    rookie_year comparison bug is fixed) plus draft capital already pulled
    into bio_cols (draft_number/draft_club/entry_year), ranked by average
    fantasy output. One row per rookie for the season, not per week.

    `_stats_df` underscore-prefixed (not hashed) - see
    apply_scoring_and_percentiles for why; `year` is the real cache key.
    """
    if _stats_df.empty or not global_rookie_names:
        return pd.DataFrame()

    df = _stats_df.copy()
    # Offense + kickers only - a standard non-IDP league doesn't roster
    # rookie defenders or O-linemen off a rookie-watch board.
    from config import OFFENSE_SKILL_POSITIONS
    if 'position' in df.columns:
        df = df[df['position'].isin(OFFENSE_SKILL_POSITIONS)]
    df['is_rookie'] = df[name_col].astype(str).str.lower().isin(global_rookie_names)
    rookies = df[df['is_rookie']]
    if rookies.empty:
        return pd.DataFrame()

    agg_spec = {
        'fantasy_points': 'mean', 'snap_pct_avg': 'max', 'position': 'first', team_col: 'first',
    }
    # 'draft_club' deliberately excluded - it's always the same as team_col
    # (a rookie's team IS who drafted them, save the ~1% traded-on-draft-day
    # exception), so a separate "Drafted By" column next to "Team" was
    # redundant almost every time.
    for opt_col in ['targets', 'receiving_yards', 'rushing_yards', 'draft_number']:
        if opt_col in rookies.columns:
            agg_spec[opt_col] = 'first' if opt_col == 'draft_number' else 'sum'
    if 'week' in rookies.columns:
        agg_spec['week'] = 'nunique'

    summary = rookies.groupby(name_col, observed=True).agg(agg_spec).reset_index()
    rename_map = {
        name_col: 'Player', team_col: 'Team', 'position': 'Pos', 'fantasy_points': 'Avg FPTS',
        'snap_pct_avg': 'Snap %', 'targets': 'Targets', 'receiving_yards': 'Rec Yds',
        'rushing_yards': 'Rush Yds', 'week': 'Games', 'draft_number': 'Draft Pick',
    }
    summary = summary.rename(columns={k: v for k, v in rename_map.items() if k in summary.columns})
    if 'Avg FPTS' in summary.columns:
        summary['Avg FPTS'] = summary['Avg FPTS'].round(1)
        summary = summary.sort_values('Avg FPTS', ascending=False)
    if 'Snap %' in summary.columns:
        summary['Snap %'] = summary['Snap %'].round(1)
    if 'Draft Pick' in summary.columns:
        draft_pick_num = pd.to_numeric(summary['Draft Pick'], errors='coerce')
        is_drafted = draft_pick_num.notna() & (draft_pick_num > 0)
        # Same sentinel-sort fix as FantasyPros Rank (data.rankings.
        # build_rankings_comparison) - undrafted rookies need a real number
        # worse than the highest actual pick, not NaN/None, so an ascending
        # sort still puts pick #1 on top and undrafted players land below
        # Mr. Irrelevant instead of wherever glide-data-grid's NaN handling
        # happens to scatter them (confirmed unreliable there before).
        sentinel = (draft_pick_num[is_drafted].max() + 1) if is_drafted.any() else 300
        summary['Draft Pick'] = draft_pick_num.where(is_drafted, sentinel).astype(int)
    for c in ['Player', 'Team', 'Pos']:
        if c in summary.columns:
            summary[c] = summary[c].astype(str)
    if 'Player' in summary.columns:
        summary = summary[summary['Player'].str.strip().ne('') & summary['Player'].ne('0')]
        summary = summary.drop_duplicates(subset=['Player'])
        # RZ Tgt Share %/RZ Rush Share % (build_redzone_usage, from raw
        # play-by-play) replace RYOE here specifically - RYOE is RB-only
        # and a rookie board is dominated by WRs, so it read as an
        # always-blank column for most rows. Red-zone share is meaningful
        # for both: target share for WR/TE, carry share for RB.
        summary = _merge_redzone_share(summary, _stats_df, name_col, year, out_name_col='Player', pos_col='Pos')
        # Explicit column order (there wasn't one before - display order
        # just fell out of agg_spec's dict-insertion order) - Player, Pos,
        # Team, Draft Pick lead, then Avg FPTS, then everything else
        # (Snap %, Targets, Rec/Rush Yds, Games, RZ share) keeps its
        # existing relative order. Filtered against summary.columns since
        # several of these are conditional on what _stats_df happens to
        # carry (see the opt_col loop above).
        lead_cols = [c for c in ['Player', 'Pos', 'Team', 'Draft Pick', 'Avg FPTS'] if c in summary.columns]
        summary = summary[lead_cols + [c for c in summary.columns if c not in lead_cols]]
    return summary


def parse_pasted_draft_picks(raw_text):
    """
    Best-effort parser for pasted draft results (Yahoo/ESPN draft pages, or
    just a plain list). Doesn't need to be perfect - match_names_to_board()
    does fuzzy two-tier matching against the actual board afterward, so this
    just needs to produce reasonably clean candidate strings. Handles:
    numbered lines ("1. Player Name"), round/pick prefixes ("Round 3, Pick
    5 - Player Name"), trailing "(POS - TEAM)" parentheticals, tab-separated
    paste from an HTML table (keeps the first cell), and comma-separated
    single-line lists.
    """
    if not raw_text or not raw_text.strip():
        return []
    candidates = []
    for line in raw_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        has_parens_tag = bool(re.search(r'\([A-Za-z]{1,3}\s*[-,]?\s*[A-Za-z]{2,3}\)', line))
        parts = [line] if has_parens_tag or len(line.split(',')) <= 1 else line.split(',')
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if '\t' in part:
                part = part.split('\t')[0].strip()
            part = re.sub(r'^(round\s*\d+[,.]?\s*pick\s*\d+\s*[-:]?\s*)', '', part, flags=re.IGNORECASE)
            part = re.sub(r'^\d+[\.\)]\s*', '', part)
            part = re.sub(r'\s*\([^)]*\)\s*$', '', part)
            part = part.strip(' -:\t')
            if part and len(part) > 1 and not part.replace('.', '').isdigit():
                candidates.append(part)
    return candidates


def match_names_to_board(candidate_names, board_players):
    """
    Two-tier match (same approach used for the snap-count name fix earlier)
    so minor formatting differences between an external source's names and
    this app's naming don't cause silent misses - tries an exact match
    first, falls back to the suffix-stripped version. Returns
    (matched_board_names, unmatched_candidate_names) so callers can report
    both without re-deriving "unmatched" via string comparison, which would
    incorrectly flag a successful fuzzy match (e.g. "AJ Terrell Jr." ->
    "A.J. Terrell") as unmatched since the strings differ.
    """
    board_exact = {clean_name_exact(pd.Series([p])).iloc[0]: p for p in board_players}
    board_loose = {clean_name_for_merge(pd.Series([p])).iloc[0]: p for p in board_players}
    matched, unmatched = [], []
    for name in candidate_names:
        if not name or not name.strip():
            continue
        exact_key = clean_name_exact(pd.Series([name])).iloc[0]
        if exact_key in board_exact:
            matched.append(board_exact[exact_key])
            continue
        loose_key = clean_name_for_merge(pd.Series([name])).iloc[0]
        if loose_key in board_loose:
            matched.append(board_loose[loose_key])
            continue
        unmatched.append(name)
    return matched, unmatched


@st.cache_data
def build_player_historical_summary(selected_player, scoring_rule, years=(2023, 2024, 2025)):
    """
    Season-by-season historical totals for one player across a fixed
    2023-2025 window. Deliberately NOT tied to whatever year is selected
    elsewhere in the app (the old version anchored off t1_target_year, which
    meant selecting 2026 in the Player Search tab pulled 2026 into a
    "3-year historical" table that's supposed to mean completed seasons
    only). Shared by the Player Search tab and (in the original single-file
    version) a since-removed duplicate view, so the numbers can't drift
    apart between the two.
    """
    from data.loaders import load_weekly_stats_history
    df_hist = load_weekly_stats_history()
    if df_hist.empty or not selected_player:
        return pd.DataFrame()

    df_hist = df_hist.copy()
    df_hist['clean_merge_name'] = clean_name_for_merge(df_hist.get('player_display_name', df_hist.get('player_name', df_hist.get('full_name'))))
    clean_target = clean_name_for_merge(pd.Series([selected_player])).iloc[0]
    p_hist = df_hist[(df_hist['clean_merge_name'] == clean_target) & (df_hist['season'].isin(list(years)))].copy()
    if p_hist.empty or 'season' not in p_hist.columns:
        return pd.DataFrame()

    # Sort so a per-season 'last' aggregation below means "team as of the
    # most recent week that season" - correct for a player traded mid-season
    # (shows where they ended up, not an arbitrary row-order artifact from
    # the raw concatenated weekly CSVs, which aren't stored in week order).
    if 'week' in p_hist.columns:
        p_hist = p_hist.sort_values(['season', 'week'])

    def h_col(df, cols):
        for c in cols:
            if c in df.columns: return df[c].fillna(0)
        return pd.Series([0]*len(df), index=df.index)

    rec_mult = 1.0 if "Full" in scoring_rule else (0.5 if "Half" in scoring_rule else 0.0)
    p_hist['calc_pass_yds'] = h_col(p_hist, ['passing_yards'])
    p_hist['calc_pass_tds'] = h_col(p_hist, ['passing_tds'])
    p_hist['calc_rush_att'] = h_col(p_hist, ['carries', 'rushing_attempts'])
    p_hist['calc_rush_yds'] = h_col(p_hist, ['rushing_yards'])
    p_hist['calc_rush_tds'] = h_col(p_hist, ['rushing_tds'])
    p_hist['calc_targets'] = h_col(p_hist, ['targets'])
    p_hist['calc_rec'] = h_col(p_hist, ['receptions'])
    p_hist['calc_rec_yds'] = h_col(p_hist, ['receiving_yards'])
    p_hist['calc_rec_tds'] = h_col(p_hist, ['receiving_tds'])
    # Same nflverse def_* column names fixed in load_year_data - a player's
    # historical totals should use the same source columns as their live
    # game log, not the old broken 'tackles'/'sacks'/'interceptions' aliases.
    if 'def_tackles_solo' in p_hist.columns or 'def_tackles_with_assist' in p_hist.columns:
        p_hist['calc_tackles'] = h_col(p_hist, ['def_tackles_solo']) + h_col(p_hist, ['def_tackles_with_assist'])
    else:
        p_hist['calc_tackles'] = h_col(p_hist, ['tackles'])
    p_hist['calc_sacks'] = h_col(p_hist, ['def_sacks', 'sacks'])
    p_hist['calc_def_ints'] = h_col(p_hist, ['def_interceptions', 'interceptions'])
    p_hist['calc_ff'] = h_col(p_hist, ['def_fumbles_forced', 'forced_fumbles'])
    # Distance-tiered FG scoring, same buckets apply_scoring_and_percentiles
    # uses for the live weekly fantasy_points column - folded into the same
    # shared calc_fp formula below rather than a kicker-only branch, since
    # these columns are naturally 0 for every non-kicker anyway (nothing to
    # gate on).
    p_hist['calc_fg_made'] = h_col(p_hist, ['fg_made'])
    p_hist['calc_fg_att'] = h_col(p_hist, ['fg_att'])
    p_hist['calc_fg_short'] = h_col(p_hist, ['fg_made_0_19']) + h_col(p_hist, ['fg_made_20_29']) + h_col(p_hist, ['fg_made_30_39'])
    p_hist['calc_fg_mid'] = h_col(p_hist, ['fg_made_40_49'])
    p_hist['calc_fg_long'] = h_col(p_hist, ['fg_made_50_59']) + h_col(p_hist, ['fg_made_60_'])
    p_hist['calc_pat_made'] = h_col(p_hist, ['pat_made'])
    p_hist['calc_pat_att'] = h_col(p_hist, ['pat_att'])
    # Same IDP + kicker scoring as apply_scoring_and_percentiles - kept in
    # sync so a player's historical totals and their live game log always
    # agree (this table previously left kicker scoring out entirely, so
    # every kicker's "Fantasy Pts" here read as 0 regardless of real makes).
    p_hist['calc_fp'] = (
        (p_hist['calc_rush_yds'] * 0.1) + (p_hist['calc_rush_tds'] * 6) +
        (p_hist['calc_rec_yds'] * 0.1) + (p_hist['calc_rec_tds'] * 6) +
        (p_hist['calc_rec'] * rec_mult) + (p_hist['calc_pass_yds'] * 0.04) +
        (p_hist['calc_pass_tds'] * 4) - (h_col(p_hist, ['passing_interceptions']) * 2) +
        (p_hist['calc_tackles'] * 1.0) + (p_hist['calc_sacks'] * 2.0) +
        (p_hist['calc_def_ints'] * 4.0) + (p_hist['calc_ff'] * 2.0) +
        (p_hist['calc_fg_short'] * 3) + (p_hist['calc_fg_mid'] * 4) + (p_hist['calc_fg_long'] * 5) +
        (p_hist['calc_pat_made'] * 1.0)
    )

    has_team = 'team' in p_hist.columns

    # Position pulled from p_hist (the weekly nflverse history already
    # loaded above), most recent row, since that's available for every
    # season this table spans regardless of PFF data completeness. Used
    # below to branch which counting stats actually apply to this player -
    # a receiver has no business seeing Pass Yds/Pass TDs (previously every
    # non-defensive position saw the exact same full offensive column set,
    # which is where a WR's passing columns - always empty/irrelevant - was
    # coming from), same idea already established for the live Game Log's
    # own position branching just above in this file's UI counterpart.
    player_position = None
    if 'position' in p_hist.columns:
        pos_vals = p_hist['position'].dropna()
        if not pos_vals.empty:
            player_position = str(pos_vals.iloc[-1]).upper()

    is_defensive = (
        (p_hist['calc_tackles'].sum() + p_hist['calc_sacks'].sum() + p_hist['calc_def_ints'].sum()) > 0
        and p_hist['calc_pass_yds'].sum() == 0 and p_hist['calc_rush_yds'].sum() == 0 and p_hist['calc_rec_yds'].sum() == 0
        # player_position != 'K' guards a real, confirmed false-positive:
        # kickers occasionally register a genuine special-teams tackle
        # (last line of defense on a kickoff/punt return), which alone
        # satisfies the tackle-sum heuristic above and - with pass/rush/rec
        # yards correctly at 0 for a kicker too - was misclassifying
        # Harrison Butker's career totals as a defensive player's (Tackles/
        # Sacks/Ints/FF instead of FG/PAT) the moment he had even one.
        and player_position != 'K'
    )

    from config import OLINE_POSITIONS
    if is_defensive:
        agg_spec = dict(
            Games_Played=('week', 'nunique'), Tackles=('calc_tackles', 'sum'),
            Sacks=('calc_sacks', 'sum'), Ints=('calc_def_ints', 'sum'), FF=('calc_ff', 'sum'),
            Fantasy_Pts=('calc_fp', 'sum'),
        )
    elif player_position == 'QB':
        agg_spec = dict(
            Games_Played=('week', 'nunique'), Pass_Yds=('calc_pass_yds', 'sum'),
            Pass_TDs=('calc_pass_tds', 'sum'), Rush_Att=('calc_rush_att', 'sum'),
            Rush_Yds=('calc_rush_yds', 'sum'), Rush_TDs=('calc_rush_tds', 'sum'),
            Fantasy_Pts=('calc_fp', 'sum'),
        )
    elif player_position in ('WR', 'TE'):
        agg_spec = dict(
            Games_Played=('week', 'nunique'), Targets=('calc_targets', 'sum'),
            Receptions=('calc_rec', 'sum'), Rec_Yds=('calc_rec_yds', 'sum'), Rec_TDs=('calc_rec_tds', 'sum'),
            Rush_Yds=('calc_rush_yds', 'sum'), Rush_TDs=('calc_rush_tds', 'sum'),
            Fantasy_Pts=('calc_fp', 'sum'),
        )
    elif player_position in ('RB', 'FB'):
        agg_spec = dict(
            Games_Played=('week', 'nunique'), Rush_Att=('calc_rush_att', 'sum'),
            Rush_Yds=('calc_rush_yds', 'sum'), Rush_TDs=('calc_rush_tds', 'sum'),
            Targets=('calc_targets', 'sum'), Receptions=('calc_rec', 'sum'),
            Rec_Yds=('calc_rec_yds', 'sum'), Rec_TDs=('calc_rec_tds', 'sum'),
            Fantasy_Pts=('calc_fp', 'sum'),
        )
    elif player_position == 'K':
        agg_spec = dict(
            Games_Played=('week', 'nunique'), FG_Made=('calc_fg_made', 'sum'),
            FG_Att=('calc_fg_att', 'sum'), PAT_Made=('calc_pat_made', 'sum'),
            PAT_Att=('calc_pat_att', 'sum'), Fantasy_Pts=('calc_fp', 'sum'),
        )
    elif player_position in OLINE_POSITIONS:
        # No offensive counting stat here is meaningful for a lineman - this
        # weekly nflverse export doesn't carry PFF-style pass-block/run-
        # block snaps, just the skill-position box score - so Games Played
        # (+ Team, added below) is genuinely all there is to show.
        agg_spec = dict(Games_Played=('week', 'nunique'))
    else:
        # Unknown/unmapped position (missing from this weekly source for
        # every row this player has) - falls back to the full offensive
        # spec rather than guessing wrong at a position this branch doesn't
        # recognize; worst case shows a stat that happens to be 0 instead of
        # hiding a real one.
        agg_spec = dict(
            Games_Played=('week', 'nunique'), Pass_Yds=('calc_pass_yds', 'sum'),
            Pass_TDs=('calc_pass_tds', 'sum'), Rush_Att=('calc_rush_att', 'sum'),
            Rush_Yds=('calc_rush_yds', 'sum'), Rush_TDs=('calc_rush_tds', 'sum'),
            Targets=('calc_targets', 'sum'), Receptions=('calc_rec', 'sum'),
            Rec_Yds=('calc_rec_yds', 'sum'), Rec_TDs=('calc_rec_tds', 'sum'),
            Fantasy_Pts=('calc_fp', 'sum'),
        )
    if has_team:
        agg_spec['Team'] = ('team', 'last')
    hist_grouped = p_hist.groupby('season').agg(**agg_spec).reset_index().sort_values('season', ascending=False)

    hist_grouped.columns = [c.replace('_', ' ') for c in hist_grouped.columns]
    if has_team:
        # Right after 'season' rather than trailing at the end - which team
        # a player was on is identifying context for the row, not just
        # another stat, so it reads more naturally leading the row.
        cols = hist_grouped.columns.tolist()
        cols.insert(1, cols.pop(cols.index('Team')))
        hist_grouped = hist_grouped[cols]

    # Overall PFF grade + Snap %/Slot %/Wide %/Inline % per season - so a
    # role/usage change over time (a slot receiver moving outside, a
    # committee back becoming the every-down guy) is visible right next to
    # the counting-stat trend above, not just the raw box score. Genuinely
    # separate data sources from the weekly nflverse history this table is
    # otherwise built from (PFF's own per-year exports for grade/alignment,
    # load_year_data's snap merge for Snap %), so this loops one season at
    # a time rather than reusing p_hist - neither source has a ready-made
    # multi-year history the way load_weekly_stats_history does.
    #
    # load_all_pff_data (NOT load_pff_data_with_fallback) deliberately -
    # pff_imports/ already covers this table's whole 2019-2025 window with
    # real per-year exports, and silently falling back to a NEIGHBORING
    # season's grade would misrepresent the exact year-over-year trend this
    # table exists to show. A season with genuinely no PFF data just shows
    # blank ('--') for that row instead - see the role_grade_cols formatting
    # step below for why that has to be a literal string, not NaN.
    from data.loaders import load_all_pff_data, load_year_data
    name_lower = selected_player.lower()
    # Slot %/Wide %/Inline % are WR/TE-specific PFF receiving-ALIGNMENT
    # concepts (per-route split: slot vs. wide vs. inline) - meaningless for
    # a QB/RB/K/defensive player, and PFF's receiving_summary.csv does carry
    # occasional rows for pass-catching RBs (a real match, not a lookup
    # miss), so "did this player match in pff['rec'] this year" alone isn't
    # a reliable WR/TE gate - a real position check is needed instead.
    # player_position was already derived above (used to branch agg_spec).
    is_wr_te = player_position in ('WR', 'TE')

    role_rows = []
    for yr in hist_grouped['season'].tolist():
        pff = load_all_pff_data(int(yr))
        overall_grade = pff['pff_grades_map'].get(name_lower)
        role_row = {'season': yr, 'Overall PFF': overall_grade}

        if is_wr_te:
            slot_rate = wide_rate = inline_rate = None
            rec_df = pff.get('rec', pd.DataFrame())
            if not rec_df.empty and 'player' in rec_df.columns:
                match = rec_df[rec_df['player'].str.lower() == name_lower]
                if not match.empty:
                    row = match.iloc[0]
                    slot_rate, wide_rate, inline_rate = row.get('slot_rate'), row.get('wide_rate'), row.get('inline_rate')
            role_row.update({'Slot %': slot_rate, 'Wide %': wide_rate, 'Inline %': inline_rate})

        snap_pct = None
        # load_year_data returns (stats, team_col, name_col, rookie_names) -
        # same 4-tuple shape as load_and_merge_data, not a bare DataFrame.
        yr_stats, _yr_team_col, yr_name_col, _yr_rookies = load_year_data(int(yr))
        if not yr_stats.empty and yr_name_col in yr_stats.columns:
            yr_match = yr_stats[yr_stats[yr_name_col].astype(str).str.lower() == name_lower]
            # has_snap_match (not just checking the value itself) - same
            # reason the bio card's own Snap % checks it: load_year_data
            # fills snap_pct_avg to a literal 0.0 sentinel when NO snap
            # source has a row for this player at all, and that's not
            # distinguishable from a genuine 0% by the number alone. Without
            # this check, a season with no snap-count coverage would show a
            # misleading "0.0%" here instead of a blank "--".
            if not yr_match.empty and bool(yr_match.iloc[0].get('has_snap_match', True)):
                snap_pct = yr_match.iloc[0].get('snap_pct_avg')
        role_row['Snap %'] = snap_pct

        role_rows.append(role_row)
    hist_grouped = hist_grouped.merge(pd.DataFrame(role_rows), on='season', how='left')

    # Sacks lands on half-increments (1 or 2.5), not whole numbers like the
    # other counting stats here - was previously getting int-cast along with
    # everything else, silently dropping half-sacks.
    decimal_cols = {'Fantasy Pts', 'Sacks'}
    # The new role/grade columns are handled separately below, not folded
    # into decimal_cols - a genuinely missing season (Slot %/Wide %/Inline %
    # for any non-receiver, or Snap % wherever the snap source has no row
    # at all) needs to end up as blank/'--', and confirmed live that
    # st.dataframe's grid (glide-data-grid) does NOT respect the Styler's
    # na_rep='--' for a real NaN float cell the way style_plain_dataframe
    # asks it to - it shows the literal string "None" instead, regardless
    # of the Styler's format() string (same class of "the grid quietly
    # ignores what the Styler asks for" limitation as border/box-shadow -
    # see style_depth_chart_table's docstring on that one). The only
    # reliable fix is the same one used there: make missing cells a real
    # '--' STRING at the value level rather than trusting the grid to
    # substitute display text for NaN.
    # Filtered against hist_grouped.columns - Slot %/Wide %/Inline % simply
    # aren't columns at all for a non-WR/TE player (role_row never set them
    # above), not just blanked out for them.
    role_grade_cols = [c for c in ['Overall PFF', 'Snap %', 'Slot %', 'Wide %', 'Inline %'] if c in hist_grouped.columns]
    for c in hist_grouped.columns:
        if c in ('season', 'Team') or c in role_grade_cols:
            continue
        elif c in decimal_cols:
            hist_grouped[c] = hist_grouped[c].round(1)
        else:
            hist_grouped[c] = hist_grouped[c].fillna(0).round(0).astype(int)

    for c in role_grade_cols:
        # pd.to_numeric first - a column that's None for EVERY row in this
        # player's window (confirmed real: Micah Parsons has no
        # receiving_summary.csv row in any season, so Slot/Wide/Inline % are
        # None every year for him) stays object dtype with literal Python
        # Nones rather than float64 NaN, since pandas only infers a numeric
        # dtype when a column has at least one real value to go on - .round()
        # on an object-dtype column falls back to calling the builtin
        # round() per element, which raises on a plain None.
        numeric = pd.to_numeric(hist_grouped[c], errors='coerce').round(1)
        hist_grouped[c] = numeric.apply(lambda v: '--' if pd.isna(v) else f"{v:.1f}")
    return hist_grouped


@st.cache_data
def fetch_intelligent_depth_chart(team, _stats_df, _df_pff_rec, year,
                                  _ourlads_rank_map=None, ourlads_sig=""):
    """
    `_stats_df`/`_df_pff_rec` underscore-prefixed (not hashed) - see
    apply_scoring_and_percentiles for why; `team`+`year` are the real cache key.

    `_ourlads_rank_map` ({clean_name_exact(name) -> listed slot, 1 = starter})
    is the Depth Charts tab's imported Ourlads chart for THIS team. It is
    underscore-prefixed (unhashable dict); `ourlads_sig` is the hashable
    stand-in the caller derives from the same map so the cache re-keys when a
    new chart is imported. It is applied ONLY for a season with no real
    weekly snap data yet (an upcoming/just-started season) - a past season
    keeps its authoritative full-season snap ordering and ignores the map.
    """
    t_col = 'recent_team' if 'recent_team' in _stats_df.columns else ('team' if 'team' in _stats_df.columns else None)
    if not t_col: return pd.DataFrame()

    # Captured BEFORE any per-player collapsing below: for a year with no
    # weekly stats file yet (e.g. an upcoming season, before games are
    # played), load_year_data() has no 'week' column at all and sets
    # weekly_snap_pct to a uniform placeholder 0.0 for every row - a real
    # value of 0, not NaN, so a later .fillna() can't recover the season
    # fallback average from it. Recording whether real weekly granularity
    # ever existed lets the snap_score block below skip that placeholder
    # entirely instead of letting it silently outrank the real
    # snap_pct_avg fallback data.
    had_real_weekly_data = 'week' in _stats_df.columns

    team_df = _stats_df[_stats_df[t_col].astype(str).str.upper() == team.upper()].copy()
    if team_df.empty: return pd.DataFrame()

    n_col = 'player_name' if 'player_name' in team_df.columns else ('full_name' if 'full_name' in team_df.columns else 'player_display_name')
    if n_col not in team_df.columns: return pd.DataFrame()

    # Dedup identity key: roster_weekly's abbreviated player_name column can
    # itself change mid-season for the SAME real player - confirmed real,
    # e.g. Michael Wilson (ARI, 2025) is "Mi.Wilson" for weeks 1-11 and
    # "M.Wilson" for weeks 12-18 (nflverse's own disambiguation logic
    # flipping, not two different players). Grouping by n_col directly
    # split him into two separate depth-chart entries, each with only half
    # a season's snaps/production. exact_name (already derived from the
    # always-consistent full-name column elsewhere in load_year_data) is
    # the stable per-player key; n_col is kept only as DISPLAY text (still
    # aggregated below, just not used to decide who's the same player).
    dedup_key = 'exact_name' if 'exact_name' in team_df.columns else n_col

    # stats_df is weekly-granularity (one row per player per week played).
    # Collapse to one row per player before building position buckets below,
    # or a single productive player's many weekly rows dominate their own
    # position group. fantasy_points is averaged per game played; everything
    # else here is already constant per player (bio fields, or season-level
    # values already broadcast across that player's weekly rows), so 'first'
    # is safe, EXCEPT weekly_snap_pct - see the MEDIAN note below. Trimmed to
    # just the columns this function actually uses - team_df can be 180+
    # columns wide at this point and the rest aren't needed for building the
    # chart.
    keep_cols = list(dict.fromkeys([n_col, dedup_key, t_col, 'position', 'depth_chart_position', 'years_exp', 'fantasy_points', 'snap_pct_avg', 'weekly_snap_pct', 'snap_data_year', 'draft_number']))
    keep_cols = [c for c in keep_cols if c in team_df.columns]
    if 'week' in team_df.columns and len(keep_cols) > 1:
        agg_spec = {}
        for c in keep_cols:
            if c == dedup_key: continue
            elif c == n_col: agg_spec[c] = 'first'
            elif c == 'fantasy_points': agg_spec[c] = 'mean'
            # MEDIAN, not mean, of the player's own weekly snap % across
            # games he actually played. A straight season-mean punishes a
            # clear starter who missed time to a late-season injury (e.g.
            # a player at 50-65% snaps most weeks but a couple of single-digit
            # weeks right before he's shut down) below a lower-ceiling
            # full-season rotational backup, even though the starter was
            # obviously the better/higher-snap player whenever healthy and
            # on the field. Median is far less sensitive to a handful of
            # extreme low weeks than a mean is - confirmed against Josh
            # Sweat (ARI, 2025): mean snap% (48.1%) ranked him BELOW a
            # backup OLB whose snaps never dipped as low, but median (51%)
            # correctly ranks him above once his late-season injury weeks
            # stop dragging the average down.
            elif c == 'weekly_snap_pct': agg_spec[c] = 'median'
            else: agg_spec[c] = 'first'
        team_df = team_df[keep_cols].groupby(dedup_key, as_index=False, observed=True).agg(agg_spec)

    if not _df_pff_rec.empty and 'slot_rate' in _df_pff_rec.columns:
        slot_map = dict(zip(_df_pff_rec['player'].str.lower(), _df_pff_rec['slot_rate']))
        team_df['slot_rate'] = team_df[n_col].astype(str).str.lower().map(slot_map).fillna(0)
    else:
        team_df['slot_rate'] = 0

    if 'years_exp' in team_df.columns:
        team_df['years_exp_num'] = pd.to_numeric(team_df['years_exp'].replace(['R', 'Rookie'], 0), errors='coerce').fillna(0)
    else:
        team_df['years_exp_num'] = 0

    # snap_score: median per-game weekly snap % where real weekly data
    # exists for this year (see above), falling back to the season-mean
    # snap_pct_avg both for players with no per-week match AND for years
    # with no weekly data at all (had_real_weekly_data False - e.g. an
    # upcoming season before any games are played, where weekly_snap_pct
    # is a meaningless uniform 0.0 for every player, not a real signal).
    # This is also what's shown in the "Snap: X%" cell text
    # (ui/tabs/depth_charts.py), so the displayed number and the starter
    # ranking always agree.
    if had_real_weekly_data and 'weekly_snap_pct' in team_df.columns:
        team_df['snap_score'] = team_df['weekly_snap_pct'].fillna(team_df.get('snap_pct_avg', 0))
    else:
        team_df['snap_score'] = team_df.get('snap_pct_avg', 0)
    if 'fantasy_points' not in team_df.columns: team_df['fantasy_points'] = 0

    # Draft-capital priority boost: a rookie taken early (roughly top-65
    # picks) should rank above career backups/journeymen even before real
    # snap-count data exists for them yet (right after the draft, or early
    # in a rookie season) - e.g. a top-10 pick shouldn't default to the
    # bottom of the depth chart just because he hasn't played a snap yet.
    # The boost fades out as snap_score climbs, so it never overrides an
    # established veteran once real playing-time data exists - a rookie who
    # has actually earned 50%+ snaps is ranked on that, not on draft slot.
    if 'draft_number' in team_df.columns:
        draft_num = pd.to_numeric(team_df['draft_number'], errors='coerce')
        is_rookie_num = (team_df['years_exp_num'] == 0)
        raw_bonus = (65 - draft_num).clip(lower=0) / 65 * 15
        fade = 1 - (team_df['snap_score'] / 50).clip(upper=1)
        team_df['draft_priority'] = (raw_bonus * fade).where(is_rookie_num, 0).fillna(0)
    else:
        team_df['draft_priority'] = 0
    team_df['depth_score'] = team_df['snap_score'] + team_df['draft_priority']

    # Ourlads preseason override. For an upcoming/just-started season the
    # snap_score above is last year's average - noisy for a team with roster
    # turnover, and the whole reason this tab existed before the Ourlads
    # importer was wired in. When the Depth Charts tab hands us an imported
    # chart (exact-name -> listed slot), let THAT drive the ordering: charted
    # players get a synthetic depth_score in listed order, above any real
    # snap number, so every downstream re-sort in this function (get_players,
    # the WR pool) that keys on depth_score follows it too. Players not on
    # the chart keep their snap-based score and sort in behind. Skipped
    # entirely once real weekly data exists (had_real_weekly_data) - a
    # played season is ordered on what actually happened.
    if _ourlads_rank_map and not had_real_weekly_data and dedup_key in team_df.columns:
        ol_rank = (clean_name_exact(team_df[dedup_key].astype(str))
                   .map(_ourlads_rank_map).astype(float))
        if ol_rank.notna().any():
            team_df['depth_score'] = np.where(
                ol_rank.notna(), 1000.0 - ol_rank.fillna(0) * 10.0, team_df['depth_score'])

    team_df = team_df.sort_values(by=['depth_score', 'fantasy_points', 'years_exp_num'], ascending=[False, False, False])
    p_col = 'depth_chart_position' if 'depth_chart_position' in team_df.columns else 'position'

    buckets = {}
    for pos in team_df[p_col].dropna().unique():
        buckets[pos] = team_df[team_df[p_col] == pos].to_dict('records')

    def get_players(allowed_pos, count=1):
        pool = []
        for p in allowed_pos:
            if p in buckets:
                pool.extend(buckets[p])
        pool = sorted(pool, key=lambda x: (x.get('depth_score', 0), x.get('fantasy_points', 0), x.get('years_exp_num', 0)), reverse=True)

        selected = pool[:count]
        for s in selected:
            for p in allowed_pos:
                if p in buckets and s in buckets[p]:
                    buckets[p].remove(s)
                    break
        return [s[n_col] for s in selected]

    dc_rows = []
    def pad(lst, n=6):
        lst = [str(x) for x in lst if x]
        return lst + [""] * (n - len(lst))

    def add_row(pos_label, names):
        row = {'Position': pos_label}
        names = pad(names, 6)
        strings = ['Starter', '2nd String', '3rd String', '4th String', '5th String', '6th String']
        for i, s in enumerate(strings):
            row[s] = names[i]
        dc_rows.append(row)

    add_row("QB", get_players(['QB'], 6))
    add_row("RB", get_players(['RB', 'FB'], 6))

    wr_pool = []
    if 'WR' in buckets:
        wr_pool = sorted(buckets['WR'], key=lambda x: (x.get('depth_score', 0), x.get('fantasy_points', 0), x.get('years_exp_num', 0)), reverse=True)

    if len(wr_pool) >= 3:
        top_3 = wr_pool[:3]
        rest = wr_pool[3:]
        top_3_slot = sorted(top_3, key=lambda x: x.get('slot_rate', 0), reverse=True)

        slot_wr = top_3_slot[0][n_col]
        out_wr1 = top_3_slot[1][n_col]
        out_wr2 = top_3_slot[2][n_col]

        add_row("WR (Outside)", [out_wr1] + [w[n_col] for w in rest[0::3]])
        add_row("WR (Outside)", [out_wr2] + [w[n_col] for w in rest[1::3]])
        add_row("WR (Slot)", [slot_wr] + [w[n_col] for w in rest[2::3]])
    else:
        names = [w[n_col] for w in wr_pool]
        add_row("WR (Outside)", names[:1] + names[3:4] if len(names)>0 else [])
        add_row("WR (Outside)", names[1:2] + names[4:5] if len(names)>1 else [])
        add_row("WR (Slot)", names[2:3] + names[5:6] if len(names)>2 else [])

    add_row("TE", get_players(['TE'], 6))

    ts = get_players(['T', 'OT', 'LT', 'RT'], 10)
    gs = get_players(['G', 'OG', 'LG', 'RG'], 10)
    add_row("LT", [ts[0]] + ts[2::2] if len(ts) > 0 else [])
    add_row("LG", [gs[0]] + gs[2::2] if len(gs) > 0 else [])
    add_row("C", get_players(['C'], 6))
    add_row("RG", [gs[1]] + gs[3::2] if len(gs) > 1 else [])
    add_row("RT", [ts[1]] + ts[3::2] if len(ts) > 1 else [])

    add_row("BREAK", [""]*6)

    # Interior/edge/LB/secondary rows below each draw ONLY from their own
    # precisely-tagged depth_chart_position pool first (NT before DT, OLB
    # before ILB, CB/SS/FS before the generic catch-alls) - the ORIGINAL
    # version let a later row's pool (e.g. "DT" in get_players(['NT','DT'])
    # for the NT row) get siphoned away by an EARLIER row's combined pool
    # (NT processed before DT, both drawing from the same DT-tagged players),
    # so a real starting DT (e.g. Nnamdi Madubuike, tagged 'DT') could get
    # mislabeled into the NT row and ranked behind the actual NT starter
    # instead of appearing in his own correct DT row. Same bug existed for
    # OLB/ILB sharing generic 'LB', and CB/SS/FS sharing generic 'S'/'DB'.
    # Fixed by giving each named row an exclusive pool and adding dedicated
    # catch-all rows (LB, S/DB) for anyone only tagged generically.
    add_row("NT", get_players(['NT'], 6))
    add_row("DT", get_players(['DT', 'DL'], 6))
    add_row("DE", get_players(['DE', 'EDGE'], 6))
    add_row("OLB", get_players(['OLB', 'EDGE'], 6))
    add_row("ILB", get_players(['ILB', 'MLB'], 6))
    add_row("LB (unclassified)", get_players(['LB'], 6))
    add_row("CB", get_players(['CB'], 6))
    add_row("SS", get_players(['SS'], 6))
    add_row("FS", get_players(['FS'], 6))
    add_row("S / DB (unclassified)", get_players(['S', 'DB'], 6))
    add_row("K", get_players(['K', 'PK'], 6))
    add_row("P", get_players(['P'], 6))
    add_row("LS", get_players(['LS'], 6))

    return pd.DataFrame(dc_rows)
