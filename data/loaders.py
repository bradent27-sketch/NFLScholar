"""
Raw data ingestion: local CSV loaders (nflverse-style stats/rosters/snaps,
PFF exports, external coverage-scheme data) and network fetchers (Odds API,
Sleeper API). See HANDOFF.md for the gotchas behind several of these
(two-tier name matching, dtype-downcast exclusions, snap-count fallback
chains) - preserved unchanged from the pre-modularization version.
"""
import os
import re
import glob
import pandas as pd
import numpy as np
import requests
import streamlit as st
import nflreadpy

# nflreadpy defaults to MEMORY caching, which means every single app restart
# re-downloads everything it pulls from nflverse - including play-by-play
# (~50k rows x 372 cols per season) - and a network outage kills those
# features entirely. FILESYSTEM mode writes downloads to the OS user cache
# dir with nflreadpy's default 24h expiry: restarts are near-instant, the
# app keeps working offline on yesterday's data, and in-season data still
# refreshes daily. This module is the app's only nflreadpy import, so
# setting it here covers every call site.
try:
    # update_config lives in the config submodule, NOT the top-level
    # package (confirmed: nflreadpy.update_config raises AttributeError) -
    # and the try/except around this would have silently swallowed exactly
    # that mistake, so if touching this, re-verify get_config().cache_mode
    # actually reports FILESYSTEM afterward.
    from nflreadpy.config import update_config as _nfl_update_config
    _nfl_update_config(cache_mode="filesystem")
except Exception:
    pass  # worst case is the old behavior (memory cache), never a crash

from data.utils import calculate_percentile, clean_name_exact, clean_name_for_merge


@st.cache_data
def _pivot_nflreadpy_snap_counts(year):
    """
    nflreadpy.load_snap_counts (PFR-sourced via nflverse, covers 2012-present)
    pivoted into the same wide "name" + "Wk N pct" + "snap_pct_avg" shape the
    local snap_counts_{year}.csv.csv export uses, so load_year_data's
    melt-based two-tier name matcher works identically against either source.

    Per-player-week pct is max(offense_pct, defense_pct) - each row in this
    export carries BOTH columns (one real, one 0) depending which side of the
    ball that player is on that week, not a single unified "pct" column like
    the local CSV. A prior version of this fallback used offense_pct alone,
    which is correct for offense/O-line but silently produced a flat 0% for
    every defensive player in any season without a local snap file (every
    year except 2025, confirmed - see load_year_data) - defense_pct existed
    in the source data the whole time, just never read.
    """
    try:
        raw = nflreadpy.load_snap_counts([year]).to_pandas()
    except Exception:
        return pd.DataFrame()
    if raw.empty or not {'player', 'week'}.issubset(raw.columns):
        return pd.DataFrame()
    raw = raw.copy()
    if 'game_type' in raw.columns:
        # REG only - same convention as every other weekly source in this
        # app (see load_year_data's season_type filter); this export numbers
        # playoff weeks continuing past 18, which would otherwise silently
        # blend a playoff-run team's postseason snaps into "season" figures.
        raw = raw[raw['game_type'] == 'REG']
    off = pd.to_numeric(raw.get('offense_pct', 0), errors='coerce').fillna(0)
    dfn = pd.to_numeric(raw.get('defense_pct', 0), errors='coerce').fillna(0)
    raw['combined_pct'] = pd.concat([off, dfn], axis=1).max(axis=1) * 100
    pivoted = raw.pivot_table(index='player', columns='week', values='combined_pct', aggfunc='mean')
    if pivoted.empty:
        return pd.DataFrame()
    pivoted.columns = [f"Wk {int(w)} pct" for w in pivoted.columns]
    pivoted = pivoted.reset_index().rename(columns={'player': 'name'})
    pct_cols = [c for c in pivoted.columns if 'pct' in c.lower()]
    pivoted['snap_pct_avg'] = pivoted[pct_cols].mean(axis=1)
    return pivoted


@st.cache_data
def build_veteran_database(target_year):
    try:
        past_years = [target_year - 1, target_year - 2, target_year - 3]
        past_rosters = nflreadpy.load_rosters(past_years).to_pandas()
        return set(past_rosters['full_name'].dropna().str.lower().unique())
    except:
        return set()


@st.cache_data
def load_year_data(year):
    """
    Loads and merges roster/stat/snap data for one season. This is the
    expensive part (multiple CSV reads + 3 merges) and none of it depends on
    scoring format, so it's cached on `year` alone. Profiled on the real
    dataset: ~1.5s wall time / ~120MB peak for a full season. See
    apply_scoring_and_percentiles() (data/transforms.py) for the part that
    used to be fused in here and busted this entire cache on every PPR-format
    toggle.
    """
    stats = pd.DataFrame()
    # Prefer an explicit weekly-granularity filename for this year (same
    # explicit-filename pattern already used for rosters/snaps below).
    local_stats = [f"stats_player_week_{year}.csv", f"player_stats_week_{year}.csv"]
    for f in local_stats:
        if os.path.exists(f):
            try:
                tmp_stats = pd.read_csv(f, low_memory=False)
                if 'season' in tmp_stats.columns:
                    tmp_stats = tmp_stats[tmp_stats['season'] == year]
                if 'player_id' in tmp_stats.columns:
                    tmp_stats = tmp_stats[tmp_stats['player_id'].notna()]
                if not tmp_stats.empty:
                    stats = tmp_stats.fillna(0)
                    break
            except: pass

    # Fallback: scan for any other stats export. IMPORTANT - only accept a
    # file that actually has weekly granularity ('week' column present).
    # Season-total exports (e.g. stats_player_reg_*.csv) match the same
    # naming pattern but have no 'week' column; previously this depended on
    # os.listdir() ordering (not guaranteed) and could silently substitute a
    # season-total file, which crashes precompute_league_percentiles and
    # breaks every weekly/per-game view downstream.
    if stats.empty:
        for f in os.listdir():
            if re.search(r'(stats_player|player_stats|player_summary)', f.lower()) and f.endswith('.csv'):
                try:
                    tmp_stats = pd.read_csv(f, low_memory=False)
                    if 'week' not in tmp_stats.columns:
                        continue
                    if 'season' in tmp_stats.columns:
                        tmp_stats = tmp_stats[tmp_stats['season'] == year]
                    if 'player_id' in tmp_stats.columns:
                        tmp_stats = tmp_stats[tmp_stats['player_id'].notna()]
                    if not tmp_stats.empty:
                        stats = tmp_stats.fillna(0)
                        break
                except: pass

    if stats.empty:
        try:
            stats = nflreadpy.load_player_stats([year]).to_pandas().fillna(0)
        except Exception:
            stats = pd.DataFrame(columns=['player_id'])

    # Regular season only - every local stats_player_week_{year}.csv (2019-
    # 2025, confirmed on all seven files) mixes REG rows in with POST
    # (playoff) rows under the SAME 'week' numbering scheme restarting at
    # 19-21 instead of continuing 18+, with no season_type filter applied
    # anywhere before this. That meant every "season" figure derived from
    # weekly stats - game logs, season/recent-form averages, VORP, Risers,
    # Rookie Watch, red-zone share, depth-chart snap%/fantasy_points
    # aggregation - was silently blending in playoff performances for any
    # team that made a playoff run, which no standard fantasy league scores
    # or drafts around. load_schedule() already had this same REG-only
    # filter (for the SOS table); the weekly PLAYER stats loader never did.
    if 'season_type' in stats.columns:
        stats = stats[stats['season_type'] == 'REG']

    if 'player_id' not in stats.columns:
        stats['player_id'] = pd.Series(dtype='str')

    rosters = pd.DataFrame()
    local_rosters = [f"roster_weekly_{year}.csv", "players.csv", "weekly_rosters.csv"]
    for r_file in local_rosters:
        if os.path.exists(r_file):
            try:
                tmp_roster = pd.read_csv(r_file, low_memory=False)
                if 'season' in tmp_roster.columns:
                    tmp_roster = tmp_roster[tmp_roster['season'] == year]
                if not tmp_roster.empty:
                    rosters = tmp_roster
                    break
            except: pass

    if rosters.empty:
        try: rosters = nflreadpy.load_rosters([year]).to_pandas()
        except Exception: rosters = pd.DataFrame(columns=['gsis_id', 'full_name', 'position', 'team'])

    snaps = pd.DataFrame()
    # NOTE: previously included a hardcoded "snap_counts_2025.csv.csv" as an
    # unconditional final fallback. That meant selecting any year other than
    # 2025 (including the default landing year on the Depth Charts / Defensive
    # tabs) would silently attach 2025 snap percentages to that year's players
    # via name matching -- wrong for anyone who changed role/team, and for
    # rookies. Only ever load a snap file that's actually for the requested year.
    local_snaps = [f"snap_counts_{year}.csv.csv", f"snap_counts_{year}.csv"]
    for s_file in local_snaps:
        if os.path.exists(s_file):
            try:
                snaps = pd.read_csv(s_file, low_memory=False)
                pct_cols = [c for c in snaps.columns if 'pct' in c.lower()]
                if pct_cols:
                    snaps['snap_pct_avg'] = snaps[pct_cols].mean(axis=1)
                break
            except: pass

    snap_source_year = year
    if snaps.empty:
        # nflreadpy.load_snap_counts (PFR-sourced via nflverse) covers
        # 2012-present - see _pivot_nflreadpy_snap_counts for the shape
        # conversion (and why it reads BOTH offense_pct and defense_pct, not
        # just offense_pct - that used to zero out every defensive player in
        # any year relying on this fallback).
        snaps = _pivot_nflreadpy_snap_counts(year)
    else:
        # Local file exists but is a real, confirmed partial export - the
        # 2025 snap_counts_2025.csv.csv file has ZERO offensive-line rows at
        # all (verified: none of its 1,828 rows carry an OL positionGroup),
        # not just sparse OL coverage - so every offensive lineman showed
        # "N/A" snap % on both Player Search and the Depth Chart, and the
        # Depth Chart's OL ordering fell back to draft-capital/experience
        # alone with no real playing-time signal at all. Top up the local
        # file with nflreadpy rows for any player NOT already present
        # locally (matched loosely, since the two sources don't necessarily
        # spell every name identically) rather than replacing anything the
        # local file already covers.
        nfl_supplement = _pivot_nflreadpy_snap_counts(year)
        if not nfl_supplement.empty and 'name' in snaps.columns:
            local_keys = set(clean_name_for_merge(snaps['name']))
            supp_keys = clean_name_for_merge(nfl_supplement['name'])
            new_rows = nfl_supplement[~supp_keys.isin(local_keys)]
            if not new_rows.empty:
                snaps = pd.concat([snaps, new_rows], ignore_index=True, sort=False)

    # If there's truly no snap data for this year yet (e.g. the upcoming
    # season before games are played), fall back to the most recent prior
    # season's averages rather than showing 0% for everyone -- this is only
    # a season-level average fallback (snap_pct_avg), never the weekly
    # breakdown, and only fires when the requested year is genuinely empty,
    # so it can't re-introduce the cross-year contamination removed above.
    if snaps.empty:
        for back in range(1, 6):
            prior_year = year - back
            if prior_year < 2012: break
            for s_file in [f"snap_counts_{prior_year}.csv.csv", f"snap_counts_{prior_year}.csv"]:
                if os.path.exists(s_file):
                    try:
                        snaps = pd.read_csv(s_file, low_memory=False)
                        pct_cols = [c for c in snaps.columns if 'pct' in c.lower()]
                        if pct_cols:
                            snaps['snap_pct_avg'] = snaps[pct_cols].mean(axis=1)
                        snap_source_year = prior_year
                        break
                    except: pass
            if not snaps.empty:
                break

    def get_col(df, cols):
        for c in cols:
            if c in df.columns: return df[c]
        return pd.Series([0]*len(df), index=df.index)

    # NOTE: fantasy_points is computed later in apply_scoring_and_percentiles()
    # since it depends on the PPR scoring multiplier, which this function no
    # longer takes as an argument (see cache-splitting note above).
    #
    # All 16 canonical columns are resolved first and attached in a single
    # concat rather than 16 sequential stats['col'] = ... assignments. On a
    # frame this wide (145+ raw columns from the stats CSV), each individual
    # single-column insert forces pandas to reallocate its internal block
    # manager -- confirmed by profiling, which reproduced pandas' own
    # "DataFrame is highly fragmented" PerformanceWarning on this exact block.
    canonical_cols = {
        'targets': get_col(stats, ['targets']),
        'receptions': get_col(stats, ['receptions', 'rec']),
        'receiving_yards': get_col(stats, ['receiving_yards', 'rec_yds']),
        'receiving_tds': get_col(stats, ['receiving_tds', 'rec_tds']),
        'rushing_attempts': get_col(stats, ['carries', 'rushing_attempts', 'rush_att']),
        'rushing_yards': get_col(stats, ['rushing_yards', 'rush_yds']),
        'rushing_tds': get_col(stats, ['rushing_tds', 'rush_tds']),
        'passing_attempts': get_col(stats, ['attempts', 'passing_attempts', 'pass_att']),
        'passing_completions': get_col(stats, ['completions', 'passing_completions', 'pass_cmp']),
        'passing_yards': get_col(stats, ['passing_yards', 'pass_yds']),
        'passing_tds': get_col(stats, ['passing_tds', 'pass_tds']),
        # BUGFIX: raw column is literally 'passing_interceptions' -- it was
        # never in its own alias list, so this was always the zero-fallback.
        'passing_interceptions': get_col(stats, ['passing_interceptions', 'interceptions', 'passing_int']),
        # BUGFIX: none of these matched the real nflverse column names
        # (def_sacks, def_interceptions, def_fumbles_forced), so every
        # defensive player showed 0 for all four stats. Tackles has no single
        # combined column in this export (def_tackles_solo +
        # def_tackles_with_assist), handled separately below.
        'sacks': get_col(stats, ['def_sacks', 'sacks', 'sack']),
        'def_ints': get_col(stats, ['def_interceptions', 'interceptions', 'def_ints', 'int', 'ints']),
        'forced_fumbles': get_col(stats, ['def_fumbles_forced', 'forced_fumbles', 'ff']),
    }
    if 'def_tackles_solo' in stats.columns or 'def_tackles_with_assist' in stats.columns:
        canonical_cols['tackles'] = get_col(stats, ['def_tackles_solo']).fillna(0) + get_col(stats, ['def_tackles_with_assist']).fillna(0)
    else:
        canonical_cols['tackles'] = get_col(stats, ['tackles', 'solo_tackles', 'combined_tackles', 'total_tackles'])
    # ypc/y-c fold into the same batch (computed from the dict values directly,
    # not stats[...], since stats hasn't been concatenated with them yet) --
    # two more one-at-a-time inserts right after the concat below was enough
    # on its own to re-trigger the fragmentation warning on a frame this wide.
    canonical_cols['ypc'] = (canonical_cols['rushing_yards'] / canonical_cols['rushing_attempts'].replace(0, np.nan)).fillna(0)
    canonical_cols['y/c'] = (canonical_cols['receiving_yards'] / canonical_cols['receptions'].replace(0, np.nan)).fillna(0)

    stats = stats.drop(columns=[c for c in canonical_cols if c in stats.columns], errors='ignore')
    stats = pd.concat([stats, pd.DataFrame(canonical_cols, index=stats.index)], axis=1)


    # roster_weekly_*.csv rows are NOT stored in week order, so keeping the
    # first-seen row per player (old behavior) effectively picked a random
    # week's snapshot -- e.g. a player's pre-trade team could "win" over
    # their current one depending on row order. Sort newest-week-first so
    # drop_duplicates keeps each player's most recent bio snapshot.
    if 'week' in rosters.columns:
        rosters = rosters.sort_values('week', ascending=False)

    bio_cols = ['gsis_id', 'pfr_id', 'espn_id', 'full_name', 'team', 'position', 'depth_chart_position', 'age', 'height', 'weight', 'jersey_number', 'years_exp', 'rookie_year', 'birth_date', 'draft_number', 'draft_club', 'entry_year']
    rosters_clean = rosters[[c for c in bio_cols if c in rosters.columns]].drop_duplicates(subset=['full_name']) if 'full_name' in rosters.columns else rosters

    # roster_weekly_*.csv's own 'position' column is a coarse group (DB/DL/LB/
    # OL for defense/O-line) compared to the granular codes
    # (DE/DT/CB/FS/SS/ILB/OLB/NT/etc.) that stats_player_week_*.csv's own
    # 'position' column already carries for anyone with recorded stats.
    # Renamed here BEFORE the overlap-drop-and-remerge below so the coarser
    # roster group doesn't clobber the stats file's granular position - it
    # was doing exactly that (every defensive/O-line player's position
    # collapsed to DB/DL/LB/OL after the merge), which silently broke every
    # position-specific defensive stat display (Player Search's positional
    # matrix + game log column selection both branch on this exact field).
    if 'position' in rosters_clean.columns:
        rosters_clean = rosters_clean.rename(columns={'position': 'roster_position_group'})

    if 'rookie_year' in rosters_clean.columns:
        rookie_year_num = pd.to_numeric(rosters_clean['rookie_year'], errors='coerce')
        rosters_clean['is_rookie_flag'] = rookie_year_num == float(year)
    elif 'years_exp' in rosters_clean.columns:
        rosters_clean['is_rookie_flag'] = rosters_clean['years_exp'].apply(lambda x: str(x).strip() in ['0', '0.0', 'R', 'Rookie'])
    else:
        rosters_clean['is_rookie_flag'] = False

    global_rookie_names = set(rosters_clean[rosters_clean['is_rookie_flag']]['full_name'].dropna().str.lower().unique()) if 'full_name' in rosters_clean.columns else set()

    overlaps = [c for c in rosters_clean.columns if c in stats.columns and c != 'gsis_id']
    stats = stats.drop(columns=overlaps, errors='ignore')

    if 'gsis_id' in rosters_clean.columns:
        stats = pd.merge(stats, rosters_clean, left_on='player_id', right_on='gsis_id', how='outer')
        stats['player_id'] = stats['player_id'].fillna(stats['gsis_id'])

    name_col = 'player_display_name' if 'player_display_name' in stats.columns else 'player_name'
    if name_col not in stats.columns: stats[name_col] = stats.get('full_name', pd.Series(['Unknown']*len(stats)))
    else: stats[name_col] = stats[name_col].fillna(stats.get('full_name'))

    team_col = 'recent_team' if 'recent_team' in stats.columns else 'team'
    stats[team_col] = stats.get(team_col, stats.get('team', pd.Series(['UNK']*len(stats)))).fillna(stats.get('team', 'UNK'))
    stats['position'] = stats.get('position', pd.Series([None]*len(stats)))
    # Fall back to the roster's coarser position GROUP only for rows with no
    # stats-side position at all (e.g. a player who never recorded a stat
    # this season and only exists via the outer-join from the roster side) -
    # never overwrites the granular stats-side code when one already exists.
    if 'roster_position_group' in stats.columns:
        stats['position'] = stats['position'].fillna(stats['roster_position_group'])
    stats['position'] = stats['position'].fillna('UNK')

    # TWO-TIER NAME MATCH FOR WEEKLY SNAPS
    # Tier 1 (exact): match names as-is (just lowercased/depunctuated). This
    # is tried first specifically so two different players who share a base
    # name but are distinguished by a real suffix - e.g. "Byron Murphy"
    # (Vikings CB) vs "Byron Murphy II" (Seahawks DT) - each match their own
    # data instead of colliding.
    # Tier 2 (loose/suffix-stripped): only for names that don't find an
    # exact counterpart on both sides - handles cases like "A.J. Terrell" vs
    # "A.J. Terrell Jr." where it's genuinely the same person spelled
    # differently across files. The tier-2 pool explicitly excludes any
    # snap-side name that exists as a tier-1 exact name anywhere in stats,
    # so an already-claimed player's data can't be borrowed by someone
    # else's fallback lookup (this is exactly what let Byron Murphy II's
    # snap rows get double-matched onto Byron Murphy's stat rows before).
    if not snaps.empty and 'name' in snaps.columns and name_col in stats.columns:
        snaps['exact_name'] = clean_name_exact(snaps['name'])
        snaps['clean_merge_name'] = clean_name_for_merge(snaps['name'])
        # Both name-key columns attached in ONE concat, not two sequential
        # inserts - same block-manager fragmentation reasoning as the
        # canonical_cols batch above (these two lines were still the top
        # PerformanceWarning source in the server log after that fix).
        stats = pd.concat([stats, pd.DataFrame({
            'exact_name': clean_name_exact(stats[name_col]),
            'clean_merge_name': clean_name_for_merge(stats[name_col]),
        }, index=stats.index)], axis=1)
        exact_names_in_stats = set(stats['exact_name'])

        # Excludes 'snap_pct_avg' itself (a season-level summary column that
        # also happens to contain "pct") - week_pct_cols is the real per-
        # week "Wk N pct" columns only, used below for snap_games_played.
        # pct_cols (unfiltered) stays as-is for the melt below, unchanged
        # from before this session's addition.
        pct_cols = [c for c in snaps.columns if 'pct' in c.lower()]
        week_pct_cols = [c for c in pct_cols if c != 'snap_pct_avg']
        if pct_cols and 'week' in stats.columns:
            melted = snaps.melt(id_vars=['exact_name', 'clean_merge_name'], value_vars=pct_cols, var_name='week_str', value_name='weekly_snap_pct')
            melted['week'] = melted['week_str'].str.extract(r'(\d+)').astype(float)
            melted = melted.dropna(subset=['weekly_snap_pct'])

            exact_pool = melted.drop_duplicates(subset=['exact_name', 'week'])
            stats = pd.merge(stats, exact_pool[['exact_name', 'week', 'weekly_snap_pct']], on=['exact_name', 'week'], how='left')

            unmatched = stats['weekly_snap_pct'].isna()
            if unmatched.any():
                fallback_pool = melted[~melted['exact_name'].isin(exact_names_in_stats)].drop_duplicates(subset=['clean_merge_name', 'week'])
                fallback_map = fallback_pool.set_index(['clean_merge_name', 'week'])['weekly_snap_pct']
                idx = pd.MultiIndex.from_arrays([stats.loc[unmatched, 'clean_merge_name'], stats.loc[unmatched, 'week']])
                stats.loc[unmatched, 'weekly_snap_pct'] = idx.map(fallback_map).to_numpy()
            stats['weekly_snap_pct'] = stats['weekly_snap_pct'].fillna(0.0)
        else:
            stats['weekly_snap_pct'] = 0.0

        # snap_pct_avg: same two-tier approach, single-key so .map() covers
        # it. Computed as a standalone Series first, then attached alongside
        # has_snap_match/snap_data_year in ONE concat below - the previous
        # three sequential single-column inserts each re-triggered pandas'
        # fragmentation warning on this very wide frame.
        snap_avg_by_exact = snaps.groupby('exact_name')['snap_pct_avg'].mean()
        snap_avg = stats['exact_name'].map(snap_avg_by_exact)

        # snap_games_played: how many real per-week entries the SNAP SOURCE
        # itself has for this player, season-level (like snap_pct_avg,
        # NOT gated by whether stats_player_week_{year}.csv happens to have
        # a row for that player-week - see the weekly_snap_pct merge above,
        # which IS gated that way). This matters because nflverse's weekly
        # box-score export is stat-triggered, not full-roster/full-snap-
        # participation: confirmed real for Kingsley Suamataia (KC OT/G,
        # 2025) - he has exactly 5 rows in stats_player_week_2025.csv all
        # season (weeks 5/11/12/16/18) despite playing the great majority of
        # KC's offensive snaps essentially every week; every other week he
        # played has no stats-side row AT ALL (no rushing/receiving/passing
        # stat to trigger one), not a real 0. A season-share calc built by
        # summing the stats-row-gated weekly_snap_pct (ui.tabs.depth_charts.
        # _build_snap_totals used to do exactly this) silently drops every
        # one of those un-rowed weeks from its numerator while still
        # dividing by the team's real week count, undercounting every O-
        # lineman's displayed share to a fraction of their real one (this
        # player's own case: summed to ~26% instead of a true-to-life ~98%
        # per-game rate). snap_games_played lets that calc use snap_pct_avg
        # (already correct - the snap source's own mean, unaffected by this
        # gap) times a real games-played count instead.
        # Per-row count first (how many of THIS row's week-columns are real),
        # then max (not sum) across any duplicate rows for the same
        # exact_name - safer than sum if local+nflreadpy both ever produced
        # a row for the same real player under slightly different name
        # spellings, since sum could double-count an overlapping week.
        snaps = snaps.copy()
        snaps['_snap_games_row'] = snaps[week_pct_cols].notna().sum(axis=1)
        games_by_exact = snaps.groupby('exact_name')['_snap_games_row'].max()
        snap_games = stats['exact_name'].map(games_by_exact)

        if snap_avg.isna().any():
            fallback_snaps = snaps[~snaps['exact_name'].isin(exact_names_in_stats)]
            snap_avg_by_loose = fallback_snaps.groupby('clean_merge_name')['snap_pct_avg'].mean()
            snap_avg = snap_avg.fillna(stats['clean_merge_name'].map(snap_avg_by_loose))
            games_by_loose = fallback_snaps.groupby('clean_merge_name')['_snap_games_row'].max()
            snap_games = snap_games.fillna(stats['clean_merge_name'].map(games_by_loose))
        # has_snap_match captures whether a real snap-count match was ever
        # found, BEFORE the fillna(0.0) collapses "no data available at all"
        # and "genuinely played 0% of snaps" into the same displayed value.
        # Downstream display code (depth chart cells, bio card) uses this to
        # show "N/A" instead of a misleading "0%" for players this source
        # simply has no row for. Confirmed this is a real, substantial gap:
        # the local snap-count export has almost no offensive-line coverage
        # (only 4 of 1,189 OL-tagged rows in the 2025 file have any snap
        # data at all) - an OL starter showing "0%" reads as "this guy
        # never plays" when the truth is just "no data for him", a
        # materially misleading signal for a depth-chart tool to show.
        stats = pd.concat([stats, pd.DataFrame({
            'has_snap_match': snap_avg.notna(),
            'snap_pct_avg': snap_avg.fillna(0.0),
            'snap_games_played': snap_games.fillna(0.0),
            'snap_data_year': snap_source_year,
        }, index=stats.index)], axis=1)
    else:
        stats = pd.concat([stats, pd.DataFrame({
            'weekly_snap_pct': 0.0,
            'snap_pct_avg': 0.0,
            'snap_games_played': 0.0,
            'has_snap_match': False,
            'snap_data_year': snap_source_year,
        }, index=stats.index)], axis=1)

    def build_headshot(eid):
        try:
            if pd.notnull(eid) and str(eid).replace('.0','').isdigit():
                val = int(float(eid))
                if val > 0: return f"https://a.espncdn.com/i/headshots/nfl/players/full/{val}.png"
        except: pass
        return "https://a.espncdn.com/combiner/i?img=/i/headshots/noplayer/nfl.png"

    stats['headshot_url'] = stats.get('espn_id', pd.Series([None]*len(stats))).apply(build_headshot)

    # Memory downcast, done last so every merge/groupby/rank above ran on
    # normal dtypes first. Any text column repeated over thousands of rows
    # with relatively few distinct values (position, team, status, ...) is
    # far cheaper as `category` than generic string objects, and pandas
    # comparisons/groupby on category columns are typically faster too;
    # near-unique columns like gsis_id/headshot_url fall below the 50%
    # threshold check and are correctly left alone. float64 is more
    # precision than any yards/tackles/pct stat needs, so float32 halves
    # that portion of memory. Measured on the real 2024 dataset: the merged
    # frame's numeric columns alone were ~28MB at float64.
    #
    # IMPORTANT: try numeric coercion before categorizing. For a year with
    # no matching local stats file yet (e.g. next season, before its data
    # exports exist), the stat columns above get built from an empty frame,
    # so they start as an empty object-dtype Series; the later outer-merge
    # with rosters backfills real rows but leaves those columns almost
    # entirely NaN. A column that's ~100% NaN has trivially low cardinality,
    # so a cardinality-only check would wrongly file a *numeric* column
    # (rushing_yards, tackles, etc.) as category and break every arithmetic
    # op on it downstream -- caught by testing this against a year with no
    # local file (2026 in this archive) before shipping it.
    # IMPORTANT: don't categorize name/identifier columns. They get run
    # through many groupby/merge/reset_index operations downstream (risers,
    # rookie watch, depth chart dedup), and a categorical that's passed
    # through enough of those can end up in a state where pyarrow's Arrow
    # conversion - which st.dataframe uses under the hood - fails outright.
    # Reproduced directly: st.dataframe(risers.set_index('Player'), ...)
    # raised "Could not convert '...' with type str: tried to convert to
    # int64" once 'Player' had picked up category dtype from this loop.
    no_categorize = {name_col, 'full_name', 'clean_merge_name', 'gsis_id', 'headshot_url'}
    obj_cols = stats.select_dtypes(include=['object']).columns
    for c in obj_cols:
        if not len(stats): continue
        if c in no_categorize: continue
        coerced = pd.to_numeric(stats[c], errors='coerce')
        if coerced.notna().sum() >= stats[c].notna().sum() * 0.99:
            stats[c] = coerced.astype('float32')
        elif stats[c].nunique(dropna=True) / len(stats) < 0.5:
            stats[c] = stats[c].astype('category')
    float_cols = stats.select_dtypes(include=['float64']).columns
    if len(float_cols):
        stats[float_cols] = stats[float_cols].astype('float32')

    # A handful of columns above (team/position defaults, snap columns,
    # headshot_url) are still set one at a time inside conditional branches,
    # which keeps re-triggering pandas' fragmentation warning on a frame
    # this wide even after the big batched concat earlier. A single .copy()
    # consolidates everything in one pass -- this is pandas' own documented
    # fix for this exact warning, cheaper than restructuring every branch
    # above to batch its one-or-two-column assignment.
    stats = stats.copy()

    return stats, team_col, name_col, global_rookie_names


@st.cache_data
def load_weekly_stats_history():
    """
    Loads every local weekly-granularity player stats export (all years) for
    the historical 3-year view. Cached because this reads several files off
    disk and previously ran, UNCACHED, on every single app rerun - Streamlit
    executes all st.tabs() bodies on every rerun regardless of which tab is
    visually active, so this cost was being paid even when just tweaking a
    dropdown in the Player Search tab.

    IMPORTANT: only matches "_week_" files, never "_reg_" (season-total)
    files. The previous version matched both with the same regex and
    concatenated them together, then summed by season - which double-counted
    every stat, since a player's season total was being added on top of the
    same total already reconstructed from summing their weekly rows.
    Verified on this dataset: a real 1,708-yard / 127-catch season was being
    shown as 3,416 yards / 254 catches (exactly 2x) before this fix.
    """
    hist_frames = []
    for f in os.listdir():
        fl = f.lower()
        if not fl.endswith('.csv'): continue
        if '_reg_' in fl or 'season_level' in fl: continue
        if re.search(r'(stats_player|player_stats|player_summary)', fl):
            try:
                tmp = pd.read_csv(f, low_memory=False)
                if 'week' in tmp.columns:
                    hist_frames.append(tmp)
            except: pass
    if hist_frames:
        return pd.concat(hist_frames, ignore_index=True)
    return pd.DataFrame()


@st.cache_data
def load_external_coverage_schemes():
    """
    Team-level man/zone coverage rates - free alternative to PFF's Shadow
    Coverage Matrix (which requires PFF Edge/Elite). Sourced from Sharp
    Football Analysis's public coverage scheme page, saved locally as
    sharp_coverage_schemes_2025.csv. This is team-level scheme tendency,
    not individual shadow-matchup tracking - a real data gap, not
    something this can fully substitute for, but useful on its own.
    """
    for p in ['sharp_coverage_schemes_2025.csv', os.path.join('external_data', 'sharp_coverage_schemes_2025.csv')]:
        if os.path.exists(p):
            try: return pd.read_csv(p)
            except: pass
    return pd.DataFrame()


@st.cache_data
def load_sumersports_tendency_data():
    """
    Team personnel-grouping (11 personnel usage rate) and formation (2X2
    usage rate) tendency, both offense and defense, pulled manually from
    SumerSports' public team-stats pages (no Cover-1/2/3/4/6 shell data is
    published there - confirmed by checking all 6 pages directly - so this
    is EPA/personnel/formation context, not a coverage-shell substitute).
    Used as the secondary "scheme context" panel in the Defensive Coverage
    Correlator (data/coverage_radar.py), alongside the man/zone PFF data
    that does the actual correlation.
    """
    files = {
        'def_overview': 'sumersports_defensive_overview_2025.csv',
        'off_overview': 'sumersports_offensive_overview_2025.csv',
        'def_personnel': 'sumersports_defensive_personnel_tendency_2025.csv',
        'off_personnel': 'sumersports_offensive_personnel_tendency_2025.csv',
        'def_formation': 'sumersports_defensive_formation_tendency_2025.csv',
        'off_formation': 'sumersports_offensive_formation_tendency_2025.csv',
    }
    out = {}
    for key, fname in files.items():
        p = os.path.join('external_data', fname)
        out[key] = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    return out


@st.cache_data
def load_sharp_positional_coverage():
    """
    Yards-per-target ALLOWED by receiver type (WR/TE/RB) and alignment
    (Outside/Slot), all 32 teams - Sharp Football Analysis's free "NFL
    Coverage Stats by Position" page (external_data/sharp_positional_coverage_2025.csv).
    Powers the "opposing defense" half of the Defensive Coverage Correlator
    radar (data/coverage_radar.build_defense_radar_data).
    """
    p = os.path.join('external_data', 'sharp_positional_coverage_2025.csv')
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def load_pff(filename):
    search_paths = [filename, os.path.join('pff_imports', filename)]
    for p in search_paths:
        if os.path.exists(p): return pd.read_csv(p)

    # Fallback for browser-download-style dedup suffixes ("file (1).csv").
    # This is exactly what silently broke receiving_summary: the code asked
    # for "receiving_summary (1).csv" but the folder only ever contained
    # "receiving_summary.csv", so load_pff returned an empty DataFrame with
    # no error -- every downstream "if not df.empty" check just quietly
    # skipped it, killing WR/TE YPRR/drop-rate/slot-rate columns and the
    # slot-rate input to the depth chart, with nothing visibly wrong.
    base = re.sub(r'\s\(\d+\)(?=\.\w+$)', '', filename)
    if base != filename:
        for p in [base, os.path.join('pff_imports', base)]:
            if os.path.exists(p): return pd.read_csv(p)
    return pd.DataFrame()


def load_pff_year(base_filename, year):
    """
    Loads one PFF export for a specific season. pff_imports/ is organized as
    one subfolder per year (pff_imports/2019/, pff_imports/2020/, ...,
    pff_imports/2025/) rather than year-suffixed filenames, since manually
    renaming every fresh PFF export would be tedious to keep up. Matched by
    PREFIX (not an exact filename) because the files inside each folder
    aren't named consistently across years: some carry a browser-download
    dedup suffix ("receiving_concept (6).csv" - confirmed byte-identical to
    its sibling copy, same finding as the pre-existing 2025
    receiving_concept_Slot/Screen duplicate - see load_all_pff_data), some
    carry an explicit "_2025" year suffix, and some (the 2024 folder) have
    neither. A plain prefix glob catches all three shapes without needing a
    separate pattern per year - verified no collisions between any of the
    15 base names this app loads (e.g. "defense_coverage_scheme" and
    "defense_coverage_summary" diverge before the wildcard, so one can't
    accidentally match the other's file).

    Falls back to load_pff() (the flat, pre-foldering pff_imports/ layout)
    if no subfolder exists yet for the requested year.
    """
    year_dir = os.path.join('pff_imports', str(year))
    if os.path.isdir(year_dir):
        candidates = sorted(glob.glob(os.path.join(year_dir, f'{base_filename}*.csv')))
        if not candidates:
            return pd.DataFrame()
        try:
            return pd.read_csv(candidates[0], low_memory=False)
        except Exception:
            return pd.DataFrame()
    return load_pff(f'{base_filename}.csv')


@st.cache_data
def load_all_pff_data(year):
    """
    Loads every PFF export for one season plus percentile columns for each,
    bundled into one dict. `year` is a real (non-underscore) cache key so
    switching seasons doesn't return another season's stale cached dict -
    see build_points_allowed_matrix's docstring elsewhere in this file for
    what a missing cache key here would look like in practice.
    """
    d = {}
    d['rec'] = load_pff_year('receiving_summary', year)
    d['pass'] = load_pff_year('passing_summary', year)
    d['rush'] = load_pff_year('rushing_summary', year)
    d['block'] = load_pff_year('offense_blocking', year)
    d['pass_block'] = load_pff_year('offense_pass_blocking', year)
    d['run_block'] = load_pff_year('offense_run_blockng', year)
    d['rec_scheme'] = load_pff_year('receiving_scheme', year)
    # NOTE: 'team_coverage_schemes.csv' doesn't exist in pff_imports and this
    # frame isn't referenced anywhere else in the app (dead weight either
    # way). The closest real file is defense_coverage_scheme, pointed at
    # here so this is at least loading real data - see data/coverage_radar.py
    # for what uses it.
    d['def_coverage_scheme'] = load_pff_year('defense_coverage_scheme', year)
    d['cov_summary'] = load_pff_year('defense_coverage_summary', year)
    d['fg'] = load_pff_year('field_goal_summary', year)
    d['punt'] = load_pff_year('punting_summary', year)
    # receiving_concept_Slot_2025.csv and receiving_concept_Screen_2025.csv
    # are byte-identical exports (verified: same 506 rows, every slot_* AND
    # screen_* column matches exactly) - PFF's concept filter didn't change
    # what got exported, so only one file needs loading; it already has
    # both concepts. Same finding holds for the "receiving_concept (N).csv"
    # duplicate pairs in the pre-2025 year folders (verified for 2024).
    d['route_concept'] = load_pff_year('receiving_concept', year)
    d['slot_cov'] = load_pff_year('slot_coverage', year)
    d['run_def'] = load_pff_year('run_defense_summary', year)
    d['pass_rush'] = load_pff_year('pass_rush_summary', year)
    # ESPN QBR (qbr_season_level.csv) is one file covering every year
    # (2006-2025) already - not part of the per-year pff_imports/ foldering,
    # so it's still loaded via the plain flat-path load_pff().
    d['qbr_season'] = load_pff('qbr_season_level.csv')

    if not d['qbr_season'].empty and 'qbr_total' in d['qbr_season'].columns:
        # qbr_season_level.csv is one row per QB PER SEASON (2006-2025) AND
        # per season_type (Regular/Playoffs) - not one row per player. A
        # flat calculate_percentile() here ranked every QB-season across all
        # 20 years and both season types in one pool, so a percentile
        # computed in, say, 2008 (a much lower-scoring era) got compared
        # directly against 2024 - confirmed real skew, not just a display
        # nit. Grouping by (season, season_type) ranks each QB only against
        # his actual same-year, same-type peers, which is the correct
        # comparison set for a single-season badge.
        group_cols = [c for c in ['season', 'season_type'] if c in d['qbr_season'].columns]
        if group_cols:
            d['qbr_season']['qbr_pct'] = d['qbr_season'].groupby(group_cols)['qbr_total'].rank(pct=True) * 100
        else:
            d['qbr_season']['qbr_pct'] = calculate_percentile(d['qbr_season'], 'qbr_total')

    if not d['pass'].empty:
        d['pass']['grades_pass_pct'] = calculate_percentile(d['pass'], 'grades_pass')
        d['pass']['grades_run_pct'] = calculate_percentile(d['pass'], 'grades_run')
        d['pass']['btt_pct'] = calculate_percentile(d['pass'], 'btt_rate')
        d['pass']['twp_pct'] = calculate_percentile(d['pass'], 'twp_rate', ascending=False)
        d['pass']['adot_pct'] = calculate_percentile(d['pass'], 'avg_depth_of_target')
        # FIXED: Finds exact mapping for Adjusted Completion Percent
        adj_col = 'adjusted_completion_percent' if 'adjusted_completion_percent' in d['pass'].columns else ('adj_completion_percent' if 'adj_completion_percent' in d['pass'].columns else None)
        if adj_col: d['pass']['adj_comp_pct'] = calculate_percentile(d['pass'], adj_col)
        else: d['pass']['adj_comp_pct'] = 0.0
        d['pass']['p2s_pct'] = calculate_percentile(d['pass'], 'pressure_to_sack_rate', ascending=False)

    if not d['rec'].empty:
        d['rec']['grades_pass_route_pct'] = calculate_percentile(d['rec'], 'grades_pass_route')
        d['rec']['yprr_pct'] = calculate_percentile(d['rec'], 'yprr')
        d['rec']['drop_pct'] = calculate_percentile(d['rec'], 'drop_rate', ascending=False)
        d['rec']['adot_pct'] = calculate_percentile(d['rec'], 'avg_depth_of_target')
        d['rec']['slot_rate_pct'] = calculate_percentile(d['rec'], 'slot_rate')
        d['rec']['wide_rate_pct'] = calculate_percentile(d['rec'], 'wide_rate')
        d['rec']['inline_rate_pct'] = calculate_percentile(d['rec'], 'inline_rate')
        # NOT 'yprr_slot'/'yprr_screen'/'grades_pass_route_slot' here - none
        # of those three columns actually exist in receiving_summary.csv
        # (confirmed by direct inspection of pff['rec'].columns), so this
        # used to silently compute an all-zero percentile column for all
        # three (calculate_percentile returns zeros, not NaN, for a missing
        # column - see its own docstring) with nothing ever reading them
        # correctly. The real slot/screen route-alignment split lives in
        # receiving_concept.csv (pff['route_concept']) under different
        # column names ('slot_yprr'/'screen_yprr'/'slot_grades_pass_route')
        # - ui.player_snapshot.build_player_snapshot's WR/TE branch computes
        # percentiles for those directly off that dataframe instead, so no
        # equivalent columns are needed here.

    if not d['rec_scheme'].empty:
        d['rec_scheme']['man_grades_pass_route_pct'] = calculate_percentile(d['rec_scheme'], 'man_grades_pass_route')
        d['rec_scheme']['zone_grades_pass_route_pct'] = calculate_percentile(d['rec_scheme'], 'zone_grades_pass_route')
        d['rec_scheme']['man_yprr_pct'] = calculate_percentile(d['rec_scheme'], 'man_yprr')
        d['rec_scheme']['zone_yprr_pct'] = calculate_percentile(d['rec_scheme'], 'zone_yprr')
        d['rec_scheme']['contested_pct'] = calculate_percentile(d['rec_scheme'], 'man_contested_catch_rate')

    if not d['rush'].empty:
        # NOT 'missed_tackles_forced' - that column doesn't exist in
        # rushing_summary.csv at all (confirmed by direct inspection); the
        # real column is 'elu_rush_mtf' (PFF's "elusiveness" rushing MTF
        # count - there's a separate 'elu_recv_mtf' for receiving plays,
        # not what this RB rushing stat means). Same silent-all-zero-
        # percentile failure mode as the WR/TE slot/screen columns above -
        # calculate_percentile doesn't raise on a missing column, it just
        # returns zeros, so 'MTF/G' displayed "--" for every single RB,
        # every season, with nothing ever visibly wrong.
        d['rush']['mtf_pct'] = calculate_percentile(d['rush'], 'elu_rush_mtf')
        d['rush']['yco_pct'] = calculate_percentile(d['rush'], 'yco_attempt')

    if not d['pass_block'].empty:
        d['pass_block']['grades_pass_block_pct'] = calculate_percentile(d['pass_block'], 'grades_pass_block')
        d['pass_block']['pbe_pct'] = calculate_percentile(d['pass_block'], 'pbe')
        if 'pressures_allowed' in d['pass_block'].columns and 'player_game_count' in d['pass_block'].columns:
            d['pass_block']['pr_allowed_per_game'] = d['pass_block']['pressures_allowed'] / d['pass_block']['player_game_count'].replace(0, 1)
            d['pass_block']['pr_allowed_per_game_pct'] = calculate_percentile(d['pass_block'], 'pr_allowed_per_game', ascending=False)

    if not d['run_block'].empty:
        d['run_block']['grades_run_block_pct'] = calculate_percentile(d['run_block'], 'grades_run_block')

    if not d['pass_rush'].empty:
        d['pass_rush']['prp_pct'] = calculate_percentile(d['pass_rush'], 'prp')
        d['pass_rush']['win_pct'] = calculate_percentile(d['pass_rush'], 'pass_rush_win_rate')

    if not d['run_def'].empty:
        d['run_def']['stop_pct'] = calculate_percentile(d['run_def'], 'stop_percent')

    if not d['cov_summary'].empty:
        d['cov_summary']['pass_rtg_pct'] = calculate_percentile(d['cov_summary'], 'qb_rating_against', ascending=False)
        d['cov_summary']['mtf_rate_pct'] = calculate_percentile(d['cov_summary'], 'missed_tackle_rate', ascending=False)

    d['pff_grades_map'], d['league_gold_players'] = _build_master_pff_grades(d)
    return d


def load_pff_data_with_fallback(year):
    """
    load_all_pff_data(year), falling back to the most recent PRIOR season
    with real PFF data if the selected year's own export is empty (e.g. an
    upcoming season before its files are uploaded - 2026 pre-kickoff, right
    now). Same "most recent season with data" pattern as the O-line
    pressure/SOS fallbacks in ui/tabs/defensive_yield.py, pulled out here
    once both Depth Charts and Player Search needed it, rather than each
    keeping its own copy of the same loop.

    Returns (pff_dict, source_year) - callers should show a caption when
    source_year != year so it's clear the grades are borrowed from a prior
    season, not the one currently selected. Self-clears the moment real
    files land in pff_imports/{year}/, since load_all_pff_data(year) then
    returns a non-empty grades map and the loop below is simply never
    entered - no separate "has this been uploaded yet" flag to maintain.
    """
    pff = load_all_pff_data(year)
    source_year = year
    if not pff.get('pff_grades_map'):
        for back in range(1, 4):
            prior_year = year - back
            if prior_year < 2019:
                break
            prior_pff = load_all_pff_data(prior_year)
            if prior_pff.get('pff_grades_map'):
                pff = prior_pff
                source_year = prior_year
                break
    return pff, source_year


def _build_master_pff_grades(d):
    pff_frames = []
    for df in [d['rec'], d['pass'], d['rush'], d['block']]:
        if not df.empty and 'player' in df.columns and 'grades_offense' in df.columns:
            cols = ['player', 'grades_offense']
            if 'position' in df.columns: cols.append('position')
            pff_frames.append(df[cols].rename(columns={'grades_offense': 'grade'}))

    if not d['pass_block'].empty and 'player' in d['pass_block'].columns and 'grades_pass_block' in d['pass_block'].columns:
        cols = ['player', 'grades_pass_block']
        if 'position' in d['pass_block'].columns: cols.append('position')
        pff_frames.append(d['pass_block'][cols].rename(columns={'grades_pass_block': 'grade'}))

    for df in [d['pass_rush'], d['run_def'], d['cov_summary'], d['slot_cov']]:
        if not df.empty and 'player' in df.columns and 'grades_defense' in df.columns:
            cols = ['player', 'grades_defense']
            if 'position' in df.columns: cols.append('position')
            pff_frames.append(df[cols].rename(columns={'grades_defense': 'grade'}))

    # K/P grades - these were never wired in, so kickers and punters always
    # showed "N/A" in the depth chart's grade column even though the
    # underlying PFF data was sitting in pff_imports unused.
    if not d['fg'].empty and 'player' in d['fg'].columns and 'grades_fgep_kicker' in d['fg'].columns:
        cols = ['player', 'grades_fgep_kicker']
        if 'position' in d['fg'].columns: cols.append('position')
        pff_frames.append(d['fg'][cols].rename(columns={'grades_fgep_kicker': 'grade'}))
    if not d['punt'].empty and 'player' in d['punt'].columns and 'grades_punter' in d['punt'].columns:
        cols = ['player', 'grades_punter']
        if 'position' in d['punt'].columns: cols.append('position')
        pff_frames.append(d['punt'][cols].rename(columns={'grades_punter': 'grade'}))

    if pff_frames:
        master = pd.concat(pff_frames, ignore_index=True)
        master = master[master['grade'] > 0]

        league_gold_players = set()
        if 'position' in master.columns:
            master_pos = master.dropna(subset=['position'])
            max_grades = master_pos.groupby('position')['grade'].transform('max')
            leaders = master_pos[master_pos['grade'] == max_grades]
            league_gold_players = set(leaders['player'].str.lower())

        master = master.sort_values('grade', ascending=False).drop_duplicates(subset=['player'])
        pff_map = dict(zip(master['player'].str.lower(), master['grade']))
        return pff_map, league_gold_players
    return {}, set()


@st.cache_data
def load_pfr_pass_block(year):
    """
    PFR advanced passing stats (season level) - times_pressured/pressure_pct
    are recorded against the QB, but since pressure allowed is overwhelmingly
    a function of the offensive line in front of him, a team's primary
    starter's pressure_pct is used as a practical team-level pass-block
    quality proxy (there's no free per-lineman pressure-allowed export).
    """
    try:
        df = nflreadpy.load_pfr_advstats([year], stat_type='pass', summary_level='season').to_pandas()
        # PFR uses placeholder codes ("2TM", "3TM") for a player's combined
        # row across multiple teams in-season - not a real team, and would
        # otherwise sometimes outrank the actual starter for team assignment.
        return df[~df['team'].isin(['2TM', '3TM'])] if 'team' in df.columns else df
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_pfr_def_pressure(year):
    """PFR advanced defensive stats (season level) - per-player pressures/sacks/hurries/knockdowns, aggregated by team below into a defense's pressure rate generated."""
    try:
        df = nflreadpy.load_pfr_advstats([year], stat_type='def', summary_level='season').to_pandas()
        return df[~df['tm'].isin(['2TM', '3TM'])] if 'tm' in df.columns else df
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_team_pass_attempts_faced(year):
    """Team-level pass attempts faced (the opponent's own passing attempts each week), used to turn raw defensive pressure counts into a rate."""
    try:
        # 'reg'-level summary collapses to one row per team per season with
        # no opponent_team/game breakdown at all - only the per-game 'week'
        # level carries opponent_team, so that's summed here instead.
        team_stats = nflreadpy.load_team_stats([year], summary_level='week').to_pandas()
        return team_stats.groupby('opponent_team', observed=True)['attempts'].sum()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data
def load_pbp(year):
    """
    Raw play-by-play (nflfastR, via nflreadpy) for one season - carries
    yardline_100/goal_to_go plus EPA/CPOE/WP/success, none of which exist in
    any of the pre-aggregated nflverse exports this app already used
    (player_stats, team_stats, snap_counts). Currently only consumed for
    red-zone usage (data.transforms.build_redzone_usage) - no local export
    anywhere has that at all, so this is a real gap-filler, not a
    duplicate of something already loaded elsewhere.
    """
    try:
        return nflreadpy.load_pbp([year]).to_pandas()
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_team_logos():
    """
    abbr -> ESPN logo URL, from the local teams_colors_logos.csv (nflverse
    export, previously on disk but unused). Same ESPN CDN the app already
    pulls player headshots from, so no new external dependency class -
    just a richer use of an asset that was already here.
    """
    try:
        df = pd.read_csv('teams_colors_logos.csv')
        if 'team_abbr' in df.columns and 'team_logo_espn' in df.columns:
            return dict(zip(df['team_abbr'].astype(str), df['team_logo_espn'].astype(str)))
    except Exception:
        pass
    return {}


@st.cache_data
def load_player_id_crosswalk():
    """
    nflverse's master player table (gsis_id / pff_id / espn_id / pfr_id /
    otc_id / nfl_id per player) - confirmed live against this app's actual
    PFF exports: PFF's own 'player_id' column (e.g. 61398 for Justin
    Jefferson) matches this table's 'pff_id' column exactly, 100% of rows
    in a full receiving_summary export. That means PFF data could in
    principle be joined onto nflverse stats by ID instead of by name string
    - more robust than the two-tier name matching used everywhere else in
    this app (see clean_name_exact/clean_name_for_merge), which has been a
    recurring source of real bugs this project (Byron Murphy vs Byron
    Murphy II, suffix mismatches, etc).

    Not yet wired into the existing PFF lookups - every one of those (~15
    call sites across player_search.py/depth_charts.py) matches by name
    today, and switching them all to ID-based joins is a real refactor in
    its own right, not something to fold into an unrelated feature pass.
    Loaded and cached here so it's available the moment that refactor is
    worth doing.
    """
    try:
        return nflreadpy.load_players().to_pandas()
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_team_pace(year):
    """
    Offensive plays run per game and defensive plays FACED per game, both
    derived from nflreadpy's weekly team stats (attempts + carries +
    sacks_suffered = a team's own offensive snaps that week; the same sum
    grouped by opponent_team instead = plays that team's DEFENSE was on the
    field for) - feeds the matchup projection model's pace adjustment in
    data.transforms.build_player_projection: a defense that faces more
    plays per game gives every offensive skill player more raw opportunities
    regardless of per-play matchup quality, independent of the points/stats-
    allowed signal build_stat_allowed_matrix already captures.

    Returns a DataFrame indexed by team abbreviation with 'off_pace' /
    'def_pace' columns (empty DataFrame if nflreadpy has nothing for this
    year, e.g. a season with no games played yet).
    """
    try:
        df = nflreadpy.load_team_stats([year], summary_level='week').to_pandas()
    except Exception:
        return pd.DataFrame()
    if df.empty or 'team' not in df.columns:
        return pd.DataFrame()
    play_cols = [c for c in ['attempts', 'carries', 'sacks_suffered'] if c in df.columns]
    if not play_cols:
        return pd.DataFrame()
    df = df.copy()
    df['plays'] = df[play_cols].sum(axis=1)

    off = df.groupby('team', observed=True).agg(off_plays=('plays', 'sum'), off_games=('week', 'nunique'))
    off['off_pace'] = off['off_plays'] / off['off_games'].replace(0, 1)

    defn = df.groupby('opponent_team', observed=True).agg(def_plays=('plays', 'sum'), def_games=('week', 'nunique'))
    defn['def_pace'] = defn['def_plays'] / defn['def_games'].replace(0, 1)
    defn.index.name = 'team'

    return pd.concat([off[['off_pace']], defn[['def_pace']]], axis=1)


@st.cache_data
def load_schedule(year):
    """Full season schedule (nflreadpy), REG season only - used to build remaining-schedule strength-of-schedule tables."""
    try:
        df = nflreadpy.load_schedules([year]).to_pandas()
        return df[df['game_type'] == 'REG'].copy()
    except Exception:
        return pd.DataFrame()


_ODDS_API_KEY_FILE = os.path.join('.streamlit', 'odds_api_key.txt')


def load_saved_odds_api_key():
    """
    Reads the API key saved locally by save_odds_api_key() below, so the
    Live Odds tab can pre-fill the input on a fresh app launch instead of
    requiring the user to paste it in every single time. Not cached - this
    is a few bytes of local disk I/O, cheaper than the caching overhead
    itself, and must see a same-run save immediately (see the note in
    ui/tabs/live_odds.py about why value= and key= are combined).
    """
    try:
        if os.path.exists(_ODDS_API_KEY_FILE):
            with open(_ODDS_API_KEY_FILE, 'r', encoding='utf-8') as fh:
                return fh.read().strip()
    except Exception:
        pass
    return ''


def save_odds_api_key(api_key):
    """
    Saves the key in plain text under .streamlit/ (created if needed) -
    fine for a single-user local app the user runs on their own machine,
    but this is NOT encrypted, so this file should be deleted (or this
    whole persistence skipped) before zipping/sharing the project folder
    with anyone else.
    """
    try:
        os.makedirs('.streamlit', exist_ok=True)
        with open(_ODDS_API_KEY_FILE, 'w', encoding='utf-8') as fh:
            fh.write(api_key.strip())
    except Exception:
        pass


@st.cache_data(ttl=900)
def fetch_nfl_odds(api_key, markets='h2h,spreads,totals'):
    """
    Refresh-on-demand design, not an always-on background pipeline: Celery +
    Redis would mean running a message broker and worker processes outside
    the Streamlit process, which is a different deployment model than
    `streamlit run app.py` locally. This gets most of the value (odds that
    don't go stale for more than 15 min while the app is open) with none of
    that infrastructure - cached for 15 minutes, with a manual refresh button
    that busts the cache on demand (see ui/tabs/live_odds.py).
    """
    try:
        resp = requests.get(
            'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds',
            params={'apiKey': api_key, 'regions': 'us', 'markets': markets, 'oddsFormat': 'american'},
            timeout=15
        )
        if resp.status_code == 200:
            return resp.json(), None, resp.headers.get('x-requests-remaining')
        return None, f"API returned {resp.status_code}: {resp.text[:300]}", None
    except Exception as e:
        return None, str(e), None


@st.cache_data(ttl=900)
def fetch_nfl_player_props(api_key, event_id, markets='player_pass_yds,player_rush_yds,player_reception_yds,player_anytime_td'):
    """
    Player props need a per-event query (a different endpoint than the bulk
    odds call above) and cost more credits per request. Live-tested against
    a real key: markets not yet posted by any book for a given game (common
    for games more than a few weeks out) just come back empty rather than
    erroring, so requesting a broad market list (see
    config.ODDS_API_PLAYER_PROP_MARKETS) is safe - whatever's actually
    posted comes back, the rest is silently omitted. Still built to fail
    gracefully with a clear message for genuine errors (bad key, unsupported
    market on your specific plan tier, etc).
    """
    try:
        resp = requests.get(
            f'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds',
            params={'apiKey': api_key, 'regions': 'us', 'markets': markets, 'oddsFormat': 'american'},
            timeout=15
        )
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"API returned {resp.status_code}: {resp.text[:300]}"
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=60)
def fetch_sleeper_draft_picks(draft_id):
    """
    Sleeper's API is fully public and read-only - no API key, no OAuth,
    nothing to register. draft_id is the string of numbers in your draft's
    URL (sleeper.com/draft/nfl/<draft_id>). 60s cache since this is safe to
    poll frequently during an actual live draft (rate limit is a generous
    90 req/min per IP).
    """
    try:
        resp = requests.get(f'https://api.sleeper.app/v1/draft/{draft_id}/picks', timeout=15)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"API returned {resp.status_code}: {resp.text[:300]}"
    except Exception as e:
        return None, str(e)
