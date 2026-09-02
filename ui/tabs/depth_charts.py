"""NFL Depth Charts tab: intelligent roster synthesizer per team."""
import pandas as pd
import streamlit as st

from config import (MASTER_TEAMS_LIST, TEAM_CONFIG, TAB_PLAYER_SEARCH,
                    AVAILABLE_SEASONS, AVAILABLE_SEASONS_WITH_UPCOMING)
from data.transforms import load_and_merge_data, fetch_intelligent_depth_chart
from data.loaders import build_veteran_database, load_pff_data_with_fallback
from data.utils import clean_name_exact, clean_name_for_merge
from data.weekly_projections import (resolve_preseason_qb1s, resolve_inseason_qb1s,
                                     save_qb1_override, clear_qb1_override,
                                     _all_played_weeks, _invalidate_weekly_projection_cache)
from data.ourlads_depth_charts import (
    load_ourlads_snapshot, save_ourlads_snapshot, save_ourlads_snapshot_from_dir,
    load_ourlads_key, save_ourlads_key, build_ourlads_projection_signal,
    OURLADS_INBOX_DIR,
)
from ui.styling import style_depth_chart_table, df_auto_height
from ui.components import switch_tab, render_team_banner, skeleton_loader


@st.cache_data
def _build_snap_totals(_df, year, team):
    """
    Per-player snap lookup (exact_name -> snap_pct_avg / games_played /
    has_snap_match) for the depth-chart cell text. Cached on `year`+`team`
    (`_df` unhashed, same convention as every other cached transform here).

    FILTERED TO ONE TEAM on purpose, not league-wide: the lookup key is
    the abbreviated "F.Last" cell name, which nflverse only guarantees
    unique WITHIN a team (its prefix-lengthening disambiguation - see
    HANDOFF gotcha #12 - is per-roster). A league-wide map silently merged
    two different same-abbreviation players from different teams into one
    entry - confirmed real: NYJ's "B.Cook" showed a 100% snap figure
    borrowed from another team's B.Cook before this filter existed.

    The DISPLAYED number is the player's share of the TEAM's season -
    his average snap rate while active, scaled by the fraction of the
    team's season he actually appeared for - NOT the median of his own
    played games that fetch_intelligent_depth_chart ranks by. The two
    answer different questions: median-of-own-games is right for ORDERING
    (an injured starter shouldn't rank below a healthy backup - the Josh
    Sweat case in that function's docstring), but as a displayed "Snap %"
    a plain per-game average produced two QBs both reading "100%" whenever
    a starter got hurt mid-season (each really did play ~100% of his own
    starts - ARI's Murray/Brissett, ATL's Penix/Cousins, 4 more teams in
    2025 alone), which reads as impossible. Season share sums to ~100%
    within a position group, the way every pro site's snap-share column
    reads; the games count is surfaced alongside it in the cell so the
    ordering (still median-based) stays explainable.

    Computed as snap_pct_avg * snap_games_played / team_weeks - NOT by
    summing weekly_snap_pct rows in `_df` (an earlier version of this
    function did exactly that, and it looked right for skill positions but
    silently undercounted every offensive lineman to roughly a third of
    their real share). weekly_snap_pct only exists on a ROW where
    stats_player_week_{year}.csv itself has an entry for that player-week,
    and nflverse's weekly box score is stat-triggered, not full-roster -
    confirmed real: Kingsley Suamataia (KC, 2025) has exactly 5 rows in
    that file all season despite playing the large majority of KC's
    offensive snaps nearly every week, so summing over only those 5 rows
    (divided by ~17 team weeks) produced ~26% instead of something close to
    his real ~98% per-game rate. snap_pct_avg/snap_games_played
    (data.loaders.load_year_data) are computed straight from the snap
    source's own per-week columns instead, unaffected by that gap.

    Groups by a key derived from the SAME 'player_name' column
    fetch_intelligent_depth_chart prefers (roster_weekly's abbreviated
    "P.Mahomes" convention), NOT the pre-existing full-name-derived
    'exact_name' column - those are different strings for the same player
    ("pmahomes" vs "patrickmahomes"), and grouping by the wrong one made
    every cell show "Snap: 0%" once before (see HANDOFF).

    IMPORTANT: only trusts weekly_snap_pct when 'week' actually exists in
    the frame - for a year with no weekly stats file yet (e.g. an upcoming
    season), load_year_data() sets weekly_snap_pct to a uniform placeholder
    0.0 (not NaN), so the fallback branch below uses the prior-season
    snap_pct_avg instead, same as before.
    """
    t_col = 'recent_team' if 'recent_team' in _df.columns else ('team' if 'team' in _df.columns else None)
    if t_col is None:
        return pd.DataFrame()
    df = _df[_df[t_col].astype(str).str.upper() == str(team).upper()]
    if df.empty:
        return pd.DataFrame()
    dc_name_col = 'player_name' if 'player_name' in df.columns else 'exact_name'
    if dc_name_col == 'player_name':
        df = df.copy()
        df['dc_exact_name'] = clean_name_exact(df['player_name'])
        dc_name_col = 'dc_exact_name'

    has_real_weekly = dc_name_col in df.columns and 'week' in df.columns and 'snap_pct_avg' in df.columns and 'snap_games_played' in df.columns
    if has_real_weekly:
        team_weeks = max(int(pd.to_numeric(df['week'], errors='coerce').dropna().nunique()), 1)
        agg_spec = {
            'snap_avg_season': ('snap_pct_avg', 'max'),
            'snap_games': ('snap_games_played', 'max'),
            # Stats-row week count kept only as a secondary/fallback games-
            # played display number (see the max() below) - snap_games is
            # the real, complete one wherever the snap source has it.
            'stats_games': ('week', 'nunique'),
        }
        if 'has_snap_match' in df.columns:
            agg_spec['has_snap_match'] = ('has_snap_match', 'max')
        totals = df.groupby(dc_name_col, observed=True).agg(**agg_spec).reset_index()
        totals['snap_pct_avg'] = (totals['snap_avg_season'] * totals['snap_games'] / team_weeks).clip(upper=100.0)
        totals['games_played'] = totals[['snap_games', 'stats_games']].max(axis=1)
        totals = totals.drop(columns=['snap_avg_season', 'snap_games', 'stats_games'])
    elif dc_name_col in df.columns and 'snap_pct_avg' in df.columns:
        agg_spec = {'snap_pct_avg': ('snap_pct_avg', 'max')}
        if 'has_snap_match' in df.columns:
            agg_spec['has_snap_match'] = ('has_snap_match', 'max')
        totals = df.groupby(dc_name_col, observed=True).agg(**agg_spec).reset_index()
        totals['games_played'] = 0
    else:
        totals = pd.DataFrame()
    return totals.rename(columns={dc_name_col: 'exact_name'})


def _make_cell_click_callback(source_df, table_key, year, team):
    """
    Depth chart cells are one PLAYER PER CELL across several slot columns
    (Starter/2nd String/.../6th String) in the same row, unlike Risers/
    Rookie Watch where each ROW is already one player - so this needs
    single-cell selection (row, column name) rather than row selection, to
    know exactly which of the several players in that row was clicked.

    Returns a closure suitable for on_select=... (must be a callback, not a
    direct call after checking the widget's return value - see the matching
    comment in ui/tabs/risers.py for why). st.dataframe's on_select doesn't
    take args/kwargs like st.button does, so source_df/table_key/year/team
    are captured via closure instead of passed in at call time.

    Passes jump_to_year=year so Player Search opens on the SAME season being
    viewed here - a click from the 2022 Depth Chart used to always land on
    whatever year Player Search last happened to show (2025 by default),
    not 2022. Also passes jump_to_team=team: this table's cell text is
    roster_weekly's abbreviated "F.Last" name, which Player Search resolves
    via an (first-initial, last-name) bridge - ambiguous LEAGUE-WIDE (e.g.
    "Mi.Wilson" also matches an unrelated "Mack Wilson" elsewhere in the
    league), so narrowing Player Search to the player's own team before
    that match runs is what actually makes it land on the right person.
    """
    def _on_cell_select():
        cells = st.session_state.get(table_key, {}).get('selection', {}).get('cells', [])
        if not cells:
            return
        row_idx, col_name = cells[0]
        if col_name == 'Position' or row_idx >= len(source_df):
            return
        clicked_player = str(source_df.iloc[row_idx][col_name]).strip()
        if clicked_player:
            switch_tab(TAB_PLAYER_SEARCH, jump_to_player=clicked_player, jump_to_year=year, jump_to_team=team)
    return _on_cell_select


def _render_projection_qb1_control(year, team, current_stats, team_col, name_col):
    """Small, explicit upcoming-game QB1 control for weekly projections.

    The generated chart below remains a roster-browsing aid.  It is not a
    source of truth for a new-season QB starter because its ordering comes
    from historical snaps.  This control instead records a named selection
    whenever current evidence cannot identify one clear starter.
    """
    required = {name_col, team_col, 'position'}
    if current_stats.empty or not required.issubset(current_stats.columns):
        return
    qbs = current_stats[
        current_stats[team_col].astype(str).str.upper().eq(str(team).upper())
        & current_stats['position'].astype(str).str.upper().eq('QB')
    ][[name_col, team_col]].dropna(subset=[name_col]).drop_duplicates(subset=[name_col])
    if qbs.empty:
        return
    try:
        prior_stats, prior_team_col, prior_name_col, _ = load_and_merge_data(year - 1, 'Full PPR')
    except Exception:
        prior_stats, prior_team_col, prior_name_col = pd.DataFrame(), team_col, name_col
    observed_weeks = pd.to_numeric(current_stats.get('week', pd.Series(dtype=float)), errors='coerce')
    latest_observed = observed_weeks[observed_weeks > 0].max() if (observed_weeks > 0).any() else None
    current_qbs = current_stats[current_stats['position'].astype(str).str.upper().eq('QB')]
    if latest_observed is not None:
        resolution = resolve_inseason_qb1s(
            current_qbs, name_col, team_col, current_stats, name_col, team_col,
            int(latest_observed) + 1, year,
        )
    else:
        ourlads_qb1s = pd.DataFrame()
        snapshot, snapshot_problem = load_ourlads_snapshot(year)
        if snapshot_problem is None and not snapshot.empty:
            signal = build_ourlads_projection_signal(snapshot, current_stats, name_col, team_col)
            ourlads_qb1s = signal['qb_starters']
        resolution = resolve_preseason_qb1s(
            current_qbs, name_col, team_col, _all_played_weeks(prior_stats), prior_name_col, prior_team_col, year,
            ourlads_qb1s=ourlads_qb1s,
        )
    status = resolution['by_team'].get(str(team).upper(), {})
    player_options = qbs[name_col].astype(str).sort_values().tolist()
    selected = status.get('player')
    selected_index = player_options.index(selected) if selected in player_options else 0

    with st.expander('Weekly projection QB1 selection', expanded=status.get('status') == 'selection_required'):
        st.caption(
            'The generated chart below is not the weekly QB1 source. A locally imported Ourlads preseason '
            'chart can resolve a healthy listed QB1; a manual selection always wins; otherwise a clear '
            'incumbent or explicit choice is required.'
        )
        if status.get('status') == 'manual_override':
            st.success(f"Manual selection: {selected}. This player receives the full projected QB workload.")
        elif status.get('status') == 'prior_season_incumbent':
            st.info(
                f"Automatic incumbent: {selected} ({status.get('prior_snap_share', 0):.0%} prior-season snap share). "
                'You can still replace this if current preseason information changes the starter.'
            )
        elif status.get('status') == 'ourlads_depth_chart':
            st.info(
                f"Imported Ourlads QB1: {selected} (source slot {status.get('source_slot', 1)}). "
                'This establishes expected starter eligibility only; the projection still models his stats independently.'
            )
        elif status.get('status') == 'observed_current_starter':
            st.info(
                f"Automatic in-season starter: {selected} "
                f"({status.get('recent_snap_share', 0):.0%} recent snap share)."
            )
        else:
            st.warning('No unambiguous QB starter. Select the expected QB1 before relying on this room’s projection.')
        if resolution['warnings']:
            for warning in resolution['warnings']:
                if str(team).upper() in warning:
                    st.caption(f'⚠️ {warning}')

        choice = st.selectbox(
            'Expected upcoming-game QB1', player_options, index=selected_index,
            key=f'projection_qb1_choice_{year}_{team}',
        )
        save_col, clear_col = st.columns(2)
        with save_col:
            if st.button('Save QB1 selection', key=f'projection_qb1_save_{year}_{team}'):
                try:
                    save_qb1_override(year, team, choice)
                    st.success(f'Saved {choice} as {team} QB1 for {year}. Weekly projections will refresh now.')
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        with clear_col:
            if status.get('status') == 'manual_override' and st.button(
                    'Clear manual selection', key=f'projection_qb1_clear_{year}_{team}'):
                try:
                    clear_qb1_override(year, team)
                    st.success('Cleared the manual selection. The automatic-incumbent rule will apply if available.')
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))


def _ourlads_rank_map_for(snapshot, team):
    """
    {clean_name_exact(player) -> listed slot} for one team's offense-skill
    rows in an imported Ourlads snapshot, lowest = starter. Feeds
    fetch_intelligent_depth_chart's preseason ordering override. Multiple
    alignment labels roll up to one fantasy position (LWR/RWR/SWR -> WR), so
    a player keeps his BEST (lowest) listed slot across them.
    """
    if snapshot is None or snapshot.empty or 'player' not in snapshot.columns:
        return {}
    rows = snapshot[(snapshot['team'].astype(str) == str(team))
                    & snapshot['position'].astype(str).str.upper().isin(
                        ['QB', 'RB', 'FB', 'WR', 'TE'])]
    if rows.empty:
        return {}
    slot = (pd.to_numeric(rows.get('source_slot'), errors='coerce')
            .fillna(pd.to_numeric(rows.get('depth_rank'), errors='coerce'))
            .fillna(9.0))
    players = rows['player'].astype(str)
    out = {}
    # BOTH key forms, because the two sources spell suffixes differently and
    # clean_name_exact deliberately PRESERVES them: Ourlads writes "GARDNER
    # MINSHEW II" / "Marvin Harrison Jr." / "DEEBO SAMUEL SR." where the
    # roster has the bare name, so an exact-only map silently dropped every
    # suffixed player from the ordering (confirmed: ARI's QB2/QB3 came out
    # reversed because Minshew never matched). The loose key strips
    # suffixes/punctuation and is what bridges them; it is written first so
    # an exact hit always wins where both exist.
    for keys, series in ((clean_name_for_merge(players), slot),
                         (clean_name_exact(players), slot)):
        for key, s in zip(keys, series):
            if key and (key not in out or s < out[key]):
                out[key] = float(s)
    return out


_OURLADS_STATUS_LABELS = {
    'lc_gold': '2026 acquisition',
    'lc_purple': '2026 rookie pick',
    'lc_aqua': '2026 UDFA',
    'lc_red': 'injured / inactive',
}


def _ourlads_status_label(value):
    values = str(value or '').split()
    return ', '.join(_OURLADS_STATUS_LABELS.get(item, item) for item in values if item) or '—'


def _render_ourlads_import_control(year, team):
    """Local Ourlads import/status panel; no live request or scraping path."""
    snapshot, problem = load_ourlads_snapshot(year)
    with st.expander('Ourlads preseason depth-chart signal', expanded=False):
        st.caption(
            'Import your saved printer-friendly Ourlads MHTML team pages here. The app stores a local normalized '
            'snapshot—not the raw browser files—and never logs in to or scrapes Ourlads.'
        )
        st.caption(
            'One import, two consumers: this same snapshot orders the depth-chart tables above for the current '
            "season AND feeds the weekly model's preseason cold-start role signal (QB1 lock + RB/WR/TE role "
            'floors) — see `weekly_projections.build_weekly_projections`.'
        )
        if problem:
            st.error(problem)
        elif snapshot.empty:
            st.info(f'No local Ourlads snapshot is imported for {year} yet.')
        else:
            teams = sorted(snapshot['team'].drop_duplicates().tolist())
            missing = sorted(set(MASTER_TEAMS_LIST) - set(teams))
            updated = [value for value in snapshot['source_updated_at'].dropna().astype(str).unique() if value]
            st.success(f'Loaded {len(snapshot):,} skill-position rows for {len(teams)}/32 teams.')
            if updated:
                st.caption(f"Source update shown in export: {', '.join(updated[:3])}{' …' if len(updated) > 3 else ''}")
            if missing:
                st.warning(f"No imported Ourlads page for: {', '.join(missing)}. Those teams fall back to the regular preseason logic.")
            team_rows = snapshot[snapshot['team'].eq(team)].copy()
            if not team_rows.empty:
                team_rows['Status'] = team_rows['status_class'].map(_ourlads_status_label)
                display = team_rows[[
                    'position_label', 'depth_rank', 'player', 'Status', 'source_file',
                ]].rename(columns={
                    'position_label': 'Formation / Pos', 'depth_rank': 'Chart slot',
                    'player': 'Ourlads player', 'source_file': 'Source export',
                })
                st.dataframe(display, hide_index=True, width='stretch', height=df_auto_height(min(len(display), 16)))

        def _report_ourlads_import(imported, report):
            if report.get('error'):
                st.error(report['error'])
                return
            _invalidate_weekly_projection_cache()
            # The depth-chart tables are a second consumer of this snapshot
            # now - drop their caches too so a fresh import redraws without a
            # manual rerun.
            fetch_intelligent_depth_chart.clear()
            _build_snap_totals.clear()
            batch = report.get('batch_teams') or []
            carried = report.get('carried_teams') or []
            st.success(
                f"Imported {len(imported):,} rows for {len(batch)} team(s) in this batch "
                f"({', '.join(batch)}). Snapshot now covers {report['team_count']}/32 teams.")
            if carried:
                st.caption(
                    f"Kept from earlier imports: {', '.join(carried)}. Batches accumulate — "
                    "a team is only replaced when a newer page for it is imported.")
            if report.get('missing_teams'):
                st.warning(
                    f"Still to import ({len(report['missing_teams'])}): "
                    f"{', '.join(report['missing_teams'])}. Import the rest in further "
                    "batches; nothing already loaded is lost.")

            # File-level accounting, shown whenever anything did not make it
            # through. Without this a batch where the browser silently dropped
            # most of the upload is indistinguishable from one where every page
            # failed to parse - the difference decides whether to retry the
            # upload or fix the pages, so it must not be buried.
            received = report.get('files_received', 0)
            parsed = report.get('files_parsed', 0)
            if received and parsed < received:
                st.error(
                    f"{received} file(s) reached the app, {parsed} parsed. "
                    f"{received - parsed} failed - details below.")
            elif received:
                st.caption(f"{received} file(s) reached the app, all {parsed} parsed.")
            if received and received < len(MASTER_TEAMS_LIST) and not report.get('carried_teams'):
                st.info(
                    f"Only {received} file(s) arrived. If you selected more than that, the "
                    "browser upload dropped the rest - a full league is ~74MB. Use the "
                    "`ourlads_inbox/` folder + the import button above instead; it reads "
                    "from disk and has no upload limit.")
            for unreadable in report.get('unreadable_files', []):
                st.caption(f"⚠️ Skipped {unreadable['source_file']}: {unreadable['error']}")
            dupes = report.get('duplicate_teams') or []
            if dupes:
                st.caption(
                    f"Multiple pages for {', '.join(dupes)} - kept the one with the newest "
                    "'Updated' timestamp. If several pages resolved to the SAME team by "
                    "mistake, that is why fewer teams landed than files sent.")
            archive = report.get('archive') or {}
            if archive.get('archived_csv') or archive.get('archived_pages'):
                st.caption(
                    f"Archived the previous snapshot"
                    f"{' + ' + str(archive['archived_pages']) + ' raw page(s)' if archive.get('archived_pages') else ''}"
                    f" to {archive.get('archive_dir')}.")
            st.rerun()

        _inbox_pages = sorted(
            p.name for p in OURLADS_INBOX_DIR.glob('*')
            if p.is_file() and p.suffix.lower() in ('.mhtml', '.mht', '.html'))
        st.caption(
            f"Drop saved team pages in `external_data/ourlads_inbox/` "
            f"({len(_inbox_pages)} there now) and import them here, or upload them below.")
        if st.button(f'Import from ourlads_inbox/ ({len(_inbox_pages)} files)',
                     key=f'ourlads_import_inbox_{year}', disabled=not _inbox_pages):
            imported, report = save_ourlads_snapshot_from_dir(OURLADS_INBOX_DIR, year=year)
            _report_ourlads_import(imported, report)

        uploads = st.file_uploader(
            'Saved Ourlads team pages (.mhtml)', type=['mhtml', 'mht', 'html'],
            accept_multiple_files=True, key=f'ourlads_pages_{year}',
        )
        key_upload = st.file_uploader(
            'Optional Ourlads depth-chart key (.mhtml)', type=['mhtml', 'mht', 'html'],
            key=f'ourlads_key_{year}',
        )
        import_col, key_col = st.columns(2)
        with import_col:
            if st.button('Import Ourlads pages', key=f'ourlads_import_{year}', disabled=not uploads):
                imported, report = save_ourlads_snapshot(uploads, year)
                _report_ourlads_import(imported, report)
        with key_col:
            if st.button('Save Ourlads key', key=f'ourlads_key_save_{year}', disabled=key_upload is None):
                _, error = save_ourlads_key(key_upload)
                if error:
                    st.error(error)
                else:
                    st.success('Saved a local plain-text Ourlads key for reference.')
                    st.rerun()
        key_text, key_problem = load_ourlads_key()
        if key_problem:
            st.caption(key_problem)
        elif key_text:
            with st.expander('Saved Ourlads key / legend'):
                st.text(key_text[:8000])
        st.caption(
            'Model use: a healthy imported QB can identify the Week 1 QB1; RB/WR/TE entries only provide a '
            'conservative low-evidence role floor. Formation starters such as LWR/RWR/SWR never imply equal or full snaps.'
        )


def render():
    st.markdown("<div class='custom-section-header'>NFL DEPTH CHARTS</div>", unsafe_allow_html=True)
    c_year, c_team = st.columns(2)
    with c_year:
        t2_target_year = st.selectbox("Season", AVAILABLE_SEASONS_WITH_UPCOMING, index=1, key="year_tab2")
    with c_team:
        sel_team_t2 = st.selectbox("Team", MASTER_TEAMS_LIST, key="team_sel_t2",
                                   format_func=lambda a: f"{TEAM_CONFIG.get(a, {}).get('name', a)} ({a})")
    pff, pff_source_year = load_pff_data_with_fallback(t2_target_year)
    if pff_source_year != t2_target_year:
        st.caption(f"⚠️ No PFF grades uploaded yet for {t2_target_year} - showing {pff_source_year} grades until real data is added to pff_imports/{t2_target_year}/.")

    with skeleton_loader("table", n_rows=12, n_cols=6):
        df_t2_stats, t2_t_col, t2_n_col, global_rookie_names = load_and_merge_data(t2_target_year, "Full PPR")

    snap_year_used = int(df_t2_stats['snap_data_year'].iloc[0]) if 'snap_data_year' in df_t2_stats.columns and not df_t2_stats.empty else t2_target_year
    if snap_year_used != t2_target_year:
        st.caption(f"⚠️ No snap-count data for {t2_target_year} yet — snap % shown below is the {snap_year_used} season average.")

    t2_player_totals = _build_snap_totals(df_t2_stats, t2_target_year, sel_team_t2)
    veteran_names = build_veteran_database(t2_target_year)

    render_team_banner(sel_team_t2, subtitle=f"{t2_target_year} depth chart — click any player to open their full profile")
    _render_ourlads_import_control(t2_target_year, sel_team_t2)
    _render_projection_qb1_control(
        t2_target_year, sel_team_t2, df_t2_stats, t2_t_col, t2_n_col,
    )

    # For an upcoming/just-started season (no real snap data yet) the depth
    # chart's ordering falls back to last season's snap averages, which is
    # noisy for a team with roster turnover. Hand fetch_intelligent_depth_chart
    # the imported Ourlads chart so the starter/backup ORDER matches it; the
    # PFF grade / snap % / GP shown in each cell stay each player's own data.
    # Past seasons pass nothing - their full-season snaps are authoritative.
    ourlads_rank_map, ourlads_sig = {}, ""
    if t2_target_year not in AVAILABLE_SEASONS:
        _ol_snap, _ol_problem = load_ourlads_snapshot(t2_target_year)
        if _ol_problem is None and not _ol_snap.empty:
            ourlads_rank_map = _ourlads_rank_map_for(_ol_snap, sel_team_t2)
            ourlads_sig = str(sorted(ourlads_rank_map.items()))

    dc_df = fetch_intelligent_depth_chart(
        sel_team_t2, df_t2_stats, pff['rec'], t2_target_year,
        _ourlads_rank_map=ourlads_rank_map or None, ourlads_sig=ourlads_sig)

    if not dc_df.empty:
        snap_map = dict(zip(t2_player_totals['exact_name'], t2_player_totals.get('snap_pct_avg', pd.Series(0)))) if not t2_player_totals.empty else {}
        has_match_map = dict(zip(t2_player_totals['exact_name'], t2_player_totals['has_snap_match'])) if not t2_player_totals.empty and 'has_snap_match' in t2_player_totals.columns else {}
        games_map = dict(zip(t2_player_totals['exact_name'], t2_player_totals['games_played'])) if not t2_player_totals.empty and 'games_played' in t2_player_totals.columns else {}

        break_idx = dc_df.index[dc_df['Position'] == 'BREAK']
        if len(break_idx) > 0:
            split_at = dc_df.index.get_loc(break_idx[0])
            offense_df = dc_df.iloc[:split_at]
            defense_df = dc_df.iloc[split_at + 1:]
        else:
            offense_df, defense_df = dc_df, dc_df.iloc[0:0]

        st.markdown("<div class='custom-section-header'>OFFENSE</div>", unsafe_allow_html=True)
        if ourlads_rank_map:
            st.caption(f"ℹ️ Starter order for {t2_target_year} follows the imported Ourlads depth chart ({len(ourlads_rank_map)} skill players matched). PFF grade, snap % and games played are still each player's own {snap_year_used} data.")
        st.caption("Each cell: **Player · PFF grade · Snap % (games played)** — \"RK\" in place of a grade means a rookie with no PFF grade yet, \"--\" means neither a rookie nor graded (common for deep backups).")
        st.caption("Snap % shows \"N/A\" (not \"0%\") for players no source has a snap-count row for at all — mostly deep backups with too little playing time to be tracked. Ordering for those rows leans on draft capital and experience instead.")
        st.dataframe(
            style_depth_chart_table(offense_df, sel_team_t2, snap_map, pff['pff_grades_map'], global_rookie_names, veteran_names, pff['league_gold_players'], has_match_map, games_map),
            width="stretch", hide_index=True, height=df_auto_height(len(offense_df)),
            on_select=_make_cell_click_callback(offense_df, 'depth_chart_offense_table', t2_target_year, sel_team_t2),
            selection_mode="single-cell", key="depth_chart_offense_table",
        )
        if not defense_df.empty:
            st.markdown("<div class='custom-section-header'>DEFENSE</div>", unsafe_allow_html=True)
            st.dataframe(
                style_depth_chart_table(defense_df, sel_team_t2, snap_map, pff['pff_grades_map'], global_rookie_names, veteran_names, pff['league_gold_players'], has_match_map, games_map),
                width="stretch", hide_index=True, height=df_auto_height(len(defense_df)),
                on_select=_make_cell_click_callback(defense_df, 'depth_chart_defense_table', t2_target_year, sel_team_t2),
                selection_mode="single-cell", key="depth_chart_defense_table",
            )
    else:
        st.error(f"Failed to load team data. Ensure your local files are correctly loaded.")
