"""NFL Depth Charts tab: intelligent roster synthesizer per team."""
import pandas as pd
import streamlit as st

from config import MASTER_TEAMS_LIST, TEAM_CONFIG, TAB_PLAYER_SEARCH, AVAILABLE_SEASONS_WITH_UPCOMING
from data.transforms import load_and_merge_data, fetch_intelligent_depth_chart
from data.loaders import build_veteran_database, load_pff_data_with_fallback
from data.utils import clean_name_exact
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

    dc_df = fetch_intelligent_depth_chart(sel_team_t2, df_t2_stats, pff['rec'], t2_target_year)

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
