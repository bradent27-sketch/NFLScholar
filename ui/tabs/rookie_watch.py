"""Rookie Watch tab: rookie leaderboard by average fantasy output."""
import streamlit as st

from config import TAB_PLAYER_SEARCH, AVAILABLE_SEASONS_WITH_UPCOMING
from data.transforms import load_and_merge_data, build_rookie_watch, build_recent_trend
from data.utils import calculate_percentile, clean_name_exact
from ui.styling import style_plain_dataframe, df_auto_height, build_column_help_config
from ui.components import switch_tab, get_drafted_players_clean_keys


def render():
    st.markdown("<div class='custom-section-header'>ROOKIE WATCH LEADERBOARD</div>", unsafe_allow_html=True)
    t5_year = st.selectbox("Season", AVAILABLE_SEASONS_WITH_UPCOMING, index=1, key="year_tab5")
    with st.spinner(f"Loading {t5_year} season data..."):
        df_t5_stats, t5_t_col, t5_n_col, t5_rookie_names = load_and_merge_data(t5_year, "Full PPR")
    rookie_board = build_rookie_watch(df_t5_stats, t5_n_col, t5_t_col, t5_rookie_names, t5_year)

    drafted_keys = get_drafted_players_clean_keys()
    if drafted_keys and not rookie_board.empty:
        name_keys = clean_name_exact(rookie_board['Player'])
        hidden_count = int(name_keys.isin(drafted_keys).sum())
        if hidden_count:
            st.caption(f"🚫 {hidden_count} rookie(s) marked drafted on the VORP sheet - hidden below.")
            rookie_board = rookie_board[~name_keys.isin(drafted_keys)]

    if not rookie_board.empty:
        pct_cols = {}
        if 'Avg FPTS' in rookie_board.columns:
            pct_cols['Avg FPTS'] = calculate_percentile(rookie_board, 'Avg FPTS')
        indexed = rookie_board.set_index('Player')

        trend_map = build_recent_trend(df_t5_stats, t5_n_col, metric='fantasy_points', n_weeks=5)
        indexed['Last 5 Wks'] = [trend_map.get(name, []) for name in indexed.index]
        meter_cols = {'Snap %': (0, 100)} if 'Snap %' in indexed.columns else {}
        column_config = build_column_help_config(indexed, pinned_cols=['Team', 'Pos'], meter_cols=meter_cols)
        column_config['Last 5 Wks'] = st.column_config.LineChartColumn(
            help="Fantasy points trend over this rookie's last 5 games played", width="small",
        )

        def _on_row_select():
            # Must run as an on_select callback - see the matching comment
            # in ui/tabs/risers.py for why a direct call after checking the
            # widget's return value raises a StreamlitAPIException here.
            rows = st.session_state.get('rookie_watch_table', {}).get('selection', {}).get('rows', [])
            if rows:
                switch_tab(TAB_PLAYER_SEARCH, jump_to_player=indexed.index[rows[0]], jump_to_year=t5_year)

        st.dataframe(
            style_plain_dataframe(indexed, numeric_pct_cols=pct_cols),
            width="stretch", height=df_auto_height(min(len(rookie_board), 40)),
            on_select=_on_row_select, selection_mode="single-row", key="rookie_watch_table",
            column_config=column_config,
        )
    else:
        st.info(f"No rookie data found for {t5_year}.")
