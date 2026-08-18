"""Risers / Waiver Wire tab: biggest week-over-week percentile jumps."""
import streamlit as st

from config import TAB_PLAYER_SEARCH, AVAILABLE_SEASONS_WITH_UPCOMING
from data.transforms import load_and_merge_data, build_risers_report, build_recent_trend
from data.utils import calculate_percentile_qualified
from ui.styling import style_plain_dataframe, df_auto_height, build_column_help_config
from ui.components import switch_tab, skeleton_loader


def render():
    st.markdown("<div class='custom-section-header'>RISERS &amp; WAIVER WIRE — BIGGEST WEEK-OVER-WEEK JUMPS</div>", unsafe_allow_html=True)
    # index=1, not 0 - same convention as Depth Charts/Player Search/Player
    # Compare: the upcoming season is selectable but isn't the default
    # landing view while it has no played games yet to compute a
    # week-over-week jump from.
    t4_year = st.selectbox("Season", AVAILABLE_SEASONS_WITH_UPCOMING, index=1, key="year_tab4")
    with skeleton_loader("table", n_rows=10, n_cols=7):
        df_t4_stats, t4_t_col, t4_n_col, _ = load_and_merge_data(t4_year, "Full PPR")
    risers = build_risers_report(df_t4_stats, t4_n_col, t4_t_col, t4_year)
    if not risers.empty:
        jump_pct = calculate_percentile_qualified(
            risers, 'Pct Jump', position_col='Pos', snap_col='Season Snap %', group_by_position=False
        )
        indexed = risers.set_index('Player')

        trend_map = build_recent_trend(df_t4_stats, t4_n_col, metric='fantasy_points', n_weeks=5)
        indexed['Last 5 Wks'] = [trend_map.get(name, []) for name in indexed.index]
        column_config = build_column_help_config(indexed, pinned_cols=['Team', 'Pos'], meter_cols={'Season Snap %': (0, 100)})
        column_config['Last 5 Wks'] = st.column_config.LineChartColumn(
            help="Fantasy points trend over this player's last 5 games played", width="small",
        )

        def _on_row_select():
            # Must run as an on_select callback, not a direct call after
            # checking the widget's return value - by the time this tab's
            # render() body executes, app.py's st.tabs(key="active_tab") has
            # already been instantiated this run, and Streamlit only allows
            # assigning to a keyed widget's session_state from a callback
            # (which runs in its own pre-script phase), not from later in
            # the same script pass. This also means it only fires on an
            # actual NEW selection - no separate "don't re-fire on every
            # later visit" guard needed like a direct-call version would.
            rows = st.session_state.get('risers_table', {}).get('selection', {}).get('rows', [])
            if rows:
                switch_tab(TAB_PLAYER_SEARCH, jump_to_player=indexed.index[rows[0]], jump_to_year=t4_year)

        st.dataframe(
            style_plain_dataframe(indexed, numeric_pct_cols={'Pct Jump': jump_pct}),
            width="stretch", height=df_auto_height(min(len(risers), 40)),
            on_select=_on_row_select, selection_mode="single-row", key="risers_table",
            column_config=column_config,
        )
    else:
        st.info("Not enough weekly data yet this season to compute risers (need at least 2 prior weeks plus a most-recent week).")
