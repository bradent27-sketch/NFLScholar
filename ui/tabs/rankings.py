"""
Weekly Rankings tab: import a WEEKLY rankings export (same column schema as
FantasyPros' draft-rankings file, just a different week's snapshot - RK,
PLAYER NAME, TEAM, POS, etc) and compare it against this app's own
recent-form ranking (average fantasy points over each player's last few
games played).

A weekly ranking answers "who's producing right now" - a different
question than the season-long draft-value ranking the VORP Draft Sheet
tab already compares against FantasyPros' DRAFT rankings. Keeping the two
comparisons on separate tabs (rather than one tab trying to serve both)
means each one's internal baseline actually matches what it's being
compared against.
"""
import streamlit as st

from config import AVAILABLE_SEASONS
from data.transforms import load_and_merge_data, build_recent_form_rank
from data.rankings import parse_fantasypros_upload, parse_custom_rankings, build_rankings_comparison
from ui.styling import style_plain_dataframe, df_auto_height, build_column_help_config
from ui.components import position_filter_multiselect, skeleton_loader, import_hint


def render():
    st.markdown("<div class='custom-section-header'>WEEKLY RANKINGS</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        wk_year = st.selectbox("Season", AVAILABLE_SEASONS, index=0, key="weekly_rank_year")
    with c2:
        wk_scoring = st.selectbox("Scoring", ["Full PPR", "Half-PPR", "Standard"], key="weekly_rank_scoring")
    with c3:
        n_weeks = st.number_input("Recent-form window (games)", min_value=1, max_value=8, value=4, key="weekly_rank_window")

    with skeleton_loader("table", n_rows=10, n_cols=7):
        df_stats, t_col, n_col, _ = load_and_merge_data(wk_year, wk_scoring)

    form_df = build_recent_form_rank(df_stats, n_col, t_col, n_weeks=n_weeks)
    if form_df.empty:
        st.info(f"Not enough {wk_year} weekly data yet to build a recent-form baseline.")
        return

    st.markdown("**Upload this week's rankings** (FantasyPros weekly export, or any CSV with a player-name column)")
    import_hint('fantasypros_weekly')
    weekly_upload = st.file_uploader("Weekly rankings CSV", type=["csv"], key="weekly_rank_upload")
    weekly_df = None
    if weekly_upload is not None:
        weekly_df = parse_fantasypros_upload(weekly_upload)
        if weekly_df.empty:
            weekly_df = parse_custom_rankings(weekly_upload)
        if weekly_df is None or weekly_df.empty:
            st.warning("Couldn't find a player-name column in that upload — expected a header like 'Player', 'Player Name', or 'Name'.")

    if weekly_df is None or weekly_df.empty:
        st.caption(f"No weekly ranking uploaded yet - showing recent-form ({n_weeks}-game) ranking only.")
        display_df = position_filter_multiselect(form_df, key="weekly_rank_pos_filter")
        indexed = display_df.set_index('Player')
        st.dataframe(
            style_plain_dataframe(indexed), width="stretch", height=df_auto_height(min(len(display_df), 40)),
            column_config=build_column_help_config(indexed, pinned_cols=['Team', 'Pos']),
        )
        return

    comparison = build_rankings_comparison(form_df, value_col='Recent Avg FPTS', rank_label='Form', fp_df=weekly_df)
    extra_cols = [c for c in comparison.columns if c not in ('Player', 'Pos', 'Team')]
    merged = form_df.merge(comparison[['Player'] + extra_cols], on='Player', how='left')

    display_df = position_filter_multiselect(merged, key="weekly_rank_pos_filter")

    diverging_cols = {}
    delta_col = 'Form vs FantasyPros'
    if delta_col in display_df.columns and display_df[delta_col].notna().any():
        max_abs = display_df[delta_col].abs().max()
        if max_abs and max_abs > 0:
            diverging_cols[delta_col] = max_abs

    indexed = display_df.set_index('Player')
    st.dataframe(
        style_plain_dataframe(indexed, diverging_cols=diverging_cols),
        width="stretch", height=df_auto_height(min(len(display_df), 40)),
        column_config=build_column_help_config(indexed, pinned_cols=['Team', 'Pos']),
    )
