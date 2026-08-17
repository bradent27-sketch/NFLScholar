"""
Weekly Rankings tab: this app's own week-by-week fantasy-points projection
model (data.weekly_projections), FantasyPros' live weekly projection pulled
straight from their API, this app's own recent-form ranking, and (still,
unchanged) an uploaded weekly-rankings export - all lined up on one table
by player.

A weekly ranking answers "who's producing right now" / "who should I start
this week" - a different question than the season-long draft-value ranking
Draft HQ compares against FantasyPros' DRAFT rankings. Keeping the two
comparisons on separate tabs (rather than one tab trying to serve both)
means each one's internal baseline actually matches what it's being
compared against.

The model, the live API pull, and the CSV upload are three INDEPENDENT
sources shown side by side, never blended into one number - same
"show market lines next to this board, don't merge them" convention Draft
HQ already uses.
"""
import pandas as pd
import streamlit as st

from config import AVAILABLE_SEASONS
from data.transforms import load_and_merge_data, build_recent_form_rank
from data.rankings import parse_fantasypros_upload, parse_custom_rankings, build_rankings_comparison
from data.utils import clean_name_exact, clean_name_for_merge
from data.weekly_projections import build_weekly_projections
from ui.styling import style_plain_dataframe, df_auto_height, build_column_help_config
from ui.components import position_filter_multiselect, skeleton_loader, import_hint

_MODEL_STAT_COLS = ['passing_yards', 'passing_tds', 'rushing_attempts', 'rushing_yards',
                    'rushing_tds', 'targets', 'receptions', 'receiving_yards', 'receiving_tds']


def _week_options(year):
    """
    [(week, label, is_next_incomplete)], from the real schedule
    (data.game_slate, already built for the Game Slate tab) - reused rather
    than a second week-numbering scheme. Falls back to a bare 1-18 range if
    the schedule feed is unreachable, so the tab still works offline.
    """
    try:
        from data.game_slate import slate_weeks
        weeks, _err = slate_weeks(year)
        reg = [w for w in weeks if w['season_type'] == 'REG']
        if reg:
            return reg
    except Exception:
        pass
    return [{'week': w, 'label': f'Week {w}', 'completed': False} for w in range(1, 19)]


def _default_week_index(weeks):
    """The next UNPLAYED week - a projection tool wants the upcoming week,
    the opposite default from Game Slate's own "last completed" one."""
    for i, w in enumerate(weeks):
        if not w['completed']:
            return i
    return max(len(weeks) - 1, 0)


def _render_fantasypros_weekly_pull(wk_year, wk_week, wk_scoring):
    """
    Button-triggered pull of FantasyPros' own weekly stat-line projection
    for the selected week, straight from their API - same secrets-aware key
    resolution and budget bookkeeping as Draft HQ's ECR/ADP pull
    (data.draft_sources.fetch_fantasypros_weekly_projections costs one call
    PER POSITION requested, see that function's own docstring for why).

    Always rendered, independent of whether this app's own model produced
    anything for the selected week - week 1 is exactly the case where the
    model CAN'T project (see build_weekly_projections' docstring) and
    FantasyPros' own number is the one worth having.
    """
    from data.draft_sources import (
        get_fantasypros_api_key, save_fantasypros_api_key,
        fantasypros_api_calls_this_month, fantasypros_effective_limit,
        fetch_fantasypros_weekly_projections,
    )
    with st.expander("📡 FantasyPros weekly projection (live API)", expanded=False):
        st.caption(
            "Pulls FantasyPros' own projected stat line for QB/RB/WR/TE for the selected week - "
            "4 calls (one per position; the endpoint requires a position per request). Shown "
            "alongside this app's model below, never blended into it."
        )
        secret_key, key_source = get_fantasypros_api_key()
        if key_source == 'secrets':
            st.caption("🔑 Using the key from `.streamlit/secrets.toml`.")
            api_key = secret_key
        else:
            api_key = st.text_input("FantasyPros API key", type="password",
                                    key="wr_fp_api_key", value=secret_key)
            if api_key and api_key != secret_key:
                save_fantasypros_api_key(api_key)

        used = fantasypros_api_calls_this_month()
        limit = fantasypros_effective_limit()
        remaining = None if limit is None else limit - used
        cost = 4
        fetch = st.button("Fetch weekly projections", key="wr_fp_fetch",
                          disabled=not api_key or (remaining is not None and remaining < cost))
        if limit is not None:
            st.caption(f"{used} of {limit} calls used this month - this pull costs {cost}.")
        if fetch:
            with st.spinner("Calling the FantasyPros API…"):
                df, meta = fetch_fantasypros_weekly_projections(api_key, wk_year, wk_week)
            if meta.get('error') and df.empty:
                st.error(f"FantasyPros API: {meta['error']}")
            else:
                st.session_state['wr_fp_weekly_df'] = df
                st.session_state['wr_fp_weekly_meta'] = (wk_year, wk_week, wk_scoring)
                if meta.get('errors'):
                    st.warning(f"Some positions failed: {meta['errors']}")
                st.success(f"Pulled {len(df):,} FantasyPros weekly projections for week {wk_week}.")
                st.rerun()

    held = st.session_state.get('wr_fp_weekly_df')
    held_meta = st.session_state.get('wr_fp_weekly_meta')
    if held is not None and not held.empty and held_meta == (wk_year, wk_week, wk_scoring):
        return held
    return None


def _attach_by_name(base_df, other_df, value_cols, prefix):
    """
    Merge `other_df`'s columns onto `base_df` by player name, two-tier
    (exact, then suffix-stripped fallback) - same matcher every cross-source
    join in this app uses, since no two sources spell every name identically.
    """
    if other_df is None or other_df.empty or 'Player' not in other_df.columns:
        return base_df
    exact_map = {clean_name_exact(pd.Series([p])).iloc[0]: p for p in other_df['Player']}
    loose_map = {clean_name_for_merge(pd.Series([p])).iloc[0]: p for p in other_df['Player']}
    lookup = other_df.set_index('Player')

    def _match(name):
        key = clean_name_exact(pd.Series([name])).iloc[0]
        if key in exact_map:
            return exact_map[key]
        key = clean_name_for_merge(pd.Series([name])).iloc[0]
        return loose_map.get(key)

    matched = base_df['Player'].map(_match)
    out = base_df.copy()
    for col in value_cols:
        if col in lookup.columns:
            out[f'{prefix}{col}'] = matched.map(lookup[col])
    return out


def _fantasypros_points_column(scoring_mode):
    return {'Full PPR': 'FP Proj Pts PPR', 'Half-PPR': 'FP Proj Pts Half',
           'Standard': 'FP Proj Pts'}[scoring_mode]


def render():
    st.markdown("<div class='custom-section-header'>WEEKLY RANKINGS</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        wk_year = st.selectbox("Season", AVAILABLE_SEASONS, index=0, key="weekly_rank_year")
    weeks = _week_options(wk_year)
    week_labels = [w['label'] for w in weeks]
    with c2:
        wk_week_idx = st.selectbox("Week", range(len(weeks)), index=_default_week_index(weeks),
                                   format_func=lambda i: week_labels[i], key="weekly_rank_week")
    wk_week = weeks[wk_week_idx]['week']
    with c3:
        wk_scoring = st.selectbox("Scoring", ["Full PPR", "Half-PPR", "Standard"], key="weekly_rank_scoring")
    n_weeks = st.number_input("Recent-form window (games)", min_value=1, max_value=8, value=4,
                              key="weekly_rank_window")

    with skeleton_loader("table", n_rows=10, n_cols=7):
        df_stats, t_col, n_col, _ = load_and_merge_data(wk_year, wk_scoring)
        model_df, model_meta = build_weekly_projections(wk_year, wk_week, wk_scoring)

    form_df = build_recent_form_rank(df_stats, n_col, t_col, n_weeks=n_weeks)

    st.markdown("### This app's weekly model")
    fp_weekly = _render_fantasypros_weekly_pull(wk_year, wk_week, wk_scoring)

    if model_df.empty:
        st.info(f"This app's model has no projection for {wk_year} week {wk_week}: "
               f"{model_meta.get('reason', 'not enough data yet')}")
        if fp_weekly is not None:
            st.caption("Showing FantasyPros' live weekly projection instead:")
            fp_pts_col = _fantasypros_points_column(wk_scoring)
            cols = ['Player', 'Pos', 'Team'] + ([fp_pts_col] if fp_pts_col in fp_weekly.columns else [])
            display_df = position_filter_multiselect(fp_weekly[cols].rename(
                columns={fp_pts_col: 'FantasyPros Proj Pts'}), key="weekly_rank_pos_filter")
            indexed = display_df.set_index('Player')
            st.dataframe(style_plain_dataframe(indexed), width="stretch",
                        height=df_auto_height(min(len(display_df), 40)))
    else:
        merged_model = model_df.copy()
        if fp_weekly is not None:
            merged_model = _attach_by_name(
                merged_model, fp_weekly,
                ['FP Proj Pts', 'FP Proj Pts PPR', 'FP Proj Pts Half'] + _MODEL_STAT_COLS, 'FP ')
            src_col = 'FP ' + _fantasypros_points_column(wk_scoring)
            if src_col in merged_model.columns:
                merged_model = merged_model.rename(columns={src_col: 'FantasyPros Proj Pts'})
        if not form_df.empty:
            merged_model = _attach_by_name(merged_model, form_df, ['Recent Avg FPTS'], '')

        display_cols = ['Player', 'Pos', 'Team', 'Opponent', 'Model Proj Pts']
        if 'FantasyPros Proj Pts' in merged_model.columns:
            display_cols.append('FantasyPros Proj Pts')
        if 'Recent Avg FPTS' in merged_model.columns:
            display_cols.append('Recent Avg FPTS')
        display_cols += ['Games This Season', 'Role Confidence', 'Injury Status']

        display_df = position_filter_multiselect(
            merged_model[[c for c in display_cols if c in merged_model.columns]],
            key="weekly_rank_pos_filter")
        indexed = display_df.set_index('Player')
        st.dataframe(
            style_plain_dataframe(indexed), width="stretch", height=df_auto_height(min(len(display_df), 40)),
            column_config=build_column_help_config(indexed, pinned_cols=['Team', 'Pos', 'Opponent']),
        )
        st.caption(
            "**Model Proj Pts** is this app's own projection (usage blended toward the current "
            "season as it grows, opponent/pace/game-script adjusted - see "
            "docs/weekly_projections_methodology.md). **FantasyPros Proj Pts** is their "
            "analysts' number, pulled live above. Shown side by side, never blended."
        )

    if form_df.empty:
        st.info(f"Not enough {wk_year} weekly data yet to build a recent-form baseline.")
        return

    st.markdown("---")
    st.markdown("**Upload a weekly rankings export** (FantasyPros weekly export, or any CSV with a player-name column)")
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
        return

    comparison = build_rankings_comparison(form_df, value_col='Recent Avg FPTS', rank_label='Form', fp_df=weekly_df)
    extra_cols = [c for c in comparison.columns if c not in ('Player', 'Pos', 'Team')]
    merged = form_df.merge(comparison[['Player'] + extra_cols], on='Player', how='left')

    display_df = position_filter_multiselect(merged, key="weekly_rank_upload_pos_filter")

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
