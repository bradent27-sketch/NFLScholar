"""
VORP Draft Sheet tab: custom league settings, replacement-level VORP, a
comparison against FantasyPros' (or your own uploaded) draft rankings, and
the Live Draft Tracker (Sleeper sync / paste-and-parse / manual multiselect).

FantasyPros' comparison lives here (not a separate tab) because the
rankings it publishes in the "draft rankings" export are, specifically,
DRAFT rankings - the same season-long draft-value question this sheet
already answers via VORP, just from an external source. A WEEKLY ranking
(FantasyPros also publishes those, in the same file format) answers a
different question - who's producing right now, not who's worth drafting -
and is compared against this app's own recent-form ranking on the separate
Weekly Rankings tab instead.

League Settings is collapsed by default (expanded=False) so the draft
tracker - the thing you're actually staring at during a live draft - is the
primary visible surface, not buried below a wall of scoring inputs.
"""
import streamlit as st

from config import AVAILABLE_SEASONS
from data.transforms import (
    load_and_merge_data, build_vorp_draft_sheet, parse_pasted_draft_picks, match_names_to_board,
    build_efficiency_volume_data,
)
from data.loaders import fetch_sleeper_draft_picks
from data.rankings import load_fantasypros_rankings, parse_custom_rankings, build_rankings_comparison
from data.utils import calculate_percentile
from ui.styling import style_plain_dataframe, df_auto_height, build_column_help_config
from ui.components import position_filter_multiselect, skeleton_loader
from ui.player_snapshot import render_efficiency_volume_quadrants


def render():
    st.markdown("<div class='custom-section-header'>CUSTOM LEAGUE VORP DRAFT SHEET</div>", unsafe_allow_html=True)
    st.caption(
        "⚠️ 'Proj Pts' below is last season's per-game pace stretched to a 17-game season - a simple, "
        "transparent stand-in for a real projection, not one. It doesn't account for injury risk, role "
        "changes, aging, or offseason moves. Treat it as a volume-based starting point, not a finished forecast. "
        "Players with fewer than 4 games played last season are left off the board entirely - a couple of games "
        "extrapolated to a full season is too noisy to be useful, and would otherwise occasionally distort "
        "replacement level for everyone else at that position too."
    )

    with st.expander("League Settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Roster**")
            vorp_num_teams = st.number_input("Number of teams", min_value=2, max_value=32, value=12, key="vorp_teams")
            vorp_qb = st.number_input("QB starters", min_value=0, max_value=3, value=1, key="vorp_qb")
            vorp_rb = st.number_input("RB starters", min_value=0, max_value=5, value=2, key="vorp_rb")
            vorp_wr = st.number_input("WR starters", min_value=0, max_value=5, value=2, key="vorp_wr")
            vorp_te = st.number_input("TE starters", min_value=0, max_value=3, value=1, key="vorp_te")
            vorp_k = st.number_input("K starters", min_value=0, max_value=2, value=1, key="vorp_k")
        with c2:
            st.markdown("**Flex slots**")
            vorp_flex = st.number_input("FLEX (RB/WR/TE)", min_value=0, max_value=4, value=1, key="vorp_flex")
            vorp_superflex = st.number_input("SUPERFLEX (QB/RB/WR/TE)", min_value=0, max_value=2, value=0, key="vorp_superflex")
        with c3:
            st.markdown("**PPR & Volume Scoring**")
            vorp_ppr = st.select_slider("PPR value", options=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5], value=1.0, key="vorp_ppr")
            vorp_ppc = st.number_input("Points per carry", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="vorp_ppc")

        c4, c5 = st.columns(2)
        with c4:
            st.markdown("**Passing**")
            vorp_pass_yd = st.number_input("Points per passing yard", min_value=0.0, max_value=0.1, value=0.04, step=0.01, format="%.2f", key="vorp_pass_yd")
            vorp_pass_td = st.number_input("Points per passing TD", min_value=0, max_value=10, value=4, key="vorp_pass_td")
            vorp_pass_int = st.number_input("Points per INT thrown", min_value=-6, max_value=0, value=-2, key="vorp_pass_int")
        with c5:
            st.markdown("**Rushing / Receiving**")
            vorp_rush_yd = st.number_input("Points per rushing yard", min_value=0.0, max_value=0.5, value=0.1, step=0.01, format="%.2f", key="vorp_rush_yd")
            vorp_rush_td = st.number_input("Points per rushing TD", min_value=0, max_value=10, value=6, key="vorp_rush_td")
            vorp_rec_yd = st.number_input("Points per receiving yard", min_value=0.0, max_value=0.5, value=0.1, step=0.01, format="%.2f", key="vorp_rec_yd")
            vorp_rec_td = st.number_input("Points per receiving TD", min_value=0, max_value=10, value=6, key="vorp_rec_td")

    vorp_year = st.selectbox("Baseline season", AVAILABLE_SEASONS, index=0, key="vorp_year")
    with skeleton_loader("table", n_rows=10, n_cols=7):
        df_vorp_stats, vorp_t_col, vorp_n_col, _ = load_and_merge_data(vorp_year, "Full PPR")

    vorp_starters = {'QB': vorp_qb, 'RB': vorp_rb, 'WR': vorp_wr, 'TE': vorp_te, 'K': vorp_k, 'FLEX': vorp_flex, 'SUPERFLEX': vorp_superflex}
    vorp_scoring = {
        'pass_yd': vorp_pass_yd, 'pass_td': vorp_pass_td, 'pass_int': vorp_pass_int,
        'rush_yd': vorp_rush_yd, 'rush_td': vorp_rush_td, 'rush_att_bonus': vorp_ppc,
        'rec': vorp_ppr, 'rec_yd': vorp_rec_yd, 'rec_td': vorp_rec_td,
    }
    draft_sheet = build_vorp_draft_sheet(df_vorp_stats, vorp_n_col, vorp_t_col, vorp_num_teams, vorp_starters, vorp_scoring, vorp_year)

    if not draft_sheet.empty:
        with st.expander("Compare against external draft rankings (optional)", expanded=False):
            fp_format = st.selectbox("FantasyPros scoring format", ["Full PPR", "Half-PPR", "Standard"], key="rank_fp_format")
            fp_df = load_fantasypros_rankings(fp_format)
            if fp_df.empty:
                st.caption(f"No FantasyPros draft-rankings file found for {fp_format} in rankings/.")

            st.markdown("**Or upload a custom draft ranking** (any CSV with a player-name column, and optionally a rank column)")
            custom_upload = st.file_uploader("Custom draft rankings CSV", type=["csv"], key="rank_custom_upload")
            custom_df = parse_custom_rankings(custom_upload) if custom_upload is not None else None
            if custom_upload is not None and (custom_df is None or custom_df.empty):
                st.warning("Couldn't find a player-name column in that upload — expected a header like 'Player', 'Player Name', or 'Name'.")

        # Merged directly onto the same draft_sheet table (Player/Team/Pos/
        # GP/Proj Pts/VORP/RZ shares) rather than shown as a separate
        # reduced comparison table - one table with everything, instead of
        # two tables a user has to cross-reference by eye.
        if not fp_df.empty or (custom_df is not None and not custom_df.empty):
            comparison = build_rankings_comparison(draft_sheet, value_col='VORP', rank_label='VORP', fp_df=fp_df, custom_df=custom_df)
            extra_cols = [c for c in comparison.columns if c not in ('Player', 'Pos', 'Team')]
            draft_sheet = draft_sheet.merge(comparison[['Player'] + extra_cols], on='Player', how='left')

        if 'drafted_players' not in st.session_state:
            st.session_state['drafted_players'] = set()

        with st.expander(f"📋 Live Draft Tracker ({len(st.session_state['drafted_players'])} drafted)", expanded=False):
            st.markdown("**Sync from Sleeper** (no login needed - fully public API)")
            sc1, sc2 = st.columns([3, 1])
            with sc1:
                sleeper_draft_id = st.text_input(
                    "Sleeper draft ID", key="sleeper_draft_id",
                    help="The number in your draft's URL: sleeper.com/draft/nfl/<this part>"
                )
            with sc2:
                st.write("")
                st.write("")
                if st.button("Sync Sleeper picks") and sleeper_draft_id:
                    picks_data, picks_err = fetch_sleeper_draft_picks(sleeper_draft_id.strip())
                    if picks_err:
                        st.error(f"Couldn't sync: {picks_err}")
                    else:
                        picked_names = []
                        for p in (picks_data or []):
                            meta = p.get('metadata', {}) or {}
                            full = f"{meta.get('first_name', '')} {meta.get('last_name', '')}".strip()
                            if full:
                                picked_names.append(full)
                        matched, unmatched = match_names_to_board(picked_names, draft_sheet['Player'].tolist())
                        st.session_state['drafted_players'].update(matched)
                        st.success(f"Synced {len(matched)} of {len(picked_names)} picks from Sleeper.")
                        if unmatched:
                            st.caption("Couldn't match: " + ", ".join(unmatched[:15]) + (" ..." if len(unmatched) > 15 else ""))
                        st.rerun()

            st.markdown("**Paste draft picks** (Yahoo draft results page, ESPN, or any list - one pick per line works best)")
            pasted_picks = st.text_area(
                "Paste here", key="pasted_draft_picks", height=120,
                placeholder="e.g.\n1. Ja'Marr Chase (WR - CIN)\n2. Bijan Robinson (RB - ATL)\n...\nor just a plain list of names, one per line"
            )
            if st.button("Parse & add pasted picks") and pasted_picks.strip():
                candidates = parse_pasted_draft_picks(pasted_picks)
                matched, unmatched = match_names_to_board(candidates, draft_sheet['Player'].tolist())
                st.session_state['drafted_players'].update(matched)
                st.success(f"Matched and added {len(matched)} of {len(candidates)} parsed lines.")
                if unmatched:
                    st.caption("Couldn't match: " + ", ".join(unmatched[:15]) + (" ..." if len(unmatched) > 15 else ""))
                st.rerun()

            st.markdown("**Or mark manually**")
            dc1, dc2 = st.columns([4, 1])
            with dc1:
                newly_drafted = st.multiselect(
                    "Mark as drafted", sorted(draft_sheet['Player'].tolist()),
                    default=[], key="draft_tracker_add"
                )
            with dc2:
                st.write("")
                st.write("")
                if st.button("Add to drafted"):
                    st.session_state['drafted_players'].update(newly_drafted)
                    st.rerun()
            if st.session_state['drafted_players']:
                st.caption("Drafted: " + ", ".join(sorted(st.session_state['drafted_players'])))
                if st.button("Reset draft"):
                    st.session_state['drafted_players'] = set()
                    st.rerun()

        display_sheet = draft_sheet[~draft_sheet['Player'].isin(st.session_state['drafted_players'])]
        display_sheet = position_filter_multiselect(display_sheet, key="vorp_pos_filter")

        numeric_pct_cols = {'VORP': calculate_percentile(display_sheet, 'VORP')}
        # Diverging red/green coloring (not the percentile-grade heatmap) for
        # the comparison delta columns - a signed disagreement metric, not
        # an absolute 0-100 grade, so color should track sign/magnitude of
        # the disagreement rather than percentile rank. See
        # get_diverging_color's docstring for why that distinction matters.
        diverging_cols = {}
        for delta_col in ['VORP vs FantasyPros', 'VORP vs Custom']:
            if delta_col in display_sheet.columns and display_sheet[delta_col].notna().any():
                max_abs = display_sheet[delta_col].abs().max()
                if max_abs and max_abs > 0:
                    diverging_cols[delta_col] = max_abs

        vorp_indexed = display_sheet.set_index('Player')
        st.dataframe(
            style_plain_dataframe(vorp_indexed, numeric_pct_cols=numeric_pct_cols, diverging_cols=diverging_cols),
            width="stretch", height=df_auto_height(min(len(display_sheet), 40)),
            column_config=build_column_help_config(vorp_indexed, pinned_cols=['Team', 'Pos']),
        )

        st.markdown("<div class='custom-section-header' style='margin-top:24px;'>EFFICIENCY VS. VOLUME (EPA-BASED)</div>", unsafe_allow_html=True)
        st.caption(
            "Each dot is a qualified player (4+ games). Dashed lines mark this season's median for each axis - "
            "the top-right quadrant is players who are both heavily used AND efficient on a per-play basis, not "
            "just compiling volume. Color follows the same red (poor) → blue (elite) percentile scale as "
            "everywhere else in this app, ranked within the position group shown."
        )
        ev_df = build_efficiency_volume_data(df_vorp_stats, vorp_n_col, vorp_t_col, vorp_year)
        quad_fig = render_efficiency_volume_quadrants(ev_df)
        if quad_fig is not None:
            st.pyplot(quad_fig, width="stretch")
        else:
            st.info("Not enough qualified players yet this season to build an efficiency-vs-volume chart.")
    else:
        st.info(f"No data available to build a draft sheet for {vorp_year}.")
