"""
Player Compare tab: head-to-head look at two players at the same position
(or position group, e.g. WR vs TE) - inspired by nflsavant.com's Compare
tool. Side-by-side bio cards + percentile-bar charts (ui.player_snapshot,
the same builder Player Search's own positional matrix uses) plus one
combined stat table with a percentile-difference "Edge" column showing
which player rates better on each shared stat, and by how much.
"""
import numpy as np
import pandas as pd
import streamlit as st

from config import TEAM_CONFIG, THEME, AVAILABLE_SEASONS_WITH_UPCOMING, get_position_color
from data.transforms import load_and_merge_data, precompute_league_percentiles
from data.loaders import load_pff_data_with_fallback
from data.utils import calculate_exact_age, parse_height, parse_weight
from ui.styling import get_pff_color, style_plain_dataframe, df_auto_height
from ui.components import render_player_card, render_bio_strip, compute_bye_weeks, build_player_search_labels, skeleton_loader
from ui.player_snapshot import (
    build_player_snapshot, render_percentile_radar_grid_compare, render_percentile_bars_figure_compare,
    position_branch_group,
)


def _load_player_side(name, df_stats, n_col, t_col, pff, df_percentiles, target_year):
    """
    Everything needed to render one player's half of the comparison: bio
    fields for render_player_card/render_bio_strip, plus the same
    build_player_snapshot list Player Search's own matrix table uses.
    Deliberately simpler than Player Search's own bio card (no gold/rookie
    treatment) - this tab is about the numbers, not a full profile.
    """
    p_data = df_stats[df_stats[n_col].astype(str) == name].copy()
    if p_data.empty:
        return None
    p_bio = p_data.iloc[0]
    pos = str(p_bio.get('position', 'N/A')).upper()
    if pos == 'NAN' or not pos:
        pos = 'N/A'
    team = str(p_bio.get(t_col, '')).upper()
    team_cfg = TEAM_CONFIG.get(team, {'color': '#00fff9'})
    overall_grade = pff['pff_grades_map'].get(name.lower(), 0.0)
    grade_color = get_pff_color(overall_grade, raw_grade=True)
    jersey_val = str(p_bio.get('jersey_number', '--')).replace('.0', '')
    if jersey_val == 'nan' or not jersey_val:
        jersey_val = '--'
    games_played = max(p_data['week'].nunique() if 'week' in p_data.columns else 1, 1)

    p_pct_row = df_percentiles[(df_percentiles[n_col].astype(str) == name) & (df_percentiles['position'] == pos)]
    snapshot = build_player_snapshot(name, pos, p_data, p_bio, pff, p_pct_row, games_played, overall_grade, grade_color, target_year)

    return {
        'name': name, 'pos': pos, 'team': team, 'team_cfg': team_cfg, 'color': team_cfg['color'],
        'grade_str': f"{overall_grade:.1f}" if overall_grade > 0 else 'N/A', 'grade_color': grade_color,
        'jersey_val': jersey_val, 'headshot_url': p_bio.get('headshot_url', ''),
        'age': calculate_exact_age(p_bio.get('birth_date'), p_bio.get('age', 0)),
        'height': parse_height(p_bio.get('height', 0)), 'weight': parse_weight(p_bio.get('weight', 0)),
        'snapshot': snapshot,
    }


def _render_player_half(side):
    render_player_card(side['name'], side['pos'], side['grade_str'], side['grade_color'],
                        side['team'], side['team_cfg']['color'], side['jersey_val'], side['headshot_url'])
    render_bio_strip(side['age'], side['height'], side['weight'])


def _render_legend(side_1, side_2):
    """Color key for the overlaid chart below - each player's own team
    color (side['color']), so the swatch someone sees on the radar/bars
    always ties back to a name without having to guess from context. A
    position tag (fixed position-identity color, config.get_position_color -
    same convention as the bio card above it) rides alongside the name as a
    second, independent signal - team color says WHO, the tag says WHAT
    POSITION, which matters here specifically since Compare allows a WR-vs-TE
    pairing where the two sides aren't the same position."""
    def swatch(side):
        # Double-quoted style="..." attributes throughout (NOT single-quoted
        # style='...') - THEME['fonts']['display']/['mono'] are themselves
        # single-quoted strings ("'Inter', sans-serif"), so a single-quoted
        # style attribute here closes prematurely at that embedded quote and
        # corrupts the rest of the tag (confirmed live: the browser was
        # parsing the leftover text as garbage bare attributes instead of a
        # style, silently dropping the font/color/padding - only invisible
        # before because the plain name span happened to still inherit a
        # correct-looking white color from its ancestor with no style
        # attribute applied at all). Same double-quote convention
        # ui.components.render_player_card already uses for this exact
        # reason.
        pos_color = get_position_color(side['pos'])
        return (
            f'<span style="display:inline-flex; align-items:center; gap:7px;">'
            f'<span style="width:13px; height:13px; border-radius:3px; background:{side["color"]}; '
            f'box-shadow:0 0 0 1px rgba(255,255,255,0.25); display:inline-block;"></span>'
            f'<span style="font-family:{THEME["fonts"]["display"]}; font-weight:700; color:#ffffff; font-size:13.5px;">{side["name"]}</span>'
            f'<span style="font-family:{THEME["fonts"]["mono"]}; font-weight:700; color:{pos_color}; font-size:10.5px; '
            f'letter-spacing:0.05em; background:rgba(255,255,255,0.06); border-radius:4px; padding:2px 6px;">{side["pos"]}</span></span>'
        )
    st.markdown(
        f"<div style='display:flex; justify-content:center; gap:26px; margin:2px 0 10px 0; flex-wrap:wrap;'>"
        f"{swatch(side_1)}{swatch(side_2)}</div>",
        unsafe_allow_html=True,
    )


def _render_skill_comparison(side_1, side_2):
    """
    ONE chart with both players overlaid (each in their own team color for
    that season) instead of two separate charts a user has to eyeball side
    by side - per explicit feedback that the previous two-column layout
    made it harder, not easier, to actually compare the two. Same
    radar-primary/bar-secondary pattern as before: positions with a
    RADAR_GRID_GROUPS entry (QB/RB/WR/TE) get the overlaid radar as the
    lead view with the overlaid bar breakdown available in an expander;
    positions without one (OL/DEFENSIVE_POSITIONS/K/[P,LS]) get the
    overlaid bar chart directly, un-demoted.
    """
    st.markdown("<div class='custom-section-header'>SKILL COMPARISON</div>", unsafe_allow_html=True)
    _render_legend(side_1, side_2)
    # Charts centered in a narrower column instead of stretched across the
    # app's full-bleed width - this section isn't inside the page's own
    # cc1/cc2 split (that ends above, at the two player cards), so without
    # this an ultra-wide monitor would stretch a matplotlib PNG well past
    # its native resolution and it'd read as soft/low-quality, the same
    # complaint that applied to the Coverage Matchup Radar (fixed there via
    # this same centered-column convention).
    radar_fig = render_percentile_radar_grid_compare(side_1, side_2)
    if radar_fig is not None:
        rc1, rc2, rc3 = st.columns([1, 6, 1])
        with rc2:
            st.pyplot(radar_fig, width="stretch")
        with st.expander("Full stat breakdown"):
            bars_fig = render_percentile_bars_figure_compare(side_1, side_2)
            if bars_fig is not None:
                bc1, bc2, bc3 = st.columns([1, 6, 1])
                with bc2:
                    st.pyplot(bars_fig, width="stretch")
    else:
        bars_fig = render_percentile_bars_figure_compare(side_1, side_2)
        if bars_fig is not None:
            bc1, bc2, bc3 = st.columns([1, 6, 1])
            with bc2:
                st.pyplot(bars_fig, width="stretch")
        else:
            st.caption("No shared percentile data available for these two players/season.")


def _render_comparison_table(side_1, side_2):
    by_label_1 = {e['label']: e for e in side_1['snapshot']}
    by_label_2 = {e['label']: e for e in side_2['snapshot']}
    # side_1's own stat order first (matches build_player_snapshot's branch
    # order, same as the matrix table on Player Search), then anything
    # side_2 has that side_1 doesn't (e.g. one player missing a PFF row
    # for a conditionally-added stat the other has).
    ordered_labels = list(by_label_1.keys()) + [l for l in by_label_2.keys() if l not in by_label_1]
    if not ordered_labels:
        st.info("No shared stats to compare for these two players.")
        return

    rows = []
    for label in ordered_labels:
        e1, e2 = by_label_1.get(label), by_label_2.get(label)
        val1 = e1['value_str'] if e1 else '--'
        val2 = e2['value_str'] if e2 else '--'
        # Percentile-point gap, not a raw-value delta - one uniform scale
        # (bounded +/-100) works across every stat regardless of its native
        # units (yards vs. % vs. counts), same "positive = this side rates
        # higher" convention already used for 'VORP vs FantasyPros' etc.
        # elsewhere in this app. NaN (not 0) whenever either side has no
        # real value to compare - checked on the DISPLAYED value ('--'/
        # 'N/A'), not just `pct is not None`: build_player_snapshot's `.get(
        # x_pct, 0)` fallback (matching the original single-player matrix
        # table's behavior) means pct is often a real 0, not None, even
        # when the stat is genuinely missing for both sides - which would
        # otherwise show a misleading "0.0" Edge (looks like "dead even")
        # between two blank "--" cells instead of "not comparable".
        has_real_value = lambda v: v not in ('--', 'N/A')
        if e1 and e2 and has_real_value(val1) and has_real_value(val2):
            edge = round(e1['pct'] - e2['pct'], 1)
        else:
            edge = np.nan
        rows.append({'Stat': label, side_1['name']: val1, side_2['name']: val2, 'Edge (Pctl)': edge})

    table_df = pd.DataFrame(rows).set_index('Stat')
    st.markdown("<div class='custom-section-header'>SHARED STAT COMPARISON</div>", unsafe_allow_html=True)
    # Column labeled "Edge (Pctl)", not bare "Edge" - it's a percentile-POINT
    # gap (see the comment above), which can look wrong at a glance next to
    # two raw values that are actually close (e.g. 97.4% vs 93.0% snap share
    # showing a "42.5" edge, since that's the gap between their SNAP %
    # PERCENTILE RANKS among position peers, not the 4.4-point raw
    # difference a bare "Edge" header implies).
    st.dataframe(
        style_plain_dataframe(table_df, diverging_cols={'Edge (Pctl)': 100}),
        width="stretch", height=df_auto_height(len(rows)),
    )


def render():
    st.markdown("<div class='custom-section-header'>PLAYER COMPARE</div>", unsafe_allow_html=True)

    c_year, c_score = st.columns(2)
    with c_year:
        cmp_year = st.selectbox("Season", AVAILABLE_SEASONS_WITH_UPCOMING, index=1, key="year_cmp")
    with c_score:
        cmp_scoring = st.selectbox("Scoring", ["Full PPR", "Half-PPR", "Standard"], key="score_cmp")

    pff, pff_source_year = load_pff_data_with_fallback(cmp_year)
    if pff_source_year != cmp_year:
        st.caption(f"⚠️ No PFF grades uploaded yet for {cmp_year} - showing {pff_source_year} grades until real data is added to pff_imports/{cmp_year}/.")

    with skeleton_loader("tiles", n=10):
        df_stats, t_col, n_col, _ = load_and_merge_data(cmp_year, cmp_scoring)
    df_percentiles = precompute_league_percentiles(df_stats, n_col, cmp_year)
    bye_weeks = compute_bye_weeks(cmp_year)

    c1, c2 = st.columns(2)
    with c1:
        labels_1, label_to_name_1 = build_player_search_labels(df_stats, n_col, t_col, bye_weeks)
        selected_label_1 = st.selectbox(
            "Player 1", labels_1, index=None, key="cmp_player_1", placeholder="— Select a player —",
        )
    player_1 = label_to_name_1.get(selected_label_1) if selected_label_1 else None

    with c2:
        branch_group = None
        if player_1:
            p1_rows = df_stats[df_stats[n_col].astype(str) == player_1]
            if not p1_rows.empty:
                p1_pos = str(p1_rows.iloc[0].get('position', '')).upper()
                branch_group = position_branch_group(p1_pos)
        labels_2, label_to_name_2 = build_player_search_labels(df_stats, n_col, t_col, bye_weeks, position_group=branch_group)
        help_2 = (
            f"Narrowed to {'/'.join(branch_group)} - the same stat set as Player 1, so the comparison is meaningful."
            if branch_group else "Pick Player 1 first."
        )
        selected_label_2 = st.selectbox(
            "Player 2", labels_2, index=None, key="cmp_player_2", placeholder="— Select a player —",
            disabled=player_1 is None, help=help_2,
        )
    player_2 = label_to_name_2.get(selected_label_2) if selected_label_2 else None

    if not player_1 or not player_2:
        return
    if player_1 == player_2:
        st.warning("Pick two different players to compare.")
        return

    side_1 = _load_player_side(player_1, df_stats, n_col, t_col, pff, df_percentiles, cmp_year)
    side_2 = _load_player_side(player_2, df_stats, n_col, t_col, pff, df_percentiles, cmp_year)
    if side_1 is None or side_2 is None:
        st.warning("Couldn't load one of the selected players' data for this season.")
        return

    cc1, cc2 = st.columns(2)
    with cc1:
        _render_player_half(side_1)
    with cc2:
        _render_player_half(side_2)

    _render_skill_comparison(side_1, side_2)
    _render_comparison_table(side_1, side_2)
