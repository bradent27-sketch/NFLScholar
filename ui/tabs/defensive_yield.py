"""
Defensive Yield Schemes tab: fantasy-points-allowed heatmap, team coverage
scheme tendencies, and the Defensive Coverage Correlator radar (man-vs-zone
WR/TE performance crossed against an opponent's man/zone coverage mix, plus
SumerSports personnel/formation tendency as secondary scheme context).
"""
import pandas as pd
import streamlit as st

from config import TEAM_CONFIG, pff_team_to_abbr, AVAILABLE_SEASONS_WITH_UPCOMING
from data.transforms import (
    load_and_merge_data, build_points_allowed_matrix, build_strength_of_schedule,
    build_oline_protection_metrics, build_def_pressure_metrics,
)
from data.loaders import (
    load_external_coverage_schemes, load_all_pff_data, load_sumersports_tendency_data, load_sharp_positional_coverage,
    load_schedule, load_pfr_pass_block, load_pfr_def_pressure, load_team_pass_attempts_faced,
)
from data.utils import calculate_percentile
from data.coverage_radar import (
    list_coverage_teams, list_receivers, build_radar_data, build_defense_radar_data, render_split_radar_figure,
    team_nickname, build_team_man_zone_rates, build_alignment_matchup_data,
)
from ui.styling import style_plain_dataframe, df_auto_height, build_column_help_config, get_pff_color, get_matchup_color
from ui.components import render_matchup_title, render_percentile_metric_tiles, skeleton_loader

# Sharp Football's coverage-tendency export keys teams by bare nickname
# (e.g. "Dolphins"), while build_points_allowed_matrix keys by nflverse
# abbreviation ("MIA") - built once from TEAM_CONFIG (the same source
# data.coverage_radar.team_nickname derives nicknames from) so the merge
# below can join the two tables on a common key.
_NICKNAME_TO_ABBR = {team_nickname(cfg.get('name', '')): abbr for abbr, cfg in TEAM_CONFIG.items()}

# Sharp Football Analysis's coverage-tendency page (the source behind
# load_external_coverage_schemes) has no season selector or archive -
# confirmed live, it only ever serves the current season's snapshot - so
# "MOF Closed %"/"MOF Open %" (middle-of-field shell tendency) can only ever
# be shown for whichever season that snapshot actually reflects, not
# retroactively attached to a specific past year. PFF's own coverage export
# doesn't carry this particular metric at all (see
# data.coverage_radar.build_team_man_zone_rates), so there's no substitute
# source for it either - a genuine, unavoidable gap for every year except
# this one.
_MOF_DATA_YEAR = 2025


def _merge_matchup_and_coverage(def_matrix, pff_man_zone_df, mof_df=None):
    """
    Combines the fantasy-points-allowed-by-position matrix with team man/
    zone coverage tendency into one table, so a defense's scoring
    vulnerability and its scheme tendency are visible side by side instead
    of in two separate tables a user has to cross-reference by eye.

    Man %/Zone % come from `pff_man_zone_df` (PFF's own defense_coverage_scheme
    export, aggregated via build_team_man_zone_rates) - already keyed by
    nflverse abbreviation - so they're available for whatever year the PFF
    data covers, not just 2025. `mof_df` (Sharp Football's MOF Closed %/Open %,
    still nickname-keyed) is optional and only ever passed for
    _MOF_DATA_YEAR, per that constant's docstring.

    Left-merged on def_matrix (points-allowed data determines which teams
    show up) - a team missing from either coverage source just gets blank
    columns for that source rather than being dropped entirely.
    """
    if def_matrix.empty:
        return pd.DataFrame()
    merged = def_matrix.copy()
    if pff_man_zone_df is not None and not pff_man_zone_df.empty:
        cov = pff_man_zone_df.rename(columns={'man_rate': 'Man %', 'zone_rate': 'Zone %'})
        merged = merged.merge(cov[['Team', 'Man %', 'Zone %']], on='Team', how='left')
    if mof_df is not None and not mof_df.empty:
        mof = mof_df.rename(columns={
            'middle_closed_rate': 'MOF Closed %', 'middle_open_rate': 'MOF Open %',
        }).copy()
        mof['Team'] = mof['team'].map(_NICKNAME_TO_ABBR)
        keep_cols = ['Team'] + [c for c in ['MOF Closed %', 'MOF Open %'] if c in mof.columns]
        merged = merged.merge(mof[keep_cols], on='Team', how='left')
    return merged


def render():
    c_t3_year, c_t3_score = st.columns(2)
    with c_t3_year:
        t3_target_year = st.selectbox("Season", AVAILABLE_SEASONS_WITH_UPCOMING, index=1, key="year_tab3")
    with c_t3_score:
        t3_scoring_rule = st.selectbox("Scoring", ["Full PPR", "Half-PPR", "Standard"], key="score_tab3")

    with skeleton_loader("table", n_rows=10, n_cols=7):
        df_t3_stats, _, _, _ = load_and_merge_data(t3_target_year, t3_scoring_rule)
    pff = load_all_pff_data(t3_target_year)

    # Coverage Correlator first - the tab's own signature visual, per user
    # request to lead with it rather than bury it below three tables.
    _render_coverage_radar(pff)

    def_matrix = build_points_allowed_matrix(df_t3_stats, t3_target_year)
    pff_man_zone = build_team_man_zone_rates(pff['def_coverage_scheme'])
    if not pff_man_zone.empty:
        pff_man_zone['Team'] = pff_man_zone['team'].apply(pff_team_to_abbr)
    # Sharp Football's MOF Closed %/Open % is only ever the CURRENT season's
    # snapshot (no archive - see _MOF_DATA_YEAR's docstring), so it's only
    # meaningful to attach when the selected year actually matches it.
    mof_df = load_external_coverage_schemes() if t3_target_year == _MOF_DATA_YEAR else pd.DataFrame()
    merged_matrix = _merge_matchup_and_coverage(def_matrix, pff_man_zone, mof_df)
    if not merged_matrix.empty:
        st.markdown(f"<div class='custom-section-header'>FANTASY POINTS ALLOWED &amp; COVERAGE TENDENCIES — {t3_target_year}</div>", unsafe_allow_html=True)
        if t3_target_year != _MOF_DATA_YEAR:
            st.caption(
                f"MOF Closed %/Open % is only available for {_MOF_DATA_YEAR} - its source (Sharp Football "
                "Analysis) has no historical archive."
            )
        # Points-allowed columns are directional (fewest allowed = best
        # defense = green) - ascending=False so the stingiest defense gets
        # the highest percentile, matching get_matchup_color's "100=green"
        # convention. Coverage-tendency columns have no inherent good/bad
        # direction (a high man rate isn't objectively better or worse than
        # a high zone rate), so they use the default ascending=True purely
        # for a consistent relative-magnitude read against the same palette.
        pos_cols = [c for c in ['QB', 'RB', 'WR', 'TE'] if c in merged_matrix.columns]
        cov_cols = [c for c in ['Man %', 'Zone %', 'MOF Closed %', 'MOF Open %'] if c in merged_matrix.columns]
        matchup_pct_cols = {col: calculate_percentile(merged_matrix, col, ascending=False) for col in pos_cols}
        matchup_pct_cols.update({col: calculate_percentile(merged_matrix, col) for col in cov_cols})
        merged_indexed = merged_matrix.set_index('Team')
        # No ProgressColumn meters on Man %/Zone %/MOF Closed %/MOF Open % -
        # tried this, but the league's actual man/zone split doesn't spread
        # far enough across 0-100 for a linear meter to show real variation
        # (most teams cluster in a narrow band), so every bar looked nearly
        # identical - the percentile heatmap coloring already does the real
        # work of showing which teams lean man/zone relative to the league.
        st.dataframe(
            style_plain_dataframe(merged_indexed, matchup_pct_cols=matchup_pct_cols),
            width="stretch", height=df_auto_height(len(merged_matrix)),
            column_config=build_column_help_config(merged_indexed),
        )
    else:
        st.info(f"No {t3_target_year} games have been played yet, so there's no fantasy-points-allowed data to show here. Try a completed season (2025, 2024, or 2023) instead.")

    _render_strength_of_schedule(t3_target_year, t3_scoring_rule, def_matrix)
    _render_oline_pressure(t3_target_year)


def _render_strength_of_schedule(target_year, scoring_rule, def_matrix):
    st.markdown(f"<div class='custom-section-header'>REST-OF-SEASON STRENGTH OF SCHEDULE — {target_year}</div>", unsafe_allow_html=True)

    sos_matrix, source_note = def_matrix, str(target_year)
    if sos_matrix.empty:
        # No fantasy-points-allowed data yet for the selected year (e.g. an
        # upcoming season before kickoff) - fall back to the most recent
        # prior season with real data so this still gives a "way too early"
        # SOS read against the current schedule rather than nothing at all.
        for back in range(1, 4):
            prior_year = target_year - back
            if prior_year < 2019:
                break
            df_prior, _, _, _ = load_and_merge_data(prior_year, scoring_rule)
            sos_matrix = build_points_allowed_matrix(df_prior, prior_year)
            if not sos_matrix.empty:
                source_note = f"{prior_year} (most recent season with data)"
                break

    schedule_df = load_schedule(target_year)
    if sos_matrix.empty:
        st.info("No fantasy-points-allowed data available yet to build a strength-of-schedule table.")
        return
    if schedule_df.empty:
        st.info(f"No {target_year} schedule available.")
        return

    remaining_only = st.checkbox("Remaining games only (unplayed)", value=True, key="sos_remaining_only")
    sos_df = build_strength_of_schedule(schedule_df, sos_matrix, target_year, remaining_only)
    if sos_df.empty:
        st.info("No games found for the selected filter.")
        return

    # build_strength_of_schedule silently falls back to the FULL season's
    # games when "remaining only" is requested but every game already has a
    # score (a completed season, e.g. browsing 2025 after it's over) - a
    # 0-game remaining table would otherwise just be empty and useless. That
    # fallback is the right behavior, but it needs its own caption (same
    # transparency pattern as every other fallback in this app - PFF grade
    # year, snap-count year) or the checkbox stays checked and reading
    # "Remaining games only" while the table quietly shows the whole season
    # instead, which looks like a bug rather than a deliberate substitution.
    if remaining_only and not schedule_df['home_score'].isna().any():
        st.caption(f"No games remain in the {target_year} season - showing the full-season schedule instead.")

    st.caption(f"Points-allowed baseline: {source_note} season. Higher = softer remaining matchups (those opponents allow more fantasy points at that position).")
    # "Games" (how many remaining/total games fed each team's average) is
    # useful for build_strength_of_schedule's own math but redundant on
    # screen - the schedule length is implicit and every other reader of
    # this table cares about the SOS numbers, not the game count.
    sos_display = sos_df.drop(columns=['Games'], errors='ignore')
    sos_indexed = sos_display.set_index('Team')
    # Same muted light-green(good)/red(bad) matchup palette as the merged
    # Defensive Yield matrix above, not get_pff_color's brighter 0-100-grade
    # rainbow - higher SOS (softer remaining schedule) should read green.
    sos_pct_cols = {c: calculate_percentile(sos_indexed, c) for c in sos_indexed.columns if pd.api.types.is_numeric_dtype(sos_indexed[c])}
    st.dataframe(
        style_plain_dataframe(sos_indexed, matchup_pct_cols=sos_pct_cols), width="stretch", height=df_auto_height(len(sos_df)),
        column_config=build_column_help_config(sos_indexed),
    )


def _relative_meter_range(series, pad_frac=0.15):
    """
    Bounds a ProgressColumn meter to this column's OWN observed range (with
    a small pad) instead of a fixed 0-100 scale. Pressure Allowed %/Pressure
    Rate % both cluster tightly (roughly 15-40%), so a 0-100 meter left
    every team's bar looking nearly identical - the real variance among
    starters/the league only shows up once the scale is stretched to match
    what actually occurs, the same idea as this table's own percentile
    heatmap coloring already uses.
    """
    vals = pd.to_numeric(series, errors='coerce').dropna()
    if vals.empty:
        return (0, 100)
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return (max(0.0, lo - 1), hi + 1)
    pad = (hi - lo) * pad_frac
    return (max(0.0, lo - pad), hi + pad)


def _render_oline_pressure(target_year):
    st.markdown(f"<div class='custom-section-header'>O-LINE PASS PROTECTION &amp; DEFENSIVE PRESSURE — {target_year}</div>", unsafe_allow_html=True)
    # Same "fall back to the most recent season with real data" pattern as
    # the SOS table above - PFR hasn't posted anything yet for a season
    # before it's been played (e.g. the 2026 default view before kickoff),
    # so without this the section would just say "not available" on the
    # very first thing most people see when opening this tab.
    pfr_pass_df, pfr_def_df, pass_attempts_faced, source_year = pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), target_year
    for back in range(0, 4):
        try_year = target_year - back
        if try_year < 2019:
            break
        pfr_pass_df = load_pfr_pass_block(try_year)
        pfr_def_df = load_pfr_def_pressure(try_year)
        pass_attempts_faced = load_team_pass_attempts_faced(try_year)
        if not pfr_pass_df.empty or not pfr_def_df.empty:
            source_year = try_year
            break
    if source_year != target_year:
        st.caption(f"No PFR data posted yet for {target_year} - showing {source_year}, the most recent season with data.")

    oc1, oc2 = st.columns(2)
    with oc1:
        st.markdown("**O-Line Pass Protection** (lower % = better protection)")
        oline_df = build_oline_protection_metrics(pfr_pass_df, source_year)
        if not oline_df.empty:
            pressure_pct = calculate_percentile(oline_df, 'Pressure Allowed %', ascending=False)
            oline_indexed = oline_df.set_index('Team')
            st.dataframe(
                style_plain_dataframe(oline_indexed, numeric_pct_cols={'Pressure Allowed %': pressure_pct}),
                width="stretch", height=df_auto_height(len(oline_df)),
                column_config=build_column_help_config(
                    oline_indexed, meter_cols={'Pressure Allowed %': _relative_meter_range(oline_df['Pressure Allowed %'])}
                ),
            )
        else:
            st.info("No PFR advanced passing data available yet.")
    with oc2:
        st.markdown("**Defensive Pressure Rate Generated**")
        def_pressure_df = build_def_pressure_metrics(pfr_def_df, pass_attempts_faced, source_year)
        if not def_pressure_df.empty:
            pr_pct = calculate_percentile(def_pressure_df, 'Pressure Rate %')
            def_pressure_indexed = def_pressure_df.set_index('Team')
            st.dataframe(
                style_plain_dataframe(def_pressure_indexed, numeric_pct_cols={'Pressure Rate %': pr_pct}),
                width="stretch", height=df_auto_height(len(def_pressure_df)),
                column_config=build_column_help_config(
                    def_pressure_indexed, meter_cols={'Pressure Rate %': _relative_meter_range(def_pressure_df['Pressure Rate %'])}
                ),
            )
        else:
            st.info("No PFR advanced defensive/team stats available yet.")


def _ordinal_percentile_label(pct):
    """
    "78th percentile" style caption for a 0-100 percentile-rank number -
    used as st.metric's `delta` text (delta_color="off" so it doesn't pick
    up a red/green up-down arrow, which would misleadingly imply a change
    over time rather than a league-relative rank) on the Coverage Matchup
    Radar's metric boxes. Returns None for a missing/unrankable value so
    the caller can skip the delta entirely instead of showing "None".
    """
    if pct is None or pd.isna(pct):
        return None
    n = int(round(pct))
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix} percentile"


def _render_coverage_radar(pff):
    st.markdown("<div class='custom-section-header'>COVERAGE MATCHUP RADAR — MAN VS. ZONE</div>", unsafe_allow_html=True)

    rec_scheme_df = pff['rec_scheme']
    def_coverage_df = pff['def_coverage_scheme']
    positional_coverage_df = load_sharp_positional_coverage()

    if rec_scheme_df.empty or def_coverage_df.empty:
        st.info("receiving_scheme_2025.csv and/or defense_coverage_scheme_2025.csv not found in pff_imports/ — radar unavailable.")
        return

    receivers = list_receivers(rec_scheme_df)
    teams = list_coverage_teams(def_coverage_df)
    if not receivers or not teams:
        st.info("No WR/TE or defensive team rows found in the loaded PFF coverage files.")
        return

    rc1, rc2 = st.columns(2)
    with rc1:
        sel_player = st.selectbox("WR / TE", receivers, key="radar_player")
    with rc2:
        sel_opponent = st.selectbox("Opposing defense (PFF team code)", teams, key="radar_opponent")

    labels, man_vals, zone_vals, matchup = build_radar_data(rec_scheme_df, sel_player, sel_opponent, def_coverage_df)

    if not labels:
        st.info(f"No man/zone receiving-scheme data found for {sel_player}.")
        return

    # sel_opponent is a PFF team code (e.g. "ARZ") from
    # defense_coverage_scheme_2025.csv; Sharp Football's + SumerSports' CSVs
    # use full/nickname team names - resolve via TEAM_CONFIG's nflverse
    # abbreviation rather than a fragile substring match.
    abbr = pff_team_to_abbr(sel_opponent)
    team_cfg = TEAM_CONFIG.get(abbr, {})
    full_name = team_cfg.get('name')
    def_labels, def_vals, def_raw = build_defense_radar_data(positional_coverage_df, full_name)

    # Chart FIRST, metric tiles below it (order flipped per explicit
    # feedback), and rendered wider - only thin side gutters now that the
    # page itself is full-bleed, since the radar is this tab's signature
    # visual. Title is a dedicated centered component (not a bare **bold**
    # markdown line) per explicit feedback that the plain text ran too
    # close together and read unpolished against the chart below it.
    render_matchup_title(sel_player, full_name or sel_opponent, team_color=team_cfg.get('color'))
    fig = render_split_radar_figure(labels, man_vals, zone_vals, sel_player, def_labels, def_vals, sel_opponent)
    if fig is not None:
        cc1, cc2, cc3 = st.columns([1, 6, 1])
        with cc2:
            st.pyplot(fig, width="stretch")
    elif not def_labels:
        st.caption("sharp_positional_coverage_2025.csv not found in external_data/ — showing player radar only.")

    # Alignment (slot-vs-outside) axis - complements the Man/Zone (coverage
    # shell) axis above with Slot %/Slot YPRR/Wide YPRR for the player, plus
    # a second matchup-aware blended number keyed on alignment instead of
    # coverage scheme.
    align_data = build_alignment_matchup_data(pff['rec'], pff['route_concept'], positional_coverage_df, sel_player, full_name)

    # Three tile columns instead of a "player row on top, defense row below"
    # stack, per explicit feedback: PLAYER (this receiver's own man/zone +
    # alignment production) on the left, the two matchup-aware BLENDED
    # numbers in the middle, and every TEAM/opponent-tendency stat (Man %/
    # Zone % coverage mix plus the Sharp Football YPT-allowed-by-type
    # figures that used to sit in their own row below) grouped on the
    # right - so "how good is HE" and "how tough is THIS DEFENSE" read as
    # two clearly separated sides of the matchup instead of top/bottom rows
    # with no visual distinction. Tile background = percentile color
    # (replaces the plain st.metric boxes' gray delta text, per earlier
    # feedback) - get_pff_color (the app's standard "how good is this
    # player" scale) for anything measuring the PLAYER's own production,
    # get_matchup_color (no inherent good/bad direction) for anything
    # measuring a TEAM's tendency/coverage quality.
    if matchup or def_raw or align_data:
        player_tiles = []
        if matchup:
            player_tiles += [
                {'label': 'Man YPRR', 'value': matchup['player_man_yprr'],
                 'sub': _ordinal_percentile_label(matchup.get('player_man_yprr_pct')),
                 'color': get_pff_color(matchup.get('player_man_yprr_pct'))},
                {'label': 'Zone YPRR', 'value': matchup['player_zone_yprr'],
                 'sub': _ordinal_percentile_label(matchup.get('player_zone_yprr_pct')),
                 'color': get_pff_color(matchup.get('player_zone_yprr_pct'))},
            ]
        if 'slot_rate' in align_data:
            player_tiles.append({
                'label': 'Slot %', 'value': f"{align_data['slot_rate']}%",
                'sub': _ordinal_percentile_label(align_data.get('slot_rate_pct')),
                'color': get_pff_color(align_data.get('slot_rate_pct')),
            })
        if 'slot_yprr' in align_data:
            player_tiles.append({
                'label': 'Slot YPRR', 'value': align_data['slot_yprr'],
                'sub': _ordinal_percentile_label(align_data.get('slot_yprr_pct')),
                'color': get_pff_color(align_data.get('slot_yprr_pct')),
            })
        if 'wide_yprr' in align_data:
            player_tiles.append({
                'label': 'Wide YPRR', 'value': align_data['wide_yprr'],
                'sub': _ordinal_percentile_label(align_data.get('wide_yprr_pct')),
                'color': get_pff_color(align_data.get('wide_yprr_pct')),
            })

        blended_tiles = []
        if matchup:
            blended_tiles.append({
                'label': 'Blended Exp. YPRR (Man/Zone)', 'value': matchup['blended_expected_yprr'],
                'sub': _ordinal_percentile_label(matchup.get('blended_expected_yprr_pct')),
                'color': get_pff_color(matchup.get('blended_expected_yprr_pct')),
            })
        if 'blended_alignment_yprr' in align_data:
            blended_tiles.append({
                'label': 'Blended Exp. YPRR (Slot Align)', 'value': align_data['blended_alignment_yprr'],
                'sub': _ordinal_percentile_label(align_data.get('blended_alignment_yprr_pct')),
                'color': get_pff_color(align_data.get('blended_alignment_yprr_pct')),
            })

        team_tiles = []
        if matchup:
            team_tiles += [
                {'label': 'Opp Man %', 'value': f"{matchup['opp_man_rate']}%",
                 'sub': _ordinal_percentile_label(matchup.get('opp_man_rate_pct')),
                 'color': get_matchup_color(matchup.get('opp_man_rate_pct'))},
                {'label': 'Opp Zone %', 'value': f"{matchup['opp_zone_rate']}%",
                 'sub': _ordinal_percentile_label(matchup.get('opp_zone_rate_pct')),
                 'color': get_matchup_color(matchup.get('opp_zone_rate_pct'))},
            ]
        if def_raw:
            # def_vals is the SAME percentile array already computed for the
            # defense-side radar chart (build_defense_radar_data), parallel
            # to def_labels/def_raw by construction - no new computation
            # needed, just reused here for the tiles' color/sub.
            # get_matchup_color (not get_pff_color): these are already an
            # INVERTED percentile (low yards allowed = good coverage = high
            # percentile, see build_defense_radar_data's docstring), so
            # "good defense = green" is the correct read.
            def_pct_map = dict(zip(def_labels, def_vals))
            team_tiles += [
                {
                    'label': label, 'value': f"{val:.1f}",
                    'sub': _ordinal_percentile_label(def_pct_map.get(label)),
                    'color': get_matchup_color(def_pct_map.get(label)),
                }
                for label, val in def_raw.items()
            ]

        col_player, col_blend, col_team = st.columns([5, 2, 6])
        with col_player:
            st.markdown("<div class='metric-col-label'>PLAYER</div>", unsafe_allow_html=True)
            if player_tiles:
                render_percentile_metric_tiles(player_tiles)
            else:
                st.caption("No player-side coverage/alignment data available.")
        with col_blend:
            st.markdown("<div class='metric-col-label'>BLENDED</div>", unsafe_allow_html=True)
            if blended_tiles:
                render_percentile_metric_tiles(blended_tiles)
            else:
                st.caption("Not enough data for a blended matchup score.")
        with col_team:
            st.markdown("<div class='metric-col-label'>TEAM</div>", unsafe_allow_html=True)
            if team_tiles:
                render_percentile_metric_tiles(team_tiles)
            else:
                st.caption("No defensive coverage data available for this team.")
    else:
        st.caption("Not enough coverage-snap data to compute matchup tiles for this pairing.")

    _render_sumersports_context(full_name)


def _render_sumersports_context(full_name):
    tendency = load_sumersports_tendency_data()
    def_personnel = tendency.get('def_personnel', pd.DataFrame())
    def_formation = tendency.get('def_formation', pd.DataFrame())
    if def_personnel.empty and def_formation.empty:
        return

    st.markdown("<div class='custom-section-header'>SCHEME CONTEXT — SUMERSPORTS 2025</div>", unsafe_allow_html=True)
    if not full_name:
        st.caption("No TEAM_CONFIG match for this PFF team code.")
        return

    cc1, cc2 = st.columns(2)
    with cc1:
        if not def_personnel.empty:
            row = def_personnel[def_personnel['team'] == full_name]
            st.markdown("**11 Personnel faced**")
            if not row.empty:
                r = row.iloc[0]
                st.write(f"Rate: {r['rate_pct']:.1f}% (league avg {r['league_avg_usage_pct']:.1f}%)")
                st.write(f"EPA allowed: {r['epa']:.1f} (rank {int(r['epa_rank'])})")
            else:
                st.caption(f"No SumerSports row for {full_name}.")
    with cc2:
        if not def_formation.empty:
            row = def_formation[def_formation['team'] == full_name]
            st.markdown("**2X2 Formation faced**")
            if not row.empty:
                r = row.iloc[0]
                st.write(f"Rate: {r['rate_pct']:.1f}% (league avg {r['league_avg_usage_pct']:.1f}%)")
                st.write(f"EPA allowed: {r['epa']:.1f} (rank {int(r['epa_rank'])})")
            else:
                st.caption(f"No SumerSports row for {full_name}.")
