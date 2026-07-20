"""
Per-player positional stat snapshot: the shared data-gathering logic behind
Player Search's "LEAGUE POSITION-RELATIVE MATRIX AVERAGES" table AND the
Player Compare tab's percentile-bar chart, so both stay in sync from one
source instead of maintaining two copies of ~60 individual per-position
stat lookups that would slowly drift apart.

build_player_snapshot(...) is a line-by-line port of what used to be
inline in ui/tabs/player_search.py's render() - same branches (QB / RB /
[WR, TE] / OLINE_POSITIONS / DEFENSIVE_POSITIONS / K / [P, LS]), same PFF
lookups, same fallback colors, same conditions for whether a stat appears
at all - just returning a list of dicts instead of building two parallel
plain-value/color dicts, so the raw percentile driving each cell's color
(previously discarded once the color string was computed) survives for
the new percentile-bar chart to use as bar length. New EPA/success-rate
entries (data.transforms.build_epa_efficiency, merged into
precompute_league_percentiles) are appended within the relevant existing
branches - additions, not modifications to what was already there.

Lives in ui/, not data/transforms.py, because it needs ui.styling.get_pff_color
and produces display-ready (not raw) output - keeps this app's existing
data/ui layering intact.
"""
import re

import pandas as pd
import streamlit as st

from config import THEME, OLINE_POSITIONS, DEFENSIVE_POSITIONS
from data.utils import get_val, calculate_percentile
from ui.styling import get_pff_color

C = THEME['colors']

# The two hardcoded fallback colors the original inline code used in
# different places - preserved verbatim rather than unified into one
# constant, since they were never actually the same color: "#313131" is
# the ternary-pattern fallback (a stat whose percentile row/column simply
# doesn't exist), "#131b38" is the matrix table's own row background
# (what a cell renders as when it was never added to cell_colors_map at
# all - a couple of K/P stats have no color entry in the original code).
_TERNARY_FALLBACK = "#313131"
_NO_COLOR_ENTRY_FALLBACK = "#131b38"


def _entry(label, value_str, pct, color=None, fallback_color=_TERNARY_FALLBACK):
    """
    One snapshot row. `pct=None` means the original code had no real
    percentile driving this cell's color at all (it fell back to a
    hardcoded gray) - preserved exactly, and the percentile-bar chart
    skips any entry with pct=None (a bar of undefined length has nothing
    meaningful to show). `color`, when given, overrides the pct-driven
    get_pff_color(pct) computation entirely - needed for 'Overall PFF',
    whose original color is grade_color (rookie-aware: a fixed color for
    rookies regardless of their numeric grade), not a fresh percentile
    lookup.
    """
    if color is not None:
        final_color = color
    elif pct is not None:
        final_color = get_pff_color(pct)
    else:
        final_color = fallback_color
    return {'label': label, 'value_str': value_str, 'pct': pct, 'color': final_color}


def position_branch_group(pos):
    """
    Which other positions share this position's exact stat branch in
    build_player_snapshot below - QB alone; RB alone; WR and TE together
    (identical branch); every OLINE_POSITIONS code together; every
    DEFENSIVE_POSITIONS code together; K alone; P and LS together. Used by
    the Player Compare tab to restrict its second player picker to
    genuinely comparable players - a QB vs. RB comparison would render an
    almost-empty shared-stat table (they only ever overlap on Snap %/
    Overall PFF), whereas WR vs. TE produces a fully aligned table.
    Anything matching none of build_player_snapshot's branches (just FB
    today) only compares against itself.
    """
    if pos == 'QB':
        return ['QB']
    if pos == 'RB':
        return ['RB']
    if pos in ('WR', 'TE'):
        return ['WR', 'TE']
    if pos in OLINE_POSITIONS:
        return list(OLINE_POSITIONS)
    if pos in DEFENSIVE_POSITIONS:
        return list(DEFENSIVE_POSITIONS)
    if pos == 'K':
        return ['K']
    if pos in ('P', 'LS'):
        return ['P', 'LS']
    return [pos]


def build_player_snapshot(selected_player, pos, p_data, p_bio, pff, p_pct_row, games_played, overall_grade, grade_color, target_year):
    """
    Returns an ordered list of {'label', 'value_str', 'pct', 'color'} dicts
    for one player's season. FB matches none of the position branches below
    (a pre-existing gap - config.OFFENSE_SKILL_POSITIONS elsewhere treats FB
    as its own code, never folded into RB) and deliberately keeps getting
    only Snap %/Overall PFF, exactly as before this function existed.
    """
    snapshot = []

    # CORE SNAP PERCENTAGE SHOWN ON EVERY PROFILE - "N/A" (not "0.0%") when
    # this source has no snap-count row for the player at all (mainly
    # offensive linemen - this local export barely tracks their snaps).
    if not bool(p_bio.get('has_snap_match', True)):
        snapshot.append(_entry('Snap %', 'N/A', None, fallback_color=C['surface_container_high']))
    else:
        val = get_val(p_bio, 'snap_pct_avg', "{:.1f}%")
        pct = p_pct_row['snap_pct_avg_max_pct'].iloc[0] if not p_pct_row.empty and 'snap_pct_avg_max_pct' in p_pct_row.columns else float(p_bio.get('snap_pct_avg', 0))
        snapshot.append(_entry('Snap %', val, pct))

    if pos not in DEFENSIVE_POSITIONS + OLINE_POSITIONS:
        snapshot.append(_entry('Overall PFF', f"{overall_grade:.1f}", overall_grade, color=grade_color))

    if pos == 'QB':
        df_qbr_season = pff['qbr_season']
        if not df_qbr_season.empty:
            # qbr_season_level.csv has one row per QB PER SEASON (2006-2025)
            # plus separate Regular/Playoffs rows - filtering to the
            # selected year's regular season avoids grabbing a frozen
            # earliest-tracked-season row.
            name_match = df_qbr_season['name_display'].str.lower() == selected_player.lower()
            if 'season' in df_qbr_season.columns:
                name_match &= df_qbr_season['season'] == target_year
            if 'season_type' in df_qbr_season.columns:
                name_match &= df_qbr_season['season_type'] == 'Regular'
            qbr_row = df_qbr_season[name_match]
            if not qbr_row.empty:
                snapshot.append(_entry('ESPN QBR', get_val(qbr_row.iloc[0], 'qbr_total', "{:.1f}"), qbr_row.iloc[0].get('qbr_pct', 0)))

        df_pff_pass = pff['pass']
        if not df_pff_pass.empty:
            p_pff = df_pff_pass[df_pff_pass['player'].str.lower() == selected_player.lower()]
            if not p_pff.empty:
                row = p_pff.iloc[0]
                snapshot.append(_entry('Pass Grade', get_val(row, 'grades_pass', "{:.1f}"), row.get('grades_pass_pct', 0)))
                snapshot.append(_entry('Run Grade', get_val(row, 'grades_run', "{:.1f}"), row.get('grades_run_pct', 0)))
                snapshot.append(_entry('BTT %', get_val(row, 'btt_rate', "{:.1f}%"), row.get('btt_pct', 0)))
                snapshot.append(_entry('TWP %', get_val(row, 'twp_rate', "{:.1f}%"), row.get('twp_pct', 0)))
                adj_comp_val = get_val(row, 'adj_comp_pct', "{:.1f}%") if 'adj_comp_pct' in df_pff_pass.columns else "--"
                snapshot.append(_entry('Adj Comp %', adj_comp_val, row.get('adj_comp_pct', 0)))
                snapshot.append(_entry('ADOT', get_val(row, 'avg_depth_of_target', "{:.1f}"), row.get('adot_pct', 0)))
                snapshot.append(_entry('P2S %', get_val(row, 'pressure_to_sack_rate', "{:.1f}%"), row.get('p2s_pct', 0)))

        # EPA/success-rate/CPOE - new, from data.transforms.build_epa_efficiency
        # via precompute_league_percentiles (p_pct_row already carries these
        # merged-in columns and their _pct siblings, same access pattern as
        # every other percentile-row lookup on this page).
        if not p_pct_row.empty:
            r = p_pct_row.iloc[0]
            if pd.notna(r.get('epa_per_dropback')):
                snapshot.append(_entry('EPA/Dropback', f"{r['epa_per_dropback']:.2f}", r.get('epa_per_dropback_pct', 0)))
            if pd.notna(r.get('success_rate_pass')):
                snapshot.append(_entry('Success Rate', f"{r['success_rate_pass'] * 100:.1f}%", r.get('success_rate_pass_pct', 0)))
            if pd.notna(r.get('cpoe_avg')):
                snapshot.append(_entry('CPOE', f"{r['cpoe_avg']:.1f}", r.get('cpoe_avg_pct', 0)))

    elif pos == 'RB':
        r_att = p_data['rushing_attempts'].sum() if 'rushing_attempts' in p_data.columns else 0
        r_yds = p_data['rushing_yards'].sum() if 'rushing_yards' in p_data.columns else 0
        tgts = p_data['targets'].sum() if 'targets' in p_data.columns else 0

        rush_att_pct = p_pct_row['rush_att_per_g_pct'].iloc[0] if not p_pct_row.empty and 'rush_att_per_g_pct' in p_pct_row.columns else None
        snapshot.append(_entry('Rush Att/G', f"{r_att / games_played:.1f}", rush_att_pct))
        targets_pct = p_pct_row['targets_per_g_pct'].iloc[0] if not p_pct_row.empty and 'targets_per_g_pct' in p_pct_row.columns else None
        snapshot.append(_entry('Targets/G', f"{tgts / games_played:.1f}", targets_pct))
        ypc_pct = p_pct_row['ypc_pct'].iloc[0] if not p_pct_row.empty and 'ypc_pct' in p_pct_row.columns else None
        snapshot.append(_entry('Yards/Carry', f"{r_yds / max(r_att, 1):.1f}", ypc_pct))

        df_pff_rec = pff['rec']
        if not df_pff_rec.empty:
            p_pff_rec = df_pff_rec[df_pff_rec['player'].str.lower() == selected_player.lower()]
            if not p_pff_rec.empty:
                snapshot.append(_entry('RecV Grade', get_val(p_pff_rec.iloc[0], 'grades_pass_route', "{:.1f}"), p_pff_rec.iloc[0].get('grades_pass_route_pct', 0)))

        df_pff_pass_block = pff['pass_block']
        if not df_pff_pass_block.empty:
            p_pff_pb = df_pff_pass_block[df_pff_pass_block['player'].str.lower() == selected_player.lower()]
            if not p_pff_pb.empty:
                snapshot.append(_entry('PBLK Grade', get_val(p_pff_pb.iloc[0], 'grades_pass_block', "{:.1f}"), p_pff_pb.iloc[0].get('grades_pass_block_pct', 0)))

        df_pff_rush = pff['rush']
        if not df_pff_rush.empty:
            p_pff = df_pff_rush[df_pff_rush['player'].str.lower() == selected_player.lower()]
            if not p_pff.empty:
                row = p_pff.iloc[0]
                # row.get(col, 0) only substitutes 0 when the KEY is
                # missing - a present column with a genuinely NaN value for
                # this row (a backup with too few carries to be tracked)
                # passes straight through float(), producing the literal
                # string "nan" once formatted. pd.notna guards that.
                # Column is 'elu_rush_mtf', not 'missed_tackles_forced' -
                # see the matching fix in data.loaders' mtf_pct computation
                # for why (that column doesn't exist in rushing_summary.csv
                # at all - this used to show "--" for every RB, every year).
                mtf_raw = row.get('elu_rush_mtf')
                if pd.notna(mtf_raw):
                    snapshot.append(_entry('MTF/G', f"{float(mtf_raw) / games_played:.1f}", row.get('mtf_pct', 0)))
                else:
                    snapshot.append(_entry('MTF/G', '--', None))
                snapshot.append(_entry('YCO/Att', get_val(row, 'yco_attempt', "{:.2f}"), row.get('yco_pct', 0)))

        # EPA/success-rate - new.
        if not p_pct_row.empty:
            r = p_pct_row.iloc[0]
            if pd.notna(r.get('epa_per_rush')):
                snapshot.append(_entry('EPA/Rush', f"{r['epa_per_rush']:.2f}", r.get('epa_per_rush_pct', 0)))
            if pd.notna(r.get('success_rate_rush')):
                snapshot.append(_entry('Success Rate', f"{r['success_rate_rush'] * 100:.1f}%", r.get('success_rate_rush_pct', 0)))

    elif pos in ['WR', 'TE']:
        tgts = p_data['targets'].sum() if 'targets' in p_data.columns else 0
        recs = p_data['receptions'].sum() if 'receptions' in p_data.columns else 0

        targets_pct = p_pct_row['targets_per_g_pct'].iloc[0] if not p_pct_row.empty and 'targets_per_g_pct' in p_pct_row.columns else None
        snapshot.append(_entry('Targets/G', f"{tgts / games_played:.1f}", targets_pct))
        rec_pct = p_pct_row['rec_per_g_pct'].iloc[0] if not p_pct_row.empty and 'rec_per_g_pct' in p_pct_row.columns else None
        snapshot.append(_entry('Rec/G', f"{recs / games_played:.1f}", rec_pct))

        df_pff_rec = pff['rec']
        df_pff_route_concept = pff['route_concept']
        if not df_pff_rec.empty:
            p_pff = df_pff_rec[df_pff_rec['player'].str.lower() == selected_player.lower()]
            if not p_pff.empty:
                row = p_pff.iloc[0]
                snapshot.append(_entry('RecV Grade', get_val(row, 'grades_pass_route', "{:.1f}"), row.get('grades_pass_route_pct', 0)))
                snapshot.append(_entry('ADOT', get_val(row, 'avg_depth_of_target', "{:.1f}"), row.get('adot_pct', 0)))
                snapshot.append(_entry('YPRR', get_val(row, 'yprr', "{:.2f}"), row.get('yprr_pct', 0)))
                snapshot.append(_entry('Drop Rate %', get_val(row, 'drop_rate', "{:.1f}%"), row.get('drop_pct', 0)))
                snapshot.append(_entry('SL %', get_val(row, 'slot_rate', "{:.1f}%"), row.get('slot_rate_pct', 0)))

                if pos == 'WR':
                    snapshot.append(_entry('WID %', get_val(row, 'wide_rate', "{:.1f}%"), row.get('wide_rate_pct', 0)))
                else:
                    snapshot.append(_entry('INL %', get_val(row, 'inline_rate', "{:.1f}%"), row.get('inline_rate_pct', 0)))

                # NOTE: 'yprr_slot'/'yprr_screen'/'grades_pass_route_slot' used
                # to be read from THIS dataframe (pff['rec'], the plain
                # receiving_summary export) right here - confirmed by direct
                # inspection that none of those three columns actually exist
                # in that file (only their already-precomputed _pct siblings
                # do, via data.loaders.load_all_pff_data's
                # calculate_percentile(df, missing_col) call, which returns
                # an all-zero Series rather than raising - see that
                # function's own docstring). That silently showed "--" for
                # every single WR/TE, every season, forever - not a sparse-
                # data edge case, a 100%-of-the-time dead lookup. The real
                # slot/screen-alignment split lives in a DIFFERENT PFF
                # export entirely (receiving_concept, loaded as
                # df_pff_route_concept below), under different column names
                # ('slot_yprr'/'screen_yprr'/'slot_grades_pass_route' - note
                # the word order flip on the last one) - 'Slot YPRR'/'Screen
                # YPRR' immediately below were already correctly reading
                # from there, making the two YPRR entries here exact
                # duplicates of a working entry two lines down. Removed
                # rather than fixed in place; 'Slot RecV Grd' (no working
                # duplicate) moved down to the route_concept block with the
                # correct source/column name instead of being deleted.

        if not df_pff_route_concept.empty:
            p_rc = df_pff_route_concept[df_pff_route_concept['player'].str.lower() == selected_player.lower()]
            if not p_rc.empty:
                row = p_rc.iloc[0]
                snapshot.append(_entry('Slot Route %', get_val(row, 'slot_route_rate', "{:.1f}%"),
                                       calculate_percentile(df_pff_route_concept, 'slot_route_rate').loc[p_rc.index[0]] if 'slot_route_rate' in df_pff_route_concept.columns else 0))
                snapshot.append(_entry('Slot YPRR', get_val(row, 'slot_yprr', "{:.2f}"),
                                       calculate_percentile(df_pff_route_concept, 'slot_yprr').loc[p_rc.index[0]] if 'slot_yprr' in df_pff_route_concept.columns else 0))
                snapshot.append(_entry('Slot RecV Grd', get_val(row, 'slot_grades_pass_route', "{:.1f}"),
                                       calculate_percentile(df_pff_route_concept, 'slot_grades_pass_route').loc[p_rc.index[0]] if 'slot_grades_pass_route' in df_pff_route_concept.columns else 0))
                snapshot.append(_entry('Screen Route %', get_val(row, 'screen_route_rate', "{:.1f}%"),
                                       calculate_percentile(df_pff_route_concept, 'screen_route_rate').loc[p_rc.index[0]] if 'screen_route_rate' in df_pff_route_concept.columns else 0))
                snapshot.append(_entry('Screen YPRR', get_val(row, 'screen_yprr', "{:.2f}"),
                                       calculate_percentile(df_pff_route_concept, 'screen_yprr').loc[p_rc.index[0]] if 'screen_yprr' in df_pff_route_concept.columns else 0))

        # EPA/success-rate (per target) - new.
        if not p_pct_row.empty:
            r = p_pct_row.iloc[0]
            if pd.notna(r.get('epa_per_target')):
                snapshot.append(_entry('EPA/Target', f"{r['epa_per_target']:.2f}", r.get('epa_per_target_pct', 0)))
            if pd.notna(r.get('success_rate_rec')):
                snapshot.append(_entry('Success Rate', f"{r['success_rate_rec'] * 100:.1f}%", r.get('success_rate_rec_pct', 0)))

    elif pos in OLINE_POSITIONS:
        snapshot.append(_entry('Overall PFF', f"{overall_grade:.1f}", overall_grade, color=grade_color))

        df_pff_pass_block = pff['pass_block']
        if not df_pff_pass_block.empty:
            p_pff = df_pff_pass_block[df_pff_pass_block['player'].str.lower() == selected_player.lower()]
            if not p_pff.empty:
                row = p_pff.iloc[0]
                snapshot.append(_entry('PBLK Grade', get_val(row, 'grades_pass_block', "{:.1f}"), row.get('grades_pass_block_pct', 0)))
                snapshot.append(_entry('Pass Block Eff', get_val(row, 'pbe', "{:.1f}"), row.get('pbe_pct', 0)))
                # Same NaN-leak guard as MTF/G above - pressures_allowed can
                # be a genuinely NaN value for this row, not just a missing
                # column, which float()/format() would otherwise turn into
                # a literal "nan" string.
                pr_allowed_raw = row.get('pressures_allowed')
                if pd.notna(pr_allowed_raw):
                    snapshot.append(_entry('PR Allowed/G', f"{float(pr_allowed_raw) / games_played:.1f}", row.get('pr_allowed_per_game_pct', 0)))
                else:
                    snapshot.append(_entry('PR Allowed/G', '--', None))

        df_pff_run_block = pff['run_block']
        if not df_pff_run_block.empty:
            p_pff = df_pff_run_block[df_pff_run_block['player'].str.lower() == selected_player.lower()]
            if not p_pff.empty:
                row = p_pff.iloc[0]
                snapshot.append(_entry('Run Block Grade', get_val(row, 'grades_run_block', "{:.1f}"), row.get('grades_run_block_pct', 0)))

    elif pos in DEFENSIVE_POSITIONS:
        snapshot.append(_entry('Overall PFF', f"{overall_grade:.1f}", overall_grade, color=grade_color))

        tkls = p_data['tackles'].sum() if 'tackles' in p_data.columns else 0
        scks = p_data['sacks'].sum() if 'sacks' in p_data.columns else 0
        dints = p_data['def_ints'].sum() if 'def_ints' in p_data.columns else 0
        ffs = p_data['forced_fumbles'].sum() if 'forced_fumbles' in p_data.columns else 0

        tackles_pct = p_pct_row['tackles_per_g_pct'].iloc[0] if not p_pct_row.empty and 'tackles_per_g_pct' in p_pct_row.columns else None
        snapshot.append(_entry('Tackles/G', f"{tkls / games_played:.1f}", tackles_pct))
        sacks_pct = p_pct_row['sacks_sum_pct'].iloc[0] if not p_pct_row.empty and 'sacks_sum_pct' in p_pct_row.columns else None
        snapshot.append(_entry('Total Sacks', f"{scks:.1f}", sacks_pct))
        ints_pct = p_pct_row['def_ints_sum_pct'].iloc[0] if not p_pct_row.empty and 'def_ints_sum_pct' in p_pct_row.columns else None
        snapshot.append(_entry('INTs', f"{int(dints)}", ints_pct))
        ff_pct = p_pct_row['forced_fumbles_sum_pct'].iloc[0] if not p_pct_row.empty and 'forced_fumbles_sum_pct' in p_pct_row.columns else None
        snapshot.append(_entry('Forced Fum', f"{int(ffs)}", ff_pct))

        df_pff_pass_rush = pff['pass_rush']
        if not df_pff_pass_rush.empty:
            p_pff = df_pff_pass_rush[df_pff_pass_rush['player'].str.lower() == selected_player.lower()]
            if not p_pff.empty:
                row = p_pff.iloc[0]
                snapshot.append(_entry('Win Rate', get_val(row, 'pass_rush_win_rate', "{:.1f}%"), row.get('win_pct', 0)))
                snapshot.append(_entry('PRP', get_val(row, 'prp', "{:.1f}"), row.get('prp_pct', 0)))

        df_pff_run_def = pff['run_def']
        if not df_pff_run_def.empty:
            p_pff = df_pff_run_def[df_pff_run_def['player'].str.lower() == selected_player.lower()]
            if not p_pff.empty:
                row = p_pff.iloc[0]
                snapshot.append(_entry('Run Stop Rate', get_val(row, 'stop_percent', "{:.1f}%"), row.get('stop_pct', 0)))

        df_pff_cov_summary = pff['cov_summary']
        if not df_pff_cov_summary.empty:
            p_pff = df_pff_cov_summary[df_pff_cov_summary['player'].str.lower() == selected_player.lower()]
            if not p_pff.empty:
                row = p_pff.iloc[0]
                snapshot.append(_entry('Missed Tkl Rate', get_val(row, 'missed_tackle_rate', "{:.1f}%"), row.get('mtf_rate_pct', 0)))
                snapshot.append(_entry('Pass Rtg Allowed', get_val(row, 'qb_rating_against', "{:.1f}"), row.get('pass_rtg_pct', 0)))

    elif pos == 'K':
        df_pff_fg = pff['fg']
        if not df_pff_fg.empty:
            p_pff = df_pff_fg[df_pff_fg['player'].str.lower() == selected_player.lower()]
            if not p_pff.empty:
                row = p_pff.iloc[0]
                snapshot.append(_entry('FG Grade', get_val(row, 'grades_fgep_kicker', "{:.1f}"),
                                       calculate_percentile(df_pff_fg, 'grades_fgep_kicker').loc[p_pff.index[0]] if 'grades_fgep_kicker' in df_pff_fg.columns else 0))
                snapshot.append(_entry('FG %', get_val(row, 'total_percent', "{:.1f}%"),
                                       calculate_percentile(df_pff_fg, 'total_percent').loc[p_pff.index[0]] if 'total_percent' in df_pff_fg.columns else 0))
                snapshot.append(_entry('PAT %', get_val(row, 'pat_percent', "{:.1f}%"),
                                       calculate_percentile(df_pff_fg, 'pat_percent').loc[p_pff.index[0]] if 'pat_percent' in df_pff_fg.columns else 0))
                snapshot.append(_entry('50+ %', get_val(row, 'fifty_percent', "{:.1f}%"),
                                       calculate_percentile(df_pff_fg, 'fifty_percent').loc[p_pff.index[0]] if 'fifty_percent' in df_pff_fg.columns else 0))
                snapshot.append(_entry('40-49 %', get_val(row, 'forty_percent', "{:.1f}%"),
                                       calculate_percentile(df_pff_fg, 'forty_percent').loc[p_pff.index[0]] if 'forty_percent' in df_pff_fg.columns else 0))

    elif pos in ['P', 'LS']:
        df_pff_punt = pff['punt']
        if not df_pff_punt.empty:
            p_pff = df_pff_punt[df_pff_punt['player'].str.lower() == selected_player.lower()]
            if not p_pff.empty:
                row = p_pff.iloc[0]
                snapshot.append(_entry('Punt Grade', get_val(row, 'grades_punter', "{:.1f}"),
                                       calculate_percentile(df_pff_punt, 'grades_punter').loc[p_pff.index[0]] if 'grades_punter' in df_pff_punt.columns else 0))
                snapshot.append(_entry('Net Yds/Punt', get_val(row, 'average_net_yards', "{:.1f}"),
                                       calculate_percentile(df_pff_punt, 'average_net_yards').loc[p_pff.index[0]] if 'average_net_yards' in df_pff_punt.columns else 0))
                snapshot.append(_entry('Hang Time', get_val(row, 'average_hangtime', "{:.2f}"),
                                       calculate_percentile(df_pff_punt, 'average_hangtime').loc[p_pff.index[0]] if 'average_hangtime' in df_pff_punt.columns else 0))
                # Inside 20s/Touchbacks never had a cell_colors_map entry in
                # the original code at all - they fell through to the
                # matrix table's own row-background default at render time
                # (`cell_colors_map.get(col, '#131b38')`), not a computed
                # color. Reproduced here with the same literal fallback and
                # pct=None (nothing meaningful to show as a bar).
                snapshot.append(_entry('Inside 20s', get_val(row, 'inside_twenties', "{:.0f}"), None, fallback_color=_NO_COLOR_ENTRY_FALLBACK))
                snapshot.append(_entry('Touchbacks', get_val(row, 'touchbacks', "{:.0f}"), None, fallback_color=_NO_COLOR_ENTRY_FALLBACK))

    return snapshot


# Curated headline stats per position group for the percentile-bar chart -
# Baseball Savant's own Percentile Rankings visual is deliberately a
# curated ~13-15 stat subset, not every tracked number, so a chart with
# every single matrix-table entry (a WR/TE branch has 15-20) would be
# noisier than the thing this is modeled on. Falls back to showing
# whatever's in the snapshot if none of the curated labels are present
# (e.g. a position/season with sparse PFF data) rather than showing nothing.
BAR_CHART_HEADLINE_STATS = {
    'QB': ['Overall PFF', 'EPA/Dropback', 'Success Rate', 'CPOE', 'ESPN QBR', 'Pass Grade', 'BTT %', 'TWP %', 'ADOT'],
    # Snap % appended per explicit feedback (Usage & Alignment promoted out
    # of the "More stats" picker into the always-visible percentile
    # rankings) - RB has no slot/wide/inline alignment equivalent to WR/TE,
    # so Snap % (the "usage" half of that group) is the only addition here.
    'RB': ['Overall PFF', 'EPA/Rush', 'Success Rate', 'Yards/Carry', 'Rush Att/G', 'Targets/G', 'RecV Grade', 'MTF/G', 'YCO/Att', 'Snap %'],
    # Snap %/SL %/WID %/INL % appended - the WR/TE 'Usage & Alignment'
    # SECONDARY_TILE_GROUPS entries below are removed in the same pass,
    # since every stat that group held now lives here instead.
    'WR': ['Overall PFF', 'EPA/Target', 'Success Rate', 'YPRR', 'Targets/G', 'Rec/G', 'RecV Grade', 'ADOT', 'Drop Rate %', 'Snap %', 'SL %', 'WID %'],
    'TE': ['Overall PFF', 'EPA/Target', 'Success Rate', 'YPRR', 'Targets/G', 'Rec/G', 'RecV Grade', 'ADOT', 'Drop Rate %', 'Snap %', 'SL %', 'INL %'],
}

# Secondary tile grouping for Player Search's season profile - everything
# NOT in BAR_CHART_HEADLINE_STATS above, split into the natural stat
# families a user would ask for separately (per explicit feedback that
# showing every snapshot stat at once - a WR/TE has ~19 - was too crowded:
# slot stats, screen stats, and alignment/usage stats should be selectable
# separately rather than all boxed on screen together). Labels listed here
# that a given player/season doesn't have are skipped at render time;
# snapshot entries not covered by any group land in a trailing "Other"
# bucket so nothing silently disappears.
SECONDARY_TILE_GROUPS = {
    'QB': [('More Stats', ['Snap %', 'Run Grade', 'Adj Comp %', 'P2S %'])],
    # RB's Snap % moved to BAR_CHART_HEADLINE_STATS above (promoted to the
    # primary percentile rankings) - only PBLK Grade is left here.
    'RB': [('More Stats', ['PBLK Grade'])],
    # WR/TE's 'Usage & Alignment' group removed entirely - every stat it
    # held (Snap %, SL %, WID %/INL %) moved to BAR_CHART_HEADLINE_STATS
    # above, per explicit feedback that those specifically belonged in the
    # always-visible percentile rankings rather than behind a picker. Slot
    # Production and Screen Game stay exactly as before ("these can stay as
    # more stats... good to have if I do want them").
    'WR': [
        ('Slot Production', ['Slot RecV Grd', 'Slot Route %', 'Slot YPRR']),
        ('Screen Game', ['Screen Route %', 'Screen YPRR']),
    ],
    'TE': [
        ('Slot Production', ['Slot RecV Grd', 'Slot Route %', 'Slot YPRR']),
        ('Screen Game', ['Screen Route %', 'Screen YPRR']),
    ],
}


def split_snapshot_for_display(snapshot, pos):
    """
    Splits a build_player_snapshot list into (primary_entries,
    [(group_name, entries), ...]) for display: primary = the same curated
    headline set the percentile-bar chart uses (BAR_CHART_HEADLINE_STATS),
    secondary = everything else bucketed by SECONDARY_TILE_GROUPS. For
    positions with no headline curation (OL/defense/K/P - already small
    stat sets), or a season so sparse that none of the headline stats
    exist, returns the whole snapshot as primary with no groups, exactly
    the old single-grid behavior.
    """
    headline = BAR_CHART_HEADLINE_STATS.get(pos)
    if not headline:
        return snapshot, []
    by_label = {e['label']: e for e in snapshot}
    primary = [by_label[l] for l in headline if l in by_label]
    if not primary:
        return snapshot, []
    used = {l for l in headline if l in by_label}
    groups = []
    for gname, labels in SECONDARY_TILE_GROUPS.get(pos, []):
        rows = [by_label[l] for l in labels if l in by_label and l not in used]
        used.update(r['label'] for r in rows)
        if rows:
            groups.append((gname, rows))
    leftovers = [e for e in snapshot if e['label'] not in used]
    if leftovers:
        groups.append(('Other', leftovers))
    return primary, groups


# Small-multiples grouping for render_percentile_radar_grid - the same
# BAR_CHART_HEADLINE_STATS curation, split into the stat categories a real
# scouting report would separate (how efficiently they play vs. how much
# they're used vs. the finer technique markers), one compact radar per
# group instead of one long bar list. Positions with no entry here
# (OLINE/DEFENSIVE_POSITIONS/K/[P,LS]) fall back to the plain bar chart as
# their ONLY view rather than being demoted into an expander with nothing
# to replace it - a 3-axis-minimum radar isn't a clear win over those
# positions' already-small stat sets.
RADAR_GRID_GROUPS = {
    'QB': [
        ('Efficiency', ['EPA/Dropback', 'Success Rate', 'CPOE', 'Pass Grade']),
        ('Ball Security', ['BTT %', 'TWP %', 'Adj Comp %']),
        ('Situational', ['ESPN QBR', 'ADOT', 'P2S %']),
    ],
    'RB': [
        ('Efficiency', ['EPA/Rush', 'Success Rate', 'Yards/Carry']),
        ('Volume', ['Rush Att/G', 'Targets/G']),
        ('Technique', ['RecV Grade', 'MTF/G', 'YCO/Att']),
    ],
    'WR': [
        ('Efficiency', ['EPA/Target', 'Success Rate', 'YPRR']),
        ('Volume', ['Targets/G', 'Rec/G', 'Slot Route %']),
        ('Technique', ['ADOT', 'Drop Rate %', 'RecV Grade']),
    ],
    'TE': [
        ('Efficiency', ['EPA/Target', 'Success Rate', 'YPRR']),
        ('Volume', ['Targets/G', 'Rec/G']),
        ('Technique', ['ADOT', 'Drop Rate %', 'RecV Grade']),
    ],
}


def _mpl_color(color_str):
    """
    get_pff_color/entry colors are CSS rgba(r, g, b, a) strings (r/g/b as
    0-255 ints) - correct for the HTML/CSS matrix table this same snapshot
    also feeds, but matplotlib's color parser doesn't understand that
    syntax (it wants hex or 0-1 float tuples) and raises ValueError on it.
    Converts rgba(...) to an (r, g, b, a) float tuple in [0, 1]; hex/named
    colors (the "#313131"/"#131b38" fallbacks) pass through unchanged.
    """
    m = re.match(r'rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)', color_str)
    if not m:
        return color_str
    r, g, b = (float(m.group(i)) / 255.0 for i in (1, 2, 3))
    a = float(m.group(4)) if m.group(4) is not None else 1.0
    return (r, g, b, a)


# Single, fixed accent for every percentile radar (card micro-radar + grid)
# rather than per-stat get_pff_color shading - a radar's SHAPE (how far
# each spoke reaches) already carries the percentile signal; recoloring
# each spoke individually the way the bar chart and matrix table do would
# fight the shape for attention instead of reinforcing it. Blue reads as
# "graded" against this app's red-poor/blue-elite convention without
# implying any single spoke here is elite on its own.
_RADAR_ACCENT = '#5b8fe8'


def _wrap_radar_label(label, width=10):
    """
    Two-line wrap for spoke labels so a long stat name ("EPA/Dropback",
    "Success Rate") doesn't run into the neighboring spoke or under the
    filled shape - wraps on spaces first, then on a '/' if the label is
    one long unbroken token.
    """
    import textwrap
    if len(label) <= width:
        return label
    if ' ' in label:
        return '\n'.join(textwrap.wrap(label, width))
    if '/' in label:
        head, _, tail = label.partition('/')
        return f"{head}/\n{tail}"
    return label


def _draw_percentile_radar(ax, labels, pcts, color, label_size=9.5):
    """
    Shared polar-radar drawing routine for render_percentile_radar_grid's
    panels - one filled series on a transparent polar axes styled to match
    the app's navy/technical palette (Agg backend, same as
    data/coverage_radar.py's render_split_radar_figure).

    Label size/padding tuned per explicit feedback (twice now): the
    original 6.5pt labels at pad=2 were both too small to read and
    physically overlapped by the filled radar shape; a later pass enlarged
    them but they still crowded each other and ran into neighboring panels'
    titles at small multiples. This pass pulls labels further from the
    plot circle (tick pad) AND shrinks the wrap width so long stat names
    break to two lines sooner instead of running wide into the next spoke's
    space - both work together with render_percentile_radar_grid's wider
    per-panel figsize/title pad to keep every label clear of the shape, its
    neighbors, and the panel title above it.
    """
    import numpy as np
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    vals = pcts + pcts[:1]
    ax.set_facecolor('none')
    ax.plot(angles, vals, color=color, linewidth=1.8, zorder=3)
    ax.fill(angles, vals, color=color, alpha=0.28, zorder=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([_wrap_radar_label(l, width=8) for l in labels], color='#e5e8ff', size=label_size)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels([])
    ax.spines['polar'].set_color('#2c3260')
    ax.grid(color='#2c3260', linewidth=0.6)
    ax.tick_params(axis='x', pad=20)


def render_percentile_radar_grid(snapshot, player_name, pos):
    """
    High-density grid of small-multiple percentile radars, grouped by stat
    category (RADAR_GRID_GROUPS) - the primary at-a-glance view for
    positions with a curated grouping, ahead of the single percentile-bar
    chart (render_percentile_bars_figure below), which stays available in
    a "Full stat breakdown" expander rather than being removed. Returns
    None for positions with no RADAR_GRID_GROUPS entry, or if fewer than 3
    real-percentile stats are available for every group - callers should
    fall back to showing the bar chart directly (not inside an expander)
    in that case, since there's nothing to demote it in favor of.
    """
    groups = RADAR_GRID_GROUPS.get(pos)
    if not groups:
        return None

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    by_label = {e['label']: e for e in snapshot}
    resolved = []
    for group_name, wanted_labels in groups:
        rows = [by_label[l] for l in wanted_labels if l in by_label and by_label[l]['pct'] is not None]
        if len(rows) >= 3:
            resolved.append((group_name, rows))
    if not resolved:
        return None

    # Panel size / DPI / paddings all raised together (2.6->3.4 wide,
    # 160->200 DPI, tight_layout 0.6->1.6): st.pyplot scales the image to
    # the column width, so a small figure with small fonts rendered the
    # labels blurry AND tiny at display size - a physically larger, denser
    # render downsamples to the same on-screen size with crisp, larger
    # text, and the extra layout padding keeps wrapped labels from
    # clipping at the figure edge.
    # st.pyplot scales the image to the COLUMN width, so what matters for
    # on-screen legibility is the font-size-to-figure-width RATIO, not the
    # absolute point size - the original 7.5pt-on-2.6in panels worked out
    # to tiny, blurry on-screen labels. 12pt on 3.2in panels is ~30%
    # larger at display size, and DPI 200 keeps the downscaled text crisp.
    # Panel width widened (3.2->3.9) and wspace/top margin set explicitly
    # (not left to tight_layout alone) - per repeated feedback that spoke
    # labels crowded each other and ran into neighboring panels/titles at
    # the tighter size. More horizontal room between polar axes plus a
    # reserved top strip for the title (title pad also raised) gives the
    # tick-label offset in _draw_percentile_radar somewhere to land without
    # colliding with either.
    # DPI raised again (200->260) per a later round of the same feedback -
    # this app's layout is full-bleed with no max-width cap, so on a wide
    # or high-DPI display the rendered CSS width can exceed what 200 DPI
    # comfortably supports before the browser's upscale starts softening it.
    fig, axes = plt.subplots(
        1, len(resolved), figsize=(3.9 * len(resolved), 3.6),
        subplot_kw=dict(polar=True), dpi=260,
    )
    fig.patch.set_alpha(0)
    if len(resolved) == 1:
        axes = [axes]
    for ax, (group_name, rows) in zip(axes, resolved):
        labels = [r['label'] for r in rows]
        pcts = [max(0.0, min(100.0, float(r['pct']))) for r in rows]
        _draw_percentile_radar(ax, labels, pcts, _RADAR_ACCENT, label_size=11)
        ax.set_title(group_name, color='#ffffff', size=13, fontweight='bold', pad=38)
    fig.subplots_adjust(wspace=0.75, top=0.78, bottom=0.08)
    return fig


def _draw_percentile_radar_overlay(ax, labels, pcts_1, pcts_2, color_1, color_2, label_size=11):
    """
    Two-series version of _draw_percentile_radar for Player Compare - both
    players' shapes drawn on the SAME polar axes (each player's own team
    color, not the single shared _RADAR_ACCENT the one-player version
    uses) so a "who wins this stat category" read comes from comparing two
    overlapping shapes directly, rather than mentally cross-referencing two
    separate side-by-side images.
    """
    import numpy as np
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    vals_1 = pcts_1 + pcts_1[:1]
    vals_2 = pcts_2 + pcts_2[:1]
    ax.set_facecolor('none')
    ax.plot(angles, vals_1, color=color_1, linewidth=2.0, zorder=4)
    ax.fill(angles, vals_1, color=color_1, alpha=0.24, zorder=3)
    ax.plot(angles, vals_2, color=color_2, linewidth=2.0, zorder=2)
    ax.fill(angles, vals_2, color=color_2, alpha=0.24, zorder=1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([_wrap_radar_label(l, width=8) for l in labels], color='#e5e8ff', size=label_size)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels([])
    ax.spines['polar'].set_color('#2c3260')
    ax.grid(color='#2c3260', linewidth=0.6)
    ax.tick_params(axis='x', pad=20)


def render_percentile_radar_grid_compare(side_1, side_2):
    """
    Overlaid version of render_percentile_radar_grid for Player Compare:
    one radar per stat-category group (RADAR_GRID_GROUPS, keyed off side_1's
    position - ui.tabs.player_compare already restricts Player 2 to the
    same position_branch_group, e.g. WR vs TE, so this is always a
    meaningful pairing), both players drawn on the SAME axes in their own
    team color (side['color'], see ui.tabs.player_compare._load_player_side)
    instead of two separate figures a user has to mentally overlay
    themselves.

    A stat axis only makes the cut if BOTH players have a real percentile
    for it (WR and TE's RADAR_GRID_GROUPS entries aren't identical - e.g.
    TE has no 'Slot Route %' - so a shared shape needs the intersection,
    not just side_1's own list). Returns None under the same "not enough
    real axes" rule as the single-player version (needs >=3 per group).
    """
    groups = RADAR_GRID_GROUPS.get(side_1['pos'])
    if not groups:
        return None

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    by_label_1 = {e['label']: e for e in side_1['snapshot']}
    by_label_2 = {e['label']: e for e in side_2['snapshot']}
    resolved = []
    for group_name, wanted_labels in groups:
        shared = [
            l for l in wanted_labels
            if l in by_label_1 and by_label_1[l]['pct'] is not None
            and l in by_label_2 and by_label_2[l]['pct'] is not None
        ]
        if len(shared) >= 3:
            resolved.append((group_name, shared))
    if not resolved:
        return None

    fig, axes = plt.subplots(
        1, len(resolved), figsize=(3.9 * len(resolved), 3.6),
        subplot_kw=dict(polar=True), dpi=260,
    )
    fig.patch.set_alpha(0)
    if len(resolved) == 1:
        axes = [axes]
    for ax, (group_name, labels) in zip(axes, resolved):
        pcts_1 = [max(0.0, min(100.0, float(by_label_1[l]['pct']))) for l in labels]
        pcts_2 = [max(0.0, min(100.0, float(by_label_2[l]['pct']))) for l in labels]
        _draw_percentile_radar_overlay(ax, labels, pcts_1, pcts_2, side_1['color'], side_2['color'])
        ax.set_title(group_name, color='#ffffff', size=13, fontweight='bold', pad=38)
    fig.subplots_adjust(wspace=0.75, top=0.78, bottom=0.08)
    return fig


def render_percentile_bars_figure(snapshot, player_name, pos):
    """
    Baseball Savant-style "Percentile Rankings" visual: one horizontal bar
    per stat, length = league percentile (0-100), color = each entry's own
    color (get_pff_color(pct), already computed by build_player_snapshot).

    Inline SVG (same convention as ui.components.render_fpts_week_strip /
    render_matchup_heat_strip - a fixed viewBox, real DOM elements, CSS
    hover) instead of matplotlib - this used to sit BELOW a stat-tile grid
    showing the same headline stats (which carried its own hover/highlight
    interaction), so the static PNG this rendered as before was never the
    only view of this data. Per explicit feedback, that duplicate tile grid
    was removed and this chart became the sole primary view - a flat
    matplotlib image can't grow a per-bar hover affordance on its own, so
    it needed a real interactive markup instead.

    Renders directly via st.markdown - no Figure to return. Entries with
    pct=None (no real percentile available - see build_player_snapshot/
    _entry) are skipped outright, same as they'd be uninterpretable as a
    bar length; renders nothing at all if that leaves zero rows, same
    "nothing to show" convention as every other render_* here/in
    ui.components.
    """
    headline_labels = BAR_CHART_HEADLINE_STATS.get(pos)
    if headline_labels:
        by_label = {e['label']: e for e in snapshot}
        rows = [by_label[l] for l in headline_labels if l in by_label and by_label[l]['pct'] is not None]
    else:
        rows = [e for e in snapshot if e['pct'] is not None]

    if not rows:
        return

    rows = list(reversed(rows))  # top stat drawn at the top of the chart
    n = len(rows)

    # SCALE_MAX (122, not 100) mirrors the old matplotlib xlim(0, 122) -
    # bars are measured on a 0-122 "scale" even though real values only
    # reach 100, so a bar at the 100th percentile still has clear room to
    # its right for the value label instead of the label doubling back onto
    # the bar itself.
    W, LABEL_END_X, PLOT_X0, PLOT_W, SCALE_MAX = 1000, 178, 195, 750, 122.0
    ROW_H, BAR_H = 34, 21
    TOP_PAD, BOTTOM_PAD = 6, 28
    H = TOP_PAD + n * ROW_H + BOTTOM_PAD

    def bar_w(pct):
        return max(2.0, (pct / SCALE_MAX) * PLOT_W)

    grid_parts = []
    for tick in (0, 25, 50, 75, 100):
        tx = PLOT_X0 + (tick / SCALE_MAX) * PLOT_W
        grid_parts.append(f'<line x1="{tx:.1f}" y1="{TOP_PAD}" x2="{tx:.1f}" y2="{H - BOTTOM_PAD + 4:.1f}" class="pbar-grid"/>')
        grid_parts.append(f'<text x="{tx:.1f}" y="{H - BOTTOM_PAD + 19:.1f}" text-anchor="middle" class="pbar-tick">{tick}</text>')

    bar_parts = []
    for i, r in enumerate(rows):
        y0 = TOP_PAD + i * ROW_H
        y_mid = y0 + ROW_H / 2
        bar_y = y0 + (ROW_H - BAR_H) / 2
        pct = max(0.0, min(100.0, float(r['pct'])))
        w = bar_w(pct)
        label, value, color = r['label'], r['value_str'], r['color']
        bar_parts.append(
            f'<g class="pbar-row">'
            f'<title>{label}: {value} — {pct:.0f}th percentile</title>'
            f'<text x="{LABEL_END_X}" y="{y_mid:.1f}" text-anchor="end" dominant-baseline="central" class="pbar-label">{label}</text>'
            f'<rect x="{PLOT_X0}" y="{bar_y:.1f}" width="{PLOT_W}" height="{BAR_H}" rx="4" class="pbar-track"/>'
            f'<rect x="{PLOT_X0}" y="{bar_y:.1f}" width="{w:.1f}" height="{BAR_H}" rx="4" class="pbar-fill" fill="{color}"/>'
            f'<text x="{PLOT_X0 + w + 10:.1f}" y="{y_mid:.1f}" dominant-baseline="central" class="pbar-value">{value}</text>'
            f'</g>'
        )

    svg = (
        f'<svg viewBox="0 0 {W} {H}" class="pbar-svg" role="img" aria-label="{pos} percentile rankings">'
        f'{"".join(grid_parts)}{"".join(bar_parts)}</svg>'
    )
    st.markdown(
        f"<div class='fpts-strip'><div class='fs-label'>{player_name} — {pos} Percentile Rankings</div>{svg}</div>",
        unsafe_allow_html=True,
    )


def render_percentile_bars_figure_compare(side_1, side_2):
    """
    Overlaid version of render_percentile_bars_figure for Player Compare -
    used for position groups with no RADAR_GRID_GROUPS entry
    (OL/DEFENSIVE_POSITIONS/K/[P,LS]), same "stays the primary, un-demoted
    view" rule as the single-player chart. Each stat gets a PAIR of bars
    (one per player, each in that player's own team color - side['color'],
    see ui.tabs.player_compare._load_player_side) on the same row instead
    of two separate charts to eyeball side by side.

    Only keeps stats BOTH players have a real percentile for - a bar with
    a value on just one side isn't a comparison.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pos = side_1['pos']
    by_label_1 = {e['label']: e for e in side_1['snapshot']}
    by_label_2 = {e['label']: e for e in side_2['snapshot']}
    headline_labels = BAR_CHART_HEADLINE_STATS.get(pos)
    if headline_labels:
        labels = [l for l in headline_labels if l in by_label_1 and by_label_1[l]['pct'] is not None]
    else:
        labels = [e['label'] for e in side_1['snapshot'] if e['pct'] is not None]
    labels = [l for l in labels if l in by_label_2 and by_label_2[l]['pct'] is not None]
    if not labels:
        return None

    labels = list(reversed(labels))  # top stat drawn at the top of the chart
    pcts_1 = [max(0.0, min(100.0, float(by_label_1[l]['pct']))) for l in labels]
    pcts_2 = [max(0.0, min(100.0, float(by_label_2[l]['pct']))) for l in labels]

    fig_height = max(2.0, 0.55 * len(labels) + 0.6)
    fig, ax = plt.subplots(figsize=(6.5, fig_height), dpi=220)
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    y_pos = list(range(len(labels)))
    bar_h = 0.36
    ax.barh([y + bar_h / 2 for y in y_pos], pcts_1, color=side_1['color'], height=bar_h, zorder=3, label=side_1['name'])
    ax.barh([y - bar_h / 2 for y in y_pos], pcts_2, color=side_2['color'], height=bar_h, zorder=3, label=side_2['name'])
    ax.barh(y_pos, [100] * len(labels), color='#1a2447', height=bar_h * 2 + 0.06, zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color='#e5e8ff', size=9)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['0', '25', '50', '75', '100'], color='#7a80a8', size=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.legend(loc='lower right', frameon=False, fontsize=8, labelcolor='#e5e8ff')
    ax.set_title(f"{side_1['name']} vs {side_2['name']} — {pos} Percentile Rankings", color='#ffffff', size=11, pad=10)

    fig.tight_layout(pad=1.4)
    return fig


# Corner callouts for _draw_quadrant_panel, in (x, y, text, ha, va) axes-
# fraction coordinates. Named for what each quadrant means on a VORP
# draft board specifically (not generic "high/low") since this chart's
# whole point is surfacing which kind of player a dot represents at a
# glance - "REPLACEMENT LEVEL" deliberately echoes the term this app's own
# VORP column already uses, not a fresh label for the same idea.
_QUADRANT_LABELS = [
    (0.03, 0.97, 'EFFICIENT, UNDERUSED', 'left', 'top'),
    (0.97, 0.97, 'ELITE & FEATURED', 'right', 'top'),
    (0.03, 0.03, 'REPLACEMENT LEVEL', 'left', 'bottom'),
    (0.97, 0.03, 'VOLUME-DEPENDENT', 'right', 'bottom'),
]


def _draw_quadrant_panel(ax, pos_df, pos):
    """
    One position's Efficiency-vs-Volume scatter with quadrant lines drawn
    at that position's own median (not zero, and not the league-wide
    median across positions - EPA/dropback and EPA/target aren't on the
    same scale, so each panel's medians only ever come from its own
    column). Point color reuses get_pff_color against a percentile ranked
    WITHIN this panel's own population, so a single-position slice always
    spans the full red-blue range instead of clustering wherever that
    position's raw EPA happens to sit relative to the app-wide tables.
    """
    x = pos_df['Volume'].astype(float)
    y = pos_df['Efficiency'].astype(float)
    x_med, y_med = x.median(), y.median()
    vol_label = pos_df['Vol Label'].iloc[0] if 'Vol Label' in pos_df.columns else 'Volume'

    eff_rank = y.rank(pct=True) * 100
    colors = [_mpl_color(get_pff_color(p)) for p in eff_rank]

    ax.set_facecolor('none')
    ax.scatter(x, y, c=colors, s=44, edgecolors='#0a0f24', linewidths=0.6, zorder=3)
    ax.axvline(x_med, color='#3a4170', linewidth=1, linestyle='--', zorder=1)
    ax.axhline(y_med, color='#3a4170', linewidth=1, linestyle='--', zorder=1)

    # Only the elite-and-featured quadrant gets player-name callouts - the
    # single most useful quadrant to name at a glance, and naming all four
    # at league-wide sample sizes would be an unreadable label soup.
    top_right = pos_df[(x >= x_med) & (y >= y_med)]
    for _, r in top_right.nlargest(6, 'Efficiency').iterrows():
        ax.annotate(
            r['Player'], (r['Volume'], r['Efficiency']),
            xytext=(4, 3), textcoords='offset points', color='#e5e8ff', size=6.5, zorder=4,
        )

    for lx, ly, text, ha, va in _QUADRANT_LABELS:
        ax.text(lx, ly, text, transform=ax.transAxes, color='#454c80', size=6.5,
                 ha=ha, va=va, fontweight='bold', zorder=0)

    ax.set_xlabel(vol_label, color='#9aa0c8', size=8.5, fontfamily='monospace')
    ax.set_ylabel('EPA Efficiency', color='#9aa0c8', size=8.5, fontfamily='monospace')
    ax.tick_params(colors='#7a80a8', labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_color('#2c3260')
    ax.grid(color='#1c2350', linewidth=0.5, zorder=0)
    ax.set_title(f"{pos} ({len(pos_df)} qualified)", color='#ffffff', size=10, fontweight='bold', pad=8)


def render_efficiency_volume_quadrants(ev_df):
    """
    2x2 small-multiples grid (QB/RB/WR/TE), one Efficiency-vs-Volume
    quadrant scatter per position - NFL Savant's "Quadrant Chart"
    convention (efficiency vs. volume, split at the median) applied to
    data.transforms.build_efficiency_volume_data's output. Each position
    gets its own panel rather than one shared scatter since EPA/dropback,
    EPA/rush, and EPA/target sit on different natural scales and "Volume"
    is a different underlying stat per position - overlaying them would
    either compress three of the four positions into a sliver of the
    chart or compare unlike things on one axis.

    squeeze=False guarantees a 2D axes array regardless of how many
    positions actually qualified this season, so the flatten()+zip below
    doesn't need a separate 1-panel special case. Returns None if no
    position has enough qualified players to plot at all.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if ev_df is None or ev_df.empty:
        return None

    MIN_QUALIFIED = 8
    panels = []
    for pos in ['QB', 'RB', 'WR', 'TE']:
        pos_df = ev_df[ev_df['Pos'] == pos]
        if len(pos_df) >= MIN_QUALIFIED:
            panels.append((pos, pos_df))
    if not panels:
        return None

    ncols = 2 if len(panels) > 1 else 1
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 5.0 * nrows), dpi=220, squeeze=False)
    fig.patch.set_alpha(0)
    axes_flat = axes.flatten()
    for ax, (pos, pos_df) in zip(axes_flat, panels):
        _draw_quadrant_panel(ax, pos_df, pos)
    for ax in axes_flat[len(panels):]:
        ax.axis('off')

    fig.tight_layout(pad=1.6)
    return fig
