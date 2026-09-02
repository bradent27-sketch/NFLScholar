"""
Defensive Coverage Correlator: cross a WR/TE's own man-vs-zone performance
split against an opposing defense's man/zone coverage mix, to flag matchups
that favor (or work against) that player's actual scheme performance.

Data sources (see PLAN.md section 0.2 for why - Cover-1/2/3/4/6 shell data
isn't published anywhere free/local this app can reach yet):
  - pff_imports/receiving_scheme_2025.csv (player-level man/zone YPRR,
    target share, route grade - via data.loaders.load_all_pff_data()['rec_scheme'])
  - pff_imports/defense_coverage_scheme_2025.csv (player-level defensive
    coverage snaps, aggregated here to team-level man/zone rate)
  - build_positional_coverage_allowed() below for the defense-side YPT
    allowed by receiver type and alignment - computed from nflverse plus the
    PFF alignment archive, replacing two hand-exported public-page scrapes
    (Sharp Football positional coverage, SumerSports tendency) that were
    removed 2026-09-02: both were frozen at 2025, had no feed behind them,
    and everything they carried was derivable from sources already loaded.
"""
import pandas as pd
import streamlit as st

from data.utils import calculate_percentile
from data.transforms import build_alignment_multiplier


def get_team_man_zone_rate(def_coverage_scheme_df, team_name):
    """
    defense_coverage_scheme_2025.csv is player-level (one row per defender).
    Aggregating man_snap_counts_coverage / zone_snap_counts_coverage across
    every defender on a team gives a team-level man% / zone% rate - the
    same idea as the team-level Sharp Football coverage export the code
    already had a (currently-missing) local-file slot for, built instead
    from real local PFF data that's actually on disk.
    """
    if def_coverage_scheme_df.empty or 'team_name' not in def_coverage_scheme_df.columns:
        return None
    team_rows = def_coverage_scheme_df[def_coverage_scheme_df['team_name'] == team_name]
    if team_rows.empty:
        return None
    man_snaps = pd.to_numeric(team_rows.get('man_snap_counts_coverage'), errors='coerce').fillna(0).sum()
    zone_snaps = pd.to_numeric(team_rows.get('zone_snap_counts_coverage'), errors='coerce').fillna(0).sum()
    total = man_snaps + zone_snaps
    if total <= 0:
        return None
    return {'man_rate': round(man_snaps / total * 100, 1), 'zone_rate': round(zone_snaps / total * 100, 1)}


def build_team_man_zone_rates(def_coverage_scheme_df):
    """
    Team-level man/zone coverage rate for ALL teams at once - same math as
    get_team_man_zone_rate above, aggregated in one groupby instead of
    looping per-team. Lets the Defensive Yield merged matchup table show
    Man%/Zone% for any year PFF data exists (2019+, per pff_imports/'s
    per-year folders), not just 2025 - the only year Sharp Football
    Analysis's free coverage-tendency page (the other source for this same
    metric) happens to have, since that site has no historical archive
    (confirmed - no season selector or year parameter). PFF's export
    doesn't carry a middle-of-field open/closed (Cover-1 vs. two-high shell)
    split at all, so that specific pair of columns is still Sharp-Football-
    only / 2025-only regardless - a genuine gap, not something this
    substitutes for.

    Returns a DataFrame with columns ['team', 'man_rate', 'zone_rate'] -
    'team' here is PFF's own team CODE (e.g. "ARZ"), matching
    defense_coverage_scheme's 'team_name' column despite that column's
    name (confirmed: it holds codes, not real names) - caller converts to
    the standard nflverse abbreviation via config.pff_team_to_abbr.
    """
    if def_coverage_scheme_df.empty or 'team_name' not in def_coverage_scheme_df.columns:
        return pd.DataFrame()
    df = def_coverage_scheme_df.copy()
    df['man_snap_counts_coverage'] = pd.to_numeric(df.get('man_snap_counts_coverage'), errors='coerce').fillna(0)
    df['zone_snap_counts_coverage'] = pd.to_numeric(df.get('zone_snap_counts_coverage'), errors='coerce').fillna(0)
    agg = df.groupby('team_name', observed=True).agg(
        man_snaps=('man_snap_counts_coverage', 'sum'),
        zone_snaps=('zone_snap_counts_coverage', 'sum'),
    ).reset_index()
    agg['total'] = agg['man_snaps'] + agg['zone_snaps']
    agg = agg[agg['total'] > 0].copy()
    if agg.empty:
        return pd.DataFrame()
    agg['man_rate'] = (agg['man_snaps'] / agg['total'] * 100).round(1)
    agg['zone_rate'] = (agg['zone_snaps'] / agg['total'] * 100).round(1)
    return agg.rename(columns={'team_name': 'team'})[['team', 'man_rate', 'zone_rate']]


def list_coverage_teams(def_coverage_scheme_df):
    if def_coverage_scheme_df.empty or 'team_name' not in def_coverage_scheme_df.columns:
        return []
    return sorted(def_coverage_scheme_df['team_name'].dropna().unique().tolist())


def list_receivers(rec_scheme_df):
    if rec_scheme_df.empty or 'player' not in rec_scheme_df.columns:
        return []
    pool = rec_scheme_df
    if 'position' in rec_scheme_df.columns:
        pool = rec_scheme_df[rec_scheme_df['position'].isin(['WR', 'TE'])]
    return sorted(pool['player'].dropna().unique().tolist())


def build_radar_data(rec_scheme_df, player_name, opponent_team_name, def_coverage_scheme_df):
    """
    Returns (axis_labels, man_values, zone_values, matchup_summary) where
    man_values/zone_values are percentile-scaled (0-100, among WR/TE with
    non-zero data) so YPRR/target-share/grade - three very different raw
    scales - can share one radar. matchup_summary blends the player's raw
    man/zone YPRR with the opponent's man/zone rate into one expected
    YPRR-against-this-defense number.
    """
    if rec_scheme_df.empty or 'player' not in rec_scheme_df.columns:
        return [], [], [], None

    pool = rec_scheme_df[rec_scheme_df['position'].isin(['WR', 'TE'])] if 'position' in rec_scheme_df.columns else rec_scheme_df
    p_row = pool[pool['player'].str.lower() == str(player_name).lower()]
    if p_row.empty:
        return [], [], [], None
    row = p_row.iloc[0]

    axis_pairs = [
        ('man_yprr', 'zone_yprr', 'YPRR'),
        ('man_targets_percent', 'zone_targets_percent', 'Target Share'),
        ('man_grades_pass_route', 'zone_grades_pass_route', 'Route Grade'),
        ('man_caught_percent', 'zone_caught_percent', 'Catch %'),
    ]
    labels, man_vals, zone_vals = [], [], []
    for man_col, zone_col, label in axis_pairs:
        if man_col not in pool.columns or zone_col not in pool.columns:
            continue
        man_pct = calculate_percentile(pool, man_col)
        zone_pct = calculate_percentile(pool, zone_col)
        idx = p_row.index[0]
        labels.append(label)
        man_vals.append(round(float(man_pct.get(idx, 0)), 1))
        zone_vals.append(round(float(zone_pct.get(idx, 0)), 1))

    matchup_summary = None
    if opponent_team_name and def_coverage_scheme_df is not None:
        # Built once via build_team_man_zone_rates (all 32 teams) instead of
        # the single-team get_team_man_zone_rate - needed anyway to rank
        # this opponent's own man/zone rate against the rest of the league
        # for the metric boxes' percentile context (explicit request: raw
        # numbers there should carry a league-relative read, not just the
        # chart itself).
        team_rates = build_team_man_zone_rates(def_coverage_scheme_df)
        team_row = team_rates[team_rates['team'] == opponent_team_name] if not team_rates.empty else pd.DataFrame()
        if not team_row.empty and 'man_yprr' in row and 'zone_yprr' in row:
            man_rate = float(team_row.iloc[0]['man_rate'])
            zone_rate = float(team_row.iloc[0]['zone_rate'])
            man_yprr = float(row.get('man_yprr', 0) or 0)
            zone_yprr = float(row.get('zone_yprr', 0) or 0)
            blended = (man_yprr * man_rate + zone_yprr * zone_rate) / 100

            p_idx = p_row.index[0]
            team_idx = team_row.index[0]
            man_yprr_pct = calculate_percentile(pool, 'man_yprr').get(p_idx)
            zone_yprr_pct = calculate_percentile(pool, 'zone_yprr').get(p_idx)
            opp_man_pct = calculate_percentile(team_rates, 'man_rate').get(team_idx)
            opp_zone_pct = calculate_percentile(team_rates, 'zone_rate').get(team_idx)
            # Blended Exp. YPRR isn't a single source column - it's this
            # player's own man/zone YPRR weighted by THIS opponent's own
            # man/zone rate, so its percentile has to be computed the same
            # way for every other qualified WR/TE against that SAME
            # opponent, then rank the selected player within that - an
            # apples-to-apples "how good is this specific matchup for him
            # vs. everyone else facing this defense" read, not a rehash of
            # his overall YPRR percentile.
            pool_blend = (
                pd.to_numeric(pool['man_yprr'], errors='coerce').fillna(0) * man_rate
                + pd.to_numeric(pool['zone_yprr'], errors='coerce').fillna(0) * zone_rate
            ) / 100
            blended_pct = calculate_percentile(pool_blend.to_frame('blend'), 'blend').get(p_idx)

            matchup_summary = {
                'opponent': opponent_team_name,
                'opp_man_rate': man_rate,
                'opp_zone_rate': zone_rate,
                'opp_man_rate_pct': round(float(opp_man_pct), 1) if pd.notna(opp_man_pct) else None,
                'opp_zone_rate_pct': round(float(opp_zone_pct), 1) if pd.notna(opp_zone_pct) else None,
                'player_man_yprr': round(man_yprr, 2),
                'player_zone_yprr': round(zone_yprr, 2),
                'player_man_yprr_pct': round(float(man_yprr_pct), 1) if pd.notna(man_yprr_pct) else None,
                'player_zone_yprr_pct': round(float(zone_yprr_pct), 1) if pd.notna(zone_yprr_pct) else None,
                'blended_expected_yprr': round(blended, 2),
                'blended_expected_yprr_pct': round(float(blended_pct), 1) if pd.notna(blended_pct) else None,
            }

    return labels, man_vals, zone_vals, matchup_summary


def build_alignment_matchup_data(rec_df, route_concept_df, positional_coverage_df, player_name, full_team_name):
    """
    WR/TE alignment profile (Slot %, Slot YPRR, Wide YPRR) plus a matchup-
    aware "Blended Exp. YPRR (Slot Align)" number - complements
    build_radar_data's own Man/Zone axis (coverage SHELL) with the alignment
    axis (slot-vs-outside) instead, for the Coverage Matchup Radar's
    PLAYER/BLENDED tile columns.

    Wide YPRR isn't its own PFF export - receiving_concept only ever ships a
    SLOT split (confirmed by direct inspection of every pff_imports/
    receiving_concept file, every year - no "Wide"/"Outside" concept file
    exists). Derived here instead as the residual of receiving_summary's
    season routes/yards minus receiving_concept's slot component. Reads as
    genuinely "wide" for WRs (inline_rate is near-zero there); for TEs it
    also nets out inline routes, so it's closer to "non-slot" than strictly
    "wide" - the same caveat PFF's own wide_rate/inline_rate split already
    carries for that position.

    The blended number reuses build_alignment_multiplier (the same function
    already driving Player Search's Next Game Projection alignment
    adjustment) rather than inventing a second formula for the same idea:
    this player's overall season YPRR, scaled up or down by how soft or
    tough THIS specific opponent plays against however this player is
    actually deployed (their own slot/wide snap split) - same "matchup-
    aware, not just raw" spirit as build_radar_data's own Man/Zone blend.

    Returns {} if receiving_summary has no row for this player. Every other
    key is filled in as far as the data allows - Slot YPRR/Wide YPRR/the
    blended number are silently omitted if their own source data is
    missing, same partial-fill convention as build_radar_data's
    matchup_summary.
    """
    if rec_df.empty or 'player' not in rec_df.columns:
        return {}
    rec_pool = rec_df[rec_df['position'].isin(['WR', 'TE'])] if 'position' in rec_df.columns else rec_df
    p_rec = rec_pool[rec_pool['player'].str.lower() == str(player_name).lower()]
    if p_rec.empty:
        return {}
    p_idx = p_rec.index[0]
    rec_row = p_rec.iloc[0]
    slot_rate = float(rec_row.get('slot_rate', 0) or 0)
    wide_rate = float(rec_row.get('wide_rate', 0) or 0)
    overall_yprr = float(rec_row.get('yprr', 0) or 0)

    result = {
        'slot_rate': round(slot_rate, 1),
        'slot_rate_pct': round(float(calculate_percentile(rec_pool, 'slot_rate').get(p_idx, 0)), 1),
    }

    rc_pool = pd.DataFrame()
    if not route_concept_df.empty and 'player' in route_concept_df.columns and 'slot_yprr' in route_concept_df.columns:
        rc_pool = route_concept_df[route_concept_df['position'].isin(['WR', 'TE'])] if 'position' in route_concept_df.columns else route_concept_df
        p_rc = rc_pool[rc_pool['player'].str.lower() == str(player_name).lower()]
        if not p_rc.empty:
            rc_idx = p_rc.index[0]
            slot_yprr = float(p_rc.iloc[0].get('slot_yprr', 0) or 0)
            result['slot_yprr'] = round(slot_yprr, 2)
            result['slot_yprr_pct'] = round(float(calculate_percentile(rc_pool, 'slot_yprr').get(rc_idx, 0)), 1)

    wide_pool = pd.DataFrame()
    if not rc_pool.empty and {'routes', 'yards'}.issubset(rec_pool.columns) and {'slot_routes', 'slot_yards'}.issubset(rc_pool.columns):
        wide_pool = rec_pool[['player', 'routes', 'yards']].merge(
            rc_pool[['player', 'slot_routes', 'slot_yards']], on='player', how='inner'
        )
        wide_pool['wide_routes'] = pd.to_numeric(wide_pool['routes'], errors='coerce') - pd.to_numeric(wide_pool['slot_routes'], errors='coerce')
        wide_pool['wide_yards'] = pd.to_numeric(wide_pool['yards'], errors='coerce') - pd.to_numeric(wide_pool['slot_yards'], errors='coerce')
        wide_pool = wide_pool[wide_pool['wide_routes'] > 0].copy()
        wide_pool['wide_yprr'] = wide_pool['wide_yards'] / wide_pool['wide_routes']
        m_row = wide_pool[wide_pool['player'].str.lower() == str(player_name).lower()]
        if not m_row.empty:
            m_idx = m_row.index[0]
            result['wide_yprr'] = round(float(m_row.iloc[0]['wide_yprr']), 2)
            result['wide_yprr_pct'] = round(float(calculate_percentile(wide_pool, 'wide_yprr').get(m_idx, 0)), 1)

    if full_team_name and positional_coverage_df is not None and not positional_coverage_df.empty:
        nickname = team_nickname(full_team_name)
        alignment_mult = build_alignment_multiplier(positional_coverage_df, nickname, slot_rate, wide_rate)
        result['blended_alignment_yprr'] = round(overall_yprr * alignment_mult, 2)

        # Percentile against every other qualified WR/TE facing this SAME
        # opponent (each run through the same opponent-specific multiplier,
        # using THEIR OWN slot/wide split) - identical "how good is THIS
        # matchup for him specifically, vs. everyone else facing this
        # defense" convention as build_radar_data's own
        # blended_expected_yprr_pct.
        pool = rec_pool[['slot_rate', 'wide_rate', 'yprr']].copy()
        pool['alignment_mult'] = rec_pool.apply(
            lambda r: build_alignment_multiplier(positional_coverage_df, nickname, r.get('slot_rate', 0), r.get('wide_rate', 0)),
            axis=1,
        )
        pool['blended_alignment_yprr'] = pd.to_numeric(pool['yprr'], errors='coerce') * pool['alignment_mult']
        blended_pct = calculate_percentile(pool, 'blended_alignment_yprr').get(p_idx)
        if pd.notna(blended_pct):
            result['blended_alignment_yprr_pct'] = round(float(blended_pct), 1)

    return result


def team_nickname(full_team_name):
    """'Arizona Cardinals' -> 'Cardinals' - matches Sharp Football Analysis's
    bare-nickname team column convention."""
    if not full_team_name:
        return None
    return full_team_name.split()[-1]


def build_positional_coverage_allowed(year, as_of_week=None):
    """Yards-per-target ALLOWED by receiver type and alignment, all 32 teams,
    computed from data this app already loads.

    REPLACES external_data/sharp_positional_coverage_{year}.csv (removed
    2026-09-02). That file was a hand-exported scrape of a public page,
    hardcoded to one season, silently frozen, and with no feed behind it -
    every season the app showed was really 2025. Everything it carried is
    derivable from sources that update themselves:

      ypt_allowed_wr / _te / _rb
          data.transforms.build_stat_allowed_matrix already returns targets
          and receiving_yards allowed per defense per position; YPT is the
          quotient. Works for any season with weekly stats.

      ypt_allowed_outside / _slot
          data.pff_alignment.load_weekly_alignment_defense_profiles carries
          observed_total for {targets, yards} x {slot, wide} per defense, so
          the same quotient gives a true per-alignment YPT. This is a
          STRICTLY BETTER source than the file it replaces - opponent-
          adjusted, shrinkage-weighted, week-sliced and time-valid - and it
          already ships inside the weekly model.

    Returns a frame with the same shape the old export had (`team` as a bare
    nickname plus the five ypt_allowed_* columns), so the radar and the
    Matchup Analyzer's positional-vulnerability panel consume it unchanged.

    The alignment half needs the PFF weekly archive, which exists for 2024
    onward. For an earlier season the two alignment columns are simply
    absent and the radar drops those axes - an honest gap, where the old
    file silently showed 2025 numbers under a 2019 heading.
    """
    from data.loaders import load_schedule
    from data.transforms import load_and_merge_data, build_stat_allowed_matrix
    from data.pff_alignment import load_weekly_alignment_defense_profiles
    from config import TEAM_CONFIG

    try:
        stats_df, _team_col, _name_col, _rookies = load_and_merge_data(int(year), 'Full PPR')
    except Exception:
        return pd.DataFrame()
    if stats_df is None or stats_df.empty:
        return pd.DataFrame()

    frames = {}
    for pos, col in (('WR', 'ypt_allowed_wr'), ('TE', 'ypt_allowed_te'), ('RB', 'ypt_allowed_rb')):
        m = build_stat_allowed_matrix(stats_df, position_filter=[pos])
        if m.empty or 'targets' not in m.columns or 'receiving_yards' not in m.columns:
            continue
        tgt = pd.to_numeric(m['targets'], errors='coerce')
        yds = pd.to_numeric(m['receiving_yards'], errors='coerce')
        frames[col] = pd.Series(
            (yds / tgt.where(tgt > 0)).to_numpy(),
            index=m['Team'].astype(str) if 'Team' in m.columns else m.index)
    if not frames:
        return pd.DataFrame()

    out = pd.DataFrame(frames)
    out.index.name = 'abbr'
    out = out.reset_index()

    # Alignment half - absent rather than guessed when there is no archive.
    try:
        prof = load_weekly_alignment_defense_profiles(
            int(year), int(as_of_week) if as_of_week else 19, load_schedule(int(year))).profiles
    except Exception:
        prof = pd.DataFrame()
    if prof is not None and not prof.empty:
        wr = prof[(prof['position'] == 'WR')
                  & (prof['alignment'].isin(['slot', 'wide']))
                  & (prof['stat'].isin(['targets', 'yards']))]
        if not wr.empty:
            piv = wr.pivot_table(index=['defense_team', 'alignment'], columns='stat',
                                 values='observed_total', aggfunc='first').reset_index()
            if {'targets', 'yards'}.issubset(piv.columns):
                piv['ypt'] = piv['yards'] / piv['targets'].where(piv['targets'] > 0)
                wide = piv[piv['alignment'] == 'wide'].set_index('defense_team')['ypt']
                slot = piv[piv['alignment'] == 'slot'].set_index('defense_team')['ypt']
                # to_numeric, not a bare .map: the pivot can hand back an
                # OBJECT-dtype column (arrow-backed source frame), and an
                # object column silently breaks .mean() downstream in
                # build_alignment_multiplier - which is a crash, not a
                # degradation, and one no unit test covers because it only
                # fires when the tab renders.
                out['ypt_allowed_outside'] = pd.to_numeric(out['abbr'].map(wide), errors='coerce')
                out['ypt_allowed_slot'] = pd.to_numeric(out['abbr'].map(slot), errors='coerce')

    # The consumers key on a bare nickname ("Seahawks"), which is what the
    # replaced export used - keep that contract rather than touching them.
    out['team'] = out['abbr'].map(
        lambda a: str(TEAM_CONFIG.get(str(a), {}).get('name', '')).split()[-1] or str(a))
    cols = ['team'] + [c for c in out.columns if c.startswith('ypt_allowed_')]
    return out[cols]


def build_defense_radar_data(positional_coverage_df, full_team_name):
    """
    Right-side "opposing defense" radar: percentile-scaled (0-100, among all
    32 teams) yards-per-target ALLOWED to each receiver type/alignment, from
    Sharp Football Analysis's free "NFL Coverage Stats by Position" page
    (external_data/sharp_positional_coverage_2025.csv - a real gap-filling
    source, not local PFF data). Percentile direction is inverted
    (ascending=False) so a LOW yards-allowed number - good coverage - still
    reads as "further out on the chart", consistent with the player-side
    radar where further out also means better performance.
    """
    if positional_coverage_df.empty or 'team' not in positional_coverage_df.columns:
        return [], [], {}
    nickname = team_nickname(full_team_name)
    row = positional_coverage_df[positional_coverage_df['team'] == nickname]
    if row.empty:
        return [], [], {}
    idx = row.index[0]

    axis_cols = [
        ('ypt_allowed_wr', 'YPT vs WR'),
        ('ypt_allowed_te', 'YPT vs TE'),
        ('ypt_allowed_rb', 'YPT vs RB'),
        ('ypt_allowed_outside', 'YPT Outside'),
        ('ypt_allowed_slot', 'YPT Slot'),
    ]
    labels, vals, raw = [], [], {}
    for col, label in axis_cols:
        if col not in positional_coverage_df.columns:
            continue
        pct = calculate_percentile(positional_coverage_df, col, ascending=False)
        labels.append(label)
        vals.append(round(float(pct.get(idx, 0)), 1))
        raw[label] = float(row.iloc[0][col])

    return labels, vals, raw


@st.cache_data(show_spinner=False)
def render_split_radar_figure(player_labels, player_man_vals, player_zone_vals, player_name,
                               defense_labels, defense_vals, defense_team_label):
    """
    Two polar charts side by side in one figure - player's own man/zone
    profile (left) and the opposing defense's coverage-quality-by-target-type
    profile (right), so both halves of the matchup are visible at once
    without needing to scroll between two separately-rendered charts.
    Higher DPI + tighter layout than the original single chart (which read
    as "large and slightly low quality" at default matplotlib DPI).

    @st.cache_data: pure function of these 7 plain values, DPI-260 two-panel
    draw - Defensive Yield's other widgets (O-Line/pressure year picker,
    strength-of-schedule controls) live in the same render() and would
    otherwise force a full redraw of this chart on every rerun even when
    neither sel_player nor sel_opponent changed.
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not player_labels and not defense_labels:
        return None

    # Sizing/DPI/fonts raised together per explicit feedback that the
    # charts read small and fuzzy: st.pyplot scales the image to its
    # container width, so on-screen legibility tracks the font-to-figure
    # ratio and the render DPI, not the absolute point sizes. DPI raised
    # again (200->260) per a later round of the same feedback - this chart
    # sits in a wide, full-bleed layout with no max-width cap, and on a
    # large/high-DPI monitor the rendered CSS width can exceed what 200 DPI
    # comfortably supports before the upscale starts looking soft.
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.9), subplot_kw=dict(polar=True), dpi=260)
    fig.patch.set_alpha(0)

    def _draw(ax, labels, series, category_label):
        ax.set_facecolor('none')
        if not labels:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', color='#7a80a8', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        n = len(labels)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]
        for vals, color, name in series:
            v = vals + vals[:1]
            ax.plot(angles, v, color=color, linewidth=2.2, label=name)
            ax.fill(angles, v, color=color, alpha=0.18)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color='#e5e8ff', size=11)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(['25', '50', '75', '100'], color='#7a80a8', size=8)
        ax.spines['polar'].set_color('#2c3260')
        ax.grid(color='#2c3260')
        ax.tick_params(axis='x', pad=12)
        # A small muted CATEGORY label ("PLAYER PROFILE"/"OPPONENT
        # DEFENSE"), not the player/team's own name - render_matchup_title
        # already names both immediately above this figure, so repeating
        # the same name here (the original version titled each panel with
        # it) was pure redundancy sitting uncomfortably close to the
        # circle. This still orients a first-time viewer to which side is
        # which without repeating text or crowding the plot - small size +
        # generous pad keeps it clearly separated from the topmost spoke.
        ax.set_title(category_label, color='#7a80a8', size=9.5, fontweight='700', pad=32,
                     fontfamily='sans-serif')
        # Legend BELOW the chart, centered - the old upper-right placement
        # sat directly on top of the top-right spoke label at the new,
        # larger label size.
        legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2,
                           facecolor='#131b38', edgecolor='#2c3260', fontsize=9)
        for text in legend.get_texts():
            text.set_color('#e5e8ff')

    _draw(axes[0], player_labels,
          [(player_man_vals, '#00fff9', 'vs. Man'), (player_zone_vals, '#ffae58', 'vs. Zone')],
          'PLAYER PROFILE')
    _draw(axes[1], defense_labels,
          [(defense_vals, '#1ed760', 'Coverage Quality')],
          'OPPONENT DEFENSE')

    # Explicit margins instead of tight_layout - tight_layout doesn't
    # account for polar tick labels or the below-axes legends, which left
    # the left-edge label clipped and the legend sitting on the bottom
    # spoke label.
    fig.subplots_adjust(left=0.145, right=0.855, top=0.87, bottom=0.17, wspace=0.6)
    return fig
