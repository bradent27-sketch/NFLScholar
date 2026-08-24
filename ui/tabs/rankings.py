"""
Weekly Rankings tab: this app's own week-by-week fantasy-points projection
model (data.weekly_projections), FantasyPros' live weekly projection pulled
straight from their API, a live player-prop-derived market projection
(data.odds_weekly), this app's own L5 recent-form ranking, and (still,
unchanged) an uploaded weekly-rankings export - all lined up on one table
by player.

A weekly ranking answers "who's producing right now" / "who should I start
this week" - a different question than the season-long draft-value ranking
Draft HQ compares against FantasyPros' DRAFT rankings. Keeping the two
comparisons on separate tabs (rather than one tab trying to serve both)
means each one's internal baseline actually matches what it's being
compared against.

The model, the live API pull, the live market projection, and the CSV
upload are four INDEPENDENT sources shown side by side, never blended into
one number - same "show market lines next to this board, don't merge them"
convention Draft HQ already uses.
"""
import numpy as np
import pandas as pd
import streamlit as st

from config import AVAILABLE_SEASONS_WITH_UPCOMING
from data.draft_board import DEFAULT_SCORING, tier_by_position
from data.transforms import load_and_merge_data, build_recent_form_rank, build_form_series
from data.rankings import parse_fantasypros_upload, parse_custom_rankings, build_rankings_comparison
from data.utils import calculate_percentile, clean_name_exact, clean_name_for_merge
from data.weekly_projections import build_weekly_projections
from data.odds_weekly import weekly_props, weekly_market_projection
from data.fantasypros_availability import canonical_status
from ui.charts import sparkline_data_uri
from ui.styling import (style_plain_dataframe, df_auto_height, build_column_help_config,
                        get_diverging_color, get_multiplier_color)
from ui.components import (position_group_buttons, apply_position_group, skeleton_loader,
                           import_hint)

_MODEL_STAT_COLS = ['passing_yards', 'passing_tds', 'rushing_attempts', 'rushing_yards',
                    'rushing_tds', 'targets', 'receptions', 'receiving_yards', 'receiving_tds']

# The stat line the model projects, shown alongside Model Proj Pts instead
# of just the point total - explicit request, so the number can be read
# rather than taken on faith. Ordered pass -> rush -> catch, the same
# volume-first sequencing the rest of the app's stat displays use. A
# position that doesn't carry a given raw stat (a WR has no passing_yards)
# just shows blank for it - the concat in build_weekly_projections already
# unions every position's columns, so these exist on the merged frame
# whenever ANY position projected that stat this week.
_STAT_DISPLAY_COLS = [
    ('passing_yards', 'Pass Yds'), ('passing_tds', 'Pass TDs'),
    ('rushing_attempts', 'Rush Att'), ('rushing_yards', 'Rush Yds'), ('rushing_tds', 'Rush TDs'),
    ('targets', 'Tgt'), ('receptions', 'Rec'), ('receiving_yards', 'Rec Yds'), ('receiving_tds', 'Rec TDs'),
]

# A fixed number of games, not a user-adjustable window - explicit request.
# "Recent" reads consistently across positions and weeks only if everyone's
# measured over the same trailing sample.
RECENT_FORM_GAMES = 5

# Rendering every skill-position player with a projection (300+ rows, each
# with several percentile-heatmapped columns) was the actual source of a
# real reported slowdown - the Styler recomputes its per-column heatmap over
# every visible row on every rerun. Defaulting to the top 50 (already-sorted
# order, so "top 50" means the 50 most relevant to whatever's currently
# filtered) with a dropdown to widen it keeps the common case fast without
# hiding the full pool from anyone who wants it.
_SHOW_N_OPTIONS = [25, 50, 100, 200, "All"]
_PROJECTION_DETAIL_KEY = 'weekly_rank_projection_detail'


def _render_distribution_chart(distribution, position_label):
    """The boom/bust curve: this player's own KDE-smoothed density (filled,
    theme accent color) plus the unadjusted position/tier baseline (dotted
    outline, 'typical distribution by position' per the explicit ask) drawn
    from the SAME family of curve so the two are visually comparable - a
    tight, narrow player fill next to a wide dotted baseline reads as "more
    consistent than typical", not just as two disconnected numbers.

    Both curves are built the same way: sample_from_band's inverse-transform
    sampling off the fitted percentile ladder, then weekly_distribution.
    kde_curve's hand-rolled Gaussian KDE (no scipy in this project) smooths
    the sample into a plottable density. Colors match .streamlit/config.toml's
    fixed dark theme (base="dark", bg #050921, accent #00fff9) - this app has
    no light-mode toggle to design a second palette for.
    """
    import matplotlib.pyplot as plt
    from data.weekly_distribution import sample_from_band, kde_curve

    player_points, position_points = distribution['points'], distribution['position_points']
    player_samples = sample_from_band(player_points, seed=42)
    position_samples = sample_from_band(position_points, seed=43)
    lo = max(0.0, min(player_samples.min(), position_samples.min()) - 1.0)
    hi = max(player_samples.max(), position_samples.max()) + 1.0
    grid = np.linspace(lo, hi, 300)
    player_density = kde_curve(player_samples, grid)
    position_density = kde_curve(position_samples, grid)

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.fill_between(grid, player_density, color="#00fff9", alpha=0.28, lw=0)
    ax.plot(grid, player_density, color="#00fff9", lw=1.8, label="This player")
    ax.plot(grid, position_density, color="#8891b0", lw=1.3, ls="--",
            label=f"Typical {distribution['tier']} {position_label}")
    for pct in (10, 50, 90):
        ax.axvline(player_points[pct], color="#ffffff", lw=1.0,
                   ls=(":" if pct != 50 else "-"), alpha=0.8 if pct == 50 else 0.5)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#8891b0")
    ax.tick_params(axis="x", colors="#c7cbe0")
    ax.set_xlabel("Fantasy points", color="#c7cbe0", fontsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor="#c7cbe0")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Bust (P10)", f"{player_points[10]:.1f}")
    c2.metric("Median (P50)", f"{player_points[50]:.1f}")
    c3.metric("Boom (P90)", f"{player_points[90]:.1f}")
    scale = distribution.get('width_scale', 1.0)
    band_note = (
        "narrower than a typical player at this tier (an established role)" if scale < 0.95 else
        "wider than a typical player at this tier (a less certain role)" if scale > 1.05 else
        "about the same width as a typical player at this tier"
    )
    st.caption(
        f"Built from how this model has actually missed for {distribution['tier']} {position_label}s "
        f"in a real out-of-sample backtest (scripts/fit_weekly_distribution.py), not an arbitrary "
        f"+/-%. This player's own band is {band_note}."
    )


def _close_projection_dialog():
    """Clear the one-shot detail state so a dismissed modal stays closed."""
    st.session_state.pop(_PROJECTION_DETAIL_KEY, None)


_DECOMPOSITION_2DP_STATS = {'passing_tds', 'rushing_tds', 'receiving_tds', 'passing_interceptions'}


def _fmt_stat(stat, value, signed=False):
    """Reception/yard/attempt-type stats display at 1 decimal, TD/INT-type
    stats at 2 - explicit request, since a TD/INT rate lives in a much
    smaller numeric range where the extra decimal carries real signal."""
    if value is None:
        return '—'
    try:
        value = float(value)
    except (TypeError, ValueError):
        return '—'
    if not np.isfinite(value):
        return '—'
    decimals = 2 if stat in _DECOMPOSITION_2DP_STATS else 1
    return f"{value:+.{decimals}f}" if signed else f"{value:.{decimals}f}"


def _render_decomposition_header(detail):
    st.caption(
        f"{detail['position']} · {detail['team']} vs {detail['opponent']} · "
        f"Week {detail['target_week']} using information through Week {detail['as_of_week'] - 1}"
    )
    raw, calibrated = detail['raw_points'], detail['calibrated_points']
    status = canonical_status(detail.get('availability', {}).get('status'))
    if detail.get('calibration', {}).get('enabled'):
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw model", f"{raw:.2f} pts")
        c2.metric("Displayed projection", f"{calibrated:.2f} pts")
        c3.metric("Availability", status.title())
    else:
        c1, c2 = st.columns(2)
        c1.metric("Projected points", f"{calibrated:.2f} pts")
        c2.metric("Availability", status.title())

    matchup = detail.get('defense_matchup')
    if matchup:
        allowed_rank, of = matchup['rank'], matchup['of']
        # data.matchup_signals.defense_stat_rank's own rank is 1 = allows
        # the MOST (softest defense) - confirmed against that function's
        # own test, test_defense_stat_rank_computes_per_game_average_
        # and_rank ("the higher (softer) of the two" -> rank 1). Flipped
        # here into a "toughness rank" (1 = toughest) because that is the
        # number a fantasy decision actually wants, and because reading
        # a raw "#1" next to the word "toughest" would otherwise say the
        # opposite of what the underlying number means.
        toughness_rank = of - allowed_rank + 1
        tier = ("toughest" if toughness_rank <= of / 3
               else "easiest" if toughness_rank > 2 * of / 3 else "middling")
        st.markdown(
            f"**{detail['opponent']} defense vs {detail['position']}: {tier} matchup** — "
            f"#{toughness_rank} of {of} toughest (allows {matchup['value']:.1f} pts/game to the "
            f"position, league avg {matchup['league_avg']:.1f})."
        )
        st.caption(f"Source: {matchup.get('source', 'unknown')}. Full breakdown in the Deep Dive tabs below.")
    else:
        st.caption("Defense matchup rank: not enough prior weeks of data yet to rank this opponent against this position.")

    if detail.get('experimental'):
        st.info("Experimental V2 run — this decomposition is auditable, but its components are not yet default-on.")


def _render_decomposition_primary_table(detail):
    """The at-a-glance table: one row per projected stat, left-to-right
    exactly stat -> projected value -> raw (season-blended) average ->
    weighted (defense-adjusted) average -> defense multiplier -> context
    multiplier -> vacancy delta - explicit request, replacing both the old
    plain stat-line table and the old 'Projection path by stat' expander."""
    stats = detail.get('stats', {})
    if not stats:
        st.caption("No projected stat line for this player.")
        return
    season_year = detail.get('season_year')
    rows = []
    for stat, values in stats.items():
        context_mult = (
            values.get('script_multiplier', 1.0) * values.get('pace_multiplier', 1.0)
            * values.get('availability_multiplier', 1.0) * values.get('environment_multiplier', 1.0)
        )
        cur_w, cur_games = values.get('current_weight'), values.get('current_games') or 0.0
        if cur_w is None:
            season_note = ''
        elif not cur_games:
            season_note = (f"100% {season_year - 1} season" if season_year else "100% prior season")
        else:
            season_note = (f"{cur_w:.0%} {season_year} ({cur_games:.0f} gm) + {1 - cur_w:.0%} prior"
                          if season_year else f"{cur_w:.0%} this season")
        vacancy_raw = float(values.get('vacancy_delta', 0.0) or 0.0)
        rows.append({
            'Stat': stat.replace('_', ' ').title(),
            'Projected value': _fmt_stat(stat, values.get('final_projection', values.get('projection'))),
            'Raw average': _fmt_stat(stat, values.get('blended_rate')),
            'Season mix': season_note,
            'Weighted average': _fmt_stat(stat, values.get('current_rate')),
            'Defense multiplier': round(float(values.get('matchup_multiplier', 1.0)), 3),
            'Context multiplier': round(float(context_mult), 3),
            'Vacancy Δ': _fmt_stat(stat, vacancy_raw, signed=True),
            '_vacancy_raw': vacancy_raw,
        })
    table = pd.DataFrame(rows)
    vacancy_vals = table['_vacancy_raw'].tolist()
    display_cols = ['Stat', 'Projected value', 'Raw average', 'Season mix',
                    'Weighted average', 'Defense multiplier', 'Context multiplier', 'Vacancy Δ']
    display_table = table[display_cols]

    # Local Styler, not a style_plain_dataframe extension - that helper is
    # shared by every other table in the app, and this coloring rule
    # (multiplier columns centered on 1.0) is specific to this one table.
    def _style_column(col):
        if col in ('Defense multiplier', 'Context multiplier'):
            return [f'background-color:{get_multiplier_color(v)}; color:#ffffff; font-weight:bold;'
                   for v in display_table[col]]
        if col == 'Vacancy Δ':
            return [f'background-color:{get_diverging_color(v, 2.0)}; color:#ffffff; font-weight:bold;'
                   for v in vacancy_vals]
        return [''] * len(display_table)

    style_grid = pd.DataFrame({col: _style_column(col) for col in display_cols})
    styler = (display_table.style.apply(lambda _: style_grid, axis=None)
             .format({'Defense multiplier': '{:.3f}×', 'Context multiplier': '{:.3f}×'}))
    st.dataframe(styler, hide_index=True, width="stretch", height=df_auto_height(len(display_table)))
    st.caption("Context multiplier = game script × pace × availability × Vegas-implied game environment.")


def _render_stat_deep_dive(detail, stat, game_log, defense_log, season_year=None):
    values = detail.get('stats', {}).get(stat, {})
    label = stat.replace('_', ' ').title()
    season_label = season_year if season_year is not None else (detail.get('season_year') or 'current')

    player_rows = [g for g in game_log if stat in g]
    if player_rows:
        pgl = pd.DataFrame(player_rows)
        pgl['_week_num'] = pd.to_numeric(pgl.get('week'), errors='coerce')
        pgl = pgl.dropna(subset=['_week_num']).sort_values('_week_num')
        eligible = (pgl['_player_history_eligible'] if '_player_history_eligible' in pgl.columns
                   else pd.Series(True, index=pgl.index))
        reason = (pgl['_player_history_reason'] if '_player_history_reason' in pgl.columns
                 else pd.Series('', index=pgl.index))
        included = [('Yes' if e else f'Excluded — {r}') for e, r in zip(eligible, reason)]
        adj_col = f'_defadj_{stat}'
        pgl_display = pd.DataFrame({
            'Week': pgl['_week_num'].astype(int),
            'Opponent': pgl.get('opponent_team', ''),
            f'Raw {label}': [_fmt_stat(stat, v) for v in pgl[stat]],
            f'Defense-adj {label}': [_fmt_stat(stat, v) for v in pgl.get(adj_col, pd.Series(dtype=float))],
            'Included': included,
        })
        spark_values = pd.to_numeric(pgl[stat], errors='coerce').fillna(0.0).tolist()
        season_avg = float(np.mean(spark_values)) if spark_values else None
        st.markdown(
            f"<img src='{sparkline_data_uri(spark_values, season_avg, width=260, height=60)}'>",
            unsafe_allow_html=True,
        )
        st.dataframe(pgl_display, hide_index=True, width="stretch", height=df_auto_height(len(pgl_display)))
        n_excluded = int((~eligible.astype(bool)).sum())
        if n_excluded:
            st.caption(f"{n_excluded} game(s) excluded from this player's rate evidence (marked above, not hidden).")
    else:
        st.caption(f"No {season_label} season games logged yet for {label.lower()}.")

    st.markdown(f"**{detail['opponent']} defense — what it's allowed to {label.lower()}, by week**")
    defense_rows = [g for g in defense_log if stat in g]
    if defense_rows:
        dgl = pd.DataFrame(defense_rows)
        dgl['_week_num'] = pd.to_numeric(dgl.get('_week'), errors='coerce')
        dgl = dgl.dropna(subset=['_week_num']).sort_values('_week_num')
        baseline_col = f'_baseline_{stat}'
        dgl_display = pd.DataFrame({
            'Week': dgl['_week_num'].astype(int),
            'Offense faced': dgl.get('_offense', ''),
            f'Allowed {label}': [_fmt_stat(stat, v) for v in dgl[stat]],
            "That offense's own average": [_fmt_stat(stat, v) for v in dgl.get(baseline_col, pd.Series(dtype=float))],
            'Recency weight': [f"{v:.2f}" if pd.notna(v) else '—' for v in dgl.get('_weight', pd.Series(dtype=float))],
        })
        st.dataframe(dgl_display, hide_index=True, width="stretch", height=df_auto_height(len(dgl_display)))
        st.caption(
            f"Defense multiplier ({values.get('matchup_multiplier', 1.0):.3f}×): each week above compares what "
            f"{detail['opponent']} allowed to that offense's own average, recency-weighted, then re-centered "
            f"to a league average of 1.0 — this table is that comparison's raw ingredients."
        )
    else:
        st.caption(f"No {label.lower()} matchup evidence available for {detail['opponent']} yet.")

    notes = []
    if values.get('prior_source'):
        role_scale = values.get('role_scale')
        notes.append(f"Prior-season source: {values['prior_source']}"
                    + (f" (role scale {role_scale:.3f})" if role_scale is not None else ""))
    if values.get('efficiency_denominator'):
        notes.append(
            f"Efficiency rebuild: projected {values['efficiency_denominator'].replace('_', ' ')} × "
            f"efficiency rate {values.get('efficiency_rate', 0.0):.3f}")
    if values.get('two_year_td_prior'):
        notes.append("Uses a comparable two-year TD prior.")
    if values.get('qb1_selection_required'):
        notes.append("QB1 selection required — volume held at zero until selected.")
    elif not values.get('qb_projected_starter', True):
        notes.append("Not the expected starter — normal QB volume held at zero.")
    if values.get('rb_segment_status') not in (None, '', 'no_clear_internal_gap'):
        notes.append(f"RB role segment: {values['rb_segment_status']}.")
    if notes:
        st.caption(" · ".join(notes))


def _render_context_deep_dive(detail):
    stats = detail.get('stats', {})
    rows = []
    for stat, values in stats.items():
        rows.append({
            'Stat': stat.replace('_', ' ').title(),
            'Game script': f"{values.get('script_multiplier', 1.0):.3f}×",
            'Pace': f"{values.get('pace_multiplier', 1.0):.3f}×",
            'Availability': f"{values.get('availability_multiplier', 1.0):.3f}×",
            'Environment': f"{values.get('environment_multiplier', 1.0):.3f}×",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=df_auto_height(len(rows)))
    sample = next(iter(stats.values()), {})
    target_margin = sample.get('target_margin')
    opp_pace, league_pace = sample.get('opponent_defensive_pace'), sample.get('league_pace')
    st.caption(
        f"**Game script** ({sample.get('script_status', 'not modeled')}"
        + (f", target margin {target_margin:+.1f}" if target_margin is not None else "")
        + "): this player's own history at similar Vegas-implied point margins. "
        "**Pace**: opponent's plays/game ("
        + (f"{opp_pace:.1f}" if opp_pace is not None else "unavailable") + " vs league "
        + (f"{league_pace:.1f}" if league_pace is not None else "unavailable") + "). "
        "**Availability**: this week's injury/role discount. **Environment**: Vegas-implied team total "
        f"and venue ({sample.get('environment_status', 'feature disabled')})."
    )
    role = detail.get('role', {})
    if role.get('alignment_available'):
        alignment_bits = []
        for lbl, key in (('Slot', 'slot_alignment_rate'), ('Non-slot', 'non_slot_alignment_rate'),
                         ('Wide', 'wide_alignment_rate'), ('Inline', 'inline_alignment_rate')):
            value = role.get(key)
            if value is not None and pd.notna(value):
                alignment_bits.append(f"{lbl}: {float(value):.0%}")
        if alignment_bits:
            st.markdown("**PFF alignment role**")
            st.caption(" · ".join(alignment_bits))
    if role.get('alignment_defense_candidate_available'):
        st.caption("PFF alignment-defense residual: audit-only, not yet scoring this projection (needs "
                  "enough weekly PFF archives, then a predeclared backtest to authorize it).")

    evidence = detail.get('alignment_scheme_evidence', {})
    if evidence.get('player_scheme_available'):
        man, zone = evidence.get('player_man_route_share'), evidence.get('player_zone_route_share')
        if man is not None and pd.notna(man):
            st.markdown("**PFF man/zone role**")
            st.caption(f"Man: {float(man):.0%}"
                      + (f" · Zone: {float(zone):.0%}" if zone is not None and pd.notna(zone) else ""))
    defense_bits = []
    if evidence.get('defense_alignment_candidate_available'):
        mult = evidence.get('defense_slot_candidate_multiplier')
        if mult is not None and pd.notna(mult):
            defense_bits.append(f"slot-weighted {float(mult):.3f}×")
    if evidence.get('defense_scheme_candidate_available'):
        mult = evidence.get('defense_man_candidate_multiplier')
        if mult is not None and pd.notna(mult):
            defense_bits.append(f"man-weighted {float(mult):.3f}×")
    if defense_bits:
        st.markdown("**Opponent alignment/scheme vulnerability (candidate only)**")
        st.caption(
            f"{detail['opponent']}'s defense, back-calculated from every offense it actually faced this "
            f"season — " + " · ".join(defense_bits) + ". Audit-only: not applied to this projection, pending "
            "a predeclared out-of-sample backtest."
        )


def _render_decomposition_audit_expander(detail):
    """Everything that isn't needed at a glance, preserved rather than
    dropped - role/depth-chart provenance, the RB allocator ledger, the
    availability and calibration sources, the injury/vacancy ledger, and the
    data-confidence/cutoff safeguards. One collapsed expander instead of
    being permanently on-page, which is what actually fixes 'too much text,
    hard to track' without losing any of the underlying audit trail."""
    with st.expander("Role, audit & data sources", expanded=False):
        role = detail.get('role', {})
        role_bits = []
        for label, key, fmt in (
            ('Expected snaps', 'expected_snap_share', '.0%'),
            ('Role confidence', 'role_confidence', '.0%'),
            ('Role-change evidence', 'role_change_confidence', '.0%'),
            ('Target-earner score', 'target_earner_score', '.2f'),
            ('Target share', 'target_share', '.1%'),
            ('Carry share', 'carry_share', '.1%'),
            ('ADOT', 'adot', '.1f'),
        ):
            value = role.get(key)
            if value is not None and pd.notna(value):
                role_bits.append(f"{label}: {float(value):{fmt}}")
        st.markdown("**Expected role**")
        st.caption(" · ".join(role_bits) if role_bits else "No time-valid role profile was available.")
        if detail.get('position') == 'QB' and role.get('qb_projected_starter'):
            st.success(f"Expected QB1: full workload from {role.get('starter_source', 'QB1 selection')}.")
        elif detail.get('position') == 'QB' and role.get('qb1_selection_required'):
            st.warning("QB1 selection required: this room receives no normal QB volume until one player is selected.")
        elif detail.get('position') == 'QB':
            st.caption("QB status: not the expected starter; projected QB volume is held at zero.")
        elif role.get('starter_source'):
            st.caption(f"Starter/workload source: {role['starter_source']}.")
        if role.get('partial_game_exclusions'):
            st.caption(f"Partial-game screen: {role['partial_game_exclusions']} clearly interrupted current-season sample(s) were excluded from rate evidence.")
        if role.get('returning_role_restored'):
            st.info(
                'Preseason role restoration: prior missed games were not treated as a reduced Week 1 workload; '
                f"the player's proven active-game role was used ({role.get('returning_role_reason', 'role recovery')})."
            )
        elif role.get('preseason_role_source') not in (None, 'Not applicable'):
            st.caption(f"Preseason role source: {role['preseason_role_source']}.")
        if role.get('ourlads_role_floor_applied'):
            label = role.get('ourlads_role_position_label') or detail.get('position')
            rank = role.get('ourlads_role_available_rank')
            floor = role.get('ourlads_role_floor')
            st.caption(
                f"Imported Ourlads evidence: available {label} depth rank "
                f"{int(rank) if rank is not None else '—'} supplied a conservative "
                f"{floor:.0%} role floor; it did not set full snaps or target share."
                if floor is not None else
                f"Imported Ourlads evidence: available {label} depth role was considered."
            )
        if role.get('identity_match_method'):
            source_name = role.get('ourlads_source_name') or detail['player']
            st.caption(
                f"Depth-chart identity: {source_name} → {detail['player']} "
                f"via {role.get('identity_match_method')} "
                f"({role.get('identity_match_confidence') or 'unrated'} confidence)."
            )
        if role.get('identity_match_warning'):
            st.warning(f"Depth-chart identity warning: {role['identity_match_warning']}")
        if role.get('ourlads_source_status_warning'):
            st.warning(
                f"Depth-chart status flag: {role['ourlads_source_status_warning']}. "
                "It is a chart-source warning only; current availability controls the projection."
            )
        if role.get('alignment_note'):
            st.caption(f"Alignment: {role['alignment_note']}")

        if detail.get('position') == 'RB' and (
                role.get('rb_allocator_applied')
                or role.get('rb_allocation_eligibility_reason')
                or role.get('rb_allocation_source')):
            st.markdown("**Preseason RB allocation**")
            st.caption(
                "Core-RB snaps, carries, and targets are reconciled separately. "
                "The residual is an explicit other/unallocated bucket rather than a hidden role for every reserve."
            )
            capacity = detail.get('rb_capacity_ledger', [])
            if capacity:
                capacity_frame = pd.DataFrame(capacity)
                display_columns = [
                    column for column in ('resource', 'capacity', 'allocated', 'unallocated',
                                           'candidate_count', 'other_fraction', 'reason')
                    if column in capacity_frame.columns
                ]
                st.dataframe(capacity_frame[display_columns], hide_index=True, width="stretch")
            allocation = detail.get('rb_team_allocation', [])
            if allocation:
                with st.expander("Team RB allocation and projected opportunities"):
                    st.dataframe(pd.DataFrame(allocation), hide_index=True, width="stretch")
            source = role.get('rb_allocation_source') or 'not recorded'
            reason = role.get('rb_allocation_eligibility_reason') or 'not recorded'
            st.caption(f"Allocator eligibility: {reason}. Source: {source}.")
            if role.get('rb_established_incumbent_backstop'):
                st.info("Established-incumbent safety backstop kept this active same-team RB eligible despite a source-match issue.")
            segment_status = role.get('rb_role_segment_status')
            if segment_status and segment_status != 'no_clear_internal_gap':
                segment_bits = [f"status {segment_status}"]
                for label, key, fmt in (
                    ('pre-gap snaps', 'rb_segment_pre_absence_snap_share', '.0%'),
                    ('gap games', 'rb_segment_gap_games', '.0f'),
                    ('return snaps', 'rb_segment_return_snap_share', '.0%'),
                    ('incumbent credit', 'rb_interrupted_incumbent_credit', '.2f'),
                    ('shared-healthy lead', 'rb_shared_healthy_lead_score', '+.2f'),
                    ('replacement downweight', 'rb_replacement_only_downweight', '.2f'),
                ):
                    value = role.get(key)
                    if value is not None and pd.notna(value):
                        segment_bits.append(f"{label} {float(value):{fmt}}")
                st.caption("Role-segment evidence: " + " · ".join(segment_bits))

        availability = detail['availability']
        st.markdown("**Availability and workload**")
        st.caption(
            f"{availability['status'] or 'No current designation'} — "
            f"plays probability {availability['plays_probability']:.0%}; "
            f"workload if active {availability['workload_if_active']:.0%}."
        )
        if availability.get('source'):
            availability_note = availability.get('note') or ''
            st.caption(
                f"Availability source: {availability['source']} "
                f"({availability.get('match_method', 'not recorded')})"
                f"{f'; {availability_note}' if availability_note else ''}."
            )

        calibration = detail.get('calibration', {})
        if calibration:
            if calibration.get('enabled'):
                st.caption(
                    f"Point calibration only: raw {calibration.get('raw_points', detail['raw_points']):.2f} → "
                    f"displayed {calibration.get('displayed_points', detail['calibrated_points']):.2f} "
                    f"({calibration.get('delta', 0.0):+.2f}); "
                    f"slope {calibration.get('slope', 1.0):.3f}, intercept {calibration.get('intercept', 0.0):.3f}."
                )
            else:
                st.caption("Point calibration is disabled for this model run; the stat line itself is unchanged either way.")

        vacancy = detail.get('vacancy', [])
        if vacancy:
            st.markdown("**Injury/vacancy ledger**")
            st.dataframe(pd.DataFrame(vacancy), hide_index=True, width="stretch")

        contract = detail.get('data_contract', {})
        st.markdown("**Data confidence and cutoff safeguards**")
        st.caption(
            f"Target classified as {'historical' if contract.get('historical_target') else 'live/upcoming'}; "
            f"latest observed week: {contract.get('latest_observed_week') or 'none'}."
        )
        for label in ('pff_season_totals', 'pff_alignment', 'pace', 'injury', 'market_script',
                      'prior_defense_recency', 'qb_starter_source',
                      'preseason_skill_role_policy', 'ourlads_preseason_depth_chart',
                      'availability', 'rb_role_segments',
                      'partial_game_history_filter'):
            st.write(f"{label.replace('_', ' ').title()}: {contract.get(label, 'unknown')}")
        st.caption("Market and FantasyPros values in the ranking table are comparisons, never inputs to this projection.")


def _open_projection_dialog(detail):
    """Open the backend-produced weekly projection decomposition: header ->
    primary at-a-glance table -> range of outcomes -> tabbed Deep Dive ->
    role/audit detail, in that order - explicit request, replacing what was
    previously a long flat stack of mostly free-text captions."""
    if not detail:
        return

    def _body():
        _render_decomposition_header(detail)
        _render_decomposition_primary_table(detail)

        distribution = detail.get('distribution')
        if distribution:
            st.markdown("**Range of outcomes**")
            _render_distribution_chart(distribution, detail['position'])
        else:
            st.caption(
                "Range of outcomes: not available yet for this run — needs scripts/fit_weekly_"
                "distribution.py's bands (see data/weekly_distribution.py)."
            )

        stats = detail.get('stats', {})
        if stats:
            st.markdown("**Deep dive**")
            game_log_by_season = detail.get('game_log_by_season') or {}
            defense_log_by_season = detail.get('defense_weekly_log_by_season') or {}
            season_years = sorted(set(game_log_by_season) | set(defense_log_by_season), reverse=True)
            if len(season_years) > 1:
                current_year = detail.get('season_year')
                season_labels = {
                    yr: (f"{yr} (this season)" if yr == current_year else f"{yr} (last season)")
                    for yr in season_years
                }
                # Default to whichever season actually has games logged - the
                # honest empty case is a cold-start CURRENT season, and
                # there's no reason to land the user on a blank tab by
                # default when last season's full log is one click away.
                default_year = next(
                    (yr for yr in season_years if game_log_by_season.get(yr)), season_years[0])
                selected_year = st.radio(
                    "Season", season_years, index=season_years.index(default_year),
                    format_func=lambda yr: season_labels[yr], horizontal=True,
                    key=(f"weekly_rank_deepdive_season_{detail['player']}_{detail['team']}_"
                        f"{detail['target_week']}_{detail['as_of_week']}"),
                )
            else:
                selected_year = season_years[0] if season_years else detail.get('season_year')
            game_log = game_log_by_season.get(selected_year) or []
            defense_log = defense_log_by_season.get(selected_year) or []
            stat_keys = list(stats.keys())
            tabs = st.tabs([s.replace('_', ' ').title() for s in stat_keys] + ['Context'])
            for stat, tab in zip(stat_keys, tabs[:-1]):
                with tab:
                    _render_stat_deep_dive(detail, stat, game_log, defense_log, selected_year)
            with tabs[-1]:
                _render_context_deep_dive(detail)

        _render_decomposition_audit_expander(detail)

    dialog = st.dialog(f"Projection decomposition — {detail['player']}", width="large",
                       on_dismiss=_close_projection_dialog)(_body)
    dialog()


def _limit_rows(df, key):
    choice = st.selectbox("Show", _SHOW_N_OPTIONS, index=1, key=key)
    return df if choice == "All" or len(df) <= choice else df.head(choice)


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

    Stamps a fetch time alongside the pull (`wr_fp_weekly_fetched_at`) so
    the table below can show a real "current to Week N, pulled at TIME"
    indicator instead of leaving whether the FantasyPros numbers are fresh
    or a stale holdover from an earlier week/scoring choice unstated.
    """
    import datetime
    from data.draft_sources import (
        get_fantasypros_api_key, save_fantasypros_api_key,
        fantasypros_api_calls_this_month,
        fetch_fantasypros_weekly_projections,
    )
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
    fetch = st.button("Fetch weekly projections", key="wr_fp_fetch", disabled=not api_key)
    st.caption(f"{used} call{'s' if used != 1 else ''} made this month - this pull costs 4.")
    if fetch:
        with st.spinner("Calling the FantasyPros API…"):
            df, meta = fetch_fantasypros_weekly_projections(api_key, wk_year, wk_week)
        if meta.get('error') and df.empty:
            st.error(f"FantasyPros API: {meta['error']}")
        else:
            st.session_state['wr_fp_weekly_df'] = df
            st.session_state['wr_fp_weekly_meta'] = (wk_year, wk_week, wk_scoring)
            st.session_state['wr_fp_weekly_fetched_at'] = datetime.datetime.now()
            if meta.get('errors'):
                st.warning(f"Some positions failed: {meta['errors']}")
            st.success(f"Pulled {len(df):,} FantasyPros weekly projections for week {wk_week}.")
            st.rerun()

    held = st.session_state.get('wr_fp_weekly_df')
    held_meta = st.session_state.get('wr_fp_weekly_meta')
    if held is not None and not held.empty and held_meta == (wk_year, wk_week, wk_scoring):
        return held
    return None


def _render_fantasypros_injury_pull(wk_year, wk_week):
    """
    FantasyPros-sourced injury/availability for the V2 model
    (v2_fantasypros_availability - see data.fantasypros_availability's own
    module docstring for why this replaced the nflverse last-designation
    feed). Two sources write the SAME file/schema, so either one - or both,
    week to week, as the user's access changes - just works with no further
    change here or in the model: an uploaded weekly export (works today) and
    a live API pull (data.fantasypros_availability.fetch_fantasypros_injury_
    report - the field name it looks for on the /nfl/players response is a
    best-effort guess, unverified against a real response; a failed fetch
    says exactly what the response DID carry so the real field name can be
    added from evidence). Absence of either for this week still means
    healthy, by design - this expander is purely optional.
    """
    from data.draft_sources import get_fantasypros_api_key, save_fantasypros_api_key
    from data.fantasypros_availability import (
        save_fantasypros_injury_upload, save_fantasypros_injury_api_pull, load_fantasypros_availability,
    )
    st.caption(
        "Feeds the V2 model's availability discount. No report for a player this week means "
        "healthy - never a stale carry-over from a prior week."
    )
    tab_api, tab_upload = st.tabs(["Live API pull", "Upload export"])
    with tab_api:
        secret_key, key_source = get_fantasypros_api_key()
        if key_source == 'secrets':
            st.caption("🔑 Using the key from `.streamlit/secrets.toml`.")
            api_key = secret_key
        else:
            api_key = st.text_input("FantasyPros API key", type="password",
                                    key="wr_fp_injury_api_key", value=secret_key)
            if api_key and api_key != secret_key:
                save_fantasypros_api_key(api_key)
        fetch = st.button("Fetch injury status", key="wr_fp_injury_fetch", disabled=not api_key)
        if fetch:
            with st.spinner("Calling the FantasyPros API…"):
                n, error = save_fantasypros_injury_api_pull(api_key, wk_year, wk_week)
            if error:
                st.error(f"FantasyPros API: {error}")
            else:
                st.success(f"Loaded {n} flagged player{'s' if n != 1 else ''} for week {wk_week}.")
                st.rerun()
    with tab_upload:
        up = st.file_uploader("FantasyPros injury export (CSV)", type=["csv"], key="wr_fp_injury_upload")
        if up is not None:
            n, error = save_fantasypros_injury_upload(up, wk_year, wk_week)
            if error:
                st.error(error)
            else:
                st.success(f"Loaded {n} flagged player{'s' if n != 1 else ''} for week {wk_week}.")
                st.rerun()

    profiles, _err = load_fantasypros_availability(wk_year, wk_week)
    if profiles:
        st.caption(f"Currently loaded for Week {wk_week}: {len(profiles)} player(s) not listed healthy.")
    else:
        st.caption(f"Nothing loaded for Week {wk_week} yet — every player defaults to healthy.")


def _render_pff_weekly_alignment_upload(wk_year, wk_week):
    """
    Upload widget for the weekly PFF slot/wide/inline archive (data.
    pff_alignment - see docs/pff_weekly_alignment_archive.md for the export
    itself). Purely additive to a season's archive: each upload here is one
    more week alongside whatever weeks are already saved, and the model only
    ever reads weeks strictly before the one being projected (the same as-of
    guard every other current-season input in this app goes through).

    This is intentionally the ONLY thing this expander does - no matchup
    preview, no defense ranking. The player role rates and defense
    vulnerability profile this feeds are audit-only in the decomposition
    dialog until they clear a predeclared backtest (see that module's own
    docstring), so nothing here changes a single displayed point yet.
    """
    from data.pff_alignment import save_weekly_alignment_export, discover_weekly_alignment_exports
    st.caption(
        "One league-wide receiving_summary.csv + receiving_concept.csv pair per played week. "
        "Builds the player role-rate and defense-vulnerability foundation for a future slot/wide/"
        "inline matchup model; audit-only today, not applied to any projection yet."
    )
    upload_week = st.number_input("Week this export covers", min_value=1, max_value=22,
                                  value=int(wk_week) if wk_week else 1, step=1, key="wr_pff_align_week")
    c1, c2 = st.columns(2)
    with c1:
        summary_up = st.file_uploader("receiving_summary.csv", type=["csv"], key="wr_pff_align_summary")
    with c2:
        concept_up = st.file_uploader("receiving_concept.csv", type=["csv"], key="wr_pff_align_concept")
    if st.button("Save this week's archive", key="wr_pff_align_save",
                 disabled=not (summary_up and concept_up)):
        ok, issues = save_weekly_alignment_export(summary_up, concept_up, wk_year, int(upload_week))
        if ok:
            st.success(f"Saved Week {int(upload_week)}, {wk_year} to the alignment archive.")
            st.rerun()
        else:
            st.error(" ".join(issues) or "Could not save that export.")

    discovered = discover_weekly_alignment_exports(wk_year)
    archives = discovered.archives
    if archives is not None and not archives.empty and 'week' in archives.columns:
        weeks_present = sorted(int(w) for w in archives['week'].dropna().unique())
        st.caption(f"Weeks archived for {wk_year}: {weeks_present}.")
    else:
        st.caption(f"No weekly archive saved for {wk_year} yet.")


def _fantasypros_freshness_caption(wk_year, wk_week, wk_scoring):
    """The "has this actually been pulled, and when" indicator - explicit
    request. Distinguishes a live pull FOR this exact year/week/scoring
    selection from a held-over one for a different combination, since the
    session can carry a stale pull from before the selectors above changed."""
    held_meta = st.session_state.get('wr_fp_weekly_meta')
    fetched_at = st.session_state.get('wr_fp_weekly_fetched_at')
    if held_meta is None or fetched_at is None:
        st.caption("📡 FantasyPros: not pulled this session — open the expander above to fetch it.")
        return
    if held_meta == (wk_year, wk_week, wk_scoring):
        st.caption(
            f"📡 FantasyPros projections current to **Week {wk_week}, {wk_year}** "
            f"({wk_scoring}) — pulled {fetched_at:%b %d, %Y at %I:%M %p}."
        )
    else:
        held_year, held_week, held_scoring = held_meta
        st.caption(
            f"📡 FantasyPros: last pull was for Week {held_week}, {held_year} ({held_scoring}) at "
            f"{fetched_at:%b %d, %I:%M %p} — that doesn't match the selection above, so it isn't shown below. "
            "Re-fetch to pull the current selection."
        )


def _scoring_dict(scoring_mode):
    """This tab's simple 'Full PPR'/'Half-PPR'/'Standard' picker, expanded
    into the full per-stat scoring dict data.draft_board.score_stats (and
    therefore the market-projection path) needs. Every non-reception weight
    here already matches what data.transforms.score_projected_stats hardcodes
    for the model's own points, so this doesn't introduce a second scoring
    opinion - it's the same rules in the dict shape the market side reads."""
    rec = 1.0 if 'Full' in scoring_mode else (0.5 if 'Half' in scoring_mode else 0.0)
    return {**DEFAULT_SCORING, 'rec': rec}


def _render_weekly_market_pull(wk_scoring, name_pool):
    """
    Live player-prop lines -> a market-implied projected-points column,
    from the SAME free weekly board (PrizePicks/Underdog/DraftKings - no
    key, no quota) the Live Odds tab already pulls
    (data.odds_weekly.weekly_props), scored the way Draft HQ scores a
    season-long book line (data.odds_weekly.weekly_market_projection).

    This is the live CURRENT slate - books post the coming weekend's board
    Tuesday/Wednesday and there is no way to ask them for a past week's
    lines, so unlike the model and the FantasyPros pull this can't be
    retargeted at an arbitrary year/week from the selectors above. A
    caption says so plainly rather than silently mismatching.
    """
    st.caption(
        "PrizePicks, Underdog and DraftKings — no key, no quota. The same live weekly board "
        "the Live Odds tab pulls, scored under the scoring mode selected above into a "
        "projected-points column here instead of a raw line list. This is always the CURRENT "
        "posted slate, not necessarily the season/week picked above."
    )
    force = st.button("🔄 Refresh player props", key="wr_market_refresh")
    with skeleton_loader("table", n_rows=5, n_cols=4):
        props, meta = weekly_props(force=force)
    stamp = meta.get('fetched_at')
    if stamp:
        age_bits = [f"Pulled {stamp:%b %d, %Y at %I:%M %p}"]
        if meta.get('stale'):
            age_bits.append("stale — a fresh pull didn't return anything")
        st.caption(" · ".join(age_bits))
    if props.empty:
        st.caption("No live player props available right now.")
        return pd.DataFrame()
    scored, mmeta = weekly_market_projection(props, _scoring_dict(wk_scoring), board=name_pool)
    if scored.empty:
        st.caption("Lines came back but none mapped to a stat this app scores.")
        return pd.DataFrame()
    st.success(f"{mmeta['players']} players priced from the live board.")
    return scored.rename(columns={'player': 'Player'})


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


def _form_entry(form_series, name):
    """(values, season average) for one player's sparkline, or ([], None)
    when this season has nothing for him - a rookie, or anyone who hasn't
    played yet. Split out so the list comprehension that builds the column
    stays readable."""
    entry = form_series.get(name)
    if not entry:
        return [], None
    values, season_avg, _weeks = entry
    return values, season_avg


# Position slot order inside one woven rank value (see _woven_rank below).
# Any position not listed shares the last slot - the model only ever emits
# QB/RB/WR/TE, the rest are here so the encoding survives a future caller.
_RANK_POSITION_ORDER = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
_RANK_SLOTS = 10


def _woven_rank(df, value_col, pos_col='Pos'):
    """
    A positional rank column ('RB4' style, matching the app's existing
    convention) from any points column - one per projection SOURCE (model,
    market, FantasyPros), so the three can be read and compared side by side
    rather than only this app's own model carrying a rank at all.

    Returns (int64 Series, {value: display label}). The column that goes
    into the grid is a NUMBER; the label is applied by
    ui.styling.style_plain_dataframe's `label_cols` at render time. That
    split is the whole point of this function, and it fixes three separate
    real complaints about the old string-valued version at once:

      1. A string rank column sorts with localeCompare in the grid, so
         "QB10" lands between "QB1" and "QB2" - a rank column that doesn't
         sort in rank order.
      2. A string rank column also sorts every QB above every RB above every
         WR (alphabetical on the position prefix), so "sort by rank" buried
         every other position under the quarterbacks instead of showing the
         positions WOVEN together.
      3. A missing rank (a source that doesn't carry this player) sorted to
         the TOP of an ascending sort, not the bottom - the grid reads a
         null cell as an empty string and an empty string compares below
         every number (HANDOFF.md gotcha #5, confirmed here by reading the
         frontend's own comparator).

    The encoding is `rank * _RANK_SLOTS + position_slot`, so ascending order
    is QB1, RB1, WR1, TE1, QB2, RB2, ... - every position's Nth-best player
    sitting together, which is what "ranked together" means for a start/sit
    table. Unranked players get `(worst_rank + 1) * _RANK_SLOTS + slot`, a
    real number worse than every real rank (so it sorts last, in either
    direction, deterministically) that labels as an em dash rather than
    pretending to be a rank.
    """
    empty = (None, {})
    if value_col not in df.columns or pos_col not in df.columns or df.empty:
        return empty
    values = pd.to_numeric(df[value_col], errors='coerce')
    if not values.notna().any():
        return empty

    positions = df[pos_col].astype(str).str.upper()
    ranks = pd.DataFrame({'_v': values, '_p': positions}).groupby('_p')['_v'].rank(
        ascending=False, method='first')
    slots = positions.map(
        lambda p: _RANK_POSITION_ORDER.index(p) if p in _RANK_POSITION_ORDER else len(_RANK_POSITION_ORDER))
    sentinel_rank = int(ranks.max()) + 1
    filled = ranks.fillna(sentinel_rank).astype(int)
    encoded = (filled * _RANK_SLOTS + slots).astype('int64')

    labels = {}
    for enc, pos, rank, is_real in zip(encoded, positions, filled, ranks.notna()):
        labels[int(enc)] = f'{pos}{int(rank)}' if is_real else '—'
    return encoded, labels


def _render_live_data_hub(wk_year, wk_week, wk_scoring, wk_week_completed, model_version, name_pool):
    """
    One expander, four tabs - consolidates what used to be four separate
    stacked accordions (FantasyPros weekly projection, FantasyPros injury/
    availability, PFF weekly alignment archive, market player-prop
    projection) so a page visit isn't four collapsed boxes deep before any
    model output appears.

    Streamlit tabs can't be conditionally hidden once declared, so the two
    gates that used to hide a whole expander (V2-only for injury/PFF,
    upcoming-week-only for market props) now show as an explanatory caption
    INSIDE that tab instead - the tab is always there, its content just says
    why there's nothing to do yet.

    Returns (fp_weekly_df, market_df), the two values render() still needs
    downstream; the injury and PFF tabs read/write straight to disk and
    session state and have no return-value contract of their own.
    """
    with st.expander("📡 Live data pulls", expanded=False):
        tab_fp, tab_injury, tab_pff, tab_market = st.tabs(
            ["FantasyPros projections", "Injury/availability", "PFF alignment", "Market props"])
        with tab_fp:
            fp_weekly = _render_fantasypros_weekly_pull(wk_year, wk_week, wk_scoring)
        with tab_injury:
            if model_version == 'v2':
                _render_fantasypros_injury_pull(wk_year, wk_week)
            else:
                st.caption("Feeds the V2 model's availability discount — switch Projection model "
                          "to V2 above to use this.")
        with tab_pff:
            if model_version == 'v2':
                _render_pff_weekly_alignment_upload(wk_year, wk_week)
            else:
                st.caption("Feeds the V2 model's future matchup work — switch Projection model "
                          "to V2 above to use this.")
        with tab_market:
            # See the (former) standalone function's docstring: sportsbooks
            # only ever post the CURRENT live slate, so this can't be
            # retargeted at an already-played week.
            if wk_week_completed:
                st.caption(
                    f"📈 Market Proj Pts isn't shown for {wk_year} Week {wk_week} — that week is "
                    "already played, and sportsbooks only post the current live slate. Pick the "
                    "upcoming week to see it."
                )
                market_df = None
            else:
                market_df = _render_weekly_market_pull(wk_scoring, name_pool)
    return fp_weekly, market_df


def render():
    st.markdown("<div class='custom-section-header'>WEEKLY RANKINGS</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        # WITH_UPCOMING (not AVAILABLE_SEASONS) so the season about to start
        # is selectable and defaults FIRST (index=0) - unlike the tabs that
        # default to index=1 to skip a season with no weekly stats yet
        # (Player Search, Depth Charts, ...), this tab's whole job is "what
        # should I do this week", so once the new season's schedule is live
        # the always-useful default is the CURRENT/upcoming week, not a
        # stale prior season's final week (see _default_week_index below -
        # a season with no games played yet opens on its Week 1).
        wk_year = st.selectbox("Season", AVAILABLE_SEASONS_WITH_UPCOMING, index=0, key="weekly_rank_year")
    weeks = _week_options(wk_year)
    week_labels = [w['label'] for w in weeks]
    with c2:
        wk_week_idx = st.selectbox("Week", range(len(weeks)), index=_default_week_index(weeks),
                                   format_func=lambda i: week_labels[i], key="weekly_rank_week")
    wk_week = weeks[wk_week_idx]['week']
    wk_week_completed = bool(weeks[wk_week_idx].get('completed')) if weeks else False
    with c3:
        wk_scoring = st.selectbox("Scoring", ["Full PPR", "Half-PPR", "Standard"], key="weekly_rank_scoring")
    with c4:
        model_choice = st.selectbox(
            "Projection model", ["V1 (released baseline)", "V2 (experimental)"],
            key="weekly_rank_model_version",
            help="V2 exposes cutoff-safe inputs and a player decomposition while it is evaluated against V1.",
        )
    model_version = 'v2' if model_choice.startswith('V2') else 'v1'

    with skeleton_loader("table", n_rows=10, n_cols=7):
        df_stats, t_col, n_col, _ = load_and_merge_data(wk_year, wk_scoring)
        model_df, model_meta = build_weekly_projections(
            wk_year, wk_week, wk_scoring, model_version=model_version)

    form_df = build_recent_form_rank(df_stats, n_col, t_col, n_weeks=RECENT_FORM_GAMES)

    st.markdown("### This app's weekly model")
    if model_version == 'v2':
        st.caption(
            "V2 is an auditable experiment. Select a player row for its decomposition; "
            "V1 remains the released baseline while V2 is backtested."
        )
    fp_weekly, market_df = _render_live_data_hub(
        wk_year, wk_week, wk_scoring, wk_week_completed, model_version,
        model_df[['Player']] if not model_df.empty else None)
    _fantasypros_freshness_caption(wk_year, wk_week, wk_scoring)

    if model_df.empty:
        st.info(f"This app's model has no projection for {wk_year} week {wk_week}: "
               f"{model_meta.get('reason', 'not enough data yet')}")
        if fp_weekly is not None:
            st.caption("Showing FantasyPros' live weekly projection instead:")
            fp_pts_col = _fantasypros_points_column(wk_scoring)
            cols = ['Player', 'Pos', 'Team'] + ([fp_pts_col] if fp_pts_col in fp_weekly.columns else [])
            fp_only = fp_weekly[cols].rename(columns={fp_pts_col: 'FantasyPros Proj Pts'})
            fp_positions, _fp_group = position_group_buttons('wrfp', default='SUPERFLEX')
            display_df = apply_position_group(fp_only, fp_positions, pos_col='Pos')
            display_df = _limit_rows(display_df, key="weekly_rank_show_n")
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
        if market_df is not None and not market_df.empty:
            merged_model = _attach_by_name(merged_model, market_df, ['Market Pts', 'Coverage'], 'Mkt ')
            merged_model = merged_model.rename(
                columns={'Mkt Market Pts': 'Market Proj Pts', 'Mkt Coverage': 'Market Coverage'})
            if 'Market Coverage' in merged_model.columns:
                # Missing stats price as ZERO in Market Proj Pts by design
                # (data.odds_projections.score_market_lines' own docstring -
                # keeping the market number independent of this app's model
                # means a gap can't be filled in), which is exactly why a
                # partial line reads "very low" with no explanation unless
                # this rides right alongside it - a book that only posted a
                # receptions prop for someone shows a real but partial
                # number, not a broken one.
                merged_model['Market Coverage'] = merged_model['Market Coverage'].map(
                    lambda v: f"{v * 100:.0f}%" if pd.notna(v) else None)
        # Ranking column, directly after Opponent, colored by TIER rather
        # than a continuous scale - explicit request. Tiers are clustered
        # per position on Model Proj Pts wherever a significant cutoff
        # actually falls (data.draft_board.tier_by_position, the same
        # k-means-on-points technique Draft HQ's board tiers with), not a
        # fixed players-per-tier bucket.
        # Tier shading for the model's own rank column - explicit request.
        # Tiers are clustered per position on Model Proj Pts wherever a
        # significant cutoff actually falls (data.draft_board.tier_by_position,
        # the same k-means-on-points technique Draft HQ's board tiers with),
        # not a fixed players-per-tier bucket.
        merged_model['_tier'] = tier_by_position(merged_model, 'Model Proj Pts', pos_col='Pos')

        # One rank column per projection SOURCE, not just this app's own
        # model - explicit request. Each is a sortable NUMBER carrying a
        # "RB4" label (see _woven_rank for why that split exists and which
        # three sorting bugs it fixes); only built when that source actually
        # produced a points column to rank. Computed on the FULL pool, before
        # the position filter and row limit below, so "RB4" always means
        # fourth among every RB rather than fourth among what's on screen.
        rank_labels = {}
        for rank_col, points_col in (('Model Rank', 'Model Proj Pts'),
                                     ('Market Rank', 'Market Proj Pts'),
                                     ('FantasyPros Rank', 'FantasyPros Proj Pts')):
            if points_col not in merged_model.columns:
                continue
            values, labels = _woven_rank(merged_model, points_col)
            if values is None:
                continue
            merged_model[rank_col] = values
            rank_labels[rank_col] = labels

        # The leading "Rank" column is the table's scan anchor and its
        # default sort - FantasyPros' rank when their projection has actually
        # been pulled this session, this app's model rank otherwise. It
        # deliberately repeats one of the three source ranks at the far right
        # rather than being a fourth, separate ranking: the requested layout
        # puts a rank first (what am I looking at) and the full three-source
        # comparison last (who disagrees with whom).
        primary_rank = next((c for c in ('FantasyPros Rank', 'Model Rank') if c in rank_labels), None)
        if primary_rank:
            merged_model['Rank'] = merged_model[primary_rank]
            rank_labels['Rank'] = rank_labels[primary_rank]
            # Woven order by construction (QB1, RB1, WR1, TE1, QB2, ...), the
            # ordering the whole encoding exists to produce - so the table
            # OPENS in it instead of only reaching it after a header click.
            merged_model = merged_model.sort_values('Rank', kind='mergesort').reset_index(drop=True)

        # Does our model's rank actually MATCH FantasyPros' own published
        # consensus rank (their real ECR, not the positional rank derived
        # above from their points projection - the two aren't guaranteed to
        # agree, since a projection reflects only the stat line while a
        # published ECR also folds in analyst judgment calls a raw point
        # total won't capture) - explicit request. Reads whatever weekly
        # FantasyPros export is already sitting in this session's uploader
        # (below) without requiring it to be re-uploaded once it's there;
        # see that uploader's own docstring for why a live per-week ECR
        # pull isn't wired in here (that endpoint exists on FantasyPros'
        # API - see draft_sources.py's "/rankings or /consensus-rankings"
        # note - but is unverified against a real response in this app and
        # a wrong guess would silently spend the call budget on garbage;
        # the CSV export is the same real ECR with zero guessing involved).
        _weekly_ecr_df = st.session_state.get('_weekly_rank_ecr_df')
        if _weekly_ecr_df is not None and not _weekly_ecr_df.empty:
            ecr_comparison = build_rankings_comparison(
                merged_model, value_col='Model Proj Pts', rank_label='Model Overall',
                fp_df=_weekly_ecr_df)
            if not ecr_comparison.empty and 'FantasyPros Rank' in ecr_comparison.columns:
                ecr_comparison = ecr_comparison.rename(columns={
                    'FantasyPros Rank': 'FantasyPros ECR',
                    'Model Overall vs FantasyPros': 'Model vs FantasyPros ECR',
                })
                keep = [c for c in ('Player', 'FantasyPros ECR', 'Model vs FantasyPros ECR')
                       if c in ecr_comparison.columns]
                merged_model = merged_model.merge(ecr_comparison[keep], on='Player', how='left')

        merged_model = merged_model.rename(columns={'Pos': 'Position'})
        merged_model = merged_model.rename(columns=dict(_STAT_DISPLAY_COLS))

        # Explicit column order, per request: identity first, then the three
        # projections side by side, then the stat line behind this app's own
        # number, then the context columns, then the three source ranks
        # together as a comparison block at the end.
        #
        # No standalone Position column - dropped to save table width, per
        # explicit request. Its info isn't lost: every rank column already
        # carries it in the "RB4"-style label (_woven_rank), and the rank
        # columns are now colored by position too (position_values/
        # position_cols below) so the same at-a-glance signal the Position
        # column gave survives without spending a column on it.
        display_cols = ['Rank', 'Player', 'Team', 'Opponent',
                        'FantasyPros Proj Pts', 'Market Proj Pts', 'Model Proj Pts']
        display_cols += [label for _col, label in _STAT_DISPLAY_COLS]
        display_cols += ['Market Coverage', 'Injury Status', 'Last 5 Weeks',
                        'FantasyPros Rank', 'Model Rank', 'Market Rank',
                        'FantasyPros ECR', 'Model vs FantasyPros ECR']

        # 'Last 5 Weeks' is built per-displayed-row further down, so it isn't
        # a real column yet at slicing time. 'Position' is kept even though
        # it's no longer in display_cols (see the comment above) - the
        # position-group filter just below and the rank-column position
        # coloring further down both still need the raw values; it's
        # dropped from the actually-rendered frame later, at the
        # display_cols re-select (indexed = indexed[[...]]).
        keep_cols = [c for c in display_cols if c in merged_model.columns] + ['_tier', 'Position']
        positions, group_label = position_group_buttons('wr', default='SUPERFLEX')
        filtered_df = apply_position_group(merged_model[keep_cols], positions, pos_col='Position')
        total_filtered = len(filtered_df)
        display_df = _limit_rows(filtered_df, key="weekly_rank_show_n")
        tier_values = display_df['_tier'].tolist()
        position_values = display_df['Position'].tolist()
        display_df = display_df.drop(columns=['_tier'])
        indexed = display_df.set_index('Player')

        # Recent-form sparkline: the last five games' fantasy points as a
        # line, with the player's SEASON average as a dotted reference line
        # under it - explicit request, and the reason this is an inline SVG
        # in an ImageColumn rather than the st.column_config.LineChartColumn
        # Rookie Watch/Risers use (that column can draw the line and nothing
        # else - see ui.charts.sparkline_data_uri).
        #
        # Gated to games BEFORE the selected week, which is also the fix for
        # "Last 5 Weeks doesn't currently show anything": this tab opens on
        # the UPCOMING season/week, and an upcoming season has no weekly stat
        # rows at all, so every cell was correctly-but-uselessly empty. When
        # that happens the series falls back to the prior season, captioned,
        # the same "based on last season" fallback the model itself already
        # makes for a cold-start week (see build_weekly_projections).
        form_series = build_form_series(df_stats, n_col, metric='fantasy_points',
                                        n_weeks=RECENT_FORM_GAMES, before_week=wk_week)
        form_source_year = wk_year
        if not form_series:
            try:
                prior_stats, _pt, _pn, _ = load_and_merge_data(wk_year - 1, wk_scoring)
                form_series = build_form_series(prior_stats, _pn, metric='fantasy_points',
                                                n_weeks=RECENT_FORM_GAMES)
                form_source_year = wk_year - 1
            except Exception:
                form_series = {}
        indexed['Last 5 Weeks'] = [
            sparkline_data_uri(*_form_entry(form_series, name)) for name in indexed.index
        ]
        # Re-apply the requested order now that Last 5 Weeks exists - it has
        # to be built per DISPLAYED row (one SVG each), which is after the
        # slice above, so it lands on the end of the frame rather than in
        # its requested slot between Injury Status and the rank block.
        indexed = indexed[[c for c in display_cols if c in indexed.columns]]

        pct_cols = {}
        for c in ('Model Proj Pts', 'Market Proj Pts', 'FantasyPros Proj Pts'):
            if c in indexed.columns and indexed[c].notna().any():
                pct_cols[c] = calculate_percentile(indexed.reset_index(), c)
        column_config = build_column_help_config(
            indexed, pinned_cols=['Rank', 'Team', 'Opponent'])
        column_config['Last 5 Weeks'] = st.column_config.ImageColumn(
            help=(f"Fantasy points over the last {RECENT_FORM_GAMES} games played "
                  f"({form_source_year}), with the dotted line at that season's average"),
            width="small",
        )
        detail_config = (wk_year, wk_week, wk_scoring, model_version)
        selected_state = st.session_state.get(_PROJECTION_DETAIL_KEY)
        if selected_state and selected_state.get('config') != detail_config:
            # A selection is meaningful only for the exact board it came
            # from.  This prevents a Week 3 explanation being displayed over
            # a Week 4 table after any selector changes.
            st.session_state.pop(_PROJECTION_DETAIL_KEY, None)
        details_by_row = [
            model_meta.get('explanations', {}).get((row['Player'], row['Position'], row['Team']))
            for _, row in display_df.iterrows()
        ]
        model_table_key = f"weekly_rank_model_table_{wk_year}_{wk_week}_{wk_scoring}_{model_version}"

        def _on_model_row_select():
            rows = st.session_state.get(model_table_key, {}).get('selection', {}).get('rows', [])
            if rows and rows[0] < len(details_by_row):
                detail = details_by_row[rows[0]]
                if detail:
                    st.session_state[_PROJECTION_DETAIL_KEY] = {
                        'config': detail_config, 'detail': detail,
                    }

        st.dataframe(
            style_plain_dataframe(indexed, numeric_pct_cols=pct_cols,
                                  tier_cols={'Model Rank': tier_values} if 'Model Rank' in indexed.columns else None,
                                  # Model Rank keeps its existing tier shading
                                  # (a separate explicit request - performance
                                  # cliffs, not position) rather than also
                                  # getting position colors here; the other
                                  # three rank columns had no coloring at all
                                  # before, so this is a clean addition there.
                                  position_cols={c: position_values for c in
                                                 ('Rank', 'FantasyPros Rank', 'Market Rank')
                                                 if c in indexed.columns},
                                  label_cols={c: labels for c, labels in rank_labels.items()
                                              if c in indexed.columns}),
            width="stretch", height=df_auto_height(min(len(display_df), 40), row_px=42),
            row_height=42, column_config=column_config,
            on_select=_on_model_row_select, selection_mode="single-row", key=model_table_key,
        )
        selected_state = st.session_state.get(_PROJECTION_DETAIL_KEY)
        if selected_state and selected_state.get('config') == detail_config:
            _open_projection_dialog(selected_state.get('detail'))
        shown_note = f"Showing {len(display_df)} of {total_filtered} {group_label} players"
        if total_filtered > len(display_df):
            st.caption(f"{shown_note} — widen \"Show\" above to see more.")
        else:
            st.caption(f"{shown_note}.")
        if form_source_year != wk_year:
            st.caption(
                f"↩︎ **Last 5 Weeks** falls back to {form_source_year} — {wk_year} has no played "
                "games before the selected week to chart yet."
            )
        st.caption(
            "**Rank** is the table's default order and its scan anchor — FantasyPros' positional "
            "rank when their projection has been pulled above, this app's model rank otherwise. "
            "Every rank column sorts WOVEN (QB1, RB1, WR1, TE1, QB2, …) rather than alphabetically, "
            "so no single position sweeps the top of the table, and a player a source doesn't rank "
            "shows \"—\" and sorts to the bottom instead of the top. "
            "**FantasyPros Rank** / **Model Rank** / **Market Rank** at the right are each source's "
            "own positional rank, side by side — Model Rank is shaded by tier, a cluster break in "
            "Model Proj Pts at that position, not a fixed players-per-tier cutoff. "
            "**FantasyPros ECR** (when a weekly FantasyPros export is uploaded below) is "
            "their own REAL published consensus rank, not derived from the points projection above "
            "it — **Model vs FantasyPros ECR** shows how far apart the two actually are, which is "
            "not always the same story a proj-pts comparison alone would tell, since a published "
            "ECR also folds in analyst judgment a raw stat-line projection doesn't capture. "
            "**Model Proj Pts** is this app's own projection (usage blended toward the current "
            "season as it grows, opponent/pace/game-script adjusted - see "
            "docs/weekly_projections_methodology.md), with the stat line it's built from shown "
            "alongside it. **Market Proj Pts** is this week's live sportsbook player-prop lines "
            "re-scored under this league's settings — missing stats price as zero, so **Market "
            "Coverage** shows how much of a typical week's points the posted lines actually cover; "
            "a low number means a partial line, not a bad projection. **FantasyPros Proj Pts** is "
            "their analysts' number, pulled live above. Independent reads, shown side by side, "
            "never blended."
        )

    if form_df.empty:
        st.info(f"Not enough {wk_year} weekly data yet to build a recent-form baseline.")
        return

    st.markdown("---")
    st.markdown("**Upload a weekly rankings export** (FantasyPros weekly export, or any CSV with a player-name column)")
    st.caption(
        "A real FantasyPros weekly export (not a custom one) also feeds the **FantasyPros ECR** / "
        "**Model vs FantasyPros ECR** columns in the model table above, on the NEXT rerun after "
        "uploading — their actual published rank, not one derived from a points projection."
    )
    import_hint('fantasypros_weekly')
    weekly_upload = st.file_uploader("Weekly rankings CSV", type=["csv"], key="weekly_rank_upload")
    weekly_df = None
    if weekly_upload is not None:
        weekly_df = parse_fantasypros_upload(weekly_upload)
        # Only a REAL FantasyPros-shaped export counts as "ECR" for the top
        # table's comparison - a generic custom upload (parsed below as a
        # fallback) could be anyone's ranking, and labelling that "FantasyPros
        # ECR" would just be wrong.
        st.session_state['_weekly_rank_ecr_df'] = weekly_df if not weekly_df.empty else None
        if weekly_df.empty:
            weekly_df = parse_custom_rankings(weekly_upload)
        if weekly_df is None or weekly_df.empty:
            st.warning("Couldn't find a player-name column in that upload — expected a header like 'Player', 'Player Name', or 'Name'.")

    if weekly_df is None or weekly_df.empty:
        return

    comparison = build_rankings_comparison(form_df, value_col='Recent Avg FPTS', rank_label='Form', fp_df=weekly_df)
    extra_cols = [c for c in comparison.columns if c not in ('Player', 'Pos', 'Team')]
    merged = form_df.merge(comparison[['Player'] + extra_cols], on='Player', how='left')
    merged = merged.rename(columns={'Recent Avg FPTS': 'L5 Avg FPTS'})

    upload_positions, _upload_group = position_group_buttons('wrup', default='SUPERFLEX')
    display_df = apply_position_group(merged, upload_positions, pos_col='Pos')
    display_df = _limit_rows(display_df, key="weekly_rank_upload_show_n")

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
