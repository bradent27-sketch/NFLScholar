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

from config import (AVAILABLE_SEASONS_WITH_UPCOMING, TEAM_CONFIG, TAB_PLAYER_SEARCH,
                    TAB_DEFENSIVE_YIELD, abbr_to_pff_team)
from data.draft_board import DEFAULT_SCORING, tier_by_position
from data.transforms import (load_and_merge_data, build_recent_form_rank, build_form_series,
                             score_projected_stats)
from data.rankings import parse_fantasypros_upload, parse_custom_rankings, build_rankings_comparison
from data.utils import calculate_percentile, clean_name_exact, clean_name_for_merge
from data.weekly_projections import build_weekly_projections
from data.odds_weekly import weekly_props, weekly_market_projection
from data.fantasypros_availability import canonical_status, FANTASYPROS_INJURY_PATH
from data.availability_overrides import availability_fingerprint, AVAILABILITY_OVERRIDE_PATH
from data.pass_capacity_allocator import (
    PASS_CAPACITY_TRUSTED_TIER, PASS_CAPACITY_TRUSTED_TIER_RB)
from ui.charts import sparkline_data_uri
from ui.styling import (style_plain_dataframe, df_auto_height, build_column_help_config,
                        get_diverging_color, get_multiplier_color)
from ui.components import (position_group_buttons, apply_position_group, skeleton_loader,
                           import_hint, switch_tab)

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
_PLAYER_DETAIL_CACHE_KEY = 'weekly_rank_player_detail_cache'


def _selected_player_detail(model_meta, detail_config, key):
    """A player's decomposition detail, memoized in st.session_state for the
    rest of THIS session - not just for the current build_weekly_projections
    cache lifetime.

    model_meta['explanations'] itself lives inside a @st.cache_data return
    value, so Streamlit hands back a deep COPY of it on every cache hit (by
    design, so callers can't mutate the canonical cached object) - a plain
    dict lookup there is fresh work every rerun, it just happens to be cheap
    today. Keying by (detail_config, key) here means a player already opened
    once this session is a session_state dict lookup on every later rerun,
    including after the underlying board itself gets rebuilt (TTL expiry, a
    week/scoring/model switch and back, etc.) - explicit request: "once
    loaded...it remains cached for the rest of the session".

    The one thing that MUST bust it is a manual injury override / FantasyPros
    pull for this same week: that rebuilds the board (a new availability
    fingerprint) and can change the decomposition of the injured player and
    everyone whose usage the vacancy redistributes to, while leaving the
    (week, scoring) head of detail_config untouched. detail_config's last
    element is that fingerprint, so a changed feed lands on a fresh cache_key;
    the stale-generation sweep below then drops the pre-injury copies for the
    same board so an affected player is re-resolved from the rebuild rather
    than served the old breakdown. Bug report 2026-08-31: the table restated
    after adding an injured player but the open decompositions did not.
    """
    cache = st.session_state.setdefault(_PLAYER_DETAIL_CACHE_KEY, {})
    cache_key = (detail_config, key)
    if cache_key in cache:
        return cache[cache_key]
    if isinstance(detail_config, tuple) and len(detail_config) > 1:
        head = detail_config[:-1]
        stale = [k for k in cache
                 if isinstance(k[0], tuple) and len(k[0]) == len(detail_config)
                 and k[0][:-1] == head and k[0] != detail_config]
        for k in stale:
            del cache[k]
    detail = model_meta.get('explanations', {}).get(key)
    if detail is not None:
        cache[cache_key] = detail
    return detail


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
    # Anchored at 0, not at this player's own min minus a margin. The
    # underlying band genuinely differs player to player (verified against
    # real data - a Henry-tier back's median sits well above a streaming
    # back's), but a per-player floating window rescales every curve to
    # fill the same physical chart width, so a low-volume player's narrow,
    # low band and a workhorse's wide, high one end up LOOKING like the
    # same generic hump at a glance. A shared zero start lets magnitude
    # show up as curve POSITION/width, not just axis tick labels.
    #
    # 'axis_max' (this position's real widest band this week, set once by
    # build_weekly_projections) fixes the RIGHT edge too - explicit request,
    # since even a shared-zero start still let matplotlib stretch each
    # player's own curve out to fill the same physical width, which is
    # exactly what made two very differently-sized bands look like the same
    # shape at a glance. Falls back to the old per-player computation only
    # for a distribution dict built before this field existed.
    hi = distribution.get('axis_max') or max(player_samples.max(), position_samples.max()) + 1.0
    grid = np.linspace(0.0, hi, 300)
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


def _render_decomposition_navigation(detail):
    """Route out of the Deep Dive into this player's full profile or this
    week's defensive matchup.

    Weekly Rankings is the tab people live in, and until 2026-09-01 it was a
    NAVIGATION DEAD END - the only tab with a player table and no way to
    click through to Player Search, because its own row-select gesture is
    already spent opening this dialog. Putting the jumps here instead of on
    the row keeps that gesture and still makes every player reachable.

    on_click callbacks, not `if st.button(...)`, for the reason every other
    cross-tab jump in this app documents: app.py's st.tabs(key="active_tab")
    is already instantiated by the time a tab body runs, and Streamlit only
    allows assigning to a keyed widget's state from a callback. switch_tab
    also records nav_back_tab, so both destinations offer "Back to Weekly
    Fantasy" once they land.
    """
    player, team = detail.get('player'), detail.get('team')
    opponent, season = detail.get('opponent'), detail.get('season_year')
    if not player:
        return
    left, right = st.columns(2)
    with left:
        kwargs = {'jump_to_player': player}
        if season:
            kwargs['jump_to_year'] = int(season)
        if team:
            kwargs['jump_to_team'] = team
        st.button(f"🔎 {player} in Player Search", key=f"wr_nav_ps_{player}_{team}",
                  width="stretch", help="Full profile: percentiles, game log, career totals.",
                  on_click=switch_tab, args=(TAB_PLAYER_SEARCH,), kwargs=kwargs)
    with right:
        # Guarded: the destination selectbox raises if handed a value that
        # isn't one of its options, so a missing/unknown opponent offers no
        # button rather than a broken jump.
        if opponent and str(opponent) in TEAM_CONFIG:
            st.button(f"🛡️ {opponent} defense breakdown",
                      key=f"wr_nav_dy_{player}_{opponent}", width="stretch",
                      help=f"How {opponent} defends this position, by scheme and alignment.",
                      on_click=switch_tab, args=(TAB_DEFENSIVE_YIELD,),
                      kwargs={'radar_opponent': abbr_to_pff_team(opponent)})


def _render_decomposition_header(detail):
    st.caption(
        f"{detail['position']} · {detail['team']} vs {detail['opponent']} · "
        f"Week {detail['target_week']} using information through Week {detail['as_of_week'] - 1}"
    )
    _render_decomposition_navigation(detail)
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


_ALIGNMENT_MIX_STATS = (
    ('targets', 'targets', 'Tgts'),
    ('receptions', 'receptions', 'Rec'),
    ('receiving_yards', 'yards', 'Rec Yds'),
)


def _render_alignment_mix(detail):
    """Player slot/wide/inline alignment mix + the worked calculation behind
    the defense-allowed multiplier that mix produces - WR/TE only, and only
    when a PFF alignment archive is actually loaded for this run.

    ALWAYS VISIBLE (no expander) as of 2026-08-26, per explicit request -
    this used to be collapsed by default. The table below shows, for each
    alignment this player uses: his own per-game rate at that alignment
    (his overall per-game rate for the stat, split by his own slot/wide/
    inline share - NOT a separately measured per-alignment rate, since a
    single-digit weekly target count per alignment would be noise; see
    _alignment_efficiency's own docstring in data/pff_alignment.py for why
    this app already withholds raw per-alignment efficiency below a minimum
    sample), the defense's allowed-by-alignment multiplier for that
    alignment, and the two multiplied together. The bottom row's multiplier
    is the SHARE-WEIGHTED blend (by this player's own alignment mix, not a
    plain average of the per-alignment multipliers) - the same number
    'Defense multiplier' uses above when this replaces it - so the bottom
    row's Combined value reconciles exactly with that stat's own after-
    defense value in the main table.
    """
    if detail.get('position') not in ('WR', 'TE'):
        return
    role = detail.get('role', {})
    if not role.get('alignment_available'):
        return
    slot = role.get('slot_alignment_rate')
    wide = role.get('wide_alignment_rate')
    inline = role.get('inline_alignment_rate')
    non_slot = role.get('non_slot_alignment_rate')
    if slot is None and wide is None and inline is None:
        return
    st.markdown("**Alignment mix (slot / wide / inline)**")
    weeks = role.get('source_week_count')
    st.caption(
        f"Source: {role.get('source_kind', 'unknown')}, {weeks if weeks else 0} week(s) "
        f"({role.get('source_weeks', '')}), confidence {role.get('alignment_confidence', 0.0):.2f}. "
        f"{role.get('alignment_semantics', '')}."
    )
    blend_mode = role.get('alignment_defense_blend_mode', 'slot_non_slot')
    if blend_mode == 'slot_wide_inline':
        aligns = [('slot', 'Slot', slot), ('wide', 'Wide', wide), ('inline', 'Inline', inline)]
    else:
        aligns = [('slot', 'Slot', slot), ('non_slot', 'Non-slot', non_slot)]
    rates = {key: (float(rate) if rate is not None and pd.notna(rate) else 0.0) for key, _, rate in aligns}
    rate_total = sum(rates.values())
    shares = ({key: (rates[key] / rate_total if rate_total > 0 else 0.0) for key, _, _ in aligns}
              if rate_total > 0 else {key: 0.0 for key, _, _ in aligns})
    if not role.get('alignment_defense_candidate_available'):
        st.caption(
            "Defense-allowed-by-alignment residual is neutral (1.0×) for this matchup: "
            f"{role.get('alignment_defense_reason') or 'insufficient comparison evidence.'}"
        )
        return

    player_label = detail.get('player', 'Player')
    splits_col = f"{player_label}'s Splits"
    table_rows = []
    for key, row_label, rate in aligns:
        row = {splits_col: f"{row_label} — {rate:.1%}" if rate is not None and pd.notna(rate) else f"{row_label} — —"}
        table_rows.append(row)
    totals_row = {splits_col: "Blend (all)"}

    for stat, pff_stat, stat_label in _ALIGNMENT_MIX_STATS:
        # NOTE: the underlying columns are named with pff_alignment.py's own
        # stat vocabulary ('targets'/'receptions'/'yards'), not this app's
        # ('receiving_yards') - use pff_stat for every role.get() lookup
        # here, same fix as _render_defense_allowed_by_alignment's
        # 2026-08-26 bug (found again while building this table: it's easy
        # to repeat since 'targets'/'receptions' happen to be spelled the
        # same in both vocabularies and only 'yards' differs).
        stat_vals = detail.get('stats', {}).get(stat, {})
        blended_rate = stat_vals.get('blended_rate')
        blended_mult = role.get(f'alignment_defense_{pff_stat}_candidate_multiplier')
        # 'Combined' silently folds in the game-context multiplier (script x
        # pace x availability x environment) on top of defense, so it lands on
        # the same scale as the main table's projected value - explicit
        # request 2026-08-29. It still stops SHORT of team-capacity and
        # vacancy (those are team-budget / OUT-teammate adjustments the main
        # table shows as their own deltas), so a small remaining gap to that
        # table's final value is expected, not a bug. The 'Final proj' column
        # this replaced was removed the same day (redundant with the main
        # table, and space-hungry).
        context_mult = (
            float(stat_vals.get('script_multiplier', 1.0))
            * float(stat_vals.get('pace_multiplier', 1.0))
            * float(stat_vals.get('availability_multiplier', 1.0))
            * float(stat_vals.get('environment_multiplier', 1.0))
        )
        per_game_col = f'{stat_label} /Game'
        mult_col = f'{stat_label} Allowed×'
        combined_col = f'{stat_label} Combined'
        for row, (key, row_label, _) in zip(table_rows, aligns):
            ratio = role.get(f'alignment_defense_{pff_stat}_{key}_ratio')
            per_game = (blended_rate * shares[key]) if blended_rate is not None else None
            row[per_game_col] = _fmt_stat(stat, per_game)
            row[mult_col] = f"{ratio:.3f}×" if ratio is not None and pd.notna(ratio) else '—'
            if per_game is not None and ratio is not None and pd.notna(ratio):
                row[combined_col] = _fmt_stat(stat, per_game * ratio * context_mult)
            else:
                row[combined_col] = '—'
        totals_row[per_game_col] = _fmt_stat(stat, blended_rate)
        totals_row[mult_col] = f"{blended_mult:.3f}×" if blended_mult is not None else '—'
        # The bottom row always uses the authoritative season-aggregate
        # blend (blended_rate * blended_mult * context), not a re-sum of the
        # per-row Combined cells above - those can individually show '—' (e.g.
        # one alignment's own comparison evidence is thin) while the aggregate
        # blend that actually scores the projection is still available.
        totals_row[combined_col] = (
            _fmt_stat(stat, blended_rate * blended_mult * context_mult)
            if blended_rate is not None and blended_mult is not None else '—')

    table = pd.DataFrame(table_rows + [totals_row])
    st.dataframe(style_plain_dataframe(table), hide_index=True, width="stretch", height=df_auto_height(len(table)))
    st.caption(
        "/Game = this player's own per-game projected rate for that stat, split by his own alignment "
        "share (not a separately measured per-alignment rate). 'Allowed×' = the opponent's allowed-by-"
        "alignment multiplier for that alignment. 'Combined' = /Game × Allowed× × game context "
        "(script/pace/availability/environment). The bottom row's multiplier is the blend weighted by "
        "this player's own alignment mix (not a plain average) - it REPLACES 'Defense multiplier' in the "
        "table above for this stat when active, not a second adjustment on top of it. 'Combined' matches "
        "the primary table's projected value except for team-capacity and vacancy adjustments, which still "
        "apply downstream there."
    )


def _render_decomposition_primary_table(detail):
    """The at-a-glance table: one row per projected stat, left-to-right in
    build order - Raw Average -> Season average (adj) -> Player Projection
    -> [Weighted average, only in-season] -> Defense multiplier -> Context
    multiplier -> [Team capacity Δ] -> Vacancy Δ -> Projected value - plus a
    trailing "Fantasy points at this stage" row, inside this SAME table,
    showing what the whole stat line was worth at each checkpoint above.

    A WR/TE 'Defense multiplier' IS the alignment-specific number whenever
    alignment evidence is available (see alignment_player_factor in
    data/weekly_projections.py - it replaces the broad matchup outright, not
    a separate factor multiplied alongside it) - a redundant "Alignment
    residual" column that duplicated this same number was removed
    2026-08-26. The always-visible "Alignment mix" section below this table
    breaks it down by slot/wide/inline for whoever wants that detail.

    RESTRUCTURED 2026-08-25 (second pass) per explicit request to walk the
    columns in build order and fold the points readout into the table
    itself rather than a second one underneath.

    Raw Average = `raw_prior_rate` - this player's own plain per-game
    history, no adjustment of any kind (see the caption below for exactly
    which seasons/weights fed it). Season average (adj) = the SAME history
    after removing each past game's own opponent's specific strength
    (`defense_adjusted_prior_rate` - literally the average of the per-game
    "Defense-adj" values the Deep Dive tab already shows) - about defenses
    ALREADY PLAYED, not the upcoming one, so it is deliberately a different
    number from 'Defense multiplier' further right. Player Projection =
    `blended_rate`, that history after role/snap-share normalization and
    (once real current-season games exist) blending with this season's own
    rate - not yet adjusted for the upcoming matchup or game context.
    'Weighted average' (`current_rate`) is only ever non-zero once real
    current-season games exist, so it's hidden entirely at cold start
    instead of printing a column of dashes."""
    stats = detail.get('stats', {})
    if not stats:
        st.caption("No projected stat line for this player.")
        return
    season_year = detail.get('season_year')
    scoring_mode = detail.get('scoring_mode') or 'Full PPR'
    has_current_season_data = any(
        (values.get('current_games') or 0.0) > 0 for values in stats.values())
    rows = []
    stage_totals = {stage: {} for stage in (
        'raw_average', 'season_adj', 'player_projection', 'after_defense',
        'after_context', 'after_capacity', 'after_vacancy', 'final')}
    raw_average_notes = set()
    prior2_weights = []
    context_ingredients = None
    for stat, values in stats.items():
        context_mult = (
            values.get('script_multiplier', 1.0) * values.get('pace_multiplier', 1.0)
            * values.get('availability_multiplier', 1.0) * values.get('environment_multiplier', 1.0)
        )
        if context_ingredients is None:
            # Player/game-level, not per-stat - every stat's dict carries
            # the same 4 numbers, so the first one seen is representative;
            # feeds the standalone "thin" table right after this one.
            context_ingredients = {
                'script': float(values.get('script_multiplier', 1.0)),
                'pace': float(values.get('pace_multiplier', 1.0)),
                'availability': float(values.get('availability_multiplier', 1.0)),
                'environment': float(values.get('environment_multiplier', 1.0)),
                'combined': float(context_mult),
            }
        matchup_mult = float(values.get('matchup_multiplier', 1.0))
        cur_w, cur_games = values.get('current_weight'), values.get('current_games') or 0.0
        if cur_w is None:
            season_note = ''
        elif not cur_games:
            season_note = (f"100% {season_year - 1} season" if season_year else "100% prior season")
        else:
            season_note = (f"{cur_w:.0%} {season_year} ({cur_games:.0f} gm) + {1 - cur_w:.0%} prior"
                          if season_year else f"{cur_w:.0%} this season")
        if season_note:
            raw_average_notes.add(season_note)
        # Independent axis from season_note above - THAT note is current
        # (2026) vs. prior (2025) season weight; this is how much of the
        # "prior" share shown there is itself 2025 blended with 2024. Added
        # 2026-08-25: without this, "100% 2025 season" reads as "0% 2024",
        # when 2024 is very likely blended in underneath it (see
        # data.weekly_projections._blend_with_prior2). NaN for a TD-type
        # stat (those use a separate two-year path) or a player with no
        # usable 2024 read.
        stat_prior2_weight = values.get('prior2_weight')
        if stat_prior2_weight is not None and pd.notna(stat_prior2_weight):
            prior2_weights.append(float(stat_prior2_weight))
        vacancy_raw = float(values.get('vacancy_delta', 0.0) or 0.0)
        # Split 2026-08-24: this used to be one 'vacancy_delta' number that
        # silently conflated two unrelated mechanisms - an OUT teammate's
        # volume actually being redistributed, and apply_pass_capacity_
        # conservation's team-target-budget fit, which runs on every V2 board
        # with 'v2_pass_capacity' on regardless of injuries and can visibly
        # squeeze a low-usage receiver (e.g. a bruising RB) even with nobody
        # out. See data/weekly_projections.py's post_capacity_snapshot
        # comment for the full story - this was a real mislabeling bug.
        capacity_raw = float(values.get('pass_capacity_delta', 0.0) or 0.0)

        raw_avg_val = values.get('raw_prior_rate')
        season_adj_val = values.get('defense_adjusted_prior_rate')
        player_proj_val = values.get('blended_rate')
        weighted_avg_val = values.get('current_rate')
        pre_vacancy = values.get('pre_vacancy_projection')
        after_defense_val = (player_proj_val * matchup_mult
                             if player_proj_val is not None else None)
        after_context_val = pre_vacancy if pre_vacancy is not None else (
            after_defense_val * context_mult if after_defense_val is not None else None)
        after_capacity_val = (after_context_val + capacity_raw
                              if after_context_val is not None else None)
        after_vacancy_val = (after_capacity_val + vacancy_raw
                             if after_capacity_val is not None else None)
        final_val = values.get('final_projection', values.get('projection'))
        for stage, val in (
            ('raw_average', raw_avg_val), ('season_adj', season_adj_val),
            ('player_projection', player_proj_val), ('after_defense', after_defense_val),
            ('after_context', after_context_val), ('after_capacity', after_capacity_val),
            ('after_vacancy', after_vacancy_val), ('final', final_val),
        ):
            if val is not None and pd.notna(val):
                stage_totals[stage][stat] = float(val)

        rows.append({
            'Stat': stat.replace('_', ' ').title(),
            'Raw Average': _fmt_stat(stat, raw_avg_val),
            'Season average (adj)': _fmt_stat(stat, season_adj_val),
            'Player Projection': _fmt_stat(stat, player_proj_val),
            'Weighted average': _fmt_stat(stat, weighted_avg_val),
            # Defense multiplier already IS the alignment-specific number for
            # a WR/TE row with available alignment evidence (it replaces the
            # broad role/defense matchup outright since the 2026-08-26
            # redesign - see alignment_player_factor in
            # data/weekly_projections.py) - a separate "Alignment residual"
            # column used to show here too, but it was always identical to
            # this one and was removed 2026-08-26 as pure redundancy. The
            # always-visible "Alignment mix" section below breaks this same
            # number down by slot/wide/inline for whoever wants that detail.
            'Defense multiplier': f"{matchup_mult:.3f}×",
            'Context multiplier': f"{context_mult:.3f}×",
            'Team capacity Δ': _fmt_stat(stat, capacity_raw, signed=True),
            'Vacancy Δ': _fmt_stat(stat, vacancy_raw, signed=True),
            'Projected value': _fmt_stat(stat, final_val),
            '_vacancy_raw': vacancy_raw,
            '_capacity_raw': capacity_raw,
            '_defense_mult': matchup_mult,
            '_context_mult': context_mult,
        })
    table = pd.DataFrame(rows)
    vacancy_vals = table['_vacancy_raw'].tolist()
    capacity_vals = table['_capacity_raw'].tolist()
    show_capacity = any(abs(v) > 0.0005 for v in capacity_vals)
    display_cols = ['Stat', 'Raw Average', 'Season average (adj)', 'Player Projection']
    if has_current_season_data:
        display_cols.append('Weighted average')
    display_cols.append('Defense multiplier')
    display_cols.append('Context multiplier')
    if show_capacity:
        display_cols.append('Team capacity Δ')
    display_cols.append('Vacancy Δ')
    display_cols.append('Projected value')

    # Trailing "Fantasy points at this stage" row, inside this same table
    # per explicit request ("not a separate table... it should be within
    # the table"). Only columns that represent an actual running LEVEL of
    # the stat line (not a bare multiplier/ingredient column) get a number -
    # Weighted average is a side ingredient already folded into Player
    # Projection, not a sequential checkpoint, so it stays blank here to
    # avoid double-counting.
    stage_by_col = {
        'Raw Average': 'raw_average', 'Season average (adj)': 'season_adj',
        'Player Projection': 'player_projection', 'Defense multiplier': 'after_defense',
        'Context multiplier': 'after_context', 'Team capacity Δ': 'after_capacity',
        'Vacancy Δ': 'after_vacancy', 'Projected value': 'final',
    }
    points_row = {'Stat': 'Fantasy points at this stage'}
    for col in display_cols[1:]:
        stage_key = stage_by_col.get(col)
        stat_dict = stage_totals.get(stage_key, {}) if stage_key else {}
        points_row[col] = (f"{max(0.0, score_projected_stats(stat_dict, scoring_mode)):.1f} pts"
                           if stat_dict else '—')
    display_table = pd.concat([table[display_cols], pd.DataFrame([points_row])], ignore_index=True)
    points_row_idx = len(display_table) - 1

    # Local Styler, not a style_plain_dataframe extension - that helper is
    # shared by every other table in the app, and this coloring rule
    # (multiplier columns centered on 1.0) is specific to this one table.
    # Colors read off the hidden numeric _*  columns (kept alongside the
    # already-formatted display strings above) rather than the display
    # column itself, since every cell in the table is now pre-formatted
    # text - and the trailing points row has no multiplier/delta value at
    # all, so it falls through to '' (no color) via the pd.notna guard.
    def _style_column(col):
        out = []
        for i in range(len(display_table)):
            if i == points_row_idx:
                out.append('font-weight:bold; border-top:2px solid rgba(128,128,128,0.5);')
                continue
            if col == 'Defense multiplier':
                v = table['_defense_mult'].iloc[i]
            elif col == 'Context multiplier':
                v = table['_context_mult'].iloc[i]
            elif col == 'Vacancy Δ':
                v = vacancy_vals[i]
                out.append(f'background-color:{get_diverging_color(v, 2.0)}; color:#ffffff; font-weight:bold;')
                continue
            elif col == 'Team capacity Δ':
                v = capacity_vals[i]
                out.append(f'background-color:{get_diverging_color(v, 2.0)}; color:#ffffff; font-weight:bold;')
                continue
            else:
                out.append('')
                continue
            out.append(f'background-color:{get_multiplier_color(v)}; color:#ffffff; font-weight:bold;'
                       if pd.notna(v) else '')
        return out

    style_grid = pd.DataFrame({col: _style_column(col) for col in display_cols})
    styler = display_table.style.apply(lambda _: style_grid, axis=None)
    st.dataframe(styler, hide_index=True, width="stretch", height=df_auto_height(len(display_table)))

    note_text = " / ".join(sorted(raw_average_notes)) if raw_average_notes else "prior-season history"
    st.caption(
        f"Raw Average = this player's own plain per-game history, no adjustment ({note_text}). "
        "Season average (adj) = that same history with each past game's OWN opponent's strength removed "
        "(what he'd average against a neutral defense) - about defenses already played, not the upcoming "
        "one. Player Projection = Raw Average after role/snap-share normalization (and, once real "
        "current-season games exist, blended with this season's own rate)."
    )
    if prior2_weights:
        lo, hi = min(prior2_weights), max(prior2_weights)
        two_years_back = (season_year - 2) if season_year else "two seasons back"
        weight_text = f"~{lo:.0%}" if abs(hi - lo) < 0.01 else f"{lo:.0%}-{hi:.0%}"
        st.caption(
            f"That \"prior season\" figure above is itself already blended with {two_years_back} "
            f"({weight_text} weight on this player's volume/yardage stats, more if 2025 was thin or a down "
            "year for him, less if 2025 was clearly the better read) - a separate axis from the current-vs-"
            "prior-season split noted above, not visible in it. TD-type stats use a different two-year "
            "blend (see the Role/audit/data sources tab) and aren't included in this weight."
        )
    if 'Team capacity Δ' in display_cols:
        st.caption(
            "Team capacity Δ: this team's RB/WR/TE targets refit to a realistic pass-attempt budget "
            "(top 8 pass catchers by current projection keep their own value; the rest share what's left). "
            "Runs on every V2 board with nobody hurt - separate from Vacancy Δ below, which is only an "
            "actual OUT teammate's volume moving. Both used to be shown as one 'Vacancy Δ' number; split "
            "2026-08-24 because that conflation was misleading a low-usage receiver's real story."
        )
    st.caption(
        f"Fantasy points row: same scoring rule as the board ({scoring_mode}), applied to the whole stat "
        "line frozen at each checkpoint above - a running total, not a per-stat breakdown. 'Projected "
        "value' should match this player's Raw Model Proj Pts; a mismatch would mean a stat is missing "
        "from the table above, not a scoring bug."
    )

    if context_ingredients is not None:
        _render_context_multiplier_table(context_ingredients)


def _render_context_multiplier_table(ingredients):
    """Standalone "thin" table of the actual numbers behind the Context
    multiplier column - added 2026-08-25 per explicit request ("right in
    that spot include the actual numbers that are being used"). One row,
    since script/pace/availability/environment are player/game-level, the
    same for every stat in the line above."""
    st.markdown("**Context multiplier - what it's made of**")
    thin = pd.DataFrame([{
        'Game script ×': f"{ingredients['script']:.3f}",
        'Pace ×': f"{ingredients['pace']:.3f}",
        'Availability ×': f"{ingredients['availability']:.3f}",
        'Vegas environment ×': f"{ingredients['environment']:.3f}",
        'Combined (Context multiplier) ×': f"{ingredients['combined']:.3f}",
    }])
    st.dataframe(style_plain_dataframe(thin), hide_index=True, width="stretch", height=df_auto_height(1))
    st.caption("Context multiplier = Game script × Pace × Availability × Vegas-implied game environment.")


def _team_cell_style(team_code):
    color = TEAM_CONFIG.get(str(team_code).strip().upper(), {}).get('color')
    return f'background-color:{color}; color:#ffffff; font-weight:bold;' if color else ''


def _style_team_column(df, col):
    """A local Styler coloring one team-code column by TEAM_CONFIG - same
    convention as the Team column elsewhere and the sticky game log's
    opponent cell (ui.styling.render_game_log_html_table), just applied to
    a plain st.dataframe table here instead of that raw-HTML one."""
    grid = pd.DataFrame('', index=df.index, columns=df.columns)
    grid[col] = [_team_cell_style(v) for v in df[col]]
    return df.style.apply(lambda _: grid, axis=None)


def _append_average_row(df, label_col, label, avg_cols, stat=None, avg_mask=None, ratio_cols=None):
    """One trailing row averaging each of ``avg_cols`` - explicit request so
    a Deep Dive table's raw/defense-adjusted (or allowed/baseline) columns
    show their own column average without the reader doing that math.
    ``stat`` (the raw stat key, e.g. 'rushing_tds') picks the same 1-vs-2
    decimal formatting _fmt_stat already uses for that stat's real values,
    so the AVG row's precision matches the rows above it.

    ``ratio_cols`` (optional) are averaged the same way but formatted as a
    plain 3-decimal x-suffixed multiplier ("1.234x"), matching how each
    row's own ratio cell is already formatted - _fmt_stat's stat-specific
    rounding is for a raw stat value, not a ratio, and would both mis-round
    it and fail to parse the trailing "x" back out. Added 2026-08-27 per
    explicit request: the allowed/offense-avg columns already got an AVG row,
    the ratio column next to them silently didn't.

    ``avg_mask`` (optional, same length/index as ``df``) restricts the mean
    to the True rows only - every row still prints in the table above (an
    excluded game is "marked, not hidden", per the caption below this call),
    but a game already labelled "Excluded" in the Included column must not
    also silently pull its own row's average down. Found real 2026-08-25 on
    Jayden Daniels: the unmasked AVG mixed in his 2 QB-split relief games and
    landed at 180.3, vs. 205.6 across only his 5 full-role starts."""
    row = {c: '' for c in df.columns}
    row[label_col] = label
    avg_df = df if avg_mask is None else df[np.asarray(avg_mask, dtype=bool)]
    for col in avg_cols:
        numeric = pd.to_numeric(avg_df[col].astype(str).str.replace('+', '', regex=False), errors='coerce')
        row[col] = _fmt_stat(stat, float(numeric.mean())) if numeric.notna().any() else '—'
    for col in (ratio_cols or ()):
        numeric = pd.to_numeric(avg_df[col].astype(str).str.replace('×', '', regex=False), errors='coerce')
        row[col] = f"{numeric.mean():.3f}×" if numeric.notna().any() else '—'
    # The label column (e.g. Week) is a real int column above this row and
    # a literal "AVG" string on it - cast to str BEFORE concatenating so
    # pandas keeps one consistent dtype throughout, not a numeric column
    # that only turns to mixed-type object once "AVG" lands in it. Arrow
    # serialization tolerates the latter (with a fallback + a noisy
    # traceback logged on every single render) but never actually needs to.
    out = df.copy()
    out[label_col] = out[label_col].astype(str)
    return pd.concat([out, pd.DataFrame([row])], ignore_index=True)


def _render_defense_allowed_by_alignment(detail, stat, label, log):
    """Slot/wide/inline breakout of the SAME defense-allowed evidence the
    whole-position weekly table just above already shows in aggregate -
    added 2026-08-25 per explicit request ("there should be 'allowed' logs
    for each alignment... it should look exactly as the [whole-position]
    defense - what it's allowed to ___, by week"). WR/TE only (the only
    positions PFF alignment data covers).

    ``log`` is the SEASON-SPECIFIC list the caller already selected via the
    Deep Dive tab's own season radio (`defense_alignment_weekly_log_by_season`
    in data/weekly_projections.py) - added 2026-08-26 to fix this table
    silently never changing when a different season (e.g. 2024) was picked,
    since it used to always read the single season loaded for SCORING
    regardless of the selector.

    This is the RAW pre-shrinkage evidence, not the season-aggregate
    candidate_multiplier shown in the Alignment mix section above - that
    number is heavily shrunk toward 1.0 for any team/stat/alignment combo
    with a thin sample (see data/pff_alignment.py's
    aggregate_alignment_defense_profiles), which is why it can read close to
    1.000 even when the raw per-week numbers below show real spread. The
    Non-slot column was dropped 2026-08-26 (per explicit request) - it only
    exists internally to let the wide/inline split be derived; once that
    split exists, non-slot itself isn't a real alignment a player lines up
    in, unlike slot/wide/inline.
    """
    if detail.get('position') not in ('WR', 'TE'):
        return
    # The log's own 'stat' values come straight from pff_alignment.py's
    # ALIGNMENT_DEFENSE_STATS naming ('targets'/'receptions'/'yards'), not
    # this app's stat keys ('receiving_yards') - targets/receptions happen
    # to be spelled the same in both places, which is exactly why this table
    # silently rendered for those two stats but never for receiving_yards
    # until this mapping was added (found 2026-08-26: matches
    # ALIGNMENT_SCORING_STAT_MAP in data/weekly_projections.py).
    log_stat = {'receiving_yards': 'yards'}.get(stat, stat)
    rows = [r for r in log if r.get('stat') == log_stat]
    if not rows:
        return
    adf = pd.DataFrame(rows)
    index_cols = ['source_week', 'offense_team']
    pivot_obs = adf.pivot_table(index=index_cols, columns='alignment', values='observed_value', aggfunc='mean')
    pivot_exp = adf.pivot_table(index=index_cols, columns='alignment', values='_expected_value', aggfunc='mean')
    week_weight = adf.drop_duplicates('source_week').set_index('source_week')['_recency_weight']
    alignment_cols = [c for c in ('slot', 'wide', 'inline') if c in pivot_obs.columns]
    if not alignment_cols:
        return
    pivot_obs = pivot_obs.reset_index().sort_values('source_week')
    pivot_exp = pivot_exp.reset_index()
    merged = pivot_obs.merge(pivot_exp, on=index_cols, how='left', suffixes=('', '_exp'))
    display_data = {
        'Week': merged['source_week'].astype(int),
        'Offense faced': merged['offense_team'],
    }
    avg_cols = []
    ratio_cols = []
    for col in alignment_cols:
        title = col.title()
        allowed_col, avg_col, ratio_col = f'{title} Allowed', f'{title} Offense Avg', f'{title} Ratio'
        observed = merged[col]
        expected = merged.get(f'{col}_exp', pd.Series(float('nan'), index=merged.index))
        display_data[allowed_col] = [_fmt_stat(stat, v) for v in observed]
        display_data[avg_col] = [_fmt_stat(stat, v) for v in expected]
        ratio = observed / expected.where(expected > 0.0)
        display_data[ratio_col] = [f"{v:.3f}×" if pd.notna(v) else '—' for v in ratio]
        avg_cols.extend([allowed_col, avg_col])
        ratio_cols.append(ratio_col)
    display_data['Recency weight'] = [
        f"{week_weight.get(w, float('nan')):.2f}" if pd.notna(week_weight.get(w, float('nan'))) else '—'
        for w in merged['source_week']
    ]
    display = pd.DataFrame(display_data)
    st.markdown(f"**{detail['opponent']} defense — what it's allowed to {label.lower()}, by alignment, by week**")
    st.dataframe(
        _style_team_column(_append_average_row(display, 'Week', 'AVG', avg_cols, stat=stat, ratio_cols=ratio_cols), 'Offense faced'),
        hide_index=True, width="stretch", height=df_auto_height(len(display) + 1))
    st.caption(
        "Raw per-week evidence, not the shrunk candidate multiplier shown in the Alignment mix section above "
        "- a thin per-alignment sample there gets pulled most of the way back to a neutral 1.000× until more "
        "weeks of PFF data exist, even when the per-week Ratio here shows real spread. 'Offense Avg' is that "
        "SAME offense's own average in its other games that season (not this defense) - the baseline 'Ratio' "
        "is compared against, same convention as the non-alignment table above. 'Recency weight' matches this "
        "app's usual within-season decay for reference; the alignment aggregation itself does not currently "
        "apply it (it weights every included game equally, using game-count shrinkage instead - see "
        "aggregate_alignment_defense_profiles's own shrinkage_weight)."
    )


def _render_stat_deep_dive(detail, stat, game_log_by_season, defense_log_by_season, default_year=None):
    """Per-stat tab, including its OWN season selector - explicit request,
    since a single season control shared across every stat tab meant
    switching season on one stat (e.g. to see last year's rushing yards)
    silently moved every OTHER stat's tab to that season too."""
    values = detail.get('stats', {}).get(stat, {})
    label = stat.replace('_', ' ').title()
    season_years = sorted(set(game_log_by_season) | set(defense_log_by_season), reverse=True)
    if len(season_years) > 1:
        current_year = detail.get('season_year')
        season_labels = {
            yr: (f"{yr} (this season)" if yr == current_year
                 else f"{yr} (last season)" if yr == current_year - 1
                 else f"{yr} (two years back)")
            for yr in season_years
        }
        default_year = default_year if default_year in season_years else season_years[0]
        season_year = st.radio(
            "Season", season_years, index=season_years.index(default_year),
            format_func=lambda yr: season_labels[yr], horizontal=True,
            key=(f"weekly_rank_deepdive_season_{stat}_{detail['player']}_{detail['team']}_"
                f"{detail['target_week']}_{detail['as_of_week']}"),
        )
    else:
        season_year = season_years[0] if season_years else detail.get('season_year')
    game_log = game_log_by_season.get(season_year) or []
    defense_log = defense_log_by_season.get(season_year) or []
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
        team_score = pd.to_numeric(pgl.get('_team_score', pd.Series(dtype=float)), errors='coerce')
        opp_score = pd.to_numeric(pgl.get('_opp_score', pd.Series(dtype=float)), errors='coerce')
        score = [f"{int(t)}-{int(o)}" if pd.notna(t) and pd.notna(o) else '—'
                for t, o in zip(team_score, opp_score)]
        result = pgl.get('_result', pd.Series('', index=pgl.index)).fillna('').replace('', '—')
        pgl_display = pd.DataFrame({
            'Week': pgl['_week_num'].astype(int),
            'Opponent': pgl.get('opponent_team', ''),
            'Score': score,
            'Result': result,
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
        pgl_with_avg = _append_average_row(
            pgl_display, 'Week', 'AVG', [f'Raw {label}', f'Defense-adj {label}'], stat=stat,
            avg_mask=eligible)
        st.dataframe(
            _style_team_column(pgl_with_avg, 'Opponent'), hide_index=True,
            width="stretch", height=df_auto_height(len(pgl_with_avg)))
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
        dgl_with_avg = _append_average_row(
            dgl_display, 'Week', 'AVG', [f'Allowed {label}', "That offense's own average"], stat=stat)
        st.dataframe(
            _style_team_column(dgl_with_avg, 'Offense faced'), hide_index=True,
            width="stretch", height=df_auto_height(len(dgl_with_avg)))
        st.caption(
            f"Defense multiplier ({values.get('matchup_multiplier', 1.0):.3f}×): each week above compares what "
            f"{detail['opponent']} allowed to that offense's own average, recency-weighted, then re-centered "
            f"to a league average of 1.0 — this table is that comparison's raw ingredients."
        )
    else:
        st.caption(f"No {label.lower()} matchup evidence available for {detail['opponent']} yet.")

    _render_defense_allowed_by_alignment(
        detail, stat, label,
        detail.get('defense_alignment_weekly_log_by_season', {}).get(season_year, []))

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
        st.dataframe(style_plain_dataframe(pd.DataFrame(rows)), hide_index=True, width="stretch", height=df_auto_height(len(rows)))
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
    # alignment_defense_scoring_active is a dead lower-level flag (always
    # False - see aggregate_alignment_defense_profiles's own hardcoded
    # 'scoring_active': False, a foundation field never actually flipped).
    # candidate_available is the real "is this replacing the general defense
    # multiplier right now" signal since the 2026-08-26 redesign - see the
    # np.where in weekly_projections.py's alignment_player_factor block.
    if role.get('alignment_defense_candidate_available'):
        st.caption("PFF alignment-defense residual: ACTIVE (v2_pff_alignment_matchup) - this player's own "
                  "allowed-by-alignment mix is replacing the general defense/role matchup for this stat, "
                  "not adding to it. See 'Defense multiplier' in the table above and the always-visible "
                  "Alignment mix section for the full breakdown.")
    elif role.get('alignment_available'):
        st.caption("PFF alignment-defense residual: no comparison evidence for this defense/stat yet - "
                  "falling back to the general defense/role matchup.")

    evidence = detail.get('alignment_scheme_evidence', {})
    if evidence.get('player_scheme_available'):
        man, zone = evidence.get('player_man_route_share'), evidence.get('player_zone_route_share')
        if man is not None and pd.notna(man):
            st.markdown("**PFF man/zone role**")
            st.caption(f"Man: {float(man):.0%}"
                      + (f" · Zone: {float(zone):.0%}" if zone is not None and pd.notna(zone) else ""))
    defense_bits = []
    slot_applied = bool(role.get('alignment_defense_candidate_available'))
    if evidence.get('defense_alignment_candidate_available'):
        mult = evidence.get('defense_slot_candidate_multiplier')
        if mult is not None and pd.notna(mult):
            defense_bits.append(f"slot-weighted {float(mult):.3f}×" + (" (applied)" if slot_applied else " (candidate)"))
    if evidence.get('defense_scheme_candidate_available'):
        mult = evidence.get('defense_man_candidate_multiplier')
        if mult is not None and pd.notna(mult):
            defense_bits.append(f"man-weighted {float(mult):.3f}× (candidate)")
    if defense_bits:
        st.markdown("**Opponent alignment/scheme vulnerability**")
        st.caption(
            f"{detail['opponent']}'s defense, back-calculated from every offense it actually faced this "
            f"season — " + " · ".join(defense_bits) + ". "
            + ("'Applied' is already folded into this stat's Defense multiplier above. "
               if slot_applied else "")
            + "'Candidate' entries remain audit-only (man/zone scheme is dormant everywhere)."
        )


def _render_pipeline_diagnostics(model_meta):
    """Run-level data-pipeline notes - NOT about any one player.

    ADDED 2026-08-24 per explicit request. This content used to live inside
    EVERY player's 'Role, audit & data sources' expander via
    _render_contract_field, even though almost all of it - the Ourlads
    import's warnings list, the availability-resolution warnings, QB1
    selection state, role-segment counts, the raw pace/injury/market_script
    mode tags - is identical for all ~900 players in a run: it describes the
    pipeline that built the whole board, not the one row someone opened.
    Shown once here instead, so the per-player panel can stay about that
    player. Reads model_meta['source_contract'], the same dict each
    player's data_contract is a shallow copy of.

    MOVED 2026-08-24 into the last tab of the "Live data pulls" expander
    (previously its own standalone expander right below it) - per explicit
    request, so the page carries one accordion for run-level info instead of
    two back-to-back. No longer opens its own st.expander: the caller
    already provides the surrounding tab."""
    contract = (model_meta or {}).get('source_contract') or {}
    if not contract:
        st.caption("No pipeline notes for this run.")
        return
    st.caption("Applies to every player below - not specific to any one of them.")

    backtest_bits = []
    if contract.get('pace') == 'weekly_box_score_proxy':
        backtest_bits.append("pace uses a same-week box-score proxy")
    if contract.get('injury') == 'disabled_historical':
        backtest_bits.append("injury discount disabled")
    if contract.get('market_script') == 'disabled_historical':
        backtest_bits.append("market script disabled")
    if backtest_bits:
        st.warning("Backtest mode: " + "; ".join(backtest_bits) +
                   " - this run is validating a past week, not projecting a live one.")
    if contract.get('prior_defense_recency'):
        st.caption(f"Prior-season defense recency weighting: {contract['prior_defense_recency']}.")

    ourlads = contract.get('ourlads_preseason_depth_chart')
    if isinstance(ourlads, dict) and ourlads:
        st.markdown("**Ourlads preseason depth chart import**")
        st.caption(
            f"{ourlads.get('status', 'unknown')} — "
            f"{ourlads.get('matched_teams', 0)}/{ourlads.get('snapshot_teams', 0)} teams matched, "
            f"{ourlads.get('matched_players', 0)} players, {ourlads.get('matched_qb1s', 0)} QB1s."
        )
        overlay = ourlads.get('roster_overlay_changes') or []
        if overlay:
            with st.expander(f"Roster overlay changes ({len(overlay)})", expanded=False):
                st.dataframe(style_plain_dataframe(pd.DataFrame(overlay)), hide_index=True, width="stretch")
        issues = ourlads.get('warnings') or []
        if issues:
            with st.expander(f"Depth-chart warnings ({len(issues)})", expanded=False):
                for issue in issues:
                    st.caption(f"• {issue}")

    availability = contract.get('availability')
    if isinstance(availability, dict) and availability:
        st.markdown("**Availability resolution**")
        st.caption(
            f"{availability.get('policy', 'unknown policy')} — "
            f"{availability.get('resolved_profiles', 0)} profiles resolved, "
            f"{availability.get('manual_overrides', 0)} manual override(s) active this week."
        )
        issues = availability.get('warnings') or []
        if issues:
            with st.expander(f"Unmatched availability sources ({len(issues)})", expanded=False):
                for issue in issues:
                    st.caption(f"• {issue}")

    if contract.get('qb1_selection_required_teams'):
        st.warning(
            "QB1 selection required for: " + ", ".join(contract['qb1_selection_required_teams']) +
            " - those teams' QB rooms receive no normal QB volume until one player is selected."
        )

    segments = contract.get('rb_role_segments')
    if isinstance(segments, dict) and segments:
        st.caption(
            f"RB role-segment detector: {segments.get('clear_interrupted_returners', 0)} players credited "
            f"with a clear interrupted/returned or season-ending role, "
            f"{segments.get('teammate_context_rows', 0)} teammate-context rows built."
        )

    if contract.get('qb_starter_source'):
        st.caption(f"QB1 selection source: {contract['qb_starter_source']}.")
    if contract.get('preseason_skill_role_policy'):
        st.caption(f"Preseason role policy: {contract['preseason_skill_role_policy']}.")
    if contract.get('partial_game_history_filter'):
        st.caption(f"Partial-game history filter: {contract['partial_game_history_filter']}.")
    exclusions = contract.get('partial_game_history_exclusions')
    if isinstance(exclusions, dict) and exclusions:
        # Confirms the SAME filter above runs on all three inputs, not just
        # the current/prior pair - explicit request 2026-08-25 ("make sure
        # the 2024 season stats are being checked... as the 2025 season
        # currently does"). two_year_prior's count here is exactly
        # (~prior2_annotated['_player_history_eligible']).sum() from
        # weekly_projections.py - the same function call as prior_season's.
        st.dataframe(pd.DataFrame([{
            'This season': exclusions.get('current_season', 0),
            'Prior season': exclusions.get('prior_season', 0),
            'Two years back': exclusions.get('two_year_prior', 0),
        }]), hide_index=True, width="stretch", height=df_auto_height(1))
        st.caption(
            "Games excluded from rate evidence by the same partial-game filter above, per season - all "
            "three seasons run through the identical eligibility check, so a 2024 QB-split or blowout-rest "
            "game is screened out exactly like a 2025 one would be."
        )


def _render_role_confidence_table(role):
    """What role_confidence (data/weekly_projections.py's `_role_confidence`)
    actually read to produce its 0-1 number, and what that number then does
    downstream - added 2026-08-25 per explicit request ("I have no idea
    what is going into this calculation"). The ingredients
    (role_confidence_recent_snap_pct/games_sampled/route_rate/method) are
    audit-only companions computed by `_role_confidence_detail`, a
    deliberate near-duplicate of the real function so a bug here can never
    change a scored value - see that function's own docstring."""
    confidence = role.get('role_confidence')
    if confidence is None or pd.isna(confidence):
        return
    snap_pct = role.get('role_confidence_recent_snap_pct')
    games_sampled = role.get('role_confidence_games_sampled')
    route_rate = role.get('role_confidence_route_rate')
    method = role.get('role_confidence_method') or 'unknown'
    st.markdown("**Role confidence — what fed this number**")
    ingredient_rows = [{
        'Ingredient': 'Recent snap share',
        'Value': f"{float(snap_pct):.0%}" if snap_pct is not None and pd.notna(snap_pct) else '—',
        'Detail': (f"mean of the last {int(games_sampled)} played game(s) before this week"
                  if games_sampled is not None and pd.notna(games_sampled) and games_sampled > 0
                  else 'no played games available'),
    }]
    if route_rate is not None and pd.notna(route_rate):
        ingredient_rows.append({
            'Ingredient': 'PFF route rate',
            'Value': f"{float(route_rate):.0%}",
            'Detail': 'season-to-date route participation (WR/TE only)',
        })
    ingredient_rows.append({
        'Ingredient': 'Role confidence (used below)',
        'Value': f"{float(confidence):.0%}",
        'Detail': method,
    })
    st.dataframe(style_plain_dataframe(pd.DataFrame(ingredient_rows)), hide_index=True, width="stretch",
                height=df_auto_height(len(ingredient_rows)))
    st.caption(
        "How it's used: role confidence shrinks how fast a stat trusts THIS player's own rate over the "
        "positional baseline. Higher confidence (closer to 100%) means his own history is trusted sooner "
        "as real current-season games accumulate; lower confidence (closer to 0%) leans harder on the "
        "position-average fallback for longer. It has no effect at cold start itself (no current-season "
        "games exist yet to blend in) - its effect shows up once the season is underway."
    )


def _render_pass_capacity_room(detail):
    """Who else shares this player's team-capacity budget, and what the
    conservation pass did to each of them - added 2026-08-25 per explicit
    request ("tell what the issue is... and show how each player in that
    room is affected"). `pass_capacity_ledger` (team-level totals) already
    existed; `pass_capacity_room` (the per-player before/after list) is new
    - see apply_pass_capacity_conservation's player_detail .attrs."""
    ledger = detail.get('pass_capacity_ledger', [])
    room = detail.get('pass_capacity_room', [])
    if not ledger and not room:
        return
    st.markdown("**Team pass-capacity room**")
    for entry in ledger:
        group = entry.get('position_group', '')
        capacity = entry.get('capacity')
        trusted = entry.get('trusted_claim')
        tail = entry.get('tail_claim')
        if capacity is None:
            st.caption(f"{group}: {entry.get('reason', 'no capacity signal')}")
            continue
        claimed = (trusted or 0.0) + (tail or 0.0)
        over_budget = claimed > capacity + 0.01
        issue = (f"claimed {claimed:.1f} targets against a {capacity:.1f} budget - "
                f"{'over' if over_budget else 'within'} budget by {abs(claimed - capacity):.1f}")
        (st.warning if over_budget else st.caption)(f"{group} room: {issue}. {entry.get('reason', '')}")
    if room:
        room_df = pd.DataFrame(room)
        room_df['Δ targets'] = (room_df['targets_after'] - room_df['targets_before']).round(3)
        display = room_df.rename(columns={
            'player': 'Player', 'position': 'Pos', 'tier': 'Tier',
            'targets_before': 'Targets before', 'targets_after': 'Targets after',
        })[['Player', 'Pos', 'Tier', 'Targets before', 'Targets after', 'Δ targets']]
        this_player = detail.get('player')
        style_grid = pd.DataFrame('', index=display.index, columns=display.columns)
        highlight_rows = room_df['player'].astype(str).eq(str(this_player))
        style_grid.loc[highlight_rows, :] = 'font-weight:bold; background-color:rgba(127,127,127,0.15);'
        st.dataframe(
            display.style.apply(lambda _: style_grid, axis=None).format(
                {'Targets before': '{:.2f}', 'Targets after': '{:.2f}', 'Δ targets': '{:+.2f}'}),
            hide_index=True, width="stretch", height=df_auto_height(len(display)))
        st.caption("This player's row is highlighted. 'Trusted' tier keeps its own value; 'tail' shares "
                  "whatever budget the trusted tier leaves behind.")


_VACANCY_VOLUME_LABELS = {
    'passing_attempts': 'Pass attempts',
    'rushing_attempts': 'Carries',
    'targets': 'Targets',
}


# A vacated per-game volume at/below this bar is not worth a full
# redistribution table - the OUT player was a bit part in that stat, and the
# reallocation is a fantasy-negligible sliver.  Per explicit request
# 2026-08-31: "only show the true reallocation for injured players that drew
# >1 target or >1 carry per game" (tightening the earlier 1.5-target /
# 3-carry bar; a 0.1-target vacancy was still rendering a full grid row when
# the open player happened to be a recipient).  QB pass attempts have no bar
# - a missing QB is never minor.
_VACANCY_NEGLIGIBLE_PER_GAME = {'targets': 1.0, 'rushing_attempts': 1.0}


def _vacancy_trusted_tier_cutoff(volume_key, source_role):
    """Depth cutoff at/above which a fill-in is worth its own row.  Carries -
    and an RB's vacated targets - use the RB room's tier; a receiver's vacated
    targets use the pass-catcher room's tier.  QB pass attempts (single named
    replacement) are never trimmed."""
    if volume_key == 'rushing_attempts':
        return PASS_CAPACITY_TRUSTED_TIER_RB
    if volume_key == 'targets':
        return (PASS_CAPACITY_TRUSTED_TIER_RB
                if str(source_role or '').upper().startswith('RB')
                else PASS_CAPACITY_TRUSTED_TIER)
    return None


def _summarize_vacancy_entry(entry, this_player):
    """Decide how one vacancy-ledger row should be shown.  Pure (no Streamlit)
    so it is unit-tested directly.  Returns a dict with 'kind':
      'skipped'    - stale provenance / no eligible recipient, nothing vacated
      'negligible' - the OUT player's per-game volume is at/below the
                     show-a-distribution bar (see _VACANCY_NEGLIGIBLE_PER_GAME);
                     collapses to the same informative one-line 'text' as the
                     'full' caption but with NO recipient grid, even for the
                     player whose decomposition this is
      'full'       - normal: 'caption', 'recipients' (table rows kept to the
                     trusted tier), and 'minor' (a one-line remainder summary
                     for the fill-ins that were trimmed, or None)
    In the 'full' case THIS player's own row is always kept, even when small
    or below the tier."""
    volume_key = str(entry.get('volume', '') or '')
    label = _VACANCY_VOLUME_LABELS.get(volume_key, volume_key or 'volume')
    source_player = str(entry.get('source_player', '') or 'OUT teammate')
    vacated = float(entry.get('vacated', 0.0) or 0.0)
    allocated = float(entry.get('allocated', 0.0) or 0.0)
    unallocated = float(entry.get('unallocated', 0.0) or 0.0)
    reason = str(entry.get('reason', '') or '')
    recips = list(entry.get('recipients', []) or [])

    if vacated <= 0 and not recips:
        return {'kind': 'skipped', 'text': f"{source_player} — {label}: {reason}"}

    caption = (f"{source_player} out — {vacated:.1f} {label.lower()} vacated; "
               f"{allocated:.1f} redistributed to active teammates, "
               f"{unallocated:.1f} left unfilled. {reason}").rstrip()

    # Below ~1 vacated target / carry per game the OUT player was a bit part
    # in that stat: keep the informative caption, drop the recipient grid -
    # even when the open player is one of the (tiny) recipients (explicit
    # request 2026-08-31). A missing QB has no bar (passing_attempts absent).
    neg = _VACANCY_NEGLIGIBLE_PER_GAME.get(volume_key)
    if neg is not None and vacated < neg:
        return {'kind': 'negligible', 'text': caption}

    cutoff = _vacancy_trusted_tier_cutoff(volume_key, entry.get('functional_source_role'))
    kept, minor_n, minor_vol = [], 0, 0.0
    for r in recips:
        rank = r.get('team_rank')
        name = str(r.get('player', ''))
        if cutoff is None or rank is None or rank <= cutoff or name == this_player:
            kept.append({
                'Out player': source_player, 'Stat': label,
                'Fills in for them': name,
                'Volume added': round(float(r.get('allocated', 0.0) or 0.0), 2),
            })
        else:
            minor_n += 1
            minor_vol += float(r.get('allocated', 0.0) or 0.0)
    minor = (f"{source_player} — {label}: {minor_n} more fill-in"
             f"{'s' if minor_n != 1 else ''} below the trusted tier absorbed "
             f"{minor_vol:+.2f} combined." if minor_n else None)
    return {'kind': 'full', 'caption': caption, 'recipients': kept, 'minor': minor}


def _render_vacancy_redistribution(detail):
    """How an OUT teammate's vacated volume is filled in - which player lost
    it and which active teammates absorbed how much of it. Added 2026-08-29
    per explicit request ("add a table that shows how the vacancy volume is
    filled in... so i understand how the volume is redistributed"). The
    per-recipient breakdown rides on each ledger row's `recipients` list -
    the RB allocator path always recorded it; `redistribute_v2_vacated_usage`
    (QB handoff, and WR/TE when the RB allocator is off) was extended to
    record it the same day. `detail['vacancy']` is already filtered to this
    player's team upstream.

    2026-08-30 per explicit request: an OUT player who vacated ~1 or fewer
    targets / carries per game (tightened from 1.5 / 3.0 on 2026-08-31)
    collapses to a one-line caption, and recipients below the trusted tier
    (PASS_CAPACITY_TRUSTED_TIER / _RB) are rolled into a single remainder line
    instead of cluttering the grid - the deep-roster fill-ins aren't getting
    a relevant look."""
    vacancy = detail.get('vacancy', [])
    if not vacancy:
        return
    st.markdown("**How vacated volume is redistributed**")
    this_player = str(detail.get('player', ''))
    recipient_rows, minor_lines = [], []
    for entry in vacancy:
        summ = _summarize_vacancy_entry(entry, this_player)
        if summ['kind'] in ('skipped', 'negligible'):
            st.caption(summ['text'])
            continue
        st.caption(summ['caption'])
        recipient_rows.extend(summ['recipients'])
        if summ['minor']:
            minor_lines.append(summ['minor'])
    if recipient_rows:
        rec_df = pd.DataFrame(recipient_rows)
        style_grid = pd.DataFrame('', index=rec_df.index, columns=rec_df.columns)
        highlight = rec_df['Fills in for them'].astype(str).eq(this_player)
        style_grid.loc[highlight, :] = 'font-weight:bold; background-color:rgba(127,127,127,0.15);'
        st.dataframe(
            rec_df.style.apply(lambda _: style_grid, axis=None).format({'Volume added': '{:+.2f}'}),
            hide_index=True, width="stretch", height=df_auto_height(len(rec_df)))
        if highlight.any():
            st.caption("This player's row is highlighted.")
    for line in minor_lines:
        st.caption(line)


def _render_decomposition_audit_body(detail):
    """Everything about THIS player that isn't needed at a glance -
    role/depth-chart provenance, the RB allocator ledger, availability, and
    the injury/vacancy ledger. Lives in its own "Role, audit & data sources"
    tab (moved out of a same-page expander 2026-08-25) instead of being
    permanently on-page.

    TRIMMED 2026-08-24 per explicit request. This used to also carry a
    'Data confidence and cutoff safeguards' block that dumped ~12 fields on
    every single player, most of which were identical across the entire
    ~900-player pool (raw pipeline-mode tags, the full Ourlads/availability
    warning lists, QB1-selection state, role-segment counts) - real audit
    trail, but about the pipeline that built the whole board, not about
    whichever player happened to be open. That content now lives once, in
    _render_pipeline_diagnostics (the last tab of the "Live data pulls"
    expander). What's left
    here only ever prints when it has something to say about THIS player:
    zero-value branches (no role change detected, no alignment profile
    active, a routine exact-name depth-chart match, no real availability
    source) are skipped rather than printed as boilerplate "nothing to
    report" lines."""
    role = detail.get('role', {})
    role_bits = []
    for label, key, fmt in (
        ('Expected snaps', 'expected_snap_share', '.0%'),
        ('Role confidence', 'role_confidence', '.0%'),
        ('Target-earner score', 'target_earner_score', '.2f'),
        ('Target share', 'target_share', '.1%'),
        ('Carry share', 'carry_share', '.1%'),
        ('ADOT', 'adot', '.1f'),
    ):
        value = role.get(key)
        if value is not None and pd.notna(value):
            role_bits.append(f"{label}: {float(value):{fmt}}")
    # Role-change evidence is 0% for nearly every player (it only fires
    # when the allocator actually detected a change) - showing it only
    # when non-zero turns a near-universal "0%" line into a real signal.
    role_change = role.get('role_change_confidence')
    if role_change is not None and pd.notna(role_change) and float(role_change) > 0:
        role_bits.append(f"Role-change evidence: {float(role_change):.0%}")
    st.markdown("**Expected role**")
    st.caption(" · ".join(role_bits) if role_bits else "No time-valid role profile was available.")

    _render_role_confidence_table(role)

    if detail.get('position') == 'QB' and role.get('qb_projected_starter'):
        st.success(f"Expected QB1: full workload from {role.get('starter_source', 'QB1 selection')}.")
    elif detail.get('position') == 'QB' and role.get('qb1_selection_required'):
        st.warning("QB1 selection required: this room receives no normal QB volume until one player is selected.")
    elif detail.get('position') == 'QB':
        st.caption("QB status: not the expected starter; projected QB volume is held at zero.")
    elif role.get('starter_source') and role.get('starter_source') != 'Not applicable':
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
    # A routine exact-name/high-confidence match is the expected case
    # for the vast majority of players and says nothing worth reading -
    # only surface this line when the match itself is uncertain.
    match_method = role.get('identity_match_method')
    match_confidence = role.get('identity_match_confidence')
    if match_method and not (match_method == 'exact name' and match_confidence == 'high'):
        source_name = role.get('ourlads_source_name') or detail['player']
        st.caption(
            f"Depth-chart identity: {source_name} → {detail['player']} "
            f"via {match_method} ({match_confidence or 'unrated'} confidence)."
        )
    if role.get('identity_match_warning'):
        st.warning(f"Depth-chart identity warning: {role['identity_match_warning']}")
    if role.get('ourlads_source_status_warning'):
        st.warning(
            f"Depth-chart status flag: {role['ourlads_source_status_warning']}. "
            "It is a chart-source warning only; current availability controls the projection."
        )
    # alignment_note reads "...alignment matchup is neutral" for
    # essentially every player right now (no 2026 alignment data is
    # loaded yet) - only worth a line once the underlying profile is
    # actually available and doing something.
    if role.get('alignment_available') and role.get('alignment_note'):
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
            st.dataframe(style_plain_dataframe(capacity_frame[display_columns]), hide_index=True, width="stretch")
        allocation = detail.get('rb_team_allocation', [])
        if allocation:
            with st.expander("Team RB allocation and projected opportunities"):
                st.dataframe(style_plain_dataframe(pd.DataFrame(allocation)), hide_index=True, width="stretch")
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

    if detail.get('position') == 'QB':
        qb_stats_detail = detail.get('stats', {})
        blend_source = next(
            (v for v in qb_stats_detail.values() if v.get('qb1_blend_applied')), None)
        if blend_source:
            st.markdown("**QB1 volume blend**")
            st.caption(
                "Passing/rushing attempts are rebuilt from team dropback capacity (raw team-game "
                "history) and this QB's own rush-share tendency, not carried forward as a flat "
                "per-game rate - a real rushing floor lowers projected passing volume even when the "
                "underlying per-game passing history is completely clean."
            )
            capacity = blend_source.get('qb1_blend_team_dropback_capacity')
            personal_db = blend_source.get('qb1_blend_personal_dropbacks')
            personal_share = blend_source.get('qb1_blend_personal_rush_share')
            league_share = blend_source.get('qb1_blend_league_rush_share')
            evidence_w = blend_source.get('qb1_blend_evidence_weight')
            prior2_w = blend_source.get('qb1_blend_prior2_weight')
            prior2_db = blend_source.get('qb1_blend_personal_dropbacks_2024')
            blend_rows = []
            if capacity is not None:
                blend_rows.append({
                    'Input': 'Team dropback capacity', 'Value': f"{capacity:.1f}/game",
                    'Source': "raw team-game history (unfiltered - see this QB's own row for why)"})
            if personal_db is not None:
                blend_rows.append({
                    'Input': 'Personal dropback sample', 'Value': f"{personal_db:.0f}",
                    'Source': 'his own prior-season pass + rush attempts, eligible games only'})
            if personal_share is not None:
                blend_rows.append({
                    'Input': 'Personal rush share', 'Value': f"{personal_share:.0%}",
                    'Source': 'his own rush attempts ÷ his own dropbacks'})
            if league_share is not None:
                blend_rows.append({
                    'Input': 'League-average rush share', 'Value': f"{league_share:.0%}",
                    'Source': 'qualified QBs, same prior season'})
            if evidence_w is not None:
                blend_rows.append({
                    'Input': 'Self-weight', 'Value': f"{evidence_w:.0%}",
                    'Source': 'how much his OWN rush share (vs. league average) counts'})
            if prior2_w:
                blend_rows.append({
                    'Input': 'Two-years-back grounding', 'Value': f"{prior2_w:.0%}",
                    'Source': (f"weight toward his prior2 season ({prior2_db:.0f} dropbacks that year)"
                              if prior2_db else 'weight toward his prior2 season')
                    + ' - active because his prior-season sample is thin relative to a full season'})
            if blend_rows:
                st.dataframe(style_plain_dataframe(pd.DataFrame(blend_rows)), hide_index=True,
                             width="stretch", height=df_auto_height(len(blend_rows)))

    availability = detail['availability']
    st.markdown("**Availability and workload**")
    st.caption(
        f"{availability['status'] or 'No current designation'} — "
        f"plays probability {availability['plays_probability']:.0%}; "
        f"workload if active {availability['workload_if_active']:.0%}."
    )
    reported = availability.get('reported_probability')
    if reported is not None and pd.notna(reported):
        st.caption(
            f"FantasyPros reported {float(reported):.0%} chance to play — shown for reference only, "
            "not an input to this projection. Only Out and Doubtful change the model (both → not playing)."
        )
    # 'no current availability source' is the sentinel for "nothing
    # to report" - printing it as if it were a real source just
    # restates the line above in a more confusing way.
    if availability.get('source') and availability['source'] != 'no current availability source':
        availability_note = availability.get('note') or ''
        st.caption(
            f"Availability source: {availability['source']} "
            f"({availability.get('match_method', 'not recorded')})"
            f"{f'; {availability_note}' if availability_note else ''}."
        )

    calibration = detail.get('calibration', {})
    if calibration.get('enabled'):
        st.caption(
            f"Point calibration: raw {calibration.get('raw_points', detail['raw_points']):.2f} → "
            f"displayed {calibration.get('displayed_points', detail['calibrated_points']):.2f} "
            f"({calibration.get('delta', 0.0):+.2f}); "
            f"slope {calibration.get('slope', 1.0):.3f}, intercept {calibration.get('intercept', 0.0):.3f}."
        )

    _render_vacancy_redistribution(detail)

    _render_pass_capacity_room(detail)

    # Historical/backtest classification only matters in the deviant
    # case - a live/upcoming target is the default for every normal
    # session and doesn't need a line saying so on every player.
    contract = detail.get('data_contract', {})
    if contract.get('historical_target'):
        st.caption(
            "Target classified as historical/backtest; "
            f"latest observed week: {contract.get('latest_observed_week') or 'none'}."
        )
    # PFF alignment/scheme fields only print once there's a real
    # profile behind them - right now none of these have data loaded
    # for any player, so none of these lines appear yet; they'll start
    # showing per-player once a real archive is imported.
    for label, key, has_signal in (
        ('PFF alignment (offense)', 'pff_alignment', lambda v: bool(v.get('included_weeks'))),
        ('PFF alignment (defense)', 'pff_alignment_defense', lambda v: (v.get('profile_rows') or 0) > 0),
        ('PFF scheme (offense)', 'pff_scheme', lambda v: bool(v.get('included_weeks'))),
        ('PFF scheme (defense)', 'pff_scheme_defense', lambda v: (v.get('profile_rows') or 0) > 0),
    ):
        value = contract.get(key)
        if isinstance(value, dict) and has_signal(value):
            st.caption(f"{label}: {value.get('adjustment', value.get('status'))}.")
    st.caption("Market and FantasyPros values in the ranking table are comparisons, never inputs to this projection.")


def _open_projection_dialog(detail):
    """Open the backend-produced weekly projection decomposition: a header
    (identity, top-line points, matchup toughness) always on top, then three
    tabs - Overview (primary at-a-glance table + tabbed Deep Dive), Range of
    outcomes (bust/median/boom), and Role, audit & data sources - replacing
    what was previously one long scroll of every section stacked in a row
    (explicit request, 2026-08-25: same content, organized so a specific
    question - "what's my range of outcomes," "why this role" - is one click
    instead of a long scroll)."""
    if not detail:
        return

    def _body():
        _render_decomposition_header(detail)

        tab_overview, tab_outcomes, tab_audit, tab_news = st.tabs(
            ["Overview", "Range of outcomes", "Role, audit & data sources", "News"])

        with tab_overview:
            _render_decomposition_primary_table(detail)
            _render_alignment_mix(detail)

            stats = detail.get('stats', {})
            if stats:
                st.markdown("**Deep dive**")
                game_log_by_season = detail.get('game_log_by_season') or {}
                defense_log_by_season = detail.get('defense_weekly_log_by_season') or {}
                # Default to whichever season actually has games logged - the
                # honest empty case is a cold-start CURRENT season, and
                # there's no reason to land the user on a blank tab by
                # default when last season's full log is one click away.
                # Each stat tab owns its OWN season selector below (explicit
                # request - a single shared control meant switching season
                # on one stat silently moved every other stat's tab too), so
                # this is only the shared starting point, not a value that
                # controls every tab at once.
                all_years = sorted(set(game_log_by_season) | set(defense_log_by_season), reverse=True)
                default_year = next(
                    (yr for yr in all_years if game_log_by_season.get(yr)),
                    all_years[0] if all_years else detail.get('season_year'))
                stat_keys = list(stats.keys())
                deep_dive_tabs = st.tabs([s.replace('_', ' ').title() for s in stat_keys] + ['Context'])
                for stat, tab in zip(stat_keys, deep_dive_tabs[:-1]):
                    with tab:
                        _render_stat_deep_dive(
                            detail, stat, game_log_by_season, defense_log_by_season, default_year)
                with deep_dive_tabs[-1]:
                    _render_context_deep_dive(detail)

        with tab_outcomes:
            distribution = detail.get('distribution')
            if distribution:
                _render_distribution_chart(distribution, detail['position'])
            else:
                st.caption(
                    "Range of outcomes: not available yet for this run — needs scripts/fit_weekly_"
                    "distribution.py's bands (see data/weekly_distribution.py)."
                )

        with tab_audit:
            _render_decomposition_audit_body(detail)

        with tab_news:
            _render_player_news(detail)

    dialog = st.dialog(f"Projection decomposition — {detail['player']}", width="large",
                       on_dismiss=_close_projection_dialog)(_body)
    dialog()


def _render_player_news(detail):
    """Last 6 FantasyPros news reports for this player, newest first, in one
    scrollable feed - explicit request 2026-08-27.

    AUTO-LOADS on first view of this tab (no button) and caches in
    st.session_state per player for the rest of the session - safe given
    this account's real 500 calls/day cap (confirmed by the user, see
    data.draft_sources' own module comment; FantasyPros' free tier is
    50/month and would NOT support this pattern, which is why every other
    live pull in this app stays button-gated). Revisiting the same player
    later in the same session costs nothing further; a fresh session re-pulls
    once per player actually opened, not the whole pool up front.
    """
    player_name = detail.get('player')
    if not player_name:
        return

    from data.draft_sources import load_player_id_map, get_fantasypros_api_key
    from data.fantasypros_availability import fetch_fantasypros_player_news, resolve_fantasypros_player_id

    api_key, _key_source = get_fantasypros_api_key()
    if not api_key:
        st.caption("No FantasyPros API key set (add one in the Draft HQ tab) - player news needs it.")
        return

    id_map, id_map_err = load_player_id_map()
    fpid = resolve_fantasypros_player_id(player_name, id_map)
    if fpid is None:
        st.caption(
            f"No FantasyPros player ID match for {player_name}"
            + (f" ({id_map_err})" if id_map_err else " (crosswalk has no matching name)")
            + " - can't fetch news for this player."
        )
        return

    cache_key = f'_fp_news_{fpid}'
    if cache_key not in st.session_state:
        with st.spinner("Loading FantasyPros news…"):
            news, error = fetch_fantasypros_player_news(api_key, fpid)
        st.session_state[cache_key] = {'news': news, 'error': error}
    cached = st.session_state[cache_key]

    if cached.get('error'):
        st.error(f"Couldn't fetch news: {cached['error']}")
        if st.button("Retry", key=f'retry_news_{fpid}'):
            del st.session_state[cache_key]
            st.rerun()
        return

    news = cached.get('news') or []
    if not news:
        st.caption("No recent FantasyPros news found for this player.")
        return

    st.caption(f"Last {len(news)} FantasyPros report(s), newest first.")
    with st.container(height=520, border=True):
        for i, item in enumerate(news):
            when = item.get('created_formatted') or item.get('created') or ''
            author = item.get('author')
            st.markdown(f"**{item.get('title') or '(untitled)'}**")
            st.caption(when + (f" — {author}" if author else ''))
            if item.get('desc'):
                st.write(item['desc'])
            if item.get('impact'):
                st.markdown(f"*Fantasy impact:* {item['impact']}")
            link = item.get('link')
            if link:
                st.markdown(f"[Full story]({link})")
            if i < len(news) - 1:
                st.divider()


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
        fantasypros_api_calls_today, FANTASYPROS_API_DAILY_LIMIT,
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

    used_today = fantasypros_api_calls_today()
    fetch = st.button("Fetch weekly projections", key="wr_fp_fetch", disabled=not api_key)
    st.caption(f"{used_today} call{'s' if used_today != 1 else ''} made today "
              f"(cap is {FANTASYPROS_API_DAILY_LIMIT}/day) - this pull costs 4.")
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


def _render_manual_availability_override(wk_year, wk_week, roster_df):
    """Let the user directly say 'this player is out' for one target week.

    Distinct from the FantasyPros pull next to it: that's an external feed
    that can lag real news (see the 2026-08-24 James Conner case - a real,
    still-unresolved 2025 season-ending foot injury that no live feed had
    flagged for Week 1 yet). This writes straight to data.availability_
    overrides' own reviewable CSV, which the model already reads FIRST -
    resolve_target_week_availability documents manual overrides winning over
    the injury report on the same player - this UI was the only missing
    piece, not new model logic.
    """
    from data.availability_overrides import (
        load_availability_overrides, save_availability_override, remove_availability_override,
    )
    st.caption(
        "Your own call for this exact week - wins over the FantasyPros feed for the same player. "
        "Use this when you know something the feed hasn't caught up to yet."
    )
    roster = roster_df if isinstance(roster_df, pd.DataFrame) else pd.DataFrame()
    players = roster['Player'].dropna().astype(str).sort_values().tolist() if 'Player' in roster.columns else []
    team_by_player = (dict(zip(roster['Player'], roster['Team']))
                      if {'Player', 'Team'}.issubset(roster.columns) else {})
    with st.form(key=f"wr_manual_availability_form_{wk_year}_{wk_week}", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            player = st.selectbox("Player", players, index=None, placeholder="Search this week's board…",
                                  key=f"wr_manual_avail_player_{wk_year}_{wk_week}")
        with c2:
            status = st.selectbox(
                "Status", ["Out", "Doubtful", "Questionable", "Healthy (clear any override)"],
                key=f"wr_manual_avail_status_{wk_year}_{wk_week}")
        note = st.text_input("Note (optional)", key=f"wr_manual_avail_note_{wk_year}_{wk_week}",
                             placeholder="e.g. still recovering from 2025 foot surgery, no return timetable")
        submitted = st.form_submit_button("Save override")
        if submitted:
            if not player:
                st.error("Pick a player first.")
            else:
                team = team_by_player.get(player, "")
                error = save_availability_override(
                    wk_year, wk_week, team, player, status.split(" ")[0], note=note)
                if error:
                    st.error(error)
                else:
                    st.success(f"{player} ({team}) set to {status.split(' ')[0]} for Week {wk_week}.")
                    # The next render() computes a fresh availability_fingerprint
                    # from this same CSV and passes it into
                    # build_weekly_projections, so the rerun below lands on a
                    # new cache key for THIS week automatically - no .clear()
                    # needed (that used to wipe every cached week, not just
                    # this one - see availability_fingerprint's docstring).
                    # Confirmed live 2026-08-24: a Zach Charbonnet 'Out'
                    # override saved here did not move his projection until
                    # the cache was invalidated for it.
                    st.rerun()

    current, _err = load_availability_overrides(wk_year, wk_week)
    if current.empty:
        st.caption(f"No manual overrides set for Week {wk_week}.")
        return
    st.caption(f"Manual overrides for Week {wk_week}:")
    for _, row in current.iterrows():
        row_c1, row_c2 = st.columns([5, 1])
        with row_c1:
            note_suffix = f" — {row['note']}" if str(row.get('note', '')).strip() else ""
            st.markdown(f"**{row['player']}** ({row['team']}) — {str(row['status']).title()}{note_suffix}")
        with row_c2:
            if st.button("Clear", key=f"wr_manual_avail_clear_{wk_year}_{wk_week}_{row['player']}_{row['team']}"):
                remove_availability_override(wk_year, wk_week, row['team'], row['player'])
                st.rerun()


def _render_fantasypros_injury_pull(wk_year, wk_week, roster_df=None):
    """
    FantasyPros-sourced injury/availability for this app's weekly model
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
        "Feeds this app's availability discount. No report for a player this week means "
        "healthy - never a stale carry-over from a prior week."
    )
    tab_api, tab_upload, tab_manual = st.tabs(["Live API pull", "Upload export", "Manual override"])
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
                # See availability_fingerprint's docstring - this CSV feeds
                # build_weekly_projections, and the fingerprint computed on
                # the next render() picks up this change on its own.
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
    with tab_manual:
        _render_manual_availability_override(wk_year, wk_week, roster_df)

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


def _render_live_data_hub(wk_year, wk_week, wk_scoring, wk_week_completed, name_pool,
                          roster_df=None, model_meta=None):
    """
    One expander, five tabs - consolidates what used to be four separate
    stacked accordions (FantasyPros weekly projection, FantasyPros injury/
    availability, PFF weekly alignment archive, market player-prop
    projection) plus the standalone "Data pipeline notes" expander that used
    to sit right below this one, so a page visit isn't multiple collapsed
    boxes deep before any model output appears.

    Streamlit tabs can't be conditionally hidden once declared, so the gate
    that used to hide a whole expander (upcoming-week-only for market props)
    shows as an explanatory caption INSIDE that tab instead - the tab is
    always there, its content just says why there's nothing to do yet.

    Returns (fp_weekly_df, market_df), the two values render() still needs
    downstream; the injury and PFF tabs read/write straight to disk and
    session state and have no return-value contract of their own.
    """
    with st.expander("📡 Live data pulls", expanded=False):
        tab_fp, tab_injury, tab_pff, tab_market, tab_pipeline = st.tabs(
            ["FantasyPros projections", "Injury/availability", "PFF alignment", "Market props",
             "Data pipeline notes"])
        with tab_fp:
            fp_weekly = _render_fantasypros_weekly_pull(wk_year, wk_week, wk_scoring)
        with tab_injury:
            _render_fantasypros_injury_pull(wk_year, wk_week, roster_df)
        with tab_pff:
            _render_pff_weekly_alignment_upload(wk_year, wk_week)
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
        with tab_pipeline:
            _render_pipeline_diagnostics(model_meta)
    return fp_weekly, market_df


def render():
    st.markdown("<div class='custom-section-header'>WEEKLY RANKINGS</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
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

    df_stats, t_col, n_col, _ = load_and_merge_data(wk_year, wk_scoring)
    form_df = build_recent_form_rank(df_stats, n_col, t_col, n_weeks=RECENT_FORM_GAMES)

    # A lightweight name/team/pos roster from the stats frame, so the Live
    # data pulls below can match FantasyPros / injury uploads BEFORE the board
    # is built.
    _roster_src = [c for c in (n_col, t_col, 'position') if c in df_stats.columns]
    hub_roster = None
    if _roster_src:
        hub_roster = (df_stats[_roster_src].dropna(subset=[n_col]).drop_duplicates()
                      .rename(columns={n_col: 'Player', t_col: 'Team', 'position': 'Pos'}))

    st.markdown("### This app's weekly model")
    # Data pulls FIRST, then an explicit build - the board no longer builds on
    # tab open (explicit request 2026-08-29): a user can stage FantasyPros
    # rankings / injury reports / PFF alignment and only then spend the model
    # build, instead of building, uploading, and rebuilding.
    fp_weekly, market_df = _render_live_data_hub(
        wk_year, wk_week, wk_scoring, wk_week_completed,
        hub_roster[['Player']] if hub_roster is not None else None,
        roster_df=hub_roster,
        model_meta=st.session_state.get('weekly_rank_last_model_meta'))
    _fantasypros_freshness_caption(wk_year, wk_week, wk_scoring)

    build_key = (wk_year, wk_week, wk_scoring)
    if st.button("🧮 Build board", key="weekly_rank_build", type="primary"):
        st.session_state['weekly_rank_built_for'] = build_key
    board_ready = st.session_state.get('weekly_rank_built_for') == build_key

    model_df, model_meta = pd.DataFrame(), {}
    avail_fp = None
    if board_ready:
        with skeleton_loader("table", n_rows=10, n_cols=7):
            # Real cache-key input, not read inside build_weekly_projections -
            # see availability_fingerprint's own docstring. Lets a saved
            # override or FantasyPros pull invalidate just THIS week's cached
            # board instead of the blanket build_weekly_projections.clear()
            # this used to require (which evicted every other cached week too).
            avail_fp = availability_fingerprint(
                wk_year, wk_week, AVAILABILITY_OVERRIDE_PATH, FANTASYPROS_INJURY_PATH)
            model_df, model_meta = build_weekly_projections(
                wk_year, wk_week, wk_scoring, availability_fingerprint=avail_fp)
        st.session_state['weekly_rank_last_model_meta'] = model_meta

    if not board_ready:
        st.info(
            "Set the season / week / scoring above, stage any FantasyPros rankings or injury "
            "reports in **📡 Live data pulls**, then click **🧮 Build board**. "
            "Changing a selector hides the board until you rebuild, so a rebuild always picks up "
            "the current settings and uploads."
        )
    elif model_df.empty:
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
        display_cols += ['Injury Status', 'Last 5 Weeks',
                        'FantasyPros Rank', 'Model Rank', 'Market Rank', 'Market Coverage',
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
        # its requested slot next to Injury Status.
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
        # The availability fingerprint is part of the key so a manual injury
        # override / FantasyPros pull for this week (which rebuilds the board
        # and can move the injured player plus every vacancy recipient) forces
        # the open decomposition and the per-player detail cache to re-resolve
        # from the rebuild instead of showing the pre-injury breakdown.
        detail_config = (wk_year, wk_week, wk_scoring, avail_fp)
        selected_state = st.session_state.get(_PROJECTION_DETAIL_KEY)
        if selected_state and selected_state.get('config') != detail_config:
            prev_config = selected_state.get('config')
            prev_key = selected_state.get('key')
            same_board = (isinstance(prev_config, tuple) and len(prev_config) == len(detail_config)
                          and prev_config[:-1] == detail_config[:-1])
            fresh = (_selected_player_detail(model_meta, detail_config, prev_key)
                     if same_board and prev_key is not None else None)
            if fresh is not None:
                # Same week/scoring, only the availability fingerprint moved -
                # an injury override / feed pull rebuilt this board. Keep the
                # decomposition open but re-resolve it against the rebuild so
                # the injured player and every vacancy recipient show fresh
                # numbers instead of the pre-injury breakdown.
                st.session_state[_PROJECTION_DETAIL_KEY] = {
                    'config': detail_config, 'detail': fresh, 'key': prev_key,
                }
            else:
                # A real board switch (week / scoring), or nothing to
                # re-resolve: a Week 3 explanation is not meaningful over a
                # Week 4 table.
                st.session_state.pop(_PROJECTION_DETAIL_KEY, None)
        # Row KEYS only here, not a model_meta['explanations'] lookup per
        # row - that dict lookup is cheap today, but every displayed row
        # (up to 40) was being looked up on EVERY rerun regardless of
        # whether anyone selects a row. Deferred into
        # _selected_player_detail below so only the one row actually
        # clicked ever gets resolved.
        row_keys = [(row['Player'], row['Position'], row['Team']) for _, row in display_df.iterrows()]
        model_table_key = f"weekly_rank_model_table_{wk_year}_{wk_week}_{wk_scoring}"

        def _on_model_row_select():
            rows = st.session_state.get(model_table_key, {}).get('selection', {}).get('rows', [])
            if rows and rows[0] < len(row_keys):
                row_key = row_keys[rows[0]]
                detail = _selected_player_detail(model_meta, detail_config, row_key)
                if detail:
                    st.session_state[_PROJECTION_DETAIL_KEY] = {
                        'config': detail_config, 'detail': detail, 'key': row_key,
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
