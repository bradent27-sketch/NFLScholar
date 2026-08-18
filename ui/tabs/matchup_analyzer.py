"""
Matchup Analyzer tab: a player-vs-team-defense prep tool.

The question it answers is "how does THIS player project against THAT
defense", which is deliberately not the question any other tab in this app
answers. Player Search profiles one player in isolation; Defensive Yield
profiles one defense in isolation; this puts the two side by side and makes
the comparison explicit.

The two pickers are fully independent on purpose. Checking a player against
a defense he isn't playing this week is a completely normal use - it's how
you price a hypothetical, scout a playoff path, or sanity-check a season-long
trade - so the defense selector offers all 32 teams rather than resolving the
player's actual next opponent. (His real next opponent is one click away on
the Game Slate.)

Layout: player on the left, defense on the right, both headed by a team
banner so a dense two-column page always says who you're looking at. The
sections below the fold - Scheme Fit, Anytime TD, Compare Board, Prop
Analysis - need BOTH sides, so they run full width underneath rather than
being crammed into one column.

Everything on this tab is real season data from files already on disk,
computed locally at zero API cost, with exactly two exceptions, both labelled
in the UI: the Matchup Curves (projections) and Live Prop Lines (opt-in,
one Odds API call). See data/matchup_signals.py for the compute layer and
docs/matchup_analyzer_data.md for the source-by-source breakdown.
"""
import pandas as pd
import streamlit as st

from config import (
    AVAILABLE_SEASONS_WITH_UPCOMING, MASTER_TEAMS_LIST, TEAM_CONFIG, THEME, abbr_to_pff_team,
    get_position_color, pff_team_to_abbr,
)
from data import matchup_signals as ms
from data.loaders import (
    load_all_pff_data, load_external_coverage_schemes,
    load_saved_odds_api_key, load_schedule, load_sharp_positional_coverage,
    load_sumersports_tendency_data, fetch_nfl_odds, fetch_nfl_player_props,
    load_pff_data_with_fallback,
)
from data.transforms import (
    build_points_allowed_matrix, load_and_merge_data, precompute_league_percentiles,
)
from data.utils import american_odds_to_prob
from ui.charts import (
    render_chart_click_overlay, render_game_log_bars, render_game_log_line,
    render_percentile_bar_list, render_split_bars, render_tier_curve,
)
from ui.components import (
    build_player_search_labels, compute_bye_weeks, render_back_button,
    render_hero_tiles, render_team_banner, skeleton_loader, switch_tab,
)
from ui.player_snapshot import build_player_snapshot
from ui.styling import get_matchup_color, get_pff_color

C = THEME['colors']

# Markets to request for the opt-in live-prop lookup, per position. Narrower
# than config.ODDS_API_PLAYER_PROP_MARKETS on purpose: this is one paid call
# per matchup and a receiver's passing-attempt line is never the question.
_POSITION_MARKETS = {
    'QB': ['player_pass_yds', 'player_pass_tds', 'player_pass_attempts', 'player_anytime_td'],
    'RB': ['player_rush_yds', 'player_rush_attempts', 'player_receptions', 'player_anytime_td'],
    'WR': ['player_reception_yds', 'player_receptions', 'player_anytime_td'],
}
_POSITION_MARKETS['TE'] = _POSITION_MARKETS['WR']
_POSITION_MARKETS['FB'] = _POSITION_MARKETS['RB']


def _section(title, help_text=None):
    st.markdown(f"<div class='custom-section-header'>{title}</div>", unsafe_allow_html=True)
    if help_text:
        st.caption(help_text)


def _team_label(abbr):
    return TEAM_CONFIG.get(abbr, {}).get('name', abbr)


def render():
    render_back_button()
    _section("MATCHUP ANALYZER", "Pick a team to filter the roster, then a player from it — or search the whole league. Pick any defense to check them against; they don't have to be playing each other.")

    # A jump from another tab (Game Slate's "{TEAM} players" button is the
    # one live caller today) can request a season - must land in
    # session_state BEFORE the season selectbox below is instantiated this
    # pass, same widget-state-ordering rule every other cross-tab trigger in
    # this app follows. Guarded on membership so an unlisted value from a
    # future caller can't make the selectbox raise - now the SAME list
    # Game Slate itself uses (AVAILABLE_SEASONS_WITH_UPCOMING), including
    # the upcoming season even before it has any played games (see the
    # stats_df.empty guard right below the selectbox, which already handles
    # that gracefully).
    jump_season = st.session_state.pop('ma_jump_season', None)
    if jump_season in AVAILABLE_SEASONS_WITH_UPCOMING and st.session_state.get('ma_season') != jump_season:
        st.session_state['ma_season'] = jump_season

    c_season, c_team, c_player, c_defense = st.columns([1, 1.3, 1.7, 1.4])
    with c_season:
        # index=1, not 0 - same convention as Depth Charts/Player Search/
        # Player Compare: the upcoming season is selectable but isn't the
        # default landing view while it has no played games yet.
        season = st.selectbox("Season", AVAILABLE_SEASONS_WITH_UPCOMING, index=1, key="ma_season")

    with skeleton_loader("tiles", n=8):
        stats_df, team_col, name_col, _ = load_and_merge_data(season, _scoring())
    if stats_df.empty:
        st.info(f"No player data available for {season} yet.")
        return

    bye_weeks = compute_bye_weeks(season)

    # Team filter narrows the player picker below it, same "team first,
    # then player" pattern Player Search uses - explicit user request,
    # since typing a name blind is slower than picking a roster first when
    # you already know which team you care about (this is exactly how a
    # jump from Game Slate arrives: a team, not yet a specific player).
    team_options = ["All Teams"] + sorted(TEAM_CONFIG.keys(), key=lambda a: TEAM_CONFIG[a].get('name', a))
    team_option_labels = {"All Teams": "All Teams (search whole league)"}
    team_option_labels.update({a: f"{TEAM_CONFIG[a]['name']} ({a})" for a in TEAM_CONFIG})

    jump_team = st.session_state.pop('ma_jump_team', None)
    if jump_team and jump_team in TEAM_CONFIG and st.session_state.get('ma_team_filter') != jump_team:
        st.session_state['ma_team_filter'] = jump_team
    with c_team:
        team_filter = st.selectbox(
            "Team filter (optional)", team_options, index=0, key="ma_team_filter",
            format_func=lambda a: team_option_labels.get(a, a),
        )

    # hide_drafted=False: this is a season-analysis tool, not a draft board.
    # Filtering out players already drafted in a mock would silently remove
    # most of the league from a picker whose whole job is "look anyone up".
    labels, label_to_name = build_player_search_labels(
        stats_df, name_col, team_col, bye_weeks, team_filter=team_filter, hide_drafted=False,
    )
    with c_player:
        # A jump from another tab lands here by pre-seeding this key, so the
        # index is resolved from session state rather than forced - a name
        # that isn't in this season's pool degrades to "nothing selected"
        # instead of raising.
        pending = st.session_state.pop('ma_jump_player', None)
        if pending:
            match = next((l for l in labels if label_to_name.get(l) == pending), None)
            if match:
                st.session_state['ma_player'] = match
        player_label = st.selectbox(
            "Player", labels, index=None, key="ma_player", placeholder="— Search any player —",
            help="Type any part of a name to filter live. Use the team filter to the left first if you don't know the exact name.",
        )
    with c_defense:
        default_def = st.session_state.pop('ma_jump_defense', None)
        if default_def in MASTER_TEAMS_LIST:
            st.session_state['ma_defense'] = default_def
        defense_team = st.selectbox(
            "Defense to check against", MASTER_TEAMS_LIST, key="ma_defense",
            format_func=lambda a: f"{a} — {_team_label(a)}",
        )

    if not player_label or player_label == "No players found":
        st.caption("Pick a player above to build the matchup.")
        return
    player_name = label_to_name.get(player_label, player_label)

    p_data = stats_df[stats_df[name_col].astype(str) == player_name]
    if p_data.empty:
        st.warning(f"No {season} data found for {player_name}.")
        return
    p_bio = p_data.iloc[0]
    position = str(p_bio.get('position', '')).upper()
    offense_team = str(p_bio.get(team_col, '')).upper()

    points_allowed = build_points_allowed_matrix(stats_df, season)
    softness_map = ms.defense_softness(points_allowed, position)
    pff, pff_year = load_pff_data_with_fallback(season)
    prowess_map = {
        pff_team_to_abbr(k): v
        for k, v in ms.team_defensive_prowess(pff.get('run_def'), pff.get('cov_summary')).items()
    }

    col_player, col_defense = st.columns(2)
    with col_player:
        render_team_banner(
            offense_team, title=player_name,
            subtitle=f"{position or '?'} · {_team_label(offense_team)} · {season}",
        )
    with col_defense:
        render_team_banner(defense_team, subtitle=f"Defense · {season}")
    if pff_year != season:
        st.caption(f"⚠️ Showing {pff_year} PFF grades — nothing uploaded for {season} yet.")

    # Rows are laid out in matched pairs rather than as two independent
    # top-to-bottom columns, so the two halves read as one comparison
    # instead of two separately-scrolling reports: each row answers the
    # SAME underlying question for the player and for the defense (his
    # tendency profile next to what the defense gives up by position, his
    # man/zone splits next to their man/zone scheme, his own usage next to
    # how the front holds up, his game log next to their week-by-week
    # allowed). Nothing here forces equal height per side - a short defense
    # panel next to a tall player one is fine, it's the ORDER that has to
    # line up, not the pixel count.
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        _render_tendency_profile(season, stats_df, name_col, p_data, p_bio, player_name, position, pff)
    with r1c2:
        _render_positional_vulnerability(stats_df, points_allowed, defense_team, position)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        _render_route_efficiency(pff, player_name, position)
    with r2c2:
        _render_coverage(defense_team, season)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        _render_usage_and_role(stats_df, name_col, team_col, player_name, offense_team)
    with r3c2:
        _render_run_defense(defense_team, season)

    r4c1, r4c2 = st.columns(2)
    with r4c1:
        selected_stat = _render_game_log_and_curves(
            season, stats_df, name_col, player_name, position, offense_team, defense_team,
            softness_map, prowess_map,
        )
    with r4c2:
        _render_defense_weekly_detail(stats_df, defense_team, position)
        _render_allowed_by_position(stats_df, defense_team, position)

    st.caption(
        "Everything above is measured season data. The two Matchup Curves are projections — "
        "they read a trend off real games, they don't predict one."
    )

    _render_scheme_fit(season, player_name, position, defense_team)
    _render_anytime_td(stats_df, name_col, player_name, position, defense_team, softness_map)
    _render_prop_analysis(season, stats_df, name_col, player_name, position, offense_team, defense_team, selected_stat)


def _scoring():
    """The app-wide scoring control lives in the header (ui.components.
    render_intro_and_glossary instantiates it), so this reads it rather than
    adding a second competing widget under the same key."""
    return st.session_state.get('score_tab1', 'Full PPR')


# ---------------------------------------------------------------------------
# Player column
# ---------------------------------------------------------------------------

def _render_tendency_profile(season, stats_df, name_col, p_data, p_bio, player_name, position, pff):
    """
    One percentile-bar chart per stat, in a fixed position-specific order.

    WR/TE goes through data.matchup_signals.receiver_tendency_entries - a
    DELIBERATE FORK of ui.player_snapshot.build_player_snapshot's WR/TE
    branch, not a change to it: that shared builder also feeds Player
    Search's matrix table and Player Compare, both explicitly off-limits
    (HANDOFF.md section 8), while this tab's receiver stat order/labels/
    set is a request specific to Matchup Analyzer alone.

    Every other position still reuses build_player_snapshot, same as
    before - a player's ADOT percentile has to be the same number on every
    tab, and three parallel implementations of "his percentile" guarantees
    it eventually isn't.
    """
    _section("TENDENCY PROFILE")
    st.caption("Percentile vs. same-position players who actually play. Order is fixed — volume first, then efficiency, then role.")
    percentiles = precompute_league_percentiles(stats_df, name_col, season)
    pct_row = percentiles[
        (percentiles[name_col].astype(str) == player_name) & (percentiles['position'] == position)
    ] if not percentiles.empty else pd.DataFrame()

    if position in ('WR', 'TE'):
        entries = ms.receiver_tendency_entries(pff, player_name, position, pct_row)
    else:
        grade = pff['pff_grades_map'].get(player_name.lower(), 0.0)
        games = max(p_data['week'].nunique() if 'week' in p_data.columns else 1, 1)
        entries = build_player_snapshot(
            player_name, position, p_data, p_bio, pff, pct_row, games, grade,
            get_pff_color(grade, raw_grade=True), season,
        )
        # PFF Blocking Grade, appended at the bottom for RBs too - not part
        # of the shared build_player_snapshot branch (Player Search/Player
        # Compare are off-limits, HANDOFF.md section 8), so this tab appends
        # it onto that function's output rather than editing it, same
        # fork-not-edit pattern the WR/TE branch above already uses.
        if position == 'RB':
            block_entry = ms.blocking_grade_entry(pff.get('block'), player_name, position)
            if block_entry:
                entries = list(entries) + [{
                    'label': 'PFF Blocking Grade',
                    'value_str': f"{block_entry['value']:.1f}",
                    'pct': block_entry['pct'],
                }]
    if not entries:
        st.caption("No percentile data for this player this season.")
        return
    # sort=False: the order is itself the message, and re-sorting by
    # percentile makes it look arbitrary.
    render_percentile_bar_list(entries, sort=False)


def _render_route_efficiency(pff, player_name, position):
    if position not in ('WR', 'TE', 'RB', 'FB'):
        return
    splits = ms.route_efficiency_splits(
        pff.get('rec'), pff.get('rec_scheme'), player_name, route_concept=pff.get('route_concept'),
        position=position,
    )
    if not splits['available']:
        return
    _section("ROUTE EFFICIENCY")
    if splits['alignment']:
        st.caption("Where he lines up and how efficient he is — percentile among receivers with 25+ routes.")
        render_percentile_bar_list(splits['alignment'], sort=False)
        # Wide YPRR isn't a PFF export column - PFF's route-concept exports
        # break out Slot specifically, never Wide as its own concept - so
        # it's CALCULATED as "the rest of his routes" once real slot
        # routes/yards are removed from the season total
        # (data.matchup_signals.wide_yprr_entry). Shown as the smaller
        # sub-bar under Wide rate, same as Slot YPRR sits under Slot rate.
        if any(e['label'] == 'Wide YPRR' for e in splits['alignment']):
            st.caption(
                "Wide YPRR isn't published by PFF — calculated as the rest of his routes once real slot "
                "routes/yards are removed from the season total. For a TE that also folds in in-line routes, "
                "which PFF has no separate yardage split for."
            )
    if splits['scheme']:
        st.caption("Same player, split by the coverage he faced. Bars are percentiles; the number is the raw value.")
        render_split_bars(splits['scheme'], 'vs Man', 'vs Zone')
        st.caption(
            "These are two independent PFF exports — alignment above, coverage faced here — with no cross-tab "
            "between them, so there is no \"slot vs zone\" number to show."
        )


def _render_usage_and_role(stats_df, name_col, team_col, player_name, offense_team):
    """
    Week-by-week share chart, one series at a time via a picker - same
    chart shape as the player's own Game By Game section, per explicit
    request, replacing three static season-average tiles that couldn't show
    a trend at all. Catch rate is dropped from this section entirely (it's
    an efficiency read, not a role/usage one, and was the odd one out among
    three share metrics) - still computed by ms.usage_and_role for anything
    else that wants it, just not displayed here.
    """
    usage = ms.usage_and_role(stats_df, name_col, player_name, offense_team, team_col=team_col)
    if not usage['available']:
        return
    _section("USAGE & ROLE")
    st.caption("Share of his own team's opportunities, week by week — not season total over season total, which halves a player who missed games.")

    weekly = usage['weekly']
    share_options = [
        ('target_share', 'Target share'), ('carry_share', 'Carry share'),
        ('opportunity_share', 'Opportunity share'),
    ]
    share_options = [(c, l) for c, l in share_options if c in weekly.columns and weekly[c].notna().any()]
    if not share_options:
        st.caption("No share data for this player this season.")
        return
    labels = [l for _, l in share_options]
    choice = st.selectbox("Share type", labels, key="ma_usage_stat")
    stat_col, stat_label = share_options[labels.index(choice)]

    plot_df = weekly.dropna(subset=[stat_col]).sort_values('week')
    if plot_df.empty:
        st.caption(f"No {stat_label.lower()} data for this player yet.")
        return
    values = plot_df[stat_col].astype(float).tolist()
    tooltips = [
        f"Wk {int(r.week)} vs {r.opponent}: {getattr(r, stat_col):.1f}% {stat_label.lower()}"
        for r in plot_df.itertuples()
    ]
    render_game_log_line(
        values, tooltips, avg=usage.get(stat_col), avg_label='season avg',
        bar_labels=[(str(r.opponent), f"W{int(r.week)}") for r in plot_df.itertuples()],
    )
    st.caption(f"{stat_label} per game he played, week by week. Dashed line = season average.")

    change = usage.get('role_change')
    if change:
        arrow = "📈" if change['direction'] == 'up' else "📉"
        verb = "up" if change['direction'] == 'up' else "down"
        st.markdown(
            f"{arrow} **Role change:** opportunity share is {verb} "
            f"{abs(change['delta']):.1f} points over his last 3 games "
            f"({change['prior']:.1f}% → {change['recent']:.1f}%)."
        )


def _render_game_log_and_curves(season, stats_df, name_col, player_name, position,
                                offense_team, defense_team, softness_map, prowess_map):
    options = ms.GAME_LOG_STATS.get(position)
    if not options:
        st.caption(f"No curated game-log stats for position {position or '?'} yet.")
        return None
    _section("GAME BY GAME")
    stat_labels = [label for _, label in options]
    choice = st.selectbox("Stat", stat_labels, key="ma_game_log_stat", label_visibility="collapsed")
    stat_col, stat_label = options[stat_labels.index(choice)]

    series = ms.player_game_series(stats_df, name_col, player_name, stat_col)
    if series.empty:
        st.caption("No played games for this player this season.")
        return stat_col, stat_label

    values = series['value'].tolist()
    tooltips = [
        f"Wk {int(r.week)} vs {r.opponent}: {r.value:.1f} {stat_label} — click to open the box score"
        for r in series.itertuples()
    ]
    render_game_log_line(
        values, tooltips, highlight=ms.highlight_games(values),
        avg=sum(values) / len(values),
        bar_labels=[(str(r.opponent), f"W{int(r.week)}") for r in series.itertuples()],
    )
    # Real Streamlit buttons, invisible, laid out in the SAME equal-width
    # st.columns(n) the chart divides its width into - clicking a point
    # opens that game's box score directly, which is what used to need a
    # separate chip strip below the chart (see render_chart_click_overlay's
    # own docstring for why this is safe to do now: it's real column layout,
    # not CSS chasing undocumented SVG hit-testing). game_link_positions,
    # not game_link_rows - this needs exactly one entry per point, in order,
    # including a None for a week that didn't resolve, or a dropped row
    # would shift every later column onto the wrong week's point.
    from data.box_score import game_link_positions
    from data.game_slate import season_slate
    slate, _err = season_slate(season)
    positions = game_link_positions(series, slate, team=offense_team)
    render_chart_click_overlay(positions, season, key_prefix="ma_game_pt")
    st.caption("Dashed line = season average. ★ = a top-quartile game for this player. Click a point to open that game's box score.")

    _render_matchup_curves(series, softness_map, prowess_map, defense_team, stat_label, season, offense_team)
    return stat_col, stat_label


def _render_matchup_curves(series, softness_map, prowess_map, defense_team, stat_label, season, offense_team):
    """All three curves plot whatever the stat picker above is set to, so
    every chart in this column always describes the same stat - reading
    charts with different y-axes side by side is a tax paid on every
    glance."""
    _section("MATCHUP CURVES", f"Projections for {stat_label} — read off real games, not a forecast.")

    tendency = ms.efficiency_elasticity_curve(series, softness_map, defense_team, stat_label)
    st.markdown("**Defensive Tendency Elasticity** — does he need a matchup soft to HIS position?")
    if tendency['available']:
        highlight = None
        if tendency['projection']:
            highlight = {**tendency['projection'], 'label': f"vs {defense_team}"}
        render_tier_curve(
            tendency['tiers'], avg=tendency['season_avg'], avg_label='season avg', highlight=highlight,
        )
        st.caption(
            f"{stat_label} per game vs. how soft the opponent is to HIS POSITION specifically (0 = "
            f"toughest, 100 = softest), across {tendency['games']} games. Hover a point for its sample size."
        )
    else:
        st.caption(tendency.get('reason', 'Not enough data yet.'))

    prowess = ms.efficiency_elasticity_curve(series, prowess_map, defense_team, stat_label)
    st.markdown("**Efficiency Elasticity** — does he need a bad defense, period?")
    if prowess['available']:
        highlight = None
        if prowess['projection']:
            highlight = {**prowess['projection'], 'label': f"vs {defense_team}"}
        render_tier_curve(
            prowess['tiers'], avg=prowess['season_avg'], avg_label='season avg', highlight=highlight,
        )
        st.caption(
            f"{stat_label} per game vs. how weak the opponent's OVERALL defense is (PFF's snap-weighted "
            f"team grade, position-independent — 0 = toughest defense in football, 100 = softest), across "
            f"{prowess['games']} games. A defense can be excellent overall and still be this player's "
            f"softest positional matchup (or the reverse) — that gap is why both curves are shown."
        )
    else:
        st.caption(prowess.get('reason', 'Not enough data yet.'))

    schedule = load_schedule(season)
    script = ms.game_script_sensitivity_curve(series, schedule, offense_team, stat_label)
    st.markdown("**Game-Script Sensitivity** — does he need the game to stay close?")
    if script['available']:
        render_tier_curve(
            script['tiers'], avg=script['season_avg'], avg_label='season avg', x_ticks=script['x_ticks'],
        )
        st.caption("Bucketed by final margin from his own team's point of view — trailing and leading are opposite scripts, not one 'blowout' bucket.")
    else:
        st.caption(script.get('reason', 'Not enough data yet.'))


# ---------------------------------------------------------------------------
# Defense column
# ---------------------------------------------------------------------------

def _render_positional_vulnerability(stats_df, points_allowed, defense_team, position):
    _section("POSITIONAL VULNERABILITY", "Which position to actually target. Rank 1 = allows the MOST, i.e. the softest matchup.")
    rows = ms.positional_vulnerability(points_allowed, defense_team)
    if not rows:
        st.caption("No points-allowed data for this defense yet.")
        return
    ypt = ms.ypt_allowed_for_team(load_sharp_positional_coverage(), _team_label(defense_team))
    cells = []
    for r in rows:
        is_subject = r['position'] == position
        marker = " ◀" if is_subject else ""
        cells.append({
            'label': f"{r['position']}{marker}",
            'value_str': f"{r['pts_allowed']:.1f}",
            'pct': r['pct'],
            'help': f"#{r['rank']} of {r['of']} in fantasy points allowed per game",
        })
        # Yards per target allowed, as a smaller sub-bar right after its own
        # position's row rather than a peer entry - moved up from the
        # Coverage panel below on explicit request, since it's a per-
        # position vulnerability read same as the bar above it. Labeled
        # "YPRR" per that same request; the `help` text says what it
        # actually measures for anyone who hovers it, since this app has no
        # real defense-side yards-per-route-run source to compute a literal
        # one from (see coverage_profile's own docstring on why yards per
        # TARGET is the closest measured equivalent).
        y = ypt.get(r['position'])
        if y is not None and y.get('pct') is not None:
            cells.append({
                'label': 'YPRR', 'value_str': f"{y['value']:.1f}", 'pct': y['pct'], 'sub': True,
                'help': "Yards per target allowed to this position (Sharp Football) — the closest "
                        "measured equivalent this app has to a defense-side YPRR; there is no "
                        "route-run-count data on the defense to compute a literal one.",
            })
    render_percentile_bar_list(cells, sort=False)
    st.caption(
        "Fantasy points allowed per game, with yards-per-target allowed as the smaller bar under each "
        "position. Green = soft, red = tough. ◀ marks the selected player's position."
    )


def _render_defense_weekly_detail(stats_df, defense_team, position):
    """
    One week-by-week chart for this defense, same shape as the player's own
    Game By Game chart on the other side of the row. Position and stat
    pickers sit side by side (not stacked) to save vertical space, and this
    is position-first only now - the old "By Stat" tab was a strictly
    worse path to the same chart (same picker, just position-second) and
    is gone per explicit request.
    """
    _section("WEEK BY WEEK DETAIL")
    positions = [p for p in ('QB', 'RB', 'WR', 'TE') if p in ms.ALLOWED_STAT_KEYS]
    if not positions:
        st.caption("No week-by-week data for this defense yet.")
        return

    c_pos, c_stat = st.columns(2)
    with c_pos:
        default_idx = positions.index(position) if position in positions else 0
        pos_choice = st.selectbox(
            "Position", positions, index=default_idx, key=f"ma_wk_pos_{defense_team}",
        )
    stat_opts = list(ms.ALLOWED_STAT_KEYS.get(pos_choice, [])) + [('fantasy_points', 'Fantasy Pts')]
    labels = [label for _, label in stat_opts]
    default_label = ms.MATCHUP_KEY.get(pos_choice, stat_opts[0])[1]
    with c_stat:
        choice = st.selectbox(
            "Stat", labels, index=labels.index(default_label) if default_label in labels else 0,
            key=f"ma_wk_pos_stat_{defense_team}",
        )
    stat_col, stat_label = stat_opts[labels.index(choice)]
    _render_one_weekly_chart(stats_df, defense_team, pos_choice, stat_col, stat_label)


def _render_one_weekly_chart(stats_df, defense_team, pos, stat_col, stat_label):
    weekly = ms.defense_weekly_allowed(stats_df, defense_team, pos, stat_col, last_n=None)
    if weekly.empty:
        st.caption(f"No week-by-week data for {pos}s allowed by this defense yet.")
        return
    values = weekly['value'].astype(float).tolist()
    team_avg = sum(values) / len(values)

    # Both reference lines live ON the chart now, not one of them in a
    # metric tile above it (the season-number-plus-rank tile that used to
    # sit here is gone per explicit request) - the neutral dashed line is
    # the LEAGUE average allowed across all 32 defenses, the second dashed
    # line in the DEFENSE'S OWN team color is this team's own season
    # average, same "saving the number and rank out of the league" info the
    # old tile carried, just read off the chart instead of a separate tile.
    league_avg = ms.league_average_allowed(stats_df, pos, stat_col)
    team_color = TEAM_CONFIG.get(defense_team, {}).get('color') or C['primary']
    st.markdown(f"**{stat_label} allowed to {pos}s, by week**")
    render_game_log_line(
        values,
        [f"Wk {int(r.week)} vs {r.offense}: {r.value:.1f} {stat_label}" for r in weekly.itertuples()],
        avg=league_avg, avg_label="league avg",
        avg2=team_avg, avg2_label=f"{defense_team} avg", avg2_color=team_color,
        bar_labels=[(str(r.offense), f"W{int(r.week)}") for r in weekly.itertuples()],
    )
    if league_avg is None:
        st.caption(f"No league pool to compare against — dashed line is {defense_team}'s own season average only.")


def _render_coverage(defense_team, season):
    pff = load_all_pff_data(season)
    profile = ms.coverage_profile(
        abbr_to_pff_team(defense_team), _team_label(defense_team),
        load_external_coverage_schemes(), load_sharp_positional_coverage(),
        pff.get('def_coverage_scheme'),
    )
    if not profile['available']:
        return
    _section("COVERAGE")
    scheme = profile['scheme']
    # Man vs Zone, laid out exactly like the player side's own vs-Man/
    # vs-Zone panel (two columns, Man left / Zone right, with the league-
    # relative percentile driving the bar length and color) - per explicit
    # request that the defense's man/zone read match the player's format
    # instead of a flat row of hero tiles with no league context at all.
    man_zone_rows = ms.man_zone_grade_rows(scheme, profile['pff'])
    if man_zone_rows:
        render_split_bars(man_zone_rows, 'Man', 'Zone')
        st.caption("Rate is how often they play each shell; grade and QB rating allowed are PFF's per-defender numbers, snap-weighted. Bar length/color = league percentile.")

    # Two-High / Single-High shell tendency, right below Man vs Zone per
    # explicit request - renamed from "MOF Closed/Open" (middle-of-field
    # closed = one deep safety = SINGLE-HIGH; middle-of-field open = two
    # deep safeties = TWO-HIGH), with each rate's own league percentile so
    # it reads relative to the rest of the league, not as a bare number.
    mof_entries = []
    if scheme.get('mof_open') is not None:
        mof_entries.append({
            'label': 'Two-High %', 'value_str': f"{scheme['mof_open']:.1f}%", 'pct': scheme.get('mof_open_pct'),
        })
    if scheme.get('mof_closed') is not None:
        mof_entries.append({
            'label': 'Single-High %', 'value_str': f"{scheme['mof_closed']:.1f}%", 'pct': scheme.get('mof_closed_pct'),
        })
    if mof_entries:
        st.markdown("**Shell tendency**")
        render_percentile_bar_list(mof_entries, sort=False)
        st.caption("How often the middle of the field is split by two deep safeties (two-high) vs covered by one (single-high), vs the rest of the league.")

    # Slot vs Wide allowed - receptions/targets/yards/TDs, not just a rate,
    # per explicit request ("yds/target is not the whole story ... lacks
    # amount of targets allowed there"). Real PFF slot-alignment defense on
    # the left; "Wide" on the right is computed the same way this tab's Wide
    # YPRR receiver-side stat is (total coverage numbers minus the real slot
    # number) - see data.matchup_signals.defense_alignment_allowed.
    alignment_allowed = ms.defense_alignment_allowed(
        pff.get('cov_summary'), pff.get('slot_cov'), abbr_to_pff_team(defense_team),
    )
    if alignment_allowed['available']:
        st.markdown("**Allowed by alignment**")
        render_split_bars(alignment_allowed['rows'], 'Slot', 'Wide')
        st.caption(
            "Real, measured slot-coverage numbers on the left. \"Wide\" is the defense's total coverage line "
            "minus that real slot number — PFF has no separate outside/wide export, so this is the closest "
            "measured equivalent this app has, not a literal boundary-only stat."
        )


def _render_run_defense(defense_team, season):
    sumer = load_sumersports_tendency_data()
    profile = ms.run_defense_profile(
        load_all_pff_data(season).get('run_def'), abbr_to_pff_team(defense_team),
        sumer.get('def_overview'), _team_label(defense_team),
    )
    if not profile['available'] or not profile['entries']:
        return
    _section("RUN DEFENSE")
    render_percentile_bar_list(profile['entries'], sort=False)
    st.caption(
        "Snap-weighted across the front seven. Higher percentile = softer to run against. "
        "No gap-vs-zone split is shown because no source in this app measures it on the DEFENSE — "
        "see the rusher's own gap/zone split in Scheme Fit below, where it is measured."
    )


def _render_allowed_by_position(stats_df, defense_team, position):
    if position not in ms.ALLOWED_STAT_KEYS:
        return
    _section(f"ALLOWED TO {position}s")
    allowed = ms.defense_allowed_by_position(stats_df, defense_team, position)
    if not allowed['available']:
        st.caption("No allowed-by-position data for this defense yet.")
        return
    render_percentile_bar_list([
        {'label': e['label'], 'value_str': f"{e['value']:.1f}", 'pct': e['pct'],
         'help': f"league avg {e['league_avg']:.1f} · #{e['rank']} of {e['of']}"}
        for e in allowed['entries']
    ], sort=False)
    st.caption(f"Per game across {allowed['games']} games. Higher percentile = softer. Hover for the league average and rank.")


# ---------------------------------------------------------------------------
# Full-width sections - these need both sides
# ---------------------------------------------------------------------------

def _render_scheme_fit(season, player_name, position, defense_team):
    pff = load_all_pff_data(season)
    coverage = ms.coverage_profile(
        abbr_to_pff_team(defense_team), _team_label(defense_team),
        load_external_coverage_schemes(), load_sharp_positional_coverage(),
    )
    run_defense = ms.run_defense_profile(pff.get('run_def'), abbr_to_pff_team(defense_team))
    fit = ms.scheme_fit(position, pff.get('rush'), pff.get('rec'), player_name, coverage, run_defense)
    if not fit['available']:
        return

    _section("SCHEME FIT")
    if fit['kind'] == 'run':
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{player_name}'s carries**")
            render_split_bars([{
                'label': 'Carry split', 'left': fit['gap_rate'], 'right': fit['zone_rate'],
                'left_str': f"{fit['gap_rate']:.0f}%", 'right_str': f"{fit['zone_rate']:.0f}%",
            }], 'Gap', 'Zone')
            render_hero_tiles([
                {'label': 'Yds after contact', 'value': f"{fit['yco_attempt']:.1f}" if fit['yco_attempt'] else '--'},
                {'label': 'Elusive rating', 'value': f"{fit['elusive_rating']:.0f}" if fit['elusive_rating'] else '--'},
                {'label': 'Breakaway %', 'value': f"{fit['breakaway_pct']:.0f}%" if fit['breakaway_pct'] else '--'},
            ])
        with c2:
            st.markdown(f"**{defense_team} run defense**")
            render_percentile_bar_list(fit['defense'], sort=False)
        st.caption(
            "Shown side by side rather than fused into one score. His gap/zone split is measured; the defense's "
            "is not published anywhere this app has, so multiplying the two would invent a comparison."
        )
        return

    delta = fit['fit_score'] - fit['neutral_score']
    verdict = "better than neutral" if delta > 2 else ("worse than neutral" if delta < -2 else "roughly neutral")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{player_name}'s alignment**")
        render_split_bars([{
            'label': 'Snap split', 'left': fit['wide_rate'], 'right': fit['slot_rate'],
            'left_str': f"{fit['wide_rate']:.0f}%", 'right_str': f"{fit['slot_rate']:.0f}%",
        }], 'Wide', 'Slot')
    with c2:
        st.markdown(f"**{defense_team} allows, by alignment**")
        render_split_bars([{
            'label': 'Softness', 'left': fit['defense_outside_pct'], 'right': fit['defense_slot_pct'],
            'left_str': f"{fit['defense_outside_pct']:.0f}", 'right_str': f"{fit['defense_slot_pct']:.0f}",
        }], 'Outside', 'Slot')
    render_hero_tiles([
        {'label': 'Alignment-weighted fit', 'value': f"{fit['fit_score']:.0f}",
         'sub': f"neutral would be {fit['neutral_score']:.0f}", 'accent': get_matchup_color(fit['fit_score'])},
        {'label': 'Verdict', 'value': verdict.split()[0].title(), 'sub': verdict},
    ])
    st.caption(
        "The defense's outside/slot softness, weighted by how often THIS receiver lines up in each spot — "
        "a defense soft over the middle matters far more to a slot receiver than to a boundary X. "
        "Both sides are measured on alignment, so this is a like-for-like comparison."
    )


def _render_anytime_td(stats_df, name_col, player_name, position, defense_team, softness_map):
    if position not in ('QB', 'RB', 'FB', 'WR', 'TE'):
        return
    td_col = 'passing_tds' if position == 'QB' else None
    if td_col is None:
        rush = ms.player_game_series(stats_df, name_col, player_name, 'rushing_tds')
        rec = ms.player_game_series(stats_df, name_col, player_name, 'receiving_tds')
        if rush.empty and rec.empty:
            return
        series = rush.copy() if not rush.empty else rec.copy()
        if not rush.empty and not rec.empty:
            merged = rush.merge(rec, on='week', how='outer', suffixes=('_rush', '_rec'))
            series = pd.DataFrame({
                'week': merged['week'],
                'value': merged['value_rush'].fillna(0) + merged['value_rec'].fillna(0),
            })
    else:
        series = ms.player_game_series(stats_df, name_col, player_name, td_col)
    if series.empty:
        return

    projection = ms.anytime_td_projection(series, softness_map.get(defense_team))
    if not projection['available']:
        return
    _section("ANYTIME TOUCHDOWN")
    render_hero_tiles([
        {'label': 'P(scores a TD)', 'value': f"{projection['probability'] * 100:.0f}%",
         'accent': get_pff_color(projection['probability'] * 100)},
        {'label': 'His TD rate', 'value': f"{projection['base_rate']:.2f}", 'sub': 'per game'},
        {'label': 'Matchup adj.', 'value': f"{(projection['adjustment'] - 1) * 100:+.0f}%",
         'sub': f"vs {defense_team}"},
        {'label': 'Season TDs', 'value': f"{projection['total_tds']:.0f}",
         'sub': f"in {projection['games']} games"},
    ])
    st.caption(
        "Poisson draw on his own per-game touchdown rate, nudged by how soft this defense is to his position "
        "(capped at ±25% — that softness number is a season-long fantasy signal, not a touchdown-specific one). "
        "Poisson rather than a plain hit rate because touchdowns are bursty: two in one game shouldn't read as two separate scoring games."
    )


def _render_prop_analysis(season, stats_df, name_col, player_name, position, offense_team, defense_team, selected_stat):
    options = ms.GAME_LOG_STATS.get(position)
    if position not in _POSITION_MARKETS or not options:
        return
    _section("PROP ANALYSIS")
    # Its own stat picker, not a read of Game By Game's selection above -
    # explicit user request, so checking a different line doesn't mean
    # scrolling back up to change that dropdown. Defaults to whatever's
    # selected up there for continuity on first load.
    stat_labels = [label for _, label in options]
    default_label = selected_stat[1] if selected_stat and selected_stat[1] in stat_labels else stat_labels[0]
    choice = st.selectbox(
        "Stat", stat_labels, index=stat_labels.index(default_label), key="ma_prop_stat",
    )
    stat_col, stat_label = options[stat_labels.index(choice)]
    series = ms.player_game_series(stats_df, name_col, player_name, stat_col)
    if series.empty:
        st.caption("No game log to check a line against.")
        return

    values = series['value'].astype(float)
    line = st.number_input(
        f"{stat_label} line", min_value=0.0, value=float(round(values.mean() * 2) / 2), step=0.5,
        key="ma_prop_line", help="Type a book's line to see how often he actually cleared it this season.",
    )
    overs = int((values > line).sum())
    st.markdown(f"**{overs} of {len(values)}** games over {line:g} ({overs / len(values) * 100:.0f}%)")
    render_game_log_bars(
        values.tolist(),
        [f"Wk {int(r.week)} vs {r.opponent}: {r.value:.1f}" for r in series.itertuples()],
        highlight=[v > line for v in values],
        avg=line, avg_label='line',
    )
    st.caption("★ = cleared the line. Dashed line = the line itself, not the season average.")

    st.caption(
        f"Season avg {values.mean():.1f} · floor (25th) {values.quantile(0.25):.1f} · "
        f"ceiling (75th) {values.quantile(0.75):.1f} · std dev {values.std():.1f}"
    )

    api_key = load_saved_odds_api_key()
    if not api_key:
        st.caption("Add an Odds API key on the Live Odds tab to pull real book lines here.")
        return
    st.caption("Live lines are one paid API call per matchup, so they're behind a button rather than automatic.")
    if not st.button("Load live prop lines", key="ma_load_props"):
        return
    with st.spinner("Fetching live props…"):
        _render_live_props(api_key, player_name, position, offense_team, defense_team)


def _render_live_props(api_key, player_name, position, offense_team, defense_team):
    events = fetch_nfl_odds(api_key)
    if not events:
        st.info("No NFL events returned by the Odds API right now.")
        return
    wanted = {_team_label(offense_team), _team_label(defense_team)}
    event = next(
        (e for e in events if wanted & {e.get('home_team'), e.get('away_team')}),
        None,
    )
    if not event:
        st.info(f"No upcoming market found involving {offense_team} or {defense_team}.")
        return
    props = fetch_nfl_player_props(api_key, event.get('id'), ','.join(_POSITION_MARKETS[position]))
    if not props:
        st.info("The book has no player props posted for this game yet.")
        return

    rows = []
    for book in props.get('bookmakers', []):
        for market in book.get('markets', []):
            for outcome in market.get('outcomes', []):
                if str(outcome.get('description', '')).lower() != player_name.lower():
                    continue
                price = outcome.get('price')
                rows.append({
                    'Book': book.get('title'), 'Market': market.get('key'),
                    'Side': outcome.get('name'), 'Line': outcome.get('point'),
                    'Price': price,
                    'Implied %': round(american_odds_to_prob(price) * 100, 1) if price is not None else None,
                })
    if not rows:
        st.info(f"No posted props for {player_name} in this game.")
        return
    from ui.styling import df_auto_height, style_plain_dataframe
    table = pd.DataFrame(rows)
    st.dataframe(style_plain_dataframe(table.set_index('Book')), width="stretch", height=df_auto_height(len(rows)))
    st.caption("Implied % includes the book's own vig — it's one side of a two-way market, not a de-vigged probability.")
