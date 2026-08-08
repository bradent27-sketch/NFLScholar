"""
Reusable UI pieces shared across tabs: the player bio card, the sticky-footer
game log (weekly rows scroll, season-average row stays pinned to the bottom
of the viewport), and the sidebar data-health diagnostics panel.
"""
import math
import os
from contextlib import contextmanager

import pandas as pd
import streamlit as st

from config import THEME, TEAM_CONFIG, get_position_color
from ui.styling import render_game_log_html_table

C = THEME['colors']
F = THEME['fonts']
R = THEME['radius']


def _skel_cells(n):
    return ''.join(f"<div class='skel-cell' style='animation-delay:{i * 45}ms;'></div>" for i in range(n))


def _render_skeleton_table(n_rows=8, n_cols=6):
    header = f"<div class='skel-row skel-header-row'>{_skel_cells(n_cols)}</div>"
    body = ''.join(
        f"<div class='skel-row' style='animation-delay:{i * 35}ms;'>{_skel_cells(n_cols)}</div>"
        for i in range(n_rows)
    )
    st.markdown(f"<div class='skel-table-wrap'>{header}{body}</div>", unsafe_allow_html=True)


def _render_skeleton_tiles(n=8):
    tiles = ''.join(f"<div class='skel-tile' style='animation-delay:{i * 40}ms;'></div>" for i in range(n))
    st.markdown(f"<div class='skel-tile-grid'>{tiles}</div>", unsafe_allow_html=True)


def _render_skeleton_block(height=220):
    st.markdown(f"<div class='skel-block' style='height:{height}px;'></div>", unsafe_allow_html=True)


@contextmanager
def skeleton_loader(kind='table', **kwargs):
    """
    Drop-in replacement for `with st.spinner(...):` around a slow
    @st.cache_data season/odds load - shows a shimmering placeholder shaped
    like the content about to appear (a table, a stat-tile grid, or a plain
    card block) instead of a bare spinner, then clears it once the `with`
    block finishes. Only ever visible on an actual cache miss (a fresh
    season/scoring/game combo) - same as the st.spinner it replaces, since
    the wrapped call is still whatever @st.cache_data function was already
    there.

    kind: 'table' (default), 'tiles', or 'block' - matched by shape to
    what's actually loading in each call site (a stats table vs. a
    card/list-shaped fetch like live odds).
    """
    placeholder = st.empty()
    with placeholder:
        if kind == 'tiles':
            _render_skeleton_tiles(**kwargs)
        elif kind == 'block':
            _render_skeleton_block(**kwargs)
        else:
            _render_skeleton_table(**kwargs)
    try:
        yield
    finally:
        placeholder.empty()


def get_drafted_players_clean_keys():
    """
    Clean-name keys (see data.utils.clean_name_exact) for the app-wide
    drafted set (st.session_state['drafted_players'], kept in step by
    Draft HQ from its own draft state - Sleeper sync, pasted picks, manual
    marking, or a mock), so Player Search and Rookie Watch can filter
    drafted players out without needing an exact string match against
    whatever name format each tab's own data happens to use - the set
    stores names in the draft board's format, which isn't guaranteed
    identical to every other tab's.
    """
    from data.utils import clean_name_exact
    drafted = st.session_state.get('drafted_players', set())
    if not drafted:
        return set()
    return set(clean_name_exact(pd.Series(list(drafted))))


@st.cache_data
def compute_bye_weeks(year):
    """
    Team -> bye week number, derived from the schedule itself (the one week
    in the season a team appears in neither home_team nor away_team) rather
    than a separately-maintained lookup table - works for any season with a
    published schedule, played or not, since a bye is a scheduling fact, not
    a results fact. Shared by Player Search and Player Compare (both build
    the same "name (POS - TEAM) | Bye: N" dropdown label).

    Cached on `year`: the 32-team loop below re-ran on every single rerun
    of Player Search/Player Compare (any widget interaction) even though a
    season's bye weeks never change within a session.
    """
    from data.loaders import load_schedule
    schedule_df = load_schedule(year)
    if schedule_df.empty:
        return {}
    all_weeks = sorted(schedule_df['week'].unique())
    all_teams = pd.concat([schedule_df['home_team'], schedule_df['away_team']]).dropna().unique()
    bye_map = {}
    for team in all_teams:
        team_weeks = set(schedule_df.loc[(schedule_df['home_team'] == team) | (schedule_df['away_team'] == team), 'week'])
        off_weeks = [w for w in all_weeks if w not in team_weeks]
        if off_weeks:
            bye_map[team] = int(off_weeks[0])
    return bye_map


def build_player_search_labels(stats_df, n_col, t_col, bye_weeks, team_filter="All Teams", position_group=None, hide_drafted=True):
    """
    Builds the "Justin Jefferson (WR - MIN) | Bye: 6"-style dropdown label
    list + a label -> plain-name lookup, shared by Player Search's own
    autocomplete and the Player Compare tab's two independent player
    pickers (previously only lived inline in Player Search's render()).

    `position_group`, when given, is an iterable of position codes (e.g.
    `['WR', 'TE']`) - Compare uses this to restrict its SECOND picker to
    the same stat-branch group as the first player, since a QB-vs-RB
    comparison would render an almost-empty shared-stat table.

    Returns (labels, label_to_name) - `labels` includes a sentinel
    "No players found" entry when the filtered pool is empty, matching
    Player Search's existing empty-state convention.
    """
    label_cols = [n_col, t_col] + (['position'] if 'position' in stats_df.columns else [])
    all_players = stats_df[label_cols].dropna(subset=[n_col]).drop_duplicates(subset=[n_col])
    all_players = all_players[all_players[n_col].astype(str).str.strip().ne('')]
    if team_filter and team_filter != "All Teams":
        all_players = all_players[all_players[t_col].astype(str).str.upper() == team_filter]
    if position_group and 'position' in all_players.columns:
        all_players = all_players[all_players['position'].astype(str).str.upper().isin(position_group)]

    # Vectorized label construction - the previous per-row iterrows() loop
    # over the full ~2,000-player league pool ran on EVERY rerun of Player
    # Search (and twice per rerun on Player Compare), which was a real,
    # felt per-click lag; whole-column string ops build the same labels in
    # roughly a millisecond.
    all_players = all_players.sort_values(n_col)
    names = all_players[n_col].astype(str)
    teams = all_players[t_col].astype(str).where(all_players[t_col].notna(), '--')
    if 'position' in all_players.columns:
        pos = all_players['position'].astype(str).str.upper().where(all_players['position'].notna(), '')
        pos_part = (pos + ' - ').where(pos.ne('') & pos.ne('NAN'), '')
    else:
        pos_part = pd.Series('', index=all_players.index)
    byes = pd.to_numeric(teams.map(bye_weeks), errors='coerce').astype('Int64')
    bye_part = (' | Bye: ' + byes.astype(str)).where(byes.notna(), '')
    label_series = names + ' (' + pos_part + teams + ')' + bye_part
    labels = label_series.tolist()
    label_to_name = dict(zip(labels, names))

    if hide_drafted:
        from data.utils import clean_name_exact
        drafted_keys = get_drafted_players_clean_keys()
        if drafted_keys and labels:
            name_keys = clean_name_exact(pd.Series([label_to_name[l] for l in labels]))
            labels = [l for l, k in zip(labels, name_keys) if k not in drafted_keys]

    if not labels:
        labels = ["No players found"]
    return labels, label_to_name


def switch_tab(tab_label, **context):
    """
    Programmatically switches the active tab (Streamlit >=1.59's st.tabs
    key-based state - see app.py and config.TAB_LABELS) and stashes any
    extra navigation context (e.g. switch_tab(TAB_PLAYER_SEARCH,
    jump_to_player="Christian McCaffrey")) into session_state for the
    destination tab to read on its next render. Destination tabs should
    consume their context key with st.session_state.pop(...) (not .get())
    so a one-time jump doesn't keep re-triggering on every later visit to
    that tab.

    MUST be invoked as a widget callback (on_click=switch_tab or
    on_select=<a closure that calls this>), never called directly inline
    after checking a widget's return value. app.py's st.tabs(key=
    "active_tab", ...) is already instantiated by the time any tab's
    render() body runs, and Streamlit raises "cannot be modified after the
    widget... is instantiated" if you assign to st.session_state["active_tab"]
    at that point in the normal script flow - callbacks run in Streamlit's
    separate pre-script phase, before the next run's st.tabs() call, which
    is the only place this assignment is actually legal. Callbacks also
    auto-trigger their own rerun afterward, so this deliberately does NOT
    call st.rerun() itself - doing so from inside a callback is redundant
    and unsupported.
    """
    # Records where we're jumping FROM so render_back_button() can offer a
    # way back - read before overwriting active_tab below, since a callback
    # is the only point that ever sees "the tab the user was just looking
    # at". Skipped when jumping to the same tab (shouldn't happen in
    # practice) so a back button never points at itself.
    current_tab = st.session_state.get('active_tab')
    if current_tab and current_tab != tab_label:
        st.session_state['nav_back_tab'] = current_tab
    st.session_state['active_tab'] = tab_label
    for key, value in context.items():
        st.session_state[key] = value


def render_back_button():
    """
    "<- Back to X" button shown at the top of a tab when it was reached via
    switch_tab (e.g. clicking a player row on Risers/Rookie Watch/Depth
    Charts jumps to Player Search) - lets you return to wherever you came
    from instead of re-navigating by hand. Not popped here - it keeps
    showing across reruns of the same tab (e.g. adjusting a filter) until
    either clicked (switch_tab then overwrites nav_back_tab with THIS tab,
    so it becomes a two-way toggle) or superseded by a fresh jump from
    somewhere else.
    """
    back_tab = st.session_state.get('nav_back_tab')
    if back_tab:
        st.button(f"← Back to {back_tab}", key="nav_back_btn", on_click=switch_tab, args=(back_tab,))


@st.cache_resource
def ensure_pff_imports_dir():
    """
    One-time local setup, not page layout - kept out of app.py's top-level
    script body (which Streamlit re-executes on every rerun, i.e. every
    widget interaction anywhere in the app) via st.cache_resource, so the
    filesystem check actually only ever runs once per server process
    instead of once per rerun.
    """
    if not os.path.exists("pff_imports"):
        os.makedirs("pff_imports")


# nflverse's own generic league shield (teams_colors_logos.csv's
# team_league_logo column - identical across all 32 rows, confirmed) -
# same GitHub org this app already sources team colors/pbp data from.
_NFL_LOGO_URL = "https://raw.githubusercontent.com/nflverse/nflverse-pbp/master/NFL.png"

_STAT_GLOSSARY_MD = """
| Term | Meaning |
|---|---|
| **PFF grade** | Pro Football Focus's 0-100 game-charting grade for a player |
| **Snap %** | Share of the team's offensive/defensive snaps a player was on the field for |
| **PRP** | Pass Rush Productivity - PFF's pressure-generation rate for pass rushers, weighted toward sacks |
| **MTF** | Missed Tackles Forced - times a runner made a defender miss |
| **YCO / Att** | Yards After Contact per rush attempt |
| **BTT % / TWP %** | Big-Time Throw rate / Turnover-Worthy Play rate (accurate high-value throws vs. should've-been-picked throws) |
| **RecV Grade** | PFF's receiving / route-running grade |
| **PBLK Grade** | PFF's pass-blocking grade |
| **Run Stop Rate** | Run plays a defender stopped at or behind the line to gain, as a % of run snaps |
| **Missed Tkl Rate** | Missed tackles as a % of tackle attempts |
| **Pass Rtg Allowed** | Opposing QB's passer rating when targeting that defender |
| **ESPN QBR** | ESPN's Total Quarterback Rating (0-100, accounts for game situation) |
| **YPRR** | Yards Per Route Run - receiving efficiency, not volume |
| **ADOT** | Average Depth of Target |
| **EPA** | Expected Points Added - how much a play's value beat expectation for that down/distance/field position |
| **VORP** | Value Over Replacement Player - this app's draft-value metric vs. a replacement-level player at the position |
| **RYOE** | Rush Yards Over Expected (NFL Next Gen Stats) - actual rush yards vs. expected given blockers/defenders in the box |
| **SOS** | Strength of Schedule |
| **MOF Open/Closed %** | Middle-of-field open (single-high safety) vs. closed (two-high/split-safety) coverage rate |
| **YPT** | Yards Per Target allowed |
"""


def render_intro_and_glossary():
    """
    App header - pulled out of app.py so that file stays what its own
    docstring says it should be (page config, theme injection, tab
    wiring), not a growing block of markdown content. Just the title line;
    the stat glossary lives in the sidebar now (render_data_health_sidebar),
    next to Local Data Health rather than taking up main-content space.

    Also renders the Full PPR/Half-PPR/Standard segmented control, top-right
    in line with the title - placed here (not inside player_search.py) per
    explicit request, even though it only actually drives Player Search's
    own scoring: this function runs once, before any tab's content, so a
    key="score_tab1" widget instantiated here satisfies Streamlit's
    "session_state[key] must be set before that widget is instantiated"
    rule for player_search.py's render() to simply read
    st.session_state['score_tab1'] later without instantiating a second,
    competing widget under the same key.
    """
    col_title, col_scoring = st.columns([3, 2])
    with col_title:
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:12px; margin-top:0;'>"
            f"<img src='{_NFL_LOGO_URL}' style='height:34px; width:auto; filter: drop-shadow(0 2px 6px rgba(0,0,0,0.5));'>"
            f"<div>"
            f"<div style='font-family:{F['display']}; font-size:21px; font-weight:800; letter-spacing:-0.02em; line-height:1.05; color:#ffffff;'>"
            f"NFL <span style='color:{C['primary']};'>SCHOLAR</span></div>"
            f"<div style='font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:0.14em; color:{C['on_surface_variant']}; margin-top:1px;'>"
            f"Fantasy Analytics &amp; Matchup Intelligence</div>"
            f"</div></div>", unsafe_allow_html=True,
        )
    with col_scoring:
        st.segmented_control(
            "Scoring", ["Full PPR", "Half-PPR", "Standard"], default="Full PPR",
            key="score_tab1", label_visibility="collapsed", required=True,
        )


def render_stat_tiles(entries):
    """
    Baseball-Savant-style stat tile grid from build_player_snapshot-style
    entries ({'label','value_str','pct','color'}): each stat is its own
    card with the value in mono, the league percentile in a colored corner
    bubble, and a thin percentile meter along the tile's bottom edge -
    replaces the old single-row horizontal-scroll HTML table, which forced
    sideways scrolling on wide positions (a WR row had 18+ columns) and
    read as a raw spreadsheet fragment rather than a stat panel. Same
    entries, same colors, same percentiles - only the presentation changed.

    Entries with pct=None (no real percentile - see ui.player_snapshot.
    _entry) render as a plain tile with no bubble/meter fill, matching the
    old table's neutral-cell treatment for those stats.
    """
    tiles = []
    for e in entries:
        label = str(e.get('label', ''))
        value = str(e.get('value_str', '--'))
        pct = e.get('pct')
        color = e.get('color') or C['surface_container_high']
        if pct is not None:
            pct_clamped = max(0.0, min(100.0, float(pct)))
            bubble = f"<div class='t-pct' style='background:{color};'>{int(round(pct_clamped))}</div>"
            meter = (
                "<div class='t-meter-track'></div>"
                f"<div class='t-meter' style='width:{max(2.0, pct_clamped):.0f}%; background:{color};'></div>"
            )
        else:
            bubble = ""
            meter = "<div class='t-meter-track'></div>"
        tiles.append(
            f"<div class='stat-tile'><div class='t-label' title='{label}'>{label}</div>"
            f"<div class='t-value'>{value}</div>{bubble}{meter}</div>"
        )
    st.markdown(f"<div class='stat-tile-grid'>{''.join(tiles)}</div>", unsafe_allow_html=True)


def render_hero_tiles(items):
    """
    Row of large headline-number tiles (fantasy PPG, position rank, etc).
    `items`: list of {'label', 'value', 'sub' (optional), 'accent'
    (optional CSS color for the value)}.
    """
    tiles = []
    for it in items:
        accent = it.get('accent')
        value_style = f" style='color:{accent};'" if accent else ""
        sub = f"<div class='h-sub'>{it['sub']}</div>" if it.get('sub') else ""
        tiles.append(
            f"<div class='hero-tile'><div class='h-label'>{it['label']}</div>"
            f"<div class='h-value'{value_style}>{it['value']}</div>{sub}</div>"
        )
    st.markdown(f"<div class='hero-tile-grid'>{''.join(tiles)}</div>", unsafe_allow_html=True)


def render_percentile_metric_tiles(items):
    """
    Percentile-colored metric tiles (Defensive Yield's Coverage Matchup
    Radar) - replaces st.metric, which has no per-instance styling hook a
    caller can drive from Python, so a per-tile percentile color was never
    reachable through it. Centered layout like render_hero_tiles, but
    `color` is REQUIRED per item and drives the tile's own background
    (a solid percentile fill, the same convention every other
    percentile-colored surface in this app uses - table cells via
    style_plain_dataframe/style_depth_chart_table - not a fixed gradient).

    A separate function rather than extending render_hero_tiles with this
    same behavior - render_hero_tiles already has callers (Player Search's
    headline band) that only tint the VALUE text via `accent`, never the
    tile background; retrofitting shared per-tile background coloring onto
    that function risked changing those unrelated tiles' look too.

    `items`: list of {'label', 'value', 'sub' (optional), 'color'} - color
    is precomputed by the caller (get_pff_color for a player-performance
    percentile, get_matchup_color for a defense/tendency percentile - see
    ui.tabs.defensive_yield._render_coverage_radar for which is used
    where), keeping this a pure render function, same layering as
    render_stat_tiles/render_hero_tiles.
    """
    tiles = []
    for it in items:
        sub = f"<div class='m-sub'>{it['sub']}</div>" if it.get('sub') else ""
        tiles.append(
            f"<div class='metric-tile' style='background:{it['color']};'>"
            f"<div class='m-label'>{it['label']}</div>"
            f"<div class='m-value'>{it['value']}</div>{sub}</div>"
        )
    st.markdown(f"<div class='metric-tile-grid'>{''.join(tiles)}</div>", unsafe_allow_html=True)


def _smooth_svg_path(pts):
    """
    SVG path string that curves THROUGH every point in `pts` (a uniform
    Catmull-Rom spline converted to cubic Bezier segments, tension 1/6) -
    an interpolating curve, not an approximating one, so the line still
    passes exactly through each week's real value instead of just trending
    near it. Endpoints are clamped by duplicating the first/last point as
    their own "neighbor" (the standard fix for a Catmull-Rom spline having
    no real neighbor to look at past the ends of the series).
    """
    if len(pts) < 3:
        # A straight segment (or a single point with nothing to connect)
        # doesn't need curve math - two points are already a straight line
        # either way.
        parts = [f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"]
        if len(pts) == 2:
            parts.append(f"L{pts[1][0]:.1f},{pts[1][1]:.1f}")
        return " ".join(parts)
    ext = [pts[0]] + list(pts) + [pts[-1]]
    d = [f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"]
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        cp1x, cp1y = p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0
        cp2x, cp2y = p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0
        d.append(f"C{cp1x:.1f},{cp1y:.1f} {cp2x:.1f},{cp2y:.1f} {p2[0]:.1f},{p2[1]:.1f}")
    return " ".join(d)


def _nice_y_ticks(max_val, target_count=4):
    """
    A handful of round-number tick values from 0 up to (at least) max_val -
    e.g. [0, 10, 20, 30] rather than the raw max sliced evenly, which would
    label a fantasy-points axis with ugly values like "27.3". Standard
    "nice number" step selection (round the raw max_val/target_count step
    up to the nearest 1/2/5/10 x a power of ten), same idea matplotlib's
    default tick locator uses.
    """
    if max_val <= 0:
        return [0]
    raw_step = max_val / target_count
    magnitude = 10 ** math.floor(math.log10(raw_step))
    residual = raw_step / magnitude
    if residual > 5:
        step = 10 * magnitude
    elif residual > 2:
        step = 5 * magnitude
    elif residual > 1:
        step = 2 * magnitude
    else:
        step = magnitude
    ticks = []
    v = 0.0
    while v <= max_val + step * 0.01:
        ticks.append(v)
        v += step
    return ticks


def render_fpts_week_strip(p_data, year, scoring_label=""):
    """
    Full-season fantasy-points-by-week LINE chart (Player Search) - shown
    first, above the hero tiles, per explicit feedback that the boards'
    own "Last 5 Wks" sparkline column (st.column_config.LineChartColumn on
    Risers/Rookie Watch) is the easiest-to-read-at-a-glance chart style in
    the app and the season trend here should lead with the same idea: one
    connected line instead of a bar per week. Each point is still colored
    by that week's league percentile (fantasy_points_wk_pct via
    get_pff_color, same scale as everywhere else) so the color signal from
    the old bar strip survives.

    Pure inline SVG (not matplotlib/Vega-Lite): a fixed viewBox with the
    default preserveAspectRatio (uniform xMidYMid scaling) so dots/text
    scale together and never stretch into ellipses/squashed glyphs the way
    a `preserveAspectRatio="none"` chart would at an arbitrary column
    width - width is 100% of the container, height follows from the fixed
    viewBox aspect ratio. Hover a point for the full
    "Wk N vs OPP — X pts" detail (native SVG <title>, same info the old
    bar strip's div title= carried).

    Silently renders nothing for seasons with no weekly data (no 'week'
    column - e.g. 2026 pre-kickoff) or a player with no played weeks.
    """
    from config import THEME
    from ui.styling import get_pff_color
    if 'week' not in p_data.columns or 'fantasy_points' not in p_data.columns:
        return
    df = p_data.copy()
    df['week'] = pd.to_numeric(df['week'], errors='coerce')
    df = df[df['week'] > 0].sort_values('week')
    if df.empty:
        return
    fpts = pd.to_numeric(df['fantasy_points'], errors='coerce').fillna(0).tolist()
    weeks = df['week'].astype(int).tolist()
    opps = df['opponent_team'].astype(str).tolist() if 'opponent_team' in df.columns else [''] * len(df)
    pcts = df['fantasy_points_wk_pct'].tolist() if 'fantasy_points_wk_pct' in df.columns else [0] * len(df)

    n = len(fpts)
    max_fpts = max(fpts)
    if max_fpts <= 0:
        return

    # Height cut roughly a third (220->140) per explicit feedback that this
    # chart ran too tall for what's meant to be a quick glance up top, not
    # a full analytical chart.
    W, H = 1000, 140
    # pad_l widened from a bare 14 to fit the y-axis tick labels added below
    # (a 1-2 digit fantasy-point value plus its tick mark) without crowding
    # the first week's point.
    pad_l, pad_r, pad_t, pad_b = 30, 14, 12, 26
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b
    y_max = max_fpts * 1.15  # headroom so the top point never touches the edge

    def x_at(i):
        return pad_l if n == 1 else pad_l + plot_w * i / (n - 1)

    def y_at(v):
        return pad_t + plot_h - plot_h * (v / y_max)

    pts = [(x_at(i), y_at(v)) for i, v in enumerate(fpts)]
    # Smooth interpolating curve (Catmull-Rom -> Bezier) instead of straight
    # segments between weeks, per explicit feedback - still passes through
    # every real weekly value, just reads as one flowing line rather than a
    # discrete connect-the-dots polyline.
    line_path = _smooth_svg_path(pts)
    area_path = f"{line_path} L{pts[-1][0]:.1f},{pad_t + plot_h:.1f} L{pts[0][0]:.1f},{pad_t + plot_h:.1f} Z"

    C = THEME['colors']
    dots, wk_labels = [], []
    for i, (x, y) in enumerate(pts):
        wk, val, opp, pct = weeks[i], fpts[i], opps[i].strip().upper(), pcts[i]
        color = get_pff_color(pct)
        opp_part = f" vs {opp}" if opp and opp != 'NAN' else ""
        dots.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="{C["surface_container"]}" stroke-width="1.5">'
            f'<title>Wk {wk}{opp_part} — {val:.1f} pts</title></circle>'
        )
        # Every week when there's room to read them; thin out to every other
        # week on a full 17/18-game season so labels don't collide.
        if n <= 10 or i % 2 == 0 or i == n - 1:
            wk_labels.append(f'<text x="{x:.1f}" y="{H - 6}" text-anchor="middle" class="fl-wklabel">{wk}</text>')

    # Faint y-axis - a handful of round-number tick marks/labels (0, 10, 20,
    # ...) so a value can be referenced against the axis when needed,
    # without the chart reading as a "real" analytical axis - just the
    # vertical line plus short tick marks, no full-width gridlines, and
    # drawn before the area/line/dots below so the translucent area fill
    # mutes it slightly further. Per explicit feedback this should stay in
    # the background, not compete with the line/dots for attention.
    axis_parts = [
        f'<line x1="{pad_l:.1f}" y1="{pad_t:.1f}" x2="{pad_l:.1f}" y2="{pad_t + plot_h:.1f}" class="fl-yaxis-line"/>'
    ]
    for tick_v in _nice_y_ticks(max_fpts):
        if tick_v > y_max:
            continue
        ty = y_at(tick_v)
        tick_label = f"{tick_v:.0f}" if tick_v == int(tick_v) else f"{tick_v:.1f}"
        axis_parts.append(f'<line x1="{pad_l - 4:.1f}" y1="{ty:.1f}" x2="{pad_l:.1f}" y2="{ty:.1f}" class="fl-ytick"/>')
        axis_parts.append(f'<text x="{pad_l - 7:.1f}" y="{ty + 3:.1f}" text-anchor="end" class="fl-yticklabel">{tick_label}</text>')
    axis_svg = ''.join(axis_parts)

    svg = (
        f'<svg viewBox="0 0 {W} {H}" class="fpts-linechart" role="img" aria-label="Fantasy points by week">'
        f'<defs><linearGradient id="fl-fill-{year}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{C["primary"]}" stop-opacity="0.30"/>'
        f'<stop offset="100%" stop-color="{C["primary"]}" stop-opacity="0"/>'
        f'</linearGradient></defs>'
        f'{axis_svg}'
        f'<path d="{area_path}" fill="url(#fl-fill-{year})" stroke="none"/>'
        f'<path d="{line_path}" fill="none" stroke="{C["primary"]}" stroke-width="2.5" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'{"".join(dots)}{"".join(wk_labels)}</svg>'
    )

    label_suffix = f" · {scoring_label}" if scoring_label else ""
    st.markdown(
        f"<div class='fpts-strip'><div class='fs-label'>Fantasy Points by Week — {year}{label_suffix}</div>{svg}</div>",
        unsafe_allow_html=True,
    )


def render_matchup_heat_strip(rows, pos, year):
    """
    Rest-of-season matchup-difficulty strip (Player Search) - one compact
    cell per remaining game, colored by how many fantasy points that
    opponent allows to this player's position. Same get_matchup_color
    (light green=soft/red=tough) and same percentile DIRECTION (ascending -
    a higher points-allowed number is a softer matchup) as Defensive
    Yield's own rest-of-season strength-of-schedule table
    (data.transforms.build_strength_of_schedule /
    ui.tabs.defensive_yield._render_strength_of_schedule) - a user who's
    already learned that convention there sees the identical read here,
    just scoped to one player's own remaining slate instead of a
    league-wide table. Reuses the .fpts-strip/.fs-label card chrome (same
    "compact glanceable widget" family as render_fpts_week_strip above)
    rather than introducing a new container style.

    `rows`: list of {'week', 'opponent', 'pct', 'raw_pts'} - `pct` (0-100
    or None) and `raw_pts` are computed by the caller from
    build_points_allowed_matrix + data.utils.calculate_percentile, since
    that's already a data/ concern, not a ui/ one. Silently renders nothing
    if `rows` is empty - same convention as render_fpts_week_strip for a
    player/season with nothing to show (e.g. the season is already over).
    """
    from ui.styling import get_matchup_color
    if not rows:
        return
    cells = []
    for r in rows:
        wk, opp = r['week'], r['opponent']
        pct = r.get('pct')
        color = get_matchup_color(pct) if pct is not None else C['surface_container_high']
        raw_pts = r.get('raw_pts')
        pts_txt = f"{raw_pts:.1f} pts allowed to {pos}s this season" if raw_pts is not None else "no matchup data yet"
        title_txt = f"Wk {wk} vs {opp} — {pts_txt}"
        cells.append(
            f"<div class='ms-cell' style='background:{color};' title='{title_txt}'>"
            f"<div class='ms-wk'>WK {wk}</div>"
            f"<div class='ms-opp'>{opp}</div>"
            f"</div>"
        )
    st.markdown(
        f"<div class='fpts-strip'><div class='fs-label'>Rest-of-Season Matchups vs. {pos} Defenses — {year}</div>"
        f"<div class='matchup-strip-row'>{''.join(cells)}</div></div>",
        unsafe_allow_html=True,
    )


def render_matchup_title(player_label, team_label, team_color=None):
    """
    Centered "{player} VS {team}" header for a matchup visual (Defensive
    Yield's Coverage Correlator radar) - replaces a bare **bold** markdown
    line (default Streamlit paragraph styling, cramped against the chart
    right below it) with typography sized/spaced/colored to match the rest
    of the app: player name and team name each read as their own headline
    word, a pill-shaped "VS" keeps them from running into each other, and
    the team side picks up that team's real color instead of plain white.
    """
    color = team_color or C['secondary']
    st.markdown(
        f"<div class='matchup-title'>"
        f"<span class='mt-player'>{player_label}</span>"
        f"<span class='mt-vs'>VS</span>"
        f"<span class='mt-team' style='color:{color};'>{team_label}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_team_banner(team_abbr, subtitle=""):
    """
    Team identity banner (Depth Charts): logo + full team name over a
    team-color gradient that fades into the app surface - the standard
    "team page" header treatment on pro sports sites. Logo comes from the
    local teams_colors_logos.csv via load_team_logos() (ESPN CDN, same
    host as the player headshots already used app-wide).
    """
    from data.loaders import load_team_logos
    cfg = TEAM_CONFIG.get(team_abbr, {})
    name = cfg.get('name', team_abbr)
    color = cfg.get('color', C['surface_container_high'])
    logo_url = load_team_logos().get(team_abbr, '')
    logo_html = f"<img src='{logo_url}' alt='{team_abbr}'>" if logo_url else ""
    sub_html = f"<div class='tb-sub'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"<div class='team-banner' style='background: linear-gradient(90deg, {color}D9 0%, {color}66 40%, {C['surface_container']} 100%);'>"
        f"{logo_html}<div><div class='tb-name'>{name}</div>{sub_html}</div></div>",
        unsafe_allow_html=True,
    )


def render_player_card(selected_player, pos, grade_str, grade_color, team_abbr, team_color, jersey_val, headshot_url, is_gold=False):
    """
    is_gold: this player is the #1 PFF-graded player at their position for
    the selected season (data.loaders._build_master_pff_grades' per-position
    league_gold_players set) - same league-leader signal the depth chart's
    gold cell border uses.

    Recolors the card's OWN existing border (outer frame + name-strip's
    border-top) to gold instead of grade_color, rather than layering a
    separate outline onto just the headshot <img> - an earlier version did
    that and the outline was largely invisible in practice: it sat right on
    top of the outer border on the sides, got clipped by the card's own
    overflow:hidden past the visible photo area, and was hidden under the
    absolute-positioned grade/team badges up top. Recoloring the
    already-working outer border guarantees one continuous, clearly visible
    gold frame around the whole card (headshot and name both included).

    No longer takes a snapshot/embedded micro-radar - that small radar
    (fused onto the card face below the name strip) was removed per user
    feedback: it only ever showed the same headline stats the (now always-
    visible, un-demoted) percentile bar chart shows anyway, at a size too
    small to read comfortably, so it was space the card wasn't using well.
    """
    card_border_color = C['tertiary'] if is_gold else grade_color
    card_glow = f"0px 8px 28px rgba(0,0,0,0.55), 0 0 18px {C['tertiary']}" if is_gold else "0px 8px 28px rgba(0,0,0,0.55)"
    name_suffix = " 🏆" if is_gold else ""
    # Position label recolored to a fixed position-identity accent instead
    # of grade_color - the grade signal is still fully carried by grade_str
    # (the big number) and the card's own border/glow, so the position text
    # is free to encode a second, position-fixed signal (same color for
    # every QB regardless of that player's individual grade) instead of
    # duplicating the grade color that's already shown two ways.
    pos_color = get_position_color(pos)

    # Corner radii tightened from the original 12px/6px (a technical scouting
    # card reads denser than a soft consumer-app card) - kept just above
    # zero rather than a hard square, since a fully square corner clashes
    # visually against a circular headshot cutout at this card's size.
    card_html = f"""
    <div class="player-card" style="position: relative; width: 100%; max-width: 320px; margin: 0 auto; background: linear-gradient(180deg, {C['surface_container']} 0%, {C['surface_dim']} 100%); border: 4px solid {card_border_color}; border-radius: {R['default']}; box-shadow: {card_glow}; overflow: hidden; display: flex; flex-direction: column;">
        <div style="position: absolute; top: 12px; left: 12px; background: rgba(2,4,9,0.85); border: 1px solid {grade_color}; border-radius: {R['sm']}; padding: 6px 12px; text-align: center; z-index: 10;">
            <div style="font-size: 24px; font-weight: 700; color: #ffffff; line-height: 1.1; font-family: {F['display']};">{grade_str}</div>
            <div style="font-size: 12px; font-weight: 600; color: {pos_color}; letter-spacing: 0.5px; margin-top: 2px;">{pos}</div>
        </div>
        <div style="position: absolute; top: 12px; right: 12px; background: rgba(2,4,9,0.85); border: 1px solid {C['outline_variant']}; border-radius: {R['sm']}; padding: 6px 10px; text-align: center; z-index: 10;">
            <div style="font-size: 14px; font-weight: 700; color: {team_color}; font-family: {F['display']};">{team_abbr}</div>
            <div style="font-size: 12px; font-weight: 500; color: #ffffff; margin-top: 2px;">#{jersey_val}</div>
        </div>
        <div style="width: 100%; display: flex; justify-content: center; align-items: flex-end; padding-top: 60px; z-index: 5;">
            <img src="{headshot_url}" style="width: 130%; max-width: 320px; margin-bottom: -10px; filter: drop-shadow(0 10px 10px rgba(0,0,0,0.8));">
        </div>
        <div style="width: 100%; background: {C['surface_container_high']}; border-top: 4px solid {card_border_color}; padding: 14px 5px; text-align: center; z-index: 10;">
            <div style="font-size: 18px; font-weight: 700; color: #ffffff; text-transform: uppercase; letter-spacing: 0.02em; font-family: {F['display']};">{selected_player}{name_suffix}</div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_bio_strip(age_str, height_str, weight_str):
    """Age/height/weight as three compact tiles in the same visual language
    as the hero/stat tiles, replacing the old single pipe-separated text bar."""
    def cell(label, value):
        return (
            f"<div style='flex:1; text-align:center; padding:10px 6px;'>"
            f"<div style='font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:{C['on_surface_variant']};'>{label}</div>"
            f"<div style='font-family:{F['mono']}; font-size:18px; font-weight:600; color:#ffffff; margin-top:2px;'>{value}</div>"
            f"</div>"
        )
    st.markdown(
        f"<div style='display:flex; gap:1px; margin-top:12px; background:{C['outline_variant']}; "
        f"border:1px solid {C['outline_variant']}; border-radius:{R['default']}; overflow:hidden;'>"
        f"<div style='flex:1; background:{C['surface_container']};'>{cell('Age', age_str)}</div>"
        f"<div style='flex:1; background:{C['surface_container']};'>{cell('Height', height_str)}</div>"
        f"<div style='flex:1; background:{C['surface_container']};'>{cell('Weight', f'{weight_str} lbs')}</div>"
        f"</div>",
        unsafe_allow_html=True
    )


def render_sticky_game_log(log_df_view, avg_source_df, log_cols, header_map, scroll_height=600):
    """
    Single unified table (not two stacked st.dataframe boxes) - weekly rows
    scroll inside one bordered container, and the AVG row is the actual
    last row of that same table, pinned to the bottom of the scrolling area
    via CSS `position: sticky` rather than living in a separate component
    below it. See ui/styling.render_game_log_html_table.
    """
    render_game_log_html_table(log_df_view, avg_source_df, log_cols, header_map, max_height=scroll_height)


POSITION_FILTER_OPTIONS = ['QB', 'RB', 'WR', 'TE', 'K', 'FLEX (RB/WR/TE)']


def position_filter_multiselect(df, key, pos_col='Pos', label="Filter by position"):
    """
    Fixed QB/RB/WR/TE/K + FLEX shortcut position filter - the alternative
    is building the multiselect off whatever position codes happen to
    appear in a particular table's data, which leaves no dedicated way to
    pull up "flex-eligible" players (RB/WR/TE together) in one click
    without hand-picking three positions every time. Returns the filtered
    df (unfiltered if nothing is selected).
    """
    selected = st.multiselect(label, POSITION_FILTER_OPTIONS, default=[], key=key)
    if not selected:
        return df
    positions = set()
    for s in selected:
        if s.startswith('FLEX'):
            positions.update(['RB', 'WR', 'TE'])
        else:
            positions.add(s)
    return df[df[pos_col].isin(positions)]


def check_data_health():
    """
    Lightweight local-file diagnostics for the sidebar panel below. Checks
    the same filenames/fallback lists that load_year_data() and load_pff()
    actually use, so a silently-missing or misnamed export shows up here
    instead of just quietly becoming an empty DataFrame three layers deep.
    This exact panel would have caught the receiving_summary (1).csv
    filename mismatch immediately instead of it silently zeroing out every
    WR/TE receiving metric in the app.
    """
    import glob
    checks = []
    def check_file(label, candidates):
        for c in candidates:
            if os.path.exists(c):
                try:
                    with open(c, 'r', encoding='utf-8', errors='ignore') as fh:
                        n = sum(1 for _ in fh) - 1
                except Exception:
                    n = None
                checks.append((label, c, True, n))
                return
        checks.append((label, " or ".join(candidates), False, None))

    # PFF-file check mirrors data.loaders.load_pff_year's prefix-glob lookup
    # inside pff_imports/{year}/ (the per-year foldering, e.g.
    # pff_imports/2025/receiving_summary.csv) rather than the old flat-path
    # check - that stale check started reporting every PFF file as
    # "not found" the moment the files moved into year subfolders, even
    # though load_all_pff_data() found them fine.
    def check_pff_file(label, base_filename, year):
        year_dir = os.path.join('pff_imports', str(year))
        matches = sorted(glob.glob(os.path.join(year_dir, f'{base_filename}*.csv')))
        if matches:
            try:
                with open(matches[0], 'r', encoding='utf-8', errors='ignore') as fh:
                    n = sum(1 for _ in fh) - 1
            except Exception:
                n = None
            checks.append((label, matches[0], True, n))
        else:
            checks.append((label, os.path.join(year_dir, f'{base_filename}*.csv'), False, None))

    for yr in [2024, 2025, 2026]:
        check_file(f"Weekly stats {yr}", [f"stats_player_week_{yr}.csv"])
        check_file(f"Rosters {yr}", [f"roster_weekly_{yr}.csv"])
        check_file(f"Snap counts {yr}", [f"snap_counts_{yr}.csv.csv", f"snap_counts_{yr}.csv"])
    check_pff_file("PFF receiving summary (2025)", "receiving_summary", 2025)
    check_pff_file("PFF passing summary (2025)", "passing_summary", 2025)
    check_pff_file("PFF rushing summary (2025)", "rushing_summary", 2025)
    return checks


def _current_nfl_season():
    """
    The season currently in progress (or most recently completed): NFL
    seasons run Sep-Feb, so before August the "current" season is still
    last calendar year's (e.g. July 2026 -> the 2025 season - the 2026
    season hasn't kicked off yet).
    """
    import datetime
    today = datetime.date.today()
    return today.year if today.month >= 8 else today.year - 1


@st.cache_data
def _local_stats_max_week(season, file_mtime):
    """
    Highest week number present in the local weekly stats file for a season.
    `file_mtime` is part of the cache key ON PURPOSE - it's how this cache
    busts itself the moment the user drops a refreshed CSV in (same path,
    new mtime), without re-reading a ~19k-row file on every single rerun
    the rest of the time. Returns None if the file doesn't exist.
    """
    path = f"stats_player_week_{season}.csv"
    if not os.path.exists(path):
        return None
    try:
        weeks = pd.read_csv(path, usecols=['week'])['week']
        return int(pd.to_numeric(weeks, errors='coerce').max())
    except Exception:
        return None


def check_current_season_freshness():
    """
    In-season staleness check: compares the latest COMPLETED week on the
    real schedule (games with a posted score, via the - now disk-cached -
    nflreadpy schedule) against the highest week in the local weekly stats
    CSV. During the season the local file only updates when the user
    re-downloads it, so "the app quietly shows 3-week-old data" is the
    single most likely real-world correctness failure - this makes it loud
    instead of silent. Returns (status, message):
      'stale'   - schedule shows completed weeks the local file lacks
      'missing' - no local file for the in-progress season (nflreadpy
                  fallback still feeds the app, so informational)
      'fresh'   - local file covers every completed week
      None      - can't determine (no schedule data; fail quiet, never loud)
    """
    from data.loaders import load_schedule
    season = _current_nfl_season()
    schedule = load_schedule(season)
    if schedule.empty or 'home_score' not in schedule.columns:
        return None, None
    completed = schedule[schedule['home_score'].notna()]
    if completed.empty:
        # Season hasn't kicked off yet - nothing can be stale.
        return None, None
    latest_completed_week = int(pd.to_numeric(completed['week'], errors='coerce').max())

    path = f"stats_player_week_{season}.csv"
    mtime = os.path.getmtime(path) if os.path.exists(path) else 0
    local_max = _local_stats_max_week(season, mtime)
    if local_max is None:
        return 'missing', (
            f"No local weekly stats file for the {season} season yet "
            f"(games through week {latest_completed_week} are complete). The app is "
            f"falling back to live nflverse pulls - drop in stats_player_week_{season}.csv "
            "for faster loads."
        )
    if local_max < latest_completed_week:
        return 'stale', (
            f"Local {season} weekly stats end at week {local_max}, but games through "
            f"week {latest_completed_week} are complete - re-download "
            f"stats_player_week_{season}.csv to see the latest games."
        )
    return 'fresh', f"Local {season} weekly stats are current through week {local_max}."


def render_data_health_sidebar():
    with st.sidebar:
        # Freshness first and OUTSIDE the expander - a stale-data warning
        # nobody sees (because it's collapsed) protects nobody. Quiet
        # one-liner when everything's fine.
        try:
            status, msg = check_current_season_freshness()
        except Exception:
            status, msg = None, None  # diagnostics must never crash the app
        if status == 'stale':
            st.warning(msg)
        elif status == 'missing':
            st.info(msg)
        elif status == 'fresh':
            st.caption(f"✅ {msg}")

        with st.expander("Local Data Health", expanded=False):
            st.caption("What the app actually found on disk just now — catches renamed exports or missing seasons before they turn into quietly blank charts.")
            for label, path, found, rows in check_data_health():
                if found:
                    row_txt = f"· {rows:,} rows" if rows is not None else ""
                    st.markdown(f"✅ **{label}**  \n`{path}` {row_txt}")
                else:
                    st.markdown(f"⚠️ **{label}** — not found  \nlooked for `{path}`")

        with st.expander("📖 Stat Glossary", expanded=False):
            st.markdown(_STAT_GLOSSARY_MD)
