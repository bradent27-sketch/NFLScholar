"""
Theme CSS injection + every Styler-based table renderer in the app. Colors
come from config.THEME (Sleeper-inspired navy/cyan, informed by Dunks &
Threes' dense-table conventions and Spotify's spacing rhythm - see
PLAN.md section 1 for the full design-language rationale).
"""
import base64
import os

import pandas as pd
import streamlit as st

from config import THEME, TEAM_CONFIG

C = THEME['colors']
F = THEME['fonts']
R = THEME['radius']
S = THEME['spacing']

_FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'fonts')
_FONT_FACES = [
    ('Inter', 400, 'inter-400.woff2'),
    ('Inter', 500, 'inter-500.woff2'),
    ('Inter', 600, 'inter-600.woff2'),
    ('Inter', 700, 'inter-700.woff2'),
    ('Inter', 800, 'inter-800.woff2'),
    ('JetBrains Mono', 400, 'jbmono-400.woff2'),
    ('JetBrains Mono', 600, 'jbmono-600.woff2'),
]


@st.cache_data
def _font_face_css():
    """
    Self-hosted @font-face declarations, base64-embedded directly in the
    stylesheet rather than an external @import. A live audit found the
    previous Google Fonts @import url(...) never actually fired a network
    request in this environment (confirmed via document.fonts.check()
    reporting false for Poppins/JetBrains Mono even with real card/table
    content on screen, and the browser's own network log showing zero
    requests to fonts.googleapis.com) - every custom typeface had been
    silently falling back to a generic sans-serif/monospace this entire
    project. Embedding the font files removes the external dependency
    entirely, so these render in every environment, not just ones that
    happen to allow that specific outbound request.
    """
    rules = []
    for family, weight, fname in _FONT_FACES:
        with open(os.path.join(_FONT_DIR, fname), 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('ascii')
        rules.append(
            f"@font-face {{ font-family: '{family}'; font-style: normal; font-weight: {weight}; "
            f"font-display: swap; src: url(data:font/woff2;base64,{b64}) format('woff2'); }}"
        )
    return '\n        '.join(rules)

# Hover-tooltip text for column headers, keyed by the EXACT column name as
# it appears in a displayed table (not the prose glossary terms in
# ui/components.render_intro_and_glossary, which cover the same ground but
# aren't 1:1 with real column names - e.g. one glossary entry covers both
# 'BTT %' and 'TWP %'). st.column_config.Column's help= renders as a native
# hover tooltip on that header - the one thing glide-data-grid headers
# actually support customizing, unlike a raw HTML title attribute (headers
# are canvas-drawn, not real DOM <th> elements - see style_plain_dataframe's
# docstring for the broader story on what is and isn't controllable there).
COLUMN_HELP = {
    'Snap %': "Share of the team's offensive/defensive snaps this player was on the field for",
    'Season Snap %': "Share of the team's offensive/defensive snaps this player was on the field for, season-to-date",
    'Overall PFF': "Pro Football Focus's 0-100 overall game-charting grade",
    'PRP': "Pass Rush Productivity - PFF's pressure-generation rate for pass rushers, weighted toward sacks",
    'MTF/G': "Missed Tackles Forced per game - times this runner made a defender miss",
    'YCO/Att': "Yards After Contact per rush attempt",
    'BTT %': "Big-Time Throw rate - accurate, high-value throws into tight windows",
    'TWP %': "Turnover-Worthy Play rate - throws that should have been intercepted",
    'RecV Grade': "PFF's receiving / route-running grade",
    'PBLK Grade': "PFF's pass-blocking grade",
    'Run Block Grade': "PFF's run-blocking grade",
    'Run Stop Rate': "Run plays this defender stopped at or behind the line to gain, as a % of run snaps",
    'Missed Tkl Rate': "Missed tackles as a % of tackle attempts",
    'Pass Rtg Allowed': "Opposing QB's passer rating when targeting this defender",
    'Win Rate': "Pass rush win rate - how often this rusher beat his blocker within PFF's charted time window",
    'ESPN QBR': "ESPN's Total Quarterback Rating (0-100, accounts for game situation)",
    'YPRR': "Yards Per Route Run - receiving efficiency, not volume",
    'Man YPRR': "Yards Per Route Run when the defense played man coverage",
    'Zone YPRR': "Yards Per Route Run when the defense played zone coverage",
    'ADOT': "Average Depth of Target",
    'EPA': "Expected Points Added - how much a play's value beat expectation for that down/distance/field position",
    'VORP': "Value Over Replacement Player - this app's draft-value metric vs. a replacement-level player at the position",
    'Age': "Age at kickoff, from the player's birthdate. Also drives the aging markdown on the projection: past the position's peak (RB 25, WR 26, QB/TE 27) his own past production is discounted at a measured rate - RB -6.5%/yr, WR -3.6%, TE -2.0%, QB -1.0%",
    'RYOE': "Rush Yards Over Expected (NFL Next Gen Stats) - actual rush yards vs. expected given blockers/defenders in the box",
    'RYOE/Att': "Rush Yards Over Expected per attempt - RYOE normalized for volume",
    'Pct Jump': "Percentile jump from this player's season-to-date average to their most recent game",
    'Man %': "Share of coverage snaps this defense played man coverage",
    'Zone %': "Share of coverage snaps this defense played zone coverage",
    'MOF Closed %': "Middle-of-field closed rate - two-high/split-safety coverage shells",
    'MOF Open %': "Middle-of-field open rate - single-high safety coverage shells",
    'Pressure Allowed %': "Share of dropbacks the starting QB was pressured on (PFR) - a team-level pass-block proxy",
    'Pressure Rate %': "Share of opponent dropbacks this defense generated a pressure on (PFR)",
    'VORP vs FantasyPros': "Positive means this app's VORP model ranks the player higher (better) than FantasyPros does",
    'VORP vs Custom': "Positive means this app's VORP model ranks the player higher (better) than your uploaded ranking does",
    'Proj Pts (17-gm pace)': "Last season's per-game pace stretched to a 17-game season - a volume-based stand-in, not a real projection",
    'Rank': "Positional rank for the selected week (e.g. \"RB4\"), shaded by tier - a cluster break in Model Proj Pts at that position, not a fixed players-per-tier cutoff",
    'Market Proj Pts': "This week's live sportsbook player-prop lines, re-scored under this league's scoring settings - independent of this app's own model",
    'Market Coverage': "Share of a typical week's fantasy points the market's posted lines actually covered for this player",
    'L5 Avg FPTS': "Average fantasy points over the player's last 5 games played, not a season-long or extrapolated number",
    'ECR': "FantasyPros consensus rank for the chosen draft FORMAT only (Redraft/Superflex/Dynasty/Best Ball) - not adjusted for your league's PPR/Half-PPR/Standard or TE-premium settings unless pulled via the FantasyPros API",
    'ADP': "Average Draft Position, matched to your Standard/Half-PPR/Full PPR setting (collapses to one Superflex page regardless of PPR if Superflex is on) - not adjusted for your exact team count unless the source is Fantasy Football Calculator",
}
for _pos in ['QB', 'RB', 'WR', 'TE']:
    COLUMN_HELP[f'{_pos} SOS'] = f"Strength of schedule for {_pos}s - average fantasy points allowed per game by teams left on the schedule (higher = softer matchups)"
for _kind in ['WR', 'TE', 'RB', 'Outside', 'Slot']:
    COLUMN_HELP[f'YPT vs {_kind}'] = f"Yards Per Target allowed to {_kind}s"
    COLUMN_HELP[f'YPT {_kind}'] = f"Yards Per Target allowed, {_kind} alignment"


def build_column_help_config(df, pinned_cols=None, meter_cols=None):
    """
    Returns the st.dataframe(column_config=...) dict for whichever of this
    table's columns have a COLUMN_HELP entry - pass alongside
    style_plain_dataframe(df) (a Styler can't carry column_config itself,
    that's a separate st.dataframe parameter). Existing callers are
    unaffected - both new params default to off.

    pinned_cols: column names to freeze on horizontal scroll (native
    st.column_config.Column(pinned=True), Streamlit 1.59+ - no custom-
    component work needed). The DataFrame's own index (every wide table
    here is already .set_index('Player') or similar) freezes on its own;
    this is for a real COLUMN worth keeping in view too on a genuinely
    wide table - e.g. Team, so you still know whose row you're reading
    after scrolling past a dozen stat columns.

    meter_cols: {column_name: (min_value, max_value)} - swaps that column
    from a plain NumberColumn to st.column_config.ProgressColumn, an
    inline linear meter instead of a bare number (NFL Savant's "Linear
    Probability Meter," also native) - ADDITIVE to the existing percentile
    heatmap coloring on the same cell (style_plain_dataframe's job),
    not a replacement for it.
    """
    pinned_cols = pinned_cols or []
    meter_cols = meter_cols or {}
    config = {}
    for col in df.columns:
        help_text = COLUMN_HELP.get(col)
        is_pinned = col in pinned_cols
        if col in meter_cols:
            lo, hi = meter_cols[col]
            config[col] = st.column_config.ProgressColumn(help=help_text, pinned=is_pinned, min_value=lo, max_value=hi, format="%.1f")
        elif help_text or is_pinned:
            config[col] = st.column_config.Column(help=help_text, pinned=is_pinned)
    return config


def inject_theme():
    st.markdown(f"""
        <style>
        {_font_face_css()}

        .stApp {{
            background:
                radial-gradient(1100px 420px at 18% -8%, rgba(56, 96, 190, 0.16) 0%, rgba(56, 96, 190, 0) 60%),
                radial-gradient(900px 380px at 85% -12%, rgba(0, 255, 249, 0.07) 0%, rgba(0, 255, 249, 0) 55%),
                linear-gradient(180deg, {C['surface']} 0%, {C['surface_dim']} 100%) !important;
            color: {C['on_surface']} !important;
            font-family: {F['body']} !important;
            /* Applied app-wide (not just mono table cells) so a proportional-
               font number - st.metric values on the Defensive Yield radar,
               odds prices, anywhere Inter renders a digit - lines up in a
               column instead of proportional-width digits (a "1" narrower
               than a "8") causing adjacent numbers to jitter left/right.
               No-op on non-numeric text. */
            font-variant-numeric: tabular-nums;
        }}

        /* ---- Streamlit chrome: make the app read as a product, not a
           notebook. The fixed header becomes a translucent glass bar
           (Sleeper's own fixed-header spec) instead of an opaque slab; the
           Deploy button and footer are development chrome and hidden.
           The main (hamburger) menu and sidebar toggle stay - they're
           genuinely useful controls. */
        header[data-testid="stHeader"] {{
            background: rgba(5, 9, 33, 0.72) !important;
            backdrop-filter: blur(8px);
            border-bottom: 1px solid rgba(44, 50, 96, 0.6);
        }}
        [data-testid="stAppDeployButton"], .stDeployButton, footer {{ display: none !important; }}

        /* padding-top clears the fixed header band (still ~60px tall even
           when translucent - the title row would otherwise start under it
           and get visually dimmed by the glass). Full-bleed width (no
           max-width cap) per explicit feedback: the 1440px cap left the
           game log table cut off horizontally and squeezed the depth
           chart's 5th/6th-string columns - data-dense tables earn the
           whole monitor. */
        .block-container {{
            padding-top: 4.5rem !important;
            padding-bottom: 2rem !important;
            padding-left: 2.5rem !important;
            padding-right: 2.5rem !important;
            max-width: none !important;
        }}

        h1, h2, h3, h4 {{
            font-family: {F['display']} !important;
            font-weight: 700 !important;
            color: {C['on_surface']};
            margin-bottom: 0.5rem;
            letter-spacing: -0.01em;
        }}

        /* Section headers - "kicker" style used by pro stat sites: small
           uppercase label with an accent tick, anchored by a hairline rule
           that runs the full row, instead of the old full-width filled
           gradient bar (which read heavy/boxy at this density). */
        .custom-section-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            color: {C['on_surface']} !important;
            font-family: {F['display']};
            font-size: 12.5px !important;
            font-weight: 800 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.1em !important;
            padding: 6px 0 8px 0 !important;
            margin-top: 26px !important;
            margin-bottom: 12px !important;
            border-bottom: 1px solid {C['outline_variant']};
        }}
        .custom-section-header::before {{
            content: "";
            display: inline-block;
            width: 4px;
            height: 14px;
            border-radius: 2px;
            background: {C['primary']};
            box-shadow: 0 0 8px rgba(0, 255, 249, 0.5);
            flex: 0 0 auto;
        }}

        /* Pill-shaped tabs, Sleeper/Spotify convention - with a hairline
           rail under the whole tab row so the nav reads as one anchored
           bar rather than floating chips.

           IMPORTANT: this Streamlit version renders st.tabs with
           react-aria ([data-testid="stTab"] divs inside a [role="tablist"],
           aria-selected on the tab itself), NOT the old BaseWeb
           button[data-baseweb="tab"] markup - confirmed live against the
           running app's actual DOM. The previous BaseWeb selectors matched
           nothing (tabs silently rendered with default styling); both
           selector generations are kept below so the styling survives in
           either direction. */
        div[data-testid="stTabs"] [role="tablist"] {{
            gap: 1px;
            border-bottom: 1px solid {C['outline_variant']};
            padding-bottom: 6px;
            flex-wrap: wrap;
        }}
        [data-testid="stTab"], div[data-testid="stTabs"] button[data-baseweb="tab"] {{
            border-radius: {R['full']} !important;
            padding: 6px 9px !important;
            transition: background-color 150ms ease-out;
            cursor: pointer;
        }}
        [data-testid="stTab"]:hover, div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {{
            background-color: rgba(255, 255, 255, 0.05) !important;
        }}
        [data-testid="stTab"] p, button[data-baseweb="tab"] p {{
            color: {C['on_surface_variant']} !important;
            font-family: {F['display']};
            font-size: 11px !important;
            font-weight: 600 !important;
            letter-spacing: 0;
            white-space: nowrap;
        }}
        [data-testid="stTab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"] {{
            background-color: rgba(0, 255, 249, 0.10) !important;
        }}
        [data-testid="stTab"][aria-selected="true"] p, button[data-baseweb="tab"][aria-selected="true"] p {{
            color: {C['primary']} !important;
        }}
        div[data-baseweb="tab-highlight"] {{ background-color: {C['primary']} !important; }}

        /* st.metric -> stat tile card (Defensive Yield radar metrics, etc).
           Default Streamlit metrics render as bare floating text, one of
           the strongest "unstyled prototype" signals on the page. */
        div[data-testid="stMetric"] {{
            background: {C['surface_container']};
            border: 1px solid {C['outline_variant']};
            border-radius: {R['md']};
            padding: 10px 14px;
        }}
        div[data-testid="stMetric"] label, div[data-testid="stMetricLabel"] p {{
            color: {C['on_surface_variant']} !important;
            font-size: 11px !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }}
        div[data-testid="stMetricValue"] {{
            font-family: {F['mono']} !important;
            font-size: 24px !important;
            font-weight: 600;
            color: {C['on_surface']} !important;
        }}

        /* Alerts (st.info/warning/error) - glass cards in the app palette
           instead of Streamlit's default flat pastels. */
        div[data-testid="stAlert"] {{
            background: rgba(19, 27, 56, 0.55) !important;
            border: 1px solid {C['outline_variant']} !important;
            border-radius: {R['md']} !important;
            backdrop-filter: blur(6px);
            color: {C['on_surface']} !important;
        }}
        div[data-testid="stAlert"] p {{ color: {C['on_surface']} !important; }}

        /* Sidebar - one shade below the canvas with a hairline edge, so the
           diagnostics panel reads as a docked rail, not a second page. */
        section[data-testid="stSidebar"] {{
            background: {C['surface_dim']} !important;
            border-right: 1px solid {C['outline_variant']};
        }}

        /* Slim dark scrollbars (webkit) - the OS-default chrome scrollbar
           inside the game log / table containers broke the dark surface. */
        ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: {C['surface_container_highest']}; border-radius: 8px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {C['outline']}; }}

        /* Stat-tile grid (Player Search season profile + hero strip) -
           static layout rules here; the per-tile percentile color/width is
           inline (data-driven). */
        .stat-tile-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(108px, 1fr));
            gap: 8px;
            margin: 4px 0 8px 0;
        }}
        .stat-tile {{
            position: relative;
            background: {C['surface_container']};
            border: 1px solid {C['outline_variant']};
            border-radius: {R['default']};
            padding: 9px 34px 10px 11px;
            overflow: hidden;
            transition: transform 200ms ease-out, border-color 200ms ease-out;
        }}
        /* Card-hover per design.md/DESIGN.md's documented "card-hover" token
           (border shifts toward the primary accent, subtle lift) - already
           applied to buttons/tabs/inputs, extended here to the tile
           components added in the July 2026 polish pass, which never picked
           it up. */
        .stat-tile:hover {{
            transform: translateY(-2px);
            border-color: rgba(0, 255, 249, 0.35);
        }}
        .stat-tile .t-label {{
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {C['on_surface_variant']};
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .stat-tile .t-value {{
            font-family: {F['mono']};
            font-size: 16.5px;
            font-weight: 600;
            color: {C['on_surface']};
            margin-top: 3px;
            line-height: 1.15;
        }}
        .stat-tile .t-pct {{
            position: absolute;
            top: 8px;
            right: 8px;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: {F['mono']};
            font-size: 10px;
            font-weight: 700;
            color: #ffffff;
            text-shadow: 0 1px 2px rgba(0,0,0,0.55);
        }}
        .stat-tile .t-meter-track {{
            position: absolute;
            left: 0; right: 0; bottom: 0;
            height: 3px;
            background: {C['surface_container_highest']};
        }}
        .stat-tile .t-meter {{ position: absolute; left: 0; bottom: 0; height: 3px; }}

        .hero-tile-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin: 10px 0 4px 0;
        }}
        .hero-tile {{
            background: linear-gradient(180deg, {C['surface_container']} 0%, {C['surface_container_low']} 100%);
            border: 1px solid {C['outline_variant']};
            border-radius: {R['md']};
            padding: 12px 14px;
            text-align: center;
            transition: transform 200ms ease-out, border-color 200ms ease-out;
        }}
        .hero-tile:hover {{
            transform: translateY(-2px);
            border-color: rgba(0, 255, 249, 0.35);
        }}
        .hero-tile .h-label {{
            font-size: 10.5px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {C['on_surface_variant']};
        }}
        .hero-tile .h-value {{
            font-family: {F['mono']};
            font-size: 26px;
            font-weight: 600;
            color: {C['on_surface']};
            line-height: 1.2;
            margin-top: 2px;
        }}
        .hero-tile .h-sub {{
            font-size: 10.5px;
            color: {C['on_surface_variant']};
            margin-top: 2px;
        }}

        /* Season fantasy-points-by-week LINE chart (Player Search) - the
           first thing shown once a player is selected, styled after the
           boards' own "Last 5 Wks" LineChartColumn sparkline per explicit
           feedback that it read cleanest "at a glance". Pure inline SVG
           (not matplotlib/Vega-Lite) with a fixed viewBox + default
           preserveAspectRatio (uniform xMidYMid scaling) so circles/text
           never distort at any container width - see
           ui.components.render_fpts_week_strip. */
        .fpts-strip {{
            background: {C['surface_container']};
            border: 1px solid {C['outline_variant']};
            border-radius: {R['md']};
            padding: 10px 14px 8px 14px;
            margin: 2px 0 10px 0;
        }}
        .fpts-strip .fs-label {{
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {C['on_surface_variant']};
            margin-bottom: 6px;
        }}
        .fpts-linechart {{
            display: block;
            width: 100%;
            height: auto;
        }}
        .fpts-linechart .fl-wklabel {{
            font-family: {F['mono']};
            font-size: 11px;
            fill: {C['on_surface_variant']};
        }}
        /* Faint y-axis (fantasy-points scale) - deliberately much quieter
           than the week labels above: a thin line + tick marks in the
           near-background outline_variant tone, and small dim labels, so
           it's there to reference a value against but never competes with
           the line/dots for attention. */
        .fpts-linechart .fl-yaxis-line, .fpts-linechart .fl-ytick {{
            stroke: {C['outline_variant']};
            stroke-width: 1;
        }}
        .fpts-linechart .fl-yticklabel {{
            font-family: {F['mono']};
            font-size: 9px;
            fill: {C['on_surface_variant']};
            opacity: 0.6;
        }}

        /* Rest-of-season matchup heat-strip (Player Search) -
           ui.components.render_matchup_heat_strip. Shares .fpts-strip/
           .fs-label above for the outer card/label; only the per-game cell
           row is new here. Cell background is inline (data-driven, one
           get_matchup_color() per game) - only the static layout/typography
           lives in these rules, same split as .stat-tile's t-pct/t-meter. */
        .matchup-strip-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }}
        .ms-cell {{
            flex: 0 0 auto;
            min-width: 56px;
            border-radius: {R['default']};
            padding: 6px 8px;
            text-align: center;
        }}
        .ms-cell .ms-wk {{
            font-family: {F['mono']};
            font-size: 9px;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: rgba(255, 255, 255, 0.75);
        }}
        .ms-cell .ms-opp {{
            font-family: {F['display']};
            font-size: 13px;
            font-weight: 800;
            color: #ffffff;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
            margin-top: 1px;
        }}

        /* Percentile bar chart (Player Search "Season Profile") -
           ui.player_snapshot.render_percentile_bars_figure. Reuses
           .fpts-strip/.fs-label for the outer card/label, same as the
           matchup heat-strip above. Bar fill color is a bare SVG `fill`
           attribute set inline per-row (data-driven, one get_pff_color()
           per stat) - only the static layout/typography/hover live here,
           same split as every other data-driven component in this file.
           Hover state (outline the bar + enlarge its value) replaces the
           interaction the now-removed stat-tile grid used to carry, since
           this chart is the sole primary view of these stats now. */
        .pbar-svg {{ display: block; width: 100%; height: auto; }}
        .pbar-grid {{ stroke: {C['outline_variant']}; stroke-width: 1; }}
        .pbar-tick {{
            font-family: {F['mono']};
            font-size: 10px;
            fill: {C['on_surface_variant']};
        }}
        .pbar-track {{ fill: {C['surface_container_high']}; }}
        .pbar-label {{
            font-family: {F['body']};
            font-size: 12px;
            font-weight: 600;
            fill: {C['on_surface_variant']};
            transition: fill 150ms ease-out;
        }}
        .pbar-value {{
            font-family: {F['mono']};
            font-size: 12px;
            font-weight: 700;
            fill: #ffffff;
            transition: transform 150ms ease-out;
            transform-box: fill-box;
            transform-origin: left center;
        }}
        .pbar-fill {{
            stroke: transparent;
            stroke-width: 0;
            transition: filter 150ms ease-out, stroke-width 150ms ease-out;
        }}
        .pbar-row:hover .pbar-fill {{ stroke: #ffffff; stroke-width: 2px; filter: brightness(1.12); }}
        .pbar-row:hover .pbar-value {{ transform: scale(1.28); }}
        .pbar-row:hover .pbar-label {{ fill: #ffffff; }}

        /* Paired split-bar chart (ui.charts.render_split_bars) - the "vs
           Man / vs Zone" and "Slot / Wide" panels. Same hover language as
           .pbar-row above (outline + brighten the filled bar, bold the
           label) so every percentile chart on Matchup Analyzer feels like
           one system rather than some rows being interactive and others not. */
        .split-bar-fill {{
            stroke: transparent;
            stroke-width: 0;
            transition: filter 150ms ease-out, stroke-width 150ms ease-out;
        }}
        .split-bar-label {{ transition: fill 150ms ease-out; }}
        .split-bar-value {{
            transition: transform 150ms ease-out;
            transform-box: fill-box;
            transform-origin: right center;
        }}
        .split-bar-row:hover .split-bar-fill {{ stroke: #ffffff; stroke-width: 2px; filter: brightness(1.12); }}
        .split-bar-row:hover .split-bar-value {{ transform: scale(1.15); }}
        .split-bar-row:hover .split-bar-label {{ fill: #ffffff; }}

        /* Percentile-colored metric tiles (Defensive Yield's Coverage
           Matchup Radar) - ui.components.render_percentile_metric_tiles.
           Centered layout like .hero-tile, but color is per-tile and
           data-driven (a percentile fill, not a fixed gradient) - a plain
           st.metric has no per-instance styling hook a caller can drive
           from Python, which is why this exists instead of just theming
           div[data-testid="stMetric"] further. */
        .metric-tile-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin: 10px 0 4px 0;
        }}
        .metric-tile {{
            border-radius: {R['md']};
            padding: 12px 14px;
            text-align: center;
            border: 1px solid {C['outline_variant']};
            transition: transform 200ms ease-out, border-color 200ms ease-out;
        }}
        /* Same card-hover token as .stat-tile/.hero-tile - this tile type
           never picked it up when the others did. */
        .metric-tile:hover {{
            transform: translateY(-2px);
            border-color: rgba(0, 255, 249, 0.35);
        }}
        .metric-tile .m-label {{
            font-size: 10.5px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: rgba(255, 255, 255, 0.85);
        }}
        .metric-tile .m-value {{
            font-family: {F['mono']};
            font-size: 22px;
            font-weight: 700;
            color: #ffffff;
            line-height: 1.2;
            margin-top: 2px;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
        }}
        .metric-tile .m-sub {{
            font-size: 10px;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.85);
            margin-top: 2px;
        }}

        /* Small column-group caption above each of the Coverage Matchup
           Radar's three tile columns (PLAYER/BLENDED/TEAM) - identifies
           which side of the matchup each group of metric-tiles belongs to
           without the visual weight of a full .custom-section-header. */
        .metric-col-label {{
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            text-align: center;
            color: {C['on_surface_variant']};
            margin: 2px 0 2px 0;
        }}

        /* Matchup title (Defensive Yield's Coverage Correlator radar) -
           centered "{{player}} VS {{team}}" header, replacing a bare bold
           markdown line that read cramped against the chart below it. */
        .matchup-title {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 14px;
            flex-wrap: wrap;
            margin: 18px 0 18px 0;
            text-align: center;
        }}
        .matchup-title .mt-player, .matchup-title .mt-team {{
            font-family: {F['display']};
            font-size: 21px;
            font-weight: 800;
            letter-spacing: -0.01em;
            line-height: 1.2;
        }}
        .matchup-title .mt-player {{ color: {C['on_surface']}; }}
        .matchup-title .mt-vs {{
            font-family: {F['display']};
            font-size: 10.5px;
            font-weight: 700;
            letter-spacing: 0.12em;
            color: {C['on_surface_variant']};
            background: {C['surface_container_high']};
            border: 1px solid {C['outline_variant']};
            border-radius: {R['full']};
            padding: 4px 11px;
            flex: 0 0 auto;
        }}

        /* Team banner (Depth Charts) */
        .team-banner {{
            display: flex;
            align-items: center;
            gap: 16px;
            border-radius: {R['md']};
            padding: 14px 18px;
            margin: 6px 0 14px 0;
            border: 1px solid {C['outline_variant']};
        }}
        .team-banner img {{ height: 52px; width: 52px; object-fit: contain; filter: drop-shadow(0 2px 6px rgba(0,0,0,0.5)); }}
        .team-banner .tb-name {{
            font-family: {F['display']};
            font-size: 22px;
            font-weight: 800;
            letter-spacing: -0.01em;
            color: #ffffff;
            text-shadow: 0 2px 4px rgba(0,0,0,0.4);
            line-height: 1.1;
        }}
        .team-banner .tb-sub {{
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: rgba(255,255,255,0.75);
            margin-top: 2px;
        }}

        /* Player bio card hover (ui.components.render_player_card) - same
           card-hover token as the stat/hero tiles above. box-shadow needs
           !important here specifically: the card's base box-shadow is set
           per-render as an inline style (card_glow, grade/gold-dependent),
           and an inline style always wins over a class selector - even
           :hover - on the same property, so the hover rule would otherwise
           be silently overridden the instant it's applied. transform has no
           such conflict (nothing sets it inline) and needs no override. */
        .player-card {{
            transition: transform 200ms ease-out, box-shadow 200ms ease-out;
        }}
        .player-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0px 16px 38px rgba(0,0,0,0.65), 0 0 24px rgba(0, 255, 249, 0.35) !important;
        }}

        /* Glass cards for expanders/containers - Sleeper's card spec
           (backdrop-filter blur + translucent surface). */
        div[data-testid="stExpander"] {{
            background: rgba(19, 27, 56, 0.4) !important;
            border: 1px solid {C['outline_variant']} !important;
            border-radius: {R['md']} !important;
            backdrop-filter: blur(8px);
        }}
        div[data-testid="stExpander"] summary {{
            font-family: {F['display']};
            font-weight: 600 !important;
        }}

        /* Pill buttons. Background/text color set explicitly - without it,
           Streamlit's own default button styling (light-theme white bg)
           leaked through here, producing white text on a white background
           (invisible button labels) since only border/radius/font were
           overridden. */
        .stButton button, .stDownloadButton button {{
            background-color: {C['surface_container']} !important;
            color: {C['on_surface']} !important;
            border-radius: {R['full']} !important;
            font-family: {F['display']};
            font-weight: 600 !important;
            border: 1px solid {C['outline_variant']} !important;
            transition: all 150ms ease-out;
        }}
        .stButton button:hover, .stDownloadButton button:hover {{
            border-color: {C['primary']} !important;
            color: {C['primary']} !important;
            background-color: rgba(0, 255, 249, 0.06) !important;
        }}
        .stButton button[kind="primary"] {{
            background-color: {C['primary']} !important;
            color: {C['on_primary']} !important;
            border: none !important;
        }}

        /* Inputs - dark field, cyan focus ring. Two selector generations:
           this Streamlit version renders selectbox/multiselect/number
           fields as react-aria groups (.stSelectbox [role="group"] etc),
           NOT the old BaseWeb div[data-baseweb="select"] markup - confirmed
           live against the running DOM (the BaseWeb selectors matched
           nothing, so every dropdown field was silently rendering with the
           default flat theme color). Both kept, same as the tab styling. */
        .stSelectbox [role="group"], .stMultiSelect [role="group"],
        .stNumberInput [role="group"], .stDateInput [role="group"],
        div[data-baseweb="select"] > div, .stTextInput input, .stTextArea textarea, .stNumberInput input {{
            background-color: rgba(0, 0, 0, 0.25) !important;
            border-radius: {R['default']} !important;
            border: 1px solid {C['outline_variant']} !important;
            color: {C['on_surface']} !important;
        }}
        .stSelectbox [role="group"] input, .stMultiSelect [role="group"] input, .stNumberInput [role="group"] input {{
            background-color: transparent !important;
            border: none !important;
            color: {C['on_surface']} !important;
        }}
        .stSelectbox [role="group"]:focus-within, .stMultiSelect [role="group"]:focus-within,
        .stNumberInput [role="group"]:focus-within, .stDateInput [role="group"]:focus-within,
        div[data-baseweb="select"] > div:focus-within, .stTextInput input:focus, .stTextArea textarea:focus {{
            border-color: {C['primary']} !important;
            box-shadow: 0 0 0 2px rgba(0, 255, 249, 0.15) !important;
        }}

        div.stSelectbox {{ margin-bottom: -10px !important; }}

        /* Hide the click-to-enlarge "Fullscreen" toolbar button on chart
           images specifically (e.g. the Coverage Radar) - the chart is
           already sized appropriately and the enlarge affordance felt
           unnecessary. Scoped with :has() to stFullScreenFrame wrappers
           that contain an stImage child, so st.dataframe's own Fullscreen
           button (a genuinely useful feature there) is left untouched. */
        div[data-testid="stFullScreenFrame"]:has(div[data-testid="stImage"]) button[data-testid="stBaseButton-elementToolbar"] {{
            display: none !important;
        }}

        /* Tables - dense, mono numerals per Dunks & Threes convention (this
           app's tables are almost entirely numeric). Radius intentionally
           tighter (R['sm']) than the glassy R['md'] expanders/inputs use -
           a data-dense grid reads as a technical instrument, not a soft
           consumer-app card; the softer radius stays reserved for
           genuinely chrome-like elements (tabs/buttons keep R['full']). */
        div[data-testid="stDataFrame"], div[data-testid="stTable"], .stDataFrame {{
            background-color: {C['surface_container']} !important;
            border: 1px solid {C['outline_variant']} !important;
            border-radius: {R['sm']} !important;
            padding: 0 6px;
            font-family: {F['mono']} !important;
            /* box-sizing + overflow are load-bearing, not cosmetic. The
               grid inside is a canvas sized to 100% of this element's
               CONTENT box; without border-box the 6px padding and 1px
               border are added on top, so the canvas renders ~14px wider
               than the frame around it and visibly hangs off its own
               background - the "table falling out of its box" effect.
               Clipping to the rounded corner finishes it. */
            box-sizing: border-box !important;
            overflow: hidden !important;
        }}
        /* No VERTICAL padding here specifically (horizontal is fine) -
           measured live (Playwright) that the grid's inner canvas is
           always exactly (requested height - 2px) tall regardless of
           padding, i.e. it does NOT shrink to fit a padded content box the
           way the width dimension does. Any top/bottom padding just pushes
           that fixed-size canvas down without shrinking it, so it overhangs
           the bottom by (padding+border amount - 2px) and gets clipped by
           overflow:hidden above - invisible on a tall table (a few px lost
           out of hundreds) but it ate the entire last row on a real 1-row
           box-score table (a single QB's passing line). Zero vertical
           padding + the 1px border leaves exactly (requested height - 2px)
           of content-box height, which matches the canvas's own size
           exactly - confirmed via getBoundingClientRect() on real tables
           from 1 row to 20+, zero overhang at every size. */
        div[data-testid="stDataFrame"] {{
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }}
        div[data-testid="stDataFrame"] > div,
        div[data-testid="stDataFrame"] canvas {{
            max-width: 100% !important;
        }}
        div[data-testid="stDataFrame"] * {{
            color: {C['on_surface']} !important;
            font-family: {F['mono']} !important;
        }}

        /* ---- Premium polish pass: skeleton loaders, content-reveal
           transitions, and pressed/focus feedback. Kept as one block at
           the end so this whole pass is easy to find (or revert) together,
           instead of scattered inline with the older per-component rules
           above. */

        /* Skeleton loaders - shimmering placeholders shaped like the
           content about to appear (a table, a stat-tile grid, or a plain
           card), swapped in via ui.components.skeleton_loader() in place
           of a bare st.spinner(). Only ever visible on an actual cache
           miss (a fresh season/scoring/game combo), same as the spinner
           it replaces. */
        @keyframes skel-shimmer {{
            0% {{ background-position: -420px 0; }}
            100% {{ background-position: 420px 0; }}
        }}
        .skel-table-wrap {{
            border: 1px solid {C['outline_variant']};
            border-radius: {R['sm']};
            background: {C['surface_container']};
            padding: 10px;
        }}
        .skel-row {{ display: flex; gap: 8px; margin-bottom: 6px; }}
        .skel-row:last-child {{ margin-bottom: 0; }}
        .skel-header-row {{ margin-bottom: 10px; }}
        .skel-header-row .skel-cell {{ height: 12px; opacity: 0.6; }}
        .skel-cell, .skel-tile, .skel-block {{
            background: linear-gradient(90deg, {C['surface_container_high']} 20%, {C['surface_container_highest']} 50%, {C['surface_container_high']} 80%);
            background-size: 420px 100%;
            animation: skel-shimmer 1.3s ease-in-out infinite;
            border-radius: {R['default']};
        }}
        .skel-row .skel-cell {{ height: 22px; flex: 1; }}
        .skel-tile-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(108px, 1fr));
            gap: 8px;
        }}
        .skel-tile {{ height: 58px; }}
        .skel-block {{ width: 100%; border: 1px solid {C['outline_variant']}; }}

        /* Content-reveal - a short fade + rise on whatever just finished
           loading (a table, an alert/info box, a section header), instead
           of an instant "pop in". Streamlit re-inserts these elements fresh
           on the rerun that follows a skeleton clearing (season/scoring/
           filter change), so this reliably replays right at the moment a
           hard instant swap used to feel harshest. */
        @keyframes content-reveal {{
            from {{ opacity: 0; transform: translateY(5px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        div[data-testid="stDataFrame"], div[data-testid="stTable"],
        div[data-testid="stAlert"], .custom-section-header,
        .skel-table-wrap, .skel-tile-grid, .skel-block {{
            animation: content-reveal 260ms ease-out;
        }}

        /* ---- Interactive feedback on buttons.
           A button that visibly responds tells you it registered the click
           before the rerun lands, which matters most under a pick clock
           where the alternative is wondering whether you actually hit it.
           Lift + glow on hover, and a real press-down on click. Kept small:
           a big scale on a dense board reads as jitter, not polish. */
        .stButton button, .stDownloadButton button {{
            transition: transform 120ms ease-out, box-shadow 160ms ease-out,
                        background-color 150ms ease-out, border-color 150ms ease-out,
                        color 150ms ease-out !important;
            will-change: transform;
        }}
        .stButton button:hover, .stDownloadButton button:hover {{
            transform: translateY(-1px) scale(1.015);
            box-shadow: 0 4px 16px rgba(0, 255, 249, 0.20);
        }}
        .stButton button[kind="primary"]:hover {{
            filter: brightness(1.12);
            box-shadow: 0 4px 18px rgba(0, 255, 249, 0.42);
        }}
        .stButton button:active, .stDownloadButton button:active {{
            transform: translateY(0) scale(0.985);
            box-shadow: none;
        }}
        /* Rows/cards that are clickable get the same language, so "this
           responds to a click" looks identical everywhere. */
        .rp-card, .pv-card, .db-cell {{
            transition: transform 120ms ease-out, box-shadow 160ms ease-out;
        }}
        .rp-card:hover, .pv-card:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
        }}

        /* Pressed/active feedback - buttons, pill tabs, and the hover-lift
           cards only had a :hover state before (see the rules above);
           nothing gave tactile feedback on the actual press. */
        .stButton button:active, .stDownloadButton button:active {{
            transform: scale(0.96) !important;
            filter: brightness(0.94);
        }}
        [data-testid="stTab"]:active, div[data-testid="stTabs"] button[data-baseweb="tab"]:active {{
            transform: scale(0.96);
        }}
        .stat-tile:active, .hero-tile:active, .player-card:active {{
            transform: scale(0.98);
        }}

        /* Focus-visible rings app-wide (keyboard nav) - previously only
           text inputs/selects had a focus ring; buttons and tabs had none. */
        .stButton button:focus-visible, .stDownloadButton button:focus-visible,
        [data-testid="stTab"]:focus-visible, div[data-testid="stTabs"] button[data-baseweb="tab"]:focus-visible {{
            outline: 2px solid {C['primary']} !important;
            outline-offset: 2px;
        }}

        /* Dropdown/number/date field hover - previously only had a focus
           ring, no cue on hover that the field is interactive. */
        .stSelectbox [role="group"]:hover, .stMultiSelect [role="group"]:hover,
        .stNumberInput [role="group"]:hover, .stDateInput [role="group"]:hover {{
            border-color: {C['outline']} !important;
        }}

        /* Expander hover - the sidebar diagnostics panel and glossary/
           technical-details disclosures had no hover cue that they're
           clickable. */
        div[data-testid="stExpander"] summary {{
            transition: color 150ms ease-out;
            cursor: pointer;
        }}
        div[data-testid="stExpander"] summary:hover {{
            color: {C['primary']} !important;
        }}

        /* Game log table rows (ui.styling.render_game_log_html_table) -
           real DOM <tr> elements (not glide-data-grid canvas - see that
           function's docstring), so a genuine per-row hover + staggered
           entrance works here, unlike the canvas-rendered st.dataframe
           tables everywhere else in the app. */
        @keyframes row-in {{
            from {{ opacity: 0; transform: translateY(4px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .gl-row {{ animation: row-in 240ms ease-out backwards; }}
        .gl-row:hover td {{ filter: brightness(1.18); }}

        /* ---- GAME SLATE matchup cards -------------------------------------
           Every rule here is scoped to div[class*="st-key-gs_card_"], the
           class Streamlit generates for st.container(key="gs_card_N"). That
           prefix is what makes a Streamlit-owned container styleable at all,
           and scoping to it keeps these rules out of every other tab's
           containers.

           The card cannot be one block of raw HTML: it has to hold real
           st.button widgets, because switching tabs is only legal from an
           on_click callback (see ui.components.switch_tab). So it is a
           styled container with markdown inside it. */
        div[class*="st-key-gs_card_"] {{
            position: relative;
            background: {C['surface_container_low']};
            border: 1px solid {C['outline_variant']};
            border-radius: {R['md']};
            padding: 14px 14px 10px 14px;
            margin-bottom: 14px;
            overflow: hidden;
            transition: border-color 160ms ease-out, transform 160ms ease-out,
                        box-shadow 160ms ease-out;
        }}
        div[class*="st-key-gs_card_"]:hover {{
            border-color: {C['outline']};
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.38);
        }}
        /* The accent bar reads the two teams' real colors, which arrive as
           custom properties from a one-line <style> the card itself emits -
           a Streamlit-owned element has no inline style attribute to set.
           Each var() carries a neutral fallback so a team with no color
           degrades to the outline instead of an invalid gradient. */
        div[class*="st-key-gs_card_"]::before {{
            content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
            background: linear-gradient(90deg,
                var(--gs-a, {C['outline']}) 0%, var(--gs-a, {C['outline']}) 48%,
                var(--gs-b, {C['outline']}) 52%, var(--gs-b, {C['outline']}) 100%);
        }}

        .gs-meta {{
            display: flex; flex-wrap: wrap; align-items: center; gap: 6px;
            font-family: {F['mono']}; font-size: 10.5px; letter-spacing: 0.04em;
            text-transform: uppercase; color: {C['on_surface_variant']};
            margin-bottom: 8px;
        }}
        .gs-dot {{ opacity: 0.45; }}
        .gs-tv {{ color: {C['tertiary']}; }}
        .gs-headline {{
            font-family: {F['display']}; font-size: 11px; font-weight: 700;
            letter-spacing: 0.06em; text-transform: uppercase;
            color: {C['tertiary']}; margin-bottom: 6px;
        }}
        .gs-status {{ color: {C['error']}; }}

        /* One row per team. .gs-name takes the free space so conference,
           score and the W pill are pushed to the right edge; everything
           else is fixed so a long name can't squeeze the logo. */
        .gs-team {{
            display: flex; align-items: center; gap: 9px;
            padding: 7px 9px; border-radius: {R['default']};
            border-left: 3px solid var(--gs-color, {C['outline']});
            background: {C['surface_container']};
            margin-bottom: 5px;
        }}
        .gs-team .gs-side {{
            flex: 0 0 34px; font-family: {F['mono']}; font-size: 9px;
            letter-spacing: 0.08em; color: {C['on_surface_variant']}; opacity: 0.7;
        }}
        .gs-team .gs-logo {{
            height: 24px; width: 24px; object-fit: contain; flex: 0 0 24px;
            filter: drop-shadow(0 1px 3px rgba(0, 0, 0, 0.55));
        }}
        .gs-team .gs-name {{
            flex: 1 1 auto; font-family: {F['display']}; font-size: 14px;
            font-weight: 600; color: {C['on_surface']};
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }}
        .gs-team .gs-score {{
            flex: 0 0 auto; font-family: {F['mono']}; font-size: 17px;
            font-weight: 700; color: {C['on_surface']};
        }}
        .gs-team .gs-win-flag {{
            flex: 0 0 auto; font-family: {F['mono']}; font-size: 9px; font-weight: 700;
            background: var(--gs-color, {C['outline']}); color: #fff;
            border-radius: {R['full']}; padding: 1px 6px; letter-spacing: 0.06em;
        }}
        .gs-team.gs-won {{ background: {C['surface_container_high']}; }}
        .gs-team.gs-won .gs-score {{ color: {C['primary']}; }}
        /* A tie dims NEITHER team - both rows get no state class at all. */
        .gs-team.gs-lost .gs-name {{ opacity: 0.55; }}
        .gs-team.gs-lost .gs-score {{ opacity: 0.6; }}

        /* ---- Box score panel (full width, above the card grid) ---------
           Same visual language as a card, scaled up: this is the one place
           in the tab that gets the whole width, so the type can be bigger
           and the two teams can sit as full rows rather than compressed
           ones. */
        .bs-header {{
            background: {C['surface_container_low']};
            border: 1px solid {C['outline_variant']};
            border-radius: {R['md']}; padding: 14px 16px 10px; margin-bottom: 10px;
        }}
        .bs-team {{
            display: flex; align-items: center; gap: 12px;
            padding: 9px 12px; border-radius: {R['default']};
            border-left: 4px solid var(--bs-color, {C['outline']});
            background: {C['surface_container']}; margin-bottom: 6px;
        }}
        .bs-team .bs-side {{
            flex: 0 0 44px; font-family: {F['mono']}; font-size: 10px;
            letter-spacing: 0.08em; color: {C['on_surface_variant']}; opacity: 0.7;
        }}
        .bs-team .bs-logo {{
            height: 32px; width: 32px; object-fit: contain; flex: 0 0 32px;
            filter: drop-shadow(0 1px 3px rgba(0, 0, 0, 0.55));
        }}
        .bs-team .bs-name {{
            flex: 1 1 auto; font-family: {F['display']}; font-size: 18px;
            font-weight: 700; color: {C['on_surface']};
        }}
        .bs-team .bs-score {{
            flex: 0 0 auto; font-family: {F['mono']}; font-size: 26px;
            font-weight: 700; color: {C['on_surface']};
        }}
        .bs-team.bs-won .bs-score {{ color: {C['primary']}; }}
        .bs-team.bs-lost .bs-name, .bs-team.bs-lost .bs-score {{ opacity: 0.6; }}
        .bs-meta {{
            display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px;
            font-family: {F['mono']}; font-size: 10.5px; letter-spacing: 0.04em;
            text-transform: uppercase; color: {C['on_surface_variant']};
        }}

        /* One SHARED track per stat, split by each side's share, rather
           than two independent bars - the comparison is between the two
           halves of one row, and two separate bars make the reader do that
           subtraction themselves. */
        .bs-compare {{
            background: {C['surface_container_low']};
            border: 1px solid {C['outline_variant']};
            border-radius: {R['md']}; padding: 12px 16px; margin-bottom: 10px;
        }}
        .bs-cmp-row {{
            display: grid; grid-template-columns: 54px 1fr 54px;
            align-items: center; gap: 8px; margin-bottom: 9px;
        }}
        .bs-cmp-label {{
            text-align: center; font-family: {F['mono']}; font-size: 10px;
            letter-spacing: 0.06em; text-transform: uppercase;
            color: {C['on_surface_variant']};
        }}
        .bs-cmp-val {{
            font-family: {F['mono']}; font-size: 14px; font-weight: 600;
            color: {C['on_surface_variant']};
        }}
        .bs-cmp-row .bs-cmp-val:last-of-type {{ text-align: right; }}
        .bs-cmp-lead {{ color: {C['on_surface']}; font-weight: 700; }}
        .bs-cmp-track {{
            grid-column: 1 / -1; display: flex; height: 7px; gap: 2px;
            border-radius: {R['full']}; overflow: hidden;
            background: {C['surface_container_high']};
        }}
        .bs-cmp-fill {{ height: 100%; }}

        /* Descendant selector, NOT `.stButton > button`. Streamlit wraps a
           button carrying help= in two extra spans, so the direct-child form
           matches nothing - with no error, while Streamlit's own accent
           hover keeps the buttons looking styled. !important because
           Streamlit injects its button CSS with emotion at runtime, AFTER
           this block, so an equally-specific rule loses the cascade. */
        div[class*="st-key-gs_card_"] .stButton button {{
            background: transparent !important;
            border: 1px solid {C['outline_variant']} !important;
            color: {C['on_surface_variant']} !important;
            font-size: 11.5px !important;
            padding: 3px 8px !important;
        }}
        div[class*="st-key-gs_card_"] .stButton button:hover:not(:disabled) {{
            background: {C['surface_container_high']} !important;
            border-color: {C['primary']} !important;
            color: {C['primary']} !important;
        }}
        div[class*="st-key-gs_card_"] .stButton button:disabled {{
            opacity: 0.4 !important;
        }}
        </style>
    """, unsafe_allow_html=True)


# Full 7-stop diverging scale, poor (0) -> elite (100), per explicit user
# color picks: Red, Softer Red, Softer Orange, Blue-Green, Light Blue,
# Darker Blue, Lavender. Replaces the earlier 5-stop version's plain
# desaturated-gray "neutral" midpoint with a real hue (Blue-Green) - with
# every stop tuned to a similar moderate lightness/saturation band (see the
# per-stop luminance check in this palette's design pass), no single stop
# reads as noticeably heavier/darker than its neighbors despite spanning
# the full red-to-violet range. "Darker Blue" is still kept light enough
# (~L 50%) to satisfy "none of the colors too dark" - it's darker only
# relative to "Light Blue" beside it, not dark in an absolute sense.
_GRADE_COLOR_STOPS = [
    (0, (208, 98, 94)),      # red - poor/bottom-tier
    (16, (214, 126, 112)),   # softer red
    (32, (223, 165, 122)),   # softer orange
    (50, (117, 181, 172)),   # blue-green - average/transition
    (68, (137, 178, 224)),   # light blue
    (84, (100, 130, 205)),   # darker blue
    (100, (172, 148, 205)),  # lavender - elite/top-tier
]
# rgba alpha < 1 rather than a flat opaque hex - softer/less neon against
# the dark navy surface than a fully saturated fill, per user feedback that
# the old colors felt harsh, while still reading clearly against white text.
_GRADE_ALPHA = 0.82

# Distinct rookie marker color - a muted gold/amber, not a shade of blue.
# Rookies used to render in a sky-blue (14, 165, 233) that sat too close to
# _GRADE_COLOR_STOPS' own "elite" blue at the top of the scale (100), which
# made a highly-graded veteran and an ungraded rookie easy to confuse at a
# glance - exactly the opposite of what a dedicated rookie color should do.
# Gold reads as "notable prospect" without implying any grade tier at all
# (no red/peach/slate/lavender/blue hue overlaps it), and is muted to the
# same alpha as every other grade color so it doesn't visually shout.
_ROOKIE_COLOR_RGB = (214, 172, 90)

# Raw PFF grades are NOT uniformly spread 0-100 the way a percentile rank
# is - real season-long grades cluster heavily in roughly 55-90, with very
# few players (rookies/backups with a handful of bad snaps aside) ever
# grading below 30. Feeding a raw grade straight into _GRADE_COLOR_STOPS'
# EVENLY spaced breakpoints wasted most of the color range on an almost-
# empty 0-50 band, so two players separated by 15 real grade points (e.g.
# 65 vs 80) still read as nearly the same washed-out neutral/lavender -
# the opposite of "make good and bad more obvious." This remaps a raw grade
# through a curve calibrated to where PFF's own grades actually cluster
# (roughly 60 = average, 75+ = good, 88+ = elite) BEFORE it reaches
# get_pff_color, so the visible color range concentrates where the real
# variation lives. Percentile-rank inputs (already uniform by construction -
# every other caller of get_pff_color) are untouched; this only applies
# when raw_grade=True.
_RAW_GRADE_NORMALIZATION = [
    (0, 0), (30, 8), (45, 18), (60, 40), (75, 65), (88, 88), (100, 100),
]


def _normalize_raw_grade(val):
    for (lo_val, lo_norm), (hi_val, hi_norm) in zip(_RAW_GRADE_NORMALIZATION, _RAW_GRADE_NORMALIZATION[1:]):
        if lo_val <= val <= hi_val:
            frac = (val - lo_val) / (hi_val - lo_val)
            return lo_norm + (hi_norm - lo_norm) * frac
    return _RAW_GRADE_NORMALIZATION[-1][1]


def get_pff_color(val, is_rookie=False, raw_grade=False):
    """
    raw_grade=True: val is a genuine 0-100 PFF grade (not a percentile
    rank) - passes through _normalize_raw_grade first so its skewed real-
    world distribution reads correctly against _GRADE_COLOR_STOPS' evenly
    spaced breakpoints. Leave False (default) for percentile-rank inputs,
    which are already uniform and need no remapping.
    """
    if is_rookie:
        r, g, b = _ROOKIE_COLOR_RGB
        return f"rgba({r}, {g}, {b}, {_GRADE_ALPHA})"
    try:
        val = float(val)
    except (TypeError, ValueError):
        return C['surface_container_high']
    if pd.isna(val) or val <= 0:
        return C['surface_container_high']
    if raw_grade:
        val = _normalize_raw_grade(val)

    # No "below the floor" branch needed - the stop table's floor is 0 and
    # anything <= 0 already returned above, so val is always in-range here.
    for (lo_val, lo_rgb), (hi_val, hi_rgb) in zip(_GRADE_COLOR_STOPS, _GRADE_COLOR_STOPS[1:]):
        if lo_val <= val <= hi_val:
            frac = (val - lo_val) / (hi_val - lo_val)
            r = round(lo_rgb[0] + (hi_rgb[0] - lo_rgb[0]) * frac)
            g = round(lo_rgb[1] + (hi_rgb[1] - lo_rgb[1]) * frac)
            b = round(lo_rgb[2] + (hi_rgb[2] - lo_rgb[2]) * frac)
            return f"rgba({r}, {g}, {b}, {_GRADE_ALPHA})"
    r, g, b = _GRADE_COLOR_STOPS[-1][1]
    return f"rgba({r}, {g}, {b}, {_GRADE_ALPHA})"


# Softer, muted diverging stops for matchup-style tables (fantasy points
# allowed, coverage scheme tendencies) - feedback was that get_pff_color's
# brighter rainbow (tuned for 0-100 PFF grades, with blue/purple "elite"
# tiers) read as too dramatic/neon for a plain good-matchup-vs-bad-matchup
# read. Light green (good) -> red (bad), several muted steps between,
# same continuous-interpolation approach as get_pff_color just with a
# narrower, softer palette and no blue/purple tier (nothing here is scored
# on a 0-100 grade scale, so an "elite" color tier doesn't apply).
_MATCHUP_COLOR_STOPS = [
    (0, (176, 62, 56)),     # muted red - worst percentile
    (25, (196, 130, 68)),   # muted orange
    (50, (196, 178, 76)),   # muted yellow
    (75, (150, 189, 104)),  # light green
    (100, (99, 163, 105)),  # green - best percentile
]
_MATCHUP_ALPHA = 0.75


def get_matchup_color(pct):
    """
    pct: 0-100, where 100 = "good" (light green) and 0 = "bad" (red) - the
    caller computes pct in whatever direction makes sense for that column
    (e.g. fewest fantasy points allowed = highest pct, via
    calculate_percentile(..., ascending=False) so the stingiest defense
    reads green, not red).
    """
    try:
        pct = float(pct)
    except (TypeError, ValueError):
        return C['surface_container_high']
    if pd.isna(pct):
        return C['surface_container_high']
    pct = max(0.0, min(100.0, pct))
    for (lo_val, lo_rgb), (hi_val, hi_rgb) in zip(_MATCHUP_COLOR_STOPS, _MATCHUP_COLOR_STOPS[1:]):
        if lo_val <= pct <= hi_val:
            frac = (pct - lo_val) / (hi_val - lo_val)
            r = round(lo_rgb[0] + (hi_rgb[0] - lo_rgb[0]) * frac)
            g = round(lo_rgb[1] + (hi_rgb[1] - lo_rgb[1]) * frac)
            b = round(lo_rgb[2] + (hi_rgb[2] - lo_rgb[2]) * frac)
            return f"rgba({r}, {g}, {b}, {_MATCHUP_ALPHA})"
    r, g, b = _MATCHUP_COLOR_STOPS[-1][1]
    return f"rgba({r}, {g}, {b}, {_MATCHUP_ALPHA})"


def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _blend_hex(c1, c2, frac):
    frac = max(0.0, min(1.0, frac))
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return f"#{round(r1 + (r2 - r1) * frac):02x}{round(g1 + (g2 - g1) * frac):02x}{round(b1 + (b2 - b1) * frac):02x}"


def get_diverging_color(val, max_abs):
    """
    Red-green diverging color for signed DELTA columns (e.g. "VORP vs
    FantasyPros"), scaled by magnitude relative to max_abs rather than by
    percentile rank. get_pff_color's percentile-heatmap scale is built for
    absolute 0-100 grades - applying it to a signed comparison metric is
    semantically backwards (a small delta near zero could land in a
    saturated color bucket just because of its percentile rank, while the
    actual sign/magnitude a viewer cares about - "does this app or the
    external ranking like this player more, and by how much" - gets lost).
    Neutral (near zero) stays a plain neutral tone; positive blends toward
    green, negative toward red, saturating at +/-max_abs.
    """
    if pd.isna(val) or not max_abs:
        return C['surface_container_high']
    frac = max(-1.0, min(1.0, float(val) / max_abs))
    if abs(frac) < 0.05:
        return C['surface_container_high']
    return _blend_hex(C['surface_container_high'], C['positive'] if frac > 0 else C['negative'], abs(frac))


def _cell_text(val):
    if pd.isna(val):
        return ''
    if isinstance(val, float):
        # Plain float columns (not run through STAT_DECIMALS-aware
        # _format_stat_value, since generic tables cover many unrelated
        # metrics) still need a sane default precision - otherwise a mean
        # or a raw API value renders as e.g. "2.500000" or
        # "1.2999999523162842" instead of "2.5"/"1.3".
        return f"{val:.2f}".rstrip('0').rstrip('.') if val != int(val) else str(int(val))
    return str(val)



# Small-magnitude ratio stats lose real signal at the default 1-decimal
# precision the auto-formatter below falls back to (e.g. two RBs at 0.85
# vs 0.95 RYOE/Att both round to "0.9") - forced to 2 decimals regardless
# of what the whole-number heuristic would otherwise pick.
_FORCED_DECIMALS = {
    'RYOE/Att': 2,
}


# Tier colours, cycling from best to worst. Deliberately a categorical ramp
# rather than a continuous gradient: a tier IS a category - the whole point
# of tiering is that everyone inside one is interchangeable and the step
# BETWEEN two is a real cliff. A smooth gradient would draw the cliff as a
# gentle slope, which is the one thing the column exists to contradict.
#
# Ordered cool-to-warm so the eye reads "elite -> deep" without a legend,
# and the band boundaries land where a scan naturally stops.
TIER_COLORS = [
    '#1b6ac9',  # 1 - elite
    '#1f8a70',  # 2
    '#3f9d3a',  # 3
    '#8a9a1e',  # 4
    '#b8860b',  # 5
    '#c26a1c',  # 6
    '#b3452c',  # 7
    '#8e3b52',  # 8+
]


def get_tier_color(tier):
    """Background for a tier cell, or None when there's no tier to show."""
    try:
        index = int(tier) - 1
    except (TypeError, ValueError):
        return None
    if index < 0:
        return None
    return TIER_COLORS[min(index, len(TIER_COLORS) - 1)]


def style_plain_dataframe(df, numeric_pct_cols=None, diverging_cols=None, matchup_pct_cols=None, tier_cols=None):
    """
    Sortable Styler for st.dataframe (historical totals, risers, rookie
    watch, rankings, VORP sheet, odds tables, coverage scheme tendencies).

    NOTE on why this is a Styler + st.dataframe again, not raw HTML: a
    prior pass switched every one of these to static HTML specifically
    because st.dataframe's interactive grid (glide-data-grid) draws its own
    header row from Streamlit's THEME CONFIG, not from page CSS or a
    Styler's `<th>` rules - so CSS overrides never reached it. The actual
    fix is a proper .streamlit/config.toml dark theme (which the grid DOES
    read for its header/cell colors), restoring native column sorting,
    search, and CSV download - all lost by going static. Header cells
    render however the active theme defines them; this Styler only
    controls per-cell data coloring (the percentile heatmap), not headers.

    numeric_pct_cols: dict of {column_name: percentile-like sequence},
    matched by ROW POSITION - the original version matched by index label
    against a differently-indexed Series and silently never matched.

    diverging_cols: dict of {column_name: max_abs} for signed delta columns
    (e.g. "VORP vs FantasyPros") - colored red/green by magnitude via
    get_diverging_color instead of the percentile-grade heatmap, since a
    signed comparison metric isn't an absolute 0-100 grade (see that
    function's docstring). Takes precedence over numeric_pct_cols for any
    column listed in both.

    matchup_pct_cols: dict of {column_name: percentile-like sequence}, same
    row-position matching as numeric_pct_cols, but colored via the softer
    get_matchup_color (light green/red) instead of get_pff_color's brighter
    0-100-grade rainbow - for matchup-style tables (fantasy points allowed,
    coverage scheme tendencies) where a PFF-grade-style blue/purple "elite"
    tier doesn't apply. Takes precedence over numeric_pct_cols for any
    column listed in both (checked after diverging_cols).

    tier_cols: dict of {column_name: tier-number sequence}, same row-position
    matching, colored via get_tier_color's fixed tier palette instead of a
    continuous percentile scale - for a column whose own displayed text
    ISN'T the tier number itself (e.g. Weekly Rankings' "Rank" column shows
    "RB4", not "3", but should still be shaded by which tier RB4 landed in).
    A column literally named 'Tier' already colors off its own value further
    down; this is for coloring a DIFFERENT column by a tier computed
    alongside it. Takes precedence over everything else, since a caller that
    passes this has already decided tier is the right read for that column.

    Every numeric column also gets an explicit, auto-detected decimal count
    (see the loop below) - without it, whatever raw float precision the
    underlying dtype happens to carry gets displayed as-is: a float32
    groupby sum can drift to something like "44.999998" from binary
    imprecision, and a genuinely whole-number count (games played, dropbacks,
    American odds) still shows a bare trailing ".0". Both read as visual
    noise/inconsistent precision across tables that otherwise look
    deliberately formatted.
    """
    numeric_pct_cols = numeric_pct_cols or {}
    diverging_cols = diverging_cols or {}
    matchup_pct_cols = matchup_pct_cols or {}
    tier_cols = tier_cols or {}
    pct_arrays = {col: list(vals) for col, vals in numeric_pct_cols.items()}
    matchup_arrays = {col: list(vals) for col, vals in matchup_pct_cols.items()}
    tier_arrays = {col: list(vals) for col, vals in tier_cols.items()}

    _DEFAULT_STYLE = f"background-color:{C['surface_container']}; color:{C['on_surface']};"

    # Built column-by-column and applied in one axis=None Styler.apply call,
    # instead of the previous df.style.apply(func, axis=1): pandas' axis=1
    # apply constructs a full pandas Series (with index alignment) for every
    # single row before the function body even runs, which is pure overhead
    # here since every branch below only ever reads one column at a time.
    # Same per-cell precedence (diverging > matchup > percentile > Team >
    # default) and same output, confirmed identical against the old
    # implementation on real data - this only changes the iteration order,
    # not what gets colored. Matters most on the app's largest tables
    # (full-season Rankings/Risers/VORP sheets, several hundred rows) which
    # get rebuilt from scratch on every widget interaction in that tab.
    def style_column(col):
        values = df[col]
        if col in diverging_cols:
            max_abs = diverging_cols[col]
            return [
                f'background-color:{get_diverging_color(v, max_abs)}; color:#ffffff; font-weight:bold;'
                for v in values
            ]
        matchup_vals = matchup_arrays.get(col)
        pct_vals = pct_arrays.get(col)
        tier_vals = tier_arrays.get(col)
        is_team = col == 'Team'
        is_position = col in ('Pos', 'Position')
        is_tier = col == 'Tier'
        out = []
        for pos, v in enumerate(values):
            if is_tier:
                # Handled before the percentile branch so a Tier column that
                # also appears in numeric_pct_cols still reads as tiers.
                tier_bg = get_tier_color(v)
                out.append(f"background-color:{tier_bg}; color:#ffffff; font-weight:bold;"
                           if tier_bg else _DEFAULT_STYLE)
                continue
            if tier_vals is not None and pos < len(tier_vals):
                tier_bg = get_tier_color(tier_vals[pos])
                out.append(f"background-color:{tier_bg}; color:#ffffff; font-weight:bold;"
                           if tier_bg else _DEFAULT_STYLE)
                continue
            if matchup_vals is not None and pos < len(matchup_vals):
                bg = get_matchup_color(matchup_vals[pos])
                out.append(f'background-color:{bg}; color:#ffffff; font-weight:bold;')
            elif pct_vals is not None and pos < len(pct_vals):
                bg = get_pff_color(pct_vals[pos])
                out.append(f'background-color:{bg}; color:#ffffff; font-weight:bold;')
            elif is_position:
                # Position chips, same mechanism as the Team column. Under a
                # pick clock the position is the first thing scanned on every
                # row - a color does that in peripheral vision where reading
                # two letters does not.
                from config import get_position_color, get_position_chip_bg
                chip_bg = get_position_chip_bg(v)
                if chip_bg:
                    out.append(f"background-color:{chip_bg}; color:{get_position_color(v)}; font-weight:bold;")
                else:
                    out.append(_DEFAULT_STYLE)
            elif is_team:
                # Same team-color convention already used for the bio card
                # accent, depth chart Position column, and game log opponent
                # cell - ties every "Team" column app-wide into one
                # consistent visual language instead of plain text, and
                # makes a given team's players easy to spot scanning down a
                # sorted table (e.g. after sorting Risers by Pct Jump).
                team_color = TEAM_CONFIG.get(str(v), {}).get('color')
                if team_color:
                    out.append(f"background-color:{team_color}; color:#ffffff; font-weight:bold;")
                else:
                    out.append(_DEFAULT_STYLE)
            else:
                out.append(_DEFAULT_STYLE)
        return out

    style_df = pd.DataFrame({col: style_column(col) for col in df.columns}, index=df.index)
    styler = df.style.apply(lambda _: style_df, axis=None)

    fmt = {}
    for col in df.columns:
        # Booleans are numeric as far as pandas is concerned, and a
        # "{:.0f}" on one turns a checkbox column's True into a literal "1".
        # They carry no decimals to decide on, so they're skipped outright.
        if pd.api.types.is_bool_dtype(df[col]) or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if col in _FORCED_DECIMALS:
            decimals = _FORCED_DECIMALS[col]
        else:
            non_null = df[col].dropna()
            # Tolerance (not an exact == check) so float32 binary-imprecision
            # on a genuinely whole-number count (e.g. 44.999998) still reads
            # as whole rather than falling through to 1 decimal.
            is_whole = non_null.empty or (non_null.round(0) - non_null).abs().max() < 0.05
            decimals = 0 if is_whole else 1
        fmt[col] = f"{{:.{decimals}f}}"
    if fmt:
        styler = styler.format(fmt, na_rep='--')

    return styler


def style_gradient_dataframe(df, cmap='RdYlGn_r', axis=0, decimals=1):
    """Sortable Styler wrapper around pandas' own background_gradient, themed
    to dark text on the (bright) gradient cells for contrast."""
    return df.style.background_gradient(cmap=cmap, axis=axis).format(f"{{:.{decimals}f}}")


def df_auto_height(n_rows, row_px=35, header_px=38, padding_px=3):
    """
    st.dataframe defaults to a fixed-height scrollable container, which is
    what was causing the depth chart to need an internal scrollbar to see
    every position even at 100% browser zoom. Passing an explicit height
    sized to the actual row count shows everything without scrolling.
    """
    return header_px + max(n_rows, 1) * row_px + padding_px


def _format_stat_value(col, val):
    from config import STAT_DECIMALS
    decimals = STAT_DECIMALS.get(col, 0)
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return val
    if pd.isna(fv):
        return 0.0 if decimals else 0
    if decimals > 0:
        return round(fv, decimals)
    # round (not truncate) before casting to int - a float32 precision
    # artifact like 244.99998 would otherwise silently truncate to 244
    # instead of the correct 245.
    return int(round(fv))


def render_game_log_html_table(log_df_view, avg_source_df, log_cols, header_map, max_height=380):
    """
    Single raw HTML table (not two stacked st.dataframe components) so the
    weekly rows and the AVG row read as ONE table, not two boxes with a
    visible seam between them. The header row and the AVG row both use
    `position: sticky` within one `overflow-y: auto` wrapper div, so the
    header stays visible while scrolling and the AVG row - the actual last
    row of the table - stays pinned to the bottom of that same scrolling
    area instead of scrolling out of view with the weekly rows above it.
    st.dataframe can't do this (each call renders as its own independent
    grid component with no way to pin one row within another component's
    scroll region), so this renders straight HTML via st.markdown instead,
    the same approach already used for the positional matrix table.

    avg_source_df determines what the AVG row averages - the caller passes
    the SAME week-range-filtered frame as log_df_view (not the full,
    unfiltered season log), so narrowing the week-range slider updates the
    average shown, not just which rows are visible. Kept as a separate
    parameter rather than always re-deriving it from log_df_view in case a
    future caller wants the two to differ again.
    """
    display_cols = [header_map.get(c, c.replace('_', ' ').upper()) for c in log_cols]

    def week_cell(col, r):
        val = _format_stat_value(col, r.get(col, 0))
        if col == 'week':
            return str(val), f"background-color:{C['surface_container']}; color:{C['on_surface']};"
        if col == 'opponent_team':
            team = str(r.get(col, '')).strip().upper() if pd.notna(r.get(col)) else '--'
            t_bg = TEAM_CONFIG.get(team, {}).get('color', C['surface_container'])
            return team, f"background-color:{t_bg}; color:#ffffff; font-weight:bold;"
        pct_val = r.get(f"{col}_wk_pct", 0)
        bg = get_pff_color(pct_val)
        return val, f"background-color:{bg}; color:#ffffff; font-weight:bold;"

    body_rows_html = []
    for i, (_, r) in enumerate(log_df_view.iterrows()):
        cells = []
        for col in log_cols:
            text, style = week_cell(col, r)
            cells.append(f"<td style='padding:8px; white-space:nowrap; text-align:center; border:1px solid {C['outline_variant']}; font-family:\"JetBrains Mono\",monospace; {style}'>{text}</td>")
        # gl-row + a staggered animation-delay (capped so a full 17+ week
        # season log still finishes revealing in well under a second) gives
        # this table a cascading entrance instead of popping in all at once
        # - see .gl-row in ui/styling.py's inject_theme for the keyframe.
        body_rows_html.append(f"<tr class='gl-row' style='animation-delay:{min(i, 20) * 18}ms;'>{''.join(cells)}</tr>")

    from config import STAT_DECIMALS
    avg_cells = []
    for col, disp in zip(log_cols, display_cols):
        if col == 'week':
            text = 'AVG'
        elif col == 'opponent_team':
            text = '--'
        else:
            mean_val = pd.to_numeric(avg_source_df[col], errors='coerce').mean()
            decimals = max(STAT_DECIMALS.get(col, 0), 1)
            # Explicit fixed-decimal string formatting, not round()'s raw
            # value - a float32 mean (e.g. 1.2999999523...) still prints
            # with its full binary-imprecision tail if you round() it and
            # let f-string interpolate the result directly; formatting with
            # `:.{decimals}f` forces the display string itself to the
            # intended precision.
            text = f"{mean_val:.{decimals}f}" if pd.notna(mean_val) else f"{0.0:.{decimals}f}"
        avg_cells.append(
            f"<td style='padding:8px; white-space:nowrap; text-align:center; "
            f"background-color:{C['surface_dim']}; color:{C['primary']}; font-weight:800; "
            f"border:2px solid {C['primary']}; font-family:\"JetBrains Mono\",monospace;'>{text}</td>"
        )

    header_cells = "".join(
        f"<th style='padding:12px 10px; white-space:nowrap; background-color:{C['surface_container_high']}; "
        f"color:{C['primary']}; font-size:12px; text-transform:uppercase; font-weight:700; letter-spacing:0.05em; "
        f"font-family:\"Poppins\",sans-serif; border:1px solid {C['outline_variant']}; position:sticky; top:0; z-index:2;'>{d}</th>"
        for d in display_cols
    )

    table_html = f"""
    <div style="max-height:{max_height}px; overflow-y:auto; border:1px solid {C['outline_variant']}; border-radius:{R['sm']}; background-color:{C['surface_container']};">
      <table style="width:100%; border-collapse:collapse;">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>
          {''.join(body_rows_html)}
          <tr style="position:sticky; bottom:0; z-index:3; box-shadow:0 -6px 12px rgba(0,0,0,0.35);">{''.join(avg_cells)}</tr>
        </tbody>
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def style_depth_chart_table(dc_df, sel_team, snap_map, pff_grades_map, global_rookie_names, veteran_names, league_gold_players, has_match_map=None, games_map=None):
    """
    Sortable Styler for the depth chart (see style_plain_dataframe's
    docstring for why this is a Styler + st.dataframe, not static HTML -
    headers now come from the .streamlit/config.toml dark theme). Same
    per-cell PFF-grade heatmap coloring, gold league-leader border, and
    Position column team color as before.

    has_match_map (exact_name -> bool) distinguishes "genuinely 0% snaps"
    from "no snap-count row for this player at all" (mainly offensive
    linemen, whose snaps this local export barely tracks) - shows "N/A"
    for the latter instead of a misleading "Snap: 0%".
    """
    from data.utils import clean_name_exact, build_last_name_index, match_abbreviated_name
    has_match_map = has_match_map or {}
    games_map = games_map or {}
    # Depth-chart cells are named from roster_weekly's abbreviated
    # "F.Last" convention (e.g. "P.Mahomes" - see
    # data.transforms.fetch_intelligent_depth_chart's docstring for why
    # that column is preferred), but pff_grades_map is keyed by full
    # lowercased name ("patrick mahomes"). A direct .get() on the
    # abbreviated cell text against that map missed 100% of the time
    # (confirmed empirically) - this index bridges initial+last-name to
    # the full name once per table render, not per cell.
    pff_name_index = build_last_name_index(pff_grades_map.keys())
    # SAME abbreviated-name bridge, for the SAME reason, on the rookie
    # check below - this was the bigger half of "PFF grades were lost":
    # is_rk fell through to True for nearly every player before this bridge
    # existed, so almost the entire table showed "PFF: ROOKIE" instead of a
    # real grade even for the league's actual PFF-graded players.
    rookie_name_index = build_last_name_index(global_rookie_names)
    # veteran_names (still accepted as a parameter - see build_cell below)
    # is no longer consulted here: the "not found in veteran_names, assume
    # rookie" fallback that used to read it was removed for misidentifying
    # real veteran backups as rookies.
    gold_name_index = build_last_name_index(league_gold_players)

    def build_cell(cell_val):
        # "nan" (not just a real NaN) - some empty depth-chart slots arrive
        # as the literal 3-character string "nan" (a stringified NaN from
        # further upstream), which pd.isna() doesn't catch and str().strip()
        # doesn't reduce to "" - those were rendering as a visible "nan"
        # player name instead of a blank slot.
        if pd.isna(cell_val) or str(cell_val).strip().lower() in ("", "nan"):
            return "", f"background-color:{C['surface_container']};"
        clean_name = str(cell_val).split(' (')[0].strip()
        merge_key = clean_name_exact(pd.Series([clean_name])).iloc[0]
        snaps = snap_map.get(merge_key, 0.0)
        # Season snap SHARE (share of the team's total season snaps - see
        # _build_snap_totals in ui/tabs/depth_charts.py), with the games
        # count alongside so a mid-season starter change reads correctly:
        # "53% · 9g" and "45% · 8g" instead of two QBs both at "100%".
        games = games_map.get(merge_key, 0)
        games_suffix = f" · {int(games)}gp" if games else ""
        snap_text = f"{snaps:.0f}%{games_suffix}" if has_match_map.get(merge_key, True) else "N/A"
        pff_grade = pff_grades_map.get(clean_name.lower())
        if pff_grade is None:
            matched_full_name = match_abbreviated_name(clean_name, pff_name_index)
            if matched_full_name:
                pff_grade = pff_grades_map.get(matched_full_name)
        # is_rk is ONLY the data-driven rookie_year/years_exp flag
        # (global_rookie_names) - a "not found in veteran_names, so assume
        # rookie" fallback used to sit here too, but veteran_names is a
        # last-3-years roster lookup with the same name-matching gaps this
        # whole codebase has had to bridge everywhere else, and a real
        # veteran BACKUP who simply has no snaps-based PFF grade yet was
        # falling through it exactly as often as a genuine rookie was -
        # showing "PFF: ROOKIE" (and the rookie-blue color) for players who
        # are neither rookies nor graded. Confirmed: the fallback predates
        # match_abbreviated_name's full-prefix fix and the rookie/veteran
        # name-index bridging both already applied above - the gap it was
        # covering for is largely closed now, and it was producing more
        # false positives (misidentified veterans) than it was catching.
        is_rk = clean_name.lower() in global_rookie_names or bool(match_abbreviated_name(clean_name, rookie_name_index))
        # Terser shorthand ("Name grade · snap%·gp" - no "(Snap:", "PFF:",
        # commas/parens) per explicit feedback that spelling out both
        # labels in every single cell, across a whole dense grid, read as
        # busy/cluttered rather than clean. The format is explained once,
        # in a caption above the table (ui/tabs/depth_charts.py), instead
        # of repeating the labels per cell. Cell color still carries the
        # PFF-grade tier on its own (get_pff_color below), same as before.
        if pd.notnull(pff_grade) and pff_grade > 0:
            cell_text = f"{clean_name} {pff_grade:.1f} · {snap_text}"
            cell_grade = float(pff_grade)
        elif is_rk:
            cell_text = f"{clean_name} RK · {snap_text}"
            cell_grade = 0.0
        else:
            # Genuinely no PFF grade and not a rookie (common for backups
            # with too few snaps to be graded) - "--", not a fabricated
            # grade riding the same rainbow scale as a real number, and
            # get_pff_color(0.0, is_rookie=False) below already falls
            # through to the neutral gray, never the rookie color or a
            # grade color, for this exact case.
            cell_text = f"{clean_name} -- · {snap_text}"
            cell_grade = 0.0
        bg = get_pff_color(cell_grade, is_rookie=is_rk, raw_grade=True)
        is_gold_name = clean_name.lower() in league_gold_players or bool(match_abbreviated_name(clean_name, gold_name_index))
        is_gold = is_gold_name and cell_grade > 0 and not is_rk
        style = f"background-color:{bg}; color:#ffffff; font-weight:bold;"
        if is_gold:
            # NOT a border/box-shadow: st.dataframe's grid (glide-data-grid)
            # draws cells on an HTML canvas, not real DOM elements - it only
            # reads background-color/color/font-weight out of the Styler's
            # CSS, silently dropping border/box-shadow declarations (they
            # show up fine in styler.to_html() since that's a separate,
            # unrelated static-HTML code path pandas Styler also supports -
            # confirmed this was the actual reason the gold border never
            # appeared in the real rendered grid despite looking correct in
            # that HTML). The trophy prefix is a real character in the cell
            # text, so it always renders - same fix already used on the
            # Player Search bio card's name line.
            cell_text = f"🏆 {cell_text}"
        return cell_text, style

    text_rows, style_rows = [], []
    team_bg = TEAM_CONFIG.get(sel_team, {}).get('color', C['surface_container_high'])
    for _, row in dc_df.iterrows():
        text_row, style_row = {}, {}
        for col, val in row.items():
            if col == 'Position':
                text_row[col] = val
                style_row[col] = f"background-color:{team_bg}; color:#ffffff; font-weight:bold;"
            else:
                text, style = build_cell(val)
                text_row[col] = text
                style_row[col] = style
        text_rows.append(text_row)
        style_rows.append(style_row)

    display_df = pd.DataFrame(text_rows, columns=dc_df.columns)
    style_df = pd.DataFrame(style_rows, columns=dc_df.columns)

    def apply_styles(row):
        return [style_df.loc[row.name, col] for col in display_df.columns]

    return display_df.style.apply(apply_styles, axis=1)
