"""
Inline-SVG chart primitives.

Why SVG rather than matplotlib or Vega-Lite, given the app already uses
both: these charts live in half-width columns on the Matchup Analyzer,
where a matplotlib PNG rendered at a fixed DPI and then stretched reads as
soft/low-quality (the same complaint that moved the Coverage Matchup Radar
into a centred narrow column), and Vega-Lite can't be styled to match the
rest of this app's surfaces without fighting its own theme. A fixed
`viewBox` with `width:100%; height:auto` scales crisply to any column
width, and every colour comes straight from THEME.

It also buys real tooltips for free - an SVG `<title>` child is a native
browser tooltip, so per-bar detail costs nothing and works on a chart the
user can't hover a Streamlit widget over.

These are pure render functions: they take already-computed numbers and
draw them. Everything that decides WHAT to draw lives in data/ (see
data/matchup_signals.py), same layering as ui.components' own
render_stat_tiles/render_hero_tiles.
"""
import html

import streamlit as st

from config import THEME
from ui.styling import get_pff_color

C = THEME['colors']
F = THEME['fonts']

# THEME's font stacks are single-quoted strings ("'Inter', sans-serif") and
# every attribute below is single-quoted, so embedding them raw closes the
# attribute early and corrupts the tag - the same failure ui.components'
# render_player_card documents. Swapped to double quotes once, here.
_BODY_FONT = F['body'].replace("'", '"')
_MONO_FONT = F['mono'].replace("'", '"')


def _esc(value):
    """Escape for both element text and a quoted attribute - tooltips carry
    real opponent/player names, and an apostrophe in one ("Ja'Marr Chase")
    would otherwise terminate a single-quoted attribute mid-string."""
    return html.escape(str(value), quote=True).replace("'", '&#39;')


def render_game_log_bars(values, tooltips, highlight=None, avg=None, avg_label="season avg", bar_labels=None):
    """
    One bar per game in season order, with the season average as a dashed
    reference line and each game's own value printed at the bar's tip.

    `highlight`: optional list of bools, same length as `values` - a game
    worth calling out (this app uses "a top-quartile game for this player
    this season") is drawn in the primary accent with a star instead of the
    muted secondary.

    `bar_labels`: optional list of (line1, line2) tuples - two small stacked
    lines under each bar, used for opponent + week. Falls back to plain
    1..N game numbering when omitted.

    Renders nothing for an empty series rather than an empty axis, matching
    ui.components.render_fpts_week_strip's convention for a player with no
    played games.
    """
    if not values:
        return
    highlight = highlight or [False] * len(values)
    W, H, MB, MT = 860, 208, 30, 36
    plot_h = H - MB - MT
    n = len(values)
    slot = W / n
    bar_w = min(46.0, slot * 0.62)
    # A season where every value is 0 (a WR with no TDs yet) would divide by
    # zero; 1 keeps every bar at the floor instead of raising.
    vmax = max(max(values), (avg or 0)) or 1
    parts = [
        f"<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' "
        f"style='width:100%; height:auto; font-family:{_BODY_FONT};'>"
    ]
    for i, v in enumerate(values):
        # A real zero gets a 2px stub rather than nothing - an absent bar
        # and a zero bar must not look the same, since "didn't play" and
        # "played and got zero" are completely different reads.
        h = max(2.0, plot_h * float(v) / vmax)
        x = slot * i + (slot - bar_w) / 2
        y = MT + plot_h - h
        is_star = bool(highlight[i]) if i < len(highlight) else False
        fill = C['primary'] if is_star else C['secondary']
        tip = _esc(tooltips[i]) if i < len(tooltips) else ''
        parts.append(
            f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{h:.1f}' rx='3' fill='{fill}' "
            f"opacity='{1.0 if is_star else 0.75}'><title>{tip}</title></rect>"
        )
        if is_star:
            parts.append(
                f"<text x='{x + bar_w / 2:.1f}' y='{y - 6:.1f}' text-anchor='middle' font-size='12' "
                f"fill='{C['primary']}'>★<title>{tip}</title></text>"
            )
        value_text = f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"
        parts.append(
            f"<text x='{x + bar_w / 2:.1f}' y='{y - (20 if is_star else 6):.1f}' text-anchor='middle' "
            f"font-size='9.5' font-weight='600' font-family='{_MONO_FONT}' fill='{C['on_surface']}'>"
            f"{value_text}<title>{tip}</title></text>"
        )
        if bar_labels and i < len(bar_labels) and bar_labels[i]:
            line1, line2 = bar_labels[i]
            for offset, text in ((17, line1), (5, line2)):
                parts.append(
                    f"<text x='{x + bar_w / 2:.1f}' y='{H - offset}' text-anchor='middle' font-size='8.5' "
                    f"font-family='{_MONO_FONT}' fill='{C['on_surface_variant']}' opacity='0.9'>{_esc(text)}"
                    f"<title>{tip}</title></text>"
                )
        else:
            parts.append(
                f"<text x='{x + bar_w / 2:.1f}' y='{H - 8}' text-anchor='middle' font-size='9.5' "
                f"font-family='{_MONO_FONT}' fill='{C['on_surface_variant']}'>{i + 1}</text>"
            )
    if avg is not None and vmax:
        ay = MT + plot_h - plot_h * float(avg) / vmax
        parts.append(
            f"<line x1='0' y1='{ay:.1f}' x2='{W}' y2='{ay:.1f}' stroke='{C['on_surface_variant']}' "
            f"stroke-width='1.2' stroke-dasharray='5,5' opacity='0.8'/>"
        )
        parts.append(
            f"<text x='{W - 4}' y='{ay - 5:.1f}' text-anchor='end' font-size='10' font-family='{_MONO_FONT}' "
            f"fill='{C['on_surface_variant']}'>{_esc(avg_label)} {avg:.1f}</text>"
        )
    parts.append("</svg>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_game_log_line(values, tooltips, highlight=None, avg=None, avg_label="season avg", bar_labels=None,
                         avg2=None, avg2_label="avg", avg2_color=None):
    """
    Line-chart twin of render_game_log_bars - same slot layout, same average
    reference line, same per-point value label and week/opponent caption,
    just a smoothed curve THROUGH the points instead of a bar per game. Per
    explicit feedback that a line reads better for a season trend than a bar
    strip does.

    `avg2` is an optional SECOND dashed reference line (e.g. Matchup
    Analyzer's Defense Week By Week Detail chart: `avg` is the league
    average allowed, `avg2` is this defense's own season average, drawn in
    its team color via `avg2_color` so the two lines read as "league" vs
    "this team" without a legend). Its label anchors on the LEFT edge
    (`avg`'s stays on the right) so the two labels don't collide when both
    lines land at a similar height.

    Points sit at the CENTER of the same equal-width per-game "slot" the bar
    chart divides the width into (`slot*i + slot/2`), not spread edge-to-edge
    across the full width the way Player Search's fpts strip does - this is
    deliberate, not an oversight: it's what lets a caller lay real Streamlit
    buttons in `st.columns(n)` underneath (equal-width columns, so column i's
    center lines up with point i's) as a click target, without duplicating
    the x-position math in two places or fighting undocumented SVG-in-HTML
    hit-testing. See ui.tabs.matchup_analyzer's game-by-game section for that
    overlay.

    Returns the list of each point's x position as a 0-1 FRACTION of the
    chart width (not pixels) - the caller needs this only if it's building
    that overlay; every other caller can ignore the return value.
    """
    if not values:
        return []
    from ui.components import _smooth_svg_path
    highlight = highlight or [False] * len(values)
    W, H, MB, MT = 860, 208, 30, 36
    plot_h = H - MB - MT
    n = len(values)
    slot = W / n
    vmax = max(max(values), (avg or 0), (avg2 or 0)) or 1

    def x_at(i):
        return slot * i + slot / 2

    def y_at(v):
        return MT + plot_h - plot_h * float(v) / vmax

    pts = [(x_at(i), y_at(v)) for i, v in enumerate(values)]
    line_path = _smooth_svg_path(pts)

    parts = [
        f"<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' "
        f"style='width:100%; height:auto; font-family:{_BODY_FONT};'>"
    ]
    if avg is not None and vmax:
        ay = MT + plot_h - plot_h * float(avg) / vmax
        parts.append(
            f"<line x1='0' y1='{ay:.1f}' x2='{W}' y2='{ay:.1f}' stroke='{C['on_surface_variant']}' "
            f"stroke-width='1.2' stroke-dasharray='5,5' opacity='0.8'/>"
        )
        parts.append(
            f"<text x='{W - 4}' y='{ay - 5:.1f}' text-anchor='end' font-size='10' font-family='{_MONO_FONT}' "
            f"fill='{C['on_surface_variant']}'>{_esc(avg_label)} {avg:.1f}</text>"
        )
    if avg2 is not None and vmax:
        a2_color = avg2_color or C['primary']
        a2y = MT + plot_h - plot_h * float(avg2) / vmax
        parts.append(
            f"<line x1='0' y1='{a2y:.1f}' x2='{W}' y2='{a2y:.1f}' stroke='{a2_color}' "
            f"stroke-width='1.4' stroke-dasharray='5,5' opacity='0.9'/>"
        )
        parts.append(
            f"<text x='4' y='{a2y - 5:.1f}' text-anchor='start' font-size='10' font-family='{_MONO_FONT}' "
            f"fill='{a2_color}'>{_esc(avg2_label)} {avg2:.1f}</text>"
        )
    area_path = f"{line_path} L{pts[-1][0]:.1f},{MT + plot_h:.1f} L{pts[0][0]:.1f},{MT + plot_h:.1f} Z"
    parts.append(f"<path d='{area_path}' fill='{C['secondary']}' opacity='0.08'/>")
    parts.append(f"<path d='{line_path}' fill='none' stroke='{C['secondary']}' stroke-width='2.4' stroke-linecap='round'/>")

    for i, (x, y) in enumerate(pts):
        is_star = bool(highlight[i]) if i < len(highlight) else False
        fill = C['primary'] if is_star else C['secondary']
        tip = _esc(tooltips[i]) if i < len(tooltips) else ''
        parts.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{4.5 if is_star else 3.5}' fill='{fill}' "
            f"stroke='{C['surface']}' stroke-width='1.3'><title>{tip}</title></circle>"
        )
        if is_star:
            parts.append(
                f"<text x='{x:.1f}' y='{y - 18:.1f}' text-anchor='middle' font-size='12' "
                f"fill='{C['primary']}'>★<title>{tip}</title></text>"
            )
        v = values[i]
        value_text = f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"
        parts.append(
            f"<text x='{x:.1f}' y='{y - (32 if is_star else 10):.1f}' text-anchor='middle' "
            f"font-size='9.5' font-weight='600' font-family='{_MONO_FONT}' fill='{C['on_surface']}'>"
            f"{value_text}<title>{tip}</title></text>"
        )
        if bar_labels and i < len(bar_labels) and bar_labels[i]:
            line1, line2 = bar_labels[i]
            for offset, text in ((17, line1), (5, line2)):
                parts.append(
                    f"<text x='{x:.1f}' y='{H - offset}' text-anchor='middle' font-size='8.5' "
                    f"font-family='{_MONO_FONT}' fill='{C['on_surface_variant']}' opacity='0.9'>{_esc(text)}"
                    f"<title>{tip}</title></text>"
                )
        else:
            parts.append(
                f"<text x='{x:.1f}' y='{H - 8}' text-anchor='middle' font-size='9.5' "
                f"font-family='{_MONO_FONT}' fill='{C['on_surface_variant']}'>{i + 1}</text>"
            )
    parts.append("</svg>")
    st.markdown("".join(parts), unsafe_allow_html=True)
    return [x / W for x, _ in pts]


def render_chart_click_overlay(entries, season, key_prefix):
    """
    Invisible per-point click targets laid directly over a chart that was
    JUST rendered by render_game_log_line (must be called immediately
    after, same script pass) - clicking anywhere in a game's column opens
    that game's box score, not just via a chip strip underneath.

    `entries` must be data.box_score.game_link_positions()'s output - one
    entry per point IN THE SAME ORDER, `None` for a point that didn't
    resolve to a real game. Positional, not the dropping game_link_rows,
    because this has to line up column-for-point with an already-drawn
    chart - dropping an unresolved row would shift every later column onto
    the wrong week.

    HOW THE OVERLAY LINES UP WITH THE CHART, with no coordinate math shared
    between Python and the SVG: render_game_log_line places point i at the
    center of an equal-width 1/n slot, and st.columns(n) below it is ALSO n
    equal-width slots - column i's center already IS point i's x position,
    by construction. No absolute positioning, no percentage-left offsets to
    get right.

    Getting the button row to sit ON TOP of the chart instead of below it
    uses a CSS negative margin-top sized as a PERCENTAGE (not a fixed
    pixel value) matching the chart's own fixed aspect ratio (H/W from its
    viewBox). Percentage margins resolve against the parent's WIDTH even
    for a top/bottom margin (real CSS behavior, not a bug) - since the
    chart is `width:100%; height:auto` off a fixed viewBox, its rendered
    pixel height is always exactly `renderedWidth * H/W`, so a percentage
    margin of that same ratio cancels it exactly AT WHATEVER WIDTH THE
    COLUMN ACTUALLY RENDERS, unlike a fixed px offset which would only be
    correct at one specific viewport width. Verified live (Playwright,
    getBoundingClientRect on both the chart and the button row) before
    shipping - this is exactly the DOM/geometry verification a past pass's
    comment on the chip-strip-only approach said wasn't available yet.
    """
    if not entries:
        return
    n = len(entries)
    wrap_key = f"{key_prefix}_overlay"
    aspect_pct = (208 / 860) * 100  # render_game_log_line's own H/W
    st.markdown(
        f"<style>"
        f".st-key-{wrap_key} {{ margin-top: -{aspect_pct:.3f}%; position: relative; z-index: 3; }}"
        f".st-key-{wrap_key} div[data-testid='stHorizontalBlock'] {{ gap: 0 !important; }}"
        f".st-key-{wrap_key} div[data-testid='stButton'] button {{"
        f"  width: 100%; min-height: 165px; background: transparent !important;"
        f"  border: none !important; box-shadow: none !important; opacity: 0;"
        f"  border-radius: 0 !important; transition: background-color 120ms ease;"
        f"}}"
        f".st-key-{wrap_key} div[data-testid='stButton'] button:hover {{"
        f"  background: {C['surface_container_high']}55 !important;"
        f"}}"
        f"</style>",
        unsafe_allow_html=True,
    )
    with st.container(key=wrap_key):
        cols = st.columns(n)
        for i, (col, entry) in enumerate(zip(cols, entries)):
            with col:
                if entry is None:
                    # Disabled, not omitted - an empty column here would
                    # shrink to content width in some Streamlit layouts and
                    # break the equal-width alignment every other column
                    # depends on.
                    st.button(
                        "​", key=f"{key_prefix}_ov_{i}", width="stretch",
                        disabled=True, help="No box score on file for this game.",
                    )
                else:
                    from ui.components import open_box_score
                    st.button(
                        "​", key=f"{key_prefix}_ov_{i}", width="stretch",
                        help=entry['help'],
                        on_click=open_box_score, args=(season, entry['game_id']),
                    )


def render_percentile_bar_list(entries, row_h=42, sort=True, sub_row_h=28):
    """
    One full-width horizontal percentile bar per row, coloured on the app's
    standard grade scale (ui.styling.get_pff_color).

    Chosen over the radar grid Player Search / Player Compare use because
    the question here is different: a radar is good at silhouette ("what
    shape of player is this") and bad at precise comparison past a handful
    of axes, and the Tendency Profile's job is precise - the gap between a
    72nd and an 80th percentile target share is the whole point, and on a
    radar those two vertices are indistinguishable.

    `entries`: list of {'label', 'value_str', 'pct' (0-100)}. Entries with
    pct=None are skipped - nothing to plot, and a bar pinned at zero would
    read as "worst in the league" rather than "not measured".

    `sort=False` keeps `entries`' own order, which is what the Tendency
    Profile needs: its sequence (volume -> efficiency -> role) is itself
    meaningful, and re-sorting by percentile scrambles it into something
    that looks arbitrary from one player to the next.

    An entry with `'sub': True` renders as a shorter, indented, smaller-font
    row directly under whatever came before it - a companion stat that
    belongs WITH the row above rather than as a peer of its own (e.g. a
    position's YPT-allowed sitting right under that position's fantasy-
    points-allowed bar in Matchup Analyzer's Positional Vulnerability list).
    `sort` only ever reorders entries within their own kind's relative
    order is undefined across a sub-row boundary, so callers that mix
    sub-rows in should always pass `sort=False` - the same convention the
    Tendency Profile already follows for its own ordering.
    """
    plotted = [e for e in entries if e.get('pct') is not None]
    if not plotted:
        return
    if sort:
        plotted = sorted(plotted, key=lambda e: e['pct'], reverse=True)
    # Font sizes here are ratios to W, not screen pixels - the <svg> scales
    # to its container, so a label sized for a full-width chart shrinks to
    # roughly 8px in a half-width column. These are sized for the narrow
    # case, which is where this chart actually lives.
    W, LABEL_W, VAL_W = 860, 250, 140
    BAR_H, LABEL_FS, VAL_FS = 22, 15, 13.5
    SUB_INDENT, SUB_SCALE = 20, 0.72
    bar_max = W - LABEL_W - VAL_W - 20
    heights = [sub_row_h if e.get('sub') else row_h for e in plotted]
    H = sum(heights) + 8
    parts = [
        f"<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' "
        f"style='width:100%; height:auto; font-family:{_BODY_FONT};'>"
    ]
    y = 4
    for e, h in zip(plotted, heights):
        is_sub = bool(e.get('sub'))
        pct = float(e['pct'])
        cy = y + h / 2
        label_x = LABEL_W - 12 - (SUB_INDENT if is_sub else 0)
        bar_x = LABEL_W + (SUB_INDENT if is_sub else 0)
        row_bar_max = bar_max - (SUB_INDENT if is_sub else 0)
        bar_h = BAR_H * (SUB_SCALE if is_sub else 1.0)
        label_fs = LABEL_FS * (SUB_SCALE if is_sub else 1.0)
        val_fs = VAL_FS * (SUB_SCALE if is_sub else 1.0)
        bar_len = max(3.0, row_bar_max * pct / 100.0)
        color = get_pff_color(pct)
        label, value_str = _esc(e['label']), _esc(e.get('value_str', '--'))
        tip = f"{label}: {value_str} — {pct:.0f}th percentile"
        if e.get('help'):
            tip += f" · {_esc(e['help'])}"
        parts.append(
            f"<text x='{label_x:.1f}' y='{cy + label_fs * 0.36:.1f}' text-anchor='end' font-size='{label_fs:.1f}' "
            f"font-weight='{500 if is_sub else 600}' fill='{C['on_surface_variant'] if is_sub else C['on_surface']}'>"
            f"{label}<title>{tip}</title></text>"
        )
        parts.append(
            f"<rect x='{bar_x}' y='{cy - bar_h / 2:.1f}' width='{row_bar_max:.1f}' height='{bar_h:.1f}' rx='4' "
            f"fill='{C['surface_container_high']}' opacity='0.5'/>"
        )
        parts.append(
            f"<rect x='{bar_x}' y='{cy - bar_h / 2:.1f}' width='{bar_len:.1f}' height='{bar_h:.1f}' rx='4' "
            f"fill='{color}' opacity='{0.85 if is_sub else 1.0}'><title>{tip}</title></rect>"
        )
        parts.append(
            f"<text x='{LABEL_W + bar_max + 14}' y='{cy + val_fs * 0.36:.1f}' font-size='{val_fs:.1f}' "
            f"font-family='{_MONO_FONT}' fill='{C['on_surface_variant'] if is_sub else C['on_surface']}'>{value_str}</text>"
        )
        y += h
    parts.append("</svg>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_tier_curve(tiers, avg=None, avg_label="season avg", highlight=None, x_ticks=None, value_fmt="{:.1f}"):
    """
    Straight-line segments through up to 4 real tier points - deliberately
    never splined. Three or four points do not justify implying a fitted
    continuous curve, and a smoothed line here would invent values between
    tiers that no game actually produced.

    `tiers`: [{'x' (0-100), 'y', 'n' (sample size), 'name' (optional)}].
    The sample size is always in the tooltip: a tier built from two games
    must not read as confidently as one built from eight, and hiding thin
    tiers entirely would silently change the shape of the curve.

    `highlight`: optional {'x', 'y', 'label'} drawn as a diamond at its own
    precise x - the specific defense being checked, placed at its real
    toughness percentile rather than snapped to the nearest tier.
    """
    pts = sorted(tiers, key=lambda t: t['x'])
    if not pts:
        return
    W, H = 860, 240
    ML, MR, MT, MB = 52, 20, 22, 32
    plot_w, plot_h = W - ML - MR, H - MT - MB

    ys = [t['y'] for t in pts] + ([avg] if avg is not None else []) + ([highlight['y']] if highlight else [])
    y_min, y_max = min(ys), max(ys)
    # A flat series (every tier identical) has zero range, which would make
    # the y-scale divide by zero; fall back to a proportional pad.
    pad = (y_max - y_min) * 0.18 or max(abs(y_max), 1.0) * 0.18
    y_min, y_max = y_min - pad, y_max + pad

    def px(x):
        return ML + plot_w * max(0.0, min(100.0, float(x))) / 100.0

    def py(y):
        return MT + plot_h * (1 - (float(y) - y_min) / (y_max - y_min))

    parts = [
        f"<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' "
        f"style='width:100%; height:auto; font-family:{_BODY_FONT};'>"
    ]
    for frac in (0.0, 0.5, 1.0):
        gy = MT + plot_h * frac
        gval = y_max - (y_max - y_min) * frac
        parts.append(
            f"<line x1='{ML}' y1='{gy:.1f}' x2='{W - MR}' y2='{gy:.1f}' stroke='{C['outline_variant']}' "
            f"stroke-width='1' stroke-dasharray='2,4' opacity='0.6'/>"
        )
        parts.append(
            f"<text x='{ML - 6}' y='{gy + 3:.1f}' text-anchor='end' font-size='10' font-family='{_MONO_FONT}' "
            f"fill='{C['on_surface_variant']}'>{value_fmt.format(gval)}</text>"
        )

    for x_pos, tick_label in (x_ticks or [(v, str(v)) for v in (0, 25, 50, 75, 100)]):
        parts.append(
            f"<text x='{px(x_pos):.1f}' y='{H - 8}' text-anchor='middle' font-size='10.5' "
            f"fill='{C['on_surface_variant']}'>{_esc(tick_label)}</text>"
        )

    if avg is not None:
        ay = py(avg)
        parts.append(
            f"<line x1='{ML}' y1='{ay:.1f}' x2='{W - MR}' y2='{ay:.1f}' stroke='{C['on_surface_variant']}' "
            f"stroke-width='1.2' stroke-dasharray='5,5' opacity='0.8'/>"
        )
        parts.append(
            f"<text x='{W - MR}' y='{ay - 6:.1f}' text-anchor='end' font-size='10' font-family='{_MONO_FONT}' "
            f"fill='{C['on_surface_variant']}'>{_esc(avg_label)} {value_fmt.format(avg)}</text>"
        )

    line_pts = [(px(t['x']), py(t['y'])) for t in pts]
    if len(line_pts) >= 2:
        area = (
            [f"{line_pts[0][0]:.1f},{MT + plot_h:.1f}"]
            + [f"{x:.1f},{y:.1f}" for x, y in line_pts]
            + [f"{line_pts[-1][0]:.1f},{MT + plot_h:.1f}"]
        )
        parts.append(f"<polygon points='{' '.join(area)}' fill='{C['primary']}' opacity='0.10'/>")
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in line_pts)
        parts.append(
            f"<polyline points='{poly}' fill='none' stroke='{C['primary']}' stroke-width='2.2' "
            f"stroke-linejoin='round'/>"
        )

    for t, (x, y) in zip(pts, line_pts):
        prefix = f"{_esc(t['name'])} — " if t.get('name') else ''
        n = int(t.get('n', 0))
        tooltip = f"{prefix}{value_fmt.format(t['y'])} (n={n} game{'s' if n != 1 else ''})"
        parts.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='5' fill='{C['primary']}' stroke='{C['surface']}' "
            f"stroke-width='1.5'><title>{tooltip}</title></circle>"
        )
        parts.append(
            f"<text x='{x:.1f}' y='{y - 11:.1f}' text-anchor='middle' font-size='10.5' font-weight='700' "
            f"font-family='{_MONO_FONT}' fill='{C['on_surface']}'>{value_fmt.format(t['y'])}</text>"
        )

    if highlight is not None:
        hx, hy = px(highlight['x']), py(highlight['y'])
        h_label = _esc(highlight.get('label', 'Projected'))
        size = 7
        diamond = f"{hx:.1f},{hy - size:.1f} {hx + size:.1f},{hy:.1f} {hx:.1f},{hy + size:.1f} {hx - size:.1f},{hy:.1f}"
        parts.append(
            f"<polygon points='{diamond}' fill='{C['tertiary']}' stroke='{C['surface']}' stroke-width='1.5'>"
            f"<title>{h_label}: {value_fmt.format(highlight['y'])}</title></polygon>"
        )
        parts.append(
            f"<text x='{hx:.1f}' y='{hy - 14:.1f}' text-anchor='middle' font-size='10.5' font-weight='700' "
            f"fill='{C['tertiary']}'>{h_label}</text>"
        )

    parts.append("</svg>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_split_bars(rows, left_label, right_label):
    """
    Paired horizontal bars sharing one track per row - built for the "vs
    man / vs zone" and "slot / wide" splits, where the comparison that
    matters is between the two halves of the SAME row, not between rows.

    `rows`: [{'label', 'left', 'right', 'left_str', 'right_str'}], where
    left/right are already on a shared 0-100 scale (percentiles, or a
    normalised share). Rows whose two sides are both None are skipped.
    """
    rows = [r for r in rows if r.get('left') is not None or r.get('right') is not None]
    if not rows:
        return
    W, LABEL_W, row_h, BAR_H = 860, 210, 40, 18
    half = (W - LABEL_W - 30) / 2
    H = row_h * len(rows) + 34
    parts = [
        f"<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' "
        f"style='width:100%; height:auto; font-family:{_BODY_FONT};'>"
    ]
    parts.append(
        f"<text x='{LABEL_W + half / 2:.1f}' y='16' text-anchor='middle' font-size='12' font-weight='700' "
        f"letter-spacing='0.08em' fill='{C['on_surface_variant']}'>{_esc(left_label).upper()}</text>"
    )
    parts.append(
        f"<text x='{LABEL_W + half + 30 + half / 2:.1f}' y='16' text-anchor='middle' font-size='12' "
        f"font-weight='700' letter-spacing='0.08em' fill='{C['on_surface_variant']}'>{_esc(right_label).upper()}</text>"
    )
    y = 26
    for r in rows:
        cy = y + row_h / 2
        parts.append(f"<g class='split-bar-row'>")
        parts.append(
            f"<text x='{LABEL_W - 12}' y='{cy + 5:.1f}' text-anchor='end' font-size='14' font-weight='600' "
            f"class='split-bar-label' fill='{C['on_surface']}'>{_esc(r['label'])}</text>"
        )
        for side, x0 in (('left', LABEL_W), ('right', LABEL_W + half + 30)):
            val = r.get(side)
            parts.append(
                f"<rect x='{x0:.1f}' y='{cy - BAR_H / 2:.1f}' width='{half:.1f}' height='{BAR_H}' rx='4' "
                f"fill='{C['surface_container_high']}' opacity='0.5'/>"
            )
            if val is None:
                continue
            length = max(3.0, half * max(0.0, min(100.0, float(val))) / 100.0)
            shown = r.get(f'{side}_str', f"{val:.0f}")
            parts.append(
                f"<rect x='{x0:.1f}' y='{cy - BAR_H / 2:.1f}' width='{length:.1f}' height='{BAR_H}' rx='4' "
                f"class='split-bar-fill' fill='{get_pff_color(val)}'><title>{_esc(r['label'])} "
                f"{_esc(left_label if side == 'left' else right_label)}: {_esc(shown)}</title></rect>"
            )
            parts.append(
                f"<text x='{x0 + half - 6:.1f}' y='{cy + 4:.1f}' text-anchor='end' font-size='12' "
                f"font-family='{_MONO_FONT}' font-weight='600' class='split-bar-value' fill='{C['on_surface']}'>{_esc(shown)}</text>"
            )
        parts.append("</g>")
        y += row_h
    parts.append("</svg>")
    st.markdown("".join(parts), unsafe_allow_html=True)
