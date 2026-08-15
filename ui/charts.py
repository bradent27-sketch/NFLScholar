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


def render_percentile_bar_list(entries, row_h=42, sort=True):
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
    bar_max = W - LABEL_W - VAL_W - 20
    H = row_h * len(plotted) + 8
    parts = [
        f"<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' "
        f"style='width:100%; height:auto; font-family:{_BODY_FONT};'>"
    ]
    y = 4
    for e in plotted:
        pct = float(e['pct'])
        cy = y + row_h / 2
        bar_len = max(3.0, bar_max * pct / 100.0)
        color = get_pff_color(pct)
        label, value_str = _esc(e['label']), _esc(e.get('value_str', '--'))
        tip = f"{label}: {value_str} — {pct:.0f}th percentile"
        if e.get('help'):
            tip += f" · {_esc(e['help'])}"
        parts.append(
            f"<text x='{LABEL_W - 12}' y='{cy + LABEL_FS * 0.36:.1f}' text-anchor='end' font-size='{LABEL_FS}' "
            f"font-weight='600' fill='{C['on_surface']}'>{label}<title>{tip}</title></text>"
        )
        parts.append(
            f"<rect x='{LABEL_W}' y='{cy - BAR_H / 2:.1f}' width='{bar_max:.1f}' height='{BAR_H}' rx='4' "
            f"fill='{C['surface_container_high']}' opacity='0.5'/>"
        )
        parts.append(
            f"<rect x='{LABEL_W}' y='{cy - BAR_H / 2:.1f}' width='{bar_len:.1f}' height='{BAR_H}' rx='4' "
            f"fill='{color}'><title>{tip}</title></rect>"
        )
        parts.append(
            f"<text x='{LABEL_W + bar_max + 14}' y='{cy + VAL_FS * 0.36:.1f}' font-size='{VAL_FS}' "
            f"font-family='{_MONO_FONT}' fill='{C['on_surface']}'>{value_str}</text>"
        )
        y += row_h
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
        parts.append(
            f"<text x='{LABEL_W - 12}' y='{cy + 5:.1f}' text-anchor='end' font-size='14' font-weight='600' "
            f"fill='{C['on_surface']}'>{_esc(r['label'])}</text>"
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
                f"fill='{get_pff_color(val)}'><title>{_esc(r['label'])} "
                f"{_esc(left_label if side == 'left' else right_label)}: {_esc(shown)}</title></rect>"
            )
            parts.append(
                f"<text x='{x0 + half - 6:.1f}' y='{cy + 4:.1f}' text-anchor='end' font-size='12' "
                f"font-family='{_MONO_FONT}' font-weight='600' fill='{C['on_surface']}'>{_esc(shown)}</text>"
            )
        y += row_h
    parts.append("</svg>")
    st.markdown("".join(parts), unsafe_allow_html=True)
