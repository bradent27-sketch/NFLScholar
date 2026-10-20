"""
Phone-friendly exports of a built weekly board (requested 2026-10-18: "export game reports and board exports to some
sort of viewable format ... have the whole weekend worth of games visible from a phone ... when I am away from the main
PC").

Two multi-page PDFs, drawn with matplotlib (already a dependency; no PDF library to install), on a narrow portrait page
that fits a phone screen:

    game_report_pdf  every game of the week in kickoff order: kickoff, venue, line / total, each side's implied points,
                     weather, and each team's key players (QB, top 2 RB, top 3 WR, top TE) with their projections -
                     the Game Slate's arrangement plus the board's context, three games a page
    board_pdf        the bare-bones board: per position, rank / player / team / opponent / projection / injury tag

Pure functions of the board frame and the week's schedule (no Streamlit here), so they are directly testable; the
Weekly Rankings tab supplies the inputs and the download / save-to-folder buttons (ui/tabs/rankings.py).
"""
import io
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from config import POSITION_COLORS

PAGE_W, PAGE_H = 4.5, 8.5          # inches: a phone-shaped portrait page
MARGIN = 0.22
HEADER_H, FOOTER_H = 0.55, 0.28
INK, MUTED, RULE, ZEBRA, PAPER = '#111827', '#6b7280', '#d1d5db', '#f3f4f6', '#ffffff'
GAMES_PER_PAGE = 3
GAME_SLOTS = (('QB', 1), ('RB', 2), ('WR', 3), ('TE', 1))
BOARD_DEPTH = {'QB': 36, 'RB': 60, 'WR': 84, 'TE': 36}
BOARD_ROW_H = 0.185
MIN_LISTED_POINTS = 1.0            # a sidelined player (0 points) is not one of a team's key players
_PRECIP_WORDS = {'dry': 'dry', 'light_rain': 'light rain', 'heavy_rain': 'heavy rain', 'snow': 'snow'}


# --- small pure helpers ------------------------------------------------------------------------------------------

def _finite(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def injury_tag(status):
    """'Questionable (95%)' -> 'Q', 'Doubtful' -> 'D', Out / reserve lists -> 'OUT', blank -> ''."""
    s = str(status or '').strip().lower()
    if not s or s in ('nan', 'none', 'unknown'):
        return ''
    if s.startswith('questionable'):
        return 'Q'
    if s.startswith('doubtful'):
        return 'D'
    if s.startswith('out') or 'reserve' in s or s.startswith('ir') or s.startswith('suspend'):
        return 'OUT'
    if s.startswith('practice squad'):
        return 'PS'
    return re.sub(r'[^A-Za-z]', '', s)[:3].upper()


def implied_totals(spread_line, total_line):
    """(away, home) implied points. nflverse's spread_line is positive when the HOME team is favored."""
    s, t = _finite(spread_line), _finite(total_line)
    if s is None or t is None:
        return None, None
    return t / 2.0 - s / 2.0, t / 2.0 + s / 2.0


def line_text(away, home, spread_line, total_line):
    parts = []
    s, t = _finite(spread_line), _finite(total_line)
    if s is not None:
        parts.append(f"{home} -{s:g}" if s > 0 else (f"{away} -{-s:g}" if s < 0 else "Pick'em"))
    if t is not None:
        parts.append(f"O/U {t:g}")
    return '  ·  '.join(parts)


def weather_text(game_weather, precip_bucket=None):
    """'Indoors', 'wind 14 mph, 41°F, light rain', or '' when there is nothing."""
    if game_weather is None:
        return ''
    if not getattr(game_weather, 'is_outdoor', True):
        return 'Indoors'
    bits = []
    wind, temp = _finite(getattr(game_weather, 'wind_mph', None)), _finite(getattr(game_weather, 'temp_f', None))
    if wind is not None:
        bits.append(f"wind {wind:.0f} mph")
    if temp is not None:
        bits.append(f"{temp:.0f}°F")
    if precip_bucket in _PRECIP_WORDS:
        bits.append(_PRECIP_WORDS[precip_bucket])
    return ', '.join(bits)


def key_players(board, team):
    """[(pos, player, points, injury tag)] - the team's QB, top 2 RB, top 3 WR and top TE by projection."""
    if board is None or board.empty:
        return []
    pts = pd.to_numeric(board['Model Proj Pts'], errors='coerce')
    t = board[(board['Team'].astype(str) == str(team)) & (pts >= MIN_LISTED_POINTS)].copy()
    t['_pts'] = pts[t.index]
    out = []
    for pos, n in GAME_SLOTS:
        for _, row in t[t['Pos'] == pos].nlargest(n, '_pts').iterrows():
            out.append((pos, str(row['Player']), float(row['_pts']), injury_tag(row.get('Injury Status'))))
    return out


def _fit(text, max_chars):
    text = str(text)
    return text if len(text) <= max_chars else text[:max_chars - 1].rstrip() + '…'


def _text_on(hex_color):
    """Black or white, whichever reads better on ``hex_color``."""
    h = str(hex_color or '').lstrip('#')
    if len(h) != 6:
        return INK
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return INK if (0.299 * r + 0.587 * g + 0.114 * b) > 0.6 else PAPER


# --- drawing -----------------------------------------------------------------------------------------------------

def _page(title, subtitle, page_no, page_count):
    fig = Figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor(PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, PAGE_H)
    ax.axis('off')
    ax.text(MARGIN, PAGE_H - MARGIN, title, fontsize=11, fontweight='bold', color=INK, va='top')
    ax.text(MARGIN, PAGE_H - MARGIN - 0.22, subtitle, fontsize=6.5, color=MUTED, va='top')
    ax.plot([MARGIN, PAGE_W - MARGIN], [PAGE_H - HEADER_H, PAGE_H - HEADER_H], color=RULE, lw=0.8)
    ax.text(PAGE_W - MARGIN, MARGIN * 0.6, f"{page_no} / {page_count}", fontsize=6.5, color=MUTED,
            ha='right', va='bottom')
    return fig, ax


def _chip(ax, x, y, w, h, color, left, right):
    color = color or '#374151'
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.06',
                                facecolor=color, edgecolor='none'))
    fg = _text_on(color)
    ax.text(x + 0.09, y + h / 2, left, fontsize=12, fontweight='bold', color=fg, va='center', ha='left')
    if right:
        ax.text(x + w - 0.09, y + h / 2, right, fontsize=11, fontweight='bold', color=fg, va='center', ha='right')


def _game_block(ax, top, game, board, weather_by_team, precip_by_team, bridge):
    """One game, drawn downward from ``top``; returns the y where it ends."""
    width = PAGE_W - 2 * MARGIN
    away, home = str(game.get('Away')), str(game.get('Home'))
    away_b, home_b = bridge.get(away, away), bridge.get(home, home)
    played = bool(game.get('Played')) or bool(game.get('Live'))

    meta = [str(game.get('Kickoff') or game.get('Kickoff Long') or ''), str(game.get('Date Display') or '')]
    venue = game.get('Venue')
    if venue and not pd.isna(venue):
        meta.append(f"{venue}{' (neutral)' if game.get('Neutral Site') else ''}")
    ax.text(MARGIN, top, _fit('  ·  '.join(m for m in meta if m), 70), fontsize=6.8, color=MUTED, va='top')

    chip_y, chip_h = top - 0.20 - 0.36, 0.36
    chip_w = (width - 0.30) / 2.0
    a_imp, h_imp = implied_totals(game.get('Spread Line'), game.get('Total Line'))
    if played:
        a_right = '' if pd.isna(game.get('Away Pts')) else f"{int(game.get('Away Pts'))}"
        h_right = '' if pd.isna(game.get('Home Pts')) else f"{int(game.get('Home Pts'))}"
    else:
        a_right = f"{a_imp:.1f}" if a_imp is not None else ''
        h_right = f"{h_imp:.1f}" if h_imp is not None else ''
    _chip(ax, MARGIN, chip_y, chip_w, chip_h, game.get('Away Color'), away, a_right)
    ax.text(MARGIN + chip_w + 0.15, chip_y + chip_h / 2, '@', fontsize=10, color=MUTED, ha='center', va='center')
    _chip(ax, MARGIN + chip_w + 0.30, chip_y, chip_w, chip_h, game.get('Home Color'), home, h_right)

    info = []
    if played:
        info.append(str(game.get('Status Detail') or 'Final'))
    else:
        lt = line_text(away, home, game.get('Spread Line'), game.get('Total Line'))
        if lt:
            info.append(lt + ('  ·  bars show implied points' if a_imp is not None else ''))
    wx = weather_text(weather_by_team.get(home_b) or weather_by_team.get(away_b),
                      precip_by_team.get(home_b) or precip_by_team.get(away_b))
    if wx:
        info.append(wx)
    y = chip_y - 0.07
    for line in info:
        ax.text(MARGIN, y, _fit(line, 74), fontsize=6.8, color=INK, va='top')
        y -= 0.16

    y -= 0.04
    row_h = 0.185
    columns = ((MARGIN, away_b), (MARGIN + chip_w + 0.30, home_b))
    deepest = y
    for x0, team in columns:
        yy = y
        players = key_players(board, team)
        if not players:
            ax.text(x0, yy, 'no projections', fontsize=6.8, color=MUTED, va='top')
            yy -= row_h
        for pos, name, pts, tag in players:
            ax.text(x0, yy, pos, fontsize=6.2, fontweight='bold', color=POSITION_COLORS.get(pos, MUTED), va='top')
            ax.text(x0 + 0.24, yy, _fit(name, 19), fontsize=7.2, color=INK, va='top')
            ax.text(x0 + chip_w - 0.02, yy, f"{pts:.1f}", fontsize=7.2, fontweight='bold', color=INK,
                    va='top', ha='right')
            if tag:
                ax.text(x0 + chip_w - 0.40, yy, tag, fontsize=5.8, fontweight='bold', color='#b45309',
                        va='top', ha='right')
            yy -= row_h
        deepest = min(deepest, yy)
    end = deepest - 0.06
    ax.plot([MARGIN, PAGE_W - MARGIN], [end, end], color=RULE, lw=0.6)
    return end - 0.14


def _render_pdf(figures, title):
    buf = io.BytesIO()
    with PdfPages(buf, metadata={'Title': title, 'Creator': 'NFL Scholar'}) as pdf:
        for fig in figures:
            pdf.savefig(fig, facecolor=fig.get_facecolor())
    return buf.getvalue()


def game_report_figures(games, board, title, subtitle, weather_by_team=None, precip_by_team=None, bridge=None):
    games = games if games is not None else pd.DataFrame()
    weather_by_team, precip_by_team, bridge = weather_by_team or {}, precip_by_team or {}, bridge or {}
    n_pages = max(1, int(np.ceil(len(games) / GAMES_PER_PAGE)))
    figures = []
    for p in range(n_pages):
        fig, ax = _page(title, subtitle, p + 1, n_pages)
        top = PAGE_H - HEADER_H - 0.14
        chunk = games.iloc[p * GAMES_PER_PAGE:(p + 1) * GAMES_PER_PAGE]
        if chunk.empty:
            ax.text(MARGIN, top, 'No games on the schedule for this week.', fontsize=8, color=MUTED, va='top')
        for _, game in chunk.iterrows():
            top = _game_block(ax, top, game, board, weather_by_team, precip_by_team, bridge)
        figures.append(fig)
    return figures


def game_report_pdf(games, board, title, subtitle, weather_by_team=None, precip_by_team=None, bridge=None):
    """PDF bytes: every game of the week, three to a page (see the module docstring)."""
    return _render_pdf(game_report_figures(games, board, title, subtitle, weather_by_team, precip_by_team, bridge),
                       title)


def _opponent_labels(games, bridge):
    """{board team: 'ATL' for home, '@ATL' for away} from the week's schedule."""
    out = {}
    if games is None or games.empty:
        return out
    bridge = bridge or {}
    for away, home in zip(games['Away'].astype(str), games['Home'].astype(str)):
        a, h = bridge.get(away, away), bridge.get(home, home)
        out[a] = f"@{h}"
        out[h] = a
    return out


def board_tables(board, depth=None):
    """{pos: frame of Rank, Player, Team, Opponent, Proj, Inj} - the top ``depth[pos]`` by projection."""
    depth = depth or BOARD_DEPTH
    pts = pd.to_numeric(board['Model Proj Pts'], errors='coerce')
    b = board.assign(_pts=pts)
    out = {}
    for pos, n in depth.items():
        sub = b[(b['Pos'] == pos) & (b['_pts'] > 0)].nlargest(n, '_pts')
        if sub.empty:
            continue
        out[pos] = pd.DataFrame({
            'Rank': np.arange(1, len(sub) + 1),
            'Player': sub['Player'].astype(str).to_numpy(),
            'Team': sub['Team'].astype(str).to_numpy(),
            'Opponent': sub['Opponent'].astype(str).to_numpy() if 'Opponent' in sub else '',
            'Proj': sub['_pts'].round(1).to_numpy(),
            'Inj': [injury_tag(s) for s in sub.get('Injury Status', pd.Series('', index=sub.index))],
        })
    return out


def board_figures(board, title, subtitle, games=None, bridge=None, depth=None):
    tables = board_tables(board, depth)
    opp = _opponent_labels(games, bridge)
    usable = PAGE_H - HEADER_H - FOOTER_H - 0.55
    per_page = int(usable // BOARD_ROW_H)
    plan = []
    for pos, table in tables.items():
        for start in range(0, len(table), per_page):
            plan.append((pos, table.iloc[start:start + per_page]))
    plan = plan or [(None, pd.DataFrame())]
    figures = []
    cols = (('#', MARGIN, 'left'), ('Player', MARGIN + 0.32, 'left'), ('Team', MARGIN + 2.30, 'left'),
            ('Opp', MARGIN + 2.75, 'left'), ('Proj', PAGE_W - MARGIN - 0.36, 'right'),
            ('', PAGE_W - MARGIN, 'right'))
    for i, (pos, chunk) in enumerate(plan):
        fig, ax = _page(title, subtitle, i + 1, len(plan))
        y = PAGE_H - HEADER_H - 0.12
        if pos is None:
            ax.text(MARGIN, y, 'No projections on this board.', fontsize=8, color=MUTED, va='top')
            figures.append(fig)
            continue
        color = POSITION_COLORS.get(pos, '#374151')
        ax.add_patch(FancyBboxPatch((MARGIN, y - 0.26), PAGE_W - 2 * MARGIN, 0.26,
                                    boxstyle='round,pad=0,rounding_size=0.05', facecolor=color, edgecolor='none'))
        first, last = int(chunk['Rank'].iloc[0]), int(chunk['Rank'].iloc[-1])
        ax.text(MARGIN + 0.08, y - 0.13, f"{pos}  {first}-{last}", fontsize=9.5, fontweight='bold',
                color=_text_on(color), va='center')
        y -= 0.36
        for label, x, ha in cols:
            ax.text(x, y, label, fontsize=6.3, fontweight='bold', color=MUTED, va='top', ha=ha)
        y -= 0.20
        for k, (_, r) in enumerate(chunk.iterrows()):
            if k % 2 == 0:
                ax.add_patch(FancyBboxPatch((MARGIN - 0.04, y - BOARD_ROW_H + 0.035), PAGE_W - 2 * MARGIN + 0.08,
                                            BOARD_ROW_H, boxstyle='square,pad=0', facecolor=ZEBRA, edgecolor='none'))
            opp_label = opp.get(str(r['Team']), str(r['Opponent']))
            cells = (str(int(r['Rank'])), _fit(r['Player'], 26), str(r['Team']), opp_label, f"{r['Proj']:.1f}",
                     r['Inj'])
            for (label, x, ha), value in zip(cols, cells):
                bold = label in ('Proj',)
                color_v = '#b45309' if label == '' else INK
                ax.text(x, y, value, fontsize=7.2 if label not in ('', '#') else 6.4,
                        fontweight='bold' if bold or label == '' else 'normal', color=color_v, va='top', ha=ha)
            y -= BOARD_ROW_H
        figures.append(fig)
    return figures


def board_pdf(board, title, subtitle, games=None, bridge=None, depth=None):
    """PDF bytes: the bare-bones board, one position block per page run (see the module docstring)."""
    return _render_pdf(board_figures(board, title, subtitle, games, bridge, depth), title)


# --- files -------------------------------------------------------------------------------------------------------

def default_export_dir():
    """The personal OneDrive (synced to the phone's OneDrive app) when there is one, else ./exports."""
    onedrive = os.environ.get('OneDriveConsumer') or os.environ.get('OneDrive')
    candidates = [Path(onedrive)] if onedrive else []
    candidates.append(Path.home() / 'OneDrive')
    for base in candidates:
        if base.is_dir() and 'university' not in base.name.lower():
            return str(base / 'NFL Scholar')
    return str(Path('exports').resolve())


def export_file_names(year, week, scoring):
    tag = {'Full PPR': 'ppr', 'Half-PPR': 'half', 'Standard': 'std'}.get(scoring, 'ppr')
    stem = f"NFL_{int(year)}_wk{int(week):02d}_{tag}"
    return {'games': f"{stem}_games.pdf", 'board': f"{stem}_board.pdf"}


def save_exports(files, folder, names):
    """Write each ``files[k]`` bytes to ``folder/names[k]`` (creating the folder); returns the paths written."""
    os.makedirs(folder, exist_ok=True)
    written = []
    for key, data in files.items():
        path = os.path.join(folder, names[key])
        with open(path, 'wb') as fh:
            fh.write(data)
        written.append(path)
    return written
