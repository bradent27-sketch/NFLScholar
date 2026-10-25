"""The week, exported for a phone: a port of CFB Scholar's ui/export_view.py (2026-10-25, user: "Board/Games export needs to be more
similar to the CFB version ... projected lines for each stat for each team ... model after the cfb version entirely if possible").
It replaces the matplotlib PDFs of 2026-10-18 (ui/report_export.py, key players' points only).

Two reports, each one self-contained HTML document (logos embedded) and, where a Chromium browser is installed, a PDF printed from it:

    games_report    the cover (when the books' lines were pulled, what is inside), the edges page (the model's median against each
                    line, biggest gap first), the slate on one page (Thursday, Sunday by window - morning, early, late, night - then
                    Monday; each row the two teams in their colours with each side's spread and the over/under, linking to its game),
                    then one block per game: the matchup card (model against market, weather), the AWAY team (banner with its projected
                    totals, availability, each position's players with a role and their full stat line), then the HOME team, then the
                    game's priced props.
    board_report    the bare board: per position, every player with a role, his game under his name, matchup rating, points and stat line.

LAYOUT RULES (CFB's, from its first export, which tore boxes across pages and clipped them on the right):
    * the page is 6.4 x 13.5 in, a phone's proportions, so a game fits on a screen or two without being shrunk;
    * a game starts a page, its home team starts a page, and no card, banner or table is split across two pages (`break-inside: avoid`);
    * nothing is wider than the page: every export block is built here, with its own stylesheet.

Differences from CFB's, each because the board or the week is different: the slate is one page (an NFL week is 13-16 games, not 60) with
Sunday's windows in place of Saturday's slates; the player and board tables carry PTS (this board ranks fantasy points) and the board is
ordered by it; the edges page has no RECORD column (no lean record is kept yet) and no line movement (lines are not archived between pulls)
- each column appears once its data exists. Nothing here calls Streamlit: ui/tabs/rankings_games.py gathers the data with the functions
the GAMES view uses.
"""
import base64
import concurrent.futures
import hashlib
import html
import io
import os
import shutil
import subprocess
import tempfile
import urllib.request

import numpy as np
import pandas as pd

from config import THEME
from ui import games_view
from ui.styling import get_matchup_color

C = THEME['colors']
F = THEME['fonts']

PAGE_WIDTH_IN = 6.4               # a phone's proportions (about 0.47): a game fills a screen or two, text readable without zooming
PAGE_HEIGHT_IN = 13.5
PDF_TIMEOUT_S = 240
# where a Chromium browser usually lives (Windows, macOS, Linux); the first that exists prints the PDF
BROWSER_PATHS = (
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
)
BROWSER_NAMES = ('google-chrome', 'chrome', 'chromium', 'chromium-browser', 'msedge', 'microsoft-edge')
# the props table's columns in the export (TEAM left out: the game is named above it)
PROP_COLUMNS = ('PLAYER', 'MARKET', 'MODEL', 'LINE', 'DIFF %', 'LEAN', 'MOVE')
EASTERN = 'America/New_York'
# Sunday's windows, by Eastern kickoff: (key, name, window text, the minute of the day it ends, exclusive)
SLATES = (('morning', 'Morning', 'international, before noon', 12 * 60),
          ('early', 'Early', '1 PM', 15 * 60),
          ('late', 'Late', '4:05 – 4:25 PM', 19 * 60),
          ('night', 'Sunday Night', '8:20 PM', 24 * 60))
WEEKDAYS = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
# a game on another day: its group's name ('Thursday Night', 'Monday Night' for a night kickoff)
NIGHT_FROM_MINUTE = 19 * 60
SUNDAY = 6


def esc(value):
    return html.escape(str(value), quote=True)


def _num(value):
    out = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
    return float(out) if pd.notna(out) and np.isfinite(out) else np.nan


def _fmt(value, pattern='{:.1f}'):
    value = _num(value)
    return '' if not np.isfinite(value) else pattern.format(value)


def team_color(colors, team):
    """The team's own colour as '#rrggbb', or a neutral when the map has none."""
    value = (colors or {}).get(team)
    return value if value and str(value).startswith('#') and len(str(value)) == 7 else '#3a4670'


def ink(color):
    """Text colour for a team-coloured background: near-black on a light colour, white on a dark one."""
    try:
        r, g, b = (int(str(color)[i:i + 2], 16) for i in (1, 3, 5))
    except ValueError:
        return '#ffffff'
    return '#0b1020' if (0.299 * r + 0.587 * g + 0.114 * b) / 255.0 > 0.62 else '#ffffff'


# ---------------------------------------------------------------------------
# kickoffs: the day, the window, the label
# ---------------------------------------------------------------------------

def eastern(start):
    """The kickoff as an Eastern Timestamp, or None when there is none."""
    if start is None or pd.isna(start):
        return None
    try:
        stamp = pd.Timestamp(start)
        return stamp.tz_localize(EASTERN) if stamp.tzinfo is None else stamp.tz_convert(EASTERN)
    except Exception:
        return None


def is_tba(local):
    """A kickoff whose time is not set sits at midnight Eastern on its day."""
    return local is not None and local.hour == 0 and local.minute == 0


def time_label(local):
    """'1:00 PM' for an Eastern kickoff, 'TBA' when there is none or it is not set."""
    if local is None or is_tba(local):
        return 'TBA'
    return local.strftime('%I:%M %p').lstrip('0')


def slate_of(local):
    """Sunday's window: 'morning' / 'early' / 'late' / 'night' by the Eastern kickoff, 'tba' when the time is not set."""
    if local is None or is_tba(local):
        return 'tba'
    minute = local.hour * 60 + local.minute
    for key, _name, _window, end in SLATES:
        if minute < end:
            return key
    return 'night'


def day_group_name(local):
    """The group a game off Sunday sits in: 'THURSDAY NIGHT', 'MONDAY NIGHT', 'SATURDAY', ..."""
    name = WEEKDAYS[local.weekday()].upper()
    return name + ' NIGHT' if not is_tba(local) and local.hour * 60 + local.minute >= NIGHT_FROM_MINUTE else name


# ---------------------------------------------------------------------------
# logos: fetched once, shrunk, embedded (a PDF would otherwise carry a full-size copy of a logo for every place it appears)
# ---------------------------------------------------------------------------

LOGO_PIXELS = 72
LOGO_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'exports', '.logos')


def _fetch(url, timeout=12):
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 NFLScholar'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _shrink(raw):
    from PIL import Image
    image = Image.open(io.BytesIO(raw)).convert('RGBA')
    image.thumbnail((LOGO_PIXELS, LOGO_PIXELS))
    out = io.BytesIO()
    image.save(out, format='PNG', optimize=True)
    return out.getvalue()


def _one_logo(url, cache_dir, fetch):
    path = os.path.join(cache_dir, hashlib.md5(url.encode('utf-8')).hexdigest() + '.png')
    try:
        if os.path.exists(path):
            with open(path, 'rb') as handle:
                data = handle.read()
        else:
            data = _shrink(fetch(url))
            os.makedirs(cache_dir, exist_ok=True)
            with open(path, 'wb') as handle:
                handle.write(data)
        return 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')
    except Exception:
        return url                                   # the original address: the browser may still fetch it


def embed_logos(logos, teams, cache_dir=LOGO_CACHE, fetch=_fetch):
    """{team: logo} for `teams`, each URL replaced by a small embedded PNG (cached on disk); one that cannot be fetched keeps its URL."""
    wanted = {t: logos[t] for t in set(teams) if (logos or {}).get(t)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        done = dict(zip(wanted, pool.map(lambda url: _one_logo(url, cache_dir, fetch), wanted.values())))
    return done


def _logo(logos, team, cls='e-logo'):
    url = (logos or {}).get(team)
    return f'<img class="{cls}" src="{esc(url)}" alt="">' if url else ''


# ---------------------------------------------------------------------------
# the slate page
# ---------------------------------------------------------------------------

LIST_ANCHOR = 'list'


def game_anchor(game):
    return f'g{esc(game["game_id"])}'


def line_numbers(line):
    """(home team's spread in a book's sign, it is the model's, the total, it is the model's): the market's number where a book has posted
    one, the model's where not, NaN where neither."""
    line = line or {}
    out = []
    for market, model in (('market_spread', 'model_spread'), ('market_total', 'model_total')):
        value = _num(line.get(market))
        from_model = False
        if not np.isfinite(value):
            value, from_model = _num(line.get(model)), True
        out += [value, from_model and np.isfinite(value)]
    return tuple(out)


def _spread_text(value):
    if not np.isfinite(value):
        return '–'
    return 'PK' if abs(value) < 0.05 else f'{value:+.1f}'


def _team_name(game, team):
    """The name a slate row shows: the team's full name when the game carries one, else its abbreviation."""
    if team == game.get('away'):
        return game.get('away_name') or team
    if team == game.get('home'):
        return game.get('home_name') or team
    return team


def _row_html(entry, colors):
    game, local = entry['game'], eastern(entry['game'].get('start'))
    home_spread, spread_model, total, total_model = line_numbers(entry.get('line'))
    cells = []
    for team, spread in ((game['away'], -home_spread), (game['home'], home_spread)):
        color = team_color(colors, team)
        cls = ' class="e-m"' if spread_model else ''
        cells.append(f'<span class="e-tm" style="background:{color};color:{ink(color)}"><b>{esc(_team_name(game, team))}</b>'
                     f'<i{cls}>{_spread_text(spread)}</i></span>')
    total_cls = ' class="e-m"' if total_model else ''
    ou = f'<span{total_cls}>{total:.1f}</span>' if np.isfinite(total) else '–'
    return (f'<a class="e-row" href="#{game_anchor(game)}"><span class="e-time">{esc(time_label(local))}</span>{cells[0]}'
            f'<span class="e-vs">{"vs" if game.get("neutral") else "@"}</span>{cells[1]}<span class="e-ou">{ou}</span></a>')


def _group_html(title, note, entries, colors):
    rows = ''.join(_row_html(e, colors) for e in entries)
    plural = 'game' if len(entries) == 1 else 'games'
    return (f'<div class="e-group"><div class="e-gh"><b>{esc(title)}</b><span>{esc(note)}</span><em>{len(entries)} {plural}</em></div>'
            f'{rows}</div>')


def _sort(entries):
    return sorted(entries, key=lambda e: (eastern(e['game'].get('start')) is None, eastern(e['game'].get('start')) or pd.Timestamp(0, tz=EASTERN),
                                          str(e['game'].get('home'))))


def _day_title(local):
    return WEEKDAYS[local.weekday()].upper() + f' · {local.strftime("%b")} {local.day}'


def schedule_html(entries, colors, intro=''):
    """The week's games on one page (id `list`): each day in kickoff order under its own heading, Sunday by window (morning, early, late,
    night, plus "time TBA"), any other day as one group (THURSDAY NIGHT, MONDAY NIGHT, SATURDAY). `entries`: [{'game': {game_id, home,
    away, home_name, away_name, neutral, start}, 'line': games_view.game_scoreline}]."""
    if not entries:
        return ''
    days, undated = {}, []
    for entry in _sort(entries):
        local = eastern(entry['game'].get('start'))
        if local is None:
            undated.append(entry)
        else:
            days.setdefault(local.normalize(), []).append(entry)
    any_model = any(line_numbers(e.get('line'))[1] or line_numbers(e.get('line'))[3] for e in entries)
    legend = ('<div class="e-legend">A spread sits beside each team (a minus sign = favoured).'
              + (" Italic numbers are the model's where no book has posted a line yet." if any_model else '')
              + ' Tap a game to jump to it.</div>')
    head = ('<div class="e-lh"><span>TIME ET</span><span>AWAY · spread</span><span></span><span>HOME · spread</span><span>O/U</span></div>')
    body = ''
    for day, group in sorted(days.items()):
        local = eastern(group[0]['game']['start'])
        body += f'<div class="e-dh">{esc(_day_title(local))}<span>{len(group)} {"game" if len(group) == 1 else "games"}</span></div>'
        if local.weekday() == SUNDAY:
            by = {}
            for entry in group:
                by.setdefault(slate_of(eastern(entry['game']['start'])), []).append(entry)
            for key, name, window, _end in SLATES:
                if by.get(key):
                    body += _group_html(name.upper(), window, by[key], colors)
            if by.get('tba'):
                body += _group_html('TIME TBA', 'kickoff not set yet', by['tba'], colors)
        else:
            by = {}
            for entry in group:
                by.setdefault(day_group_name(eastern(entry['game']['start'])), []).append(entry)
            for name, part in by.items():
                body += _group_html(name, '', part, colors)
    if undated:
        body += _group_html('DATE TBA', '', undated, colors)
    roomy = ' e-roomy' if len(entries) <= 18 else ''
    return (f'<section class="e-list{roomy}" id="{LIST_ANCHOR}">{intro}<div class="e-ph">THE SLATE<span>{len(entries)} games</span></div>'
            f'{head}{body}{legend}</section>')


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def player_table_html(frame):
    """A games_view.position_table frame as an HTML table, coloured as the GAMES grid colours it (MATCHUP on the board's red-to-green
    scale, HEALTH by its tone, PTS bold, LINE in the accent, a player who is out dimmed); the POS chip is left out (the table is one
    position)."""
    if frame is None or frame.empty:
        return ''
    columns = [c for c in frame.columns if not c.startswith('_') and c != 'POS']
    widths = {'PLAYER': 'x-l x-cp', 'MATCHUP': 'x-cm', 'HEALTH': 'x-l x-ch', 'PTS': 'x-cpt', 'LINE': 'x-cl'}
    head = ''.join(f'<th class="{widths.get(c, "")}">{esc(c)}</th>' for c in columns)
    rows = []
    for rec in frame.to_dict('records'):
        out = bool(rec.get('_out'))
        cells = []
        for column in columns:
            value = rec.get(column)
            if column == 'PLAYER':
                cells.append(f'<td class="x-l x-p">{esc(value)}</td>')
            elif column == 'HEALTH':
                fg, bg = games_view.TONES.get(rec.get('_tone'), (C['on_surface'], 'transparent'))
                cells.append(f'<td class="x-l x-h" style="color:{fg};background:{bg}">{esc(value)}</td>' if value else '<td></td>')
            elif column == 'MATCHUP':
                rating = _num(value)
                cells.append(f'<td class="x-m" style="background:{get_matchup_color(rating)}">{rating:.0f}</td>' if np.isfinite(rating)
                             else '<td></td>')
            elif column == 'PTS':
                cells.append(f'<td class="x-pts">{_fmt(value)}</td>')
            elif column == 'LINE':
                cells.append(f'<td class="x-line">{_fmt(value)}</td>')
            else:
                cells.append(f'<td>{_fmt(value)}</td>')
        rows.append(f'<tr{" class=x-out" if out else ""}>{"".join(cells)}</tr>')
    return f'<table class="x-t"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def props_table_html(frame):
    """A games_view.props_table frame as an HTML table: DIFF % and LEAN shaded green (model over the line) or red (under) by size."""
    if frame is None or frame.empty:
        return ''
    columns = [c for c in PROP_COLUMNS if c in frame.columns]
    head = ''.join(f'<th class="{"x-l" if c in ("PLAYER", "MARKET") else ""}">{esc(c)}</th>' for c in columns)
    rows = []
    for rec in frame.to_dict('records'):
        diff = _num(rec.get('DIFF %'))
        side = C['positive'] if np.isfinite(diff) and diff > 0 else C['negative']
        shade = games_view.blend(C['surface_container'], side, min(abs(diff) / 0.25, 1.0) * 0.55) if np.isfinite(diff) else 'transparent'
        cells = []
        for column in columns:
            value = rec.get(column)
            if column in ('PLAYER', 'MARKET'):
                cells.append(f'<td class="x-l{" x-p" if column == "PLAYER" else ""}">{esc(value if pd.notna(value) else "")}</td>')
            elif column == 'DIFF %':
                cells.append(f'<td class="x-s" style="background:{shade}">{_fmt(value, "{:+.0%}")}</td>')
            elif column == 'LEAN':
                cells.append(f'<td class="x-s" style="background:{shade}">{esc(value or "")}</td>')
            elif column == 'LINE':
                cells.append(f'<td class="x-line">{_fmt(value)}</td>')
            elif column == 'MOVE':
                cells.append(f'<td class="x-mvc">{move_html(value)}</td>')
            else:
                cells.append(f'<td>{_fmt(value)}</td>')
        rows.append(f'<tr>{"".join(cells)}</tr>')
    return f'<table class="x-t"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


# ---------------------------------------------------------------------------
# one game: the matchup card, the away team, the home team, the props
# ---------------------------------------------------------------------------

def _scoreboard(rows, colors):
    body = []
    for i, r in enumerate(rows):
        team = r.get('team')
        label = esc(r['label'])
        tint = ''
        if team:
            color = team_color(colors, team)
            label = f'<span class="e-sw" style="background:{color}"></span>{label}'
            if r.get('fav'):
                label += f'<span class="e-fav" style="background:{color};color:{ink(color)}">PROJECTED WIN</span>'
                tint = f' style="background:{color}30"'
        delta = f'<td class="e-d{" e-warn" if r.get("tone") else ""}">{esc(str(r["delta"]).replace("-0.0", "0.0").replace("+0.0", "0.0"))}</td>'
        body.append(f'<tr{" class=e-sep" if i == 2 else ""}{tint}><td class="e-l">{label}</td><td>{esc(r["model"])}</td>'
                    f'<td>{esc(r["market"])}</td>{delta}</tr>')
    return ('<table class="e-mm"><thead><tr><th></th><th>MODEL</th><th>MARKET</th><th>DIFF</th></tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def _weather(view):
    if view['state'] != 'data':
        return f'<div class="e-wxmsg">{esc(view["message"])}</div>'
    cells = []
    for cell in view['cells']:
        chip = ''
        if cell['label']:
            fg, bg = games_view.TONES.get(cell['tone'], games_view.TONES['plain'])
            chip = f'<span class="e-chip" style="color:{fg};background:{bg}">{esc(cell["label"])}</span>'
        cells.append(f'<div class="e-wx"><div class="e-k">{cell["k"]}</div><div class="e-wv">{esc(cell["v"])}<small>{esc(cell["unit"])}</small></div>'
                     f'<div class="e-ws">{esc(cell["sub"]) or "&nbsp;"}</div>{chip}</div>')
    foot = f'<div class="e-wfoot">{esc(view["foot"])}</div>' if view['foot'] else ''
    return f'<div class="e-wxrow">{"".join(cells)}</div>{foot}'


def card_html(entry, colors, logos):
    """The matchup card: the two teams on a bar in their colours, the model-against-market table, the weather."""
    game = entry['game']
    away, home = game['away'], game['home']
    bar = f'linear-gradient(90deg,{team_color(colors, away)} 0 50%,{team_color(colors, home)} 50% 100%)'
    sides = []
    for team, tag in ((away, 'AWAY'), (home, 'NEUTRAL' if game.get('neutral') else 'HOME')):
        sides.append(f'<div class="e-side">{_logo(logos, team)}<div><div class="e-nm">{esc(_team_name(game, team))}</div>'
                     f'<div class="e-tag">{tag}</div></div></div>')
    venue = f'<div class="e-venue">{esc(game["venue"])}</div>' if game.get('venue') else ''
    return (f'<div class="e-card"><div class="e-bar" style="background:{bar}"></div>'
            f'<div class="e-vsrow">{sides[0]}<div class="e-at">{"vs" if game.get("neutral") else "@"}</div>{sides[1]}</div>{venue}'
            f'<div class="e-k e-sec">MODEL VS MARKET</div>{_scoreboard(entry["rows"], colors)}'
            f'<div class="e-k e-sec">WEATHER</div>{_weather(entry["weather"])}</div>')


def _status(items):
    out = sum(1 for i in items if i['tone'] in ('out', 'assumed'))
    half = sum(1 for i in items if i['tone'] == 'half')
    flagged = sum(1 for i in items if i['tone'] in ('warn', 'note'))
    parts = [f'{out} out'] * bool(out) + [f'{half} limited'] * bool(half) + [f'{flagged} flagged'] * bool(flagged)
    fg, bg = games_view.TONES['info' if not parts else ('out' if out else 'warn')]
    return f'<span class="e-status" style="color:{fg};background:{bg}">{esc(" · ".join(parts).upper() or "HEALTHY")}</span>'


def _item_html(item):
    fg, bg = games_view.TONES.get(item['tone'], games_view.TONES['plain'])
    also = f' · also: {", ".join(item["also"])}' if item['also'] else ''
    return (f'<div class="e-it"><span class="e-b" style="color:{fg};background:{bg}">{esc(item["label"])}</span>'
            f'<b>{esc(item["player"])}</b><i>{esc(item["position"])}</i>'
            f'<div class="e-d2">{esc(item["source"])} · {esc(item["detail"])}{esc(also)}</div></div>')


def team_html(block, colors, logos, view='Average', second=False):
    """One team: a banner in its colours (name, side, status), its projected totals, its availability, then each position's players with a role.
    The banner, totals and availability are one unit (never split across pages), each position's table is one, and `second` starts a page."""
    team = block['team']
    color = team_color(colors, team)
    fg = ink(color)
    surface = C['surface_container']
    banner = f'linear-gradient(90deg,{color} 0%,{games_view.blend(surface, color, 0.72)} 40%,{games_view.blend(surface, color, 0.3)} 75%,{surface} 100%)'
    chips = ''.join(f'<div class="e-ch"><b>{esc(label)}</b><span>{"–" if not np.isfinite(_num(value)) else f"{float(value):.0f}"}</span></div>'
                    for label, value in block['chips'])
    items = block['items']
    if items:
        rows = ''.join(_item_html(i) for i in items)
        if block['hidden']:
            rows += f'<div class="e-ok">+ {block["hidden"]} more with little role out (their rows say so below)</div>'
    else:
        rows = ('<div class="e-ok">No one flagged: nothing in the injury reports or your rulings says a player here is out or in doubt'
                + (f' ({block["hidden"]} reserves with little role are out).' if block['hidden'] else '.') + '</div>')
    head = (f'<div class="e-head"><div class="e-th" style="background:{banner};color:{fg}">{_logo(logos, team)}<div class="e-thn">'
            f'<div class="e-tn">{esc(block.get("name") or team)}</div><div class="e-ts">{esc(block["side"])}</div></div>{_status(items)}</div>'
            f'<div class="e-chips">{chips}</div><div class="e-av"><div class="e-k">AVAILABILITY</div>{rows}</div></div>')
    tables = ''.join(f'<div class="e-tbl"><div class="e-tt"><b>{esc(label)}</b><span>{len(frame)} · '
                     f'{"averages" if view == "Average" else "medians"}</span></div>{player_table_html(frame)}</div>'
                     for label, frame in block.get('tables', ()) if frame is not None and not frame.empty)
    if not tables:
        tables = '<div class="e-none">No players with a role at these positions.</div>'
    return f'<div class="e-team{" e-second" if second else ""}" style="--tc:{color}">{head}{tables}</div>'


def _kick_line(game, week=''):
    local = eastern(game.get('start'))
    when = (f'{local.strftime("%a").upper()} {time_label(local)}' + ('' if is_tba(local) else ' ET')) if local is not None else 'DATE TBA'
    name = ''
    if local is not None and local.weekday() == SUNDAY:
        slate = slate_of(local)
        name = {k: n for k, n, _w, _e in SLATES}.get(slate, 'Time TBA' if slate == 'tba' else '')
    return ' · '.join(part for part in (f'WEEK {week}' if week else '', when, name.upper()) if part)


def game_html(entry, colors, logos, view='Average', week=''):
    """One game's block, starting a page: a kickoff line with a link back to the slate, the matchup card, the away team, the home team (a
    page of its own), and the priced props."""
    game = entry['game']
    teams = entry.get('teams', [])
    blocks = ''.join(team_html(b, colors, logos, view, second=i > 0) for i, b in enumerate(teams))
    props = entry.get('props')
    props_html = ''
    if props is not None and not props.empty:
        props_html = ('<div class="e-props"><div class="e-tt"><b>Priced props</b><span>the median against the line, largest gap first</span></div>'
                      f'{props_table_html(props)}</div>')
    return (f'<section class="e-game" id="{game_anchor(game)}"><div class="e-kick"><span>{esc(_kick_line(game, week))}</span>'
            f'<a href="#{LIST_ANCHOR}">↑ THE SLATE</a></div>{card_html(entry, colors, logos)}{blocks}{props_html}</section>')


def games_report(title, subtitle, entries, colors, logos, view='Average', week='', freshness=None, priced=0, on_board=0,
                 edge_rows=None, selection='Full slate'):
    """The whole games report as one HTML document: the cover (how fresh the lines are, what is inside), the edges page, the slate page,
    then every game in kickoff order."""
    ordered = _sort(entries)
    rows = edge_rows or []
    props = sum(len(e['props']) for e in ordered if e.get('props') is not None)
    shown = sum(1 for r in rows if np.isfinite(_num(r.get('diff_pct'))) and abs(r['diff_pct']) >= EDGE_FLOOR)
    body = (cover_html(title, subtitle, freshness, priced, on_board, props, selection, len(ordered), shown, view)
            + edges_html(rows, colors) + schedule_html(ordered, colors, '')
            + ''.join(game_html(e, colors, logos, view, week) for e in ordered))
    return document(title, body)


# ---------------------------------------------------------------------------
# the board report
# ---------------------------------------------------------------------------

# the stat a book's line on a player is for (the board's LINE column): passing yards, rushing yards, receiving yards
LINE_STAT = games_view.HEADLINE_STAT


def board_rows(board, position, view, game_of=None, lined_only=False):
    """The players of `position` with a role on the board (his volume stat at least games_view.ROLE_BARS' bar, or a back's targets 1.5+;
    a player projected out has none): [{'player', 'team', 'opponent', 'away', 'matchup', 'line', 'gap', 'PTS', stat label: value}], most points
    first. `view`: Average or Median (games_view.value_column). `game_of`: {team: (opponent, away)} from the slate (the board's own
    Opponent column otherwise, without the side). `lined_only`: every player a book has a line on (a role or not), by the size of the
    gap between the model's number for the stat the line is on (LINE_STAT) and the line (`gap`, a share of the line)."""
    if board is None or board.empty or 'Pos' not in board.columns:
        return []
    volume_stat, bar = games_view.ROLE_BARS[position]
    stats = games_view.POSITION_STATS[position]
    game_of = game_of or {}
    out = []
    for rec in board[board['Pos'] == position].to_dict('records'):
        volume = _num(rec.get(games_view.value_column(volume_stat, view)))
        targets = _num(rec.get(games_view.value_column('targets', view)))
        line = games_view.headline_line(rec, position)
        if lined_only:
            if not np.isfinite(line):
                continue
        elif not ((np.isfinite(volume) and volume >= bar) or (position == 'RB' and np.isfinite(targets) and targets >= 1.5)):
            continue
        team = rec.get('Team')
        opponent, away = game_of.get(team, (rec.get('Opponent'), None))
        row = {'player': rec.get('Player'), 'team': team, 'opponent': opponent, 'away': away, 'matchup': _num(rec.get('Matchup Pct')),
               'line': line, 'PTS': _num(rec.get(games_view.value_column('points', view)))}
        for stat, label in stats:
            row[label] = _num(rec.get(games_view.value_column(stat, view)))
        head = _num(rec.get(games_view.value_column(LINE_STAT[position], view)))
        row['gap'] = (head - line) / line if np.isfinite(head) and np.isfinite(line) and line > 0 else np.nan
        out.append(row)
    if lined_only:
        out.sort(key=lambda r: -(abs(r['gap']) if np.isfinite(r['gap']) else -1.0))
    else:
        out.sort(key=lambda r: -(r['PTS'] if np.isfinite(r['PTS']) else -1e9))
    return out


def _game_text(r):
    if not r.get('opponent'):
        return str(r['team'])
    side = '@' if r['away'] else 'vs' if r['away'] is not None else 'v'
    return f'{r["team"]} {side} {r["opponent"]}'


def board_table_html(rows, position, colors=None, lined=False):
    """One position's board table: #, the player with his team (in its colour) and game under the name, MATCHUP, PTS, the position's stats,
    LINE."""
    if not rows:
        return '<div class="e-none">No players with a role.</div>'
    labels = ['PTS'] + [label for _stat, label in games_view.POSITION_STATS[position]]
    has_line = any(np.isfinite(r['line']) for r in rows)
    has_gap = lined and has_line
    head = ('<th>#</th><th class="x-l">PLAYER</th><th>MATCHUP</th>' + ''.join(f'<th>{esc(l)}</th>' for l in labels)
            + ('<th>LINE</th>' if has_line else '') + ('<th>GAP</th>' if has_gap else ''))
    body = []
    for i, r in enumerate(rows, 1):
        rating = r['matchup']
        swatch = f'<span class="e-sw" style="background:{team_color(colors, r["team"])}"></span>' if colors else ''
        cells = [f'<td class="x-n">{i}</td>', f'<td class="x-l"><div class="x-p">{esc(r["player"])}</div><div class="x-g">{swatch}{esc(_game_text(r))}</div></td>',
                 f'<td class="x-m" style="background:{get_matchup_color(rating)}">{rating:.0f}</td>' if np.isfinite(rating) else '<td></td>']
        cells += [f'<td class="x-pts">{_fmt(r[l])}</td>' if l == 'PTS' else f'<td>{_fmt(r[l])}</td>' for l in labels]
        if has_line:
            cells.append(f'<td class="x-line">{_fmt(r["line"])}</td>')
        if has_gap:
            gap = r['gap']
            side = C['positive'] if np.isfinite(gap) and gap > 0 else C['negative']
            shade = games_view.blend(C['surface_container'], side, min(abs(gap) / 0.25, 1.0) * 0.55) if np.isfinite(gap) else 'transparent'
            cells.append(f'<td class="x-s" style="background:{shade}">{_fmt(gap, "{:+.0%}")}</td>')
        body.append(f'<tr>{"".join(cells)}</tr>')
    return f'<table class="x-t x-board x-b{position}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def board_report(title, subtitle, board, view, colors=None, game_of=None, lined_only=False):
    """The bare board as one HTML document: a section per position (each starts a page), a link to each at the top."""
    sections, links = [], []
    for position in games_view.POSITIONS:
        rows = board_rows(board, position, view, game_of, lined_only)
        label = games_view.POSITION_LABELS[position]
        links.append(f'<a href="#p{position}">{esc(label)} · {len(rows)}</a>')
        sections.append(f'<section class="e-pos" id="p{position}"><div class="e-ph">{esc(label.upper())}<span>{len(rows)} players</span></div>'
                        f'{board_table_html(rows, position, colors, lined_only)}</section>')
    body = (f'<header class="e-title" id="top"><div class="e-eyebrow">NFL SCHOLAR</div><h1>{esc(title)}</h1>'
            f'<div class="e-sub">{esc(subtitle)}</div>'
            + ('<div class="e-sub">Only players a book has a line on, largest gap between the model and the line first.</div>' if lined_only else '')
            + f'</header><nav class="e-jump">{"".join(links)}</nav>{"".join(sections)}')
    return document(title, body)


# ---------------------------------------------------------------------------
# the document
# ---------------------------------------------------------------------------

_CSS_PAGE = """
@page{size:__PW__in __PH__in;margin:0.24in;background:__BG__}
html,body{background:__BG__;-webkit-print-color-adjust:exact;print-color-adjust:exact}
*{box-sizing:border-box}
body{margin:0;color:__TEXT__;font-family:__BODY__;font-size:13px;line-height:1.3}
@media screen and (min-width:700px){body{max-width:640px;margin:0 auto;padding:14px}}
a{color:__ACCENT__;text-decoration:none}
.e-title{margin:0 0 12px 0;padding-bottom:9px;border-bottom:1px solid __LINE__}
.e-eyebrow{font-size:10px;font-weight:800;letter-spacing:.22em;color:__ACCENT__}
.e-title h1{font-size:28px;line-height:1.1;margin:3px 0 2px 0;color:#fff;font-weight:800}
.e-sub{font-size:11px;color:__MUTED__}
.e-ph{display:flex;justify-content:space-between;align-items:baseline;font-size:20px;font-weight:800;letter-spacing:.04em;color:#fff;margin:4px 0 8px 0}
.e-ph span{font-size:11px;font-weight:600;color:__MUTED__;letter-spacing:0}
.e-list{break-after:page}
.e-lh,.e-row{display:grid;grid-template-columns:58px minmax(0,1fr) 16px minmax(0,1fr) 42px;gap:4px;align-items:center}
.e-lh{font-size:8.5px;font-weight:700;letter-spacing:.1em;color:__MUTED__;margin:0 0 2px 0}
.e-lh span:nth-child(5){text-align:right}
.e-dh{display:flex;justify-content:space-between;align-items:baseline;font-size:17px;font-weight:800;color:#fff;letter-spacing:.03em;margin:10px 0 0 0;padding-bottom:3px;border-bottom:2px solid __LINE__}
.e-dh span{font-size:11px;font-weight:600;color:__MUTED__}
.e-group{margin:0 0 6px 0;break-inside:avoid}
.e-gh{display:flex;align-items:baseline;gap:8px;margin:9px 0 4px 0;padding-left:8px;border-left:4px solid __ACCENT__}
.e-gh b{font-size:12px;letter-spacing:.1em;color:#fff}
.e-gh span{font-size:11px;color:__MUTED__}
.e-gh em{margin-left:auto;font-style:normal;font-size:10px;color:__MUTED__}
.e-row{height:21px;margin:0 0 1px 0}
.e-time{font-family:__MONO__;font-size:10.5px;font-weight:700;color:__TEXT__;white-space:nowrap}
.e-tm{display:flex;justify-content:space-between;align-items:center;gap:6px;height:21px;padding:0 7px;border-radius:4px;border:1px solid rgba(255,255,255,.22);min-width:0}
.e-tm b{font-size:11.5px;font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.e-tm i{font-family:__MONO__;font-style:normal;font-size:11px;font-weight:700;white-space:nowrap}
.e-vs{text-align:center;font-size:9px;color:__MUTED__}
.e-ou{text-align:right;font-family:__MONO__;font-size:12px;font-weight:700;color:#fff}
.e-m{font-style:italic;opacity:.8}
.e-legend{font-size:9.5px;color:__MUTED__;margin-top:8px}
.e-roomy .e-row{height:30px;margin:0 0 4px 0}
.e-roomy .e-tm{height:30px}
.e-roomy .e-tm b{font-size:13px}
.e-roomy .e-tm i,.e-roomy .e-ou{font-size:13px}
.e-roomy .e-time{font-size:12px}
.e-roomy .e-gh b{font-size:14px}
.e-roomy .e-gh{margin:14px 0 6px 0}
.e-game{break-before:page}
.e-kick{display:flex;justify-content:space-between;font-size:9.5px;font-weight:700;letter-spacing:.1em;color:__MUTED__;margin:0 0 6px 0}
.e-card{background:__SURFACE__;border:1px solid __LINE__;border-radius:10px;padding:0 0 8px 0;margin:0 0 8px 0;break-inside:avoid}
.e-bar{height:5px;border-radius:9px 9px 0 0}
.e-vsrow{display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);gap:8px;align-items:center;padding:10px 12px 2px 12px}
.e-side{display:flex;align-items:center;gap:8px;min-width:0}
.e-side:last-child{flex-direction:row-reverse;text-align:right}
.e-logo{width:34px;height:34px;object-fit:contain;flex:none}
.e-nm{font-size:16px;font-weight:800;line-height:1.1;color:#fff}
.e-tag{font-size:8.5px;font-weight:700;letter-spacing:.12em;color:__MUTED__;margin-top:2px}
.e-at{font-size:13px;font-weight:700;color:__MUTED__}
.e-venue{font-size:9.5px;color:__MUTED__;text-align:center;margin:2px 12px 0 12px}
.e-k{font-size:9px;font-weight:800;letter-spacing:.14em;color:__MUTED__}
.e-sec{margin:8px 12px 2px 12px}
table.e-mm{width:calc(100% - 24px);margin:0 12px;border-collapse:collapse;table-layout:fixed}
.e-mm th{font-size:8.5px;font-weight:700;letter-spacing:.1em;color:__MUTED__;text-align:right;padding:0 6px 3px 6px}
.e-mm th:nth-child(n+2),.e-mm td:nth-child(n+2){width:64px}
.e-mm td{font-family:__MONO__;font-size:16px;font-weight:700;text-align:right;padding:4px 6px;border-top:1px solid __LINE__;color:#fff}
.e-mm td.e-l{font-family:__BODY__;font-size:13px;font-weight:800;text-align:left;white-space:normal}
.e-mm td.e-d{font-size:12px;color:__MUTED__}
.e-mm td.e-warn{color:__WARNFG__;background:__WARNBG__;font-weight:700}
.e-mm tr.e-sep td{border-top:2px solid __LINE__}
.e-sw{display:inline-block;width:5px;height:15px;border-radius:2px;vertical-align:middle;margin-right:7px}
.e-fav{display:inline-block;margin-left:7px;font-family:__BODY__;font-size:7.5px;font-weight:800;letter-spacing:.08em;border-radius:9999px;padding:2px 7px;vertical-align:middle}
.e-wxrow{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:0 12px}
.e-wv{font-family:__MONO__;font-size:21px;font-weight:700;color:#fff;line-height:1.1}
.e-wv small{font-size:12px;font-weight:600;color:__MUTED__;margin-left:2px}
.e-ws{font-size:10px;color:__MUTED__;min-height:13px}
.e-chip{display:inline-block;margin-top:3px;font-size:10.5px;font-weight:700;border-radius:9999px;padding:1px 10px}
.e-wfoot{font-size:9px;color:__MUTED__;margin:6px 12px 0 12px}
.e-wxmsg{font-size:12px;color:__MUTED__;margin:0 12px}
.e-team{margin:0 0 8px 0}
.e-second{break-before:page}
.e-head{break-inside:avoid;break-after:avoid;border:1px solid __LINE__;border-radius:10px;overflow:hidden;background:__SURFACE__;margin:0 0 8px 0}
.e-th{display:flex;align-items:center;gap:10px;padding:8px 12px}
.e-th .e-logo{width:32px;height:32px}
.e-tn{font-size:18px;font-weight:800;line-height:1.1}
.e-ts{font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;opacity:.85}
.e-status{margin-left:auto;font-size:8.5px;font-weight:800;letter-spacing:.06em;border-radius:9999px;padding:2px 9px;white-space:nowrap}
.e-chips{display:flex;gap:1px;background:__LINE__}
.e-ch{flex:1 1 0;min-width:0;background:__SURFACE__;text-align:center;padding:5px 2px}
.e-ch b{display:block;font-size:8px;letter-spacing:.08em;color:__MUTED__;font-weight:700}
.e-ch span{font-family:__MONO__;font-size:14px;font-weight:700;color:#fff}
.e-av{padding:7px 12px 8px 12px}
.e-it{padding:4px 0;border-top:1px solid __LINE__;break-inside:avoid}
.e-it b{font-size:12px;color:#fff}
.e-it i{font-style:normal;font-size:10px;color:__MUTED__;margin-left:6px}
.e-b{display:inline-block;font-size:8px;font-weight:800;letter-spacing:.06em;border-radius:3px;padding:2px 6px;margin-right:7px;text-transform:uppercase;vertical-align:1px}
.e-d2{font-size:9.5px;color:__MUTED__;line-height:1.3;margin-top:2px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.e-ok{font-size:10.5px;color:__MUTED__;border-top:1px solid __LINE__;padding-top:6px;margin-top:2px}
.e-tbl{break-inside:avoid;margin:0 0 8px 0}
.e-tt{display:flex;align-items:baseline;gap:8px;break-after:avoid;margin:8px 0 3px 0;padding-left:8px;border-left:4px solid var(--tc,__ACCENT__)}
.e-tt b{font-size:13px;color:#fff}
.e-tt span{font-size:10px;color:__MUTED__}
.e-props{margin-top:8px}
.e-none{font-size:11px;color:__MUTED__;margin:4px 0}
table.x-t{border-collapse:collapse;width:100%;table-layout:fixed;font-variant-numeric:tabular-nums}
.x-t th.x-cp{width:118px}
.x-t th.x-cm{width:50px}
.x-t th.x-ch{width:78px}
.x-t th.x-cpt{width:38px}
.x-t th.x-cl{width:42px}
.x-t th{font-size:8.5px;font-weight:700;letter-spacing:.05em;color:__MUTED__;text-align:right;padding:3px 3px;border-bottom:1px solid __LINE__;vertical-align:bottom}
.x-t td{font-family:__MONO__;font-size:11.5px;text-align:right;padding:3px 3px;border-bottom:1px solid __LINE__;color:__TEXT__;white-space:nowrap}
.x-t tr{break-inside:avoid}
.x-t tbody tr:nth-child(even) td{background-color:__ZEBRA__}
.x-t .x-l{text-align:left}
.x-t td.x-p,.x-t .x-p{font-family:__BODY__;font-size:12px;font-weight:700;color:#fff;white-space:normal}
.x-t td.x-h{font-family:__BODY__;font-size:9.5px;font-weight:700;white-space:normal;line-height:1.15}
.x-t td.x-m{color:#fff;font-weight:800;text-align:center;width:36px}
.x-t td.x-s{color:#fff;font-weight:700}
.x-t td.x-pts{color:#fff;font-weight:800}
.x-t td.x-line{color:__ACCENT__;font-weight:700}
.x-t td.x-n{color:__MUTED__;font-size:10px}
.x-t tr.x-out td{color:__MUTED__}
.x-g{font-family:__BODY__;font-size:10px;color:__MUTED__;font-weight:400;white-space:normal}
.x-board th.x-l{width:138px}
.x-board th:first-child{width:24px}
.x-board th:nth-child(3){width:48px}
.x-board td{font-size:11px;padding:3px 2px}
.x-board th{padding:3px 2px}
.x-board td.x-l{white-space:normal}
.x-bQB td,.x-bRB td{font-size:10px}
.x-bQB th.x-l,.x-bRB th.x-l{width:118px}
.e-jump{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:12px;font-weight:700;margin:0 0 6px 0}
.e-cover{break-after:page}
.e-ph2{margin-top:18px}
.e-fresh{margin:2px 0 8px 0}
.e-fresh th:nth-child(1){width:120px}
.e-fresh th:nth-child(2){width:56px}
.e-fresh th:nth-child(4){width:96px}
.e-fresh th:nth-child(3),.e-fresh td:nth-child(3){padding-left:16px}
.e-age{display:inline-block;font-size:10px;font-weight:700;border-radius:9999px;padding:1px 9px}
.e-cov{font-size:12px;color:__MUTED__;margin:0 0 4px 0}
.e-cov b{color:#fff}
.e-facts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
.e-fact{background:__SURFACE__;border:1px solid __LINE__;border-radius:8px;padding:8px 12px}
.e-fv{font-size:18px;font-weight:800;color:#fff;margin-top:2px}
.e-help{font-size:11.5px;line-height:1.5;color:__MUTED__}
.e-help p{margin:0 0 8px 0}
.e-help b{color:#fff}
.e-edges{break-after:page}
.e-egroup{margin:0 0 12px 0}
.e-top{margin:0 0 6px 0}
.e-mv{font-family:__MONO__;font-size:10.5px;font-weight:700;white-space:nowrap}
.e-up{color:__POS__}
.e-dn{color:__NEG__}
.x-t td.x-mvc{text-align:left;width:52px}
.e-dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px;vertical-align:1px}
.e-basis{font-size:8px;font-weight:700;color:__WARNFG__;letter-spacing:.04em}
.x-edge th.x-cpe{width:190px}
.x-edge th.x-cmk{width:96px}
.x-edge th.x-cn{width:46px}
.x-edge th.x-cr{width:94px}
.x-edge th.x-cmv{width:52px}
.x-edge td{padding:3px 2px}
.x-edge td.x-mk{font-family:__BODY__;font-size:10.5px;white-space:normal;line-height:1.15}
.x-edge td.x-rc{font-family:__BODY__;font-size:10px;white-space:normal;line-height:1.15}
.x-t td.x-over{color:__POS__;font-weight:800;font-size:10px}
.x-t td.x-under{color:__NEG__;font-weight:800;font-size:10px}

.e-pos{break-before:page}
.e-jump + .e-pos{break-before:auto}
"""


def document(title, body):
    """A complete HTML document around `body`, in the app's dark palette, printed at a phone's proportions."""
    sheet = _CSS_PAGE
    warn_fg, warn_bg = games_view.TONES['warn']
    for token, value in {'__PW__': f'{PAGE_WIDTH_IN:g}', '__PH__': f'{PAGE_HEIGHT_IN:g}', '__BG__': C['surface'], '__TEXT__': C['on_surface'],
                         '__SURFACE__': C['surface_container'], '__MUTED__': C['on_surface_variant'], '__LINE__': C['outline_variant'],
                         '__ACCENT__': C['primary'], '__ZEBRA__': games_view.blend(C['surface'], C['surface_container_high'], 0.35),
                         '__WARNFG__': warn_fg, '__WARNBG__': warn_bg, '__POS__': C['positive'], '__NEG__': C['negative'], '__BODY__': F['body'], '__MONO__': F['mono']}.items():
        sheet = sheet.replace(token, value)
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title)}</title><style>{sheet}</style></head><body>{body}</body></html>')


def find_browser():
    """The path of an installed Chrome or Edge, or None."""
    for path in BROWSER_PATHS:
        if os.path.exists(path):
            return path
    for name in BROWSER_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def to_pdf(document_html, pdf_path, browser=None, timeout=PDF_TIMEOUT_S):
    """Print `document_html` to `pdf_path` with a headless Chrome / Edge (its own throwaway profile, so an open browser window is not
    touched). -> True when the PDF was written; False with no browser, on a timeout or on any failure (the HTML still stands)."""
    browser = browser or find_browser()
    if not browser:
        return False
    work = tempfile.mkdtemp(prefix='nfls_export_')
    try:
        page = os.path.join(work, 'page.html')
        with open(page, 'w', encoding='utf-8') as handle:
            handle.write(document_html)
        target = os.path.abspath(pdf_path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.exists(target):
            os.remove(target)
        command = [browser, '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
                   f'--user-data-dir={os.path.join(work, "profile")}', '--no-pdf-header-footer', '--disable-pdf-tagging', '--virtual-time-budget=15000',
                   f'--print-to-pdf={target}', 'file:///' + page.replace(os.sep, '/')]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout, check=False)
        return os.path.exists(target) and os.path.getsize(target) > 0
    except Exception:
        return False
    finally:
        shutil.rmtree(work, ignore_errors=True)


EXPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'exports')


def save(document_html, stem, pdf=True, folder=EXPORT_DIR):
    """Write `stem`.html (and `stem`.pdf when `pdf` and a browser can print it) into `folder` (exports/, git-ignored).
    -> {'html': path, 'pdf': path or None}."""
    os.makedirs(folder, exist_ok=True)
    html_path = os.path.join(folder, f'{stem}.html')
    with open(html_path, 'w', encoding='utf-8') as handle:
        handle.write(document_html)
    pdf_path = os.path.join(folder, f'{stem}.pdf')
    return {'html': html_path, 'pdf': pdf_path if pdf and to_pdf(document_html, pdf_path) else None}


# ---------------------------------------------------------------------------
# line movement: an arrow and the amount, since the line was first pulled
# ---------------------------------------------------------------------------

MOVE_MIN = 0.05                      # a line that has moved less than this has not moved


def move_html(move):
    """'▲ +1.5' (green) or '▼ -1.0' (red): how far the consensus line has moved since it was first seen; empty when it has not moved."""
    move = _num(move)
    if not np.isfinite(move) or abs(move) < MOVE_MIN:
        return ''
    cls, arrow = ('e-up', '▲') if move > 0 else ('e-dn', '▼')
    return f'<span class="e-mv {cls}">{arrow} {move:+.1f}</span>'


# ---------------------------------------------------------------------------
# the edges page: the slate's biggest gaps between the model and the line
# ---------------------------------------------------------------------------

EDGE_GROUPS = (('Passing', games_view.PROP_GROUPS['Passing']), ('Rushing', games_view.PROP_GROUPS['Rushing']),
               ('Receiving', games_view.PROP_GROUPS['Receiving']))
EDGE_FLOOR = 0.10                    # a gap of 10% of the line (CFB's Edges floor)
EDGE_CAP = 30                        # rows a group shows before the rest are counted
KIND_COLOR = {'solid': '#1ed760', 'leaning': '#ffb020', 'flat': '#8a93b8', 'wrong': '#ef4444', 'thin': '#565d8c'}


def record_text(record, kind, conditional=False):
    """The lean record as a short cell: '56% (30/54)', '3/9 so far' while thin, 'n/a' for an if-he-plays number or no record."""
    if conditional or not record:
        return 'n/a'
    if kind == 'thin':
        return f"{record['hits']}/{record['n']} so far"
    return f"{record['rate']:.0%} ({record['hits']}/{record['n']})"


def edges_html(rows, colors, floor=EDGE_FLOOR):
    """The edges page(s): per prop group (Passing, Rushing, Receiving) the lines the model is `floor` or more away from, biggest gap first.
    `rows`: [{'player', 'team', 'opponent', 'away', 'start' (label), 'group', 'market', 'model', 'line', 'diff_pct', 'lean', and when
    kept, 'record_text', 'kind', 'move', 'basis'}]. The RECORD and MOVE columns appear when a row carries them. Empty when there is none."""
    rows = [r for r in rows if np.isfinite(_num(r.get('diff_pct')))]
    if not rows:
        return ''
    has_record = any(r.get('record_text') for r in rows)
    has_move = any(np.isfinite(_num(r.get('move'))) and abs(_num(r.get('move'))) >= MOVE_MIN for r in rows)
    sections = []
    for name, _stats in EDGE_GROUPS:
        part = sorted((r for r in rows if r['group'] == name), key=lambda r: -abs(r['diff_pct']))
        big = [r for r in part if abs(r['diff_pct']) >= floor]
        if not part:
            continue
        shown = big[:EDGE_CAP]
        body = []
        for r in shown:
            color = team_color(colors, r['team'])
            side = C['positive'] if r['lean'] == 'OVER' else C['negative']
            game = _game_text(r) + (f' · {str(r["start"]).replace(" ET", "")}' if r.get('start') else '')
            lean_cls = 'x-over' if r['lean'] == 'OVER' else 'x-under'
            basis = f' <span class="e-basis">{esc(r["basis"])}</span>' if r.get('basis') else ''
            record = ''
            if has_record:
                dot = f'<span class="e-dot" style="background:{KIND_COLOR.get(r.get("kind"), KIND_COLOR["thin"])}"></span>'
                record = f'<td class="x-l x-rc">{dot}{esc(r.get("record_text") or "n/a")}</td>'
            move = f'<td class="x-mvc">{move_html(r.get("move"))}</td>' if has_move else ''
            body.append(
                f'<tr><td class="x-l"><div class="x-p">{esc(r["player"])}{basis}</div>'
                f'<div class="x-g"><span class="e-sw" style="background:{color}"></span>{esc(game)}</div></td>'
                f'<td class="x-l x-mk">{esc(r["market"])}</td><td>{_fmt(r["model"])}</td><td class="x-line">{_fmt(r["line"])}</td>'
                f'<td class="x-s" style="background:{games_view.blend(C["surface_container"], side, min(abs(r["diff_pct"]) / 0.25, 1.0) * 0.55)}">'
                f'{r["diff_pct"]:+.0%}</td><td class="{lean_cls}">{esc(r["lean"])}</td>{record}{move}</tr>')
        more = len(big) - len(shown)
        under = len(part) - len(big)
        note = ' · '.join(x for x in (f'{more} more over the floor' if more > 0 else '',
                                      f'{under} under {floor:.0%} not shown' if under > 0 else '') if x)
        head = ('<tr><th class="x-l x-cpe">PLAYER</th><th class="x-l x-cmk">PROP</th><th class="x-cn">MODEL</th><th class="x-cn">LINE</th>'
                '<th class="x-cn">GAP</th><th class="x-cn">LEAN</th>' + ('<th class="x-l x-cr">RECORD</th>' if has_record else '')
                + ('<th class="x-cmv">MOVE</th>' if has_move else '') + '</tr>')
        table = (f'<table class="x-t x-edge"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>' if body
                 else f'<div class="e-none">No {name.lower()} line is {floor:.0%} or more from the model.</div>')
        sections.append(f'<div class="e-egroup"><div class="e-tt"><b>{name}</b><span>{len(big)} over {floor:.0%} of the line'
                        f'{" · " + note if note and big else ""}</span></div>{table}</div>')
    legend = ('MODEL is the median projection; GAP is its distance from the line as a share of the line.'
              + (' RECORD is how leans of this kind (position, prop and side) have done so far.' if has_record else '')
              + (' ▲ ▼ show how far the line has moved since it was first pulled.' if has_move else ''))
    return (f'<section class="e-edges" id="edges"><div class="e-ph">EDGES<span>model vs line, biggest gap first</span></div>'
            f'<div class="e-legend e-top">{legend}</div>{"".join(sections)}</section>')


# ---------------------------------------------------------------------------
# which games: the whole slate by default, or a day or a Sunday window
# ---------------------------------------------------------------------------

SELECTIONS = (('all', 'Full slate'), ('thursday', 'Thursday'), ('friday', 'Friday'), ('saturday', 'Saturday'), ('sunday', 'Sunday'),
              ('morning', 'Sun morning'), ('early', 'Sun early'), ('late', 'Sun late'), ('night', 'Sun night'), ('monday', 'Monday'))
_SELECTION_DAY = {'monday': 0, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6}


def _in_selection(entry, key):
    local = eastern(entry['game'].get('start'))
    day = local.weekday() if local is not None else None
    if key == 'all':
        return True
    if key in _SELECTION_DAY:
        return day == _SELECTION_DAY[key]
    return day == SUNDAY and slate_of(local) == key


def selection_options(entries):
    """[(key, label)] of the selections that have at least one game, 'all' first. A Sunday window is offered only when Sunday has more
    than one window (otherwise it is the same as 'Sunday')."""
    options = [(k, label) for k, label in SELECTIONS if k == 'all' or any(_in_selection(e, k) for e in entries)]
    windows = [k for k, _l in options if k in {s[0] for s in SLATES}]
    if len(windows) < 2:
        options = [(k, label) for k, label in options if k not in windows]
    return options


def select_games(entries, key):
    """The entries of one selection (see SELECTIONS); an unknown key is the whole slate."""
    if key not in dict(SELECTIONS):
        return list(entries)
    return [e for e in entries if _in_selection(e, key)]


def selection_label(key):
    return dict(SELECTIONS).get(key, 'Full slate')


# ---------------------------------------------------------------------------
# the cover: what the lines are as of, what is inside
# ---------------------------------------------------------------------------

_AGE_TONE = {'fresh': ('#8ff0b3', '#1ed760'), 'aging': ('#ffd27a', '#ffb020'), 'stale': ('#ff9aa6', '#ef4444'), 'none': ('#7d86ad', '#565d8c')}
FRESH_HOURS, AGING_HOURS = 3.0, 24.0         # a pull under 3 hours old is fresh, under a day aging, older stale


def age_text(hours):
    """'10 min ago', '3 h ago', '2 days ago'; '' with no age."""
    if hours is None or not np.isfinite(hours):
        return ''
    if hours < 1:
        return f'{max(int(round(hours * 60)), 1)} min ago'
    if hours < 48:
        return f'{hours:.0f} h ago'
    return f'{hours / 24:.0f} days ago'


def freshness_rows(lines_by_book, fetched_at, now=None):
    """The cover's lines table from a market pull: [{'provider', 'lines', 'updated', 'age', 'state'}] for each book with its line count,
    every book stamped with the pull's time (one pull fetches every book). `fetched_at`: a Timestamp (UTC when naive) or None."""
    stamp = None
    if fetched_at is not None:
        try:
            stamp = pd.Timestamp(fetched_at)
            stamp = stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp
        except Exception:
            stamp = None
    now = pd.Timestamp.now(tz='UTC') if now is None else now
    hours = (now - stamp).total_seconds() / 3600.0 if stamp is not None else None
    state = 'none' if hours is None else 'fresh' if hours < FRESH_HOURS else 'aging' if hours < AGING_HOURS else 'stale'
    updated = stamp.tz_convert(EASTERN).strftime('%b %d, %I:%M %p ET').replace(' 0', ' ') if stamp is not None else '—'
    return [{'provider': book, 'lines': f'{int(n):,}' if n else '—', 'updated': updated if n else '—', 'age': age_text(hours) if n else '',
             'state': state if n else 'none'} for book, n in lines_by_book.items()]


def cover_html(title, subtitle, freshness, priced, on_board, props, selection, games, edges, view='Average'):
    """The first page: the title, how fresh each book's lines are, how many players have a line, and what the export holds.
    `freshness`: [{'provider', 'lines', 'updated', 'age', 'state'}]; `priced` / `on_board`: players with a line / on the board; `props`: priced
    props in the games; `selection`: the label of the games chosen; `games`, `edges`: counts."""
    rows = []
    for r in freshness or []:
        fg, base = _AGE_TONE.get(r.get('state'), _AGE_TONE['none'])
        chip = f'<span class="e-age" style="color:{fg};background:{base}33">{esc(r["age"])}</span>' if r.get('age') else ''
        rows.append(f'<tr><td class="x-l x-p">{esc(r["provider"])}</td><td>{esc(r["lines"])}</td><td class="x-l">{esc(r["updated"])}</td>'
                    f'<td class="x-l">{chip}</td></tr>')
    lines = (f'<table class="x-t e-fresh"><thead><tr><th class="x-l">BOOK</th><th>LINES</th><th class="x-l">LAST UPDATED</th>'
             f'<th class="x-l">AGE</th></tr></thead><tbody>{"".join(rows)}</tbody></table>' if rows
             else '<div class="e-none">No market lines were loaded when this was built.</div>')
    cover = [('Selection', esc(selection)), ('Games', f'{games}'), ('Priced props in these games', f'{props}'),
             ('Edges listed (10%+ from the line)', f'{edges}')]
    cells = ''.join(f'<div class="e-fact"><div class="e-k">{esc(k).upper()}</div><div class="e-fv">{v}</div></div>' for k, v in cover)
    cov = (f'<div class="e-cov"><b>{priced}</b> of <b>{on_board}</b> players on the board have a line.</div>' if on_board else '')
    numbers = ('Player tables are averages (the expected points and stats, what fantasy adds up) unless the header says medians; the '
               'edges and props always compare the median (the number a line is priced against).' if view == 'Average' else
               'Player tables are medians (the middle game: the number a line is priced against, and a floor reference); PTS is the median '
               'game too.')
    return (f'<section class="e-cover"><header class="e-title"><div class="e-eyebrow">NFL SCHOLAR</div><h1>{esc(title)}</h1>'
            f'<div class="e-sub">{esc(subtitle)}</div></header>'
            f'<div class="e-ph">LINES<span>when the books were last pulled</span></div>{lines}{cov}'
            f'<div class="e-ph e-ph2">THIS EXPORT</div><div class="e-facts">{cells}</div>'
            '<div class="e-ph e-ph2">HOW TO READ IT</div>'
            '<div class="e-help"><p><b>Edges</b> lists where the model and the line disagree most. <b>The slate</b> follows by day and '
            'Sunday window; tap a game to jump to it. Inside a game, the away team comes first, then the home team and the game\'s props.</p>'
            f'<p>{esc(numbers)} Every number assumes the player holds his projected role.</p>'
            '<p><b>MATCHUP</b> is the opponent\'s defense against the position, 1-100 (100 = allows the most fantasy points). '
            '<b>LINE</b> is the books\' line on the headline stat: passing yards (QB), rushing yards (RB), receiving yards (WR/TE).</p>'
            '</div></section>')


# ---------------------------------------------------------------------------
# where the files go: the exports folder, and optionally a OneDrive folder so a phone has them
# ---------------------------------------------------------------------------

def default_copy_folder():
    """The personal OneDrive's 'NFL Scholar' folder (where the 2026-10-18 export saved) when OneDrive is on this PC, else ''."""
    root = os.environ.get('OneDriveConsumer') or os.path.join(os.path.expanduser('~'), 'OneDrive')
    return os.path.join(root, 'NFL Scholar') if os.path.isdir(root) else ''


def _settings_path(folder=None):
    return os.path.join(folder or EXPORT_DIR, 'settings.json')


def load_settings(folder=None):
    """{'copy': bool, 'folder': str}: the copy-to-folder choice, remembered between sessions (exports/settings.json)."""
    import json
    try:
        with open(_settings_path(folder), encoding='utf-8') as handle:
            saved = json.load(handle)
        return {'copy': bool(saved.get('copy')), 'folder': str(saved.get('folder') or '')}
    except Exception:
        found = default_copy_folder()
        return {'copy': bool(found), 'folder': found}


def save_settings(settings, folder=None):
    import json
    try:
        os.makedirs(folder or EXPORT_DIR, exist_ok=True)
        with open(_settings_path(folder), 'w', encoding='utf-8') as handle:
            json.dump({'copy': bool(settings.get('copy')), 'folder': str(settings.get('folder') or '')}, handle)
    except OSError:
        pass


def copy_files(files, target):
    """Copy the PDF and HTML of a saved export into `target` (created if need be). -> the paths written, [] when `target` is empty or fails."""
    if not target:
        return []
    out = []
    try:
        os.makedirs(target, exist_ok=True)
        for key in ('pdf', 'html'):
            if files.get(key) and os.path.exists(files[key]):
                out.append(shutil.copy2(files[key], target))
    except OSError:
        return []
    return out
