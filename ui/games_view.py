"""The Weekly Rankings GAMES view's pure parts: no Streamlit calls, so every rule here is tested on a small frame (2026-10-23, user:
"this is the main difference between the two ... replicate this very closely" - a port of CFB Scholar's ui/games_view.py, the same
cards, colours and layout on this app's board).

What lives here: a player's HEALTH (one short label for the table, the fuller reason for the availability box), who goes in a team's
availability box and where his volume went, the weather as the numbers a person reads, the model-against-market numbers as one small
table, a team's projected totals, the player table (every player on the board for the team) and its styling, the props table, and the
HTML of the header card and the team blocks. The tab (ui/tabs/rankings.py) owns the widgets and the projection dialog.

Differences from CFB's, each because the board is different: the player table has a PTS column (this board ranks fantasy points); the
health flags read this app's availability columns (injury report, FantasyPros, your rulings, the QB1 choice), not a conference
report or an assumed-out rule; the MODEL score is data/points_fit.py (display only).

THE DARK THEME ONLY: the app is dark (config.THEME); every colour here comes from its tokens.
"""
import html

import numpy as np
import pandas as pd

from config import THEME, get_position_chip_bg, get_position_color
from ui.styling import get_matchup_color

C = THEME['colors']
F = THEME['fonts']
POSITIONS = ('QB', 'RB', 'WR', 'TE')

# ---------------------------------------------------------------------------
# colours
# ---------------------------------------------------------------------------


def _rgb(color):
    color = str(color).lstrip('#')
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def blend(low, high, fraction):
    """The hex colour `fraction` (0-1) of the way from `low` to `high`."""
    fraction = min(max(float(fraction), 0.0), 1.0)
    a, b = _rgb(low), _rgb(high)
    return '#%02x%02x%02x' % tuple(round(x + (y - x) * fraction) for x, y in zip(a, b))


_BASE = C['surface_container']
# tone -> (text colour, background): the health labels and the weather chips share them
TONES = {
    'out': ('#ff9aa6', blend(_BASE, '#ef4444', 0.34)),
    'assumed': ('#f0b3ba', blend(_BASE, '#ef4444', 0.16)),
    'half': ('#ffd27a', blend(_BASE, '#ffb020', 0.30)),
    'warn': ('#ffd27a', blend(_BASE, '#ffb020', 0.20)),
    'note': ('#bccaf5', blend(_BASE, '#3860be', 0.34)),
    'info': ('#8ff0b3', blend(_BASE, '#1ed760', 0.18)),
    'plain': (C['on_surface_variant'], C['surface_container_high']),
    'cold': ('#9be7ff', blend(_BASE, '#38bdf8', 0.26)),
    'cool': ('#a5c8ff', blend(_BASE, '#60a5fa', 0.24)),
    'hot': ('#ffb27a', blend(_BASE, '#fb923c', 0.28)),
    'wet': ('#8fb8ff', blend(_BASE, '#3b82f6', 0.30)),
    'wet2': ('#b3b1ff', blend(_BASE, '#6366f1', 0.34)),
    'snow': ('#e6f6ff', blend(_BASE, '#e0f2fe', 0.22)),
}
TONE_RANK = {'out': 0, 'assumed': 1, 'half': 2, 'warn': 3, 'note': 4, 'info': 5}
_MUTED_TEXT = '#7d86ad'                       # a player who is out: his row is dimmed

# ---------------------------------------------------------------------------
# a player's health
# ---------------------------------------------------------------------------

SIDELINED = 0.01                               # data.weekly_projections.SIDELINED_AVAILABILITY: at or below, projected out
OVERRIDE_SOURCE = 'manual target-week availability override'
# a player is on the "with a role" panel when his own volume says so: (stat, minimum)
ROLE_BARS = {'QB': ('passing_attempts', 3.0), 'RB': ('rushing_attempts', 2.0), 'WR': ('targets', 1.0), 'TE': ('targets', 1.0)}
# a player who is out is in the availability box when his normal role (expected snap share) was at least this
BOX_MIN_SNAP_SHARE = 0.15
_VOLUME_UNIT = {'passing_attempts': 'attempts', 'rushing_attempts': 'carries', 'targets': 'targets'}
_STATUS_LABELS = {'ir': 'IR', 'pup': 'PUP', 'nfi': 'NFI', 'gtd': 'Game-time decision', 'dnp': 'DNP'}

# the headline of a player's flags: the first of these that applies
FLAG_PRIORITY = ('ruled_out', 'report_out', 'partial', 'questionable', 'qb1_choice', 'probable', 'full_role')


def _num(value):
    return pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]


def _truthy(value):
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def _status_word(status):
    """'Questionable' from 'Questionable (45%)'; 'IR' from 'Ir'."""
    word = str(status or '').split(' (')[0].strip()
    return _STATUS_LABELS.get(word.lower(), word.title() if word else '')


def _origin(source):
    source = str(source or '')
    if source == OVERRIDE_SOURCE:
        return 'Your ruling'
    if 'fantasypros' in source.lower():
        return 'FantasyPros injury report'
    if 'injury report' in source.lower():
        return 'Injury report'
    return 'Report'


def player_health(row, detail=None):
    """What is known about whether a player plays, from the board row (Injury Status, Availability, Workload If Active, the
    availability source and note, the QB1 flag) and `detail` (his explanation, for the Questionable workload step).

    -> {'flags': [...], 'label', 'tone'}: `flags` are every condition that applies, the headline first (FLAG_PRIORITY), each
    {'key', 'label', 'tone', 'source', 'detail'}; `label` / `tone` are the headline's ('' when he is healthy as far as anything here
    can tell). Nothing here changes a projection, it only says why one is what it is."""
    detail = detail or {}
    status = str(row.get('Injury Status') or '').strip()
    word = _status_word(status)
    source = str(row.get('_availability_source') or '')
    own = source == OVERRIDE_SOURCE
    origin = _origin(source)
    note = str(row.get('_availability_note') or '').strip()
    note_text = f' ({note})' if note and note.lower() != 'nan' else ''
    availability = _num(row.get('Availability'))
    workload = _num(row.get('Workload If Active'))
    flags = {}

    def add(key, label, tone, where, text):
        flags[key] = {'key': key, 'label': label, 'tone': tone, 'source': where, 'detail': text}

    if np.isfinite(availability) and availability <= SIDELINED:
        add('ruled_out' if own else 'report_out', word or 'Out', 'out', origin,
            'projected out; his volume went to his teammates' + note_text)
    elif (np.isfinite(workload) and workload < 0.995) or (np.isfinite(availability) and availability < 0.995):
        share = (workload if np.isfinite(workload) else 1.0) * (availability if np.isfinite(availability) else 1.0)
        add('partial', f'{share:.0%}' + (' (yours)' if own else ''), 'half', origin,
            f'projected at {share:.0%} of his normal workload' + (f' ({word})' if word else '') + note_text)
    elif word.lower() == 'questionable':
        step = detail.get('questionable_workload') or {}
        before, after = _num(step.get('before')), _num(step.get('after'))
        text = 'still projected to play'
        if np.isfinite(before) and np.isfinite(after) and abs(before - after) > 0.05:
            text += f'; points {before:.1f} -> {after:.1f} for the Questionable workload step'
        add('questionable', status or 'Questionable', 'warn', origin, text + note_text)
    elif word.lower() == 'probable':
        add('probable', 'Probable', 'note', origin, 'projected at a full role')
    elif own and word:
        add('full_role', f'{word} (you)', 'info', 'Your ruling', 'no change to the projection (a full role)' + note_text)
    if _truthy(row.get('QB1 Selection Required')) and str(row.get('Pos')) == 'QB':
        add('qb1_choice', 'QB1?', 'warn', 'Model',
            'more than one quarterback could start: pick the starter in Live data pulls -> QB1 overrides')
    ordered = [flags[key] for key in FLAG_PRIORITY if key in flags]
    head = ordered[0] if ordered else {}
    return {'flags': ordered, 'label': head.get('label', ''), 'tone': head.get('tone', '')}


def is_out(health):
    """True when the player is projected out for the game (his stats are zero), as opposed to flagged."""
    return bool(health['flags']) and health['flags'][0]['key'] in ('ruled_out', 'report_out')


# ---------------------------------------------------------------------------
# the availability box
# ---------------------------------------------------------------------------

def _handoff_index(ledger):
    """{(team, vacated player): ['his targets went to Smith +2.9, Jones +1.1', ...]} from the board's vacancy ledger (both the
    allocator's and the QB hand-off's entries: 'team', 'volume', 'source_player' - comma-joined when two QBs are out - and
    'recipients' [{'player', 'allocated'}])."""
    out = {}
    for entry in ledger or []:
        rows = []
        for r in entry.get('recipients') or []:
            added = _num(r.get('allocated'))
            if np.isfinite(added) and added >= 0.05:
                rows.append((str(r.get('player', '')), float(added)))
        rows = sorted(rows, key=lambda x: -x[1])[:3]
        if not rows:
            continue
        stat = str(entry.get('volume') or '')
        unit = _VOLUME_UNIT.get(stat, stat.replace('_', ' ') or 'volume')
        text = f'his {unit} went to ' + ', '.join(f'{name} +{added:.1f}' for name, added in rows)
        for player in str(entry.get('source_player') or '').split(', '):
            if player:
                out.setdefault((str(entry.get('team', '')), player), []).append(text)
    return out


def availability_items(group, explanations=None, ledger=None, min_snap_share=BOX_MIN_SNAP_SHARE):
    """-> (items, hidden) for one team's availability box. `group`: the team's rows of the board; `explanations`: the board's
    {(player, pos, team): detail}; `ledger`: the board's vacancy ledger. An item is {'player', 'position', 'label', 'tone', 'source',
    'detail', 'also'}, the most serious first; `hidden` counts the players who are out with too little role to list (their row in the
    player table still says so)."""
    explanations = explanations or {}
    handoff = _handoff_index(ledger)
    items, hidden = [], 0
    for row in group.to_dict('records'):
        key = (row.get('Player'), row.get('Pos'), row.get('Team'))
        health = player_health(row, explanations.get(key))
        if not health['flags']:
            continue
        head = health['flags'][0]
        share = _num(row.get('Expected Snap Share'))
        share = float(share) if np.isfinite(share) else 0.0
        if is_out(health) and head['key'] != 'ruled_out' and share < min_snap_share:
            hidden += 1
            continue
        text = head['detail']
        moved = handoff.get((str(row.get('Team')), str(row.get('Player'))))
        if is_out(health) and moved:
            text = text.replace('; his volume went to his teammates', '') + '; ' + '; '.join(moved)
        items.append({'player': row.get('Player'), 'position': row.get('Pos'), 'label': head['label'], 'tone': head['tone'],
                      'source': head['source'], 'detail': text, 'also': [flag['label'] for flag in health['flags'][1:]],
                      '_share': share})
    items.sort(key=lambda i: (TONE_RANK.get(i['tone'], 9), -i['_share'], str(i['player'])))
    for item in items:
        item.pop('_share')
    return items, hidden


# ---------------------------------------------------------------------------
# the player table
# ---------------------------------------------------------------------------

# the stat columns when every position is on one table (a stat a position does not use is blank)
ALL_STATS = (('passing_attempts', 'ATT'), ('passing_completions', 'COMP'), ('passing_yards', 'PASS YD'), ('passing_tds', 'PASS TD'),
             ('passing_interceptions', 'INT'), ('rushing_attempts', 'CAR'), ('rushing_yards', 'RUSH YD'), ('rushing_tds', 'RUSH TD'),
             ('targets', 'TGT'), ('receptions', 'REC'), ('receiving_yards', 'REC YD'), ('receiving_tds', 'REC TD'))
ALL_POSITION_STATS = {
    'QB': ('passing_attempts', 'passing_completions', 'passing_yards', 'passing_tds', 'passing_interceptions', 'rushing_attempts',
           'rushing_yards', 'rushing_tds'),
    'RB': ('rushing_attempts', 'rushing_yards', 'rushing_tds', 'targets', 'receptions', 'receiving_yards', 'receiving_tds'),
    'WR': ('targets', 'receptions', 'receiving_yards', 'receiving_tds'),
    'TE': ('targets', 'receptions', 'receiving_yards', 'receiving_tds'),
}
# a position's own table, the headline stat first (a running back's receiving touchdowns added)
POSITION_STATS = {
    'QB': (('passing_yards', 'PASS YD'), ('passing_attempts', 'ATT'), ('passing_completions', 'COMP'), ('passing_tds', 'PASS TD'),
           ('passing_interceptions', 'INT'), ('rushing_attempts', 'CAR'), ('rushing_yards', 'RUSH YD'), ('rushing_tds', 'RUSH TD')),
    'RB': (('rushing_attempts', 'CAR'), ('rushing_yards', 'RUSH YD'), ('rushing_tds', 'RUSH TD'), ('targets', 'TGT'),
           ('receptions', 'REC'), ('receiving_yards', 'REC YD'), ('receiving_tds', 'REC TD')),
    'WR': (('receiving_yards', 'REC YD'), ('receptions', 'REC'), ('targets', 'TGT'), ('receiving_tds', 'REC TD')),
    'TE': (('receiving_yards', 'REC YD'), ('receptions', 'REC'), ('targets', 'TGT'), ('receiving_tds', 'REC TD')),
}
# the market line the LINE column shows: the position's headline stat
HEADLINE_STAT = {'QB': 'passing_yards', 'RB': 'rushing_yards', 'WR': 'receiving_yards', 'TE': 'receiving_yards'}
# the grid's column widths in pixels (the PLAYER column is pinned). The two teams' tables sit side by side, so each column is as
# narrow as its header and a three-digit number allow.
COLUMN_WIDTHS = {'PLAYER': 116, 'POS': 40, 'MATCHUP': 60, 'HEALTH': 88, 'PTS': 46, 'LINE': 50}
STAT_WIDTHS = {'ATT': 40, 'COMP': 46, 'PASS YD': 58, 'PASS TD': 58, 'INT': 36, 'CAR': 40, 'RUSH YD': 58, 'RUSH TD': 58, 'TGT': 40, 'REC': 40,
               'REC YD': 56, 'REC TD': 56}
STAT_COLUMN_WIDTH = 54
POSITION_LABELS = {'QB': 'Quarterbacks', 'RB': 'Running backs', 'WR': 'Wide receivers', 'TE': 'Tight ends'}


def value_column(stat, view='Average'):
    """The board column a stat (or 'points') reads in the chosen view: the mean, or its median from data/weekly_median.py."""
    base = 'Model Proj Pts' if stat == 'points' else stat
    if view != 'Median':
        return base
    return 'Median Proj Pts' if stat == 'points' else 'Median ' + stat


def team_table(group, display_stats, position='ALL', role_only=False, explanations=None, view='Average'):
    """-> the team's player table: EVERY QB/RB/WR/TE on the board for the team, in position order, the players projected in a game
    first by volume and the players who are out last, with a HEALTH column (blank when healthy), PTS (Model Proj Pts), the stats and
    the market's line on the headline stat, in the chosen `view` (Average or Median). A player who is out has blank stats.
    `position`: 'ALL' (the compact stat set) or one position (that position's own set). `role_only` drops the players with no role who are not flagged. Hidden columns (`_key`,
    `_tone`, `_out`) carry the row click and the styling. `group` may carry 'Matchup Pct' (0-100, 100 = the easiest matchup) and
    'Mkt <stat>' columns."""
    explanations = explanations or {}
    positions = POSITIONS if position == 'ALL' else (position,)
    rows = []
    for index, pos in enumerate(positions):
        stats = ALL_STATS if position == 'ALL' else tuple(display_stats[pos])
        volume_stat, bar = ROLE_BARS[pos]
        for rec in group[group['Pos'] == pos].to_dict('records'):
            key = (rec.get('Player'), pos, rec.get('Team'))
            health = player_health(rec, explanations.get(key))
            out = is_out(health)
            volume = _num(rec.get(volume_stat))
            volume = float(volume) if np.isfinite(volume) else 0.0
            share = _num(rec.get('Expected Snap Share'))
            share = float(share) if np.isfinite(share) else 0.0
            secondary = _num(rec.get('targets')) if pos == 'RB' else 0.0
            has_role = volume >= bar or (pos == 'RB' and np.isfinite(secondary) and secondary >= 1.5)
            # a flagged player stays when he is in the game or his role was real
            keep = bool(health['flags']) and (not out or share >= BOX_MIN_SNAP_SHARE or health['flags'][0]['key'] == 'ruled_out')
            if role_only and not has_role and not keep:
                continue
            rating = _num(rec.get('Matchup Pct'))
            points = _num(rec.get(value_column('points', view)))
            shown = {'PLAYER': rec.get('Player'), 'POS': pos, 'MATCHUP': float(rating) if np.isfinite(rating) else np.nan,
                     'HEALTH': health['label'], 'PTS': float(points) if np.isfinite(points) and not out else np.nan}
            for stat, label in stats:
                applies = position != 'ALL' or stat in ALL_POSITION_STATS[pos]
                value = _num(rec.get(value_column(stat, view)))
                shown[label] = float(value) if applies and not out and np.isfinite(value) else np.nan
            line = headline_line(rec, pos)
            shown['LINE'] = float(line) if np.isfinite(line) else np.nan
            shown['_key'] = key
            shown['_tone'] = health['tone']
            shown['_out'] = out
            shown['_order'] = (index, out, -(share if out else volume))
            rows.append(shown)
    rows.sort(key=lambda r: r['_order'])
    for r in rows:
        r.pop('_order')
    return pd.DataFrame(rows)


def position_table(group, position, role_only=False, explanations=None, view='Average'):
    """A team's players at ONE position, compacted for a half-width grid: the team_table of that position's own stat set, with the
    small coloured POS chip and the matchup rating (1-100, coloured the same way), without a HEALTH or LINE column when nobody in it
    has a value there. Empty when the team has no one at the position for this filter."""
    frame = team_table(group, POSITION_STATS, position, role_only=role_only, explanations=explanations, view=view)
    if frame.empty:
        return frame
    drop = []
    if not frame['HEALTH'].fillna('').astype(str).str.len().gt(0).any():
        drop.append('HEALTH')
    if frame['LINE'].isna().all():
        drop.append('LINE')
    if frame['MATCHUP'].isna().all():
        drop.append('MATCHUP')
    return frame.drop(columns=drop)


def _blankable(frame):
    """-> (the table's visible columns with every missing number as 0.0, a frame of which cells are blank). The data grid prints a
    missing number as 'None' whatever a Styler's na_rep says, and a stat a position does not have is not missing, it does not exist:
    so those cells hold 0.0 and the styling paints their text in their own background, which reads as blank and leaves the
    column numeric, so a click on its header sorts it as numbers."""
    show = frame[[c for c in frame.columns if not c.startswith('_')]].reset_index(drop=True)
    blank = show.isna()
    numeric = [c for c in show.columns if pd.api.types.is_numeric_dtype(show[c])]
    show[numeric] = show[numeric].fillna(0.0)
    return show, blank


def styled_table(frame):
    """The player table as a Styler: position chips, a coloured HEALTH cell, a bold PTS, an amber LINE, a dimmed row for a player who
    is out, zebra banding, blank cells (not 'None' or '--') where a position has no such stat. Numbers stay numbers, so the grid sorts
    them."""
    show, blank = _blankable(frame)
    tones, outs = list(frame['_tone']), list(frame['_out'])
    zebra = blend(C['surface_container'], C['surface_container_high'], 0.4)

    def style_row(row):
        i = int(row.name)
        base = zebra if i % 2 else C['surface_container']
        text = _MUTED_TEXT if outs[i] else C['on_surface']
        styles = []
        for column in show.columns:
            value = row[column]
            if blank.at[i, column]:
                styles.append(f'background-color:{base}; color:{base};')
            elif column == 'POS':
                styles.append(f"background-color:{get_position_chip_bg(value) or base}; "
                              f"color:{get_position_color(value) or text}; font-weight:bold;")
            elif column == 'MATCHUP':
                styles.append(f'background-color:{get_matchup_color(value)}; color:#ffffff; font-weight:bold;')
            elif column == 'HEALTH' and value:
                fg, bg = TONES.get(tones[i], (text, base))
                styles.append(f'background-color:{bg}; color:{fg}; font-weight:bold;')
            elif column in ('PLAYER', 'PTS'):
                styles.append(f'background-color:{base}; color:{text}; font-weight:bold;')
            elif column == 'LINE' and pd.notna(value):
                styles.append(f"background-color:{base}; color:{C['primary']}; font-weight:bold;")
            else:
                styles.append(f'background-color:{base}; color:{text};')
        return styles

    numeric = {c: '{:.1f}' for c in show.columns if pd.api.types.is_numeric_dtype(show[c])}
    if 'MATCHUP' in numeric:
        numeric['MATCHUP'] = '{:.0f}'
    return show.style.apply(style_row, axis=1).format(numeric)


def team_chips(group, view='Average'):
    """[(label, value)] for a team's projected totals: its plays (attempts + carries + sacks are not on the board, so pass attempts
    plus carries) and the sums of its players' attempts and yards in the chosen view (a sum of medians is not the team's median;
    the Average view adds up)."""
    def total(stat, positions=None):
        part = group if positions is None else group[group['Pos'].isin(positions)]
        column = value_column(stat, view)
        return float(pd.to_numeric(part[column], errors='coerce').sum()) if column in part.columns and len(part) else np.nan
    pass_att, rush_att = total('passing_attempts', ('QB',)), total('rushing_attempts')
    plays = pass_att + rush_att if np.isfinite(pass_att) and np.isfinite(rush_att) else np.nan
    return [('PLAYS', plays), ('PASS ATT', pass_att), ('PASS YD', total('passing_yards', ('QB',))),
            ('RUSH ATT', rush_att), ('RUSH YD', total('rushing_yards'))]


# ---------------------------------------------------------------------------
# the props table
# ---------------------------------------------------------------------------

PROP_GROUPS = {
    'Passing': ('passing_yards', 'passing_attempts', 'passing_completions', 'passing_tds', 'passing_interceptions'),
    'Rushing': ('rushing_yards', 'rushing_attempts', 'rushing_tds'),
    'Receiving': ('receiving_yards', 'receptions', 'targets', 'receiving_tds'),
}
STAT_LABELS = {'passing_yards': 'Pass yards', 'passing_attempts': 'Pass attempts', 'passing_completions': 'Completions',
               'passing_tds': 'Pass TDs', 'passing_interceptions': 'Interceptions', 'rushing_yards': 'Rush yards',
               'rushing_attempts': 'Carries', 'rushing_tds': 'Rush TDs', 'receiving_yards': 'Receiving yards',
               'receptions': 'Receptions', 'targets': 'Targets', 'receiving_tds': 'Receiving TDs'}


# the market pull's name for a board stat where the two differ (data.draft_projections.PROJECTED_STATS)
MARKET_STAT_NAME = {'rushing_attempts': 'carries', 'passing_attempts': 'attempts'}

# THE POSTED LINE (2026-10-25). A board's 'Mkt <stat>' is the market's IMPLIED MEAN (data/market_devig.py: the vig removed and a
# Poisson mean solved for a count, so a 0.5 interceptions line priced at even money reads 0.69), the right number beside a mean,
# the wrong one beside a median or as "the line". 'Line <stat>' is the line a book actually posted: the sharpest book that posted
# one (this order), so the LINE column is a number that can be bet and a median can be read against it.
LINE_BOOKS = ('Pinnacle', 'DraftKings', 'FanDuel', 'Underdog', 'PrizePicks')
# market stats whose implied mean is a Poisson mean (above the line it came from); a yardage implied mean is the market's median
_MEAN_STATS = {'receptions', 'targets', 'passing_tds', 'rushing_tds', 'receiving_tds', 'passing_interceptions', 'passing_completions',
               'passing_attempts', 'rushing_attempts'}
# A line too low to read a median against: a count's 0.5 line (an anytime TD, an interception, "a catch") is a yes/no bet priced
# far from even money, where the price carries the market's view and the median of a rare count sits near 0 (every one would read
# as a -95% UNDER), and a yardage line under 5 is a special, not the stat's market. Below these (counts: 1.0) a line is left out.
MIN_LEAN_LINE = {'passing_yards': 5.0, 'rushing_yards': 5.0, 'receiving_yards': 5.0}
# A rushing or receiving TD line is an anytime-TD market (0.5) or its "2+ TDs" version (1.5, priced around 1-5% to go over): a
# yes/no bet carried by its price, never a median line, so the props and edges leave it out.
NO_LEAN_STATS = {'rushing_tds', 'receiving_tds'}


def posted_lines(book_lines):
    """One row per player with each market stat's posted line from the sharpest book that posted it (LINE_BOOKS order): a frame with
    'Player' and one column per market stat (the market's own names: 'carries', 'attempts', ...). Empty without book lines.
    `book_lines`: data.odds_projections.market_book_stat_lines (player, market, provider, line)."""
    if book_lines is None or len(book_lines) == 0 or not {'player', 'market', 'provider', 'line'} <= set(book_lines.columns):
        return pd.DataFrame()
    frame = book_lines[['player', 'market', 'provider', 'line']].copy()
    frame['line'] = pd.to_numeric(frame['line'], errors='coerce')
    book = frame['provider'].astype(str).str.split(' (', regex=False).str[0]
    frame['_rank'] = book.map({b: i for i, b in enumerate(LINE_BOOKS)}).fillna(len(LINE_BOOKS))
    frame = frame.dropna(subset=['line']).sort_values(['_rank', 'line'])
    best = frame.drop_duplicates(['player', 'market'], keep='first')
    wide = best.pivot(index='player', columns='market', values='line')
    wide.columns.name = None
    return wide.reset_index().rename(columns={'player': 'Player'})


def long_market(board):
    """The board's priced props, one row per (player, stat): player, team, Pos, stat, model, line, diff, diff_pct, _key. The line is
    the posted one ('Line <stat>', see posted_lines) and the model side the MEDIAN ('Median <stat>'): a line is priced against the
    middle game. A board with only the implied means ('Mkt <stat>') compares like with like instead: the median against a yardage
    mean (a symmetric market, so its median) and the model's mean against a count's Poisson mean. Under the market's own stat names;
    empty when there are none."""
    if board is None or board.empty:
        return pd.DataFrame()
    rows = []
    records = board.to_dict('records')
    for stat in STAT_LABELS:
        name = MARKET_STAT_NAME.get(stat, stat)
        posted, implied = 'Line ' + name, 'Mkt ' + name
        if stat in NO_LEAN_STATS or stat not in board.columns or (posted not in board.columns and implied not in board.columns):
            continue
        for rec in records:
            # no number to read against the line: a stat the model does not project at his position (a receiver's rushing), a player
            # it has out, or no volume at all (his zeros would list as -100% edges)
            if (stat not in ALL_POSITION_STATS.get(rec.get('Pos'), ()) or _num(rec.get('Availability')) <= SIDELINED
                    or not _num(rec.get(stat)) > 0):
                continue
            line = _num(rec.get(posted))
            model = _num(rec.get('Median ' + stat, rec.get(stat)))
            if not np.isfinite(line):
                line = _num(rec.get(implied))
                if stat in _MEAN_STATS:
                    model = _num(rec.get(stat))
            if not np.isfinite(line) or not np.isfinite(model) or line < MIN_LEAN_LINE.get(stat, 1.0 if stat in _MEAN_STATS else 0.0):
                continue
            rows.append({'player': rec['Player'], 'team': rec['Team'], 'Pos': rec['Pos'], 'stat': stat, 'model': float(model),
                         'line': float(line), 'diff': float(model) - line, 'diff_pct': (float(model) - line) / line if line else np.nan,
                         '_key': (rec['Player'], rec['Pos'], rec['Team'])})
    return pd.DataFrame(rows)


def headline_line(rec, position):
    """The LINE column's number for a player: the posted line on his headline stat, else the market's number for it (a yardage implied
    mean is the market's median), NaN with neither."""
    stat = MARKET_STAT_NAME.get(HEADLINE_STAT[position], HEADLINE_STAT[position])
    line = _num(rec.get('Line ' + stat))
    return line if np.isfinite(line) else _num(rec.get('Mkt ' + stat))


def props_table(market, teams, group='All'):
    """The priced props of the game's players, the largest disagreement with the book first: PLAYER, TEAM, MARKET, MODEL, LINE, DIFF %
    (a number) and LEAN. `group`: All / Passing / Rushing / Receiving."""
    if market is None or market.empty:
        return pd.DataFrame()
    g = market[market['team'].isin(teams)].copy()
    if group in PROP_GROUPS:
        g = g[g['stat'].isin(PROP_GROUPS[group])]
    if g.empty:
        return pd.DataFrame()
    g['_gap'] = g['diff_pct'].abs()
    g = g.sort_values('_gap', ascending=False)
    out = pd.DataFrame({'PLAYER': g['player'], 'TEAM': g['team'], 'MARKET': g['stat'].map(lambda x: STAT_LABELS.get(x, x)),
                        'MODEL': g['model'], 'LINE': g['line'], 'DIFF %': g['diff_pct'],
                        'LEAN': g['diff'].map(lambda v: 'OVER' if v > 0 else 'UNDER' if v < 0 else '')})
    out['_key'] = g['_key'].to_numpy()
    return out.reset_index(drop=True)


def styled_props(frame):
    """The props table as a Styler: DIFF % and LEAN shaded green (the model above the line) or red (below), by size."""
    show, blank = _blankable(frame)
    zebra = blend(C['surface_container'], C['surface_container_high'], 0.4)

    def style_row(row):
        base = zebra if int(row.name) % 2 else C['surface_container']
        diff = _num(row.get('DIFF %'))
        side = C['positive'] if np.isfinite(diff) and diff > 0 else C['negative']
        shade = blend(base, side, min(abs(diff) / 0.25, 1.0) * 0.55) if np.isfinite(diff) else base
        styles = []
        for column in show.columns:
            if blank.at[int(row.name), column]:
                styles.append(f'background-color:{base}; color:{base};')
            elif column in ('DIFF %', 'LEAN'):
                styles.append(f'background-color:{shade}; color:#ffffff; font-weight:bold;')
            elif column == 'LINE':
                styles.append(f"background-color:{base}; color:{C['primary']}; font-weight:bold;")
            elif column == 'PLAYER':
                styles.append(f"background-color:{base}; color:{C['on_surface']}; font-weight:bold;")
            else:
                styles.append(f"background-color:{base}; color:{C['on_surface']};")
        return styles

    formats = {'MODEL': '{:.1f}', 'LINE': '{:.1f}', 'DIFF %': '{:+.1%}'}
    return show.style.apply(style_row, axis=1).format(formats)


# ---------------------------------------------------------------------------
# the days of the slate
# ---------------------------------------------------------------------------

def game_day(date):
    """'Sun' for a game's local date ('2026-10-25'), 'TBD' when there is none."""
    try:
        stamp = pd.Timestamp(date)
    except Exception:
        return 'TBD'
    return 'TBD' if pd.isna(stamp) else stamp.strftime('%a')


def day_options(games):
    """[(day, number of games)] in kickoff order, for the day filter."""
    order, counts = [], {}
    for g in games:
        day = game_day(g.get('date'))
        if day not in counts:
            order.append(day)
            counts[day] = 0
        counts[day] += 1
    return [(day, counts[day]) for day in order]


# ---------------------------------------------------------------------------
# the model against the market, and the weather
# ---------------------------------------------------------------------------
DELTA_WARN = 3.0                 # a model number this far from the market's is shaded


def _finite(value):
    return value is not None and np.isfinite(value)


def scoreboard_rows(game, line):
    """The model-against-market table: both teams' points, the home team's spread in a book's sign (negative = favored) and the
    total. -> [{'label', 'model', 'market', 'delta', 'tone'}]; the delta is model minus market, shaded past DELTA_WARN."""
    def fmt(v):
        return f'{v:.1f}' if _finite(v) else '—'

    def sgn(v):
        return f'{v:+.1f}' if _finite(v) else '—'

    def row(label, model, market, signed=False):
        both = _finite(model) and _finite(market)
        gap = model - market if both else np.nan
        return {'label': label, 'model': (sgn if signed else fmt)(model), 'market': (sgn if signed else fmt)(market),
                'delta': f'{gap:+.1f}' if both else '', 'tone': 'warn' if both and abs(gap) >= DELTA_WARN else ''}
    # the nickname ("Steelers"): a full name and the PROJECTED WIN tag overflow the panel's number columns
    away = row(game.get('away_short') or game['away_name'], line['model_away'], line['market_away'])
    home = row(game.get('home_short') or game['home_name'], line['model_home'], line['market_home'])
    # the two team rows say who they are (a logo and the team's colour in the table) and which of them the model has winning
    away['team'], home['team'] = game['away'], game['home']
    ahead = line['model_home'] - line['model_away'] if _finite(line['model_home']) and _finite(line['model_away']) else np.nan
    away['fav'], home['fav'] = bool(_finite(ahead) and ahead < 0), bool(_finite(ahead) and ahead > 0)
    return [away, home,
            row(f"Spread · {game['home']}", line['model_spread'], line['market_spread'], signed=True),
            row('Total', line['model_total'], line['market_total'])]


def game_scoreline(game, points_fit):
    """The model's and the market's score for one game: the points fit's (data/points_fit.py, display only) and the market's implied
    points from its spread and total (nflverse's spread_line is positive when the HOME team is favored), the spread for the home team in
    a book's sign (negative = favored) and the total."""
    out = {'model_home': np.nan, 'model_away': np.nan, 'market_home': np.nan, 'market_away': np.nan,
           'model_spread': np.nan, 'market_spread': np.nan, 'model_total': np.nan, 'market_total': np.nan}
    if points_fit is not None:
        out['model_home'], out['model_away'] = points_fit.game(game['home'], game['away'], bool(game.get('neutral')))
        out['model_spread'] = -(out['model_home'] - out['model_away'])
        out['model_total'] = out['model_home'] + out['model_away']
    spread, total = _num(game.get('spread_line')), _num(game.get('total_line'))
    if np.isfinite(spread):
        out['market_spread'] = -float(spread)
    if np.isfinite(total):
        out['market_total'] = float(total)
        if np.isfinite(spread):
            out['market_home'], out['market_away'] = total / 2.0 + spread / 2.0, total / 2.0 - spread / 2.0
    return out


# the board's own weather bands (ui/tabs/rankings.py's Wind/Temp columns use the same)
WIND_CALM_MAX_MPH, WIND_WINDY_MAX_MPH = 10.0, 20.0
TEMP_FREEZING_MAX_F, TEMP_COLD_MAX_F, TEMP_HOT_MIN_F = 32.0, 50.0, 85.0
_PRECIP_VIEW = {'dry': ('Dry', 'plain'), 'light_rain': ('Light rain', 'wet'), 'heavy_rain': ('Heavy rain', 'wet2'),
                'snow': ('Snow', 'snow')}
_FORECAST_PROVIDERS = {'open-meteo': 'Open-Meteo', 'visual-crossing': 'Visual Crossing', 'nws': 'National Weather Service'}


def _weather_source(source):
    """The weather box's foot line from a GameWeather source ('schedule-recorded', 'open-meteo', 'open-meteo-cache', ...)."""
    source = str(source or '')
    if source.startswith('schedule'):
        return 'Recorded at the game (nflverse schedule)'
    for key, name in _FORECAST_PROVIDERS.items():
        if source.startswith(key):
            return f'Forecast for kickoff ({name})'
    return ''


def weather_view(weather, precip=None):
    """The game's weather as the numbers a person reads, from a data.weather.GameWeather (or None) and the precipitation bucket
    ('dry' / 'light_rain' / 'heavy_rain' / 'snow', forecast only). -> {'state': 'data' | 'indoors' | 'none', 'message', 'cells':
    [{'k', 'v', 'unit', 'sub', 'label', 'tone'}], 'foot'}."""
    if weather is None:
        return {'state': 'none', 'message': 'No weather for this game.', 'cells': [], 'foot': ''}
    if not weather.is_outdoor:
        return {'state': 'indoors', 'message': 'Indoors: the weather does not apply.', 'cells': [], 'foot': ''}
    temp, wind = weather.temp_f, weather.wind_mph
    if temp is None and wind is None and precip is None:
        return {'state': 'none', 'message': 'No forecast for this game yet (forecasts reach about 16 days out).', 'cells': [], 'foot': ''}
    if temp is None:
        temp_label, temp_tone = '', 'plain'
    elif temp <= TEMP_FREEZING_MAX_F:
        temp_label, temp_tone = 'Freezing', 'cold'
    elif temp <= TEMP_COLD_MAX_F:
        temp_label, temp_tone = 'Cold', 'cool'
    elif temp >= TEMP_HOT_MIN_F:
        temp_label, temp_tone = 'Hot', 'hot'
    else:
        temp_label, temp_tone = 'Mild', 'info'
    if wind is None:
        wind_label, wind_tone = '', 'plain'
    elif wind < WIND_CALM_MAX_MPH:
        wind_label, wind_tone = 'Calm', 'plain'
    elif wind < WIND_WINDY_MAX_MPH:
        wind_label, wind_tone = 'Windy', 'warn'
    else:
        wind_label, wind_tone = 'Very windy', 'out'
    precip_word, precip_tone = _PRECIP_VIEW.get(precip, ('—', 'plain'))
    cells = [
        {'k': 'TEMP', 'v': f'{temp:.0f}' if temp is not None else '—', 'unit': '°F', 'sub': '', 'label': temp_label, 'tone': temp_tone},
        {'k': 'PRECIP', 'v': precip_word, 'unit': '', 'sub': 'forecast' if precip else 'no forecast', 'label': '', 'tone': precip_tone},
        {'k': 'WIND', 'v': f'{wind:.0f}' if wind is not None else '—', 'unit': ' mph', 'sub': '', 'label': wind_label, 'tone': wind_tone},
    ]
    return {'state': 'data', 'message': '', 'cells': cells, 'foot': _weather_source(weather.source)}


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def esc(text):
    return html.escape(str(text), quote=True)


_CSS = """
<style>
.gm-card{background:__SURFACE__;border:1px solid __LINE__;border-radius:10px;overflow:hidden;margin:6px 0 14px 0}
.gm-bar{height:4px}
.gm-top{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px 18px;padding:12px 16px 2px 16px}
.gm-teams{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.gm-tm{display:flex;align-items:center;gap:8px;font-weight:800;font-size:21px;line-height:1.15;color:#fff}
.gm-tm img{width:30px;height:30px;object-fit:contain}
.gm-tm small{font-size:10px;font-weight:700;letter-spacing:.1em;color:__MUTED__;border:1px solid __LINE__;border-radius:9999px;padding:1px 7px}
.gm-at{color:__MUTED__;font-weight:700;font-size:16px}
.gm-meta{display:flex;gap:6px;flex-wrap:wrap}
.gm-pill{font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:__MUTED__;background:__HIGH__;border-radius:9999px;padding:3px 10px}
.gm-body{display:grid;grid-template-columns:minmax(300px,1.15fr) minmax(300px,1fr);gap:12px;padding:10px 16px 14px 16px}
.gm-panel{background:__LOW__;border:1px solid __LINE__;border-radius:8px;padding:10px 14px}
.gm-k{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:__MUTED__}
table.gm-mm{width:100%;border-collapse:collapse;margin-top:4px}
.gm-mm th{font-size:10px;font-weight:700;letter-spacing:.1em;color:__MUTED__;text-align:right;padding:2px 6px 4px 6px}
.gm-mm th{font-size:12px}
.gm-mm td{padding:9px 8px;border-top:1px solid __LINE__;text-align:right;font-family:__MONO__;font-size:28px;font-weight:700;color:#fff}
.gm-mm td.gm-l{text-align:left;font-family:__BODY__;font-size:19px;font-weight:800;color:#fff;white-space:nowrap}
.gm-mm td.gm-l img{width:26px;height:26px;object-fit:contain;vertical-align:middle;margin-right:8px}
.gm-mm td.gm-l .gm-sw{display:inline-block;width:6px;height:26px;border-radius:3px;vertical-align:middle;margin-right:8px}
.gm-mm td.gm-l .gm-fav{display:inline-block;margin-left:10px;font-size:10px;font-weight:800;letter-spacing:.1em;border-radius:9999px;padding:2px 9px;color:#fff;vertical-align:middle}
.gm-mm td.gm-d{font-size:18px;color:__MUTED__}
.gm-mm td.gm-d.warn{color:__WARNFG__;background:__WARNBG__;border-radius:4px;font-weight:700}
.gm-mm tr.gm-sep td{border-top:2px solid __LINE__}
.gm-wxrow{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:8px}
.gm-wxrow .gm-k{font-size:13px}
.gm-wxv{font-family:__MONO__;font-size:46px;font-weight:700;color:#fff;line-height:1.05}
.gm-wxv small{font-size:22px;font-weight:600;color:__MUTED__;margin-left:2px}
.gm-wxs{font-size:17px;color:__MUTED__;min-height:24px;margin-top:2px}
.gm-chip{display:inline-block;margin-top:6px;font-size:19px;font-weight:700;border-radius:9999px;padding:3px 16px}
.gm-wxfoot{margin-top:12px;font-size:14px;color:__MUTED__}
.gm-wxmsg{margin-top:10px;font-size:18px;color:__MUTED__}
.gm-th{display:flex;align-items:baseline;gap:12px;margin:14px 0 6px 0;padding-left:10px;border-left:5px solid __LINE__}
.gm-th b{font-size:17px;color:#fff;font-weight:800}
.gm-th span{font-size:13px;color:__MUTED__}
.gm-grid2{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:2px 0 10px 0}
.gm-cell{display:flex;flex-direction:column;border:1px solid __LINE__;border-radius:10px;overflow:hidden;background:__SURFACE__}
.gm-banner{display:flex;align-items:center;gap:10px;padding:10px 14px}
.gm-banner img{width:34px;height:34px;object-fit:contain}
.gm-bn{font-weight:800;font-size:18px;color:#fff;line-height:1.1}
.gm-bs{font-size:11px;color:#ffffffd0;letter-spacing:.08em;text-transform:uppercase}
.gm-status{margin-left:auto;font-size:11px;font-weight:700;letter-spacing:.06em;border-radius:9999px;padding:2px 10px;white-space:nowrap}
.gm-chips{display:flex;gap:1px;background:__LINE__}
.gm-chipc{flex:1 1 0;min-width:0;background:__SURFACE__;padding:6px 6px;text-align:center}
.gm-chipc b{display:block;font-size:10px;letter-spacing:.08em;color:__MUTED__;font-weight:700}
.gm-chipc span{font-family:__MONO__;font-size:16px;font-weight:700;color:#fff}
.gm-av{padding:8px 14px 10px 14px;flex:1}
.gm-row{display:grid;grid-template-columns:max-content 1fr;gap:1px 10px;align-items:baseline;padding:6px 0;border-top:1px solid __LINE__}
.gm-b{font-size:10px;font-weight:800;letter-spacing:.07em;border-radius:4px;padding:2px 7px;text-transform:uppercase;white-space:nowrap}
.gm-n{font-weight:700;color:#fff;font-size:14px}
.gm-n i{font-style:normal;font-weight:700;font-size:11px;color:__MUTED__;margin-left:7px}
.gm-d{grid-column:2;font-size:12px;color:__MUTED__;line-height:1.35}
.gm-ok{border-top:1px solid __LINE__;padding:8px 0 2px 0;font-size:13px;color:__MUTED__}
@media (max-width:900px){.gm-body,.gm-grid2{grid-template-columns:1fr}}
</style>
"""


def css():
    """The stylesheet of the game view (inject once per render)."""
    sheet = _CSS
    for token, value in {'__SURFACE__': C['surface_container'], '__LOW__': C['surface_container_low'], '__HIGH__': C['surface_container_high'],
                         '__LINE__': C['outline_variant'], '__MUTED__': C['on_surface_variant'], '__MONO__': F['mono'],
                         '__BODY__': F['body'], '__WARNFG__': TONES['warn'][0], '__WARNBG__': TONES['warn'][1]}.items():
        sheet = sheet.replace(token, value)
    return sheet.replace('\n', '')


def _color(colors, team):
    value = (colors or {}).get(team)
    return value if value and str(value).startswith('#') and len(str(value)) == 7 else C['surface_container_high']


def _logo(logos, team):
    url = (logos or {}).get(team)
    return f'<img src="{esc(url)}" alt="">' if url else ''


def _badge(label, tone):
    fg, bg = TONES.get(tone, TONES['plain'])
    return f'<span class="gm-b" style="color:{fg};background:{bg}">{esc(label)}</span>'


def weather_html(view):
    if view['state'] != 'data':
        return f'<div class="gm-panel"><div class="gm-k">WEATHER</div><div class="gm-wxmsg">{esc(view["message"])}</div></div>'
    cells = []
    for cell in view['cells']:
        chip = ''
        if cell['label']:
            fg, bg = TONES.get(cell['tone'], TONES['plain'])
            chip = f'<span class="gm-chip" style="color:{fg};background:{bg}">{esc(cell["label"])}</span>'
        cells.append(f'<div><div class="gm-k">{cell["k"]}</div><div class="gm-wxv">{esc(cell["v"])}<small>{esc(cell["unit"])}</small></div>'
                     f'<div class="gm-wxs">{esc(cell["sub"])}</div>{chip}</div>')
    foot = f'<div class="gm-wxfoot">{esc(view["foot"])}</div>' if view['foot'] else ''
    return f'<div class="gm-panel"><div class="gm-k">WEATHER</div><div class="gm-wxrow">{"".join(cells)}</div>{foot}</div>'


def scoreboard_html(rows, colors=None, logos=None):
    """The model-against-market table. A team's row carries its colour bar and logo, and the team the model has winning is tinted in its
    own colour with a PROJECTED WIN tag, so who the model likes reads at a glance."""
    body = []
    for i, r in enumerate(rows):
        delta_class = f'gm-d {r["tone"]}'.strip()
        label, tint = esc(r['label']), ''
        team = r.get('team')
        if team:
            color = _color(colors, team)
            label = f'<span class="gm-sw" style="background:{color}"></span>{_logo(logos, team)}{label}'
            if r.get('fav'):
                label += f'<span class="gm-fav" style="background:{color}">PROJECTED WIN</span>'
                tint = f' style="background:{color}2B"'
        body.append(f'<tr{" class=gm-sep" if i == 2 else ""}{tint}><td class="gm-l">{label}</td><td>{esc(r["model"])}</td>'
                    f'<td>{esc(r["market"])}</td><td class="{delta_class}">{esc(r["delta"])}</td></tr>')
    return ('<div class="gm-panel"><div class="gm-k">MODEL VS MARKET</div><table class="gm-mm"><thead><tr><th></th><th>MODEL</th>'
            f'<th>MARKET</th><th>MODEL − MKT</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def header_html(game, rows, weather, kickoff='', colors=None, logos=None):
    """The game's header card: the matchup (logos, sides, kickoff), the model-against-market table and the weather box."""
    away, home = game['away'], game['home']
    bar = f'linear-gradient(90deg,{_color(colors, away)} 0 50%,{_color(colors, home)} 50% 100%)'
    side = 'NEUTRAL' if game.get('neutral') else 'HOME'
    pills = ''.join(f'<span class="gm-pill">{esc(text)}</span>'
                    for text in (kickoff, 'Neutral site' if game.get('neutral') else '', game.get('venue') or '') if text)
    return (f'{css()}<div class="gm-card"><div class="gm-bar" style="background:{bar}"></div><div class="gm-top"><div class="gm-teams">'
            f'<div class="gm-tm">{_logo(logos, away)}{esc(game["away_name"])}<small>AWAY</small></div>'
            f'<div class="gm-at">{"vs" if game.get("neutral") else "@"}</div>'
            f'<div class="gm-tm">{_logo(logos, home)}{esc(game["home_name"])}<small>{side}</small></div></div><div class="gm-meta">{pills}</div></div>'
            f'<div class="gm-body">{scoreboard_html(rows, colors, logos)}{weather_html(weather)}</div></div>')


def _status_pill(items):
    """The banner's status pill: HEALTHY, or the counts of who is out and who is flagged."""
    out = sum(1 for i in items if i['tone'] in ('out', 'assumed'))
    half = sum(1 for i in items if i['tone'] == 'half')
    flagged = sum(1 for i in items if i['tone'] in ('warn', 'note'))
    parts = [f'{out} out'] * bool(out) + [f'{half} limited'] * bool(half) + [f'{flagged} flagged'] * bool(flagged)
    if not parts:
        fg, bg = TONES['info']
        return f'<span class="gm-status" style="color:{fg};background:{bg}">HEALTHY</span>'
    tone = 'out' if out else 'warn'
    fg, bg = TONES[tone]
    return f'<span class="gm-status" style="color:{fg};background:{bg}">{" · ".join(parts).upper()}</span>'


def _item_html(item):
    also = f' · also: {", ".join(item["also"])}' if item['also'] else ''
    return (f'<div class="gm-row">{_badge(item["label"], item["tone"])}<span class="gm-n">{esc(item["player"])}<i>{esc(item["position"])}</i></span>'
            f'<span class="gm-d">{esc(item["source"])} · {esc(item["detail"])}{esc(also)}</span></div>')


def team_block_html(block, colors=None, logos=None):
    """One team's cell: a banner in its colours (name, side, status), its projected totals, and its availability box."""
    team = block['team']
    color = _color(colors, team)
    banner = f'linear-gradient(90deg,{color}D9 0%,{color}66 42%,{C["surface_container"]} 100%)'
    chips = ''.join(f'<div class="gm-chipc"><b>{esc(label)}</b><span>{"—" if not _finite(value) else f"{value:.0f}"}</span></div>'
                    for label, value in block['chips'])
    items = block['items']
    if items:
        rows = ''.join(_item_html(i) for i in items)
        if block['hidden']:
            rows += f'<div class="gm-ok">+ {block["hidden"]} more with little role out (their rows say so below)</div>'
    else:
        rows = ('<div class="gm-ok">No one flagged: nothing in the injury reports or your rulings says a player here is out or in doubt'
                + (f' ({block["hidden"]} reserves with little role are out).' if block['hidden'] else '.') + '</div>')
    return (f'<div class="gm-cell"><div class="gm-banner" style="background:{banner}">{_logo(logos, team)}<div><div class="gm-bn">{esc(block["name"])}</div>'
            f'<div class="gm-bs">{esc(block["side"])}</div></div>{_status_pill(items)}</div><div class="gm-chips">{chips}</div>'
            f'<div class="gm-av"><div class="gm-k">AVAILABILITY</div>{rows}</div></div>')


def table_header_html(team, name, count, colors=None, label='', view='Average'):
    """The line above a team's player table: its name beside a bar in its colour, how many players (of the position `label`, when one
    is given)."""
    what = f'{label} · {int(count)}' if label else f'{int(count)} players'
    return (f'<div class="gm-th" style="border-left-color:{_color(colors, team)}"><b>{esc(name)}</b>'
            f'<span>{esc(what)} · {"medians" if view == "Median" else "averages"}</span></div>')


def teams_html(blocks, colors=None, logos=None):
    """Both teams' cells in ONE grid, so the two availability boxes are the same height and the player tables below start on the same line."""
    return f'<div class="gm-grid2">{"".join(team_block_html(b, colors, logos) for b in blocks)}</div>'
