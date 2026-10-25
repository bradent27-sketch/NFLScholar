"""
The Weekly Rankings GAMES view (2026-10-23, user: "this is the main difference between the two ... I have really enjoyed how it works
in the CFB app. Try to replicate this very closely"): one game at a time - a header with the model's score against the market's and the
weather, each team's banner, projected totals and AVAILABILITY box, both teams' players side by side by position, and the game's priced
props. The widget half of ui/games_view.py, a port of CFB Scholar's _render_games. Everything comes from the built board; nothing here
rebuilds it. Selecting a row opens the projection dialog (the tab opens it; this view only says which player).

Also the phone export (2026-10-25, CFB's _render_export and _export_*_html): the games report and the board as a PDF (and HTML), laid
out by ui/export_view.py from the same numbers this view draws (game_data).
"""
import os
import re

import pandas as pd
import streamlit as st

from data.game_slate import load_slate, team_display_name
from data.points_fit import week_points_fit
from data.weekly_median import attach_medians
from ui import export_view, games_view

_GAME_PICK_KEY = 'wr_game_pick'
_GAME_TABLE_MAX_ROWS = 16            # rows of a team's player table shown before it scrolls
_POSITION_ORDER = ('QB', 'RB', 'WR', 'TE')
# the widgets of this view that carry a value worth keeping across the BOARD/GAMES switch, with the value each starts at
_PERSISTED_WIDGETS = {'wr_game_day': 'All', _GAME_PICK_KEY: None, 'wr_game_pos': 'ALL', 'wr_game_rows': 'With a role',
                      'wr_game_props_group': 'All', 'weekly_rank_projection_view': 'Average'}


def keep_widget_state(keys=_PERSISTED_WIDGETS):
    """Streamlit forgets a widget's value on a run that does not draw it, and only the open view is drawn: switching to BOARD and
    back reset the day, the game and the filters. Writing a value back keeps it as an ordinary session entry, which survives; a
    widget is then created WITHOUT a `default`, from the entry (a first run gets its starting value). Call on every run."""
    for key, start in keys.items():
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
        elif start is not None:
            st.session_state[key] = start


def _games_table_height(rows):
    """A data grid's height for `rows` rows: its 36 px header, 35 px rows and its own 1 px border, nothing left over below the last
    row (a spare few pixels showed as a strip under the table)."""
    return 38 + 35 * min(max(int(rows), 1), _GAME_TABLE_MAX_ROWS)


def _game_column_config(frame):
    """The grid's columns for a player table: the PLAYER column pinned so the names stay put while the stats scroll, fixed widths (a
    stat column fits its header), and a tooltip on MATCHUP, HEALTH, PTS and LINE."""
    config = {}
    for column in [c for c in frame.columns if not c.startswith('_')]:
        width = games_view.COLUMN_WIDTHS.get(column) or games_view.STAT_WIDTHS.get(column, games_view.STAT_COLUMN_WIDTH)
        help_text = {'MATCHUP': "The opponent's defense against this position, 1-100: 100 = allows the most fantasy points (the "
                                "softest matchup), 1 = the fewest (the board's Def Rank, as a percentile).",
                     'HEALTH': 'Blank = nothing in the injury reports or your rulings says he is out or in doubt. Red = projected out, '
                               'amber = in doubt or limited.',
                     'PTS': "This app's projected fantasy points (the board's Model Proj Pts).",
                     'LINE': "The market's line on his headline stat: passing yards (QB), rushing yards (RB), receiving yards "
                             "(WR/TE)."}.get(column)
        config[column] = st.column_config.Column(column, width=width, pinned=(column == 'PLAYER') or None, help=help_text)
    return config


def _step_game(delta, ids):
    """on_click of the previous / next game buttons: move the game picker by `delta` within `ids`."""
    current = st.session_state.get(_GAME_PICK_KEY)
    if current in ids:
        st.session_state[_GAME_PICK_KEY] = ids[min(max(ids.index(current) + delta, 0), len(ids) - 1)]


_CLOCK = re.compile(r'(\d{1,2}):(\d{2})\s*([AP]M)', re.IGNORECASE)


def kickoff_start(date, kickoff, time_tbd=False):
    """The kickoff as an Eastern Timestamp from the slate's local 'Date' and its 'Kickoff' text ('Sun 1:00 PM EST'; nflverse's game time
    is Eastern): midnight on its day when the time is not set (the export's TBA), None without a date."""
    try:
        day = pd.Timestamp(date)
    except Exception:
        return None
    if pd.isna(day):
        return None
    day = day.normalize()
    match = _CLOCK.search(str(kickoff or ''))
    if not time_tbd and match:
        hour = int(match.group(1)) % 12 + (12 if match.group(3).upper() == 'PM' else 0)
        day = day + pd.Timedelta(hours=hour, minutes=int(match.group(2)))
    return day.tz_localize(export_view.EASTERN)


def game_list(year, week, board):
    """The week's games with a team on the board, in kickoff order: [{'game_id', 'home', 'away', 'home_name', 'away_name', 'neutral',
    'date', 'kickoff', 'start' (Eastern Timestamp), 'venue', 'spread_line', 'total_line', 'teams', 'colors', 'logos'}]."""
    slate, _err = load_slate(year, week)
    if slate is None or slate.empty or board is None or board.empty:
        return []
    on_board = set(board['Team'].astype(str))
    games = []
    for rec in slate.to_dict('records'):
        home, away = str(rec.get('Home') or ''), str(rec.get('Away') or '')
        teams = [t for t in (away, home) if t in on_board]
        if not teams:
            continue
        games.append({'game_id': rec.get('Game Id') or f'{away}@{home}', 'home': home, 'away': away,
                      'home_name': team_display_name(home), 'away_name': team_display_name(away),
                      'home_short': team_display_name(home).split()[-1], 'away_short': team_display_name(away).split()[-1],
                      'neutral': bool(rec.get('Neutral Site')), 'date': rec.get('Date'), 'kickoff': rec.get('Kickoff') or '',
                      'start': kickoff_start(rec.get('Date'), rec.get('Kickoff'), bool(rec.get('Time TBD'))),
                      'venue': rec.get('Venue') or '', 'spread_line': rec.get('Spread Line'), 'total_line': rec.get('Total Line'),
                      'teams': teams,
                      'colors': {away: rec.get('Away Color'), home: rec.get('Home Color')},
                      'logos': {away: rec.get('Away Logo'), home: rec.get('Home Logo')}})
    return games


def _game_label(game):
    return f"{game['away_name']} {'vs' if game['neutral'] else '@'} {game['home_name']}" + (
        f" · {game['kickoff']}" if game['kickoff'] else '')


def game_data(game, board, meta, year, week, weather_by_team, precip_by_team, view='Average', points_fit=None):
    """One game's numbers, before any HTML: {'line' (games_view.game_scoreline), 'rows' (the model-against-market table), 'weather',
    'blocks' (each team's name, side, totals and availability, away first)}. The GAMES view and the phone export both draw from this.
    `points_fit`: the week's data.points_fit fit (read here when not given)."""
    points_fit = week_points_fit(year, week) if points_fit is None else points_fit
    line = games_view.game_scoreline(game, points_fit)
    weather = games_view.weather_view((weather_by_team or {}).get(game['home']), (precip_by_team or {}).get(game['home']))
    explanations = (meta or {}).get('explanations') or {}
    ledger = (meta or {}).get('vacancy_ledger') or []
    blocks = []
    for team in game['teams']:
        group = board[board['Team'] == team]
        items, hidden = games_view.availability_items(group, explanations, ledger)
        side = ('Home' if team == game['home'] else 'Away') + (' · neutral site' if game['neutral'] else '')
        blocks.append({'team': team, 'name': team_display_name(team), 'side': side, 'chips': games_view.team_chips(group, view),
                       'items': items, 'hidden': hidden})
    return {'line': line, 'rows': games_view.scoreboard_rows(game, line), 'weather': weather, 'blocks': blocks}


def render(board, meta, year, week, weather_by_team, precip_by_team, select_player, team_filter=None, view='Average'):
    """The GAMES view. `board`: the built board with 'Matchup Pct' and any 'Mkt <stat>' columns (rankings._games_board); `meta`: its
    meta (explanations, vacancy ledger); `select_player(key)`: what a row click calls with (Player, Pos, Team); `team_filter`: the
    board's matchup filter, as a set of teams (None = every game); `view`: Average or Median for the player tables and team totals
    (the props table always compares the median with the line)."""
    games = game_list(year, week, board)
    if team_filter:
        games = [g for g in games if set(g['teams']) & set(team_filter)]
    if not games:
        st.caption('No games for the current filters.')
        return

    # THE CONTROLS: a day filter, the game picker, previous / next (clicking through the slate as the week goes on)
    days = games_view.day_options(games)
    counts = dict(days)
    c_day, c_pick, c_prev, c_next = st.columns([3.4, 4.6, 0.7, 0.7], vertical_alignment='center')
    with c_day:
        if st.session_state.get('wr_game_day') not in ['All', *counts]:       # a day the slate no longer has (the filter)
            st.session_state['wr_game_day'] = 'All'
        day = st.segmented_control('Day', ['All'] + [d for d, _n in days], key='wr_game_day', label_visibility='collapsed',
                                   format_func=lambda d: f'All · {len(games)}' if d == 'All' else f'{d} · {counts[d]}')
    if day not in counts:                                         # 'All', or nothing picked
        day = 'All'
    shown = games if day == 'All' else [g for g in games if games_view.game_day(g['date']) == day]
    ids = [g['game_id'] for g in shown]
    labels = {g['game_id']: _game_label(g) for g in shown}
    if st.session_state.get(_GAME_PICK_KEY) not in ids:
        st.session_state[_GAME_PICK_KEY] = ids[0]
    with c_pick:
        chosen = st.selectbox('Game', ids, format_func=labels.get, key=_GAME_PICK_KEY, label_visibility='collapsed')
    with c_prev:
        st.button('◀', key='wr_game_prev', on_click=_step_game, args=(-1, ids), help='Previous game',
                  disabled=ids.index(chosen) == 0, width='stretch')
    with c_next:
        st.button('▶', key='wr_game_next', on_click=_step_game, args=(1, ids), help='Next game',
                  disabled=ids.index(chosen) == len(ids) - 1, width='stretch')
    game = next(g for g in shown if g['game_id'] == chosen)
    colors, logos = game['colors'], game['logos']

    # THE HEADER CARD: the matchup, the model against the market in one small table, and the weather as numbers
    data = game_data(game, board, meta, year, week, weather_by_team, precip_by_team, view)
    st.markdown(games_view.header_html(game, data['rows'], data['weather'], game['kickoff'], colors, logos), unsafe_allow_html=True)

    # EACH TEAM'S BANNER, TOTALS AND AVAILABILITY BOX, in one grid so the two boxes are the same height and the tables below start on
    # the same line, however many players are out on either side
    explanations = (meta or {}).get('explanations') or {}
    st.markdown(games_view.teams_html(data['blocks'], colors, logos), unsafe_allow_html=True)

    # THE PLAYERS: every QB/RB/WR/TE on the board for each team, blank health = healthy
    k1, k2, _spare = st.columns([4, 3, 5], vertical_alignment='center')
    with k1:
        position = st.segmented_control('Position', ['ALL', *_POSITION_ORDER], key='wr_game_pos',
                                        label_visibility='collapsed') or 'ALL'
    with k2:
        # WITH A ROLE is the starting view: the players a projection is for; All players is one click away
        mode = st.segmented_control('Players', ['All players', 'With a role'], key='wr_game_rows',
                                    label_visibility='collapsed') or 'With a role'

    def _on_select_for(key, frame):
        def _on_select():
            rows = st.session_state.get(key, {}).get('selection', {}).get('rows', [])
            if rows and rows[0] < len(frame):
                select_player(frame.iloc[rows[0]].get('_key'))
        return _on_select

    board_key = f"{year}_{week}_{len(board)}"
    # BOTH TEAMS SIDE BY SIDE: a row of two tables per position, the away team's on the left and the home team's on the right, each
    # with only the stats its position has, the columns as narrow as their headers allow
    for pos in (_POSITION_ORDER if position == 'ALL' else (position,)):
        frames = {team: games_view.position_table(board[board['Team'] == team], pos, role_only=(mode == 'With a role'),
                                                  explanations=explanations, view=view) for team in game['teams']}
        if all(frame.empty for frame in frames.values()):
            if position != 'ALL':
                st.caption(f'No {games_view.POSITION_LABELS[pos].lower()} for this filter.')
            continue
        for team, column in zip(game['teams'], st.columns(len(game['teams']))):
            with column:
                frame = frames[team]
                if frame.empty:
                    st.caption(f'{team_display_name(team)}: no {games_view.POSITION_LABELS[pos].lower()} for this filter.')
                    continue
                st.markdown(games_view.table_header_html(team, team_display_name(team), len(frame), colors, label=pos, view=view),
                            unsafe_allow_html=True)
                key = f"wr_game_{game['game_id']}_{team}_{pos}_{mode}_{view}_{board_key}"
                st.dataframe(games_view.styled_table(frame), width='stretch', hide_index=True, height=_games_table_height(len(frame)),
                             on_select=_on_select_for(key, frame), selection_mode='single-row', key=key,
                             column_config=_game_column_config(frame))

    # THE PROPS in the game, the largest disagreement with the book first
    st.markdown("**Priced props in this game** (the model's median against the line, largest disagreement first)")
    group = st.segmented_control('Props', ['All', 'Passing', 'Rushing', 'Receiving'], key='wr_game_props_group',
                                 label_visibility='collapsed') or 'All'
    props = games_view.props_table(games_view.long_market(board), game['teams'], group)
    if props.empty:
        st.caption('No market lines are matched to this game: pull them in 📡 Live data pulls → Market props (sportsbooks post only '
                   'the coming slate).' if group == 'All' else f'No {group.lower()} lines are matched to this game.')
    else:
        key = f"wr_game_props_{game['game_id']}_{group}_{board_key}"
        st.dataframe(games_view.styled_props(props), width='stretch', hide_index=True, height=_games_table_height(len(props)),
                     on_select=_on_select_for(key, props), selection_mode='single-row', key=key)


def games_board(model_df, market_df, matchup_difficulty, attach_by_name):
    """The board the GAMES view reads: the built board with 'Matchup Pct' (0-100, 100 = the softest matchup, the Def Rank
    percentile), the medians beside the means (data/weekly_median.py), and when a market pull is staged the market's implied mean
    per stat as 'Mkt <stat>' and the posted line as 'Line <stat>' (games_view.posted_lines: what the props compare the median with)."""
    board = attach_medians(model_df)
    pct = [(matchup_difficulty.get((str(p).upper(), str(o))) or (None, None))[1]
           for p, o in zip(board['Pos'], board['Opponent'])]
    board['Matchup Pct'] = pd.to_numeric(pd.Series(pct, index=board.index, dtype=object), errors='coerce')
    if market_df is not None and not market_df.empty:
        stats = [s for s in market_df.attrs.get('consensus_stats', []) if s in market_df.columns]
        if stats:
            board = attach_by_name(board, market_df, stats, 'Mkt ')
        posted = games_view.posted_lines(market_df.attrs.get('book_lines'))
        if not posted.empty:
            board = attach_by_name(board, posted, [c for c in posted.columns if c != 'Player'], 'Line ')
    return board


# ---------------------------------------------------------------------------
# the phone export (CFB Scholar's _render_export / _export_games_html / _export_board_html)
# ---------------------------------------------------------------------------

_EXPORT_KEY = 'wr_export_files'
_SCORING_TAG = {'Full PPR': 'ppr', 'Half-PPR': 'half', 'Standard': 'std'}


def _kickoff_label(start):
    """'Sun 1:00 PM ET' for an Eastern kickoff, '' with none."""
    local = export_view.eastern(start)
    if local is None:
        return ''
    return f"{local.strftime('%a')} {export_view.time_label(local)}" + ('' if export_view.is_tba(local) else ' ET')


def export_subtitle(meta, year, week, scoring, view, now=None):
    now = pd.Timestamp.now(tz=export_view.EASTERN) if now is None else now
    stamp = f"{now.strftime('%a %b')} {now.day}, {now.strftime('%I:%M %p').lstrip('0')} ET"
    as_of = (meta or {}).get('as_of_week', week)
    return (f"{year} · {scoring} · built from weeks before {as_of} · {'averages' if view == 'Average' else 'medians'} · "
            f"exported {stamp}")


def export_edge_rows(board, games):
    """The edges page's rows: every priced prop of the games' players (ui/export_view.py): who, his game and its kickoff, the model's
    median against the line and the lean. No lean record or line movement is kept yet, so those columns stay off the page."""
    market = games_view.long_market(board)
    if market.empty:
        return []
    side_of = {}
    for g in games:
        for team in g['teams']:
            side_of[team] = (g['home'] if team == g['away'] else g['away'], team == g['away'], _kickoff_label(g['start']))
    group_of = {s: name for name, stats in export_view.EDGE_GROUPS for s in stats}
    rows = []
    for rec in market[market['team'].isin(side_of)].to_dict('records'):
        group = group_of.get(rec['stat'])
        diff = rec.get('diff')
        if group is None or not pd.notna(rec.get('diff_pct')) or not diff:
            continue
        opponent, away, start = side_of[rec['team']]
        rows.append({'player': rec['player'], 'team': rec['team'], 'opponent': opponent, 'away': away, 'start': start, 'group': group,
                     'market': games_view.STAT_LABELS.get(rec['stat'], rec['stat']), 'model': rec['model'], 'line': rec['line'],
                     'diff_pct': float(rec['diff_pct']), 'lean': 'OVER' if diff > 0 else 'UNDER'})
    return rows


def priced_players(board):
    """How many players on the board carry at least one market line ('Mkt <stat>')."""
    columns = [c for c in board.columns if str(c).startswith('Mkt ')]
    if not columns:
        return 0
    return int(board[columns].apply(pd.to_numeric, errors='coerce').notna().any(axis=1).sum())


def export_games_html(board, meta, year, week, scoring, view, weather_by_team, precip_by_team, selection='all', freshness=None):
    """The games report (ui/export_view.py): the cover, the edges page, the slate and every game of `selection` (the whole slate by
    default), each with its numbers gathered the way the GAMES view gathers them (game_data), the 'With a role' rows of each position,
    the away team's before the home team's. The report lays them out."""
    games = game_list(year, week, board)
    keep = {e['game']['game_id'] for e in export_view.select_games([{'game': g} for g in games], selection)}
    games = [g for g in games if g['game_id'] in keep]
    colors = {t: c for g in games for t, c in g['colors'].items()}
    logos = export_view.embed_logos({t: u for g in games for t, u in g['logos'].items()}, [t for g in games for t in g['teams']])
    explanations = (meta or {}).get('explanations') or {}
    points_fit = week_points_fit(year, week)
    market = games_view.long_market(board)
    entries = []
    for game in games:
        data = game_data(game, board, meta, year, week, weather_by_team, precip_by_team, view, points_fit)
        teams = []
        for block in data['blocks']:
            group = board[board['Team'] == block['team']]
            tables = [(games_view.POSITION_LABELS[pos],
                       games_view.position_table(group, pos, role_only=True, explanations=explanations, view=view))
                      for pos in _POSITION_ORDER]
            teams.append({**block, 'tables': tables})
        entries.append({'game': game, 'line': data['line'], 'rows': data['rows'], 'weather': data['weather'], 'teams': teams,
                        'props': games_view.props_table(market, game['teams'], 'All')})
    title = f"Week {week} games"
    return export_view.games_report(
        title, f'{len(games)} games · ' + export_subtitle(meta, year, week, scoring, view), entries, colors, logos, view, week,
        freshness=freshness, priced=priced_players(board), on_board=len(board), edge_rows=export_edge_rows(board, games),
        selection=export_view.selection_label(selection))


def export_board_html(board, meta, year, week, scoring, view, lined_only=False):
    games = game_list(year, week, board)
    colors = {t: c for g in games for t, c in g['colors'].items()}
    game_of = {}
    for g in games:
        game_of[g['away']] = (g['home'], True)
        game_of[g['home']] = (g['away'], False)
    return export_view.board_report(f"Week {week} board", export_subtitle(meta, year, week, scoring, view), board, view, colors,
                                    game_of, lined_only)


def market_freshness(market_df):
    """The cover's lines table from the staged market pull (its attrs: lines per book and the pull's time); [] with no pull."""
    if market_df is None or not hasattr(market_df, 'attrs'):
        return []
    lines = market_df.attrs.get('lines_by_book') or {}
    return export_view.freshness_rows(lines, market_df.attrs.get('fetched_at')) if lines else []


def render_export(model_df, board_fn, meta, year, week, scoring, view, weather_by_team, precip_by_team, freshness=None):
    """Export for phone: the games report (the whole slate, or one day or Sunday window) and the bare board as a PDF (and HTML) to
    download, kept in exports/ and, if chosen, copied to a OneDrive folder so a phone has it. Built on a click, not on every rerun, and
    kept for the board it was built from. `board_fn()`: the GAMES view's board (games_board), only built on a click."""
    if model_df is None or model_df.empty:
        return
    build_key = (year, week, scoring, view, len(model_df),
                 round(float(pd.to_numeric(model_df['Model Proj Pts'], errors='coerce').sum()), 2))
    options = export_view.selection_options([{'game': g} for g in game_list(year, week, model_df)])
    keys = [k for k, _label in options]
    labels = dict(options)
    if st.session_state.get('wr_export_sel') not in keys:
        st.session_state['wr_export_sel'] = 'all'
    if 'wr_export_copy' not in st.session_state:                  # the saved choice is the source of truth (a widget the page skipped forgets)
        saved = export_view.load_settings()
        st.session_state['wr_export_copy'], st.session_state['wr_export_folder'] = ('On' if saved['copy'] else 'Off'), saved['folder']
    with st.popover('Export for phone'):
        st.caption('A PDF a phone can open anywhere (and an HTML copy). Saved in the exports folder too.')
        selection = st.segmented_control('Games', keys, format_func=labels.get, key='wr_export_sel',
                                         label_visibility='collapsed') or 'all'
        c1, c2 = st.columns(2)
        build = None
        with c1:
            if st.button('Games report', key='wr_export_games', width='stretch',
                         help='The edges, then every game of the selection: model vs market, weather, both teams, availability, each '
                              'position with a role and its full stat line, priced props. Full slate by default.'):
                build = 'games'
        with c2:
            if st.button('Board', key='wr_export_board', width='stretch',
                         help='Bare bones: every player with a role, his points and projected stat line, by position.'):
                build = 'board'
        st.caption('Board shows')
        lined = (st.segmented_control('Board shows', ['With a role', 'Lined only'], key='wr_export_board_mode',
                                      label_visibility='collapsed',
                                      help='With a role: every player with a role, most points first. Lined only: every player a book has '
                                           'a line on, largest gap between the model and the line first.') or 'With a role') == 'Lined only'
        st.caption('Also copy the files to a folder (OneDrive), so they reach your phone')
        copy = (st.segmented_control('Copy', ['On', 'Off'], key='wr_export_copy', label_visibility='collapsed',
                                     help='The copy overwrites the last export of the same name.') or 'Off') == 'On'
        folder = st.text_input('Folder', key='wr_export_folder', disabled=not copy, label_visibility='collapsed',
                               placeholder='C:/Users/you/OneDrive/NFL Scholar')
        settings = {'copy': copy, 'folder': folder}
        if st.session_state.get('_wr_export_settings') != settings:
            export_view.save_settings(settings)
            st.session_state['_wr_export_settings'] = settings
        if build:
            kind = build + ('_' + selection if build == 'games' and selection != 'all' else '') + ('_lined' if build == 'board' and lined else '')
            stem = f"NFL_{int(year)}_week{int(week):02d}_{_SCORING_TAG.get(scoring, 'ppr')}_{kind}"
            with st.spinner('Building the export...'):
                board = board_fn()
                document = (export_games_html(board, meta, year, week, scoring, view, weather_by_team, precip_by_team, selection,
                                              freshness) if build == 'games'
                            else export_board_html(board, meta, year, week, scoring, view, lined))
                files = export_view.save(document, stem)
                copied = export_view.copy_files(files, folder.strip()) if copy and folder.strip() else []
            store = st.session_state.setdefault(_EXPORT_KEY, {})
            store[build] = {'build_key': build_key, 'copied': copied, 'copy_failed': bool(copy and folder.strip() and not copied), **files}
        for kind, label in (('games', 'Games report'), ('board', 'Board')):
            entry = (st.session_state.get(_EXPORT_KEY) or {}).get(kind)
            if not entry or entry.get('build_key') != build_key:
                continue
            for ext, mime in (('pdf', 'application/pdf'), ('html', 'text/html')):
                path = entry.get(ext)
                if not path:
                    if ext == 'pdf':
                        st.caption(f'{label}: no PDF (no Chrome or Edge found to print it); the HTML is below.')
                    continue
                try:
                    with open(path, 'rb') as handle:
                        data = handle.read()
                except OSError:
                    continue
                st.download_button(f'{label} · {ext.upper()}', data, file_name=os.path.basename(path), mime=mime,
                                   key=f'wr_export_dl_{kind}_{ext}', on_click='ignore', width='stretch')
            st.caption(f'Saved: {entry["pdf"] or entry["html"]}')
            if entry.get('copied'):
                st.caption(f'Copied to: {os.path.dirname(entry["copied"][0])}')
            elif entry.get('copy_failed'):
                st.caption('Could not copy to that folder: check the path.')
