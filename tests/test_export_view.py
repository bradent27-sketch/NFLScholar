"""ui/export_view.py: the games report and the bare board as HTML (and PDF) for a phone - CFB Scholar's export ported (2026-10-25)."""
import os

import numpy as np
import pandas as pd

from ui import export_view as ev
from ui import games_view

COLORS = {'KC': '#e31837', 'BUF': '#00338d', 'PIT': '#ffb612', 'NO': '#d3bc8d'}


def _et(text):
    return pd.Timestamp(text, tz=ev.EASTERN)


def _game(game_id, away, home, start, neutral=False):
    return {'game_id': game_id, 'away': away, 'home': home, 'neutral': neutral, 'start': start, 'teams': [away, home],
            'away_name': f'{away} team', 'home_name': f'{home} team', 'venue': ''}


def _entry(game, market_spread=np.nan, market_total=np.nan, model_spread=-3.0, model_total=47.0):
    line = {'market_spread': market_spread, 'market_total': market_total, 'model_spread': model_spread, 'model_total': model_total,
            'model_home': 25.0, 'model_away': 22.0, 'market_home': np.nan, 'market_away': np.nan}
    return {'game': game, 'line': line, 'rows': games_view.scoreboard_rows(game, line), 'weather': games_view.weather_view(None),
            'teams': [], 'props': pd.DataFrame()}


def test_a_sunday_kickoff_lands_in_its_window_by_eastern_time():
    assert ev.slate_of(_et('2026-11-01 09:30')) == 'morning'
    assert ev.slate_of(_et('2026-11-01 13:00')) == 'early'
    assert ev.slate_of(_et('2026-11-01 16:05')) == 'late' and ev.slate_of(_et('2026-11-01 16:25')) == 'late'
    assert ev.slate_of(_et('2026-11-01 20:20')) == 'night'
    assert ev.slate_of(_et('2026-11-01 00:00')) == 'tba' and ev.slate_of(None) == 'tba'
    assert ev.time_label(_et('2026-11-01 20:20')) == '8:20 PM' and ev.time_label(_et('2026-11-01 00:00')) == 'TBA'
    assert ev.day_group_name(_et('2026-10-29 20:15')) == 'THURSDAY NIGHT' and ev.day_group_name(_et('2026-12-26 16:30')) == 'SATURDAY'
    assert ev.eastern(pd.Timestamp('2026-11-01 18:00', tz='UTC')).hour == 13      # a UTC kickoff reads in Eastern


def test_the_slate_is_one_page_by_day_with_sunday_in_windows():
    games = [_game(1, 'KC', 'BUF', _et('2026-10-29 20:15')),               # Thu night
             _game(2, 'PIT', 'NO', _et('2026-11-01 09:30')),               # Sun morning
             _game(3, 'NO', 'KC', _et('2026-11-01 13:00')),
             _game(4, 'BUF', 'PIT', _et('2026-11-01 16:25')),
             _game(5, 'KC', 'PIT', _et('2026-11-01 20:20')),
             _game(6, 'BUF', 'NO', _et('2026-11-02 20:15'))]               # Mon night
    page = ev.schedule_html([_entry(g, market_spread=-3.5, market_total=44.5) for g in games], COLORS)
    assert page.count('<section class="e-list') == 1 and f'id="{ev.LIST_ANCHOR}"' in page
    heads = [f'<b>{name}</b>' for name in ('THURSDAY NIGHT', 'MORNING', 'EARLY', 'LATE', 'SUNDAY NIGHT', 'MONDAY NIGHT')]
    assert all(head in page for head in heads)
    assert [page.index(head) for head in heads] == sorted(page.index(head) for head in heads)
    assert 'href="#g3"' in page and '44.5' in page and '-3.5' in page and '+3.5' in page
    assert 'KC team' in page and '#e31837' in page                       # the full name, in the team's colour
    assert 'e-roomy' in page                                              # an NFL week fits roomy rows
    assert ev.schedule_html([], COLORS) == ''


def test_a_spread_the_book_has_not_posted_shows_the_models_in_italics():
    game = _game(1, 'KC', 'BUF', _et('2026-11-01 13:00'))
    page = ev.schedule_html([_entry(game, model_spread=-2.0, model_total=49.0)], COLORS)
    assert 'class="e-m"' in page and '49.0' in page and 'Italic numbers' in page
    assert ev.line_numbers({'market_spread': -1.0, 'market_total': 50.5, 'model_spread': -9.0, 'model_total': 40.0}) == (-1.0, False, 50.5, False)
    assert ev._spread_text(0.0) == 'PK'


def test_team_text_is_dark_on_a_light_colour_and_white_on_a_dark_one():
    assert ev.ink('#ffb612') == '#0b1020' and ev.ink('#00338d') == '#ffffff' and ev.ink('nonsense') == '#ffffff'


def _team_block(team, label='Quarterbacks'):
    frame = pd.DataFrame([{'PLAYER': 'Ace Passer', 'POS': 'QB', 'MATCHUP': 71.0, 'PTS': 19.4, 'PASS YD': 251.0, 'LINE': np.nan,
                           '_key': ('Ace Passer', 'QB', team), '_tone': 'plain', '_out': False}])
    return {'team': team, 'name': f'{team} team', 'side': 'Home', 'chips': [('PLAYS', 64.0), ('PASS ATT', 34.0)], 'items': [],
            'hidden': 0, 'tables': [(label, frame), ('Tight ends', pd.DataFrame())]}


def test_a_game_is_the_card_then_each_team_whole_then_the_props():
    game = _game(9, 'KC', 'BUF', _et('2026-11-01 16:25'))
    entry = _entry(game, market_spread=-3.5, market_total=47.5)
    entry['teams'] = [_team_block('KC'), _team_block('BUF')]
    entry['props'] = pd.DataFrame([{'PLAYER': 'Ace Passer', 'TEAM': 'KC', 'MARKET': 'Pass yards', 'MODEL': 251.0, 'LINE': 244.5,
                                    'DIFF %': 0.027, 'LEAN': 'OVER', '_key': None}])
    block = ev.game_html(entry, COLORS, {}, 'Average', 8)
    assert block.startswith('<section class="e-game" id="g9">')
    assert 'WEEK 8 · SUN 4:25 PM ET · LATE' in block and f'href="#{ev.LIST_ANCHOR}"' in block
    monday = ev.game_html(_entry(_game(8, 'KC', 'BUF', _et('2026-11-02 20:15'))), COLORS, {}, 'Average', 8)
    assert 'MON 8:15 PM ET' in monday and 'NIGHT' not in monday.split('</span>')[0]      # windows are a Sunday idea
    order = [block.index(token) for token in ('class="e-card"', 'class="e-team"', 'e-team e-second', 'Priced props')]
    assert order == sorted(order)
    assert block.count('class="e-team"') == 1 and block.count('e-team e-second') == 1      # the home team starts its own page
    assert 'Tight ends' not in block and 'KC team' in block and 'averages' in block
    assert '--tc:#e31837' in block


def test_player_table_drops_the_pos_chip_colours_the_matchup_and_bolds_points():
    frame = pd.DataFrame([{'PLAYER': 'Lead Back', 'POS': 'RB', 'MATCHUP': 40.0, 'PTS': 14.0, 'CAR': 14.2, 'LINE': np.nan,
                           '_key': 1, '_tone': 'plain', '_out': False},
                          {'PLAYER': 'Hurt Guy', 'POS': 'RB', 'MATCHUP': 40.0, 'PTS': np.nan, 'CAR': np.nan, 'LINE': np.nan,
                           '_key': 2, '_tone': 'out', '_out': True}])
    table = ev.player_table_html(frame)
    assert '>POS<' not in table and '_key' not in table
    assert 'x-m' in table and '>40<' in table and 'class="x-pts">14.0' in table
    assert 'class=x-out' in table
    assert ev.player_table_html(pd.DataFrame()) == ''


def test_props_table_shades_the_gap_and_formats_it():
    frame = pd.DataFrame([{'PLAYER': 'Ace Passer', 'TEAM': 'KC', 'MARKET': 'Pass yards', 'MODEL': 251.0, 'LINE': 244.5, 'DIFF %': 0.027,
                           'LEAN': 'OVER', '_key': None}])
    table = ev.props_table_html(frame)
    assert '+3%' in table and 'OVER' in table and '>TEAM<' not in table and '>MOVE<' not in table


def _board():
    rows = [
        {'Player': 'Ace Passer', 'Team': 'KC', 'Opponent': 'BUF', 'Pos': 'QB', 'passing_attempts': 33.0, 'passing_yards': 251.0,
         'Model Proj Pts': 19.0, 'Matchup Pct': 71.0, 'Line passing_yards': 244.5},
        {'Player': 'Lead Back', 'Team': 'KC', 'Opponent': 'BUF', 'Pos': 'RB', 'rushing_attempts': 14.2, 'rushing_yards': 66.0,
         'targets': 2.0, 'Model Proj Pts': 11.0, 'Matchup Pct': 40.0},
        {'Player': 'Deep Bench', 'Team': 'KC', 'Opponent': 'BUF', 'Pos': 'RB', 'rushing_attempts': 0.4, 'rushing_yards': 1.5,
         'targets': 0.1, 'Model Proj Pts': 0.4, 'Matchup Pct': 40.0},
        {'Player': 'Big Back', 'Team': 'BUF', 'Opponent': 'KC', 'Pos': 'RB', 'rushing_attempts': 18.0, 'rushing_yards': 90.0,
         'targets': 1.0, 'Model Proj Pts': 15.0, 'Matchup Pct': 62.0, 'Mkt rushing_yards': 84.5},
    ]
    return pd.DataFrame(rows)


GAME_OF = {'KC': ('BUF', True), 'BUF': ('KC', False)}


def test_board_rows_keep_players_with_a_role_most_points_first():
    rows = ev.board_rows(_board(), 'RB', 'Average', GAME_OF)
    assert [r['player'] for r in rows] == ['Big Back', 'Lead Back']
    assert not rows[0]['away'] and rows[1]['away']
    assert rows[0]['CAR'] == 18.0 and rows[1]['RUSH YD'] == 66.0 and rows[0]['PTS'] == 15.0
    assert rows[0]['line'] == 84.5                                       # no posted line: the market's yardage number


def test_board_report_is_one_document_with_a_section_per_position():
    page = ev.board_report('Week 8 board', '2026 · averages', _board(), 'Average', COLORS, GAME_OF)
    assert page.startswith('<!doctype html>') and '</html>' in page
    for position in games_view.POSITIONS:
        assert f'id="p{position}"' in page
    assert 'KC @ BUF' in page and 'BUF vs KC' in page and 'Deep Bench' not in page and '244.5' in page and '>PTS<' in page
    assert f'size:{ev.PAGE_WIDTH_IN:g}in {ev.PAGE_HEIGHT_IN:g}in' in page and 'NFL SCHOLAR' in page


def test_the_board_can_be_only_the_lined_players_by_the_size_of_the_gap():
    board = _board()
    board['Line rushing_yards'] = np.nan
    board.loc[board['Player'] == 'Lead Back', 'Line rushing_yards'] = 50.0        # model 66 vs 50: +32%
    board.loc[board['Player'] == 'Deep Bench', 'Line rushing_yards'] = 2.0        # no role, but a book has a line
    rows = ev.board_rows(board, 'RB', 'Average', GAME_OF, lined_only=True)
    assert {r['player'] for r in rows} == {'Lead Back', 'Big Back', 'Deep Bench'}
    assert abs(rows[0]['gap']) >= abs(rows[1]['gap']) >= abs(rows[2]['gap'])
    assert abs({r['player']: r['gap'] for r in rows}['Lead Back'] - 16.0 / 50.0) < 1e-9     # his YARDS against the yards line
    page = ev.board_report('Week 8 board', 's', board, 'Average', COLORS, GAME_OF, lined_only=True)
    assert '>GAP<' in page and 'Only players a book has a line on' in page
    assert '>GAP<' not in ev.board_report('Week 8 board', 's', board, 'Average', COLORS, GAME_OF)


def test_logos_are_fetched_once_shrunk_and_embedded(tmp_path):
    import io
    from PIL import Image
    buffer = io.BytesIO()
    Image.new('RGBA', (500, 500), (255, 0, 0, 255)).save(buffer, format='PNG')
    calls = []

    def fetch(url):
        calls.append(url)
        return buffer.getvalue()

    def broken(url):
        raise OSError('offline')
    logos = {'KC': 'https://x/kc.png', 'BUF': 'https://x/buf.png'}
    first = ev.embed_logos(logos, ['KC', 'KC', 'BUF', 'Nowhere'], str(tmp_path), fetch)
    assert set(first) == {'KC', 'BUF'} and first['KC'].startswith('data:image/png;base64,')
    ev.embed_logos(logos, ['KC'], str(tmp_path), fetch)
    assert len(calls) == 2                                                        # the second build read the cache
    assert ev.embed_logos(logos, ['KC'], str(tmp_path / 'x'), broken)['KC'] == 'https://x/kc.png'


def test_save_writes_the_html_and_skips_the_pdf_without_a_browser(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, 'find_browser', lambda: None)
    files = ev.save('<!doctype html><html><body>x</body></html>', 'week08_board', folder=str(tmp_path))
    assert os.path.exists(files['html']) and files['pdf'] is None
    assert ev.to_pdf('<html></html>', str(tmp_path / 'x.pdf')) is False


def _edge(player, group, diff_pct, lean='OVER', **extra):
    return {'player': player, 'team': 'KC', 'opponent': 'BUF', 'away': True, 'start': 'Sun 4:25 PM ET', 'group': group,
            'market': 'Pass yards', 'model': 260.0, 'line': 244.5, 'diff_pct': diff_pct, 'lean': lean, **extra}


def test_the_edges_page_lists_each_group_biggest_gap_first_with_the_floor_counted():
    rows = [_edge('Small Gap', 'Passing', 0.04), _edge('Medium', 'Passing', -0.15, 'UNDER'),
            _edge('Biggest', 'Passing', 0.30), _edge('A Back', 'Rushing', 0.12)]
    page = ev.edges_html(rows, COLORS)
    assert page.count('class="e-egroup"') == 2 and 'Receiving' not in page            # a group with no prop is left out
    assert page.index('Biggest') < page.index('Medium') and 'Small Gap' not in page
    assert '1 under 10% not shown' in page and '+30%' in page and '-15%' in page and 'x-under' in page
    assert 'KC @ BUF · Sun 4:25 PM' in page
    assert '>RECORD<' not in page and '>MOVE<' not in page and 'RECORD is' not in page    # no lean record or movement kept yet
    with_both = ev.edges_html([_edge('Biggest', 'Passing', 0.3, record_text='56% (30/54)', kind='solid', move=2.0)], COLORS)
    assert '>RECORD<' in with_both and '56% (30/54)' in with_both and '▲ +2.0' in with_both
    assert ev.edges_html([], COLORS) == ''
    capped = ev.edges_html([_edge(f'P{i}', 'Passing', 0.2 + i / 1000) for i in range(ev.EDGE_CAP + 5)], COLORS)
    assert '5 more over the floor' in capped


def test_a_selection_is_a_day_or_a_sunday_window_and_the_full_slate_by_default():
    games = [_game(1, 'KC', 'BUF', _et('2026-10-29 20:15')), _game(2, 'PIT', 'NO', _et('2026-11-01 13:00')),
             _game(3, 'NO', 'KC', _et('2026-11-01 16:25')), _game(4, 'BUF', 'PIT', _et('2026-11-01 20:20')),
             _game(5, 'KC', 'NO', _et('2026-11-02 20:15'))]
    entries = [_entry(g) for g in games]

    def ids(key):
        return [e['game']['game_id'] for e in ev.select_games(entries, key)]
    assert ids('all') == [1, 2, 3, 4, 5] and ids('nonsense') == [1, 2, 3, 4, 5]
    assert ids('thursday') == [1] and ids('sunday') == [2, 3, 4] and ids('monday') == [5]
    assert ids('early') == [2] and ids('late') == [3] and ids('night') == [4]
    assert [k for k, _ in ev.selection_options(entries)] == ['all', 'thursday', 'sunday', 'early', 'late', 'night', 'monday']
    assert [k for k, _ in ev.selection_options(entries[:2])] == ['all', 'thursday', 'sunday']   # one Sunday window: no window choice


def test_the_cover_shows_how_fresh_the_lines_are_and_what_is_inside():
    now = pd.Timestamp('2026-10-25 12:00', tz='UTC')
    fresh = ev.freshness_rows({'Pinnacle': 667, 'DraftKings': 0}, pd.Timestamp('2026-10-25 11:50', tz='UTC'), now)
    assert fresh[0]['age'] == '10 min ago' and fresh[0]['state'] == 'fresh' and fresh[0]['lines'] == '667'
    assert fresh[1]['state'] == 'none' and fresh[1]['lines'] == '—'
    assert ev.freshness_rows({'Pinnacle': 5}, pd.Timestamp('2026-10-24 06:00'), now)[0]['state'] == 'stale'     # naive = UTC
    page = ev.cover_html('Week 8 games', '14 games', fresh, 40, 451, 313, 'Sunday', 13, 25)
    assert 'Pinnacle' in page and '10 min ago' in page and '<b>40</b> of <b>451</b>' in page
    assert 'Sunday' in page and '>13<' in page and '>25<' in page and 'averages' in page
    assert 'medians' in ev.cover_html('t', 's', [], 0, 0, 0, 'Full slate', 0, 0, view='Median')
    assert 'No market lines' in ev.cover_html('t', 's', [], 0, 0, 0, 'Full slate', 0, 0)


def test_the_whole_report_is_cover_then_edges_then_slate_then_games():
    game = _game(9, 'KC', 'BUF', _et('2026-11-01 13:00'))
    entry = _entry(game, market_spread=-3.5, market_total=47.5)
    entry['teams'] = [_team_block('KC'), _team_block('BUF')]
    page = ev.games_report('Week 8 games', 's', [entry], COLORS, {}, 'Average', 8, freshness=[], priced=3, on_board=10,
                           edge_rows=[_edge('Biggest', 'Passing', 0.3)], selection='Sunday')
    order = [page.index(token) for token in ('class="e-cover"', 'id="edges"', f'id="{ev.LIST_ANCHOR}"', 'id="g9"')]
    assert order == sorted(order) and 'Sunday' in page


def test_the_copy_folder_is_remembered_and_the_files_are_copied(tmp_path):
    settings = tmp_path / 'settings'
    ev.save_settings({'copy': True, 'folder': str(tmp_path / 'one')}, str(settings))
    assert ev.load_settings(str(settings)) == {'copy': True, 'folder': str(tmp_path / 'one')}
    pdf, page = tmp_path / 'a.pdf', tmp_path / 'a.html'
    pdf.write_bytes(b'%PDF')
    page.write_text('x')
    written = ev.copy_files({'pdf': str(pdf), 'html': str(page)}, str(tmp_path / 'one' / 'deeper'))
    assert sorted(os.path.basename(w) for w in written) == ['a.html', 'a.pdf']
    assert ev.copy_files({'pdf': str(pdf)}, '') == [] and ev.copy_files({'pdf': None}, str(tmp_path / 'two')) == []


def test_the_slate_kickoff_text_becomes_an_eastern_time():
    from ui.tabs.rankings_games import kickoff_start
    start = kickoff_start('2026-11-01', 'Sun 4:25 PM EST')
    assert (start.hour, start.minute, str(start.tz)) == (16, 25, ev.EASTERN)
    assert kickoff_start('2026-10-29', 'Thu 8:15 PM EDT').hour == 20 and kickoff_start('2026-11-01', 'Sun 12:00 PM EST').hour == 12
    assert ev.is_tba(ev.eastern(kickoff_start('2026-11-01', '', time_tbd=True)))
    assert kickoff_start('', 'Sun 1:00 PM') is None
