"""Phone exports (2026-10-18): ui/report_export.py draws the week's game report and the bare-bones board as PDFs; the
projection breakdown shows the matchup rank as its own metric tile."""
import os
import re
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

from ui import report_export as rx  # noqa: E402


def _board():
    rows = [
        ('Starter QB', 'QB', 'AAA', 'BBB', 21.0, ''), ('Backup QB', 'QB', 'AAA', 'BBB', 0.0, ''),
        ('Lead Back', 'RB', 'AAA', 'BBB', 15.0, 'Questionable (80%)'), ('Second Back', 'RB', 'AAA', 'BBB', 7.0, ''),
        ('Third Back', 'RB', 'AAA', 'BBB', 2.0, ''),
        ('Alpha', 'WR', 'AAA', 'BBB', 18.0, ''), ('Beta', 'WR', 'AAA', 'BBB', 12.0, ''),
        ('Gamma', 'WR', 'AAA', 'BBB', 8.0, ''), ('Delta', 'WR', 'AAA', 'BBB', 4.0, ''),
        ('Out Receiver', 'WR', 'AAA', 'BBB', 0.0, 'Out'),
        ('Seam', 'TE', 'AAA', 'BBB', 9.0, ''),
        ('Other QB', 'QB', 'BBB', 'AAA', 17.0, ''), ('Other WR', 'WR', 'BBB', 'AAA', 14.0, 'Doubtful'),
    ]
    return pd.DataFrame(rows, columns=['Player', 'Pos', 'Team', 'Opponent', 'Model Proj Pts', 'Injury Status'])


def _games():
    return pd.DataFrame([{
        'Away': 'BBB', 'Home': 'AAA', 'Kickoff': 'Sun 1:00 PM EDT', 'Date Display': '10/25/2026', 'Venue': 'Field',
        'Neutral Site': False, 'Spread Line': 3.5, 'Total Line': 47.5, 'Played': False, 'Live': False,
        'Away Color': '#002244', 'Home Color': '#ffb612', 'Away Pts': None, 'Home Pts': None, 'Status Detail': '',
    }])


def _pages(pdf_bytes):
    return len(re.findall(rb'/Type\s*/Page[^s]', pdf_bytes))


def test_injury_tags():
    assert rx.injury_tag('Questionable (95%)') == 'Q'
    assert rx.injury_tag('Doubtful') == 'D'
    assert rx.injury_tag('Out') == 'OUT'
    assert rx.injury_tag('Reserve List (Res/R01)') == 'OUT'
    assert rx.injury_tag('') == '' and rx.injury_tag(None) == '' and rx.injury_tag(float('nan')) == ''


def test_implied_totals_follow_the_home_favored_sign():
    away, home = rx.implied_totals(3.5, 47.5)          # spread_line > 0: home favored by 3.5
    assert (away, home) == (22.0, 25.5)
    assert rx.implied_totals(None, 47.5) == (None, None)
    assert rx.line_text('BBB', 'AAA', 3.5, 47.5) == 'AAA -3.5  ·  O/U 47.5'
    assert rx.line_text('BBB', 'AAA', -2.0, None) == 'BBB -2'


def test_weather_text():
    assert rx.weather_text(None) == ''
    assert rx.weather_text(SimpleNamespace(is_outdoor=False, wind_mph=None, temp_f=None)) == 'Indoors'
    assert rx.weather_text(SimpleNamespace(is_outdoor=True, wind_mph=14.2, temp_f=41.0), 'light_rain') == \
        'wind 14 mph, 41°F, light rain'


def test_key_players_take_the_starting_shape_and_skip_sidelined_players():
    got = rx.key_players(_board(), 'AAA')
    assert [p for p, *_ in got] == ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE']
    names = [n for _, n, _, _ in got]
    assert 'Backup QB' not in names and 'Out Receiver' not in names and 'Delta' not in names
    assert got[1] == ('RB', 'Lead Back', 15.0, 'Q')


def test_board_tables_rank_by_projection_and_drop_zero_rows():
    tables = rx.board_tables(_board(), depth={'QB': 5, 'WR': 2})
    assert list(tables) == ['QB', 'WR']
    assert tables['QB']['Player'].tolist() == ['Starter QB', 'Other QB']        # the 0.0 backup is left out
    assert tables['WR']['Player'].tolist() == ['Alpha', 'Other WR'] and tables['WR']['Inj'].tolist() == ['', 'D']


def test_opponent_labels_mark_away_teams():
    assert rx._opponent_labels(_games(), {}) == {'BBB': '@AAA', 'AAA': 'BBB'}


def test_pdfs_are_real_multi_page_files():
    games = pd.concat([_games()] * 4, ignore_index=True)     # 4 games -> 2 pages at 3 a page
    pdf = rx.game_report_pdf(games, _board(), 'Week 7 · games', 'sub', {}, {}, {})
    assert pdf[:5] == b'%PDF-' and _pages(pdf) == 2
    board = rx.board_pdf(_board(), 'Week 7 · board', 'sub', _games(), {})
    assert board[:5] == b'%PDF-' and _pages(board) == 4          # QB, RB, WR, TE: one page each here


def test_empty_inputs_still_make_a_readable_page():
    assert _pages(rx.game_report_pdf(pd.DataFrame(), _board(), 't', 's')) == 1
    assert _pages(rx.board_pdf(_board().iloc[0:0], 't', 's')) == 1


def test_file_names_and_saving(tmp_path):
    names = rx.export_file_names(2026, 7, 'Half-PPR')
    assert names == {'games': 'NFL_2026_wk07_half_games.pdf', 'board': 'NFL_2026_wk07_half_board.pdf'}
    paths = rx.save_exports({'games': b'%PDF-1', 'board': b'%PDF-2'}, str(tmp_path / 'sub'), names)
    assert [open(p, 'rb').read() for p in paths] == [b'%PDF-1', b'%PDF-2']
    assert isinstance(rx.default_export_dir(), str)


# --- the breakdown header's matchup tile ---------------------------------------------------------------------

class _Col:
    def __init__(self, log):
        self.log = log

    def metric(self, label, value, delta=None, **kw):
        self.log.append(('metric', label, value, delta, kw.get('delta_color')))


class _FakeSt:
    def __init__(self):
        self.calls = []

    def columns(self, n):
        return [_Col(self.calls) for _ in range(n if isinstance(n, int) else len(n))]

    def caption(self, text, *a, **k):
        self.calls.append(('caption', text))

    def markdown(self, text, *a, **k):
        self.calls.append(('markdown', text))


def _header(monkeypatch, matchup, calibration=True):
    import ui.tabs.rankings as rk
    fake = _FakeSt()
    monkeypatch.setattr(rk, 'st', fake)
    monkeypatch.setattr(rk, '_render_decomposition_navigation', lambda detail: None)
    rk._render_decomposition_header({
        'position': 'WR', 'team': 'AAA', 'opponent': 'BBB', 'target_week': 7, 'as_of_week': 7,
        'raw_points': 15.0, 'calibrated_points': 14.2, 'availability': {'status': 'healthy'},
        'calibration': {'enabled': calibration}, 'defense_matchup': matchup})
    return [c for c in fake.calls if c[0] == 'metric'], fake.calls


def test_matchup_is_a_full_size_metric_tile(monkeypatch):
    # defense_stat_rank's rank 1 = allows the most; 30 of 32 allowed -> 3rd toughest
    metrics, calls = _header(monkeypatch, {'rank': 30, 'of': 32, 'value': 12.1, 'league_avg': 15.0, 'source': 's'})
    assert [m[1] for m in metrics] == ['Raw model', 'Displayed projection', 'Availability', 'Matchup']
    tile = metrics[-1]
    assert tile[2] == '3/32' and tile[3] == 'toughest vs WR' and tile[4] == 'inverse'
    assert any('#3 of 32 toughest' in str(c[1]) for c in calls if c[0] == 'caption')


def test_easy_matchup_is_green_and_a_missing_one_shows_a_dash(monkeypatch):
    metrics, _ = _header(monkeypatch, {'rank': 2, 'of': 32, 'value': 20.0, 'league_avg': 15.0}, calibration=False)
    assert [m[1] for m in metrics] == ['Projected points', 'Availability', 'Matchup']
    assert metrics[-1][2] == '31/32' and metrics[-1][4] == 'normal'
    metrics, _ = _header(monkeypatch, None)
    assert metrics[-1][1:3] == ('Matchup', '—')
