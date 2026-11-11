"""Audits B4/B5/B6 helpers (2026-11-11): the FanDuel weekly parser, the Odds API budget rules, line shopping, the line shift,
closing-line value, and the pre-kickoff refresh's small pieces."""
import datetime
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest

from data import line_shift, line_shopping, odds_api_props, odds_sources
from data import prop_probability as pp
from data import prop_tiers as pt

HAS_ENGINE = np.isfinite(pp.prob_over('receiving_yards', np.array(['WR']), np.array([60.0]), np.array([54.5]))).all()


def _fd_market(mid, name, mtype, line, over, under, nested=False):
    def odds(v):
        return {'trueOdds': {'decimalOdds': {'decimalOdds': v} if nested else v}}
    return {'marketId': mid, 'marketName': name, 'marketType': mtype,
            'runners': [{'runnerName': f'{name.split(" - ")[0]} Over', 'handicap': line, 'winRunnerOdds': odds(over)},
                        {'runnerName': f'{name.split(" - ")[0]} Under', 'handicap': line, 'winRunnerOdds': odds(under)}]}


def test_fanduel_parser_keeps_main_two_way_lines():
    payload = {'attachments': {'markets': {
        '1': _fd_market('1', 'Puka Nacua - Receiving Yds', 'PLAYER_X_TOTAL_RECEIVING_YARDS', 84.5, 1.87, 1.95, nested=True),
        '2': _fd_market('2', 'Puka Nacua - Alt Receiving Yds', 'PLAYER_X_ALT_RECEIVING_YARDS', 99.5, 2.5, 1.5),
        '3': _fd_market('3', 'Puka Nacua - Total Receptions', 'PLAYER_X_TOTAL_RECEPTIONS', 7.5, 1.9, 1.9),
        '4': {'marketId': '4', 'marketName': 'Moneyline', 'marketType': 'MONEY_LINE', 'runners': []}}}}
    props, err = odds_sources.parse_fanduel_event_payloads([payload, payload])     # the same market twice: one row
    assert err is None
    assert len(props) == 2
    rec = props.set_index('market_raw').loc['Receiving Yds']
    assert rec['provider'] == 'FanDuel' and rec['line'] == 84.5 and rec['over_payout'] == 1.87
    assert 0.5 < rec['p_over'] < 0.52
    assert props.set_index('market_raw').loc['Total Receptions', 'market'] == 'receptions'


def test_fanduel_parser_drops_mismatched_lines():
    m = _fd_market('9', 'A B - Rushing Yds', 'PLAYER_X_TOTAL_RUSHING_YARDS', 50.5, 1.9, 1.9)
    m['runners'][1]['handicap'] = 51.5
    props, err = odds_sources.parse_fanduel_event_payloads([{'attachments': {'markets': {'9': m}}}])
    assert props.empty and err


def test_fanduel_week_events_window():
    now = pd.Timestamp('2026-11-11T12:00Z')
    page = {'attachments': {'events': {
        '1': {'name': 'Rams @ Seahawks', 'openDate': '2026-11-15T21:25:00Z'},
        '2': {'name': 'Rams @ Seahawks', 'openDate': '2026-11-25T21:25:00Z'},
        '3': {'name': 'NFL Futures', 'openDate': '2026-11-15T21:25:00Z'},
        '4': {'name': 'Bills @ Jets', 'openDate': '2026-11-10T21:25:00Z'}}}}
    assert odds_sources.fanduel_week_events(page, now) == ['1']


class _Resp:
    def __init__(self, body, remaining, status=200):
        self._body, self.status_code, self.text = body, status, json.dumps(body)
        self.headers = {'x-requests-remaining': str(remaining)}

    def json(self):
        return self._body


class _Session:
    def __init__(self, n_events, remaining):
        self.calls, self.remaining, self.n = [], remaining, n_events

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        start = '2026-11-15T18:00:00Z'
        if url.endswith('/events'):
            return _Resp([{'id': f'e{i}', 'commence_time': start} for i in range(self.n)], self.remaining)
        self.remaining -= len(odds_api_props.MARKETS)
        return _Resp({'id': url.split('/')[-2], 'bookmakers': [{'title': 'BetMGM', 'markets': [
            {'key': 'player_reception_yds', 'outcomes': [
                {'name': 'Over', 'description': 'Puka Nacua', 'point': 84.5, 'price': 1.9},
                {'name': 'Under', 'description': 'Puka Nacua', 'point': 84.5, 'price': 1.9}]}]}]}, self.remaining)


@pytest.fixture
def api_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(odds_api_props, 'STATE_PATH', str(tmp_path / 'state.json'))
    return tmp_path


NOW = datetime.datetime(2026, 11, 11, 12, tzinfo=datetime.timezone.utc)


def test_odds_api_skips_without_key_or_budget(api_paths):
    assert odds_api_props.pull_and_archive(api_key='', now=NOW).startswith('skipped: no Odds API key')
    s = _Session(14, remaining=120)                       # 14 games x 3 markets = 42 credits; 120 - 42 < reserve 100
    msg = odds_api_props.pull_and_archive(api_key='k', now=NOW, session=s, archive_dir=str(api_paths / 'arc'))
    assert msg.startswith('skipped: 42 credits needed') and len(s.calls) == 1   # only the free events call


def test_odds_api_pulls_and_then_waits(api_paths):
    s = _Session(2, remaining=500)
    msg = odds_api_props.pull_and_archive(api_key='k', now=NOW, session=s, archive_dir=str(api_paths / 'arc'))
    assert msg.startswith('pulled 2 games, 2 lines') and len(s.calls) == 3
    rows = odds_api_props.load_odds_api_archive(str(api_paths / 'arc'))
    assert set(rows['provider']) == {'OddsAPI BetMGM'} and rows['p_over'].round(3).eq(0.5).all()
    again = odds_api_props.pull_and_archive(api_key='k', now=NOW + datetime.timedelta(hours=5), session=s,
                                            archive_dir=str(api_paths / 'arc'))
    assert again.startswith('skipped: last pull 5.0h ago') and len(s.calls) == 3


def test_odds_api_rows_stay_out_of_the_props_archive():
    assert os.path.normpath(odds_api_props.ARCHIVE_DIR) != os.path.normpath(os.path.join('external_data', 'props_archive'))


def test_best_prices_per_side():
    rows = pd.DataFrame([
        {'player': 'A', 'market': 'receiving_yards', 'provider': 'Pinnacle', 'line': 54.5, 'over_payout': 1.91, 'under_payout': 1.91},
        {'player': 'A', 'market': 'receiving_yards', 'provider': 'DraftKings', 'line': 52.5, 'over_payout': 1.87, 'under_payout': 1.95},
        {'player': 'A', 'market': 'receiving_yards', 'provider': 'FanDuel', 'line': 54.5, 'over_payout': 1.95, 'under_payout': 1.95},
        {'player': 'A', 'market': 'receiving_yards', 'provider': 'PrizePicks (standard)', 'line': 53.5, 'over_payout': None,
         'under_payout': None}])
    best = line_shopping.best_prices(rows).iloc[0]
    assert (best['over_book'], best['over_line']) == ('DraftKings', 52.5)
    assert (best['under_book'], best['under_line'], best['under_price']) == ('FanDuel', 54.5, 1.95)
    assert best['books'] == 3 and best['line_spread'] == 2.0
    assert line_shopping.american(2.2) == '+120' and line_shopping.american(1.909) == '-110' and line_shopping.american(None) == ''


@pytest.mark.skipif(not HAS_ENGINE, reason='no prop engine on this machine')
def test_line_shift_round_trip_and_direction():
    m = line_shift.implied_mean('receiving_yards', ['WR'], [54.5], [0.5])
    assert np.isclose(pp.prob_over('receiving_yards', np.array(['WR']), m, np.array([54.5])), 0.5, atol=0.005).all()
    lower, same, higher = line_shift.p_over_at('receiving_yards', ['WR'] * 3, [54.5] * 3, [0.5] * 3, [49.5, 54.5, 59.5])
    assert lower > 0.5 > higher and same == 0.5
    assert np.isnan(line_shift.p_over_at('receiving_yards', ['WR'], [np.nan], [0.5], [50.5])).all()


@pytest.mark.skipif(not HAS_ENGINE, reason='no prop engine on this machine')
def test_clv_values_price_and_line_value():
    bets = pd.DataFrame({'week': [10, 10], '_key': ['a', 'b'], 'stat': ['receiving_yards'] * 2, 'book': ['DraftKings'] * 2,
                         'Pos': ['WR', 'WR'], 'side': ['over', 'under'], 'line': [54.5, 54.5], 'price': [2.0, 1.8]})
    close = pd.DataFrame({'week': [10, 10, 10], '_key': ['a', 'b', 'a'], 'stat': ['receiving_yards'] * 3,
                          'book': ['DraftKings', 'DraftKings', 'Pinnacle'], 'line': [56.5, 54.5, 54.5],
                          'p_book': [0.5, 0.6, 0.55], 'sharpest': [False, True, True]})
    v = pt.clv_values(bets, close)
    assert v['line_value'].tolist() == [2.0, 0.0]            # the over's line rose 2 (good), the under's stayed
    assert np.isclose(v.at[0, 'price_value'], 0.55 - 0.5)    # the sharp close at the same line: 0.55 against 1/2.0
    assert np.isclose(v.at[1, 'price_value'], 0.4 - 1 / 1.8)
    table = pt.clv_table(bets.assign(**{c: v[c] for c in v.columns}), ('side',))
    assert table.iloc[0]['with close'] == 2 and table.iloc[0]['status'] == 'tracking'


def _refresh_module():
    spec = importlib.util.spec_from_file_location('refresh_pre_kickoff', os.path.join('scripts', 'refresh_pre_kickoff.py'))
    mod = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    return mod


def test_refresh_fp_weekly_and_what_moved():
    r = _refresh_module()
    before = pd.DataFrame({'Player': ['A', 'B', 'C'], 'Pos': ['WR', 'RB', 'TE'], 'Team': ['LAR'] * 3,
                           'Raw Model Proj Pts': [15.0, 10.0, 6.0], 'receiving_yards': [80.0, 10.0, 30.0],
                           'Availability': ['active'] * 3, 'Injury Status': ['', 'Q', ''],
                           'FP Proj Pts': [14.0, np.nan, 5.0], 'FP receiving_yards': [75.0, np.nan, 28.0]})
    after = before.copy()
    after.loc[1, ['Raw Model Proj Pts', 'Injury Status']] = [7.5, 'D']
    after.loc[0, 'receiving_yards'] = 70.0
    after = after[after['Player'] != 'C']
    fp = r.fp_weekly_from(before)
    assert list(fp.columns) == ['Player', 'FP Proj Pts', 'receiving_yards'] and fp['Player'].tolist() == ['A', 'C']
    text = r.what_moved(before, after)
    assert 'pts 10.0 -> 7.5' in text and 'injury status Q -> D' in text and 'receiving_yards 80.0 -> 70.0' in text
    assert 'Off the board: C' in text
    assert r.what_moved(before, before) == ''


def test_pickem_break_evens_from_config():
    from data import pickem
    be = pickem.break_evens({'PrizePicks': {'power': {'2': 3.0, '6': 37.5}}})
    best, entry, two = be['PrizePicks']
    assert entry == '6-pick power' and np.isclose(best, 37.5 ** (-1 / 6)) and np.isclose(two, 3 ** -0.5)
    assert set(pickem.load_payouts()) >= {'PrizePicks', 'Underdog'}


def test_pickem_sides_and_table():
    from data import pickem
    payouts = {'PrizePicks': {'power': {'2': 3.0}}, 'Underdog': {'power': {'2': 3.0}}}
    legs = pd.DataFrame({'week': [10] * 4, '_key': list('abcd'), 'stat': ['receptions'] * 4,
                         'book': ['PrizePicks', 'PrizePicks', 'Underdog', 'PrizePicks'],
                         'kind': ['standard', 'standard', 'shaded', 'goblin'], 'line': [4.5] * 4,
                         'over_mult': [1.0, 1.0, 0.9, 1.0], 'under_mult': [1.0, 1.0, 1.1, np.nan],
                         'fair_over': [0.62, 0.40, 0.45, 0.70], 'Pos': ['WR'] * 4})
    s = pickem.choose_sides(legs, payouts).set_index('_key')
    assert s.at['a', 'side'] == 'over' and s.at['a', 'flagged']             # 0.62 >= 0.577
    assert s.at['b', 'side'] == 'under' and s.at['b', 'flagged']            # 0.60 >= 0.577
    assert s.at['c', 'side'] == 'under' and s.at['c', 'flagged']            # 0.55 >= 0.577 / 1.1 (the 1.1x multiplier)
    assert s.at['d', 'side'] == 'over' and np.isnan(s.at['d', 'break_even'])
    res = pickem.attach_results(s.reset_index(), pd.DataFrame({'week': [10] * 4, '_key': list('abcd'),
                                                               'receptions': [6, 3, 4.5, 5]}))
    assert set(res['_key']) == {'a', 'b', 'd'}                              # c pushed
    t = pickem.leg_table(res, ('kind',)).set_index('kind')
    assert t.at['goblin', 'status'] == 'calibration only' and t.at['standard', 'legs'] == 2
