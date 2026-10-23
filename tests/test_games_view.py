"""ui/games_view.py (2026-10-23): the Weekly Rankings GAMES view's pure parts, on small frames."""
import unittest

import numpy as np
import pandas as pd

from data.weather import GameWeather
from ui import games_view as gv


def _row(**kw):
    base = {'Player': 'P', 'Pos': 'WR', 'Team': 'KC', 'Injury Status': '', 'Availability': 1.0, 'Workload If Active': 1.0,
            '_availability_source': 'no current availability source', '_availability_note': '', 'QB1 Selection Required': False,
            'Expected Snap Share': 0.8, 'Model Proj Pts': 12.0, 'targets': 7.0, 'receptions': 5.0, 'receiving_yards': 60.0,
            'receiving_tds': 0.4, 'rushing_attempts': 0.0, 'passing_attempts': 0.0}
    base.update(kw)
    return base


class PlayerHealthTest(unittest.TestCase):
    def test_healthy_player_has_no_label(self):
        self.assertEqual(gv.player_health(_row())['label'], '')

    def test_out_by_report_and_by_your_ruling(self):
        report = gv.player_health(_row(**{'Injury Status': 'Out', 'Availability': 0.0,
                                          '_availability_source': 'FantasyPros injury report'}))
        self.assertEqual((report['flags'][0]['key'], report['label'], report['tone']), ('report_out', 'Out', 'out'))
        self.assertEqual(report['flags'][0]['source'], 'FantasyPros injury report')
        yours = gv.player_health(_row(**{'Injury Status': 'Ir', 'Availability': 0.0,
                                         '_availability_source': gv.OVERRIDE_SOURCE, '_availability_note': 'ACL'}))
        self.assertEqual((yours['flags'][0]['key'], yours['label']), ('ruled_out', 'IR'))
        self.assertIn('ACL', yours['flags'][0]['detail'])
        self.assertTrue(gv.is_out(yours))

    def test_questionable_reports_the_workload_step(self):
        health = gv.player_health(_row(**{'Injury Status': 'Questionable (45%)'}),
                                  {'questionable_workload': {'before': 14.0, 'after': 12.6}})
        self.assertEqual((health['label'], health['tone']), ('Questionable (45%)', 'warn'))
        self.assertIn('14.0 -> 12.6', health['flags'][0]['detail'])
        self.assertFalse(gv.is_out(health))

    def test_limited_workload_and_the_qb1_choice(self):
        limited = gv.player_health(_row(**{'Workload If Active': 0.5}))
        self.assertEqual((limited['label'], limited['tone']), ('50%', 'half'))
        qb = gv.player_health(_row(Pos='QB', **{'QB1 Selection Required': True}))
        self.assertEqual(qb['label'], 'QB1?')


class AvailabilityItemsTest(unittest.TestCase):
    def test_most_serious_first_reserves_hidden_and_where_the_volume_went(self):
        group = pd.DataFrame([
            _row(Player='Star', **{'Injury Status': 'Out', 'Availability': 0.0, '_availability_source': 'FantasyPros injury report'}),
            _row(Player='Doubt', **{'Injury Status': 'Questionable'}),
            _row(Player='Deep', **{'Injury Status': 'Ir', 'Availability': 0.0, 'Expected Snap Share': 0.02,
                                   '_availability_source': 'FantasyPros injury report'}),
            _row(Player='Fine'),
        ])
        ledger = [{'team': 'KC', 'volume': 'targets', 'source_player': 'Star',
                   'recipients': [{'player': 'Fine', 'allocated': 3.2}, {'player': 'Doubt', 'allocated': 1.1},
                                  {'player': 'Nobody', 'allocated': 0.01}]}]
        items, hidden = gv.availability_items(group, ledger=ledger)
        self.assertEqual([i['player'] for i in items], ['Star', 'Doubt'])
        self.assertEqual(hidden, 1)
        self.assertIn('his targets went to Fine +3.2, Doubt +1.1', items[0]['detail'])
        self.assertNotIn('Nobody', items[0]['detail'])


class PlayerTableTest(unittest.TestCase):
    def setUp(self):
        self.group = pd.DataFrame([
            _row(Player='WR1', targets=8.0, **{'Mkt receiving_yards': 64.5, 'Matchup Pct': 80.0}),
            _row(Player='WR2', targets=4.0),
            _row(Player='WR5', targets=0.3, **{'Expected Snap Share': 0.05}),
            _row(Player='Hurt', targets=0.0, **{'Injury Status': 'Out', 'Availability': 0.0,
                                                '_availability_source': 'FantasyPros injury report'}),
        ])

    def test_order_role_filter_and_columns(self):
        frame = gv.position_table(self.group, 'WR', role_only=True)
        self.assertEqual(list(frame['PLAYER']), ['WR1', 'WR2', 'Hurt'])
        self.assertEqual([c for c in frame.columns if not c.startswith('_')],
                         ['PLAYER', 'POS', 'MATCHUP', 'HEALTH', 'PTS', 'REC YD', 'REC', 'TGT', 'REC TD', 'LINE'])
        hurt = frame[frame['PLAYER'] == 'Hurt'].iloc[0]
        self.assertTrue(np.isnan(hurt['PTS']) and np.isnan(hurt['REC YD']))
        self.assertTrue(bool(hurt['_out']))

    def test_all_players_keeps_the_reserve_and_empty_columns_drop(self):
        group = self.group.drop(columns=['Mkt receiving_yards', 'Matchup Pct']).iloc[:3]
        frame = gv.position_table(group, 'WR')
        self.assertIn('WR5', list(frame['PLAYER']))
        self.assertFalse({'HEALTH', 'LINE', 'MATCHUP'} & set(frame.columns))

    def test_styled_table_renders(self):
        html = gv.styled_table(gv.position_table(self.group, 'WR')).to_html()
        self.assertIn('WR1', html)


class PropsTest(unittest.TestCase):
    def test_largest_disagreement_first(self):
        board = pd.DataFrame([
            _row(Player='A', receiving_yards=70.0, **{'Mkt receiving_yards': 60.0}),
            _row(Player='B', receiving_yards=50.0, **{'Mkt receiving_yards': 49.5, 'Mkt receptions': 4.5}),
            _row(Player='C', Team='LV', receiving_yards=30.0, **{'Mkt receiving_yards': 40.0}),
        ])
        market = gv.long_market(board)
        props = gv.props_table(market, ['KC'])
        self.assertEqual(list(props['PLAYER']), ['A', 'B', 'B'])
        self.assertEqual(props.iloc[0]['LEAN'], 'OVER')
        self.assertEqual(len(gv.props_table(market, ['KC'], 'Passing')), 0)
        self.assertIn('LEAN', gv.styled_props(props).to_html())


class ScoreboardAndWeatherTest(unittest.TestCase):
    def test_market_points_from_spread_and_total(self):
        game = {'home': 'KC', 'away': 'LV', 'home_name': 'Kansas City Chiefs', 'away_name': 'Las Vegas Raiders',
                'spread_line': 7.0, 'total_line': 45.0, 'neutral': False}
        line = gv.game_scoreline(game, None)
        self.assertEqual((line['market_home'], line['market_away'], line['market_spread']), (26.0, 19.0, -7.0))
        rows = gv.scoreboard_rows(game, line)
        self.assertEqual(rows[0]['market'], '19.0')
        self.assertEqual(rows[2]['market'], '-7.0')
        self.assertEqual(rows[0]['model'], '—')

    def test_weather_states(self):
        self.assertEqual(gv.weather_view(None)['state'], 'none')
        self.assertEqual(gv.weather_view(GameWeather(is_outdoor=False))['state'], 'indoors')
        view = gv.weather_view(GameWeather(wind_mph=22.0, temp_f=28.0, source='open-meteo-cache'), 'snow')
        labels = {c['k']: (c['v'], c['label']) for c in view['cells']}
        self.assertEqual(labels['TEMP'], ('28', 'Freezing'))
        self.assertEqual(labels['WIND'], ('22', 'Very windy'))
        self.assertEqual(labels['PRECIP'][0], 'Snow')
        self.assertIn('Open-Meteo', view['foot'])

    def test_days_in_kickoff_order(self):
        games = [{'date': '2026-10-22'}, {'date': '2026-10-25'}, {'date': '2026-10-25'}, {'date': '2026-10-26'}]
        self.assertEqual(gv.day_options(games), [('Thu', 1), ('Sun', 2), ('Mon', 1)])

    def test_header_and_team_html_escape_and_render(self):
        game = {'home': 'KC', 'away': 'LV', 'home_name': 'Kansas City Chiefs', 'away_name': 'Las <Vegas>', 'neutral': False,
                'spread_line': 3.0, 'total_line': 44.0}
        html = gv.header_html(game, gv.scoreboard_rows(game, gv.game_scoreline(game, None)), gv.weather_view(None), 'Sun 1:00 PM EDT')
        self.assertIn('Las &lt;Vegas&gt;', html)
        self.assertIn('MODEL VS MARKET', html)
        block = {'team': 'KC', 'name': 'Kansas City Chiefs', 'side': 'Home', 'chips': gv.team_chips(pd.DataFrame([_row(Pos='QB', passing_attempts=35.0)])),
                 'items': [], 'hidden': 0}
        self.assertIn('HEALTHY', gv.teams_html([block]))


if __name__ == '__main__':
    unittest.main()
