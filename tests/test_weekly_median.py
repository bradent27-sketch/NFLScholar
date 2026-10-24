"""data/weekly_median.py (2026-10-23): a median beside every projection, and the Average/Median switch of the Weekly Rankings
board and GAMES view."""
import math
import unittest

import pandas as pd

from data import weekly_median as wm
from ui import games_view as gv


class PoissonMedianTest(unittest.TestCase):
    def test_continuity_corrected_values(self):
        # P(0) = e^-0.4 = 0.670 > 0.5, so the median sits inside [-0.5, 0.5]: -0.5 + 0.5 / 0.670
        self.assertAlmostEqual(wm.poisson_median(0.4), -0.5 + 0.5 / math.exp(-0.4), places=9)
        self.assertEqual(wm.poisson_median(0.0), 0.0)
        self.assertLess(wm.poisson_median(0.4), 0.4)

    def test_line_reading(self):
        # median above 0.5 <=> P(1 or more) above one half <=> P(0) below one half <=> mean above ln 2
        self.assertGreater(wm.poisson_median(0.70), 0.5)
        self.assertLess(wm.poisson_median(0.68), 0.5)

    def test_increasing(self):
        values = [wm.poisson_median(m / 10) for m in range(1, 60)]
        self.assertEqual(values, sorted(values))


class StatMedianTest(unittest.TestCase):
    def test_fitted_curve_and_clip(self):
        a, b = wm.MEDIAN_CURVES[('RB', 'rushing_yards')]
        self.assertAlmostEqual(wm.stat_median('RB', 'rushing_yards', 60.0), 60.0 * (a + b * math.log(60.0)), places=9)
        self.assertGreaterEqual(wm.stat_median('RB', 'rushing_yards', 0.5) / 0.5, wm.RATIO_CLIP[0])

    def test_skewed_stats_sit_below_the_mean_and_tds_use_poisson(self):
        self.assertLess(wm.stat_median('WR', 'receiving_yards', 55.0), 55.0)
        self.assertLess(wm.stat_median('WR', 'points', 12.0), 12.0)
        self.assertAlmostEqual(wm.stat_median('WR', 'receiving_tds', 0.45), wm.poisson_median(0.45))

    def test_fallbacks(self):
        self.assertAlmostEqual(wm.stat_median('WR', 'rushing_attempts', 0.6), wm.poisson_median(0.6))
        self.assertEqual(wm.stat_median('QB', 'passing_completions', 21.0), 21.0)
        self.assertEqual(wm.stat_median('WR', 'receiving_yards', 0.0), 0.0)


class AttachMediansTest(unittest.TestCase):
    def test_columns_and_a_player_who_is_out(self):
        board = pd.DataFrame([{'Player': 'A', 'Pos': 'WR', 'Availability': 1.0, 'Model Proj Pts': 14.0, 'targets': 8.0,
                               'receiving_yards': 70.0, 'receiving_tds': 0.5},
                              {'Player': 'B', 'Pos': 'WR', 'Availability': 0.0, 'Model Proj Pts': 0.0, 'targets': 0.0,
                               'receiving_yards': 0.0, 'receiving_tds': 0.0}])
        out = wm.attach_medians(board)
        for col in ('Median targets', 'Median receiving_yards', 'Median receiving_tds', 'Median Proj Pts'):
            self.assertIn(col, out.columns)
        self.assertLess(out.loc[0, 'Median Proj Pts'], 14.0)
        self.assertEqual(out.loc[1, 'Median receiving_yards'], 0.0)
        self.assertEqual(list(out['receiving_yards']), [70.0, 0.0])

    def test_a_stat_the_position_does_not_project_stays_blank(self):
        board = pd.DataFrame([{'Player': 'A', 'Pos': 'WR', 'Availability': 1.0, 'Model Proj Pts': 14.0,
                               'receiving_yards': 70.0, 'passing_yards': float('nan')}])
        self.assertTrue(pd.isna(wm.attach_medians(board).loc[0, 'Median passing_yards']))


class GamesViewSwitchTest(unittest.TestCase):
    def test_tables_read_the_chosen_view_and_props_read_the_median(self):
        row = {'Player': 'A', 'Pos': 'WR', 'Team': 'KC', 'Injury Status': '', 'Availability': 1.0, 'Workload If Active': 1.0,
               '_availability_source': '', '_availability_note': '', 'Expected Snap Share': 0.9, 'Model Proj Pts': 14.0,
               'targets': 8.0, 'receptions': 5.5, 'receiving_yards': 70.0, 'receiving_tds': 0.5, 'Mkt receiving_yards': 60.5}
        board = wm.attach_medians(pd.DataFrame([row]))
        avg = gv.position_table(board, 'WR')
        med = gv.position_table(board, 'WR', view='Median')
        self.assertEqual(avg.iloc[0]['REC YD'], 70.0)
        self.assertAlmostEqual(med.iloc[0]['REC YD'], board.loc[0, 'Median receiving_yards'])
        self.assertAlmostEqual(med.iloc[0]['PTS'], board.loc[0, 'Median Proj Pts'])
        props = gv.long_market(board)
        self.assertAlmostEqual(props.iloc[0]['model'], board.loc[0, 'Median receiving_yards'])
        self.assertIn('medians', gv.table_header_html('KC', 'Kansas City Chiefs', 1, view='Median'))


if __name__ == '__main__':
    unittest.main()
