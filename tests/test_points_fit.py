"""data/points_fit.py (2026-10-23): the display-only points fit behind the GAMES view's MODEL column."""
import unittest

import pandas as pd

from data.points_fit import fit_points


def _schedule(games, season=2026):
    return pd.DataFrame([{'season': season, 'week': w, 'game_type': 'REG', 'home_team': h, 'away_team': a, 'home_score': hs,
                          'away_score': as_, 'location': 'Home'} for w, h, a, hs, as_ in games])


class PointsFitTest(unittest.TestCase):
    def test_a_strong_offense_and_a_weak_defense_show(self):
        games = []
        for week in range(1, 9):
            games += [(week, 'AAA', 'BBB', 34, 17), (week, 'CCC', 'DDD', 20, 20), (week, 'BBB', 'CCC', 17, 24),
                      (week, 'DDD', 'AAA', 21, 31)]
        fit = fit_points(_schedule(games), 9)
        self.assertGreater(fit.offense['AAA'], fit.offense['DDD'])
        home, away = fit.game('AAA', 'BBB')
        self.assertGreater(home, away)
        self.assertAlmostEqual(home + away, fit.expected('AAA', 'BBB', True) + fit.expected('BBB', 'AAA', False))

    def test_only_games_before_the_week_count(self):
        early = _schedule([(1, 'AAA', 'BBB', 20, 20)])
        later = _schedule([(1, 'AAA', 'BBB', 20, 20), (2, 'AAA', 'BBB', 60, 0)])
        self.assertAlmostEqual(fit_points(early, 2).offense['AAA'], fit_points(later, 2).offense['AAA'])

    def test_ratings_shrink_toward_average_with_little_evidence(self):
        # home and away once each, so home field cannot carry the 40-10 gap
        fit = fit_points(_schedule([(1, 'AAA', 'BBB', 40, 10), (2, 'BBB', 'AAA', 10, 40)]), 3)
        self.assertLess(fit.offense['AAA'], 15.0)
        self.assertGreater(fit.offense['AAA'], 0.0)

    def test_no_scores_no_fit(self):
        self.assertIsNone(fit_points(_schedule([]), 1))
        unplayed = _schedule([(1, 'AAA', 'BBB', None, None)])
        self.assertIsNone(fit_points(unplayed, 3))


if __name__ == '__main__':
    unittest.main()
