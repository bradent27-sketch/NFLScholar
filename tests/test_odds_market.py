"""
Offline tests for data/odds_market.py's estimate_full_season_scoring - the
matchup-adjusted fill-in for games without a posted Vegas line, built for
Draft HQ's "Market lines vs this board" panel. Monkeypatches fetch_game_lines
with hand-built fixtures rather than hitting the network, same convention
every other odds/market test in this repo uses.

Runs two ways: `python tests/test_odds_market.py` needs nothing but the
app's own dependencies, and `pytest tests/` works if pytest is installed.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.odds_market as om  # noqa: E402


def _games(rows):
    frame = pd.DataFrame(rows)
    frame['season'] = 2025
    frame['game_type'] = 'REG'
    return frame


def test_fully_posted_season_matches_team_scoring_environment_exactly():
    games = _games([
        {'week': 1, 'home_team': 'AAA', 'away_team': 'BBB', 'spread_line': 3.0, 'total_line': 50.0},
        {'week': 2, 'home_team': 'BBB', 'away_team': 'AAA', 'spread_line': -1.0, 'total_line': 46.0},
    ])
    om.fetch_game_lines = lambda season: (games, {
        'error': None, 'games': 2, 'posted': 2, 'weeks_posted': [1, 2],
    })
    full, meta = om.estimate_full_season_scoring(2025)
    env, _ = om.team_scoring_environment(2025)
    assert meta['error'] is None
    assert (full.set_index('team')['games_posted'] == full.set_index('team')['games']).all()
    merged = full.set_index('team')['full_season_ppg']
    reference = env.set_index('team')['implied_ppg']
    for team in merged.index:
        assert abs(merged[team] - reference[team]) < 1e-9


def test_unposted_game_is_estimated_from_matchup_not_left_blank():
    games = _games([
        # Week 1: both games posted.
        {'week': 1, 'home_team': 'AAA', 'away_team': 'BBB', 'spread_line': 3.0, 'total_line': 50.0},
        {'week': 1, 'home_team': 'CCC', 'away_team': 'DDD', 'spread_line': -7.0, 'total_line': 40.0},
        # Week 2: AAA (strong offense) at DDD (defense that allowed a lot
        # in week 1) - no line posted yet.
        {'week': 2, 'home_team': 'AAA', 'away_team': 'DDD', 'spread_line': None, 'total_line': None},
    ])
    om.fetch_game_lines = lambda season: (games, {
        'error': None, 'games': 3, 'posted': 2, 'weeks_posted': [1],
    })
    full, meta = om.estimate_full_season_scoring(2025)
    row = full.set_index('team').loc['AAA']
    # AAA has one posted game (26.5) and one MODELED game - the average of
    # the two should differ from the posted-only number, proving the gap
    # actually got filled rather than silently ignored.
    assert row['games'] == 2
    assert row['games_posted'] == 1
    assert row['modeled_games'] == 1
    assert abs(row['full_season_ppg'] - row['posted_ppg']) > 1e-6
    # Every team appears with a real (non-NaN) full-season number, even
    # BBB and CCC which only ever appear in a single posted game.
    assert full['full_season_ppg'].notna().all()


def test_a_team_with_no_posted_games_falls_back_to_league_average():
    games = _games([
        {'week': 1, 'home_team': 'AAA', 'away_team': 'BBB', 'spread_line': 0.0, 'total_line': 44.0},
        {'week': 2, 'home_team': 'AAA', 'away_team': 'ZZZ', 'spread_line': None, 'total_line': None},
    ])
    om.fetch_game_lines = lambda season: (games, {
        'error': None, 'games': 2, 'posted': 1, 'weeks_posted': [1],
    })
    full, meta = om.estimate_full_season_scoring(2025)
    # ZZZ never appears in a posted game at all - its estimate should still
    # be a finite, sane number (built off the league-average fallback), not
    # a crash or a NaN.
    zzz = full.set_index('team').loc['ZZZ']
    assert pd.notna(zzz['full_season_ppg'])
    assert 0 < zzz['full_season_ppg'] < 60


def test_no_lines_posted_at_all_returns_an_error_not_a_crash():
    games = _games([{'week': 1, 'home_team': 'AAA', 'away_team': 'BBB',
                     'spread_line': None, 'total_line': None}])
    om.fetch_game_lines = lambda season: (games, {
        'error': None, 'games': 1, 'posted': 0, 'weeks_posted': [],
    })
    full, meta = om.estimate_full_season_scoring(2025)
    assert full.empty
    assert meta.get('error')


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failures.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
