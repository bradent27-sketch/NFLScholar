"""
Offline tests for data/loaders.py.

Same convention as tests/test_weekly_projections.py: hand-built fixtures,
no network - nflreadpy's own load functions are monkeypatched per test.

Runs two ways: `python tests/test_loaders.py` needs nothing but the app's
own dependencies, and `pytest tests/` works if pytest is installed.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.loaders as loaders  # noqa: E402


class _FakeFrame:
    """Stand-in for nflreadpy's return value - only needs to_pandas()."""
    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df


def test_load_team_pace_through_week_drops_the_in_progress_week():
    # Regression (2026-09-14): a Thursday opener going final gives
    # nflreadpy real week-1 rows for ONLY the two teams that already
    # played, while the other ~30 have none yet. Without a cutoff,
    # load_team_pace(year) returned those two teams' real one-game pace
    # and silently omitted everyone else - corrupting the league-average
    # denominator and (via build_weekly_projections' pace_mult) handing
    # exactly those two teams' own players a live, small-sample pace
    # multiplier while the rest of the board stayed neutral.
    # through_week must drop the in-progress week entirely (same strict
    # cutoff _played_weeks_before uses elsewhere), so a partially-played
    # target week reads as no current-season data at all - not a
    # two-team sample - and falls back to prior season upstream.
    week1 = pd.DataFrame([
        {'team': 'NE', 'opponent_team': 'SEA', 'week': 1,
         'attempts': 30, 'carries': 25, 'sacks_suffered': 2},
        {'team': 'SEA', 'opponent_team': 'NE', 'week': 1,
         'attempts': 35, 'carries': 20, 'sacks_suffered': 1},
    ])
    week2 = pd.DataFrame([
        {'team': 'KC', 'opponent_team': 'DEN', 'week': 2,
         'attempts': 28, 'carries': 22, 'sacks_suffered': 1},
        {'team': 'DEN', 'opponent_team': 'KC', 'week': 2,
         'attempts': 32, 'carries': 18, 'sacks_suffered': 3},
    ])
    both_weeks = pd.concat([week1, week2], ignore_index=True)

    original_stats = loaders.nflreadpy.load_team_stats
    original_sched = loaders.nflreadpy.load_schedules
    try:
        loaders.nflreadpy.load_team_stats = (
            lambda years, summary_level='week': _FakeFrame(both_weeks))
        # No OT games in this fixture - an empty schedule keeps
        # _overtime_team_weeks a no-op without ever hitting the network.
        loaders.nflreadpy.load_schedules = lambda years: _FakeFrame(pd.DataFrame())

        # Only week 1 exists (in progress); as-of week 1 must see nothing
        # this season, not NE/SEA's one game.
        loaders.load_team_pace.clear()
        assert loaders.load_team_pace(2099, through_week=1).empty

        # Weeks 1-2 both real/complete: as-of week 3 should see every
        # team from both weeks, not just the most recent week's pair.
        loaders.load_team_pace.clear()
        through3 = loaders.load_team_pace(2099, through_week=3)
        assert set(through3.index) == {'NE', 'SEA', 'KC', 'DEN'}

        # No through_week keeps the old whole-history-to-date behavior
        # (used only for a fully-complete prior season).
        loaders.load_team_pace.clear()
        unfiltered = loaders.load_team_pace(2099)
        assert set(unfiltered.index) == {'NE', 'SEA', 'KC', 'DEN'}
    finally:
        loaders.nflreadpy.load_team_stats = original_stats
        loaders.nflreadpy.load_schedules = original_sched


def test_load_team_pace_reports_games_played_alongside_each_average():
    stats = pd.DataFrame([
        {'team': 'KC', 'opponent_team': 'DEN', 'week': 1,
         'attempts': 28, 'carries': 22, 'sacks_suffered': 1},
        {'team': 'DEN', 'opponent_team': 'KC', 'week': 1,
         'attempts': 32, 'carries': 18, 'sacks_suffered': 3},
        {'team': 'KC', 'opponent_team': 'LAC', 'week': 2,
         'attempts': 30, 'carries': 20, 'sacks_suffered': 2},
        {'team': 'LAC', 'opponent_team': 'KC', 'week': 2,
         'attempts': 25, 'carries': 25, 'sacks_suffered': 1},
    ])
    original_stats = loaders.nflreadpy.load_team_stats
    original_sched = loaders.nflreadpy.load_schedules
    try:
        loaders.nflreadpy.load_team_stats = (
            lambda years, summary_level='week': _FakeFrame(stats))
        loaders.nflreadpy.load_schedules = lambda years: _FakeFrame(pd.DataFrame())
        loaders.load_team_pace.clear()
        pace = loaders.load_team_pace(2099)
    finally:
        loaders.nflreadpy.load_team_stats = original_stats
        loaders.nflreadpy.load_schedules = original_sched
    # KC has two games on the books, DEN and LAC one each - the shrinkage
    # build_weekly_projections applies to pace_mult (PACE_PRIOR_GAMES) needs
    # this count sitting right next to the average it was built from.
    assert pace.loc['KC', 'off_games'] == 2
    assert pace.loc['DEN', 'def_games'] == 1
    assert pace.loc['LAC', 'def_games'] == 1


def test_load_team_pace_discounts_a_game_that_went_to_overtime():
    # A and B's Week 1 game ran long (70 combined plays); C and D's did not
    # (60). Raw, A/B would look like the faster pair purely from bonus OT
    # time - reported 2026-09-15 (the Lions' Week 1 opener went to OT).
    stats = pd.DataFrame([
        {'team': 'A', 'opponent_team': 'B', 'week': 1,
         'attempts': 40, 'carries': 28, 'sacks_suffered': 2},
        {'team': 'B', 'opponent_team': 'A', 'week': 1,
         'attempts': 38, 'carries': 30, 'sacks_suffered': 2},
        {'team': 'C', 'opponent_team': 'D', 'week': 1,
         'attempts': 35, 'carries': 23, 'sacks_suffered': 2},
        {'team': 'D', 'opponent_team': 'C', 'week': 1,
         'attempts': 32, 'carries': 26, 'sacks_suffered': 2},
    ])
    schedule = pd.DataFrame([
        {'week': 1, 'game_type': 'REG', 'home_team': 'A', 'away_team': 'B', 'overtime': 1},
        {'week': 1, 'game_type': 'REG', 'home_team': 'C', 'away_team': 'D', 'overtime': 0},
    ])
    original_stats = loaders.nflreadpy.load_team_stats
    original_sched = loaders.nflreadpy.load_schedules
    try:
        loaders.nflreadpy.load_team_stats = (
            lambda years, summary_level='week': _FakeFrame(stats))
        loaders.nflreadpy.load_schedules = lambda years: _FakeFrame(schedule)
        loaders.load_team_pace.clear()
        loaders.load_schedule.clear()
        loaders._overtime_team_weeks.clear()
        pace = loaders.load_team_pace(2099)

        loaders.load_team_weekly_plays.clear()
        plays = loaders.load_team_weekly_plays(2099)
    finally:
        loaders.nflreadpy.load_team_stats = original_stats
        loaders.nflreadpy.load_schedules = original_sched
    # 70 combined plays scaled by OVERTIME_PLAY_DISCOUNT (60/70) lands on the
    # same 60 the non-OT pair already shows - not still-inflated.
    for team in ('A', 'B', 'C', 'D'):
        assert abs(pace.loc[team, 'off_pace'] - 60.0) < 1e-9
        assert abs(pace.loc[team, 'def_pace'] - 60.0) < 1e-9
        row = plays.loc[plays['team'] == team].iloc[0]
        assert abs(row['plays'] - 60.0) < 1e-9


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
