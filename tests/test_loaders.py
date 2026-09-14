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

    original = loaders.nflreadpy.load_team_stats
    try:
        loaders.nflreadpy.load_team_stats = (
            lambda years, summary_level='week': _FakeFrame(both_weeks))

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
        loaders.nflreadpy.load_team_stats = original


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
