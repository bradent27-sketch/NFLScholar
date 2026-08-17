"""
Offline tests for two real bugs found in data/draft_sources.py's live
FantasyPros API parsing - both confirmed against a real API response, not
theoretical:

  - _rank_or_unranked: the API sends 0 (not null) for a player with no real
    ECR/ADP on a scoring column, and 0 used to sort before rank 1 - see
    fetch_fantasypros_players' own docstring for the board-breaking result.
  - _first_stats_block: the OpenAPI spec for /nfl/{season}/projections says
    a player's `stats` field is an array, but a live response sent a plain
    object instead - `stats[0]` on that raised `KeyError: 0`, crashing the
    whole Weekly Rankings tab.

Runs two ways: `python tests/test_draft_sources.py` needs nothing but the
app's own dependencies, and `pytest tests/` works if pytest is installed.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

from data.draft_sources import _rank_or_unranked, _first_stats_block  # noqa: E402


def test_zero_rank_becomes_missing_not_literal_rank_zero():
    out = _rank_or_unranked(pd.Series([0, 5, 12]))
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == 5.0 and out.iloc[2] == 12.0


def test_negative_and_null_ranks_are_also_treated_as_missing():
    out = _rank_or_unranked(pd.Series([-1, None, float('nan'), 7]))
    assert out.isna().sum() == 3
    assert out.iloc[3] == 7.0


def test_a_real_rank_of_one_survives():
    # Rank 1 is a real, legitimate value - only <= 0 is the sentinel.
    out = _rank_or_unranked(pd.Series([1]))
    assert out.iloc[0] == 1.0


def test_stats_block_handles_the_documented_list_shape():
    assert _first_stats_block([{'points': 12.5}]) == {'points': 12.5}


def test_stats_block_handles_the_live_dict_shape():
    # The shape that actually crashed - stats sent as a plain object, not
    # wrapped in a list, contrary to the OpenAPI spec.
    assert _first_stats_block({'points': 12.5}) == {'points': 12.5}


def test_stats_block_handles_missing_or_empty_gracefully():
    assert _first_stats_block([]) == {}
    assert _first_stats_block(None) == {}


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
