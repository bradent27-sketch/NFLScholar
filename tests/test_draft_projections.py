"""
Offline regression tests for a real bug found in Draft HQ's projection
engine (data/draft_projections.py, data/draft_board.py): a player with a
missing or zero Pos Rank was defaulting to rank_idx 0 - the BEST slot on
the volume curve (RB1/WR1-level production, 300+ fantasy points) - instead
of the worst. Confirmed live: several deep-bench/rookie names whose
ECR/ADP came back 0 from FantasyPros' API (see
data.draft_sources._rank_or_unranked's own docstring) rendered as the
board's #1 overall values off exactly this fallback.

Hand-built curve/board fixtures rather than real season files, same
convention as tests/test_matchup_signals.py - what's pinned here is the
fallback direction, which would fail SILENTLY against real data (the board
still renders, the number is just wrong).

Runs two ways: `python tests/test_draft_projections.py` needs nothing but
the app's own dependencies, and `pytest tests/` works if pytest is
installed.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

from data.draft_projections import project_stat_lines  # noqa: E402
from data.draft_board import _rank_uncertainty  # noqa: E402


def _rb_curves(depth=5):
    """A tiny hand-built volume curve: rank 1 is a workhorse, rank `depth`
    is a non-contributor - descending, like a real one."""
    rushing_yards = np.linspace(1400.0, 20.0, depth)
    games = np.linspace(17.0, 8.0, depth)
    return {'RB': {'rushing_yards': rushing_yards, 'games': games}}


def test_a_missing_pos_rank_lands_on_the_worst_curve_slot_not_the_best():
    curves = _rb_curves(depth=5)
    board = pd.DataFrame([
        {'Player': 'Real RB1', 'Pos': 'RB', 'Team': 'DET', 'Pos Rank': 1},
        {'Player': 'Unranked Rookie', 'Pos': 'RB', 'Team': 'BAL', 'Pos Rank': np.nan},
    ])
    out = project_stat_lines(board, curves, rates=pd.DataFrame(), latest_season=2025)
    rb1 = out[out['Player'] == 'Real RB1'].iloc[0]['rushing_yards']
    unranked = out[out['Player'] == 'Unranked Rookie'].iloc[0]['rushing_yards']
    # The worst slot on this fixture's curve is 20.0 yards - the unranked
    # player must land there (or within the curve's own noise floor), never
    # anywhere near the RB1 slot's 1400.
    assert unranked < 100
    assert rb1 > 1000
    assert unranked < rb1


def test_a_zero_pos_rank_is_treated_the_same_as_missing():
    # FantasyPros' API sends 0 (not null) for an unranked player - see
    # data.draft_sources._rank_or_unranked. Confirm 0 gets the same
    # worst-slot treatment as a genuine NaN, not literal rank 0.
    curves = _rb_curves(depth=5)
    board = pd.DataFrame([{'Player': 'Zero Rank Guy', 'Pos': 'RB', 'Team': 'SEA', 'Pos Rank': 0}])
    out = project_stat_lines(board, curves, rates=pd.DataFrame(), latest_season=2025)
    assert out.iloc[0]['rushing_yards'] < 100


def test_a_real_pos_rank_still_indexes_the_curve_normally():
    curves = _rb_curves(depth=5)
    board = pd.DataFrame([{'Player': 'RB3 Guy', 'Pos': 'RB', 'Team': 'KC', 'Pos Rank': 3}])
    out = project_stat_lines(board, curves, rates=pd.DataFrame(), latest_season=2025)
    # rank 3 of 5 -> the middle of a linspace(1400, 20, 5) curve -> 710.
    assert abs(out.iloc[0]['rushing_yards'] - 710.0) < 1.0


def test_rank_uncertainty_widens_not_narrows_for_a_missing_rank():
    board = pd.DataFrame([
        {'Pos': 'RB', 'Pos Rank': 1, 'ECR SD': np.nan, 'ECR Best': np.nan, 'ECR Worst': np.nan},
        {'Pos': 'RB', 'Pos Rank': 40, 'ECR SD': np.nan, 'ECR Best': np.nan, 'ECR Worst': np.nan},
        {'Pos': 'RB', 'Pos Rank': np.nan, 'ECR SD': np.nan, 'ECR Best': np.nan, 'ECR Worst': np.nan},
    ])
    out = _rank_uncertainty(board)
    # The missing-rank player's uncertainty must be at least as wide as the
    # deepest REAL rank on the board, never as tight as the rank-1 player's.
    assert out[2] >= out[1] > out[0]


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
