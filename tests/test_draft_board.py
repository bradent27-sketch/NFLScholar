"""
Offline tests for data/draft_board.py's tier_by_position - the per-position
significant-cutoff clustering Weekly Rankings' Rank column reuses from the
draft board's own tiering. Runs two ways: `python tests/test_draft_board.py`
needs nothing but the app's own dependencies, and `pytest tests/` works if
pytest is installed.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.draft_board import tier_by_position  # noqa: E402


def test_tiers_split_at_the_real_gap_not_at_a_fixed_bucket_size():
    df = pd.DataFrame({
        'Pos': ['RB'] * 8,
        'Model Proj Pts': [27.2, 24.4, 21.3, 19.3, 12.0, 11.5, 4.0, 3.5],
    })
    tiers = tier_by_position(df, 'Model Proj Pts', pos_col='Pos').tolist()
    # A clear gap sits between 19.3 and 12.0 - everyone above it should
    # share one tier and everyone below it another, not a fixed group-of-N.
    assert tiers[:4] == [tiers[0]] * 4
    assert tiers[4:] == [tiers[4]] * 4
    assert tiers[0] != tiers[4]
    # Tier 1 is always the MOST valuable cluster.
    assert tiers[0] < tiers[4]


def test_tiers_are_computed_independently_per_position():
    df = pd.DataFrame({
        'Pos': ['QB', 'QB', 'QB', 'RB', 'RB', 'RB'],
        'Model Proj Pts': [30.0, 29.0, 28.0, 10.0, 9.0, 8.0],
    })
    tiers = tier_by_position(df, 'Model Proj Pts', pos_col='Pos')
    # Three QBs bunched together and three RBs bunched together, at
    # completely different point levels - each position's own tier scale
    # should not be dragged around by the other position's numbers.
    assert tiers.iloc[0:3].nunique() == 1
    assert tiers.iloc[3:6].nunique() == 1


def test_missing_value_or_no_rows_does_not_raise():
    df = pd.DataFrame({'Pos': ['RB'], 'Model Proj Pts': [np.nan]})
    out = tier_by_position(df, 'Model Proj Pts', pos_col='Pos')
    assert out.isna().all()
    empty = pd.DataFrame({'Pos': [], 'Model Proj Pts': []})
    assert tier_by_position(empty, 'Model Proj Pts', pos_col='Pos').empty
    assert tier_by_position(df, 'not_a_column', pos_col='Pos').isna().all()


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
