"""harness_v2 every-game scopes (2026-10-22, backlog L7): every player the board expected to play is scored, and a week
with no box-score line counts as 0 points."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import harness_v2  # noqa: E402


def _board(pts):
    return pd.DataFrame({'Player': list(pts), 'Pos': 'TE', 'Model Proj Pts': list(pts.values()),
                         'Availability': [1.0, 1.0, 0.0, 0.8]})


def test_every_game_pool_keeps_statless_players_and_drops_the_ones_ruled_out():
    base = _board({'Starter': 10.0, 'Blocker': 2.0, 'Out': 0.0, 'Scratch': 1.5})
    pool = harness_v2.every_game_pool(base, base, 'TE', False)
    assert pool == ['Blocker', 'Scratch', 'Starter']           # 'Out' (Availability 0) is not expected to play
    actual = pd.Series({'Starter': 12.0})                       # only the starter has a box-score line
    act = harness_v2.every_game_actual(actual, pool)
    assert act.to_dict() == {'Blocker': 0.0, 'Scratch': 0.0, 'Starter': 12.0}


def test_every_game_scopes_mirror_the_box_score_scopes():
    assert [s[0] for s in harness_v2.EVERY_GAME_SCOPES] == ['EVERY-' + s[0] for s in harness_v2.SCOPES]
    assert harness_v2.expected_players(pd.DataFrame({'Player': ['A']})) == {'A'}   # no column: everyone
