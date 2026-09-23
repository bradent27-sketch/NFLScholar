"""weekly_market_projection: the weekly-board counterpart of Draft HQ's
season-long score_market_lines call.

Found 2026-09-22 from a live Week 3 board: Josh Allen's row showed a blank
Market Coverage in the Weekly Rankings table despite DraftKings/Underdog/
PrizePicks all pricing him fully. Root cause was that this weekly path never
passed a `positions` fallback into score_market_lines the way
ui/tabs/draft_hq.py's season-long pull already does - so a player whose
first-sorted provider row happened to omit `position` fell outside
_points_weighted_coverage's KEY_STATS lookup and computed Coverage as NaN,
even though every stat was actually priced.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

from data.draft_board import DEFAULT_SCORING  # noqa: E402
from data.odds_weekly import weekly_market_projection  # noqa: E402


def _props_missing_position():
    # Mirrors a real book export where one provider's row has no position
    # tag - market_stat_lines' per-player `position` is the FIRST row after
    # sorting by provider, so DraftKings (alphabetically first) landing here
    # blank is what actually starves Coverage even though every stat is
    # priced by someone.
    rows = []
    for provider, market, line, position in (
        ('DraftKings', 'passing_yards', 245.5, ''),
        ('DraftKings', 'passing_tds', 1.5, ''),
        ('Underdog', 'passing_yards', 244.5, 'QB'),
        ('PrizePicks', 'rushing_yards', 35.5, 'QB'),
    ):
        rows.append({
            'provider': provider, 'player': 'Josh Allen', 'player_key': 'joshallen',
            'team': 'BUF', 'position': position, 'market': market, 'market_raw': 'x',
            'scorable': True, 'line': line, 'period': 'game', 'source_id': f'{provider}-{market}',
        })
    return pd.DataFrame(rows)


def test_coverage_is_not_blank_when_the_board_can_backfill_position():
    board = pd.DataFrame([{'Player': 'Josh Allen', 'Pos': 'QB'}])
    scored, meta = weekly_market_projection(_props_missing_position(), DEFAULT_SCORING, board=board)
    assert meta['players'] == 1
    assert pd.notna(scored.iloc[0]['Coverage'])
    assert scored.iloc[0]['Coverage'] > 0


def test_coverage_stays_blank_without_a_position_capable_board():
    # No board at all: same starved-position row, no fallback available -
    # documents the boundary this fix actually moves rather than a case it
    # is expected to fix.
    scored, meta = weekly_market_projection(_props_missing_position(), DEFAULT_SCORING, board=None)
    assert meta['players'] == 1
    assert pd.isna(scored.iloc[0]['Coverage'])


def test_empty_props_returns_empty_frame():
    scored, meta = weekly_market_projection(pd.DataFrame(), {}, board=None)
    assert scored.empty
    assert meta == {'players': 0}
