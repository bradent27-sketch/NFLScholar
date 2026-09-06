"""market_book_stat_lines: the un-blended per-(player, stat, book) companion
to market_stat_lines that the Weekly Rankings decomposition's 'Market lines'
tab is built from.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

from data.odds_projections import market_book_stat_lines, market_stat_lines  # noqa: E402
from data.odds_weekly import weekly_market_book_lines  # noqa: E402


def _props():
    return pd.DataFrame([
        {'provider': 'DraftKings', 'player': "De'Von Achane", 'player_key': 'devonachane',
         'team': 'MIA', 'position': 'RB', 'market': 'rushing_yards', 'market_raw': 'x',
         'scorable': True, 'line': 64.5, 'period': 'game', 'source_id': 'd1'},
        {'provider': 'PrizePicks', 'player': "De'Von Achane", 'player_key': 'devonachane',
         'team': 'MIA', 'position': 'RB', 'market': 'rushing_yards', 'market_raw': 'x',
         'scorable': True, 'line': 66.5, 'period': 'game', 'source_id': 'p1'},
        {'provider': 'PrizePicks', 'player': "De'Von Achane", 'player_key': 'devonachane',
         'team': 'MIA', 'position': 'RB', 'market': 'receptions', 'market_raw': 'x',
         'scorable': True, 'line': 3.5, 'period': 'game', 'source_id': 'p2'},
        {'provider': 'Underdog', 'player': "De'Von Achane", 'player_key': 'devonachane',
         'team': 'MIA', 'position': 'RB', 'market': 'receiving_yards', 'market_raw': 'x',
         'scorable': True, 'line': 27.5, 'period': 'game', 'source_id': 'u1'},
        # non-standard shade: filtered out before it can be shown as a book line
        {'provider': 'PrizePicks (demon)', 'player': "De'Von Achane", 'player_key': 'devonachane',
         'team': 'MIA', 'position': 'RB', 'market': 'rushing_yards', 'market_raw': 'x',
         'scorable': False, 'line': 75.5, 'period': 'game', 'source_id': 'p3'},
    ])


def test_one_row_per_player_stat_book_scorable_only():
    bl = market_book_stat_lines(_props(), season_only=False)
    assert list(bl.columns) == ['player_key', 'player', 'team', 'position',
                                'market', 'provider', 'line', 'p_over', 'implied_mean']
    # 4 scorable (player, stat, book) combos; the (demon) row is gone
    assert len(bl) == 4
    assert 'PrizePicks (demon)' not in set(bl['provider'])
    ry = bl[bl['market'] == 'rushing_yards'].set_index('provider')['line'].to_dict()
    assert ry == {'DraftKings': 64.5, 'PrizePicks': 66.5}
    # No prices in this fixture -> implied_mean falls back to the raw line
    # for a yardage stat (MEDIAN_TO_MEAN has no entry), so it equals `line`.
    im = bl[bl['market'] == 'rushing_yards'].set_index('provider')['implied_mean'].to_dict()
    assert im == {'DraftKings': 64.5, 'PrizePicks': 66.5}


def test_average_matches_market_stat_lines_consensus():
    """The tab shows these rows next to market_stat_lines' weighted consensus
    as the 'Average' column - the two must be the same view of one board."""
    bl = market_book_stat_lines(_props(), season_only=False)
    wide = market_stat_lines(_props(), season_only=False).iloc[0]
    # DraftKings priced rushing_yards too, so PrizePicks is dropped from the
    # blended consensus (fallback-only); the number is DK's raw 64.5 (no
    # prices in the fixture -> no devig shift on a yardage line).
    assert abs(float(wide['rushing_yards']) - 64.5) < 1e-9
    assert set(bl[bl['market'] == 'rushing_yards']['line']) == {64.5, 66.5}


def test_empty_and_unusable_inputs_degrade_quietly():
    cols = ['player_key', 'player', 'team', 'position', 'market', 'provider',
            'line', 'p_over', 'implied_mean']
    assert market_book_stat_lines(pd.DataFrame()).columns.tolist() == cols
    assert market_book_stat_lines(None).empty
    assert weekly_market_book_lines(pd.DataFrame()).empty
    # all rows non-scorable
    junk = _props().assign(scorable=False)
    assert market_book_stat_lines(junk, season_only=False).empty


def test_weekly_wrapper_is_game_period():
    """weekly_market_book_lines must not silently drop weekly ('game') rows
    the way season_only=True would."""
    out = weekly_market_book_lines(_props())
    assert len(out) == 4
