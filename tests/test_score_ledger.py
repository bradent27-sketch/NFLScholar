"""scripts.score_ledger: model vs. market vs. FantasyPros on a real ledger."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.score_ledger import _score_week  # noqa: E402


_RB_NAMES = ['RB Alpha', 'RB Bravo', 'RB Charlie', 'RB Delta',
            'RB Echo', 'RB Foxtrot', 'RB Golf', 'RB Hotel']
_WR_NAMES = ['WR Alpha', 'WR Bravo', 'WR Charlie', 'WR Delta',
            'WR Echo', 'WR Foxtrot', 'WR Golf', 'WR Hotel']


def _ledger():
    return pd.DataFrame({
        'Player': _RB_NAMES + _WR_NAMES,
        'Pos': ['RB'] * 8 + ['WR'] * 8,
        'Model Proj Pts': [18, 16, 14, 12, 10, 8, 6, 4] + [17, 15, 13, 11, 9, 7, 5, 3],
        'Mkt Market Pts': [17, 15, 15, 11, 9, 9, 5, 3] + [16, 16, 12, 12, 8, 8, 4, 4],
        'FP Proj Pts PPR': [19, 15, 13, 13, 11, 7, 7, 5] + [18, 14, 14, 10, 10, 6, 6, 2],
        'rushing_yards': [80, 70, 60, 50, 40, 30, 20, 10] + [0] * 8,
        'Mkt rushing_yards': [75, 72, 58, 48, 38, 32, 18, 12] + [0] * 8,
        'receiving_yards': [10] * 8 + [90, 80, 70, 60, 50, 40, 30, 20],
        'Mkt receiving_yards': [10] * 8 + [88, 78, 68, 62, 48, 42, 28, 22],
    })


def _stats_df():
    rows = []
    for i, name in enumerate(_RB_NAMES):
        rows.append({'name': name, 'week': 3, 'fantasy_points_ppr': 18 - i * 1.5,
                    'rushing_yards': 80 - i * 9})
    for i, name in enumerate(_WR_NAMES):
        rows.append({'name': name, 'week': 3, 'fantasy_points_ppr': 17 - i * 1.4,
                    'receiving_yards': 90 - i * 9})
    return pd.DataFrame(rows)


def test_score_week_runs_and_scopes_by_position(capsys):
    ok = _score_week(2026, 3, _ledger(), _stats_df(), 'name', 'fantasy_points_ppr')
    assert ok is True
    out = capsys.readouterr().out
    assert '2026 week 3' in out
    assert 'RB' in out and 'WR' in out
    assert 'Model:' in out and 'Market:' in out and 'FantasyPros:' in out
    assert 'rushing_yards' in out and 'receiving_yards' in out


def test_score_week_returns_none_with_no_actuals_for_that_week():
    empty_stats = pd.DataFrame({'name': ['RB Alpha'], 'week': [99], 'fantasy_points_ppr': [10.0]})
    assert _score_week(2026, 3, _ledger(), empty_stats, 'name', 'fantasy_points_ppr') is None
