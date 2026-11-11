"""data.pff_alignment.detect_export_week: which played week a weekly receiving_summary export covers (audit D4)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pff_alignment import detect_export_week  # noqa: E402


def _box():
    rows = []
    for week in (1, 2, 3):
        for i in range(30):
            rows.append({'player_display_name': f'Player {i}', 'week': week, 'receiving_yards': float(i * week + week)})
    return pd.DataFrame(rows)


def _export(week):
    return pd.DataFrame({'player': [f'Player {i}' for i in range(30)], 'yards': [float(i * week + week) for i in range(30)]})


def test_detects_the_week_the_yards_match():
    week, share, n = detect_export_week(_export(2), _box(), 'player_display_name')
    assert week == 2 and share == 1.0 and n == 30


def test_too_few_common_players_or_missing_columns_return_none():
    assert detect_export_week(_export(2).head(5), _box(), 'player_display_name')[0] is None
    assert detect_export_week(_export(2).drop(columns='yards'), _box(), 'player_display_name') == (None, 0.0, 0)
    assert detect_export_week(_export(2), _box(), 'no_such_column') == (None, 0.0, 0)
