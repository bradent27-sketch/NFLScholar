"""v2_alignment_recency (shipped 2026-11-16): the weekly PFF alignment mix weights each game by decay ** games_ago."""
import pandas as pd

import data.pff_alignment as pa
import data.weekly_projections as wp


def _rows(weeks, slots):
    n = len(weeks)
    frame = pd.DataFrame({
        'identity_key': ['p1'] * n, 'source_week': weeks, 'source_path': [f'w{w}' for w in weeks],
        'position': ['WR'] * n, '_alignment_weight': [40.0] * n, '_slot_rate': slots,
        '_wide_rate': [1.0 - s for s in slots], '_inline_rate': [0.0] * n,
        'source_regular_season': [True] * n, 'source_confidence': ['test'] * n})
    for col in ('slot_routes', 'slot_targets', 'slot_receptions', 'slot_yards', 'slot_touchdowns', 'slot_snaps',
                'wide_snaps', 'inline_snaps', 'routes'):
        frame[col] = 0.0
    return frame


def _slot(profiles):
    return float(profiles['slot_alignment_rate'].iloc[0])


def _aggregate(rows, decay):
    return pa._aggregate_profiles(rows, source_kind='weekly_archive', source_year=2025, source_time_valid=True,
                                  source_notes='test', recency_decay=decay)


def test_flag_is_shipped_with_per_position_decays():
    assert 'v2_alignment_recency' in wp.MODEL_FEATURES and 'v2_alignment_recency' in wp.DEFAULT_FEATURES
    assert wp.ALIGNMENT_RECENCY_DECAY == {'WR': 0.80, 'TE': 0.85}
    assert all(0.5 < d < 1.0 for d in wp.ALIGNMENT_RECENCY_DECAY.values())


def test_a_position_dict_applies_only_to_its_positions():
    rows = _rows([1, 2, 3, 4], [0.2, 0.2, 0.8, 0.8])                    # a WR row
    assert abs(_slot(_aggregate(rows, {'WR': 0.80})) - _slot(_aggregate(rows, 0.80))) < 1e-12
    assert abs(_slot(_aggregate(rows, {'TE': 0.85})) - 0.5) < 1e-9      # WR not in the dict: season to date


def test_decay_leans_toward_the_latest_games_and_none_is_season_to_date():
    rows = _rows([1, 2, 3, 4], [0.2, 0.2, 0.8, 0.8])          # moved into the slot two games ago
    assert abs(_slot(_aggregate(rows, None)) - 0.5) < 1e-9
    assert 0.5 < _slot(_aggregate(rows, 0.85)) < _slot(_aggregate(rows, 0.70)) < 0.8
    steady = _rows([1, 2, 3], [0.4, 0.4, 0.4])                # no move: decay changes nothing
    assert abs(_slot(_aggregate(steady, 0.70)) - 0.4) < 1e-9
