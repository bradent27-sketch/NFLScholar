"""v2_te_fill_in_role (candidate 2026-10-22, the user's hypothesis): a backup TE's games played while a TE now ahead of him
was out do not carry his fill-in role into the weeks after that TE returns."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.loaders as ld  # noqa: E402
import data.weekly_projections as wp  # noqa: E402


def _snaps(rows):
    return pd.DataFrame(rows, columns=['player', 'week', 'offense_pct']).assign(position='TE', game_type='REG')


def test_flag_is_registered_and_not_default():
    assert 'v2_te_fill_in_role' in wp.MODEL_FEATURES
    assert 'v2_te_fill_in_role' not in wp.DEFAULT_FEATURES


def test_fill_in_games_are_left_out_of_the_backups_role(monkeypatch):
    snaps = _snaps([
        ('Starter', 1, 0.90), ('Starter', 2, 0.88),                                        # out weeks 3-4, back now
        ('Backup', 1, 0.30), ('Backup', 2, 0.30), ('Backup', 3, 0.80), ('Backup', 4, 0.80),
        ('Fill Only', 3, 0.40), ('Fill Only', 4, 0.40),                                    # only played while out
        ('Steady', 1, 0.35), ('Steady', 2, 0.35), ('Steady', 3, 0.35), ('Steady', 4, 0.35),
        ('Other Lead', 1, 0.9), ('Other Lead', 2, 0.9), ('Other Lead', 3, 0.9), ('Other Lead', 4, 0.9),
    ])
    monkeypatch.setattr(ld, '_load_snap_counts_raw', lambda year: snaps)
    wp._te_snap_weeks.cache_clear()
    names = np.array(['Starter', 'Backup', 'Fill Only', 'Steady', 'Other Lead'], dtype=object)
    teams = np.array(['KC', 'KC', 'KC', 'BUF', 'BUF'], dtype=object)
    shares = np.array([0.89, 0.55, 0.40, 0.35, 0.90])
    live = np.ones(5, dtype=bool)
    out, rank = wp.te_fill_in_role_multiplier(names, teams, shares, live, 2099, 5)
    wp._te_snap_weeks.cache_clear()
    assert out[0] == 1.0                                       # room leader
    assert np.isclose(out[1], 0.30 / np.mean([0.30, 0.30, 0.80, 0.80]))
    assert out[2] == wp.FILL_IN_ONLY_FACTOR
    assert out[3] == 1.0 and out[4] == 1.0                     # no teammate ahead of him missed a game
    assert list(rank) == [1.0, 2.0, 3.0, 2.0, 1.0]


def test_an_out_teammate_does_not_make_fill_in_games(monkeypatch):
    snaps = _snaps([('Starter', 1, 0.9), ('Backup', 1, 0.3), ('Backup', 2, 0.8)])
    monkeypatch.setattr(ld, '_load_snap_counts_raw', lambda year: snaps)
    wp._te_snap_weeks.cache_clear()
    out, _ = wp.te_fill_in_role_multiplier(np.array(['Starter', 'Backup'], dtype=object), np.array(['KC', 'KC'], dtype=object),
                                           np.array([0.9, 0.5]), np.array([False, True]), 2099, 3)
    wp._te_snap_weeks.cache_clear()
    assert out[1] == 1.0                                       # the starter is still out: the backup's role stands
