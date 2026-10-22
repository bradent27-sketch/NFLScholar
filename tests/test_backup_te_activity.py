"""v2_backup_te_activity (candidate 2026-10-22, model audit lead L2, TE part): a TE below the room leader is scaled by how
often backups with his last snap share really play and see targets."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.weekly_projections as wp  # noqa: E402


def test_flag_is_registered_and_not_default():
    assert 'v2_backup_te_activity' in wp.MODEL_FEATURES
    assert 'v2_backup_te_activity' not in wp.DEFAULT_FEATURES


def test_bands_by_last_snap_share_and_room_rank(monkeypatch):
    snaps = pd.DataFrame({'player': ['Lead TE', 'Blocker TE', 'Deep TE', 'Two TE Set', 'Lead TE', 'Blocker TE'],
                          'position': 'TE', 'game_type': 'REG', 'week': [3, 3, 3, 3, 4, 4],
                          'offense_pct': [0.9, 0.5, 0.0, 0.7, 0.88, 0.10]})
    import data.loaders as ld
    monkeypatch.setattr(ld, '_load_snap_counts_raw', lambda year: snaps)
    wp._te_last_snap_shares.cache_clear()
    names = np.array(['Lead TE', 'Blocker TE', 'Deep TE', 'New TE', 'Two TE Set', 'Out TE'], dtype=object)
    teams = np.array(['KC', 'KC', 'KC', 'KC', 'BUF', 'BUF'], dtype=object)
    shares = np.array([0.85, 0.30, 0.10, 0.05, 0.40, 0.90])
    live = np.array([True, True, True, True, True, False])
    out = wp.backup_te_activity_multiplier(names, teams, shares, live, 2099, 5)
    wp._te_last_snap_shares.cache_clear()
    assert out[0] == 1.0                    # room leader untouched
    assert out[1] == 0.53                   # last row week 4 at 10%
    assert out[2] == 0.25                   # special teams only
    assert out[3] == 0.73                   # no snap row yet this season
    assert out[4] == 1.0                    # BUF leader: the OUT TE does not count in the ranking
    assert out[5] == 1.0                    # sidelined rows are left to availability


def test_reads_only_weeks_before_the_target_week(monkeypatch):
    snaps = pd.DataFrame({'player': ['B'], 'position': 'TE', 'game_type': 'REG', 'week': [5], 'offense_pct': [0.9]})
    import data.loaders as ld
    monkeypatch.setattr(ld, '_load_snap_counts_raw', lambda year: snaps)
    wp._te_last_snap_shares.cache_clear()
    assert wp._te_last_snap_shares(2099, 5) == {}
    wp._te_last_snap_shares.cache_clear()


def test_freed_backup_volume_goes_to_the_te1_and_keeps_the_room_total():
    proj = {'targets': np.array([5.0, 2.0, 1.0, 4.0]), 'receptions': np.array([3.5, 1.4, 0.7, 3.0]),
            'receiving_yards': np.array([40.0, 15.0, 8.0, 35.0]), 'receiving_tds': np.array([0.3, 0.1, 0.05, 0.25])}
    teams = np.array(['KC', 'KC', 'KC', 'BUF'], dtype=object)
    activity = np.array([1.0, 0.85, 0.53, 1.0])        # projections already carry the factor
    rank = np.array([1.0, 2.0, 3.0, 1.0])
    before = 5.0 + 2.0 / 0.85 + 1.0 / 0.53             # the KC room's claim without the factor
    scale = wp.transfer_te_backup_volume(proj, teams, activity, rank)
    assert np.isclose(proj['targets'][:3].sum(), before)
    assert np.isclose(proj['receptions'][0] / proj['targets'][0], 0.7)      # TE1 keeps his catch rate
    assert scale[1] == scale[2] == 1.0 and scale[3] == 1.0


def test_discount_only_variant_is_registered_and_not_default():
    assert 'v2_backup_te_discount' in wp.MODEL_FEATURES
    assert 'v2_backup_te_discount' not in wp.DEFAULT_FEATURES
