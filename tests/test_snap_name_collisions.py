"""Snap-count joins (2026-10-26/27): two players with one name in the same week are read by team, not averaged (Justin
Jefferson MIN WR 89% read 44.5% in 2026 week 5 next to a CLE linebacker of the same name), and a name the snap counts
spell differently is found by PFR id (Bam Knight / "Zonovan Knight")."""
import numpy as np
import pandas as pd

import data.loaders as loaders
from data.loaders import apply_snap_name_collisions, snap_name_collisions
from data.utils import clean_name_exact


def _raw():
    return pd.DataFrame({
        'player': ['Justin Jefferson', 'Justin Jefferson', 'Justin Jefferson', 'Garrett Wilson'],
        'pfr_player_id': ['JeffJu00', 'JeffJu01', 'JeffJu00', 'WilsGa00'],
        'team': ['MIN', 'CLE', 'MIN', 'NYJ'],
        'week': [5, 5, 6, 5],
        'game_type': 'REG',
        'offense_pct': [0.89, 0.0, 0.95, 0.96],
        'defense_pct': [0.0, 0.0, 0.0, 0.0],
    })


def _stats():
    s = pd.DataFrame({
        'player_display_name': ['Justin Jefferson', 'Justin Jefferson', 'Justin Jefferson', 'Garrett Wilson'],
        'recent_team': ['MIN', None, 'MIN', 'NYJ'],
        'week': [5.0, 5.0, 6.0, 5.0],
        'weekly_snap_pct': [44.5, 44.5, 95.0, 96.0],  # what the name-keyed merge gives
    })
    s['exact_name'] = clean_name_exact(s['player_display_name'])
    return s


def test_collision_week_is_read_by_team(monkeypatch):
    monkeypatch.setattr(loaders, '_load_snap_counts_raw', lambda year: _raw())
    coll = snap_name_collisions(2026)
    assert sorted(coll['team']) == ['CLE', 'MIN'] and set(coll['week']) == {5.0}
    out = apply_snap_name_collisions(_stats(), coll, 'recent_team')
    assert np.isclose(out.loc[0, 'weekly_snap_pct'], 89.0)
    # no team on the stats side: left as the merge gave it; other weeks and other players untouched
    assert out.loc[1, 'weekly_snap_pct'] == 44.5
    assert out.loc[2, 'weekly_snap_pct'] == 95.0 and out.loc[3, 'weekly_snap_pct'] == 96.0


def test_no_collisions_changes_nothing(monkeypatch):
    raw = _raw()
    raw = raw[raw['team'].ne('CLE')]
    monkeypatch.setattr(loaders, '_load_snap_counts_raw', lambda year: raw)
    coll = snap_name_collisions(2026)
    assert coll.empty
    stats = _stats()
    pd.testing.assert_frame_equal(apply_snap_name_collisions(stats, coll, 'recent_team'), stats)
    monkeypatch.setattr(loaders, '_load_snap_counts_raw', lambda year: pd.DataFrame())
    assert snap_name_collisions(2026).empty


def test_game_team_wins_when_the_merged_team_is_blank(monkeypatch):
    # the roster merge blanked Michael Thomas's (NO WR, 2022) visible team; the raw feed's game_team still has it
    monkeypatch.setattr(loaders, '_load_snap_counts_raw', lambda year: _raw())
    stats = _stats()
    stats['game_team'] = ['MIN', 'CLE', 'MIN', 'NYJ']
    stats.loc[0, 'recent_team'] = None
    out = apply_snap_name_collisions(stats, snap_name_collisions(2026), 'recent_team')
    assert np.isclose(out.loc[0, 'weekly_snap_pct'], 89.0) and np.isclose(out.loc[1, 'weekly_snap_pct'], 0.0)


def test_same_name_same_team_is_left_out(monkeypatch):
    raw = _raw()
    raw.loc[1, 'team'] = 'MIN'  # two Justin Jeffersons on one team that week: team cannot tell them apart
    monkeypatch.setattr(loaders, '_load_snap_counts_raw', lambda year: raw)
    assert snap_name_collisions(2026).empty


def test_pfr_id_fills_a_name_the_snap_counts_spell_differently(monkeypatch):
    # Bam Knight (ARI) is "Zonovan Knight" in the snap counts; the roster's pfr_id is the same PFR id
    raw = pd.DataFrame({'player': ['Zonovan Knight', 'Zonovan Knight'], 'pfr_player_id': ['KnigZo00', 'KnigZo00'],
                        'team': ['ARI', 'ARI'], 'week': [6, 7], 'game_type': 'REG',
                        'offense_pct': [0.44, 0.53], 'defense_pct': [0.0, 0.0]})
    monkeypatch.setattr(loaders, '_load_snap_counts_raw', lambda year: raw)
    by_id = loaders.snap_pct_by_pfr_id(2026)
    stats = pd.DataFrame({'player_display_name': ['Bam Knight', 'Bam Knight', 'Garrett Wilson'],
                          'pfr_id': ['KnigZo00', 'KnigZo00', 'WilsGa00'], 'week': [6.0, 7.0, 7.0],
                          'weekly_snap_pct': [np.nan, np.nan, 96.0]})
    out = loaders.fill_snap_pct_by_pfr_id(stats, by_id)
    assert np.allclose(out['weekly_snap_pct'], [44.0, 53.0, 96.0])
    # a row the name match already found is never overwritten
    stats.loc[0, 'weekly_snap_pct'] = 40.0
    assert loaders.fill_snap_pct_by_pfr_id(stats, by_id).loc[0, 'weekly_snap_pct'] == 40.0
