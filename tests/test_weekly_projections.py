"""
Offline tests for the weekly projection model's compute layer
(data/weekly_projections.py).

Same convention as tests/test_matchup_signals.py: hand-built fixtures, no
network, no real season files - what's pinned here is the vectorized logic
that would fail SILENTLY against real data (a groupby keyed wrong, a
shrinkage formula pulling the wrong direction, a sign flipped on a margin).
Functions that touch the network or another loader (_target_margins_by_team,
_injury_multipliers, _load_pff_receiving, build_weekly_projections itself)
are exercised live instead, by scripts/validate_weekly_projections.py and by
actually opening the Weekly Rankings tab - not here.

Runs two ways: `python tests/test_weekly_projections.py` needs nothing but
the app's own dependencies, and `pytest tests/` works if pytest is installed.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

import data.weekly_projections as wp  # noqa: E402
import data.ourlads_depth_charts as odc  # noqa: E402


def weekly(rows):
    """A minimal weekly-stats frame in the shape load_and_merge_data emits."""
    frame = pd.DataFrame(rows)
    for col in ('targets', 'receptions', 'rushing_attempts', 'rushing_yards',
                'receiving_yards', 'receiving_tds', 'rushing_tds',
                'weekly_snap_pct'):
        if col not in frame.columns:
            frame[col] = 0.0
    if 'season_type' not in frame.columns:
        frame['season_type'] = 'REG'
    if 'position' not in frame.columns:
        frame['position'] = pd.Series(dtype=str)
    return frame


def _ourlads_mhtml(team='ARZ'):
    """Minimal printer-friendly page covering parser/status behavior."""
    return f'''From: <Saved by Blink>
MIME-Version: 1.0
Content-Type: multipart/related; boundary="chart"

--chart
Content-Type: text/html; charset=UTF-8
Content-Location: https://www.ourlads.com/nfldepthcharts/pfdepthchart/{team}

<html><head><title>Arizona Cardinals Depth Chart</title></head><body>
Updated: 08/19/2026 2:16PM ET
<table><tr><th>Pos</th><th>No.</th><th>Player 1</th><th>No.</th><th>Player 2</th></tr>
<tr><td>LWR</td><td>1</td><td><a href="https://www.ourlads.com/nfldepthcharts/player/1/" class="lc_purple">Rookie, Wide 26/1</a></td><td>2</td><td><a href="https://www.ourlads.com/nfldepthcharts/player/2/">Reserve, Wide CF26</a></td></tr>
<tr><td>SWR</td><td>3</td><td><a href="https://www.ourlads.com/nfldepthcharts/player/3/">Slot, Wide U/NE</a></td><td>4</td><td></td></tr>
<tr><td>QB</td><td>9</td><td><a href="https://www.ourlads.com/nfldepthcharts/player/4/" class="lc_red">Penix Jr., Michael 24/1</a></td><td>8</td><td><a href="https://www.ourlads.com/nfldepthcharts/player/5/">Tagovailoa, Tua CC/Mia</a></td></tr>
<tr><td>RB</td><td>4</td><td><a href="https://www.ourlads.com/nfldepthcharts/player/6/">Back, New 26/1</a></td><td>5</td><td><a href="https://www.ourlads.com/nfldepthcharts/player/7/">Back, Reserve SF26</a></td></tr>
<tr><td>TE</td><td>85</td><td><a href="https://www.ourlads.com/nfldepthcharts/player/8/">End, Tight 25/2</a></td><td>84</td><td></td></tr>
</table></body></html>
--chart--
'''.encode('utf-8')


# --- local Ourlads preseason depth-chart import -----------------------------

def test_ourlads_mhtml_parser_preserves_formation_and_inactive_status():
    chart, report = odc.parse_ourlads_depth_chart(_ourlads_mhtml(), 'Arizona.mhtml')
    assert report['error'] == '' and report['team'] == 'ARI'
    assert set(chart.loc[chart['position'].eq('WR'), 'position_label']) == {'LWR', 'SWR'}
    tua = chart.loc[chart['player'].eq('Tua Tagovailoa')].iloc[0]
    penix = chart.loc[chart['player'].eq('Michael Penix Jr.')].iloc[0]
    assert tua['source_slot'] == 2 and not tua['is_inactive']
    assert penix['is_listed_starter'] and penix['is_inactive']
    assert penix['status_class'] == 'lc_red'
    assert chart.loc[chart['position'].eq('RB'), 'position_occurrence'].eq(0).all()


def test_ourlads_signal_retains_first_qb_and_red_source_status_for_current_availability_layer():
    chart, _ = odc.parse_ourlads_depth_chart(_ourlads_mhtml(), 'Arizona.mhtml')
    roster = pd.DataFrame([
        {'name': 'Michael Penix Jr.', 'team': 'ARI', 'position': 'QB'},
        {'name': 'Tua Tagovailoa', 'team': 'ARI', 'position': 'QB'},
        {'name': 'New Back', 'team': 'ARI', 'position': 'RB'},
        {'name': 'Reserve Back', 'team': 'ARI', 'position': 'RB'},
        {'name': 'Wide Rookie', 'team': 'ARI', 'position': 'WR'},
        {'name': 'Wide Slot', 'team': 'ARI', 'position': 'WR'},
        {'name': 'Tight End', 'team': 'ARI', 'position': 'TE'},
    ])
    signal = odc.build_ourlads_projection_signal(chart, roster, 'name', 'team')
    assert len(signal['qb_starters']) == 1
    assert signal['qb_starters'].iloc[0]['matched_player'] == 'Michael Penix Jr.'
    assert signal['qb_starters'].iloc[0]['source_is_inactive']
    assert 'current availability must confirm' in signal['qb_starters'].iloc[0]['source_status_warning']
    assert set(signal['skill_roles']['matched_player']) >= {'New Back', 'Wide Rookie', 'Wide Slot'}


def test_ourlads_role_floor_only_supports_new_or_thin_role_evidence():
    roles = pd.DataFrame([
        {'team': 'KC', 'position': 'RB', 'position_label': 'RB', 'source_row': 1,
         'source_slot': 1, 'position_occurrence': 0, 'matched_player_key': 'New Lead'},
        {'team': 'KC', 'position': 'RB', 'position_label': 'RB', 'source_row': 1,
         'source_slot': 2, 'position_occurrence': 0, 'matched_player_key': 'New Reserve'},
        {'team': 'KC', 'position': 'WR', 'position_label': 'LWR', 'source_row': 2,
         'source_slot': 1, 'position_occurrence': 0, 'matched_player_key': 'Outside Wide'},
        {'team': 'KC', 'position': 'WR', 'position_label': 'SWR', 'source_row': 3,
         'source_slot': 1, 'position_occurrence': 0, 'matched_player_key': 'Slot Wide'},
    ])
    rb_share, rb_used, rb_rank, rb_floor, _ = wp.apply_ourlads_preseason_role_floor(
        np.array([0.12, 0.05, 0.35]), np.array([np.nan, np.nan, 0.35]),
        np.array(['', '', 'KC']), np.array(['KC', 'KC', 'KC']),
        np.array(['New Lead', 'New Reserve', 'Established Same-Team']), 'RB', roles)
    assert rb_share.tolist() == [0.55, 0.20, 0.35]
    assert rb_used.tolist() == [True, True, False]
    assert rb_rank.tolist()[:2] == [1.0, 2.0] and rb_floor.tolist()[:2] == [0.55, 0.20]

    wr_share, wr_used, _, _, labels = wp.apply_ourlads_preseason_role_floor(
        np.array([0.10, 0.10]), np.array([np.nan, np.nan]), np.array(['', '']),
        np.array(['KC', 'KC']), np.array(['Outside Wide', 'Slot Wide']), 'WR', roles)
    assert wr_used.all() and wr_share.tolist() == [0.45, 0.45]
    assert labels.tolist() == ['LWR', 'SWR']
    assert max(wr_share) < 1.0  # listed formations are not 100%-snap claims


def test_buried_veteran_dock_reads_ourlads_slot_rank_not_team_depth():
    # The bug this pins (found 2026-08-30 on the real 2026 board): Ourlads
    # ranks WR/TE PER ALIGNMENT SLOT, so Marquise Brown (charted RWR-2 in
    # PHI) and Troy Franklin (SWR-2 in DEN) are slot rank 2, not "WR4/5".
    # The first version of the dock checked rank >= 4 and never fired.
    prior = np.array([0.55, 0.55, 0.80, 0.10, 0.55, 0.55])
    share = np.array([0.47, 0.47, 0.80, 0.47, 0.15, 0.47])
    rank = np.array([2.0, 3.0, 1.0, 2.0, 2.0, np.nan])  # Ourlads slot ranks
    out, applied = wp.apply_buried_veteran_dock(share, prior, rank, depth_chart_decay=1.0)

    # [0] proven vet, slot rank 2, real current role -> half his share
    assert applied[0] and abs(out[0] - 0.47 * wp.RECEIVER_BURIED_VET_KEEP_FRACTION) < 1e-9
    # [1] proven vet, slot rank 3+ -> pulled to the deep-bench cutoff
    assert applied[1] and abs(out[1] - wp.RECEIVER_DEPTH_CUTOFF_SHARE_CAP) < 1e-9
    # [2] slot rank 1 = that slot's listed starter -> never docked
    assert not applied[2] and abs(out[2] - 0.80) < 1e-9
    # [3] thin prior (0.10) -> not a "proven" vet -> untouched
    assert not applied[3] and abs(out[3] - 0.47) < 1e-9
    # [4] proven but the model already has him at 0.15 -> nothing to dock
    assert not applied[4] and abs(out[4] - 0.15) < 1e-9
    # [5] no chart rank -> untouched
    assert not applied[5] and abs(out[5] - 0.47) < 1e-9


def test_buried_veteran_dock_fades_with_depth_chart_decay_and_is_inert_in_season():
    prior, share, rank = np.array([0.55]), np.array([0.40]), np.array([2.0])
    # Half decay -> half the pull toward 0.40*0.5=0.20: 0.40 + 0.5*(0.20-0.40) = 0.30
    out_half, applied_half = wp.apply_buried_veteran_dock(share, prior, rank, depth_chart_decay=0.5)
    assert applied_half[0] and abs(out_half[0] - 0.30) < 1e-9
    # No pull left (in-season) -> exact no-op
    out_off, applied_off = wp.apply_buried_veteran_dock(share, prior, rank, depth_chart_decay=0.0)
    assert not applied_off[0] and abs(out_off[0] - 0.40) < 1e-9


def test_buried_veteran_dock_te_slot_is_one_deeper_than_wr():
    # A charted TE-2 is often the receiving TE (TE-1 blocks). Passing the TE
    # backup slot rank (3) must leave a proven TE-2 alone and only dock TE-3+.
    prior = np.array([0.55, 0.55, 0.55])
    share = np.array([0.45, 0.45, 0.45])
    rank = np.array([2.0, 3.0, 4.0])
    te_slot = wp.RECEIVER_BURIED_VET_BACKUP_SLOT_RANK_TE
    out, applied = wp.apply_buried_veteran_dock(share, prior, rank, 1.0, backup_slot_rank=te_slot)
    assert not applied[0] and abs(out[0] - 0.45) < 1e-9          # TE-2: untouched
    assert applied[1] and abs(out[1] - 0.45 * wp.RECEIVER_BURIED_VET_KEEP_FRACTION) < 1e-9  # TE-3: half
    assert applied[2] and abs(out[2] - wp.RECEIVER_DEPTH_CUTOFF_SHARE_CAP) < 1e-9           # TE-4: hard cap
    # Same players under the WR default (slot 2) WOULD dock the rank-2 one.
    out_wr, applied_wr = wp.apply_buried_veteran_dock(share, prior, rank, 1.0)
    assert applied_wr[0]


def test_ourlads_starter_overlay_adds_only_a_uniquely_verified_starter():
    chart, _ = odc.parse_ourlads_depth_chart(_ourlads_mhtml(), 'Arizona.mhtml')
    # New Back is absent from the current roster but exists uniquely in the
    # prior-year history. Reserve Back is not a chart starter and must stay out.
    roster = pd.DataFrame([
        {'name': 'Tua Tagovailoa', 'team': 'ARI', 'position': 'QB'},
        {'name': 'Rookie Wide', 'team': 'ARI', 'position': 'WR'},
    ])
    prior = weekly([
        {'name': 'New Back', 'team': 'SEA', 'position': 'RB', 'week': 1},
        {'name': 'Reserve Back', 'team': 'SEA', 'position': 'RB', 'week': 1},
    ])
    overlaid, changes, warnings = odc.apply_ourlads_starter_roster_overlay(
        chart[chart['position'].eq('RB')], roster, 'name', 'team', prior, 'name', 'team')
    assert not warnings
    assert any(change['player'] == 'New Back' and change['action'] == 'added verified missing starter'
               for change in changes)
    assert 'New Back' in set(overlaid['name'])
    assert 'Reserve Back' not in set(overlaid['name'])


def test_preseason_qb1_manual_selection_beats_ourlads_then_ourlads_beats_old_incumbent():
    current = pd.DataFrame([
        {'name': 'Old Incumbent', 'team': 'KC', 'position': 'QB'},
        {'name': 'New Starter', 'team': 'KC', 'position': 'QB'},
    ])
    prior = weekly([
        {'name': 'Old Incumbent', 'team': 'OLD', 'position': 'QB', 'week': 1, 'weekly_snap_pct': 95.0},
        {'name': 'New Starter', 'team': 'NEW', 'position': 'QB', 'week': 1, 'weekly_snap_pct': 10.0},
    ])
    chart_qb = pd.DataFrame([{
        'team': 'KC', 'matched_player_key': 'New Starter', 'matched_player': 'New Starter',
        'source_slot': 1, 'status_class': '',
    }])
    imported = wp.resolve_preseason_qb1s(
        current, 'name', 'team', prior, 'name', 'team', 2026,
        overrides=pd.DataFrame(columns=wp.QB1_OVERRIDE_COLUMNS), ourlads_qb1s=chart_qb)
    assert imported['selected'][('KC', 'newstarter')] == 'ourlads_depth_chart'
    manual = wp.resolve_preseason_qb1s(
        current, 'name', 'team', prior, 'name', 'team', 2026,
        overrides=pd.DataFrame([{'year': 2026, 'team': 'KC', 'player': 'Old Incumbent'}]),
        ourlads_qb1s=chart_qb)
    assert manual['selected'][('KC', 'oldincumbent')] == 'manual_override'


# --- no-leakage week filters --------------------------------------------

def test_played_weeks_before_excludes_target_week_and_later():
    df = weekly([{'week': w} for w in (1, 2, 3, 4)])
    out = wp._played_weeks_before(df, as_of_week=3)
    assert sorted(out['week']) == [1, 2]


def test_played_weeks_before_excludes_week_zero_placeholder_rows():
    # A roster-only season (no games played yet) leaves week=0 placeholder
    # rows - these must never count as "history".
    df = weekly([{'week': 0}, {'week': 1}])
    out = wp._played_weeks_before(df, as_of_week=5)
    assert list(out['week']) == [1]


def test_all_played_weeks_keeps_everything_real():
    df = weekly([{'week': w} for w in (1, 5, 18)])
    assert sorted(wp._all_played_weeks(df)['week']) == [1, 5, 18]


# --- season totals / recent rate -----------------------------------------

def test_season_totals_sums_per_player_and_counts_distinct_games():
    df = weekly([
        {'name': 'A', 'week': 1, 'team': 'KC', 'position': 'WR', 'targets': 5, 'receiving_yards': 40},
        {'name': 'A', 'week': 2, 'team': 'KC', 'position': 'WR', 'targets': 7, 'receiving_yards': 60},
        {'name': 'B', 'week': 1, 'team': 'BUF', 'position': 'WR', 'targets': 3, 'receiving_yards': 20},
    ])
    out = wp._season_totals(df, 'name', 'team', 'WR', ['targets', 'receiving_yards'])
    a = out[out['name'] == 'A'].iloc[0]
    assert a['Games'] == 2 and a['targets'] == 12 and a['receiving_yards'] == 100 and a['Team'] == 'KC'


def test_season_totals_uses_most_recent_team_after_a_trade():
    df = weekly([
        {'name': 'A', 'week': 1, 'team': 'KC', 'position': 'WR', 'targets': 5},
        {'name': 'A', 'week': 5, 'team': 'BUF', 'position': 'WR', 'targets': 5},
    ])
    out = wp._season_totals(df, 'name', 'team', 'WR', ['targets'])
    assert out.iloc[0]['Team'] == 'BUF'


# --- quality-adjusted matchup / weighted own-history rate -----------------

def test_quality_adjusted_matchup_centers_on_one_and_favors_the_tougher_defense():
    # Player A faces KC (allows him half his normal level) and BUF (allows
    # him his normal level) once each; a league-average defense should read
    # 1.0 and KC (the tougher matchup) should read below BUF.
    df = weekly([
        {'name': 'A', 'week': 1, 'position': 'WR', 'opponent_team': 'KC', 'receiving_yards': 40},
        {'name': 'A', 'week': 2, 'position': 'WR', 'opponent_team': 'BUF', 'receiving_yards': 80},
        {'name': 'A', 'week': 3, 'position': 'WR', 'opponent_team': 'BUF', 'receiving_yards': 80},
    ])
    out = wp.build_quality_adjusted_matchup(df, 'name', ['receiving_yards'], as_of_week=4)
    assert out.loc['KC', 'receiving_yards'] < out.loc['BUF', 'receiving_yards']


def test_quality_adjusted_matchup_empty_without_opponent_column():
    df = weekly([{'name': 'A', 'week': 1, 'position': 'WR', 'receiving_yards': 40}])
    assert wp.build_quality_adjusted_matchup(df, 'name', ['receiving_yards'], as_of_week=2).empty


def test_qb_passing_matchup_uses_one_team_game_not_replacement_qb_rows():
    # Each offense totals only 100 yards against HOU and 200 against KC, so
    # HOU should look tough. A 75-yard backup whose own two-game mean is
    # only 37.5 used to flip the player-row estimator in the wrong direction.
    rows = []
    for team in ('A', 'B'):
        rows.extend([
            {'name': f'{team} Starter', 'team': team, 'opponent_team': 'HOU', 'week': 1,
             'position': 'QB', 'passing_yards': 25.0},
            {'name': f'{team} Backup', 'team': team, 'opponent_team': 'HOU', 'week': 1,
             'position': 'QB', 'passing_yards': 75.0},
            {'name': f'{team} Starter', 'team': team, 'opponent_team': 'KC', 'week': 2,
             'position': 'QB', 'passing_yards': 200.0},
            {'name': f'{team} Backup', 'team': team, 'opponent_team': 'KC', 'week': 2,
             'position': 'QB', 'passing_yards': 0.0},
        ])
    df = weekly(rows)
    player_row = wp.build_quality_adjusted_matchup(df, 'name', ['passing_yards'], as_of_week=3)
    team_game = wp.build_qb_passing_quality_adjusted_matchup(
        df, 'team', ['passing_yards'], as_of_week=3)
    assert player_row.loc['HOU', 'passing_yards'] > 1.0
    assert team_game.loc['HOU', 'passing_yards'] < 1.0
    assert team_game.loc['HOU', 'passing_yards'] < team_game.loc['KC', 'passing_yards']


def test_qb_passing_matchup_drops_an_all_zero_td_baseline_safely():
    df = weekly([
        {'name': 'A QB', 'team': 'A', 'opponent_team': 'HOU', 'week': 1,
         'position': 'QB', 'passing_tds': 0.0},
        {'name': 'A QB', 'team': 'A', 'opponent_team': 'KC', 'week': 2,
         'position': 'QB', 'passing_tds': 0.0},
    ])
    out = wp.build_qb_passing_quality_adjusted_matchup(df, 'team', ['passing_tds'], as_of_week=3)
    assert out.empty


def test_team_game_plays_lookup_normalizes_team_keys_and_columns():
    raw = pd.DataFrame([{'team': 'OAK', 'week': 1, 'plays': 65}])
    out = wp._team_game_plays_lookup(raw)
    assert list(out.columns) == ['_offense', '_week', '_plays']
    assert out.iloc[0]['_offense'] == 'LV'
    assert out.iloc[0]['_plays'] == 65


def test_team_game_quality_profile_removes_pace_volume_when_plays_supplied():
    # Two offenses (A, B) each play HOU once and FAST once. Against HOU the
    # game had 40 total plays and each offense gained 40 receiving yards -
    # par, a 1.0-yard-per-play rate. Against FAST the game had DOUBLE the
    # plays (80) and each offense gained double the yards (80) - the exact
    # same 1.0-yard-per-play rate, just a bigger game. Raw totals alone make
    # FAST look like a much worse matchup than HOU; per-play, they are
    # identical, which is what pace_mult (built from FAST's own higher play
    # count) is supposed to be the ONLY place that volume difference shows
    # up.
    rows = []
    for team in ('A', 'B'):
        rows.append({'name': f'{team} WR', 'team': team, 'opponent_team': 'HOU',
                     'week': 1, 'position': 'WR', 'receiving_yards': 40.0})
        rows.append({'name': f'{team} WR', 'team': team, 'opponent_team': 'FAST',
                     'week': 2, 'position': 'WR', 'receiving_yards': 80.0})
    df = weekly(rows)
    plays_lookup = wp._team_game_plays_lookup(pd.DataFrame([
        {'team': 'A', 'week': 1, 'plays': 40}, {'team': 'B', 'week': 1, 'plays': 40},
        {'team': 'A', 'week': 2, 'plays': 80}, {'team': 'B', 'week': 2, 'plays': 80},
    ]))

    without = wp.build_team_game_quality_adjusted_matchup(
        df, 'team', ['receiving_yards'], as_of_week=3)
    with_plays = wp.build_team_game_quality_adjusted_matchup(
        df, 'team', ['receiving_yards'], as_of_week=3, plays=plays_lookup)

    assert without.loc['FAST', 'receiving_yards'] > without.loc['HOU', 'receiving_yards']
    assert np.isclose(with_plays.loc['HOU', 'receiving_yards'],
                       with_plays.loc['FAST', 'receiving_yards'], atol=1e-6)


def test_team_game_quality_profile_drops_a_game_with_unknown_plays_instead_of_raw_scale():
    rows = [
        {'name': 'A WR', 'team': 'A', 'opponent_team': 'HOU', 'week': 1,
         'position': 'WR', 'receiving_yards': 40.0},
        {'name': 'B WR', 'team': 'B', 'opponent_team': 'HOU', 'week': 2,
         'position': 'WR', 'receiving_yards': 40.0},
    ]
    df = weekly(rows)
    # Only team A's week-1 play count is known; team B's week-2 game has no
    # matching row at all in the plays table.
    plays_lookup = wp._team_game_plays_lookup(pd.DataFrame([{'team': 'A', 'week': 1, 'plays': 40}]))
    out = wp.build_team_game_quality_adjusted_matchup(
        df, 'team', ['receiving_yards'], as_of_week=3, plays=plays_lookup)
    assert not out.empty and np.isfinite(out.loc['HOU', 'receiving_yards'])


def test_role_matchup_pace_normalization_survives_partition_recursion():
    # Same HOU/FAST par-per-play setup as the team-game test above, but
    # role-partitioned - exercises _team_game_quality_profile's recursive
    # partition_keys branch, which must also forward `plays` to each
    # recursive call rather than silently reverting to raw totals.
    rows = []
    for team in ('A', 'B'):
        rows.append({'name': f'{team} WR', 'team': team, 'opponent_team': 'HOU',
                     'week': 1, 'position': 'WR', 'receiving_yards': 40.0})
        rows.append({'name': f'{team} WR', 'team': team, 'opponent_team': 'FAST',
                     'week': 2, 'position': 'WR', 'receiving_yards': 80.0})
    df = weekly(rows)
    roles = {'A WR': 'primary', 'B WR': 'primary'}
    plays_lookup = wp._team_game_plays_lookup(pd.DataFrame([
        {'team': 'A', 'week': 1, 'plays': 40}, {'team': 'B', 'week': 1, 'plays': 40},
        {'team': 'A', 'week': 2, 'plays': 80}, {'team': 'B', 'week': 2, 'plays': 80},
    ]))
    role_tables, _sizes = wp.build_role_matchup(
        df, 'name', 'team', ['receiving_yards'], as_of_week=3, roles=roles, plays=plays_lookup)
    table = role_tables['primary']
    assert np.isclose(table.loc['HOU', 'receiving_yards'],
                       table.loc['FAST', 'receiving_yards'], atol=1e-6)


def _position_team_fixture(pos, stat, split_spot_starter=False):
    """Same team-position totals, optionally split across a spot starter."""
    rows = []
    for team in ('A', 'B'):
        values = [('HOU', 1, 10.0), ('KC', 2, 20.0)]
        for opponent, week, total in values:
            if split_spot_starter:
                rows.extend([
                    {'name': f'{team} Main', 'team': team, 'opponent_team': opponent,
                     'week': week, 'position': pos, stat: 2.0 if week == 1 else total},
                    {'name': f'{team} Spot', 'team': team, 'opponent_team': opponent,
                     'week': week, 'position': pos, stat: total - 2.0 if week == 1 else 0.0},
                ])
            else:
                rows.append({'name': f'{team} Main', 'team': team, 'opponent_team': opponent,
                             'week': week, 'position': pos, stat: total})
    return weekly(rows)


def test_every_projected_stat_uses_a_team_game_defense_profile_not_player_rows():
    # This is the generalized version of the Houston QB regression.  The
    # position-team total is 10 vs HOU and 20 vs KC in both fixtures.  In the
    # split fixture, a spot player owns most of the HOU game and zero in KC;
    # the old individual-baseline estimator reverses the defensive signal.
    # Every projected stat must be partition-invariant instead.
    for pos, stats in wp.OFFENSE_PROJECTION_STATS.items():
        for stat in stats:
            unsplit = wp.build_team_game_quality_adjusted_matchup(
                _position_team_fixture(pos, stat), 'team', [stat], as_of_week=3)
            split = wp.build_team_game_quality_adjusted_matchup(
                _position_team_fixture(pos, stat, split_spot_starter=True),
                'team', [stat], as_of_week=3)
            assert np.allclose(unsplit.sort_index()[stat], split.sort_index()[stat]), (pos, stat)
            assert unsplit.loc['HOU', stat] < 1.0 < unsplit.loc['KC', stat], (pos, stat)


def test_role_profile_is_also_invariant_to_a_spot_starter_split():
    # The forward role overlay is part of the final multiplier, so its table
    # must use the same team-game grain as the broad defense profile.
    roles = {'A Main': 'WR_DEEP', 'A Spot': 'WR_DEEP',
             'B Main': 'WR_DEEP', 'B Spot': 'WR_DEEP'}
    unsplit, unsplit_sizes = wp.build_role_matchup(
        _position_team_fixture('WR', 'receiving_yards'), 'name', 'team',
        ['receiving_yards'], 3, roles)
    split, split_sizes = wp.build_role_matchup(
        _position_team_fixture('WR', 'receiving_yards', split_spot_starter=True), 'name', 'team',
        ['receiving_yards'], 3, roles)
    assert np.allclose(unsplit['WR_DEEP'].sort_index()['receiving_yards'],
                       split['WR_DEEP'].sort_index()['receiving_yards'])
    assert unsplit_sizes == split_sizes


def test_qb_rushing_profile_ignores_negative_kneel_denominators():
    # A QB team can have negative net rushing yards because kneels live in
    # the box score. That is not a valid negative baseline for a forward
    # player rushing projection; it must never produce sign-flipped or
    # infinite defensive ratings.
    rows = []
    for team in ('A', 'B'):
        rows.extend([
            {'name': f'{team} QB', 'team': team, 'opponent_team': 'HOU', 'week': 1,
             'position': 'QB', 'rushing_yards': -4.0},
            {'name': f'{team} QB', 'team': team, 'opponent_team': 'KC', 'week': 2,
             'position': 'QB', 'rushing_yards': 20.0},
        ])
    out = wp.build_team_game_quality_adjusted_matchup(
        weekly(rows), 'team', ['rushing_yards'], as_of_week=3)
    assert np.isfinite(out['rushing_yards']).all()
    assert out.loc['HOU', 'rushing_yards'] < out.loc['KC', 'rushing_yards']


def test_broad_position_profile_keeps_a_zero_output_game_from_the_full_universe():
    # Weekly player files are stat-triggered. The TE group has no row in the
    # HOU games, but the offense still played and that zero is real defensive
    # evidence for a broad TE profile rather than a missing game.
    universe = weekly([
        {'name': 'A QB', 'team': 'A', 'opponent_team': 'HOU', 'week': 1, 'position': 'QB'},
        {'name': 'A TE', 'team': 'A', 'opponent_team': 'KC', 'week': 2, 'position': 'TE', 'targets': 4},
        {'name': 'B QB', 'team': 'B', 'opponent_team': 'HOU', 'week': 1, 'position': 'QB'},
        {'name': 'B TE', 'team': 'B', 'opponent_team': 'KC', 'week': 2, 'position': 'TE', 'targets': 4},
    ])
    tes = universe[universe['position'].eq('TE')]
    games, _ = wp._position_team_games(tes, 'team', ['targets'], game_universe=universe)
    assert len(games) == 4
    assert games.loc[games['_defense'].eq('HOU'), 'targets'].eq(0.0).all()
    evidence = wp._defense_game_evidence(tes, game_universe=universe, team_col='team')
    assert evidence.to_dict() == {'HOU': 1.0, 'KC': 1.0}


def test_historical_game_team_beats_latest_roster_team_and_game_id_fallback():
    # The explicit raw game team wins even when an old game_id uses OAK and
    # the current NFL abbreviation is LV. A pre-contract cached frame still
    # reconstructs the offense from game_id + opponent as a safe fallback.
    explicit = pd.DataFrame([{
        'team': 'DEN', 'game_team': 'LV', 'opponent_team': 'DEN',
        'game_id': '2019_01_DEN_OAK',
    }])
    fallback = pd.DataFrame([{
        'team': 'CIN', 'opponent_team': 'BAL', 'game_id': '2025_01_CLE_BAL',
    }])
    assert wp._historical_game_team(explicit, 'team').iloc[0] == 'LV'
    assert wp._historical_game_team(fallback, 'team').iloc[0] == 'CLE'


def test_qb_passing_efficiency_bypasses_role_tables_when_enabled():
    overall = pd.DataFrame({'passing_yards': [0.90], 'passing_attempts': [1.00]}, index=['HOU'])
    role_tables = {'QB_DOWNFIELD': pd.DataFrame(
        {'passing_yards': [1.30], 'passing_attempts': [0.75]}, index=['HOU'])}
    out = wp._efficiency_matchup(
        overall, role_tables, {('HOU', 'QB_DOWNFIELD'): 100.0},
        np.array(['HOU']), np.array(['QB_DOWNFIELD']),
        'passing_yards', 'passing_attempts', position='QB')
    assert abs(out[0] - 0.90) < 1e-9


def test_prior_season_defense_recency_keeps_a_broad_full_season_baseline():
    weeks = pd.Series([1, 18])
    ordinary = wp.defense_recency_weights(weeks, as_of_week=19)
    prior = wp.defense_recency_weights(
        weeks, as_of_week=19, recency_floor=wp.PRIOR_SEASON_DEFENSE_RECENCY_FLOOR)
    # In-season the finale legitimately carries much more weight.  Across an
    # offseason, the new profile may modestly favor it but cannot act like a
    # trailing-few-games sample.
    assert ordinary.iloc[1] / ordinary.iloc[0] > 10
    assert prior.iloc[1] == 1.0
    assert prior.iloc[1] / prior.iloc[0] < 1.35


def test_weighted_player_rates_weighs_recent_games_more():
    # Same player, an old low game and a recent high game - the weighted
    # rate should sit closer to the RECENT value than a flat average would.
    df = weekly([
        {'name': 'A', 'week': 1, 'position': 'WR', 'opponent_team': 'KC', 'targets': 2},
        {'name': 'A', 'week': 6, 'position': 'WR', 'opponent_team': 'BUF', 'targets': 10},
    ])
    out, _totals = wp._weighted_player_rates(df, 'name', ['targets'], as_of_week=7,
                                    matchup_matrix=pd.DataFrame(), upcoming_opponent={'A': 'MIA'})
    flat_avg = 6.0
    assert out.loc['A', 'targets'] > flat_avg


def test_scheme_matchup_override_is_scoped_to_tight_ends():
    """v2_scheme_matchup ships TE-only (2026-09-02).

    The scheme PREVIEW is still computed for WR - it stays visible in the
    audit panel and testable by a sweep - but only TE is allowed to have its
    projection moved by it, because the cross-season sweep found a clean
    replicating TE win and no startable WR win at any weight. This pins the
    two constants apart so a later edit cannot quietly widen the override
    back to WR by reusing SCHEME_DEFENSE_SUPPORTED_POSITIONS.
    """
    assert set(wp.SCHEME_MATCHUP_SCORING_POSITIONS) == {'TE'}
    assert 'WR' in wp.SCHEME_DEFENSE_SUPPORTED_POSITIONS,         "the preview must still be COMPUTED for WR, only the scoring is scoped"
    assert wp.SCHEME_MATCHUP_SCORING_POSITIONS < set(wp.SCHEME_DEFENSE_SUPPORTED_POSITIONS),         "scoring scope must be a strict subset of where the preview exists"
    assert 'v2_scheme_matchup' in wp.DEFAULT_FEATURES


def test_rematch_bump_is_neutral_at_the_shipped_multiplier():
    """REMATCH_WEIGHT_MULT ships at 1.0 as of 2026-09-02, so a rematch game
    must weigh EXACTLY the same as an equally-recent ordinary game.

    This test previously asserted the opposite - that a rematch is upweighted
    - which was the shipped behaviour at 1.6. That value was never backtested
    until now, and when it was, every value below 1.6 helped and every value
    above hurt, monotonically, across two independent windows. Opponent
    quality is already adjusted for in
    build_team_game_quality_adjusted_matchup, so a second helping of "same
    opponent" was double-counting. The mechanism is retained and still
    exercised by the test below; only its shipped strength changed."""
    df_rematch = weekly([
        {'name': 'A', 'week': 5, 'position': 'WR', 'opponent_team': 'KC', 'targets': 10},
        {'name': 'A', 'week': 4, 'position': 'WR', 'opponent_team': 'BUF', 'targets': 2},
    ])
    with_rematch, _ = wp._weighted_player_rates(
        df_rematch, 'name', ['targets'], as_of_week=6,
        matchup_matrix=pd.DataFrame(), upcoming_opponent={'A': 'KC'})

    df_no_rematch = weekly([
        {'name': 'A', 'week': 5, 'position': 'WR', 'opponent_team': 'SEA', 'targets': 10},
        {'name': 'A', 'week': 4, 'position': 'WR', 'opponent_team': 'BUF', 'targets': 2},
    ])
    without_rematch, _ = wp._weighted_player_rates(
        df_no_rematch, 'name', ['targets'], as_of_week=6,
        matchup_matrix=pd.DataFrame(), upcoming_opponent={'A': 'KC'})

    assert with_rematch.loc['A', 'targets'] == pytest.approx(
        without_rematch.loc['A', 'targets']),         "at REMATCH_WEIGHT_MULT=1.0 a rematch must carry no extra weight"


def test_rematch_mechanism_still_works_when_the_multiplier_is_raised(monkeypatch):
    """The code path is retained, not deleted - 1.0 only makes it inert. If
    the constant is ever raised again this proves the wiring still bites."""
    monkeypatch.setattr(wp, 'REMATCH_WEIGHT_MULT', 2.0)
    df_rematch = weekly([
        {'name': 'A', 'week': 5, 'position': 'WR', 'opponent_team': 'KC', 'targets': 10},
        {'name': 'A', 'week': 4, 'position': 'WR', 'opponent_team': 'BUF', 'targets': 2},
    ])
    with_rematch, _ = wp._weighted_player_rates(
        df_rematch, 'name', ['targets'], as_of_week=6,
        matchup_matrix=pd.DataFrame(), upcoming_opponent={'A': 'KC'})

    df_no_rematch = weekly([
        {'name': 'A', 'week': 5, 'position': 'WR', 'opponent_team': 'SEA', 'targets': 10},
        {'name': 'A', 'week': 4, 'position': 'WR', 'opponent_team': 'BUF', 'targets': 2},
    ])
    without_rematch, _ = wp._weighted_player_rates(
        df_no_rematch, 'name', ['targets'], as_of_week=6,
        matchup_matrix=pd.DataFrame(), upcoming_opponent={'A': 'KC'})

    assert with_rematch.loc['A', 'targets'] > without_rematch.loc['A', 'targets']


def test_weighted_player_rates_empty_input_returns_empty():
    # Returns (rates, weighted totals) - the totals ride along so an
    # efficiency RATIO can be formed from two weighted sums.
    rates, totals = wp._weighted_player_rates(weekly([]), 'name', ['targets'], as_of_week=2,
                                              matchup_matrix=pd.DataFrame(), upcoming_opponent={})
    assert rates.empty and totals.empty


# --- shrinkage --------------------------------------------------------------


def test_blended_rate_leans_on_prior_with_zero_games():
    # A player with no current-season games yet should land entirely on
    # the prior - w_current = 0/(0+K) = 0.
    out = wp._blended_rate(np.array([50.0]), np.array([0.0]), np.array([50.0]),
                           np.array([10.0]), 'targets', np.array([0.5]))
    assert abs(out[0] - 50.0) < 1e-9


def test_blended_rate_converges_toward_current_rate_with_more_games():
    # Same prior/current-rate gap, more games played -> closer to the
    # current-season rate. This is the literal "current season outweighs
    # the past as the sample grows" mechanism.
    few = wp._blended_rate(np.array([20.0]), np.array([1.0]), np.array([5.0]),
                           np.array([5.0]), 'targets', np.array([0.5]))
    many = wp._blended_rate(np.array([20.0]), np.array([10.0]), np.array([5.0]),
                            np.array([5.0]), 'targets', np.array([0.5]))
    assert many[0] > few[0]


def test_role_confidence_shrinks_k_less_for_a_confident_role():
    # Higher role_confidence -> smaller effective K -> more weight on the
    # player's own current rate for the SAME games-played count.
    low_conf = wp._blended_rate(np.array([20.0]), np.array([4.0]), np.array([5.0]),
                                np.array([5.0]), 'targets', np.array([0.0]))
    high_conf = wp._blended_rate(np.array([20.0]), np.array([4.0]), np.array([5.0]),
                                 np.array([5.0]), 'targets', np.array([1.0]))
    assert high_conf[0] > low_conf[0]


def test_blended_rate_falls_back_to_position_rate_with_no_prior_season():
    out = wp._blended_rate(np.array([20.0]), np.array([0.0]), np.array([np.nan]),
                           np.array([8.0]), 'targets', np.array([0.5]))
    assert abs(out[0] - 8.0) < 1e-9


# --- game script -----------------------------------------------------------

def test_team_week_margins_sign_convention():
    # Home team wins 30-10 (margin +20 for home, -20 for away).
    sched = pd.DataFrame([{'week': 1, 'home_team': 'KC', 'away_team': 'BUF',
                           'home_score': 30, 'away_score': 10}])
    out = wp._team_week_margins(sched)
    kc = out[out['Team'] == 'KC'].iloc[0]
    buf = out[out['Team'] == 'BUF'].iloc[0]
    assert kc['margin'] == 20 and buf['margin'] == -20


def test_team_week_margins_drops_unplayed_games():
    sched = pd.DataFrame([{'week': 1, 'home_team': 'KC', 'away_team': 'BUF',
                           'home_score': np.nan, 'away_score': np.nan}])
    assert wp._team_week_margins(sched).empty


def test_week_opponents_maps_both_directions_and_omits_byes():
    sched = pd.DataFrame([{'week': 3, 'home_team': 'KC', 'away_team': 'BUF'}])
    out = wp._week_opponents(sched, 3)
    assert out == {'KC': 'BUF', 'BUF': 'KC'}
    assert 'DEN' not in out  # DEN has no game this week -> on a bye


def test_vectorized_game_script_multiplier_neutral_without_enough_history():
    df = weekly([{'name': 'A', 'week': 1, 'team': 'KC', 'position': 'WR', 'receiving_yards': 50}])
    sched = pd.DataFrame([{'week': 1, 'home_team': 'KC', 'away_team': 'BUF',
                           'home_score': 20, 'away_score': 17}])
    out = wp._vectorized_game_script_multiplier(
        df, 'name', 'team', as_of_week=2, schedule_df=sched,
        target_margins=pd.Series({'A': 3.0}), stat='receiving_yards')
    # Only one game of history - below the 4-game floor - so no multiplier
    # is produced for this player at all.
    assert out.empty


# --- missing-data safety ----------------------------------------------------

def test_season_totals_empty_input_returns_empty_not_a_crash():
    out = wp._season_totals(weekly([]), 'name', 'team', 'WR', ['targets'])
    assert out.empty


def test_role_confidence_handles_no_snap_column_gracefully():
    df = pd.DataFrame({'name': ['A'], 'week': [1], 'position': ['WR']})
    out = wp._role_confidence(df, 'name', as_of_week=2, pos='WR', pff_rec=pd.DataFrame())
    assert out.empty


# --- expected snap share / role volume ---------------------------------------

def test_expected_snap_share_separates_a_backup_from_a_starter():
    # The measured failure this exists for: a backup's PER-GAME rate looks
    # like a starter's on a small sample, and only snap share separates them.
    df = weekly([
        {'name': 'Starter', 'week': w, 'position': 'QB', 'team': 'KC',
         'opponent_team': 'DEN', 'weekly_snap_pct': 100.0} for w in (1, 2, 3, 4, 5)
    ] + [
        {'name': 'Backup', 'week': w, 'position': 'QB', 'team': 'KC',
         'opponent_team': 'DEN', 'weekly_snap_pct': 12.0} for w in (2, 5)
    ])
    share = wp.expected_snap_share(df, 'name', 'team', as_of_week=6)
    assert share['Starter'] == 1.0
    assert 0.0 < share['Backup'] < 0.2


def test_expected_snap_share_reads_a_role_takeover_from_recent_games_only():
    # Tyler Shough's real 2025 shape: 4% -> 54% -> 90% -> 95%. A season
    # average would still call him a backup; a four-appearance window
    # shouldn't.
    df = weekly([
        {'name': 'Riser', 'week': 1, 'position': 'QB', 'team': 'NO',
         'opponent_team': 'ATL', 'weekly_snap_pct': 4.0},
        {'name': 'Riser', 'week': 2, 'position': 'QB', 'team': 'NO',
         'opponent_team': 'ATL', 'weekly_snap_pct': 54.0},
        {'name': 'Riser', 'week': 3, 'position': 'QB', 'team': 'NO',
         'opponent_team': 'ATL', 'weekly_snap_pct': 90.0},
        {'name': 'Riser', 'week': 4, 'position': 'QB', 'team': 'NO',
         'opponent_team': 'ATL', 'weekly_snap_pct': 95.0},
        {'name': 'Riser', 'week': 5, 'position': 'QB', 'team': 'NO',
         'opponent_team': 'ATL', 'weekly_snap_pct': 99.0},
    ])
    share = wp.expected_snap_share(df, 'name', 'team', as_of_week=6, lookback=4)
    assert share['Riser'] > 0.8  # weeks 2-5, not weeks 1-5


def test_expected_snap_share_does_not_punish_a_returning_starter():
    # Measured regression the team-weeks version caused: a starter who
    # missed two weeks must not read as a part-time player on his return.
    df = weekly([
        # 'Hurt' misses weeks 3-4 entirely; 'Healthy' plays all four.
        {'name': 'Hurt', 'week': 1, 'position': 'RB', 'team': 'SF',
         'opponent_team': 'LA', 'weekly_snap_pct': 85.0},
        {'name': 'Hurt', 'week': 2, 'position': 'RB', 'team': 'SF',
         'opponent_team': 'LA', 'weekly_snap_pct': 88.0},
    ] + [
        {'name': 'Healthy', 'week': w, 'position': 'RB', 'team': 'SF',
         'opponent_team': 'LA', 'weekly_snap_pct': 86.0} for w in (1, 2, 3, 4)
    ])
    share = wp.expected_snap_share(df, 'name', 'team', as_of_week=5)
    assert share['Hurt'] > 0.8
    assert abs(share['Hurt'] - share['Healthy']) < 0.06


def test_cold_start_returning_skill_role_restores_only_a_proven_same_team_role():
    restored, used = wp.restore_cold_start_returning_role_share(
        np.array([0.34, 0.15, 0.33, 0.40]),
        np.array([0.91, 0.90, 0.90, 0.90]),
        np.array([7, 3, 8, 7]),
        np.array(['NYJ', 'KC', 'HOU', 'TB']),
        np.array(['NYJ', 'KC', 'SEA', 'TB']),
        'WR',
    )
    assert used.tolist() == [True, False, False, True]
    # Recovery is deliberately continuous rather than the old all-or-nothing
    # restore.  Proven same-team roles gain material credit for active-game
    # work, but do not jump straight to a full-share assumption from a single
    # preseason signal.
    assert 0.34 < restored[0] < 0.91
    assert restored[1] == 0.15
    assert restored[2] == 0.33
    assert 0.40 < restored[3] < 0.90
    # RBs need the stricter eight-game evidence threshold, preserving the
    # conservative treatment for a short fill-in sample.
    rb_restored, rb_used = wp.restore_cold_start_returning_role_share(
        np.array([0.35]), np.array([0.90]), np.array([7]),
        np.array(['KC']), np.array(['KC']), 'RB',
    )
    assert not rb_used[0]
    assert rb_restored[0] == 0.35


def test_charted_short_sample_returning_starter_recovers_without_treating_source_red_as_out():
    # Nabers-shaped history: three genuine full-snap games, then a clear
    # partial exit and a terminal absence.  Three games are not enough for a
    # generic WR returning-role claim, but a resolved current WR1 chart rank
    # gives a bounded live-preseason recovery path.  It remains well below
    # the raw 95% active sample and never fires when an actual availability
    # source says the player is out.
    restored, used, reasons = wp.restore_cold_start_returning_role_share(
        np.array([0.168, 0.168]), np.array([0.95, 0.95]), np.array([3, 3]),
        np.array(['NYG', 'NYG']), np.array(['NYG', 'NYG']), 'WR',
        pre_absence_share=np.array([0.95, 0.95]),
        depth_rank=np.array([1.0, np.nan]),
        terminal_gap_weeks=np.array([14.0, 14.0]),
        return_details=True,
    )
    assert used.tolist() == [True, False]
    assert reasons.tolist() == ['charted short-sample returning-starter recovery', 'none']
    assert 0.50 < restored[0] <= 0.75
    assert restored[1] == 0.168


def test_injury_shortened_prior_year_does_not_suppress_a_proven_multi_year_role():
    # Garrett Wilson, NYJ 2026 wk1: 6 played games in 2025 (all 84-100% snaps)
    # then a season-ending injury; a full 2024 at ~95% snaps; rank-1 on the
    # current chart; role_confidence 0.93 off his last healthy games. The
    # pre-fix path pinned `evidence` at its 0.20 floor purely because 6 == the
    # WR minimum-games threshold, recovering him to only ~0.71 team snaps.
    # With N-2 corroboration + confidence he should land near his real role.
    restored, used, reasons = wp.restore_cold_start_returning_role_share(
        np.array([0.516, 0.516]),          # injury-depressed whole-season share
        np.array([0.954, 0.954]),          # active-game share
        np.array([6.0, 6.0]),              # exactly the WR minimum
        np.array(['NYJ', 'NYJ']), np.array(['NYJ', 'NYJ']), 'WR',
        pre_absence_share=np.array([0.94, 0.94]),
        depth_rank=np.array([1.0, 1.0]),
        terminal_gap_weeks=np.array([12.0, 12.0]),
        prior2_games=np.array([17.0, 2.0]),        # row 1: no real N-2 sample
        prior2_active_share=np.array([0.95, 0.95]),
        role_confidence=np.array([0.93, 0.93]),
        return_details=True,
    )
    # Row 0 is fully corroborated -> pulled at least 85% of the way to the role.
    assert restored[0] >= 0.516 + 0.85 * (0.95 - 0.516) - 1e-6
    assert restored[0] > 0.85
    assert reasons[0] == 'proven multi-year every-down role restored (injury-shortened prior year)'
    # Row 1 has the same prior year but no qualifying N-2 season, so it stays
    # on the ordinary (more conservative) recovery.
    assert restored[1] < restored[0] - 0.05
    assert reasons[1] == 'continuous returning active/pre-absence role recovery'


def test_returning_role_recovery_has_no_more_evidence_lower_projection_cliff():
    # A rank-1 charted returning starter one game either side of the
    # minimum-games threshold must not see his projection DROP as he crosses
    # from the short-sample path to the standard path.
    common = dict(
        pre_absence_share=np.array([0.95, 0.95]),
        depth_rank=np.array([1.0, 1.0]),
        terminal_gap_weeks=np.array([12.0, 12.0]),
        return_details=True,
    )
    below, _, _ = wp.restore_cold_start_returning_role_share(
        np.array([0.30, 0.30]), np.array([0.95, 0.95]), np.array([5.0, 5.0]),
        np.array(['NYG', 'NYG']), np.array(['NYG', 'NYG']), 'WR', **common)
    at, _, _ = wp.restore_cold_start_returning_role_share(
        np.array([0.30, 0.30]), np.array([0.95, 0.95]), np.array([6.0, 6.0]),
        np.array(['NYG', 'NYG']), np.array(['NYG', 'NYG']), 'WR', **common)
    assert at[0] >= below[0] - 1e-6


def test_regular_season_late_games_remain_in_a_returning_receiver_prior():
    # Drake London audit regression: the late low-output games must remain
    # rate evidence when they were real full-role games.  This is deliberately
    # a no-op on judgment—the participation screen should only drop a clear
    # partial/relief game, not quietly select the high-production portion of
    # a season.
    rows = []
    for week in range(1, 10):
        rows.append({'name': 'Drake London', 'team': 'ATL', 'opponent_team': 'NO',
                     'week': week, 'position': 'WR', 'weekly_snap_pct': 90.0,
                     'has_snap_match': True, 'targets': 94.0 / 9.0,
                     'receptions': 6.67, 'receiving_yards': 90.0})
    rows.extend([
        {'name': 'Drake London', 'team': 'ATL', 'opponent_team': 'NO', 'week': 16,
         'position': 'WR', 'weekly_snap_pct': 69.0, 'has_snap_match': True,
         'targets': 8.0, 'receptions': 3.0, 'receiving_yards': 27.0},
        {'name': 'Drake London', 'team': 'ATL', 'opponent_team': 'NO', 'week': 17,
         'position': 'WR', 'weekly_snap_pct': 98.0, 'has_snap_match': True,
         'targets': 2.0, 'receptions': 1.0, 'receiving_yards': 4.0},
        {'name': 'Drake London', 'team': 'ATL', 'opponent_team': 'NO', 'week': 18,
         'position': 'WR', 'weekly_snap_pct': 94.0, 'has_snap_match': True,
         'targets': 8.0, 'receptions': 4.0, 'receiving_yards': 78.0},
    ])
    eligible = wp.annotate_player_history_participation(weekly(rows), 'name', 'team')
    assert eligible['_player_history_eligible'].all()
    totals = wp._season_totals(
        eligible, 'name', 'team', 'WR', ['targets', 'receptions', 'receiving_yards'])
    london = totals.iloc[0]
    assert london['Games'] == 12
    assert np.isclose(london['targets'], 112.0, atol=0.02)
    assert np.isclose(london['receiving_yards'], 919.0, atol=0.02)


def test_default_features_include_the_former_v2_only_components():
    # The separate "V1 released baseline" / "V2 experimental" toggle was
    # retired 2026-08-26 - DEFAULT_FEATURES is now the single standard model
    # and includes what used to be V2-only components.
    feats = wp.resolve_model_features()
    assert 'v2_preseason_rb_allocator' in feats
    assert 'calibration' in feats


def test_pff_alignment_matchup_ships_active_despite_the_measured_loss():
    # Measured 2026-08-24 against the 2025 startable pool (see
    # DEFAULT_FEATURES's own comment): loses MAE and rank-corr on both
    # START-WR and START-TE. DESPITE that measured loss, the user explicitly
    # asked (2026-08-26) to ship it ON by default anyway so it stays live
    # and inspectable rather than gated behind a mode nobody selects by
    # default - see DEFAULT_FEATURES's own comment.
    feats = wp.resolve_model_features()
    assert 'v2_pff_alignment_matchup' in feats
    assert 'v2_pff_alignment_matchup' in wp.MODEL_FEATURES


def test_preseason_qb1_resolution_auto_selects_one_clear_full_season_incumbent():
    current = pd.DataFrame([
        {'name': 'Joe Burrow', 'team': 'CIN', 'position': 'QB'},
        {'name': 'Jake Browning', 'team': 'CIN', 'position': 'QB'},
        {'name': 'Lamar Jackson', 'team': 'BAL', 'position': 'QB'},
    ])
    prior = weekly([
        *[{'name': 'Joe Burrow', 'team': 'CIN', 'opponent_team': 'BAL', 'week': w,
           'position': 'QB', 'weekly_snap_pct': 96.0} for w in range(1, 11)],
        *[{'name': 'Jake Browning', 'team': 'CIN', 'opponent_team': 'BAL', 'week': w,
           'position': 'QB', 'weekly_snap_pct': 4.0} for w in range(1, 11)],
        *[{'name': 'Lamar Jackson', 'team': 'BAL', 'opponent_team': 'CIN', 'week': w,
           'position': 'QB', 'weekly_snap_pct': 98.0} for w in range(1, 11)],
    ])
    resolution = wp.resolve_preseason_qb1s(
        current, 'name', 'team', prior, 'name', 'team', 2026,
        overrides=pd.DataFrame(columns=wp.QB1_OVERRIDE_COLUMNS),
    )
    assert resolution['selected'][('CIN', 'joeburrow')] == 'prior_season_incumbent'
    assert resolution['selected'][('BAL', 'lamarjackson')] == 'prior_season_incumbent'
    assert ('CIN', 'jakebrowning') not in resolution['selected']
    assert not resolution['selection_required_teams']


def test_preseason_qb1_resolution_requires_a_manual_choice_for_an_ambiguous_room():
    current = pd.DataFrame([
        {'name': 'Jaxson Dart', 'team': 'NYG', 'position': 'QB'},
        {'name': 'Brandon Allen', 'team': 'NYG', 'position': 'QB'},
    ])
    prior = weekly([
        *[{'name': 'Jaxson Dart', 'team': 'NYG', 'opponent_team': 'DAL', 'week': w,
           'position': 'QB', 'weekly_snap_pct': 60.0} for w in range(1, 11)],
        *[{'name': 'Brandon Allen', 'team': 'NYG', 'opponent_team': 'DAL', 'week': w,
           'position': 'QB', 'weekly_snap_pct': 40.0} for w in range(1, 11)],
    ])
    unresolved = wp.resolve_preseason_qb1s(
        current, 'name', 'team', prior, 'name', 'team', 2026,
        overrides=pd.DataFrame(columns=wp.QB1_OVERRIDE_COLUMNS),
    )
    assert 'NYG' in unresolved['selection_required_teams']
    assert not unresolved['selected']
    resolved = wp.resolve_preseason_qb1s(
        current, 'name', 'team', prior, 'name', 'team', 2026,
        overrides=pd.DataFrame([{'year': 2026, 'team': 'NYG', 'player': 'Jaxson Dart'}]),
    )
    assert resolved['selected'][('NYG', 'jaxsondart')] == 'manual_override'
    assert 'NYG' not in resolved['selection_required_teams']


def test_cold_start_manual_qb1_receives_full_prior_per_game_workload():
    # A named upcoming QB1 may only have half of the prior team's season. The
    # manual selection must restore a full workload without promoting backups.
    current = pd.DataFrame([
        {'name': 'Starter', 'team': 'KC', 'position': 'QB'},
    ])
    prior_rows = []
    for week in (1, 2, 3):
        prior_rows.append({
            'name': 'Starter', 'team': 'KC', 'opponent_team': 'DEN', 'week': week,
            'position': 'QB', 'weekly_snap_pct': 80.0,
            'passing_attempts': 30.0, 'passing_completions': 20.0,
            'passing_yards': 200.0, 'passing_tds': 1.0,
            'passing_interceptions': 0.5, 'rushing_attempts': 3.0,
            'rushing_yards': 15.0, 'rushing_tds': 0.0,
        })
    for week in (4, 5, 6):
        prior_rows.append({
            'name': 'Backup', 'team': 'KC', 'opponent_team': 'DEN', 'week': week,
            'position': 'QB', 'weekly_snap_pct': 80.0,
            'passing_attempts': 20.0, 'passing_completions': 12.0,
            'passing_yards': 130.0, 'passing_tds': 0.5,
            'passing_interceptions': 0.5, 'rushing_attempts': 2.0,
            'rushing_yards': 8.0, 'rushing_tds': 0.0,
        })
    prior = weekly(prior_rows)
    schedule = pd.DataFrame([{'week': 1, 'home_team': 'KC', 'away_team': 'DEN'}])
    original = (wp.load_and_merge_data, wp.load_schedule, wp._load_pff_receiving,
                wp.load_team_pace, wp.load_qb1_overrides, wp._target_margins_by_team)
    try:
        wp.load_and_merge_data = lambda year, scoring: (
            (current.copy() if year == 2026 else prior.copy()), 'team', 'name', None)
        wp.load_schedule = lambda year: schedule.copy()
        wp._load_pff_receiving = lambda year, allow_season_totals=True: pd.DataFrame()
        wp.load_team_pace = lambda year: pd.DataFrame()
        wp.load_qb1_overrides = lambda _year: (
            pd.DataFrame([{'year': 2026, 'team': 'KC', 'player': 'Starter'}]), None)
        wp._target_margins_by_team = lambda year, week: {}
        # availability_fingerprint is cache-key-only (see its own docstring) -
        # given a distinct value here purely so this test's mocked fixture
        # can't collide in @st.cache_data with another test's (year, week,
        # scoring, as_of_week, apply_injury) tuple that happens to match.
        out, meta = wp.build_weekly_projections(
            2026, 1, 'Full PPR', as_of_week=1, apply_injury=False,
            availability_fingerprint='test_cold_start_manual_qb1')
    finally:
        (wp.load_and_merge_data, wp.load_schedule, wp._load_pff_receiving,
         wp.load_team_pace, wp.load_qb1_overrides, wp._target_margins_by_team) = original
    starter = out.loc[out['Player'] == 'Starter'].iloc[0]
    detail = meta['explanations'][('Starter', 'QB', 'KC')]
    passing = detail['stats']['passing_yards']
    assert starter['Expected Snap Share'] == 1.0
    assert starter['QB1 Workload Override']
    assert passing['qb1_workload_override']
    assert passing['qb1_workload_source'] == 'Manual QB1 selection'
    assert passing['role_scale'] == 1.0
    # raw_prior_rate, not prior_rate: the latter now also carries
    # v2_qb_volume_blend's evidence-weighted adjustment (part of
    # DEFAULT_FEATURES since 2026-08-26), so it's no longer exactly the raw
    # per-game average even though the manual override is applying the full,
    # undiluted-by-the-backup prior workload correctly - that's what
    # raw_prior_rate isolates and confirms.
    assert passing['raw_prior_rate'] == 200.0
    assert meta['source_contract']['qb_starter_source'] == 'manual_qb1_overrides_plus_unambiguous_prior_season_incumbents'


# --- interrupted player-game screen / QB starter gate ----------------------

def test_partial_game_screen_excludes_a_qb_split_from_player_rate_evidence():
    rows = []
    for week, share in ((1, 95.0), (2, 96.0), (3, 94.0), (4, 48.0)):
        rows.append({
            'name': 'Starter', 'team': 'KC', 'opponent_team': 'DEN', 'week': week,
            'position': 'QB', 'weekly_snap_pct': share, 'has_snap_match': True,
            'passing_yards': 250.0 if week < 4 else 80.0,
        })
    rows.append({
        'name': 'Backup', 'team': 'KC', 'opponent_team': 'DEN', 'week': 4,
        'position': 'QB', 'weekly_snap_pct': 52.0, 'has_snap_match': True,
        'passing_yards': 90.0,
    })
    annotated = wp.annotate_player_history_participation(weekly(rows), 'name', 'team')
    split = annotated[annotated['week'].eq(4)]
    assert not split['_player_history_eligible'].any()
    assert set(split['_player_history_reason']) == {'QB split/relief game'}
    eligible = annotated[annotated['_player_history_eligible']]
    totals = wp._season_totals(eligible, 'name', 'team', 'QB', ['passing_yards'])
    starter = totals.loc[totals['name'].eq('Starter')].iloc[0]
    assert starter['Games'] == 3
    assert starter['passing_yards'] == 750.0


def test_partial_game_screen_excludes_paired_replacement_but_not_normal_rotation():
    rows = []
    for week, lead_share, reserve_share in ((1, 92.0, 8.0), (2, 90.0, 10.0),
                                            (3, 94.0, 6.0), (4, 45.0, 55.0)):
        rows.extend([
            {'name': 'Lead RB', 'team': 'KC', 'opponent_team': 'DEN', 'week': week,
             'position': 'RB', 'weekly_snap_pct': lead_share, 'has_snap_match': True,
             'rushing_attempts': 14.0},
            {'name': 'Reserve RB', 'team': 'KC', 'opponent_team': 'DEN', 'week': week,
             'position': 'RB', 'weekly_snap_pct': reserve_share, 'has_snap_match': True,
             'rushing_attempts': 3.0},
        ])
    for week, share in ((1, 42.0), (2, 39.0), (3, 45.0), (4, 40.0)):
        rows.append({
            'name': 'Committee RB', 'team': 'DEN', 'opponent_team': 'KC', 'week': week,
            'position': 'RB', 'weekly_snap_pct': share, 'has_snap_match': True,
            'rushing_attempts': 7.0,
        })
    annotated = wp.annotate_player_history_participation(weekly(rows), 'name', 'team')
    lead = annotated[(annotated['name'] == 'Lead RB') & annotated['week'].eq(4)].iloc[0]
    reserve = annotated[(annotated['name'] == 'Reserve RB') & annotated['week'].eq(4)].iloc[0]
    committee = annotated[annotated['name'].eq('Committee RB')]
    assert not lead['_player_history_eligible']
    assert lead['_player_history_reason'] == 'abrupt partial role after established workload'
    assert not reserve['_player_history_eligible']
    assert reserve['_player_history_reason'] == 'partial replacement after teammate exit'
    assert committee['_player_history_eligible'].all()


def test_partial_game_screen_uses_a_final_margin_only_for_extreme_rest_case():
    rows = [
        {'name': 'WR', 'team': 'KC', 'opponent_team': 'DEN', 'week': week,
         'position': 'WR', 'weekly_snap_pct': share, 'has_snap_match': True,
         'targets': 7.0}
        for week, share in ((1, 94.0), (2, 95.0), (3, 93.0), (4, 60.0))
    ]
    schedule = pd.DataFrame([
        {'week': 4, 'home_team': 'KC', 'away_team': 'DEN', 'home_score': 35, 'away_score': 7},
    ])
    without_score = wp.annotate_player_history_participation(weekly(rows), 'name', 'team')
    with_score = wp.annotate_player_history_participation(weekly(rows), 'name', 'team', schedule)
    assert without_score.loc[without_score['week'].eq(4), '_player_history_eligible'].iloc[0]
    rested = with_score.loc[with_score['week'].eq(4)].iloc[0]
    assert not rested['_player_history_eligible']
    assert rested['_player_history_reason'] == 'severe blowout rest'


def test_partial_game_screen_never_treats_an_unmatched_snap_source_as_an_exit():
    rows = [
        {'name': 'WR', 'team': 'KC', 'opponent_team': 'DEN', 'week': week,
         'position': 'WR', 'weekly_snap_pct': share, 'has_snap_match': False,
         'targets': 7.0}
        for week, share in ((1, 95.0), (2, 94.0), (3, 96.0), (4, 20.0))
    ]
    annotated = wp.annotate_player_history_participation(weekly(rows), 'name', 'team')
    assert annotated['_player_history_eligible'].all()


def test_inseason_qb1_resolver_requires_one_clear_recent_starter_or_manual_choice():
    current = pd.DataFrame([
        {'name': 'Starter', 'team': 'KC', 'position': 'QB'},
        {'name': 'Backup', 'team': 'KC', 'position': 'QB'},
    ])
    history = weekly([
        *[{'name': 'Starter', 'team': 'KC', 'opponent_team': 'DEN', 'week': week,
           'position': 'QB', 'weekly_snap_pct': 95.0} for week in (1, 2, 3)],
        *[{'name': 'Backup', 'team': 'KC', 'opponent_team': 'DEN', 'week': week,
           'position': 'QB', 'weekly_snap_pct': 5.0} for week in (1, 2, 3)],
    ])
    automatic = wp.resolve_inseason_qb1s(
        current, 'name', 'team', history, 'name', 'team', 4, 2026,
        overrides=pd.DataFrame(columns=wp.QB1_OVERRIDE_COLUMNS))
    assert automatic['selected'][('KC', 'starter')] == 'observed_current_starter'
    manual = wp.resolve_inseason_qb1s(
        current, 'name', 'team', history, 'name', 'team', 4, 2026,
        overrides=pd.DataFrame([{'year': 2026, 'team': 'KC', 'player': 'Backup'}]))
    assert manual['selected'][('KC', 'backup')] == 'manual_override'
    split_history = history.copy()
    split_history.loc[split_history['name'].eq('Starter'), 'weekly_snap_pct'] = 55.0
    split_history.loc[split_history['name'].eq('Backup'), 'weekly_snap_pct'] = 45.0
    unresolved = wp.resolve_inseason_qb1s(
        current, 'name', 'team', split_history, 'name', 'team', 4, 2026,
        overrides=pd.DataFrame(columns=wp.QB1_OVERRIDE_COLUMNS))
    assert 'KC' in unresolved['selection_required_teams']


def test_nonstarter_qb_has_zero_projected_volume_not_a_relief_rate_projection():
    rows = []
    for week in (1, 2, 3):
        rows.extend([
            {'name': 'Starter', 'team': 'KC', 'opponent_team': 'DEN', 'week': week,
             'position': 'QB', 'weekly_snap_pct': 95.0, 'has_snap_match': True,
             'passing_attempts': 32.0, 'passing_completions': 21.0,
             'passing_yards': 245.0, 'passing_tds': 1.5, 'passing_interceptions': 0.5,
             'rushing_attempts': 3.0, 'rushing_yards': 15.0, 'rushing_tds': 0.1},
            {'name': 'Backup', 'team': 'KC', 'opponent_team': 'DEN', 'week': week,
             'position': 'QB', 'weekly_snap_pct': 5.0, 'has_snap_match': True,
             'passing_attempts': 3.0, 'passing_completions': 2.0,
             'passing_yards': 35.0, 'passing_tds': 0.2, 'passing_interceptions': 0.0,
             'rushing_attempts': 1.0, 'rushing_yards': 5.0, 'rushing_tds': 0.0},
        ])
    current = weekly(rows)
    prior = current.copy()
    schedule = pd.DataFrame([{'week': 4, 'home_team': 'KC', 'away_team': 'DEN'}])
    original = (wp.load_and_merge_data, wp.load_schedule, wp._load_pff_receiving,
                wp.load_team_pace, wp.load_qb1_overrides, wp._target_margins_by_team)
    try:
        wp.load_and_merge_data = lambda year, scoring: (
            (current.copy() if year == 2026 else prior.copy()), 'team', 'name', None)
        wp.load_schedule = lambda year: schedule.copy()
        wp._load_pff_receiving = lambda year, allow_season_totals=True: pd.DataFrame()
        wp.load_team_pace = lambda year: pd.DataFrame()
        wp.load_qb1_overrides = lambda _year: (pd.DataFrame(columns=wp.QB1_OVERRIDE_COLUMNS), None)
        wp._target_margins_by_team = lambda year, week: {}
        out, meta = wp.build_weekly_projections(
            2026, 4, 'Full PPR', as_of_week=4, apply_injury=False)
    finally:
        (wp.load_and_merge_data, wp.load_schedule, wp._load_pff_receiving,
         wp.load_team_pace, wp.load_qb1_overrides, wp._target_margins_by_team) = original
    starter = out.loc[out['Player'].eq('Starter')].iloc[0]
    backup = out.loc[out['Player'].eq('Backup')].iloc[0]
    assert starter['QB Projected Starter']
    assert not backup['QB Projected Starter']
    for stat in wp.OFFENSE_PROJECTION_STATS['QB']:
        assert backup[stat] == 0.0
    assert backup['Raw Model Proj Pts'] == 0.0
    detail = meta['explanations'][('Backup', 'QB', 'KC')]
    assert detail['stats']['passing_yards']['qb_nonstarter_volume_factor'] == 0.0


# --- roles / role-conditioned matchup ----------------------------------------

def test_player_roles_split_receivers_by_depth_of_target():
    rows = []
    for i in range(12):
        # 12 receivers on an evenly-spread ADOT ladder from 4 to 15 yards
        adot = 4 + i
        rows.append({'name': f'W{i}', 'week': 1, 'position': 'WR', 'team': 'KC',
                     'opponent_team': 'DEN', 'targets': 30,
                     'receiving_air_yards': 30 * adot})
    roles = wp.build_player_roles(weekly(rows), 'name', 'WR')
    assert roles['W0'] == 'WR_SHORT'
    assert roles['W11'] == 'WR_DEEP'
    assert roles['W5'] in ('WR_SHORT', 'WR_MID')


def test_player_roles_park_a_thin_sample_in_the_middle_bucket():
    # Two targets is not a measured role, whatever the ADOT says.
    rows = [{'name': f'W{i}', 'week': 1, 'position': 'WR', 'team': 'KC',
             'opponent_team': 'DEN', 'targets': 30, 'receiving_air_yards': 30 * (4 + i)}
            for i in range(12)]
    rows.append({'name': 'Thin', 'week': 1, 'position': 'WR', 'team': 'KC',
                 'opponent_team': 'DEN', 'targets': 2, 'receiving_air_yards': 80})
    roles = wp.build_player_roles(weekly(rows), 'name', 'WR')
    assert roles['Thin'] == 'WR_MID'


def test_role_matchup_blend_shrinks_toward_the_overall_rating():
    overall = pd.DataFrame({'targets': [1.0]}, index=['DEN'])
    role_tables = {'WR_DEEP': pd.DataFrame({'targets': [1.4]}, index=['DEN'])}
    opponents = np.array(['DEN'])
    roles = np.array(['WR_DEEP'])
    thin = wp._role_adjusted_multiplier(overall, role_tables, {('DEN', 'WR_DEEP'): 1.0},
                                        opponents, roles, 'targets')
    thick = wp._role_adjusted_multiplier(overall, role_tables, {('DEN', 'WR_DEEP'): 40.0},
                                         opponents, roles, 'targets')
    # More role-specific evidence -> closer to the role rating, never past it.
    assert 1.0 < thin[0] < thick[0] <= 1.4


def test_role_matchup_falls_back_when_the_defense_never_faced_that_role():
    overall = pd.DataFrame({'targets': [1.2]}, index=['DEN'])
    out = wp._role_adjusted_multiplier(overall, {'WR_DEEP': pd.DataFrame()}, {},
                                       np.array(['DEN']), np.array(['WR_SHORT']), 'targets')
    assert abs(out[0] - 1.2) < 1e-9


# --- game environment ---------------------------------------------------------

def test_game_environment_sign_convention_favours_the_home_team():
    # spread_line is POSITIVE when the HOME team is favored - the one thing
    # that silently inverts this whole component if it's read backwards.
    sched = pd.DataFrame({'week': [1], 'home_team': ['KC'], 'away_team': ['DEN'],
                          'total_line': [46.0], 'spread_line': [7.0], 'roof': ['outdoors']})
    env = wp.game_environment(sched, 1)
    assert env['KC']['implied'] == 26.5
    assert env['DEN']['implied'] == 19.5
    assert env['KC']['indoor'] is False


def test_game_environment_multiplier_is_neutral_without_a_posted_line():
    sched = pd.DataFrame({'week': [1], 'home_team': ['KC'], 'away_team': ['DEN'],
                          'total_line': [np.nan], 'spread_line': [np.nan], 'roof': ['dome']})
    env = wp.game_environment(sched, 1)
    out = wp._game_env_multiplier(env, np.array(['KC']), 'QB', league_implied=22.0)
    # Venue still applies; the total does not, because there isn't one.
    assert abs(out[0] - wp.VENUE_MULT['QB']['indoor']) < 1e-9


# --- calibration --------------------------------------------------------------

def test_calibration_is_one_sided():
    slope, intercept = wp.WEEKLY_CALIBRATION['WR']
    crossover = intercept / (1 - slope)
    high, low = crossover + 10, crossover - 5
    assert min(high, intercept + slope * high) < high      # the top is shrunk
    assert min(low, intercept + slope * low) == low        # the bottom is left alone


# --- V2: priors, continuous roles, and cutoff-safe sources -----------------

def test_v2_defense_prior_starts_at_last_year_then_adapts():
    current = pd.DataFrame({'targets': [1.4]}, index=['DEN'])
    prior = pd.DataFrame({'targets': [0.8]}, index=['DEN'])
    # Week 1 (no current evidence) sits exactly on last year regardless of
    # the shrinkage constant.
    week_one = wp.blend_defense_prior(current, prior, pd.Series({'DEN': 0.0}))
    assert abs(week_one.loc['DEN', 'targets'] - 0.8) < 1e-9
    # The blend weight is evidence / (evidence + prior_games). Pin the
    # mechanism with an explicit constant so this does not silently encode
    # whatever DEFENSE_PRIOR_GAMES currently is: 4 games against a prior_games
    # of 4 is a 50/50 blend -> 1.1.
    even = wp.blend_defense_prior(current, prior, pd.Series({'DEN': 4.0}), prior_games=4.0)
    assert abs(even.loc['DEN', 'targets'] - 1.1) < 1e-9
    # The shipped constant (12.0 as of 2026-08-29) deliberately adapts
    # SLOWER: 4 current games is only a quarter-weight, so the profile is
    # still much closer to last year than to the current sample.
    shipped = wp.blend_defense_prior(current, prior, pd.Series({'DEN': 4.0}))
    expected = 0.8 + (4.0 / (4.0 + wp.DEFENSE_PRIOR_GAMES)) * (1.4 - 0.8)
    assert abs(shipped.loc['DEN', 'targets'] - expected) < 1e-9
    assert shipped.loc['DEN', 'targets'] < even.loc['DEN', 'targets']


def test_v2_confirmed_role_change_allows_aggressive_volume_but_not_td_blend():
    # The agreed target behavior: three real 9-target games after a
    # comparable 6-target prior should be roughly 8--8.25 when role evidence
    # is strong.  TD shrinkage must remain unchanged by that role signal.
    volume = wp._blended_rate(np.array([9.0]), np.array([3.0]), np.array([6.0]),
                              np.array([6.0]), 'targets', np.array([1.0]),
                              role_change_confidence=np.array([0.75]))
    td_with_signal = wp._blended_rate(np.array([1.0]), np.array([3.0]), np.array([0.2]),
                                      np.array([0.2]), 'receiving_tds', np.array([1.0]),
                                      role_change_confidence=np.array([1.0]))
    td_without_signal = wp._blended_rate(np.array([1.0]), np.array([3.0]), np.array([0.2]),
                                         np.array([0.2]), 'receiving_tds', np.array([1.0]))
    assert 8.0 <= volume[0] <= 8.25
    assert abs(td_with_signal[0] - td_without_signal[0]) < 1e-9


def test_v2_two_year_td_prior_requires_a_comparable_opportunity_role():
    comparable, used = wp.blend_comparable_td_priors(
        np.array([0.50]), np.array([0.30]), np.array([7.0]), np.array([6.0]))
    changed_role, rejected = wp.blend_comparable_td_priors(
        np.array([0.50]), np.array([0.30]), np.array([7.0]), np.array([1.0]))
    assert used[0] and 0.30 < comparable[0] < 0.50
    assert not rejected[0] and changed_role[0] == 0.50


def test_td_prior_credibility_regresses_a_thin_one_season_rate_but_spares_a_veteran():
    pos_rate = np.array([0.030])
    # RJ Harvey shape: ~120 carries, one role season, hot 0.055 rate. After the
    # 2026-08-30 K softening (rushing K 220->130) a thin one-season sample still
    # lands close to a coin-flip credibility - about half its rate regressed.
    harvey, cred_h, long_h = wp.credibility_shrunk_td_prior(
        np.array([0.055]), np.array([120.0]), np.array([1.0]), pos_rate, 'rushing_attempts')
    assert cred_h[0] < 0.52 and long_h[0] == 1.0
    assert pos_rate[0] < harvey[0] < 0.048          # still pulled well toward the mean

    # Derrick Henry shape: ~1100 carries over 4 role seasons, 0.045 rate.
    henry, cred_y, long_y = wp.credibility_shrunk_td_prior(
        np.array([0.045]), np.array([1100.0]), np.array([4.0]), pos_rate, 'rushing_attempts')
    assert cred_y[0] > 0.8 and long_y[0] == 1.0 + wp.TD_PRIOR_LONGEVITY_BONUS_MAX
    assert abs(henry[0] - 0.045) < 0.002            # essentially untouched

    # Two solid seasons: regressed, but NOT punished for lacking a third year.
    two_yr, cred_2, long_2 = wp.credibility_shrunk_td_prior(
        np.array([0.040]), np.array([450.0]), np.array([2.0]), pos_rate, 'rushing_attempts')
    assert long_2[0] == 1.0 and 0.55 < cred_2[0] < 0.80
    # More history keeps a LARGER fraction of the player's own rate.
    kept_2yr = (two_yr[0] - pos_rate[0]) / (0.040 - pos_rate[0])
    kept_harvey = (harvey[0] - pos_rate[0]) / (0.055 - pos_rate[0])
    assert kept_2yr > kept_harvey

    # NaN prior passes straight through (league fallback handled elsewhere).
    nan_out, nan_cred, _ = wp.credibility_shrunk_td_prior(
        np.array([np.nan]), np.array([0.0]), np.array([0.0]), pos_rate, 'rushing_attempts')
    assert np.isnan(nan_out[0]) and np.isnan(nan_cred[0])


def test_prior2_blend_weight_full_when_2024_raises_the_value():
    # Lamar Jackson/Jayden Daniels shape: a down/injury-shortened 2025 (4
    # games, so fraction_missing=(8-4)/8=0.5) with a BETTER 2024 - kept at
    # full base weight, no asymmetric cut.
    weight = wp.prior2_blend_weight(
        games_2025=np.array([4.0]), games_2024=np.array([17.0]),
        current=np.array([0.20]), prior2_value=np.array([0.30]))
    expected_base = wp.PRIOR2_BLEND_BASE_WEIGHT + (wp.PRIOR2_BLEND_MAX_WEIGHT - wp.PRIOR2_BLEND_BASE_WEIGHT) * 0.5
    assert abs(weight[0] - expected_base) < 1e-9


def test_prior2_blend_weight_dampened_when_2024_lowers_the_value():
    # A full 2025 season (>=8 games -> floor weight) with a WORSE 2024 (a
    # 2025 breakout off a thinner/backup prior role) - cut to
    # PRIOR2_BLEND_DECREASE_DAMPENING of the base weight, staying bullish on
    # the ascending player rather than docking him at full weight.
    weight = wp.prior2_blend_weight(
        games_2025=np.array([16.0]), games_2024=np.array([17.0]),
        current=np.array([0.30]), prior2_value=np.array([0.10]))
    expected = wp.PRIOR2_BLEND_BASE_WEIGHT * wp.PRIOR2_BLEND_DECREASE_DAMPENING
    assert abs(weight[0] - expected) < 1e-9


def test_prior2_blend_weight_decays_to_zero_across_the_2026_season():
    games_2025 = np.array([16.0])
    games_2024 = np.array([17.0])
    current = np.array([0.20])
    prior2_value = np.array([0.30])  # pulls up, so no asymmetric cut muddies the decay check
    at_cold_start = wp.prior2_blend_weight(games_2025, games_2024, current, prior2_value,
                                           games_2026=np.array([0.0]))
    at_half_decay = wp.prior2_blend_weight(games_2025, games_2024, current, prior2_value,
                                           games_2026=np.array([wp.PRIOR2_DECAY_GAMES_2026 / 2.0]))
    at_full_decay = wp.prior2_blend_weight(games_2025, games_2024, current, prior2_value,
                                           games_2026=np.array([wp.PRIOR2_DECAY_GAMES_2026]))
    no_decay_requested = wp.prior2_blend_weight(games_2025, games_2024, current, prior2_value)
    assert abs(at_cold_start[0] - no_decay_requested[0]) < 1e-9  # zero 2026 games is a no-op
    assert abs(at_half_decay[0] - no_decay_requested[0] * 0.5) < 1e-9
    assert at_full_decay[0] == 0.0
    # A player who has barely played (1 of 8 games) keeps most of his 2024
    # read - an early-season absence must not burn the fade down for him.
    barely_played = wp.prior2_blend_weight(games_2025, games_2024, current, prior2_value,
                                           games_2026=np.array([1.0]))
    assert barely_played[0] > no_decay_requested[0] * 0.8


def test_prior2_blend_weight_zero_below_minimum_2024_games():
    below_min = wp.prior2_blend_weight(
        games_2025=np.array([4.0]), games_2024=np.array([0.0]),
        current=np.array([0.20]), prior2_value=np.array([0.30]))
    missing = wp.prior2_blend_weight(
        games_2025=np.array([4.0]), games_2024=np.array([np.nan]),
        current=np.array([0.20]), prior2_value=np.array([np.nan]))
    assert below_min[0] == 0.0
    assert missing[0] == 0.0


def test_v2_qb_and_rb_rushing_channels_are_explicitly_separate():
    assert wp.projection_channel('QB', 'rushing_yards') == 'QB rushing'
    assert wp.projection_channel('RB', 'rushing_yards') == 'RB rushing'
    assert wp.projection_channel('RB', 'receiving_yards') == 'RB receiving'


def test_v2_role_profile_keeps_target_earner_rank_continuous():
    df = weekly([
        {'name': 'WR1', 'week': 1, 'team': 'KC', 'position': 'WR', 'targets': 10, 'weekly_snap_pct': 90},
        {'name': 'WR2', 'week': 1, 'team': 'KC', 'position': 'WR', 'targets': 5, 'weekly_snap_pct': 80},
        {'name': 'WR3', 'week': 1, 'team': 'KC', 'position': 'WR', 'targets': 1, 'weekly_snap_pct': 30},
        {'name': 'RB', 'week': 1, 'team': 'KC', 'position': 'RB', 'targets': 4, 'weekly_snap_pct': 60},
    ])
    profile = wp.build_continuous_role_profiles(df, 'name', 'team', 'WR')
    assert profile.loc['WR1', 'target_earner_rank'] == 1
    assert profile.loc['WR1', 'target_share'] == 0.5  # all KC targets, not just WR targets
    assert profile.loc['WR1', 'target_earner_score'] > profile.loc['WR2', 'target_earner_score']
    assert profile.loc['WR2', 'target_earner_score'] > profile.loc['WR3', 'target_earner_score']
    assert not profile.loc['WR1', 'alignment_available']


def test_v2_continuous_role_weights_sum_to_one_without_a_bucket_cliff():
    rows = []
    for i in range(12):
        rows.append({'name': f'W{i}', 'week': 1, 'team': 'KC', 'position': 'WR',
                     'opponent_team': 'DEN', 'targets': 30, 'receiving_air_yards': 30 * (4 + i)})
    weights = wp.build_continuous_role_weights(weekly(rows), 'name', 'WR')
    assert np.allclose(weights.sum(axis=1).to_numpy(), 1.0)
    assert (weights.loc['W5'] > 0).sum() >= 1


def test_v2_as_of_pace_excludes_future_week_rows():
    df = weekly([
        {'name': 'QB', 'week': 1, 'team': 'KC', 'opponent_team': 'DEN', 'position': 'QB',
         'passing_attempts': 30},
        {'name': 'RB', 'week': 1, 'team': 'KC', 'opponent_team': 'DEN', 'position': 'RB',
         'rushing_attempts': 20},
        {'name': 'QB', 'week': 2, 'team': 'KC', 'opponent_team': 'DEN', 'position': 'QB',
         'passing_attempts': 60},
        {'name': 'RB', 'week': 2, 'team': 'KC', 'opponent_team': 'DEN', 'position': 'RB',
         'rushing_attempts': 40},
    ])
    pace = wp.as_of_team_pace(df, 'team', as_of_week=2)
    assert pace.loc['KC', 'off_pace'] == 50
    assert pace.loc['DEN', 'def_pace'] == 50


def test_as_of_team_weekly_plays_excludes_future_week_rows_and_stays_per_game():
    df = weekly([
        {'name': 'QB', 'week': 1, 'team': 'KC', 'opponent_team': 'DEN', 'position': 'QB',
         'passing_attempts': 30},
        {'name': 'RB', 'week': 1, 'team': 'KC', 'opponent_team': 'DEN', 'position': 'RB',
         'rushing_attempts': 20},
        {'name': 'QB', 'week': 2, 'team': 'KC', 'opponent_team': 'DEN', 'position': 'QB',
         'passing_attempts': 60},
        {'name': 'RB', 'week': 2, 'team': 'KC', 'opponent_team': 'DEN', 'position': 'RB',
         'rushing_attempts': 40},
    ])
    plays = wp.as_of_team_weekly_plays(df, 'team', as_of_week=2)
    # Week 2 (>= as_of_week) is excluded, same cutoff as_of_team_pace uses -
    # only the single week-1 game (50 plays) survives.
    assert list(plays['week']) == [1]
    row = plays[(plays['team'] == 'KC') & (plays['week'] == 1)].iloc[0]
    assert row['plays'] == 50


def test_v2_defense_matchup_falls_back_to_prior_season_at_true_cold_start():
    # as_of_week=1 has no current-season "before" week by construction (week
    # numbering starts at 1), so this is every Week 1 decomposition until the
    # first 2026 games are played - defense_matchup must fall back to the
    # full prior season rather than silently staying empty for the entire
    # opening week of a new season, which is exactly when a user would look
    # at this the most.
    current = weekly([
        {'name': 'KC WR', 'team': 'KC', 'opponent_team': 'DEN', 'week': 1,
         'position': 'WR', 'weekly_snap_pct': 85.0, 'targets': 6.0,
         'receptions': 4.0, 'receiving_yards': 50.0},
        {'name': 'DEN WR', 'team': 'DEN', 'opponent_team': 'KC', 'week': 1,
         'position': 'WR', 'weekly_snap_pct': 85.0, 'targets': 6.0,
         'receptions': 4.0, 'receiving_yards': 50.0},
    ])
    prior = weekly([
        {'name': 'KC WR P', 'team': 'KC', 'opponent_team': 'DEN', 'week': 1,
         'position': 'WR', 'weekly_snap_pct': 85.0, 'targets': 6.0,
         'receptions': 4.0, 'receiving_yards': 50.0, 'fantasy_points': 10.0},
        {'name': 'DEN WR P', 'team': 'DEN', 'opponent_team': 'KC', 'week': 1,
         'position': 'WR', 'weekly_snap_pct': 85.0, 'targets': 6.0,
         'receptions': 4.0, 'receiving_yards': 90.0, 'fantasy_points': 20.0},
    ])
    schedule = pd.DataFrame([
        {'week': 1, 'home_team': 'KC', 'away_team': 'DEN', 'home_score': np.nan, 'away_score': np.nan},
    ])
    original = (wp.load_and_merge_data, wp.load_schedule, wp._load_pff_receiving)
    try:
        wp.load_and_merge_data = lambda year, scoring: (
            (current.copy() if year == 2026 else prior.copy()), 'team', 'name', None)
        wp.load_schedule = lambda year: schedule.copy()
        wp._load_pff_receiving = lambda year, allow_season_totals=True: pd.DataFrame()
        # See test_cold_start_manual_qb1_receives_full_prior_per_game_workload's
        # own comment: distinct availability_fingerprint avoids an
        # @st.cache_data collision with that test's identical (year, week,
        # scoring, as_of_week, apply_injury) tuple.
        result, meta = wp.build_weekly_projections(
            2026, 1, 'Full PPR', as_of_week=1, apply_injury=False,
            availability_fingerprint='test_v2_defense_matchup_cold_start')
    finally:
        wp.load_and_merge_data, wp.load_schedule, wp._load_pff_receiving = original
    assert not result.empty
    kc_wr = meta['explanations'][('KC WR', 'WR', 'KC')]
    matchup = kc_wr['defense_matchup']
    assert matchup is not None
    assert matchup['source'] == '2025 full season (no 2026 games played yet)'
    assert matchup['of'] == 2
    # kc_wr's opponent is DEN, so this is DEN's defense profile. DEN's
    # defense faced one prior WR (KC WR P, opponent_team=DEN) who scored
    # 10.0 fantasy_points; KC's defense faced the other (DEN WR P,
    # opponent_team=KC) who scored 20.0. defense_stat_rank's OWN convention
    # (see its test in test_matchup_signals.py) is rank 1 = allows the MOST -
    # DEN allowed fewer (10 < 20), so DEN is rank 2 of 2 by that raw
    # convention (the tougher defense, ranked last by "how much it allows").
    assert matchup['rank'] == 2
    assert matchup['value'] == 10.0


def test_v2_full_projection_contract_is_cutoff_safe_and_explained():
    # Integration fixture: patch only I/O boundaries, then exercise the
    # real V2 build end to end.  Future Week 2 values are deliberately huge;
    # changing them must not change a Week 2 projection built as of Week 2.
    stats = []
    all_stats = set(sum(wp.OFFENSE_PROJECTION_STATS.values(), []))
    for season, week_values in ((2026, {1: 1.0, 2: 100.0}), (2025, {1: 0.9, 2: 1.1})):
        for week, multiplier in week_values.items():
            for team, opponent in (('KC', 'DEN'), ('DEN', 'KC')):
                for pos in wp.DRAFTABLE_POSITIONS:
                    row = {stat: 0.0 for stat in all_stats}
                    row.update({
                        'season': season, 'week': week, 'name': f'{team} {pos}',
                        'team': team, 'opponent_team': opponent, 'position': pos,
                        'weekly_snap_pct': 85.0, 'receiving_air_yards': 20.0 * multiplier,
                    })
                    if pos == 'QB':
                        row.update({'passing_attempts': 30.0 * multiplier,
                                    'passing_completions': 20.0 * multiplier,
                                    'passing_yards': 220.0 * multiplier,
                                    'passing_tds': 1.5 * multiplier,
                                    'rushing_attempts': 4.0 * multiplier,
                                    'rushing_yards': 20.0 * multiplier})
                    elif pos == 'RB':
                        row.update({'rushing_attempts': 14.0 * multiplier,
                                    'rushing_yards': 60.0 * multiplier,
                                    'targets': 4.0 * multiplier,
                                    'receptions': 3.0 * multiplier,
                                    'receiving_yards': 25.0 * multiplier})
                    else:
                        row.update({'targets': 6.0 * multiplier,
                                    'receptions': 4.0 * multiplier,
                                    'receiving_yards': 50.0 * multiplier})
                    stats.append(row)
    current = pd.DataFrame([r for r in stats if r['season'] == 2026])
    prior = pd.DataFrame([r for r in stats if r['season'] == 2025])
    schedule = pd.DataFrame([
        {'week': 2, 'home_team': 'KC', 'away_team': 'DEN', 'home_score': np.nan, 'away_score': np.nan},
    ])
    original = (wp.load_and_merge_data, wp.load_schedule, wp._load_pff_receiving)
    try:
        wp.load_and_merge_data = lambda year, scoring: (
            (current.copy() if year == 2026 else prior.copy()), 'team', 'name', None)
        wp.load_schedule = lambda year: schedule.copy()
        wp._load_pff_receiving = lambda year, allow_season_totals=True: pd.DataFrame()
        first, meta = wp.build_weekly_projections(
            2026, 2, 'Full PPR', as_of_week=2, apply_injury=False)
        current.loc[current['week'] == 2, list(all_stats)] = 99999.0
        second, _ = wp.build_weekly_projections(
            2026, 2, 'Full PPR', as_of_week=2, apply_injury=False)
    finally:
        wp.load_and_merge_data, wp.load_schedule, wp._load_pff_receiving = original
    assert not first.empty
    assert first[['Player', 'Raw Model Proj Pts', 'Calibrated Model Proj Pts']].equals(
        second[['Player', 'Raw Model Proj Pts', 'Calibrated Model Proj Pts']])
    assert meta['source_contract']['historical_target']
    assert meta['source_contract']['pace'] == 'weekly_box_score_proxy'
    assert meta['source_contract']['prior_defense_recency'].startswith('80% full-season')
    assert len(meta['explanations']) == len(first)
    detail = meta['explanations'][('KC QB', 'QB', 'KC')]
    trace = detail['stats']['passing_yards']
    for key in ('build_path', 'raw_prior_rate', 'role_scale', 'blended_rate',
                'defense_estimator', 'defense_current_games', 'defense_prior_games',
                'script_multiplier', 'pace_multiplier', 'availability_multiplier',
                'environment_multiplier', 'pre_vacancy_projection',
                'vacancy_delta', 'final_projection'):
        assert key in trace
    for explanation in meta['explanations'].values():
        for stat in wp.OFFENSE_PROJECTION_STATS[explanation['position']]:
            stat_trace = explanation['stats'][stat]
            assert stat_trace['defense_estimator'] == 'offense-position team-game normalized production'
    assert trace['final_projection'] == detail['stat_line']['passing_yards']


def test_v2_decomposition_refreshes_the_stat_line_after_vacancy_redistribution():
    # The final explanation must describe the selected ranking row, not the
    # pre-vacancy stat line captured inside the per-position loop.
    current = weekly([
        {'name': 'Out WR', 'team': 'KC', 'opponent_team': 'DEN', 'week': 1,
         'position': 'WR', 'weekly_snap_pct': 85.0, 'targets': 8.0,
         'receptions': 5.0, 'receiving_yards': 70.0, 'receiving_tds': 0.5},
        {'name': 'Healthy WR', 'team': 'KC', 'opponent_team': 'DEN', 'week': 1,
         'position': 'WR', 'weekly_snap_pct': 85.0, 'targets': 5.0,
         'receptions': 3.0, 'receiving_yards': 40.0, 'receiving_tds': 0.2},
        {'name': 'DEN WR', 'team': 'DEN', 'opponent_team': 'KC', 'week': 1,
         'position': 'WR', 'weekly_snap_pct': 85.0, 'targets': 5.0,
         'receptions': 3.0, 'receiving_yards': 40.0, 'receiving_tds': 0.2},
    ])
    prior = current.copy()
    prior['week'] = 18
    schedule = pd.DataFrame([{'week': 2, 'home_team': 'KC', 'away_team': 'DEN'}])
    original = (wp.load_and_merge_data, wp.load_schedule, wp._load_pff_receiving,
                wp.load_team_pace, wp._target_margins_by_team, wp._injury_profiles,
                wp.load_fantasypros_availability)
    try:
        wp.load_and_merge_data = lambda year, scoring: (
            (current.copy() if year == 2026 else prior.copy()), 'team', 'name', None)
        wp.load_schedule = lambda year: schedule.copy()
        wp._load_pff_receiving = lambda year, allow_season_totals=True: pd.DataFrame()
        wp.load_team_pace = lambda year: pd.DataFrame()
        wp._target_margins_by_team = lambda year, week: {}
        wp._injury_profiles = lambda year, week: {
            'Out WR': {'plays_probability': 0.0, 'workload_if_active': 1.0, 'status': 'out'},
        }
        # v2_fantasypros_availability is in DEFAULT_FEATURES and takes
        # precedence over v2_availability's nflverse-backed _injury_profiles -
        # mock the source this run actually calls, not the one it superseded.
        wp.load_fantasypros_availability = lambda year, week: (
            {'Out WR': {'plays_probability': 0.0, 'workload_if_active': 1.0, 'status': 'out',
                       'source': 'FantasyPros injury report'}}, None)
        out, meta = wp.build_weekly_projections(
            2026, 2, 'Full PPR', as_of_week=2, apply_injury=True)
    finally:
        (wp.load_and_merge_data, wp.load_schedule, wp._load_pff_receiving,
         wp.load_team_pace, wp._target_margins_by_team, wp._injury_profiles,
         wp.load_fantasypros_availability) = original
    healthy = out.loc[out['Player'] == 'Healthy WR'].iloc[0]
    detail = meta['explanations'][('Healthy WR', 'WR', 'KC')]
    trace = detail['stats']['targets']
    assert trace['vacancy_delta'] > 0
    assert detail['stat_line']['targets'] == float(healthy['targets'])
    assert trace['final_projection'] == float(healthy['targets'])


# --- teammate vacancy ---------------------------------------------------------

def test_vacated_targets_move_to_healthy_teammates_in_proportion():
    frame = pd.DataFrame({
        'Player': ['Out', 'Big', 'Small', 'OtherTeam'],
        'Pos': ['WR', 'WR', 'WR', 'WR'],
        'Team': ['KC', 'KC', 'KC', 'DEN'],
        'targets': [0.0, 8.0, 4.0, 8.0],
        'receiving_yards': [0.0, 80.0, 40.0, 80.0],
    })
    # `_full_targets` is what the position loop stashes before the injury
    # discount zeroes an Out player's own line - without it there is nothing
    # left to redistribute, which is the whole point of carrying it.
    frame['_full_targets'] = [9.0, 8.0, 4.0, 8.0]
    out, n = wp.redistribute_vacated_usage(frame, {'Out': 0.0})
    assert n == 2
    # 9 vacated targets, 75% of them re-used, split 2:1 by existing usage.
    assert out.loc[1, 'targets'] > out.loc[2, 'targets']
    assert out.loc[1, 'targets'] > 8.0 and out.loc[2, 'targets'] > 4.0
    assert out.loc[3, 'targets'] == 8.0          # a different team is untouched
    # Dependent stats ride the same factor, so yards per target is unchanged.
    assert abs(out.loc[1, 'receiving_yards'] / out.loc[1, 'targets'] - 10.0) < 1e-6


def test_vacancy_growth_is_capped():
    frame = pd.DataFrame({
        'Player': ['Out', 'Only'],
        'Pos': ['RB', 'RB'],
        'Team': ['KC', 'KC'],
        'rushing_attempts': [20.0, 4.0],
        'rushing_yards': [80.0, 16.0],
    })
    frame['_full_rushing_attempts'] = [50.0, 4.0]
    out, _n = wp.redistribute_vacated_usage(frame, {'Out': 0.4})
    assert out.loc[1, 'rushing_attempts'] <= 4.0 * wp.VACANCY_MAX_GROWTH + 1e-9


def test_v2_vacancy_keeps_qb_passes_and_wr_targets_in_their_own_roles():
    frame = pd.DataFrame({
        'Player': ['Out QB', 'Backup QB', 'Out WR', 'WR Two', 'RB One'],
        'Pos': ['QB', 'QB', 'WR', 'WR', 'RB'],
        'Team': ['KC', 'KC', 'KC', 'KC', 'KC'],
        'Expected Snap Share': [1.0, 0.7, 0.9, 0.8, 0.7],
        'passing_attempts': [0.0, 12.0, np.nan, np.nan, np.nan],
        'passing_yards': [0.0, 100.0, np.nan, np.nan, np.nan],
        'targets': [np.nan, np.nan, 0.0, 5.0, 3.0],
        'receptions': [np.nan, np.nan, 0.0, 3.0, 2.0],
        'receiving_yards': [np.nan, np.nan, 0.0, 35.0, 20.0],
    })
    frame['_full_passing_attempts'] = [32.0, 12.0, np.nan, np.nan, np.nan]
    frame['_full_targets'] = [np.nan, np.nan, 9.0, 5.0, 3.0]
    profiles = {
        'Out QB': {'plays_probability': 0.0},
        'Out WR': {'plays_probability': 0.0},
    }
    out, n, ledger = wp.redistribute_v2_vacated_usage(frame, profiles)
    assert n >= 2 and ledger
    assert out.loc[1, 'passing_attempts'] > 12.0
    assert out.loc[3, 'targets'] > 5.0
    assert out.loc[4, 'targets'] == 3.0  # WR absence does not feed the RB pool.


def test_v2_qb_vacancy_never_rehydrates_a_nonstarter_qb_projection():
    """An injured QB's pass volume can only go to the selected QB1."""
    frame = pd.DataFrame({
        'Player': ['Out QB', 'Selected QB', 'Backup QB'],
        'Pos': ['QB', 'QB', 'QB'],
        'Team': ['KC', 'KC', 'KC'],
        'Expected Snap Share': [1.0, 1.0, 0.2],
        'QB Projected Starter': [True, True, False],
        'passing_attempts': [0.0, 12.0, 9.0],
        'passing_yards': [0.0, 96.0, 72.0],
    })
    frame['_full_passing_attempts'] = [30.0, 12.0, 9.0]
    out, n, ledger = wp.redistribute_v2_vacated_usage(
        frame, {'Out QB': {'plays_probability': 0.0}})
    assert n == 1
    assert out.loc[1, 'passing_attempts'] > 12.0
    assert out.loc[2, 'passing_attempts'] == 9.0
    assert out.loc[2, 'passing_yards'] == 72.0
    assert any(entry['volume'] == 'passing_attempts' and entry['allocated'] > 0 for entry in ledger)

    unresolved = frame.copy()
    unresolved['QB Projected Starter'] = False
    unresolved_out, unresolved_n, unresolved_ledger = wp.redistribute_v2_vacated_usage(
        unresolved, {'Out QB': {'plays_probability': 0.0}})
    assert unresolved_n == 0
    assert unresolved_out.loc[1, 'passing_attempts'] == 12.0
    assert unresolved_out.loc[2, 'passing_attempts'] == 9.0
    assert any(entry['volume'] == 'passing_attempts' and entry['unallocated'] > 0
               for entry in unresolved_ledger)


def test_scoring_a_cross_position_frame_survives_missing_stat_columns():
    # Regression: the assembled frame is the UNION of four positions' stat
    # lists, so a receiver's row carries NaN for every passing stat.
    # score_projected_stats reads with .get(stat, 0), which returns the NaN
    # for a key that exists and is NaN - and max(0.0, nan) is 0.0, so a
    # whole position silently projected exactly zero points while showing a
    # sensible stat line. Anything re-scoring a mixed frame must fill first.
    from data.transforms import score_projected_stats
    row = {'receiving_yards': 80.0, 'receptions': 6.0, 'receiving_tds': 0.5,
           'passing_yards': float('nan'), 'passing_tds': float('nan')}
    assert np.isnan(score_projected_stats(row, 'Full PPR'))
    filled = {k: (0.0 if v != v else v) for k, v in row.items()}
    assert score_projected_stats(filled, 'Full PPR') > 10


def test_vacancy_is_a_no_op_with_nobody_out():
    frame = pd.DataFrame({'Player': ['A'], 'Pos': ['WR'], 'Team': ['KC'], 'targets': [6.0]})
    out, n = wp.redistribute_vacated_usage(frame, {'A': 1.0})
    assert n == 0 and out.loc[0, 'targets'] == 6.0


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
