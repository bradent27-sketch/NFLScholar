"""Regression contracts for the experimental preseason RB role allocator.

These tests deliberately target the small, data-frame-only seam in
``data.rb_role_allocator`` rather than the Streamlit app or the full weekly
projection build.  That makes the football contracts below auditable:

* a fullback is not silently treated as a normal running back;
* credible backs share a finite team backfield rather than each receiving an
  independent league-average role;
* an interrupted returning role fades continuously instead of jumping at an
  arbitrary 60% active-snap cutoff; and
* injury vacancy volume is tied to a current, role-compatible source.

Shares in these fixtures are fractions (``0.75`` means 75%), not percentages.
The allocator's public contract is intentionally compact:

``allocate_preseason_rb_roles(candidates) -> (allocations, ledger)``
``redistribute_rb_vacancy_with_allocator(result, injury_profiles, ...) -> (result, ledger)``

The ledger is accepted as either a DataFrame or a list of dictionaries, but
must expose the audit fields asserted below.
"""
import os
import sys

import numpy as np
import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.rb_role_allocator as rba  # noqa: E402


def _candidate(team, player, **overrides):
    """One Week-1 candidate using the allocator's reviewable input schema."""
    row = {
        'team': team,
        'Player': player,
        'player_id': f'id-{player.lower().replace(" ", "-")}',
        'position': 'RB',
        'depth_chart_position': 'RB',
        'ourlads_position': 'RB',
        'prior_pff_position': 'RB',
        'ourlads_depth_rank': 9,
        'prior_active_snap_share': 0.12,
        'prior_whole_snap_share': 0.08,
        'pre_absence_snap_share': np.nan,
        'prior_active_carry_share': 0.06,
        'prior_active_target_share': 0.03,
        'prior_games': 3,
        'same_team': False,
        'draft_capital': 0.0,
        'availability': 1.0,
        'is_active': True,
        # A team owns three distinct capacities.  They intentionally need
        # not equal one another because empty-back / two-back formations are
        # real and a receiving back need not be an early-down carry leader.
        'core_rb_snap_capacity': 0.90,
        'rb_carry_capacity': 0.92,
        'rb_target_capacity': 0.72,
    }
    row.update(overrides)
    return row


def _ledger_frame(ledger):
    frame = ledger.copy() if isinstance(ledger, pd.DataFrame) else pd.DataFrame(ledger)
    # ``metric`` is an equally clear implementation name for the three
    # capacity resources.  Normalize it here so this test protects behavior,
    # not a cosmetic ledger heading.
    if 'resource' not in frame.columns and 'metric' in frame.columns:
        frame = frame.rename(columns={'metric': 'resource'})
    return frame


def _column(frame, *names):
    for name in names:
        if name in frame.columns:
            return name
    raise AssertionError(f'Missing one of {names}; found {list(frame.columns)}')


def _player_row(frame, player):
    player_col = _column(frame, 'Player', 'player')
    rows = frame.loc[frame[player_col].eq(player)]
    assert len(rows) == 1, f'Expected exactly one allocation for {player}, got {len(rows)}'
    return rows.iloc[0]


def _team_rows(frame, team):
    return frame.loc[frame[_column(frame, 'Team', 'team')].astype(str).eq(team)]


def _allocation_and_ledger(candidates):
    allocations, ledger = rba.allocate_preseason_rb_roles(pd.DataFrame(candidates))
    assert isinstance(allocations, pd.DataFrame)
    required = {'expected_snap_share', 'carry_share', 'target_share', 'allocated_carries',
                'allocated_targets', 'core_rb', 'allocation_source'}
    assert required.issubset(allocations.columns), (
        f'Allocation output must expose {sorted(required)}, got {list(allocations.columns)}')
    ledger = _ledger_frame(ledger)
    required_ledger = {'team', 'resource', 'capacity', 'allocated', 'unallocated'}
    assert required_ledger.issubset(ledger.columns), (
        f'Allocator ledger must expose {sorted(required_ledger)}, got {list(ledger.columns)}')
    return allocations, ledger


def _assert_team_capacity_conservation(allocations, ledger, team):
    """The player allocations plus explicit unallocated bucket equal capacity."""
    team_allocations = _team_rows(allocations, team)
    core = team_allocations.loc[team_allocations['core_rb'].astype(bool)]
    expected = {
        'core_rb_snaps': 'expected_snap_share',
        # The share columns are fractions of these two count capacities;
        # physical allocation is what must reconcile to a count ledger.
        'rb_carries': 'allocated_carries',
        'rb_targets': 'allocated_targets',
    }
    for resource, metric in expected.items():
        rows = ledger.loc[(ledger['team'].astype(str) == team)
                          & (ledger['resource'].astype(str) == resource)]
        assert len(rows) == 1, f'{team} needs one {resource} capacity ledger row'
        entry = rows.iloc[0]
        allocated = float(core[metric].sum())
        assert np.isclose(allocated, float(entry['allocated']), atol=1e-9), (
            f'{team} {resource}: player allocations must reconcile to the ledger')
        assert np.isclose(allocated + float(entry['unallocated']), float(entry['capacity']), atol=1e-9), (
            f'{team} {resource}: no volume may appear or disappear outside the explicit bucket')


def test_functional_position_prioritizes_depth_chart_then_ourlads_then_prior_source():
    frame = pd.DataFrame([
        # Roster says FB even though the fantasy roster's broad position is RB.
        {'Player': 'Alec Ingold', 'position': 'RB', 'depth_chart_position': 'FB',
         'ourlads_position': 'RB', 'prior_pff_position': 'RB'},
        # An absent roster subposition can be resolved from the imported chart.
        {'Player': 'Chart Fullback', 'position': 'RB', 'depth_chart_position': '',
         'ourlads_position': 'FB', 'prior_pff_position': 'RB'},
        # Older PFF/source history is only the final fallback.
        {'Player': 'Historical Fullback', 'position': 'RB', 'depth_chart_position': '',
         'ourlads_position': '', 'prior_pff_position': 'FB'},
        {'Player': 'Core Back', 'position': 'RB', 'depth_chart_position': 'RB',
         'ourlads_position': 'FB', 'prior_pff_position': 'FB'},
    ])
    functional = rba.classify_functional_position(frame)
    assert list(pd.Series(functional).astype(str)) == ['FB', 'FB', 'FB', 'RB']


def test_fullback_is_not_given_core_rb_fallback_and_inactive_rows_do_not_enter_pool():
    candidates = [
        _candidate('MIA', 'Devon Achane', ourlads_depth_rank=1,
                   prior_active_snap_share=0.78, prior_whole_snap_share=0.70,
                   pre_absence_snap_share=0.78, prior_active_carry_share=0.62,
                   prior_active_target_share=0.38, prior_games=16, same_team=True),
        _candidate('MIA', 'Reserve Back', ourlads_depth_rank=2,
                   prior_active_snap_share=0.22, prior_whole_snap_share=0.18,
                   prior_active_carry_share=0.12, prior_active_target_share=0.08,
                   prior_games=12, same_team=True),
        # This is the regression: the broad roster position is RB, but he is
        # functionally a fullback and must not inherit the RB fallback.
        _candidate('MIA', 'Alec Ingold', depth_chart_position='FB', ourlads_position='FB',
                   ourlads_depth_rank=1, prior_active_snap_share=0.18,
                   prior_whole_snap_share=0.16, prior_active_carry_share=0.02,
                   prior_active_target_share=0.03, prior_games=17, same_team=True),
        _candidate('MIA', 'Retired Fullback', depth_chart_position='FB', ourlads_position='FB',
                   is_active=False, prior_active_snap_share=0.20, prior_games=12),
    ]
    allocations, ledger = _allocation_and_ledger(candidates)
    ingold = _player_row(allocations, 'Alec Ingold')
    assert not bool(ingold['core_rb'])
    # Own-use may be nonzero, but it cannot become a generic RB population
    # role (historical 2% carry / 3% target share here).
    assert float(ingold['carry_share']) <= 0.03
    assert float(ingold['target_share']) <= 0.04
    assert 'fullback' in str(ingold['allocation_source']).lower()
    assert 'Retired Fullback' not in set(allocations[_column(allocations, 'Player', 'player')])
    _assert_team_capacity_conservation(allocations, ledger, 'MIA')


def test_player_id_and_literal_ourlads_rank_keep_gainwell_ahead_of_tucker():
    """Kenny/Kenneth is one player; an unresolved alias must not promote RB3."""
    common = {
        'prior_active_snap_share': 0.28,
        'prior_whole_snap_share': 0.22,
        'prior_active_carry_share': 0.14,
        'prior_active_target_share': 0.11,
        'prior_games': 14,
        'same_team': False,
    }
    candidates = [
        _candidate('TB', 'Bucky Irving', player_id='00-0040001', ourlads_depth_rank=1,
                   **common),
        # Current roster spelling differs from the historical/Ourlads spelling,
        # but GSIS is stable and this remains the literal #2 row.
        _candidate('TB', 'Kenneth Gainwell', player_id='00-0036919',
                   historical_name='Kenny Gainwell', ourlads_player='Kenny Gainwell',
                   ourlads_depth_rank=2, **common),
        _candidate('TB', 'Sean Tucker', player_id='00-0039924', ourlads_depth_rank=3,
                   **common),
    ]
    allocations, _ = _allocation_and_ledger(candidates)
    player_col = _column(allocations, 'Player', 'player')
    assert not allocations[player_col].eq('Kenny Gainwell').any()
    gainwell = _player_row(allocations, 'Kenneth Gainwell')
    tucker = _player_row(allocations, 'Sean Tucker')
    # Holding all other evidence equal makes this a direct test of preserving
    # the imported order rather than re-enumerating only successful name joins.
    assert float(gainwell['expected_snap_share']) > float(tucker['expected_snap_share'])
    assert float(gainwell['carry_share']) > float(tucker['carry_share'])


def test_jacksonville_core_backfield_reconciles_to_three_separate_team_capacities():
    candidates = [
        _candidate('JAX', 'Bhayshul Tuten', ourlads_depth_rank=1, draft_capital=3,
                   prior_active_snap_share=0.21, prior_whole_snap_share=0.18,
                   prior_active_carry_share=0.18, prior_active_target_share=0.10,
                   prior_games=10),
        _candidate('JAX', 'Chris Rodriguez', ourlads_depth_rank=2,
                   prior_active_snap_share=0.33, prior_whole_snap_share=0.23,
                   prior_active_carry_share=0.28, prior_active_target_share=0.06,
                   prior_games=13),
        _candidate('JAX', 'LeQuint Allen', ourlads_depth_rank=3,
                   prior_active_snap_share=0.30, prior_whole_snap_share=0.25,
                   prior_active_carry_share=0.10, prior_active_target_share=0.26,
                   prior_games=14),
        # No corroborating role, draft, or chart signal: this player must not
        # receive an automatic league-median role merely for being rostered.
        _candidate('JAX', 'Unknown Reserve', ourlads_depth_rank=9,
                   prior_active_snap_share=0.0, prior_whole_snap_share=0.0,
                   prior_active_carry_share=0.0, prior_active_target_share=0.0,
                   prior_games=0, availability=1.0),
    ]
    allocations, ledger = _allocation_and_ledger(candidates)
    for player in ('Bhayshul Tuten', 'Chris Rodriguez', 'LeQuint Allen'):
        assert bool(_player_row(allocations, player)['core_rb'])
    reserve = _player_row(allocations, 'Unknown Reserve')
    assert not bool(reserve['core_rb']) or float(reserve['expected_snap_share']) <= 0.05
    _assert_team_capacity_conservation(allocations, ledger, 'JAX')


def test_uncharted_team_does_not_turn_a_median_placeholder_into_a_reserve_role():
    """Missing DET/PIT pages must not fabricate a role for every roster RB."""
    candidates = [
        _candidate('DET', 'Proven Veteran', ourlads_depth_rank=np.nan,
                   prior_active_snap_share=0.55, prior_whole_snap_share=0.50,
                   prior_games=12, has_observed_prior_role=True, same_team=True),
        # A highly drafted rookie can remain a credible no-chart candidate,
        # but a generic cold-pool median (about 16%) is not evidence by itself.
        _candidate('DET', 'High Draft Rookie', ourlads_depth_rank=np.nan,
                   prior_active_snap_share=np.nan, prior_whole_snap_share=np.nan,
                   prior_games=0, has_observed_prior_role=False, draft_capital=35,
                   is_rookie=True),
        _candidate('DET', 'Unobserved Reserve', ourlads_depth_rank=np.nan,
                   base_snap_share=0.166, prior_active_snap_share=np.nan,
                   prior_whole_snap_share=np.nan, prior_games=0,
                   has_observed_prior_role=False, draft_capital=np.nan),
    ]
    allocations, _ = _allocation_and_ledger(candidates)
    assert bool(_player_row(allocations, 'Proven Veteran')['eligible_core_rb'])
    assert bool(_player_row(allocations, 'High Draft Rookie')['eligible_core_rb'])
    reserve = _player_row(allocations, 'Unobserved Reserve')
    assert not bool(reserve['eligible_core_rb'])
    assert float(reserve['expected_snap_share']) == 0.0


def test_returning_role_recovery_is_continuous_on_both_sides_of_old_sixty_percent_cliff():
    candidates = [
        _candidate('A59', 'Returning 59', ourlads_depth_rank=1, same_team=True, prior_games=11,
                   prior_whole_snap_share=0.35, prior_active_snap_share=0.59,
                   pre_absence_snap_share=0.80, prior_active_carry_share=0.48,
                   prior_active_target_share=0.20),
        _candidate('A59', 'A59 Reserve', ourlads_depth_rank=2, same_team=True, prior_games=10,
                   prior_whole_snap_share=0.25, prior_active_snap_share=0.30,
                   prior_active_carry_share=0.20, prior_active_target_share=0.12),
        _candidate('A61', 'Returning 61', ourlads_depth_rank=1, same_team=True, prior_games=11,
                   prior_whole_snap_share=0.35, prior_active_snap_share=0.61,
                   pre_absence_snap_share=0.80, prior_active_carry_share=0.48,
                   prior_active_target_share=0.20),
        _candidate('A61', 'A61 Reserve', ourlads_depth_rank=2, same_team=True, prior_games=10,
                   prior_whole_snap_share=0.25, prior_active_snap_share=0.30,
                   prior_active_carry_share=0.20, prior_active_target_share=0.12),
    ]
    allocations, _ = _allocation_and_ledger(candidates)
    below = float(_player_row(allocations, 'Returning 59')['expected_snap_share'])
    above = float(_player_row(allocations, 'Returning 61')['expected_snap_share'])
    # 59% active evidence still receives real returning-role credit; a two
    # percentage-point input difference cannot create a discontinuous 25% role jump.
    assert below > 0.43
    assert above > 0.43
    assert 0.0 <= above - below < 0.08


def _vacancy_result_frame():
    """A narrow MIA frame that isolates improper FB/WR -> RB vacancy leakage."""
    frame = pd.DataFrame([
        {'Player': 'Devon Achane', 'Pos': 'RB', 'Team': 'MIA', 'functional_position': 'RB',
         'Expected Snap Share': 0.78, 'rushing_attempts': 15.0, 'rushing_yards': 72.0,
         'targets': 6.0, 'receptions': 4.5, 'receiving_yards': 38.0,
         '_full_rushing_attempts': 15.0, '_full_targets': 6.0},
        {'Player': 'Reserve RB', 'Pos': 'RB', 'Team': 'MIA', 'functional_position': 'RB',
         'Expected Snap Share': 0.20, 'rushing_attempts': 4.0, 'rushing_yards': 17.0,
         'targets': 2.0, 'receptions': 1.5, 'receiving_yards': 11.0,
         '_full_rushing_attempts': 4.0, '_full_targets': 2.0},
        {'Player': 'Alec Ingold', 'Pos': 'RB', 'Team': 'MIA', 'functional_position': 'FB',
         'Expected Snap Share': 0.18, 'rushing_attempts': 0.0, 'rushing_yards': 0.0,
         'targets': 0.0, 'receptions': 0.0, 'receiving_yards': 0.0,
         '_full_rushing_attempts': 1.0, '_full_targets': 1.0},
        {'Player': 'Out WR', 'Pos': 'WR', 'Team': 'MIA', 'functional_position': 'WR',
         'Expected Snap Share': 0.85, 'rushing_attempts': np.nan, 'rushing_yards': np.nan,
         'targets': 0.0, 'receptions': 0.0, 'receiving_yards': 0.0,
         '_full_rushing_attempts': np.nan, '_full_targets': 10.0},
        {'Player': 'Healthy WR', 'Pos': 'WR', 'Team': 'MIA', 'functional_position': 'WR',
         'Expected Snap Share': 0.82, 'rushing_attempts': np.nan, 'rushing_yards': np.nan,
         'targets': 5.0, 'receptions': 3.5, 'receiving_yards': 44.0,
         '_full_rushing_attempts': np.nan, '_full_targets': 5.0},
    ])
    return frame


def _vacancy_ledger(ledger):
    frame = _ledger_frame(ledger)
    aliases = {
        'functional_role': 'functional_source_role',
        'injury_source_provenance': 'injury_provenance',
    }
    frame = frame.rename(columns={old: new for old, new in aliases.items()
                                  if new not in frame.columns and old in frame.columns})
    required = {'team', 'source_player', 'functional_source_role', 'injury_provenance',
                'recipients', 'allocated', 'unallocated'}
    assert required.issubset(frame.columns), (
        f'Vacancy ledger must expose {sorted(required)}, got {list(frame.columns)}')
    return frame


def test_vacancy_ignores_fullback_and_wr_sources_for_achane_core_rb_volume():
    original = _vacancy_result_frame()
    profiles = {
        'Alec Ingold': {'plays_probability': 0.0, 'status': 'out'},
        'Out WR': {'plays_probability': 0.0, 'status': 'out'},
    }
    provenance = {
        'Alec Ingold': {'year': 2026, 'week': 1, 'source': 'current_week_report'},
        'Out WR': {'year': 2026, 'week': 1, 'source': 'current_week_report'},
    }
    out, ledger = rba.redistribute_rb_vacancy_with_allocator(
        original, profiles, as_of_year=2026, injury_provenance=provenance)
    before = _player_row(original, 'Devon Achane')
    achane = _player_row(out, 'Devon Achane')
    # This protects the reported 2.63-carry bug: FB and WR availability must
    # not create core-RB carries (nor generic RB targets) for Achane.
    for metric in ('rushing_attempts', 'rushing_yards', 'targets', 'receptions', 'receiving_yards'):
        assert float(achane[metric]) == float(before[metric]), metric
    # A WR vacancy is allowed to reallocate targets, but only within the
    # healthy WR/TE pool.  Its essential contract here is that it cannot
    # leak into Achane's carries or targets.
    assert float(_player_row(out, 'Healthy WR')['targets']) > 5.0
    ledger = _vacancy_ledger(ledger)
    # The decomposition should retain the fullback source and explicitly
    # show that none of his nominal volume was allocated to core RBs.
    ingold = ledger.loc[ledger['source_player'].eq('Alec Ingold')]
    assert not ingold.empty
    assert float(ingold['allocated'].sum()) == 0.0
    assert ingold['functional_source_role'].eq('FB').all()


def test_core_rb_vacancy_keeps_carries_and_targets_in_separate_core_rb_recipient_pools():
    frame = pd.DataFrame([
        {'Player': 'Out Lead RB', 'Pos': 'RB', 'Team': 'KC', 'functional_position': 'RB',
         '_rb_core': True, 'Expected Snap Share': 0.60, 'rushing_attempts': 0.0,
         'rushing_yards': 0.0, 'targets': 0.0, 'receptions': 0.0, 'receiving_yards': 0.0,
         '_full_rushing_attempts': 13.0, '_full_targets': 4.0,
         '_rb_carry_allocation_share': 0.60, '_rb_target_allocation_share': 0.25},
        {'Player': 'Early Down RB', 'Pos': 'RB', 'Team': 'KC', 'functional_position': 'RB',
         '_rb_core': True, 'Expected Snap Share': 0.50, 'rushing_attempts': 10.0,
         'rushing_yards': 45.0, 'targets': 1.0, 'receptions': 0.7, 'receiving_yards': 6.0,
         '_full_rushing_attempts': 10.0, '_full_targets': 1.0,
         '_rb_carry_allocation_share': 0.82, '_rb_target_allocation_share': 0.10},
        {'Player': 'Receiving RB', 'Pos': 'RB', 'Team': 'KC', 'functional_position': 'RB',
         '_rb_core': True, 'Expected Snap Share': 0.35, 'rushing_attempts': 3.0,
         'rushing_yards': 12.0, 'targets': 5.0, 'receptions': 3.7, 'receiving_yards': 30.0,
         '_full_rushing_attempts': 3.0, '_full_targets': 5.0,
         '_rb_carry_allocation_share': 0.08, '_rb_target_allocation_share': 0.90},
        {'Player': 'Fullback', 'Pos': 'RB', 'Team': 'KC', 'functional_position': 'FB',
         '_rb_core': False, 'Expected Snap Share': 0.20, 'rushing_attempts': 0.0,
         'rushing_yards': 0.0, 'targets': 0.0, 'receptions': 0.0, 'receiving_yards': 0.0,
         '_full_rushing_attempts': 1.0, '_full_targets': 1.0,
         '_rb_carry_allocation_share': 0.0, '_rb_target_allocation_share': 0.0},
        {'Player': 'Wide Receiver', 'Pos': 'WR', 'Team': 'KC', 'functional_position': 'WR',
         '_rb_core': False, 'Expected Snap Share': 0.80, 'rushing_attempts': np.nan,
         'rushing_yards': np.nan, 'targets': 6.0, 'receptions': 4.0, 'receiving_yards': 55.0,
         '_full_rushing_attempts': np.nan, '_full_targets': 6.0,
         '_rb_carry_allocation_share': 0.0, '_rb_target_allocation_share': 0.0},
    ])
    out, ledger = rba.redistribute_rb_vacancy_with_allocator(
        frame, {'Out Lead RB': {'plays_probability': 0.0, 'status': 'out'}},
        as_of_year=2026,
        injury_provenance={'Out Lead RB': {'year': 2026, 'week': 1, 'source': 'current_week_report'}})
    assert float(_player_row(out, 'Early Down RB')['rushing_attempts']) > 10.0
    assert float(_player_row(out, 'Receiving RB')['targets']) > 5.0
    assert float(_player_row(out, 'Fullback')['rushing_attempts']) == 0.0
    assert float(_player_row(out, 'Wide Receiver')['targets']) == 6.0
    ledger = _vacancy_ledger(ledger)
    source = ledger.loc[ledger['source_player'].eq('Out Lead RB')]
    assert len(source) == 2
    assert source['functional_source_role'].astype(str).str.contains('core', case=False).all()
    assert source['injury_provenance'].astype(str).str.contains('current_week_report').all()
    assert float(source['allocated'].sum()) > 0.0
    assert source['recipients'].map(bool).all()


def test_stale_prior_season_injury_cannot_trigger_week_one_vacancy_redistribution():
    original = _vacancy_result_frame()
    # Put a plausible historical RB absence into the frame.  Without a
    # provenance gate it would incorrectly boost Achane in the next season.
    old = {
        'Player': 'Old Out RB', 'Pos': 'RB', 'Team': 'MIA', 'functional_position': 'RB',
        'Expected Snap Share': 0.40, 'rushing_attempts': 0.0, 'rushing_yards': 0.0,
        'targets': 0.0, 'receptions': 0.0, 'receiving_yards': 0.0,
        '_full_rushing_attempts': 12.0, '_full_targets': 3.0,
    }
    original = pd.concat([original, pd.DataFrame([old])], ignore_index=True)
    profiles = {'Old Out RB': {'plays_probability': 0.0, 'status': 'out'}}
    out, ledger = rba.redistribute_rb_vacancy_with_allocator(
        original, profiles, as_of_year=2026,
        injury_provenance={'Old Out RB': {'year': 2025, 'week': 17, 'source': 'historical_cache'}})
    # Metadata columns are allowed, but no stat line may change because this
    # designation came from the preceding season.
    visible = ['rushing_attempts', 'rushing_yards', 'targets', 'receptions', 'receiving_yards']
    pd.testing.assert_frame_equal(out[visible], original[visible])
    ledger = _vacancy_ledger(ledger)
    stale = ledger.loc[ledger['source_player'].eq('Old Out RB')]
    assert not stale.empty
    assert stale['injury_provenance'].astype(str).str.contains('stale', case=False).any()
    assert float(stale['allocated'].sum()) == 0.0


def _hampton_vidal_role_segment_history():
    """A clear lead-back injury gap, then a return, on immutable LAC game rows."""
    rows = []
    hampton_snaps = {1: 80, 2: 62, 3: 79, 4: 89, 5: 58, 13: 31, 14: 36, 15: 55, 16: 81}
    vidal_snaps = {4: 3, 5: 21, 6: 67, 7: 64, 8: 74, 9: 72, 10: 93, 11: 52,
                   12: 76, 13: 69, 14: 64, 15: 33, 16: 20}
    for player, player_id, snaps in (
        ('Omarion Hampton', '00-0040666', hampton_snaps),
        ('Kimani Vidal', '00-0040123', vidal_snaps),
    ):
        for week, snap in snaps.items():
            rows.append({
                'player_display_name': player,
                'player_id': player_id,
                'position': 'RB',
                # Deliberately wrong latest-roster value: game_team must win.
                'team': 'XXX',
                'game_team': 'LAC',
                'season': 2025,
                'week': week,
                'weekly_snap_pct': snap,
                'has_snap_match': True,
                'rushing_attempts': max(1, round(snap / 5)),
                'targets': max(0, round(snap / 25)),
            })
    calendar = pd.DataFrame({
        'game_team': ['LAC'] * 16,
        'season': [2025] * 16,
        'week': list(range(1, 17)),
    })
    return pd.DataFrame(rows), calendar


def test_role_segments_detect_internal_absence_return_and_retain_healthy_teammate_context():
    history, calendar = _hampton_vidal_role_segment_history()
    segments, context = rba.analyze_rb_role_segments(history, team_game_calendar=calendar)
    required_segment = {
        'rb_segment_identity_key', 'rb_segment_team', 'interrupted_season',
        'pre_absence_snap_share', 'absence_team_games', 'return_recovery_snap_share',
        'interrupted_incumbent_role_credit', 'absence_replacement_top_rb_snap_share',
    }
    required_context = {
        'incumbent_identity_key', 'teammate_identity_key', 'shared_healthy_games',
        'shared_healthy_lead_score', 'teammate_absence_replacement_snap_share',
        'replacement_only_era_downweight',
    }
    assert required_segment.issubset(segments.columns)
    assert required_context.issubset(context.columns)
    hampton = segments.loc[segments['rb_segment_player'].eq('Omarion Hampton')].iloc[0]
    assert hampton['rb_segment_team'] == 'LAC'  # not the deliberately stale XXX roster value
    assert bool(hampton['interrupted_season'])
    assert int(hampton['absence_team_games']) == 7
    # The last four pre-gap games, not the final recovery appearances, define
    # the incumbent's pre-absence role: (62 + 79 + 89 + 58) / 4 = 72%.
    assert np.isclose(float(hampton['pre_absence_snap_share']), 0.72)
    assert np.isclose(float(hampton['return_recovery_snap_share']), (0.31 + 0.36 + 0.55 + 0.81) / 4)
    assert float(hampton['interrupted_incumbent_role_credit']) > 0.90
    pair = context.loc[(context['incumbent_player'].eq('Omarion Hampton'))
                       & (context['teammate_player'].eq('Kimani Vidal'))].iloc[0]
    # Only W4/W5 were genuine shared healthy work; Vidal's seven-game surge
    # happened while Hampton was absent and cannot be read as an equal split.
    assert int(pair['shared_healthy_games']) == 2
    assert np.isclose(float(pair['incumbent_shared_healthy_snap_share']), (0.89 + 0.58) / 2)
    assert np.isclose(float(pair['teammate_shared_healthy_snap_share']), (0.03 + 0.21) / 2)
    assert float(pair['shared_healthy_lead_score']) > 0.65
    assert float(pair['teammate_absence_replacement_snap_share']) > 0.65
    assert float(pair['replacement_only_era_downweight']) > 0.25


def test_role_segment_allocator_fields_expose_incumbent_credit_and_replacement_downweight():
    history, calendar = _hampton_vidal_role_segment_history()
    segments, context = rba.analyze_rb_role_segments(history, team_game_calendar=calendar)
    candidates = pd.DataFrame([
        {'Player': 'Omarion Hampton', 'player_id': '00-0040666', 'team': 'LAC'},
        {'Player': 'Kimani Vidal', 'player_id': '00-0040123', 'team': 'LAC'},
    ])
    attached = rba.derive_rb_allocator_segment_fields(candidates, segments, context)
    hampton = _player_row(attached, 'Omarion Hampton')
    vidal = _player_row(attached, 'Kimani Vidal')
    assert bool(hampton['rb_segment_match_found'])
    assert np.isclose(float(hampton['rb_segment_pre_absence_snap_share']), 0.72)
    assert float(hampton['interrupted_incumbent_role_credit']) > 0.90
    assert float(hampton['shared_healthy_lead_score']) > 0.65
    assert float(hampton['replacement_only_era_downweight']) == 0.0
    assert float(vidal['shared_healthy_lead_score']) < -0.65
    assert float(vidal['replacement_only_era_downweight']) > 0.25


def test_role_segments_do_not_infer_absence_without_snap_coverage_or_for_two_game_rotation_gap():
    no_snap = pd.DataFrame([
        {'player_display_name': 'No Snap Back', 'player_id': 'none-1', 'position': 'RB',
         'game_team': 'SEA', 'season': 2025, 'week': week, 'weekly_snap_pct': 0,
         'has_snap_match': False}
        for week in range(1, 7)
    ])
    no_snap_calendar = pd.DataFrame({'game_team': ['SEA'] * 6, 'season': [2025] * 6,
                                     'week': list(range(1, 7))})
    no_snap_segments, _ = rba.analyze_rb_role_segments(no_snap, team_game_calendar=no_snap_calendar)
    no_snap_row = no_snap_segments.iloc[0]
    assert no_snap_row['rb_segment_status'] == 'no_snap_data'
    assert not bool(no_snap_row['interrupted_season'])

    # Four real games then two team games off is a normal short absence / role
    # variation, not the clear three-game internal gap required by this model.
    rotation = pd.DataFrame([
        {'player_display_name': 'Rotation Back', 'player_id': 'rotation-1', 'position': 'RB',
         'game_team': 'SEA', 'season': 2025, 'week': week, 'weekly_snap_pct': 35,
         'has_snap_match': True}
        for week in (1, 2, 3, 4, 7, 8)
    ])
    rotation_calendar = pd.DataFrame({'game_team': ['SEA'] * 8, 'season': [2025] * 8,
                                      'week': list(range(1, 9))})
    rotation_segments, _ = rba.analyze_rb_role_segments(rotation, team_game_calendar=rotation_calendar)
    rotation_row = rotation_segments.iloc[0]
    assert rotation_row['rb_segment_status'] == 'no_clear_internal_gap'
    assert not bool(rotation_row['interrupted_season'])


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f'  PASS  {name}')
        except Exception as exc:  # pragma: no cover - standalone test report
            failures.append((name, exc))
            print(f'  FAIL  {name}: {type(exc).__name__}: {exc}')
    print(f'\n{len(tests) - len(failures)}/{len(tests)} passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
