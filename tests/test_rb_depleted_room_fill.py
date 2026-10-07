"""v2_rb_depleted_room_fill: an emptied RB room no longer loses the carries its survivors' growth cap cannot place.

The live case (PHI, week 5 of 2026, 2026-10-07): Saquon Barkley (14.9 carries/game) and Tank Bigsby (1.5) out, Will
Shipley (2.9) the only healthy back. The 2x growth cap let him take 2.9 more; ten carries vanished and no other back
was projected. With the flag the survivor takes 80% of what the cap could not place and the practice-squad backs the
build added split the rest as an expected value."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

import data.rb_role_allocator as rba  # noqa: E402
import data.weekly_projections as wp  # noqa: E402


def _rb(player, team, *, core, snap, carries, targets, ps=False, full_carries=None, full_targets=None,
        carry_share=0.0, target_share=0.0):
    return {'Player': player, 'Pos': 'RB', 'Team': team, 'functional_position': 'RB', '_rb_core': core,
            'Expected Snap Share': snap, 'rushing_attempts': carries, 'rushing_yards': carries * 4.5,
            'rushing_tds': carries * 0.03, 'targets': targets, 'receptions': targets * 0.8,
            'receiving_yards': targets * 7.0, 'receiving_tds': targets * 0.04,
            '_full_rushing_attempts': carries if full_carries is None else full_carries,
            '_full_targets': targets if full_targets is None else full_targets,
            '_rb_carry_allocation_share': carry_share, '_rb_target_allocation_share': target_share,
            '_ps_elevation_candidate': ps}


def _phi(with_practice_squad=True):
    rows = [
        _rb('Lead RB', 'PHI', core=True, snap=0.7, carries=0.0, targets=0.0, full_carries=14.9, full_targets=2.2),
        _rb('Backup RB', 'PHI', core=True, snap=0.1, carries=0.0, targets=0.0, full_carries=1.5, full_targets=0.3),
        _rb('Lone Survivor', 'PHI', core=True, snap=0.22, carries=2.9, targets=0.8, carry_share=1.0, target_share=1.0),
    ]
    if with_practice_squad:
        rows += [_rb('PS One', 'PHI', core=False, snap=0.05, carries=0.0, targets=0.0, ps=True),
                 _rb('PS Two', 'PHI', core=False, snap=0.05, carries=0.0, targets=0.0, ps=True)]
    return pd.DataFrame(rows)


_OUT = {'Lead RB': {'plays_probability': 0.0, 'status': 'out'}, 'Backup RB': {'plays_probability': 0.0, 'status': 'out'}}
_PROV = {n: {'year': 2026, 'source': 'current_week_report'} for n in _OUT}


def _run(frame, **kwargs):
    return rba.redistribute_rb_vacancy_with_allocator(frame, _OUT, as_of_year=2026, injury_provenance=_PROV, **kwargs)


def _row(frame, name):
    return frame.loc[frame['Player'].eq(name)].iloc[0]


def test_default_behaviour_is_unchanged_without_the_flag():
    out, ledger = _run(_phi())
    survivor = _row(out, 'Lone Survivor')
    # the 2x cap binds per vacated back (two backs out -> at most 2x twice), and nothing reaches the practice squad
    assert survivor['rushing_attempts'] <= 2.9 * rba.VACANCY_MAX_GROWTH ** 2 + 1e-6
    assert survivor['rushing_attempts'] < 8.0
    assert _row(out, 'PS One')['rushing_attempts'] == 0.0 and _row(out, 'PS Two')['rushing_attempts'] == 0.0
    assert 'Depleted room' not in ' '.join(ledger['reason'].astype(str))


def test_the_survivor_takes_what_the_growth_cap_could_not_place_and_practice_squad_backs_share_the_rest():
    out, ledger = _run(_phi(), depleted_room_fill=True)
    vacated_reusable = (14.9 + 1.5) * rba.VACANCY_SURVIVAL
    survivor, one, two = _row(out, 'Lone Survivor'), _row(out, 'PS One'), _row(out, 'PS Two')
    # nothing is dropped any more: survivor + practice squad account for every reusable carry
    placed = (survivor['rushing_attempts'] - 2.9) + one['rushing_attempts'] + two['rushing_attempts']
    assert abs(placed - vacated_reusable) < 0.05
    assert survivor['rushing_attempts'] > 2.9 * rba.VACANCY_MAX_GROWTH ** 2 + 1.0       # far past the old cap
    assert abs(one['rushing_attempts'] - two['rushing_attempts']) < 1e-9 and one['rushing_attempts'] > 0
    assert one['rushing_attempts'] < survivor['rushing_attempts']
    # dependent stats follow at sane per-carry / per-target rates (league rates for a practice-squad back)
    assert abs(one['rushing_yards'] / one['rushing_attempts'] - 4.2) < 0.2
    assert one['receptions'] <= one['targets'] + 1e-9
    # the ledger says so, and names the practice-squad backs as recipients
    ps_recipients = [r for entries in ledger['recipients'] for r in entries if r.get('practice_squad')]
    assert {r['player'] for r in ps_recipients} == {'PS One', 'PS Two'}
    assert 'Depleted room' in ' '.join(ledger['reason'].astype(str))


def test_without_practice_squad_rows_the_unplaced_remainder_stays_unallocated_not_invented():
    out, ledger = _run(_phi(with_practice_squad=False), depleted_room_fill=True)
    survivor = _row(out, 'Lone Survivor')
    leftover_share = 1.0 - rba.DEPLETED_ROOM_SURVIVOR_ABSORB
    assert survivor['rushing_attempts'] > 2.9 * rba.VACANCY_MAX_GROWTH ** 2 + 1.0
    carries = ledger.loc[ledger['volume'].eq('rushing_attempts')]
    assert (carries['unallocated'] > 0).any()                                             # the call-up share
    assert float(carries['unallocated'].sum()) < leftover_share * (14.9 + 1.5) * rba.VACANCY_SURVIVAL


def test_a_room_with_a_real_second_back_is_untouched_by_the_flag():
    """Nothing is left over when the survivors can absorb it under the growth cap, so the flag changes nothing."""
    rows = [
        _rb('Lead RB', 'KC', core=True, snap=0.6, carries=0.0, targets=0.0, full_carries=6.0, full_targets=1.0),
        _rb('Second RB', 'KC', core=True, snap=0.4, carries=8.0, targets=2.0, carry_share=1.0, target_share=1.0),
    ]
    base, _ = rba.redistribute_rb_vacancy_with_allocator(
        pd.DataFrame(rows), {'Lead RB': {'plays_probability': 0.0}}, as_of_year=2026,
        injury_provenance={'Lead RB': {'year': 2026, 'source': 'current_week_report'}})
    filled, _ = rba.redistribute_rb_vacancy_with_allocator(
        pd.DataFrame(rows), {'Lead RB': {'plays_probability': 0.0}}, as_of_year=2026,
        injury_provenance={'Lead RB': {'year': 2026, 'source': 'current_week_report'}}, depleted_room_fill=True)
    assert abs(_row(base, 'Second RB')['rushing_attempts'] - _row(filled, 'Second RB')['rushing_attempts']) < 1e-9


def test_with_every_back_out_the_practice_squad_takes_the_whole_pool():
    frame = _phi()
    frame.loc[frame['Player'].eq('Lone Survivor'), ['rushing_attempts', 'targets']] = 0.0
    both_out = dict(_OUT, **{'Lone Survivor': {'plays_probability': 0.0, 'status': 'out'}})
    frame.loc[frame['Player'].eq('Lone Survivor'), '_full_rushing_attempts'] = 2.9
    out, ledger = rba.redistribute_rb_vacancy_with_allocator(
        frame, both_out, as_of_year=2026,
        injury_provenance={n: {'year': 2026, 'source': 'current_week_report'} for n in both_out},
        depleted_room_fill=True)
    assert _row(out, 'PS One')['rushing_attempts'] > 0 and _row(out, 'PS Two')['rushing_attempts'] > 0


def test_depleted_rb_room_teams_needs_an_out_back_and_at_most_one_healthy_back():
    cur = pd.DataFrame({'Player': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                        'Team': ['PHI', 'PHI', 'PHI', 'KC', 'KC', 'KC', 'DAL']})
    profiles = {'A': {'plays_probability': 0.0}, 'B': {'plays_probability': 0.0},     # PHI: two out, one healthy
                'D': {'plays_probability': 0.0},                                      # KC: one out, two healthy
                'G': {'plays_probability': 1.0}}                                      # DAL: nobody out
    assert wp._depleted_rb_room_teams(cur, profiles, 'Player') == {'PHI'}
    assert wp._depleted_rb_room_teams(cur, {}, 'Player') == set()
    assert wp._depleted_rb_room_teams(pd.DataFrame(), profiles, 'Player') == set()


def test_practice_squad_backs_join_the_pool_only_for_a_depleted_team():
    def roster(status_by_team):
        rows = []
        for player, team, status in status_by_team:
            rows.append({'player_display_name': player, 'position': 'RB', 'team': team, 'status': status, 'week': 1})
        return pd.DataFrame(rows)

    stats_df = roster([('Ps Phi Rb', 'PHI', 'DEV'), ('Ps Kc Rb', 'KC', 'DEV'), ('Active Phi Rb', 'PHI', 'ACT'),
                       ('Ps Phi Nohist', 'PHI', 'DEV')])
    prior = pd.DataFrame({'player_display_name': ['Ps Phi Rb', 'Ps Kc Rb', 'Active Phi Rb'], 'position': ['RB'] * 3,
                          'week': [1, 1, 1]})
    cur = pd.DataFrame(columns=['player_display_name', 'Games', 'Team'])
    stats = ['rushing_attempts']
    plain = wp._add_missing_returning_player_rows(
        cur, stats_df, prior, 'player_display_name', 'player_display_name', 'team', stats, 'RB')
    assert set(plain['player_display_name']) == {'Active Phi Rb'}                         # the practice squad stays off
    gated = wp._add_missing_returning_player_rows(
        cur, stats_df, prior, 'player_display_name', 'player_display_name', 'team', stats, 'RB',
        practice_squad_teams={'PHI'})
    # only PHI's, only DEV - including a back with no prior-season NFL game (an anonymous call-up candidate)
    assert set(gated['player_display_name']) == {'Active Phi Rb', 'Ps Phi Rb', 'Ps Phi Nohist'}
    flagged = gated.set_index('player_display_name')['_ps_elevation_candidate'].fillna(False)
    assert bool(flagged['Ps Phi Rb']) and not bool(flagged['Active Phi Rb'])


def test_the_flag_is_a_named_model_feature():
    assert 'v2_rb_depleted_room_fill' in wp.MODEL_FEATURES


def test_the_practice_squad_read_is_point_in_time_and_never_touches_the_base_rows():
    rows = [
        {'player_display_name': 'Active Phi Rb', 'position': 'RB', 'team': 'PHI', 'status': 'ACT', 'week': 1},
        {'player_display_name': 'Always Dev', 'position': 'RB', 'team': 'PHI', 'status': 'DEV', 'week': 1},
        {'player_display_name': 'Always Dev', 'position': 'RB', 'team': 'PHI', 'status': 'DEV', 'week': 9},
        {'player_display_name': 'Signed Later', 'position': 'RB', 'team': 'PHI', 'status': 'DEV', 'week': 8},
        {'player_display_name': 'Elevated Later', 'position': 'RB', 'team': 'PHI', 'status': 'DEV', 'week': 1},
        {'player_display_name': 'Elevated Later', 'position': 'RB', 'team': 'PHI', 'status': 'ACT', 'week': 6},
    ]
    stats_df = pd.DataFrame(rows)
    names = ['Active Phi Rb', 'Always Dev', 'Signed Later', 'Elevated Later']
    prior = pd.DataFrame({'player_display_name': names, 'position': ['RB'] * 4, 'week': [1] * 4})
    cur = pd.DataFrame(columns=['player_display_name', 'Games', 'Team'])
    args = (cur, stats_df, prior, 'player_display_name', 'player_display_name', 'team', ['rushing_attempts'], 'RB')
    plain = wp._add_missing_returning_player_rows(*args)
    at_week_3 = wp._add_missing_returning_player_rows(*args, practice_squad_teams={'PHI'}, as_of_week=3)
    at_week_10 = wp._add_missing_returning_player_rows(*args, practice_squad_teams={'PHI'}, as_of_week=10)
    base_names = set(plain['player_display_name'])
    assert base_names == {'Active Phi Rb', 'Elevated Later'}            # the default read never takes a DEV-only back
    # only EXTRA rows are ever added: the flag-on frame starts with exactly the flag-off rows
    for flagged in (at_week_3, at_week_10):
        assert list(flagged['player_display_name'][:len(plain)]) == list(plain['player_display_name'])
    assert set(at_week_3['player_display_name']) - base_names == {'Always Dev'}                  # Signed Later: no row yet
    assert set(at_week_10['player_display_name']) - base_names == {'Always Dev', 'Signed Later'}  # Elevated Later is in base


def test_a_live_board_reads_the_weekless_roster_snapshot_and_a_backtest_does_not():
    """On a live board the practice-squad backs are roster rows with NO week (no stats row); in a backtest such a
    row has no known date, so it is dropped (the harness-tested behaviour)."""
    rows = [
        {'player_display_name': 'Active Phi Rb', 'position': 'RB', 'team': 'PHI', 'status': 'ACT', 'week': 1.0},
        {'player_display_name': 'Snapshot Dev', 'position': 'RB', 'team': 'PHI', 'status': 'DEV', 'week': np.nan},
        {'player_display_name': 'Stale Weekly', 'position': 'RB', 'team': 'PHI', 'status': 'DEV', 'week': 2.0},
        {'player_display_name': 'Stale Weekly', 'position': 'RB', 'team': 'PHI', 'status': 'ACT', 'week': np.nan},
    ]
    stats_df = pd.DataFrame(rows)
    cur = pd.DataFrame(columns=['player_display_name', 'Games', 'Team'])
    args = (cur, stats_df, 'player_display_name', 'team', ['rushing_attempts'], 'RB', {'PHI'}, 5)
    live = wp._add_practice_squad_candidates(*args, current_snapshot=True)
    backtest = wp._add_practice_squad_candidates(*args, current_snapshot=False)
    assert set(live['player_display_name']) == {'Snapshot Dev'}      # the snapshot overrides Stale Weekly's old DEV row
    assert set(backtest['player_display_name']) == {'Stale Weekly'}   # weekless rows are not dated: only week 2 counts
