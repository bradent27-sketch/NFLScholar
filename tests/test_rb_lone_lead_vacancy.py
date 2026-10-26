"""v2_rb_lone_lead_vacancy (candidate 2026-10-25): a lone lead back takes a quarter of an out backup's volume, not 80%.

The live case (ATL, week 7 of 2026): Brian Robinson (8.8 carries/game) out, Bijan Robinson (17.1) the only healthy back
with a role. The vacancy pass handed Bijan 80% of Robinson's carries (24.1) and the practice-squad backs on the board
got 0. In 2021-2025 an RB1 picked up ~24% of an absent RB2's carries and the rest of the room ~57%."""
import os
import sys

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


def _atl(with_practice_squad=True):
    rows = [
        _rb('Lead RB', 'ATL', core=True, snap=0.73, carries=17.1, targets=6.0, carry_share=1.0, target_share=1.0),
        _rb('Backup RB', 'ATL', core=True, snap=0.27, carries=0.0, targets=0.0, full_carries=8.8, full_targets=2.0),
    ]
    if with_practice_squad:
        rows += [_rb('PS One', 'ATL', core=False, snap=0.03, carries=0.0, targets=0.0, ps=True),
                 _rb('PS Two', 'ATL', core=False, snap=0.03, carries=0.0, targets=0.0, ps=True)]
    return pd.DataFrame(rows)


def _run(frame, out_players=('Backup RB',), **kwargs):
    injuries = {p: {'plays_probability': 0.0, 'status': 'out'} for p in out_players}
    provenance = {p: {'year': 2026, 'source': 'current_week_report'} for p in out_players}
    return rba.redistribute_rb_vacancy_with_allocator(frame, injuries, as_of_year=2026, injury_provenance=provenance,
                                                      depleted_room_fill=True, **kwargs)


def _row(frame, name):
    return frame.loc[frame['Player'].eq(name)].iloc[0]


def test_flag_is_registered_and_shipped():
    assert 'v2_rb_lone_lead_vacancy' in wp.MODEL_FEATURES
    assert 'v2_rb_lone_lead_vacancy' in wp.DEFAULT_FEATURES     # shipped 2026-10-25 (user: "all three on")


def test_without_the_flag_the_lead_back_takes_the_reusable_volume_and_the_practice_squad_nothing():
    out, _ = _run(_atl())
    assert abs(_row(out, 'Lead RB')['rushing_attempts'] - (17.1 + 8.8 * rba.VACANCY_SURVIVAL)) < 0.02
    assert _row(out, 'PS One')['rushing_attempts'] == 0.0


def test_the_lone_lead_back_takes_his_share_and_the_practice_squad_splits_the_rest():
    share = rba.LONE_LEAD_VACANCY_SHARE
    out, ledger = _run(_atl(), lone_lead_share=share)
    lead, one, two = _row(out, 'Lead RB'), _row(out, 'PS One'), _row(out, 'PS Two')
    assert abs(lead['rushing_attempts'] - (17.1 + 8.8 * share)) < 0.02
    assert abs(lead['targets'] - (6.0 + 2.0 * share)) < 0.02
    # the rest of the reusable volume is not dropped: the practice-squad backs split it equally
    rest = 8.8 * (rba.VACANCY_SURVIVAL - share)
    assert abs(one['rushing_attempts'] - rest / 2) < 0.02 and abs(two['rushing_attempts'] - rest / 2) < 0.02
    assert abs(one['rushing_yards'] / one['rushing_attempts'] - 4.2) < 0.05
    # dependent stats of the lead back scale with his volume
    assert abs(lead['rushing_yards'] / lead['rushing_attempts'] - 4.5) < 0.01
    carries = ledger[ledger['volume'].eq('rushing_attempts')].iloc[0]
    assert 'Lone lead back' in carries['reason']
    assert carries['unallocated'] < 1e-6


def test_without_practice_squad_rows_the_rest_stays_unallocated():
    share = rba.LONE_LEAD_VACANCY_SHARE
    out, ledger = _run(_atl(with_practice_squad=False), lone_lead_share=share)
    assert abs(_row(out, 'Lead RB')['rushing_attempts'] - (17.1 + 8.8 * share)) < 0.02
    carries = ledger[ledger['volume'].eq('rushing_attempts')].iloc[0]
    assert abs(carries['unallocated'] - 8.8 * (rba.VACANCY_SURVIVAL - share)) < 0.02


def test_a_lone_backup_whose_lead_is_out_is_not_capped():
    # Lead RB out, the backup survives: next man up, the normal split (and the depleted-room fill) applies
    frame = pd.DataFrame([
        _rb('Lead RB', 'ATL', core=True, snap=0.73, carries=0.0, targets=0.0, full_carries=17.1, full_targets=6.0),
        _rb('Backup RB', 'ATL', core=True, snap=0.27, carries=8.8, targets=2.0, carry_share=1.0, target_share=1.0),
        _rb('PS One', 'ATL', core=False, snap=0.03, carries=0.0, targets=0.0, ps=True),
    ])
    with_flag, _ = _run(frame, out_players=('Lead RB',), lone_lead_share=rba.LONE_LEAD_VACANCY_SHARE)
    without, _ = _run(frame, out_players=('Lead RB',))
    pd.testing.assert_frame_equal(with_flag, without)
    assert _row(with_flag, 'Backup RB')['rushing_attempts'] > 8.8 + 8.0


def test_the_lead_is_read_before_redistribution_so_a_backup_grown_by_the_lead_vacancy_stays_uncapped():
    # PHI week 5 shape: lead and a deep backup out, a small back the only survivor. The lead's carries grow him past
    # the deep backup's, but he was never the lead: neither out back's volume is capped.
    frame = pd.DataFrame([
        _rb('Lead RB', 'PHI', core=True, snap=0.7, carries=0.0, targets=0.0, full_carries=14.9, full_targets=2.2),
        _rb('Deep RB', 'PHI', core=True, snap=0.1, carries=0.0, targets=0.0, full_carries=1.5, full_targets=0.3),
        _rb('Lone Survivor', 'PHI', core=True, snap=0.22, carries=2.9, targets=0.8, carry_share=1.0, target_share=1.0),
        _rb('PS One', 'PHI', core=False, snap=0.05, carries=0.0, targets=0.0, ps=True),
    ])
    out_players = ('Lead RB', 'Deep RB')
    with_flag, _ = _run(frame, out_players=out_players, lone_lead_share=rba.LONE_LEAD_VACANCY_SHARE)
    without, _ = _run(frame, out_players=out_players)
    pd.testing.assert_frame_equal(with_flag, without)


def test_a_room_with_a_second_healthy_back_is_untouched():
    frame = _atl()
    frame = pd.concat([frame, pd.DataFrame([_rb('Third RB', 'ATL', core=True, snap=0.15, carries=4.0, targets=1.0,
                                                carry_share=0.2, target_share=0.2)])], ignore_index=True)
    frame.loc[frame['Player'].eq('Lead RB'), ['_rb_carry_allocation_share', '_rb_target_allocation_share']] = 0.8
    with_flag, _ = _run(frame, lone_lead_share=rba.LONE_LEAD_VACANCY_SHARE)
    without, _ = _run(frame)
    pd.testing.assert_frame_equal(with_flag, without)


# v2_ps_line_routing (candidate 2026-10-25; user: a practice-squad back with a market line should get the volume, an
# even split only when none has one). ATL week 7: Khalil Herbert has lines at four books, Jam Miller none.

def test_line_routing_flag_is_registered_and_shipped():
    assert 'v2_ps_line_routing' in wp.MODEL_FEATURES
    assert 'v2_ps_line_routing' in wp.DEFAULT_FEATURES     # shipped 2026-10-25


def test_the_practice_squad_back_with_a_line_takes_the_whole_practice_squad_share():
    share = rba.LONE_LEAD_VACANCY_SHARE
    out, ledger = _run(_atl(), lone_lead_share=share, rb_line_holders=frozenset({'PS One'}))
    rest = 8.8 * (rba.VACANCY_SURVIVAL - share)
    assert abs(_row(out, 'PS One')['rushing_attempts'] - rest) < 0.02
    assert _row(out, 'PS Two')['rushing_attempts'] == 0.0
    assert abs(_row(out, 'Lead RB')['rushing_attempts'] - (17.1 + 8.8 * share)) < 0.02   # the lead back is unchanged
    carries = ledger[ledger['volume'].eq('rushing_attempts')].iloc[0]
    assert 'posted line' in carries['reason'] and 'PS One' in carries['reason']
    assert [r['player'] for r in carries['recipients'] if r.get('practice_squad')] == ['PS One']


def test_with_no_line_or_a_line_for_nobody_on_the_team_the_even_split_stands():
    share = rba.LONE_LEAD_VACANCY_SHARE
    even, _ = _run(_atl(), lone_lead_share=share)
    for holders in (None, frozenset(), frozenset({'Someone Else'})):
        routed, _ = _run(_atl(), lone_lead_share=share, rb_line_holders=holders)
        pd.testing.assert_frame_equal(routed, even)


def test_line_routing_also_applies_to_the_depleted_room_fill():
    # PHI week 5 shape (lead out, lone small survivor): what the growth cap could not place goes to the lined back
    frame = pd.DataFrame([
        _rb('Lead RB', 'PHI', core=True, snap=0.7, carries=0.0, targets=0.0, full_carries=14.9, full_targets=2.2),
        _rb('Lone Survivor', 'PHI', core=True, snap=0.22, carries=2.9, targets=0.8, carry_share=1.0, target_share=1.0),
        _rb('PS One', 'PHI', core=False, snap=0.05, carries=0.0, targets=0.0, ps=True),
        _rb('PS Two', 'PHI', core=False, snap=0.05, carries=0.0, targets=0.0, ps=True),
    ])
    even, _ = _run(frame, out_players=('Lead RB',))
    routed, _ = _run(frame, out_players=('Lead RB',), rb_line_holders=frozenset({'PS Two'}))
    assert _row(routed, 'PS One')['rushing_attempts'] == 0.0
    assert abs(_row(routed, 'PS Two')['rushing_attempts']
               - (_row(even, 'PS One')['rushing_attempts'] + _row(even, 'PS Two')['rushing_attempts'])) < 0.02
    assert _row(routed, 'Lone Survivor')['rushing_attempts'] == _row(even, 'Lone Survivor')['rushing_attempts']


def test_players_with_lines_reads_every_pull_of_the_week(tmp_path):
    import json
    from data.odds_weekly import players_with_lines
    row = {'provider': 'DraftKings', 'team': 'ATL', 'position': 'RB', 'market': 'rushing_yards', 'scorable': True,
           'line': 7.5, 'period': 'game'}
    (tmp_path / '2026_wk07_20261024T000000Z.json').write_text(json.dumps({'rows': [{**row, 'player': 'Khalil Herbert'}]}))
    (tmp_path / '2026_wk07_20261025T000000Z.json').write_text(json.dumps({'rows': [{**row, 'player': 'Bijan Robinson'}]}))
    (tmp_path / '2026_wk06_20261017T000000Z.json').write_text(json.dumps({'rows': [{**row, 'player': 'Jam Miller'}]}))
    lined = players_with_lines(2026, 7, archive_dir=str(tmp_path))
    assert lined == frozenset({'khalilherbert', 'bijanrobinson'})     # a later partial pull does not erase Herbert
    assert players_with_lines(2025, 7, archive_dir=str(tmp_path)) == frozenset()


def test_rb_backs_with_lines_names_every_back_with_a_line(monkeypatch):
    import data.odds_weekly as ow
    monkeypatch.setattr(ow, 'players_with_lines', lambda year, week: frozenset({'khalilherbert', 'bijanrobinson', 'drakelondon'}))
    board = pd.DataFrame({'Player': ['Bijan Robinson', 'Khalil Herbert', 'Jam Miller', 'Drake London'],
                          'Pos': ['RB', 'RB', 'RB', 'WR']})
    assert wp._rb_backs_with_lines(board, 2026, 7) == frozenset({'Bijan Robinson', 'Khalil Herbert'})
    monkeypatch.setattr(ow, 'players_with_lines', lambda year, week: frozenset())
    assert wp._rb_backs_with_lines(board, 2026, 7) is None


def _ten(depth_snaps=(0.12, 0.04)):
    # TEN week 7 shape: the backup out, the lead back the only one with an allocator share, two active depth backs on the
    # board with small snap shares and no allocation of their own (not core) - no practice-squad rows (room not depleted)
    return pd.DataFrame([
        _rb('Lead RB', 'TEN', core=True, snap=0.70, carries=19.0, targets=3.0, carry_share=1.0, target_share=1.0),
        _rb('Backup RB', 'TEN', core=True, snap=0.25, carries=0.0, targets=0.0, full_carries=3.7, full_targets=1.0),
        _rb('Depth One', 'TEN', core=False, snap=depth_snaps[0], carries=0.5, targets=0.2),
        _rb('Depth Two', 'TEN', core=False, snap=depth_snaps[1], carries=0.1, targets=0.0),
    ])


def test_the_rest_goes_to_the_active_depth_backs_by_snap_share():
    share = rba.LONE_LEAD_VACANCY_SHARE
    out, ledger = _run(_ten(), lone_lead_share=share)
    rest = 3.7 * (rba.VACANCY_SURVIVAL - share)
    one, two = _row(out, 'Depth One'), _row(out, 'Depth Two')
    assert abs(_row(out, 'Lead RB')['rushing_attempts'] - (19.0 + 3.7 * share)) < 0.02
    assert abs((one['rushing_attempts'] - 0.5) - rest * 0.12 / 0.16) < 0.02
    assert abs((two['rushing_attempts'] - 0.1) - rest * 0.04 / 0.16) < 0.02
    carries = ledger[ledger['volume'].eq('rushing_attempts')].iloc[0]
    assert carries['unallocated'] < 1e-6 and 'Depth One' in carries['reason']


def test_a_depth_back_with_a_line_takes_the_whole_rest():
    share = rba.LONE_LEAD_VACANCY_SHARE
    out, _ = _run(_ten(), lone_lead_share=share, rb_line_holders=frozenset({'Depth Two', 'Lead RB'}))
    assert abs((_row(out, 'Depth Two')['rushing_attempts'] - 0.1) - 3.7 * (rba.VACANCY_SURVIVAL - share)) < 0.02
    assert _row(out, 'Depth One')['rushing_attempts'] == 0.5


def test_rb_lines_by_team_and_its_fingerprint(tmp_path):
    import json
    from data.odds_weekly import rb_lines_by_team, rb_lines_fingerprint
    row = {'provider': 'DraftKings', 'market': 'rushing_yards', 'scorable': True, 'line': 7.5, 'period': 'game'}
    rows = [{**row, 'player': 'Khalil Herbert', 'team': 'ATL', 'position': 'RB'},
            {**row, 'player': 'Drake London', 'team': 'ATL', 'position': 'WR'},       # not a back
            {**row, 'player': 'No Team RB', 'team': '', 'position': 'RB'}]            # no team: skipped
    (tmp_path / '2026_wk07_20261024T000000Z.json').write_text(json.dumps({'rows': rows}))
    assert rb_lines_by_team(2026, 7, archive_dir=str(tmp_path)) == {'ATL': frozenset({'Khalil Herbert'})}
    before = rb_lines_fingerprint(2026, 7, archive_dir=str(tmp_path))
    (tmp_path / '2026_wk07_20261025T000000Z.json').write_text(json.dumps({'rows': [
        {**row, 'player': 'Jam Miller', 'team': 'ATL', 'position': 'RB'}]}))
    assert rb_lines_fingerprint(2026, 7, archive_dir=str(tmp_path)) != before   # a new line changes the build's cache key
