"""_identity_by_name and drop_shared_name_defenders (2026-10-26): a name two players share maps to the offensive player's
id, not whichever row sorts last, and the defender's rows leave the history (a 2026 CLE rookie linebacker named Justin
Jefferson took the Minnesota receiver's identity, so his prior fell back to the WR position average, and his 0% week
joined the receiver's snap-share window)."""
import pandas as pd

from data.weekly_projections import (_identity_by_name, attach_player_identity, drop_shared_name_defenders,
                                     expected_snap_share)


def _hist(lb_last=True):
    wr = [('Justin Jefferson', '00-0036322', 'WR', w) for w in (1, 2, 3, 5)]
    lb = [('Justin Jefferson', '00-0041075', 'LB', 5)]
    other = [('Garrett Wilson', '00-0037740', 'WR', 5)]
    rows = wr + lb + other if lb_last else lb + wr + other
    return pd.DataFrame(rows, columns=['player_display_name', 'player_id', 'position', 'week'])


def test_shared_name_maps_to_the_offensive_player():
    for lb_last in (True, False):
        m = _identity_by_name(_hist(lb_last), 'player_display_name')
        assert set(m.values) == {'gsis_id:00-0036322', 'gsis_id:00-0037740'}


def test_totals_get_the_receiver_id():
    totals = pd.DataFrame({'player_display_name': ['Justin Jefferson', 'Garrett Wilson'], 'targets': [28.0, 9.0]})
    out = attach_player_identity(totals, _hist(), 'player_display_name')
    assert out['_identity_key'].tolist() == ['gsis_id:00-0036322', 'gsis_id:00-0037740']


def test_shared_name_defender_rows_are_dropped():
    h = _hist()
    h['weekly_snap_pct'] = [92.0, 100.0, 12.0, 89.0, 0.0, 96.0]
    out, names = drop_shared_name_defenders(h, 'player_display_name')
    assert names == ['Justin Jefferson'] and len(out) == 5 and not out['position'].eq('LB').any()
    # the receiver's own share no longer averages the linebacker's 0% week in
    assert expected_snap_share(out, 'player_display_name', 'team', as_of_week=8)['Justin Jefferson'] > 0.7
    assert expected_snap_share(h, 'player_display_name', 'team', as_of_week=8)['Justin Jefferson'] < 0.6


def test_two_way_player_and_unshared_names_are_kept():
    h = pd.DataFrame([('Travis Hunter', '00-0040000', 'WR', 1), ('Travis Hunter', '00-0040000', 'CB', 2),
                      ('Fred Warner', '00-0034000', 'LB', 1)],
                     columns=['player_display_name', 'player_id', 'position', 'week'])
    out, names = drop_shared_name_defenders(h, 'player_display_name')
    assert names == [] and len(out) == 3
    out, names = drop_shared_name_defenders(_hist().drop(columns=['player_id']), 'player_display_name')
    assert names == [] and len(out) == 6


def test_history_without_position_keeps_the_old_order():
    h = _hist().drop(columns=['position'])
    m = _identity_by_name(h, 'player_display_name')
    assert len(m) == 2
