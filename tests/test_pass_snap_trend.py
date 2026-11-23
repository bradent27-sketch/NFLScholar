"""v2_recal_pass_snap_trend (shipped 2026-11-23, accuracy ideas 1-2): the recal's WR / TE / RB receiving fits add a weight x
r x season to date, r = his latest game's pass-snap share over his earlier games' - 1 (data/pass_snaps.py)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.weekly_projections as wp  # noqa: E402
from data.pass_snaps import TREND_CLIP, pass_snap_trend, trend_ratios  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402

SEVEN = (0.5, 0.4, 0.1, 0.1, 0.1, 1.0, 0.2)


def _rows(spec):
    """spec: (week, team, key, snaps, team_db)"""
    return pd.DataFrame(spec, columns=['week', 'Team', '_key', 'pass_snaps', 'team_db'])


def test_trend_is_latest_game_over_earlier_games():
    t = pass_snap_trend(_rows([(1, 'AAA', 'a', 10, 40), (2, 'AAA', 'a', 10, 40), (3, 'AAA', 'a', 30, 40),
                               (1, 'AAA', 'b', 20, 40), (2, 'AAA', 'b', 20, 40)]))
    a = t.set_index('_key').loc['a']
    assert a['last_week'] == 3 and a['games_before'] == 2
    assert np.isclose(a['ps_last'], 0.75) and np.isclose(a['ps_before'], 0.25)
    assert 'b' not in set(t['_key'])          # one earlier game only: left out


def test_ratios_are_clipped_and_need_the_teams_latest_game():
    t = pass_snap_trend(_rows([(1, 'AAA', 'a', 5, 40), (2, 'AAA', 'a', 5, 40), (3, 'AAA', 'a', 40, 40),
                               (1, 'BBB', 'c', 20, 40), (2, 'BBB', 'c', 20, 40), (3, 'BBB', 'c', 10, 40)]))
    r = trend_ratios(t, {'AAA': 3, 'BBB': 4})
    assert r[('AAA', 'a')][0] == TREND_CLIP[1]         # 1.0 / 0.125 - 1 = 7, clipped
    assert ('BBB', 'c') not in r                         # his latest game is not his team's latest (week 4 missing)


def test_table_appends_two_weights_to_the_receiving_entries():
    form, table = wp.stat_recal_table(frozenset({'v2_stat_recal_own_history', 'v2_recal_pass_snap_trend'}), 9)
    assert form == 'linear'
    for pos, stats in wp.STAT_RECAL_PASS_SNAP_TREND.items():
        for stat, weights in stats.items():
            assert table[pos][stat] == tuple(wp.STAT_RECAL_OWN_HISTORY[pos][stat]) + tuple(weights)
    assert table['RB']['rushing_attempts'] == wp.STAT_RECAL_OWN_HISTORY['RB']['rushing_attempts']
    _, plain = wp.stat_recal_table(frozenset({'v2_stat_recal_own_history'}), 9)
    assert len(plain['WR']['targets']) == 7
    assert 'v2_recal_pass_snap_trend' in wp.MODEL_FEATURES and 'v2_recal_pass_snap_trend' in wp.POST_CALIBRATION_FEATURES
    assert 'v2_recal_pass_snap_trend' in wp.DEFAULT_FEATURES          # shipped 2026-11-23
    assert not ({'v2_te_route_rise', 'v2_wr_route_rise'} & wp.DEFAULT_FEATURES)   # it replaced them


def _apply(week, trend):
    board = pd.DataFrame([['Rising One', 'WR', 'AAA', 10.0, 1.0, 9.0], ['No Trend', 'WR', 'AAA', 10.0, 1.0, 9.0]],
                         columns=['Player', 'Pos', 'Team', 'targets', 'Availability', 'Model Proj Pts'])
    keys = clean_name_exact(board['Player'])
    hist = pd.DataFrame([{'h_games': 4, 'h_ps_games': 17, 'h_s2d_targets': 6.0, 'h_l3_targets': 7.0, 'h_ps_targets': 9.0}] * 2,
                        index=pd.MultiIndex.from_arrays([keys, board['Pos']], names=['_key', 'pos']))
    table = {'WR': {'targets': SEVEN + (0.2, 0.1)}}
    t = {('AAA', keys.iloc[0]): (0.5, 0.0)} if trend else None
    out, info = wp.apply_stat_recal_own_history(board, hist, 'linear', table, trend=t, week=week)
    return out.set_index('Player')['targets'], info


def test_trend_weight_by_week_and_missing_rows_read_zero():
    blend = (4 * 6.0 + 6 * 9.0) / 10
    seven = 0.5 + 0.4 * 10 + 0.1 * 6 + 0.1 * 7 + 0.1 * 9 + 0.2 * blend
    late, info = _apply(9, True)
    early, _ = _apply(5, True)
    none, _ = _apply(9, False)
    assert np.isclose(late['Rising One'], round(seven + 0.2 * 0.5 * 6.0, 3))
    assert np.isclose(early['Rising One'], round(seven + 0.1 * 0.5 * 6.0, 3))
    assert np.isclose(late['No Trend'], round(seven, 3)) and np.isclose(none['Rising One'], round(seven, 3))
    assert info[('Rising One', 'WR', 'AAA')]['pass_snap_trend'] == 0.5
    assert 'pass_snap_trend' not in info[('No Trend', 'WR', 'AAA')]
