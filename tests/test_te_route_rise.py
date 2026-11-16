"""v2_te_route_rise / v2_wr_route_rise (shipped 2026-11-16): a TE whose latest game's route participation jumped gets his receiving line scaled."""
import pandas as pd

import data.weekly_projections as wp
from data.route_participation import latest_game_moves, route_rise_factors


def _routes(weeks, routes, team='KC', key='traviskelce', db=40.0):
    return pd.DataFrame({'week': weeks, 'Team': team, '_key': key, 'position': 'TE', 'routes': routes,
                         'pass_plays': db, 'team_db': db})


def _board(pos='TE'):
    return pd.DataFrame({'Player': ['Travis Kelce'], 'Pos': [pos], 'Team': ['KC'], 'targets': [5.0], 'receptions': [4.0],
                         'receiving_yards': [40.0], 'receiving_tds': [0.3], 'rushing_yards': [0.0]})


def _factors(board, moves, last_week):
    return route_rise_factors(board, moves, {'KC': last_week}, {'TE': (wp.TE_ROUTE_RISE_THRESHOLD, wp.TE_ROUTE_RISE_EXPONENT)},
                              wp.TE_ROUTE_RISE_MAX_FACTOR)


def test_flags_are_shipped_after_the_calibration():
    for flag in ('v2_te_route_rise', 'v2_wr_route_rise'):
        assert flag in wp.MODEL_FEATURES and flag in wp.DEFAULT_FEATURES
        assert flag in wp.POST_CALIBRATION_FEATURES and flag not in wp.CALIBRATION_INPUT_FEATURES
    assert (wp.TE_ROUTE_RISE_EXPONENT, wp.WR_ROUTE_RISE_EXPONENT) == (0.20, 0.15)


def test_moves_pool_the_earlier_games():
    m = latest_game_moves(_routes([1, 2, 3], [16.0, 24.0, 36.0])).iloc[0]
    assert m.last_week == 3 and m.games_before == 2
    assert abs(m.rp_before - 0.5) < 1e-9 and abs(m.rp_last - 0.9) < 1e-9
    assert latest_game_moves(_routes([1, 2], [16.0, 36.0])).empty      # one earlier game is not enough


def test_a_rise_scales_the_receiving_line_only():
    moves = latest_game_moves(_routes([1, 2, 3], [16.0, 24.0, 36.0]))
    factors = _factors(_board(), moves, 3)
    spec = factors[('Travis Kelce', 'TE', 'KC')]
    assert abs(spec['factor'] - (0.9 / 0.5) ** wp.TE_ROUTE_RISE_EXPONENT) < 1e-9
    out, info = wp.apply_te_route_rise(_board(), factors)
    assert abs(out['receiving_yards'].iloc[0] - 40.0 * spec['factor']) < 1e-9
    assert out['rushing_yards'].iloc[0] == 0.0 and set(info[('Travis Kelce', 'TE', 'KC')]['stats']) == set(wp.TE_ROUTE_RISE_STATS)


def test_no_rise_a_drop_a_stale_export_or_a_wr_is_left_alone():
    steady = latest_game_moves(_routes([1, 2, 3], [20.0, 20.0, 24.0]))           # +0.10, under the threshold
    drop = latest_game_moves(_routes([1, 2, 3], [36.0, 36.0, 16.0]))
    rise = latest_game_moves(_routes([1, 2, 3], [16.0, 24.0, 36.0]))
    assert not _factors(_board(), steady, 3)
    assert not _factors(_board(), drop, 3)
    assert not _factors(_board(), rise, 4)                                         # team played week 4, export missing
    assert not _factors(_board('WR'), rise, 3)


def test_the_factor_is_capped():
    moves = latest_game_moves(_routes([1, 2, 3], [2.0, 2.0, 40.0]))
    assert _factors(_board(), moves, 3)[('Travis Kelce', 'TE', 'KC')]['factor'] == wp.TE_ROUTE_RISE_MAX_FACTOR


def test_wr_arm_uses_its_own_threshold():
    moves = latest_game_moves(_routes([1, 2, 3], [16.0, 24.0, 32.0]))             # +0.30 over 0.50
    wr = {'WR': (wp.WR_ROUTE_RISE_THRESHOLD, wp.WR_ROUTE_RISE_EXPONENT)}
    spec = route_rise_factors(_board('WR'), moves, {'KC': 3}, wr, wp.TE_ROUTE_RISE_MAX_FACTOR)[('Travis Kelce', 'WR', 'KC')]
    assert abs(spec['factor'] - 1.6 ** wp.WR_ROUTE_RISE_EXPONENT) < 1e-9
    small = latest_game_moves(_routes([1, 2, 3], [16.0, 24.0, 27.0]))             # +0.175: a TE rise, not a WR one
    assert not route_rise_factors(_board('WR'), small, {'KC': 3}, wr, wp.TE_ROUTE_RISE_MAX_FACTOR)
