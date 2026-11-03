"""v2_rb_capacity_relief: each back takes only RB_CAPACITY_ELASTICITY of the pass-capacity factor's move."""
import pandas as pd

import data.pass_capacity_allocator as pca
import data.weekly_projections as wp


def _row(team, player, pos, targets, attempts=0.0, rush=0.0):
    return {'Team': team, 'Player': player, 'Pos': pos, 'targets': targets, 'receptions': targets * 0.7,
            'receiving_yards': targets * 8.0, 'receiving_tds': targets * 0.05, 'passing_attempts': attempts,
            'rushing_attempts': rush}


BOARD = pd.DataFrame([_row('SF', 'QB1', 'QB', 0.0, 24.0), _row('SF', 'WR1', 'WR', 9.0), _row('SF', 'WR2', 'WR', 7.0),
                      _row('SF', 'TE1', 'TE', 8.0), _row('SF', 'RB1', 'RB', 7.0, rush=14), _row('SF', 'RB2', 'RB', 4.0, rush=8)])
get = lambda f, p, c='targets': float(f.loc[f['Player'] == p, c].iloc[0])


def test_flag_is_a_candidate():
    assert 'v2_rb_capacity_relief' in wp.MODEL_FEATURES and 'v2_rb_capacity_relief' not in wp.DEFAULT_FEATURES


def test_each_back_takes_the_elastic_share_and_nobody_else_moves():
    off, _ = pca.apply_pass_capacity_conservation(BOARD, prior_history=None)
    on, ledger = pca.apply_pass_capacity_conservation(BOARD, prior_history=None, rb_elasticity=0.6)
    factor = get(off, 'RB1') / 7.0
    assert factor < 1.0
    for back, base in (('RB1', 7.0), ('RB2', 4.0)):
        assert abs(get(on, back) - base * (1 + 0.6 * (factor - 1))) < 2e-3
        assert abs(get(on, back, 'receiving_yards') - base * 8.0 * (1 + 0.6 * (factor - 1))) < 2e-2
    for p in ('WR1', 'WR2', 'TE1', 'QB1'):
        assert abs(get(on, p) - get(off, p)) < 1e-9
    assert 'Backs x' in ' '.join(ledger['reason'].astype(str))


def test_none_is_the_old_fit():
    a, _ = pca.apply_pass_capacity_conservation(BOARD, prior_history=None)
    b, _ = pca.apply_pass_capacity_conservation(BOARD, prior_history=None, rb_elasticity=None)
    pd.testing.assert_frame_equal(a, b)
