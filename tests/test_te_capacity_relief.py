"""v2_te_capacity_relief: a tight end takes only TE_CAPACITY_ELASTICITY of the pass-capacity factor's move."""
import pandas as pd

import data.pass_capacity_allocator as pca
import data.weekly_projections as wp


def _row(player, pos, targets, attempts=0.0):
    return {'Team': 'SF', 'Player': player, 'Pos': pos, 'targets': targets, 'receptions': targets * 0.7,
            'receiving_yards': targets * 8.0, 'receiving_tds': targets * 0.05, 'passing_attempts': attempts}


BOARD = pd.DataFrame([_row('QB1', 'QB', 0.0, attempts=24.0), _row('WR1', 'WR', 9.0), _row('WR2', 'WR', 7.0),
                      _row('TE1', 'TE', 6.0), _row('WR3', 'WR', 4.0)])


def test_flag_is_a_candidate():
    assert 'v2_te_capacity_relief' in wp.MODEL_FEATURES and 'v2_te_capacity_relief' not in wp.DEFAULT_FEATURES


def test_tight_end_takes_half_the_move_and_nothing_is_reassigned():
    off, _ = pca.apply_pass_capacity_conservation(BOARD, prior_history=None)
    on, ledger = pca.apply_pass_capacity_conservation(BOARD, prior_history=None, te_elasticity=0.5)
    get = lambda f, p, c='targets': float(f.loc[f['Player'] == p, c].iloc[0])
    factor = get(off, 'WR1') / 9.0
    assert factor < 1.0
    assert abs(get(on, 'TE1') - 6.0 * (1 + 0.5 * (factor - 1))) < 2e-3
    assert abs(get(on, 'TE1', 'receiving_yards') - 48.0 * (1 + 0.5 * (factor - 1))) < 2e-2
    for p in ('WR1', 'WR2', 'WR3'):
        assert abs(get(on, p) - get(off, p)) < 1e-9          # the WRs are not cut harder
    assert 'Tight ends x' in ' '.join(ledger['reason'].astype(str))


def test_none_is_the_old_fit():
    a, _ = pca.apply_pass_capacity_conservation(BOARD, prior_history=None)
    b, _ = pca.apply_pass_capacity_conservation(BOARD, prior_history=None, te_elasticity=None)
    pd.testing.assert_frame_equal(a, b)
