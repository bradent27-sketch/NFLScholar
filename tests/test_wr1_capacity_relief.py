"""v2_wr1_capacity_relief: each room's lead WR takes only WR1_CAPACITY_ELASTICITY of the pass-capacity factor's move."""
import pandas as pd

import data.pass_capacity_allocator as pca
import data.weekly_projections as wp


def _row(team, player, pos, targets, attempts=0.0):
    return {'Team': team, 'Player': player, 'Pos': pos, 'targets': targets, 'receptions': targets * 0.7,
            'receiving_yards': targets * 8.0, 'receiving_tds': targets * 0.05, 'passing_attempts': attempts}


BOARD = pd.DataFrame([_row('SF', 'QB1', 'QB', 0.0, attempts=24.0), _row('SF', 'WR2', 'WR', 7.0),
                      _row('SF', 'WR1', 'WR', 9.0), _row('SF', 'TE1', 'TE', 10.0), _row('SF', 'WR3', 'WR', 4.0),
                      _row('KC', 'QBk', 'QB', 0.0, attempts=24.0), _row('KC', 'Bwr', 'WR', 8.0),
                      _row('KC', 'Awr', 'WR', 8.0), _row('KC', 'TEk', 'TE', 6.0)])
get = lambda f, p, c='targets': float(f.loc[f['Player'] == p, c].iloc[0])


def test_flag_ships():
    assert 'v2_wr1_capacity_relief' in wp.MODEL_FEATURES and 'v2_wr1_capacity_relief' in wp.DEFAULT_FEATURES


def test_lead_wr_takes_the_elastic_share_and_nothing_is_reassigned():
    off, _ = pca.apply_pass_capacity_conservation(BOARD, prior_history=None)
    on, ledger = pca.apply_pass_capacity_conservation(BOARD, prior_history=None, wr1_elasticity=0.6)
    factor = get(off, 'WR2') / 7.0
    assert factor < 1.0
    # the lead WR is the WR with the most targets, not the room's top receiver (TE1 has 10)
    assert abs(get(on, 'WR1') - 9.0 * (1 + 0.6 * (factor - 1))) < 2e-3
    assert abs(get(on, 'WR1', 'receiving_yards') - 72.0 * (1 + 0.6 * (factor - 1))) < 2e-2
    for p in ('WR2', 'WR3', 'TE1'):
        assert abs(get(on, p) - get(off, p)) < 1e-9          # nobody else is cut harder
    assert 'Lead WR WR1 x' in ' '.join(ledger['reason'].astype(str))


def test_one_lead_per_team_and_name_breaks_a_tie():
    off, _ = pca.apply_pass_capacity_conservation(BOARD, prior_history=None)
    on, _ = pca.apply_pass_capacity_conservation(BOARD, prior_history=None, wr1_elasticity=0.6)
    factor = get(off, 'Awr') / 8.0
    assert abs(factor - 1.0) > 0.01                          # KC's room is moved (up: under budget)
    assert abs(get(on, 'Awr') - 8.0 * (1 + 0.6 * (factor - 1))) < 2e-3   # Awr before Bwr by name
    assert abs(get(on, 'Bwr') - get(off, 'Bwr')) < 1e-9
    shuffled, _ = pca.apply_pass_capacity_conservation(BOARD.iloc[::-1].reset_index(drop=True), prior_history=None,
                                                       wr1_elasticity=0.6)
    for p in ('Awr', 'Bwr', 'WR1'):
        assert abs(get(on, p) - get(shuffled, p)) < 1e-9


def test_none_is_the_old_fit():
    a, _ = pca.apply_pass_capacity_conservation(BOARD, prior_history=None)
    b, _ = pca.apply_pass_capacity_conservation(BOARD, prior_history=None, wr1_elasticity=None)
    pd.testing.assert_frame_equal(a, b)
