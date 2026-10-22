"""2026-10-21 error-first model audit: the oracle split, the as-of join and the information slope."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.audit_error_budget as eb  # noqa: E402
from scripts.audit_features import asof  # noqa: E402
from scripts.audit_market_gap import slope  # noqa: E402


def _two_receivers():
    d = pd.DataFrame({'year': 2025, 'week': 5, 'team': 'KC', 'pos': 'WR', 'played': True, 'player': ['A', 'B']})
    for s in eb.WEIGHTS:
        d['p_' + s] = 0.0
        d['a_' + s] = 0.0
    for pre in ('p_', 'a_'):
        for s in ('passing_attempts', 'passing_completions', 'rushing_attempts'):
            d[pre + s] = 0.0
    d['p_targets'], d['a_targets'] = [8.0, 4.0], [6.0, 9.0]
    d['p_receptions'], d['a_receptions'] = [5.6, 2.4], [3.0, 6.0]
    d['p_receiving_yards'], d['a_receiving_yards'] = [80.0, 32.0], [40.0, 120.0]
    d['p_receiving_tds'], d['a_receiving_tds'] = [0.4, 0.2], [0.0, 2.0]
    return d


def test_oracle_without_pieces_is_the_projection_and_with_all_is_the_actual():
    d = _two_receivers()
    none = eb.oracle_line(d, ())
    full = eb.oracle_line(d, ('team', 'share', 'eff', 'tds'))
    for s in ('receptions', 'receiving_yards', 'receiving_tds'):
        assert np.allclose(none[s], d['p_' + s])
        assert np.allclose(full[s], d['a_' + s])


def test_team_oracle_keeps_the_projected_split_at_the_actual_team_total():
    d = _two_receivers()
    team = eb.oracle_line(d, ('team',))
    # targets 8/4 scaled to the actual team total 15 -> 10/5; catch rate and yards per target stay projected
    assert np.allclose(team['receptions'], [10 * 0.7, 5 * 0.6])
    assert np.allclose(team['receiving_yards'], [100.0, 40.0])


def test_asof_reads_only_earlier_weeks():
    board = pd.DataFrame({'_rid': [0, 1], 'year': [2025, 2025], 'week': [5, 6], 'key': ['a|WR', 'a|WR']})
    logs = pd.DataFrame({'year': [2025, 2025], 'week': [4, 5], 'key': ['a|WR', 'a|WR'], 'v': [1.0, 2.0]})
    out = asof(board, logs, ['v'])
    assert out['v'].tolist() == [1.0, 2.0]          # week 5 reads week 4, never its own week
    assert out['_lw'].tolist() == [4, 5]


def test_information_slope():
    rng = np.random.default_rng(0)
    n = 400
    truth = rng.normal(10, 3, n)
    model = truth + rng.normal(0, 2, n)
    weeks = np.repeat(np.arange(8), n // 8)
    right = pd.DataFrame({'week': weeks, 'model_pts': model, 'bench': truth, 'actual_pts': truth})
    noise = right.assign(bench=model + rng.normal(0, 2, n))
    assert abs(slope(right, 'bench')[0] - 1.0) < 0.05
    assert abs(slope(noise, 'bench')[0]) < 0.2
