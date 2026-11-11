"""data.prop_tiers (2026-11-11, audit B2): the v0 tiers, their forward record, and the board columns the props table reads."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import prop_tiers as pt  # noqa: E402


def test_tier_rules():
    cases = [
        # stat, model p, book q, tier
        ('receiving_yards', 0.42, 0.47, 'A'),       # under, the price leans under, gap 0.05
        ('receiving_yards', 0.56, 0.47, 'B'),       # over against an under lean
        ('receptions', 0.55, 0.50, 'B'),            # no lean (0.5) counts as against
        ('receiving_yards', 0.49, 0.47, 'Track'),   # gap 0.02 < 0.03
        ('rushing_yards', 0.40, 0.46, 'A'),         # a rushing-yards under with the lean
        ('rushing_yards', 0.60, 0.53, 'Track'),     # a rushing-yards over is never A or B
        ('targets', 0.40, 0.48, 'Track'),           # not a tier stat
        ('passing_yards', 0.40, 0.48, 'No bet'),    # a switched-off stat
        ('receptions', 0.20, 0.50, 'No bet'),       # gap 0.30 > 0.25
        ('receptions', np.nan, 0.50, ''),
    ]
    got = pt.assign_tiers([c[0] for c in cases], [c[1] for c in cases], [c[2] for c in cases])
    assert list(got) == [c[3] for c in cases]


def _rows(week, n=40, timing='at board'):
    over = np.array([1, 0, 0, 0] * (n // 4), dtype=float)            # 25% overs: the model's unders mostly win
    return pd.DataFrame({'timing': timing, 'week': week, 'stat': 'receptions', 'book': 'Pinnacle', 'sharpest': True,
                         '_key': [f'p{i}' for i in range(n)], 'line': 4.5, 'p_over': 0.40, 'p_book': 0.45, 'over': over,
                         'over_payout': 2.0, 'under_payout': 1.8})


def test_forward_record_reads_the_status_from_forward_weeks_only():
    rows = pd.concat([_rows(8), _rows(8, timing='at close'), _rows(10), _rows(10, timing='at close')], ignore_index=True)
    fwd = pt.forward_rows(rows)
    assert set(fwd['tier']) == {'A'} and len(fwd) == 80
    assert (fwd['side'] == 'under').all() and (fwd['price'] == 1.8).all()
    rec = pt.tier_record(fwd, 2026).set_index('tier')
    assert rec.at['A', 'bets'] == 80 and rec.at['A', 'hit'] == 0.75 and rec.at['A', 'status'] == 'tracking'
    assert rec.at['A', 'forward bets'] == 40
    assert abs(rec.at['A', 'roi_price'] - (0.75 * 0.8 - 0.25)) < 1e-12
    assert abs(rec.at['A', 'hit shown'] - (60 + 100) / (80 + 200)) < 1e-12
    live = pt.tier_record(fwd, 2026, forward_only=True).set_index('tier')
    assert live.at['A', 'bets'] == 40 and live.at['A', 'status'] == 'tracking'       # 40 < 300 forward bets
    assert live.at['A', 'clv toward'] == 0.0 and live.at['A', 'clv away'] == 0.0     # the close did not move


def test_a_tier_turns_bettable_only_on_300_forward_bets_clear_of_zero():
    rows = pd.concat([_rows(w, n=80) for w in range(10, 14)], ignore_index=True)    # 320 forward bets, 75% at 1.8
    rec = pt.tier_record(pt.forward_rows(rows), 2026, forward_only=True).set_index('tier')
    assert rec.at['A', 'bets'] == 320 and rec.at['A', 'roi 90% low'] > 0 and rec.at['A', 'status'] == 'bettable'
    assert pt.tier_record(pt.forward_rows(rows), 2026).set_index('tier').at['A', 'status'] == 'tracking'


def test_shrink_uses_earlier_weeks_only():
    rows = pd.concat([_rows(w, n=100, timing='at close') for w in (3, 4, 5)], ignore_index=True)
    assert pt.shrinks_before(rows, 3) == {'_all': pt.DEFAULT_SHRINK}
    s = pt.shrinks_before(rows, 6)                  # 300 rows: fit, and receptions has 200+
    assert set(s) == {'_all', 'receptions'} and s['receptions'] > 0


def test_board_tiers_and_the_props_table_column():
    from ui import games_view
    board = pd.DataFrame({'Player': ['B Wideout'], 'Team': ['DET'], 'Pos': ['WR'], 'Availability': [1.0],
                          'receiving_yards': [60.0], 'Median receiving_yards': [58.0], 'Line receiving_yards': [62.5],
                          'Model P Over receiving_yards': [0.42], 'Book P Over receiving_yards': [0.47]})
    board = pt.board_tiers(board)
    assert board.at[0, 'Tier receiving_yards'] == 'A'
    table = games_view.props_table(games_view.long_market(board), ['DET'])
    assert list(table['TIER']) == ['A']
    plain = games_view.props_table(games_view.long_market(board.drop(columns=['Tier receiving_yards'])), ['DET'])
    assert 'TIER' not in plain.columns                              # no tiers on the board: the table is as before
