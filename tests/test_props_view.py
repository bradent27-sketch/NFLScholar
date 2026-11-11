"""C1 (2026-11-11): the PROPS view's rows - the ledger's pricing, the side, best price, EV, sorting and filters."""
import numpy as np
import pandas as pd
import pytest

from data import prop_probability as pp
from ui import props_view as pv

HAS_ENGINE = np.isfinite(pp.prob_over('receiving_yards', np.array(['WR']), np.array([60.0]), np.array([54.5]))).all()
pytestmark = pytest.mark.skipif(not HAS_ENGINE, reason='no prop engine on this machine')


def _inputs():
    board = pd.DataFrame({'Player': ['Al Wide', 'Bo Back', 'Cy Arm'], 'Pos': ['WR', 'RB', 'QB'], 'Team': ['LAR', 'SEA', 'SEA'],
                          'Opponent': ['SEA', 'LAR', 'LAR'], 'receiving_yards': [75.0, 12.0, 0.0],
                          'rushing_yards': [0.0, 55.0, 12.0], 'passing_yards': [0.0, 0.0, 250.0]})
    lines = pd.DataFrame([
        ('Al Wide', 'receiving_yards', 'Pinnacle', 60.5, 1.91, 1.91, 0.5),
        ('Al Wide', 'receiving_yards', 'FanDuel', 59.5, 1.87, 1.95, 0.51),
        ('Al Wide', 'receiving_yards', 'Underdog', 61.5, 1.0, 1.0, np.nan),
        ('Bo Back', 'rushing_yards', 'DraftKings', 62.5, 1.87, 1.95, 0.49),
        ('Cy Arm', 'passing_yards', 'Pinnacle', 240.5, 1.91, 1.91, 0.5)],
        columns=['player', 'market', 'provider', 'line', 'over_payout', 'under_payout', 'p_over'])
    return board, lines


def test_rows_price_like_the_ledger():
    from data.prediction_ledger import build_ledger_frame
    board, lines = _inputs()
    rows = pv.props_rows(board, lines)
    assert set(rows['_stat']) == {'receiving_yards', 'rushing_yards', 'passing_yards'}
    mkt = pd.DataFrame({'Player': board['Player'], 'Market Pts': np.nan, 'Coverage': np.nan})
    mkt.attrs['book_lines'], mkt.attrs['consensus_stats'] = lines, []
    led = build_ledger_frame(2026, 10, board, mkt).set_index('Player')
    for _, r in rows.iterrows():
        assert r['LINE'] == led.at[r['PLAYER'], f"Priced Line {r['_stat']}"]
        assert r['TIER'] == led.at[r['PLAYER'], f"Tier {r['_stat']}"]
        p = led.at[r['PLAYER'], f"Model P Over {r['_stat']}"]
        assert r['SIDE'] == ('OVER' if p > led.at[r['PLAYER'], f"Book P Over {r['_stat']}"] else 'UNDER')
    assert rows.set_index('_stat').at['passing_yards', 'TIER'] == 'No bet'


def test_best_price_ev_and_order():
    board, lines = _inputs()
    rows = pv.props_rows(board, lines, shrinks={'_all': 0.5}, first_lines={('alwide', 'receiving_yards', 'Pinnacle'): 58.5})
    al = rows[rows['PLAYER'] == 'Al Wide'].iloc[0]
    assert al['SIDE'] == 'OVER' and al['BEST'] == 'FanDuel 59.5'            # the lowest line for an over; Underdog is no price
    assert al['MOVE'] == 2.0
    p_side = al['_side_p']
    assert 0.5 < p_side < 1 and al['EV %'] > p_side * 1.87 - 1               # one point lower than the priced line: worth more
    ranks = rows['_tier_rank'].tolist()
    assert ranks == sorted(ranks)
    shown = pv.filter_rows(rows, tiers=None, side='OVER')
    assert set(shown['SIDE']) == {'OVER'}
    assert pv.filter_rows(rows, group='Passing', tiers=None)['_stat'].eq('passing_yards').all()
    assert pv.filter_rows(rows)['TIER'].isin(pv.SHOWN_TIERS).all()


def test_no_lines_no_rows():
    board, _ = _inputs()
    assert pv.props_rows(board, pd.DataFrame()).empty
