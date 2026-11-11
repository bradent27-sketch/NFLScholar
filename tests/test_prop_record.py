"""data.prop_record (2026-11-11, audit B3): the prop record at the real price, by book and side, against blind baselines."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import prop_record as pr  # noqa: E402


def _snapshot(path, fetched_at, rows):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump({'fetched_at': fetched_at, 'season': 2026, 'week': 5, 'status': 'ok', 'rows': rows}, fh)


def _line(provider, line, over, under, p_over):
    return {'provider': provider, 'player': 'B Wideout', 'market': 'receiving_yards', 'period': 'game', 'scorable': True,
            'line': line, 'over_payout': over, 'under_payout': under, 'p_over': p_over}


def test_archive_keeps_both_prices_and_marks_the_sharpest_book(tmp_path):
    _snapshot(tmp_path / '2026_wk05_a.json', '2026-10-10T12:00:00Z',
              [_line('DraftKings', 70.5, 1.87, 1.95, 0.49), _line('Pinnacle', 69.5, 1.91, 1.95, 0.505)])
    _snapshot(tmp_path / '2026_wk05_b.json', '2026-10-11T12:00:00Z', [_line('Pinnacle', 71.5, 1.95, 1.91, 0.49)])
    arc = pr.load_archive(2026, archive=str(tmp_path))
    assert {'over_payout', 'under_payout'} <= set(arc.columns)
    when = {(5, 'bwideout'): pd.Timestamp('2026-10-10T18:00:00Z')}
    lines = pr.book_lines_as_of(arc, when).set_index('book')
    assert lines.loc['Pinnacle', 'sharpest'] and not lines.loc['DraftKings', 'sharpest']
    assert lines.loc['Pinnacle', 'line'] == 69.5 and lines.loc['DraftKings', 'over_payout'] == 1.87   # the later pull is not yet out
    assert pr.priced_lines(arc, when) == {(5, 'bwideout', 'receiving_yards'): (69.5, 0.505, 'Pinnacle')}


def _rows():
    # 40 lines: the book says 50/50 on each; the model leans under on the first 30 (wrong on 12) and over on the last 10 (right on 4)
    n = 40
    over = np.array([1] * 12 + [0] * 18 + [1] * 4 + [0] * 6, dtype=float)
    p_over = np.array([0.40] * 30 + [0.60] * 10)
    return pd.DataFrame({'timing': 'at board', 'stat': 'receiving_yards', 'week': 5, 'Pos': 'WR', 'book': 'Pinnacle',
                         'sharpest': True, 'p_over': p_over, 'p_book': 0.5, 'over': over,
                         'over_payout': 2.0, 'under_payout': 1.8, 'line': 60.5, '_key': [f'p{i}' for i in range(n)]})


def test_roi_is_at_the_price_of_the_side_taken():
    _summary, bets, _bands = pr.record_tables(_rows())
    b = bets[(bets['stat'] == 'ALL') & (bets['edge_min'] == 0.05)].iloc[0]
    # 18 unders won at 1.8 (+0.8 each), 12 lost; 4 overs won at 2.0 (+1.0 each), 6 lost
    assert b['bets'] == 40 and b['hit'] == 22 / 40
    assert abs(b['roi_price'] - (18 * 0.8 - 12 + 4 * 1.0 - 6) / 40) < 1e-12
    assert abs(b['roi_-110'] - (22 * 100 / 110 - 18) / 40) < 1e-12


def test_baselines_and_the_model_lift():
    base = pr.baseline_table(_rows()).set_index('stat').loc['receiving_yards']
    # 16 overs, 24 unders in all
    assert base['always under hit'] == 24 / 40 and abs(base['always under roi'] - (24 * 0.8 - 16) / 40) < 1e-12
    assert base['always over hit'] == 16 / 40 and abs(base['always over roi'] - (16 * 1.0 - 24) / 40) < 1e-12
    assert base['book favourite hit'] == 24 / 40          # p_book 0.5 is a tie: the favourite is the under
    assert abs(base['hit lift vs best blind'] - (22 / 40 - 24 / 40)) < 1e-12


def test_book_and_side_tables():
    rows = pd.concat([_rows(), _rows().assign(book='DraftKings', sharpest=False)], ignore_index=True)
    books = pr.book_table(rows).set_index('book')
    assert set(books.index) == {'Pinnacle', 'DraftKings'} and books.loc['DraftKings', 'bets'] == 40
    sides = pr.side_table(rows)
    s = sides[sides['edge_min'] == 0.0].set_index('side')
    assert s.loc['under', 'bets'] == 30 and s.loc['under', 'hit'] == 18 / 30      # the sharpest rows only
    assert 'over' not in s.index                                                   # 10 overs: under the 15-bet minimum
