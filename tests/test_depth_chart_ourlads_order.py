"""fetch_intelligent_depth_chart's Ourlads preseason ordering override.

The tab hands the function an imported Ourlads chart (exact-name -> listed
slot) for an upcoming season with no real snap data. Charted players must
then lead the position row in listed order, regardless of last season's
snap average - and a PAST season (real weekly data) must ignore the map
entirely.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

from data.transforms import fetch_intelligent_depth_chart  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402
from ui.tabs.depth_charts import _ourlads_rank_map_for  # noqa: E402


def _roster_only_stats():
    """No 'week' column => had_real_weekly_data is False (upcoming season)."""
    names = ['Veteran WR', 'Rookie WR', 'Camp WR',
             'Backup QB', 'Starter QB', 'Lead RB', 'Change RB']
    pos = ['WR', 'WR', 'WR', 'QB', 'QB', 'RB', 'RB']
    # Snap average would rank Veteran WR first and Rookie WR last.
    snap = [78.0, 5.0, 30.0, 60.0, 20.0, 70.0, 40.0]
    return pd.DataFrame({
        'recent_team': ['BUF'] * len(names),
        'player_name': names,
        'exact_name': clean_name_exact(pd.Series(names)),
        'position': pos,
        'depth_chart_position': pos,
        'years_exp': [6, 0, 3, 4, 1, 5, 2],
        'fantasy_points': [12.0, 3.0, 6.0, 8.0, 4.0, 14.0, 7.0],
        'snap_pct_avg': snap,
        'weekly_snap_pct': snap,
        'snap_data_year': [2025] * len(names),
        'draft_number': [40, 20, 200, 120, 33, 25, 150],
    })


def _played_stats():
    df = _roster_only_stats()
    df['week'] = 3           # any real weekly granularity flips the guard
    return df


def _snapshot():
    rows = [
        ('BUF', 'WR', 'LWR', 1, 'Rookie WR'),
        ('BUF', 'WR', 'RWR', 1, 'Camp WR'),
        ('BUF', 'WR', 'SWR', 1, 'Veteran WR'),
        ('BUF', 'QB', 'QB', 1, 'Starter QB'),
        ('BUF', 'QB', 'QB', 2, 'Backup QB'),
        ('BUF', 'RB', 'RB', 1, 'Change RB'),
        ('BUF', 'RB', 'RB', 2, 'Lead RB'),
    ]
    return pd.DataFrame(rows, columns=['team', 'position', 'position_label',
                                       'source_slot', 'player']).assign(depth_rank=lambda d: d['source_slot'])


def _starter(dc_df, pos_label):
    row = dc_df[dc_df['Position'] == pos_label].iloc[0]
    return row['Starter']


def test_rank_map_rolls_alignment_labels_up_and_keeps_best_slot():
    m = _ourlads_rank_map_for(_snapshot(), 'BUF')
    assert m[clean_name_exact(pd.Series(['Starter QB'])).iloc[0]] == 1.0
    assert m[clean_name_exact(pd.Series(['Backup QB'])).iloc[0]] == 2.0
    # all three alignment WR1s roll up to slot 1
    assert m[clean_name_exact(pd.Series(['Rookie WR'])).iloc[0]] == 1.0
    assert m[clean_name_exact(pd.Series(['Camp WR'])).iloc[0]] == 1.0


def test_ourlads_order_overrides_snap_average_for_upcoming_season():
    fetch_intelligent_depth_chart.clear()
    m = _ourlads_rank_map_for(_snapshot(), 'BUF')
    dc = fetch_intelligent_depth_chart(
        'BUF', _roster_only_stats(), pd.DataFrame(), 2026,
        _ourlads_rank_map=m, ourlads_sig=str(sorted(m.items())))
    # QB: Ourlads says Starter QB is QB1 even though Backup QB has 3x the snaps.
    assert _starter(dc, 'QB') == 'Starter QB'
    # RB: Ourlads lists Change RB ahead of Lead RB.
    assert _starter(dc, 'RB') == 'Change RB'


def test_past_season_ignores_the_map():
    fetch_intelligent_depth_chart.clear()
    m = _ourlads_rank_map_for(_snapshot(), 'BUF')
    dc = fetch_intelligent_depth_chart(
        'BUF', _played_stats(), pd.DataFrame(), 2024,
        _ourlads_rank_map=m, ourlads_sig=str(sorted(m.items())))
    # Real weekly data present => snap average wins: Backup QB (60) > Starter QB (20).
    assert _starter(dc, 'QB') == 'Backup QB'


def test_no_map_is_a_no_op():
    fetch_intelligent_depth_chart.clear()
    dc = fetch_intelligent_depth_chart('BUF', _roster_only_stats(), pd.DataFrame(), 2026)
    assert _starter(dc, 'QB') == 'Backup QB'   # snap-driven, unchanged


def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith('test_') and callable(f)]
    bad = []
    for n, f in tests:
        try:
            f()
            print(f"  PASS  {n}")
        except Exception as exc:  # noqa: BLE001
            bad.append((n, exc))
            print(f"  FAIL  {n}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(bad)}/{len(tests)} passed")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
