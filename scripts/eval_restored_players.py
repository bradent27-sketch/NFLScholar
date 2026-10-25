"""
The players a candidate puts back on the board (2026-10-25, v2_keep_fully_excluded_history; user: Ja'Kobi Lane missing from the
week 7 board). The harness scores the players BOTH arms projected, so a player only the variant has is never in it; this scores them
from the run's --variant-boards file: every (year, week, player) on the variant's board and not on the base board.

    played   games he played (a box-score row, or a silent game = 0 points) - his projection against the result, and against 0
             (what leaving him off the board amounts to)
    every    every week the variant expected him to play (Availability > 0.5; no game = 0)

    python scripts/eval_restored_players.py --boards .sweeps/harness_boards/keep_excluded_2022-2025_wk3-17.parquet --variant keep_excluded
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.silent_games import played_players  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402

EXPECTED = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boards', required=True)
    ap.add_argument('--variant', required=True, help="the variant's label (with or without the 'variant:' prefix)")
    ap.add_argument('--out', default='.sweeps/eval_restored_players.txt')
    a = ap.parse_args()
    label = a.variant if a.variant.startswith('variant:') else f'variant:{a.variant}'
    b = pd.read_parquet(a.boards)
    base = b[b['variant'] == 'base']
    var = b[b['variant'] == label]
    keys = ['year', 'week', 'Player']
    restored = var.merge(base[keys].drop_duplicates(), on=keys, how='left', indicator=True)
    restored = restored[restored['_merge'] == 'left_only'].drop(columns='_merge')
    rows = []
    for year, part in restored.groupby('year'):
        stats, _t, name_col, _ = load_and_merge_data(int(year), 'Full PPR')
        wk = pd.to_numeric(stats['week'], errors='coerce')
        pts = stats.assign(_w=wk).groupby([name_col, '_w'])['fantasy_points_ppr'].sum()
        for week, g in part.groupby('week'):
            played = played_players(int(year), int(week), stats, name_col)
            for _, r in g.iterrows():
                rows.append({'year': int(year), 'week': int(week), 'player': r['Player'], 'pos': r['Pos'], 'team': r['Team'],
                             'avail': float(pd.to_numeric(r['Availability'], errors='coerce')),
                             'proj': float(pd.to_numeric(r['Model Proj Pts'], errors='coerce')),
                             'played': r['Player'] in played,
                             'actual': float(pts.get((r['Player'], float(week)), 0.0))})
    d = pd.DataFrame(rows)
    out = [f"{a.boards} [{label}]: {len(d):,} player-weeks on the variant's board and not the base's "
           f"({d['player'].nunique() if len(d) else 0} players)", '']
    if d.empty:
        out.append('none')
    else:
        for scope, m in (('played', d['played']), ('every', d['avail'] > EXPECTED)):
            s = d[m]
            out.append(f"{scope}: n = {len(s):,}")
            t = s.groupby('pos').apply(lambda g: pd.Series({
                'n': len(g), 'proj': g['proj'].mean(), 'actual': g['actual'].mean(), 'bias': (g['proj'] - g['actual']).mean(),
                'rmse': float(np.sqrt(((g['proj'] - g['actual']) ** 2).mean())),
                'rmse_if_0': float(np.sqrt((g['actual'] ** 2).mean()))}), include_groups=False)
            out.append(t.to_string(float_format=lambda v: f'{v:.2f}'))
            out.append('')
        out.append('largest projections:')
        out.append(d.sort_values('proj', ascending=False).head(25).to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    text = '\n'.join(out)
    print(text)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
