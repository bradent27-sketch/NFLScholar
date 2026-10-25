"""
The backup-TE bands on the silent-games model (2026-10-25, user: "if you think there are pieces of the RB layer and backup TE
discount maybe that should be included too"). BACKUP_TE_ACTIVITY_BANDS is actual / projected targets for a TE below his room's
leader, by his offensive snap share in his latest snap-count row this season, fit on the 60-flag board - before silent games
were in his history, so part of what it took off (his statless games) is now in the rate. This re-reads the bands on the
silent-games boards with the discount OFF (the silent-games run's `silent_snapcurve_lite` variant boards), every game he was
expected to play (Availability > 0.5; no snap = 0 targets), by season, so what is left of each band shows.

    python scripts/diag_backup_te_silent.py [--boards .sweeps/harness_boards/silent_games_2022-2025_wk3-17.parquet] [--variant silent_snapcurve_lite]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.transforms import load_and_merge_data  # noqa: E402
from data.utils import clean_name_for_merge  # noqa: E402
from data.weekly_projections import BACKUP_TE_ACTIVITY_BANDS, _te_last_snap_shares, te_room_rank  # noqa: E402

EXPECTED = 0.5


def band_of(value):
    for i, (edge, _f) in enumerate(BACKUP_TE_ACTIVITY_BANDS):
        if (edge is None and value is None) or (edge is not None and value is not None and value <= edge):
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boards', default='.sweeps/harness_boards/silent_games_2022-2025_wk3-17.parquet')
    ap.add_argument('--variant', default='silent_snapcurve_lite')
    ap.add_argument('--out', default='.sweeps/diag_backup_te_silent.txt')
    a = ap.parse_args()
    b = pd.read_parquet(a.boards)
    b = b[(b['variant'] == a.variant) & (b['Pos'] == 'TE')].copy()
    rows = []
    for year, part in b.groupby('year'):
        stats, _t, name_col, _ = load_and_merge_data(int(year), 'Full PPR')
        wk = pd.to_numeric(stats['week'], errors='coerce')
        tgt = stats.assign(_w=wk).groupby([name_col, '_w'])['targets'].sum()
        for week, g in part.groupby('week'):
            avail = pd.to_numeric(g['Availability'], errors='coerce').fillna(1.0).to_numpy()
            live = avail > 0.01
            share = pd.to_numeric(g['Expected Snap Share'], errors='coerce').to_numpy()
            rank = te_room_rank(g['Team'].to_numpy(dtype=object), share, live)
            last = _te_last_snap_shares(int(year), int(week))
            keys = clean_name_for_merge(g['Player'].astype(str)).to_numpy(dtype=object)
            for i, (player, key) in enumerate(zip(g['Player'], keys)):
                if not np.isfinite(rank[i]) or rank[i] < 2 or avail[i] <= EXPECTED:
                    continue
                rows.append({'year': int(year), 'week': int(week), 'player': player, 'rank': int(rank[i]),
                             'band': band_of(last.get(key)), 'proj': float(g['targets'].iloc[i]),
                             'actual': float(tgt.get((player, float(week)), 0.0))})
    d = pd.DataFrame(rows).dropna(subset=['band'])
    d['band'] = d['band'].astype(int)
    labels = {i: ('no snap row yet' if e is None else f'last share <= {e:g}') for i, (e, _f) in enumerate(BACKUP_TE_ACTIVITY_BANDS)}
    out = [f"{a.boards} [{a.variant}]: TE room rank >= 2, expected to play, {len(d):,} rows; factor = actual / projected targets", '']
    t = d.groupby('band').agg(n=('proj', 'size'), proj=('proj', 'sum'), actual=('actual', 'sum'))
    t['factor_silent'] = t['actual'] / t['proj']
    t['factor_shipped'] = [BACKUP_TE_ACTIVITY_BANDS[i][1] for i in t.index]
    for year in sorted(d['year'].unique()):
        y = d[d['year'] == year].groupby('band').agg(p=('proj', 'sum'), a=('actual', 'sum'))
        t[str(year)] = (y['a'] / y['p']).reindex(t.index)
    t.index = [labels[i] for i in t.index]
    out.append(t.to_string(float_format=lambda v: f'{v:.3f}'))
    text = '\n'.join(out)
    print(text)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
