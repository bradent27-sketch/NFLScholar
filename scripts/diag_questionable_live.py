"""Out-of-sample check of the Questionable shortfall on LIVE 2026 boards (2026-10-08).

Every board saved to data/ledger/ keeps the 'Injury Status' it showed when it was built (FantasyPros tag + its reported
probability), so a Questionable tag can be compared with what the player then did, split by when in the week the board
was built. Weeks 1-3 were backfilled on 2026-09-24 from the stored FantasyPros imports (their timing is unknown), week
4 has six builds from Tuesday to Sunday evening. Played rows only (the harness criterion).

    python scripts/diag_questionable_live.py --year 2026
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.loaders import load_schedule  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=2026)
    a = ap.parse_args()
    stats, _t, name_col, _ = load_and_merge_data(a.year, 'Full PPR')
    sched = load_schedule(a.year)
    sched['week'] = pd.to_numeric(sched['week'], errors='coerce')
    first_kick = {}
    for wk, g in sched.groupby('week'):
        day = pd.to_datetime(g['gameday'], errors='coerce')
        time = g['gametime'].astype(str).where(g['gametime'].astype(str).str.match(r'^\d{1,2}:\d{2}$', na=False), '13:00')
        k = pd.to_datetime(day.dt.strftime('%Y-%m-%d') + ' ' + time).dt.tz_localize('US/Eastern', ambiguous='NaT',
                                                                                    nonexistent='NaT').dt.tz_convert('UTC')
        first_kick[int(wk)] = k.min()
    rows = []
    for path in sorted(glob.glob(os.path.join('data', 'ledger', f'{a.year}_wk*.parquet'))):
        m = re.match(rf'{a.year}_wk(\d+)_(\d{{8}}T\d{{6}})', os.path.basename(path))
        week = int(m.group(1))
        built = pd.Timestamp(m.group(2), tz='UTC')
        led = pd.read_parquet(path)
        act = (stats[pd.to_numeric(stats['week'], errors='coerce') == week]
               .groupby(name_col, observed=True)['fantasy_points_ppr'].sum())
        if act.empty:
            continue
        act.index = clean_name_exact(pd.Series(act.index))
        led['_key'] = clean_name_exact(led['Player'])
        led['actual'] = led['_key'].map(act)
        tag = led['Injury Status'].fillna('').astype(str)
        led['tag'] = np.where(tag.str.startswith('Questionable'), 'Q', np.where(tag.eq(''), 'none', 'other'))
        led['prob'] = pd.to_numeric(tag.str.extract(r'\((\d+)%\)')[0], errors='coerce') / 100.0
        led['hours_before_first_kick'] = (first_kick[week] - built).total_seconds() / 3600.0
        led['week'], led['built'] = week, built
        rows.append(led[['week', 'built', 'hours_before_first_kick', 'Player', 'Pos', 'Model Proj Pts', 'Raw Model Proj Pts',
                         'Availability', 'tag', 'prob', 'actual']])
    d = pd.concat(rows, ignore_index=True)
    d = d[(d['Availability'] > 0.01) & (d['Model Proj Pts'] > 0.5)]
    pd.set_option('display.width', 200)

    print('Questionable-tagged rows that played, per ledger build (all positions), then WR only')
    out = []
    for (week, built), g in d.groupby(['week', 'built']):
        q = g[(g['tag'] == 'Q') & g['actual'].notna()]
        n = g[(g['tag'] == 'none') & g['actual'].notna() & g['Pos'].isin(['QB', 'RB', 'WR', 'TE'])]
        qw = q[q['Pos'] == 'WR']
        out.append({'week': week, 'built (UTC)': built.strftime('%m-%d %H:%M'), 'hrs before 1st kick': round(g['hours_before_first_kick'].iloc[0]),
                    'Q played': len(q), 'Q act/proj': round(q['actual'].sum() / max(q['Model Proj Pts'].sum(), 1e-9), 3) if len(q) else np.nan,
                    'WR Q played': len(qw), 'WR Q act/proj': round(qw['actual'].sum() / max(qw['Model Proj Pts'].sum(), 1e-9), 3) if len(qw) else np.nan,
                    'healthy act/proj': round(n['actual'].sum() / n['Model Proj Pts'].sum(), 3)})
    print(pd.DataFrame(out).to_string(index=False))

    # one board per week (the earliest and the latest build) pooled
    for label, pick in (('earliest build of each week', 'min'), ('latest build of each week', 'max')):
        keep = d.groupby('week')['built'].transform(pick) == d['built']
        x = d[keep & (d['tag'] == 'Q') & d['actual'].notna()]
        print(f"\n{label}: Q rows that played {len(x)}; act/proj all {x['actual'].sum() / x['Model Proj Pts'].sum():.3f}; "
              + '; '.join(f"{p} n={len(g)} {g['actual'].sum() / g['Model Proj Pts'].sum():.3f}" for p, g in x.groupby('Pos') if len(g) >= 5))
    latest = d[d.groupby('week')['built'].transform('max') == d['built']]
    x = latest[(latest['tag'] == 'Q')]
    print('\nlatest builds, ALL Q-tagged rows (played or not): share that played, by the reported probability band')
    x = x.assign(band=pd.cut(x['prob'], [-0.01, 0.25, 0.5, 0.75, 0.9, 1.01]))
    print(x.groupby('band', observed=True).agg(n=('Player', 'size'), played=('actual', lambda s: s.notna().mean())).round(3).to_string())
    xp = x[x['actual'].notna() & x['Pos'].isin(['WR', 'RB', 'TE', 'QB'])]
    print('\nlatest builds, Q rows that played, act/proj by reported probability band:')
    print(xp.groupby('band', observed=True).apply(lambda g: pd.Series({'n': len(g), 'act/proj': g['actual'].sum() / g['Model Proj Pts'].sum()}),
                                                 include_groups=False).round(3).to_string())


if __name__ == '__main__':
    main()
