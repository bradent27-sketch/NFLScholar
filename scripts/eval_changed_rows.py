"""
The rows a candidate changes (2026-10-25, v2_rb_lone_lead_vacancy; user: ATL week 7, Bijan Robinson given all of an out
Brian Robinson's carries). The paired harness scores every player on both boards, so a rule that fires a few times a
season is diluted to nothing; this scores only the (year, week, player) rows whose projection the variant moved by more
than --min-change points, base against variant, from the run's --variant-boards file.

    played   games he played (a box-score row, or a silent game = 0 points)
    every    every week the variant expected him to play (Availability > 0.5; no game = 0)

By position and by direction (lowered / raised), with the paired RMSE change and a 95% bootstrap interval over team-weeks,
plus carries (projected vs actual) for the RBs.

    python scripts/eval_changed_rows.py --boards .sweeps/harness_boards/lone_lead_2022-2025_wk3-17.parquet --variant lone_lead
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
BOOTSTRAP = 2000


def _paired(g, rng):
    """RMSE base, variant, their difference and a team-week cluster bootstrap interval for it."""
    eb, ev = (g['base'] - g['actual']) ** 2, (g['var'] - g['actual']) ** 2
    rb, rv = float(np.sqrt(eb.mean())), float(np.sqrt(ev.mean()))
    clusters = g.assign(_eb=eb, _ev=ev).groupby(['year', 'week', 'team'])[['_eb', '_ev']].agg(['sum', 'count'])
    sb, sv, n = clusters[('_eb', 'sum')].to_numpy(), clusters[('_ev', 'sum')].to_numpy(), clusters[('_eb', 'count')].to_numpy()
    draws = []
    for _ in range(BOOTSTRAP):
        pick = rng.integers(0, len(n), len(n))
        draws.append(np.sqrt(sv[pick].sum() / n[pick].sum()) - np.sqrt(sb[pick].sum() / n[pick].sum()))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return pd.Series({'n': len(g), 'base': g['base'].mean(), 'var': g['var'].mean(), 'actual': g['actual'].mean(),
                      'rmse_base': rb, 'rmse_var': rv, 'delta': rv - rb, 'lo': lo, 'hi': hi})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boards', required=True)
    ap.add_argument('--variant', required=True, help="the variant's label (with or without the 'variant:' prefix)")
    ap.add_argument('--min-change', type=float, default=0.05)
    ap.add_argument('--out', default='.sweeps/eval_changed_rows.txt')
    a = ap.parse_args()
    label = a.variant if a.variant.startswith('variant:') else f'variant:{a.variant}'
    b = pd.read_parquet(a.boards)
    keys = ['year', 'week', 'Player']
    cols = keys + ['Pos', 'Team', 'Availability', 'Model Proj Pts', 'rushing_attempts']
    m = b[b['variant'] == label][cols].merge(b[b['variant'] == 'base'][keys + ['Model Proj Pts', 'rushing_attempts']],
                                             on=keys, suffixes=('', '_base'))
    m['var'] = pd.to_numeric(m['Model Proj Pts'], errors='coerce')
    m['base'] = pd.to_numeric(m['Model Proj Pts_base'], errors='coerce')
    m = m[(m['var'] - m['base']).abs() > a.min_change].copy()
    rows = []
    for year, part in m.groupby('year'):
        stats, _t, name_col, _ = load_and_merge_data(int(year), 'Full PPR')
        wk = pd.to_numeric(stats['week'], errors='coerce')
        by = stats.assign(_w=wk).groupby([name_col, '_w'])
        pts, car = by['fantasy_points_ppr'].sum(), by['rushing_attempts'].sum()
        for week, g in part.groupby('week'):
            played = played_players(int(year), int(week), stats, name_col)
            for _, r in g.iterrows():
                rows.append({'year': int(year), 'week': int(week), 'player': r['Player'], 'pos': r['Pos'], 'team': r['Team'],
                             'avail': float(pd.to_numeric(r['Availability'], errors='coerce')),
                             'base': r['base'], 'var': r['var'], 'played': r['Player'] in played,
                             'actual': float(pts.get((r['Player'], float(week)), 0.0)),
                             'car_base': float(pd.to_numeric(r['rushing_attempts_base'], errors='coerce')),
                             'car_var': float(pd.to_numeric(r['rushing_attempts'], errors='coerce')),
                             'car_actual': float(car.get((r['Player'], float(week)), 0.0))})
    d = pd.DataFrame(rows)
    out = [f"{a.boards} [{label}]: {len(d):,} player-weeks the variant moved by more than {a.min_change} pts "
           f"({d['player'].nunique() if len(d) else 0} players)", '']
    if d.empty:
        out.append('none')
    else:
        rng = np.random.default_rng(0)
        d['direction'] = np.where(d['var'] < d['base'], 'lowered', 'raised')
        fmt = lambda v: f'{v:.3f}'  # noqa: E731
        for scope, mask in (('every', d['avail'] > EXPECTED), ('played', d['played'])):
            s = d[mask]
            out.append(f"{scope}: n = {len(s):,}")
            out.append('  all: ' + _paired(s, rng).to_frame().T.to_string(index=False, float_format=fmt))
            out.append(s.groupby(['pos', 'direction']).apply(lambda g: _paired(g, rng), include_groups=False)
                       .to_string(float_format=fmt))
            rb = s[s['pos'] == 'RB']
            if len(rb):
                out.append(rb.groupby('direction').apply(lambda g: pd.Series({
                    'n': len(g), 'carries_base': g['car_base'].mean(), 'carries_var': g['car_var'].mean(),
                    'carries_actual': g['car_actual'].mean(), 'bias_base': (g['car_base'] - g['car_actual']).mean(),
                    'bias_var': (g['car_var'] - g['car_actual']).mean()}), include_groups=False).to_string(float_format=fmt))
            out.append('')
        out.append('largest moves:')
        d['move'] = d['var'] - d['base']
        out.append(d.reindex(d['move'].abs().sort_values(ascending=False).index).head(25)
                   [['year', 'week', 'player', 'pos', 'team', 'base', 'var', 'actual', 'played', 'car_base', 'car_var',
                     'car_actual']].to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    text = '\n'.join(out)
    print(text)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
