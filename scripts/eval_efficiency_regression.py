"""
Top-end efficiency regression, offline (2026-10-13). User: top-end players feel a hair over-projected, projections
above ~115 yards are unrealistic; the board's stat columns are used for prop betting.

What the lab dump shows (2021-2025 weeks 3-17, played rows): projected yards per opportunity is far too spread out -
regressing the actual rate on the projected rate gives a slope of 0.32 (WR yards/target), 0.30 (TE), 0.48 (RB
yards/carry), 0.54 (QB yards/attempt). Volume is close. So 100+ yard receiving projections average 84 and 300+ passing
projections 275. The stat columns are never calibrated (the points line only moves points), so the yardage on the board
is the most inflated number on it.

Candidate (one change per channel, volume untouched):

    new_yards = volume * (m + lam * (projected_yards / volume - m))

m = the position's volume-weighted projected rate on the fit seasons (so the league level does not move, only the
spread), lam fit by least squares on the fit seasons' yards (closed form, every played row with volume, weighted by
volume through the yards themselves). Points are re-scored from the new line, then the SHIPPED points line and season
anchor are applied unchanged (exactly what the harness would do with the flag on). Everything is held out
leave-one-season-out: each season's lam and m come from the other four.

    python scripts/eval_efficiency_regression.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_projections import SEASON_ANCHOR_S  # noqa: E402
from scripts.eval_calibration_schemes import (anchor, boot_ci, load_dump, paired_scores,  # noqa: E402
                                              shipped_line)

# channel -> (positions, volume stat, yards stat, points per yard (Full PPR), min projected volume)
CHANNELS = {
    'rec': (('WR', 'TE', 'RB'), 'targets', 'receiving_yards', 0.1, 1.0),
    'rush': (('RB',), 'rushing_attempts', 'rushing_yards', 0.1, 3.0),
    'pass': (('QB',), 'passing_attempts', 'passing_yards', 0.04, 10.0),
}
# optional: catches per target (PPR points + reception props)
CATCH = (('WR', 'TE', 'RB'), 'targets', 'receptions', 1.0, 1.0)


def channel_rows(d, pos, vol, mn):
    return (d['pos'] == pos).to_numpy() & (d['p_' + vol] >= mn).to_numpy()


def fit_channel(d, fit_mask, pos, vol, yds, mn):
    """(m, lam) for one position/channel from the fit rows (played, weeks 3-17)."""
    m_rows = fit_mask & channel_rows(d, pos, vol, mn)
    v = d.loc[m_rows, 'p_' + vol].to_numpy(float)
    y = d.loc[m_rows, 'p_' + yds].to_numpy(float)
    a = d.loc[m_rows, 'a_' + yds].to_numpy(float)
    m = y.sum() / v.sum()
    x = y - v * m                     # projected yards above the volume-at-mean line = v * (eff - m)
    lam = float((x * (a - v * m)).sum() / (x * x).sum())
    return m, float(np.clip(lam, 0.0, 1.0))


def apply_channel(d, stats, pos, vol, yds, mn, m, lam):
    rows = channel_rows(d, pos, vol, mn)
    v = d.loc[rows, 'p_' + vol].to_numpy(float)
    y = stats[yds][rows]
    stats[yds][rows] = np.clip(v * (m + lam * (y / v - m)), 0.0, None)


def loyo(d, channels, strength=1.0):
    """Held-out new stat lines + params. `strength` scales the move: lam_used = 1 - strength * (1 - lam)."""
    stats = {s: d['p_' + s].to_numpy(float).copy() for s in
             ('receiving_yards', 'rushing_yards', 'passing_yards', 'receptions')}
    params = {}
    years = sorted(d['year'].unique())
    base_fit = d['played'].to_numpy() & d['week'].between(3, 17).to_numpy() & d['actual'].notna().to_numpy()
    for y in years:
        fit = base_fit & (d['year'] != y).to_numpy()
        tgt = (d['year'] == y).to_numpy()
        sub = d[tgt]
        sub_stats = {k: v[tgt].copy() for k, v in stats.items()}
        for ch in channels:
            positions, vol, yds, _w, mn = CATCH if ch == 'catch' else CHANNELS[ch]
            for pos in positions:
                m, lam = fit_channel(d, fit, pos, vol, yds, mn)
                lam = 1.0 - strength * (1.0 - lam)
                params.setdefault((ch, pos), []).append((y, round(m, 3), round(lam, 3)))
                apply_channel(sub, sub_stats, pos, vol, yds, mn, m, lam)
        for k in stats:
            stats[k][tgt] = sub_stats[k]
    return stats, params


def rescore(d, stats):
    raw = d['raw'].to_numpy(float).copy()
    for s, w in (('receiving_yards', 0.1), ('rushing_yards', 0.1), ('passing_yards', 0.04), ('receptions', 1.0)):
        raw += w * (stats[s] - d['p_' + s].to_numpy(float))
    raw = np.clip(raw, 0.0, None)
    dd = d.copy()
    dd['raw'] = raw
    return anchor(shipped_line(dd), dd, SEASON_ANCHOR_S), raw


def yard_bins(d, proj, yds, positions, edges, window):
    m = window & d['pos'].isin(positions).to_numpy() & (proj > 0)
    act = d['a_' + yds].to_numpy(float)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = m & (proj >= lo) & (proj < hi)
        if b.sum() < 5:
            out.append(f'{lo}-{hi}: n<5')
            continue
        out.append(f'{lo}+' if hi > 900 else f'{lo}-{hi}')
        out[-1] += (f': n={int(b.sum())} proj {proj[b].mean():.1f} act {act[b].mean():.1f} '
                    f'({act[b].mean() / proj[b].mean() - 1:+.0%})')
    return out


def stat_rmse(d, proj, yds, positions, window, top=None):
    m = window & d['pos'].isin(positions).to_numpy() & (proj > 0)
    if top is not None:
        m &= top
    e = proj[m] - d['a_' + yds].to_numpy(float)[m]
    return float(np.sqrt((e ** 2).mean())), float(e.mean()), int(m.sum())


def rank_bias(d, pred, window):
    out = {}
    rk = pd.Series(pred, index=d.index).groupby([d['wk_key'], d['pos']]).rank(ascending=False, method='first').to_numpy()
    act = d['actual'].to_numpy(float)
    for pos, lo, hi in (('WR', 1, 3), ('WR', 4, 8), ('WR', 9, 16), ('RB', 1, 3), ('RB', 4, 8), ('TE', 1, 3),
                        ('QB', 1, 3), ('QB', 4, 8)):
        m = window & (d['pos'] == pos).to_numpy() & (rk >= lo) & (rk <= hi)
        out[f'{pos}{lo}-{hi}'] = float((pred[m] - act[m]).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default='.sweeps/calibration_lab_v7.csv')
    ap.add_argument('--week1', default='.sweeps/calibration_lab_v7_wk1_ourlads.csv')
    a = ap.parse_args()
    d = load_dump(a.dump, a.week1)
    d['final'] = anchor(shipped_line(d), d, SEASON_ANCHOR_S)       # the shipped board (Q cut aside)
    played = d['played'].to_numpy() & d['actual'].notna().to_numpy()
    in_season = d['week'].between(3, 17).to_numpy()
    window = played & in_season                                      # 2021-2025 (stat checks)
    harness_window = window & (d['year'] >= 2022).to_numpy()         # harness window (points)

    variants = {
        'rec only': (['rec'], 1.0),
        'rec+rush+pass': (['rec', 'rush', 'pass'], 1.0),
        'rec+rush+pass, half': (['rec', 'rush', 'pass'], 0.5),
        'rec+rush+pass+catch': (['rec', 'rush', 'pass', 'catch'], 1.0),
    }
    base_stats = {s: d['p_' + s].to_numpy(float) for s in ('receiving_yards', 'rushing_yards', 'passing_yards',
                                                         'receptions')}
    results = {'shipped': (base_stats, d['final'].to_numpy(float), None)}
    for name, (chs, k) in variants.items():
        st, params = loyo(d, chs, k)
        pts, _raw = rescore(d, st)
        results[name] = (st, pts, params)

    _, _, p = results['rec+rush+pass+catch']
    print('Held-out params (m, lam) by season:')
    for key, vals in p.items():
        print(f'  {key[0]:<5}{key[1]:<3} ' + '  '.join(f'{y}: m={m:.2f} lam={l:.2f}' for y, m, l in vals))

    print('\n== Stat columns (2021-2025 wk3-17, played), binned by what the board would SHOW ==')
    for label, yds, pos, edges in (('receiving yards', 'receiving_yards', ('WR', 'TE', 'RB'), [60, 80, 100, 115, 999]),
                                   ('rushing yards', 'rushing_yards', ('RB',), [60, 80, 100, 115, 999]),
                                   ('passing yards', 'passing_yards', ('QB',), [250, 275, 300, 999])):
        print(f'  {label}')
        for name, (st, _pts, _p) in results.items():
            print(f'    {name:<22} ' + ' | '.join(yard_bins(d, st[yds], yds, pos, edges, window)))
    print('\n  stat RMSE / bias, all rows with volume | startable rows (top-N by shown points):')
    top = {name: None for name in results}
    from scripts.eval_calibration_schemes import top_n_mask
    for name, (st, pts, _p) in results.items():
        top[name] = top_n_mask(d, pts)
    for yds, pos in (('receiving_yards', ('WR', 'TE', 'RB')), ('rushing_yards', ('RB',)), ('passing_yards', ('QB',)),
                     ('receptions', ('WR', 'TE', 'RB'))):
        line = f'    {yds:<16}'
        for name, (st, _pts, _p) in results.items():
            r_all, b_all, _ = stat_rmse(d, st[yds], yds, pos, window)
            r_top, b_top, _ = stat_rmse(d, st[yds], yds, pos, window, top[name])
            line += f' | {name}: {r_all:.2f}/{b_all:+.2f}, start {r_top:.2f}/{b_top:+.2f}'
        print(line)

    print('\n== Points: projected-rank bias (shown - actual), harness window 2022-2025 ==')
    rows = []
    for name, (_st, pts, _p) in results.items():
        r = {'variant': name}
        r.update({k: f'{v:+.2f}' for k, v in rank_bias(d, pts, harness_window).items()})
        rows.append(r)
    pd.set_option('display.width', 250)
    print(pd.DataFrame(rows).to_string(index=False))

    print('\n== Points: harness-style paired START pools vs shipped, 2022-2025 wk3-17 [bootstrap CI over weeks] ==')
    base = d['final'].to_numpy(float)
    rows = []
    for name, (_st, pts, _p) in results.items():
        if name == 'shipped':
            continue
        sc = paired_scores(d, base, pts, harness_window)
        r = {'variant': name}
        for scope in ('START-ALL', 'START-QB', 'START-RB', 'START-WR', 'START-TE', 'ALL'):
            x = sc[scope]
            dr = x['rmse_v'] - x['rmse_b']
            if scope in ('START-ALL', 'ALL'):
                lo, hi = boot_ci(x['weeks'])
                r[scope] = f'{dr:+.3f} [{lo:+.3f},{hi:+.3f}]'
            else:
                r[scope] = f'{dr:+.3f}'
        sa = sc['START-ALL']
        r['bias'] = f"{sa['bias_b']:+.2f}->{sa['bias_v']:+.2f}"
        r['pairwise WR'] = f"{sc['START-WR']['pw_b']:.3f}->{sc['START-WR']['pw_v']:.3f}"
        r['wk W-L'] = f"{sa['wins']}-{sa['losses']}"
        by_year = []
        for y in (2022, 2023, 2024, 2025):
            s2 = paired_scores(d, base, pts, harness_window & (d['year'] == y).to_numpy())['START-ALL']
            by_year.append(s2['rmse_v'] - s2['rmse_b'])
        r['yrs better'] = f"{sum(v < 0 for v in by_year)}/4"
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
