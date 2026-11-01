"""
Medians under v2_stat_recal_own_history (2026-11-01, bar in .sweeps/stat_recal_median_bar.md, set before the run).

data/weekly_median.py turns a mean into a median with a curve fit on the board's means (median / mean as a function of the
mean). The recal moves the stat columns toward the player's own history, so the same curve may no longer sit at the middle game:
the recal'd mean is a different, tighter estimate. On the deep-audit panel (today's model) each moved row is scored three ways,
leave-one-season-out over 2021-2025 (2026 on a 2021-2025 fit): today's curve on the recal'd mean (A), a curve re-fit on recal'd
means with scripts/eval_stat_medians.py's method (B), and today's curve on the pre-recal mean (A0, the reference).

    python scripts/eval_stat_recal_medians.py [--emit]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data.weekly_projections as wp  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402
from data.weekly_median import MEDIAN_CURVES, RATIO_CLIP  # noqa: E402
from scripts.eval_stat_medians import MIN_PROJECTION, fit_curve  # noqa: E402

PANEL = '.sweeps/deep_audit/panel.parquet'
OUT = '.sweeps/stat_recal'
HIST = (2021, 2022, 2023, 2024, 2025)
STATS = sorted({s for v in wp.STAT_RECAL_OWN_HISTORY.values() for s in v})


def recal_rows():
    """One row per (moved player-week, stat): pos, stat, year, week, pre (board mean), m (recal'd mean), a (actual)."""
    p = pd.read_parquet(PANEL)
    p = p[p['played'].astype(bool) & (pd.to_numeric(p['Availability'], errors='coerce').fillna(1.0) >= 0.99)]
    hcols = ['h_games', 'h_ps_games'] + [f'h_{k}_{s}' for s in STATS for k in ('s2d', 'l3', 'ps')]
    feats = {'v2_stat_recal_own_history', 'v2_stat_recal_own_history_edges'}
    out = []
    for (yr, wk), g in p.groupby(['year', 'week']):
        form, table = wp.stat_recal_table(feats, wk)
        if not table:
            continue
        res = pd.DataFrame({'Player': g['player'].values, 'Pos': g['pos'].values, 'Team': g['team'].values,
                            'Model Proj Pts': g['p_points'].values, 'Availability': g['Availability'].values,
                            'Workload If Active': g['Workload If Active'].values,
                            'QB Projected Starter': g['QB Projected Starter'].values})
        for s in STATS:
            res[s] = g[f'p_{s}'].values
        hist = g[hcols].copy()
        hist.index = pd.MultiIndex.from_arrays([clean_name_exact(g['player'].astype(str)).values, g['pos'].values])
        hist = hist[~hist.index.duplicated(keep=False)]
        new, _ = wp.apply_stat_recal_own_history(res, hist, form, table)
        for s in STATS:
            pre_col = f'_pre_recal_{s}'
            if pre_col not in new:
                continue
            moved = new[pre_col].notna().to_numpy()
            out.append(pd.DataFrame({'pos': res['Pos'].to_numpy()[moved], 'stat': s, 'year': yr, 'week': wk,
                                     'pre': new[pre_col].to_numpy(float)[moved], 'm': new[s].to_numpy(float)[moved],
                                     'a': pd.to_numeric(g[f'y_{s}'], errors='coerce').to_numpy()[moved]}))
    d = pd.concat(out, ignore_index=True)
    d = d[d['a'].notna() & (d['m'] > d['stat'].map(MIN_PROJECTION).clip(lower=0.05))].reset_index(drop=True)
    # weeks 3-17 (the fit) and weeks 1-2 jittered by their own generators, so a change in one never moves the other's draws
    d['aj'] = np.nan
    for seed, rows in ((7, d['week'] >= 3), (8, d['week'] <= 2)):
        d.loc[rows, 'aj'] = d.loc[rows, 'a'] + np.random.default_rng(seed).uniform(-0.5, 0.5, int(rows.sum()))
    return d


def curve_median(coef, m):
    m = np.maximum(np.asarray(m, float), 1e-6)
    return m * np.clip(coef[0] + coef[1] * np.log(m), *RATIO_CLIP)


def score(aj, med):
    return float(np.mean(np.abs(aj - med))), float(np.mean(aj > med))


def main():
    d = recal_rows()
    early, d = d[d['week'] <= 2], d[d['week'] >= 3]
    recs = []
    for (pos, stat), g in d.groupby(['pos', 'stat']):
        cur = MEDIAN_CURVES[(pos, stat)]
        for t in (*HIST, 2026):
            te, tr = g[g['year'] == t], g[g['year'].isin(HIST) & (g['year'] != t)]
            if len(te) < 30:
                continue
            fit = fit_curve(tr)
            for name, med in (('A', curve_median(cur, te['m'])), ('B', curve_median(fit, te['m'])),
                              ('A0', curve_median(cur, te['pre']))):
                mae, over = score(te['aj'].to_numpy(), med)
                recs.append({'pos': pos, 'stat': stat, 'season': t, 'method': name, 'n': len(te), 'mae': mae, 'over': over})
        # weeks 1-2 with the weeks 3-17 curves fit on 2021-2025
        e = early[(early['pos'] == pos) & (early['stat'] == stat)]
        if len(e) >= 30:
            fit = fit_curve(g[g['year'].isin(HIST)])
            for name, med in (('A', curve_median(cur, e['m'])), ('B', curve_median(fit, e['m'])),
                              ('A0', curve_median(cur, e['pre']))):
                mae, over = score(e['aj'].to_numpy(), med)
                recs.append({'pos': pos, 'stat': stat, 'season': 'wk1-2', 'method': name, 'n': len(e), 'mae': mae,
                             'over': over})
    r = pd.DataFrame(recs)
    os.makedirs(OUT, exist_ok=True)
    r.to_csv(os.path.join(OUT, 'medians.csv'), index=False)
    hist = r[r['season'].isin(HIST)]
    pooled = hist.assign(w_mae=hist['mae'] * hist['n'], w_over=hist['over'] * hist['n']).groupby(
        ['pos', 'stat', 'method'])[['w_mae', 'w_over', 'n']].sum()
    pooled['mae'], pooled['over'] = pooled['w_mae'] / pooled['n'], pooled['w_over'] / pooled['n']
    t = pooled[['mae', 'over']].unstack('method')
    t[('mae', 'B_vs_A_%')] = 100 * (t[('mae', 'B')] / t[('mae', 'A')] - 1)
    with pd.option_context('display.width', 200):
        print('pooled 2021-2025, leave-one-season-out (weeks 3-17)')
        print(t.round(4).to_string())
        for season in (2026, 'wk1-2'):
            s = r[r['season'] == season].pivot_table(index=['pos', 'stat'], columns='method', values=['over', 'mae'])
            print(f'\n{season}'); print(s.round(4).to_string())
    dev = (t['over'] - 0.5).abs().mean()
    dev26 = (r[r['season'] == 2026].pivot_table(index=['pos', 'stat'], columns='method', values='over') - 0.5).abs().mean()
    dev12 = (r[r['season'] == 'wk1-2'].pivot_table(index=['pos', 'stat'], columns='method', values='over') - 0.5).abs().mean()
    print('\nmean |over - 0.5|  pooled:', dev.round(4).to_dict(), ' 2026:', dev26.round(4).to_dict(),
          ' wk1-2:', dev12.round(4).to_dict())
    worst = t[('mae', 'B_vs_A_%')].max()
    a_tight = ((t[('over', 'A')] - 0.5).abs() <= 0.01).all()
    passes = (dev['B'] < dev['A']) and worst <= 0.5 and (dev26['B'] - dev26['A'] <= 0.01) and not a_tight
    print(f'bar: B dev {dev["B"]:.4f} vs A {dev["A"]:.4f}; worst MAE B vs A {worst:+.2f}%; 2026 B-A {dev26["B"] - dev26["A"]:+.4f};'
          f' A all within 0.49-0.51: {a_tight} -> {"SHIP B" if passes else "keep A"}')
    if '--emit' in sys.argv:
        print('\nMEDIAN_CURVES_RECAL = {   # on recal\'d means, 2021-2026 weeks 3-17')
        for (pos, stat), g in d.groupby(['pos', 'stat']):
            ca, cb = fit_curve(g)
            print(f"    ('{pos}', '{stat}'): ({ca:.4f}, {cb:.4f}),")


if __name__ == '__main__':
    main()
