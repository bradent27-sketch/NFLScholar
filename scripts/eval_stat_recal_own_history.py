"""
Offline replay of the board flag 'v2_stat_recal_own_history' (backlog M13; bar in .sweeps/stat_recal_bar.md, set before this ran).

The deep audit (2026-10-28) found the board's stat columns overshoot whenever they leave the player's own level: only 58-81% of
a WR/TE/RB volume departure came true, about 45% of a QB rushing one. The flag moves each stat to

    b + k (p - b)

on the if-active level (p / (Availability x Workload If Active)), b = the player's box-score season to date after 3+ games, else
last season's per-game after 4+ games. Points are untouched.

On the deep-audit panel (.sweeps/deep_audit/panel.parquet, today's 76-flag board replayed 2021-2025 and 2026 weeks 1-7) this fits
k per (position, stat, week window), held out by season, and scores the stat MSE against the board as is; a symmetric k against
one k for departures up and one for departures down. Writes .sweeps/stat_recal/: heldout.csv (per season), verdict.csv (the bar
per position/stat/window), constants.csv (the 2021-2025 fits to ship) and leans_2026.csv (the posted-line side record).

    python scripts/eval_stat_recal_own_history.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.deep_audit import (PANEL, PROP_SPECS, clustered_mean_se, follow_through, own_baseline,  # noqa: E402
                                posted_join, posted_lines_2026)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.sweeps', 'stat_recal')
STATS = {'QB': ('passing_attempts', 'passing_yards', 'rushing_attempts', 'rushing_yards'),
         'RB': ('rushing_attempts', 'rushing_yards', 'targets', 'receptions', 'receiving_yards'),
         'WR': ('targets', 'receptions', 'receiving_yards'),
         'TE': ('targets', 'receptions', 'receiving_yards')}
WINDOWS = {'wk1': (1, 1), 'wk2': (2, 2), 'wk3-17': (3, 17), 'wk18': (18, 18)}
HIST_SEASONS = (2021, 2022, 2023, 2024, 2025)
MIN_POINTS = 3.0


def load():
    p = pd.read_parquet(PANEL)
    p['played'] = p['played'].astype(bool)
    starter = p['QB Projected Starter'].astype(str).str.lower().isin(('true', '1', '1.0'))
    p = p[p['played'] & (p['pos'].ne('QB') | starter)].reset_index(drop=True)
    avail = (pd.to_numeric(p['Availability'], errors='coerce').fillna(1.0)
             * pd.to_numeric(p['Workload If Active'], errors='coerce').fillna(1.0))
    p['_avail'] = avail
    return p[avail > 0.01].reset_index(drop=True)


def frame(p, pos, stat, lo, hi):
    d = p[(p['pos'] == pos) & p['week'].between(lo, hi) & p[f'p_{stat}'].notna() & p[f'y_{stat}'].notna()]
    b = own_baseline(d, stat)
    ok = ~np.isnan(b)
    d = d[ok].reset_index(drop=True)
    d['_b'] = b[ok]
    d['_m'] = d[f'p_{stat}'].to_numpy(float) / d['_avail'].to_numpy(float)     # the if-active level
    d['_y'] = d[f'y_{stat}'].to_numpy(float)
    d['_cut'] = pd.to_numeric(d['p_points'], errors='coerce').fillna(0).to_numpy() >= MIN_POINTS
    d['_cl'] = (d['year'] * 100 + d['week']).to_numpy()
    return d


def fit(d):
    """(k, k_up, k_down) on the p_points >= 3 rows of d."""
    d = d[d['_cut']]
    x, z, cl = (d['_m'] - d['_b']).to_numpy(), (d['_y'] - d['_b']).to_numpy(), d['_cl'].to_numpy()
    k = follow_through(x, z, cl)[0]
    up, dn = x > 0, x < 0
    k_up = follow_through(x[up], z[up], cl[up])[0] if up.sum() >= 30 else k
    k_dn = follow_through(x[dn], z[dn], cl[dn])[0] if dn.sum() >= 30 else k
    return k, k_up, k_dn


def apply(d, k, k_up, k_dn):
    """(symmetric, asymmetric) recalibrated board stat for the rows of d (back on the board's availability-weighted scale)."""
    x = (d['_m'] - d['_b']).to_numpy()
    b, a = d['_b'].to_numpy(), d['_avail'].to_numpy()
    sym = np.maximum(b + k * x, 0.0) * a
    asym = np.maximum(b + np.where(x > 0, k_up, k_dn) * x, 0.0) * a
    return sym, asym


def score(d, pred, scope):
    sel = d['_cut'].to_numpy() if scope == 'cut' else np.ones(len(d), dtype=bool)
    y, m = d['_y'].to_numpy()[sel], d[f'p_{d.attrs["stat"]}'].to_numpy(float)[sel]
    q = pred[sel]
    if len(y) < 30:
        return None
    mse_m, mse_q = float(np.mean((m - y) ** 2)), float(np.mean((q - y) ** 2))
    diff, se = clustered_mean_se((q - y) ** 2 - (m - y) ** 2, d['_cl'].to_numpy()[sel])
    return {'n': int(sel.sum()), 'mse_model': mse_m, 'gain_pct': 100 * (1 - mse_q / mse_m), 'z': diff / se if se else np.nan}


def main():
    os.makedirs(OUT, exist_ok=True)
    p = load()
    rows, consts = [], []
    for pos, stats in STATS.items():
        for stat in stats:
            for wname, (lo, hi) in WINDOWS.items():
                d = frame(p, pos, stat, lo, hi)
                d.attrs['stat'] = stat
                yr = d['year'].to_numpy()
                hist = d[np.isin(yr, HIST_SEASONS)]
                if hist['_cut'].sum() < 150:
                    continue
                k, k_up, k_dn = fit(hist)
                consts.append({'pos': pos, 'stat': stat, 'window': wname, 'k': k, 'k_up': k_up, 'k_down': k_dn,
                               'n_fit': int(hist['_cut'].sum())})
                for t in (*HIST_SEASONS, 2026):
                    te = d[yr == t].reset_index(drop=True)
                    te.attrs['stat'] = stat
                    if not len(te):
                        continue
                    tr = d[np.isin(yr, HIST_SEASONS) & (yr != t)]
                    kk = fit(tr)
                    sym, asym = apply(te, *kk)
                    for variant, pred in (('sym', sym), ('asym', asym)):
                        for scope in ('cut', 'all'):
                            s = score(te, pred, scope)
                            if s:
                                rows.append({'pos': pos, 'stat': stat, 'window': wname, 'season': t, 'variant': variant,
                                             'scope': scope, 'k': kk[0], 'k_up': kk[1], 'k_down': kk[2], **s})
    held = pd.DataFrame(rows)
    held.to_csv(os.path.join(OUT, 'heldout.csv'), index=False)
    pd.DataFrame(consts).to_csv(os.path.join(OUT, 'constants.csv'), index=False)
    verdict(held)
    leans(p, pd.DataFrame(consts))


def pooled(h):
    """Pooled 2021-2025 gain from per-season rows (MSE weighted by n)."""
    h = h[h['season'].isin(HIST_SEASONS)]
    mse_m = (h['mse_model'] * h['n']).sum()
    mse_q = (h['mse_model'] * (1 - h['gain_pct'] / 100) * h['n']).sum()
    return 100 * (1 - mse_q / mse_m) if mse_m else np.nan


def verdict(held):
    out = []
    for (pos, stat, w), g in held.groupby(['pos', 'stat', 'window'], sort=False):
        rec = {'pos': pos, 'stat': stat, 'window': w}
        for v in ('sym', 'asym'):
            c = g[(g['variant'] == v) & (g['scope'] == 'cut')]
            a = g[(g['variant'] == v) & (g['scope'] == 'all')]
            hc = c[c['season'].isin(HIST_SEASONS)]
            rec[f'{v}_pooled'] = pooled(c)
            rec[f'{v}_seasons_better'] = int((hc['gain_pct'] > 0).sum())
            rec[f'{v}_n_seasons'] = len(hc)
            r26 = c[c['season'] == 2026]
            rec[f'{v}_2026'] = float(r26['gain_pct'].iloc[0]) if len(r26) else np.nan
            rec[f'{v}_2026_n'] = int(r26['n'].iloc[0]) if len(r26) else 0
            rec[f'{v}_all_pooled'] = pooled(a)
        sc, ac = g[(g['variant'] == 'sym') & (g['scope'] == 'cut')], g[(g['variant'] == 'asym') & (g['scope'] == 'cut')]
        m = sc.merge(ac, on='season', suffixes=('_s', '_a'))
        m = m[m['season'].isin(HIST_SEASONS)]
        rec['asym_beats_sym_seasons'] = int((m['gain_pct_a'] > m['gain_pct_s']).sum())
        use_asym = rec['asym_beats_sym_seasons'] >= 4 and rec['asym_pooled'] > rec['sym_pooled']
        v = 'asym' if use_asym else 'sym'
        rec['variant'] = v
        if w == 'wk3-17':
            passes = (rec[f'{v}_pooled'] > 0 and rec[f'{v}_seasons_better'] >= 4
                      and (np.isnan(rec[f'{v}_2026']) or rec[f'{v}_2026'] >= -1.0) and rec[f'{v}_all_pooled'] >= 0)
        else:
            passes = rec[f'{v}_pooled'] > 0 and rec[f'{v}_seasons_better'] >= 4
        rec['passes'] = bool(passes)
        out.append(rec)
    v = pd.DataFrame(out)
    v.to_csv(os.path.join(OUT, 'verdict.csv'), index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(v.round(2).to_string(index=False))


def leans(p, consts):
    """2026 weeks 3-7: the board stat's side against the last priced line before kickoff (Pinnacle, then DraftKings), as is and
    recalibrated as the board flag does it (2021-2025 k, the verdict's variant, only stats that passed, only rows projected 3+
    points). Side = the stat against the market's implied mean; a push is no bet."""
    from data.market_devig import implied_mean_from_line
    v = pd.read_csv(os.path.join(OUT, 'verdict.csv'))
    v = v[v['window'] == 'wk3-17']
    variants = {(r.pos, r.stat): r.variant for r in v.itertuples()}
    passes = {(r.pos, r.stat): bool(r.passes) for r in v.itertuples()}
    try:
        arc, kick = posted_lines_2026()
    except Exception as exc:                                              # noqa: BLE001
        print(f'leans: no archive ({exc})')
        return
    recs = []
    for pos, stats in STATS.items():
        for stat in stats:
            c = consts[(consts['pos'] == pos) & (consts['stat'] == stat) & (consts['window'] == 'wk3-17')]
            if not len(c):
                continue
            k = float(c['k'].iloc[0])
            d = frame(p, pos, stat, 3, 17)
            d = d[d['year'] == 2026].reset_index(drop=True)
            if not len(d):
                continue
            spec = PROP_SPECS[stat]
            ku, kd = (float(c['k_up'].iloc[0]), float(c['k_down'].iloc[0])) if variants.get((pos, stat)) == 'asym' else (k, k)
            dep = d['_m'] - d['_b']
            rec_stat = np.maximum(d['_b'] + np.where(dep > 0, ku, kd) * dep, 0.0) * d['_avail']
            rec_stat = np.where(d['_cut'] & passes.get((pos, stat), False), rec_stat, d[f'p_{stat}'])   # as the board applies it
            for _, L in posted_join(d, spec, arc, kick).iterrows():
                i = int(L['i'])
                imp = implied_mean_from_line(L['line'], L['p_book'], spec['market'], pos)
                y = float(d.at[i, '_y'])
                if y == float(L['line']) or imp is None:
                    continue
                recs.append({'pos': pos, 'stat': stat, 'week': int(L['week']), 'line': float(L['line']),
                             'implied': float(imp), 'raw': float(d.at[i, f'p_{stat}']), 'recal': float(rec_stat[i]),
                             'over': y > float(L['line'])})
    r = pd.DataFrame(recs)
    if not len(r):
        print('leans: no rows')
        return
    r.to_csv(os.path.join(OUT, 'leans_2026.csv'), index=False)
    out = []
    for (pos, stat), g in [(('ALL', 'ALL'), r)] + list(r.groupby(['pos', 'stat'])):
        row = {'pos': pos, 'stat': stat, 'n': len(g)}
        for col in ('raw', 'recal'):
            gap = g[col] / g['implied'] - 1
            hit = np.where(gap > 0, g['over'], ~g['over'])
            row[f'{col}_hit'] = float(hit.mean())
            big = gap.abs() >= 0.05
            row[f'{col}_hit_5pct'] = float(hit[big].mean()) if big.any() else np.nan
            row[f'{col}_n_5pct'] = int(big.sum())
        out.append(row)
    with pd.option_context('display.width', 250):
        print(pd.DataFrame(out).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
