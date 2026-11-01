"""
Is 'v2_stat_recal_own_history' in its best form? (2026-11-01, user: "ship recal own history if you are sure this is the best it
can get"; bar in .sweeps/stat_recal_forms_bar.md, set before this ran.)

Same rows for every form (deep-audit panel, weeks 3-17, projected 3+ points, the flag's 13 position/stat pairs, rows with the
flag's baseline). Fits are held out by season (2026 on 2021-2025). Forms:
  F1   as built: b = season to date after 3 games, else last season after 4; b + k (p - b) (k_up / k_down where chosen).
  F2_K blended baseline (g s2d + K ps) / (g + K) when last season has 4+ games, else season to date.
  F3   F1 plus an intercept.
  F4   the prop engine's linear form on the projection, season to date, last 3, last season and missing flags.
  F5   F1 with its own k for weeks 3-8 and 9-17.
  F6   F1 with its own k by baseline source.
Writes .sweeps/stat_recal/forms.csv (per stat and season) and prints the verdict.

    python scripts/eval_stat_recal_forms.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.weekly_projections import STAT_RECAL_OWN_HISTORY  # noqa: E402
from scripts.deep_audit import PANEL, follow_through  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.sweeps', 'stat_recal')
HIST = (2021, 2022, 2023, 2024, 2025)
# games this season a row needs, by the first week of the window loaded (weeks 2-17: 1 - a player's first game back loses
# under F4B and F1); week 1 has none by definition
MIN_GAMES = {3: 1, 2: 1}


def load(weeks=(3, 17)):
    p = pd.read_parquet(PANEL)
    p['played'] = p['played'].astype(bool)
    starter = p['QB Projected Starter'].astype(str).str.lower().isin(('true', '1', '1.0'))
    return p[p['played'] & p['week'].between(*weeks) & (p['pos'].ne('QB') | starter)
             & (pd.to_numeric(p['p_points'], errors='coerce') >= 3)].reset_index(drop=True)


def rows(p, pos, stat):
    d = p[(p['pos'] == pos) & p[f'p_{stat}'].notna() & p[f'y_{stat}'].notna()].rename_axis('_row').reset_index()
    g, pg = d['h_games'].fillna(0).to_numpy(), d['h_ps_games'].fillna(0).to_numpy()
    s2d, ps = d[f'h_s2d_{stat}'].to_numpy(float), d[f'h_ps_{stat}'].to_numpy(float)
    b = np.where(g >= 3, s2d, np.where(pg >= 4, ps, np.nan))
    keep = ~np.isnan(b) & (g >= MIN_GAMES.get(int(d['week'].min()) if len(d) else 3, 0))
    d = d[keep].reset_index(drop=True)
    d['_b'], d['_g'], d['_pg'] = b[keep], g[keep], pg[keep]
    d['_season_src'] = d['_g'] >= 3
    return d


def kfit(x, z, cl, asym):
    if not asym:
        k = follow_through(x, z, cl)[0]
        return k, k
    up, dn = x > 0, x < 0
    return follow_through(x[up], z[up], cl[up])[0], follow_through(x[dn], z[dn], cl[dn])[0]


def kapply(x, ku, kd):
    return np.where(x > 0, ku, kd) * x


def blended_b(d, stat, K):
    s2d, ps = d[f'h_s2d_{stat}'].to_numpy(float), d[f'h_ps_{stat}'].to_numpy(float)
    g = d['_g'].to_numpy()
    blend = (np.nan_to_num(s2d) * g + K * ps) / (g + K)
    return np.where(d['_pg'].to_numpy() >= 4, blend, s2d)


def f4_design(d, stat):
    m = d[f'p_{stat}'].to_numpy(float)
    cols, flags = [np.ones(len(d)), m], []
    for c in ('h_s2d_', 'h_l3_', 'h_ps_'):
        v = d[f'{c}{stat}'].to_numpy(float)
        miss = np.isnan(v)
        cols.append(np.where(miss, m, v))
        flags.append(miss.astype(float))
    return np.column_stack(cols + flags)


def predict(form, tr, te, stat, asym):
    """Held-out prediction for te from a fit on tr."""
    y_tr = tr[f'y_{stat}'].to_numpy(float)
    m_tr, m_te = tr[f'p_{stat}'].to_numpy(float), te[f'p_{stat}'].to_numpy(float)
    cl = (tr['year'] * 100 + tr['week']).to_numpy()
    if form.startswith('F2_'):
        K = float(form.split('_')[1])
        b_tr, b_te = blended_b(tr, stat, K), blended_b(te, stat, K)
        ku, kd = kfit(m_tr - b_tr, y_tr - b_tr, cl, asym)
        return b_te + kapply(m_te - b_te, ku, kd)
    b_tr, b_te = tr['_b'].to_numpy(), te['_b'].to_numpy()
    x_tr, z_tr, x_te = m_tr - b_tr, y_tr - b_tr, m_te - b_te
    if form == 'F1':
        ku, kd = kfit(x_tr, z_tr, cl, asym)
        return b_te + kapply(x_te, ku, kd)
    if form == 'F3':
        if asym:
            X = np.column_stack([np.where(x_tr > 0, x_tr, 0), np.where(x_tr < 0, x_tr, 0), np.ones(len(x_tr))])
            Xt = np.column_stack([np.where(x_te > 0, x_te, 0), np.where(x_te < 0, x_te, 0), np.ones(len(x_te))])
        else:
            X, Xt = np.column_stack([x_tr, np.ones(len(x_tr))]), np.column_stack([x_te, np.ones(len(x_te))])
        coef = np.linalg.lstsq(X, z_tr, rcond=None)[0]
        return b_te + Xt @ coef
    if form == 'F4':
        coef = np.linalg.lstsq(f4_design(tr, stat), y_tr, rcond=None)[0]
        COEFS.setdefault((tr.attrs.get('key'), te['year'].iloc[0]), coef)
        return f4_design(te, stat) @ coef
    if form in ('F4B', 'F4B_noL3'):
        X_tr = np.column_stack([f4_design(tr, stat), blended_b(tr, stat, 6.0)])
        X_te = np.column_stack([f4_design(te, stat), blended_b(te, stat, 6.0)])
        if form == 'F4B_noL3':                     # drop last 3 (column 3) and its missing flag (column 6)
            X_tr, X_te = np.delete(X_tr, [3, 6], axis=1), np.delete(X_te, [3, 6], axis=1)
        coef = np.linalg.lstsq(X_tr, y_tr, rcond=None)[0]
        COEFS_B.setdefault((form, tr.attrs.get('key'), te['year'].iloc[0]), coef)
        return X_te @ coef
    if form == 'F4P':
        out = np.empty(len(te))
        for early in (True, False):
            s_tr, s_te = (tr['week'].to_numpy() <= 8) == early, (te['week'].to_numpy() <= 8) == early
            coef = np.linalg.lstsq(f4_design(tr[s_tr], stat), y_tr[s_tr], rcond=None)[0]
            out[s_te] = f4_design(te[s_te], stat) @ coef
        return out
    if form in ('F5', 'F6'):
        if form == 'F5':
            grp_tr, grp_te = tr['week'].to_numpy() <= 8, te['week'].to_numpy() <= 8
        else:
            grp_tr, grp_te = tr['_season_src'].to_numpy(), te['_season_src'].to_numpy()
        out = np.empty(len(te))
        for gval in (True, False):
            s_tr, s_te = grp_tr == gval, grp_te == gval
            ku, kd = kfit(x_tr[s_tr], z_tr[s_tr], cl[s_tr], asym) if s_tr.sum() >= 100 else kfit(x_tr, z_tr, cl, asym)
            out[s_te] = b_te[s_te] + kapply(x_te[s_te], ku, kd)
        return out
    raise ValueError(form)


FORMS = ('F1', 'F2_3', 'F2_6', 'F2_12', 'F3', 'F4', 'F5', 'F6', 'F4P', 'F4B', 'F4B_noL3')
COEFS, COEFS_B = {}, {}


def main():
    p = load()
    v = pd.read_csv(os.path.join(OUT, 'verdict.csv'))
    v = v[v['window'] == 'wk3-17']
    asym_of = {(r.pos, r.stat): r.variant == 'asym' for r in v.itertuples()}
    recs, pulled = [], {}
    for pos, stats in STAT_RECAL_OWN_HISTORY.items():
        for stat in stats:
            d = rows(p, pos, stat)
            yr = d['year'].to_numpy()
            for t in (*HIST, 2026):
                te = d[yr == t]
                tr = d[np.isin(yr, HIST) & (yr != t)]
                tr.attrs['key'] = (pos, stat)
                if not len(te):
                    continue
                y, m = te[f'y_{stat}'].to_numpy(float), te[f'p_{stat}'].to_numpy(float)
                mse0 = np.mean((m - y) ** 2)
                for form in FORMS:
                    q = np.maximum(predict(form, tr, te, stat, asym_of[(pos, stat)]), 0.0)
                    recs.append({'pos': pos, 'stat': stat, 'season': t, 'form': form, 'n': len(te),
                                 'gain_pct': 100 * (1 - np.mean((q - y) ** 2) / mse0)})
                    if form == 'F1':
                        pulled.setdefault(pos, {}).setdefault(stat, []).append(pd.Series(q, index=te['_row'].to_numpy()))
    f = pd.DataFrame(recs)
    f.to_csv(os.path.join(OUT, 'forms.csv'), index=False)
    # mean gain over the 13 stats per season; pooled 2021-2025 = mean over stats of the n-weighted pooled gain
    per = f.pivot_table(index=['pos', 'stat', 'form'], columns='season', values='gain_pct')
    ns = f.pivot_table(index=['pos', 'stat', 'form'], columns='season', values='n')
    per['pooled'] = (per[list(HIST)] * ns[list(HIST)]).sum(axis=1) / ns[list(HIST)].sum(axis=1)
    by_form = per.groupby(level='form').mean()
    base = by_form.loc['F1']
    out = []
    for form, row in by_form.iterrows():
        better = int(sum(row[t] > base[t] for t in HIST))
        passes = form != 'F1' and row['pooled'] > base['pooled'] and better >= 4 and row[2026] >= base[2026] - 0.5
        f4 = by_form.loc['F4']
        beat_f4 = int(sum(row[t] > f4[t] for t in HIST))
        out.append({'form': form, 'pooled': row['pooled'], **{str(t): row[t] for t in (*HIST, 2026)},
                    'seasons_beat_F1': better, 'passes': passes, 'seasons_beat_F4': beat_f4,
                    'beats_F4': bool(form.startswith('F4') and form != 'F4' and row['pooled'] > f4['pooled'] and beat_f4 >= 4
                                     and row[2026] >= f4[2026] - 0.5)})
    with pd.option_context('display.width', 250):
        print(pd.DataFrame(out).round(3).to_string(index=False))
        print(per[['pooled', 2026]].unstack('form').round(2).to_string())
    names = ['const', 'proj', 's2d', 'l3', 'last_season', 'miss_s2d', 'miss_l3', 'miss_ps']
    cf = pd.DataFrame([{'pos': k[0][0], 'stat': k[0][1], 'fold': k[1], **dict(zip(names, c))} for k, c in COEFS.items()])
    cf.to_csv(os.path.join(OUT, 'f4_coefs.csv'), index=False)
    nb = {'F4B': names + ['blend'], 'F4B_noL3': ['const', 'proj', 's2d', 'last_season', 'miss_s2d', 'miss_ps', 'blend']}
    pd.DataFrame([{'form': k[0], 'pos': k[1][0], 'stat': k[1][1], 'fold': k[2], **dict(zip(nb[k[0]], c))}
                  for k, c in COEFS_B.items()]).to_csv(os.path.join(OUT, 'f4b_coefs.csv'), index=False)
    with pd.option_context('display.width', 250):
        print(cf.groupby(['pos', 'stat'])[names[:5]].agg(['min', 'max']).round(3).to_string())
    # receptions above targets after the pull (F1, held out)
    for pos in ('RB', 'WR', 'TE'):
        if 'receptions' in pulled.get(pos, {}) and 'targets' in pulled.get(pos, {}):
            r, t = pd.concat(pulled[pos]['receptions']), pd.concat(pulled[pos]['targets'])
            j = pd.concat([r.rename('r'), t.rename('t')], axis=1).dropna()
            print(f'{pos}: pulled receptions > pulled targets on {int((j["r"] > j["t"] + 1e-9).sum())} of {len(j)} rows')


def edges():
    """Weeks 1, 2 and 18: F1 (one k) against F4B, each window on its own rows; the 2026 gain with a player bootstrap interval."""
    rng = np.random.default_rng(7)
    out = []
    for wk in (1, 2, 18):
        p = load((wk, wk))
        for pos, stats in STAT_RECAL_OWN_HISTORY.items():
            for stat in stats:
                d = rows(p, pos, stat)
                yr = d['year'].to_numpy()
                if np.isin(yr, HIST).sum() < 150:
                    continue
                for t in (*HIST, 2026):
                    te = d[yr == t]
                    tr = d[np.isin(yr, HIST) & (yr != t)]
                    if len(te) < 20:
                        continue
                    y, m = te[f'y_{stat}'].to_numpy(float), te[f'p_{stat}'].to_numpy(float)
                    for form in ('F1', 'F4B'):
                        q = np.maximum(predict(form, tr, te, stat, False), 0.0)
                        e0, e1 = (m - y) ** 2, (q - y) ** 2
                        rec = {'week': wk, 'pos': pos, 'stat': stat, 'season': t, 'form': form, 'n': len(te),
                               'gain_pct': 100 * (1 - e1.mean() / e0.mean())}
                        if t == 2026:
                            bs = [100 * (1 - e1[i].mean() / e0[i].mean())
                                  for i in (rng.integers(0, len(te), len(te)) for _ in range(2000))]
                            rec['ci_lo'], rec['ci_hi'] = np.percentile(bs, [5, 95])
                        out.append(rec)
    e = pd.DataFrame(out)
    e.to_csv(os.path.join(OUT, 'edges_forms.csv'), index=False)
    for wk, g in e.groupby('week'):
        per = g.pivot_table(index=['pos', 'stat', 'form'], columns='season', values='gain_pct')
        ns = g.pivot_table(index=['pos', 'stat', 'form'], columns='season', values='n')
        hist = [t for t in HIST if t in per.columns]
        per['pooled'] = (per[hist] * ns[hist]).sum(axis=1) / ns[hist].sum(axis=1)
        per['seasons_better'] = (per[hist] > 0).sum(axis=1)
        means = per.groupby(level='form')['pooled'].mean()
        form = 'F1' if means['F1'] > means['F4B'] else 'F4B'
        print(f'week {wk}: mean pooled gain F1 {means["F1"]:.2f}, F4B {means["F4B"]:.2f} -> {form}')
        sel = per.xs(form, level='form').copy()
        sel['passes'] = (sel['pooled'] > 0) & (sel['seasons_better'] >= 4)
        if wk in (1, 2):
            ci = g[(g['form'] == form) & (g['season'] == 2026)].set_index(['pos', 'stat'])[['ci_lo', 'ci_hi']]
            sel = sel.join(ci)
        with pd.option_context('display.width', 250):
            print(sel.round(2).to_string())


# What ships (2026-11-01): F4B in weeks 3-17 and week 1, one k in week 2 (the edges verdict; RB receptions out, since RB targets
# did not pass and catches moved alone outran the targets), week 18 left out. Final fits on
# every season 2021-2026 (the method was validated held out by season above).
SHIP = {'wk3-17': ((3, 17), 'F4B', None),
        1: ((1, 1), 'F4B', ('RB', 'receiving_yards', 'receptions', 'rushing_attempts', 'rushing_yards', 'targets',
                            'TE', 'targets', 'receptions', 'receiving_yards', 'WR', 'targets', 'receptions', 'receiving_yards')),
        2: ((2, 2), 'F1', ('RB', 'rushing_attempts', 'rushing_yards', 'TE', 'targets', 'receptions',
                           'receiving_yards', 'WR', 'targets', 'receptions', 'receiving_yards'))}


def _pairs(spec):
    if spec is None:
        return [(pos, stat) for pos, stats in STAT_RECAL_OWN_HISTORY.items() for stat in stats]
    out, pos = [], None
    for tok in spec:
        if tok in ('QB', 'RB', 'WR', 'TE'):
            pos = tok
        else:
            out.append((pos, tok))
    return out


def emit():
    """Print the constants that ship: per window, {pos: {stat: coefficients}} (F4B: const, projection, season to date, last 3,
    last season, three missing flags, blended baseline) or {pos: {stat: k}} (F1)."""
    for window, (weeks, form, spec) in SHIP.items():
        p = load(weeks)
        table = {}
        for pos, stat in _pairs(spec):
            d = rows(p, pos, stat)
            y = d[f'y_{stat}'].to_numpy(float)
            if form == 'F4B':
                X = np.column_stack([f4_design(d, stat), blended_b(d, stat, 6.0)])
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                table.setdefault(pos, {})[stat] = [round(float(c), 4) for c in coef]
            else:
                m, b = d[f'p_{stat}'].to_numpy(float), d['_b'].to_numpy()
                k = follow_through(m - b, y - b, (d['year'] * 100 + d['week']).to_numpy())[0]
                table.setdefault(pos, {})[stat] = round(float(k), 3)
        print(f'# {window}: {form}, rows {weeks}')
        if window == 'wk3-17':      # 1+ game this season: the two season-missing flags never fire; drop them
            table = {ps: {st: tuple(c[:5] + c[7:]) for st, c in v.items()} for ps, v in table.items()}
            print('# (const, projection, season to date, last 3, last season, last season missing, blended baseline)')
        elif form == 'F4B':         # week 1: no game yet, last season present -> const + projection + last season
            table = {ps: {st: (round(c[0] + c[5] + c[6], 4), round(c[1] + c[2] + c[3], 4), round(c[4] + c[8], 4))
                          for st, c in v.items()} for ps, v in table.items()}
            print('# (const, projection, last season)')
        print(repr(table))


if __name__ == '__main__':
    if '--edges' in sys.argv:
        edges()
    elif '--emit' in sys.argv:
        emit()
    else:
        main()
