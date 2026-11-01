"""
Game script in the stat recal (backlog M15, 2026-11-01; bar in .sweeps/stat_recal_game_script_bar.md, set before the run).

The QB stat columns run high for big favorites: favored by 7.5+, rushing yards +6 and passing yards +12 on the deep-audit panel.
Holding the other fixed, the QB rushing miss follows the implied team total and the passing miss follows the margin - a big
favorite runs the ball with its backs to kill the clock instead of scrambling or throwing. This adds the margin m (own implied
points - opponent's) and the total t (own implied points - the week's league mean) to the recal's linear form (F4B) and scores
it held out by season, per stat:

  F4B  the shipped form;  G1  F4B + m + t;  G2  G1 + max(m - 7, 0)
  H    F4B + max(m - 7, 0) alone, for QB rushing (the bar's addendum: the MSE rule barely sees a bias in 10% of rows)

    python scripts/eval_stat_recal_game_script.py [--qb-only] [--emit]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.eval_stat_recal_forms import HIST, OUT, blended_b, f4_design, load, rows  # noqa: E402

QB_STATS = ('passing_attempts', 'passing_completions', 'passing_yards', 'passing_tds', 'passing_interceptions',
            'rushing_attempts', 'rushing_yards')
OTHER = {'RB': ('rushing_attempts', 'rushing_yards', 'targets', 'receptions', 'receiving_yards'),
         'WR': ('targets', 'receptions', 'receiving_yards'), 'TE': ('targets', 'receptions', 'receiving_yards')}
FORMS = ('F4B', 'G1', 'G2')
ADDENDUM_STATS = ('rushing_attempts', 'rushing_yards')   # QB stats the addendum form H is tried on
BIG_FAV = 7.0
BUCKETS = [-99, -3, 3, 7.5, 99]
BUCKET_LABELS = ['dog 3+', 'pk', 'fav 3-7.5', 'fav 7.5+']


def with_script(p):
    """p with m (margin) and t (implied points - the week's league mean over teams with a line)."""
    p = p.copy()
    teams = p[['year', 'week', 'team', 'imp_team']].dropna().drop_duplicates(['year', 'week', 'team'])
    league = teams.groupby(['year', 'week'])['imp_team'].mean().rename('_league')
    p = p.join(league, on=['year', 'week'])
    p['m'] = p['imp_team'] - p['imp_opp']
    p['t'] = p['imp_team'] - p['_league']
    return p


def design(d, stat, form):
    X = np.column_stack([f4_design(d, stat), blended_b(d, stat, 6.0)])
    if form == 'F4B':
        return X
    if form == 'H':
        return np.column_stack([X, np.maximum(d['m'].to_numpy(float) - BIG_FAV, 0.0)])
    extra = [d['m'].to_numpy(float), d['t'].to_numpy(float)]
    if form == 'G2':
        extra.append(np.maximum(d['m'].to_numpy(float) - BIG_FAV, 0.0))
    return np.column_stack([X] + extra)


def predict(form, tr, te, stat):
    coef = np.linalg.lstsq(design(tr, stat, form), tr[f'y_{stat}'].to_numpy(float), rcond=None)[0]
    return np.maximum(design(te, stat, form) @ coef, 0.0)


def heldout(p, pos, stat):
    """One row per player-week of (pos, stat): board, actual and each form's held-out prediction."""
    d = rows(p, pos, stat)
    d = d[d['m'].notna() & d['t'].notna()].reset_index(drop=True)
    yr = d['year'].to_numpy()
    out = []
    for t in (*HIST, 2026):
        te, tr = d[yr == t], d[np.isin(yr, HIST) & (yr != t)]
        if len(te) < 20:
            continue
        rec = te[['_row', 'year', 'week', 'key', 'team', 'm', 't']].copy()
        rec['pos'], rec['stat'] = pos, stat
        rec['board'], rec['y'] = te[f'p_{stat}'].to_numpy(float), te[f'y_{stat}'].to_numpy(float)
        for form in FORMS + (('H',) if pos == 'QB' and stat in ADDENDUM_STATS else ()):
            rec[form] = predict(form, tr, te, stat)
        out.append(rec)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def gains(h):
    recs = []
    for (pos, stat, season), g in h.groupby(['pos', 'stat', 'year']):
        e0 = ((g['board'] - g['y']) ** 2).mean()
        rec = {'pos': pos, 'stat': stat, 'season': season, 'n': len(g)}
        for form in FORMS:
            rec[form] = 100 * (1 - ((g[form] - g['y']) ** 2).mean() / e0)
        recs.append(rec)
    return pd.DataFrame(recs)


def verdict(gn):
    """Per (pos, stat): pooled gains, seasons better, 2026, and the form the bar picks (None = leave the stat alone)."""
    out = []
    for (pos, stat), g in gn.groupby(['pos', 'stat']):
        h, y26 = g[g['season'].isin(HIST)], g[g['season'] == 2026]
        rec = {'pos': pos, 'stat': stat}
        for form in FORMS:
            rec[f'{form}_pooled'] = float((h[form] * h['n']).sum() / h['n'].sum())
            rec[f'{form}_better'] = int((h[form] > 0).sum())
            rec[f'{form}_2026'] = float(y26[form].iloc[0]) if len(y26) else np.nan
        for a, b in (('G1', 'F4B'), ('G2', 'G1')):
            rec[f'{a}_beats_{b}'] = int((h[a] > h[b]).sum())

        def replaces(a, b):
            return (rec[f'{a}_pooled'] > rec[f'{b}_pooled'] and rec[f'{a}_beats_{b}'] >= 4
                    and (np.isnan(rec[f'{a}_2026']) or rec[f'{a}_2026'] >= rec[f'{b}_2026'] - 0.5))
        form = 'F4B'
        if replaces('G1', 'F4B'):
            form = 'G2' if replaces('G2', 'G1') else 'G1'
        ok = (rec[f'{form}_pooled'] > 0 and rec[f'{form}_better'] >= 4
              and (np.isnan(rec[f'{form}_2026']) or rec[f'{form}_2026'] >= -1.0))
        rec['form'] = form if ok else None
        out.append(rec)
    return pd.DataFrame(out)


def bucket_bias(h, forms):
    h = h.assign(bucket=pd.cut(h['m'], BUCKETS, labels=BUCKET_LABELS))
    rows_ = []
    for (stat, bucket), g in h.groupby(['stat', 'bucket'], observed=True):
        hist, y26 = g[g['year'].isin(HIST)], g[g['year'] == 2026]
        rec = {'stat': stat, 'bucket': bucket, 'n': len(hist)}
        for f in ('board', *forms):
            rec[f] = float((hist[f] - hist['y']).mean())
            rec[f'{f}_2026'] = float((y26[f] - y26['y']).mean()) if len(y26) else np.nan
        rows_.append(rec)
    return pd.DataFrame(rows_)


def addendum(h):
    """The addendum's rules for H against F4B on each QB rushing stat, held out; the bucket is rule 3's 'fav 7.5+' (m above
    7.5), with m >= 7.5 reported alongside. {stat: True when H replaces F4B}."""
    out, recs = {}, []
    for stat in ADDENDUM_STATS:
        g = h[(h['pos'] == 'QB') & (h['stat'] == stat)]
        if not len(g) or 'H' not in g:
            continue
        for label, fav in (('m > 7.5', g['m'] > 7.5), ('m >= 7.5', g['m'] >= 7.5)):
            hist = g['year'].isin(HIST)
            err = {f: g[f] - g['y'] for f in ('board', 'F4B', 'H')}
            bias = {f: float(e[hist & fav].mean()) for f, e in err.items()}
            mse = {f: float((err[f][hist & fav] ** 2).mean()) for f in ('F4B', 'H')}
            closer = sum(abs(err['H'][(g['year'] == y) & fav].mean()) < abs(err['F4B'][(g['year'] == y) & fav].mean())
                         for y in HIST)
            gain = {}
            for f in ('F4B', 'H'):
                per = g.groupby('year').apply(lambda x: 100 * (1 - ((x[f] - x['y']) ** 2).mean()
                                                               / ((x['board'] - x['y']) ** 2).mean()), include_groups=False)
                n = g.groupby('year').size()
                gain[f] = (float((per[list(HIST)] * n[list(HIST)]).sum() / n[list(HIST)].sum()), float(per.get(2026, np.nan)))
            a = abs(bias['H']) <= 0.5 * abs(bias['F4B']) and closer >= 4
            b = mse['H'] < mse['F4B']
            c = gain['H'][0] >= gain['F4B'][0] - 0.1 and not gain['H'][1] < gain['F4B'][1] - 0.5
            y26 = (g['year'] == 2026) & fav
            recs.append({'stat': stat, 'bucket': label, 'n_fav': int((hist & fav).sum()), 'bias board': bias['board'],
                         'bias F4B': bias['F4B'], 'bias H': bias['H'], 'H closer (of 5)': closer,
                         'bias 2026 F4B': float(err['F4B'][y26].mean()), 'bias 2026 H': float(err['H'][y26].mean()),
                         'fav MSE F4B': mse['F4B'], 'fav MSE H': mse['H'], 'gain F4B': gain['F4B'][0], 'gain H': gain['H'][0],
                         'gain 2026 F4B': gain['F4B'][1], 'gain 2026 H': gain['H'][1], 'a': a, 'b': b, 'c': c})
            if label == 'm > 7.5':
                out[stat] = a and b and c
    with pd.option_context('display.width', 260, 'display.max_columns', 30):
        print(pd.DataFrame(recs).round(2).to_string(index=False))
    return out


def market(h, chosen):
    """2026: each stat's side against the last priced line before kickoff (Pinnacle, then DraftKings), board vs chosen form."""
    from data.market_devig import implied_mean_from_line
    from scripts.deep_audit import PROP_SPECS, posted_join, posted_lines_2026
    try:
        arc, kick = posted_lines_2026()
    except Exception as exc:                                              # noqa: BLE001
        print(f'market: no archive ({exc})')
        return
    recs = []
    for stat in ('passing_yards', 'passing_attempts', 'passing_completions', 'rushing_yards', 'rushing_attempts'):
        d = h[(h['pos'] == 'QB') & (h['stat'] == stat) & (h['year'] == 2026)].reset_index(drop=True)
        form = chosen.get(('QB', stat))
        if not len(d) or not form:
            continue
        spec = PROP_SPECS[stat]
        for _, L in posted_join(d, spec, arc, kick).iterrows():
            i = int(L['i'])
            imp = implied_mean_from_line(L['line'], L['p_book'], spec['market'], 'QB')
            y = float(d.at[i, 'y'])
            if imp is None or y == float(L['line']):
                continue
            recs.append({'stat': stat, 'm': float(d.at[i, 'm']), 'implied': float(imp), 'board': float(d.at[i, 'board']),
                         'new': float(d.at[i, form]), 'over': y > float(L['line']), 'y': y})
    r = pd.DataFrame(recs)
    if not len(r):
        print('market: no rows')
        return
    r.to_csv(os.path.join(OUT, 'game_script_market_2026.csv'), index=False)
    out = []
    for (stat, fav), g in r.groupby(['stat', r['m'] >= 7.5]):
        rec = {'stat': stat, 'fav 7.5+': bool(fav), 'n': len(g), 'board-mkt': float((g['board'] - g['implied']).mean()),
               'new-mkt': float((g['new'] - g['implied']).mean()), 'actual-mkt': float((g['y'] - g['implied']).mean())}
        for col in ('board', 'new'):
            hit = np.where(g[col] > g['implied'], g['over'], ~g['over'])
            rec[f'{col}_hit'] = float(hit.mean())
        out.append(rec)
    with pd.option_context('display.width', 220):
        print(pd.DataFrame(out).round(3).to_string(index=False))


def main():
    p = with_script(load((3, 17)))
    parts = [heldout(p, 'QB', s) for s in QB_STATS]
    if '--qb-only' not in sys.argv:
        parts += [heldout(p, pos, s) for pos, stats in OTHER.items() for s in stats]
    h = pd.concat([x for x in parts if len(x)], ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    h.to_parquet(os.path.join(OUT, 'game_script_heldout.parquet'))
    gn = gains(h)
    gn.to_csv(os.path.join(OUT, 'game_script_gains.csv'), index=False)
    v = verdict(gn)
    v.to_csv(os.path.join(OUT, 'game_script_verdict.csv'), index=False)
    with pd.option_context('display.width', 260, 'display.max_columns', 30):
        print('held out by season, stat MSE gain vs the board (%), pooled 2021-2025 / seasons better / 2026')
        print(v.round(2).to_string(index=False))
        qb = h[h['pos'] == 'QB']
        print('\nQB bias (projection - actual) by spread bucket, held out, 2021-2025 pooled and 2026')
        b = bucket_bias(qb[qb['stat'].isin(['passing_yards', 'passing_attempts', 'rushing_yards', 'rushing_attempts'])], FORMS)
        print(b.round(2).to_string(index=False))
    chosen = {(r.pos, r.stat): (r.form if isinstance(r.form, str) else None) for r in v.itertuples()}
    print('\naddendum (QB rushing, H = F4B + max(m - 7, 0) against F4B, held out):')
    for stat, ok in addendum(h).items():
        if ok and chosen.get(('QB', stat)) == 'F4B':
            chosen[('QB', stat)] = 'H'
    print('chosen:', {k: f for k, f in chosen.items() if f and (k[0] == 'QB' or f != 'F4B')})
    # consistency: completions above attempts after the chosen forms
    qa = h[(h['pos'] == 'QB') & h['stat'].isin(['passing_attempts', 'passing_completions'])]
    if len(qa):
        w = qa.pivot_table(index='_row', columns='stat', values=['board', *FORMS])
        att = w[(chosen.get(('QB', 'passing_attempts')) or 'board', 'passing_attempts')]
        comp = w[(chosen.get(('QB', 'passing_completions')) or 'board', 'passing_completions')]
        print(f'\ncompletions > attempts after the chosen forms: {int((comp > att + 1e-9).sum())} of {int(comp.notna().sum())}')
    print('\n2026 posted-line check (QB):')
    market(h, chosen)
    if '--emit' in sys.argv:
        emit(p, chosen)


def emit(p, chosen):
    """The coefficients that ship: each chosen stat's form fit on every season 2021-2026."""
    print('\n# fit on 2021-2026; F4B order (const, projection, s2d, l3, last season, s2d/l3/last-season missing, blend), then'
          ' m, t (G1, G2) and max(m - 7, 0) (G2, H)')
    for (pos, stat), form in chosen.items():
        if not form or (pos != 'QB' and form == 'F4B'):
            continue
        d = rows(p, pos, stat)
        d = d[d['m'].notna() & d['t'].notna()].reset_index(drop=True)
        coef = np.linalg.lstsq(design(d, stat, form), d[f'y_{stat}'].to_numpy(float), rcond=None)[0]
        print(pos, stat, form, len(d), [round(float(c), 4) for c in coef])


if __name__ == '__main__':
    main()
