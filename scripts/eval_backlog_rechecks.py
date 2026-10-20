"""
The 2026-10-19 backlog read-only checks in one place, so they can be re-run on any board file with the lab-dump columns
(the M2 capture writes one: .sweeps/stat_blend_traces.parquet.board.csv, built by the CURRENT model; the older
calibration_lab_v7.csv no longer matches it).

    M3  TD levels by position, season and projection tier (START pool, played)
    P1  slope of actual on projected for every displayed stat line, with a leave-one-season-out fitted line
    M4  QB passing yards by projection bucket and a top-end shrink (display only)
    D8  coverage of the Range-of-outcomes bands (Starter tier) in the fit window and the held-out seasons
    M9  tier bands vs role-confidence heuristic vs per-player width, mean pinball loss

Everything is scored on the board's own final points (`final`), START pool = the top N per (year, week, position) by raw
(harness v2 STARTABLE_N), rows with a box score, weeks 3-17 (D8 and M9: weeks 5-17, the fit window of the bands).

    python scripts/eval_backlog_rechecks.py [--board .sweeps/stat_blend_traces.parquet.board.csv] [--only m3,p1,m4,d8,m9]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_distribution import (WEEKLY_DISTRIBUTION_BANDS, _width_scale, player_width_scale,  # noqa: E402
                                      tier_for_rank)
from scripts.eval_calibration_schemes import load_dump, top_n_mask  # noqa: E402

TDS = {'QB': 'passing_tds', 'RB': 'rushing_tds', 'WR': 'receiving_tds', 'TE': 'receiving_tds'}
SLOPE_CASES = [('QB', 'passing_yards'), ('QB', 'passing_attempts'), ('QB', 'rushing_yards'), ('RB', 'rushing_yards'),
               ('RB', 'receiving_yards'), ('RB', 'receptions'), ('WR', 'receiving_yards'), ('WR', 'receptions'),
               ('TE', 'receiving_yards'), ('TE', 'receptions'), ('QB', 'passing_tds'), ('WR', 'receiving_tds'),
               ('RB', 'rushing_tds'), ('TE', 'receiving_tds')]


def start_pool(d):
    d = d[d['week'].between(3, 17)].copy()
    d['start'] = top_n_mask(d, d['raw'].to_numpy())
    return d[d['start'] & d['played'] & (d['avail'] > 0.5)]


def m3(S):
    print('\n=== M3: TD levels, projected vs actual per game (START pool, played) ===')
    rows = []
    for pos, st in TDS.items():
        for y in [*sorted(S['year'].unique()), 'all']:
            s = S[(S['pos'] == pos) & ((S['year'] == y) if y != 'all' else True)]
            rows.append((pos, y, len(s), s['p_' + st].mean(), s['a_' + st].mean()))
    r = pd.DataFrame(rows, columns=['pos', 'year', 'n', 'proj', 'act'])
    r['bias%'] = 100 * (r['proj'] / r['act'] - 1)
    print(r.round(3).to_string(index=False))
    print('\nby projection quintile:')
    for pos, st in TDS.items():
        s = S[S['pos'] == pos].copy()
        s['tier'] = pd.qcut(s['p_' + st], 5, duplicates='drop')
        g = s.groupby('tier', observed=True).agg(n=('p_' + st, 'size'), proj=('p_' + st, 'mean'), act=('a_' + st, 'mean'))
        g['bias%'] = 100 * (g['proj'] / g['act'] - 1)
        print(pos)
        print(g.round(3).to_string())


def p1(S):
    print('\n=== P1: slope of actual on projected, and a leave-one-season-out fitted line ===')
    rows = []
    for pos, st in SLOPE_CASES:
        s = S[S['pos'] == pos]
        s = s[s['p_' + st] > 0.05]
        p, a = s['p_' + st].to_numpy(float), s['a_' + st].to_numpy(float)
        b, a0 = np.polyfit(p, a, 1)
        dr, dm = [], []
        for y in sorted(s['year'].unique()):
            tr, te = s[s['year'] != y], s[s['year'] == y]
            bb, aa = np.polyfit(tr['p_' + st].to_numpy(float), tr['a_' + st].to_numpy(float), 1)
            pt, at = te['p_' + st].to_numpy(float), te['a_' + st].to_numpy(float)
            cal = np.clip(aa + bb * pt, 0, None)
            dr.append(np.sqrt(((cal - at) ** 2).mean()) - np.sqrt(((pt - at) ** 2).mean()))
            dm.append(np.abs(cal - at).mean() - np.abs(pt - at).mean())
        rows.append({'pos': pos, 'stat': st, 'n': len(s), 'mean proj': p.mean(), 'mean act': a.mean(), 'slope': b,
                     'rmse base': np.sqrt(((p - a) ** 2).mean()), 'LOYO dRMSE': np.mean(dr), 'LOYO dMAE': np.mean(dm),
                     'yrs better': f'{sum(x < 0 for x in dr)}/{len(dr)}'})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def m4(S):
    print('\n=== M4: QB passing yards by projection bucket (starters: 15+ projected attempts) ===')
    q = S[(S['pos'] == 'QB') & (S['p_passing_attempts'] > 15)].copy()
    q['py'], q['ay'] = q['p_passing_yards'], q['a_passing_yards']
    q['b'] = pd.cut(q['py'], [0, 200, 225, 250, 275, 300, 400])
    g = q.groupby('b', observed=True).agg(n=('py', 'size'), proj=('py', 'mean'), act=('ay', 'mean'))
    g['bias'] = g['proj'] - g['act']
    g['bias%'] = 100 * g['bias'] / g['act']
    g['beat the projection %'] = q.groupby('b', observed=True).apply(lambda x: 100 * (x['ay'] > x['py']).mean())
    print(g.round(1).to_string())

    def tr(y, knot, share):
        return np.where(y > knot, y - share * (y - knot), y)
    grid = [(k, s_) for k in (225, 240, 250, 260, 275) for s_ in (0.1, 0.2, 0.3, 0.4, 0.5)]
    out = []
    for y in sorted(q['year'].unique()):
        fit, te = q[q['year'] != y], q[q['year'] == y]
        _, k, s_ = min((np.sqrt(((tr(fit['py'].to_numpy(), k, s_) - fit['ay']) ** 2).mean()), k, s_) for k, s_ in grid)
        d_ = (np.sqrt(((tr(te['py'].to_numpy(), k, s_) - te['ay']) ** 2).mean()) - np.sqrt(((te['py'] - te['ay']) ** 2).mean()))
        out.append(d_)
        print(f'  held-out {y}: chosen knot {k} share {s_} dRMSE {d_:+.3f}')
    print(f'  mean {np.mean(out):+.3f}, better in {sum(x < 0 for x in out)} of {len(out)}')


def band_frame(d):
    d = d[d['week'].between(5, 17) & d['played'] & (d['avail'] > 0.5) & d['actual'].notna() & (d['final'] > 0.5)].copy()
    d['rank'] = d.groupby(['year', 'week', 'pos'])['final'].rank(ascending=False, method='first')
    d['tier'] = d['rank'].map(tier_for_rank)
    return d


def d8(d):
    print('\n=== D8: share of actuals below each band edge (target 10/25/50/75/90) ===')
    d = band_frame(d)
    rows = []
    for (pos, tier), g in d.groupby(['pos', 'tier']):
        band = WEEKLY_DISTRIBUTION_BANDS.get(pos, {}).get(tier)
        if not band or len(g) < 100:
            continue
        for label, sub in (('2021-23 (fit)', g[g['year'] <= 2023]), ('2024-25 (held out)', g[g['year'] >= 2024])):
            a, p = sub['actual'].to_numpy(), sub['final'].to_numpy()
            r = {'pos': pos, 'tier': tier, 'sample': label, 'n': len(sub)}
            for q in (10, 25, 50, 75, 90):
                r[f'<P{q}'] = 100 * (a < p * band[q]).mean()
            rows.append(r)
    print(pd.DataFrame(rows).round(1).to_string(index=False))


def pin(a, qv, q):
    e = a - qv
    return np.maximum(q / 100 * e, (q / 100 - 1) * e)


def m9(d):
    from data.transforms import load_and_merge_data
    print('\n=== M9: band width, mean pinball loss over P10-P90 (Starter tier) ===')
    d = band_frame(d)
    d = d[d['tier'] == 'Starter']
    logs = {}
    for y in sorted(d['year'].unique()):
        for yy in (y - 1, y):
            if yy in logs:
                continue
            st, _tc, nc, _ = load_and_merge_data(int(yy), 'Full PPR')
            st = st.copy()
            st['w'] = pd.to_numeric(st['week'], errors='coerce')
            st = st[st['w'].between(1, 18)]
            logs[yy] = {n: g.sort_values('w')[['w', 'fantasy_points_ppr']].to_numpy() for n, g in st.groupby(nc, observed=True)}

    def own(r):
        pts = []
        for yy in (r.year - 1, r.year):
            a = logs.get(yy, {}).get(r.player)
            if a is None:
                continue
            if yy == r.year:
                a = a[a[:, 0] < r.week]
            pts.extend(a[:, 1].tolist())
        return np.array(pts[-16:])
    d['own'] = [own(r) for r in d.itertuples()]
    Q = (10, 25, 50, 75, 90)
    rows = []
    for scheme in ('tier', 'heuristic', 'personalized'):
        tot = {q: [] for q in Q}
        cov = {q: [] for q in Q}
        for r in d.itertuples():
            b = WEEKLY_DISTRIBUTION_BANDS[r.pos]['Starter']
            sc = 1.0 if scheme == 'tier' else _width_scale(r.role_conf) if scheme == 'heuristic' else player_width_scale(r.role_conf, b, r.own)
            for q in Q:
                qv = max(0.0, r.final * (b[50] + (b[q] - b[50]) * sc))
                tot[q].append(pin(r.actual, qv, q))
                cov[q].append(r.actual < qv)
        row = {'scheme': scheme, 'mean pinball': np.mean([np.mean(tot[q]) for q in Q])}
        row.update({f'<P{q}': 100 * np.mean(cov[q]) for q in Q})
        rows.append(row)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--board', default='.sweeps/stat_blend_traces.parquet.board.csv')
    ap.add_argument('--only', default='m3,p1,m4,d8,m9')
    a = ap.parse_args()
    pd.set_option('display.width', 220)
    d = load_dump(a.board)
    S = start_pool(d)
    print(f'board {a.board}: {len(d):,} player-weeks, START pool played rows {len(S):,}, years {sorted(d["year"].unique())}')
    todo = set(a.only.split(','))
    if 'm3' in todo:
        m3(S)
    if 'p1' in todo:
        p1(S)
    if 'm4' in todo:
        m4(S)
    if 'd8' in todo:
        d8(d)
    if 'm9' in todo:
        m9(d)


if __name__ == '__main__':
    main()
