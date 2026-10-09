"""Opening-week calibration, position by position (2026-10-08).

`eval_calibration_schemes.py` on the 2026-10-08 dump: the shipped board runs +1.8 pts high on startable players in
week 1 and the in-season lines have nothing left to give, so this looks only at weeks 1-2. For each candidate it fits
on three seasons, predicts the fourth (week 1: 2022-2025, the seasons with an archived pre-Week-1 depth chart; week 2:
2021-2025), and reports per position and per held-out season the START-pool dRMSE vs the shipped board, the bias, and
- for the per-stat candidates - the stat line itself. Also prints the lines a ship would use (fit on every season).

    python scripts/eval_opening_week_calibration.py --dump-path .sweeps/calibration_lab_v7.csv --week1-path .sweeps/calibration_lab_v7_wk1_ourlads.csv
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

import scripts.eval_calibration_schemes as ecs  # noqa: E402

POS = ecs.POS
CANDIDATES = {
    'line_half': ecs.make_line_scheme(0.5, 'all', 'none'),
    'line_full': ecs.make_line_scheme(1.0, 'all', 'none'),
    'stat_half': ecs.make_stat_scheme(0.5, 'all', 'none'),
    'stat_0.75': ecs.make_stat_scheme(0.75, 'all', 'none'),
    'stat_full': ecs.make_stat_scheme(1.0, 'all', 'none'),
    # level only (a ratio per stat, no intercept): the QB week-1 miss looks like a league-wide passing level
    'ratio_half': ecs.make_stat_scheme(0.5, 'all', 'none', intercept=False),
    'ratio_full': ecs.make_stat_scheme(1.0, 'all', 'none', intercept=False),
}


def run_week(d, week):
    wk = (d['week'] == week).to_numpy()
    years = sorted(d.loc[wk, 'year'].unique())
    base = d['final'].to_numpy(dtype=float)
    preds, lines = {}, {}
    for name, (fit, predict) in CANDIDATES.items():
        pred = base.copy()
        line_out = {s: d[f'p_{s}'].to_numpy(dtype=float).copy() for s in ecs.WEIGHTS}
        for y in years:
            test = wk & (d['year'] == y).to_numpy()
            train = wk & (d['year'] != y).to_numpy()
            cal, ln = predict(d, fit(d, train))
            pred[test] = cal[test]
            if ln is not None:
                for s in ln:
                    line_out[s][test] = ln[s][test]
        preds[name], lines[name] = pred, line_out
    print('\n' + '=' * 110 + f"\nWEEK {week}: held-out (leave-one-season-out over {years[0]}-{years[-1]}) vs SHIPPED\n" + '=' * 110)
    rows = []
    for name, pred in preds.items():
        for scope in [f'START-{p}' for p in POS] + ['START-ALL', 'ALL']:
            per = []
            for y in years:
                sc = ecs.paired_scores(d, base, pred, wk & (d['year'] == y).to_numpy())
                if scope in sc:
                    per.append(sc[scope]['rmse_v'] - sc[scope]['rmse_b'])
            sc = ecs.paired_scores(d, base, pred, wk)
            if scope not in sc:
                continue
            x = sc[scope]
            lo, hi = ecs.boot_ci(x['weeks']) if len(x['weeks']) > 2 else (np.nan, np.nan)
            rows.append({'scheme': name, 'scope': scope, 'n': x['n'], 'RMSE shipped': round(x['rmse_b'], 3),
                         'dRMSE': round(x['rmse_v'] - x['rmse_b'], 3), 'CI': f"[{lo:+.3f},{hi:+.3f}]",
                         'bias': f"{x['bias_b']:+.2f}->{x['bias_v']:+.2f}",
                         'dRMSE by season': ' '.join(f"{v:+.3f}" for v in per),
                         'better in': f"{sum(v < 0 for v in per)}/{len(per)}"})
    pd.set_option('display.width', 250)
    print(pd.DataFrame(rows).to_string(index=False))
    # stat line on the START pool (shipped board's top-N), played
    m0 = ecs.top_n_mask(d, base) & d['played'].to_numpy() & wk
    srows = []
    for pos in POS:
        m = m0 & (d['pos'] == pos).to_numpy()
        for s in ecs.POS_STATS[pos]:
            a = d[f'a_{s}'].to_numpy()[m]
            r = {'pos': pos, 'stat': s, 'n': int(m.sum()), 'actual mean': round(float(a.mean()), 2),
                 'bias shipped line': round(float((d[f'p_{s}'].to_numpy()[m] - a).mean()), 2)}
            for name in ('stat_half', 'stat_full'):
                r[f'bias {name}'] = round(float((lines[name][s][m] - a).mean()), 2)
            srows.append(r)
    print(f"\nweek {week} stat line, START pool, played (bias = projected - actual; the shipped stat line is uncalibrated):")
    print(pd.DataFrame(srows).to_string(index=False))
    # the lines a ship would use
    for name in ('line_full', 'stat_full', 'line_half'):
        fit, _ = CANDIDATES[name]
        params = fit(d, wk)
        print(f"\n{name}, fit on every season of week {week}:")
        for k, v in params.items():
            print(f"   {k}: ({v[0]:.3f}, {v[1]:.3f})")


# ---- the configuration a ship would use -------------------------------------------------------------------------
CANDIDATE_STRENGTH = 0.75
CANDIDATE_SCOPE = {1: ('RB', 'WR', 'TE'), 2: ('RB',)}       # QB never improved; week 2 only RB did
CANDIDATE_STATS = {   # scoring stats move the points; volume (attempts/targets) is calibrated so the line stays coherent
    'RB': ('rushing_attempts', 'rushing_yards', 'rushing_tds', 'targets', 'receptions', 'receiving_yards', 'receiving_tds'),
    'WR': ('targets', 'receptions', 'receiving_yards', 'receiving_tds'),
    'TE': ('targets', 'receptions', 'receiving_yards', 'receiving_tds'),
}


def fit_candidate(d, train):
    """{week: {pos: {stat: (slope, intercept)}}} on played, non-sidelined, non-zero rows of `train`."""
    out = {}
    base = train & d['played'].to_numpy() & d['live'].to_numpy()
    for week, positions in CANDIDATE_SCOPE.items():
        out[week] = {}
        for pos in positions:
            m = base & (d['week'] == week).to_numpy() & (d['pos'] == pos).to_numpy()
            out[week][pos] = {s: ecs._line(d[f'p_{s}'].to_numpy()[m], d[f'a_{s}'].to_numpy()[m], CANDIDATE_STRENGTH)
                              for s in CANDIDATE_STATS[pos]}
    return out


def apply_candidate(d, table):
    """(points, stat line) - mirrors data.weekly_projections.apply_opening_week_stat_calibration."""
    pts = d['final'].to_numpy(dtype=float).copy()
    line = {s: d[f'p_{s}'].to_numpy(dtype=float).copy() for s in set(ecs.WEIGHTS) | {'targets', 'rushing_attempts'}}
    live = d['live'].to_numpy()
    for week, by_pos in table.items():
        for pos, lines in by_pos.items():
            m = live & (d['week'] == week).to_numpy() & (d['pos'] == pos).to_numpy()
            for s, (sl, ic) in lines.items():
                line[s][m] = np.clip(sl * line[s][m] + ic, 0.0, None)
            if 'receptions' in lines and 'targets' in lines:
                line['receptions'][m] = np.minimum(line['receptions'][m], line['targets'][m])
            sc = np.zeros(m.sum())
            for s, w in ecs.WEIGHTS.items():
                sc += w * line[s][m]
            pts[m] = np.round(np.clip(sc, 0.0, None), 1)
    return pts, line


def run_candidate(d):
    print('\n' + '#' * 110 + f"\n# CANDIDATE: per-stat lines at strength {CANDIDATE_STRENGTH}, scope {CANDIDATE_SCOPE}\n" + '#' * 110)
    base = d['final'].to_numpy(dtype=float)
    pred = base.copy()
    line = {s: d[f'p_{s}'].to_numpy(dtype=float).copy() for s in set(ecs.WEIGHTS) | {'targets', 'rushing_attempts'}}
    for y in sorted(d['year'].unique()):
        test = (d['year'] == y).to_numpy()
        p, ln = apply_candidate(d, fit_candidate(d, ~test))
        pred[test] = p[test]
        for s in ln:
            line[s][test] = ln[s][test]
    rows = []
    for week in (1, 2, 3):
        wk = (d['week'] == week).to_numpy()
        years = sorted(d.loc[wk, 'year'].unique())
        for scope in [f'START-{p}' for p in POS] + ['START-ALL', 'ALL']:
            sc = ecs.paired_scores(d, base, pred, wk)
            if scope not in sc:
                continue
            per = []
            for y in years:
                s2 = ecs.paired_scores(d, base, pred, wk & (d['year'] == y).to_numpy())
                if scope in s2:
                    per.append(s2[scope]['rmse_v'] - s2[scope]['rmse_b'])
            x = sc[scope]
            lo, hi = ecs.boot_ci(x['weeks'])
            rows.append({'week': week, 'scope': scope, 'n': x['n'], 'RMSE shipped': round(x['rmse_b'], 3),
                         'dRMSE': round(x['rmse_v'] - x['rmse_b'], 3), 'CI': f"[{lo:+.3f},{hi:+.3f}]",
                         'bias': f"{x['bias_b']:+.2f}->{x['bias_v']:+.2f}",
                         'pairwise': f"{x['pw_b']:.3f}->{x['pw_v']:.3f}" if np.isfinite(x['pw_b']) else '',
                         'dRMSE by season': ' '.join(f"{v:+.3f}" for v in per),
                         'better in': f"{sum(v < 0 for v in per)}/{len(per)}"})
    pd.set_option('display.width', 250)
    print(pd.DataFrame(rows).to_string(index=False))
    m0 = ecs.top_n_mask(d, base) & d['played'].to_numpy()
    srows = []
    for week, positions in CANDIDATE_SCOPE.items():
        for pos in positions:
            m = m0 & (d['week'] == week).to_numpy() & (d['pos'] == pos).to_numpy()
            for s in CANDIDATE_STATS[pos]:
                a = d[f'a_{s}'].to_numpy()[m]
                p0 = d[f'p_{s}'].to_numpy()[m]
                srows.append({'week': week, 'pos': pos, 'stat': s, 'n': int(m.sum()), 'actual': round(float(a.mean()), 2),
                              'shipped': round(float(p0.mean()), 2), 'calibrated': round(float(line[s][m].mean()), 2),
                              'RMSE shipped': round(float(np.sqrt(((p0 - a) ** 2).mean())), 3),
                              'RMSE cal': round(float(np.sqrt(((line[s][m] - a) ** 2).mean())), 3)})
    print('\nheld-out stat line, START pool, played:')
    print(pd.DataFrame(srows).to_string(index=False))
    live = d['live'].to_numpy()
    for week, positions in CANDIDATE_SCOPE.items():
        m = live & (d['week'] == week).to_numpy() & d['pos'].isin(positions).to_numpy()
        bad = int((line['receptions'][m] > line['targets'][m] + 1e-9).sum())
        print(f"week {week}: rows with receptions > targets after calibration: {bad} of {int(m.sum())}")
    table = fit_candidate(d, np.ones(len(d), dtype=bool))
    print('\nOPENING_WEEK_STAT_CALIBRATION = {   # fit on every season (week 1: 2022-2025, week 2: 2021-2025)')
    for week, by_pos in table.items():
        print(f"    {week}: {{")
        for pos, lines in by_pos.items():
            body = ', '.join(f"'{s}': ({sl:.3f}, {ic:.3f})" for s, (sl, ic) in lines.items())
            print(f"        '{pos}': {{{body}}},")
        print('    },')
    print('}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-path', default=os.path.join('.sweeps', 'calibration_lab_v7.csv'))
    ap.add_argument('--week1-path', default='')
    ap.add_argument('--candidate-only', action='store_true')
    a = ap.parse_args()
    d = ecs.load_dump(a.dump_path, a.week1_path)
    if not a.candidate_only:
        for week in (1, 2):
            run_week(d, week)
    run_candidate(d)


if __name__ == '__main__':
    main()
