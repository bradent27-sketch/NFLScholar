"""
Does the model over-react to a team's own tempo?  (2026-10-06, from the pace question.)

A team's plays per game regress hard toward the league mean (2016-2025: this game's plays move only
~0.37 per play of the team's own to-date deviation; scripts/diag_pace_signal.py), but the model's per-game
volume rates carry the team's own tempo in full. If that is a real bias, the actual-minus-projected team
volume should fall as the team's to-date tempo rises, most of all early in the season.

Reads a calibration dump built with `fit_seasonal_calibration.py --mode dump --all-rows` AFTER 2026-10-05
(it carries team, opp, p_/a_ passing_attempts, rushing_attempts, targets; a live row with no box score is
0). Aggregates to team-games (QB passing attempts, all rushing attempts, WR/TE+RB targets), joins each
team's own to-date plays per game, and regresses (actual - projected) on the deviation from the league mean.

    python scripts/analyze_team_tempo.py --dump-path .sweeps/seasonal_calibration_allrows.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.diag_pace_signal import team_weeks  # noqa: E402

CHANNELS = (('pass attempts (QB)', 'passing_attempts', 'QB'), ('rush attempts (all)', 'rushing_attempts', None),
            ('targets (all)', 'targets', None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-path', default='.sweeps/seasonal_calibration_allrows.csv')
    a = ap.parse_args()
    d = pd.read_csv(a.dump_path)
    need = {'team', 'p_passing_attempts', 'a_passing_attempts', 'p_rushing_attempts', 'a_rushing_attempts'}
    if not need.issubset(d.columns):
        raise SystemExit(f"dump lacks {sorted(need - set(d.columns))}: rebuild it with the current fit_seasonal_calibration.py")
    d['team'] = d['team'].astype(str).str.upper()
    years = sorted(d['year'].unique())
    tw = team_weeks(years)
    feat = {}
    for year, g in tw.groupby('year'):
        for week in sorted(g['week'].unique()):
            prior = g[g['week'] < week]
            if prior.empty:
                continue
            off = prior.groupby('team')['plays'].agg(['mean', 'count'])
            m = off['mean'].mean()
            for t, r in off.iterrows():
                feat[(year, week, t)] = (r['mean'] - m, r['count'])
    pd.set_option('display.width', 200)
    for label, stat, pos in CHANNELS:
        s = d if pos is None else d[d['pos'] == pos]
        p, act = f'p_{stat}', f'a_{stat}'
        s = s.assign(**{p: pd.to_numeric(s[p], errors='coerce').fillna(0.0), act: pd.to_numeric(s[act], errors='coerce').fillna(0.0)})
        g = s.groupby(['year', 'week', 'team']).agg(p=(p, 'sum'), a=(act, 'sum')).reset_index()
        g['dev'] = [feat.get((r.year, r.week, r.team), (np.nan, np.nan))[0] for r in g.itertuples()]
        g['G'] = [feat.get((r.year, r.week, r.team), (np.nan, np.nan))[1] for r in g.itertuples()]
        g = g.dropna(subset=['dev'])
        g = g[g['p'] > 0]
        print(f"\n=== {label}: team-games {len(g)}, projected {g['p'].mean():.2f} vs actual {g['a'].mean():.2f} per game ===")
        rows = []
        for name, (lo, hi) in {'all weeks': (2, 17), 'games 2-5': (2, 5), '6-10': (6, 10), '11-17': (11, 17)}.items():
            x = g[(g['G'] >= lo) & (g['G'] <= hi)]
            if len(x) < 40:
                continue
            X = np.column_stack([np.ones(len(x)), x['dev']])
            y = (x['a'] - x['p']).to_numpy()
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ beta
            se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * res.var(ddof=2)))
            rows.append({'window': name, 'n': len(x), 'slope (act-proj per play of tempo dev)': round(beta[1], 3),
                         'se': round(se[1], 3), 't': round(beta[1] / se[1], 1),
                         'implied by "tempo fully carried, 0.37 persists"': round(-0.63 * x['p'].mean() / 61.4, 3)})
        print(pd.DataFrame(rows).to_string(index=False))
        g['bin'] = pd.qcut(g['dev'], 5, labels=['slowest', 'slow', 'mid', 'fast', 'fastest'])
        t = g.groupby('bin', observed=True).agg(n=('p', 'size'), tempo_dev=('dev', 'mean'), proj=('p', 'mean'), act=('a', 'mean'))
        t['act - proj'] = t['act'] - t['proj']
        print(t.round(2).to_string())


if __name__ == '__main__':
    main()
