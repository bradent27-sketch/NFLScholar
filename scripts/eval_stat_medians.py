"""
A median beside every projection (2026-10-23, user: "the median version toggle that the cfb app has ... by general position, value,
but also specific to that player's history ... some are more consistently high and some are more boom or bust ... average for
fantasy, median as a floor reference and for O/U bets").

The board's numbers are means. A median of the same outcome is the mean times a ratio:

    median = mean x r(position, stat, mean)  x  player shape factor

  r          the population curve, smooth in log(mean): fit by weighted least squares to the median of what players at each
             projected level actually did over their mean projection (equal-count bins), by position and stat. Touchdowns use the
             Poisson continuity-corrected median instead (TD_POISSON), the bins being too thin for a curve;
  factor     the player's own history: his own median over his own mean, relative to r at his own mean, shrunk toward 1 by
             n / (n + K) games (a boom-or-bust receiver below 1, a steady one near or above).

Counts are jittered by U(-0.5, 0.5) before a median is taken (the continuity correction that makes a median comparable with an x.5
prop line: a model median above 4.5 receptions <=> P(5 or more) above one half).

Sample: played games where the board projected a full role (Availability >= 0.99) and at least MIN_PROJECTION of the stat - below
it a box-score row exists mainly when the fringe player DID record something, which bends the curve. Scored leave-one-season-out
over 2022-2025 (fit on three seasons, score the fourth): MAE (the loss a median minimizes) and the over-rate (share of results above
the median; 0.500 is right).

    python scripts/eval_stat_medians.py [--out .sweeps/eval_stat_medians.txt] [--emit]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.eval_calibration_schemes import load_dump  # noqa: E402

BOARDS = '.sweeps/harness_boards/leanness_families_2022-2025_wk3-17.parquet'
ACTUALS = '.sweeps/qb_style_capture.parquet.board.csv'
YEARS = (2022, 2023, 2024, 2025)
PAIRS = {'QB': ('passing_attempts', 'passing_yards', 'passing_tds', 'rushing_attempts', 'rushing_yards', 'points'),
         'RB': ('rushing_attempts', 'rushing_yards', 'rushing_tds', 'targets', 'receptions', 'receiving_yards', 'points'),
         'WR': ('targets', 'receptions', 'receiving_yards', 'receiving_tds', 'points'),
         'TE': ('targets', 'receptions', 'receiving_yards', 'receiving_tds', 'points')}
MIN_PROJECTION = {'passing_attempts': 15.0, 'passing_yards': 100.0, 'rushing_attempts': 1.5, 'rushing_yards': 5.0,
                  'targets': 1.5, 'receptions': 1.0, 'receiving_yards': 8.0, 'points': 3.0,
                  'passing_tds': 0.0, 'rushing_tds': 0.0, 'receiving_tds': 0.0}
TD_POISSON = ('passing_tds', 'rushing_tds', 'receiving_tds')
WEEKLY_COLS = {'passing_attempts': 'attempts', 'rushing_attempts': 'carries', 'points': 'fantasy_points_ppr'}
K_GRID = (20.0, 40.0, 80.0)
N_BINS = 15
RATIO_CLIP = (0.35, 1.25)
RNG = np.random.default_rng(7)


def jitter(values, stat):
    values = np.asarray(values, dtype=float)
    return values if stat == 'points' else values + RNG.uniform(-0.5, 0.5, len(values))


def poisson_median(m):
    """The continuity-corrected median of Poisson(m): the point where a count spread evenly over [k - 0.5, k + 0.5] reaches one half."""
    from scipy.stats import poisson
    m = np.asarray(m, dtype=float)
    out = np.zeros_like(m)
    for i, lam in enumerate(m):
        if not np.isfinite(lam) or lam <= 0:
            continue
        k = int(poisson.ppf(0.5, lam))
        below = poisson.cdf(k - 1, lam) if k > 0 else 0.0
        out[i] = k - 0.5 + (0.5 - below) / poisson.pmf(k, lam)
    return out


def load_rows():
    b = pd.read_parquet(BOARDS)
    d = load_dump(ACTUALS)
    keep = ['player', 'team', 'year', 'week', 'actual', 'played'] + [c for c in d.columns if c.startswith('a_')]
    m = b.merge(d[keep], left_on=['Player', 'Team', 'year', 'week'], right_on=['player', 'team', 'year', 'week'], how='inner')
    m = m[m['played'].astype(bool) & (pd.to_numeric(m['Availability'], errors='coerce').fillna(1.0) >= 0.99)]
    rows = []
    for pos, stats in PAIRS.items():
        g = m[m['Pos'] == pos]
        for stat in stats:
            proj = g['Model Proj Pts'] if stat == 'points' else g[stat]
            act = g['actual'] if stat == 'points' else g['a_' + stat]
            part = pd.DataFrame({'player': g['Player'].to_numpy(), 'pos': pos, 'stat': stat, 'year': g['year'].to_numpy(),
                                 'week': g['week'].to_numpy(), 'm': pd.to_numeric(proj, errors='coerce').to_numpy(),
                                 'a': pd.to_numeric(act, errors='coerce').to_numpy()})
            rows.append(part[(part['m'] > max(MIN_PROJECTION[stat], 0.05)) & part['a'].notna()])
    out = pd.concat(rows, ignore_index=True)
    out['aj'] = np.nan
    for stat, idx in out.groupby('stat').groups.items():
        out.loc[idx, 'aj'] = jitter(out.loc[idx, 'a'], stat)
    return out


def fit_curve(part):
    """(a, b): ratio = a + b * log(mean), weighted least squares on N_BINS equal-count bins of median(actual) / mean(projection)."""
    part = part.sort_values('m')
    xs, ys, ws = [], [], []
    for idx in np.array_split(np.arange(len(part)), N_BINS):
        if len(idx) < 25:
            continue
        sub = part.iloc[idx]
        mean_m = float(sub['m'].mean())
        xs.append(np.log(mean_m))
        ys.append(float(np.median(sub['aj'])) / mean_m)
        ws.append(len(idx))
    if len(xs) < 3:
        return (1.0, 0.0)
    w = np.sqrt(np.array(ws, dtype=float))
    design = np.column_stack([np.ones(len(xs)), xs]) * w[:, None]
    a, b = np.linalg.lstsq(design, np.array(ys) * w, rcond=None)[0]
    return (float(a), float(b))


def curve_ratio(coef, m):
    a, b = coef
    m = np.maximum(np.asarray(m, dtype=float), 1e-6)
    return np.clip(a + b * np.log(m), *RATIO_CLIP)


def population_median(stat, coef, m):
    return poisson_median(m) if stat in TD_POISSON else np.asarray(m, dtype=float) * curve_ratio(coef, m)


def weekly_history():
    frames = []
    for year in range(2021, 2026):
        path = f'stats_player_week_{year}.csv'
        if os.path.exists(path):
            w = pd.read_csv(path, low_memory=False)
            frames.append(w[w['season_type'].astype(str) == 'REG'] if 'season_type' in w.columns else w)
    w = pd.concat(frames, ignore_index=True)
    extra = {stat: pd.to_numeric(w[col], errors='coerce') for stat, col in WEEKLY_COLS.items()}
    return pd.concat([w.drop(columns=[c for c in extra if c in w.columns]), pd.DataFrame(extra)], axis=1)


def own_shape(rows, history, curves):
    """Per row: n (own games before the week: this season so far and all of last season) and the raw factor
    (own median / own mean) / (population median at his own mean / his own mean)."""
    hist = history.rename(columns={'player_display_name': 'player', 'season': 'hyear', 'week': 'hweek'})
    hist = {p: g for p, g in hist.groupby('player')}
    rows = rows.reset_index(drop=True)
    n_own, f_raw = np.zeros(len(rows)), np.ones(len(rows))
    for (player, stat, pos), idx in rows.groupby(['player', 'stat', 'pos']).groups.items():
        h = hist.get(player)
        if h is None or stat not in h.columns:
            continue
        coef = curves[(pos, stat)]
        for i in idx:
            year, week = rows.at[i, 'year'], rows.at[i, 'week']
            games = h.loc[((h['hyear'] == year) & (h['hweek'] < week)) | (h['hyear'] == year - 1), stat].dropna().to_numpy()
            if len(games) < 3 or games.mean() <= 0.05:
                continue
            own_mean = games.mean()
            pop = float(population_median(stat, coef, [own_mean])[0])
            if pop <= 0:
                continue
            n_own[i] = len(games)
            f_raw[i] = np.median(jitter(games, stat)) / pop
    rows['n_own'], rows['f_raw'] = n_own, np.clip(f_raw, 0.3, 2.0)
    return rows


def score(part, median):
    err = part['aj'].to_numpy() - median
    return float(np.mean(np.abs(err))), float(np.mean(err > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='.sweeps/eval_stat_medians.txt')
    a = ap.parse_args()
    rows = load_rows()
    history = weekly_history()
    scored = []
    for test_year in YEARS:
        fit = rows[rows['year'] != test_year]
        curves = {key: fit_curve(g) for key, g in fit.groupby(['pos', 'stat'])}
        test = own_shape(rows[rows['year'] == test_year].copy(), history, curves)
        test['pop'] = np.nan
        for (pos, stat), idx in test.groupby(['pos', 'stat']).groups.items():
            test.loc[idx, 'pop'] = population_median(stat, curves[(pos, stat)], test.loc[idx, 'm'])
        scored.append(test)
    test = pd.concat(scored, ignore_index=True)
    lines = [f'leave-one-season-out over {YEARS}; MAE (lower is better) and over-rate (share above the median; 0.500 is right)', '',
             f"{'pos':4} {'stat':17} {'n':>6}  {'mean as median':>15}  {'population':>15}  "
             + '  '.join(f'{"+player K=" + str(int(k)):>15}' for k in K_GRID)]
    for (pos, stat), g in test.groupby(['pos', 'stat']):
        cells = [score(g, g['m'].to_numpy()), score(g, g['pop'].to_numpy())]
        for k in K_GRID:
            n = g['n_own'].to_numpy()
            cells.append(score(g, g['pop'].to_numpy() * (1.0 + (g['f_raw'].to_numpy() - 1.0) * n / (n + k))))
        lines.append(f"{pos:4} {stat:17} {len(g):6d}  " + '  '.join(f'{mae:7.3f} {over:5.3f}  ' for mae, over in cells))
    curves = {key: fit_curve(g) for key, g in rows.groupby(['pos', 'stat'])}
    lines += ['', 'Population curves on all four seasons: ratio = a + b * log(mean) (touchdowns: Poisson median)']
    for (pos, stat), (ca, cb) in sorted(curves.items()):
        lines.append(f"    ('{pos}', '{stat}'): ({ca:.4f}, {cb:.4f}),")
    text = '\n'.join(lines)
    print(text)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
