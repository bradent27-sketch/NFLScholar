"""
Market gap (model audit step 3, 2026-10-21): what do the market and FantasyPros know that the model does not?

Reads the live ledger through data/ledger_scorecard.py (each player's last pre-kickoff board, paired pools: the same players
for both sides). Diagnostic only - the market and FantasyPros stay benchmarks, never inputs.

  1. Points, START pool (union of both sides' top-N per week and position): RMSE by position.
  2. By stat: RMSE of each projected stat, model vs benchmark, same rows; and the swap test - the model's points with one
     stat replaced by the benchmark's number (weight x difference). The stats whose swap helps are where the benchmark knows
     more.
  3. Information slope: (actual - model) regressed on (benchmark - model). 0 = the disagreement carries no information,
     1 = the benchmark is right whenever they disagree. Weekly bootstrap CI.
  4. Where the edge sits: the same slope and both RMSEs by situation (injury status, games this season, teammates sidelined,
     size and direction of the disagreement).
  5. The largest disagreements of each week with the result, for a cause-by-cause read.

    python scripts/audit_market_gap.py [--year 2026] [--benchmark Market|FantasyPros] [--out .sweeps/audit_market_gap.txt]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data.ledger_scorecard as lsc  # noqa: E402
from data.loaders import load_schedule  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402

POSITIONS = ('QB', 'RB', 'WR', 'TE')
WEIGHTS = {'passing_yards': 0.04, 'passing_tds': 4.0, 'passing_interceptions': -2.0, 'rushing_yards': 0.1,
           'rushing_tds': 6.0, 'receptions': 1.0, 'receiving_yards': 0.1, 'receiving_tds': 6.0}
# ledger benchmark column per model stat (the market's carries/attempts columns are named differently)
BENCH_STAT = {'Market': {'rushing_attempts': 'Mkt carries', 'passing_attempts': 'Mkt attempts'},
              'FantasyPros': {}}


def bench_col(benchmark, stat):
    return BENCH_STAT[benchmark].get(stat, lsc.BENCHMARK_STAT_COLS[benchmark].format(stat))


def start_pool(frame, col):
    """Pre-kickoff rows paired on model and `col`, union of each side's top-N per week and position."""
    live = frame[frame['pre_kickoff']] if 'pre_kickoff' in frame.columns else frame
    out = [lsc._scope_rows(wk, None, True, col) for _, wk in live.groupby('week')]
    return pd.concat(out) if out else live.iloc[0:0]


def rmse(e):
    e = np.asarray(e, float)
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else float('nan')


def slope(sub, col, actual='actual_pts', model='model_pts', n_boot=1000, seed=5):
    x = (sub[col] - sub[model]).to_numpy(float)
    y = (sub[actual] - sub[model]).to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, wk = x[ok], y[ok], sub['week'].to_numpy()[ok]
    if len(x) < 20 or np.var(x) == 0:
        return np.nan, np.nan, np.nan
    b = float(np.sum(x * y) / np.sum(x * x))
    rng = np.random.default_rng(seed)
    weeks = np.unique(wk)
    idx = {w: np.where(wk == w)[0] for w in weeks}
    bs = []
    for _ in range(n_boot):
        ii = np.concatenate([idx[w] for w in rng.choice(weeks, len(weeks))])
        if np.sum(x[ii] ** 2) > 0:
            bs.append(np.sum(x[ii] * y[ii]) / np.sum(x[ii] ** 2))
    return b, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def section_points(pool, col, name):
    rows = []
    for pos in POSITIONS + (None,):
        s = pool if pos is None else pool[pool['Pos'] == pos]
        b, lo, hi = slope(s, col)
        rows.append({'pos': pos or 'ALL', 'n': len(s), 'model RMSE': rmse(s['model_pts'] - s['actual_pts']),
                     f'{name} RMSE': rmse(s[col] - s['actual_pts']), 'model bias': float((s['model_pts'] - s['actual_pts']).mean()),
                     f'{name} bias': float((s[col] - s['actual_pts']).mean()), 'info slope': b, 'slope CI': f'[{lo:+.2f}, {hi:+.2f}]'})
    return pd.DataFrame(rows).set_index('pos').round(3).to_string()


def section_stats(pool, benchmark):
    rows = []
    for pos in POSITIONS:
        s = pool[pool['Pos'] == pos]
        for stat in ('passing_attempts', 'passing_yards', 'passing_tds', 'passing_interceptions', 'rushing_attempts',
                     'rushing_yards', 'rushing_tds', 'targets', 'receptions', 'receiving_yards', 'receiving_tds'):
            bc, ac = bench_col(benchmark, stat), f'actual_{stat}'
            if stat not in s or bc not in s or ac not in s:
                continue
            m, b, a = (pd.to_numeric(s[c], errors='coerce') for c in (stat, bc, ac))
            ok = m.notna() & b.notna() & a.notna()
            if ok.sum() < 20 or a[ok].abs().sum() == 0:
                continue
            sw = np.nan
            if stat in WEIGHTS:
                p0 = s.loc[ok, 'model_pts'].to_numpy(float)
                p1 = p0 + WEIGHTS[stat] * (b[ok] - m[ok]).to_numpy(float)
                act = s.loc[ok, 'actual_pts'].to_numpy(float)
                sw = rmse(p1 - act) - rmse(p0 - act)
            sl = slope(pd.DataFrame({'week': s.loc[ok, 'week'], 'b': b[ok], 'm': m[ok], 'a': a[ok]}), 'b', 'a', 'm')
            rows.append({'pos': pos, 'stat': stat, 'n': int(ok.sum()), 'model RMSE': rmse(m[ok] - a[ok]),
                         'bench RMSE': rmse(b[ok] - a[ok]), 'model bias': float((m[ok] - a[ok]).mean()),
                         'bench bias': float((b[ok] - a[ok]).mean()), 'info slope': sl[0],
                         'slope CI': f'[{sl[1]:+.2f}, {sl[2]:+.2f}]', 'pts swap dRMSE': sw})
    return pd.DataFrame(rows).round(3).to_string(index=False)


def add_situations(frame):
    f = frame.copy()
    inj = f['Injury Status'].astype(str).str.lower() if 'Injury Status' in f else pd.Series('', index=f.index)
    f['seg_injury'] = np.select([inj.str.startswith('questionable'), inj.str.startswith('doubtful')],
                                ['Questionable', 'Doubtful'], 'none')
    g = pd.to_numeric(f.get('Games This Season'), errors='coerce')
    f['seg_games'] = np.select([g <= 1, g <= 3], ['0-1 games', '2-3 games'], '4+ games')
    av = pd.to_numeric(f.get('Availability'), errors='coerce').fillna(1.0)
    out = (av <= 0.05).astype(float)
    mp = pd.to_numeric(f['model_pts'], errors='coerce').fillna(0.0)
    # teammates at the same position the board has sidelined this week (a vacancy the model had to redistribute)
    f['_side'] = out
    side_pos = f.groupby(['week', 'Team', 'Pos'])['_side'].transform('sum') - out
    f['seg_teammate_out'] = np.where(side_pos > 0, 'same-position teammate out', 'none out')
    f['_mp'] = mp
    return f


def section_segments(pool, col, name):
    f = add_situations(pool)
    gap = f['model_pts'] - f[col]
    f['seg_gap'] = np.select([gap >= 3, gap <= -3], [f'model {">="}3 above', f'model {">="}3 below'], 'within 3')
    rows = []
    for seg in ('seg_injury', 'seg_games', 'seg_teammate_out', 'seg_gap'):
        for val, s in f.groupby(seg):
            for pos in (None, 'WR'):
                ss = s if pos is None else s[s['Pos'] == pos]
                if len(ss) < 15:
                    continue
                b, lo, hi = slope(ss, col)
                rows.append({'segment': seg[4:], 'value': val, 'pos': pos or 'ALL', 'n': len(ss),
                             'model RMSE': rmse(ss['model_pts'] - ss['actual_pts']), f'{name} RMSE': rmse(ss[col] - ss['actual_pts']),
                             'model bias': float((ss['model_pts'] - ss['actual_pts']).mean()),
                             f'{name} bias': float((ss[col] - ss['actual_pts']).mean()), 'info slope': b})
    return pd.DataFrame(rows).round(3).to_string(index=False)


def section_disagreements(pool, col, name, per_week=10):
    f = add_situations(pool)
    f['gap'] = f['model_pts'] - f[col]
    f['closer'] = np.where((f['model_pts'] - f['actual_pts']).abs() < (f[col] - f['actual_pts']).abs(), 'model', name)
    keep = ['week', 'Player', 'Pos', 'Team', 'Opponent', 'model_pts', col, 'actual_pts', 'gap', 'closer', 'seg_injury',
            'seg_games', 'seg_teammate_out']
    out = []
    for wk, s in f.groupby('week'):
        out.append(s.reindex(s['gap'].abs().sort_values(ascending=False).index).head(per_week)[keep])
    t = pd.concat(out)
    tally = t['closer'].value_counts().to_dict()
    return t.round(2).to_string(index=False) + f'\n   closer on these: {tally}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=2026)
    ap.add_argument('--benchmark', default='Market', choices=('Market', 'FantasyPros'))
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    pd.set_option('display.width', 240)
    pd.set_option('display.max_columns', 30)
    stats_df, _team_col, name_col, _ = load_and_merge_data(a.year, 'Full PPR')
    frame, info = lsc.load_scored_ledger(a.year, stats_df, name_col, load_schedule(a.year))
    col = lsc.SOURCES[a.benchmark]
    name = 'mkt' if a.benchmark == 'Market' else 'FP'
    pool = start_pool(frame, col)
    weeks = sorted(pool['week'].unique())
    lines = [f'{a.year} ledger: {info["files"]} files, scoring {info["scoring"]}; pre-kickoff START pool paired with '
             f'{a.benchmark}: {len(pool):,} player-weeks, weeks {weeks}', '',
             '1. Points (info slope: actual - model on benchmark - model; CI by week):', section_points(pool, col, name), '',
             '2. By stat (same START rows; swap = model points with this stat taken from the benchmark):',
             section_stats(pool, a.benchmark), '',
             '3./4. By situation:', section_segments(pool, col, name), '',
             f'5. Largest disagreements per week (model - {name}):', section_disagreements(pool, col, name)]
    text = '\n'.join(lines)
    print(text)
    out = a.out or f'.sweeps/audit_market_gap_{a.benchmark.lower()}.txt'
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
