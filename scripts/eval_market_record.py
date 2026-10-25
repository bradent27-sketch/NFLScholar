"""
The live record against the market (2026-10-24; its leans and movement read the implied mean as the line - superseded by
scripts/eval_posted_line_record.py, 2026-10-25. User: "How accurate are we compared to the market ... are our leans sharp compared to
the market ... does the market shift towards our lines from opening to close"), from the saved pre-kickoff boards (data/ledger,
data/ledger_scorecard.py's rules: the last board before each player's own kickoff, paired pools).

  1. fantasy points: model vs market vs FantasyPros, RMSE and pairwise ranking, start-worthy and all, pooled over the live weeks;
  2. stats: model vs the market's line per stat;
  3. leans: the model's side of each market line (over when the model is higher) and how often the result landed there - with the
     model's MEAN and with its MEDIAN (data/weekly_median.py; a line is priced at the median), by stat and by the size of the gap;
  4. movement: for each player-stat line seen on two boards of the same week, the earliest board's line (the "open" here - the
     earliest board built that week, not the book's true opening number) against the last pre-kickoff board's (the "close"), and
     whether the line moved toward the model's number on the earliest board. Only lines that moved count; a market snapshot that
     was not re-pulled between builds carries the same number and shows no movement.

    python scripts/eval_market_record.py [--year 2026] [--out .sweeps/eval_market_record.txt]
"""
import argparse
import glob
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import ledger_scorecard as sc  # noqa: E402
from data.loaders import load_schedule  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402
from data.weekly_median import stat_median  # noqa: E402

LINE_STATS = {'receiving_yards': 'receiving_yards', 'rushing_yards': 'rushing_yards', 'passing_yards': 'passing_yards',
              'receptions': 'receptions', 'targets': 'targets', 'rushing_attempts': 'carries', 'passing_attempts': 'attempts',
              'passing_tds': 'passing_tds'}
GAPS = {'receiving_yards': (0, 5, 10), 'rushing_yards': (0, 5, 10), 'passing_yards': (0, 10, 20), 'receptions': (0, 0.5, 1.0),
        'targets': (0, 0.5, 1.0), 'rushing_attempts': (0, 1, 2), 'passing_attempts': (0, 1, 2), 'passing_tds': (0, 0.1, 0.2)}


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan)
    p, z = k / n, 1.96
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (mid - half, mid + half)


def fmt_table(df):
    return df.to_string(index=False, float_format=lambda v: f'{v:.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=2026)
    ap.add_argument('--out', default='.sweeps/eval_market_record.txt')
    a = ap.parse_args()
    stats_df, _team_col, name_col, _ = load_and_merge_data(a.year, 'Full PPR')
    schedule = load_schedule(a.year)
    frame, info = sc.load_scored_ledger(a.year, stats_df, name_col, schedule)
    live = frame[frame['pre_kickoff']].copy()
    weeks = sorted(live['week'].unique())
    out = [f"{a.year} live record: weeks {weeks} (pre-kickoff boards), scoring {info['scoring']}, {len(live)} player-weeks", '']

    # 1. fantasy points
    out.append('1. FANTASY POINTS, paired pools (lower RMSE / higher pairwise is better)')
    for bench in ('Market', 'FantasyPros'):
        for startable in (True, False):
            t = sc.weekly_comparison(frame, bench, startable=startable)
            if t.empty:
                continue
            allrow = t[t['week'] == 'All'].iloc[0]
            wk = t[t['week'] != 'All']
            better = int((wk['model_rmse'] < wk['bench_rmse']).sum())
            out.append(f"   vs {bench:11} {'start-worthy' if startable else 'all players':12} n={int(allrow['n']):5d}  RMSE model "
                       f"{allrow['model_rmse']:.3f} vs {allrow['bench_rmse']:.3f}  bias {allrow['model_bias']:+.2f} vs "
                       f"{allrow['bench_bias']:+.2f}  pairwise {allrow['model_pairwise']:.3f} vs {allrow['bench_pairwise']:.3f}  "
                       f"weeks model better {better} of {len(wk)}")
    out.append('')
    out.append('   when the model and the market disagree by 2+ / 4+ points: how often the model ended closer')
    d = sc.disagreement_table(frame, positions=(None,))
    out.append(fmt_table(d[['Gap', 'Side', 'n', 'Model closer', 'Model avg', 'Market avg', 'Actual avg']]))
    out.append('')

    # 2. stats
    out.append("2. STAT LINES vs the market's number (paired, played games)")
    s = sc.stat_comparison(frame, 'Market')
    out.append(fmt_table(s))
    out.append('')

    # 3. leans, mean and median
    out.append("3. LEANS: the model's side of the market line; hit = the result landed on that side (pushes and exact ties out)")
    rows = []
    for stat, mkt in LINE_STATS.items():
        mcol, acol = f'Mkt {mkt}', f'actual_{stat}'
        if stat not in live.columns or mcol not in live.columns or acol not in live.columns:
            continue
        mean = pd.to_numeric(live[stat], errors='coerce')
        line = pd.to_numeric(live[mcol], errors='coerce')
        act = pd.to_numeric(live[acol], errors='coerce')
        med = pd.Series([stat_median(p, stat, m) if np.isfinite(m) else np.nan for p, m in zip(live['Pos'], mean)],
                        index=live.index)
        for label, model in (('mean', mean), ('median', med)):
            for gap in GAPS[stat]:
                ok = model.notna() & line.notna() & act.notna() & ((model - line).abs() > max(gap, 1e-9)) & (act != line)
                n = int(ok.sum())
                if n < 15:
                    continue
                over = model[ok] > line[ok]
                hit = np.where(over, act[ok] > line[ok], act[ok] < line[ok])
                lo, hi = wilson(int(hit.sum()), n)
                rows.append({'stat': stat, 'model': label, 'min gap': gap, 'n': n, 'overs': int(over.sum()),
                             'hit': hit.mean(), 'CI low': lo, 'CI high': hi,
                             'over hit': hit[over.to_numpy()].mean() if over.any() else np.nan,
                             'under hit': hit[~over.to_numpy()].mean() if (~over).any() else np.nan})
    leans = pd.DataFrame(rows)
    out.append(fmt_table(leans))
    out.append('   (a lean has to hit about 52.4% to beat a standard -110 price)')
    out.append('')

    # 4. movement
    out.append('4. MOVEMENT: earliest board of the week ("open") to the last pre-kickoff board ("close"), lines that moved')
    paths = sorted(glob.glob(os.path.join(sc.LEDGER_DIR, f'{a.year}_wk*_*.parquet')))
    builds = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    builds['_key'] = clean_name_exact(builds['Player'])
    builds['build_ts'] = pd.to_datetime(builds['build_ts'], utc=True)
    kick = sc.kickoffs_utc(schedule)
    builds['_kick'] = [kick.get((int(w), str(t))) for w, t in zip(builds['week'], builds['Team'])]
    builds = builds[builds['_kick'].notna()]
    builds = builds[builds['build_ts'] <= pd.to_datetime(builds['_kick'], utc=True)]
    mrows = []
    for stat, mkt in LINE_STATS.items():
        mcol = f'Mkt {mkt}'
        if mcol not in builds.columns or stat not in builds.columns:
            continue
        b = builds[['week', '_key', 'Pos', 'build_ts', stat, mcol]].dropna(subset=[mcol])
        b = b.sort_values('build_ts')
        first = b.groupby(['week', '_key']).first()
        last = b.groupby(['week', '_key']).last()
        nb = b.groupby(['week', '_key']).size()
        j = first.join(last, lsuffix='_o', rsuffix='_c')
        j = j[(nb.reindex(j.index) >= 2)]
        move = j[mcol + '_c'] - j[mcol + '_o']
        moved = move.abs() > 1e-9
        model_o = pd.to_numeric(j[stat + '_o'], errors='coerce')
        med_o = pd.Series([stat_median(p, stat, m) if np.isfinite(m) else np.nan for p, m in zip(j['Pos_o'], model_o)],
                          index=j.index)
        for label, model in (('mean', model_o), ('median', med_o)):
            lean = model - j[mcol + '_o']
            ok = moved & lean.notna() & (lean.abs() > 1e-9)
            n = int(ok.sum())
            if n < 10:
                mrows.append({'stat': stat, 'model': label, 'lines seen twice': int(len(j)), 'moved': int(moved.sum()), 'n': n})
                continue
            toward = (np.sign(move[ok]) == np.sign(lean[ok]))
            lo, hi = wilson(int(toward.sum()), n)
            corr = float(np.corrcoef(lean[ok], move[ok])[0, 1]) if n > 2 else np.nan
            mrows.append({'stat': stat, 'model': label, 'lines seen twice': int(len(j)), 'moved': int(moved.sum()), 'n': n,
                          'moved toward model': toward.mean(), 'CI low': lo, 'CI high': hi, 'corr(lean, move)': corr,
                          'mean |move|': float(move[ok].abs().mean())})
    out.append(fmt_table(pd.DataFrame(mrows)))
    text = '\n'.join(out)
    print(text)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
