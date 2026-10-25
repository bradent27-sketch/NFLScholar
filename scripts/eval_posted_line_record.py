"""
The live lean record against the POSTED lines (2026-10-25). scripts/eval_market_record.py read a board's 'Mkt <stat>' as the line,
but that column is the market's implied MEAN (data/market_devig.py: the vig removed and, for a count, a Poisson mean solved; a bare
DFS line is multiplied up), so its leans and movement were measured against a number nobody posted. This re-scores them against the
lines the books actually posted, from the raw pulls archived in external_data/props_archive/.

  1. leans: the model's MEDIAN (data/weekly_median.py; a line is priced against the middle game) against the posted line, from the
     sharpest book that posted one (Pinnacle, DraftKings, FanDuel, Underdog, PrizePicks; scorable lines only - no demon/goblin,
     shaded or partial-game lines), and whether the result landed on the model's side (pushes out). Two lines per lean: the line
     posted when the model's board was built (what could have been bet then) and the last line before kickoff (the close). The same
     lines the app leaves out are left out here (games_view: rushing/receiving TD markets, 0.5 count lines, yardage under 5).
  2. movement: for each player-stat one book posted at two or more times before kickoff, the first line against the last, and
     whether it moved toward the model's median on the week's first board.

The model's numbers are the ledger's pre-kickoff boards (data/ledger_scorecard.py: the last board before each player's kickoff).

    python scripts/eval_posted_line_record.py [--year 2026] [--out .sweeps/eval_posted_line_record.txt]
"""
import argparse
import glob
import json
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
from scripts.eval_market_record import GAPS, wilson  # noqa: E402
from ui.games_view import LINE_BOOKS, MIN_LEAN_LINE, _MEAN_STATS  # noqa: E402

ARCHIVE = os.path.join('external_data', 'props_archive')
# board stat -> the archive's market name
LINE_STATS = {'receiving_yards': 'receiving_yards', 'rushing_yards': 'rushing_yards', 'passing_yards': 'passing_yards',
              'receptions': 'receptions', 'targets': 'targets', 'rushing_attempts': 'carries', 'passing_attempts': 'attempts',
              'passing_completions': 'passing_completions', 'passing_tds': 'passing_tds', 'passing_interceptions': 'passing_interceptions'}


def load_archive(year):
    """Every scorable full-game line ever pulled for `year`: week, _key, market, book, line, p_over, fetched_at (UTC)."""
    frames = []
    for path in sorted(glob.glob(os.path.join(ARCHIVE, f'{year}_wk*.json'))):
        with open(path, encoding='utf-8') as handle:
            snap = json.load(handle)
        rows = pd.DataFrame(snap.get('rows') or [])
        if rows.empty:
            continue
        rows['week'] = int(snap['week'])
        rows['fetched_at'] = pd.Timestamp(snap['fetched_at'])
        frames.append(rows)
    arc = pd.concat(frames, ignore_index=True)
    arc = arc[arc['scorable'].fillna(False).astype(bool) & arc['period'].astype(str).eq('game')
              & arc['market'].isin(set(LINE_STATS.values()))].copy()
    arc['book'] = arc['provider'].astype(str).str.split(' (', regex=False).str[0]
    arc = arc[arc['book'].isin(LINE_BOOKS)]
    arc['_rank'] = arc['book'].map({b: i for i, b in enumerate(LINE_BOOKS)})
    arc['_key'] = clean_name_exact(arc['player'])
    arc['line'] = pd.to_numeric(arc['line'], errors='coerce')
    arc['fetched_at'] = pd.to_datetime(arc['fetched_at'], utc=True)
    return arc.dropna(subset=['line'])[['week', '_key', 'market', 'book', '_rank', 'line', 'p_over', 'fetched_at']]


def line_as_of(arc, when):
    """{(week, _key, market): line}: per book, its latest line pulled at or before `when` ((week, _key) -> Timestamp), then the
    sharpest book's."""
    a = arc.copy()
    a['_when'] = [when.get((w, k)) for w, k in zip(a['week'], a['_key'])]
    a = a[a['_when'].notna()]
    a = a[a['fetched_at'] <= pd.to_datetime(a['_when'], utc=True)]
    a = a.sort_values('fetched_at').groupby(['week', '_key', 'market', 'book']).last().reset_index()
    a = a.sort_values('_rank').groupby(['week', '_key', 'market']).first()
    return a['line'].to_dict()


def leanable(stat, line):
    return np.isfinite(line) and line >= MIN_LEAN_LINE.get(stat, 1.0 if stat in _MEAN_STATS else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=2026)
    ap.add_argument('--out', default='.sweeps/eval_posted_line_record.txt')
    a = ap.parse_args()
    stats_df, _team_col, name_col, _ = load_and_merge_data(a.year, 'Full PPR')
    schedule = load_schedule(a.year)
    frame, _info = sc.load_scored_ledger(a.year, stats_df, name_col, schedule)
    live = frame[frame['pre_kickoff']].copy()
    live['build_ts'] = pd.to_datetime(live['build_ts'], utc=True)
    arc = load_archive(a.year)
    out = [f"{a.year} lean record against the POSTED lines (external_data/props_archive), weeks {sorted(live['week'].unique())}, "
           f"{len(live)} player-weeks on pre-kickoff boards, {len(arc):,} archived lines", '']

    at_build = line_as_of(arc, {(w, k): t for w, k, t in zip(live['week'], live['_key'], live['build_ts'])})
    at_close = line_as_of(arc, {(w, k): pd.Timestamp(t) for w, k, t in zip(live['week'], live['_key'], live['kickoff'])
                                if pd.notna(t)})

    out.append("1. LEANS: the model's median against the posted line (sharpest book); hit = the result landed on that side")
    rows = []
    for stat, market in LINE_STATS.items():
        acol = f'actual_{stat}'
        if stat not in live.columns or acol not in live.columns:
            continue
        mean = pd.to_numeric(live[stat], errors='coerce')
        med = pd.Series([stat_median(p, stat, m) if np.isfinite(m) else np.nan for p, m in zip(live['Pos'], mean)], index=live.index)
        act = pd.to_numeric(live[acol], errors='coerce')
        for label, lines in (('at board', at_build), ('at close', at_close)):
            line = pd.Series([lines.get((w, k, market), np.nan) for w, k in zip(live['week'], live['_key'])], index=live.index)
            ok_line = pd.Series([leanable(stat, v) for v in line], index=live.index)
            for gap in GAPS.get(stat, (0,)):
                ok = (ok_line & med.notna() & act.notna() & (mean > 0) & ((med - line).abs() > max(gap, 1e-9)) & (act != line))
                n = int(ok.sum())
                if n < 15:
                    continue
                over = med[ok] > line[ok]
                hit = np.where(over, act[ok] > line[ok], act[ok] < line[ok])
                lo, hi = wilson(int(hit.sum()), n)
                rows.append({'stat': stat, 'line': label, 'min gap': gap, 'n': n, 'overs': int(over.sum()), 'hit': hit.mean(),
                             'CI low': lo, 'CI high': hi,
                             'over hit': hit[over.to_numpy()].mean() if over.any() else np.nan,
                             'under hit': hit[~over.to_numpy()].mean() if (~over).any() else np.nan})
    out.append(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    out.append('   (a lean has to hit about 52.4% to beat a standard -110 price)')
    out.append('')

    # 2. movement: per (week, player, market, book) with two or more pulls before kickoff, first vs last; the sharpest such book
    out.append("2. MOVEMENT: one book's first posted line against its last before kickoff, lines that moved; toward = toward the "
               "model's median on the week's first board")
    kick = {(w, k): pd.Timestamp(t) for w, k, t in zip(live['week'], live['_key'], live['kickoff']) if pd.notna(t)}
    m = arc.copy()
    m['_kick'] = [kick.get((w, k)) for w, k in zip(m['week'], m['_key'])]
    m = m[m['_kick'].notna()]
    m = m[m['fetched_at'] <= pd.to_datetime(m['_kick'], utc=True)].sort_values('fetched_at')
    g = m.groupby(['week', '_key', 'market', 'book'])
    path = pd.DataFrame({'open': g['line'].first(), 'close': g['line'].last(), 'pulls': g['fetched_at'].nunique(),
                         'opened': g['fetched_at'].first(), '_rank': g['_rank'].first()}).reset_index()
    path = path[path['pulls'] >= 2].sort_values('_rank').groupby(['week', '_key', 'market']).first().reset_index()
    paths = sorted(glob.glob(os.path.join(sc.LEDGER_DIR, f'{a.year}_wk*_*.parquet')))
    builds = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    builds['_key'] = clean_name_exact(builds['Player'])
    builds['build_ts'] = pd.to_datetime(builds['build_ts'], utc=True)
    first = builds.sort_values('build_ts').groupby(['week', '_key']).first()
    mrows = []
    for stat, market in LINE_STATS.items():
        p = path[path['market'] == market].copy()
        if p.empty or stat not in first.columns:
            continue
        idx = pd.MultiIndex.from_arrays([p['week'], p['_key']])
        mean = pd.to_numeric(first[stat].reindex(idx), errors='coerce').to_numpy()
        pos = first['Pos'].reindex(idx).to_numpy()
        p['model'] = [stat_median(ps, stat, mu) if isinstance(ps, str) and np.isfinite(mu) and mu > 0 else np.nan
                      for ps, mu in zip(pos, mean)]
        p = p[[leanable(stat, v) for v in p['open']]]
        p['move'] = p['close'] - p['open']
        p['lean'] = p['model'] - p['open']
        moved = p['move'].abs() > 1e-9
        ok = moved & p['lean'].notna() & (p['lean'].abs() > 1e-9)
        n = int(ok.sum())
        row = {'stat': stat, 'seen twice': len(p), 'moved': int(moved.sum()), 'n': n}
        if n >= 10:
            toward = np.sign(p.loc[ok, 'move']) == np.sign(p.loc[ok, 'lean'])
            lo, hi = wilson(int(toward.sum()), n)
            row.update({'moved toward model': toward.mean(), 'CI low': lo, 'CI high': hi,
                        'corr(lean, move)': float(np.corrcoef(p.loc[ok, 'lean'], p.loc[ok, 'move'])[0, 1]),
                        'mean |move|': float(p.loc[ok, 'move'].abs().mean())})
        mrows.append(row)
    out.append(pd.DataFrame(mrows).to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    text = '\n'.join(out)
    print(text)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
