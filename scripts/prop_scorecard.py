"""
The weekly prop record (2026-10-28), plus the prop sheet for an upcoming week. data/prop_probability.py's P(over) on the live boards
is scored against the books' de-vigged price and the result.

Record: each player-stat on a pre-kickoff ledger board (data/ledger_scorecard.py) with a priced line, taken from the sharpest book
whose pulls carry a de-vigged p_over (Pinnacle, then DraftKings). Lines come from external_data/props_archive at two times: the
line posted when the board was built (what could have been bet then) and the last line before kickoff (the close). For each:
  - Brier of the model's P(over) and of the book's, the over rate, and calibration (the model's P(over) in bands against the hit
    rate);
  - the shrink toward 1/2 that minimises the model's Brier (p' = 1/2 + s (p - 1/2); 1 = trust as is, 0 = no information);
  - the record betting the model's side where its P(over) differs from the book's price by 3%, 5%, 8% or 12%: hit rate, 95% CI,
    and ROI at -110 and at the book's fair price.
Yardage lines under 5 are left out, as the app does. Pushes are out.

Sheet (--sheet WEEK): the latest ledger board for WEEK against the latest priced lines: the recalibrated mean, P(over), the book's
price, the gap, and the P(over) shrunk by the record's closing-line factor. A gap is a lean to track until the record says otherwise.

    python scripts/prop_scorecard.py [--year 2026] [--out .sweeps/prop_scorecard.txt]
    python scripts/prop_scorecard.py --sheet 9 [--year 2026]
    python scripts/prop_scorecard.py --install-engines [--engines-from .sweeps/deep_audit/prop_models_v2]
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
from data import prop_probability as pp  # noqa: E402
from data.loaders import load_schedule  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402
from scripts.eval_market_record import wilson  # noqa: E402
from scripts.eval_posted_line_record import load_archive  # noqa: E402
from ui.games_view import MIN_LEAN_LINE  # noqa: E402

EDGES = (0.03, 0.05, 0.08, 0.12)
# the closing-line shrink measured by the deep audit on 2026 weeks 3-7 (today's model rebuilt), until the record has its own
DEFAULT_SHRINK = 0.34


def install_engines(src):
    engines = {}
    for path in sorted(glob.glob(os.path.join(src, '*.json'))):
        with open(path, encoding='utf-8') as fh:
            eng = json.load(fh)
        engines[eng['stat']] = {k: eng[k] for k in ('spec', 'seasons', 'recal', 'M1', 'mean') if k in eng}
    with open(pp.ENGINE_PATH, 'w', encoding='utf-8') as fh:
        json.dump(engines, fh, separators=(',', ':'))
    print(f'{len(engines)} engines -> {pp.ENGINE_PATH}: {", ".join(engines)}')


def priced_lines(arc, when):
    """{(week, _key, market): (line, p_over, book)}: per book its latest priced line pulled at or before `when` ((week, _key) ->
    Timestamp), then the sharpest book's."""
    a = arc[arc['p_over'].notna()].copy()
    a['_when'] = [when.get((w, k)) for w, k in zip(a['week'], a['_key'])]
    a = a[a['_when'].notna()]
    a = a[a['fetched_at'] <= pd.to_datetime(a['_when'], utc=True)]
    a = a.sort_values('fetched_at').groupby(['week', '_key', 'market', 'book']).last().reset_index()
    a = a.sort_values('_rank').groupby(['week', '_key', 'market']).first()
    return {k: (float(r['line']), float(r['p_over']), r['book']) for k, r in a.iterrows()}


def histories(stats_df, name_col, weeks, prior, prior_name_col):
    return {int(w): pp.player_history(stats_df, name_col, int(w), prior, prior_name_col) for w in weeks}


def scored_rows(live, lines, hist, label):
    """One row per (player-week, stat) with a priced line: the model's P(over), the book's, the result."""
    frames = []
    for stat, market in pp.PROP_MARKETS.items():
        acol = f'actual_{stat}'
        if stat not in live.columns or acol not in live.columns:
            continue
        pre = [f'_pre_recal_{stat}'] if f'_pre_recal_{stat}' in live.columns else []   # v2_stat_recal_own_history boards
        f = live[['week', '_key', 'Player', 'Pos', stat, acol] + pre].copy()
        got = [lines.get((w, k, market)) for w, k in zip(f['week'], f['_key'])]
        f['line'] = [g[0] if g else np.nan for g in got]
        f['p_book'] = [g[1] if g else np.nan for g in got]
        f['book'] = [g[2] if g else None for g in got]
        f = f[f['line'].notna() & pd.to_numeric(f[stat], errors='coerce').gt(0) & pd.to_numeric(f[acol], errors='coerce').notna()]
        f = f[f['line'] >= MIN_LEAN_LINE.get(stat, 0.0)]
        f = f[pd.to_numeric(f[acol], errors='coerce') != f['line']]
        parts = []
        for w, fw in f.groupby('week'):
            parts.append(pp.prop_probabilities(fw, stat, history=hist.get(int(w))))
        if not parts:
            continue
        f = pd.concat(parts)
        f['actual'] = pd.to_numeric(f[acol], errors='coerce')
        f['over'] = (f['actual'] > f['line']).astype(float)
        f['stat'] = stat
        f['timing'] = label
        frames.append(f.rename(columns={stat: 'projection'}).drop(columns=[acol]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def best_shrink(p, o):
    grid = np.linspace(0, 1.2, 61)
    return float(grid[np.argmin([np.mean((0.5 + g * (p - 0.5) - o) ** 2) for g in grid])])


def record_tables(rows):
    summary, bets, bands = [], [], []
    for (timing, stat), s in [((t, 'ALL'), g) for t, g in rows.groupby('timing')] + list(rows.groupby(['timing', 'stat'])):
        s = s[s['p_over'].notna()]
        if len(s) < 15:
            continue
        p, q, o = s['p_over'].to_numpy(), s['p_book'].to_numpy(), s['over'].to_numpy()
        summary.append({'timing': timing, 'stat': stat, 'n': len(s), 'over_rate': o.mean(), 'model_mean_p': p.mean(),
                        'brier_model': np.mean((p - o) ** 2), 'brier_book': np.mean((q - o) ** 2),
                        'best_shrink': best_shrink(p, o) if len(s) >= 50 else np.nan})
        for thr in EDGES:
            edge = p - q
            sel = np.abs(edge) >= thr
            n = int(sel.sum())
            if n < 15:
                continue
            over_side = edge[sel] > 0
            win = np.where(over_side, o[sel] == 1, o[sel] == 0)
            lo, hi = wilson(int(win.sum()), n)
            fair = np.where(over_side, q[sel], 1 - q[sel])
            bets.append({'timing': timing, 'stat': stat, 'edge_min': thr, 'bets': n, 'overs': int(over_side.sum()),
                         'hit': win.mean(), 'CI low': lo, 'CI high': hi,
                         'roi_-110': float(np.mean(np.where(win, 100 / 110, -1.0))),
                         'roi_fair': float(np.mean(np.where(win, 1 / fair - 1, -1.0)))})
        if stat == 'ALL':
            band = pd.cut(s['p_over'], [0, 0.2, 0.35, 0.45, 0.55, 0.65, 0.8, 1.0], include_lowest=True)
            for b, g in s.groupby(band, observed=True):
                bands.append({'timing': timing, 'P(over) band': str(b), 'n': len(g), 'model_p': g['p_over'].mean(),
                              'book_p': g['p_book'].mean(), 'hit_rate': g['over'].mean()})
    return pd.DataFrame(summary), pd.DataFrame(bets), pd.DataFrame(bands)


def load_inputs(year):
    stats_df, _team_col, name_col, _ = load_and_merge_data(year, 'Full PPR')
    try:
        prior, _t, prior_name_col, _ = load_and_merge_data(year - 1, 'Full PPR')
    except Exception:
        prior, prior_name_col = None, None
    return stats_df, name_col, prior, prior_name_col


def run_record(a):
    stats_df, name_col, prior, prior_name_col = load_inputs(a.year)
    frame, _info = sc.load_scored_ledger(a.year, stats_df, name_col, load_schedule(a.year))
    live = frame[frame['pre_kickoff']].copy()
    live['build_ts'] = pd.to_datetime(live['build_ts'], utc=True)
    arc = load_archive(a.year)
    hist = histories(stats_df, name_col, sorted(live['week'].unique()), prior, prior_name_col)
    at_build = priced_lines(arc, {(w, k): t for w, k, t in zip(live['week'], live['_key'], live['build_ts'])})
    at_close = priced_lines(arc, {(w, k): pd.Timestamp(t) for w, k, t in zip(live['week'], live['_key'], live['kickoff'])
                                  if pd.notna(t)})
    rows = pd.concat([scored_rows(live, at_build, hist, 'at board'), scored_rows(live, at_close, hist, 'at close')],
                     ignore_index=True)
    summary, bets, bands = record_tables(rows)
    fmt = lambda v: f'{v:.3f}'  # noqa: E731
    out = [f"{a.year} prop record: data/prop_probability.py's P(over) on the pre-kickoff boards against the priced posted lines "
           f"(Pinnacle, then DraftKings), weeks {sorted(rows['week'].unique()) if len(rows) else []}, {len(rows):,} line-results", '',
           '1. BRIER (lower is better; 0.25 = a coin flip) and the best shrink toward 1/2 (1 = trust the model as is)',
           summary.to_string(index=False, float_format=fmt), '',
           '2. CALIBRATION, all stats: the model\'s P(over) in bands against the hit rate',
           bands.to_string(index=False, float_format=fmt), '',
           "3. BETTING THE MODEL'S SIDE where its P(over) differs from the book's de-vigged price by at least edge_min "
           "(a -110 bet needs 52.4%)",
           bets.to_string(index=False, float_format=fmt)]
    text = '\n'.join(out)
    print(text)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')
    rows.to_csv(a.csv, index=False)


def run_sheet(a):
    stats_df, name_col, prior, prior_name_col = load_inputs(a.year)
    paths = sorted(glob.glob(os.path.join(sc.LEDGER_DIR, f'{a.year}_wk{int(a.sheet):02d}_*.parquet')))
    if not paths:
        print(f'no ledger board for {a.year} week {a.sheet}')
        return
    board = pd.read_parquet(paths[-1])
    board['_key'] = clean_name_exact(board['Player'])
    board['week'] = int(a.sheet)
    arc = load_archive(a.year)
    arc = arc[arc['week'] == int(a.sheet)]
    now = pd.Timestamp.now(tz='UTC')
    lines = priced_lines(arc, {(int(a.sheet), k): now for k in board['_key']})
    hist = pp.player_history(stats_df, name_col, int(a.sheet), prior, prior_name_col)
    shrink = DEFAULT_SHRINK
    if os.path.exists(a.csv):
        rec = pd.read_csv(a.csv)
        rec = rec[(rec['timing'] == 'at close') & rec['p_over'].notna()]
        if len(rec) >= 300:
            shrink = best_shrink(rec['p_over'].to_numpy(), rec['over'].to_numpy())
    rows = []
    for stat, market in pp.PROP_MARKETS.items():
        if stat not in board.columns:
            continue
        pre = [f'_pre_recal_{stat}'] if f'_pre_recal_{stat}' in board.columns else []
        f = board[['Player', 'Pos', 'Team', '_key', stat] + pre].copy()
        got = [lines.get((int(a.sheet), k, market)) for k in f['_key']]
        f['line'] = [g[0] if g else np.nan for g in got]
        f['p_book'] = [g[1] if g else np.nan for g in got]
        f['book'] = [g[2] if g else None for g in got]
        f = f[f['line'].notna() & pd.to_numeric(f[stat], errors='coerce').gt(0)]
        f = f[f['line'] >= MIN_LEAN_LINE.get(stat, 0.0)]
        if f.empty:
            continue
        f = pp.prop_probabilities(f, stat, history=hist)
        f['stat'] = stat
        rows.append(f.rename(columns={stat: 'projection'}))
    if not rows:
        print('no priced lines for that week')
        return
    sheet = pd.concat(rows, ignore_index=True)
    sheet['p_over_shrunk'] = 0.5 + shrink * (sheet['p_over'] - 0.5)
    sheet['gap'] = sheet['p_over_shrunk'] - sheet['p_book']
    sheet = sheet.sort_values('gap', key=np.abs, ascending=False)
    path = os.path.join('.sweeps', f'prop_sheet_{a.year}_wk{int(a.sheet):02d}.csv')
    sheet.to_csv(path, index=False)
    print(f'board {os.path.basename(paths[-1])}; {len(sheet)} priced lines; shrink {shrink:.2f}; -> {path}')
    print(sheet.head(25)[['Player', 'Pos', 'stat', 'line', 'book', 'projection', 'prop_mean', 'p_over', 'p_over_shrunk', 'p_book',
                          'gap']].to_string(index=False, float_format=lambda v: f'{v:.3f}'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=2026)
    ap.add_argument('--out', default='.sweeps/prop_scorecard.txt')
    ap.add_argument('--csv', default='.sweeps/prop_scorecard_rows.csv')
    ap.add_argument('--sheet', type=int, default=None, help='write the prop sheet for this week instead of the record')
    ap.add_argument('--engines', default=None, help='score with this engine file instead of data/prop_engines.json')
    ap.add_argument('--install-engines', action='store_true')
    ap.add_argument('--engines-from', default=os.path.join('.sweeps', 'deep_audit', 'prop_models_v2'))
    a = ap.parse_args()
    if a.engines:
        pp.ENGINE_PATH = a.engines
    if a.install_engines:
        install_engines(a.engines_from)
    elif a.sheet is not None:
        run_sheet(a)
    else:
        run_record(a)


if __name__ == '__main__':
    main()
