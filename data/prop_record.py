"""
The prop record (2026-11-11, audit B3, docs/audit_leads_2026-11-10.md): data/prop_probability.py's P(over) on the pre-kickoff
ledger boards scored against the lines the books posted (external_data/props_archive), shared by scripts/prop_scorecard.py and
the Track Record page. Moved out of scripts/prop_scorecard.py and scripts/eval_posted_line_record.py; nothing about the model
changes.

One row per (player-week, stat, book) with a priced line - both prices kept - so the record can be read three ways:
  - the sharpest book's line (Pinnacle, then DraftKings, ...: data.odds_projections.LINE_BOOKS), graded as before: Brier,
    calibration, the shrink toward 1/2, and betting the model's side at an edge, with ROI at -110 and at the book's fair price;
  - the same bets at the price actually posted for the side taken (`roi_price`), by book and by side;
  - against three blind baselines per stat (always under, always over, always the side the book's price favours), with the
    model's lift over the best of them. A record that only matches "always under" has found nothing.

Yardage lines under 5 are left out, as the app does. Pushes are out. Prices are decimal odds (1.91 = -110).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from data import prop_probability as pp
from data.odds_projections import LINE_BOOKS
from data.utils import clean_name_exact

ARCHIVE = os.path.join('external_data', 'props_archive')
# board stat -> the archive's market name
LINE_STATS = {'receiving_yards': 'receiving_yards', 'rushing_yards': 'rushing_yards', 'passing_yards': 'passing_yards',
              'receptions': 'receptions', 'targets': 'targets', 'rushing_attempts': 'carries', 'passing_attempts': 'attempts',
              'passing_completions': 'passing_completions', 'passing_tds': 'passing_tds', 'passing_interceptions': 'passing_interceptions'}
# the app's own minimum line for a lean (ui.games_view.MIN_LEAN_LINE; repeated here so data/ does not import ui/)
MIN_LEAN_LINE = {'receiving_yards': 5.0, 'rushing_yards': 5.0, 'passing_yards': 5.0}
EDGES = (0.03, 0.05, 0.08, 0.12)
MIN_ROWS = 15


def wilson(k, n, z=1.96):
    """95% Wilson interval for k hits in n."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load_archive(year, archive=None):
    """Every scorable full-game line ever pulled for `year`: week, _key, market, book, _rank, line, p_over, over_payout,
    under_payout, fetched_at (UTC).

    A snapshot that cannot be read (e.g. 2026_wk09_20261103T001625Z.json, cut off mid-write) is skipped and named on stderr;
    the file itself is left in place (the archive is a permanent record). The skipped names are in attrs['skipped_files']."""
    frames, skipped = [], []
    for path in sorted(glob.glob(os.path.join(archive or ARCHIVE, f'{year}_wk*.json'))):
        try:
            with open(path, encoding='utf-8') as handle:
                snap = json.load(handle)
            if not isinstance(snap, dict) or 'week' not in snap or 'fetched_at' not in snap:
                raise ValueError('not a props snapshot')
        except (OSError, ValueError) as exc:
            skipped.append(os.path.basename(path))
            print(f'props archive: skipped unreadable snapshot {os.path.basename(path)} ({exc.__class__.__name__})',
                  file=sys.stderr)
            continue
        rows = pd.DataFrame(snap.get('rows') or [])
        if rows.empty:
            continue
        rows['week'] = int(snap['week'])
        rows['fetched_at'] = pd.Timestamp(snap['fetched_at'])
        # A book read from a saved payload carries when that file was written (data.odds_weekly, 2026-11-11): its lines are
        # that old, however late the pull that re-read them. Snapshots before then have no stamp and keep the pull's time.
        status = snap.get('status') if isinstance(snap.get('status'), dict) else {}
        saved = {b: s.get('saved_at') for b, s in status.items() if isinstance(s, dict) and s.get('saved_at')}
        if saved:
            book = rows['provider'].astype(str).str.split(' (', regex=False).str[0]
            stamp = book.map(saved)
            rows.loc[stamp.notna(), 'fetched_at'] = pd.to_datetime(stamp[stamp.notna()], utc=True).map(pd.Timestamp)
        frames.append(rows)
    cols = ['week', '_key', 'market', 'book', '_rank', 'line', 'p_over', 'over_payout', 'under_payout', 'fetched_at']
    if not frames:
        out = pd.DataFrame(columns=cols)
        out.attrs['skipped_files'] = skipped
        return out
    arc = pd.concat(frames, ignore_index=True)
    for c in ('over_payout', 'under_payout', 'p_over'):
        if c not in arc.columns:
            arc[c] = np.nan
    arc = arc[arc['scorable'].fillna(False).astype(bool) & arc['period'].astype(str).eq('game')
              & arc['market'].isin(set(LINE_STATS.values()))].copy()
    arc['book'] = arc['provider'].astype(str).str.split(' (', regex=False).str[0]
    arc = arc[arc['book'].isin(LINE_BOOKS)]
    arc['_rank'] = arc['book'].map({b: i for i, b in enumerate(LINE_BOOKS)})
    arc['_key'] = clean_name_exact(arc['player'])
    for c in ('line', 'p_over', 'over_payout', 'under_payout'):
        arc[c] = pd.to_numeric(arc[c], errors='coerce')
    arc['fetched_at'] = pd.to_datetime(arc['fetched_at'], utc=True)
    out = arc.dropna(subset=['line'])[cols].copy()
    out.attrs['skipped_files'] = skipped
    return out


def book_lines_as_of(arc, when, priced=True):
    """Per (week, _key, market, book) the book's latest line pulled at or before `when` ((week, _key) -> Timestamp), with its
    p_over and both prices; `priced` keeps only lines that carry a de-vigged p_over. Column 'sharpest' marks the one row per
    (week, _key, market) from the first book in LINE_BOOKS order."""
    a = arc[arc['p_over'].notna()].copy() if priced else arc.copy()
    a['_when'] = [when.get((w, k)) for w, k in zip(a['week'], a['_key'])]
    a = a[a['_when'].notna()]
    a = a[a['fetched_at'] <= pd.to_datetime(a['_when'], utc=True)]
    a = a.sort_values('fetched_at').groupby(['week', '_key', 'market', 'book']).last().reset_index()
    a = a.drop(columns=['_when']).sort_values(['week', '_key', 'market', '_rank']).reset_index(drop=True)
    a['sharpest'] = ~a.duplicated(['week', '_key', 'market'])
    return a


def priced_lines(arc, when):
    """{(week, _key, market): (line, p_over, book)}: the sharpest book's latest priced line at or before `when`."""
    a = book_lines_as_of(arc, when)
    a = a[a['sharpest']]
    return {(w, k, m): (float(li), float(p), b)
            for w, k, m, li, p, b in zip(a['week'], a['_key'], a['market'], a['line'], a['p_over'], a['book'])}


def histories(stats_df, name_col, weeks, prior, prior_name_col):
    return {int(w): pp.player_history(stats_df, name_col, int(w), prior, prior_name_col) for w in weeks}


def scored_rows(live, lines, hist, label):
    """One row per (player-week, stat, book) in `lines` (book_lines_as_of): the model's P(over) at that book's line, the book's
    p_over ('p_book'), both prices, whether it is the sharpest book's line, and the result."""
    frames = []
    for stat, market in pp.PROP_MARKETS.items():
        acol = f'actual_{stat}'
        if stat not in live.columns or acol not in live.columns:
            continue
        pre = [f'_pre_recal_{stat}'] if f'_pre_recal_{stat}' in live.columns else []   # v2_stat_recal_own_history boards
        f = live[['week', '_key', 'Player', 'Pos', stat, acol] + pre]
        m = lines.loc[lines['market'] == market, ['week', '_key', 'book', 'line', 'p_over', 'over_payout', 'under_payout',
                                                   'sharpest']].rename(columns={'p_over': 'p_book'})
        f = f.merge(m, on=['week', '_key'], how='inner')
        f = f[pd.to_numeric(f[stat], errors='coerce').gt(0) & pd.to_numeric(f[acol], errors='coerce').notna()]
        f = f[f['line'] >= MIN_LEAN_LINE.get(stat, 0.0)]
        f = f[pd.to_numeric(f[acol], errors='coerce') != f['line']]
        parts = [pp.prop_probabilities(fw, stat, history=hist.get(int(w))) for w, fw in f.groupby('week')]
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


def _groups(rows, by=('stat',)):
    """[(timing, *by values), frame] with an 'ALL' group per timing first."""
    out = [((t, *(['ALL'] * len(by))), g) for t, g in rows.groupby('timing')]
    out += [((k if isinstance(k, tuple) else (k,)), g) for k, g in rows.groupby(['timing', *by])]
    return out


def _bet_returns(win, price):
    """Per bet: the profit on 1 unit at decimal `price` (NaN price -> NaN)."""
    return np.where(win, price - 1.0, -1.0) * np.where(np.isfinite(price), 1.0, np.nan)


def _side_record(win, price, fair=None):
    n = len(win)
    lo, hi = wilson(int(win.sum()), n)
    rec = {'bets': n, 'hit': float(win.mean()) if n else np.nan, 'CI low': lo, 'CI high': hi,
           'roi_-110': float(np.mean(np.where(win, 100 / 110, -1.0))) if n else np.nan}
    if fair is not None:
        rec['roi_fair'] = float(np.mean(np.where(win, 1 / fair - 1, -1.0))) if n else np.nan
    r = _bet_returns(win, price)
    rec['priced'] = int(np.isfinite(r).sum())
    rec['roi_price'] = float(np.nanmean(r)) if np.isfinite(r).any() else np.nan
    return rec


def _model_side(s):
    """(over_side, win, price taken, fair p of the side) for betting the model's side against the book's price on frame s."""
    p, q, o = s['p_over'].to_numpy(), s['p_book'].to_numpy(), s['over'].to_numpy()
    over_side = p - q > 0
    win = np.where(over_side, o == 1, o == 0)
    price = np.where(over_side, s['over_payout'].to_numpy(float), s['under_payout'].to_numpy(float))
    fair = np.where(over_side, q, 1 - q)
    return over_side, win, price, fair


def record_tables(rows):
    """(summary, bets, bands) on the sharpest book's line, as the record has always been read; `bets` adds the ROI at the
    price posted for the side taken (roi_price) and how many of the bets had one (priced)."""
    if 'sharpest' in rows.columns:
        rows = rows[rows['sharpest'].astype(bool)]
    summary, bets, bands = [], [], []
    for (timing, stat), s in _groups(rows):
        s = s[s['p_over'].notna()]
        if len(s) < MIN_ROWS:
            continue
        p, q, o = s['p_over'].to_numpy(), s['p_book'].to_numpy(), s['over'].to_numpy()
        summary.append({'timing': timing, 'stat': stat, 'n': len(s), 'over_rate': o.mean(), 'model_mean_p': p.mean(),
                        'brier_model': np.mean((p - o) ** 2), 'brier_book': np.mean((q - o) ** 2),
                        'best_shrink': best_shrink(p, o) if len(s) >= 50 else np.nan})
        edge = p - q
        over_side, win, price, fair = _model_side(s)
        for thr in EDGES:
            sel = np.abs(edge) >= thr
            if int(sel.sum()) < MIN_ROWS:
                continue
            rec = {'timing': timing, 'stat': stat, 'edge_min': thr, 'overs': int(over_side[sel].sum())}
            rec.update(_side_record(win[sel], price[sel], fair[sel]))
            bets.append(rec)
        if stat == 'ALL':
            band = pd.cut(s['p_over'], [0, 0.2, 0.35, 0.45, 0.55, 0.65, 0.8, 1.0], include_lowest=True)
            for b, g in s.groupby(band, observed=True):
                bands.append({'timing': timing, 'P(over) band': str(b), 'n': len(g), 'model_p': g['p_over'].mean(),
                              'book_p': g['p_book'].mean(), 'hit_rate': g['over'].mean()})
    bets = pd.DataFrame(bets)
    if len(bets):
        bets = bets[['timing', 'stat', 'edge_min', 'bets', 'overs', 'hit', 'CI low', 'CI high', 'roi_-110', 'roi_fair',
                     'priced', 'roi_price']]
    return pd.DataFrame(summary), bets, pd.DataFrame(bands)


def book_table(rows, edge_min=0.05):
    """The model's side at each book's own line and price where its P(over) differs from that book's price by edge_min:
    one row per (timing, book), every book that posted a priced line (not only the sharpest)."""
    out = []
    for (timing, book), s in rows[rows['p_over'].notna()].groupby(['timing', 'book']):
        s = s[np.abs(s['p_over'] - s['p_book']) >= edge_min]
        if len(s) < MIN_ROWS:
            continue
        over_side, win, price, fair = _model_side(s)
        rec = {'timing': timing, 'book': book, 'lines': int(len(s)), 'overs': int(over_side.sum())}
        rec.update(_side_record(win, price, fair))
        rec['mean price'] = float(np.nanmean(price)) if np.isfinite(price).any() else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def side_table(rows, edges=(0.0, 0.05)):
    """The model's overs and unders separately, on the sharpest book's line, at each minimum edge."""
    if 'sharpest' in rows.columns:
        rows = rows[rows['sharpest'].astype(bool)]
    out = []
    for timing, s in rows[rows['p_over'].notna()].groupby('timing'):
        over_side, win, price, _fair = _model_side(s)
        edge = np.abs(s['p_over'] - s['p_book']).to_numpy()
        for thr in edges:
            for side, mask in (('over', over_side), ('under', ~over_side)):
                sel = mask & (edge >= thr)
                if int(sel.sum()) < MIN_ROWS:
                    continue
                rec = {'timing': timing, 'edge_min': thr, 'side': side}
                rec.update(_side_record(win[sel], price[sel]))
                out.append(rec)
    return pd.DataFrame(out)


def baseline_table(rows, edge_min=0.05):
    """Per stat, on the sharpest book's line: three blind strategies bet on every line (always under, always over, always the
    side the book's price favours - the higher de-vigged p, the under on a tie), the model's side on every line and at
    edge_min, and the model's lift over the best blind strategy (hit rate and ROI at the posted price)."""
    if 'sharpest' in rows.columns:
        rows = rows[rows['sharpest'].astype(bool)]
    out = []
    for (timing, stat), s in _groups(rows):
        s = s[s['p_over'].notna()]
        if len(s) < MIN_ROWS:
            continue
        o = s['over'].to_numpy()
        po, pu = s['over_payout'].to_numpy(float), s['under_payout'].to_numpy(float)
        fav_over = s['p_book'].to_numpy() > 0.5
        blind = {'always under': (o == 0, pu), 'always over': (o == 1, po),
                 'book favourite': (np.where(fav_over, o == 1, o == 0), np.where(fav_over, po, pu))}
        rec = {'timing': timing, 'stat': stat, 'lines': len(s)}
        best_hit, best_roi = -np.inf, -np.inf
        for name, (win, price) in blind.items():
            r = _side_record(win, price)
            rec[f'{name} hit'], rec[f'{name} roi'] = r['hit'], r['roi_price']
            best_hit = max(best_hit, r['hit'])
            if np.isfinite(r['roi_price']):
                best_roi = max(best_roi, r['roi_price'])
        _side, win, price, _fair = _model_side(s)
        r = _side_record(win, price)
        rec['model hit'], rec['model roi'] = r['hit'], r['roi_price']
        sel = np.abs(s['p_over'] - s['p_book']).to_numpy() >= edge_min
        r5 = _side_record(win[sel], price[sel]) if sel.sum() >= MIN_ROWS else {'bets': int(sel.sum()), 'hit': np.nan,
                                                                                'roi_price': np.nan}
        rec[f'model {edge_min:.0%}+ bets'], rec[f'model {edge_min:.0%}+ hit'] = r5['bets'], r5['hit']
        rec[f'model {edge_min:.0%}+ roi'] = r5['roi_price']
        rec['hit lift vs best blind'] = rec['model hit'] - best_hit
        rec['roi lift vs best blind'] = rec['model roi'] - best_roi if np.isfinite(best_roi) else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def build_record(year, stats_df, name_col, prior, prior_name_col, schedule, archive=None, ledger_dir=None):
    """(rows, info): every scored (player-week, stat, book) line at two times - when the board was built ('at board') and the
    last pull before kickoff ('at close') - from the pre-kickoff ledger boards. info: weeks, skipped snapshot names."""
    from data import ledger_scorecard as sc
    kwargs = {'ledger_dir': ledger_dir} if ledger_dir else {}
    frame, _info = sc.load_scored_ledger(year, stats_df, name_col, schedule, **kwargs)
    live = frame[frame['pre_kickoff']].copy()
    live['build_ts'] = pd.to_datetime(live['build_ts'], utc=True)
    arc = load_archive(year, archive)
    info = {'skipped_files': arc.attrs.get('skipped_files') or []}
    if live.empty or arc.empty:
        info['weeks'] = []
        return pd.DataFrame(), info
    hist = histories(stats_df, name_col, sorted(live['week'].unique()), prior, prior_name_col)
    at_build = book_lines_as_of(arc, {(w, k): t for w, k, t in zip(live['week'], live['_key'], live['build_ts'])})
    at_close = book_lines_as_of(arc, {(w, k): pd.Timestamp(t) for w, k, t in zip(live['week'], live['_key'], live['kickoff'])
                                      if pd.notna(t)})
    rows = pd.concat([scored_rows(live, at_build, hist, 'at board'), scored_rows(live, at_close, hist, 'at close')],
                     ignore_index=True)
    info['weeks'] = sorted(int(w) for w in rows['week'].unique()) if len(rows) else []
    return rows, info
