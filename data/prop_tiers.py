"""
Prop tiers v0 (2026-11-11, audit B2, docs/audit_leads_2026-11-10.md; definitions and bar fixed before scoring in
.sweeps/prop_tiers_v0_bar.md). A label for a priced prop in place of the gap %: edge size did not predict the hit rate, while the
stat, the side and agreement with the book's price lean did (data/prop_record.py, B3). A tier never moves a projection.

  No bet  passing yards / attempts / completions / TDs, interceptions, carries (their B8 switch is off); or a gap over 0.25
  A       receiving yards, receptions, or a rushing-yards under; the model's side matches the price lean; gap 0.03-0.25
  B       the same stats and sides against the lean (or no lean); gap 0.03-0.25
  Track   everything else: shown, never recommended

Each tier carries a forward record (bets, hit rate, ROI at the posted price with a 90% lower bound, closing-line value) and
reads "bettable" only with 300+ bets and that lower bound above 0; otherwise "tracking". Changing a definition here is a new
version whose record starts over.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import prop_probability as pp
from data.prop_record import best_shrink, wilson

VERSION = 'v0'
# The v0 rules were chosen from what weeks 3-8 of 2026 showed (B3), so those weeks cannot test them: only boards from this week
# on count toward the status. Earlier weeks are shown as in-sample background.
FORWARD_FROM = (2026, 10)
NO_BET_STATS = frozenset({'passing_yards', 'passing_attempts', 'passing_completions', 'passing_tds', 'passing_interceptions',
                          'rushing_attempts'})
A_STATS = frozenset({'receiving_yards', 'receptions'})
UNDER_ONLY_STATS = frozenset({'rushing_yards'})
GAP_MIN, GAP_MAX = 0.03, 0.25
ORDER = ('A', 'B', 'Track', 'No bet')
DEFAULT_SHRINK = 0.34
SHRINK_MIN_STAT, SHRINK_MIN_ALL = 200, 300
SHOWN_PRIOR_BETS = 200
BETTABLE_MIN_BETS = 300


def assign_tiers(stat, p, q):
    """The v0 tier per line: `stat` (scalar or array), the model's raw P(over) `p` and the book's no-vig P(over) `q`. '' where
    either probability is missing."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    stat = np.broadcast_to(np.asarray(stat, dtype=object), p.shape)
    gap = np.abs(p - q)
    model_over = p > q
    lean = np.where(q > 0.5, 1, np.where(q < 0.5, -1, 0))
    with_lean = np.where(model_over, lean == 1, lean == -1)
    eligible = np.array([s in A_STATS for s in stat]) | (np.array([s in UNDER_ONLY_STATS for s in stat]) & ~model_over)
    in_band = (gap >= GAP_MIN) & (gap <= GAP_MAX)
    no_bet = np.array([s in NO_BET_STATS for s in stat]) | (gap > GAP_MAX)
    tier = np.where(no_bet, 'No bet', np.where(eligible & in_band & with_lean, 'A',
                                               np.where(eligible & in_band, 'B', 'Track')))
    return np.where(np.isfinite(p) & np.isfinite(q), tier, '').astype(object)


def shrinks_before(rows, week):
    """{stat: s} for p' = 0.5 + s (p - 0.5), fit on the 'at close' sharpest-book record rows of weeks before `week`: per stat
    with 200+ rows, else all stats pooled with 300+, else DEFAULT_SHRINK. Key '_all' holds the pooled value."""
    r = rows[(rows['timing'] == 'at close') & (rows['week'] < week) & rows['p_over'].notna()]
    if 'sharpest' in r.columns:
        r = r[r['sharpest'].astype(bool)]
    pooled = best_shrink(r['p_over'].to_numpy(), r['over'].to_numpy()) if len(r) >= SHRINK_MIN_ALL else DEFAULT_SHRINK
    out = {'_all': pooled}
    for stat, g in r.groupby('stat'):
        out[stat] = best_shrink(g['p_over'].to_numpy(), g['over'].to_numpy()) if len(g) >= SHRINK_MIN_STAT else pooled
    return out


def _clv(bets, close):
    """Per bet: +1 when the close from the same book moved toward the side taken (a higher line or, at the same line, a higher
    no-vig P(over), for an over; the mirror for an under), -1 when it moved away, 0 when nothing moved, NaN with no close."""
    key = ['week', '_key', 'stat', 'book']
    c = close[key + ['line', 'p_book']].drop_duplicates(key).rename(columns={'line': '_close_line', 'p_book': '_close_q'})
    m = bets[key].merge(c, on=key, how='left')
    sign = np.where(bets['side'].eq('over').to_numpy(), 1.0, -1.0)
    dline = m['_close_line'].to_numpy(float) - bets['line'].to_numpy(float)
    dq = m['_close_q'].to_numpy(float) - bets['p_book'].to_numpy(float)
    move = np.where(np.abs(dline) > 1e-9, np.sign(dline), np.where(np.abs(dq) > 1e-9, np.sign(dq), 0.0))
    return np.where(np.isfinite(dline), sign * move, np.nan)


def forward_rows(rows):
    """The record's at-board sharpest-book rows (data.prop_record.build_record) with the v0 tier, the side taken, its price,
    the shrunk P(over) and EV (week w's shrink fit on weeks before w only), the result and closing-line value."""
    if rows is None or rows.empty:
        return pd.DataFrame()
    board = rows[(rows['timing'] == 'at board') & rows['p_over'].notna() & rows['p_book'].notna()]
    if 'sharpest' in board.columns:
        board = board[board['sharpest'].astype(bool)]
    board = board.copy()
    board['tier'] = assign_tiers(board['stat'].to_numpy(), board['p_over'], board['p_book'])
    over = (board['p_over'] > board['p_book']).to_numpy()
    board['side'] = np.where(over, 'over', 'under')
    board['price'] = np.where(over, board['over_payout'], board['under_payout'])
    board['win'] = np.where(over, board['over'] == 1, board['over'] == 0)
    shrink = {int(w): shrinks_before(rows, int(w)) for w in board['week'].unique()}
    s = np.array([shrink[int(w)].get(st, shrink[int(w)]['_all']) for w, st in zip(board['week'], board['stat'])])
    board['shrink'] = s
    board['p_shrunk'] = 0.5 + s * (board['p_over'] - 0.5)
    side_p = np.where(over, board['p_shrunk'], 1 - board['p_shrunk'])
    board['ev'] = side_p * board['price'].astype(float) - 1.0
    close = rows[rows['timing'] == 'at close']
    board['clv'] = _clv(board.reset_index(drop=True), close) if len(close) else np.nan
    return board.reset_index(drop=True)


def tier_record(fwd, year=None, forward_only=False):
    """One row per tier (ORDER) of the forward rows: bets, hit rate and its 95% interval, the hit rate shown (shrunk toward 50%
    with SHOWN_PRIOR_BETS), ROI at the posted price and its 90% lower bound, the mean EV, closing-line value (the share of bets
    with a close that it moved toward, and away from) and the status: 'bettable' only with BETTABLE_MIN_BETS+ bets and the lower
    bound above 0, else 'tracking'. forward_only keeps weeks from FORWARD_FROM on (`year` = the rows' season); the status
    always reads those weeks only - an in-sample record is never labelled bettable."""
    if forward_only and len(fwd):
        fwd = fwd[is_forward(year, fwd['week'])]
    out = []
    for tier in ORDER:
        g = fwd[fwd['tier'] == tier] if len(fwd) else fwd
        n = len(g)
        rec = {'tier': tier, 'bets': n}
        if n:
            win = g['win'].to_numpy(bool)
            ret = np.where(win, g['price'].to_numpy(float) - 1.0, -1.0)
            ret = ret[np.isfinite(g['price'].to_numpy(float))]
            lo, hi = wilson(int(win.sum()), n)
            lb = float(ret.mean() - 1.645 * ret.std(ddof=1) / np.sqrt(len(ret))) if len(ret) > 1 else np.nan
            clv = g['clv'].dropna()
            fwd_n = int(is_forward(year, g['week']).sum())
            rec.update({'hit': float(win.mean()), 'CI low': lo, 'CI high': hi,
                        'hit shown': (win.sum() + SHOWN_PRIOR_BETS / 2) / (n + SHOWN_PRIOR_BETS),
                        'roi_price': float(ret.mean()) if len(ret) else np.nan, 'roi 90% low': lb,
                        'mean ev': float(g['ev'].mean()),
                        'clv toward': float((clv > 0).mean()) if len(clv) else np.nan,
                        'clv away': float((clv < 0).mean()) if len(clv) else np.nan,
                        'clv n': int(len(clv)), 'overs': int(g['side'].eq('over').sum()), 'forward bets': fwd_n})
            rec['status'] = ('bettable' if forward_only and n >= BETTABLE_MIN_BETS and np.isfinite(lb) and lb > 0
                             else 'tracking')
        else:
            rec['status'] = 'tracking'
        out.append(rec)
    return pd.DataFrame(out)


def is_forward(year, weeks):
    """Boolean array: the week is on or after FORWARD_FROM for season `year` (None = unknown: nothing is forward)."""
    weeks = np.asarray(weeks, dtype=float)
    if year is None:
        return np.zeros(len(weeks), dtype=bool)
    return (int(year) > FORWARD_FROM[0]) | ((int(year) == FORWARD_FROM[0]) & (weeks >= FORWARD_FROM[1]))


def board_tiers(frame):
    """'Tier <stat>' for every row and prop stat of a frame carrying 'Model P Over <stat>' and 'Book P Over <stat>' (a ledger
    frame, or a board run through prop_board_columns); '' where either is missing."""
    out = frame.copy()
    for stat in pp.PROP_MARKETS:
        p, q = f'Model P Over {stat}', f'Book P Over {stat}'
        if p in out.columns and q in out.columns:
            out[f'Tier {stat}'] = assign_tiers(stat, pd.to_numeric(out[p], errors='coerce'), pd.to_numeric(out[q], errors='coerce'))
    return out


def prop_board_columns(board, book_lines, history=None):
    """Player plus, per prop stat, the book's and the model's P(over) and the tier, on the board's own index, for a built board
    and the staged pull's book lines (data.odds_weekly.weekly_market_book_lines). The same pricing the ledger stores (D1). Empty
    frame when nothing is priced; never raises."""
    try:
        from data.prediction_ledger import _attach_model_p_over, _line_frame
        from data.utils import clean_name_exact
        lines = _line_frame(book_lines)
        if lines.empty or board is None or board.empty:
            return pd.DataFrame()
        keep = ['Player', 'Pos'] + [c for c in board.columns if c in pp.PROP_MARKETS or c.startswith('_pre_recal_')]
        b = board[keep].copy()
        b['_key'] = clean_name_exact(b['Player'].astype(str))
        b = b.merge(lines.drop_duplicates('_key'), on='_key', how='left')
        b.index = board.index                       # a left merge on unique keys keeps the board's rows and order
        b = board_tiers(_attach_model_p_over(b, history=history))
        cols = ['Player'] + [c for c in b.columns if c.startswith(('Tier ', 'Model P Over ', 'Book P Over '))]
        return b[cols]
    except Exception:
        return pd.DataFrame()
