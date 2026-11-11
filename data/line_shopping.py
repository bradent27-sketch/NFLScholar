"""
Line shopping (2026-11-11, audit B4): per (player, stat), the best price on each side across every book that posted a priced
line, and how far the books' lines spread. Input: rows with player, market, provider, line, over_payout, under_payout (the
staged pull's book lines, data.odds_weekly.weekly_market_book_lines, plus any Odds API rows from data/odds_api_props.py).

The best over is the lowest line (an over wants it low), the best price at that line breaking a tie; the best under is the
highest line, then the best price. Pick'em rows (no prices) are left out: they pay a fixed amount, not a price (B5 reads them).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PICKEM_BOOKS = ('PrizePicks', 'Underdog')


def _book(provider):
    return provider.astype(str).str.split(' (', regex=False).str[0]


def best_prices(rows):
    """One row per (player, market): over_line, over_price, over_book, under_line, under_price, under_book, books (how many
    priced the stat), line_spread (highest minus lowest line). Empty frame without priced rows."""
    need = {'player', 'market', 'provider', 'line', 'over_payout', 'under_payout'}
    if rows is None or len(rows) == 0 or not need <= set(rows.columns):
        return pd.DataFrame()
    r = rows[list(need)].copy()
    for c in ('line', 'over_payout', 'under_payout'):
        r[c] = pd.to_numeric(r[c], errors='coerce')
    r = r.dropna(subset=['line', 'over_payout', 'under_payout'])
    if r.empty:
        return pd.DataFrame()
    r['book'] = _book(r['provider'])
    r = r[~r['book'].isin(PICKEM_BOOKS) & (r['over_payout'] > 1) & (r['under_payout'] > 1)]   # multipliers are not prices
    if r.empty:
        return pd.DataFrame()
    over = r.sort_values(['line', 'over_payout'], ascending=[True, False]).drop_duplicates(['player', 'market'])
    under = r.sort_values(['line', 'under_payout'], ascending=[False, False]).drop_duplicates(['player', 'market'])
    g = r.groupby(['player', 'market'])
    out = pd.DataFrame({'books': g['book'].nunique(), 'line_spread': g['line'].max() - g['line'].min()}).reset_index()
    out = out.merge(over[['player', 'market', 'line', 'over_payout', 'book']].rename(
        columns={'line': 'over_line', 'over_payout': 'over_price', 'book': 'over_book'}), on=['player', 'market'])
    out = out.merge(under[['player', 'market', 'line', 'under_payout', 'book']].rename(
        columns={'line': 'under_line', 'under_payout': 'under_price', 'book': 'under_book'}), on=['player', 'market'])
    return out


def american(decimal_price):
    """Decimal odds -> American text ('+120', '-110'); '' for a missing price."""
    d = float(decimal_price) if decimal_price is not None else np.nan
    if not np.isfinite(d) or d <= 1:
        return ''
    return f'+{round((d - 1) * 100):d}' if d >= 2 else f'-{round(100 / (d - 1)):d}'
