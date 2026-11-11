"""
PROPS view (2026-11-11, audit C1): every priced prop on the built board, one row each, as a betting list. Opened from the
PROPS tab beside BOARD and GAMES (not the default view). The pricing is the ledger's (data/prediction_ledger.py D1 columns:
the sharpest book's priced line, its no-vig P(over), the model's P(over) at that line, the v0 tier), so a row matches what
the ledger stores for the build.

Per row: player, team and opponent, stat, the line and its book, the model's median, the side, the model's chance of that
side shrunk toward 50% (data/prop_tiers.py), fair odds, the best price for the side across the books that priced it
(data/line_shopping.py) with EV % at that price, the tier and its record, and how far that book's line has moved since its
first pull of the week. Sorted by tier, then EV. Track and No bet rows are hidden unless asked for. Colour: the tier and EV only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import prop_probability as pp
from data import prop_tiers as pt
from data.line_shift import p_over_at
from data.line_shopping import american, best_prices
from ui.games_view import PROP_GROUPS, STAT_LABELS, TIER_ORDER, C, blend

SHOWN_TIERS = ('A', 'B')
COLUMNS = ['PLAYER', 'TEAM', 'OPP', 'STAT', 'LINE', 'BOOK', 'MEDIAN', 'SIDE', 'P', 'FAIR', 'BEST', 'PRICE', 'EV %', 'TIER',
           'RECORD', 'MOVE']


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def priced_rows(model_df, book_lines, history=None):
    """Long frame, one row per (player, prop stat) with a priced line: _key, Player, Pos, Team, Opponent, stat, line, book,
    over/under price, p_book, p_model, tier, median. The ledger's pricing (prediction_ledger._line_frame /
    _attach_model_p_over / prop_tiers.board_tiers). Empty frame without priced lines."""
    from data.prediction_ledger import _attach_model_p_over, _line_frame
    from data.utils import clean_name_exact
    from data.weekly_median import attach_medians
    lines = _line_frame(book_lines)
    if lines.empty or model_df is None or model_df.empty:
        return pd.DataFrame()
    keep = [c for c in ('Player', 'Pos', 'Team', 'Opponent') if c in model_df.columns]
    keep += [c for c in model_df.columns if c in pp.PROP_MARKETS or str(c).startswith('_pre_recal_')]
    b = model_df[keep].copy()
    try:
        med = attach_medians(model_df)
        for c in [c for c in med.columns if str(c).startswith('Median ')]:
            b[c] = med[c].to_numpy()
    except Exception:
        pass
    b['_key'] = clean_name_exact(b['Player'].astype(str))
    b = b.drop_duplicates('_key').merge(lines.drop_duplicates('_key'), on='_key', how='inner')
    b = pt.board_tiers(_attach_model_p_over(b, history=history))
    parts = []
    for stat in pp.PROP_MARKETS:
        need = [f'Priced Line {stat}', f'Book P Over {stat}', f'Model P Over {stat}']
        if not all(c in b.columns for c in need):
            continue
        s = b[b[need].notna().all(axis=1)]
        if s.empty:
            continue
        parts.append(pd.DataFrame({
            '_key': s['_key'], 'Player': s['Player'], 'Pos': s.get('Pos'), 'Team': s.get('Team'),
            'Opponent': s.get('Opponent'), 'stat': stat, 'line': s[f'Priced Line {stat}'].astype(float),
            'book': s[f'Priced Book {stat}'], 'over_price': s[f'Over Price {stat}'], 'under_price': s[f'Under Price {stat}'],
            'p_book': s[f'Book P Over {stat}'].astype(float), 'p_model': s[f'Model P Over {stat}'].astype(float),
            'tier': s.get(f'Tier {stat}', ''), 'median': s.get(f'Median {stat}', np.nan)}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def props_rows(model_df, book_lines, history=None, shrinks=None, tier_rec=None, first_lines=None):
    """The view's rows (every tier; the renderer filters): COLUMNS plus _key, _stat, _tier_rank, _ev, _side_p.
    shrinks: {stat: s, '_all': s} (prop_tiers.shrinks_before); default prop_tiers.DEFAULT_SHRINK. tier_rec: prop_tiers.tier_record
    output (its forward record is shown per tier). first_lines: {(_key, stat, book): that book's first line of the week}."""
    r = priced_rows(model_df, book_lines, history)
    if r.empty:
        return pd.DataFrame(columns=COLUMNS)
    shrinks = shrinks or {}
    s = r['stat'].map(lambda st: shrinks.get(st, shrinks.get('_all', pt.DEFAULT_SHRINK))).astype(float)
    over = r['p_model'] > r['p_book']
    r['side'] = np.where(over, 'over', 'under')
    p_shrunk = 0.5 + s * (r['p_model'] - 0.5)
    side_p = np.where(over, p_shrunk, 1 - p_shrunk)
    # the best price for the side across books; the model's chance moved to that book's line when it differs
    best = best_prices(book_lines)
    from data.utils import clean_name_exact
    if len(best):
        best['_key'] = clean_name_exact(best['player'].astype(str))
        best['stat'] = best['market'].map({m: st for st, m in pp.PROP_MARKETS.items()})
        best = best.dropna(subset=['stat']).drop_duplicates(['_key', 'stat'])
        m = r[['_key', 'stat']].merge(best, on=['_key', 'stat'], how='left')
    else:
        m = pd.DataFrame(index=r.index, columns=['over_line', 'over_price', 'over_book', 'under_line', 'under_price',
                                                 'under_book'])
    b_line = np.where(over, m['over_line'], m['under_line']).astype(float)
    b_price = np.where(over, m['over_price'], m['under_price']).astype(float)
    b_book = np.where(over, m['over_book'], m['under_book'])
    own_price = np.where(over, r['over_price'], r['under_price']).astype(float)
    use_own = ~np.isfinite(b_price)
    b_line = np.where(use_own, r['line'], b_line)
    b_price = np.where(use_own, own_price, b_price)
    b_book = np.where(use_own, r['book'], b_book)
    p_at_best = side_p.copy()
    for stat in r['stat'].unique():
        idx = np.where((r['stat'] == stat).to_numpy() & ~np.isclose(b_line, r['line'].to_numpy(float)))[0]
        if len(idx):
            q = p_over_at(stat, r['Pos'].astype(str).to_numpy()[idx], r['line'].to_numpy(float)[idx], p_shrunk.to_numpy()[idx],
                          b_line[idx])
            p_at_best[idx] = np.where(over.to_numpy()[idx], q, 1 - q)
    ev = p_at_best * b_price - 1.0
    rec = {}
    if tier_rec is not None and len(tier_rec):
        for _, t in tier_rec.iterrows():
            rec[t['tier']] = f"{100 * t['hit shown']:.0f}% ({int(t['bets'])})" if t.get('bets') else '-'
    first = first_lines or {}
    move = [r_line - first[(k, st, bk)] if (k, st, bk) in first else np.nan
            for k, st, bk, r_line in zip(r['_key'], r['stat'], r['book'], r['line'])]
    out = pd.DataFrame({
        'PLAYER': r['Player'], 'TEAM': r['Team'], 'OPP': r['Opponent'], 'STAT': r['stat'].map(lambda x: STAT_LABELS.get(x, x)),
        'LINE': r['line'], 'BOOK': r['book'], 'MEDIAN': pd.to_numeric(r['median'], errors='coerce'),
        'SIDE': np.where(over, 'OVER', 'UNDER'), 'P': side_p,
        'FAIR': [american(1 / p) if p > 0 else '' for p in side_p],
        'BEST': [f'{bk} {ln:g}' if isinstance(bk, str) else '' for bk, ln in zip(b_book, b_line)],
        'PRICE': [american(p) for p in b_price], 'EV %': ev, 'TIER': r['tier'].fillna(''),
        'RECORD': r['tier'].map(lambda t: rec.get(t, '')), 'MOVE': move})
    out['_key'], out['_stat'], out['_ev'], out['_side_p'] = r['_key'], r['stat'], ev, side_p
    out['_tier_rank'] = out['TIER'].map({t: i for i, t in enumerate(TIER_ORDER)}).fillna(len(TIER_ORDER))
    return out.sort_values(['_tier_rank', '_ev'], ascending=[True, False]).reset_index(drop=True)


def filter_rows(rows, game=None, group='All', tiers=SHOWN_TIERS, book=None, side=None):
    """The rows shown: one game (a set of teams), a stat group (games_view.PROP_GROUPS), tiers, a best-price book, a side."""
    out = rows
    if game:
        out = out[out['TEAM'].isin(game)]
    if group in PROP_GROUPS:
        out = out[out['_stat'].isin(PROP_GROUPS[group])]
    if tiers:
        out = out[out['TIER'].isin(tiers)]
    if book:
        out = out[out['BEST'].astype(str).str.startswith(book)]
    if side:
        out = out[out['SIDE'] == side]
    return out.reset_index(drop=True)


def styled(rows):
    show = rows[[c for c in COLUMNS if c in rows.columns]]
    zebra = blend(C['surface_container'], C['surface_container_high'], 0.4)

    def style_row(row):
        base = zebra if int(row.name) % 2 else C['surface_container']
        ev = _num(row.get('EV %'))
        tone = C['positive'] if np.isfinite(ev) and ev > 0 else C['negative']
        shade = blend(base, tone, min(abs(ev) / 0.15, 1.0) * 0.55) if np.isfinite(ev) else base
        tier = row.get('TIER')
        out = []
        for col in show.columns:
            if col == 'EV %':
                out.append(f'background-color:{shade}; color:#ffffff; font-weight:bold;')
            elif col == 'TIER':
                out.append(f"background-color:{blend(base, C['primary'], 0.45 if tier == 'A' else 0.2 if tier == 'B' else 0)};"
                           f" color:{C['on_surface']}; font-weight:bold;")
            elif col == 'PLAYER':
                out.append(f"background-color:{base}; color:{C['on_surface']}; font-weight:bold;")
            else:
                out.append(f"background-color:{base}; color:{C['on_surface']};")
        return out

    fmt = {'LINE': '{:g}', 'MEDIAN': '{:.1f}', 'P': '{:.0%}', 'EV %': '{:+.1%}',
           'MOVE': lambda v: '' if not np.isfinite(_num(v)) or abs(_num(v)) < 1e-9 else f'{_num(v):+g}'}
    return show.style.apply(style_row, axis=1).format(fmt, na_rep='-')


def render(rows, games=None):
    """The PROPS view: filters, a one-line header, the table. `games`: [(label, {teams})] for the game filter."""
    import streamlit as st
    from ui.components import quiet_note
    from ui.styling import df_auto_height
    if rows is None or rows.empty:
        st.caption('No priced props for this board yet: pull Market props, then build.')
        return
    c1, c2, c3, c4, c5 = st.columns([2, 1.2, 1.2, 1.2, 1])
    labels = ['All games'] + [g[0] for g in (games or [])]
    pick = c1.selectbox('Game', labels, key='props_game')
    group = c2.selectbox('Stats', ['All'] + list(PROP_GROUPS), key='props_group')
    side = c3.selectbox('Side', ['Both', 'OVER', 'UNDER'], key='props_side')
    books = sorted({str(b).split(' ')[0] for b in rows['BEST'] if b})
    book = c4.selectbox('Best at', ['Any'] + books, key='props_book')
    every = c5.toggle('All tiers', value=False, key='props_all_tiers')
    game = dict(games or []).get(pick) if pick != 'All games' else None
    shown = filter_rows(rows, game, group, None if every else SHOWN_TIERS, None if book == 'Any' else book,
                        None if side == 'Both' else side)
    counts = rows['TIER'].value_counts()
    st.caption(f"{len(shown)} shown · " + ' · '.join(f"{t} {int(counts.get(t, 0))}" for t in TIER_ORDER)
               + ' · passing stats and carries are No bet')
    if shown.empty:
        st.caption('Nothing in these filters.')
        return
    st.dataframe(styled(shown), hide_index=True, width='stretch', height=df_auto_height(min(len(shown), 40)))
    quiet_note("P = the model's chance of the side, pulled toward 50% by the factor the record fits (the raw engine is too "
               "confident). FAIR = the odds that chance is worth. BEST = the book and line with the best price for the side "
               "(EV % there, the chance moved to that line). RECORD = the tier's forward hit rate and bets, from 2026 week 10 "
               "(tracking until 300 bets clear zero). MOVE = how far that book's line has moved since its first pull of the "
               "week. A tier never moves a projection.", 'About these columns')
