"""
Top-down pick'em legs (2026-11-11, audit B5, docs/audit_leads_2026-11-10.md). A PrizePicks or Underdog line pays a fixed amount,
so a leg is worth playing when its fair chance - the sharp book's de-vigged price moved to the pick'em line (data/line_shift.py)
- clears the leg's break-even (data/pickem_payouts.json). No projection enters the selection; the model's side is shown beside it
as agree / disagree only (the no-blend rule: the market selects bets, it never moves a projection).

Products: PrizePicks standard (more or less), PrizePicks demon and goblin (more only; their payouts are not in the archive, so
they are scored for calibration, not against a break-even), Underdog (both sides, each with its payout multiplier: a leg's
break-even is the entry's divided by its multiplier; 'shaded' rows are lines Underdog has already moved, with uneven
multipliers).

Bar (fixed before scoring): a product and stat is surfaced only when its flagged legs beat their break-even with a 90% lower
bound above it on 300+ legs.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from data import prop_probability as pp
from data.line_shift import p_over_at
from data.prop_record import ARCHIVE, MIN_LEAN_LINE, wilson
from data.utils import clean_name_exact

PAYOUTS_PATH = os.path.join(os.path.dirname(__file__), 'pickem_payouts.json')
PICKEM_BOOKS = ('PrizePicks', 'Underdog')
SHARP_BOOKS = ('Pinnacle', 'DraftKings', 'FanDuel')
MARKET_STAT = {m: s for s, m in pp.PROP_MARKETS.items()}
SURFACE_MIN_LEGS = 300
OVER_ONLY = ('demon', 'goblin')


def load_payouts(path=PAYOUTS_PATH):
    with open(path, encoding='utf-8') as fh:
        return {k: v for k, v in json.load(fh).items() if not k.startswith('_')}


def leg_break_even(payout, picks):
    return float(payout) ** (-1.0 / int(picks))


def break_evens(payouts=None):
    """{book: (best per-leg break-even over its power entries, the entry, the 2-pick break-even)}."""
    payouts = payouts or load_payouts()
    out = {}
    for book, cfg in payouts.items():
        be = {int(n): leg_break_even(m, n) for n, m in (cfg.get('power') or {}).items()}
        if be:
            n = min(be, key=be.get)
            out[book] = (be[n], f'{n}-pick power', be.get(2, np.nan))
    return out


def _kind(provider):
    p = str(provider)
    if '(' not in p:
        return 'standard'
    return p.split('(', 1)[1].rstrip(')').strip()


def load_lines(year, archive=None):
    """Every pick'em row and every priced sharp-book row archived for `year`, full game, prop stats only: week, _key, player,
    position, stat, book, kind, line, over_mult / under_mult (Underdog's payout multipliers; 1 elsewhere), p_over (sharp rows),
    fetched_at (a saved payload's own time where the snapshot carries it)."""
    frames = []
    for path in sorted(glob.glob(os.path.join(archive or ARCHIVE, f'{year}_wk*.json'))):
        try:
            with open(path, encoding='utf-8') as fh:
                snap = json.load(fh)
            if not isinstance(snap, dict) or 'week' not in snap or 'fetched_at' not in snap:
                continue
        except (OSError, ValueError):
            continue
        rows = pd.DataFrame(snap.get('rows') or [])
        if rows.empty or 'provider' not in rows.columns:
            continue
        rows['week'] = int(snap['week'])
        rows['fetched_at'] = pd.Timestamp(snap['fetched_at'])
        status = snap.get('status') if isinstance(snap.get('status'), dict) else {}
        saved = {b: s.get('saved_at') for b, s in status.items() if isinstance(s, dict) and s.get('saved_at')}
        rows['book'] = rows['provider'].astype(str).str.split(' (', regex=False).str[0]
        if saved:
            stamp = rows['book'].map(saved)
            rows.loc[stamp.notna(), 'fetched_at'] = pd.to_datetime(stamp[stamp.notna()], utc=True).map(pd.Timestamp)
        frames.append(rows)
    if not frames:
        return pd.DataFrame()
    a = pd.concat(frames, ignore_index=True)
    a = a[a['period'].astype(str).eq('game') & a['market'].isin(MARKET_STAT) & a['book'].isin(PICKEM_BOOKS + SHARP_BOOKS)].copy()
    a['stat'] = a['market'].map(MARKET_STAT)
    a['kind'] = a['provider'].map(_kind)
    a['fetched_at'] = pd.to_datetime(a['fetched_at'], utc=True)
    for c in ('line', 'p_over', 'over_payout', 'under_payout'):
        a[c] = pd.to_numeric(a.get(c), errors='coerce')
    ud = a['book'].eq('Underdog')
    a['over_mult'] = np.where(ud, a['over_payout'], np.where(a['book'].eq('PrizePicks'), 1.0, np.nan))
    a['under_mult'] = np.where(ud, a['under_payout'],
                               np.where(a['book'].eq('PrizePicks') & a['kind'].eq('standard'), 1.0, np.nan))
    a['_key'] = clean_name_exact(a['player'].astype(str))
    a = a[a['line'].notna() & ~(a['book'].isin(SHARP_BOOKS) & a['p_over'].isna())]
    a = a[a['line'] >= a['stat'].map(MIN_LEAN_LINE).fillna(0.0)]
    return a.reset_index(drop=True)


def legs_at(lines, cutoff):
    """Each pick'em line's latest pull at or before `cutoff` ({(week, _key): Timestamp}), with the sharp book's latest priced
    line as of THAT pick'em line's own time (Pinnacle, then DraftKings, then FanDuel): a line saved by hand hours earlier is
    compared with the price of its own moment, not one that came later. Adds the fair P(over) at the pick'em line."""
    a = lines.copy()
    a['_cut'] = pd.to_datetime(pd.Series([cutoff.get((w, k)) for w, k in zip(a['week'], a['_key'])], index=a.index), utc=True)
    a = a[a['_cut'].notna() & (a['fetched_at'] <= a['_cut'])]
    a = a.sort_values('fetched_at')
    keys = ['week', '_key', 'stat']
    pick = a[a['book'].isin(PICKEM_BOOKS)].groupby(keys + ['book', 'kind']).last().reset_index()
    pick = pick.drop(columns=['p_over'], errors='ignore').sort_values('fetched_at')
    sharp = a[a['book'].isin(SHARP_BOOKS)].sort_values('fetched_at')
    legs = pick.assign(sharp_book=None, sharp_line=np.nan, sharp_p=np.nan, _spos='')
    for book in reversed(SHARP_BOOKS):            # the sharpest book last, so it wins where it has a line
        sb = sharp[sharp['book'] == book][keys + ['fetched_at', 'line', 'p_over', 'position']].rename(
            columns={'line': '_l', 'p_over': '_p', 'position': '_pp', 'fetched_at': '_t'})
        if sb.empty:
            continue
        m = pd.merge_asof(legs[keys + ['fetched_at']].reset_index(), sb.sort_values('_t'), left_on='fetched_at',
                          right_on='_t', by=keys, direction='backward').set_index('index').reindex(legs.index)
        hit = m['_p'].notna()
        legs.loc[hit, 'sharp_book'] = book
        legs.loc[hit, 'sharp_line'] = m.loc[hit, '_l']
        legs.loc[hit, 'sharp_p'] = m.loc[hit, '_p']
        legs.loc[hit, '_spos'] = m.loc[hit, '_pp']
    legs = legs[legs['sharp_p'].notna()].copy()
    pos = legs['position'].where(legs['position'].astype(str).str.len() > 0, legs['_spos']).astype(str).str.upper()
    legs['Pos'] = pos
    legs['fair_over'] = np.nan
    for stat, idx in legs.groupby('stat').groups.items():
        g = legs.loc[idx]
        legs.loc[idx, 'fair_over'] = p_over_at(stat, g['Pos'].to_numpy(), g['sharp_line'].to_numpy(float),
                                               g['sharp_p'].to_numpy(float), g['line'].to_numpy(float))
    return legs.drop(columns=['_spos', '_cut'], errors='ignore').reset_index(drop=True)


def choose_sides(legs, payouts=None):
    """The side to play per leg (the one with the higher fair chance that the product offers), its fair chance, its
    break-even (best power entry / the side's multiplier; NaN for demons and goblins) and whether it clears it."""
    be = break_evens(payouts)
    out = legs.copy()
    over_ok = out['over_mult'].notna() & (out['over_mult'] > 0)
    under_ok = out['under_mult'].notna() & (out['under_mult'] > 0) & ~out['kind'].isin(OVER_ONLY)
    fair_o, fair_u = out['fair_over'], 1 - out['fair_over']
    base = out['book'].map({b: v[0] for b, v in be.items()})
    be_o = base / out['over_mult'].where(over_ok)
    be_u = base / out['under_mult'].where(under_ok)
    # the side whose fair chance beats its own break-even by more; without break-evens (demon/goblin), the over
    take_over = over_ok & (~under_ok | ((fair_o - be_o.fillna(0.5)) >= (fair_u - be_u.fillna(0.5))))
    out['side'] = np.where(take_over, 'over', np.where(under_ok, 'under', ''))
    out['fair'] = np.where(take_over, fair_o, fair_u)
    out['break_even'] = np.where(take_over, be_o, be_u)
    out.loc[out['kind'].isin(OVER_ONLY), 'break_even'] = np.nan
    out['flagged'] = out['fair'] >= out['break_even']
    out['edge'] = out['fair'] - out['break_even']
    return out[out['side'] != '']


def attach_results(legs, actuals):
    """Join each leg's result: `actuals` has week, _key and one column per stat. Pushes and missing results are dropped."""
    long = actuals.melt(id_vars=['week', '_key'], value_vars=[s for s in MARKET_STAT.values() if s in actuals.columns],
                        var_name='stat', value_name='actual').dropna(subset=['actual'])
    m = legs.merge(long.drop_duplicates(['week', '_key', 'stat']), on=['week', '_key', 'stat'], how='inner')
    m = m[m['actual'] != m['line']].copy()
    m['win'] = np.where(m['side'].eq('over'), m['actual'] > m['line'], m['actual'] < m['line'])
    return m


def model_agrees(legs, model_rows):
    """'agree' / 'disagree' / '' per leg: the model's P(over) at the sharp line (the record's 'at close' rows), moved to the
    pick'em line, on the leg's side of 1/2."""
    r = model_rows[['week', '_key', 'stat', 'line', 'p_over']].drop_duplicates(['week', '_key', 'stat']).rename(
        columns={'line': '_ml', 'p_over': '_mp'})
    m = legs.merge(r, on=['week', '_key', 'stat'], how='left')
    p = np.full(len(m), np.nan)
    for stat, idx in m.groupby('stat').groups.items():
        g = m.loc[idx]
        ok = g['_mp'].notna().to_numpy()
        if ok.any():
            pos = np.where(ok)[0]
            p[np.asarray(m.index.get_indexer(idx))[pos]] = p_over_at(
                stat, g['Pos'].to_numpy()[pos], g['_ml'].to_numpy(float)[pos], g['_mp'].to_numpy(float)[pos],
                g['line'].to_numpy(float)[pos])
    side_p = np.where(m['side'].eq('over'), p, 1 - p)
    out = legs.copy()
    out['model'] = np.where(np.isfinite(side_p), np.where(side_p > 0.5, 'agree', 'disagree'), '')
    return out


def leg_table(scored, by=('book', 'kind')):
    """Flagged legs per group: legs, hit rate (95% interval), mean fair chance, mean break-even, hit minus break-even with its
    90% lower bound, and 'surface' when that bound is above 0 on SURFACE_MIN_LEGS+ legs (else 'tracking'). Demon and goblin
    groups have no break-even: every leg is listed, with hit against fair chance (calibration)."""
    out = []
    for k, g in scored.groupby(list(by)):
        k = k if isinstance(k, tuple) else (k,)
        cal = g['break_even'].isna().all()
        f = g if cal else g[g['flagged']]
        rec = dict(zip(by, k))
        rec.update({'lines': int(len(g)), 'legs': int(len(f))})
        if len(f):
            w = f['win'].to_numpy(bool)
            lo, hi = wilson(int(w.sum()), len(f))
            rec.update({'hit': float(w.mean()), 'CI low': lo, 'CI high': hi, 'fair': float(f['fair'].mean())})
            if not cal:
                d = w.astype(float) - f['break_even'].to_numpy(float)
                lb = float(d.mean() - 1.645 * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan
                rec.update({'break_even': float(f['break_even'].mean()), 'hit - be': float(d.mean()), '90% low': lb,
                            'status': 'surface' if len(f) >= SURFACE_MIN_LEGS and np.isfinite(lb) and lb > 0
                            else 'tracking'})
            else:
                rec['status'] = 'calibration only'
            if 'model' in f.columns:
                for side in ('agree', 'disagree'):
                    s = f[f['model'] == side]
                    rec[f'{side} legs'] = int(len(s))
                    rec[f'{side} hit'] = float(s['win'].mean()) if len(s) else np.nan
        out.append(rec)
    return pd.DataFrame(out)
