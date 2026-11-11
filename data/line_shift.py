"""
A book's probability moved to another line (2026-11-11, audits B5 and B6). The stat's empirical shape (data/prop_probability.py:
the actual / mean ratio quantiles, by position and mean level) is centred so that P(over the book's line) equals the book's
no-vig price, then read at the other line. No model projection enters: the mean is solved from the price alone.

Used for the fair probability of a pick'em line from the sharp book's price (B5), and for the closing fair probability of a bet
whose line moved before the close (B6 price value).
"""
from __future__ import annotations

import numpy as np

from data import prop_probability as pp

ITERATIONS = 40


def implied_mean(stat, pos, line, p_over, iterations=ITERATIONS):
    """The mean at which the stat's empirical shape gives P(over `line`) = `p_over`, per row (bisection on a log scale; P(over)
    rises with the mean). NaN where the engine has no table for the position or the inputs are missing."""
    line = np.asarray(line, dtype=float)
    target = np.asarray(p_over, dtype=float)
    pos = np.asarray(pos).astype(str)
    lo = np.full(len(line), np.log(0.05))
    hi = np.log(np.maximum(line, 1.0) * 6.0 + 5.0)
    ok = np.isfinite(line) & np.isfinite(target) & (target > 0) & (target < 1)
    for _ in range(iterations):
        mid = (lo + hi) / 2
        p = pp.prob_over(stat, pos, np.exp(mid), line)
        up = p < target                       # P(over) too low: the mean must rise
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    mean = np.exp((lo + hi) / 2)
    check = pp.prob_over(stat, pos, mean, line)
    return np.where(ok & np.isfinite(check), mean, np.nan)


def p_over_at(stat, pos, from_line, from_p_over, to_line):
    """P(over `to_line`) implied by a no-vig P(over `from_line`) = from_p_over, per row. Where the two lines are equal it is
    from_p_over itself."""
    from_line = np.asarray(from_line, dtype=float)
    to_line = np.asarray(to_line, dtype=float)
    from_p = np.asarray(from_p_over, dtype=float)
    same = np.isclose(from_line, to_line)
    out = np.where(same, from_p, np.nan)
    move = ~same & np.isfinite(from_line) & np.isfinite(to_line) & np.isfinite(from_p)
    if move.any():
        idx = np.where(move)[0]
        pos = np.asarray(pos).astype(str)
        mean = implied_mean(stat, pos[idx], from_line[idx], from_p[idx])
        out[idx] = pp.prob_over(stat, pos[idx], mean, to_line[idx])
    return out
