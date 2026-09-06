"""Turn a posted prop line + its de-vigged P(over) into the mean the market implies.

A sportsbook line is the point where the book wants equal money on each
side, priced with two-sided odds. Two separate things push that number away
from a fantasy MEAN:

  1. THE VIG LEAN. "1.5 passing TDs, Over -140 / Under +115" is not a coin
     flip on 1.5. De-vigged, the over is ~56% - the book's real number is
     higher than 1.5. Only an exactly even market (Over/Under priced the
     same) has its listed number as the central estimate.

  2. THE MEDIAN<MEAN SKEW. Even a genuinely even line is a MEDIAN. For a
     right-skewed count (touchdowns, receptions) the mean sits above it: an
     even 1.5 passing-TD line (P(>=2) = 0.5) implies a Poisson mean of ~1.68,
     not 1.5.

Both fall out of one step: pick a distribution for the stat, and solve for
the parameter that puts the de-vigged tail probability where the market put
it. That parameter IS the mean.

  - COUNTS (TDs, receptions, INTs, completions, attempts): Poisson. Solve
    P(Poisson(lambda) >= ceil(line)) = p_over  ->  lambda.
  - YARDS: Normal. mu = line + Phi^-1(p_over) * sigma. p_over = 0.5 -> mu =
    line exactly (a yardage line really is ~symmetric, so the skew term is
    zero and only the vig lean moves it).

No p_over (PrizePicks' bare board, a book that didn't publish prices): fall
back to line * MEDIAN_TO_MEAN, the old timid multiplier, so those sources
degrade to exactly the previous behaviour instead of guessing at odds.

Dependency-free on purpose (no scipy): Acklam's rational approximation for
the inverse normal CDF, a short bisection on the Poisson upper tail. Both
are accurate well past what a betting line's precision could justify.
"""
import math

# Stat -> distribution family. Anything not listed falls back to the
# multiplier path (see implied_mean_from_line).
_COUNT_STATS = frozenset({
    'passing_tds', 'rushing_tds', 'receiving_tds', 'receptions',
    'passing_interceptions', 'passing_completions', 'passing_attempts',
    'rushing_attempts', 'targets',
})
_YARD_STATS = frozenset({'passing_yards', 'rushing_yards', 'receiving_yards'})

# Rough single-GAME standard deviation per (position, yard stat). Only scales
# the SIZE of the vig-lean shift, and only when a line is not evenly priced -
# an even line moves zero regardless of sigma - so a rough figure is fine and
# in keeping with this module's "don't manufacture precision" rule. Season
# lines scale these by sqrt(17).
_YARD_SIGMA_GAME = {
    ('QB', 'passing_yards'): 74.0,
    ('QB', 'rushing_yards'): 18.0,
    ('RB', 'rushing_yards'): 31.0,
    ('RB', 'receiving_yards'): 20.0,
    ('WR', 'receiving_yards'): 36.0,
    ('TE', 'receiving_yards'): 26.0,
}
_YARD_SIGMA_DEFAULT = 32.0
_SEASON_GAMES = 17.0

# Fallback multiplier for a line with no usable P(over). Matches the old
# data.odds_projections.MEDIAN_TO_MEAN exactly.
MEDIAN_TO_MEAN_FALLBACK = {
    'passing_tds': 1.02, 'rushing_tds': 1.05, 'receiving_tds': 1.05,
}

_EPS = 1e-6


def norm_ppf(p):
    """Inverse standard-normal CDF. Acklam's rational approximation;
    absolute error < 1.15e-9 across (0, 1)."""
    if p <= 0.0:
        return float('-inf')
    if p >= 1.0:
        return float('inf')
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def _poisson_upper_tail(k, mu):
    """P(Poisson(mu) >= k), computed as 1 - sum_{i<k} pmf(i)."""
    if k <= 0:
        return 1.0
    if mu <= 0.0:
        return 0.0
    term = math.exp(-mu)      # i = 0
    cdf = term
    for i in range(1, k):
        term *= mu / i
        cdf += term
    return min(1.0, max(0.0, 1.0 - cdf))


def poisson_mean_for_upper_tail(line, p_over, iters=100, tol=1e-7):
    """lambda such that P(Poisson(lambda) >= ceil(line)) == p_over.

    The upper tail is strictly increasing in lambda, so a bounded bisection
    converges cleanly. ``line`` is the posted number (usually X.5, so
    ceil(line) is the smallest winning count for the over)."""
    k = int(math.ceil(line - 1e-9))
    if k <= 0:
        return max(0.0, float(line))
    p = min(max(float(p_over), _EPS), 1 - _EPS)
    lo, hi = 1e-6, k + 10.0 * math.sqrt(k) + 20.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _poisson_upper_tail(k, mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def _yard_sigma(position, market, period):
    sigma = _YARD_SIGMA_GAME.get((str(position).upper(), market), _YARD_SIGMA_DEFAULT)
    if period == 'season':
        sigma *= math.sqrt(_SEASON_GAMES)
    return sigma


def implied_mean_from_line(line, p_over, market, position=None, period='game'):
    """The market's implied MEAN for one posted line.

    ``p_over`` is the de-vigged probability the over hits (0..1), or None.
    ``period`` is 'game' or 'season' and only affects the yardage sigma.

    None / non-finite p_over  -> line * MEDIAN_TO_MEAN_FALLBACK.get(market, 1)
    count stat  -> Poisson upper-tail inversion (handles skew AND vig lean)
    yard stat   -> line + Phi^-1(p_over) * sigma  (vig lean only; even -> line)
    other       -> line unchanged
    """
    try:
        value = float(line)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None

    usable_p = p_over is not None
    if usable_p:
        try:
            p = float(p_over)
            usable_p = math.isfinite(p) and 0.0 < p < 1.0
        except (TypeError, ValueError):
            usable_p = False

    if not usable_p:
        return value * MEDIAN_TO_MEAN_FALLBACK.get(market, 1.0)

    if market in _COUNT_STATS:
        return poisson_mean_for_upper_tail(value, p)
    if market in _YARD_STATS:
        return value + norm_ppf(p) * _yard_sigma(position, market, period)
    return value
