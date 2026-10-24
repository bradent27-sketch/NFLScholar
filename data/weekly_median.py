"""
A median beside every weekly projection (2026-10-23, user: "the median version toggle that the cfb app has ... average for fantasy,
median as a floor reference and the median for O/U bets").

The board's numbers are MEANS (expected value - what adds up over a fantasy season). A median is the middle game: half of games land
above it, which is what an over/under line is priced against. For one player and stat:

    median = mean x ratio(position, stat, mean)        ratio = a + b * log(mean), clipped to RATIO_CLIP

fit by scripts/eval_stat_medians.py on the 2022-2025 weeks 3-17 base boards (played games, full role, at least that script's
MIN_PROJECTION of the stat): the median of what players at each projected level actually did, over their mean projection. Touchdowns,
and small counts with no fitted curve, use the Poisson median instead. Every median is CONTINUITY-CORRECTED (a count read as spread
evenly over [k - 0.5, k + 0.5]), so it compares directly with an x.5 line: a median above 4.5 receptions means P(5 or more) is over
one half.

Scored leave-one-season-out: the share of results above the median is 0.495-0.512 for every position and stat (the mean itself
scores 0.38-0.54), with an equal or lower absolute error everywhere. The player's own history (a steady player's median closer to his
mean, a boom-or-bust one's further below) was tested the same way and did not help out of sample at any shrinkage - so the median
depends on position and projected level only. A player's own spread still drives the width of his boom/bust band
(data/weekly_distribution.py).

Fantasy points: fit on Full PPR; applied to Half-PPR and Standard as is (the shape is set by the same games).
"""
import math

import numpy as np
import pandas as pd

# (position, stat) -> (a, b) of ratio = a + b * log(mean); 'points' is fantasy points
MEDIAN_CURVES = {
    ('QB', 'passing_attempts'): (1.8966, -0.2638),
    ('QB', 'passing_yards'): (2.6392, -0.3015),
    ('QB', 'points'): (0.6015, 0.1286),
    ('QB', 'rushing_attempts'): (1.0340, -0.1214),
    ('QB', 'rushing_yards'): (0.5579, 0.0506),
    ('RB', 'points'): (0.5665, 0.1307),
    ('RB', 'receiving_yards'): (0.8902, -0.0469),
    ('RB', 'receptions'): (0.9412, -0.0594),
    ('RB', 'rushing_attempts'): (0.8719, 0.0358),
    ('RB', 'rushing_yards'): (0.3610, 0.1428),
    ('RB', 'targets'): (0.9831, -0.0664),
    ('TE', 'points'): (0.6865, 0.1173),
    ('TE', 'receiving_yards'): (1.1510, -0.0455),
    ('TE', 'receptions'): (1.1826, -0.1122),
    ('TE', 'targets'): (1.2117, -0.1106),
    ('WR', 'points'): (0.6030, 0.1183),
    ('WR', 'receiving_yards'): (0.6646, 0.0668),
    ('WR', 'receptions'): (0.9823, -0.0049),
    ('WR', 'targets'): (1.0970, -0.0610),
}
RATIO_CLIP = (0.35, 1.25)
POISSON_STATS = ('passing_tds', 'rushing_tds', 'receiving_tds', 'passing_interceptions')
# a count with no fitted curve and a mean below this takes the Poisson median; above it the median is read as the mean
POISSON_FALLBACK_MAX_MEAN = 3.0
MEDIAN_STATS = ('passing_attempts', 'passing_completions', 'passing_yards', 'passing_tds', 'passing_interceptions',
                'rushing_attempts', 'rushing_yards', 'rushing_tds', 'targets', 'receptions', 'receiving_yards', 'receiving_tds')
SIDELINED = 0.01


def poisson_median(mean):
    """The continuity-corrected median of Poisson(mean): k - 0.5 + (0.5 - P(X <= k - 1)) / P(X = k) for the first k with
    P(X <= k) >= 0.5. 0 for a mean of 0."""
    lam = float(mean)
    if not math.isfinite(lam) or lam <= 0:
        return 0.0
    pmf = math.exp(-lam)
    below, k = 0.0, 0
    while below + pmf < 0.5:
        below += pmf
        k += 1
        pmf *= lam / k
    return k - 0.5 + (0.5 - below) / pmf


def median_ratio(position, stat, mean):
    """The fitted median / mean ratio at this projected level, or None when there is no curve for (position, stat)."""
    coef = MEDIAN_CURVES.get((str(position), stat))
    if coef is None:
        return None
    m = max(float(mean), 1e-6)
    return float(min(max(coef[0] + coef[1] * math.log(m), RATIO_CLIP[0]), RATIO_CLIP[1]))


def stat_median(position, stat, mean):
    """The median of one projected stat (or 'points') for a player at `position` whose mean projection is `mean`."""
    m = float(mean) if mean is not None and np.isfinite(mean) else 0.0
    if m <= 0:
        return 0.0
    if stat in POISSON_STATS:
        return poisson_median(m)
    ratio = median_ratio(position, stat, m)
    if ratio is not None:
        return m * ratio
    if stat != 'points' and m < POISSON_FALLBACK_MAX_MEAN:
        return poisson_median(m)
    return m


def attach_medians(board):
    """`board` with 'Median <stat>' beside every projected stat column and 'Median Proj Pts' beside Model Proj Pts. A player
    projected out (Availability at or below SIDELINED) has a median of 0. Nothing else changes."""
    if board is None or board.empty:
        return board
    out = board.copy()
    positions = out['Pos'].astype(str).to_numpy()
    avail = (pd.to_numeric(out['Availability'], errors='coerce').fillna(1.0).to_numpy()
             if 'Availability' in out.columns else np.ones(len(out)))
    live = avail > SIDELINED
    for stat in [s for s in MEDIAN_STATS if s in out.columns]:
        means = pd.to_numeric(out[stat], errors='coerce').fillna(0.0).to_numpy()
        out['Median ' + stat] = [round(stat_median(p, stat, m), 2) if ok else 0.0 for p, m, ok in zip(positions, means, live)]
    if 'Model Proj Pts' in out.columns:
        points = pd.to_numeric(out['Model Proj Pts'], errors='coerce').fillna(0.0).to_numpy()
        out['Median Proj Pts'] = [round(stat_median(p, 'points', m), 2) if ok else 0.0 for p, m, ok in zip(positions, points, live)]
    return out
