"""Points-total steps retired from the projection (wave 2 of the flag retirement, 2026-11-03; docs/retired_flags.md).

The season anchor (a calibrated total shrunk toward the player's own to-date mean), the expected-points luck step (a WR/TE total
pulled against how far earlier games ran ahead of nflverse's expected points) and their refits were SHIPPED, measured to help, and
retired 2026-11-02 because the stat recal plus v2_points_from_stat_line replace the weeks 3-17 total (the `lean` harness arm, pfs minus
these steps, tied pfs). They are not failed ideas: if the points total ever goes back to a calibration line, these are the corrections
that were worth having. Nothing in the model calls this module; it is kept because the calibration scripts (audit_calibration.py,
eval_*.py, fit_points_scale.py ...) replay and re-fit them, and data.weekly_projections re-exports the names so those scripts need
no change. The retired flag names (v2_season_anchor, v2_anchor_s_refit, v2_expected_points_luck, v2_luck_te_refit, v2_luck_qb) are no
longer in MODEL_FEATURES; the selectors below still read them from a feature set so an old replay can be reproduced.
"""
import numpy as np
import pandas as pd

from data.pass_capacity_allocator import SIDELINED_AVAILABILITY
from data.utils import clean_name_exact


# --- v2_season_anchor ---------------------------------------------------------
# The calibrated projection shrinks toward the POOL mean (WEEKLY_CALIBRATION), not toward the player's own
# average, so a WR/RB/TE who has played a few games is still moved too far from his own usual level. 2022-2025
# startable pool, played rows, the model's departure from the player's to-date mean (known before the game) against
# what then happened: slope WR 0.72 (se 0.04), RB 0.76 (0.04), TE 0.83 (0.09), QB 0.95 (0.08); right-sized in weeks
# 3-6 and 0.5-0.7 from week 7 (scripts/analyze_model_weak_spots.py). Shrinking toward the to-date mean,
#     final = mean + s * (calibrated - mean),   s fit by least squares per position,
# improved held-out RMSE in all six year splits (mean -0.027 ALL, WR -0.050 in every split, TE every split, RB 5/6)
# - scripts/eval_season_anchor.py. QB (s ~ 0.98) is left alone. A phase-by-phase s was unstable across fits and
# is not used. Points total only, like the calibration itself: the stat line is untouched.
SEASON_ANCHOR_S = {'RB': 0.75, 'WR': 0.69, 'TE': 0.84}
SEASON_ANCHOR_MIN_GAMES = 2


def apply_season_anchor(result, hist, name_col, factors=None, min_games=None):
    """(result, info): shrink each eligible row's calibrated 'Model Proj Pts' toward the player's mean fantasy
    points over his games BEFORE the target week (``hist``, scoring-aware 'fantasy_points'). Eligible: a position in
    ``factors``, at least ``min_games`` played games, not sidelined, and a non-zero projection. ``info`` maps
    (Player, Pos, Team) -> {'mean', 'games', 'factor', 'before', 'after'} for the rows changed."""
    factors = SEASON_ANCHOR_S if factors is None else factors
    min_games = SEASON_ANCHOR_MIN_GAMES if min_games is None else min_games
    if (result is None or result.empty or hist is None or hist.empty or name_col not in hist.columns
            or 'fantasy_points' not in hist.columns or 'Model Proj Pts' not in result.columns):
        return result, {}
    pts = pd.to_numeric(hist['fantasy_points'], errors='coerce')
    grouped = pts.groupby(hist[name_col]).agg(['mean', 'count'])
    out = result.copy()
    mean = out['Player'].map(grouped['mean'])
    games = out['Player'].map(grouped['count']).fillna(0)
    factor = out['Pos'].map(factors)
    cal = pd.to_numeric(out['Model Proj Pts'], errors='coerce')
    avail = (pd.to_numeric(out['Availability'], errors='coerce').fillna(1.0)
             if 'Availability' in out.columns else pd.Series(1.0, index=out.index))
    eligible = (factor.notna() & mean.notna() & (games >= min_games) & (cal > 0.005) & (avail > SIDELINED_AVAILABILITY))
    if not eligible.any():
        return out, {}
    new = np.round(np.clip(mean + factor * (cal - mean), 0.0, None), 2)
    info = {}
    for idx in out.index[eligible]:
        info[(out.at[idx, 'Player'], out.at[idx, 'Pos'], out.at[idx, 'Team'])] = {
            'mean': float(mean[idx]), 'games': int(games[idx]), 'factor': float(factor[idx]),
            'before': float(cal[idx]), 'after': float(new[idx])}
    for col in ('Model Proj Pts', 'Calibrated Model Proj Pts'):
        if col in out.columns:
            out.loc[eligible, col] = new[eligible]
    return out, info


# --- v2_expected_points_luck ----------------------------------------------------------------------------------------
# nflverse's ff_opportunity gives every player-game an EXPECTED point total from the context of his touches. A receiver
# whose earlier games ran ahead of that ("luck" = actual - expected, per game) is over-projected the next week, one that ran
# behind is under-projected. 2021-2025 weeks 3-17, shipped board, played rows: the miss (actual - projected) moves by -0.18
# points per point of mean luck for WR (every season, CI [-0.26, -0.10]) and -0.16 for TE (4 of 5 seasons); RB shows nothing
# (-0.06, CI spans 0) and a QB correction made START-QB worse in held-out runs, so only WR and TE are corrected. Held out by
# season (scripts/eval_expected_points.py, b fit on the other four seasons, harness-style paired START pools): START-ALL
# -0.0055 [-0.013, +0.001], START-WR -0.010, START-TE -0.010, ALL -0.0045 [-0.009, -0.000], better in 4 of 4 seasons. No
# games-played shrink was best (a K of 4 or 8 only lost gain). Points total only, like the anchor and the Questionable cut: the
# stat line is untouched. A player with no expected-points row yet this season (a rookie debut, a name the feed lacks) is left
# alone. Caveat: the expectation model is trained on all seasons, so past-season luck is in-sample for it. 2026-10-28: on today's
# model the QB correction helps (2024-2026) and ships centered on the average QB's luck ('v2_luck_qb', EXPECTED_POINTS_LUCK_QB).
EXPECTED_POINTS_LUCK = {'WR': -0.18, 'TE': -0.16}


def apply_expected_points_luck(result, table, factors=None, centers=None):
    """(result, info): add ``factors[pos] * (luck_pg - centers[pos])`` (center 0 for a position not in ``centers``,
    default EXPECTED_POINTS_LUCK_CENTER) to the point total of each eligible row. ``table`` is
    data.expected_points.luck_per_game (index = clean player name, columns games / luck_pg). Eligible: a position in
    ``factors``, at least one earlier game in the table, not sidelined (Availability above SIDELINED_AVAILABILITY) and a
    non-zero total. ``info`` maps (Player, Pos, Team) -> {'luck_pg', 'games', 'factor', 'center', 'before', 'after'} for
    the rows changed."""
    factors = EXPECTED_POINTS_LUCK if factors is None else factors
    centers = EXPECTED_POINTS_LUCK_CENTER if centers is None else centers
    if result is None or result.empty or table is None or len(table) == 0 or 'Model Proj Pts' not in result.columns:
        return result, {}
    out = result.copy()
    keys = clean_name_exact(out['Player'].astype(str))
    luck = keys.map(table['luck_pg'])
    games = keys.map(table['games']).fillna(0)
    factor = out['Pos'].map(factors)
    center = out['Pos'].map(centers).fillna(0.0)
    avail = (pd.to_numeric(out['Availability'], errors='coerce').fillna(1.0)
             if 'Availability' in out.columns else pd.Series(1.0, index=out.index))
    pts = pd.to_numeric(out['Model Proj Pts'], errors='coerce')
    eligible = luck.notna() & factor.notna() & (games >= 1) & (avail > SIDELINED_AVAILABILITY) & (pts > 0.005)
    if not eligible.any():
        return out, {}
    new = np.round(np.clip(pts + factor * (luck - center), 0.0, None), 2)
    info = {}
    for idx in out.index[eligible]:
        info[(out.at[idx, 'Player'], out.at[idx, 'Pos'], out.at[idx, 'Team'])] = {
            'luck_pg': float(luck[idx]), 'games': int(games[idx]), 'factor': float(factor[idx]),
            'center': float(center[idx]), 'before': float(pts[idx]), 'after': float(new[idx])}
    for col in ('Model Proj Pts', 'Calibrated Model Proj Pts'):
        if col in out.columns:
            out.loc[eligible, col] = new[eligible]
    return out, info


# 'v2_luck_te_refit' (SHIPPED 2026-10-28, calibration audit): the TE luck factor re-fit on today's model. Re-fit in sample on
# 2021-2025 it is -0.33, not -0.16 (WR -0.21, about as shipped); forward-chained with only TE re-fit: START-TE -0.0092, START-ALL
# -0.0012 [-0.0033, +0.0006], better in 2023, 2024 and 2025; 2026 weeks 3-7: START-TE -0.076, START-ALL -0.011. Under the
# weaker anchor ('v2_anchor_s_refit') the in-sample fit drops to -0.24 (2021-2025 START-TE -0.004 against -0.33) but 2026 weeks
# 3-7 still want -0.33 (START-TE +0.018 at -0.24), and the fit has grown more negative season by season: kept at -0.33.
EXPECTED_POINTS_LUCK_TE_REFIT = -0.33

# 'v2_luck_qb' (SHIPPED 2026-10-28, user: "the QB luck correlation is really close and generally helps ... probably worthwhile";
# MISSED its pre-set bar by 0.0001 on 2023). The luck step at QB on today's model: the miss of a played START-pool QB moves by
# -0.11 per point of earlier-game luck (2021-2025 in sample; by season 2021 -0.18, 2022 +0.08, 2023 -0.02, 2024 -0.18, 2025
# -0.50, 2026 -0.30). Unlike WR/TE, a QB's luck runs below 0 on average (the feed over-expects QBs: mean -0.50 on the fit rows,
# 2022 -1.26, 2025 +0.08), so the plain step would nudge every QB up (START-QB bias +0.32 -> +0.38) and, at actual 0, every late
# scratch; the step is centered on that mean, + b x (luck - center), which keeps the bias where it was. Forward-chained (fit on
# earlier seasons, centered) against today's pipeline: START-QB -0.0072 [-0.0151, +0.0007], START-ALL -0.0012, by season 2023
# +0.0002, 2024 -0.0006, 2025 -0.0033; EVERY-QB -0.0019; 2026 weeks 3-7 START-QB -0.027, START-ALL -0.0044. In sample at
# -0.11 2022 and 2023 are worse at QB (+0.028, +0.006). Typical move +-0.3 points (QB luck sd 2.8 per game).
EXPECTED_POINTS_LUCK_QB = -0.11
EXPECTED_POINTS_LUCK_CENTER = {'QB': -0.50}

# 'v2_anchor_s_refit' (SHIPPED 2026-10-28, user: "go ahead and weaken the season anchor"; MISSED its pre-set bar): the season
# anchor's s re-fit on today's model (fit 2021-2025: RB 0.88, WR 0.82, TE 1.03, capped at 1 = no pull). Since the anchor shipped
# (2026-10-06) the raw projection has taken in silent games, participation and share recency, and the newer seasons want a weaker
# pull toward the player's own mean: forward-chained against today's pipeline 2023 +0.0119 (worse in every phase), 2024 -0.0079, 2025 -0.0102; 2026 weeks 3-7
# -0.0356 [-0.0399, -0.0268] (RB -0.048, WR -0.030, TE -0.065). An s by games-in-the-mean failed (unstable). Half the move
# (RB 0.815, WR 0.755, TE 0.92) costs 2021/2023 less (+0.0028/+0.0020 START-ALL against +0.0109/+0.0079) but gains less in
# every season since (2024 -0.0062, 2025 -0.0069, 2026 -0.0188 against -0.0081/-0.0099/-0.0284), and a fit on 2024-2025 alone is
# weaker still (RB 0.92, WR 0.88, TE 1.13): the full re-fit shipped.
SEASON_ANCHOR_S_REFIT = {'RB': 0.88, 'WR': 0.82, 'TE': 1.0}


def season_anchor_factors(feats):
    """The anchor's s by position: SEASON_ANCHOR_S_REFIT when 'v2_anchor_s_refit' is in ``feats``, else None (SEASON_ANCHOR_S)."""
    return dict(SEASON_ANCHOR_S_REFIT) if 'v2_anchor_s_refit' in feats else None


def expected_points_luck_factors(feats):
    """The luck step's factor by position: EXPECTED_POINTS_LUCK, with TE re-fit under 'v2_luck_te_refit' and QB added under
    'v2_luck_qb'."""
    factors = dict(EXPECTED_POINTS_LUCK)
    if 'v2_luck_te_refit' in feats:
        factors['TE'] = EXPECTED_POINTS_LUCK_TE_REFIT
    if 'v2_luck_qb' in feats:
        factors['QB'] = EXPECTED_POINTS_LUCK_QB
    return factors
