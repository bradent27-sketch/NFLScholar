"""Per-player boom/bust percentile bands from this model's own backtest residuals.

WHY THIS EXISTS. A single "Model Proj Pts" number is a median-ish guess, and
reads that way whether it comes from a high-floor possession receiver or a
boom/bust deep threat - the same 12.0 means something different for each.
Per the user's own framing: not an arbitrary +/-X% band, but a real
distribution grounded in how this model has actually missed in the past.

WHERE THE SHAPE COMES FROM. scripts/fit_weekly_distribution.py backtests the
V2 model on 2021-2023 (out-of-sample from the 2024-2025 seasons this model is
evaluated on - same discipline as WEEKLY_CALIBRATION in data.weekly_
projections, which this ALSO depends on: the ratios below are actual points
divided by CALIBRATED points, so a stale calibration line changes what they
mean). WEEKLY_DISTRIBUTION_BANDS stores, per position x usage tier (the same
Elite/Starter/Flex/Streamer buckets scripts/backtest_five_weeks.py reports
on), the empirical P10/P25/P50/P75/P90 of that ratio. Multiplying a band onto
any player's own calibrated points gives that tier's typical spread - this is
the "typical distribution by position" reference line.

PERSONALIZING IT. A flat per-tier band is still not "this specific player's"
distribution - a well-established, high-snap-share role is more predictable
than a thin or newly-changed one. width_scale below stretches or compresses
the fitted band around its own median using role_confidence, which this
model already computes for every player (see build_weekly_projections'
'role_confidence' trace). This scaling is a bounded, monotonic HEURISTIC on
top of an empirically fit base - clearly labelled as such below, not claimed
to be measured with the same rigor as the percentiles themselves.
"""

from __future__ import annotations

import numpy as np

# Fitted by scripts/fit_weekly_distribution.py --years 2021,2022,2023 (fit
# window strictly out-of-sample from the 2024-2025 seasons this model is
# evaluated on). Re-run the script (and replace this block) whenever
# WEEKLY_CALIBRATION is re-fit - these ratios are of actual-vs-CALIBRATED
# points, so a new calibration line changes what they mean.
#
# QB 'Streamer/Bench' has no band: only n=2 backtested player-weeks ever
# landed there (qb1_override projects real backup QBs at ~0, by design - see
# that feature's own docstring - so there is almost never a meaningful QB
# projection this deep to backtest against). player_distribution() returns
# None for that cell rather than fabricate one from 2 data points; the
# decomposition dialog shows "not available" for those rows, same as any
# other genuinely-missing input in this app.
WEEKLY_DISTRIBUTION_BANDS = {
    'QB': {
        'Elite': {10: 0.531, 25: 0.763, 50: 1.022, 75: 1.326, 90: 1.603},  # n=224
        'Starter': {10: 0.435, 25: 0.697, 50: 0.992, 75: 1.247, 90: 1.593},  # n=325
        'Flex': {10: 0.367, 25: 0.637, 50: 0.939, 75: 1.348, 90: 1.701},  # n=384
    },
    'RB': {
        'Elite': {10: 0.361, 25: 0.547, 50: 0.889, 75: 1.234, 90: 1.518},  # n=200
        'Starter': {10: 0.339, 25: 0.573, 50: 0.990, 75: 1.404, 90: 1.697},  # n=286
        'Flex': {10: 0.321, 25: 0.605, 50: 1.008, 75: 1.432, 90: 1.933},  # n=481
        'Streamer/Bench': {10: 0.084, 25: 0.440, 50: 1.062, 75: 2.079, 90: 3.758},  # n=1584
    },
    'WR': {
        'Elite': {10: 0.325, 25: 0.580, 50: 0.847, 75: 1.321, 90: 1.688},  # n=212
        'Starter': {10: 0.361, 25: 0.617, 50: 0.907, 75: 1.340, 90: 1.864},  # n=312
        'Flex': {10: 0.333, 25: 0.556, 50: 0.940, 75: 1.405, 90: 1.910},  # n=501
        'Streamer/Bench': {10: 0.000, 25: 0.381, 50: 0.901, 75: 1.698, 90: 2.621},  # n=2882
    },
    'TE': {
        'Elite': {10: 0.319, 25: 0.547, 50: 0.903, 75: 1.329, 90: 1.926},  # n=198
        'Starter': {10: 0.234, 25: 0.507, 50: 0.989, 75: 1.500, 90: 2.082},  # n=289
        'Flex': {10: 0.227, 25: 0.428, 50: 0.921, 75: 1.625, 90: 2.373},  # n=456
        'Streamer/Bench': {10: 0.000, 25: 0.394, 50: 0.900, 75: 1.809, 90: 2.869},  # n=750
    },
}
NEEDS_REFIT = False

TIER_BOUNDS = (('Elite', 1, 6), ('Starter', 7, 15), ('Flex', 16, 30), ('Streamer/Bench', 31, 10_000))
MIN_WIDTH_SCALE, MAX_WIDTH_SCALE = 0.7, 1.4


def tier_for_rank(rank_within_position: int | float | None) -> str:
    """Which usage tier a projected positional rank falls into - the SAME
    cutoffs scripts/backtest_five_weeks.py and fit_weekly_distribution.py
    both bucket on, so a player's band always matches the tier its own
    accuracy was actually measured in."""
    if rank_within_position is None or not np.isfinite(rank_within_position):
        return 'Streamer/Bench'
    rank = int(rank_within_position)
    for name, lo, hi in TIER_BOUNDS:
        if lo <= rank <= hi:
            return name
    return 'Streamer/Bench'


def position_band_ratios(position: str, rank_within_position) -> dict[int, float] | None:
    """The unadjusted position/tier ratio ladder - this IS the 'typical
    distribution by position' overlay the decomposition dialog draws as a
    dotted reference line next to a player's own (personalized) band."""
    tier = tier_for_rank(rank_within_position)
    return WEEKLY_DISTRIBUTION_BANDS.get(str(position), {}).get(tier)


def _width_scale(role_confidence: float | None) -> float:
    if role_confidence is None or not np.isfinite(role_confidence):
        return 1.0
    # role_confidence in [0,1]; 1.0 (fully established role) -> 0.7x
    # (narrower than the tier average), 0.0 (no established role) -> 1.3x
    # (wider). Centered so a mid-confidence player gets back the fitted
    # tier band essentially unchanged.
    return float(np.clip(1.3 - 0.6 * float(role_confidence), MIN_WIDTH_SCALE, MAX_WIDTH_SCALE))


def player_distribution(position: str, rank_within_position, calibrated_points: float,
                        role_confidence: float | None = None) -> dict | None:
    """This player's own boom/bust band, in real points.

    Returns {'tier', 'width_scale', 'points': {10: v, 25: v, 50: v, 75: v, 90: v},
    'position_points': {...same shape, unadjusted, for the overlay...}} or
    None when this position/tier has no fitted band (e.g. NEEDS_REFIT, or too
    few backtested players in a tier to trust - see the fit script's own
    n>=30 floor).
    """
    if NEEDS_REFIT:
        return None
    ratios = position_band_ratios(position, rank_within_position)
    if not ratios or calibrated_points is None or not np.isfinite(calibrated_points):
        return None
    scale = _width_scale(role_confidence)
    median_ratio = ratios[50]
    points = {}
    position_points = {}
    for p, r in ratios.items():
        position_points[p] = max(0.0, calibrated_points * r)
        scaled_ratio = median_ratio + (r - median_ratio) * scale
        points[p] = max(0.0, calibrated_points * scaled_ratio)
    return {
        'tier': tier_for_rank(rank_within_position),
        'width_scale': round(scale, 3),
        'points': points,
        'position_points': position_points,
    }


def sample_from_band(points: dict, n: int = 4000, seed: int = 42) -> np.ndarray:
    """Inverse-transform samples from a {10,25,50,75,90: value} band, via a
    piecewise-linear quantile function extended to (1, 99) by holding the
    outermost measured slope - NOT a claim that P1/P99 are separately
    measured, just a finite, reasonable tail so a downstream density estimate
    doesn't need to guess what happens past P90. A fixed seed keeps the
    rendered curve stable across Streamlit reruns of the same player/data
    instead of visibly jittering on every rerun.
    """
    pcts = sorted(points.keys())
    vals = [points[p] for p in pcts]
    lo_slope = (vals[1] - vals[0]) / max(pcts[1] - pcts[0], 1e-6)
    hi_slope = (vals[-1] - vals[-2]) / max(pcts[-1] - pcts[-2], 1e-6)
    ext_pcts = [1.0] + list(pcts) + [99.0]
    ext_vals = [max(0.0, vals[0] - lo_slope * (pcts[0] - 1.0))] + list(vals) + [
        max(vals[-1], vals[-1] + hi_slope * (99.0 - pcts[-1]))]
    rng = np.random.default_rng(seed)
    u = rng.uniform(1.0, 99.0, size=n)
    return np.interp(u, ext_pcts, ext_vals)


def kde_curve(samples: np.ndarray, grid: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    """Gaussian KDE, hand-rolled (this project has no scipy dependency - see
    scripts/eval_weekly_model.py's own note on Spearman-via-Pearson-on-ranks
    for the same constraint elsewhere). Silverman's rule of thumb for
    bandwidth, floored so a very tight sample doesn't render as a spike."""
    samples = np.asarray(samples, dtype=float)
    if bandwidth is None:
        std = float(samples.std()) or 1.0
        bandwidth = max(1.06 * std * len(samples) ** (-1 / 5), 0.5)
    diffs = (grid[:, None] - samples[None, :]) / bandwidth
    weights = np.exp(-0.5 * diffs ** 2)
    return weights.sum(axis=1) / (len(samples) * bandwidth * np.sqrt(2 * np.pi))
