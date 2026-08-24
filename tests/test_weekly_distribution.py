"""Offline tests for data.weekly_distribution - the per-player boom/bust band."""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import weekly_distribution as wd  # noqa: E402


def test_needs_refit_withholds_a_band_rather_than_fabricating_one():
    original_needs_refit = wd.NEEDS_REFIT
    try:
        wd.NEEDS_REFIT = True
        assert wd.player_distribution('WR', 3, 15.0, role_confidence=0.8) is None
    finally:
        wd.NEEDS_REFIT = original_needs_refit


def test_tier_for_rank_matches_the_backtest_bucket_boundaries():
    assert wd.tier_for_rank(1) == 'Starter'
    assert wd.tier_for_rank(15) == 'Starter'
    assert wd.tier_for_rank(30) == 'Starter'
    assert wd.tier_for_rank(31) == 'Streamer/Bench'
    assert wd.tier_for_rank(None) == 'Streamer/Bench'
    assert wd.tier_for_rank(float('nan')) == 'Streamer/Bench'


def test_player_distribution_scales_calibrated_points_by_the_fitted_ratios():
    original_bands, original_needs_refit = wd.WEEKLY_DISTRIBUTION_BANDS, wd.NEEDS_REFIT
    try:
        wd.NEEDS_REFIT = False
        wd.WEEKLY_DISTRIBUTION_BANDS = {
            'WR': {'Starter': {10: 0.30, 25: 0.60, 50: 1.00, 75: 1.30, 90: 1.70}},
        }
        result = wd.player_distribution('WR', 10, calibrated_points=10.0, role_confidence=None)
        assert result is not None
        assert result['tier'] == 'Starter'
        # role_confidence=None -> width_scale 1.0 -> points match the raw ratios exactly.
        assert math.isclose(result['width_scale'], 1.0)
        assert math.isclose(result['points'][50], 10.0)
        assert math.isclose(result['points'][10], 3.0)
        assert math.isclose(result['points'][90], 17.0)
        for pct in result['points']:
            assert math.isclose(result['points'][pct], result['position_points'][pct])
    finally:
        wd.WEEKLY_DISTRIBUTION_BANDS, wd.NEEDS_REFIT = original_bands, original_needs_refit


def test_high_role_confidence_narrows_the_band_around_its_own_median():
    original_bands, original_needs_refit = wd.WEEKLY_DISTRIBUTION_BANDS, wd.NEEDS_REFIT
    try:
        wd.NEEDS_REFIT = False
        wd.WEEKLY_DISTRIBUTION_BANDS = {
            'RB': {'Starter': {10: 0.40, 25: 0.70, 50: 1.00, 75: 1.30, 90: 1.60}},
        }
        confident = wd.player_distribution('RB', 2, calibrated_points=20.0, role_confidence=1.0)
        uncertain = wd.player_distribution('RB', 2, calibrated_points=20.0, role_confidence=0.0)
        # Median is unaffected by width scaling either way.
        assert math.isclose(confident['points'][50], 20.0)
        assert math.isclose(uncertain['points'][50], 20.0)
        # A fully-established role narrows the band; a totally unestablished
        # one widens it - both relative to the position_points overlay, which
        # never moves regardless of role_confidence.
        assert confident['points'][90] < confident['position_points'][90]
        assert uncertain['points'][90] > uncertain['position_points'][90]
        assert confident['points'][90] < uncertain['points'][90]
        assert confident['points'][10] > uncertain['points'][10]
        assert math.isclose(confident['position_points'][90], 32.0)
    finally:
        wd.WEEKLY_DISTRIBUTION_BANDS, wd.NEEDS_REFIT = original_bands, original_needs_refit


def test_band_points_never_go_negative_even_with_a_low_floor_and_wide_scale():
    original_bands, original_needs_refit = wd.WEEKLY_DISTRIBUTION_BANDS, wd.NEEDS_REFIT
    try:
        wd.NEEDS_REFIT = False
        wd.WEEKLY_DISTRIBUTION_BANDS = {
            'TE': {'Streamer/Bench': {10: 0.0, 25: 0.1, 50: 0.6, 75: 1.6, 90: 2.9}},
        }
        result = wd.player_distribution('TE', 40, calibrated_points=2.0, role_confidence=0.0)
        assert all(v >= 0.0 for v in result['points'].values())
    finally:
        wd.WEEKLY_DISTRIBUTION_BANDS, wd.NEEDS_REFIT = original_bands, original_needs_refit


def test_sample_from_band_respects_the_measured_median_and_stays_within_a_wide_tail():
    points = {10: 4.0, 25: 8.0, 50: 12.0, 75: 16.0, 90: 20.0}
    samples = wd.sample_from_band(points, n=5000)
    assert len(samples) == 5000
    assert abs(np.median(samples) - 12.0) < 1.0
    # Extrapolated 1/99 tails hold the outermost measured slope, so samples
    # should stay well within a generous multiple of the measured spread -
    # this is a sanity bound against a runaway extrapolation, not a precise claim.
    spread = points[90] - points[10]
    assert samples.min() > points[10] - 2 * spread
    assert samples.max() < points[90] + 2 * spread


def test_sample_from_band_is_deterministic_across_calls():
    points = {10: 4.0, 25: 8.0, 50: 12.0, 75: 16.0, 90: 20.0}
    a = wd.sample_from_band(points, n=200)
    b = wd.sample_from_band(points, n=200)
    assert np.array_equal(a, b)


def test_kde_curve_peaks_near_the_sample_center_and_integrates_to_roughly_one():
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=12.0, scale=2.0, size=3000)
    grid = np.linspace(0.0, 24.0, 400)
    density = wd.kde_curve(samples, grid)
    peak_location = grid[int(np.argmax(density))]
    assert abs(peak_location - 12.0) < 1.5
    # Trapezoidal integral of a density over its support should be close to 1.
    integral = np.trapezoid(density, grid)
    assert 0.85 < integral < 1.15


def test_shipped_bands_are_internally_monotonic_and_needs_refit_is_off():
    """Guards the pasted-in fit output itself, not just the mechanism around
    it - a future bad paste (wrong column order, stale run) should fail here
    rather than silently ship a backwards or flat band."""
    assert wd.NEEDS_REFIT is False
    assert set(wd.WEEKLY_DISTRIBUTION_BANDS) == {'QB', 'RB', 'WR', 'TE'}
    for position, tiers in wd.WEEKLY_DISTRIBUTION_BANDS.items():
        for tier, ratios in tiers.items():
            ordered = [ratios[p] for p in (10, 25, 50, 75, 90)]
            assert ordered == sorted(ordered), f"{position}/{tier} percentiles are not monotonic: {ordered}"
            assert 0.5 <= ratios[50] <= 1.5, f"{position}/{tier} median ratio {ratios[50]} is implausibly far from 1.0"


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failures.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
