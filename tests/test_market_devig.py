"""data.market_devig: posted line + de-vigged P(over) -> implied market mean."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.market_devig import (  # noqa: E402
    implied_mean_from_line, norm_ppf, poisson_mean_for_upper_tail,
    _poisson_upper_tail,
)


def test_norm_ppf_known_points():
    assert abs(norm_ppf(0.5) - 0.0) < 1e-9
    assert abs(norm_ppf(0.975) - 1.959963985) < 1e-6
    assert abs(norm_ppf(0.025) + 1.959963985) < 1e-6
    # symmetric
    assert abs(norm_ppf(0.3) + norm_ppf(0.7)) < 1e-9


def test_poisson_tail_matches_hand_computation():
    # P(Poisson(mu) >= 2) = 1 - e^-mu (1 + mu)
    for mu in (0.5, 1.0, 1.8, 3.2):
        expected = 1 - math.exp(-mu) * (1 + mu)
        assert abs(_poisson_upper_tail(2, mu) - expected) < 1e-12


def test_poisson_inversion_round_trips():
    # pick a mean, read off its tail, invert, get the mean back
    for k, mu in ((2, 1.7), (4, 3.1), (11, 9.4), (71, 68.0)):
        p = _poisson_upper_tail(k, mu)
        recovered = poisson_mean_for_upper_tail(k - 0.5, p)
        assert abs(recovered - mu) < 1e-3


def test_stafford_passing_td_example():
    # 1.5 line, de-vigged P(over) ~ 0.56 -> mean well above 1.5, ~1.85
    mu = implied_mean_from_line(1.5, 0.56, 'passing_tds')
    assert 1.80 < mu < 1.95


def test_even_count_line_still_lifts_for_skew():
    # An exactly even 1.5 TD line is a MEDIAN; the Poisson mean sits above it.
    mu = implied_mean_from_line(1.5, 0.5, 'passing_tds')
    assert 1.60 < mu < 1.75
    # and a juiced-under line pulls the mean back down toward / below the number
    lower = implied_mean_from_line(1.5, 0.42, 'passing_tds')
    assert lower < mu


def test_even_yardage_line_is_unchanged():
    # A yardage line priced evenly implies its own number - no skew term.
    assert abs(implied_mean_from_line(74.5, 0.5, 'receiving_yards', 'WR') - 74.5) < 1e-6
    # Over favoured -> mean above the number, scaled by the WR sigma (36).
    hi = implied_mean_from_line(74.5, 0.60, 'receiving_yards', 'WR')
    assert 74.5 + 8 < hi < 74.5 + 11        # 0.253 * 36 ~ 9.1


def test_no_p_over_falls_back_to_multiplier():
    # Counts: the old MEDIAN_TO_MEAN multiplier, exactly.
    assert implied_mean_from_line(1.5, None, 'passing_tds') == 1.5 * 1.02
    assert implied_mean_from_line(0.5, None, 'receiving_tds') == 0.5 * 1.05
    # Yards: unchanged (no multiplier entry).
    assert implied_mean_from_line(64.5, None, 'rushing_yards', 'RB') == 64.5
    # Garbage p_over is treated as "no p_over", not an error.
    assert implied_mean_from_line(1.5, float('nan'), 'passing_tds') == 1.5 * 1.02
    assert implied_mean_from_line(2.5, 1.4, 'receptions') == 2.5 * 1.0


def test_bad_line_returns_none():
    assert implied_mean_from_line(None, 0.5, 'passing_tds') is None
    assert implied_mean_from_line('x', 0.5, 'passing_tds') is None
    assert implied_mean_from_line(float('inf'), 0.5, 'passing_tds') is None


def test_season_yardage_uses_wider_sigma():
    game = implied_mean_from_line(900.5, 0.60, 'receiving_yards', 'WR', period='game')
    season = implied_mean_from_line(900.5, 0.60, 'receiving_yards', 'WR', period='season')
    # same p_over, but the season sigma is sqrt(17)x, so the shift is bigger
    assert (season - 900.5) > (game - 900.5) * 3
