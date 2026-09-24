"""data.market_devig: posted line + de-vigged P(over) -> implied market mean."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.market_devig import (  # noqa: E402
    implied_mean_from_line, norm_ppf, poisson_mean_for_upper_tail,
    _poisson_upper_tail, _yard_skew_factor,
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


def test_attempts_and_carries_are_devigged_as_counts():
    # The app's canonical prop names for pass/rush attempts are 'attempts'
    # and 'carries' (not 'passing_attempts'/'rushing_attempts'). They must
    # hit the Poisson path, not fall straight through un-devigged.
    # Dak Prescott: 33.5 pass-att, both books lean UNDER -> mean below 33.5.
    under = implied_mean_from_line(33.5, 0.481, 'attempts', 'QB')
    assert 33.0 < under < 33.45
    # An exactly even line is a MEDIAN; the large-count Poisson mean sits a
    # hair above the number, and a juiced under still pulls below that.
    even = implied_mean_from_line(33.5, 0.5, 'attempts', 'QB')
    assert 33.5 <= even < 34.0
    assert under < even
    # 'carries' takes the same path.
    rb = implied_mean_from_line(15.5, 0.44, 'carries', 'RB')
    assert 14.5 < rb < 15.5
    assert implied_mean_from_line(15.5, None, 'carries', 'RB') == 15.5   # bare board unchanged


def test_even_yardage_line_gets_the_median_to_mean_skew_correction():
    # 2026-09-23: an evenly priced yardage line is a MEDIAN, and weekly
    # yardage is right-skewed, so even a perfectly even line now lifts by the
    # bucketed factor fit in scripts/fit_yard_skew.py (YARD_MEDIAN_TO_MEAN).
    factor = _yard_skew_factor('WR', 'receiving_yards', 50.5)
    assert abs(implied_mean_from_line(50.5, 0.5, 'receiving_yards', 'WR') - 50.5 * factor) < 1e-6
    # A season-period line skips the skew term entirely (17-game sums are
    # already close to symmetric) - the even line really is unchanged there.
    assert abs(implied_mean_from_line(50.5, 0.5, 'receiving_yards', 'WR', period='season') - 50.5) < 1e-6
    # Over favoured -> vig-lean shift (WR sigma 36) THEN the skew multiplier.
    hi_factor = _yard_skew_factor('WR', 'receiving_yards', 74.5)
    hi = implied_mean_from_line(74.5, 0.60, 'receiving_yards', 'WR')
    vig_only = 74.5 + norm_ppf(0.60) * 36.0
    assert abs(hi - vig_only * hi_factor) < 1e-6
    assert hi > vig_only > 74.5   # skew strictly adds on top of the vig lean


def test_yard_skew_factor_falls_back_to_no_correction_for_unknown_position():
    # A missing or unrecognized position (not one of QB/RB/WR/TE) isn't
    # guessed at - the caller (weekly_market_projection) is expected to have
    # already tried to backfill it from the board.
    assert _yard_skew_factor(None, 'receiving_yards', 74.5) == 1.0
    assert _yard_skew_factor('', 'receiving_yards', 74.5) == 1.0
    assert _yard_skew_factor('K', 'receiving_yards', 74.5) == 1.0
    assert implied_mean_from_line(74.5, 0.5, 'receiving_yards', None) == 74.5
    assert implied_mean_from_line(74.5, 0.5, 'receiving_yards', 'K') == 74.5


def test_yard_skew_shrinks_as_the_player_line_grows():
    # The whole point of bucketing: a low-volume player's line needs a much
    # bigger correction than a workhorse's (E2/fit_yard_skew.py finding).
    low = _yard_skew_factor('RB', 'receiving_yards', 5.0)
    high = _yard_skew_factor('RB', 'receiving_yards', 70.0)
    assert low > high > 1.0


def test_tiny_yardage_line_with_heavy_under_never_goes_negative():
    # Matthew Stafford, 0.5 rushing yards, Over deep plus-money -> de-vigged
    # P(over) well under 0.5. The raw Normal(0.5, 18) vig-lean term is about
    # -6; the result must not be negative - a yardage mean is physical.
    mu = implied_mean_from_line(0.5, 0.33, 'rushing_yards', 'QB')
    assert mu >= 0.0
    assert mu == 0.5                                  # falls back to the posted line
    # A normal-sized line with the same lean still gets the ordinary shift.
    big = implied_mean_from_line(245.5, 0.33, 'passing_yards', 'QB')
    assert big < 245.5                                # under-lean pulls it down
    assert big > 200.0                                # ...but nowhere near zero
    # A small line whose lean leaves it barely positive is left alone.
    small_ok = implied_mean_from_line(8.5, 0.45, 'receiving_yards', 'RB')
    assert 0.0 < small_ok < 8.5


def test_anytime_td_cap_applies_only_to_the_real_05_line_anytime_market():
    # A real "does he score anytime" market is posted at 0.5 - the dispersion
    # cap (_ANYTIME_TD_MEAN_CAP) belongs there: real single-game TD counts
    # are less dispersed than Poisson at the top, so an elite back's 0.5-line
    # price shouldn't invert past ~1.0/game.
    anytime = implied_mean_from_line(0.5, 0.7, 'rushing_tds', 'RB')
    assert anytime == 1.0   # capped - raw Poisson inversion would be ~1.2

    # BUG FIX 2026-09-16: a genuine 1.5-line multi-TD market (Derrick Henry
    # "Over/Under 1.5 rushing TDs") is a DIFFERENT, higher-confidence market,
    # not a dispersion-inflated anytime line - it used to hit the same cap
    # (`value <= 1.5`) on the mistaken assumption its mean "lands well under
    # the cap anyway." An evenly priced 1.5 line does not: its raw Poisson
    # mean is ~1.68 (matching test_even_count_line_still_lifts_for_skew's
    # passing_tds case, same math), and that real signal must survive
    # uncapped for rushing/receiving TDs same as it already does for passing.
    multi_td = implied_mean_from_line(1.5, 0.5, 'rushing_tds', 'RB')
    assert 1.60 < multi_td < 1.75
    assert multi_td == poisson_mean_for_upper_tail(1.5, 0.5)   # truly uncapped

    # receiving_tds takes the same path.
    multi_td_rec = implied_mean_from_line(1.5, 0.5, 'receiving_tds', 'WR')
    assert 1.60 < multi_td_rec < 1.75


def test_no_p_over_falls_back_to_multiplier():
    # Counts: the old MEDIAN_TO_MEAN multiplier, exactly.
    assert implied_mean_from_line(1.5, None, 'passing_tds') == 1.5 * 1.02
    assert implied_mean_from_line(0.5, None, 'receiving_tds') == 0.5 * 1.05
    # Yards with a known position: the median->mean skew factor now applies
    # here too (2026-09-23) - a bare board with no odds is still a median.
    factor = _yard_skew_factor('RB', 'rushing_yards', 64.5)
    assert abs(implied_mean_from_line(64.5, None, 'rushing_yards', 'RB') - 64.5 * factor) < 1e-6
    # No known position: unchanged, same as before.
    assert implied_mean_from_line(64.5, None, 'rushing_yards') == 64.5
    # Season period: unchanged regardless of position (skew is game-only).
    assert implied_mean_from_line(64.5, None, 'rushing_yards', 'RB', period='season') == 64.5
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
    # 'game' also carries the skew multiplier, 'season' never does (skew is
    # game-only) - divide it back out before comparing the sigma scaling.
    game_vig_only = game / _yard_skew_factor('WR', 'receiving_yards', 900.5)
    # same p_over, but the season sigma is sqrt(17)x, so the shift is bigger
    assert (season - 900.5) > (game_vig_only - 900.5) * 3
