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


def test_even_yardage_line_is_unchanged():
    # A yardage line priced evenly implies its own number - no skew term.
    assert abs(implied_mean_from_line(74.5, 0.5, 'receiving_yards', 'WR') - 74.5) < 1e-6
    # Over favoured -> mean above the number, scaled by the WR sigma (36).
    hi = implied_mean_from_line(74.5, 0.60, 'receiving_yards', 'WR')
    assert 74.5 + 8 < hi < 74.5 + 11        # 0.253 * 36 ~ 9.1


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
