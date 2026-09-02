"""Coaching-change labels + the coaching-aware defense prior weighting.

The labels feed v2_coaching_aware_defense_prior, so a wrong one silently
mis-weights a whole team's defense prior. These tests pin the derivation
logic (change = different name vs the same team's previous season; missing
baseline => NaN; cohort mapping) and the safe-fallback contract that makes
the flag a no-op wherever coordinator data is absent.
"""
import numpy as np
import pandas as pd
import pytest

from data import coaching_changes as cc
from data.weekly_projections import blend_defense_prior, DEFENSE_PRIOR_GAMES


def test_cohort_mapping_is_exhaustive_and_directional():
    m = cc.COHORT_DEFENSE_PRIOR_GAMES
    assert m["none"] is None and m["unknown"] is None      # -> caller default
    assert m["both"] < DEFENSE_PRIOR_GAMES                  # staff reset: shorter leash
    assert m["hc_only"] < DEFENSE_PRIOR_GAMES
    assert m["dc_only"] > DEFENSE_PRIOR_GAMES               # coordinator promo: longer leash


def test_defense_prior_games_by_team_falls_back_to_default_without_coordinator_data():
    # 2015 is the first backfilled season, so it has no prior-year DC baseline
    # for anyone -> every team 'unknown' -> everyone gets the passed-in default
    # (the flag does nothing).
    s = cc.defense_prior_games_by_team(2015, 12.0, season_lo=2015)
    assert len(s) == 32
    assert (s == 12.0).all()


def test_defense_prior_games_by_team_splits_a_real_season():
    # 2024 has coordinator data: at least one team on each of the three
    # weights, and every value is one of {default, dc_only, both/hc_only}.
    s = cc.defense_prior_games_by_team(2024, 12.0)
    allowed = {12.0, cc.COHORT_DEFENSE_PRIOR_GAMES["dc_only"], cc.COHORT_DEFENSE_PRIOR_GAMES["both"]}
    assert set(np.unique(s.values)).issubset(allowed)
    assert (s == 12.0).any() and (s > 12.0).any() and (s < 12.0).any()


def test_change_flags_need_a_prior_baseline():
    tbl = cc.coaching_change_table(2016, 2025)
    # nflverse HC history runs back to 1999, so hc_changed is populated for
    # essentially every row - the only gaps are a franchise's first season
    # under a new abbreviation (relocation: SD->LAC 2017, OAK->LV 2020).
    assert tbl["hc_changed"].notna().mean() > 0.95
    for stable in ("KC", "GB", "PIT", "BAL"):
        assert tbl.loc[tbl["team"] == stable, "hc_changed"].notna().all()
    # DC history now runs back to 2015 (Wikipedia backfill under Ourlads
    # 2022+), so dc_changed is populated for a good share of every season from
    # 2016 on - but still NaN wherever that season or the prior one has no
    # known DC (coverage is ~75%).
    assert tbl.loc[tbl["season"] == 2024, "dc_changed"].notna().all()   # Ourlads: complete
    cov_2018 = tbl.loc[tbl["season"] == 2018, "dc_changed"].notna().mean()
    assert 0.4 < cov_2018 < 1.0                                          # Wikipedia: partial
    # The very first backfilled season has no prior baseline for anyone.
    early = cc.coaching_change_table(2015, 2025)
    assert early.loc[early["season"] == 2015, "dc_changed"].isna().all()


def test_blend_defense_prior_accepts_a_per_team_series():
    cur = pd.DataFrame({"WR": [1.30, 1.30]}, index=["AAA", "BBB"])
    old = pd.DataFrame({"WR": [0.80, 0.80]}, index=["AAA", "BBB"])
    evidence = pd.Series({"AAA": 6.0, "BBB": 6.0})
    # AAA keeps a long prior leash (30), BBB a short one (3): with equal
    # current evidence BBB must sit closer to its (higher) current value.
    pg = pd.Series({"AAA": 30.0, "BBB": 3.0})
    out = blend_defense_prior(cur, old, evidence, prior_games=pg)
    assert out.loc["AAA", "WR"] < out.loc["BBB", "WR"]
    # A team missing from the Series falls back to the scalar default, not NaN.
    pg_partial = pd.Series({"AAA": 30.0})
    out2 = blend_defense_prior(cur, old, evidence, prior_games=pg_partial)
    assert np.isfinite(out2.loc["BBB", "WR"])
    scalar = blend_defense_prior(cur, old, evidence, prior_games=float(DEFENSE_PRIOR_GAMES))
    assert out2.loc["BBB", "WR"] == pytest.approx(scalar.loc["BBB", "WR"])


def test_blend_defense_prior_scalar_path_unchanged():
    cur = pd.DataFrame({"QB": [1.2]}, index=["ZZZ"])
    old = pd.DataFrame({"QB": [0.9]}, index=["ZZZ"])
    ev = pd.Series({"ZZZ": 8.0})
    a = 8.0 / (8.0 + float(DEFENSE_PRIOR_GAMES))
    exp = a * 1.2 + (1 - a) * 0.9
    out = blend_defense_prior(cur, old, ev)
    assert out.loc["ZZZ", "QB"] == pytest.approx(exp)
