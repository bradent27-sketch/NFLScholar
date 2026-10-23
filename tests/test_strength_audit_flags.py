"""Subtraction candidates from the 2026-10-23 multiplier strength audit (CFB comparison): v2_no_td_matchup,
v2_no_qb_rush_script, v2_no_pace - registered and off."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.weekly_projections as wp  # noqa: E402


def test_flags_are_registered_and_not_default():
    for flag in ('v2_no_td_matchup', 'v2_no_qb_rush_script', 'v2_no_pace'):
        assert flag in wp.MODEL_FEATURES
        assert flag not in wp.DEFAULT_FEATURES
    assert wp.STRENGTH_AUDIT_TD_STATS == ('passing_tds', 'rushing_tds', 'receiving_tds')
