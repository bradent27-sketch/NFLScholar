"""Subtraction candidates from the 2026-10-23 multiplier strength audit (CFB comparison): v2_no_td_matchup and v2_no_pace
registered and off; v2_no_qb_rush_script shipped 2026-10-24 (docs/mechanism_review_2026-10-24.md)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.weekly_projections as wp  # noqa: E402


def test_flags_are_registered_and_not_default():
    for flag in ('v2_no_td_matchup', 'v2_no_pace'):
        assert flag in wp.MODEL_FEATURES
        assert flag not in wp.DEFAULT_FEATURES


def test_shipped_on_2026_10_24():
    assert 'v2_no_qb_rush_script' in wp.DEFAULT_FEATURES
    assert 'v2_channel_matchups' not in wp.DEFAULT_FEATURES
    assert wp.STRENGTH_AUDIT_TD_STATS == ('passing_tds', 'rushing_tds', 'receiving_tds')


def test_td_matchup_strength_variants():
    feats = frozenset(wp.DEFAULT_FEATURES)
    assert wp.td_matchup_strength(feats, 'passing_tds') == 1.0
    assert wp.td_matchup_strength(feats | {'v2_no_td_matchup'}, 'rushing_tds') == 0.0
    audit = feats | {'v2_td_matchup_audit'}
    assert (wp.td_matchup_strength(audit, 'receiving_tds'), wp.td_matchup_strength(audit, 'rushing_tds')) == (0.0, 0.5)
    assert wp.td_matchup_strength(feats | {'v2_td_matchup_pass_rec_off'}, 'rushing_tds') == 1.0
    assert wp.td_matchup_strength(feats | {'v2_td_matchup_half'}, 'passing_tds') == 0.5
    for flag in wp.TD_MATCHUP_STRENGTH:
        assert flag in wp.MODEL_FEATURES and flag not in wp.DEFAULT_FEATURES
