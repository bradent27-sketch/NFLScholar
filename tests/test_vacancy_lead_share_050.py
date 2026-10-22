"""v2_vacancy_lead_share_050 (shipped 2026-10-22, model audit lead L2, WR part): when a WR/TE is out, the room's lead
pass catcher gets weight 0.5 of the next man up's in the vacancy split instead of 0.24."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.rb_role_allocator as rba  # noqa: E402
import data.weekly_projections as wp  # noqa: E402
from tests.test_rb_role_allocator_v2 import _player_row, _receiver_room_frame  # noqa: E402


def _gains(frame, lead_share):
    profiles = {'Out WR2': {'plays_probability': 0.0, 'status': 'out'}}
    provenance = {'Out WR2': {'year': 2026, 'week': 1, 'source': 'current_week_report'}}
    out, _ = rba.redistribute_rb_vacancy_with_allocator(
        frame, profiles, as_of_year=2026, injury_provenance=provenance, receiver_pecking_order=True,
        receiver_lead_share=lead_share)
    return {r['Player']: float(r['targets']) - float(_player_row(frame, r['Player'])['targets']) for _, r in out.iterrows()}


def test_flag_is_registered_and_shipped():
    assert 'v2_vacancy_lead_share_050' in wp.MODEL_FEATURES
    assert 'v2_vacancy_lead_share_050' in wp.DEFAULT_FEATURES          # shipped 2026-10-22
    assert rba.RECEIVER_VACANCY_LEAD_SHARE == 0.24
    assert rba.RECEIVER_VACANCY_LEAD_SHARE_CANDIDATE == 0.50


def test_default_lead_share_is_unchanged_and_the_candidate_gives_the_alpha_more():
    frame = _receiver_room_frame()
    shipped = _gains(frame, rba.RECEIVER_VACANCY_LEAD_SHARE)
    candidate = _gains(frame, rba.RECEIVER_VACANCY_LEAD_SHARE_CANDIDATE)
    total_s = sum(v for v in shipped.values() if v > 0)
    total_c = sum(v for v in candidate.values() if v > 0)
    assert abs(total_s - total_c) < 1e-6                     # same vacated volume, only the split moves
    assert candidate['Alpha WR'] > shipped['Alpha WR']
    assert candidate['WR3'] < shipped['WR3']
    assert candidate['WR3'] > candidate['Alpha WR']           # the next man up still gets the most
