"""v2_xtd_missing_prior_share: a player absent from last season's zone usage has no prior zone share in v2_xtd."""
import numpy as np

import data.weekly_projections as wp

RATES = {'td_rate': {'rush': {'RB': {'rz5': {'rate': 0.4}, 'rz10': {'rate': 0.1}, 'rz20': {'rate': 0.04}}}},
         'zone_elasticity': {'rush': {}}}
CTX = {
    'cur_player': {'rookie': {'rz5_carries': 2.0, 'rz10_carries': 1.0, 'rz20_carries': 2.0},
                   'vet': {'rz5_carries': 2.0, 'rz10_carries': 1.0, 'rz20_carries': 2.0},
                   'backup': {'rz5_carries': 2.0, 'rz10_carries': 1.0, 'rz20_carries': 2.0}},
    'cur_team': {'LV': {'rz5_carries': 4.0, 'rz10_carries': 2.0, 'rz20_carries': 4.0}},
    'prior_player': {'vet': {'rz5_carries': 10.0, 'rz10_carries': 5.0, 'rz20_carries': 10.0},
                     'backup': {'rz5_carries': 0.0, 'rz10_carries': 0.0, 'rz20_carries': 0.0}},
    'prior_team': {'LV': {'games': 17, 'rz5_carries': 20.0, 'rz10_carries': 10.0, 'rz20_carries': 20.0}},
}


def _xtd(monkeypatch, missing_prior_unknown, games=3.0):
    monkeypatch.setattr(wp, '_load_xtd_rates', lambda: RATES)
    monkeypatch.setattr(wp, '_xtd_zone_context', lambda *a, **k: CTX)
    names = np.array(['rookie', 'vet', 'backup'], dtype=object)
    return wp.xtd_blended_rate('rushing_tds', 'RB', np.zeros(3), np.full(3, games), np.zeros(3), np.zeros(3),
                               names, np.array(['LV'] * 3, dtype=object), {}, None, 2025, 4,
                               missing_prior_unknown=missing_prior_unknown)


def _full_share_xtd(share):
    return sum(CTX['prior_team']['LV'][f'{z}_carries'] / 17 * share * RATES['td_rate']['rush']['RB'][z]['rate']
               for z in ('rz5', 'rz10', 'rz20'))


def test_flag_is_a_candidate():
    assert 'v2_xtd_missing_prior_share' in wp.MODEL_FEATURES and 'v2_xtd_missing_prior_share' not in wp.DEFAULT_FEATURES


def test_rookie_keeps_his_current_share(monkeypatch):
    on = _xtd(monkeypatch, True)
    off = _xtd(monkeypatch, False)
    assert abs(on[0] - _full_share_xtd(0.5)) < 1e-9                  # current share 0.5 in every zone
    assert abs(off[0] - _full_share_xtd(0.5) * 3.0 / 7.0) < 1e-9     # the old read: 0.5 blended with a fake 0
    assert on[0] > off[0]


def test_players_present_last_season_are_unchanged(monkeypatch):
    on = _xtd(monkeypatch, True)
    off = _xtd(monkeypatch, False)
    assert abs(on[1] - off[1]) < 1e-12        # vet: real prior share 0.5
    assert abs(on[2] - off[2]) < 1e-12        # backup: present with no zone carries keeps his real 0 prior share
    assert on[2] < on[0]
