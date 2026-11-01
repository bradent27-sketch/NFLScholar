"""data/prop_probability.py (2026-10-28): the prop engine's recalibrated mean, its P(over) and the history inputs it reads."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.prop_probability as pp  # noqa: E402

# one WR engine: mean = 0.5 projection + 0.2 s2d + 0.1 l3 + 0.2 last season (+ 1.0 when s2d is missing); one projection bin
# whose ratios actual / mean run evenly from 0 to 2
ENGINE = {'receptions': {'spec': {'kind': 'count'},
                         'recal': {'WR': {'coef': [0.5, 0.2, 0.1, 0.2, 1.0, 0.0, 0.0], 'intercept': 0.0, 'floor': 0.05}},
                         'M1': {'WR': {'edges': [], 'ratio_quantiles': {'0': list(np.linspace(0, 2, 201))}}}}}


@pytest.fixture
def engine(tmp_path, monkeypatch):
    path = tmp_path / 'engines.json'
    path.write_text(json.dumps(ENGINE), encoding='utf-8')
    monkeypatch.setattr(pp, 'ENGINE_PATH', str(path))
    return path


def test_recalibrated_mean_blends_the_projection_with_the_players_own_history(engine):
    got = pp.recalibrated_mean('receptions', ['WR', 'WR', 'RB'], [6.0, 6.0, 3.0], s2d=[4.0, np.nan, 2.0], l3=[5.0, np.nan, 2.0],
                               ps=[3.0, np.nan, 2.0])
    assert got[0] == pytest.approx(0.5 * 6 + 0.2 * 4 + 0.1 * 5 + 0.2 * 3)
    assert got[1] == pytest.approx(6.0 + 1.0)            # every history input missing: the projection stands in, plus the flag
    assert got[2] == 3.0                                 # no fit for RB: the projection as is


def test_prob_over_reads_the_ratio_table_at_the_line(engine):
    # mean 4: P(result <= 3) = P(ratio < 3.5 / 4) = 0.4375 on the 0-2 uniform ratios
    p = pp.prob_over('receptions', ['WR'] * 3, [4.0, 4.0, 4.0], [3.5, 3.0, 7.5])
    assert p[0] == pytest.approx(1 - 3.5 / 8, abs=1e-6)
    assert p[1] == pytest.approx(p[0])                   # a whole-number line: a push is not an over
    assert p[2] == pytest.approx(1 - 7.5 / 8, abs=1e-6)
    assert np.isnan(pp.prob_over('receptions', ['TE'], [4.0], [3.5])[0])


def test_no_engine_file_leaves_the_projection_and_no_probability(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, 'ENGINE_PATH', str(tmp_path / 'missing.json'))
    assert pp.recalibrated_mean('receptions', ['WR'], [5.0])[0] == 5.0
    assert np.isnan(pp.prob_over('receptions', ['WR'], [5.0], [4.5])[0])


def test_player_history_uses_only_games_before_the_week():
    rows = [{'player_display_name': 'Ace Wideout', 'position': 'WR', 'week': w, 'receptions': r, 'season_type': 'REG'}
            for w, r in ((1, 2.0), (2, 4.0), (3, 6.0), (4, 8.0), (5, 100.0))]
    prior = [{'player_display_name': 'Ace Wideout', 'position': 'WR', 'week': w, 'receptions': 5.0, 'season_type': 'REG'}
             for w in range(1, 4)]
    h = pp.player_history(pd.DataFrame(rows), 'player_display_name', 5, pd.DataFrame(prior), 'player_display_name')
    row = h.iloc[0]
    assert row['h_s2d_receptions'] == pytest.approx(5.0)    # weeks 1-4 only
    assert row['h_l3_receptions'] == pytest.approx(6.0)     # weeks 2-4
    assert row['h_ps_receptions'] == pytest.approx(5.0)


def test_prop_probabilities_joins_history_by_player_and_position(engine):
    hist = pd.DataFrame({'h_s2d_receptions': [4.0], 'h_l3_receptions': [5.0], 'h_ps_receptions': [3.0]},
                        index=pd.MultiIndex.from_tuples([('ace wideout', 'WR')], names=['_key', 'pos']))
    frame = pd.DataFrame({'_key': ['ace wideout', 'someone else'], 'Pos': ['WR', 'WR'], 'receptions': [6.0, 6.0],
                          'line': [4.5, 4.5]})
    out = pp.prop_probabilities(frame, 'receptions', history=hist)
    assert out['prop_mean'].tolist() == pytest.approx([4.9, 7.0])
    assert out['p_over'].iloc[0] < out['p_over'].iloc[1]
