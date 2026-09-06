"""Weekly Rankings projection decomposition: the TE scheme (man/zone) mix
table and the demoted-to-reference alignment table.

For a tight end 'Defense multiplier' in the primary table is the man/zone
(scheme) blend, not the slot/wide/inline (alignment) blend, wherever scheme
evidence exists (v2_scheme_matchup, SCHEME_MATCHUP_SCORING_POSITIONS).
_render_scheme_mix is the worked calc that reconciles with it; the alignment
table is kept underneath in reference_only mode. This locks in:

* the scheme table renders Man / Zone / Blend rows for a TE with evidence,
* its Blend row's Allowed x equals the scored candidate multiplier,
* it is a no-op for a WR and for a TE with no scheme evidence,
* the alignment table's reference_only caption stops claiming it reconciles.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

import ui.tabs.rankings as rankings  # noqa: E402


class _Sink:
    """Capture st.markdown / st.caption strings and the st.dataframe frame."""

    def __init__(self, monkeypatch):
        self.text = []
        self.frames = []
        monkeypatch.setattr(rankings.st, "markdown", lambda body, *a, **k: self.text.append(str(body)))
        monkeypatch.setattr(rankings.st, "caption", lambda body, *a, **k: self.text.append(str(body)))
        monkeypatch.setattr(rankings.st, "dataframe", lambda frame, *a, **k: self.frames.append(frame))
        # style_plain_dataframe would return a Styler; keep the raw frame so
        # the test can read cells back.
        monkeypatch.setattr(rankings, "style_plain_dataframe", lambda frame, *a, **k: frame)
        monkeypatch.setattr(rankings, "df_auto_height", lambda *a, **k: 200)

    @property
    def joined(self):
        return "\n".join(self.text)


def _te_detail(scheme_available=True, defense_evidence=None):
    """A tight end with man/zone route split and a live scheme-defense
    residual for every _ALIGNMENT_MIX_STATS entry.

    scheme_available -> whether the PLAYER has a man/zone route profile.
    defense_evidence -> whether the DEFENSE has man/zone comparison
    evidence (defaults to scheme_available); the early-season case is
    player yes / defense no, which shows the fallback caption not a table.
    """
    if defense_evidence is None:
        defense_evidence = scheme_available
    stat_block = {
        'blended_rate': 6.0,
        'script_multiplier': 1.0, 'pace_multiplier': 1.0,
        'availability_multiplier': 1.0, 'environment_multiplier': 1.0,
    }
    return {
        'position': 'TE',
        'player': 'Sample TE',
        'opponent': 'DEN',
        'stats': {
            'targets': dict(stat_block, blended_rate=6.0),
            'receptions': dict(stat_block, blended_rate=4.0),
            'receiving_yards': dict(stat_block, blended_rate=45.0),
        },
        'alignment_scheme_evidence': {
            'player_scheme_available': scheme_available,
            'player_man_route_share': 0.40,
            'player_zone_route_share': 0.60,
            'player_scheme_sample_weight': 5.0,
            'defense_scheme_candidate_available': defense_evidence,
            'defense_scheme_reason': '' if defense_evidence else 'no man/zone comparison evidence',
            'defense_man_candidate_multiplier': 1.12,
            'defense_scheme_targets_candidate_multiplier': 1.12,
            'defense_scheme_receptions_candidate_multiplier': 1.08,
            'defense_scheme_yards_candidate_multiplier': 1.20,
            'defense_scheme_targets_man_ratio': 1.30, 'defense_scheme_targets_zone_ratio': 1.00,
            'defense_scheme_receptions_man_ratio': 1.20, 'defense_scheme_receptions_zone_ratio': 1.00,
            'defense_scheme_yards_man_ratio': 1.45, 'defense_scheme_yards_zone_ratio': 1.05,
        },
        'role': {},
    }


def test_scheme_mix_renders_man_zone_blend_rows_for_a_te(monkeypatch):
    sink = _Sink(monkeypatch)
    rankings._render_scheme_mix(_te_detail())
    assert sink.frames, "expected a scheme-mix dataframe"
    frame = sink.frames[0]
    splits = list(frame.iloc[:, 0])
    assert splits[0].startswith("Man")
    assert splits[1].startswith("Zone")
    assert splits[2] == "Blend (all)"
    assert "Scheme mix (man / zone)" in sink.joined


def test_scheme_mix_blend_row_reconciles_with_the_scored_multiplier(monkeypatch):
    sink = _Sink(monkeypatch)
    rankings._render_scheme_mix(_te_detail())
    frame = sink.frames[0].set_index(sink.frames[0].columns[0])
    # Blend row Allowed x for each stat == the candidate multiplier the TE
    # projection actually scored (evidence -> primary table 'Defense
    # multiplier').
    assert frame.loc['Blend (all)', 'Tgts Allowed×'] == "1.120×"
    assert frame.loc['Blend (all)', 'Rec Allowed×'] == "1.080×"
    assert frame.loc['Blend (all)', 'Rec Yds Allowed×'] == "1.200×"
    # Blend /Game is the unsplit projected rate, Combined = rate * mult.
    assert frame.loc['Blend (all)', 'Rec Yds /Game'] == "45.0"
    assert frame.loc['Blend (all)', 'Rec Yds Combined'] == "54.0"   # 45 * 1.20


def test_scheme_mix_man_row_uses_the_man_side_ratio_and_route_share(monkeypatch):
    sink = _Sink(monkeypatch)
    rankings._render_scheme_mix(_te_detail())
    frame = sink.frames[0].set_index(sink.frames[0].columns[0])
    man_label = [s for s in frame.index if s.startswith("Man")][0]
    # 40% man route share of a 45.0 rate -> 18.0 /Game on the man row.
    assert frame.loc[man_label, 'Rec Yds /Game'] == "18.0"
    assert frame.loc[man_label, 'Rec Yds Allowed×'] == "1.450×"
    # Combined = 18.0 * 1.45 = 26.1
    assert frame.loc[man_label, 'Rec Yds Combined'] == "26.1"


def test_scheme_mix_is_a_noop_for_a_wr(monkeypatch):
    sink = _Sink(monkeypatch)
    detail = _te_detail()
    detail['position'] = 'WR'
    rankings._render_scheme_mix(detail)
    assert not sink.frames
    assert not sink.text


def test_scheme_mix_explains_the_fallback_when_no_scheme_evidence(monkeypatch):
    sink = _Sink(monkeypatch)
    rankings._render_scheme_mix(_te_detail(scheme_available=True, defense_evidence=False))
    assert not sink.frames                      # no table
    assert "neutral (1.0" in sink.joined
    assert "falls back to the alignment number" in sink.joined


def test_alignment_mix_reference_mode_drops_the_reconciles_claim(monkeypatch):
    sink = _Sink(monkeypatch)
    detail = {
        'position': 'TE',
        'player': 'Sample TE',
        'role': {
            'alignment_available': True,
            'alignment_defense_candidate_available': True,
            'slot_alignment_rate': 0.2, 'wide_alignment_rate': 0.1, 'inline_alignment_rate': 0.7,
            'non_slot_alignment_rate': 0.8,
            'alignment_defense_blend_mode': 'slot_non_slot',
            'source_week_count': 4, 'source_weeks': '1-4', 'alignment_confidence': 0.5,
            'alignment_defense_targets_candidate_multiplier': 1.1,
            'alignment_defense_receptions_candidate_multiplier': 1.1,
            'alignment_defense_yards_candidate_multiplier': 1.1,
        },
        'stats': {'targets': {'blended_rate': 6.0}, 'receptions': {'blended_rate': 4.0},
                  'receiving_yards': {'blended_rate': 45.0}},
    }
    rankings._render_alignment_mix(detail, reference_only=True)
    assert "reference only" in sink.joined
    assert "CONTEXT ONLY" in sink.joined
    assert "REPLACES 'Defense multiplier'" not in sink.joined


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
