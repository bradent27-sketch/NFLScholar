"""The projection decomposition's audit notes (2026-10-07): only what bears on THIS player's number is printed.

Covers the pure-ish renderers in ui/tabs/rankings.py with a recording stand-in for ``st``: the availability line
(silent for a healthy player), the raw -> calibrated -> anchor points table, and the vacancy section (silent for a
teammate whose absence vacated nothing or too little to matter, unless this player received some of it)."""
import contextlib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

import ui.tabs.rankings as rk  # noqa: E402


class _FakeSt:
    def __init__(self):
        self.calls = []

    def markdown(self, text, *a, **k):
        self.calls.append(('markdown', text))

    def caption(self, text, *a, **k):
        self.calls.append(('caption', text))

    def dataframe(self, frame, *a, **k):
        self.calls.append(('dataframe', frame))

    def expander(self, label, expanded=False):
        self.calls.append(('expander', label))
        return contextlib.nullcontext()

    def texts(self):
        return ' '.join(str(c[1]) for c in self.calls if c[0] in ('markdown', 'caption'))


def _frame(fake):
    styled = next(c[1] for c in fake.calls if c[0] == 'dataframe')
    return getattr(styled, 'data', styled)           # st.dataframe is handed a Styler


def _fake(monkeypatch):
    fake = _FakeSt()
    monkeypatch.setattr(rk, 'st', fake)
    return fake


def test_a_healthy_player_gets_no_availability_section(monkeypatch):
    fake = _fake(monkeypatch)
    rk._render_availability_note({'status': 'No current designation', 'plays_probability': 1.0,
                                  'workload_if_active': 1.0, 'source': 'no current availability source'})
    rk._render_availability_note({'status': '', 'plays_probability': 1.0, 'workload_if_active': 1.0})
    assert fake.calls == []


def test_a_real_designation_a_discount_or_a_source_is_printed(monkeypatch):
    fake = _fake(monkeypatch)
    rk._render_availability_note({'status': 'Questionable', 'plays_probability': 0.6, 'workload_if_active': 0.8,
                                  'source': 'FantasyPros injury report', 'match_method': 'exact name',
                                  'reported_probability': 0.6})
    text = fake.texts()
    assert 'Availability and workload' in text and 'Questionable' in text and '60%' in text and '80%' in text
    assert 'reference only' in text and 'FantasyPros injury report' in text
    fake2 = _fake(monkeypatch)
    rk._render_availability_note({'status': None, 'plays_probability': 1.0, 'workload_if_active': 0.7})
    assert 'workload if active 70%' in fake2.texts()                                  # a workload discount alone counts


def test_points_path_lists_only_the_stages_that_fired(monkeypatch):
    fake = _fake(monkeypatch)
    detail = {'raw_points': 5.20, 'calibrated_points': 6.03,
              'calibration': {'enabled': True, 'raw_points': 5.20, 'displayed_points': 6.03, 'slope': 0.969,
                              'intercept': 0.990},
              'season_anchor': {'before': 6.03, 'after': 6.39, 'mean': 7.2, 'games': 4, 'factor': 0.69}}
    rk._render_points_path(detail)
    frame = _frame(fake)
    assert [r for r in frame['Stage']] == ['Raw model', 'Calibrated', 'Season anchor']
    assert list(frame['Points']) == ['5.20', '6.03', '6.39']
    assert list(frame['Change']) == ['—', '+0.83', '+0.36']
    assert ('expander', 'What these mean:') in fake.calls                              # the long explanation is folded away

    only_cal = _fake(monkeypatch)
    rk._render_points_path({'raw_points': 5.0, 'calibrated_points': 5.5,
                            'calibration': {'enabled': True, 'raw_points': 5.0, 'displayed_points': 5.5,
                                            'slope': 1.0, 'intercept': 0.5}})
    assert len(_frame(only_cal)) == 2
    assert ('expander', 'What these mean:') not in only_cal.calls

    neither = _fake(monkeypatch)
    rk._render_points_path({'raw_points': 5.0, 'calibrated_points': 5.0, 'calibration': {'enabled': False}})
    assert neither.calls == []


def _entry(volume, source, vacated, recipients=(), reason='Allocator-weighted core-RB injury redistribution.',
           allocated=None, role='core RB'):
    allocated = sum(r['allocated'] for r in recipients) if allocated is None else allocated
    return {'team': 'TB', 'volume': volume, 'source_player': source, 'vacated': vacated, 'allocated': allocated,
            'unallocated': max(0.0, vacated - allocated), 'recipients': list(recipients), 'reason': reason,
            'functional_source_role': role}


def test_vacancy_section_is_silent_for_nothing_vacated_or_too_little_to_matter(monkeypatch):
    fake = _fake(monkeypatch)
    rk._render_vacancy_redistribution({'player': 'Some WR', 'vacancy': [
        _entry('passing_attempts', 'Baker Mayfield', 0.0, reason='No projected, active role-compatible replacement.'),
        _entry('targets', 'Jalen McMillan', 0.2, recipients=[{'player': 'Other WR', 'allocated': 0.2, 'team_rank': 2}]),
    ]})
    assert fake.calls == []


def test_a_small_absence_is_one_line_only_when_this_player_received_some_of_it(monkeypatch):
    fake = _fake(monkeypatch)
    rk._render_vacancy_redistribution({'player': 'Other WR', 'vacancy': [
        _entry('targets', 'Jalen McMillan', 0.2, recipients=[{'player': 'Other WR', 'allocated': 0.2, 'team_rank': 2}]),
    ]})
    text = fake.texts()
    assert 'How vacated volume is redistributed' in text and 'Jalen McMillan' in text and '+0.20' in text
    assert not any(c[0] == 'dataframe' for c in fake.calls)                           # no recipient grid for a sliver


def test_a_real_vacancy_still_gets_its_caption_and_table(monkeypatch):
    fake = _fake(monkeypatch)
    rk._render_vacancy_redistribution({'player': 'Will Shipley', 'vacancy': [
        _entry('rushing_attempts', 'Saquon Barkley', 14.9,
               recipients=[{'player': 'Will Shipley', 'allocated': 9.9, 'team_rank': 1}]),
    ]})
    assert 'Saquon Barkley out' in fake.texts()
    assert any(c[0] == 'dataframe' for c in fake.calls)


def test_what_these_mean_is_a_collapsed_dropdown(monkeypatch):
    fake = _fake(monkeypatch)
    rk._what_these_mean('**Combined** = /Game × Allowed×.')
    assert fake.calls[0] == ('expander', 'What these mean:') and 'Combined' in fake.calls[1][1]
