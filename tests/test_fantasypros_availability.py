"""Offline tests for data.fantasypros_availability - the FantasyPros-sourced
injury/availability signal.

WHY THESE EXIST. The live-API path here was originally wired to the wrong
endpoint entirely: it called GET /nfl/players (which has no injury field at
all - confirmed against a real response dump) and guessed at a status field
name that was never going to appear. The real endpoint, GET /nfl/injuries,
has its own dedicated schema (confirmed against FantasyPros' own OpenAPI
spec, not guessed). These tests pin that schema down with a monkeypatched
requests.get returning a payload shaped exactly like the real one, so a
future accidental revert back to /nfl/players - or a typo in a field name -
fails loudly here instead of silently in production with an empty result.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import fantasypros_availability as fpa  # noqa: E402


class _FakeResponse:
    def __init__(self, status_code, payload=None, text=''):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def test_parse_maps_the_real_fantasypros_status_enum_and_drops_healthy_rows():
    raw = pd.DataFrame({
        'player': ['A Hurt', 'B Doubt', 'C Reserve', 'D Susp', 'E Covid', 'F Bench', 'G Fine'],
        'team': ['SF', 'MIN', 'DAL', 'KC', 'PHI', 'BUF', 'GB'],
        'status': ['Questionable', 'Doubtful', 'IR', 'Suspended', 'COV-IR', 'Not Starting', 'Active'],
        'note': ['knee'] * 7,
        'plays_probability': [0.7, 0.2, None, None, None, None, 1.0],
    })
    out, error = fpa.parse_fantasypros_injury_export(raw)
    assert error is None
    statuses = dict(zip(out['player'], out['status']))
    assert statuses['A Hurt'] == 'questionable'
    assert statuses['B Doubt'] == 'doubtful'
    assert statuses['C Reserve'] == 'out'
    assert statuses['D Susp'] == 'out'
    assert statuses['E Covid'] == 'out'
    assert statuses['F Bench'] == 'doubtful'
    # A clean "Active" row is dropped entirely - absence means healthy.
    assert 'G Fine' not in set(out['player'])
    assert len(out) == 6


def test_parse_carries_through_a_real_probability_of_playing_when_present():
    raw = pd.DataFrame({
        'player': ['A Hurt'],
        'status': ['Questionable'],
        'probability_of_playing': ['0.65'],
    })
    out, error = fpa.parse_fantasypros_injury_export(raw)
    assert error is None
    assert abs(float(out.iloc[0]['plays_probability']) - 0.65) < 1e-9


def test_canonical_status_maps_real_designations_to_one_of_four_buckets():
    assert fpa.canonical_status('Questionable') == 'questionable'
    assert fpa.canonical_status('Doubtful') == 'doubtful'
    assert fpa.canonical_status('IR') == 'out'
    assert fpa.canonical_status('Suspended') == 'out'
    assert fpa.canonical_status('COV-IR') == 'out'
    assert fpa.canonical_status('Not Starting') == 'doubtful'
    assert fpa.canonical_status('Active') == 'healthy'


def test_canonical_status_treats_no_report_as_healthy_not_flagged():
    # 'No current designation' is data.weekly_projections' own fallback
    # string for a player with nothing on the report (row['Injury Status']
    # or 'No current designation') - not a real designation. A regression
    # here would misread every healthy player in the app as Questionable.
    assert fpa.canonical_status('No current designation') == 'healthy'
    assert fpa.canonical_status('') == 'healthy'
    assert fpa.canonical_status(None) == 'healthy'
    assert fpa.canonical_status(float('nan')) == 'healthy'


def test_canonical_status_treats_a_genuinely_unrecognized_status_as_questionable():
    # A non-empty status this app doesn't recognize (e.g. an nflverse-
    # specific code from the older V1 availability path) must never silently
    # present as full health - it should read as "something's flagged".
    assert fpa.canonical_status('some_unmapped_nflverse_code') == 'questionable'


def test_fetch_injury_report_reads_the_real_nfl_injuries_schema(monkeypatch):
    """Pins the fix: name/team_id/status/comment/probability_of_playing on
    GET /nfl/injuries - not player_name/team/injury_status on /nfl/players."""
    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured['url'] = url
        captured['params'] = params
        return _FakeResponse(200, {
            'sport': 'NFL', 'count': 1,
            'injuries': [{
                'player_id': 15901, 'yahoo_id': '5529', 'name': 'Brandon Saad',
                'team_id': 'PIT', 'position_id': 'RB', 'rank': 24,
                'injury_type': 'Lower body', 'comment': 'Game-time decision.',
                'injury_update_date': '2025-05-14', 'status': 'Questionable',
                'status_short': 'Q', 'ir_weeks': [], 'probability_of_playing': '0.75',
                'practice_1': 'Limit', 'practice_2': 'Limit', 'practice_3': 'Full',
                'practice_report_injury_type': 'Abdomen',
                'team_practice_1_submitted': True, 'team_practice_2_submitted': True,
                'team_practice_3_submitted': True,
            }],
        })

    monkeypatch.setattr(fpa.requests, 'get', fake_get)
    raw, error = fpa.fetch_fantasypros_injury_report('key123', 2025, 6)
    assert error is None
    assert captured['url'].endswith('/nfl/injuries')
    assert captured['params']['year'] == 2025
    assert captured['params']['week'] == 6
    assert len(raw) == 1
    row = raw.iloc[0]
    assert row['player'] == 'Brandon Saad'
    assert row['team'] == 'PIT'
    assert row['status'] == 'Questionable'
    assert row['note'] == 'Game-time decision.'
    assert abs(float(row['plays_probability']) - 0.75) < 1e-9


def test_fetch_injury_report_treats_an_empty_week_as_success_not_an_error(monkeypatch):
    monkeypatch.setattr(fpa.requests, 'get',
                        lambda *a, **k: _FakeResponse(200, {'sport': 'NFL', 'count': 0, 'injuries': []}))
    raw, error = fpa.fetch_fantasypros_injury_report('key123', 2025, 1)
    assert error is None
    assert raw.empty


def test_fetch_injury_report_surfaces_a_rejected_api_key(monkeypatch):
    monkeypatch.setattr(fpa.requests, 'get', lambda *a, **k: _FakeResponse(401))
    raw, error = fpa.fetch_fantasypros_injury_report('bad-key', 2025, 1)
    assert raw.empty
    assert 'rejected' in error.lower()


def test_save_injury_api_pull_persists_a_clean_week_as_zero_rows(monkeypatch):
    monkeypatch.setattr(fpa.requests, 'get',
                        lambda *a, **k: _FakeResponse(200, {'sport': 'NFL', 'count': 0, 'injuries': []}))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'fp_injuries.csv')
        n, error = fpa.save_fantasypros_injury_api_pull('key123', 2025, 3, path=path)
        assert error is None
        assert n == 0
        profiles, load_error = fpa.load_fantasypros_availability(2025, 3, path=path)
        assert load_error is None
        assert profiles == {}


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            if 'monkeypatch' in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                fn(_SimpleMonkeypatch())
            else:
                fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failures.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


class _SimpleMonkeypatch:
    """Minimal pytest-monkeypatch-alike for the standalone `python
    tests/test_fantasypros_availability.py` run path (no pytest dependency)."""
    def __init__(self):
        self._restores = []

    def setattr(self, obj, name, value):
        self._restores.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def __del__(self):
        for obj, name, old in reversed(self._restores):
            try:
                setattr(obj, name, old)
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
