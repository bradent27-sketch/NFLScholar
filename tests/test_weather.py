"""data/weather.py provider abstraction + the weekly model's per-STAT weather
redistribution.

Contract:
  * recorded_game_weather reads the schedule's post-game temp/wind/roof and is
    keyed by BOTH teams in a game (it is the game's environment, not a
    stadium property);
  * an indoor game / missing weather / feed down -> empty dict, never a crash;
  * wind pushes pass stats DOWN and QB+RB rush stats UP (a redistribution, not
    a flat penalty), monotone in wind speed, clamped per stat.
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from data import weather as wx
from data.weekly_projections import weather_stat_multipliers, WEATHER_STAT_CLAMP


def _schedule(rows):
    return pd.DataFrame(rows)


def test_recorded_weather_keys_both_teams_and_reads_roof():
    sch = _schedule([
        {"season": 2024, "week": 12, "home_team": "CLE", "away_team": "PIT",
         "roof": "outdoors", "temp": 36, "wind": 13},
        {"season": 2024, "week": 12, "home_team": "DET", "away_team": "IND",
         "roof": "dome", "temp": None, "wind": None},
    ])
    out = wx.recorded_game_weather(sch, 12)
    assert out["CLE"].wind_mph == 13 and out["PIT"].wind_mph == 13   # game, not stadium
    assert out["CLE"].is_outdoor and out["CLE"].temp_f == 36
    assert not out["DET"].is_outdoor and not out["IND"].is_outdoor


def test_resolve_prefers_recorded_and_never_calls_forecast_when_complete(monkeypatch):
    sch = _schedule([
        {"season": 2024, "week": 5, "home_team": "GB", "away_team": "CHI",
         "roof": "outdoors", "temp": 55, "wind": 9},
    ])

    def _boom(*a, **k):
        raise AssertionError("forecast must not be called when the row is recorded")

    monkeypatch.setattr(wx, "forecast_game_weather", _boom)
    out = wx.resolve_game_weather(sch, 5, allow_forecast=True)
    assert out["GB"].source == "schedule-recorded" and out["GB"].wind_mph == 9


def test_forecast_provider_failure_is_silent():
    class DeadProvider(wx.WeatherProvider):
        name = "dead"

        def fetch(self, lat, lon, when):
            raise RuntimeError("network down")

    sch = _schedule([
        {"season": 2026, "week": 1, "home_team": "BUF", "away_team": "NYJ",
         "roof": "outdoors", "temp": None, "wind": None, "gameday": "2026-09-13",
         "gametime": "13:00"},
    ])
    out = wx.forecast_game_weather(sch, 1, provider=DeadProvider(), use_cache=False)
    assert out["BUF"].wind_mph is None and out["BUF"].temp_f is None  # no crash


def test_open_meteo_parses_a_stubbed_response(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"hourly": {
                "time": ["2026-09-13T16:00", "2026-09-13T17:00", "2026-09-13T18:00"],
                "temperature_2m": [70.0, 68.0, 66.0],
                "wind_speed_10m": [6.0, 19.0, 8.0]}}

    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp())
    p = wx.OpenMeteoProvider()
    w, t = p.fetch(42.77, -78.79, dt.datetime(2026, 9, 13, 17, 0))
    assert w == 19.0 and t == 68.0   # nearest hour to 17:00


def test_visual_crossing_needs_a_key():
    p = wx.VisualCrossingProvider(api_key=None)
    assert p.fetch(40.0, -80.0, dt.datetime(2026, 1, 1, 13)) == (None, None)


def test_wind_pushes_pass_stats_down_and_leaves_rb_alone():
    m = weather_stat_multipliers("QB", 22, is_outdoor=True)
    assert m["passing_attempts"] < 1.0
    assert m["passing_yards"] < m["passing_completions"] < 1.0   # yards fall fastest
    assert m["passing_tds"] < 1.0
    wr = weather_stat_multipliers("WR", 22, is_outdoor=True)
    assert wr["receiving_yards"] < wr["receptions"] < wr["targets"] < 1.0
    te = weather_stat_multipliers("TE", 22, is_outdoor=True)
    assert te["receiving_yards"] < 1.0 and te["receptions"] < 1.0
    assert weather_stat_multipliers("RB", 30, is_outdoor=True) == {}   # RB: no measured wind effect


def test_wind_ignores_temperature():
    warm = weather_stat_multipliers("QB", 20, temp_f=70, is_outdoor=True)
    freezing = weather_stat_multipliers("QB", 20, temp_f=5, is_outdoor=True)
    assert warm == freezing and warm                       # identical, and non-empty
    assert "passing_interceptions" not in warm             # INT is not modelled


def test_wind_neutral_at_or_below_knee_indoor_and_missing():
    assert weather_stat_multipliers("QB", 5, is_outdoor=True) == {}      # at the 5 mph QB knee
    assert weather_stat_multipliers("TE", 7, is_outdoor=True) == {}      # under the 8 mph TE knee
    assert weather_stat_multipliers("QB", 25, is_outdoor=False) == {}    # indoor
    assert weather_stat_multipliers("WR", np.nan, is_outdoor=True) == {}


def test_wind_is_monotone_and_clamped():
    prev = 2.0
    for w in range(6, 70, 4):
        m = weather_stat_multipliers("QB", w, is_outdoor=True)["passing_yards"]
        assert m <= prev + 1e-9 and m >= WEATHER_STAT_CLAMP[0] - 1e-9
        prev = m
    assert weather_stat_multipliers("QB", 120, is_outdoor=True)["passing_yards"] == pytest.approx(
        WEATHER_STAT_CLAMP[0])
