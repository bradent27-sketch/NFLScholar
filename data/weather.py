"""Game-day weather for the weekly model (wind + temperature).

Two sources behind one interface:

  * RECORDED - nflverse `load_schedules()` fills `temp` / `wind` / `roof`
    AFTER a game is played. Perfect for a backtest (the recorded value is
    what a good forecast would have converged to) and free.
  * FORECAST - a `WeatherProvider` for games that have not happened yet.
    `OpenMeteoProvider` is the default: keyless, no signup, hourly wind/temp
    by lat/lon (https://open-meteo.com). `VisualCrossingProvider` and
    `NWSProvider` are adapters you can enable later (the first needs a key).

`resolve_game_weather(schedule_df, week, provider=...)` returns
    {team_abbr: GameWeather(wind_mph, temp_f, is_outdoor, source)}
using the recorded columns when they are present for that week and falling
back to the forecast provider only for the rows that are still blank. Every
failure path degrades to an empty dict / neutral GameWeather so a caller can
treat "no weather" as "no adjustment" - never as a crash.

The module is import-safe with no network at import time; a provider only
touches the network when `.fetch()` is called.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
import time
from typing import Iterable

import pandas as pd

# --- stadium coordinates ---------------------------------------------------
# lat, lon, is_dome_default. Retractable roofs (ATL, DAL, HOU, ARI, LV, LAC/LA
# SoFi is a fixed open-air canopy) are marked dome=False here and the per-game
# `roof` column decides open/closed; a fixed dome (MIN, DET, NO, IND) is
# dome=True. Coordinates are the stadium centroid to ~0.02 deg (~1.5 mi),
# which is far finer than any weather grid this feeds.
STADIUM_COORDS: dict[str, tuple[float, float, bool]] = {
    "ARI": (33.53, -112.26, False),  # State Farm - retractable
    "ATL": (33.76, -84.40, False),   # Mercedes-Benz - retractable
    "BAL": (39.28, -76.62, False),
    "BUF": (42.77, -78.79, False),
    "CAR": (35.23, -80.85, False),
    "CHI": (41.86, -87.62, False),
    "CIN": (39.10, -84.52, False),
    "CLE": (41.51, -81.70, False),
    "DAL": (32.75, -97.09, False),   # AT&T - retractable
    "DEN": (39.74, -105.02, False),
    "DET": (42.34, -83.05, True),    # Ford Field - fixed dome
    "GB": (44.50, -88.06, False),
    "HOU": (29.68, -95.41, False),   # NRG - retractable
    "IND": (39.76, -86.16, True),    # Lucas Oil - retractable but ~always closed; treat dome
    "JAX": (30.32, -81.64, False),
    "KC": (39.05, -94.48, False),
    "LA": (33.95, -118.34, True),    # SoFi - roofed (open-air sides); no wind/rain -> dome-like
    "LAC": (33.95, -118.34, True),
    "LV": (36.09, -115.18, True),    # Allegiant - fixed dome
    "MIA": (25.96, -80.24, False),
    "MIN": (44.97, -93.26, True),    # U.S. Bank - fixed
    "NE": (42.09, -71.26, False),
    "NO": (29.95, -90.08, True),     # Superdome - fixed
    "NYG": (40.81, -74.07, False),
    "NYJ": (40.81, -74.07, False),
    "PHI": (39.90, -75.17, False),
    "PIT": (40.45, -80.02, False),
    "SEA": (47.60, -122.33, False),
    "SF": (37.40, -121.97, False),
    "TB": (27.98, -82.50, False),
    "TEN": (36.17, -86.77, False),
    "WAS": (38.91, -76.86, False),
}

INDOOR_ROOF_VALUES = {"dome", "closed", "indoor", "indoors", "retractable-closed"}
OUTDOOR_ROOF_VALUES = {"outdoors", "open", "outdoor"}

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "nflscholar_weather")
_FORECAST_TTL = 6 * 3600  # a forecast older than 6h for an upcoming game is refetched


@dataclasses.dataclass(frozen=True)
class GameWeather:
    wind_mph: float | None = None
    temp_f: float | None = None
    is_outdoor: bool = True
    source: str = "none"

    @property
    def usable(self) -> bool:
        return self.is_outdoor and (self.wind_mph is not None or self.temp_f is not None)


# --- recorded (schedule) path -------------------------------------------------
def _roof_is_outdoor(roof: object) -> bool:
    r = str(roof or "").strip().lower()
    if r in INDOOR_ROOF_VALUES:
        return False
    if r in OUTDOOR_ROOF_VALUES:
        return True
    return True  # unknown roof string -> assume outdoor (conservative: weather can apply)


def recorded_game_weather(schedule_df: pd.DataFrame, week: int) -> dict[str, GameWeather]:
    """{team: GameWeather} from the schedule's post-game `temp`/`wind`/`roof`."""
    if schedule_df is None or schedule_df.empty:
        return {}
    need = {"week", "home_team", "away_team", "roof"}
    if not need.issubset(schedule_df.columns):
        return {}
    wk = schedule_df[pd.to_numeric(schedule_df["week"], errors="coerce") == week]
    out: dict[str, GameWeather] = {}
    for _, g in wk.iterrows():
        outdoor = _roof_is_outdoor(g.get("roof"))
        temp = pd.to_numeric(pd.Series([g.get("temp")]), errors="coerce").iloc[0]
        wind = pd.to_numeric(pd.Series([g.get("wind")]), errors="coerce").iloc[0]
        obs = GameWeather(
            wind_mph=float(wind) if pd.notna(wind) else None,
            temp_f=float(temp) if pd.notna(temp) else None,
            is_outdoor=outdoor,
            source="schedule-recorded" if (pd.notna(temp) or pd.notna(wind) or not outdoor) else "none",
        )
        for team in (g.get("home_team"), g.get("away_team")):
            if isinstance(team, str) and team:
                out[team] = obs
    return out


# --- forecast providers -----------------------------------------------------
class WeatherProvider:
    name = "abstract"

    def fetch(self, lat: float, lon: float, when_utc: _dt.datetime) -> tuple[float | None, float | None]:
        """Return (wind_mph, temp_f) nearest `when_utc`, or (None, None)."""
        raise NotImplementedError


class OpenMeteoProvider(WeatherProvider):
    """https://open-meteo.com - keyless. Hourly `wind_speed_10m` + `temperature_2m`
    in imperial units; we pick the hour closest to kickoff. Forecast horizon is
    ~16 days, which covers every scheduled NFL week."""
    name = "open-meteo"
    ENDPOINT = "https://api.open-meteo.com/v1/forecast"

    def fetch(self, lat, lon, when_utc):
        try:
            import requests
        except Exception:
            return (None, None)
        day = when_utc.date().isoformat()
        params = {
            "latitude": round(lat, 3), "longitude": round(lon, 3),
            "hourly": "temperature_2m,wind_speed_10m",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "start_date": day, "end_date": day, "timezone": "UTC",
        }
        try:
            r = requests.get(self.ENDPOINT, params=params, timeout=12)
            r.raise_for_status()
            h = r.json().get("hourly", {})
            times = h.get("time", [])
            if not times:
                return (None, None)
            target = when_utc.replace(minute=0, second=0, microsecond=0, tzinfo=None)
            idx = min(range(len(times)),
                      key=lambda i: abs(_dt.datetime.fromisoformat(times[i]) - target))
            temps = h.get("temperature_2m", [])
            winds = h.get("wind_speed_10m", [])
            return (float(winds[idx]) if idx < len(winds) else None,
                    float(temps[idx]) if idx < len(temps) else None)
        except Exception:
            return (None, None)


class VisualCrossingProvider(WeatherProvider):
    """https://www.visualcrossing.com/weather-api - needs an API key
    (VISUAL_CROSSING_API_KEY env var or explicit arg). Historical + forecast in
    one call; enable it only once a key is provided."""
    name = "visual-crossing"
    ENDPOINT = ("https://weather.visualcrossing.com/VisualCrossing/rest/services/"
                "timeline/{lat},{lon}/{date}")

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("VISUAL_CROSSING_API_KEY")

    def fetch(self, lat, lon, when_utc):
        if not self.api_key:
            return (None, None)
        try:
            import requests
            url = self.ENDPOINT.format(lat=lat, lon=lon, date=when_utc.date().isoformat())
            r = requests.get(url, params={"key": self.api_key, "unitGroup": "us",
                                          "include": "hours", "elements": "datetime,temp,windspeed"},
                             timeout=12)
            r.raise_for_status()
            hours = (r.json().get("days") or [{}])[0].get("hours", [])
            if not hours:
                return (None, None)
            tgt = when_utc.hour
            hh = min(hours, key=lambda h: abs(int(str(h.get("datetime", "12:00:00"))[:2]) - tgt))
            return (float(hh.get("windspeed")) if hh.get("windspeed") is not None else None,
                    float(hh.get("temp")) if hh.get("temp") is not None else None)
        except Exception:
            return (None, None)


class NWSProvider(WeatherProvider):
    """https://www.weather.gov/documentation/services-web-api - keyless, US only.
    Two hops (point -> gridpoint -> forecastHourly); left as an adapter, Open-Meteo
    is lighter for this use."""
    name = "nws"

    def fetch(self, lat, lon, when_utc):
        return (None, None)


PROVIDERS = {"open-meteo": OpenMeteoProvider, "visual-crossing": VisualCrossingProvider,
             "nws": NWSProvider}


def get_provider(name: str | None = None, **kw) -> WeatherProvider:
    return PROVIDERS.get((name or "open-meteo").lower(), OpenMeteoProvider)(**kw)


def _cache_path(key: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, key + ".json")


def _kickoff_utc(row) -> _dt.datetime:
    """Best-effort kickoff in UTC from the schedule row; if gametime/gameday are
    missing, fall back to 18:00 UTC (~1pm ET) on the season-week's Sunday."""
    gd = str(row.get("gameday") or row.get("gsis") or "")
    gt = str(row.get("gametime") or "13:00")
    try:
        base = _dt.datetime.fromisoformat(gd)
    except ValueError:
        return _dt.datetime.now()
    try:
        hh, mm = (int(x) for x in gt.split(":")[:2])
    except Exception:
        hh, mm = 13, 0
    # schedule gametime is US Eastern; +5h is a fine approximation for a
    # weather hour-bucket (DST error is at most 1h, sub-bucket).
    return base.replace(hour=hh, minute=mm) + _dt.timedelta(hours=5)


def forecast_game_weather(schedule_df: pd.DataFrame, week: int,
                          provider: WeatherProvider | str | None = None,
                          use_cache: bool = True) -> dict[str, GameWeather]:
    if schedule_df is None or schedule_df.empty:
        return {}
    prov = provider if isinstance(provider, WeatherProvider) else get_provider(provider)
    wk = schedule_df[pd.to_numeric(schedule_df["week"], errors="coerce") == week]
    if wk.empty:
        return {}
    season = int(pd.to_numeric(wk["season"], errors="coerce").dropna().iloc[0]) if "season" in wk else 0
    ckey = f"{prov.name}_{season}_wk{week}"
    cache = {}
    if use_cache:
        p = _cache_path(ckey)
        if os.path.exists(p) and (time.time() - os.path.getmtime(p)) < _FORECAST_TTL:
            try:
                cache = json.load(open(p))
            except Exception:
                cache = {}

    out: dict[str, GameWeather] = {}
    dirty = False
    for _, g in wk.iterrows():
        home = g.get("home_team")
        if not isinstance(home, str) or home not in STADIUM_COORDS:
            continue
        outdoor = _roof_is_outdoor(g.get("roof"))
        if not outdoor:
            obs = GameWeather(is_outdoor=False, source=f"{prov.name}-roof")
        elif home in cache:
            c = cache[home]
            obs = GameWeather(c.get("wind_mph"), c.get("temp_f"), True, f"{prov.name}-cache")
        else:
            lat, lon, _dome = STADIUM_COORDS[home]
            try:
                wind, temp = prov.fetch(lat, lon, _kickoff_utc(g))
            except Exception:
                wind, temp = None, None
            obs = GameWeather(wind, temp, True, prov.name)
            cache[home] = {"wind_mph": wind, "temp_f": temp}
            dirty = True
        for team in (g.get("home_team"), g.get("away_team")):
            if isinstance(team, str) and team:
                out[team] = obs
    if use_cache and dirty:
        try:
            json.dump(cache, open(_cache_path(ckey), "w"))
        except Exception:
            pass
    return out


def resolve_game_weather(schedule_df: pd.DataFrame, week: int,
                         provider: WeatherProvider | str | None = None,
                         allow_forecast: bool = True) -> dict[str, GameWeather]:
    """Recorded weather where the schedule already has it; forecast only for
    the rows still blank (and only if `allow_forecast`)."""
    rec = recorded_game_weather(schedule_df, week)
    have = {t: o for t, o in rec.items() if o.source != "none"}
    missing_outdoor = [t for t, o in rec.items()
                       if o.source == "none" and o.is_outdoor]
    if allow_forecast and (missing_outdoor or not rec):
        fc = forecast_game_weather(schedule_df, week, provider=provider)
        for t, o in fc.items():
            if t not in have:
                have[t] = o
    # carry through any indoor rows the recorded pass already resolved
    for t, o in rec.items():
        have.setdefault(t, o)
    return have


if __name__ == "__main__":
    import sys
    import nflreadpy as nfl
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    wk = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    sch = nfl.load_schedules(seasons=[yr]).to_pandas()
    print(f"--- recorded {yr} wk{wk} ---")
    for t, o in sorted(recorded_game_weather(sch, wk).items()):
        print(f"  {t:>4}  {o}")
