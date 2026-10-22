"""data.loaders._cached_season_download (2026-10-22): play-by-play and weekly team stats cached on disk like snap counts,
so a finished season never depends on a download during a harness run."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.historical_availability as ha  # noqa: E402
import data.loaders as ld  # noqa: E402


class _Frame:
    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df


def _season(season):
    return pd.DataFrame({'season': season, 'week': list(range(1, 19)), 'posteam': 'KC', 'play_id': range(18)})


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ha, 'CACHE_DIR', str(tmp_path))
    return tmp_path


def test_pbp_is_cached_and_a_failed_download_keeps_the_copy(cache_dir, monkeypatch):
    monkeypatch.setattr(ld.nflreadpy, 'load_pbp', lambda years: _Frame(_season(years[0])))
    ld.load_pbp.clear()
    first = ld.load_pbp(2019)
    assert (cache_dir / 'pbp_2019.parquet').exists() and len(first) == 18

    def fail(years):
        raise ConnectionError('offline')
    monkeypatch.setattr(ld.nflreadpy, 'load_pbp', fail)
    ld.load_pbp.clear()
    assert len(ld.load_pbp(2019)) == 18


def test_team_stats_week_is_cached_and_never_redownloaded_for_a_finished_season(cache_dir, monkeypatch):
    calls = []

    def fake(years, summary_level=None):
        calls.append((tuple(years), summary_level))
        return _Frame(_season(years[0]))
    monkeypatch.setattr(ld.nflreadpy, 'load_team_stats', fake)
    assert len(ld._load_team_stats_week(2019)) == 18
    assert len(ld._load_team_stats_week(2019)) == 18
    assert calls == [((2019,), 'week')]
    assert (cache_dir / 'team_stats_week_2019.parquet').exists()


def test_unreachable_source_with_no_cache_returns_empty(cache_dir):
    def fail():
        raise ConnectionError('offline')
    assert ld._cached_season_download('pbp', 2019, fail).empty
