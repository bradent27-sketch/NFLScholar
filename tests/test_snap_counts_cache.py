"""data.loaders._load_snap_counts_raw (backlog D9, 2026-10-21): snap counts cached on disk so a failed download can no longer
silently empty a season."""
import os
import sys
import time

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


def _season(season, last_week):
    return pd.DataFrame({'season': season, 'week': list(range(1, last_week + 1)), 'player': 'A',
                         'offense_pct': 0.9, 'defense_pct': 0.0, 'game_type': 'REG'})


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ha, 'CACHE_DIR', str(tmp_path))
    return tmp_path


def _download(monkeypatch, result):
    calls = []

    def fake(years):
        calls.append(years)
        if isinstance(result, Exception):
            raise result
        return _Frame(result)
    monkeypatch.setattr(ld.nflreadpy, 'load_snap_counts', fake)
    return calls


def test_download_is_cached(cache_dir, monkeypatch):
    calls = _download(monkeypatch, _season(2023, 22))
    out = ld._load_snap_counts_raw(2023)
    assert len(out) == 22 and len(calls) == 1
    assert os.path.exists(os.path.join(cache_dir, 'snap_counts_2023.parquet'))


def test_finished_season_is_read_from_disk(cache_dir, monkeypatch):
    _season(2023, 22).to_parquet(os.path.join(cache_dir, 'snap_counts_2023.parquet'), index=False)
    calls = _download(monkeypatch, RuntimeError('network down'))
    assert len(ld._load_snap_counts_raw(2023)) == 22
    assert calls == []


def test_failed_refresh_keeps_the_partial_season_copy(cache_dir, monkeypatch):
    path = os.path.join(cache_dir, 'snap_counts_2026.parquet')
    _season(2026, 6).to_parquet(path, index=False)
    old = time.time() - 7 * 3600
    os.utime(path, (old, old))
    calls = _download(monkeypatch, RuntimeError('network down'))
    assert len(ld._load_snap_counts_raw(2026)) == 6
    assert len(calls) == 1


def test_empty_download_keeps_the_cached_copy(cache_dir, monkeypatch):
    path = os.path.join(cache_dir, 'snap_counts_2026.parquet')
    _season(2026, 6).to_parquet(path, index=False)
    old = time.time() - 7 * 3600
    os.utime(path, (old, old))
    _download(monkeypatch, pd.DataFrame())
    assert len(ld._load_snap_counts_raw(2026)) == 6


def test_fresh_partial_season_copy_is_not_downloaded_again(cache_dir, monkeypatch):
    _season(2026, 6).to_parquet(os.path.join(cache_dir, 'snap_counts_2026.parquet'), index=False)
    calls = _download(monkeypatch, _season(2026, 7))
    assert len(ld._load_snap_counts_raw(2026)) == 6
    assert calls == []


def test_no_cache_and_no_download_is_empty(cache_dir, monkeypatch):
    _download(monkeypatch, RuntimeError('network down'))
    assert ld._load_snap_counts_raw(2022).empty
