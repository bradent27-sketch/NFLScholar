"""v2_suspension_return_full_role (2026-10-14): a zero-game RB/WR/TE back from a suspension gets his full active role."""
import os
import sys
import time
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

import data.historical_availability as ha  # noqa: E402
import data.weekly_projections as wp  # noqa: E402


def key(name):
    return wp.clean_name_exact(pd.Series([name])).iloc[0]


def _rosters(rows):
    return pd.DataFrame(rows, columns=['season', 'week', 'team', 'full_name', 'status', 'status_description_abbr'])


def test_flag_is_registered_and_shipped():
    assert 'v2_suspension_return_full_role' in wp.MODEL_FEATURES
    assert 'v2_suspension_return_full_role' in wp.DEFAULT_FEATURES


def test_suspended_before_reads_the_latest_earlier_row():
    r = _rosters([
        [2023, 1, 'NO', 'Alvin Kamara', 'RES', 'R40'], [2023, 2, 'NO', 'Alvin Kamara', 'RES', 'R40'],
        [2023, 3, 'NO', 'Alvin Kamara', 'RES', 'R40'], [2023, 4, 'NO', 'Alvin Kamara', 'ACT', 'A01'],
        [2023, 1, 'GB', 'Healthy Guy', 'ACT', 'A01'], [2023, 2, 'GB', 'Healthy Guy', 'ACT', 'A01'],
        [2023, 1, 'IND', 'Injured Guy', 'RES', 'R01'], [2023, 2, 'IND', 'Injured Guy', 'RES', 'R01'],
    ])
    # week 3: suspended through week 2 and still suspended in week 3's own row -> has not returned
    assert key('Alvin Kamara') not in ha.suspended_before(2023, 3, rosters=r)
    # week 4: his week-3 row is the latest earlier one and week 4's row says activated -> a return
    assert key('Alvin Kamara') in ha.suspended_before(2023, 4, rosters=r)
    # week 2: week 2's own row is still R40 -> still suspended
    assert key('Alvin Kamara') not in ha.suspended_before(2023, 2, rosters=r)
    out = ha.suspended_before(2023, 4, rosters=r)
    assert key('Healthy Guy') not in out and key('Injured Guy') not in out      # only the suspension code


def test_a_player_activated_and_then_gone_again_is_not_a_return():
    r = _rosters([[2024, 1, 'ARI', 'Zay Jones', 'RES', 'R40'], [2024, 2, 'ARI', 'Zay Jones', 'ACT', 'A01'],
                  [2024, 3, 'ARI', 'Zay Jones', 'ACT', 'A01']])
    assert key('Zay Jones') not in ha.suspended_before(2024, 4, rosters=r)


def test_without_the_current_weeks_row_the_latest_earlier_status_decides():
    # the live feed may not have week W's row yet: suspended through the last published week counts as a return
    r = _rosters([[2026, 4, 'NO', 'Some Receiver', 'RES', 'R40'], [2026, 5, 'NO', 'Some Receiver', 'RES', 'R40']])
    assert key('Some Receiver') in ha.suspended_before(2026, 6, rosters=r)


def test_suspended_before_never_raises():
    assert ha.suspended_before(2026, 6, rosters=pd.DataFrame()) == set()
    assert ha.suspended_before(2026, 6, rosters=pd.DataFrame({'week': [1]})) == set()


def test_mask_needs_the_suspension_and_availability():
    names = pd.Series(['Alvin Kamara', 'Other Back', 'Still Out'])
    mask = wp.suspension_return_mask(names, {key('Alvin Kamara'), key('Still Out')}, [1.0, 1.0, 0.0])
    assert mask.tolist() == [True, False, False]
    assert wp.suspension_return_mask(names, set(), [1.0, 1.0, 1.0]).tolist() == [False, False, False]


def _restore(full):
    args = dict(whole_season_share=np.array([0.05, 0.05, 0.05]), active_game_share=np.array([0.80, 0.80, 0.10]),
                prior_games=np.array([14, 14, 3]), prior_team=['A', 'A', 'A'], current_team=['A', 'A', 'A'], pos='WR')
    return wp.restore_cold_start_returning_role_share(**args, full_recovery=full, return_details=True)


def test_full_recovery_promotes_only_eligible_marked_rows():
    base_share, base_ok, _ = _restore(None)
    share, ok, reason = _restore(np.array([False, True, True]))
    assert share[0] == base_share[0]                                    # eligible, not marked: the partial pull
    assert share[1] > base_share[1] and np.isclose(share[1], 0.80)      # eligible and marked: the full active role
    assert share[2] == base_share[2] and not ok[2]                      # marked but ineligible (3 prior games): untouched
    assert reason[1] == 'suspension return: full active role restored'
    assert (ok == base_ok).all()


def test_full_recovery_is_capped_like_every_other_restoration():
    share, _, _ = wp.restore_cold_start_returning_role_share(
        np.array([0.05]), np.array([1.0]), np.array([15]), ['A'], ['A'], 'WR',
        full_recovery=np.array([True]), return_details=True)
    assert share[0] == wp.COLD_START_RETURNING_ROLE_CAP


# --- the stale roster cache ---------------------------------------------------------------------------------------

class _Polars:
    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df


def _fake_nflreadpy(monkeypatch, frame=None, boom=False):
    calls = []

    def load_rosters_weekly(seasons):
        calls.append(seasons)
        if boom:
            raise OSError('offline')
        return _Polars(frame)
    monkeypatch.setitem(sys.modules, 'nflreadpy', types.SimpleNamespace(load_rosters_weekly=load_rosters_weekly))
    return calls


def _write_cache(tmp_path, season, max_week, age_hours):
    rows = [[season, w, 'NO', 'P', 'ACT', 'A01'] for w in range(1, max_week + 1)]
    path = tmp_path / f'rosters_weekly_{season}.parquet'
    _rosters(rows).to_parquet(path, index=False)
    old = time.time() - age_hours * 3600
    os.utime(path, (old, old))
    return path


def test_a_partial_season_cache_is_refreshed_when_old(tmp_path, monkeypatch):
    monkeypatch.setattr(ha, 'CACHE_DIR', str(tmp_path))
    _write_cache(tmp_path, 2026, 4, age_hours=30)
    fresh = _rosters([[2026, w, 'NO', 'P', 'ACT', 'A01'] for w in range(1, 7)])
    fresh['game_type'] = 'REG'
    calls = _fake_nflreadpy(monkeypatch, fresh)
    assert pd.to_numeric(ha.load_weekly_rosters(2026)['week']).max() == 6
    assert len(calls) == 1
    # now fresh on disk: a second read does not fetch again
    assert pd.to_numeric(ha.load_weekly_rosters(2026)['week']).max() == 6
    assert len(calls) == 1


def test_a_young_partial_cache_and_a_finished_season_are_not_refetched(tmp_path, monkeypatch):
    monkeypatch.setattr(ha, 'CACHE_DIR', str(tmp_path))
    _write_cache(tmp_path, 2026, 4, age_hours=1)
    _write_cache(tmp_path, 2025, 18, age_hours=24 * 200)
    calls = _fake_nflreadpy(monkeypatch, _rosters([]))
    assert pd.to_numeric(ha.load_weekly_rosters(2026)['week']).max() == 4
    assert pd.to_numeric(ha.load_weekly_rosters(2025)['week']).max() == 18
    assert calls == []


def test_a_failed_refresh_keeps_the_cached_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(ha, 'CACHE_DIR', str(tmp_path))
    _write_cache(tmp_path, 2026, 4, age_hours=30)
    _fake_nflreadpy(monkeypatch, boom=True)
    assert pd.to_numeric(ha.load_weekly_rosters(2026)['week']).max() == 4
    _fake_nflreadpy(monkeypatch, _rosters([]))                       # an empty answer also keeps the cache
    assert pd.to_numeric(ha.load_weekly_rosters(2026)['week']).max() == 4


# --- the injury-report cache (2026-10-19) ----------------------------------------------------------------------------

def _write_injury_cache(tmp_path, season, max_week, age_hours):
    rows = [[season, w, 'NO', 'P', 'Out'] for w in range(1, max_week + 1)]
    path = tmp_path / f'injuries_{season}.parquet'
    pd.DataFrame(rows, columns=['season', 'week', 'team', 'full_name', 'report_status']).to_parquet(path, index=False)
    old = time.time() - age_hours * 3600
    os.utime(path, (old, old))
    return path


def _fake_injuries(monkeypatch, frame=None, boom=False):
    calls = []

    def load_injuries(seasons):
        calls.append(seasons)
        if boom:
            raise OSError('offline')
        return _Polars(frame)
    monkeypatch.setitem(sys.modules, 'nflreadpy', types.SimpleNamespace(load_injuries=load_injuries))
    return calls


def test_a_partial_injury_cache_is_refreshed_when_old(tmp_path, monkeypatch):
    monkeypatch.setattr(ha, 'CACHE_DIR', str(tmp_path))
    _write_injury_cache(tmp_path, 2026, 4, age_hours=30)
    fresh = pd.DataFrame([[2026, w, 'NO', 'P', 'Out', 'REG'] for w in range(1, 7)],
                         columns=['season', 'week', 'team', 'full_name', 'report_status', 'game_type'])
    calls = _fake_injuries(monkeypatch, fresh)
    assert pd.to_numeric(ha.load_injury_reports(2026)['week']).max() == 6
    assert len(calls) == 1
    assert pd.to_numeric(ha.load_injury_reports(2026)['week']).max() == 6      # fresh on disk now: no second fetch
    assert len(calls) == 1


def test_a_young_partial_or_finished_injury_cache_is_not_refetched(tmp_path, monkeypatch):
    monkeypatch.setattr(ha, 'CACHE_DIR', str(tmp_path))
    _write_injury_cache(tmp_path, 2026, 4, age_hours=1)
    _write_injury_cache(tmp_path, 2025, 18, age_hours=24 * 200)
    calls = _fake_injuries(monkeypatch, pd.DataFrame())
    assert pd.to_numeric(ha.load_injury_reports(2026)['week']).max() == 4
    assert pd.to_numeric(ha.load_injury_reports(2025)['week']).max() == 18
    assert calls == []


def test_a_failed_injury_refresh_keeps_the_cached_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(ha, 'CACHE_DIR', str(tmp_path))
    _write_injury_cache(tmp_path, 2026, 4, age_hours=30)
    _fake_injuries(monkeypatch, boom=True)
    assert pd.to_numeric(ha.load_injury_reports(2026)['week']).max() == 4
    _fake_injuries(monkeypatch, pd.DataFrame())
    assert pd.to_numeric(ha.load_injury_reports(2026)['week']).max() == 4
    # no cache at all and no source: still an empty frame, never an exception
    (tmp_path / 'injuries_2026.parquet').unlink()
    _fake_injuries(monkeypatch, boom=True)
    assert ha.load_injury_reports(2026).empty


def test_a_finished_17_week_season_is_never_refetched(tmp_path, monkeypatch):
    monkeypatch.setattr(ha, 'CACHE_DIR', str(tmp_path))
    _write_cache(tmp_path, 2020, 17, age_hours=24 * 200)
    calls = _fake_nflreadpy(monkeypatch, _rosters([]))
    assert pd.to_numeric(ha.load_weekly_rosters(2020)['week']).max() == 17
    assert calls == []
