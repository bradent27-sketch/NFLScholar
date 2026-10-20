"""v2_expected_points_luck (candidate 2026-10-20, backlog N1): WR/TE points pulled against how far earlier results ran
ahead of the expected points their usage was worth."""
import os
import sys
import time
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

import data.expected_points as ep  # noqa: E402
import data.historical_availability as ha  # noqa: E402
import data.weekly_projections as wp  # noqa: E402


def key(name):
    return wp.clean_name_exact(pd.Series([name])).iloc[0]


def _ff(rows):
    return pd.DataFrame(rows, columns=['week', 'full_name', 'total_fantasy_points', 'total_fantasy_points_exp',
                                       'receptions', 'receptions_exp'])


def test_flag_is_registered_and_not_yet_default():
    # when it ships it must also join POST_CALIBRATION_FEATURES (a calibration dump must not record already-adjusted totals)
    assert 'v2_expected_points_luck' in wp.MODEL_FEATURES
    assert 'v2_expected_points_luck' not in wp.DEFAULT_FEATURES
    assert 'v2_expected_points_luck' not in wp.CALIBRATION_INPUT_FEATURES


def test_luck_uses_only_games_before_the_target_week():
    ff = _ff([[1, 'A Receiver', 20.0, 10.0, 6, 5], [2, 'A Receiver', 8.0, 10.0, 4, 5], [3, 'A Receiver', 30.0, 10.0, 8, 5]])
    t = ep.luck_per_game(ff, 3)
    assert t.loc[key('A Receiver'), 'games'] == 2
    assert np.isclose(t.loc[key('A Receiver'), 'luck_pg'], (10.0 - 2.0) / 2)      # week 3 itself is not seen
    assert ep.luck_per_game(ff, 1).empty                                           # nothing before week 1


def test_reception_part_of_luck_follows_the_scoring_mode():
    ff = _ff([[1, 'A Receiver', 20.0, 10.0, 8, 5]])                                # 10 points of luck, 3 from extra catches
    full = ep.luck_per_game(ff, 2, 'Full PPR').loc[key('A Receiver'), 'luck_pg']
    half = ep.luck_per_game(ff, 2, 'Half-PPR').loc[key('A Receiver'), 'luck_pg']
    std = ep.luck_per_game(ff, 2, 'Standard').loc[key('A Receiver'), 'luck_pg']
    assert np.isclose(full, 10.0) and np.isclose(half, 8.5) and np.isclose(std, 7.0)


def test_a_name_twice_in_one_week_is_left_out():
    ff = _ff([[1, 'Same Name', 5.0, 4.0, 1, 1], [1, 'Same Name', 9.0, 3.0, 2, 1], [1, 'Solo Guy', 5.0, 4.0, 1, 1]])
    t = ep.luck_per_game(ff, 2)
    assert key('Same Name') not in t.index and key('Solo Guy') in t.index


def _board():
    return pd.DataFrame({
        'Player': ['Lucky WR', 'Unlucky TE', 'Lucky RB', 'Out WR', 'Zero WR', 'New WR'],
        'Pos': ['WR', 'TE', 'RB', 'WR', 'WR', 'WR'], 'Team': ['A', 'B', 'C', 'D', 'E', 'F'],
        'Availability': [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        'Model Proj Pts': [14.0, 9.0, 12.0, 0.0, 0.0, 10.0]})


def _table():
    idx = [key(n) for n in ('Lucky WR', 'Unlucky TE', 'Lucky RB', 'Out WR', 'Zero WR')]
    return pd.DataFrame({'games': [5, 5, 5, 5, 5], 'luck_pg': [5.0, -4.0, 6.0, 5.0, 5.0]}, index=idx)


def test_wr_and_te_move_against_their_luck_and_nobody_else_does():
    out, info = wp.apply_expected_points_luck(_board(), _table())
    pts = dict(zip(out['Player'], out['Model Proj Pts']))
    assert np.isclose(pts['Lucky WR'], 14.0 - 0.18 * 5.0)
    assert np.isclose(pts['Unlucky TE'], 9.0 + 0.16 * 4.0)
    assert pts['Lucky RB'] == 12.0 and pts['Out WR'] == 0.0 and pts['Zero WR'] == 0.0 and pts['New WR'] == 10.0
    assert set(k[0] for k in info) == {'Lucky WR', 'Unlucky TE'}
    row = info[('Lucky WR', 'WR', 'A')]
    assert row['before'] == 14.0 and np.isclose(row['after'], 13.1) and row['games'] == 5


def test_never_below_zero_and_no_table_changes_nothing():
    board = pd.DataFrame({'Player': ['Lucky WR'], 'Pos': ['WR'], 'Team': ['A'], 'Availability': [1.0], 'Model Proj Pts': [0.5]})
    out, _ = wp.apply_expected_points_luck(board, pd.DataFrame({'games': [3], 'luck_pg': [40.0]}, index=[key('Lucky WR')]))
    assert out['Model Proj Pts'].iloc[0] == 0.0
    same, info = wp.apply_expected_points_luck(_board(), pd.DataFrame(columns=['games', 'luck_pg']))
    assert info == {} and same['Model Proj Pts'].tolist() == _board()['Model Proj Pts'].tolist()


class _Polars:
    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df


def _fake_feed(monkeypatch, frame=None, boom=False):
    calls = []

    def load_ff_opportunity(seasons, stat_type):
        calls.append(seasons)
        if boom:
            raise OSError('offline')
        return _Polars(frame)
    monkeypatch.setitem(sys.modules, 'nflreadpy', types.SimpleNamespace(load_ff_opportunity=load_ff_opportunity))
    return calls


def _write_cache(tmp_path, season, max_week, age_hours):
    rows = [[season, w, 'A Receiver', 10.0, 8.0] for w in range(1, max_week + 1)]
    path = tmp_path / f'ff_opportunity_{season}.parquet'
    pd.DataFrame(rows, columns=['season', 'week', 'full_name', 'total_fantasy_points', 'total_fantasy_points_exp']).to_parquet(path, index=False)
    old = time.time() - age_hours * 3600
    os.utime(path, (old, old))


def test_a_partial_season_refreshes_when_old_and_a_failed_refresh_keeps_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ep, 'CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(ha, 'CACHE_DIR', str(tmp_path))
    _write_cache(tmp_path, 2026, 4, age_hours=30)
    fresh = pd.DataFrame([[2026, w, 'A Receiver', 10.0, 8.0] for w in range(1, 7)],
                         columns=['season', 'week', 'full_name', 'total_fantasy_points', 'total_fantasy_points_exp'])
    calls = _fake_feed(monkeypatch, fresh)
    assert pd.to_numeric(ep.load_ff_opportunity_weekly(2026)['week']).max() == 6 and len(calls) == 1
    assert pd.to_numeric(ep.load_ff_opportunity_weekly(2026)['week']).max() == 6 and len(calls) == 1   # fresh now
    _write_cache(tmp_path, 2026, 4, age_hours=30)
    _fake_feed(monkeypatch, boom=True)
    assert pd.to_numeric(ep.load_ff_opportunity_weekly(2026)['week']).max() == 4                      # kept
    (tmp_path / 'ff_opportunity_2026.parquet').unlink()
    assert ep.load_ff_opportunity_weekly(2026).empty                                                 # nothing, no crash


def test_a_finished_season_is_never_refetched(tmp_path, monkeypatch):
    monkeypatch.setattr(ep, 'CACHE_DIR', str(tmp_path))
    _write_cache(tmp_path, 2024, 18, age_hours=24 * 200)
    calls = _fake_feed(monkeypatch, pd.DataFrame())
    assert pd.to_numeric(ep.load_ff_opportunity_weekly(2024)['week']).max() == 18 and calls == []
