"""2026-10-21 app audit and speed pass: output-neutral helpers and the data fixes behind them."""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.loaders as ld  # noqa: E402
import data.pff_alignment as pa  # noqa: E402
import data.weekly_projections as wp  # noqa: E402
from data.transforms import build_rookie_watch  # noqa: E402


def test_frame_records_matches_to_dict_value_and_type():
    frame = pd.DataFrame({
        'f32': np.array([1.5, np.nan], dtype='float32'), 'f64': [2.25, 3.0], 'i': [1, 2], 'b': [True, False],
        's': pd.Series(['a', None], dtype=object), 'cat': pd.Categorical(['KC', 'BUF']),
        'when': pd.to_datetime(['2026-10-01', '2026-10-08']),
    })
    want, got = frame.to_dict('records'), wp._frame_records(frame)
    assert len(want) == len(got)
    for a, b in zip(want, got):
        assert a.keys() == b.keys()
        for k in a:
            assert type(a[k]) is type(b[k])
            assert a[k] == b[k] or (a[k] != a[k] and b[k] != b[k])
    assert wp._frame_records(None) == []
    assert wp._frame_records(pd.DataFrame()) == []


def test_attach_by_name_exact_then_suffix_stripped():
    from ui.tabs.rankings import _attach_by_name
    base = pd.DataFrame({'Player': ['Marvin Harrison Jr.', 'Kenneth Walker III', 'Nobody Here']})
    other = pd.DataFrame({'Player': ['Marvin Harrison', 'Kenneth Walker III'], 'v': [1, 2]})
    out = _attach_by_name(base, other, ['v'], 'p_')
    assert out['p_v'].tolist()[:2] == [1, 2]
    assert pd.isna(out['p_v'].iloc[2])


def test_rookie_watch_reads_each_rows_rookie_year_not_the_name():
    stats = pd.DataFrame({
        'player_display_name': ['DeVonta Smith', 'Omar Cooper Jr.'], 'team': ['PHI', 'NYJ'], 'position': ['WR', 'WR'],
        'rookie_year': [2021.0, 2026.0], 'fantasy_points': [12.0, 10.0], 'snap_pct_avg': [80.0, 50.0], 'week': [6, 6],
    })
    names = {'devonta smith', 'omar cooper jr.'}         # a 2026 rookie DB shares the veteran WR's name
    board = build_rookie_watch(stats, 'player_display_name', 'team', names, 2026)
    assert set(board['Player']) == {'Omar Cooper Jr.'}


def test_pfr_pass_season_rolls_up_weekly_logs(monkeypatch):
    weekly = pd.DataFrame({'pfr_player_name': ['A', 'A', 'B'], 'team': ['KC', 'KC', 'BUF'], 'game_type': 'REG',
                           'times_pressured': [10.0, 5.0, 0.0], 'times_pressured_pct': [0.25, 0.125, 0.0]})
    monkeypatch.setattr(ld, '_pfr_weekly', lambda year, stat_type: weekly)
    out = ld._pfr_pass_season_from_weeks(2026)
    row = out[out['player'] == 'A'].iloc[0]
    assert row['pass_attempts'] == 80.0                  # 10 / 0.25 + 5 / 0.125
    assert row['pressure_pct'] == round(100 * 15 / 80, 1)
    assert 'B' not in set(out['player'])                 # no measurable dropbacks
    assert out.attrs['source'] == 'weekly'


def test_default_season_index(monkeypatch):
    import ui.components as comp
    monkeypatch.setattr(comp, 'season_has_played_games', lambda season: True)
    assert comp.default_season_index([2026, 2025]) == 0
    monkeypatch.setattr(comp, 'season_has_played_games', lambda season: False)
    assert comp.default_season_index([2026, 2025]) == 1


def test_season_being_prepared():
    import datetime
    from ui.components import _season_being_prepared
    assert _season_being_prepared(datetime.date(2026, 10, 21)) == 2026
    assert _season_being_prepared(datetime.date(2027, 1, 20)) == 2026
    assert _season_being_prepared(datetime.date(2027, 3, 15)) == 2027


def test_pff_load_memo_sees_a_changed_export(tmp_path):
    year_dir = tmp_path / '2026'
    year_dir.mkdir()
    export = year_dir / 'receiving_scheme.csv'
    export.write_text('a\n1\n')
    before = pa._files_signature(tmp_path, 2026)
    time.sleep(0.01)
    export.write_text('a\n1\n2\n')
    assert pa._files_signature(tmp_path, 2026) != before
