"""data.prediction_ledger: a permanent per-week record of model/market/FP."""
import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.prediction_ledger as pl  # noqa: E402


def _model_df():
    return pd.DataFrame({
        'Player': ['A Back', 'B Wideout'],
        'Pos': ['RB', 'WR'],
        'Team': ['KC', 'DAL'],
        'Opponent': ['DEN', 'PHI'],
        'Games This Season': [2, 2],
        'Availability': [1.0, 1.0],
        'Injury Status': ['', ''],
        'Raw Model Proj Pts': [14.2, 11.8],
        'Model Proj Pts': [14.2, 11.8],
        'rushing_attempts': [15.0, 0.0],
        'rushing_yards': [70.0, 0.0],
        'targets': [3.0, 8.0],
        'receiving_yards': [20.0, 90.0],
    })


def _market_df():
    df = pd.DataFrame({
        'Player': ['A Back', 'B Wideout'],
        'Market Pts': [13.0, 12.5],
        'Coverage': [0.9, 0.95],
        'rushing_yards': [65.0, 0.0],
        'receiving_yards': [18.0, 95.0],
    })
    df.attrs['consensus_stats'] = ['rushing_yards', 'receiving_yards']
    return df


def _fp_df():
    return pd.DataFrame({
        'Player': ['A Back', 'B Wideout'],
        'FP Proj Pts PPR': [13.5, 12.0],
        'rushing_yards': [68.0, 0.0],
        'receiving_yards': [19.0, 88.0],
    })


def test_build_ledger_frame_carries_model_stats_and_metadata():
    out = pl.build_ledger_frame(2026, 3, _model_df())
    assert list(out['Player']) == ['A Back', 'B Wideout']
    assert 'rushing_attempts' in out.columns and 'targets' in out.columns
    assert (out['year'] == 2026).all() and (out['week'] == 3).all()
    assert out['feature_hash'].iloc[0] == pl.feature_hash(None)
    assert 'build_ts' in out.columns


def test_build_ledger_frame_merges_market_and_fp_by_name():
    out = pl.build_ledger_frame(2026, 3, _model_df(), market_df=_market_df(), fp_weekly=_fp_df())
    row = out.set_index('Player').loc['A Back']
    assert row['Mkt Market Pts'] == 13.0
    assert row['Mkt Coverage'] == 0.9
    assert row['Mkt rushing_yards'] == 65.0
    assert row['Mkt receiving_yards'] == 18.0
    assert row['FP Proj Pts PPR'] == 13.5
    assert row['FP rushing_yards'] == 68.0
    # The model's OWN stat column is untouched by the market/FP merge - it
    # keeps its own name, the market/FP versions are prefixed instead.
    assert row['rushing_yards'] == 70.0


def test_build_ledger_frame_tolerates_missing_market_and_fp():
    out = pl.build_ledger_frame(2026, 3, _model_df(), market_df=pd.DataFrame(), fp_weekly=None)
    assert len(out) == 2
    assert 'Mkt Market Pts' not in out.columns


def test_feature_hash_is_stable_and_order_independent():
    a = pl.feature_hash({'v2_foo', 'v2_bar'})
    b = pl.feature_hash({'v2_bar', 'v2_foo'})
    c = pl.feature_hash({'v2_foo'})
    assert a == b
    assert a != c
    assert pl.feature_hash(None) == 'unspecified'


def test_record_board_writes_a_file_and_throttles_reruns():
    with tempfile.TemporaryDirectory() as tmp:
        original_dir = pl.LEDGER_DIR
        pl.LEDGER_DIR = os.path.join(tmp, 'ledger')
        try:
            path1 = pl.record_board(2026, 3, _model_df(), feats={'v2_foo'})
            assert path1 is not None and os.path.exists(path1)
            df = pd.read_parquet(path1)
            assert len(df) == 2 and (df['week'] == 3).all()

            # A second call for the SAME (year, week, feature hash) within the
            # throttle window reuses the same file - a fragment rerunning on
            # every click must not spam the ledger directory.
            path2 = pl.record_board(2026, 3, _model_df(), feats={'v2_foo'})
            assert path2 == path1
            assert len(os.listdir(pl.LEDGER_DIR)) == 1

            # A DIFFERENT feature set is a different build worth keeping.
            path3 = pl.record_board(2026, 3, _model_df(), feats={'v2_bar'})
            assert path3 is not None and path3 != path1
            assert len(os.listdir(pl.LEDGER_DIR)) == 2
        finally:
            pl.LEDGER_DIR = original_dir


def test_record_board_never_raises_on_bad_input():
    assert pl.record_board(2026, 3, None) is None
    assert pl.record_board(2026, 3, pd.DataFrame()) is None
    assert pl.record_board(2026, 3, pd.DataFrame({'not_player': [1]})) is None


def test_latest_ledger_paths_picks_the_newest_per_week():
    with tempfile.TemporaryDirectory() as tmp:
        original_dir = pl.LEDGER_DIR
        pl.LEDGER_DIR = tmp
        try:
            os.makedirs(tmp, exist_ok=True)
            older = os.path.join(tmp, '2026_wk01_20260901T000000000000Z.parquet')
            newer = os.path.join(tmp, '2026_wk01_20260902T000000000000Z.parquet')
            other_week = os.path.join(tmp, '2026_wk02_20260901T000000000000Z.parquet')
            for p in (older, newer, other_week):
                _model_df().to_parquet(p, index=False)
            os.utime(older, (1000, 1000))
            os.utime(newer, (2000, 2000))
            latest = pl.latest_ledger_paths(2026)
            assert latest[(2026, 1)] == newer
            assert latest[(2026, 2)] == other_week
        finally:
            pl.LEDGER_DIR = original_dir
