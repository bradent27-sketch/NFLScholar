"""data.ledger_scorecard (2026-10-13): the live track record - each player's last pre-kickoff board, paired pools."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

from data import ledger_scorecard as lsc  # noqa: E402
from data.prediction_ledger import STAT_COLS  # noqa: E402
from data.transforms import score_projected_stats  # noqa: E402

SCHEDULE = pd.DataFrame({
    'game_type': ['REG', 'REG'], 'week': [5, 5],
    'gameday': ['2026-10-08', '2026-10-11'], 'gametime': ['20:15', '13:00'],
    'home_team': ['AAA', 'CCC'], 'away_team': ['BBB', 'DDD'],
})


def _row(player, team, pts, build_ts, pos='WR', market=None, fp=None, **stats):
    rec = {'Player': player, 'Pos': pos, 'Team': team, 'week': 5, 'Model Proj Pts': pts,
           'Raw Model Proj Pts': pts, 'build_ts': pd.Timestamp(build_ts, tz='UTC'),
           'Mkt Market Pts': market, 'Mkt Coverage': 1.0 if market is not None else np.nan, 'FP Proj Pts PPR': fp}
    rec.update(stats)
    return rec


def test_kickoffs_are_eastern_converted_to_utc():
    ko = lsc.kickoffs_utc(SCHEDULE)
    assert ko[(5, 'AAA')] == ko[(5, 'BBB')] == pd.Timestamp('2026-10-09 00:15', tz='UTC')
    assert ko[(5, 'CCC')] == pd.Timestamp('2026-10-11 17:00', tz='UTC')


def test_each_player_gets_his_last_board_before_his_own_kickoff():
    ledger = pd.DataFrame([
        _row('Thursday Guy', 'AAA', 10.0, '2026-10-08 12:00'),
        _row('Thursday Guy', 'AAA', 99.0, '2026-10-10 12:00'),     # built after his Thursday game: never used
        _row('Sunday Guy', 'CCC', 12.0, '2026-10-08 12:00'),
        _row('Sunday Guy', 'CCC', 14.0, '2026-10-10 12:00'),       # Saturday board, before his Sunday game: used
        _row('Late Guy', 'CCC', 7.0, '2026-10-12 12:00'),          # only a post-kickoff board exists
    ])
    ledger['_key'] = ledger['Player']
    picked = lsc.select_pre_kickoff_rows(ledger, lsc.kickoffs_utc(SCHEDULE)).set_index('Player')
    assert picked.at['Thursday Guy', 'Model Proj Pts'] == 10.0 and picked.at['Thursday Guy', 'pre_kickoff']
    assert picked.at['Sunday Guy', 'Model Proj Pts'] == 14.0 and picked.at['Sunday Guy', 'pre_kickoff']
    assert picked.at['Late Guy', 'Model Proj Pts'] == 7.0 and not picked.at['Late Guy', 'pre_kickoff']


def test_infer_scoring_reads_the_mode_off_the_stat_line():
    line = {'targets': 8.0, 'receptions': 6.0, 'receiving_yards': 70.0, 'receiving_tds': 0.5}
    rows = []
    for mode in ('Full PPR', 'Half-PPR'):
        rec = {s: line.get(s, 0.0) for s in STAT_COLS}
        rec['Raw Model Proj Pts'] = score_projected_stats(line, mode)
        rows.append(pd.DataFrame([rec]))
    assert lsc.infer_scoring(rows[0]) == 'Full PPR'
    assert lsc.infer_scoring(rows[1]) == 'Half-PPR'


def _scored_frame():
    rng = np.random.default_rng(0)
    recs = []
    for i in range(30):
        actual = float(rng.uniform(2, 25))
        recs.append({'Player': f'P{i}', 'Pos': 'WR', 'week': 5, 'pre_kickoff': True,
                     'model_pts': actual + 3.0, 'market_pts': actual + 1.0 if i < 20 else np.nan,
                     'fp_pts': actual - 2.0, 'actual_pts': actual})
    return pd.DataFrame(recs)


def test_weekly_comparison_is_paired_on_rows_both_sources_have():
    f = _scored_frame()
    comp = lsc.weekly_comparison(f, 'Market', pos='WR', startable=False)
    wk = comp[comp['week'] == 5].iloc[0]
    assert wk['n'] == 20                                  # the 10 rows without a market number are left out
    assert np.isclose(wk['model_rmse'], 3.0) and np.isclose(wk['bench_rmse'], 1.0)
    assert np.isclose(wk['model_bias'], 3.0)
    fp = lsc.weekly_comparison(f, 'FantasyPros', pos='WR', startable=False)
    assert fp[fp['week'] == 5].iloc[0]['n'] == 30
    assert list(comp['week'])[-1] == 'All'


def test_rebuilt_weeks_stay_out_of_the_live_comparison():
    f = _scored_frame()
    f.loc[:, 'pre_kickoff'] = False
    assert lsc.weekly_comparison(f, 'Market', startable=False).empty


def test_disagreement_table_counts_who_was_closer():
    f = pd.DataFrame([
        {'Pos': 'WR', 'pre_kickoff': True, 'model_pts': 15.0, 'market_pts': 10.0, 'actual_pts': 14.0},  # model
        {'Pos': 'WR', 'pre_kickoff': True, 'model_pts': 15.0, 'market_pts': 10.0, 'actual_pts': 9.0},   # market
        {'Pos': 'WR', 'pre_kickoff': True, 'model_pts': 5.0, 'market_pts': 10.0, 'actual_pts': 4.0},    # model
        {'Pos': 'WR', 'pre_kickoff': True, 'model_pts': 10.5, 'market_pts': 10.0, 'actual_pts': 4.0},   # no gap
    ])
    t = lsc.disagreement_table(f, gaps=(2.0,), positions=(None,)).set_index('Side')
    assert t.at['Model higher', 'n'] == 2 and t.at['Model higher', 'Model closer'] == 0.5
    assert t.at['Model lower', 'n'] == 1 and t.at['Model lower', 'Model closer'] == 1.0


def test_line_calls_score_the_side_of_the_market_number():
    f = pd.DataFrame({
        'pre_kickoff': True,
        'receiving_yards': [80.0, 80.0, 50.0, 60.0],
        'Mkt receiving_yards': [70.0, 70.0, 60.0, 60.0],      # last row: no lean
        'actual_receiving_yards': [90.0, 40.0, 30.0, 99.0],
    })
    calls = lsc.line_calls(f, stats=('receiving_yards',)).iloc[0]
    assert calls['n'] == 3 and calls['overs'] == 2 and calls['unders'] == 1
    assert calls['over_hit'] == 0.5 and calls['under_hit'] == 1.0
    assert np.isclose(calls['hit'], 2 / 3)


def test_score_ledger_reads_fantasypros_in_the_scoring_being_scored():
    # regression: it used to take the first 'FP Proj Pts*' column (standard scoring) and score it against PPR
    from scripts.score_ledger import _fp_points_col
    ledger = pd.DataFrame(columns=['FP Proj Pts', 'FP Proj Pts PPR', 'FP Proj Pts Half'])
    assert _fp_points_col(ledger, 'Full PPR') == 'FP Proj Pts PPR'
    assert _fp_points_col(ledger, 'Half-PPR') == 'FP Proj Pts Half'
    assert _fp_points_col(ledger, 'Standard') == 'FP Proj Pts'
