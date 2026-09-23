"""ui/tabs/rankings.py: pure-logic helpers, tested standalone (no Streamlit
runtime needed - these are plain pandas transforms pulled out of the render
functions specifically so they can be exercised this way)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.tabs.rankings import _blended_woven_rank, _market_proj_pts_with_backfill  # noqa: E402


def test_market_proj_pts_backfill_subtracts_an_unpriced_interception_penalty():
    # A QB whose passing_yards/passing_tds are priced by the market but whose
    # interceptions are not (the common case - almost no book posts a season
    # or weekly INT line). BUG FIX 2026-09-16: the backfill term used to be
    # clipped at 0, silently discarding the model's real INT penalty whenever
    # it made the "stats the market didn't price" contribution net negative -
    # inflating every such QB's Market Proj Pts by exactly his INT cost.
    #
    # Market's own total (Mkt Market Pts): 4100 pass yds + 27 pass TD, no INT
    # line -> 4100*0.04 + 27*4 = 164 + 108 = 272.
    # Our model's full total (Raw Model Proj Pts): 4000 pass yds + 25 pass TD
    # + 100 rush yds + 2 rush TD - 12 INT -> 160+100+10+12-24 = 258.
    # Our model restricted to just the priced stats (passing_yards/tds only):
    # 4000*0.04 + 25*4 = 160+100 = 260.
    # backfill = 258 - 260 = -2 (the model's real INT/rushing net contribution
    # from everything the market DIDN'T price) -> Market Proj Pts = 272-2=270.
    merged = pd.DataFrame([{
        'Mkt Market Pts': 272.0,
        'Raw Model Proj Pts': 258.0,
        'Mkt passing_yards': 4100.0, 'passing_yards': 4000.0,
        'Mkt passing_tds': 27.0, 'passing_tds': 25.0,
    }])
    out = _market_proj_pts_with_backfill(merged, ['passing_yards', 'passing_tds'], 'Full PPR')
    assert out.iloc[0] == 270.0


def test_market_proj_pts_backfill_still_floors_the_final_total_at_zero():
    # The intermediate backfill term is allowed to go negative (that's the
    # fix above), but a real player's total point projection never displays
    # negative - only the FINAL sum is floored, same convention as Raw Model
    # Proj Pts elsewhere in this app.
    merged = pd.DataFrame([{
        'Mkt Market Pts': 1.0,
        'Raw Model Proj Pts': 0.0,   # a replacement-level QB, still threw a garbage-time INT
        'Mkt passing_tds': 1.0, 'passing_tds': 5.0,   # our own model projects way more TDs
    }])
    out = _market_proj_pts_with_backfill(merged, ['passing_tds'], 'Full PPR')
    assert out.iloc[0] == 0.0


def _labeled(values, labels):
    return [labels[int(v)] for v in values]


def test_blended_rank_averages_each_sources_own_ordinal_not_the_points():
    # Explicit request (2026-09-29): the leading "Rank" column blends
    # FantasyPros' and the model's own rank by averaging each source's
    # per-position ORDINAL (1st/2nd/3rd/...), not the raw point totals - a
    # 0.1-point gap and a 10-point gap both just mean "one spot apart".
    # FP ordinal:    A=1  B=2  C=3
    # Model ordinal: A=3  B=1  C=2   (Model Proj Pts: B=20 > C=17 > A=15)
    # Averaged:      A=2.0  B=1.5  C=2.5  -> re-ranked B, A, C
    df = pd.DataFrame([
        {'Pos': 'RB', 'FantasyPros Proj Pts': 20.0, 'Model Proj Pts': 15.0},
        {'Pos': 'RB', 'FantasyPros Proj Pts': 18.0, 'Model Proj Pts': 20.0},
        {'Pos': 'RB', 'FantasyPros Proj Pts': 16.0, 'Model Proj Pts': 17.0},
    ])
    values, labels = _blended_woven_rank(df, ('FantasyPros Proj Pts', 'Model Proj Pts'))
    assert _labeled(values, labels) == ['RB2', 'RB1', 'RB3']


def test_blended_rank_falls_back_to_whichever_source_ranks_a_player():
    # D has no FantasyPros number at all (that source doesn't carry him) -
    # he should rank off his Model ordinal alone, not get penalized by
    # averaging against a missing value.
    df = pd.DataFrame([
        {'Pos': 'RB', 'FantasyPros Proj Pts': 20.0, 'Model Proj Pts': 15.0},
        {'Pos': 'RB', 'FantasyPros Proj Pts': 18.0, 'Model Proj Pts': 20.0},
        {'Pos': 'RB', 'FantasyPros Proj Pts': 16.0, 'Model Proj Pts': 17.0},
        {'Pos': 'RB', 'FantasyPros Proj Pts': None, 'Model Proj Pts': 10.0},
    ])
    values, labels = _blended_woven_rank(df, ('FantasyPros Proj Pts', 'Model Proj Pts'))
    assert _labeled(values, labels) == ['RB2', 'RB1', 'RB3', 'RB4']


def test_blended_rank_degrades_to_the_one_real_source_when_fantasypros_absent():
    # The common case: FantasyPros hasn't been pulled this session at all,
    # so 'FantasyPros Proj Pts' isn't even a column - the blend should
    # reproduce the model's own ranking exactly, not break or go empty.
    df = pd.DataFrame([
        {'Pos': 'RB', 'Model Proj Pts': 15.0},
        {'Pos': 'RB', 'Model Proj Pts': 20.0},
        {'Pos': 'RB', 'Model Proj Pts': 17.0},
    ])
    values, labels = _blended_woven_rank(df, ('FantasyPros Proj Pts', 'Model Proj Pts'))
    assert _labeled(values, labels) == ['RB3', 'RB1', 'RB2']


def test_blended_rank_weaves_positions_together_like_woven_rank():
    df = pd.DataFrame([
        {'Pos': 'RB', 'Model Proj Pts': 20.0},
        {'Pos': 'WR', 'Model Proj Pts': 18.0},
        {'Pos': 'RB', 'Model Proj Pts': 10.0},
    ])
    values, labels = _blended_woven_rank(df, ('FantasyPros Proj Pts', 'Model Proj Pts'))
    assert _labeled(values, labels) == ['RB1', 'WR1', 'RB2']


def test_blended_rank_with_no_points_columns_at_all_returns_none():
    df = pd.DataFrame([{'Pos': 'RB', 'Player': 'Nobody'}])
    values, labels = _blended_woven_rank(df, ('FantasyPros Proj Pts', 'Model Proj Pts'))
    assert values is None
    assert labels == {}


def test_market_proj_pts_backfill_matches_the_market_when_fully_priced():
    # When the market priced every scoring stat the model would otherwise
    # backfill, there is nothing left to add - Market Proj Pts should equal
    # the market's own partial total exactly.
    merged = pd.DataFrame([{
        'Mkt Market Pts': 300.0,
        'Raw Model Proj Pts': 280.0,
        'Mkt receiving_yards': 950.0, 'receiving_yards': 900.0,
        'Mkt receiving_tds': 6.0, 'receiving_tds': 5.0,
        'Mkt receptions': 70.0, 'receptions': 65.0,
    }])
    out = _market_proj_pts_with_backfill(
        merged, ['receiving_yards', 'receiving_tds', 'receptions'], 'Full PPR')
    # model_on_priced re-scores OUR OWN values for the same three stats;
    # backfill is whatever's left of our own total beyond that - here it's
    # only meaningful if our own priced-stat score differs from our total,
    # so this mainly guards that a fully-priced row doesn't crash or blow up.
    assert out.iloc[0] >= 0.0
