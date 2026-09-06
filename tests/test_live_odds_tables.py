"""Offline tests for the pure table/consensus helpers behind the Live Odds tab.

No Streamlit runtime - these operate on the raw Odds-API payload shape and the
frames the tab renders. The tab's own filter/sort wiring is exercised only
indirectly (the helpers it calls), but that is where the logic that can be
wrong actually lives.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

import ui.tabs.live_odds as live_odds  # noqa: E402
from ui.tabs.live_odds import (  # noqa: E402
    _build_lines_table, _lines_consensus, _build_props_long_table,
    _build_props_comparison_table, _render_odds_api_devig,
)


def _game():
    return {
        'home_team': 'Buffalo Bills', 'away_team': 'Kansas City Chiefs',
        'bookmakers': [
            {'title': 'DraftKings', 'markets': [
                {'key': 'h2h', 'outcomes': [
                    {'name': 'Buffalo Bills', 'price': -160},
                    {'name': 'Kansas City Chiefs', 'price': 140}]},
                {'key': 'spreads', 'outcomes': [
                    {'name': 'Buffalo Bills', 'point': -3.5},
                    {'name': 'Kansas City Chiefs', 'point': 3.5}]},
                {'key': 'totals', 'outcomes': [
                    {'name': 'Over', 'point': 48.5}, {'name': 'Under', 'point': 48.5}]},
            ]},
            {'title': 'FanDuel', 'markets': [
                {'key': 'spreads', 'outcomes': [
                    {'name': 'Buffalo Bills', 'point': -2.5},
                    {'name': 'Kansas City Chiefs', 'point': 2.5}]},
                {'key': 'totals', 'outcomes': [{'name': 'Over', 'point': 49.0}]},
            ]},
        ],
    }


def _props():
    return {'bookmakers': [
        {'title': 'DraftKings', 'markets': [{'key': 'player_pass_yds', 'outcomes': [
            {'name': 'Over', 'description': 'Josh Allen', 'point': 270.5, 'price': -115},
            {'name': 'Under', 'description': 'Josh Allen', 'point': 270.5, 'price': -105}]}]},
        {'title': 'FanDuel', 'markets': [{'key': 'player_pass_yds', 'outcomes': [
            {'name': 'Over', 'description': 'Josh Allen', 'point': 272.5, 'price': -110}]}]},
    ]}


# --- game lines ----------------------------------------------------------

def test_lines_table_is_one_row_per_book():
    df = _build_lines_table(_game())
    assert list(df['Book']) == ['DraftKings', 'FanDuel']
    assert df.loc[df['Book'] == 'DraftKings', 'Home Spread'].iloc[0] == -3.5


def test_consensus_names_the_home_favorite_and_median_total():
    line = _lines_consensus(_build_lines_table(_game()), 'Buffalo Bills', 'Kansas City Chiefs')
    assert 'Buffalo Bills' in line          # negative home spread => home favored
    assert 'Kansas City Chiefs' not in line
    assert 'O/U 48.75' in line              # median(48.5, 49.0)
    assert 'median of 2 books' in line


def test_consensus_flips_to_away_when_home_spread_is_positive():
    g = _game()
    for bk in g['bookmakers']:
        for m in bk['markets']:
            if m['key'] == 'spreads':
                for o in m['outcomes']:
                    o['point'] = -o['point']   # away now favored
    line = _lines_consensus(_build_lines_table(g), 'Buffalo Bills', 'Kansas City Chiefs')
    assert 'Kansas City Chiefs' in line
    assert line.split(' · ')[0].startswith('Kansas City Chiefs -')


def test_consensus_pickem_when_spread_is_zero():
    g = _game()
    for bk in g['bookmakers']:
        for m in bk['markets']:
            if m['key'] == 'spreads':
                for o in m['outcomes']:
                    o['point'] = 0.0
    line = _lines_consensus(_build_lines_table(g), 'Buffalo Bills', 'Kansas City Chiefs')
    assert "Pick'em" in line


def test_consensus_tolerates_a_book_with_no_spread_or_total():
    df = _build_lines_table({'home_team': 'A', 'away_team': 'B',
                             'bookmakers': [{'title': 'X', 'markets': []}]})
    # No spread/total columns at all - must not raise.
    line = _lines_consensus(df, 'A', 'B')
    assert 'median of 1 book' in line


# --- player props ------------------------------------------------------

def test_props_comparison_pivots_one_row_per_bet_one_col_per_book():
    wide = _build_props_comparison_table(_build_props_long_table(_props()))
    assert set(['Market', 'Player', 'Selection']).issubset(wide.columns)
    assert 'DraftKings' in wide.columns and 'FanDuel' in wide.columns
    over = wide[wide['Selection'] == 'Over'].iloc[0]
    assert '270.5' in str(over['DraftKings'])
    assert '272.5' in str(over['FanDuel'])


def test_props_comparison_sorts_by_player_then_market():
    long_df = _build_props_long_table(_props())
    # add a second player so a sort is observable
    long_df = pd.concat([long_df, long_df.assign(Player='Aaron Aaronson')], ignore_index=True)
    wide = _build_props_comparison_table(long_df)
    wide = wide.sort_values([c for c in ('Player', 'Market') if c in wide.columns])
    assert list(wide['Player'])[0] == 'Aaron Aaronson'


def test_empty_props_long_table_yields_empty_comparison():
    assert _build_props_comparison_table(pd.DataFrame()).empty


class _Sink:
    def __init__(self, monkeypatch):
        self.text, self.frames = [], []
        monkeypatch.setattr(live_odds.st, "markdown", lambda b, *a, **k: self.text.append(str(b)))
        monkeypatch.setattr(live_odds.st, "caption", lambda b, *a, **k: self.text.append(str(b)))
        monkeypatch.setattr(live_odds.st, "dataframe", lambda f, *a, **k: self.frames.append(f))
        monkeypatch.setattr(live_odds, "style_plain_dataframe", lambda f, *a, **k: f)
        monkeypatch.setattr(live_odds, "df_auto_height", lambda *a, **k: 200)

    @property
    def joined(self):
        return "\n".join(self.text)


def _leaned_props_payload():
    """One event, two books; DK leans the receiving-yards over, FanDuel is even.
    A yardage line has no median->mean skew term, so an even price implies its
    own number and a leaned one does not."""
    return {
        'id': 'evt-9',
        'bookmakers': [
            {'key': 'draftkings', 'title': 'DraftKings', 'markets': [
                {'key': 'player_reception_yds', 'outcomes': [
                    {'name': 'Over', 'description': 'Stefon Diggs', 'point': 68.5, 'price': -170},
                    {'name': 'Under', 'description': 'Stefon Diggs', 'point': 68.5, 'price': 135},
                ]}]},
            {'key': 'fanduel', 'title': 'FanDuel', 'markets': [
                {'key': 'player_reception_yds', 'outcomes': [
                    {'name': 'Over', 'description': 'Stefon Diggs', 'point': 68.5, 'price': -110},
                    {'name': 'Under', 'description': 'Stefon Diggs', 'point': 68.5, 'price': -110},
                ]}]},
        ],
    }


def test_odds_api_devig_view_shows_the_leaned_line_shift(monkeypatch):
    sink = _Sink(monkeypatch)
    _render_odds_api_devig(_leaned_props_payload())
    assert sink.frames, "expected a de-vigged table"
    frame = sink.frames[0].reset_index()
    cells = " ".join(str(v) for v in frame.values.ravel())
    # DK's -170/+135 over lean -> its cell carries an arrow to a number above 68.5
    assert "→" in cells
    dk_col = [c for c in frame.columns if 'DraftKings' in str(c)][0]
    assert "→" in str(frame[dk_col].iloc[0])
    # FanDuel's even price on a yardage line -> implies its own number, no arrow
    fd_col = [c for c in frame.columns if 'FanDuel' in str(c)][0]
    assert "→" not in str(frame[fd_col].iloc[0])
    assert "no extra API credit" in sink.joined


def test_odds_api_devig_view_is_silent_on_an_empty_payload(monkeypatch):
    sink = _Sink(monkeypatch)
    _render_odds_api_devig({'id': 'x', 'bookmakers': []})
    assert not sink.frames and not sink.text


def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith('test_') and callable(f)]
    bad = []
    for n, f in tests:
        try:
            f()
            print(f"  PASS  {n}")
        except Exception as exc:  # noqa: BLE001
            bad.append((n, exc))
            print(f"  FAIL  {n}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(bad)}/{len(tests)} passed")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
