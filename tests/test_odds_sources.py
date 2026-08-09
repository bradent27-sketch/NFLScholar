"""
Offline tests for the odds adapters.

WHY THESE EXIST IN THIS SHAPE. The sportsbook hosts are unreachable from
some networks (they were entirely blocked from the sandbox this was written
in), so the fetch path cannot be exercised everywhere. That is survivable
because the fetch path is four lines of `requests.get`; the bugs in an
adapter live in the PARSE - a renamed key, a nesting level that moved, a
join that silently produces zero rows. All of that is pure functions over a
recorded payload, and all of it is tested here with no network at all.

Runs two ways, on purpose: `python tests/test_odds_sources.py` needs nothing
but the app's own dependencies, and `pytest tests/` works if pytest is
installed. The project has no test framework, and requiring one to check a
parser would mean these never get run.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

from data.odds_sources import (  # noqa: E402
    parse_underdog_payload, parse_prizepicks_payload, parse_odds_api_props,
    standardize_team, normalize_stat, combine_props, load_props_fixture,
    odds_api_bookmakers, PROP_COLUMNS, prizepicks_leagues,
    implausible_period_rows,
)
from data.odds_projections import (  # noqa: E402
    market_stat_lines, score_market_lines, compare_to_board,
    blend_market_into_projection, attach_board_player, build_book_projection,
    attach_book_projection,
)

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')
SCORING = {
    'pass_yd': 0.04, 'pass_td': 4, 'pass_int': -2, 'pass_2pt': 2,
    'rush_yd': 0.1, 'rush_td': 6, 'rush_att': 0.0, 'rush_2pt': 2,
    'rec': 1.0, 'rec_yd': 0.1, 'rec_td': 6, 'rec_2pt': 2,
    'fumble_lost': -2, 'fg_0_39': 3, 'fg_40_49': 4, 'fg_50_plus': 5, 'pat': 1,
    'te_premium': 0.0, 'bonus_mode': 'cumulative',
}


def _underdog():
    return parse_underdog_payload(
        load_props_fixture(os.path.join(FIXTURES, 'underdog_over_under_lines.json')))


def _prizepicks():
    return parse_prizepicks_payload(
        load_props_fixture(os.path.join(FIXTURES, 'prizepicks_projections.json')))


# ---------------------------------------------------------------------------

def test_team_standardization():
    for raw, expected in (
        ('KC', 'KC'), ('Kansas City Chiefs', 'KC'), ('Chiefs', 'KC'),
        ('JAC', 'JAX'), ('JAX', 'JAX'), ('Jaguars', 'JAX'),
        ('WSH', 'WAS'), ('Commanders', 'WAS'),
        ('LAR', 'LA'), ('Rams', 'LA'), ('SD', 'LAC'), ('OAK', 'LV'),
        ('kansas city', 'KC'),
    ):
        assert standardize_team(raw) == expected, f"{raw!r} -> {standardize_team(raw)!r}"
    # Ambiguous and unknown both resolve to '' rather than guessing - a prop
    # joined to the wrong team is worse than one that visibly didn't join.
    for ambiguous in ('New York', 'Los Angeles', '', None, 'Toronto Raptors'):
        assert standardize_team(ambiguous) == '', f"{ambiguous!r} should not resolve"


def test_stat_normalization():
    assert normalize_stat('Receiving Yards') == ('receiving_yards', True)
    assert normalize_stat('receiving_yards') == ('receiving_yards', True)
    assert normalize_stat('rec_yds') == ('receiving_yards', True)
    assert normalize_stat('interceptions') == ('passing_interceptions', True)
    assert normalize_stat('Rushing Attempts') == ('carries', True)
    # Combination markets are recognized but never scorable: mapping
    # "rush+rec yards" onto rushing_yards would corrupt the projection.
    name, scorable = normalize_stat('rushing_receiving_yards')
    assert scorable is False and name == 'rush_rec_yards'
    assert normalize_stat('Total Bases') == (None, False)


def test_underdog_parses_and_joins():
    """Structure verified against a real 12.6MB payload in Aug 2026."""
    props, err = _underdog()
    assert err is None, err
    assert list(props.columns) == PROP_COLUMNS
    # 13 lines in. Dropped: 1 suspended, 1 orphan appearance_id, 1 appearance
    # whose player is missing, 1 MLB line (the payload carries every sport at
    # once and this adapter is asked for NFL). 8 survive.
    assert len(props) == 8, props[['player', 'market_raw', 'period']].to_string()
    assert 'Some Shortstop' not in set(props['player']), "MLB must be filtered out"

    jj = props[props['player'] == 'Justin Jefferson']
    season = jj[jj['period'] == 'season']
    assert len(season) == 3, "three season-long lines for the receiver"
    assert set(season['market']) == {'receiving_yards', 'receptions', 'receiving_tds'}
    assert float(season[season['market'] == 'receiving_yards']['line'].iloc[0]) == 1275.5
    assert (season['player_key'] == 'justinjefferson').all()
    # A single-game line must not be mistaken for a season-long one.
    assert (jj['period'] == 'game').sum() == 2


def test_underdog_season_match_type_is_series():
    """
    THE BUG THIS LOCKS IN. Underdog's match_type values are 'Game', 'Series'
    and 'SoloGame' - season-long is 'Series', and nothing about the word says
    so. The original code looked for the substring "season", matched none of
    them, and labelled all 356 season-long NFL lines as single-game, so the
    season projection silently had nothing to work with.
    """
    props, _ = _underdog()
    season = props[props['period'] == 'season']
    assert len(season) > 0, "match_type 'Series' must read as season-long"
    assert set(season['market_raw']) >= {'season_receiving_yards', 'season_pass_yards'}


def test_underdog_team_ids_are_uuids_resolved_via_games():
    """
    THE OTHER BUG. A player's team_id is a UUID and the abbreviation exists
    only on the games array, in abbreviated_title ("NE @ SEA") beside
    home_team_id / away_team_id. Passing the UUID straight to
    standardize_team resolved 0 of 5,314 real lines.
    """
    props, _ = _underdog()
    assert props[props['player'] == 'Justin Jefferson']['team'].iloc[0] == 'MIN'
    assert props[props['player'] == 'Patrick Mahomes']['team'].iloc[0] == 'SEA'
    # 'JAC' in Underdog's title, 'JAX' in this app.
    assert props[props['player'] == 'Travis Etienne']['team'].iloc[0] == 'JAX'
    assert (props['team'] != '').all(), "every NFL line should resolve a team"
    # Position comes from position_name, not the UUID position_id.
    assert props[props['player'] == 'Patrick Mahomes']['position'].iloc[0] == 'QB'


def test_underdog_payouts_and_partial_game_markets():
    props, _ = _underdog()
    recs = props[(props['player'] == 'Justin Jefferson') & (props['market'] == 'receptions')]
    assert float(recs['over_payout'].iloc[0]) == 0.95
    assert float(recs['under_payout'].iloc[0]) == 1.05
    # Quarter/half markets are real and must never reach the scoring path.
    partial = props[props['market_raw'] == 'period_1_receiving_yds']
    assert len(partial) == 1 and bool(partial['scorable'].iloc[0]) is False


def test_underdog_combo_market_not_scorable():
    """"Rush + Rec TDs" is Underdog's highest-volume NFL market and it is a
    SUM - mapping it onto rushing_tds would inflate every back on the board."""
    props, _ = _underdog()
    combo = props[props['market_raw'] == 'rush_rec_tds']
    assert len(combo) == 1
    assert bool(combo['scorable'].iloc[0]) is False


def test_prizepicks_parses_included_lookup():
    props, err = _prizepicks()
    assert err is None, err
    # 6 in, minus the unresolvable player pointer and the null line_score.
    assert len(props) == 4, props[['player', 'market_raw']].to_string()
    jj = props[props['player'] == 'Justin Jefferson']
    assert len(jj) == 3
    assert (jj['period'] == 'season').all()
    assert props[props['player'] == 'Patrick Mahomes']['period'].iloc[0] == 'game'


def test_bad_payloads_do_not_raise():
    for payload in (None, {}, [], 'nonsense', {'players': [], 'over_under_lines': []}):
        props, err = parse_underdog_payload(payload)
        assert props.empty and err
        props, err = parse_prizepicks_payload(payload)
        assert props.empty and err


def test_odds_api_bookmakers_and_props():
    payload = [{
        'id': 'evt-1',
        'bookmakers': [
            {'key': 'fanduel', 'title': 'FanDuel', 'markets': [
                {'key': 'player_pass_yds', 'outcomes': [
                    {'name': 'Over', 'description': 'Patrick Mahomes', 'point': 275.5, 'price': -110},
                    {'name': 'Under', 'description': 'Patrick Mahomes', 'point': 275.5, 'price': -110},
                ]}]},
            {'key': 'draftkings', 'title': 'DraftKings', 'markets': [
                {'key': 'player_reception_yds', 'outcomes': [
                    {'name': 'Over', 'description': 'Justin Jefferson', 'point': 88.5, 'price': -115},
                ]}]},
        ]}]
    books = odds_api_bookmakers(payload)
    assert set(books['key']) == {'fanduel', 'draftkings'}
    assert books[books['key'] == 'fanduel']['markets'].iloc[0] == 'player_pass_yds'

    props, err = parse_odds_api_props(payload)
    assert err is None and len(props) == 3
    # Per-game, always - these must never reach the season-long comparison.
    assert (props['period'] == 'game').all()

    only_fd, _ = parse_odds_api_props(payload, bookmaker='fanduel')
    assert len(only_fd) == 2 and only_fd['provider'].str.contains('FanDuel').all()


def test_market_projection_scores_and_measures_coverage():
    ud, _ = _underdog()
    pp, _ = _prizepicks()
    combined = combine_props(ud, pp)
    rows = market_stat_lines(combined, season_only=True)
    assert not rows.empty

    jj = rows[rows['player_key'] == 'justinjefferson'].iloc[0]
    # Two providers priced the same receiver; the median of the two is taken
    # rather than either one alone.
    assert float(jj['receiving_yards']) == (1275.5 + 1240.5) / 2
    assert float(jj['receptions']) == (92.5 + 90.5) / 2
    assert 'Underdog' in jj['providers'] and 'PrizePicks' in jj['providers']

    scored = score_market_lines(rows, SCORING)
    jj_scored = scored[scored['player_key'] == 'justinjefferson'].iloc[0]
    assert float(jj_scored['Coverage']) == 1.0, "all three WR key stats priced"

    yards = (1275.5 + 1240.5) / 2
    recs = (92.5 + 90.5) / 2
    tds = (8.5 + 9.5) / 2 * 1.05          # median-to-mean bump on TDs
    expected = round(yards * 0.1 + recs * 1.0 + tds * 6, 1)
    assert abs(float(jj_scored['Market Pts']) - expected) < 0.15, (
        f"{jj_scored['Market Pts']} vs {expected}")

    # The quarterback was only priced season-long on passing stats, so he is
    # covered but not fully - and that has to be visible, not silent.
    mahomes = scored[scored['player_key'] == 'patrickmahomes'].iloc[0]
    assert 0 < float(mahomes['Coverage']) < 1.0


def test_compare_to_board_scores_like_for_like():
    """
    THE MOST DANGEROUS BUG THIS MODULE HAD. Underdog's real season-long board
    posts receiving yards and receiving TDs but NO receptions, so a market
    total is missing a fifth of a receiver's half-PPR value through no error.
    Compared against our FULL projection, every receiver showed a +40% to
    +60% edge, all in the same direction - which is exactly what a genuine
    market inefficiency looks like. Our side is now re-scored over only the
    stats that market priced.
    """
    ud, _ = _underdog()
    rows = market_stat_lines(combine_props(ud), season_only=True)
    scored = score_market_lines(rows, SCORING)
    board = pd.DataFrame({
        'Player': ['Justin Jefferson', 'Patrick Mahomes', 'Somebody Else'],
        'Pos': ['WR', 'QB', 'RB'],
        'Team': ['MIN', 'KC', 'DAL'],
        'Proj Pts': [300.0, 320.0, 200.0],
        'Board Rank': [3, 20, 40],
        'receiving_yards': [1300.0, 0.0, 400.0],
        'receptions': [95.0, 0.0, 40.0],
        'receiving_tds': [9.0, 0.0, 2.0],
        'passing_yards': [0.0, 4400.0, 0.0],
        'passing_tds': [0.0, 33.0, 0.0],
        'rushing_yards': [0.0, 300.0, 900.0],
    })
    comparison, meta = compare_to_board(board, scored, SCORING)
    assert meta['scope'] == 'matched'
    assert 'Justin Jefferson' in set(comparison['Player'])
    assert 'Somebody Else' not in set(comparison['Player'])

    # The receiver was priced on all three of his stats here, so matched
    # scope equals the sum of those three under this scoring.
    jj = comparison[comparison['Player'] == 'Justin Jefferson'].iloc[0]
    expected = 1300.0 * 0.1 + 95.0 * 1.0 + 9.0 * 6
    assert abs(float(jj['Ours (matched)']) - expected) < 0.05, jj['Ours (matched)']
    assert abs(float(jj['Edge']) - (expected - float(jj['Market Pts']))) < 0.05

    # The QB was priced on passing only, so his RUSHING yards must be
    # excluded from our side - that is the whole point of matched scope.
    qb = comparison[comparison['Player'] == 'Patrick Mahomes'].iloc[0]
    passing_only = 4400.0 * 0.04 + 33.0 * 4
    assert abs(float(qb['Ours (matched)']) - passing_only) < 0.05, qb['Ours (matched)']

    edges = comparison['Edge %'].abs().tolist()
    assert edges == sorted(edges, reverse=True)


def test_compare_falls_back_when_board_has_no_stat_columns():
    """A board with no raw stats would score every matched total as 0 and
    report enormous negative edges. It falls back and flags itself instead."""
    ud, _ = _underdog()
    scored = score_market_lines(market_stat_lines(combine_props(ud)), SCORING)
    board = pd.DataFrame({'Player': ['Justin Jefferson'], 'Pos': ['WR'],
                          'Proj Pts': [300.0]})
    comparison, meta = compare_to_board(board, scored, SCORING)
    assert 'full projection' in meta['scope']
    if not comparison.empty:
        assert float(comparison.iloc[0]['Ours (matched)']) == 300.0


def test_blend_is_off_by_default():
    """
    Off is the shipped default and a considered one: the board already blends
    toward the market through ADP and ECR, so a second dose inside the
    projection double-counts the same opinion invisibly.
    """
    ud, _ = _underdog()
    scored = score_market_lines(market_stat_lines(combine_props(ud)), SCORING)
    board = _board_with_stats()
    same, moved = blend_market_into_projection(board, scored, weight=0.0,
                                               scoring=SCORING)
    assert moved == 0
    assert same['Proj Pts'].tolist() == board['Proj Pts'].tolist()


def test_two_tier_name_match_handles_suffixes():
    """
    The bug this locks in: the board says "Patrick Mahomes II" and every
    book says "Patrick Mahomes", so an exact-key join drops him - and with
    him every Jr./Sr./II/III in the league. Caught by a real miss in the UI,
    not by inspection.
    """
    ud, _ = _underdog()
    scored = score_market_lines(market_stat_lines(combine_props(ud)), SCORING)
    board = pd.DataFrame({
        'Player': ['Patrick Mahomes II', 'Justin Jefferson'],
        'Pos': ['QB', 'WR'],
        'Proj Pts': [320.0, 300.0],
    })
    resolved = attach_board_player(scored, board)
    mahomes = resolved[resolved['player_key'] == 'patrickmahomes'].iloc[0]
    assert mahomes['board_player'] == 'Patrick Mahomes II', "suffix fallback must match"
    jefferson = resolved[resolved['player_key'] == 'justinjefferson'].iloc[0]
    assert jefferson['board_player'] == 'Justin Jefferson', "exact match still wins"

    comparison, meta = compare_to_board(board, scored, SCORING)
    assert 'Patrick Mahomes II' in set(comparison['Player'])
    assert meta['unmatched'] == 0


def test_loose_match_refuses_ambiguous_names():
    """
    The reason the fallback is guarded: "Byron Murphy" and "Byron Murphy II"
    are two different real players who collapse to one key once suffixes are
    stripped. An ambiguous fallback must stay unmatched rather than attach
    one player's lines to the other's projection.
    """
    scored = pd.DataFrame({
        'player_key': ['byronmurphy'],
        'player': ['Byron Murphy'],
        'position': ['WR'],
        'Market Pts': [200.0],
        'Coverage': [1.0],
    })
    board = pd.DataFrame({
        'Player': ['Byron Murphy II', 'Byron Murphy Sr.'],
        'Pos': ['WR', 'WR'],
        'Proj Pts': [180.0, 90.0],
    })
    resolved = attach_board_player(scored, board)
    assert resolved['board_player'].isna().all(), "ambiguous stripped key must not resolve"

    blended, moved = blend_market_into_projection(board, scored, weight=0.5)
    assert moved == 0
    assert blended['Proj Pts'].tolist() == [180.0, 90.0]


def test_underdog_endpoint_ladder():
    """
    Underdog versions this path and the version moves - the reference the
    original URL came from was 22 months stale. The ladder tries newest
    first and stops at the first version that answers.
    """
    import data.odds_sources as osrc

    calls = []

    def fake_get(url, params=None, headers=None):
        calls.append(url)
        # Pretend only v5 is live.
        if '/v5/' in url:
            return {'players': [], 'over_under_lines': []}, None
        return None, f"{url} returned 404: not found"

    original = osrc._get_json
    try:
        osrc._get_json = fake_get
        payload, url, err = osrc.fetch_underdog_payload()
        assert err is None and payload is not None
        assert '/v5/' in url, url
        # Newest first, and it stops rather than walking the whole ladder.
        assert calls == [osrc.UNDERDOG_LINE_ENDPOINTS[0],
                         osrc.UNDERDOG_LINE_ENDPOINTS[1],
                         osrc.UNDERDOG_LINE_ENDPOINTS[2]]

        # A refusal is about the client, not the version: walking on would
        # just collect four more 403s from a host that already said no.
        calls.clear()
        osrc._get_json = lambda url, params=None, headers=None: (
            calls.append(url) or (None, f"{url} refused the request (403)."))
        payload, url, err = osrc.fetch_underdog_payload()
        assert payload is None and 'refused' in err
        assert len(calls) == 1, f"should stop on a refusal, tried {calls}"

        # Every version dead: one actionable message naming what was tried.
        calls.clear()
        osrc._get_json = lambda url, params=None, headers=None: (
            calls.append(url) or (None, f"{url} returned 404: not found"))
        payload, url, err = osrc.fetch_underdog_payload()
        assert payload is None
        assert 'UNDERDOG_LINE_ENDPOINTS' in err, err
        assert len(calls) == len(osrc.UNDERDOG_LINE_ENDPOINTS)
    finally:
        osrc._get_json = original


def test_empty_inputs_are_safe():
    assert combine_props().empty
    assert market_stat_lines(pd.DataFrame()).empty
    assert score_market_lines(pd.DataFrame(), SCORING).empty
    comparison, meta = compare_to_board(pd.DataFrame(), pd.DataFrame(), SCORING)
    assert comparison.empty and meta['matched'] == 0


def _board_with_stats():
    """A board carrying raw stat columns, as the real one does."""
    return pd.DataFrame({
        'Player': ['Justin Jefferson', 'Patrick Mahomes', 'Somebody Else'],
        'Pos': ['WR', 'QB', 'RB'],
        'Team': ['MIN', 'SEA', 'DAL'],
        'Proj Pts': [300.0, 320.0, 200.0],
        'receiving_yards': [1300.0, 0.0, 400.0],
        'receptions': [95.0, 0.0, 40.0],
        'receiving_tds': [9.0, 0.0, 2.0],
        'passing_yards': [0.0, 4400.0, 0.0],
        'passing_tds': [0.0, 33.0, 0.0],
        'rushing_yards': [0.0, 300.0, 900.0],
        'rushing_tds': [0.0, 2.0, 8.0],
    })


def test_book_projection_uses_book_where_priced_and_us_where_not():
    """
    The hybrid: book number for every stat it priced, ours for the rest, so
    the total covers a WHOLE player and sits on the same scale as Proj Pts.
    """
    ud, _ = _underdog()
    scored = score_market_lines(market_stat_lines(combine_props(ud)), SCORING)
    board = _board_with_stats()
    book = build_book_projection(board, scored, SCORING)
    assert not book.empty

    jj = book[book['board_player'] == 'Justin Jefferson'].iloc[0]
    # Underdog priced yards (1275.5), receptions (92.5) and TDs (8.5) for him
    # in this fixture, so all three come from the book; the TD line gets the
    # median-to-mean bump.
    expected = 1275.5 * 0.1 + 92.5 * 1.0 + 8.5 * 1.05 * 6
    assert abs(float(jj['Book Proj']) - expected) < 0.2, jj['Book Proj']
    assert int(jj['Book Stats']) == 3


def test_book_projection_falls_back_to_our_receptions():
    """
    THE WHOLE POINT. Underdog's real board posts receiving yards and TDs but
    NO receptions, so a hybrid that treated the gap as zero would collapse
    every receiver by a third in full PPR. Our reception estimate has to
    survive into the total.
    """
    ud, _ = _underdog()
    props = ud[~((ud['market'] == 'receptions') & (ud['period'] == 'season'))]
    scored = score_market_lines(market_stat_lines(combine_props(props)), SCORING)
    board = _board_with_stats()
    book = build_book_projection(board, scored, SCORING)

    jj = book[book['board_player'] == 'Justin Jefferson'].iloc[0]
    # Book yards + book TDs + OUR 95 receptions, not zero receptions.
    expected = 1275.5 * 0.1 + 9.0 * 1.05 * 0 + 8.5 * 1.05 * 6 + 95.0 * 1.0
    assert abs(float(jj['Book Proj']) - expected) < 0.2, jj['Book Proj']
    assert int(jj['Book Stats']) == 2, "only two stats came from the book"
    # And the book is now responsible for less of the total.
    assert float(jj['Book Share']) < 1.0


def test_book_share_reports_how_much_is_actually_the_market():
    ud, _ = _underdog()
    scored = score_market_lines(market_stat_lines(combine_props(ud)), SCORING)
    book = build_book_projection(_board_with_stats(), scored, SCORING)
    shares = book['Book Share'].dropna()
    assert len(shares) and (shares.between(0, 1.01)).all(), shares.tolist()


def test_attach_book_projection_leaves_unpriced_players_blank():
    ud, _ = _underdog()
    scored = score_market_lines(market_stat_lines(combine_props(ud)), SCORING)
    board = _board_with_stats()
    out = attach_book_projection(board, build_book_projection(board, scored, SCORING))
    assert 'Book Proj' in out.columns
    somebody = out[out['Player'] == 'Somebody Else'].iloc[0]
    assert pd.isna(somebody['Book Proj']), "the book never priced him"
    assert out[out['Player'] == 'Justin Jefferson']['Book Proj'].notna().all()
    # The board's own projection must be untouched by attaching a column.
    assert out['Proj Pts'].tolist() == board['Proj Pts'].tolist()


def test_blend_moves_toward_the_complete_hybrid_not_the_partial_total():
    """
    Blending toward Market Pts dragged every receiver down by his whole
    receptions total, because that number was never a projection of a whole
    player. The blend targets Book Proj now.
    """
    ud, _ = _underdog()
    scored = score_market_lines(market_stat_lines(combine_props(ud)), SCORING)
    board = _board_with_stats()
    book = build_book_projection(board, scored, SCORING)
    target = float(book[book['board_player'] == 'Justin Jefferson']['Book Proj'].iloc[0])

    blended, moved = blend_market_into_projection(board, scored, weight=0.5,
                                                  scoring=SCORING)
    assert moved >= 1
    jj = blended[blended['Player'] == 'Justin Jefferson'].iloc[0]
    assert abs(float(jj['Proj Pts']) - (0.5 * 300.0 + 0.5 * target)) < 0.2
    # Nobody the book was silent about should move.
    assert float(blended[blended['Player'] == 'Somebody Else']['Proj Pts'].iloc[0]) == 200.0

    # Without scoring there is no way to build the hybrid, so it must no-op
    # rather than fall back to the partial number it used to use.
    same, moved = blend_market_into_projection(board, scored, weight=0.5)
    assert moved == 0 and same['Proj Pts'].tolist() == board['Proj Pts'].tolist()


def test_book_projection_needs_board_stat_columns():
    ud, _ = _underdog()
    scored = score_market_lines(market_stat_lines(combine_props(ud)), SCORING)
    bare = pd.DataFrame({'Player': ['Justin Jefferson'], 'Pos': ['WR'],
                         'Proj Pts': [300.0]})
    assert build_book_projection(bare, scored, SCORING).empty
    assert build_book_projection(pd.DataFrame(), scored, SCORING).empty
    assert build_book_projection(_board_with_stats(), pd.DataFrame(), SCORING).empty


def test_prizepicks_demon_and_goblin_are_not_scorable():
    """
    Demon and goblin are PrizePicks' ALTERED lines - a demon sits above the
    true median and pays more, a goblin below and pays less. Scoring one as
    though it were the book's honest middle would bias a player by design.
    """
    payload = {
        'data': [
            {'type': 'projection', 'id': '1',
             'attributes': {'stat_type': 'Receiving Yards', 'line_score': 60.5,
                            'odds_type': 'standard'},
             'relationships': {'new_player': {'data': {'type': 'new_player', 'id': 'p1'}}}},
            {'type': 'projection', 'id': '2',
             'attributes': {'stat_type': 'Receiving Yards', 'line_score': 85.5,
                            'odds_type': 'demon'},
             'relationships': {'new_player': {'data': {'type': 'new_player', 'id': 'p1'}}}},
            {'type': 'projection', 'id': '3',
             'attributes': {'stat_type': 'Receiving Yards', 'line_score': 40.5,
                            'odds_type': 'goblin'},
             'relationships': {'new_player': {'data': {'type': 'new_player', 'id': 'p1'}}}},
        ],
        'included': [{'type': 'new_player', 'id': 'p1',
                      'attributes': {'display_name': '', 'name': 'Justin Jefferson',
                                     'team': 'MIN', 'position': 'WR'}}],
    }
    props, err = parse_prizepicks_payload(payload)
    assert err is None and len(props) == 3
    # display_name is an empty STRING on real payloads, not absent.
    assert set(props['player']) == {'Justin Jefferson'}
    scorable = props[props['scorable']]
    assert len(scorable) == 1 and float(scorable['line'].iloc[0]) == 60.5
    assert set(props['provider']) == {'PrizePicks', 'PrizePicks (demon)',
                                      'PrizePicks (goblin)'}


def test_prizepicks_filters_combined_player_props():
    """"Tristan Jarry + Cam Talbot" can never match a board row."""
    payload = {
        'data': [{'type': 'projection', 'id': '1',
                  'attributes': {'stat_type': 'Receiving Yards', 'line_score': 60.5,
                                 'odds_type': 'standard'},
                  'relationships': {'new_player': {'data': {'type': 'new_player',
                                                            'id': 'combo'}}}}],
        'included': [{'type': 'new_player', 'id': 'combo',
                      'attributes': {'name': 'Player One + Player Two', 'combo': True,
                                     'team': 'MIN', 'position': 'WR'}}],
    }
    props, _ = parse_prizepicks_payload(payload)
    assert props.empty


def test_prizepicks_per_game_lines_are_not_read_as_season():
    """
    THE BUG THIS LOCKS IN. Nothing in a PrizePicks payload says "season" -
    odds_type is standard/demon/goblin, which describes how a line is SHADED.
    Their real NFL board is week-one game props (Pass Yards median 229), and
    reading those as season-long would put a 52-yard receiving projection on
    a draft board.
    """
    payload = {
        'data': [{'type': 'projection', 'id': '1',
                  'attributes': {'stat_type': 'Pass Yards', 'line_score': 229.5,
                                 'odds_type': 'standard', 'description': 'SF'},
                  'relationships': {'new_player': {'data': {'type': 'new_player', 'id': 'p1'}},
                                    'duration': {'data': {'type': 'duration', 'id': 'd1'}}}}],
        'included': [{'type': 'new_player', 'id': 'p1',
                      'attributes': {'name': 'Brock Purdy', 'team': 'SF', 'position': 'QB'}},
                     {'type': 'duration', 'id': 'd1', 'attributes': {'name': 'Full'}}],
    }
    props, _ = parse_prizepicks_payload(payload)
    assert props['period'].tolist() == ['game']
    # A genuinely season-long label must still be recognised.
    payload['data'][0]['attributes']['stat_type'] = 'Season Pass Yards'
    props, _ = parse_prizepicks_payload(payload)
    assert props['period'].tolist() == ['season']


def test_two_books_average_per_stat_not_per_projection():
    """
    Multi-book averaging happens at the STAT level. If one book prices yards
    and the other prices receptions, the player keeps BOTH - more of his line
    comes from the market instead of from us. Averaging finished projections
    would have thrown that away.
    """
    ud, _ = _underdog()
    pp = pd.DataFrame([
        # Same stat as Underdog -> should average to the midpoint.
        {'provider': 'PrizePicks', 'player': 'Justin Jefferson',
         'player_key': 'justinjefferson', 'team': 'MIN', 'position': 'WR',
         'market': 'receiving_yards', 'market_raw': 'Season Receiving Yards',
         'scorable': True, 'line': 1375.5, 'over_payout': None, 'under_payout': None,
         'period': 'season', 'source_id': 'pp1'},
        # A stat Underdog never priced for the QB -> must be ADDED, not lost.
        {'provider': 'PrizePicks', 'player': 'Patrick Mahomes',
         'player_key': 'patrickmahomes', 'team': 'SEA', 'position': 'QB',
         'market': 'rushing_yards', 'market_raw': 'Season Rush Yards',
         'scorable': True, 'line': 300.5, 'over_payout': None, 'under_payout': None,
         'period': 'season', 'source_id': 'pp2'},
    ])
    rows = market_stat_lines(combine_props(ud, pp), season_only=True)

    jj = rows[rows['player_key'] == 'justinjefferson'].iloc[0]
    assert float(jj['receiving_yards']) == (1275.5 + 1375.5) / 2, "two books -> midpoint"
    assert int(jj['Books']) == 2
    assert 'Underdog' in jj['providers'] and 'PrizePicks' in jj['providers']

    qb = rows[rows['player_key'] == 'patrickmahomes'].iloc[0]
    assert float(qb['rushing_yards']) == 300.5, "a stat only one book priced survives"
    assert float(qb['passing_yards']) == 4225.5, "and the other book's stats are kept"
    assert int(qb['Books']) == 2

    # A player only one book priced uses that book, unchanged.
    rb = rows[rows['player_key'] == 'travisetienne']
    if not rb.empty:
        assert int(rb.iloc[0]['Books']) == 1


def test_nflszn_league_makes_plain_stat_labels_read_as_season():
    """
    THE TRAP. PrizePicks runs season-long as a SEPARATE LEAGUE (NFLSZN), not
    a flag on a projection. So inside an NFLSZN payload the stat labels read
    perfectly ordinary - "Receiving Yards", not "Season Receiving Yards" -
    and every label-based heuristic classifies the whole board as per-game,
    silently. The league name has to be what decides it.
    """
    def payload(league_name):
        return {
            'data': [{'type': 'projection', 'id': '1',
                      'attributes': {'stat_type': 'Receiving Yards',
                                     'line_score': 1240.5, 'odds_type': 'standard'},
                      'relationships': {
                          'new_player': {'data': {'type': 'new_player', 'id': 'p1'}},
                          'league': {'data': {'type': 'league', 'id': 'L'}}}}],
            'included': [
                {'type': 'new_player', 'id': 'p1',
                 'attributes': {'name': 'Justin Jefferson', 'team': 'MIN',
                                'position': 'WR', 'league': league_name}},
                {'type': 'league', 'id': 'L', 'attributes': {'name': league_name}},
            ],
        }

    season, _ = parse_prizepicks_payload(payload('NFLSZN'))
    assert season['period'].tolist() == ['season'], "NFLSZN must read as season-long"
    assert float(season['line'].iloc[0]) == 1240.5

    weekly, _ = parse_prizepicks_payload(payload('NFL'))
    assert weekly['period'].tolist() == ['game'], "the weekly board must stay per-game"


def test_league_discovery_and_implausible_period_warning():
    leagues_payload = {'data': [
        {'type': 'league', 'id': '9', 'attributes': {'name': 'NFL'}},
        {'type': 'league', 'id': '241', 'attributes': {'name': 'NFLSZN'}},
        {'type': 'league', 'id': '7', 'attributes': {'name': 'NBA'}},
    ]}
    found = prizepicks_leagues(leagues_payload)
    assert found['241'] == 'NFLSZN' and found['9'] == 'NFL'

    # The magnitude cross-check: a 1,240-yard "game" line is impossible, and
    # period detection has been wrong twice, so it gets surfaced.
    props = pd.DataFrame({
        'player': ['A', 'B'], 'market': ['receiving_yards', 'receiving_yards'],
        'line': [1240.5, 62.5], 'period': ['game', 'game'], 'market_raw': ['x', 'y'],
    })
    odd = implausible_period_rows(props)
    assert len(odd) == 1 and odd['player'].iloc[0] == 'A'


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failures.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
