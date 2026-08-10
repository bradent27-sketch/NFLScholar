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
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

from data.odds_sources import (  # noqa: E402
    parse_underdog_payload, parse_prizepicks_payload, parse_odds_api_props,
    standardize_team, normalize_stat, combine_props, load_props_fixture,
    odds_api_bookmakers, PROP_COLUMNS, prizepicks_leagues,
    implausible_period_rows, parse_fanduel_payload, parse_pinnacle_payload,
    devig_two_way, american_to_decimal, parse_draftkings_payload,
    parse_draftkings_payloads,
)
from data.odds_projections import (  # noqa: E402
    market_stat_lines, score_market_lines, compare_to_board,
    blend_market_into_projection, attach_board_player, build_book_projection,
    attach_book_projection, canonicalize_props, resolve_names_to_board,
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


def test_two_books_spelling_one_player_differently_become_one_row():
    """
    THE BUG. PrizePicks writes "James Cook III", Underdog writes "James
    Cook". Left alone that is two market rows, so the books never average
    against each other, Books reads 1 for a player both priced, AND the loose
    board match refuses both because two unresolved rows share one stripped
    key. Kyle Pitts and Omar Cooper failed identically.
    """
    board = pd.DataFrame({
        'Player': ['James Cook III', 'Byron Murphy', 'Byron Murphy II'],
        'Pos': ['RB', 'WR', 'DT'], 'Proj Pts': [250.0, 100.0, 10.0],
    })
    props = pd.DataFrame([
        {'provider': 'Underdog', 'player': 'James Cook', 'player_key': 'jamescook',
         'team': 'BUF', 'position': 'RB', 'market': 'rushing_yards',
         'market_raw': 'season_rush_yards', 'scorable': True, 'line': 1000.5,
         'over_payout': None, 'under_payout': None, 'period': 'season', 'source_id': 'u1'},
        {'provider': 'PrizePicks', 'player': 'James Cook III', 'player_key': 'jamescookiii',
         'team': 'BUF', 'position': 'RB', 'market': 'rushing_yards',
         'market_raw': 'Rush Yards', 'scorable': True, 'line': 1100.5,
         'over_payout': None, 'under_payout': None, 'period': 'season', 'source_id': 'p1'},
    ])
    rows = market_stat_lines(props, season_only=True, board=board)
    assert len(rows) == 1, "the two spellings must collapse to one player"
    row = rows.iloc[0]
    assert row['player'] == 'James Cook III', "canonicalised to the board's spelling"
    assert int(row['Books']) == 2
    assert float(row['rushing_yards']) == (1000.5 + 1100.5) / 2, "and they average"


def test_canonicalisation_still_refuses_two_genuinely_different_players():
    """The guard that must survive: "Byron Murphy" and "Byron Murphy II" are
    two real people and neither may absorb the other."""
    board = pd.DataFrame({
        'Player': ['Byron Murphy', 'Byron Murphy II'],
        'Pos': ['CB', 'DT'], 'Proj Pts': [10.0, 8.0],
    })
    # Neither book name matches a board row exactly, and both strip to the
    # same key - so both must stay unresolved.
    names = pd.Series(['Byron Murphy Jr.', 'Byron Murphy Sr.'])
    resolved = resolve_names_to_board(names, board)
    assert resolved.isna().all(), resolved.tolist()

    # An exact match is never disturbed by a suffixed sibling.
    exact = resolve_names_to_board(pd.Series(['Byron Murphy II']), board)
    assert exact.tolist() == ['Byron Murphy II']


def test_canonicalize_keeps_names_the_board_does_not_know():
    board = pd.DataFrame({'Player': ['James Cook III'], 'Pos': ['RB'], 'Proj Pts': [250.0]})
    props = pd.DataFrame([
        {'provider': 'PrizePicks', 'player': 'Jack Fox', 'player_key': 'jackfox',
         'team': 'DET', 'position': 'P', 'market': 'punts_inside_20',
         'market_raw': 'Punts Inside 20', 'scorable': False, 'line': 24.5,
         'over_payout': None, 'under_payout': None, 'period': 'season', 'source_id': 'x'},
    ])
    out = canonicalize_props(props, board)
    assert out['player'].tolist() == ['Jack Fox'], "a punter we don't rank stays visible"


def test_int_is_defensive_unless_the_player_is_a_quarterback():
    """
    On PrizePicks' season board "INT" is DEFENSIVE interceptions - the lines
    belong to corners, safeties and linebackers - while a quarterback's
    thrown picks are a separate "Pass INTs" market. Mapping the bare label to
    passing_interceptions gave a cornerback a -3 point penalty for throwing
    them, and dropped the real QB stat entirely.
    """
    def payload(stat, position):
        return {
            'data': [{'type': 'projection', 'id': '1',
                      'attributes': {'stat_type': stat, 'line_score': 1.5,
                                     'odds_type': 'standard'},
                      'relationships': {'new_player': {'data': {'type': 'new_player',
                                                                'id': 'p'}}}}],
            'included': [{'type': 'new_player', 'id': 'p',
                          'attributes': {'name': 'Somebody', 'team': 'MIN',
                                         'position': position}}],
        }
    db, _ = parse_prizepicks_payload(payload('INT', 'CB'))
    assert bool(db['scorable'].iloc[0]) is False, "a corner's picks are not a QB stat"

    qb, _ = parse_prizepicks_payload(payload('INT', 'QB'))
    assert qb['market'].iloc[0] == 'passing_interceptions'
    assert bool(qb['scorable'].iloc[0]) is True

    thrown, _ = parse_prizepicks_payload(payload('Pass INTs', 'QB'))
    assert thrown['market'].iloc[0] == 'passing_interceptions'
    assert bool(thrown['scorable'].iloc[0]) is True


def _fixture(name):
    with open(os.path.join(FIXTURES, name), encoding='utf-8') as handle:
        return json.load(handle)


# --- FanDuel ---------------------------------------------------------------

def test_fanduel_reads_season_player_props():
    props, err = parse_fanduel_payload(_fixture('fanduel_nfl_page.json'))
    assert err is None, err
    assert list(props.columns) == PROP_COLUMNS
    assert set(props['provider']) == {'FanDuel'}
    assert set(props['period']) == {'season'}
    assert bool(props['scorable'].all()), "every FanDuel season stat should map"
    assert set(props['market']) == {'passing_yards', 'passing_tds', 'rushing_yards',
                                    'rushing_tds', 'receiving_yards'}


def test_fanduel_line_comes_from_the_runner_name_not_the_handicap_field():
    # Every one of these markets carries handicap: 0.0 alongside a real line
    # in the runner name. Reading the field would give a board of zeroes that
    # still looked structurally fine.
    props, _ = parse_fanduel_payload(_fixture('fanduel_nfl_page.json'))
    rodgers = props[props['player'] == 'Aaron Rodgers'].iloc[0]
    assert rodgers['market'] == 'passing_yards'
    assert rodgers['line'] == 3050.5
    assert bool((props['line'] > 0).all())


def test_fanduel_rejects_award_and_leader_markets_that_match_the_name_pattern():
    # "AP NFL Regular Season MVP 2026-27" -> player "AP NFL", stat "MVP".
    # "Most Regular Season Rookie Receiving Yards 2026-27" -> player "Most".
    # Both parse cleanly as names and are winner markets, not over/unders.
    props, _ = parse_fanduel_payload(_fixture('fanduel_nfl_page.json'))
    assert 'AP NFL' not in set(props['player'])
    assert 'Most' not in set(props['player'])
    assert 'MVP' not in set(props['market_raw'])


def test_fanduel_ignores_game_markets():
    props, _ = parse_fanduel_payload(_fixture('fanduel_nfl_page.json'))
    assert 'Moneyline' not in set(props['market_raw'])
    assert (props['period'] == 'season').all()


def test_fanduel_devigs_asymmetric_prices():
    props, _ = parse_fanduel_payload(_fixture('fanduel_nfl_page.json'))
    even = props[props['over_payout'] == props['under_payout']]
    assert not even.empty
    assert all(abs(p - 0.5) < 1e-9 for p in even['p_over']), \
        "equal prices both sides means the posted line IS the median"

    skewed = props[props['over_payout'] != props['under_payout']]
    assert not skewed.empty, "fixture should include an unbalanced market"
    assert all(0.0 < p < 1.0 for p in skewed['p_over'])
    assert any(abs(p - 0.5) > 0.01 for p in skewed['p_over']), \
        "unbalanced prices should move the true median off the posted line"


def test_devig_removes_the_margin():
    # -110 both sides implies 0.5238 each, summing to 1.0476. Divided out,
    # the over is exactly even money.
    assert abs(devig_two_way(-110, -110) - 0.5) < 1e-12
    assert devig_two_way(-148, 112) > 0.5, "the shorter price is the likelier side"
    assert devig_two_way(-110, None) is None, "no price, no probability - not a guess"
    assert abs(american_to_decimal(100) - 2.0) < 1e-12
    assert abs(american_to_decimal(-200) - 1.5) < 1e-12


# --- Pinnacle --------------------------------------------------------------

def test_pinnacle_reads_season_player_props():
    props, err = parse_pinnacle_payload(_fixture('pinnacle_nfl_matchups.json'))
    assert err is None, err
    assert list(props.columns) == PROP_COLUMNS
    assert set(props['provider']) == {'Pinnacle'}
    assert set(props['period']) == {'season'}
    assert bool(props['scorable'].all())
    assert set(props['market']) == {'receiving_yards', 'rushing_yards', 'passing_yards'}
    flowers = props[props['player'] == 'Zay Flowers'].iloc[0]
    assert flowers['line'] == 974.5, "line is inside the participant name"


def test_pinnacle_ignores_team_specials_and_game_matchups():
    # The same feed carries "Pittsburgh Steelers Total Regular Season Wins",
    # "New York Giants To Make the Playoffs" and ordinary game matchups.
    props, _ = parse_pinnacle_payload(_fixture('pinnacle_nfl_matchups.json'))
    joined = ' '.join(props['player']) + ' ' + ' '.join(props['market_raw'])
    assert 'Steelers' not in joined
    assert 'Playoffs' not in joined
    assert len(props) == 3


def test_pinnacle_publishes_no_prices_so_claims_no_probability():
    # The matchup feed carries lines but not odds; prices need a second call.
    # A placeholder here would read downstream as a real even-money quote.
    props, _ = parse_pinnacle_payload(_fixture('pinnacle_nfl_matchups.json'))
    assert props['p_over'].isna().all()
    assert props['over_payout'].isna().all()


# --- DraftKings ------------------------------------------------------------

def test_draftkings_reads_season_player_props():
    props, err = parse_draftkings_payloads(_fixture('draftkings_player_futures.json'))
    assert err is None, err
    assert list(props.columns) == PROP_COLUMNS
    assert set(props['provider']) == {'DraftKings'}
    assert set(props['period']) == {'season'}
    assert set(props['market']) == {'receptions', 'receiving_yards'}
    assert bool(props['scorable'].all())


def test_draftkings_prices_use_a_unicode_minus():
    # displayOdds.american is written "−115" with U+2212 MINUS SIGN, which
    # float() rejects outright - this does not degrade a parser, it kills it.
    raw = json.dumps(_fixture('draftkings_player_futures.json'), ensure_ascii=False)
    assert '−' in raw, "fixture must keep the real character"
    assert american_to_decimal('−110') is not None
    assert abs(american_to_decimal('−100') - 2.0) < 1e-12
    assert abs(devig_two_way('−110', '−110') - 0.5) < 1e-12

    props, _ = parse_draftkings_payloads(_fixture('draftkings_player_futures.json'))
    assert props['over_payout'].notna().all(), "every price should have parsed"
    assert props['p_over'].notna().all()
    assert all(0.3 < p < 0.7 for p in props['p_over'])


def test_draftkings_player_comes_from_the_event_not_the_market_name():
    """
    THE BUG THIS LOCKS IN. Splitting the market name on " - " looks like the
    obvious way to get the player and silently drops three of 319: the
    separator is an ASCII hyphen on most rows, an EN DASH on Travis Kelce's,
    and a double-spaced hyphen on Romeo Doubs'. The event's participants have
    no such problem.
    """
    payloads = _fixture('draftkings_player_futures.json')
    names = [m['name'] for p in payloads for m in p['markets']]
    assert any('–' in n for n in names), "fixture must keep the en-dash market"
    assert any('  -  ' in n for n in names), "fixture must keep the double-spaced market"

    props, _ = parse_draftkings_payloads(payloads)
    assert 'Travis Kelce' in set(props['player'])
    assert 'Romeo Doubs' in set(props['player'])
    # And no market-name debris leaked into a player name.
    assert not any('NFL 20' in p or 'Regular Season' in p for p in props['player'])


def test_draftkings_team_comes_from_the_flagged_participant():
    # Both participants are typed "Team" and the order varies; only one
    # carries metadata.rosettaTeamName, and that one is the club.
    props, _ = parse_draftkings_payloads(_fixture('draftkings_player_futures.json'))
    assert (props['team'] != '').all(), "every row should resolve a team"
    # Read off the payload rather than from memory of who plays where - the
    # book knows about the offseason and a hardcoded roster does not.
    assert dict(zip(props['player'], props['team'])) == {
        'Mike Evans': 'SF', 'Travis Kelce': 'KC',
        'Romeo Doubs': 'NE', 'Amon-Ra St. Brown': 'DET',
    }
    # Two rows for Kelce, one team - the join is per event, not per market.
    assert (props['player'] == 'Travis Kelce').sum() == 2


def test_draftkings_stat_comes_from_market_type():
    # The market name says "Receiving TDs" where marketType says "Receiving
    # Touchdowns" for the same stat; marketType is uniform, the name is not.
    props, _ = parse_draftkings_payloads(_fixture('draftkings_player_futures.json'))
    assert set(props['market_raw']) == {'Receptions', 'Receiving Yards'}


def test_draftkings_accepts_one_payload_or_many():
    payloads = _fixture('draftkings_player_futures.json')
    many, _ = parse_draftkings_payloads(payloads)
    one, _ = parse_draftkings_payloads(payloads[0])
    single, _ = parse_draftkings_payload(payloads[0])
    assert len(one) == len(single) < len(many)
    assert len(many) == sum(len(parse_draftkings_payload(p)[0]) for p in payloads)


def test_draftkings_line_is_in_the_label_not_a_points_field():
    # Futures selections carry no `points` key at all, unlike the game-lines
    # feed from the same API where it is populated.
    payloads = _fixture('draftkings_player_futures.json')
    assert not any('points' in s for p in payloads for s in p['selections'])
    props, _ = parse_draftkings_payloads(payloads)
    assert bool((props['line'] > 0).all())
    kelce = props[(props['player'] == 'Travis Kelce') & (props['market'] == 'receptions')]
    assert len(kelce) == 1 and kelce['line'].iloc[0] > 10


def test_new_books_combine_with_the_existing_ones():
    fd, _ = parse_fanduel_payload(_fixture('fanduel_nfl_page.json'))
    pin, _ = parse_pinnacle_payload(_fixture('pinnacle_nfl_matchups.json'))
    ud, _ = parse_underdog_payload(_fixture('underdog_over_under_lines.json'))
    both = combine_props(fd, pin, ud)
    assert list(both.columns) == PROP_COLUMNS
    assert {'FanDuel', 'Pinnacle'} <= set(both['provider'])
    assert len(both) == len(fd) + len(pin) + len(ud)


# --- weekly ----------------------------------------------------------------

def _dk_game_payload(market_name, label_prefix=''):
    """A DraftKings weekly market: a GAME event, so two club participants."""
    return {
        'events': [{'id': 'ev1', 'name': 'KC Chiefs @ BUF Bills', 'participants': [
            {'name': 'KC Chiefs', 'type': 'Team',
             'metadata': {'rosettaTeamName': 'Chiefs', 'shortName': 'KC'}},
            {'name': 'BUF Bills', 'type': 'Team',
             'metadata': {'rosettaTeamName': 'Bills', 'shortName': 'BUF'}},
        ]}],
        'markets': [{'id': 'm1', 'eventId': 'ev1', 'name': market_name,
                     'subcategoryId': 999,
                     'marketType': {'name': 'Passing Yards OU'}}],
        'selections': [
            {'id': 's1', 'marketId': 'm1', 'label': f'{label_prefix}Over 275.5',
             'trueOdds': 1.91, 'outcomeType': 'Over'},
            {'id': 's2', 'marketId': 'm1', 'label': f'{label_prefix}Under 275.5',
             'trueOdds': 1.91, 'outcomeType': 'Under'},
        ],
    }


def test_draftkings_classifies_game_vs_season_from_event_shape():
    """
    A season market hangs off a PLAYER event (one club participant plus the
    man); a weekly market hangs off a GAME event (two clubs). Counting the
    team-flagged participants classifies it without reading any wording,
    which is the property worth having given how often a label has moved.
    """
    weekly, err = parse_draftkings_payload(_dk_game_payload('Patrick Mahomes Passing Yards'))
    assert err is None, err
    assert list(weekly['period']) == ['game']
    assert weekly['player'].iloc[0] == 'Patrick Mahomes'
    assert weekly['market'].iloc[0] == 'passing_yards'
    assert weekly['line'].iloc[0] == 275.5

    season, _ = parse_draftkings_payloads(_fixture('draftkings_player_futures.json'))
    assert set(season['period']) == {'season'}


def test_draftkings_weekly_player_can_come_from_the_selection_label():
    # The other shape books use: one market for the game, the man named on
    # each selection. Neither shape is confirmed live, so both are handled.
    props, err = parse_draftkings_payload(
        _dk_game_payload('Passing Yards', label_prefix='Patrick Mahomes '))
    assert err is None, err
    assert props['player'].iloc[0] == 'Patrick Mahomes'
    assert props['period'].iloc[0] == 'game'


def test_draftkings_drops_a_game_market_with_no_recoverable_player():
    # Better a missing row than one filed under a wrong name.
    props, err = parse_draftkings_payload(_dk_game_payload('Passing Yards'))
    assert props.empty and err


def test_weekly_posting_anchor_walks_back_to_tuesday():
    import datetime as dt
    from data.odds_weekly import posting_anchor, is_stale, POST_HOUR_UTC

    def at(text):
        return dt.datetime.fromisoformat(text).replace(tzinfo=dt.timezone.utc)

    # Before Tuesday's posting hour, the current slate is still last week's.
    assert posting_anchor(at('2026-08-11 14:00')) == at('2026-08-04 15:00')
    # After it, the new slate is up.
    assert posting_anchor(at('2026-08-11 16:00')) == at('2026-08-11 15:00')
    # And it stays the anchor all the way to the following Tuesday.
    for when in ('2026-08-14 09:00', '2026-08-16 23:00', '2026-08-17 09:00'):
        assert posting_anchor(at(when)) == at('2026-08-11 15:00'), when
    assert posting_anchor(at('2026-08-11 16:00')).hour == POST_HOUR_UTC

    # Staleness is about which slate, not about elapsed hours: a snapshot
    # taken Friday is current on Sunday and stale on Wednesday.
    friday = at('2026-08-14 09:00')
    assert is_stale(friday, at('2026-08-16 12:00')) is False
    assert is_stale(friday, at('2026-08-18 16:00')) is True
    assert is_stale(None) is True


def test_weekly_snapshot_round_trips_and_survives_a_bad_file():
    import tempfile
    from data.odds_weekly import save_snapshot, load_snapshot, weekly_summary, weekly_consensus

    props, _ = parse_underdog_payload(_fixture('underdog_over_under_lines.json'))
    game = props[props['period'] == 'game']
    assert not game.empty, "fixture should carry single-game lines"

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'snap.json')
        assert save_snapshot(game, {'Underdog': {'rows': len(game)}}, path=path) is None
        back, when, status = load_snapshot(path)
        assert len(back) == len(game)
        assert when is not None and status['Underdog']['rows'] == len(game)
        assert list(back['line']) == list(game['line'])

        # A missing or corrupt file reads as "no snapshot" so the next call
        # refetches, rather than raising on a tab that is drawing odds.
        empty, when2, _ = load_snapshot(os.path.join(tmp, 'nope.json'))
        assert empty.empty and when2 is None
        with open(path, 'w') as handle:
            handle.write('{not json')
        empty2, _, _ = load_snapshot(path)
        assert empty2.empty

    summary = weekly_summary(game)
    assert set(summary.columns) == {'Book', 'Lines', 'Players', 'Markets'}
    consensus = weekly_consensus(game)
    assert list(consensus.columns[:6]) == ['Player', 'Team', 'Market',
                                           'Consensus', 'Books', 'Spread']
    assert (consensus['Books'] >= 1).all()


def test_weekly_consensus_repeats_a_player_across_stats():
    """
    One row per player PER STAT, so player names repeat. This is the thing
    that broke the tab: indexing that frame by Player gives a non-unique
    index, and pandas' Styler refuses to apply against one - which surfaced
    as the whole Live Odds tab failing to render rather than as a bad table.
    """
    from data.odds_weekly import weekly_consensus
    # Built rather than taken from the fixture: the recorded payload happens
    # to give each player one stat, so it cannot show the case that broke.
    rows = pd.DataFrame([
        {'provider': 'Underdog', 'player': 'A Back', 'player_key': 'aback',
         'team': 'KC', 'position': 'RB', 'market': market, 'market_raw': market,
         'scorable': True, 'line': line, 'over_payout': None, 'under_payout': None,
         'p_over': None, 'period': 'game', 'source_id': market}
        for market, line in (('rushing_yards', 62.5), ('receiving_yards', 21.5))
    ])
    out = weekly_consensus(rows)
    assert len(out) == 2
    assert out['Player'].duplicated().any(), \
        "a player with two stats must appear twice - do not index by Player"
    assert out.index.is_unique


def test_weekly_prefers_a_saved_payload_over_the_network():
    """
    A saved file wins, and its own separate slot matters: PrizePicks' weekly
    board is a different league from its season board, so one file must not
    be able to overwrite the other.
    """
    from data.odds_sources import SAVED_PAYLOADS
    from data.odds_weekly import WEEKLY_SAVED_SOURCES

    season_path = SAVED_PAYLOADS['PrizePicks'][0]
    weekly_path = SAVED_PAYLOADS['PrizePicks Weekly'][0]
    assert season_path != weekly_path, \
        "weekly must not overwrite the season payload Book Proj is built from"

    # Underdog deliberately has no separate weekly file: one endpoint returns
    # both periods, so the season file already carries the weekly board.
    assert WEEKLY_SAVED_SOURCES['Underdog'][0] == 'Underdog'
    assert WEEKLY_SAVED_SOURCES['PrizePicks'][0] == 'PrizePicks Weekly'
    assert WEEKLY_SAVED_SOURCES['DraftKings'][0] == 'DraftKings Weekly'


def test_weekly_consensus_medians_across_books_and_reports_the_spread():
    from data.odds_weekly import weekly_consensus
    rows = pd.DataFrame([
        {'provider': b, 'player': 'A Back', 'player_key': 'aback', 'team': 'KC',
         'position': 'RB', 'market': 'rushing_yards', 'market_raw': 'Rush Yards',
         'scorable': True, 'line': line, 'over_payout': None, 'under_payout': None,
         'p_over': None, 'period': 'game', 'source_id': str(i)}
        for i, (b, line) in enumerate([('Underdog', 60.5), ('PrizePicks', 64.5),
                                       ('DraftKings', 62.5)])
    ])
    out = weekly_consensus(rows)
    assert len(out) == 1
    assert out['Consensus'].iloc[0] == 62.5, "median of the three, not the mean"
    assert out['Books'].iloc[0] == 3
    assert out['Spread'].iloc[0] == 4.0


def test_browser_fallback_is_switchable_and_reports_itself():
    """
    The fallback must be visible in both directions: off means a 403 is
    still the final answer, and unavailable must say so rather than looking
    like the book was down.
    """
    import data.odds_sources as src

    original = os.environ.get('NFLSCHOLAR_BROWSER_FALLBACK')
    try:
        os.environ['NFLSCHOLAR_BROWSER_FALLBACK'] = '0'
        assert src.browser_fallback_enabled() is False
        payload, err = src._browser_retry('https://example.invalid/x')
        assert payload is None and 'switched off' in err

        os.environ['NFLSCHOLAR_BROWSER_FALLBACK'] = '1'
        assert src.browser_fallback_enabled() is True
    finally:
        if original is None:
            os.environ.pop('NFLSCHOLAR_BROWSER_FALLBACK', None)
        else:
            os.environ['NFLSCHOLAR_BROWSER_FALLBACK'] = original

    # Nested shared_browser is a no-op rather than a second browser.
    with src.shared_browser():
        with src.shared_browser():
            pass


def test_browser_module_makes_no_evasion_claims_it_breaks():
    """
    data/odds_browser lists things it does not do. This checks the one that
    was actually violated once: the launch args must not carry the flag that
    turns off navigator.webdriver, because the docstring says they don't.
    """
    import inspect
    import data.odds_browser as browser

    source = inspect.getsource(browser)
    assert 'no navigator.webdriver patching' in source

    # The flag is named in the docstring, which explains why it is absent -
    # so check the CODE, not the prose. Everything after the docstring is
    # what actually runs.
    body = inspect.getsource(browser._launch)
    body = body.split('"""')[-1]
    assert 'AutomationControlled' not in body, \
        "the module claims not to patch navigator.webdriver - keep that true"

    for banned in ('undetected_chromedriver', 'playwright_stealth', 'stealth('):
        assert banned not in source, banned


def test_bad_payload_shapes_are_reported_not_raised():
    for bad in ({}, {'attachments': {}}, [], 'nonsense', None):
        props, err = parse_fanduel_payload(bad)
        assert props.empty and err, f"FanDuel accepted {bad!r}"
    for bad in ({}, [], 'nonsense', None, [{'type': 'matchup'}]):
        props, err = parse_pinnacle_payload(bad)
        assert props.empty and err, f"Pinnacle accepted {bad!r}"


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
