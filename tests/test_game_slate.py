"""
Offline tests for the Game Slate tab.

THESE ASSERT ON RENDERED HTML, which is unusual here and deliberate: this
card's requirements are visual, and the important ones fail SILENTLY. A card
printing `nan` where a score goes still renders, still raises nothing, and
still looks like a working tab. So does one that dims both teams on a tie,
or one whose style block was broken by a malformed colour.

Fixtures build scores as float/NaN, which is what pandas actually hands the
renderer - an int column holding one missing value is promoted to float64,
and the None written by the loader becomes NaN on the way.

Runs two ways: `python tests/test_game_slate.py` needs nothing but the app's
own dependencies, and `pytest tests/` works if pytest is installed.
"""
import datetime
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

from data.game_slate import (  # noqa: E402
    SLATE_COLUMNS, _hex_color, format_kickoff, slate_team_bridge, team_color,
    team_logo, default_week_index, ROUND_LABELS,
)
from ui.tabs.game_slate import card_html  # noqa: E402


def _row(**over):
    base = {
        'Away': 'KC', 'Home': 'BUF',
        'Away Pts': 27.0, 'Home Pts': 24.0, 'Winner': 'KC',
        'Date': '2026-09-13', 'Date Display': '09/13/2026',
        'Kickoff': 'Sun 1:00 PM EDT', 'Kickoff Long': 'Sunday at 1:00 PM EDT',
        'Time TBD': False, 'Status': 'post', 'Status Detail': 'Final',
        'Played': True, 'Live': False,
        'Away Color': '#e31837', 'Home Color': '#00338d',
        'Away Logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/kc.png',
        'Home Logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/buf.png',
        'Venue': 'Highmark Stadium', 'Neutral Site': False,
        'Headline': None, 'Matchup': 'KC @ BUF',
        'Week': 2, 'Season Type': 'REG', '_source': 'nflverse schedule',
    }
    base.update(over)
    return pd.Series(base)


def _unplayed(**over):
    # Score 0, not null, is the harder fixture on purpose - the sibling apps'
    # source sends "0" before kickoff, and gating on presence would render it.
    # Here the gate is Played, so 0 must still produce no score node.
    defaults = {'Away Pts': 0.0, 'Home Pts': 0.0, 'Winner': None,
                'Played': False, 'Status': 'pre', 'Status Detail': 'Scheduled'}
    defaults.update(over)
    return _row(**defaults)


# --- played games ----------------------------------------------------------

def test_played_game_renders_both_scores_and_marks_one_winner():
    out = card_html(0, _row())
    assert '>27<' in out and '>24<' in out
    assert out.count('gs-win-flag') == 1, "exactly one W flag"
    assert 'gs-won' in out and 'gs-lost' in out
    assert out.count('gs-won') == 1 and out.count('gs-lost') == 1


def test_played_game_carries_each_team_colour_and_binds_both_to_the_card():
    out = card_html(3, _row())
    assert "--gs-color:#e31837" in out
    assert "--gs-color:#00338d" in out
    assert ".st-key-gs_card_3{--gs-a:#e31837;--gs-b:#00338d;}" in out


def test_date_and_kickoff_render_as_specified():
    out = card_html(0, _row())
    assert '09/13/2026' in out
    assert 'Sunday at 1:00 PM EDT' in out, "long form names the weekday"


# --- unplayed, postponed, live, tie ---------------------------------------

def test_unplayed_game_emits_no_score_node_and_never_the_string_nan():
    out = card_html(0, _unplayed())
    assert 'gs-score' not in out, "no score node at all - not '--', not 'NA'"
    assert 'nan' not in out.lower().replace('gs-name', '')
    assert 'gs-won' not in out and 'gs-lost' not in out
    assert 'gs-win-flag' not in out


def test_unplayed_game_with_real_nan_scores_is_also_clean():
    out = card_html(0, _unplayed(**{'Away Pts': float('nan'), 'Home Pts': float('nan')}))
    assert 'gs-score' not in out
    assert 'nan' not in out.lower()


def test_postponed_game_shows_the_status_where_the_kickoff_would_go():
    out = card_html(0, _unplayed(**{'Status Detail': 'Postponed'}))
    assert 'Postponed' in out
    assert 'Sunday at 1:00 PM EDT' not in out, "don't advertise a start it won't have"
    assert 'gs-score' not in out


def test_live_game_shows_a_running_score_and_no_winner():
    out = card_html(0, _row(**{'Played': False, 'Live': True, 'Winner': None,
                               'Status': 'in', 'Status Detail': '4th 2:11'}))
    assert '>27<' in out and '>24<' in out
    assert 'gs-win-flag' not in out
    assert 'gs-won' not in out and 'gs-lost' not in out


def test_a_tie_dims_neither_team():
    out = card_html(0, _row(**{'Away Pts': 20.0, 'Home Pts': 20.0, 'Winner': None}))
    assert out.count('>20<') == 2, "both scores render"
    assert 'gs-lost' not in out, "a tie has no loser - is_winner must stay None"
    assert 'gs-won' not in out


def test_winner_that_is_nan_is_treated_as_no_winner():
    # A NaN sails through `is not None`, which is exactly the trap.
    out = card_html(0, _row(**{'Winner': float('nan')}))
    assert 'gs-lost' not in out and 'gs-won' not in out


# --- markup safety ---------------------------------------------------------

def test_ampersands_in_names_and_venues_are_escaped():
    out = card_html(0, _row(**{'Venue': 'M&T Bank Stadium'}))
    assert 'M&amp;T Bank Stadium' in out
    assert 'M&T' not in out.replace('M&amp;T', '')


def test_a_missing_colour_falls_back_instead_of_emitting_a_broken_rule():
    out = card_html(1, _row(**{'Away Color': None}))
    assert '--gs-a' not in out, "absent, so the stylesheet's var() fallback applies"
    assert '--gs-b:#00338d;' in out
    out_none = card_html(1, _row(**{'Away Color': None, 'Home Color': None}))
    assert '<style>' not in out_none, "no colours means no style tag at all"


def test_malformed_colours_are_rejected_before_reaching_css():
    for bad in ('red', '#12345', 'ff0000; } body {', '', None, '#zzzzzz'):
        assert _hex_color(bad) is None, bad
    assert _hex_color('#E31837') == '#e31837'
    assert _hex_color('e31837') == '#e31837'


def test_a_missing_venue_is_omitted_not_printed_as_nan():
    for missing in (None, float('nan')):
        out = card_html(0, _row(**{'Venue': missing}))
        assert 'nan' not in out.lower()
        assert out.count('gs-dot') == 1, "only date · kickoff remain"


def test_neutral_site_is_labelled():
    out = card_html(0, _row(**{'Neutral Site': True, 'Venue': 'Wembley Stadium'}))
    assert 'Wembley Stadium (neutral)' in out


def test_playoff_headline_renders_and_regular_season_has_none():
    assert 'Super Bowl' in card_html(0, _row(**{'Headline': 'Super Bowl'}))
    assert 'gs-headline' not in card_html(0, _row())


# --- schema drift ----------------------------------------------------------

def test_dropping_optional_columns_does_not_crash_the_card():
    row = _row()
    for optional in ('Headline', 'Venue', 'Neutral Site', 'Broadcast', 'Live'):
        trimmed = row.drop(labels=[optional], errors='ignore')
        out = card_html(0, trimmed)
        assert 'gs-team' in out, optional


# --- logos -----------------------------------------------------------------

def test_logos_render_and_the_colour_stripe_survives_beside_them():
    out = card_html(0, _row())
    assert out.count('gs-logo') == 2
    assert '--gs-color:#e31837' in out, "stripe kept alongside the logo, deliberately"


def test_a_card_with_no_logos_renders_text_only():
    out = card_html(0, _row(**{'Away Logo': None, 'Home Logo': None}))
    assert 'gs-logo' not in out
    assert 'Kansas City Chiefs' in out


def test_one_missing_logo_drops_only_that_team():
    out = card_html(0, _row(**{'Away Logo': None}))
    assert out.count('gs-logo') == 1


def test_team_logo_uses_the_nfl_abbreviation_pattern_not_the_college_one():
    # Verified against a real URL seen in live data. The college leagues key
    # off a numeric id in an /ncaa/ directory; NFL does not.
    assert team_logo('KC') == 'https://a.espncdn.com/i/teamlogos/nfl/500/kc.png'
    assert team_logo('LA') == 'https://a.espncdn.com/i/teamlogos/nfl/500/la.png'
    # A relocated franchise this app has no entry for is OMITTED, not
    # mapped to a guessed URL that would render as a broken image.
    for gone in ('OAK', 'SD', 'STL', '', None):
        assert team_logo(gone) is None, gone


def test_team_colour_is_validated_through_the_same_gate():
    assert team_color('KC') is not None
    assert team_color('OAK') is None
    assert team_color(None) is None


# --- kickoff formatting and timezones -------------------------------------

def test_kickoff_strips_the_leading_zero_and_labels_the_zone():
    day = datetime.date(2026, 9, 13)
    out = format_kickoff(day, datetime.time(8, 0))
    assert '8:00 PM' in out or '8:00 AM' in out
    assert ' 08:' not in out


def test_zone_label_follows_dst_in_both_directions():
    # An NFL season straddles the boundary: September is EDT, the postseason
    # is EST. One hardcoded label would be wrong for every playoff game.
    september = format_kickoff(datetime.date(2026, 9, 13), datetime.time(13, 0))
    february = format_kickoff(datetime.date(2027, 2, 7), datetime.time(18, 30))
    assert september.endswith('EDT') or september.endswith('ET'), september
    assert february.endswith('EST') or february.endswith('ET'), february
    if september.endswith('EDT'):
        assert february.endswith('EST'), "must not use one fixed label all season"


def test_a_tbd_kickoff_takes_its_weekday_from_the_date_not_a_conversion():
    day = datetime.date(2026, 9, 13)          # a Sunday
    assert format_kickoff(day, None) == 'TBD'
    long = format_kickoff(day, None, long=True)
    assert long == 'Sunday · time TBD', long
    assert 'Saturday' not in long, "a placeholder time must never shift the day"


def test_missing_day_degrades_rather_than_raising():
    assert format_kickoff(None, datetime.time(13, 0)) == 'TBD'


# --- week selection --------------------------------------------------------

def test_default_week_is_the_last_one_with_a_played_game():
    weeks = [
        {'week': 1, 'games': 16, 'any_played': True},
        {'week': 2, 'games': 16, 'any_played': True},
        {'week': 3, 'games': 16, 'any_played': False},
    ]
    assert default_week_index(weeks) == 1


def test_a_season_with_nothing_played_opens_on_the_first_week_not_an_empty_screen():
    weeks = [{'week': w, 'games': 16, 'any_played': False} for w in (1, 2, 3)]
    assert default_week_index(weeks) == 0
    assert default_week_index([]) == 0


def test_playoff_rounds_have_labels():
    for code in ('WC', 'DIV', 'CON', 'SB'):
        assert ROUND_LABELS[code]


# --- team bridge -----------------------------------------------------------

def test_bridge_omits_an_unresolvable_team_rather_than_mapping_it_to_none():
    games = pd.DataFrame({'Away': ['KC', 'OAK'], 'Home': ['BUF', 'STL']})
    bridge = slate_team_bridge(games)
    assert bridge == {'KC': 'KC', 'BUF': 'BUF'}
    # The distinction that matters: a None VALUE would sail through the
    # `.get(x) is None` check every call site uses and seed a null.
    assert 'OAK' not in bridge and 'STL' not in bridge
    assert None not in bridge.values()


def test_an_empty_game_list_yields_an_empty_bridge():
    assert slate_team_bridge(pd.DataFrame()) == {}
    assert slate_team_bridge(None) == {}


# --- button wiring ---------------------------------------------------------

def test_each_button_seeds_its_own_team_and_an_unbridged_team_is_disabled():
    """
    AppTest can't reliably drive a tab switch followed by another
    interaction with this tabs pattern, so the wiring is verified by calling
    the renderer with st.button captured.
    """
    import streamlit as st
    import ui.tabs.game_slate as tab

    captured = []
    real_button, real_container, real_columns, real_markdown = (
        st.button, st.container, st.columns, st.markdown)

    class _Ctx:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    st.button = lambda label, **kw: captured.append((label, kw)) or False
    st.container = lambda **kw: _Ctx()
    st.columns = lambda n, **kw: [_Ctx() for _ in range(n if isinstance(n, int) else len(n))]
    st.markdown = lambda *a, **kw: None
    try:
        tab._render_card(0, _row(), 2026, {'KC': 'KC', 'BUF': 'BUF'})
        assert len(captured) == 2
        away_label, away_kw = captured[0]
        assert away_label == 'KC players'
        assert away_kw['disabled'] is False
        # Routes into the Matchup Analyzer with THIS team as the roster
        # filter and the actual opponent pre-filled as the defense to check
        # against - the whole point of task 5's routing change.
        assert away_kw['kwargs']['ma_jump_team'] == 'KC'
        assert away_kw['kwargs']['ma_jump_defense'] == 'BUF'
        assert away_kw['kwargs']['ma_jump_season'] == 2026
        assert away_kw['on_click'] is tab.switch_tab
        home_label, home_kw = captured[1]
        assert home_label == 'BUF players'
        assert home_kw['kwargs']['ma_jump_team'] == 'BUF'
        assert home_kw['kwargs']['ma_jump_defense'] == 'KC'

        captured.clear()
        tab._render_card(0, _row(**{'Away': 'OAK'}), 2026, {'BUF': 'BUF'})
        oak_label, oak_kw = captured[0]
        assert oak_kw['disabled'] is True, "an unbridgeable team disables its button"
        assert oak_kw['kwargs'] == {}, "and seeds nothing at all"
        assert oak_kw['help']
    finally:
        st.button, st.container = real_button, real_container
        st.columns, st.markdown = real_columns, real_markdown


def test_seeded_team_filter_is_a_valid_option_in_the_destination():
    """
    This app has no sticky wrappers, so a cross-tab seed writes the
    destination's widget key directly - and Streamlit RAISES on the next run
    if that value isn't one of the selectbox's options. The bridge only ever
    emits TEAM_CONFIG keys, which is exactly the option list.
    """
    from config import TEAM_CONFIG
    games = pd.DataFrame({'Away': ['KC', 'LA'], 'Home': ['BUF', 'SF']})
    options = ['All Teams'] + list(TEAM_CONFIG)
    for seeded in slate_team_bridge(games).values():
        assert seeded in options, seeded


def test_slate_columns_contract_is_what_the_card_reads():
    row = _row()
    for column in ('Away', 'Home', 'Away Pts', 'Home Pts', 'Winner', 'Date',
                   'Date Display', 'Kickoff Long', 'Played', 'Away Color'):
        assert column in SLATE_COLUMNS, column
        assert column in row.index, column


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
