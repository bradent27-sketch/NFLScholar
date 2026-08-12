"""
Game Slate: who is playing this week, and a way into every other tab with
both teams already filled in.

It is a LAUNCHPAD, NOT A REPORT. It answers the first question of a prep
session and then hands the chosen game onward.

WHY CARDS AND NOT A TABLE, since the obvious move is st.dataframe and it was
tried first in the app this was ported from. Streamlit's dataframe renders on
a <canvas>, so it cannot hold a button or a link. Acting on a game from a
table therefore means re-picking it from a second widget underneath - you
select a row with your eyes, then select the same game again with a
dropdown. The card design exists to delete that second selection. A table
also gives every column equal weight; a card can make the teams and the
score big and the metadata small.

THE ONE CONSTRAINT THAT SHAPES EVERYTHING: switch_tab must run as an
on_click callback (see ui.components.switch_tab - Streamlit raises if
active_tab is reassigned during the run that already read it). Raw HTML
cannot fire a Python callback, so the card cannot be one HTML block. It is a
KEYED st.container, which Streamlit renders with the class `st-key-<key>`,
and ui/styling.py styles that class prefix. If this is ever "simplified"
into pure HTML with <a> tags, the pre-seeding - the entire point of the tab
- goes with it.
"""
import pandas as pd
import streamlit as st

from config import AVAILABLE_SEASONS_WITH_UPCOMING, TAB_PLAYER_SEARCH, TEAM_CONFIG
from data.game_slate import (
    escape, escape_attr, load_slate, refresh_slate, slate_source, slate_team_bridge,
    slate_weeks, default_week_index, team_display_name, team_abbrev_label,
)
from ui.components import skeleton_loader, switch_tab

# A full NFL week is ~16 games, so there is no paging, no conference filter
# and no "analyzable game" checkbox. Those exist in the college versions of
# this tab because a college date can carry 169 games, half of them against
# opponents with no stats. Porting them here would add widgets that can
# never do anything.
_CARDS_PER_ROW = 2


def _team_row(side, abbr, color, logo, points, is_winner, show_score):
    """
    One team's row inside a card.

    `is_winner` is deliberately three-valued. None means "no winner" - an
    unplayed game or a TIE - and must not collapse to False, because False
    would mark BOTH teams as losers and dim them.
    """
    classes = ['gs-team']
    if is_winner is True:
        classes.append('gs-won')
    elif is_winner is False:
        classes.append('gs-lost')

    style = f" style='--gs-color:{color};'" if color else ''
    parts = [f"<div class='{' '.join(classes)}'{style}>",
             f"<span class='gs-side'>{side}</span>"]
    if logo:
        parts.append(f"<img class='gs-logo' src='{escape_attr(logo)}' alt=''>")
    parts.append(f"<span class='gs-name'>{escape(team_display_name(abbr))}</span>")

    # An unplayed game emits NO score node at all - not '--', not 'NA', not
    # an empty span. A blank right edge is the honest reading of "hasn't
    # happened yet"; a placeholder reads as missing data.
    if show_score and pd.notna(points):
        parts.append(f"<span class='gs-score'>{int(points)}</span>")
        if is_winner is True:
            parts.append("<span class='gs-win-flag'>W</span>")
    parts.append('</div>')
    return ''.join(parts)


def card_html(idx, row):
    """The card's markup. Pure function of one row, so it is directly testable."""
    away, home = row['Away'], row['Home']
    away_color, home_color = row.get('Away Color'), row.get('Home Color')

    # Emitted only when at least one real color survived validation - an
    # empty rule is harmless but pointless, and an unvalidated one would
    # break the whole style block.
    style_tag = ''
    if away_color or home_color:
        decls = ''.join(filter(None, [
            f'--gs-a:{away_color};' if away_color else '',
            f'--gs-b:{home_color};' if home_color else '',
        ]))
        style_tag = f"<style>.st-key-gs_card_{idx}{{{decls}}}</style>"

    played = bool(row.get('Played'))
    live = bool(row.get('Live'))
    show_score = played or live

    winner = row.get('Winner')
    winner = winner if (winner is not None and pd.notna(winner)) else None
    # None, not False - see _team_row.
    away_won = (winner == away) if winner else None
    home_won = (winner == home) if winner else None

    meta = [f"<span class='gs-date'>{escape(row.get('Date Display'))}</span>"]
    status_detail = str(row.get('Status Detail') or '')
    if status_detail in ('Postponed', 'Canceled', 'Suspended'):
        # Show the status where the kickoff would go, rather than
        # advertising a start time the game will never have.
        meta.append(f"<span class='gs-status'>{escape(status_detail)}</span>")
    else:
        meta.append(f"<span>{escape(row.get('Kickoff Long'))}</span>")
    venue = row.get('Venue')
    if venue and pd.notna(venue):
        text = f"{venue} (neutral)" if row.get('Neutral Site') else str(venue)
        meta.append(f"<span>{escape(text)}</span>")
    if row.get('Broadcast') and pd.notna(row.get('Broadcast')):
        meta.append(f"<span class='gs-tv'>{escape(row['Broadcast'])}</span>")

    dotted = "<span class='gs-dot'>·</span>".join(meta)
    headline = row.get('Headline')
    headline_html = (f"<div class='gs-headline'>{escape(headline)}</div>"
                     if headline and pd.notna(headline) else '')

    return (
        f"{style_tag}"
        f"<div class='gs-meta'>{dotted}</div>"
        f"{headline_html}"
        f"{_team_row('AWAY', away, away_color, row.get('Away Logo'), row.get('Away Pts'), away_won, show_score)}"
        f"{_team_row('HOME', home, home_color, row.get('Home Logo'), row.get('Home Pts'), home_won, show_score)}"
    )


def _render_card(idx, row, season, bridge):
    with st.container(key=f"gs_card_{idx}"):
        st.markdown(card_html(idx, row), unsafe_allow_html=True)
        cols = st.columns(2)
        for col, side in zip(cols, ('Away', 'Home')):
            abbr = row[side]
            resolved = bridge.get(abbr)
            with col:
                st.button(
                    f"{team_abbrev_label(abbr)} players",
                    key=f"gs_go_{side.lower()}_{idx}",
                    width="stretch",
                    # Disabled rather than seeding a guess: a best-guess team
                    # opens Player Search on the WRONG team with nothing
                    # anywhere saying so.
                    disabled=resolved is None,
                    help=(f"Open Player Search filtered to {team_display_name(abbr)}"
                          if resolved else
                          f"{abbr} isn't in this app's team list, so it can't be opened"),
                    on_click=switch_tab,
                    args=(TAB_PLAYER_SEARCH,),
                    kwargs={'jump_to_year': int(season),
                            'player_search_team_filter': resolved} if resolved else {},
                )


def render():
    st.markdown("<div class='custom-section-header'>GAME SLATE</div>", unsafe_allow_html=True)

    weeks, err = [], None
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        season = st.selectbox("Season", AVAILABLE_SEASONS_WITH_UPCOMING, index=0,
                              key="gs_season")

    with skeleton_loader("table", n_rows=4, n_cols=4):
        weeks, err = slate_weeks(season)

    if err or not weeks:
        st.info(err or f"No {season} games in the schedule file yet.")
        return

    labels = [f"{w['label']} · {w['games']} games" for w in weeks]
    with c2:
        chosen = st.selectbox("Week", labels, index=default_week_index(weeks),
                              key=f"gs_week_{season}")
    week = weeks[labels.index(chosen)]

    with c3:
        st.write("")
        if st.button("🔄 Refresh", key="gs_refresh", help="Re-download the schedule."):
            refresh_slate()
            st.rerun()

    games, _ = load_slate(season, week['week'])
    if games.empty:
        st.info("No games in that week.")
        return

    bridge = slate_team_bridge(games)

    # Row by row, not column by column. Filling column 0 with the first half
    # of the week and column 1 with the second half makes the two sides of
    # the screen scroll out of chronological order.
    #
    # Streamlit allows exactly one level of column nesting and this spends
    # it: grid column -> button row. Nothing inside a card may open more.
    for start in range(0, len(games), _CARDS_PER_ROW):
        cols = st.columns(_CARDS_PER_ROW)
        for offset, col in enumerate(cols):
            idx = start + offset
            if idx >= len(games):
                continue
            with col:
                _render_card(idx, games.iloc[idx], season, bridge)

    played = int(games['Played'].sum())
    tbd = int(games['Time TBD'].sum())
    bits = [f"{len(games)} games", f"{played} played", f"source: {slate_source(season)}"]
    if tbd:
        bits.insert(2, f"{tbd} kickoff{'s' if tbd != 1 else ''} TBD")
    st.caption(' · '.join(bits))
