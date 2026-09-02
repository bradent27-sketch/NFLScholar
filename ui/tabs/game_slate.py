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

from config import (
    AVAILABLE_SEASONS_WITH_UPCOMING, TAB_MATCHUP_ANALYZER, TEAM_CONFIG, THEME,
)
from data.box_score import (
    BOX_SECTIONS, TOTALS_COMPARISON, game_players, section_rows, team_totals,
)
from data.game_slate import (
    escape, escape_attr, find_slate_game, load_slate, refresh_slate, slate_source,
    slate_team_bridge, slate_weeks, default_week_index, team_display_name,
    team_abbrev_label, team_color, team_logo,
)
from ui.components import (
    close_box_score, open_box_score, render_back_button, skeleton_loader, switch_tab,
)

# A full NFL week is ~16 games, so there is no paging, no conference filter
# and no "analyzable game" checkbox. Those exist in the college versions of
# this tab because a college date can carry 169 games, half of them against
# opponents with no stats. Porting them here would add widgets that can
# never do anything.
_CARDS_PER_ROW = 2


def _team_row(side, abbr, color, logo, points, is_winner, show_score, line_note=None):
    """
    One team's row inside a card.

    `is_winner` is deliberately three-valued. None means "no winner" - an
    unplayed game or a TIE - and must not collapse to False, because False
    would mark BOTH teams as losers and dim them.

    `line_note` is the Vegas snippet for THIS team - the point spread on the
    favorite's row ("-3.5"), the total on the other ("O/U 44.5"). It takes
    the same right-edge slot the score occupies, and only for an unplayed
    game: a played game shows the score there instead, and a game with no
    line posted yet passes None and shows nothing.
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
    elif line_note:
        parts.append(f"<span class='gs-line'>{escape(line_note)}</span>")
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

    # Vegas snippet per team, only when there's no score to show. nflverse's
    # `spread_line` is positive when the HOME team is favored (see
    # data/odds_market.py's sign note). The favorite's row gets the spread,
    # the other row gets the game total; a game with neither field posted
    # yet (too far out) passes None both ways and shows nothing.
    away_line = home_line = None
    if not show_score:
        spread = row.get('Spread Line')
        total = row.get('Total Line')
        if spread is not None and pd.notna(spread):
            spread = float(spread)
            if spread > 0:
                home_line = f"{-spread:g}"
            elif spread < 0:
                away_line = f"{spread:g}"
            else:
                home_line = 'PK'
        if total is not None and pd.notna(total):
            ou = f"O/U {float(total):g}"
            if home_line is not None and away_line is None:
                away_line = ou
            elif away_line is not None and home_line is None:
                home_line = ou
            elif home_line is None and away_line is None:
                home_line = ou

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
        f"{_team_row('AWAY', away, away_color, row.get('Away Logo'), row.get('Away Pts'), away_won, show_score, away_line)}"
        f"{_team_row('HOME', home, home_color, row.get('Home Logo'), row.get('Home Pts'), home_won, show_score, home_line)}"
    )


def _render_card(idx, row, season, bridge):
    with st.container(key=f"gs_card_{idx}"):
        st.markdown(card_html(idx, row), unsafe_allow_html=True)
        # The box-score control sits on its own row ABOVE the two team
        # buttons rather than beside them. Streamlit allows exactly one
        # level of column nesting and the team row already spends it, so a
        # third column here would have to share that row and squeeze both
        # team names to an unreadable width.
        game_id = str(row.get('Game Id') or '')
        if bool(row.get('Played')) and game_id:
            st.button(
                "📋 Box score", key=f"gs_box_{idx}", width="stretch",
                help="Full box score for this game, above the grid",
                on_click=open_box_score, args=(season, game_id),
                kwargs={'_from_slate': True},
            )
        cols = st.columns(2)
        for col, side in zip(cols, ('Away', 'Home')):
            abbr = row[side]
            other_abbr = row['Home'] if side == 'Away' else row['Away']
            resolved = bridge.get(abbr)
            other_resolved = bridge.get(other_abbr)
            with col:
                # Routes into the Matchup Analyzer with this team's roster
                # AND this game's actual opponent pre-filled as the defense
                # to check against - automates exactly the matchup this card
                # already represents, instead of dropping you in Player
                # Search with no defense context at all. Disabled rather
                # than seeding a guess when the team itself doesn't resolve:
                # a best-guess team opens the wrong roster with nothing
                # anywhere saying so. A missing OPPONENT resolution (a
                # relocated-franchise code from an old season, not in this
                # app's team list) only skips pre-filling the defense - the
                # team side still works.
                kwargs = {}
                if resolved:
                    kwargs['ma_jump_team'] = resolved
                    kwargs['ma_jump_season'] = int(season)
                    if other_resolved:
                        kwargs['ma_jump_defense'] = other_resolved
                st.button(
                    f"{team_abbrev_label(abbr)} players",
                    key=f"gs_go_{side.lower()}_{idx}",
                    width="stretch",
                    disabled=resolved is None,
                    help=(f"Open the Matchup Analyzer with {team_display_name(abbr)} vs {team_display_name(other_abbr)}'s defense"
                          if resolved else
                          f"{abbr} isn't in this app's team list, so it can't be opened"),
                    on_click=switch_tab,
                    args=(TAB_MATCHUP_ANALYZER,),
                    kwargs=kwargs,
                )


def _box_header_html(game):
    """Both teams, logos, scores, with the winner's score brightened."""
    away, home = game['Away'], game['Home']
    winner = game.get('Winner')
    winner = winner if (winner is not None and pd.notna(winner)) else None
    rows = []
    for side, abbr, points in (('AWAY', away, game.get('Away Pts')), ('HOME', home, game.get('Home Pts'))):
        color = team_color(abbr) or THEME['colors']['outline']
        logo = team_logo(abbr)
        won = (winner == abbr) if winner else None
        classes = 'bs-team' + (' bs-won' if won is True else (' bs-lost' if won is False else ''))
        score = f"<span class='bs-score'>{int(points)}</span>" if pd.notna(points) else ''
        logo_html = f"<img class='bs-logo' src='{escape_attr(logo)}' alt=''>" if logo else ''
        rows.append(
            f"<div class='{classes}' style='--bs-color:{color};'>"
            f"<span class='bs-side'>{side}</span>{logo_html}"
            f"<span class='bs-name'>{escape(team_display_name(abbr))}</span>{score}</div>"
        )
    meta = [escape(game.get('Date Display')), escape(game.get('Status Detail') or '')]
    if game.get('Venue') and pd.notna(game.get('Venue')):
        meta.append(escape(str(game['Venue'])))
    if game.get('Broadcast') and pd.notna(game.get('Broadcast')):
        meta.append(escape(str(game['Broadcast'])))
    meta_html = "<span class='gs-dot'>·</span>".join(m for m in meta if m)
    return f"<div class='bs-header'>{''.join(rows)}<div class='bs-meta'>{meta_html}</div></div>"


def _totals_bar_html(label, away_value, home_value, more_is_better, away_color, home_color):
    """
    One shared track per stat, split by each side's share.

    A 0-0 row does not collapse the track - it splits evenly, because an
    empty bar reads as missing data rather than as "neither team did this".
    `more_is_better=False` (turnovers, penalties) flips which side is
    highlighted, so the team with MORE turnovers isn't shown winning the row.
    """
    total = float(away_value) + float(home_value)
    away_share = 50.0 if total <= 0 else float(away_value) / total * 100
    leader = None
    if away_value != home_value:
        away_ahead = away_value > home_value
        leader = 'away' if (away_ahead == more_is_better) else 'home'
    fmt = (lambda v: f"{v:.0f}")
    return (
        f"<div class='bs-cmp-row'>"
        f"<span class='bs-cmp-val{' bs-cmp-lead' if leader == 'away' else ''}'>{fmt(away_value)}</span>"
        f"<span class='bs-cmp-label'>{escape(label)}</span>"
        f"<span class='bs-cmp-val{' bs-cmp-lead' if leader == 'home' else ''}'>{fmt(home_value)}</span>"
        f"<div class='bs-cmp-track'>"
        f"<div class='bs-cmp-fill' style='width:{away_share:.1f}%; background:{away_color};'></div>"
        f"<div class='bs-cmp-fill' style='width:{100 - away_share:.1f}%; background:{home_color};'></div>"
        f"</div></div>"
    )


def _render_box_panel(season):
    """
    The box score, FULL WIDTH, called from one of two spots in render():

      - Inline, right after the grid row that holds the clicked card - the
        common case (you're already looking at the slate and click a card's
        own "Box score" button). This is what lets the panel expand from
        the row you clicked instead of the page jumping to the top, and
        what puts you back at that same row when you close it - it's
        DOM position, not a scroll hack.
      - Above the grid, before the week's games are even loaded, when the
        open game isn't part of the week currently on screen at all - a
        deep link from another tab (Player Search's game log, the Matchup
        Analyzer) can legitimately name a game from a DIFFERENT week or
        season than the selectors above happen to be showing. render()
        resolves which case applies; this function itself doesn't care
        which spot it was called from.

    Full width either way, because a two-team box is unreadable at half
    width, and because a card has already spent Streamlit's single level of
    column nesting on its own button row.

    Resolved against the WHOLE season (data.game_slate.find_slate_game),
    never just the week on screen, because this panel is the destination
    for every cross-tab link in the app - see that function's docstring.
    """
    game_id = st.session_state.get('gs_box_game')
    if not game_id:
        return
    game = find_slate_game(season, game_id)
    if game is None:
        # The link named a game this season doesn't have - most likely the
        # season selector moved. Say so and clear it, rather than leaving a
        # control that silently does nothing.
        st.info("That game isn't in the selected season. Pick the right season, or open another game.")
        st.button("Close", key="gs_box_close_missing", on_click=close_box_score)
        return

    st.markdown(_box_header_html(game), unsafe_allow_html=True)
    away, home = game['Away'], game['Home']
    away_color = team_color(away) or THEME['colors']['secondary']
    home_color = team_color(home) or THEME['colors']['primary']

    # Loaded HERE, not in render(), so a slate with no box open pays nothing
    # for this. A dedicated raw loader, not the six-tabs-share
    # load_and_merge_data() - that one runs every row through
    # load_year_data()'s REG-only filter (see its docstring), which drops
    # every playoff game before a box score ever gets a chance to look one
    # up. load_box_score_stats() reads the same file without that filter -
    # see its docstring for why that's safe for a single-game lookup.
    from data.loaders import load_box_score_stats
    with skeleton_loader("table", n_rows=6, n_cols=6):
        stats_df = load_box_score_stats(season)
    players = game_players(stats_df, game_id)
    if players.empty:
        st.caption("No player box score on file for this game yet.")
        st.button("Close box score", key="gs_box_close", on_click=close_box_score)
        return

    away_totals, home_totals = team_totals(players, away), team_totals(players, home)
    if away_totals and home_totals:
        bars = ''.join(
            _totals_bar_html(label, away_totals.get(label, 0), home_totals.get(label, 0),
                             more_is_better, away_color, home_color)
            for label, more_is_better in TOTALS_COMPARISON
            if label in away_totals
        )
        st.markdown(f"<div class='bs-compare'>{bars}</div>", unsafe_allow_html=True)

    from ui.styling import df_auto_height, style_plain_dataframe
    team_tabs = st.tabs([team_display_name(away), team_display_name(home)])
    for tab, abbr in zip(team_tabs, (away, home)):
        with tab:
            rendered_any = False
            for title, columns, gate in BOX_SECTIONS:
                table = section_rows(players, abbr, columns, gate)
                if table.empty:
                    continue
                rendered_any = True
                st.markdown(f"**{title}**")
                st.dataframe(
                    style_plain_dataframe(table.set_index('Player')),
                    width="stretch", height=df_auto_height(len(table)),
                )
            if not rendered_any:
                st.caption("No stat lines on file for this team in this game.")

    action_cols = st.columns(3)
    with action_cols[0]:
        st.button("Close box score", key="gs_box_close", width="stretch", on_click=close_box_score)
    for col, abbr, other in ((action_cols[1], away, home), (action_cols[2], home, away)):
        with col:
            st.button(
                f"{team_abbrev_label(abbr)} vs {team_abbrev_label(other)} defense",
                key=f"gs_box_ma_{abbr}", width="stretch",
                help=f"Open the Matchup Analyzer with the {other} defense selected",
                on_click=switch_tab, args=(TAB_MATCHUP_ANALYZER,),
                kwargs={'ma_jump_defense': other},
            )
    st.divider()


def render():
    # This tab is now a jump DESTINATION as well as a launchpad - a box-score
    # chip on Player Search or the Matchup Analyzer lands here - so it needs
    # the same way back every other destination tab already offers. Rendered
    # here rather than once above the tab bar in app.py, because
    # render_back_button uses a fixed widget key and instantiating it in two
    # places at once is a duplicate-key error.
    render_back_button()
    st.markdown("<div class='custom-section-header'>GAME SLATE</div>", unsafe_allow_html=True)

    weeks, err = [], None
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        season = st.selectbox("Season", AVAILABLE_SEASONS_WITH_UPCOMING, index=0,
                              key="gs_season")

    # Whichever game is open, if any - resolved once, up front, so both the
    # "does it belong to the week on screen" check below and the panel
    # itself (called from one of two spots) agree on the same value.
    open_game_id = st.session_state.get('gs_box_game')

    with skeleton_loader("table", n_rows=4, n_cols=4):
        weeks, err = slate_weeks(season)

    if err or not weeks:
        # No grid to anchor into - same "deep-link destination, not a page
        # element" reasoning as the inline case below, just with nowhere to
        # inline it into.
        if open_game_id:
            _render_box_panel(season)
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
        if open_game_id:
            _render_box_panel(season)
        st.info("No games in that week.")
        return

    # Where the open game sits IN THIS WEEK'S GRID, if it's here at all. A
    # box score opened by clicking a card on THIS page is always here - this
    # is the common case, and it's what lets the panel expand from the same
    # row you clicked instead of the page yanking you to the top. A box
    # score opened by a deep link from another tab (Player Search's game
    # log, the Matchup Analyzer) can legitimately name a game from a
    # DIFFERENT week or season than whatever the selectors above happen to
    # be showing right now - see find_slate_game's own docstring - so that
    # case still falls back to rendering above the grid, exactly as before,
    # rather than silently not opening at all.
    inline_row_start = None
    if open_game_id:
        hits = games.index[games['Game Id'].astype(str) == str(open_game_id)]
        if len(hits):
            row_pos = games.index.get_loc(hits[0])
            inline_row_start = (row_pos // _CARDS_PER_ROW) * _CARDS_PER_ROW
        else:
            _render_box_panel(season)

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
        # Full width, right after the row that contains the clicked card -
        # not squeezed into either column, for the same "unreadable at half
        # width" reason _render_box_panel's own docstring gives. Closing it
        # (close_box_score just pops gs_box_game) leaves you exactly here,
        # at this row, instead of back at the top of the page.
        if start == inline_row_start:
            _render_box_panel(season)

    played = int(games['Played'].sum())
    tbd = int(games['Time TBD'].sum())
    bits = [f"{len(games)} games", f"{played} played", f"source: {slate_source(season)}"]
    if tbd:
        bits.insert(2, f"{tbd} kickoff{'s' if tbd != 1 else ''} TBD")
    st.caption(' · '.join(bits))
