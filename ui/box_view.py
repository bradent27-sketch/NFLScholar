"""
A game's full box score, opened from INSIDE the Weekly Rankings projection decomposition.

WHAT THE USER ASKED FOR (2026-10-06): in the decomposition, next to the opponent on the player's own game log and on the
defense-allowed logs, show the game's final score for context (blowout? trailing?), and let a click on that score open
the whole box score - every player, like the Game Slate tab - without leaving the Weekly model tab.

WHY IT IS A VIEW INSIDE THE DIALOG AND NOT A SECOND DIALOG. Streamlit allows one dialog at a time, and the
decomposition already is one. Clicking a Score therefore swaps the dialog's content for the box score (with a Back
button that returns to exactly the tab and season the reader left); the projection breakdown itself is untouched and
is rendered exactly as before whenever no box is open.

WHY A CLICK ON A CELL WORKS NOW. The rest of the app opens box scores with a strip of buttons under a table
(ui.components.render_game_links) because a canvas table could not fire a Python callback. st.dataframe's
`on_select="rerun", selection_mode="single-cell"` (this app is on 1.59.1) reports which cell was clicked as
(row position, column name), so the Score cell itself is the control.

STATE. One session key holds the open box ({'owner', 'season', 'game_id'}); `owner` ties it to the player whose
dialog opened it, so a stale box can never show on another player's breakdown. A generation counter is appended to
each table's widget key and bumped on close, so a table that comes back after the box starts with nothing selected
(otherwise the old selection would reopen the box straight away).
"""
import pandas as pd
import streamlit as st

BOX_VIEW_KEY = 'wr_box_view'
BOX_GEN_KEY = 'wr_box_gen'
NO_SCORE = '—'


def owner_key(detail):
    """Identity of the open decomposition: the same player, team and weeks."""
    return (str(detail.get('player')), str(detail.get('team')),
            int(detail.get('target_week') or 0), int(detail.get('as_of_week') or 0))


def open_box_view(owner, season, game_id):
    st.session_state[BOX_VIEW_KEY] = {'owner': owner, 'season': int(season), 'game_id': str(game_id)}


def close_box_view():
    """Back to the breakdown. A callback (button on_click) and also what the dialog's dismiss uses."""
    st.session_state.pop(BOX_VIEW_KEY, None)
    st.session_state[BOX_GEN_KEY] = int(st.session_state.get(BOX_GEN_KEY, 0)) + 1


def active_box_view(owner):
    """The open box for THIS decomposition, or None."""
    view = st.session_state.get(BOX_VIEW_KEY)
    if view and view.get('owner') == owner:
        return view
    return None


def table_generation():
    return int(st.session_state.get(BOX_GEN_KEY, 0))


def score_entries(df, season, team, week_col, id_col=None):
    """One entry per row of ``df``, in order: the resolved game (data.box_score.game_link_positions) or None.

    Resolves by ``id_col`` when the log has it, else by (week, ``team``) - exact in the NFL, a team plays once a
    week. ``team`` is the side the score is read from: the player's team for his own log, the DEFENSE for a
    defense-allowed log (its own points first)."""
    from data.box_score import game_link_positions
    from data.game_slate import season_slate
    if df is None or len(df) == 0:
        return []
    slate, _err = season_slate(season)
    return game_link_positions(df, slate, team=team, id_col=id_col or '__no_game_id__', week_col=week_col)


def score_texts(entries):
    """The Score cell text for each entry: "24-17" (team first), or a dash for a game with no final yet."""
    return [(e.get('score') or NO_SCORE) if e else NO_SCORE for e in entries]


def clicked_entry(selection, entries, score_col):
    """The entry whose Score cell was clicked, or None.

    ``selection`` is the object st.dataframe returns for ``on_select`` (attribute or dict access); only a cell in
    ``score_col`` counts, a row past the games (the AVG row) or one that resolved to no game counts for nothing."""
    cells = None
    try:
        cells = selection.selection.cells
    except AttributeError:
        try:
            cells = selection['selection']['cells']
        except (KeyError, TypeError):
            cells = None
    for row, col in (cells or ()):
        if col == score_col and isinstance(row, int) and 0 <= row < len(entries) and entries[row]:
            return entries[row]
    return None


def render_game_table(styled, entries, *, score_col, key, owner, season, height):
    """``st.dataframe`` with a clickable Score column. A click opens that game's box score and reruns the dialog
    so the box shows at once; with nothing resolvable it is just a table."""
    if not any(entries):
        st.dataframe(styled, hide_index=True, width='stretch', height=height)
        return
    event = st.dataframe(
        styled, hide_index=True, width='stretch', height=height,
        on_select='rerun', selection_mode='single-cell', key=f'{key}_{table_generation()}')
    hit = clicked_entry(event, entries, score_col)
    if hit:
        open_box_view(owner, season, hit['game_id'])
        try:
            st.rerun(scope='fragment')
        except Exception:   # not inside a fragment (a plain page, a test): a whole-app rerun does the same job
            st.rerun()


def box_view_context(detail, game):
    """(focus_team, highlight_player, description) for the open game: whose tab to show first, whether the
    decomposition's player can be bolded, and one line saying what game this is.

    The clicked game is either the player's own (his team is in it) or one of the opposing DEFENSE's weekly
    games from the defense-allowed logs (his team is not in it), and the caption must not call that second kind
    "his" box score."""
    away, home = str(game['Away']).upper(), str(game['Home']).upper()
    team = str(detail.get('team') or '').upper()
    opponent = str(detail.get('opponent') or '').upper()
    player = detail.get('player')
    matchup = f"{game['Away']} @ {game['Home']}" if not bool(game.get('Neutral Site')) else f"{game['Away']} vs {game['Home']}"
    head = f"{int(game['Week'])} {matchup}"
    if team in (away, home):
        return detail.get('team'), player, f"Week {head} · {player}'s game"
    if opponent in (away, home):
        return detail.get('opponent'), None, (
            f"Week {head} · a {detail.get('opponent')} game, the defense {player} faces this week "
            f"(not {player}'s own game)")
    return None, None, f"Week {head}"


def render_box_view(detail, view):
    """The open game's full box score in place of the breakdown, with a way back."""
    from data.game_slate import find_slate_game
    from ui.tabs.game_slate import render_box_body
    st.button('← Back to the projection breakdown', key='wr_box_back_top', on_click=close_box_view)
    game = find_slate_game(view['season'], view['game_id'])
    if game is None:
        st.info("That game isn't on the schedule this season - go back and pick another.")
        return
    focus_team, highlight, description = box_view_context(detail, game)
    st.caption(f"{view['season']} {description}")
    shown = render_box_body(view['season'], game, focus_team=focus_team, highlight_player=highlight)
    if shown and highlight:
        st.caption(f"{highlight}'s line is in bold; every other player in the game is in the team tabs.")
    st.button('← Back to the projection breakdown', key='wr_box_back_bottom', on_click=close_box_view)
