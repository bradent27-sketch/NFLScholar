"""
Live Odds tab: refresh-on-demand game lines (moneyline/spread/total across
every bookmaker) and player props (every market The Odds API documents for
NFL - see config.ODDS_API_PLAYER_PROP_MARKETS) for a specific selected game,
pivoted so the same bet's price can be compared across every book at a glance.
"""
import datetime
import pandas as pd
import streamlit as st

from config import ODDS_API_PLAYER_PROP_MARKETS
from data.loaders import fetch_nfl_odds, fetch_nfl_player_props, load_saved_odds_api_key, save_odds_api_key
from ui.styling import style_plain_dataframe, df_auto_height
from ui.components import skeleton_loader


def _fmt_kickoff(iso_str):
    try:
        dt = datetime.datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        return dt.strftime('%a %b %d, %I:%M %p UTC')
    except Exception:
        return iso_str


def _pt(val):
    """Round a spread/total point value to 1 decimal - raw API floats
    otherwise render as e.g. "3.500000" (a float column with no natural
    precision hint renders with a long default decimal tail)."""
    return round(float(val), 1) if val is not None else None


def _build_lines_table(game):
    """One row per bookmaker for the selected game - moneyline, spread, total."""
    home, away = game.get('home_team'), game.get('away_team')
    rows = []
    for book in game.get('bookmakers', []):
        row = {'Book': book.get('title', book.get('key', '?'))}
        for market in book.get('markets', []):
            if market['key'] == 'h2h':
                for o in market['outcomes']:
                    if o['name'] == home: row['Home ML'] = o['price']
                    if o['name'] == away: row['Away ML'] = o['price']
            elif market['key'] == 'spreads':
                for o in market['outcomes']:
                    if o['name'] == home: row['Home Spread'] = _pt(o.get('point'))
                    if o['name'] == away: row['Away Spread'] = _pt(o.get('point'))
            elif market['key'] == 'totals':
                for o in market['outcomes']:
                    if o['name'] == 'Over': row['Total (O/U)'] = _pt(o.get('point'))
        rows.append(row)
    return pd.DataFrame(rows)


def _build_props_long_table(props_data):
    """One row per (book, market, player, selection) - the raw shape,
    kept for the market filter/summary and as the source for the pivot
    below."""
    rows = []
    for book in props_data.get('bookmakers', []):
        for market in book.get('markets', []):
            for o in market.get('outcomes', []):
                rows.append({
                    'Market': market.get('key', '').replace('player_', '').replace('_', ' ').title(),
                    'Player': o.get('description') or o.get('name'),
                    'Selection': o.get('name'),
                    'Line': o.get('point'),
                    'Odds': o.get('price'),
                    'Book': book.get('title', book.get('key', '?')),
                })
    return pd.DataFrame(rows)


def _build_props_comparison_table(props_long_df):
    """
    Pivots the long table so each row is ONE bet (Market + Player +
    Selection) and each column is a bookmaker, so the same bet's price
    (and line, if that market has one) can be compared across every book
    that posted it in a single glance, instead of scanning one row per
    book. Cell text combines odds + line (when present) since different
    books can quote slightly different lines for the same market/player -
    collapsing to odds alone would silently hide that.
    """
    if props_long_df.empty:
        return pd.DataFrame()

    def fmt_cell(r):
        try:
            odds_txt = f"{int(r['Odds']):+d}"
        except (TypeError, ValueError):
            return ''
        if pd.notna(r['Line']):
            return f"{odds_txt} ({r['Line']:g})"
        return odds_txt

    work = props_long_df.copy()
    work['_cell'] = work.apply(fmt_cell, axis=1)
    pivot = work.pivot_table(
        index=['Market', 'Player', 'Selection'], columns='Book', values='_cell', aggfunc='first'
    ).reset_index()
    pivot.columns.name = None
    return pivot


def render():
    st.markdown("<div class='custom-section-header'>LIVE NFL ODDS</div>", unsafe_allow_html=True)

    # value= only seeds the widget on its very first render this session
    # (session_state takes over after that) - so this pre-fills the box
    # from the saved file on a fresh app launch without fighting Streamlit
    # over who owns the keyed widget's value on later reruns.
    saved_key = load_saved_odds_api_key()
    odds_api_key = st.text_input(
        "The Odds API key", type="password", key="odds_api_key", value=saved_key,
        help="Saved locally in .streamlit/odds_api_key.txt so you don't have to re-enter it every launch. "
             "That file is plain text, not encrypted - delete it (or clear this field) before sharing this project folder with anyone.",
    )
    if odds_api_key and odds_api_key != saved_key:
        save_odds_api_key(odds_api_key)

    if not odds_api_key:
        st.info("Enter an API key above to load odds. Free tier available at the-odds-api.com.")
        return

    oc1, oc2 = st.columns([1, 3])
    with oc1:
        if st.button("🔄 Refresh Odds"):
            fetch_nfl_odds.clear()
            fetch_nfl_player_props.clear()
            st.session_state.pop('odds_props_game_id', None)
            st.session_state.pop('odds_props_data', None)
            st.rerun()

    with skeleton_loader("table", n_rows=6, n_cols=5):
        odds_data, odds_err, requests_left = fetch_nfl_odds(odds_api_key)

    if odds_err:
        st.error(f"Couldn't fetch odds: {odds_err}")
        return
    if not odds_data:
        st.info("No upcoming NFL games with odds right now.")
        return

    if requests_left:
        st.caption(f"API requests remaining this period: {requests_left}")

    game_labels = {}
    for g in odds_data:
        label = f"{g.get('away_team','?')} @ {g.get('home_team','?')} — {_fmt_kickoff(g.get('commence_time',''))}"
        game_labels[label] = g
    sel_label = st.selectbox("Select a game", list(game_labels.keys()), key="odds_game_select")
    game = game_labels[sel_label]

    st.markdown(f"<div class='custom-section-header'>{game.get('away_team')} @ {game.get('home_team')}</div>", unsafe_allow_html=True)
    st.caption(f"Kickoff: {_fmt_kickoff(game.get('commence_time',''))}")

    lines_df = _build_lines_table(game)
    if not lines_df.empty:
        st.markdown("**Game Lines — Moneyline / Spread / Total (every bookmaker, click a header to sort)**")
        st.dataframe(style_plain_dataframe(lines_df.set_index('Book')), width="stretch", height=df_auto_height(len(lines_df)))
    else:
        st.info("No bookmakers have posted lines for this game yet.")

    st.markdown("**Player Props**")

    # Fetched props are stored in session_state (keyed to this game's id) so
    # the market filter/sort widgets below - which trigger their own
    # reruns - don't lose the fetched data. st.button() only returns True on
    # the exact rerun it was clicked; without this, changing the filter
    # dropdown would silently make the whole props section disappear on the
    # very next interaction, since it would have been gated entirely behind
    # `if st.button(...)`.
    if st.button("Load player props for this game") or st.session_state.get('odds_props_game_id') == game['id']:
        if st.session_state.get('odds_props_game_id') != game['id']:
            with skeleton_loader("table", n_rows=8, n_cols=5):
                props_data, props_err = fetch_nfl_player_props(
                    odds_api_key, game['id'], markets=','.join(ODDS_API_PLAYER_PROP_MARKETS)
                )
            st.session_state['odds_props_game_id'] = game['id']
            st.session_state['odds_props_data'] = props_data
            st.session_state['odds_props_err'] = props_err

        props_err = st.session_state.get('odds_props_err')
        props_data = st.session_state.get('odds_props_data')

        if props_err:
            st.warning(f"Player props unavailable: {props_err}")
        elif props_data:
            props_long = _build_props_long_table(props_data)
            if not props_long.empty:
                markets_found = sorted(props_long['Market'].unique().tolist())
                books_found = sorted(props_long['Book'].unique().tolist())
                st.caption(f"Markets posted for this game: {', '.join(markets_found)} — across {len(books_found)} book(s): {', '.join(books_found)}")
                market_filter = st.multiselect("Filter by market", markets_found, default=[], key="odds_market_filter")
                filtered_long = props_long[props_long['Market'].isin(market_filter)] if market_filter else props_long

                comparison_df = _build_props_comparison_table(filtered_long)
                st.markdown("**Cross-book comparison** — one row per bet, one column per bookmaker (odds shown as `price (line)`; click a column header to sort)")
                st.dataframe(
                    style_plain_dataframe(comparison_df.set_index('Player')),
                    width="stretch", height=df_auto_height(min(len(comparison_df), 30))
                )
            else:
                st.info("No player props posted for this game yet by any tracked bookmaker.")
