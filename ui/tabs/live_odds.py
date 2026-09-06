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
from data.odds_sources import parse_prizepicks_payload, parse_draftkings_payloads
from ui.components import skeleton_loader, import_hint


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


def _lines_consensus(lines_df, home, away):
    """
    One-line market consensus across every book that posted this game -
    shown above the per-book table so the headline number is readable
    without scanning the grid. Medians, not means: one book with a stale
    line shouldn't drag the shown spread.
    """
    def med(col):
        if col not in lines_df.columns:
            return None
        s = pd.to_numeric(lines_df[col], errors='coerce').dropna()
        return float(s.median()) if not s.empty else None

    hs, tot = med('Home Spread'), med('Total (O/U)')
    parts = []
    if hs is not None:
        if hs < 0:
            parts.append(f"{home} {hs:.1f}")
        elif hs > 0:
            parts.append(f"{away} {-hs:.1f}")
        else:
            parts.append("Pick'em")
    if tot is not None:
        parts.append(f"O/U {tot:g}")
    parts.append(f"median of {len(lines_df)} book{'s' if len(lines_df) != 1 else ''}")
    return " · ".join(parts)


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


def _render_odds_api_devig(props_data, player_q='', market_filter=None):
    """The sharp books (FanDuel / DraftKings / Pinnacle / BetMGM / ...) this
    fetch just returned, de-vigged the same way the free weekly board is:
    each book's Over/Under prices -> a p_over -> an implied market mean, then
    a reliability-weighted consensus across books.

    This reuses the fetch that already happened for the cross-book table
    above - it does NOT spend another credit. It only appears once player
    props have been loaded for the game.
    """
    from data.odds_sources import parse_odds_api_props
    from data.odds_projections import market_book_stat_lines, market_stat_lines

    props, err = parse_odds_api_props(props_data)
    if props is None or props.empty:
        return
    props = props[props['scorable'].astype(bool)]
    if props.empty:
        st.caption("De-vigged view: none of these markets map to a stat this app scores.")
        return

    book_lines = market_book_stat_lines(props, season_only=False)
    consensus_wide = market_stat_lines(props, season_only=False)
    if book_lines.empty:
        return
    consensus = consensus_wide.iloc[0].to_dict() if not consensus_wide.empty else {}

    bl = book_lines.copy()
    bl['book'] = bl['provider'].astype(str).str.split(' (', regex=False).str[0].str.split(': ').str[-1]
    if player_q:
        bl = bl[bl['player'].str.contains(player_q, case=False, na=False)]
    if market_filter:
        want = {m.lower().replace(' ', '_') for m in market_filter}
        bl = bl[bl['market'].str.lower().isin(want) | bl['market_raw'].str.lower().isin(want)] \
            if 'market_raw' in bl.columns else bl[bl['market'].str.lower().isin(want)]
    if bl.empty:
        return

    bl['line'] = pd.to_numeric(bl['line'], errors='coerce')
    bl['implied_mean'] = pd.to_numeric(bl.get('implied_mean'), errors='coerce').fillna(bl['line'])

    def _cell(raw, imp):
        if pd.isna(raw):
            return '—'
        if pd.notna(imp) and abs(float(imp) - float(raw)) >= 0.05:
            return f"{float(raw):.2f} → {float(imp):.2f}"
        return f"{float(raw):.2f}"

    rows = []
    for (player, market), chunk in bl.groupby(['player', 'market']):
        row = {'Player': player, 'Market': market.replace('_', ' ').title()}
        for _, r in chunk.iterrows():
            row[r['book']] = _cell(r['line'], r['implied_mean'])
        avg = consensus.get(market)
        row['Market proj'] = f"{float(avg):.2f}" if avg is not None and pd.notna(avg) else '—'
        rows.append(row)
    table = pd.DataFrame(rows).fillna('—')
    front = ['Player', 'Market']
    book_cols = [c for c in table.columns if c not in front + ['Market proj']]
    table = table[front + sorted(book_cols) + ['Market proj']].sort_values(['Player', 'Market'])

    st.markdown(
        "**De-vigged market projection** — each book cell is its raw posted line, with an arrow to "
        "the mean its Over/Under prices actually imply. **Market proj** is the reliability-weighted "
        "consensus of those de-vigged numbers, the same treatment the free weekly board gets."
    )
    st.dataframe(style_plain_dataframe(table.set_index('Player')),
                 width="stretch", height=df_auto_height(min(len(table), 30)))
    st.caption("Reuses the props fetch above — no extra API credit spent.")


def _fmt_age(stamp):
    if stamp is None:
        return "never"
    delta = datetime.datetime.now(datetime.timezone.utc) - stamp
    hours = delta.total_seconds() / 3600
    if hours < 1:
        return f"{int(delta.total_seconds() // 60)} min ago"
    if hours < 48:
        return f"{int(hours)}h ago"
    return f"{int(hours // 24)}d ago"


def _render_weekly_uploads():
    """
    Drop a weekly board in by hand, per book. Returns True if anything was
    saved, so the caller can rebuild the snapshot instead of serving the one
    that predates the upload.

    PrizePicks is the reason this exists. Cloudflare refuses this app whether
    it asks as a script or through a browser, and the remaining ways past
    that are ones this project won't use - so their weekly board arrives as a
    file or not at all. The file is deliberately separate from the season
    one: their weekly product is a different league, and saving one over the
    other would swap season totals for single-game props, which would parse
    perfectly and be wrong by two orders of magnitude.
    """
    from data.odds_sources import save_book_payload, MULTI_FILE_BOOKS

    saved_any = False
    from data.import_sources import markdown_list

    with st.expander("📥 Load a weekly board from a saved file", expanded=False):
        st.caption(
            "Click a source, save the JSON it returns, drop it here. A saved file "
            "wins over the live fetch for that book and keeps working when the book "
            "refuses us."
        )
        st.markdown(markdown_list('prizepicks_weekly', 'underdog', 'draftkings_weekly'))
        for provider, label in (('PrizePicks Weekly', 'PrizePicks weekly projections JSON'),
                                ('DraftKings Weekly', 'DraftKings weekly boards JSON (one or more)')):
            multi = provider in MULTI_FILE_BOOKS
            upload = st.file_uploader(label, type=["json"], accept_multiple_files=multi,
                                      key=f"weekly_upload_{provider.replace(' ', '_').lower()}")
            if multi and upload:
                payload, err = save_book_payload([f.getvalue() for f in upload], provider)
            elif not multi and upload is not None:
                payload, err = save_book_payload(upload.getvalue(), provider)
            else:
                continue

            if err and payload is None:
                st.error(err)
                continue
            if err:
                st.warning(err)
            parser = (parse_draftkings_payloads if provider.startswith('DraftKings')
                      else parse_prizepicks_payload)
            parsed, perr = parser(payload)
            game = parsed[parsed['period'] == 'game'] if not parsed.empty else parsed
            if perr and game.empty:
                st.warning(f"Saved it, but: {perr}")
            elif game.empty:
                st.warning(
                    "Saved it, but there are no single-game lines in that file. For "
                    "PrizePicks make sure it's `league_id=9` — the season board parses "
                    "fine and is the wrong product for this panel."
                )
            else:
                st.success(f"Saved {len(game)} weekly {provider.split()[0]} lines "
                           f"({game['player'].nunique()} players).")
                saved_any = True
    return saved_any


def _render_weekly_props():
    """
    This week's player props from the books that cost nothing to ask.

    ABOVE the paid section and outside its key check on purpose: these three
    need no key and no quota, so making them wait behind a text box someone
    may never fill in would hide the part of this tab that can be used every
    day of the season.
    """
    from data.odds_weekly import weekly_props, weekly_summary, weekly_consensus, posting_anchor

    st.markdown("#### This week's player props")
    st.caption(
        "PrizePicks, Underdog, DraftKings and Pinnacle — no key, no quota, no request limit. "
        "Pinnacle is the sharpest book here and prices its per-game lines with real "
        "over/under odds, so its lean feeds `Market proj` (the de-vigged column) most. "
        "Books post the coming weekend on Tuesday or Wednesday, so this reloads on its "
        "own the first time you open it after Tuesday morning. Hit refresh whenever you "
        "want the current numbers rather than the ones posted at the start of the week."
    )

    # PrizePicks has always been browser-only, and DraftKings started
    # refusing scripts after a burst of requests. Both serve the same URL to
    # a browser, so there is a fallback - but it needs Playwright installed,
    # and silently doing nothing would look identical to the book being down.
    from data.odds_browser import browser_available
    have_browser, why = browser_available()
    if not have_browser:
        st.warning(
            f"**PrizePicks and DraftKings may refuse a plain request.** {why}",
            icon="🧭",
        )

    force = st.button("🔄 Refresh weekly lines", key="weekly_props_refresh",
                      help="Goes back to all four books now, ignoring the saved snapshot.")
    force = _render_weekly_uploads() or force
    if force:
        for fn in ('fetch_prizepicks_lines', 'fetch_underdog_lines',
                   'fetch_draftkings_weekly_lines', 'dk_weekly_subcategories',
                   'fetch_pinnacle_lines'):
            getattr(__import__('data.odds_sources', fromlist=[fn]), fn).clear()

    with skeleton_loader("table", n_rows=5, n_cols=5):
        props, meta = weekly_props(force=force)

    stamp = meta.get('fetched_at')
    bits = [f"Pulled {_fmt_age(stamp)}"]
    if meta.get('from_network'):
        bits.append("refetched just now" if not meta.get('stale') else "refetch returned nothing")
    else:
        bits.append("from the saved snapshot")
    bits.append(f"slate posted {posting_anchor():%a %d %b}")
    st.caption(" · ".join(bits))

    status = meta.get('status') or {}
    problems = {k: v.get('error') for k, v in status.items() if v.get('error')}
    if props.empty:
        st.info(
            "No weekly player props right now. Outside the season that is the expected "
            "answer — the books post a slate on Tuesday and pull it after the games."
        )
        for book, err in problems.items():
            st.caption(f"**{book}** — {err}")
        return

    summary = weekly_summary(props)
    if not summary.empty:
        sources = {k: v.get('source') for k, v in status.items() if v.get('source')}
        if sources:
            summary = summary.assign(
                Source=summary['Book'].map(lambda b: sources.get(b, '')))
        st.dataframe(summary, width="stretch", hide_index=True)
    for book, err in problems.items():
        st.caption(f"⚠️ **{book}** — {err}")

    consensus = weekly_consensus(props)
    if consensus.empty:
        st.info("Lines came back but none mapped to a stat this app scores.")
        return

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        markets = sorted(consensus['Market'].unique().tolist())
        chosen = st.multiselect("Stat", markets, default=[], key="weekly_props_market")
    with c2:
        teams = sorted(t for t in consensus.get('Team', pd.Series(dtype=str)).dropna().unique().tolist() if t)
        team_pick = st.multiselect("Team", teams, default=[], key="weekly_props_team")
    with c3:
        search = st.text_input("Player", key="weekly_props_player",
                               placeholder="filter by name")

    sort_by = st.radio("Sort by", ["Player", "Team", "Stat", "Widest spread"],
                       horizontal=True, key="weekly_props_sort")

    view = consensus
    if chosen:
        view = view[view['Market'].isin(chosen)]
    if team_pick and 'Team' in view.columns:
        view = view[view['Team'].isin(team_pick)]
    if search:
        view = view[view['Player'].str.contains(search, case=False, na=False)]

    _sort_spec = {
        "Player": (["Player", "Market"], True),
        "Team": (["Team", "Player", "Market"], True),
        "Stat": (["Market", "Player"], True),
        "Widest spread": (["Spread"], False),
    }
    _by, _asc = _sort_spec[sort_by]
    _by = [c for c in _by if c in view.columns]
    if _by:
        view = view.sort_values(_by, ascending=_asc)

    st.markdown(
        "**Consensus board** — one row per player and stat. `Consensus` is the median posted "
        "line across whichever books priced him; `Market proj` is that same median AFTER "
        "de-vigging — where a book gives over/under prices, a leaned line implies a number off "
        "to one side, and `Market proj` carries that shift while `Consensus` does not (they match "
        "for a pure pick'em board). `Books` is how many priced him, `Spread` is the gap between "
        "the highest and lowest posted line. A stat every book agrees on is settled; the wide "
        "ones are where a disagreement is worth reading."
    )
    # NOT set_index('Player'). A player has one row per stat here, so that
    # index is non-unique and pandas' Styler refuses to apply against it -
    # which surfaces as the whole tab failing to render, not as a bad table.
    # Draft HQ can index by player because there it IS one row per player.
    st.dataframe(style_plain_dataframe(view.reset_index(drop=True)),
                 width="stretch", hide_index=True,
                 height=df_auto_height(min(len(view), 25)))
    st.caption(f"{len(view)} of {len(consensus)} player-stat rows.")


def render():
    st.markdown("<div class='custom-section-header'>LIVE NFL ODDS</div>", unsafe_allow_html=True)

    _render_weekly_props()

    st.divider()
    st.markdown("#### More books, via The Odds API")
    st.caption(
        "Everything above is free and uncapped. This section is the metered one — it "
        "adds the sportsbooks the three above don't cover (BetMGM, Caesars, BetRivers "
        "and the rest) and it spends credits from your plan to do it, so it only ever "
        "fetches when you ask."
    )

    # value= only seeds the widget on its very first render this session
    # (session_state takes over after that) - so this pre-fills the box
    # from the saved file on a fresh app launch without fighting Streamlit
    # over who owns the keyed widget's value on later reruns.
    import_hint('odds_api')

    saved_key = load_saved_odds_api_key()
    odds_api_key = st.text_input(
        "The Odds API key", type="password", key="odds_api_key", value=saved_key,
        help="Saved locally in .streamlit/odds_api_key.txt so you don't have to re-enter it every launch. "
             "That file is plain text, not encrypted - delete it (or clear this field) before sharing this project folder with anyone.",
    )
    if odds_api_key and odds_api_key != saved_key:
        save_odds_api_key(odds_api_key)

    if not odds_api_key:
        st.info("Enter an API key above to add these books. Free tier at the-odds-api.com. "
                "The weekly board above works without one.")
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

    # Which books this key ACTUALLY returns, read off the response rather
    # than off a docs page. Coverage varies by plan tier, by region and by
    # how close a game is, so the published list and what a given key sees at
    # a given moment are not the same list - and "which books am I paying
    # for" is otherwise a surprisingly hard question to answer.
    with st.expander("📚 Which sportsbooks does this key cover?", expanded=False):
        from data.odds_sources import odds_api_bookmakers
        books = odds_api_bookmakers(odds_data)
        if books.empty:
            st.info("No bookmakers in the current response.")
        else:
            st.caption(
                f"{len(books)} books across {len(odds_data)} upcoming games. "
                "'events' is how many of those games each book has posted, which is the "
                "honest measure of coverage — a book that appears once is not a book you "
                "can shop against."
            )
            st.dataframe(books, width="stretch", hide_index=True)
            st.caption(
                "**On the DFS books:** PrizePicks and Underdog are pick'em operators rather "
                "than sportsbooks and generally are NOT in this feed. Underdog's season-long "
                "player lines are pulled directly instead — see Draft HQ → Market lines vs "
                "this board. FanDuel and DraftKings ARE here under licence, which is why "
                "neither is scraped anywhere in this app."
            )

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
        st.markdown(
            f"**Game lines** — {_lines_consensus(lines_df, game.get('home_team'), game.get('away_team'))}"
        )
        with st.expander(f"Per-book breakdown — moneyline / spread / total ({len(lines_df)} books, click a header to sort)",
                         expanded=False):
            st.dataframe(style_plain_dataframe(lines_df.set_index('Book')),
                         width="stretch", height=df_auto_height(len(lines_df)))
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
                fc1, fc2, fc3 = st.columns([3, 2, 2])
                with fc1:
                    market_filter = st.multiselect("Filter by market", markets_found, default=[], key="odds_market_filter")
                with fc2:
                    player_q = st.text_input("Player", key="odds_props_player", placeholder="filter by name")
                with fc3:
                    sort_by = st.radio("Sort by", ["Player", "Market"], horizontal=True, key="odds_props_sort")

                filtered_long = props_long
                if market_filter:
                    filtered_long = filtered_long[filtered_long['Market'].isin(market_filter)]
                if player_q:
                    filtered_long = filtered_long[filtered_long['Player'].str.contains(player_q, case=False, na=False)]

                # Team filter is intentionally omitted here: a single-game
                # props fetch spans only this matchup's two teams, and mapping
                # every sportsbook player string back to a roster team is more
                # work than it saves for a two-team board.
                comparison_df = _build_props_comparison_table(filtered_long)
                if comparison_df.empty:
                    st.info("No player props match the current filters.")
                else:
                    secondary = "Market" if sort_by == "Player" else "Player"
                    comparison_df = comparison_df.sort_values(
                        [c for c in (sort_by, secondary) if c in comparison_df.columns])
                    st.markdown("**Cross-book comparison** — one row per bet, one column per bookmaker (odds shown as `price (line)`; click a column header to sort)")
                    st.dataframe(
                        style_plain_dataframe(comparison_df.set_index('Player')),
                        width="stretch", height=df_auto_height(min(len(comparison_df), 30))
                    )

                _render_odds_api_devig(props_data, player_q, market_filter)
            else:
                st.info("No player props posted for this game yet by any tracked bookmaker.")
