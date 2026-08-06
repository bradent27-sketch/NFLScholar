"""
Draft HQ: the live draft room, big board, and mock draft simulator.

WHY THIS REPLACES THE OLD VORP SHEET AS THE DRAFT SURFACE: the previous
tab's board was built from last season's per-game pace extrapolated to 17
games. That's a backward-looking stat summary, not a projection - it can't
know about an offseason trade, a new coordinator, a rookie who hasn't
played a snap, or a torn ACL in August, all of which are the things that
actually move draft value. It also had no ADP, so it could tell you who was
good but never who would still be there at your next pick, which is the
question you're actually answering on the clock.

LAYOUT REASONING: the sub-tabs are ordered by how urgently you need them
with a clock running. Draft Room first (recommendations, your roster,
what's left), Big Board second (the full sortable table for when you want
to look something up yourself), Mock Draft third (preparation, not draft
night), News last. League Settings sits in a collapsed expander above all
of it - it's configured once and then never touched again during a draft,
so it should not occupy the screen you're staring at while on the clock.
"""
import numpy as np
import pandas as pd
import streamlit as st

from config import AVAILABLE_SEASONS_WITH_UPCOMING
from data.draft_sources import (
    ECR_BOARDS, load_ecr_raw, build_ecr_board, fetch_adp, fetch_injury_report,
    fetch_player_news, load_dynasty_values,
)
from data.draft_board import (
    DEFAULT_SCORING, DEFAULT_ROSTER, build_draft_board, recommend_picks,
    roster_needs, snake_pick_numbers, next_pick_for, DRAFTABLE_POSITIONS,
)
from data.draft_sim import (
    prepare_sim_pool, init_draft_state, run_until_user_pick, team_on_clock,
    current_round, record_pick, available_players, grade_draft, autopick_for_user,
    run_many_drafts, pick_slot_comparison, optimal_lineup_points,
)
from data.loaders import fetch_sleeper_draft_picks
from data.transforms import parse_pasted_draft_picks, match_names_to_board
from data.utils import calculate_percentile
from ui.styling import style_plain_dataframe, df_auto_height, build_column_help_config
from ui.components import skeleton_loader

DRAFTED_KEY = 'dhq_drafted'          # ordered list of pick dicts (live draft tracker)
SIM_KEY = 'dhq_sim_state'

# Columns shown on the board by default, in the order a drafter reads them:
# who, what, how good, how much better than replacement, what it costs, and
# then - the actual decision inputs - where the market has him and whether
# he'll still be there next time.
BOARD_COLUMNS = [
    'Player', 'Pos', 'Team', 'Tier', 'Proj Pts', 'VORP', 'VONA', 'Auction $',
    'ADP', 'Value vs ADP', 'Avail Next %', 'Ceiling', 'Floor', 'Risk', 'Bye', 'ECR',
]

# The draft room runs the board in a narrower column alongside your roster,
# where all 16 columns get squeezed to unreadable. This is the subset that
# actually drives an on-the-clock decision; the full set stays available one
# click away on the Big Board sub-tab, which is full width.
ROOM_COLUMNS = [
    'Player', 'Pos', 'Team', 'Tier', 'Proj Pts', 'VORP', 'VONA',
    'ADP', 'Value vs ADP', 'Avail Next %', 'Bye',
]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _drafted_list():
    if DRAFTED_KEY not in st.session_state:
        st.session_state[DRAFTED_KEY] = []
    return st.session_state[DRAFTED_KEY]


def _drafted_names():
    return {p['Player'] for p in _drafted_list()}


def _my_roster():
    return [p for p in _drafted_list() if p.get('mine')]


def _sync_legacy_drafted_set():
    """
    Keep the app-wide st.session_state['drafted_players'] set in step with
    this tab's richer pick log.

    Player Search and Rookie Watch already filter drafted players out using
    that plain set (via ui.components.get_drafted_players_clean_keys), and
    they should keep working off THIS tab's tracker now that it's the one
    being used during a draft - otherwise a player drafted here still shows
    up as available everywhere else in the app.
    """
    st.session_state['drafted_players'] = _drafted_names()


def _record_pick(row, mine):
    _drafted_list().append({
        'Player': row['Player'], 'Pos': str(row['Pos']).upper(), 'Team': row.get('Team'),
        'Bye': row.get('Bye'), 'Proj Pts': row.get('Proj Pts'), 'VORP': row.get('VORP'),
        'ADP': row.get('ADP'), 'Tier': row.get('Tier'), 'mine': bool(mine),
    })
    _sync_legacy_drafted_set()


def _undo_last():
    lst = _drafted_list()
    if lst:
        lst.pop()
        _sync_legacy_drafted_set()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _league_settings_ui():
    """League configuration. Returns a settings dict the whole engine keys off."""
    # Collapsed by default, matching the convention the VORP tab already
    # set: during a draft the board is what you stare at, and a full wall of
    # scoring inputs above it pushes the actual decision surface off-screen.
    # Settings get configured once before the draft, then never touched.
    with st.expander("⚙️ League Settings & Data Sources", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**League**")
            num_teams = st.number_input("Teams", 4, 32, 12, key="dhq_teams")
            draft_type = st.selectbox("Draft type", ["Snake", "Linear", "Auction"], key="dhq_draft_type")
            my_slot = st.number_input("Your draft slot", 1, int(num_teams), min(5, int(num_teams)), key="dhq_slot")
            auction_budget = st.number_input("Auction budget ($)", 1, 1000, 200, key="dhq_budget")
        with c2:
            st.markdown("**Starting lineup**")
            qb = st.number_input("QB", 0, 3, 1, key="dhq_qb")
            rb = st.number_input("RB", 0, 5, 2, key="dhq_rb")
            wr = st.number_input("WR", 0, 6, 2, key="dhq_wr")
            te = st.number_input("TE", 0, 3, 1, key="dhq_te")
        with c3:
            st.markdown("**Flex & bench**")
            flex = st.number_input("FLEX (RB/WR/TE)", 0, 4, 1, key="dhq_flex")
            superflex = st.number_input("SUPERFLEX (QB too)", 0, 2, 0, key="dhq_superflex")
            k = st.number_input("K", 0, 2, 1, key="dhq_k")
            dst = st.number_input("DST", 0, 2, 1, key="dhq_dst")
            bench = st.number_input("Bench spots", 0, 20, 6, key="dhq_bench")
        with c4:
            st.markdown("**Scoring**")
            ppr = st.select_slider("PPR", options=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5], value=1.0, key="dhq_ppr")
            te_prem = st.select_slider("TE premium (extra per rec)", options=[0.0, 0.25, 0.5, 0.75, 1.0], value=0.0, key="dhq_te_prem")
            pass_td = st.number_input("Pass TD", 0, 10, 4, key="dhq_pass_td")
            pass_yd = st.number_input("Pts/pass yd", 0.0, 0.2, 0.04, step=0.01, format="%.2f", key="dhq_pass_yd")
            ppc = st.number_input("Pts/carry", 0.0, 1.0, 0.0, step=0.05, key="dhq_ppc")

        st.markdown("---")
        st.markdown("**Per-game yardage bonuses**")
        st.caption(
            "Points awarded in any single game clearing each threshold — leave at 0 if your "
            "league doesn't use them. These are scored week by week, not off season totals, so "
            "a back with eight 100-yard games is correctly worth more than one with the same "
            "yardage spread evenly. Turning these on genuinely re-prices the board toward "
            "boom-week players."
        )
        bonus_mode = st.radio(
            "When several thresholds are cleared in one game",
            ["cumulative", "highest"], horizontal=True, key="dhq_bonus_mode",
            help="Cumulative: a 210-yard game pays the 100, 150 and 200 bonuses (Sleeper/ESPN "
                 "default). Highest: it pays only the 200 bonus.",
        )
        bcols = st.columns(3)
        bonuses = {}
        with bcols[0]:
            st.caption("**Rushing yards**")
            for threshold in (100, 150, 200, 250):
                bonuses[f'bonus_rush_{threshold}'] = st.number_input(
                    f"{threshold}+ rush yds", 0.0, 20.0, 0.0, step=0.5, key=f"dhq_br{threshold}")
        with bcols[1]:
            st.caption("**Receiving yards**")
            for threshold in (100, 150, 200, 250):
                bonuses[f'bonus_rec_{threshold}'] = st.number_input(
                    f"{threshold}+ rec yds", 0.0, 20.0, 0.0, step=0.5, key=f"dhq_bc{threshold}")
        with bcols[2]:
            st.caption("**Passing yards**")
            for threshold in (300, 400, 500, 600):
                bonuses[f'bonus_pass_{threshold}'] = st.number_input(
                    f"{threshold}+ pass yds", 0.0, 20.0, 0.0, step=0.5, key=f"dhq_bp{threshold}")

        st.markdown("---")
        c5, c6, c7 = st.columns(3)
        with c5:
            st.markdown("**More scoring**")
            rush_yd = st.number_input("Pts/rush yd", 0.0, 0.5, 0.1, step=0.01, format="%.2f", key="dhq_rush_yd")
            rec_yd = st.number_input("Pts/rec yd", 0.0, 0.5, 0.1, step=0.01, format="%.2f", key="dhq_rec_yd")
            rush_td = st.number_input("Rush TD", 0, 10, 6, key="dhq_rush_td")
            rec_td = st.number_input("Rec TD", 0, 10, 6, key="dhq_rec_td")
            pass_int = st.number_input("INT thrown", -6, 0, -2, key="dhq_int")
            fumble = st.number_input("Fumble lost", -6, 0, -2, key="dhq_fum")
        with c6:
            st.markdown("**Board & market**")
            board_format = st.selectbox("Ranking board", list(ECR_BOARDS.keys()), key="dhq_board_fmt")
            adp_source = st.selectbox("ADP source", ["Auto", "Fantasy Football Calculator", "Sleeper"], key="dhq_adp_src")
            adp_year = st.selectbox("ADP season", AVAILABLE_SEASONS_WITH_UPCOMING, index=0, key="dhq_adp_year")
            adp_upload = st.file_uploader("Or upload ADP CSV", type=["csv"], key="dhq_adp_upload")
        with c7:
            st.markdown("**Model**")
            st.caption(
                "Projection uncertainty scales how widely a player's plausible finishes are "
                "spread around his consensus rank. Higher = flatter board (stars regress toward "
                "the pack, deep sleepers gain). See the Ceiling/Floor columns move with it."
            )
            uncertainty = st.slider("Projection uncertainty", 0.5, 2.0, 1.0, 0.1, key="dhq_uncertainty")
            tiers = st.slider("Max tiers per position", 3, 12, 8, key="dhq_tiers")
            baseline_season = st.selectbox(
                "Value-curve baseline through", AVAILABLE_SEASONS_WITH_UPCOMING[1:], index=0,
                key="dhq_curve_season",
                help="Last completed season used to build the historical points-by-positional-finish curves.",
            )

    scoring = dict(DEFAULT_SCORING)
    scoring.update({
        'rec': float(ppr), 'te_premium': float(te_prem), 'pass_td': float(pass_td),
        'pass_yd': float(pass_yd), 'rush_att': float(ppc), 'rush_yd': float(rush_yd),
        'rec_yd': float(rec_yd), 'rush_td': float(rush_td), 'rec_td': float(rec_td),
        'pass_int': float(pass_int), 'fumble_lost': float(fumble),
        'bonus_mode': bonus_mode,
    })
    scoring.update({k: float(v) for k, v in bonuses.items()})
    roster = dict(DEFAULT_ROSTER)
    roster.update({'QB': int(qb), 'RB': int(rb), 'WR': int(wr), 'TE': int(te), 'K': int(k),
                   'DST': int(dst), 'FLEX': int(flex), 'SUPERFLEX': int(superflex), 'BENCH': int(bench)})

    return {
        'num_teams': int(num_teams), 'roster': roster, 'scoring': scoring,
        'auction_budget': float(auction_budget), 'draft_type': draft_type,
        'my_slot': int(my_slot), 'board_format': board_format,
        'adp_source': adp_source, 'adp_year': int(adp_year), 'adp_upload': adp_upload,
        'uncertainty': float(uncertainty), 'tiers': int(tiers),
        'baseline_season': int(baseline_season),
    }


# ---------------------------------------------------------------------------
# Board assembly
# ---------------------------------------------------------------------------

def _board_cache_key(settings, next_pick, adp_meta):
    """
    A fully hashable summary of everything that changes the board.

    Built explicitly rather than by hashing the settings dict, because that
    dict carries Streamlit's UploadedFile object for the ADP upload, which
    st.cache_data cannot hash - it would raise on every rerun the moment a
    file was attached. The uploaded ADP still participates in the key via
    adp_meta's source label plus the row count of the frame it produced, so
    swapping one upload for another does invalidate the cache.
    """
    scoring = settings['scoring']
    roster = settings['roster']
    return (
        settings['num_teams'], settings['board_format'], settings['draft_type'],
        settings['auction_budget'], next_pick, settings['uncertainty'],
        settings['tiers'], settings['baseline_season'],
        # str() rather than float() - scoring now carries 'bonus_mode',
        # which is a string, and float()-ing every value would raise the
        # moment the settings panel is opened.
        tuple(sorted((k, str(v)) for k, v in scoring.items())),
        tuple(sorted((k, int(v)) for k, v in roster.items())),
        str(adp_meta.get('source')), adp_meta.get('teams'), adp_meta.get('rows'),
    )


@st.cache_data(show_spinner=False)
def _cached_board(_ecr_board, _adp_df, _settings, cache_key):
    """
    Cache the assembled board on the SETTINGS, not on the DataFrames.

    Same reasoning as data.transforms.apply_scoring_and_percentiles: the
    underscore-prefixed frames are excluded from the hash because hashing a
    700-row board (and re-hashing it on every widget interaction anywhere in
    the app) costs more than the transform it's protecting. cache_key fully
    determines the output for a given source snapshot, which is what makes
    that safe here.
    """
    return build_draft_board(
        _ecr_board, _settings, adp_df=_adp_df, next_pick=cache_key[4],
        tiers_per_position=_settings['tiers'], rank_sd_scale=_settings['uncertainty'],
        latest_season=_settings['baseline_season'],
    )


def _load_board(settings, next_pick):
    """Fetch sources and assemble the board, reporting what did and didn't load."""
    status = {}
    ecr_raw, ecr_err = load_ecr_raw()
    status['ecr'] = ecr_err
    if ecr_raw is None or ecr_raw.empty:
        return pd.DataFrame(), {}, pd.DataFrame(), {}, status

    ecr_board = build_ecr_board(ecr_raw, settings['board_format'])
    is_superflex = int(settings['roster'].get('SUPERFLEX', 0)) > 0
    scoring_label = ('Full PPR' if settings['scoring']['rec'] >= 0.75
                     else 'Half-PPR' if settings['scoring']['rec'] >= 0.25 else 'Standard')
    adp_df, adp_meta = fetch_adp(scoring_label, settings['num_teams'], is_superflex,
                                 settings['adp_year'], source=settings['adp_source'],
                                 uploaded=settings.get('adp_upload'))
    adp_meta = dict(adp_meta or {})
    adp_meta['rows'] = int(len(adp_df))
    status['adp'] = adp_meta

    board, meta = _cached_board(ecr_board, adp_df, settings,
                                _board_cache_key(settings, next_pick, adp_meta))
    return board, meta, adp_df, adp_meta, status


def _render_source_status(status, meta):
    """One honest line about where every number on this board came from."""
    bits = []
    if status.get('ecr'):
        bits.append(f"⚠️ Rankings source unreachable ({status['ecr']}) — board is empty.")
    else:
        bits.append("✅ FantasyPros consensus rankings loaded")
    adp_meta = status.get('adp') or {}
    if adp_meta.get('error'):
        # Truncated: a blocked-network failure returns a full urllib3
        # ProxyError repr, several hundred characters of stack detail that
        # buries the one line that matters ("no ADP, here's what to do").
        reason = str(adp_meta['error']).split('(')[0].strip(' :,)').strip()[:90]
        bits.append(
            f"⚠️ No live ADP ({adp_meta.get('source', 'unknown')}: {reason}). "
            "Value-vs-market, availability and VONA columns are blank — upload an ADP CSV in "
            "League Settings to turn them back on."
        )
    else:
        src = adp_meta.get('source', 'ADP')
        extra = ""
        if adp_meta.get('teams') and adp_meta.get('requested_teams') and adp_meta['teams'] != adp_meta['requested_teams']:
            extra = (f" — showing {adp_meta['teams']}-team ADP for your "
                     f"{adp_meta['requested_teams']}-team league (nearest published size)")
        bits.append(f"✅ ADP from {src}{extra}")
    if meta.get('has_curves'):
        bits.append(f"✅ Value curves built from local history for {', '.join(sorted(meta.get('curves', {})))}")
    st.caption("  •  ".join(bits))


# ---------------------------------------------------------------------------
# Draft room
# ---------------------------------------------------------------------------

def _pick_context(settings):
    """Where the draft is: picks made, your next pick, and the round."""
    picks_made = len(_drafted_list())
    rounds = sum(int(settings['roster'].get(k, 0)) for k in
                 ['QB', 'RB', 'WR', 'TE', 'K', 'DST', 'FLEX', 'SUPERFLEX', 'BENCH'])
    rounds = max(rounds, 1)
    my_picks = snake_pick_numbers(settings['my_slot'], settings['num_teams'], rounds,
                                  draft_type=settings['draft_type'])
    nxt = next_pick_for(my_picks, picks_made)
    return {'picks_made': picks_made, 'rounds': rounds, 'my_picks': my_picks,
            'next_pick': nxt, 'on_clock': picks_made + 1,
            'round': picks_made // max(settings['num_teams'], 1) + 1}


def _render_roster_panel(settings):
    """Your roster as it fills, with the holes and bye stacking called out."""
    mine = _my_roster()
    st.markdown("**Your roster**")
    if not mine:
        st.caption("No picks yet. Use *Draft to my team* on the board below.")
    else:
        rdf = pd.DataFrame(mine)[['Player', 'Pos', 'Team', 'Bye', 'Proj Pts', 'VORP']]
        st.dataframe(rdf, width="stretch", hide_index=True,
                     height=df_auto_height(min(len(rdf), 18)))
        starters_pts, _ = optimal_lineup_points(mine, settings)
        st.caption(f"Best starting lineup projects **{starters_pts:.0f} pts**")

    needs = roster_needs(mine, settings)
    open_slots = [f"{n}x {pos}" for pos, n in needs.items() if n > 0]
    if open_slots:
        st.caption("**Still need:** " + ", ".join(open_slots))
    else:
        st.caption("**Starting lineup is full** — everything from here is depth and upside.")

    byes = [p.get('Bye') for p in mine if p.get('Bye') and not pd.isna(p.get('Bye'))]
    if byes:
        counts = pd.Series(byes).value_counts()
        stacked = counts[counts >= 3]
        if not stacked.empty:
            st.warning("Bye week stacking: " + ", ".join(
                f"{int(w)} ({int(c)} players)" for w, c in stacked.items()))


def _render_live_sync(board):
    """Sleeper sync / paste / manual entry for a real draft happening elsewhere."""
    with st.expander(f"🔗 Sync from your live draft ({len(_drafted_list())} picks logged)", expanded=False):
        st.markdown("**Sleeper** (public API, no login needed)")
        c1, c2 = st.columns([3, 1])
        with c1:
            draft_id = st.text_input("Sleeper draft ID", key="dhq_sleeper_id",
                                     help="The number in sleeper.com/draft/nfl/<this part>")
        with c2:
            st.write("")
            st.write("")
            sync = st.button("Sync picks", key="dhq_sync_btn")
        if sync and draft_id.strip():
            picks_data, err = fetch_sleeper_draft_picks(draft_id.strip())
            if err:
                st.error(f"Couldn't sync: {err}")
            else:
                _apply_synced_picks(picks_data or [], board)

        st.markdown("**Or paste picks** (Yahoo/ESPN draft results, or one name per line)")
        pasted = st.text_area("Paste here", key="dhq_paste", height=110,
                              placeholder="1. Ja'Marr Chase (WR - CIN)\n2. Bijan Robinson (RB - ATL)")
        if st.button("Parse & add", key="dhq_parse_btn") and pasted.strip():
            candidates = parse_pasted_draft_picks(pasted)
            matched, unmatched = match_names_to_board(candidates, board['Player'].tolist())
            _add_names_as_picks(matched, board, mine=False)
            st.success(f"Added {len(matched)} of {len(candidates)} parsed lines.")
            if unmatched:
                st.caption("Couldn't match: " + ", ".join(unmatched[:12]))
            st.rerun()

        if _drafted_list():
            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button("↩ Undo last pick", key="dhq_undo"):
                    _undo_last()
                    st.rerun()
            with cc2:
                if st.button("🗑 Reset draft", key="dhq_reset"):
                    st.session_state[DRAFTED_KEY] = []
                    _sync_legacy_drafted_set()
                    st.rerun()


def _apply_synced_picks(picks_data, board):
    """
    Fold a Sleeper pick feed into the tracker, keeping YOUR picks distinct.

    Sleeper reports which roster/slot made each pick, so picks made from your
    own draft slot are marked as yours automatically - which is what makes
    the roster panel and the recommendations correct without you having to
    re-enter your own picks by hand while the clock runs.
    """
    my_slot = st.session_state.get('dhq_slot', 1)
    names, mine_flags = [], []
    for p in picks_data:
        meta = p.get('metadata', {}) or {}
        full = f"{meta.get('first_name', '')} {meta.get('last_name', '')}".strip()
        if not full:
            continue
        names.append(full)
        mine_flags.append(p.get('draft_slot') == my_slot)

    already = _drafted_names()
    matched, unmatched = match_names_to_board(names, board['Player'].tolist())
    matched_set = set(matched)
    added = 0
    for name, is_mine in zip(names, mine_flags):
        hit = next((m for m in matched_set if m.lower().replace('.', '') in name.lower().replace('.', '')
                    or name.lower().replace('.', '') in m.lower().replace('.', '')), None)
        if hit and hit not in already:
            row = board[board['Player'] == hit]
            if not row.empty:
                _record_pick(row.iloc[0], mine=is_mine)
                already.add(hit)
                added += 1
    st.success(f"Synced {added} new picks from Sleeper.")
    if unmatched:
        st.caption("Couldn't match: " + ", ".join(unmatched[:12]))
    st.rerun()


def _add_names_as_picks(names, board, mine):
    already = _drafted_names()
    for name in names:
        if name in already:
            continue
        row = board[board['Player'] == name]
        if not row.empty:
            _record_pick(row.iloc[0], mine=mine)
            already.add(name)


def _render_recommendations(board, settings, ctx):
    """The 'who should I take' panel - the reason to open this tab at all."""
    available = board[~board['Player'].isin(_drafted_names())]
    recs = recommend_picks(available, _my_roster(), settings, next_pick=ctx['next_pick'], top_n=6)
    if recs.empty:
        st.info("No recommendations available — the board didn't load any valued players.")
        return

    st.markdown("<div class='custom-section-header'>BEST PICKS FOR YOU RIGHT NOW</div>", unsafe_allow_html=True)
    st.caption(
        f"Ranked by fit with your roster, not raw value. Assumes your next pick is #{ctx['next_pick']} "
        f"(slot {settings['my_slot']} of {settings['num_teams']}, {settings['draft_type'].lower()})."
        if ctx['next_pick'] else "Ranked by value — your picks are all used up."
    )
    show = recs[['Player', 'Pos', 'Team', 'Tier', 'Proj Pts', 'VORP', 'VONA', 'ADP',
                 'Avail Next %', 'Fit Score', 'Why']]
    st.dataframe(
        show, width="stretch", hide_index=True, height=df_auto_height(len(show)),
        column_config={
            'Why': st.column_config.TextColumn("Why", width="large"),
            'Avail Next %': st.column_config.NumberColumn("Avail Next %", format="%d%%",
                help="Chance this player is still on the board at your next pick"),
            'VONA': st.column_config.NumberColumn("VONA",
                help="Value Over Next Available: points you lose by passing on him now"),
        },
    )


def _render_board_table(board, settings, ctx, key_prefix, columns=None):
    """The board itself, with drafted players removed and draft actions attached."""
    drafted = _drafted_names()
    available = board[~board['Player'].isin(drafted)].copy()

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        positions = st.multiselect("Positions", DRAFTABLE_POSITIONS, default=[],
                                   key=f"{key_prefix}_pos")
    with c2:
        sort_by = st.selectbox("Sort by", ['VORP', 'VONA', 'Proj Pts', 'Value vs ADP', 'ADP', 'ECR', 'Auction $', 'Ceiling'],
                               key=f"{key_prefix}_sort")
    with c3:
        limit = st.number_input("Rows", 10, 400, 60, step=10, key=f"{key_prefix}_limit")

    if positions:
        available = available[available['Pos'].astype(str).str.upper().isin(positions)]
    ascending = sort_by in ('ADP', 'ECR')
    if sort_by in available.columns:
        available = available.sort_values(sort_by, ascending=ascending, na_position='last')
    view = available.head(int(limit))

    cols = [c for c in (columns or BOARD_COLUMNS) if c in view.columns]
    display = view[cols].set_index('Player')

    pct_cols = {}
    for c in ('VORP', 'VONA', 'Proj Pts'):
        if c in display.columns and display[c].notna().any():
            pct_cols[c] = calculate_percentile(display.reset_index(), c)
    diverging = {}
    if 'Value vs ADP' in display.columns and display['Value vs ADP'].notna().any():
        max_abs = display['Value vs ADP'].abs().max()
        if max_abs and max_abs > 0:
            diverging['Value vs ADP'] = max_abs

    column_config = build_column_help_config(display, pinned_cols=['Pos', 'Team'])
    if 'Avail Next %' in display.columns:
        column_config['Avail Next %'] = st.column_config.NumberColumn(
            "Avail Next %", format="%d%%",
            help=f"Chance he lasts to your next pick (#{ctx['next_pick']})" if ctx['next_pick'] else "No next pick",
        )
    if 'Auction $' in display.columns:
        column_config['Auction $'] = st.column_config.NumberColumn("Auction $", format="$%d")
    if 'Risk' in display.columns:
        column_config['Risk'] = st.column_config.NumberColumn(
            "Risk", format="%d%%",
            help="Width of the ceiling-to-floor band as a share of the projection")

    # The selected player is read back off the widget's return value on every
    # run, rather than being latched into session_state by an on_select
    # callback.
    #
    # This is what makes drafting several players in a row work. A callback
    # only fires when the selection CHANGES, and Streamlit keeps the grid's
    # selected ROW INDEX across reruns - so after drafting the top player and
    # removing him from the board, row 0 is still "selected" but now points at
    # a different player. A latched value would go stale, and clicking that
    # same top row again would change nothing and fire nothing, leaving the
    # board unresponsive exactly when you're picking fastest. Re-reading the
    # index each run instead means the selection simply follows down the board
    # as players come off it.
    event = st.dataframe(
        style_plain_dataframe(display, numeric_pct_cols=pct_cols, diverging_cols=diverging),
        width="stretch", height=df_auto_height(min(len(display), 26)),
        on_select="rerun", selection_mode="single-row", key=f"{key_prefix}_table",
        column_config=column_config,
    )
    rows = []
    try:
        rows = list(event.selection.rows)
    except Exception:
        rows = []
    selected = display.index[rows[0]] if rows and rows[0] < len(display) else None

    if selected and selected in set(board['Player']):
        row = board[board['Player'] == selected].iloc[0]
        st.markdown(f"**Selected:** {selected} ({row['Pos']} — {row.get('Team')})")
        b1, b2, b3 = st.columns([1, 1, 3])
        with b1:
            if st.button("➕ Draft to my team", key=f"{key_prefix}_draft_mine", type="primary"):
                _record_pick(row, mine=True)
                st.rerun()
        with b2:
            if st.button("❌ Taken by someone else", key=f"{key_prefix}_draft_other"):
                _record_pick(row, mine=False)
                st.rerun()
    else:
        st.caption("Click a row to draft that player to your team or mark him as taken.")


def _render_positional_scarcity(board, settings):
    """
    How many players above replacement are left at each position.

    The single most useful "what should I be worried about" readout on a
    draft board: it turns "should I take a TE?" into a number. Two startable
    tight ends left and eleven receivers is a completely different situation
    from the reverse, and it's invisible from a ranked list.
    """
    drafted = _drafted_names()
    available = board[~board['Player'].isin(drafted)]
    rows = []
    for pos in DRAFTABLE_POSITIONS:
        pos_rows = available[available['Pos'].astype(str).str.upper() == pos]
        if pos_rows.empty:
            continue
        above = pos_rows[pos_rows['VORP'] > 0] if 'VORP' in pos_rows.columns else pos_rows
        top_tier = pos_rows['Tier'].min() if 'Tier' in pos_rows.columns and pos_rows['Tier'].notna().any() else np.nan
        in_top_tier = int((pos_rows['Tier'] == top_tier).sum()) if pd.notna(top_tier) else 0
        rows.append({
            'Pos': pos,
            'Above replacement': len(above),
            'Best tier left': int(top_tier) if pd.notna(top_tier) else None,
            'Left in that tier': in_top_tier,
            'Best available': pos_rows.nlargest(1, 'VORP')['Player'].iloc[0] if 'VORP' in pos_rows.columns and pos_rows['VORP'].notna().any() else None,
        })
    if rows:
        st.markdown("**Positional scarcity**")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                     height=df_auto_height(len(rows)))


def _render_draft_room(board, settings, ctx):
    top = st.columns(4)
    top[0].metric("Picks made", ctx['picks_made'])
    top[1].metric("Round", ctx['round'])
    top[2].metric("Your next pick", ctx['next_pick'] if ctx['next_pick'] else "—")
    top[3].metric("Your roster", len(_my_roster()))

    _render_live_sync(board)
    _render_recommendations(board, settings, ctx)

    left, right = st.columns([3, 1])
    with left:
        _render_board_table(board, settings, ctx, key_prefix="dhq_room", columns=ROOM_COLUMNS)
    with right:
        _render_roster_panel(settings)
    # Full width rather than in the sidebar column - it's a 5-column table
    # and the narrow column truncated its last two columns entirely.
    _render_positional_scarcity(board, settings)


# ---------------------------------------------------------------------------
# Mock draft
# ---------------------------------------------------------------------------

def _render_mock(board, settings):
    st.markdown("<div class='custom-section-header'>MOCK DRAFT SIMULATOR</div>", unsafe_allow_html=True)
    st.caption(
        "Runs your league's real settings against opponents who draft from ADP with noise, so "
        "positional runs and reaches happen the way they do in a real room. Bots respect roster "
        "legality — they fill their lineups and don't take three kickers — which is what stops "
        "studs from unrealistically falling to you."
    )

    pool, order_col, has_adp = prepare_sim_pool(board)
    if pool.empty:
        st.info("No board loaded to simulate against.")
        return
    if not has_adp:
        st.warning(
            "No ADP loaded, so the simulated opponents are drafting off this board's own value "
            "ranking instead of the market. That's a useful stress test, but it is not a "
            "realistic draft room — treat 'who fell to me' with suspicion until ADP is available."
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rounds = st.number_input("Rounds", 5, 25, min(15, sum(int(settings['roster'].get(k, 0)) for k in
                                 ['QB', 'RB', 'WR', 'TE', 'K', 'DST', 'FLEX', 'SUPERFLEX', 'BENCH']) or 15),
                                 key="dhq_sim_rounds")
    with c2:
        slot = st.number_input("Your slot", 1, settings['num_teams'], settings['my_slot'], key="dhq_sim_slot")
    with c3:
        reach = st.slider("Room chaos", 1.0, 8.0, 3.0, 0.5, key="dhq_sim_reach",
                          help="How far opponents stray from ADP. 1 = chalk, 8 = wild.")
    with c4:
        n_sims = st.number_input("Simulations", 1, 100, 20, key="dhq_sim_n")

    mode = st.radio("Mode", ["Run many drafts (summary)", "Draft interactively", "Compare every draft slot"],
                    horizontal=True, key="dhq_sim_mode")

    if mode == "Run many drafts (summary)":
        if st.button("▶ Run simulations", key="dhq_sim_run", type="primary"):
            with st.spinner(f"Running {int(n_sims)} drafts..."):
                summary, outcomes = run_many_drafts(board, settings, int(slot), int(rounds),
                                                    n_sims=int(n_sims), reach_window=float(reach))
            if outcomes.empty:
                st.info("Simulation produced no results.")
                return
            m1, m2, m3 = st.columns(3)
            m1.metric("Median starters projection", f"{outcomes['Starters Proj'].median():.0f}")
            m2.metric("Range", f"{outcomes['Starters Proj'].min():.0f} – {outcomes['Starters Proj'].max():.0f}")
            m3.metric("Avg league finish", f"{outcomes['League Rank'].mean():.1f}")
            st.markdown("**Who you ended up with, by round**")
            st.caption("Read this as tendencies, not predictions — a player showing up in 60% of "
                       "drafts is one your slot reliably gets; 10% is a coin flip you shouldn't plan around.")
            st.dataframe(summary[['round', 'Player', 'Pos', '% of drafts']], width="stretch",
                         hide_index=True, height=df_auto_height(min(len(summary), 30)))

    elif mode == "Draft interactively":
        _render_interactive_mock(board, settings, pool, order_col, int(slot), int(rounds), float(reach))

    else:
        st.caption("Runs every draft slot to see which ones YOUR settings favour — the answer "
                   "moves with league size, superflex and TE premium, so the usual received wisdom "
                   "about 'the turn' often doesn't apply.")
        sims_each = st.number_input("Sims per slot", 1, 40, 8, key="dhq_slot_sims")
        if st.button("▶ Compare slots", key="dhq_slot_run"):
            with st.spinner(f"Running {settings['num_teams']} slots x {int(sims_each)} drafts..."):
                cmp_df = pick_slot_comparison(board, settings, int(rounds),
                                              n_sims=int(sims_each), reach_window=float(reach))
            if cmp_df.empty:
                st.info("No results.")
            else:
                st.dataframe(cmp_df, width="stretch", hide_index=True,
                             height=df_auto_height(len(cmp_df)))


def _render_interactive_mock(board, settings, pool, order_col, slot, rounds, reach):
    """A mock you actually pick in, so you can test a strategy rather than watch one."""
    state = st.session_state.get(SIM_KEY)
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("🎲 New mock draft", key="dhq_mock_new", type="primary"):
            state = init_draft_state(settings, slot, rounds)
            run_until_user_pick(state, settings, pool, order_col, reach_window=reach)
            st.session_state[SIM_KEY] = state
            st.rerun()
    if state is None:
        st.caption("Start a new mock to draft against the simulated room.")
        return

    with c2:
        st.caption(f"Pick {state['pick_no']} of {state['num_teams'] * state['rounds']} · "
                   f"Round {current_round(state)} · Team on the clock: {team_on_clock(state)}")

    if state['complete']:
        st.success("Mock complete.")
        grades = grade_draft(state, settings)
        st.dataframe(grades[['Rank', 'Team', 'Starters Proj', 'Bench Proj']], width="stretch",
                     hide_index=True, height=df_auto_height(len(grades)))
        mine = state['rosters'][state['my_slot']]
        st.markdown("**Your roster**")
        st.dataframe(pd.DataFrame(mine)[['round', 'pick', 'Player', 'Pos', 'Proj Pts']],
                     width="stretch", hide_index=True, height=df_auto_height(len(mine)))
        return

    avail = available_players(state, pool)
    recs = recommend_picks(avail, state['rosters'][state['my_slot']], settings,
                           next_pick=state['pick_no'] + state['num_teams'], top_n=8)
    st.markdown("**You're on the clock**")
    if not recs.empty:
        choice = st.selectbox("Pick a player", recs['Player'].tolist() + ["— someone else —"],
                              key="dhq_mock_choice")
        st.dataframe(recs[['Player', 'Pos', 'Tier', 'Proj Pts', 'VORP', 'ADP', 'Why']],
                     width="stretch", hide_index=True, height=df_auto_height(len(recs)))
        if choice == "— someone else —":
            choice = st.selectbox("Full board", avail['Player'].head(200).tolist(), key="dhq_mock_choice_all")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("✅ Make pick", key="dhq_mock_pick"):
                row = avail[avail['Player'] == choice]
                if not row.empty:
                    record_pick(state, state['my_slot'], row.iloc[0])
                    run_until_user_pick(state, settings, pool, order_col, reach_window=reach)
                    st.session_state[SIM_KEY] = state
                    st.rerun()
        with b2:
            if st.button("⏭ Auto-pick for me", key="dhq_mock_auto"):
                row = autopick_for_user(state, settings, pool)
                if row is not None:
                    record_pick(state, state['my_slot'], row)
                    run_until_user_pick(state, settings, pool, order_col, reach_window=reach)
                    st.session_state[SIM_KEY] = state
                    st.rerun()

    recent = state['picks'][-min(len(state['picks']), 12):]
    if recent:
        st.markdown("**Since your last pick**")
        st.dataframe(pd.DataFrame(recent)[['pick', 'round', 'team', 'Player', 'Pos']],
                     width="stretch", hide_index=True, height=df_auto_height(len(recent)))


# ---------------------------------------------------------------------------
# News & injuries
# ---------------------------------------------------------------------------

def _render_news(board, settings):
    st.markdown("<div class='custom-section-header'>NEWS &amp; INJURY REPORT</div>", unsafe_allow_html=True)
    st.caption(
        "Both feeds are live and both are allowed to fail — if a source is unreachable this "
        "section degrades to whatever did load rather than breaking the draft room."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Injury designations**")
        inj, inj_err = fetch_injury_report(settings['adp_year'])
        if inj_err or inj.empty:
            st.caption(f"No injury data available ({inj_err or 'empty'}).")
        else:
            on_board = inj[inj['Player'].isin(set(board['Player']))] if not board.empty else inj
            merged = on_board.merge(board[['Player', 'Pos', 'Team', 'ECR']], on='Player', how='left')
            merged = merged.sort_values('ECR', na_position='last')
            st.dataframe(merged[['Player', 'Pos', 'Team', 'Injury Status', 'Injury Detail', 'ECR']],
                         width="stretch", hide_index=True,
                         height=df_auto_height(min(len(merged), 20)))
    with c2:
        st.markdown("**Latest headlines**")
        news, news_err = fetch_player_news()
        if news_err or news.empty:
            st.caption(f"No news feed available ({news_err or 'empty'}).")
        else:
            for _, row in news.head(15).iterrows():
                link = row.get('Link')
                headline = row['Headline']
                st.markdown(f"- [{headline}]({link})" if link else f"- {headline}")

    with st.expander("Dynasty / keeper trade values", expanded=False):
        st.caption(
            "Redraft value and dynasty value are different questions — a 29-year-old back and a "
            "23-year-old with identical projections are the same redraft pick and very different "
            "keeper assets. Shown separately rather than blended into the board so a one-year "
            "league's rankings never quietly get contaminated by age."
        )
        vals, verr = load_dynasty_values()
        if verr or vals.empty:
            st.caption(f"Dynasty values unavailable ({verr or 'empty'}).")
        else:
            show = vals[['player', 'pos', 'team', 'age', 'ecr_1qb', 'ecr_2qb', 'value_1qb', 'value_2qb']]
            show = show.rename(columns={'player': 'Player', 'pos': 'Pos', 'team': 'Team', 'age': 'Age'})
            st.dataframe(show.head(120), width="stretch", hide_index=True, height=df_auto_height(24))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def render():
    st.markdown("<div class='custom-section-header'>DRAFT HQ</div>", unsafe_allow_html=True)

    settings = _league_settings_ui()
    ctx = _pick_context(settings)

    with skeleton_loader("table", n_rows=12, n_cols=8):
        board, meta, adp_df, adp_meta, status = _load_board(settings, ctx['next_pick'])

    _render_source_status(status, meta)

    if board.empty:
        st.error(
            "Couldn't build a draft board — the consensus rankings source was unreachable. "
            "Everything here is live-fetched, so this usually means no network access rather "
            "than anything wrong with the app. The VORP Draft Sheet tab still works fully "
            "offline from local data."
        )
        return

    room, big_board, mock, news = st.tabs(
        ["🎯 Draft Room", "📋 Big Board", "🎲 Mock Draft", "📰 News & Injuries"])
    with room:
        _render_draft_room(board, settings, ctx)
    with big_board:
        st.caption(
            "The full board, every column. Proj Pts are season totals under YOUR scoring, built "
            "from what players who actually finished at each positional rank have scored in "
            "recent seasons — so changing PPR or TE premium genuinely re-prices the board rather "
            "than just re-sorting it."
        )
        _render_board_table(board, settings, ctx, key_prefix="dhq_big")
    with mock:
        _render_mock(board, settings)
    with news:
        _render_news(board, settings)
