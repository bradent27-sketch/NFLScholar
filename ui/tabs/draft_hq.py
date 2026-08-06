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

LAYOUT REASONING: there is ONE board, not a "draft room" and a separate
"big board" - those were the same table with different columns squeezed in,
which just read as two boards that might disagree. Before you start
drafting it is the big board; as picks come in it thins into the live room.
Mock Draft and News follow because they're preparation, not draft night.
League Settings sits in a collapsed expander above all of it: it's
configured once and then never touched again while a clock is running.
"""
import numpy as np
import pandas as pd
import streamlit as st

from config import AVAILABLE_SEASONS_WITH_UPCOMING
from data.draft_sources import (
    ECR_BOARDS, load_ecr_raw, build_ecr_board, fetch_adp, fetch_injury_report,
    fetch_player_news, load_dynasty_values, ecr_age_days, ADP_SOURCE_NOTES,
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
from data.draft_projections import build_projected_board
from data.draft_sos import build_team_sos, attach_sos_to_board, adp_quartiles, WEEK_PRESETS
from data.draft_intel import (
    classify_strategy, pick_intel, outcome_distribution, roster_percentile,
    positional_run_pressure,
)
from data.ffa_import import load_ffa_import, save_ffa_import, merge_ffa_into_board
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
# 'Pos Rk' sits immediately beside Proj Pts on purpose. Projected points are
# not comparable across positions - every starting QB outscores every running
# back - so a points column read on its own makes the board look like it
# rates the QB1 the best player in football. Seeing "QB1" and "RB3" next to
# the number keeps it in the only context where it means anything.
BOARD_COLUMNS = [
    'Player', 'Pos', 'Team', 'Pos Rk', 'Tier', 'Proj Pts', 'VORP', 'VONA',
    'FFA Rank', 'ADP', 'Value vs ADP', 'Avail Next %', 'Ceiling', 'Floor',
    'Risk', 'SOS', 'Bye', 'ECR',
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
            adp_year = st.selectbox("ADP season", AVAILABLE_SEASONS_WITH_UPCOMING, index=0, key="dhq_adp_year")
            adp_source = st.selectbox(
                "ADP source", ["Auto", "FFA import", "Uploaded CSV", "Fantasy Football Calculator"],
                key="dhq_adp_source",
                help="Auto prefers an uploaded CSV, then an FFA import, then Fantasy Football "
                     "Calculator. FFC's ADP comes from mock drafts on its own free site, which "
                     "skews casual — tight ends and quarterbacks slide later there than in real "
                     "drafts.",
            )
            sos_window = st.selectbox("Schedule window", list(WEEK_PRESETS.keys()), key="dhq_sos_window",
                help="Strength of schedule is graded per position group — backs against run "
                     "defenses, passers and pass catchers against pass defenses.")
            adp_upload = st.file_uploader(
                "Upload ADP CSV (overrides live)", type=["csv"], key="dhq_adp_upload",
                help="Any CSV with a player-name column and an ADP/rank column. Overrides the "
                     "live Fantasy Football Calculator feed.",
            )
            market_weight = st.slider(
                "Market blend", 0, 100, 40, 5, key="dhq_market_weight",
                help="How much the board's ORDER defers to ADP. 0 = pure model, 100 = pure ADP. "
                     "VORP itself is never blended — only the ordering moves.",
            ) / 100.0
            st.caption(
                "Value-based drafting and the market disagree hardest at QB: with replacement "
                "set at the last starting QB, the model prices an elite QB as a top-15 overall "
                "pick in a 1QB league and real drafts take him ~20 picks later. This blend lets "
                "you decide how much deference the market gets."
            )
        with c7:
            st.markdown("**Model**")
            uncertainty = st.slider(
                "Projection uncertainty", 0.5, 2.0, 1.0, 0.1, key="dhq_uncertainty",
                help="Multiplier on the measured spread of where players actually finish "
                     "relative to their consensus rank. 1.0 = use the measured value as-is.",
            )
            st.caption(
                "The baseline is measured, not guessed: from your own weekly history, how far "
                "players land from where they ranked, per position and rank. Top-6 QBs and TEs "
                "come in around ±10 finish slots; RBs and WRs scatter more than twice as far. "
                "This slider scales that. Higher widens Ceiling/Floor and flattens the board."
            )
            tiers = st.slider("Max tiers per position", 3, 12, 8, key="dhq_tiers")
            baseline_season = st.selectbox(
                "Projection baseline through", AVAILABLE_SEASONS_WITH_UPCOMING[1:], index=0,
                key="dhq_curve_season",
                help="Last completed season used to build the usage curves and player rates.",
            )

        st.markdown("---")
        st.markdown("**Import Fantasy Football Advice projections** (optional)")
        st.caption(
            "Drop in an FFA player export and the board will use their analysts' projected "
            "STAT LINE — carries, yards, receptions — re-scored under your league settings, "
            "plus their FFA Value and written player notes. Importing the stat line rather "
            "than their point total is what keeps it correct: their export is half-PPR, so "
            "reading their points straight off would be wrong in any other format."
        )
        f1, f2 = st.columns([2, 1])
        with f1:
            ffa_upload = st.file_uploader("FFA players JSON", type=["json"], key="dhq_ffa_upload")
        with f2:
            ffa_weight = st.slider("Blend FFA stat line into projections", 0, 100, 0, 5, key="dhq_ffa_weight",
                help="100 = use their projections outright, 0 = keep this app's own and take "
                     "only their notes and value score.") / 100.0

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
        'draft_type': draft_type,
        'my_slot': int(my_slot), 'board_format': board_format,
        'adp_year': int(adp_year), 'adp_upload': adp_upload, 'adp_source': adp_source,
        'uncertainty': float(uncertainty), 'tiers': int(tiers),
        'baseline_season': int(baseline_season), 'market_weight': float(market_weight),
        'sos_window': sos_window, 'ffa_upload': ffa_upload, 'ffa_weight': float(ffa_weight),
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
        next_pick, settings['uncertainty'], settings['tiers'],
        settings['baseline_season'], settings['market_weight'],
        settings['sos_window'], settings['ffa_weight'], settings.get('ffa_rows', 0),
        settings.get('adp_source', 'Auto'),
        # str() rather than float() - scoring now carries 'bonus_mode',
        # which is a string, and float()-ing every value would raise the
        # moment the settings panel is opened.
        tuple(sorted((k, str(v)) for k, v in scoring.items())),
        tuple(sorted((k, int(v)) for k, v in roster.items())),
        str(adp_meta.get('source')), adp_meta.get('teams'), adp_meta.get('rows'),
    )


@st.cache_data(show_spinner=False)
def _cached_board(_ecr_board, _adp_df, _ffa_df, _settings, cache_key):
    """
    Cache the assembled board on the SETTINGS, not on the DataFrames.

    Same reasoning as data.transforms.apply_scoring_and_percentiles: the
    underscore-prefixed frames are excluded from the hash because hashing a
    700-row board (and re-hashing it on every widget interaction anywhere in
    the app) costs more than the transform it's protecting. cache_key fully
    determines the output for a given source snapshot, which is what makes
    that safe here.
    """
    scoring = _settings['scoring']
    season = _settings['baseline_season']

    # Volume-based stat-line projections first, so build_draft_board sees a
    # board that already carries Proj Pts and only adds the outcome range
    # around it (see add_outcome_range_from_projections).
    projected, proj_meta = build_projected_board(
        _ecr_board, scoring, latest_season=season, ppr_for_ranking=float(scoring.get('rec', 1.0)))

    ffa_meta = {}
    if _ffa_df is not None and not _ffa_df.empty:
        projected, ffa_meta = merge_ffa_into_board(
            projected, _ffa_df, stat_line_weight=_settings['ffa_weight'], scoring=scoring)

    board, meta = build_draft_board(
        projected, _settings, adp_df=_adp_df, next_pick=cache_key[3],
        tiers_per_position=_settings['tiers'], rank_sd_scale=_settings['uncertainty'],
        latest_season=season, market_weight=_settings['market_weight'],
    )

    week_start, week_end = WEEK_PRESETS.get(_settings['sos_window'], (1, 17))
    sos = build_team_sos(_settings['adp_year'], season, week_start, week_end,
                         scoring_ppr=float(scoring.get('rec', 1.0)))
    board = attach_sos_to_board(board, sos)
    board = adp_quartiles(board)

    # FFA's ranking sits beside VORP and VONA as just another column, rather
    # than in a tab of its own. Two independent rankings are most useful read
    # side by side on the same row - the disagreement is the informative
    # part, and it's invisible if you have to switch screens to see it.
    if 'FFA Value' in board.columns and board['FFA Value'].notna().any():
        board['FFA Rank'] = pd.to_numeric(board['FFA Value'], errors='coerce').rank(
            ascending=False, method='first').astype('Int64')

    meta['projection'] = proj_meta
    meta['ffa'] = ffa_meta
    meta['sos_window'] = (week_start, week_end)
    return board, meta


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
    # FFA import resolves FIRST, because it is now a candidate ADP source -
    # their composite is drawn across real drafts on the major platforms,
    # which is a far better market read than a free mock site.
    ffa_df, ffa_err = pd.DataFrame(), None
    upload = settings.get('ffa_upload')
    if upload is not None:
        ffa_df, ffa_err = save_ffa_import(upload)
    else:
        ffa_df, ffa_err = load_ffa_import()
    status['ffa'] = {'rows': int(len(ffa_df)), 'error': ffa_err}
    settings = {**settings, 'ffa_rows': int(len(ffa_df))}

    adp_df, adp_meta = fetch_adp(scoring_label, settings['num_teams'], is_superflex,
                                 settings['adp_year'], uploaded=settings.get('adp_upload'),
                                 ffa_df=ffa_df, source=settings.get('adp_source', 'Auto'))
    adp_meta = dict(adp_meta or {})
    adp_meta['rows'] = int(len(adp_df))
    status['adp'] = adp_meta
    status['ecr_age'] = ecr_age_days(ecr_raw)

    board, meta = _cached_board(ecr_board, adp_df, ffa_df, settings,
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

    age = status.get('ecr_age')
    if age is not None and age >= 3:
        bits.append(f"⚠️ Consensus rankings are {age} days old (source refreshes nightly; "
                    "it hasn't). Injuries and depth-chart news since then aren't in them.")
    projection = meta.get('projection') or {}
    if projection.get('volume_projections'):
        n = projection.get('players_with_history', 0)
        bits.append(f"✅ Stat-line projections from {n:,} players of local history")
    ffa = status.get('ffa') or {}
    if ffa.get('error'):
        bits.append(f"⚠️ FFA import: {str(ffa['error'])[:70]}")
    elif ffa.get('rows'):
        matched = (meta.get('ffa') or {}).get('matched', 0)
        bits.append(f"✅ FFA import: {matched} of {ffa['rows']} players matched")
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
    # Your next pick is the one after however many YOU have taken - not the
    # first pick number greater than the total picks logged.
    #
    # The old form broke the availability model in normal use: if you only
    # mark your own picks (which is what most people do), the total logged
    # stays tiny while your real position in the draft advances, so
    # next_pick_for kept returning the same pick number and Avail Next %
    # froze after the first selection. Counting your own picks is correct
    # whether or not the rest of the room is tracked.
    taken_by_me = len(_my_roster())
    nxt = my_picks[taken_by_me] if taken_by_me < len(my_picks) else None
    return {'picks_made': picks_made, 'rounds': rounds, 'my_picks': my_picks,
            'next_pick': nxt, 'on_clock': picks_made + 1,
            'round': picks_made // max(settings['num_teams'], 1) + 1}


def _render_roster_panel(settings):
    """Your roster as it fills, with the holes and bye stacking called out."""
    mine = _my_roster()
    st.markdown("**Your roster**")
    _render_roster_slots(settings)
    if mine:
        starters_pts, _ = optimal_lineup_points(mine, settings)
        # A raw point total is unreadable on its own - 1,850 means nothing
        # without knowing this slot in this format produces 1,700-1,950. The
        # percentile against a simulated distribution is the number that can
        # actually be acted on, and it's what every reference tool reports.
        grade = None
        distribution = st.session_state.get('dhq_outcome_dist')
        if distribution:
            grade = roster_percentile(starters_pts, distribution)
        if grade is not None:
            st.caption(f"Best starting lineup projects **{starters_pts:.0f} pts** "
                       f"— **{grade:.0f}th percentile** for this slot")
        else:
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


def _render_selectable_board(available, key_prefix, next_pick=None, columns=None, row_limit=60):
    """
    The sortable, filterable board grid, returning whichever player is
    currently selected (or None).

    Shared by the live draft room and the mock draft on purpose. They are the
    same surface answering the same question - the only difference is whether
    the picks are real - so they should not be two separately-drifting tables
    with two different sets of columns and two different sort behaviours. The
    mock in particular used to be a pair of dropdowns, which made it useless
    for the thing a mock is actually for: rehearsing the read you'll be doing
    live, on the board you'll be doing it on.
    """
    available = available.copy()
    # No position control here - the button row above owns that now, and two
    # filters for one thing is how you end up with a board that disagrees
    # with itself.
    c2, c3 = st.columns([3, 1])
    positions = []
    with c2:
        sort_by = st.selectbox("Sort by", ['Board Rank', 'VORP', 'VONA', 'Proj Pts',
                                           'Value vs ADP', 'ADP', 'ECR', 'Ceiling'],
                               key=f"{key_prefix}_sort")
    with c3:
        limit = st.number_input("Rows", 10, 400, row_limit, step=10, key=f"{key_prefix}_limit")

    if positions:
        available = available[available['Pos'].astype(str).str.upper().isin(positions)]
    ascending = sort_by in ('ADP', 'ECR', 'Board Rank')
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
            help=f"Chance he lasts to your next pick (#{next_pick})" if next_pick else "No next pick",
        )
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
    return display.index[rows[0]] if rows and rows[0] < len(display) else None


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



def _render_strategy_panel(settings, roster=None):
    """
    The archetype your picks have actually committed you to.

    Descriptive, not prescriptive, on purpose - "you are two picks into Zero
    RB" is a fact you can act on, where an abstract recommendation mid-draft
    is just noise competing with the board. It also makes the pick odds
    below readable, since what counts as a good next pick depends entirely
    on which shape you're already building.
    """
    mine = _my_roster() if roster is None else roster
    strategy = classify_strategy(mine, settings)
    st.markdown("**Your strategy**")
    if not strategy['primary']:
        st.caption("Not classified yet — make a pick or two.")
        return
    confidence = strategy['confidence']
    st.markdown(f"### {strategy['primary']}")
    st.caption(strategy['label'])
    st.progress(min(1.0, confidence), text=f"confidence {confidence:.0%}")
    alternates = [c for c in strategy['candidates'][1:] if c['weight'] > 0.08]
    if alternates:
        st.caption("Or pivot to: " + ", ".join(f"{c['id']} ({c['weight']:.0%})" for c in alternates))


def _render_pick_odds(board, settings, ctx):
    """
    What positions actually get taken at your upcoming picks - overall, and
    in the simulated drafts that finished in the top quartile.

    The two columns are the whole point. The first describes what usually
    happens; the second describes what happens when things go WELL from
    exactly where you're sitting. When they disagree, that gap is the
    advice.

    Run on demand rather than automatically: it simulates the rest of the
    draft dozens of times, which is seconds of work, and nobody wants their
    board to stall every time they mark a player gone.
    """
    st.markdown("**Odds at your next picks**")
    c1, c2 = st.columns([1, 1])
    with c1:
        n_sims = st.number_input("Sims", 10, 200, 40, step=10, key="dhq_intel_sims")
    with c2:
        st.write("")
        run = st.button("Run pick odds", key="dhq_intel_run")

    if run:
        signature = tuple((p['Player'], bool(p.get('mine'))) for p in _drafted_list())
        with st.spinner(f"Simulating {int(n_sims)} drafts forward from here..."):
            st.session_state['dhq_intel'] = pick_intel(
                board, settings, settings['my_slot'], ctx['rounds'], signature,
                n_sims=int(n_sims))
            # Same run also gives the scale the roster grade is read against,
            # so the percentile costs nothing extra.
            st.session_state['dhq_outcome_dist'] = outcome_distribution(
                board, settings, settings['my_slot'], ctx['rounds'],
                n_sims=max(12, int(n_sims) // 2))

    intel = st.session_state.get('dhq_intel')
    if not intel:
        st.caption("Run this to see which positions land at your next picks, and which ones "
                   "the best-finishing drafts took there.")
        return

    for offset, entry in enumerate(intel.get('rounds', [])):
        overall = ", ".join(f"{x['pos']} {x['freq']:.0%}" for x in entry['positions'][:4]) or "—"
        best = ", ".join(f"{x['pos']} {x['freq']:.0%}" for x in entry['top_quartile'][:4]) or "—"
        label = "Your next pick" if offset == 0 else f"Pick +{offset}"
        st.markdown(f"**{label}**")
        st.caption(f"typical: {overall}")
        st.caption(f"best drafts: {best}")
    st.caption(f"From {intel['sims']} simulated drafts. Top quartile = final starting lineup "
               f"above {intel['top_quartile_cutoff']:.0f} pts.")


def _render_run_pressure(settings):
    """Which positions the room is currently draining faster than baseline demand."""
    rows = positional_run_pressure(_drafted_list(), settings)
    if not rows:
        return
    hot = [r for r in rows if r['Pressure'] >= 12]
    if hot:
        st.warning("Run in progress: " + ", ".join(
            f"{r['Pos']} ({r['Recent share']:.0f}% of the last 12 picks vs {r['Baseline share']:.0f}% normal)"
            for r in hot[:2]))


def _render_player_detail(board, selected):
    """Notes and the projected stat line for whoever is selected."""
    if not selected:
        return
    row = board[board['Player'] == selected]
    if row.empty:
        return
    row = row.iloc[0]
    notes = row.get('Notes')
    stat_bits = []
    for label, column in (('carries', 'carries'), ('targets', 'targets'),
                          ('rush yds', 'rushing_yards'), ('rec', 'receptions'),
                          ('rec yds', 'receiving_yards'), ('pass yds', 'passing_yards')):
        value = row.get(column)
        if pd.notna(value) and float(value) > 0:
            stat_bits.append(f"{float(value):.0f} {label}")
    if stat_bits:
        st.caption("Projected: " + " · ".join(stat_bits) +
                   f"  ({row.get('proj_basis', 'projection')})")
    if isinstance(notes, str) and notes.strip():
        with st.expander(f"Player note — {selected}", expanded=False):
            st.write(notes)



def _roster_slot_plan(settings):
    """The starting lineup as an ordered list of slot labels, from settings."""
    roster = settings['roster']
    plan = []
    for pos in ('QB', 'RB', 'WR', 'TE'):
        plan += [pos] * int(roster.get(pos, 0))
    plan += ['FLEX'] * int(roster.get('FLEX', 0))
    plan += ['SUPERFLEX'] * int(roster.get('SUPERFLEX', 0))
    for pos in ('K', 'DST'):
        plan += [pos] * int(roster.get(pos, 0))
    plan += ['BENCH'] * int(roster.get('BENCH', 0))
    return plan


SLOT_ELIGIBLE = {
    'QB': ('QB',), 'RB': ('RB',), 'WR': ('WR',), 'TE': ('TE',),
    'K': ('K',), 'DST': ('DST',),
    'FLEX': ('RB', 'WR', 'TE'),
    'SUPERFLEX': ('QB', 'RB', 'WR', 'TE'),
    'BENCH': ('QB', 'RB', 'WR', 'TE', 'K', 'DST'),
}


def _fill_roster_slots(my_roster, settings):
    """
    Assign drafted players to concrete lineup slots.

    Dedicated slots are filled before FLEX and FLEX before BENCH, and within
    each slot the best remaining eligible player is taken. That ordering
    matters: a roster of RB/RB/WR filled naively can drop a back into FLEX
    while a real RB slot sits empty, which then reports a positional need
    that doesn't exist and skews every recommendation downstream.

    Returns [(slot_label, player_or_None)] in lineup order, so the panel can
    render empty slots from the very first pick instead of appearing only
    once something fills them.
    """
    plan = _roster_slot_plan(settings)
    remaining = sorted(list(my_roster or []),
                       key=lambda p: (p.get('Proj Pts') if pd.notna(p.get('Proj Pts')) else 0),
                       reverse=True)
    filled = []
    for slot in plan:
        eligible = SLOT_ELIGIBLE.get(slot, ())
        chosen = None
        for candidate in remaining:
            if str(candidate.get('Pos', '')).upper() in eligible:
                chosen = candidate
                break
        if chosen is not None:
            remaining.remove(chosen)
        filled.append((slot, chosen))
    # Anyone left over (more players than slots) is still on the roster and
    # shouldn't silently vanish from the panel.
    for leftover in remaining:
        filled.append(('BENCH', leftover))
    return filled


def _pos_chip(pos):
    from config import get_position_color, get_position_chip_bg
    pos = str(pos or '').upper()
    bg = get_position_chip_bg(pos) or 'rgba(255,255,255,0.06)'
    fg = get_position_color(pos)
    return (f"<span style='background:{bg};color:{fg};border-radius:4px;padding:1px 6px;"
            f"font-size:10px;font-weight:700;letter-spacing:.04em'>{pos or '--'}</span>")


def _render_roster_slots(settings, roster=None):
    """
    Your lineup as slots that fill in, rather than a flat list of picks.

    `roster` is explicit so the mock draft can render its own simulated
    roster through the same component - reading the live tracker there would
    show your real draft's players inside a mock.
    """
    mine = _my_roster() if roster is None else roster
    filled = _fill_roster_slots(mine, settings)
    rows = []
    for slot, player in filled:
        if player is None:
            rows.append(
                f"<div class='rs-row rs-empty'><span class='rs-slot'>{slot}</span>"
                f"<span class='rs-name'>—</span></div>")
        else:
            bye = player.get('Bye')
            bye_txt = f"bye {int(bye)}" if pd.notna(bye) else ""
            pts = player.get('Proj Pts')
            pts_txt = f"{float(pts):.0f}" if pd.notna(pts) else ""
            rows.append(
                f"<div class='rs-row'><span class='rs-slot'>{slot}</span>"
                f"<span class='rs-name'>{_pos_chip(player.get('Pos'))} {player['Player']}</span>"
                f"<span class='rs-meta'>{bye_txt}</span><span class='rs-pts'>{pts_txt}</span></div>")
    st.markdown(
        "<style>"
        ".rs-row{display:flex;align-items:center;gap:8px;padding:3px 6px;border-radius:4px;"
        "font-size:12px;border-bottom:1px solid rgba(255,255,255,0.05)}"
        ".rs-empty{opacity:.45}"
        ".rs-slot{min-width:64px;font-weight:700;font-size:10px;letter-spacing:.06em;opacity:.75}"
        ".rs-name{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}"
        ".rs-meta{opacity:.6;font-size:10px}"
        ".rs-pts{min-width:34px;text-align:right;font-variant-numeric:tabular-nums;opacity:.8}"
        "</style>" + "".join(rows), unsafe_allow_html=True)


def _team_for_pick(pick_number, num_teams, draft_type='Snake'):
    """Which slot owns an overall pick number, honoring snake order."""
    idx = int(pick_number) - 1
    rnd, slot = divmod(idx, int(num_teams))
    if draft_type == 'Snake' and rnd % 2 == 1:
        return int(num_teams) - slot
    return slot + 1


def _render_draft_board_grid(picks, num_teams, my_slot, draft_type='Snake', highlight_mine=True):
    """
    The full draft board: teams across, rounds down, every pick in place.

    This is the view a real draft room puts on the wall, and it answers
    questions no sorted list can - who is hoarding running backs, whether the
    turn is about to strip a position, what the room's shape looks like three
    rounds from now. Snake order is drawn as it actually runs, so a round
    reads left-to-right or right-to-left exactly as the picks happened.
    """
    if not picks:
        st.caption("No picks yet — the board fills in as the draft runs.")
        return

    by_cell = {}
    max_round = 1
    for pick in picks:
        number = pick.get('pick')
        if number is None:
            continue
        team = pick.get('team') or _team_for_pick(number, num_teams, draft_type)
        rnd = (int(number) - 1) // int(num_teams) + 1
        max_round = max(max_round, rnd)
        by_cell[(rnd, int(team))] = pick

    from config import get_position_color, get_position_chip_bg
    html = ["<div class='db-wrap'><table class='db'>", "<tr><th class='db-rnd'></th>"]
    for team in range(1, int(num_teams) + 1):
        label = f"Team {team}" + (" ★" if team == my_slot else "")
        html.append(f"<th class='{'db-mine' if team == my_slot else ''}'>{label}</th>")
    html.append("</tr>")

    for rnd in range(1, max_round + 1):
        html.append(f"<tr><td class='db-rnd'>{rnd}</td>")
        for team in range(1, int(num_teams) + 1):
            pick = by_cell.get((rnd, team))
            if not pick:
                html.append("<td class='db-cell db-empty'></td>")
                continue
            pos = str(pick.get('Pos', '')).upper()
            bg = get_position_chip_bg(pos) or '#1a2447'
            fg = get_position_color(pos)
            mine = highlight_mine and team == my_slot
            name = str(pick.get('Player', ''))
            short = name if len(name) <= 17 else name[:16] + '…'
            html.append(
                f"<td class='db-cell{' db-mine-cell' if mine else ''}' "
                f"style='background:{bg};border-left:3px solid {fg}'>"
                f"<div class='db-name'>{short}</div>"
                f"<div class='db-sub' style='color:{fg}'>{pos} · {pick.get('Team NFL') or pick.get('Team') or ''}"
                f" · {int(pick['pick'])}</div></td>")
        html.append("</tr>")
    html.append("</table></div>")

    st.markdown(
        "<style>"
        ".db-wrap{overflow-x:auto;max-width:100%}"
        ".db{border-collapse:separate;border-spacing:3px;font-size:11px}"
        ".db th{font-size:10px;letter-spacing:.06em;text-transform:uppercase;opacity:.7;"
        "padding:2px 4px;white-space:nowrap}"
        ".db th.db-mine{color:#00fff9;opacity:1}"
        ".db-rnd{font-size:10px;opacity:.5;text-align:center;min-width:20px}"
        ".db-cell{min-width:118px;max-width:118px;padding:4px 6px;border-radius:4px;vertical-align:top}"
        ".db-empty{background:rgba(255,255,255,0.03)}"
        ".db-mine-cell{outline:1px solid rgba(0,255,249,0.55)}"
        ".db-name{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}"
        ".db-sub{font-size:9px;opacity:.85;letter-spacing:.03em}"
        "</style>" + "".join(html), unsafe_allow_html=True)



# Position filter buttons. FLEX expands to the positions a flex slot can
# actually take, which is the filter people reach for constantly and that a
# plain position dropdown can't express at all.
POSITION_FILTERS = [
    ('All', None),
    ('QB', ['QB']),
    ('RB', ['RB']),
    ('WR', ['WR']),
    ('TE', ['TE']),
    ('FLEX', ['RB', 'WR', 'TE']),
    ('K', ['K']),
    ('DST', ['DST']),
]
POS_FILTER_KEY = 'dhq_pos_filter'


def _render_position_filter():
    """
    A row of buttons rather than a dropdown.

    A dropdown costs a click to open, a read to find the option, and a click
    to choose - three actions to answer "show me receivers", repeated dozens
    of times per draft. Buttons make it one, and they show the current state
    without being opened.
    """
    current = st.session_state.get(POS_FILTER_KEY, 'All')
    cols = st.columns(len(POSITION_FILTERS))
    for col, (label, _) in zip(cols, POSITION_FILTERS):
        with col:
            if st.button(label, key=f"posfilter_{label}", width="stretch",
                         type="primary" if current == label else "secondary"):
                st.session_state[POS_FILTER_KEY] = label
                st.rerun()
    return dict(POSITION_FILTERS).get(current), current


def _apply_position_filter(df, positions):
    if not positions or df.empty:
        return df
    return df[df['Pos'].astype(str).str.upper().isin(positions)]


def _render_recent_picks_strip(picks, num_teams, current_pick, window=12):
    """
    The last dozen picks as colored cards, plus the upcoming slots as empty
    ones.

    This is the thing a drafter actually keeps glancing at - not a table of
    everyone taken, just "what just happened and what's about to". Showing
    the empty upcoming boxes alongside is what makes it readable as a
    position on the clock rather than a log: you can see how many picks
    stand between you and your turn without counting.
    """
    start = max(1, int(current_pick) - window + 1)
    by_number = {int(p['pick']): p for p in picks if p.get('pick')}

    from config import get_position_color, get_position_chip_bg
    cards = []
    for number in range(start, start + window):
        pick = by_number.get(number)
        if pick:
            pos = str(pick.get('Pos', '')).upper()
            bg = get_position_chip_bg(pos) or '#1a2447'
            fg = get_position_color(pos)
            name = str(pick.get('Player', ''))
            short = name if len(name) <= 15 else name.split()[-1][:14]
            cards.append(
                f"<div class='rp-card' style='background:{bg};border-color:{fg}'>"
                f"<div class='rp-num'>PICK {number}</div>"
                f"<div class='rp-name'>{short}</div>"
                f"<div class='rp-sub' style='color:{fg}'>{pos} · {pick.get('Team NFL') or pick.get('Team') or ''}</div>"
                f"</div>")
        else:
            on_clock = number == int(current_pick)
            cards.append(
                f"<div class='rp-card rp-empty{' rp-clock' if on_clock else ''}'>"
                f"<div class='rp-num'>PICK {number}</div>"
                f"<div class='rp-name'>{'ON THE CLOCK' if on_clock else '—'}</div>"
                f"<div class='rp-sub'></div></div>")

    st.markdown(
        "<style>"
        ".rp-strip{display:flex;gap:4px;overflow-x:auto;padding:2px 0 6px 0}"
        ".rp-card{min-width:96px;max-width:96px;border-radius:5px;padding:4px 6px;"
        "border-left:3px solid rgba(255,255,255,0.15);font-size:11px}"
        ".rp-empty{background:rgba(255,255,255,0.035);opacity:.55}"
        ".rp-clock{outline:1px solid rgba(0,255,249,0.7);opacity:1}"
        ".rp-num{font-size:9px;letter-spacing:.08em;opacity:.6;font-weight:700}"
        ".rp-name{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}"
        ".rp-sub{font-size:9px;opacity:.9;letter-spacing:.03em;min-height:11px}"
        "</style>"
        f"<div class='rp-strip'>{''.join(cards)}</div>", unsafe_allow_html=True)


def _render_single_recommendation(board, settings, my_roster, next_pick, positions, label):
    """
    ONE recommendation, scoped to whatever the position filter is showing.

    A five-row table of suggestions duplicated most of the board directly
    beneath it and ate the screen for it. The board is already sorted by
    value - what it can't tell you is which single player best fits THIS
    roster right now, and that's a one-line answer. Narrowing by position
    turns it into "best tight end available" without a second control.
    """
    available = board[~board['Player'].isin(_drafted_names())]
    available = _apply_position_filter(available, positions)
    if available.empty:
        st.info("Nothing available at that position.")
        return None

    recs = recommend_picks(available, my_roster, settings, next_pick=next_pick, top_n=1,
                           allow_late_round=bool(positions))
    if recs.empty:
        recs = available.nlargest(1, 'VORP' if 'VORP' in available.columns else 'Proj Pts')
        recs = recs.assign(**{'Fit Score': np.nan, 'Why': 'best available'})
    pick = recs.iloc[0]

    from config import get_position_color, get_position_chip_bg
    pos = str(pick['Pos']).upper()
    bg = get_position_chip_bg(pos) or '#1a2447'
    fg = get_position_color(pos)
    scope = "overall" if label == 'All' else label
    bits = []
    for column, fmt in (('Proj Pts', '{:.0f} pts'), ('VORP', 'VORP {:.0f}'),
                        ('VONA', 'VONA {:.0f}'), ('ADP', 'ADP {:.0f}')):
        value = pick.get(column)
        if pd.notna(value):
            bits.append(fmt.format(float(value)))
    avail_pct = pick.get('Avail Next %')
    if pd.notna(avail_pct):
        bits.append(f"{float(avail_pct):.0f}% to last")

    st.markdown(
        "<style>.reco{display:flex;align-items:center;gap:12px;border-radius:8px;"
        "padding:8px 12px;margin-bottom:6px}"
        ".reco-pos{font-size:11px;font-weight:800;letter-spacing:.06em;padding:2px 8px;border-radius:4px}"
        ".reco-name{font-size:17px;font-weight:700}"
        ".reco-meta{font-size:11px;opacity:.85}"
        ".reco-why{font-size:11px;opacity:.7;font-style:italic}</style>"
        f"<div class='reco' style='background:{bg};border-left:4px solid {fg}'>"
        f"<span class='reco-pos' style='background:{fg};color:#0b1020'>BEST {scope.upper()}</span>"
        f"<span class='reco-name'>{pick['Player']}</span>"
        f"<span class='reco-meta' style='color:{fg}'>{pos} · {pick.get('Team') or ''}</span>"
        f"<span class='reco-meta'>{' · '.join(bits)}</span>"
        f"<span class='reco-why'>{pick.get('Why', '')}</span>"
        "</div>", unsafe_allow_html=True)
    return pick['Player']


def _draft_context(board, settings, ctx, mode):
    """
    One description of "where the draft is", whether it's real or simulated.

    Both modes answer the same questions - who's gone, what's my roster,
    which pick am I on, what happens when I take someone - so the surface
    below is written once against this, rather than as two screens that
    drift apart. The drafted lists stay strictly separate: a mock can never
    write into your real draft log.
    """
    if mode == 'Mock draft':
        state = st.session_state.get(SIM_KEY)
        if state is None:
            return None
        pool, order_col, _ = prepare_sim_pool(board)
        avail = available_players(state, pool) if not pool.empty else board.iloc[0:0]
        my_roster = [dict(p) for p in state['rosters'][state['my_slot']]]
        return {
            'mode': mode, 'state': state, 'pool': pool, 'order_col': order_col,
            'picks': state['picks'], 'my_roster': my_roster,
            'available': avail,
            'current_pick': state['pick_no'],
            'next_pick': state['pick_no'] + state['num_teams'],
            'num_teams': state['num_teams'], 'my_slot': state['my_slot'],
            'complete': state['complete'],
            'on_clock_me': team_on_clock(state) == state['my_slot'],
        }

    picks = []
    for i, pick in enumerate(_drafted_list()):
        entry = dict(pick)
        entry['pick'] = i + 1
        entry['team'] = _team_for_pick(i + 1, settings['num_teams'], settings['draft_type'])
        picks.append(entry)
    return {
        'mode': mode, 'state': None, 'pool': None, 'order_col': None,
        'picks': picks, 'my_roster': _my_roster(),
        'available': board[~board['Player'].isin(_drafted_names())],
        'current_pick': len(picks) + 1,
        'next_pick': ctx['next_pick'],
        'num_teams': settings['num_teams'], 'my_slot': settings['my_slot'],
        'complete': False, 'on_clock_me': True,
    }


def _commit_pick(dc, settings, player_name, mine=True, reach=3.0):
    """Take a player, in whichever mode is active."""
    row = dc['available'][dc['available']['Player'] == player_name]
    if row.empty:
        return
    if dc['mode'] == 'Mock draft':
        state = dc['state']
        record_pick(state, state['my_slot'], row.iloc[0])
        run_until_user_pick(state, settings, dc['pool'], dc['order_col'], reach_window=reach)
        st.session_state[SIM_KEY] = state
    else:
        _record_pick(row.iloc[0], mine=mine)
    st.session_state['dhq_selected'] = None
    st.rerun()



@st.dialog("Player profile", width="large")
def _player_profile_dialog(player_name, board):
    """
    A player's profile in a modal, without leaving the draft.

    WHY A PURPOSE-BUILT PANEL RATHER THAN THE PLAYER SEARCH TAB: that tab's
    render() is a single function wired to fixed widget keys (year_tab1,
    player_search_team_filter, player_sel_t1_name). Calling it from inside a
    dialog while the tab itself exists registers those keys twice and
    Streamlit raises. Rebuilding it would mean editing that tab, which is
    exactly what was asked not to happen. So this reads the same underlying
    data through the same cached loaders and shows the parts that matter on
    the clock, with a jump to the full tab for a deeper look.

    Streamlit reruns the script when a dialog opens, but session state
    survives it - so an in-progress mock draft is untouched.
    """
    row = board[board['Player'] == player_name]
    if row.empty:
        st.info("No board data for that player.")
        return
    row = row.iloc[0]
    pos = str(row['Pos']).upper()

    from config import get_position_color, get_position_chip_bg
    fg = get_position_color(pos)
    bg = get_position_chip_bg(pos) or '#1a2447'
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;background:{bg};"
        f"border-left:4px solid {fg};border-radius:8px;padding:8px 12px'>"
        f"<span style='font-size:20px;font-weight:700'>{player_name}</span>"
        f"<span style='color:{fg};font-weight:700'>{pos} · {row.get('Team') or ''}</span>"
        f"<span style='opacity:.8;font-size:12px'>{row.get('Pos Rk') or ''}"
        f"{' · bye ' + str(int(row['Bye'])) if pd.notna(row.get('Bye')) else ''}</span></div>",
        unsafe_allow_html=True)

    tiles = []
    for label, column, fmt in (('Proj Pts', 'Proj Pts', '{:.0f}'), ('VORP', 'VORP', '{:.0f}'),
                               ('VONA', 'VONA', '{:.0f}'), ('ADP', 'ADP', '{:.1f}'),
                               ('FFA Rank', 'FFA Rank', '{:.0f}'), ('Tier', 'Tier', '{:.0f}'),
                               ('Ceiling', 'Ceiling', '{:.0f}'), ('Floor', 'Floor', '{:.0f}')):
        value = row.get(column)
        if pd.notna(value):
            tiles.append((label, fmt.format(float(value))))
    if tiles:
        cols = st.columns(len(tiles))
        for col, (label, value) in zip(cols, tiles):
            col.metric(label, value)

    stat_bits = []
    for label, column in (('carries', 'carries'), ('targets', 'targets'),
                          ('rush yds', 'rushing_yards'), ('rec', 'receptions'),
                          ('rec yds', 'receiving_yards'), ('rush TD', 'rushing_tds'),
                          ('rec TD', 'receiving_tds'), ('pass yds', 'passing_yards'),
                          ('pass TD', 'passing_tds')):
        value = row.get(column)
        if pd.notna(value) and float(value) > 0:
            stat_bits.append(f"**{float(value):.0f}** {label}")
    if stat_bits:
        st.markdown("**Projected line** — " + " · ".join(stat_bits))
        st.caption(f"Source: {row.get('proj_basis', 'projection')}")

    notes = row.get('Notes')
    if isinstance(notes, str) and notes.strip():
        st.markdown("**Scouting note**")
        st.write(notes)

    st.markdown("---")
    st.markdown("**Recent game log**")
    try:
        from data.transforms import load_and_merge_data
        scoring_rule = st.session_state.get('score_tab1', 'Full PPR')
        season = int(st.session_state.get('dhq_curve_season', 2025))
        stats, team_col, name_col, _ = load_and_merge_data(season, scoring_rule)
        log = stats[stats[name_col] == player_name]
        log = log[pd.to_numeric(log['week'], errors='coerce').fillna(0) > 0]
        if log.empty:
            st.caption(f"No {season} game log for this player (rookie, or he didn't play).")
        else:
            wanted = ['week', 'opponent_team', 'fantasy_points', 'carries', 'rushing_yards',
                      'rushing_tds', 'targets', 'receptions', 'receiving_yards', 'receiving_tds',
                      'passing_yards', 'passing_tds', 'passing_interceptions']
            cols = [c for c in wanted if c in log.columns]
            view = log[cols].sort_values('week', ascending=False).head(10)
            view = view.loc[:, (view.fillna(0) != 0).any(axis=0)]
            st.dataframe(view, width="stretch", hide_index=True,
                         height=df_auto_height(min(len(view), 10)))
    except Exception as exc:
        st.caption(f"Couldn't load the game log ({type(exc).__name__}).")

    st.caption("For the full profile — percentile radars, splits, matchup history — open the "
               "Player Search tab. This draft stays exactly where it is.")


def _render_draft_room(board, settings, ctx):
    """
    The single draft surface, live or simulated.

    Previously two tabs that were 90% the same screen. Keeping them apart
    meant every improvement had to be built twice and they slowly diverged -
    the mock had a draft board, the live room didn't; the live room had
    strategy and pick odds, the mock didn't. One surface with a mode switch
    is the same product with half the code and none of the drift.
    """
    mode = st.radio("Mode", ["Live draft", "Mock draft"], horizontal=True,
                    key="dhq_mode", label_visibility="collapsed")

    reach = 3.0
    if mode == "Mock draft":
        controls = st.columns([1.1, 1, 1, 1, 2])
        with controls[0]:
            start_mock = st.button("🎲 New mock", key="dhq_new_mock", type="primary",
                                   width="stretch")
        with controls[1]:
            slot = st.number_input("Slot", 1, int(settings['num_teams']),
                                   int(settings['my_slot']), key="dhq_mock_slot")
        with controls[2]:
            rounds = st.number_input("Rounds", 5, 25, int(ctx['rounds']), key="dhq_mock_rounds")
        with controls[3]:
            reach = st.slider("Chaos", 1.0, 8.0, 3.0, 0.5, key="dhq_mock_reach",
                              help="How far simulated opponents stray from ADP.")
        if start_mock:
            pool, order_col, _ = prepare_sim_pool(board)
            if pool.empty:
                st.error("No board to simulate against.")
            else:
                state = init_draft_state({**settings, 'my_slot': int(slot)}, int(slot), int(rounds))
                run_until_user_pick(state, settings, pool, order_col, reach_window=reach)
                st.session_state[SIM_KEY] = state
                st.session_state['dhq_selected'] = None
                st.rerun()

    dc = _draft_context(board, settings, ctx, mode)
    if dc is None:
        st.info("Start a new mock to draft against a simulated room.")
        return

    if dc['complete']:
        st.success("Mock complete.")
        grades = grade_draft(dc['state'], settings)
        st.dataframe(grades[['Rank', 'Team', 'Starters Proj', 'Bench Proj']], width="stretch",
                     hide_index=True, height=df_auto_height(len(grades)))

    top = st.columns(4)
    top[0].metric("Pick", dc['current_pick'])
    top[1].metric("Round", (dc['current_pick'] - 1) // max(dc['num_teams'], 1) + 1)
    top[2].metric("Your next pick", dc['next_pick'] if dc['next_pick'] else "—")
    top[3].metric("Your roster", len(dc['my_roster']))

    _render_recent_picks_strip(dc['picks'], dc['num_teams'], dc['current_pick'])
    if mode == "Live draft":
        _render_run_pressure(settings)

    positions, label = _render_position_filter()
    _render_single_recommendation(board, settings, dc['my_roster'], dc['next_pick'],
                                  positions, label)

    selected = st.session_state.get('dhq_selected')
    if selected and selected not in set(dc['available']['Player']):
        selected = None

    action = st.columns([1.3, 1.3, 1.3, 3])
    if mode == "Mock draft":
        with action[0]:
            if st.button("✅ Draft selected", key="dhq_act_draft", type="primary",
                         disabled=not selected, width="stretch"):
                _commit_pick(dc, settings, selected, reach=reach)
        with action[1]:
            if st.button("⏭ Auto-pick", key="dhq_act_auto", width="stretch"):
                row = autopick_for_user(dc['state'], settings, dc['pool'])
                if row is not None:
                    _commit_pick(dc, settings, row['Player'], reach=reach)
    else:
        with action[0]:
            if st.button("➕ Draft to my team", key="dhq_act_mine", type="primary",
                         disabled=not selected, width="stretch"):
                _commit_pick(dc, settings, selected, mine=True)
        with action[1]:
            if st.button("❌ Taken by others", key="dhq_act_other",
                         disabled=not selected, width="stretch"):
                _commit_pick(dc, settings, selected, mine=False)
        with action[2]:
            if st.button("↩ Undo", key="dhq_act_undo", disabled=not dc['picks'],
                         width="stretch"):
                _undo_last()
                st.rerun()
    with action[2 if mode == "Mock draft" else 3]:
        if selected and st.button("🔍 Player profile", key="dhq_act_profile", width="stretch"):
            _player_profile_dialog(selected, board)
    with action[3]:
        if selected:
            st.caption(f"Selected: **{selected}**")
        else:
            st.caption("Click a row below to select a player.")

    left, right = st.columns([3, 1])
    with left:
        view = _apply_position_filter(dc['available'], positions)
        picked = _render_selectable_board(view, "dhq_board", next_pick=dc['next_pick'],
                                          row_limit=40)
        if picked and picked != selected:
            st.session_state['dhq_selected'] = picked
            st.rerun()
        _render_player_detail(board, selected)
    with right:
        _render_roster_slots(settings, roster=dc['my_roster'])
        st.markdown("---")
        _render_strategy_panel(settings, roster=dc['my_roster'])

    with st.expander("🗂 Full draft board", expanded=False):
        _render_draft_board_grid(dc['picks'], dc['num_teams'], dc['my_slot'],
                                 settings['draft_type'])
    with st.expander("📊 Pick odds & positional scarcity", expanded=False):
        _render_pick_odds(board, settings, ctx)
        _render_positional_scarcity(board, settings)
    if mode == "Live draft":
        _render_live_sync(board)
    else:
        with st.expander("🔁 Run many mocks / compare slots", expanded=False):
            _render_mock_tools(board, settings, ctx)


# ---------------------------------------------------------------------------
# Mock draft
# ---------------------------------------------------------------------------

def _render_mock_tools(board, settings, ctx):
    """
    Batch simulation tools: many drafts at once, and per-slot comparison.

    Separate from the interactive mock because they answer a different
    question. One mock tells you what happened; fifty tell you what usually
    happens, which is the only version worth planning around.
    """
    pool, _, has_adp = prepare_sim_pool(board)
    if pool.empty:
        st.info("No board loaded to simulate against.")
        return
    if not has_adp:
        st.warning(
            "No ADP loaded, so simulated opponents draft off this board's own value ranking "
            "rather than the market. Useful as a stress test, but not a realistic room — treat "
            "'who fell to me' with suspicion."
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rounds = st.number_input("Rounds", 5, 25, int(ctx['rounds']), key="dhq_batch_rounds")
    with c2:
        slot = st.number_input("Slot", 1, int(settings['num_teams']), int(settings['my_slot']),
                               key="dhq_batch_slot")
    with c3:
        reach = st.slider("Chaos", 1.0, 8.0, 3.0, 0.5, key="dhq_batch_reach")
    with c4:
        n_sims = st.number_input("Simulations", 1, 100, 20, key="dhq_batch_n")

    b1, b2 = st.columns(2)
    with b1:
        run_many = st.button("▶ Run simulations", key="dhq_batch_run", width="stretch")
    with b2:
        run_slots = st.button("▶ Compare every slot", key="dhq_slot_run", width="stretch")

    if run_many:
        with st.spinner(f"Running {int(n_sims)} drafts..."):
            summary, outcomes = run_many_drafts(board, settings, int(slot), int(rounds),
                                                n_sims=int(n_sims), reach_window=float(reach))
        if outcomes.empty:
            st.info("Simulation produced no results.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Median starters projection", f"{outcomes['Starters Proj'].median():.0f}")
            m2.metric("Range", f"{outcomes['Starters Proj'].min():.0f} – {outcomes['Starters Proj'].max():.0f}")
            m3.metric("Avg league finish", f"{outcomes['League Rank'].mean():.1f}")
            st.caption("Read as tendencies, not predictions — 60% of drafts is a player your "
                       "slot reliably gets; 10% is a coin flip you shouldn't plan around.")
            st.dataframe(summary[['round', 'Player', 'Pos', '% of drafts']], width="stretch",
                         hide_index=True, height=df_auto_height(min(len(summary), 26)))

    if run_slots:
        sims_each = max(4, int(n_sims) // 3)
        with st.spinner(f"Running {settings['num_teams']} slots x {sims_each} drafts..."):
            cmp_df = pick_slot_comparison(board, settings, int(rounds),
                                          n_sims=sims_each, reach_window=float(reach))
        if cmp_df.empty:
            st.info("No results.")
        else:
            st.caption("Which slots your settings actually favour — the answer moves with league "
                       "size, superflex and TE premium, so the usual wisdom about 'the turn' "
                       "often doesn't apply.")
            st.dataframe(cmp_df, width="stretch", hide_index=True,
                         height=df_auto_height(len(cmp_df)))


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
            st.caption(f"No injury data available ({str(inj_err or 'empty')[:90]}).")
        else:
            season_used = inj.attrs.get('season')
            if season_used and int(season_used) != int(settings['adp_year']):
                st.caption(
                    f"Showing {season_used} designations — nflverse has no {settings['adp_year']} "
                    "injury rows until the season starts. Stale for lineups, still useful for "
                    "who finished last year hurt."
                )
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
            reason = str(news_err or 'empty').split('(')[0].strip(' :,)')[:70]
            st.caption(
                f"No news feed available ({reason}). ESPN's public feed is the only source "
                "wired in here and it refuses some networks outright — the injury designations "
                "on the left come from nflverse and are unaffected."
            )
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

    # The big board is no longer its own sub-tab. It was the same table as
    # the draft room's, one click away, differing only in how many columns
    # got squeezed in - so it read as two boards that might disagree. There
    # is one board now: before you start drafting it IS the big board, and
    # as picks come in it thins out into the live room.
    # ONE draft surface. The draft board and mock draft were separate tabs
    # showing the same thing at different moments; both now live inside the
    # draft room, where you're already looking.
    room, news = st.tabs(["🎯 Draft Room", "📰 News & Injuries"])
    with room:
        _render_draft_room(board, settings, ctx)
    with news:
        _render_news(board, settings)
