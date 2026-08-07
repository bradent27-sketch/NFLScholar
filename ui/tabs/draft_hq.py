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

from config import AVAILABLE_SEASONS_WITH_UPCOMING, TAB_PLAYER_SEARCH
from data.draft_sources import (
    ECR_BOARDS, load_ecr_raw, build_ecr_board, fetch_adp, fetch_injury_report,
    fetch_player_news, load_dynasty_values, ecr_age_days,
    ADP_SOURCE_CHOICES, parse_adp_upload, parse_ecr_upload,
)
from data.draft_board import (
    DEFAULT_SCORING, DEFAULT_ROSTER, build_draft_board, recommend_picks,
    roster_needs, snake_pick_numbers, next_pick_for, DRAFTABLE_POSITIONS,
    refresh_pick_context,
)
from data.draft_sim import (
    prepare_sim_pool, init_draft_state, run_until_user_pick, team_on_clock,
    current_round, record_pick, available_players, grade_draft, autopick_for_user,
    run_many_drafts, pick_slot_comparison, optimal_lineup_points,
)
from data.draft_projections import build_projected_board
from data.draft_sos import build_team_sos, attach_sos_to_board, adp_quartiles, WEEK_PRESETS
from data.draft_weekly import attach_consistency
from data.draft_season_sim import grade_roster_wins, simulate_seasons
from data.draft_intel import (
    pick_intel, outcome_distribution, roster_percentile,
    positional_run_pressure, positional_value_add,
)
from data.ffa_import import load_ffa_import, save_ffa_import, merge_ffa_into_board
from data.fantasypros_import import (
    load_cheatsheet, merge_cheatsheet_into_board, save_upload as save_fp_upload,
)
from data.loaders import fetch_sleeper_draft_picks
from data.transforms import parse_pasted_draft_picks, match_names_to_board
from data.utils import calculate_percentile
from ui.styling import style_plain_dataframe, df_auto_height, build_column_help_config
from ui.components import skeleton_loader, switch_tab

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
# ADP and ECR sit side by side: they are the two market/expert reads of the
# same player, and their DISAGREEMENT is the informative part. Split to
# opposite ends of a nineteen-column table that comparison needed a
# horizontal scroll and a memory of the first number.
BOARD_COLUMNS = [
    'Player', 'Pos', 'Team', 'Age', 'Pos Rk', 'Tier', 'FP Tier', 'Auction $',
    'Proj Pts', 'VORP', 'VONA',
    'FFA Rank', 'ADP', 'ECR', 'Value vs ADP', 'Avail Next %', 'Ceiling', 'Floor',
    'Risk', 'Start %', 'Boom %', 'Health', 'SOS', 'Bye',
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

# Every league setting, with the value used before the panel has ever been
# opened.
#
# This table exists because the settings panel is now genuinely absent from
# the page when it's closed, rather than merely collapsed. An st.expander
# still runs its whole body while collapsed, so the old code could read the
# widgets' return values unconditionally; a conditionally-rendered panel
# can't, and Streamlit garbage-collects a widget's session_state entry after
# a run in which that widget wasn't instantiated. So the values live here
# instead, in a plain dict that nothing cleans up, and the widgets read from
# and write back to it. That also means closing the panel genuinely stops
# executing ~50 widgets on every rerun during a draft.
SETTINGS_KEY = 'dhq_cfg'
SETTINGS_OPEN_KEY = 'dhq_settings_open'
ADP_UPLOAD_KEY = 'dhq_adp_upload_df'
ECR_UPLOAD_KEY = 'dhq_ecr_upload_df'
ECR_UPLOAD_ERROR_KEY = 'dhq_ecr_upload_error'

SETTING_DEFAULTS = {
    'teams': 12, 'draft_type': 'Snake', 'slot': 5,
    'qb': 1, 'rb': 2, 'wr': 2, 'te': 1,
    'flex': 1, 'superflex': 0, 'k': 1, 'dst': 1, 'bench': 6,
    'ppr': 1.0, 'te_prem': 0.0, 'pass_td': 4, 'pass_yd': 0.04, 'ppc': 0.0,
    'bonus_mode': 'cumulative',
    'rush_yd': 0.1, 'rec_yd': 0.1, 'rush_td': 6, 'rec_td': 6, 'int': -2, 'fum': -2,
    'board_fmt': 'Redraft 1QB', 'adp_year': AVAILABLE_SEASONS_WITH_UPCOMING[0],
    'adp_source': 'Auto', 'sos_window': None, 'market_weight': 40,
    'uncertainty': 1.0, 'tiers': 8, 'curve_season': AVAILABLE_SEASONS_WITH_UPCOMING[1],
    'ffa_weight': 0,
    # Simulated-opponent ranking blend, as percentages. See
    # data.draft_sim.build_opponent_ranking.
    'bot_adp': 50, 'bot_ecr': 50, 'bot_ffa': 0,
}
for _threshold in (100, 150, 200, 250):
    SETTING_DEFAULTS[f'bonus_rush_{_threshold}'] = 0.0
    SETTING_DEFAULTS[f'bonus_rec_{_threshold}'] = 0.0
for _threshold in (300, 400, 500, 600):
    SETTING_DEFAULTS[f'bonus_pass_{_threshold}'] = 0.0


def _cfg():
    """The persisted league configuration, seeded with defaults on first use."""
    cfg = st.session_state.setdefault(SETTINGS_KEY, {})
    for key, value in SETTING_DEFAULTS.items():
        cfg.setdefault(key, value)
    if cfg.get('sos_window') not in WEEK_PRESETS:
        cfg['sos_window'] = list(WEEK_PRESETS.keys())[0]
    return cfg


def _pick_index(options, value, default=0):
    """Index of a stored choice in a selectbox's options, tolerant of drift."""
    try:
        return list(options).index(value)
    except (ValueError, TypeError):
        return default


def _render_settings_panel(cfg):
    """
    The full-width settings surface, rendered only while it's open.

    Every widget reads its current value out of `cfg` and writes the new one
    straight back, so the dict is the single source of truth and the panel
    can disappear entirely between uses without losing anything.
    """
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**League**")
            cfg['teams'] = st.number_input("Teams", 4, 32, int(cfg['teams']))
            cfg['draft_type'] = st.selectbox(
                "Draft type", ["Snake", "Linear", "Auction"],
                index=_pick_index(["Snake", "Linear", "Auction"], cfg['draft_type']))
            cfg['slot'] = st.number_input("Your draft slot", 1, int(cfg['teams']),
                                          min(int(cfg['slot']), int(cfg['teams'])))
        with c2:
            st.markdown("**Starting lineup**")
            cfg['qb'] = st.number_input("QB", 0, 3, int(cfg['qb']))
            cfg['rb'] = st.number_input("RB", 0, 5, int(cfg['rb']))
            cfg['wr'] = st.number_input("WR", 0, 6, int(cfg['wr']))
            cfg['te'] = st.number_input("TE", 0, 3, int(cfg['te']))
        with c3:
            st.markdown("**Flex & bench**")
            cfg['flex'] = st.number_input("FLEX (RB/WR/TE)", 0, 4, int(cfg['flex']))
            cfg['superflex'] = st.number_input("SUPERFLEX (QB too)", 0, 2, int(cfg['superflex']))
            cfg['k'] = st.number_input("K", 0, 2, int(cfg['k']))
            cfg['dst'] = st.number_input("DST", 0, 2, int(cfg['dst']))
            cfg['bench'] = st.number_input("Bench spots", 0, 20, int(cfg['bench']))
        with c4:
            st.markdown("**Scoring**")
            cfg['ppr'] = st.select_slider("PPR", options=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5],
                                          value=float(cfg['ppr']))
            cfg['te_prem'] = st.select_slider("TE premium (extra per rec)",
                                              options=[0.0, 0.25, 0.5, 0.75, 1.0],
                                              value=float(cfg['te_prem']))
            cfg['pass_td'] = st.number_input("Pass TD", 0, 10, int(cfg['pass_td']))
            cfg['pass_yd'] = st.number_input("Pts/pass yd", 0.0, 0.2, float(cfg['pass_yd']),
                                             step=0.01, format="%.2f")
            cfg['ppc'] = st.number_input("Pts/carry", 0.0, 1.0, float(cfg['ppc']), step=0.05)

        st.markdown("---")
        st.markdown("**Per-game yardage bonuses**")
        st.caption(
            "Points awarded in any single game clearing each threshold — leave at 0 if your "
            "league doesn't use them. These are scored week by week, not off season totals, so "
            "a back with eight 100-yard games is correctly worth more than one with the same "
            "yardage spread evenly. Turning these on genuinely re-prices the board toward "
            "boom-week players."
        )
        cfg['bonus_mode'] = st.radio(
            "When several thresholds are cleared in one game",
            ["cumulative", "highest"], horizontal=True,
            index=_pick_index(["cumulative", "highest"], cfg['bonus_mode']),
            help="Cumulative: a 210-yard game pays the 100, 150 and 200 bonuses (Sleeper/ESPN "
                 "default). Highest: it pays only the 200 bonus.",
        )
        bcols = st.columns(3)
        with bcols[0]:
            st.caption("**Rushing yards**")
            for threshold in (100, 150, 200, 250):
                key = f'bonus_rush_{threshold}'
                cfg[key] = st.number_input(f"{threshold}+ rush yds", 0.0, 20.0,
                                           float(cfg[key]), step=0.5)
        with bcols[1]:
            st.caption("**Receiving yards**")
            for threshold in (100, 150, 200, 250):
                key = f'bonus_rec_{threshold}'
                cfg[key] = st.number_input(f"{threshold}+ rec yds", 0.0, 20.0,
                                           float(cfg[key]), step=0.5)
        with bcols[2]:
            st.caption("**Passing yards**")
            for threshold in (300, 400, 500, 600):
                key = f'bonus_pass_{threshold}'
                cfg[key] = st.number_input(f"{threshold}+ pass yds", 0.0, 20.0,
                                           float(cfg[key]), step=0.5)

        st.markdown("---")
        c5, c6, c7 = st.columns(3)
        with c5:
            st.markdown("**More scoring**")
            cfg['rush_yd'] = st.number_input("Pts/rush yd", 0.0, 0.5, float(cfg['rush_yd']),
                                             step=0.01, format="%.2f")
            cfg['rec_yd'] = st.number_input("Pts/rec yd", 0.0, 0.5, float(cfg['rec_yd']),
                                            step=0.01, format="%.2f")
            cfg['rush_td'] = st.number_input("Rush TD", 0, 10, int(cfg['rush_td']))
            cfg['rec_td'] = st.number_input("Rec TD", 0, 10, int(cfg['rec_td']))
            cfg['int'] = st.number_input("INT thrown", -6, 0, int(cfg['int']))
            cfg['fum'] = st.number_input("Fumble lost", -6, 0, int(cfg['fum']))
        with c6:
            st.markdown("**Board & market**")
            boards = list(ECR_BOARDS.keys())
            cfg['board_fmt'] = st.selectbox("Ranking board", boards,
                                            index=_pick_index(boards, cfg['board_fmt']))
            ecr_upload = st.file_uploader(
                "Upload FantasyPros rankings CSV", type=["csv"], key="dhq_ecr_upload",
                help="Overrides the live consensus feed. The live one comes from a nightly "
                     "third-party mirror of FantasyPros, and that mirror can stall — it has. "
                     "Export the Draft Rankings view from FantasyPros and drop it here to be "
                     "certain the board is current. Include BEST/WORST/STD DEV if the export "
                     "offers them; that spread is what Upside, Bust and Risk are built from.",
            )
            if ecr_upload is not None:
                parsed, ecr_error = parse_ecr_upload(ecr_upload)
                # Parsed and held rather than kept as a file handle, for the
                # same reason as the ADP upload: this widget stops existing
                # the moment the panel closes.
                st.session_state[ECR_UPLOAD_KEY] = None if parsed.empty else parsed
                st.session_state[ECR_UPLOAD_ERROR_KEY] = ecr_error
            if st.session_state.get(ECR_UPLOAD_ERROR_KEY):
                st.error(st.session_state[ECR_UPLOAD_ERROR_KEY])
            held_ecr = st.session_state.get(ECR_UPLOAD_KEY)
            if held_ecr is not None and not held_ecr.empty:
                spread = 'with' if held_ecr['ECR SD'].notna().any() else 'without'
                st.caption(f"✅ Using your export — {len(held_ecr)} players, {spread} "
                           "expert spread")
                if st.button("Back to the live feed", key="dhq_ecr_clear"):
                    st.session_state[ECR_UPLOAD_KEY] = None
                    st.session_state[ECR_UPLOAD_ERROR_KEY] = None
                    st.rerun()
            cfg['adp_year'] = st.selectbox(
                "ADP season", AVAILABLE_SEASONS_WITH_UPCOMING,
                index=_pick_index(AVAILABLE_SEASONS_WITH_UPCOMING, cfg['adp_year']))
            cfg['adp_source'] = st.selectbox(
                "ADP source", ADP_SOURCE_CHOICES,
                index=_pick_index(ADP_SOURCE_CHOICES, cfg['adp_source']),
                help="Auto tries your uploaded CSV, then an FFA import, then FantasyPros' live "
                     "consensus, then the same consensus recovered from the local ranking "
                     "exports. Fantasy Football Calculator is a manual choice only — its ADP "
                     "comes from free mock drafts on its own site and slides tight ends and "
                     "quarterbacks well past where real leagues take them.",
            )
            windows = list(WEEK_PRESETS.keys())
            cfg['sos_window'] = st.selectbox(
                "Schedule window", windows, index=_pick_index(windows, cfg['sos_window']),
                help="Strength of schedule is graded per position group — backs against run "
                     "defenses, passers and pass catchers against pass defenses.")
            adp_upload = st.file_uploader(
                "Upload ADP CSV (overrides live)", type=["csv"], key="dhq_adp_upload",
                help="Any CSV with a player-name column and an ADP/rank column. Overrides every "
                     "live source.",
            )
            if adp_upload is not None:
                # Parsed and stashed immediately rather than handed on as a
                # file object, because the uploader widget itself disappears
                # the moment this panel is closed - and an ADP source that
                # silently reverts when you collapse the settings would be a
                # nasty thing to discover mid-draft.
                parsed = parse_adp_upload(adp_upload)
                st.session_state[ADP_UPLOAD_KEY] = parsed if not parsed.empty else None
            if st.session_state.get(ADP_UPLOAD_KEY) is not None:
                st.caption(f"✅ {len(st.session_state[ADP_UPLOAD_KEY])} rows held from your upload")
                if st.button("Clear uploaded ADP", key="dhq_adp_clear"):
                    st.session_state[ADP_UPLOAD_KEY] = None
                    st.rerun()
            cfg['market_weight'] = st.slider(
                "Market blend", 0, 100, int(cfg['market_weight']), 5,
                help="How much the board's ORDER defers to ADP. 0 = pure model, 100 = pure ADP. "
                     "VORP itself is never blended — only the ordering moves.",
            )
            st.caption(
                "Value-based drafting and the market disagree hardest at QB: with replacement "
                "set at the last starting QB, the model prices an elite QB as a top-15 overall "
                "pick in a 1QB league and real drafts take him ~20 picks later. This blend lets "
                "you decide how much deference the market gets."
            )
        with c7:
            st.markdown("**Model**")
            cfg['uncertainty'] = st.slider(
                "Projection uncertainty", 0.5, 2.0, float(cfg['uncertainty']), 0.1,
                help="Multiplier on the measured spread of where players actually finish "
                     "relative to their consensus rank. 1.0 = use the measured value as-is.",
            )
            st.caption(
                "The baseline is measured, not guessed: from your own weekly history, how far "
                "players land from where they ranked, per position and rank. Top-6 QBs and TEs "
                "come in around ±10 finish slots; RBs and WRs scatter more than twice as far. "
                "This slider scales that. Higher widens Ceiling/Floor and flattens the board."
            )
            cfg['tiers'] = st.slider("Max tiers per position", 3, 12, int(cfg['tiers']))
            seasons = AVAILABLE_SEASONS_WITH_UPCOMING[1:]
            cfg['curve_season'] = st.selectbox(
                "Projection baseline through", seasons,
                index=_pick_index(seasons, cfg['curve_season']),
                help="Last completed season used to build the usage curves and player rates.",
            )

        st.markdown("---")
        s1, s2 = st.columns([1, 1])
        with s1:
            st.markdown("**How simulated opponents rank players**")
            st.caption(
                "Mock-draft opponents order the board by a weighted blend of these three. Pure "
                "ADP gives you a room that has memorised the market and never has an opinion; "
                "pure consensus rank gives you a room of analysts who all read the same page and "
                "ignore each other. Real drafters do some of both, which is why this starts at "
                "an even split. Weights are normalized, so only their ratio matters."
            )
            cfg['bot_adp'] = st.slider("Weight on ADP", 0, 100, int(cfg['bot_adp']), 5)
            cfg['bot_ecr'] = st.slider("Weight on consensus rank (ECR)", 0, 100,
                                       int(cfg['bot_ecr']), 5)
            cfg['bot_ffa'] = st.slider(
                "Weight on FFA rank", 0, 100, int(cfg['bot_ffa']), 5,
                help="Only has an effect once an FFA export has been imported below.")
        with s2:
            st.markdown("**Import Fantasy Football Advice projections** (optional)")
            st.caption(
                "Drop in an FFA player export and the board will use their analysts' projected "
                "STAT LINE — carries, yards, receptions — re-scored under your league settings, "
                "plus their FFA Value and written player notes. Importing the stat line rather "
                "than their point total is what keeps it correct: their export is half-PPR, so "
                "reading their points straight off would be wrong in any other format."
            )
            ffa_upload = st.file_uploader("FFA players JSON", type=["json"], key="dhq_ffa_upload")
            st.markdown("**Import a FantasyPros cheat sheet** (optional)")
            st.caption(
                "Adds auction dollar values, their analysts' tiers alongside this board's own, "
                "and a written note per player. Auction values are the piece nothing else here "
                "can produce — VORP says who is worth more, not whether that's a $47 player or "
                "a $12 one, which is the entire question in an auction. Drop in the "
                "getCheatSheet response; add the draft room's player array too and every player "
                "resolves instead of ~80%."
            )
            fp_upload = st.file_uploader("FantasyPros cheat sheet / player array",
                                         type=["json", "html", "txt"], key="dhq_fp_upload")
            if fp_upload is not None:
                kind, fp_error = save_fp_upload(fp_upload)
                if fp_error:
                    st.error(fp_error)
                elif kind:
                    st.success(f"Saved the {kind}.")
            cfg['ffa_weight'] = st.slider(
                "Blend FFA stat line into projections", 0, 100, int(cfg['ffa_weight']), 5,
                help="100 = use their projections outright, 0 = keep this app's own and take "
                     "only their notes and value score.")

    return ffa_upload


def _settings_from_cfg(cfg, ffa_upload=None):
    """Turn the stored configuration into the dict the whole engine keys off."""
    scoring = dict(DEFAULT_SCORING)
    scoring.update({
        'rec': float(cfg['ppr']), 'te_premium': float(cfg['te_prem']),
        'pass_td': float(cfg['pass_td']), 'pass_yd': float(cfg['pass_yd']),
        'rush_att': float(cfg['ppc']), 'rush_yd': float(cfg['rush_yd']),
        'rec_yd': float(cfg['rec_yd']), 'rush_td': float(cfg['rush_td']),
        'rec_td': float(cfg['rec_td']), 'pass_int': float(cfg['int']),
        'fumble_lost': float(cfg['fum']), 'bonus_mode': cfg['bonus_mode'],
    })
    scoring.update({k: float(v) for k, v in cfg.items() if k.startswith('bonus_')
                    and k != 'bonus_mode'})
    roster = dict(DEFAULT_ROSTER)
    roster.update({'QB': int(cfg['qb']), 'RB': int(cfg['rb']), 'WR': int(cfg['wr']),
                   'TE': int(cfg['te']), 'K': int(cfg['k']), 'DST': int(cfg['dst']),
                   'FLEX': int(cfg['flex']), 'SUPERFLEX': int(cfg['superflex']),
                   'BENCH': int(cfg['bench'])})

    return {
        'num_teams': int(cfg['teams']), 'roster': roster, 'scoring': scoring,
        'draft_type': cfg['draft_type'],
        'my_slot': int(min(cfg['slot'], cfg['teams'])), 'board_format': cfg['board_fmt'],
        'adp_year': int(cfg['adp_year']),
        'adp_upload': st.session_state.get(ADP_UPLOAD_KEY),
        'adp_source': cfg['adp_source'],
        'uncertainty': float(cfg['uncertainty']), 'tiers': int(cfg['tiers']),
        'baseline_season': int(cfg['curve_season']),
        'market_weight': float(cfg['market_weight']) / 100.0,
        'sos_window': cfg['sos_window'], 'ffa_upload': ffa_upload,
        'ffa_weight': float(cfg['ffa_weight']) / 100.0,
        'bot_weights': {'ADP': float(cfg['bot_adp']), 'ECR': float(cfg['bot_ecr']),
                        'FFA': float(cfg['bot_ffa'])},
        # Part of the board cache key, so swapping the consensus source
        # rebuilds the board instead of serving the previous one.
        'ecr_signature': _ecr_signature(),
    }


def _ecr_signature():
    """A hashable summary of which consensus rankings are in play."""
    uploaded = st.session_state.get(ECR_UPLOAD_KEY)
    if uploaded is None or uploaded.empty:
        return ('live', 0)
    return ('upload', int(len(uploaded)), float(uploaded['ECR'].sum()))


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
        settings.get('adp_source', 'Auto'), settings.get('ecr_signature'),
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
    # How his points ARRIVE, which nothing else on the board describes: every
    # other spread here is a season-total percentile.
    board = attach_consistency(board, _settings)

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
    # An uploaded FantasyPros export wins over the live mirror outright. It's
    # a deliberate act by someone who has just looked at the rankings, which
    # is better evidence of freshness than anything this code can check - and
    # the mirror is exactly the thing that can silently stop updating.
    uploaded_ecr = st.session_state.get(ECR_UPLOAD_KEY)
    if uploaded_ecr is not None and not uploaded_ecr.empty:
        ecr_board = uploaded_ecr
        status['ecr'] = None
        status['ecr_source'] = 'your FantasyPros export'
        status['ecr_age'] = None
    else:
        ecr_raw, ecr_err = load_ecr_raw()
        status['ecr'] = ecr_err
        if ecr_raw is None or ecr_raw.empty:
            return pd.DataFrame(), {}, pd.DataFrame(), {}, status
        ecr_board = build_ecr_board(ecr_raw, settings['board_format'])
        status['ecr_source'] = 'FantasyPros via DynastyProcess'
        status['ecr_age'] = ecr_age_days(ecr_raw)

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

    board, meta = _cached_board(ecr_board, adp_df, ffa_df, settings,
                                _board_cache_key(settings, next_pick, adp_meta))

    # Merged after the cached build rather than inside it: this is pure
    # annotation - auction values, their analysts' tiers, their notes - and
    # none of it feeds a calculation, so it has no business invalidating a
    # board rebuild when the file changes.
    sheet, sheet_err = load_cheatsheet()
    status['cheatsheet'] = {'rows': int(len(sheet)), 'error': sheet_err}
    if not sheet.empty:
        board, sheet_meta = merge_cheatsheet_into_board(board, sheet)
        status['cheatsheet'].update(sheet_meta)
    return board, meta, adp_df, adp_meta, status


def _render_source_status(status, meta):
    """
    One honest line about where every number on this board came from.

    Deliberately terse, with the explanations moved into the tooltip: this
    now shares a row with the settings button rather than owning a line of
    its own, and a four-sentence paragraph there would wrap to three lines
    and give back all the vertical space that layout just bought. The short
    form still names every source and still flags every degradation - it
    just stops explaining each one in place.
    """
    bits, details = [], []
    if status.get('ecr'):
        bits.append("⚠️ No rankings")
        details.append(f"Rankings source unreachable ({status['ecr']}) — the board is empty.")
    elif status.get('ecr_source') == 'your FantasyPros export':
        bits.append("✅ ECR: your export")
        details.append("Consensus rankings are coming from the CSV you uploaded, not the live "
                       "mirror. Clear it in League Settings to go back to the feed.")
    else:
        age = status.get('ecr_age')
        stale = age is not None and age >= 3
        bits.append(f"{'⚠️' if stale else '✅'} FantasyPros ECR" + (f" ({age}d old)" if stale else ""))
        if stale:
            details.append(
                f"Consensus rankings are {age} days old. They come from DynastyProcess's "
                "nightly mirror of FantasyPros, and that mirror has stopped refreshing — "
                "there is no fresher copy to fetch. Injuries and depth-chart news since then "
                "aren't in them. Upload a FantasyPros export in League Settings to override it.")

    adp_meta = status.get('adp') or {}
    if adp_meta.get('error'):
        # Truncated: a blocked-network failure returns a full urllib3
        # ProxyError repr, several hundred characters of stack detail that
        # buries the one line that matters ("no ADP, here's what to do").
        reason = str(adp_meta['error']).split('(')[0].strip(' :,)').strip()[:90]
        bits.append("⚠️ No ADP")
        details.append(
            f"No ADP available ({adp_meta.get('source', 'unknown')}: {reason}). "
            "Value-vs-market, availability and VONA are blank — pick another ADP source or "
            "upload a CSV in League Settings to turn them back on."
        )
    else:
        src = adp_meta.get('source', 'ADP')
        bits.append(f"✅ ADP: {src}")
        if adp_meta.get('note'):
            details.append(f"{src} — {adp_meta['note']}")
        if (adp_meta.get('teams') and adp_meta.get('requested_teams')
                and adp_meta['teams'] != adp_meta['requested_teams']):
            details.append(f"Showing {adp_meta['teams']}-team ADP for your "
                           f"{adp_meta['requested_teams']}-team league (nearest published size).")

    projection = meta.get('projection') or {}
    if projection.get('volume_projections'):
        n = projection.get('players_with_history', 0)
        bits.append(f"✅ Projections ({n:,})")
        details.append(f"Stat-line projections built from {n:,} players of local history.")
    sheet = status.get('cheatsheet') or {}
    if sheet.get('error'):
        bits.append("⚠️ Cheat sheet")
        details.append(f"FantasyPros cheat sheet: {str(sheet['error'])[:140]}")
    elif sheet.get('rows'):
        bits.append(f"✅ FP sheet {sheet.get('matched', 0)}/{sheet['rows']}")
        details.append(f"FantasyPros cheat sheet matched {sheet.get('matched', 0)} of "
                       f"{sheet['rows']} players — auction values, their tiers and analyst notes.")
    ffa = status.get('ffa') or {}
    if ffa.get('error'):
        bits.append("⚠️ FFA import")
        details.append(f"FFA import: {str(ffa['error'])[:120]}")
    elif ffa.get('rows'):
        matched = (meta.get('ffa') or {}).get('matched', 0)
        bits.append(f"✅ FFA {matched}/{ffa['rows']}")
        details.append(f"FFA import matched {matched} of {ffa['rows']} players onto the board.")
    st.caption("  •  ".join(bits), help="\n\n".join(details) or None)


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

    # Where the ROOM is, which is not the same as how many picks you've
    # logged. Someone who only marks their own selections has logged four
    # picks while the draft has actually run 40 - and the availability model
    # conditions on the current pick, so believing the low number would have
    # it reporting round-1 odds in round 5. Your own last pick number is a
    # hard floor on how far the draft has gone, whatever else is tracked.
    current = picks_made + 1
    if taken_by_me:
        current = max(current, my_picks[taken_by_me - 1] + 1)
    if nxt is not None:
        # Never past your own next pick: you can't be on the clock for a
        # pick that hasn't come around yet.
        current = min(current, nxt)

    return {'picks_made': picks_made, 'rounds': rounds, 'my_picks': my_picks,
            'next_pick': nxt, 'on_clock': picks_made + 1, 'current_pick': current,
            'round': (current - 1) // max(settings['num_teams'], 1) + 1}


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
    my_slot = _cfg().get('slot', 1)
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


# The per-row action columns, as (column label, action id, tooltip). These
# replace the old select-a-row-then-click-a-button flow entirely: marking a
# player gone is the single most repeated motion in a draft (eleven of every
# twelve picks in a 12-team league are someone else's), and making it two
# clicks and a scan for the right button was the wrong cost to pay dozens of
# times an hour.
BOARD_ACTIONS = {
    'Live draft': [
        ('✅', 'mine', 'Draft this player to YOUR team'),
        ('❌', 'gone', 'Mark this player taken by another team'),
        ('🔍', 'profile', 'Open this player in Player Search'),
    ],
    'Mock draft': [
        ('✅', 'draft', 'Draft this player'),
        ('🔍', 'profile', 'Open this player in Player Search'),
    ],
}
BOARD_NONCE_KEY = 'dhq_board_nonce'
BOARD_OPEN_KEY = 'dhq_board_open'


BOARD_ROWS_KEY = 'dhq_board_rows'


def _board_editor_changed(widget_key, action_map):
    """
    Widget callback for the board grid, which exists solely to make the 🔍
    row button able to change tabs.

    switch_tab writes st.session_state['active_tab'], and app.py has already
    instantiated that keyed st.tabs widget by the time any tab body runs -
    Streamlit only permits writes to a keyed widget's state from a callback,
    which runs in its own pre-script phase before the next run's st.tabs()
    call. So the jump has to happen here rather than in the normal flow. The
    drafting actions need no such privilege and are handled in the script
    body, where the draft context they operate on actually exists.
    """
    edits = (st.session_state.get(widget_key) or {}).get('edited_rows') or {}
    players = st.session_state.get(BOARD_ROWS_KEY) or []
    for row_index, changes in edits.items():
        for label, value in (changes or {}).items():
            if not value or action_map.get(label) != 'profile':
                continue
            try:
                player = players[int(row_index)]
            except (ValueError, IndexError):
                continue
            # Bumped so the tick doesn't survive the round trip and re-fire
            # the moment you come back to this tab.
            _bump_board_nonce()
            switch_tab(TAB_PLAYER_SEARCH, jump_to_player=player)
            return


def _bump_board_nonce():
    """
    Rotate the board widget's key so its pending edits are discarded.

    A data editor keeps the cells you ticked in its own widget state, keyed
    by ROW INDEX. Act on a tick and rerun, and that same row index is still
    ticked - but the board has thinned by one, so the tick now points at a
    different player and fires again immediately. Giving the widget a fresh
    key after every action makes it a brand-new widget with no history,
    which is the only reliable way to make a one-shot row button out of a
    persistent checkbox.
    """
    st.session_state[BOARD_NONCE_KEY] = st.session_state.get(BOARD_NONCE_KEY, 0) + 1


def _render_board_grid(available, key_prefix, mode, next_pick=None, columns=None, row_limit=40):
    """
    The sortable, filterable board grid, returning (action, player) for
    whichever row button was pressed.

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
    with c2:
        sort_by = st.selectbox("Sort by", ['Board Rank', 'VORP', 'VONA', 'Proj Pts',
                                           'Value vs ADP', 'ADP', 'ECR', 'Ceiling'],
                               key=f"{key_prefix}_sort")
    with c3:
        limit = st.number_input("Rows", 10, 400, row_limit, step=10, key=f"{key_prefix}_limit")

    ascending = sort_by in ('ADP', 'ECR', 'Board Rank')
    if sort_by in available.columns:
        available = available.sort_values(sort_by, ascending=ascending, na_position='last')
    view = available.head(int(limit))

    cols = [c for c in (columns or BOARD_COLUMNS) if c in view.columns]
    # Player stays a real column rather than the index so the action buttons
    # can sit to its LEFT, which is where a hand already is when it's reading
    # down a board. It's pinned instead, so it stays put on horizontal
    # scroll exactly as the index used to.
    display = view[cols].reset_index(drop=True)
    actions = BOARD_ACTIONS.get(mode, BOARD_ACTIONS['Mock draft'])
    for offset, (label, _, _) in enumerate(actions):
        display.insert(offset, label, False)

    pct_cols = {}
    for c in ('VORP', 'VONA', 'Proj Pts'):
        if c in display.columns and display[c].notna().any():
            pct_cols[c] = calculate_percentile(display, c)
    diverging = {}
    if 'Value vs ADP' in display.columns and display['Value vs ADP'].notna().any():
        max_abs = display['Value vs ADP'].abs().max()
        if max_abs and max_abs > 0:
            diverging['Value vs ADP'] = max_abs

    column_config = build_column_help_config(display, pinned_cols=['Pos', 'Team'])
    column_config['Player'] = st.column_config.TextColumn("Player", pinned=True)
    for label, _, _tooltip in actions:
        # No help= on these three. The tooltip renders an ⓘ badge INSIDE the
        # header cell, and next to a one-character emoji in a 44px column
        # there isn't room for both - the header came out as a clipped icon
        # and a sliver of emoji. What each button does is written in the
        # caption above the table, where there's room to say it properly.
        column_config[label] = st.column_config.CheckboxColumn(label, width="small",
                                                               pinned=True)
    if 'Avail Next %' in display.columns:
        column_config['Avail Next %'] = st.column_config.NumberColumn(
            "Avail Next %", format="%d%%",
            help=f"Chance he lasts to your next pick (#{next_pick})" if next_pick else "No next pick",
        )
    if 'Risk' in display.columns:
        column_config['Risk'] = st.column_config.NumberColumn(
            "Risk", format="%d%%",
            help="Width of the ceiling-to-floor band as a share of the projection")
    if 'Health' in display.columns:
        column_config['Health'] = st.column_config.TextColumn(
            "Health",
            help="Games played last season. A back or tight end who lost much of a year is "
                 "marked down twice over — likelier to miss time again, and worse per game "
                 "when he plays.")
    if 'Auction $' in display.columns:
        column_config['Auction $'] = st.column_config.NumberColumn(
            "Auction $", format="$%d",
            help="FantasyPros auction value under a $200 budget, from your imported cheat sheet")
    if 'FP Tier' in display.columns:
        column_config['FP Tier'] = st.column_config.NumberColumn(
            "FP Tier", format="%d",
            help="FantasyPros analysts' own tier. Shown beside this board's computed Tier "
                 "on purpose — where they disagree is the interesting part.")

    action_labels = [label for label, _, _ in actions]
    action_map = {label: action_id for label, action_id, _ in actions}
    # Stashed for the callback, which only ever sees a row INDEX and has no
    # other way back to a player name.
    st.session_state[BOARD_ROWS_KEY] = display['Player'].tolist()
    widget_key = f"{key_prefix}_editor_{st.session_state.get(BOARD_NONCE_KEY, 0)}"
    edited = st.data_editor(
        style_plain_dataframe(display, numeric_pct_cols=pct_cols, diverging_cols=diverging),
        width="stretch", height=df_auto_height(min(len(display), 26)),
        hide_index=True, column_config=column_config,
        disabled=[c for c in display.columns if c not in action_labels],
        key=widget_key, on_change=_board_editor_changed, args=(widget_key, action_map),
    )

    for label, action_id, _ in actions:
        if action_id == 'profile':
            continue
        if label not in edited.columns:
            continue
        hits = edited.index[edited[label].fillna(False).astype(bool)]
        if len(hits):
            return action_id, str(display.loc[hits[0], 'Player'])
    return None, None


def _render_positional_scarcity(available, settings):
    """
    How many players above replacement are left at each position.

    The single most useful "what should I be worried about" readout on a
    draft board: it turns "should I take a TE?" into a number. Two startable
    tight ends left and eleven receivers is a completely different situation
    from the reverse, and it's invisible from a ranked list.

    Takes the already-thinned available pool rather than the full board, so
    it describes whichever draft is actually on screen - reading the live
    tracker here reported the preseason board's scarcity in the middle of a
    mock, i.e. none at all.
    """
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



def _render_roster_outlook(board, settings, dc):
    """
    What your roster is worth in the currency the league actually pays out
    in: wins.

    A projected point total is a real answer to the wrong question. Seasons
    are seventeen head-to-head games, so a steadier roster beats a
    boom-and-bust one of identical projection - measurably, by about half a
    win - and no points figure can show that. This runs the season a few
    hundred times against a field of competently-drafted teams and reports
    the record.

    Run on demand rather than on every pick: it's a second of work, and a
    board that stalls each time you mark a player gone is worse than one that
    waits to be asked.
    """
    roster = dc.get('my_roster') or []
    if not roster:
        return
    st.markdown("---")
    signature = tuple((p.get('Player'), str(p.get('Pos', '')).upper(),
                       float(p['Proj Pts']) if pd.notna(p.get('Proj Pts')) else None,
                       float(p['Bye']) if pd.notna(p.get('Bye')) else None)
                      for p in roster)

    if st.button("📈 Project my season", key="dhq_wins_run", width="stretch"):
        with st.spinner("Simulating seasons..."):
            st.session_state['dhq_wins'] = grade_roster_wins(
                board, settings, signature, dc['my_slot'])
        st.session_state['dhq_wins_sig'] = signature

    outlook = st.session_state.get('dhq_wins')
    if not outlook:
        st.caption("Projects your record against a field of drafted teams — wins, not points.")
        return
    stale = st.session_state.get('dhq_wins_sig') != signature
    st.markdown(
        "<style>.wo{display:flex;gap:14px;padding:6px 10px;border-radius:6px;"
        "background:rgba(255,255,255,0.045);border-left:3px solid rgba(0,255,249,0.55)}"
        ".wo-b{display:flex;flex-direction:column;line-height:1.15}"
        ".wo-k{font-size:9px;letter-spacing:.09em;opacity:.6;font-weight:700}"
        ".wo-v{font-size:17px;font-weight:700;font-variant-numeric:tabular-nums}</style>"
        f"<div class='wo'>"
        f"<div class='wo-b'><span class='wo-k'>WINS</span>"
        f"<span class='wo-v'>{outlook['wins']:.1f}</span></div>"
        f"<div class='wo-b'><span class='wo-k'>PLAYOFFS</span>"
        f"<span class='wo-v'>{outlook['playoff_pct']:.0f}%</span></div>"
        f"<div class='wo-b'><span class='wo-k'>IN LEAGUE</span>"
        f"<span class='wo-v'>{outlook['rank']}/{outlook['teams']}</span></div>"
        "</div>", unsafe_allow_html=True)
    st.caption(
        f"Typical record {outlook['wins_p25']:.0f}–{outlook['wins_p75']:.0f} wins, "
        f"{outlook['points']:,.0f} pts. The field averages {outlook['field_wins']:.1f}."
        + ("  ⚠️ Out of date — you've drafted since." if stale else ""))


def _render_pick_odds(board, settings, ctx, dc=None):
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
        # Conditioned on THIS draft, whichever one is running. Reading the
        # live tracker while a mock is open would simulate forward from a
        # draft that isn't on screen.
        if dc and dc['mode'] == 'Mock draft':
            signature = tuple((p['Player'], p.get('team') == dc['my_slot'])
                              for p in dc['picks'])
            my_slot = dc['my_slot']
        else:
            signature = tuple((p['Player'], bool(p.get('mine'))) for p in _drafted_list())
            my_slot = settings['my_slot']
        with st.spinner(f"Simulating {int(n_sims)} drafts forward from here..."):
            st.session_state['dhq_intel'] = pick_intel(
                board, settings, my_slot, ctx['rounds'], signature,
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
    # Two independent sets of analyst notes where both exist. Shown side by
    # side rather than merged: they're written by different people at
    # different times, and where they disagree that IS the read.
    written = [(source, text) for source, text in
               (('FFA', notes), ('FantasyPros', row.get('FP Note')))
               if isinstance(text, str) and text.strip()]
    if written:
        label = " / ".join(source for source, _ in written)
        with st.expander(f"Analyst notes — {selected} ({label})", expanded=False):
            for source, text in written:
                st.markdown(f"**{source}**")
                st.write(text)



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
        # width:100% + table-layout:fixed is what makes the board actually
        # FILL its container. With cells pinned to a fixed 118px the table
        # was as wide as 12 x 118px and no wider, leaving a quarter of the
        # expander empty on a wide screen while the names inside were still
        # being truncated. Fixed layout divides the real width evenly
        # instead, and min-width on the table keeps the horizontal scroll
        # for genuinely large leagues rather than crushing 16 teams into
        # unreadable slivers.
        ".db{border-collapse:separate;border-spacing:3px;font-size:11px;width:100%;"
        "table-layout:fixed;min-width:760px}"
        ".db th{font-size:10px;letter-spacing:.06em;text-transform:uppercase;opacity:.7;"
        "padding:2px 4px;white-space:nowrap}"
        ".db th.db-mine{color:#00fff9;opacity:1}"
        ".db-rnd{font-size:10px;opacity:.5;text-align:center;width:22px}"
        ".db-cell{padding:4px 6px;border-radius:4px;vertical-align:top}"
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


def _current_position_filter():
    """
    The active filter as (positions, label), without drawing anything.

    Split from the buttons because the recommendation banner is scoped to
    this filter but renders ABOVE it, so the state has to be readable before
    the controls exist.
    """
    current = st.session_state.get(POS_FILTER_KEY, 'All')
    return dict(POSITION_FILTERS).get(current), current


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


def _render_recent_picks_strip(picks, num_teams, current_pick, round_no=None,
                               next_pick=None, window=11):
    """
    The last several picks as colored cards, the upcoming slots as empty
    ones, and where you are in the draft pinned to the right of them.

    This is the thing a drafter actually keeps glancing at - not a table of
    everyone taken, just "what just happened, what's about to, and when am I
    up". Showing the empty upcoming boxes alongside is what makes it readable
    as a position on the clock rather than a log: you can see how many picks
    stand between you and your turn without counting.

    The pick/round/next-pick readout lives INSIDE this strip rather than in a
    row of metric tiles above it. Three st.metric boxes spend an enormous
    amount of screen height to display three short numbers, and those numbers
    are describing exactly the same thing the strip already draws - so they
    belong on the same line as it, in the space at its right end.
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
            short = name if len(name) <= 16 else name.split()[-1][:15]
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

    if round_no is None:
        round_no = (int(current_pick) - 1) // max(int(num_teams), 1) + 1
    # "In N picks" rather than only the pick number: on the clock you're
    # deciding whether a player comes back to you, and the answer is about
    # the GAP, not the absolute pick number. Working it out by subtracting
    # two three-digit numbers is exactly the arithmetic a draft clock
    # shouldn't make you do.
    away = max(0, int(next_pick) - int(current_pick)) if next_pick else None
    stats = [('PICK', str(int(current_pick))), ('ROUND', str(int(round_no))),
             ('YOU’RE UP', str(int(next_pick)) if next_pick else '—'),
             ('IN', 'NOW' if away == 0 else (f"{away}" if away is not None else '—'))]
    status = "".join(
        f"<div class='rp-stat'><span class='rp-stat-k'>{k}</span>"
        f"<span class='rp-stat-v'>{v}</span></div>" for k, v in stats)

    st.markdown(
        "<style>"
        ".rp-row{display:flex;gap:10px;align-items:stretch;padding:2px 0 8px 0}"
        ".rp-strip{display:flex;gap:5px;overflow-x:auto;flex:1;min-width:0}"
        ".rp-card{min-width:112px;max-width:112px;border-radius:6px;padding:6px 8px;"
        "border-left:3px solid rgba(255,255,255,0.15);font-size:12px}"
        ".rp-empty{background:rgba(255,255,255,0.035);opacity:.55}"
        ".rp-clock{outline:1px solid rgba(0,255,249,0.7);opacity:1}"
        ".rp-num{font-size:10px;letter-spacing:.08em;opacity:.6;font-weight:700}"
        ".rp-name{font-weight:600;font-size:13px;white-space:nowrap;overflow:hidden;"
        "text-overflow:ellipsis}"
        ".rp-sub{font-size:10px;opacity:.9;letter-spacing:.03em;min-height:12px}"
        ".rp-status{flex:0 0 auto;display:flex;gap:14px;align-items:center;padding:4px 12px;"
        "border-radius:6px;background:rgba(255,255,255,0.045);"
        "border-left:3px solid rgba(0,255,249,0.55)}"
        ".rp-stat{display:flex;flex-direction:column;line-height:1.15}"
        ".rp-stat-k{font-size:9px;letter-spacing:.09em;opacity:.6;font-weight:700}"
        ".rp-stat-v{font-size:17px;font-weight:700;font-variant-numeric:tabular-nums}"
        "</style>"
        f"<div class='rp-row'><div class='rp-strip'>{''.join(cards)}</div>"
        f"<div class='rp-status'>{status}</div></div>", unsafe_allow_html=True)


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
        pool, order_col, _ = prepare_sim_pool(board, settings.get('bot_weights'))
        avail = available_players(state, pool) if not pool.empty else board.iloc[0:0]
        my_roster = [dict(p) for p in state['rosters'][state['my_slot']]]
        # Snake order, not "this pick plus one lap": at the turn those differ
        # by up to a full round, which is exactly where the wait-or-take
        # decision is hardest.
        my_picks = snake_pick_numbers(state['my_slot'], state['num_teams'], state['rounds'],
                                      draft_type=state.get('draft_type', 'Snake'))
        avail_pick = _availability_pick(my_picks, state['pick_no'])
        return {
            'mode': mode, 'state': state, 'pool': pool, 'order_col': order_col,
            'picks': state['picks'], 'my_roster': my_roster,
            'available': refresh_pick_context(avail, avail_pick, state['pick_no']),
            'current_pick': state['pick_no'],
            'next_pick': state['pick_no'], 'avail_pick': avail_pick,
            'picks_left': max(0, state['rounds'] - len(my_roster)),
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
    available = board[~board['Player'].isin(_drafted_names())]
    current = ctx['current_pick']
    avail_pick = _availability_pick(ctx['my_picks'], current)
    return {
        'mode': mode, 'state': None, 'pool': None, 'order_col': None,
        'picks': picks, 'my_roster': _my_roster(),
        'available': refresh_pick_context(available, avail_pick, current),
        'current_pick': current,
        'next_pick': ctx['next_pick'], 'avail_pick': avail_pick,
        'picks_left': max(0, ctx['rounds'] - len(_my_roster())),
        'num_teams': settings['num_teams'], 'my_slot': settings['my_slot'],
        'complete': False, 'on_clock_me': True,
    }


def _availability_pick(my_picks, current_pick):
    """
    The pick number the availability model measures TO: your first pick
    strictly after where the draft is now.

    Strictly after, not "at or after", because the question the column
    answers is "if I pass on him with this pick, does he come back to me".
    When you're the one on the clock those two readings differ by a full
    turn, and the 'at or after' version degenerates - it compares the board
    to the very pick it's already conditioned on, which makes every player
    read 100%.
    """
    for number in (my_picks or []):
        if number > int(current_pick):
            return number
    return None


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
    _bump_board_nonce()
    st.rerun()


def _handle_board_action(action, player, dc, settings, reach):
    """
    Route a row button on the board to the thing it means.

    'profile' is deliberately absent: jumping tabs has to happen inside a
    widget callback (see _board_editor_changed), and by the time this runs
    the jump has already been made.
    """
    if action in ('mine', 'draft'):
        _commit_pick(dc, settings, player, mine=True, reach=reach)
    elif action == 'gone':
        _commit_pick(dc, settings, player, mine=False, reach=reach)



def _render_positional_value_add(board, settings, dc):
    """
    Which position this pick should go to, in points and as a share of your
    projected lineup.

    The single most useful thing on the screen once you've made a couple of
    picks: it collapses the whole board into six numbers that answer "what
    do I need" rather than "who is good". Reading it is deliberately
    literal - the bar is how much your projected starting lineup improves by
    spending THIS pick there instead of waiting one more turn.
    """
    # `board` here is already the available pool for whichever draft is
    # running, so nothing further needs excluding.
    rows = positional_value_add(board, dc['my_roster'], settings, dc['avail_pick'],
                                drafted_names=set(),
                                picks_left=dc.get('picks_left'))
    if not rows:
        return
    from config import get_position_color, get_position_chip_bg
    top = max((r['Points added'] for r in rows), default=1.0) or 1.0

    cards = []
    for row in rows:
        pos = row['Pos']
        fg = get_position_color(pos)
        bg = get_position_chip_bg(pos) or '#1a2447'
        width = max(2.0, min(100.0, row['Points added'] / top * 100))
        cards.append(
            f"<div class='pv-card' style='background:{bg};border-left:3px solid {fg}'>"
            f"<div class='pv-top'><span class='pv-pos' style='color:{fg}'>{pos}</span>"
            f"<span class='pv-pct'>+{row['Team %']:.1f}%</span></div>"
            f"<div class='pv-bar'><span style='width:{width:.0f}%;background:{fg}'></span></div>"
            f"<div class='pv-pts'>+{row['Points added']:.0f} pts</div>"
            f"<div class='pv-name'>{row['Best available']}</div>"
            f"<div class='pv-wait'>wait &rarr; {row['If you wait']:.0f}</div>"
            "</div>")
    st.markdown(
        "<style>"
        ".pv-wrap{display:flex;gap:6px;overflow-x:auto;padding:2px 0 8px 0}"
        ".pv-card{min-width:132px;flex:1;border-radius:6px;padding:6px 8px}"
        ".pv-top{display:flex;justify-content:space-between;align-items:baseline}"
        ".pv-pos{font-size:12px;font-weight:800;letter-spacing:.06em}"
        ".pv-pct{font-size:13px;font-weight:700}"
        ".pv-bar{height:4px;border-radius:2px;background:rgba(255,255,255,0.09);margin:4px 0}"
        ".pv-bar span{display:block;height:100%;border-radius:2px}"
        ".pv-pts{font-size:10px;opacity:.75}"
        ".pv-name{font-size:11px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}"
        ".pv-wait{font-size:9px;opacity:.55}"
        "</style>"
        f"<div class='pv-wrap'>{''.join(cards)}</div>", unsafe_allow_html=True)


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
            pool, order_col, _ = prepare_sim_pool(board, settings.get('bot_weights'))
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
        # Every roster in the room is known here, so the season can be played
        # out against the teams that actually drafted rather than against a
        # reconstructed field - the one place this simulation has no
        # guesswork in its opponents at all.
        if st.button("📈 Play the season out", key="dhq_mock_season", width="stretch"):
            with st.spinner("Simulating seasons for all teams..."):
                rosters = {team: [dict(p) for p in players]
                           for team, players in dc['state']['rosters'].items()}
                table = simulate_seasons(rosters, board, settings, n_sims=200)
            if not table.empty:
                table = table.reset_index().rename(columns={'Team': 'Slot'})
                table['Slot'] = table['Slot'].map(
                    lambda t: f"Team {t}" + (" ★" if t == dc['my_slot'] else ""))
                st.dataframe(table.sort_values('Wins', ascending=False), width="stretch",
                             hide_index=True, height=df_auto_height(len(table)))
                st.caption(
                    "Wins, not points. Each season redraws the schedule and re-rolls every "
                    "player's weeks from the real distribution of scores at his projected rank, "
                    "with lineups set on projections rather than on hindsight."
                )

    round_no = (dc['current_pick'] - 1) // max(dc['num_teams'], 1) + 1
    # The board lives beside the strip rather than in an expander at the
    # bottom of the page: it's a glance-at-it-mid-pick view ("who's hoarding
    # backs", "is the turn about to strip receivers"), and a view you have
    # to scroll past the whole draft table to reach may as well not exist
    # while a clock is running.
    #
    # A plain toggle button rather than st.popover, because a popover panel
    # is anchored to its trigger and capped at a few hundred pixels - a
    # twelve-column draft board rendered inside one is a postage stamp with
    # its own scrollbar. Toggling a full-width block below the row gives the
    # grid the whole screen, which is the only width it's readable at.
    strip_col, board_col = st.columns([9, 1.5], vertical_alignment="center")
    with strip_col:
        _render_recent_picks_strip(dc['picks'], dc['num_teams'], dc['current_pick'],
                                   round_no=round_no, next_pick=dc['next_pick'])
    with board_col:
        board_open = st.session_state.get(BOARD_OPEN_KEY, False)
        if st.button("🗂 Draft board", key="dhq_board_toggle", width="stretch",
                     type="primary" if board_open else "secondary"):
            st.session_state[BOARD_OPEN_KEY] = not board_open
            st.rerun()
    if st.session_state.get(BOARD_OPEN_KEY):
        with st.container(border=True):
            _render_draft_board_grid(dc['picks'], dc['num_teams'], dc['my_slot'],
                                     settings['draft_type'])
    if mode == "Live draft":
        _render_run_pressure(settings)

    _render_positional_value_add(dc['available'], settings, dc)
    # Recommendation directly under the positional cards, filter buttons
    # below it: the cards say which position to spend the pick on and the
    # banner names the player, so they read as one thought. The filter is a
    # control, and controls belong next to the thing they filter - which is
    # the board underneath, not the answer above.
    #
    # The filter still has to be READ first, since the banner is scoped to
    # it, so its state comes out of session_state here and the buttons that
    # set it are drawn afterwards.
    positions, label = _current_position_filter()
    # Scoped to what's actually left in THIS draft, not to the full board.
    # Reading the live tracker here was the bug that had a mock ten rounds
    # deep still recommending the consensus 1.01: in a mock, the live
    # tracker is empty, so "available" was the entire preseason board.
    recommended = _render_single_recommendation(dc['available'], settings, dc['my_roster'],
                                                dc['avail_pick'], positions, label)
    _render_position_filter()

    action = st.columns([1.3, 1.3, 4.4])
    if mode == "Mock draft":
        with action[0]:
            if st.button("⏭ Auto-pick", key="dhq_act_auto", width="stretch"):
                row = autopick_for_user(dc['state'], settings, dc['pool'])
                if row is not None:
                    _commit_pick(dc, settings, row['Player'], reach=reach)
    else:
        with action[0]:
            if st.button("↩ Undo last pick", key="dhq_act_undo", disabled=not dc['picks'],
                         width="stretch"):
                _undo_last()
                st.rerun()
    with action[2]:
        st.caption("Draft straight from the board — the ✅ / ❌ / 🔍 boxes on each row take "
                   "the player, mark him gone, or open his full profile.")

    left, right = st.columns([3, 1])
    with left:
        view = _apply_position_filter(dc['available'], positions)
        act, player = _render_board_grid(view, "dhq_board", mode, next_pick=dc['avail_pick'],
                                         row_limit=40)
        if act and player:
            _handle_board_action(act, player, dc, settings, reach)
        _render_player_detail(board, recommended)
    with right:
        st.markdown("**Your roster**")
        # Nudged down so the QB line starts level with the top of the board
        # table, which sits below that column's sort/rows controls. Without
        # it the roster floats a control-height above everything it's meant
        # to be read alongside.
        st.markdown("<div style='height:52px'></div>", unsafe_allow_html=True)
        _render_roster_slots(settings, roster=dc['my_roster'])
        _render_roster_outlook(board, settings, dc)

    with st.expander("📊 Pick odds & positional scarcity", expanded=False):
        _render_pick_odds(board, settings, ctx, dc)
        _render_positional_scarcity(dc['available'], settings)
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
    pool, _, has_market = prepare_sim_pool(board, settings.get('bot_weights'))
    if pool.empty:
        st.info("No board loaded to simulate against.")
        return
    if not has_market:
        st.warning(
            "No market ranking loaded, so simulated opponents draft off this board's own value "
            "ranking rather than the market. Useful as a stress test, but not a realistic room — "
            "treat 'who fell to me' with suspicion."
        )
    else:
        blend = (pool.attrs.get('opponent_ranking') or {}).get('components') or {}
        st.caption("Opponents rank players by " + ", ".join(
            f"{name} {weight:.0%}" for name, weight in blend.items()) +
            " — change the mix in League Settings.")

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
    # No "DRAFT HQ" heading here: the tab you clicked to get here is already
    # labelled DRAFT HQ, and repeating it costs a line of vertical space on
    # the one screen where vertical space is scarcest.
    #
    # The settings control is a plain button rather than a full-width
    # expander bar for the same reason. The status line renders into the
    # column beside it - the two columns are laid out first and filled in
    # afterwards, because the status text can only be written once the board
    # has actually loaded, and it belongs on this row rather than a line of
    # its own.
    head_left, head_right = st.columns([1.35, 5.5], vertical_alignment="center")
    cfg = _cfg()
    with head_left:
        if st.button("⚙️ League Settings & Data Sources", key="dhq_settings_btn",
                     width="stretch",
                     type="primary" if st.session_state.get(SETTINGS_OPEN_KEY) else "secondary"):
            st.session_state[SETTINGS_OPEN_KEY] = not st.session_state.get(SETTINGS_OPEN_KEY, False)
            st.rerun()

    ffa_upload = None
    if st.session_state.get(SETTINGS_OPEN_KEY):
        ffa_upload = _render_settings_panel(cfg)

    settings = _settings_from_cfg(cfg, ffa_upload)
    ctx = _pick_context(settings)

    with skeleton_loader("table", n_rows=12, n_cols=8):
        board, meta, adp_df, adp_meta, status = _load_board(settings, ctx['next_pick'])

    with head_right:
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
