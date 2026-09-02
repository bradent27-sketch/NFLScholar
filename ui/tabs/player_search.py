"""
Player Search tab: team-agnostic player lookup, PFF-grade bio card, positional
percentile matrix, weekly game log (sticky season-average footer), and
2023-2025 historical totals.
"""
import pandas as pd
import streamlit as st

from config import TEAM_CONFIG, THEME, OLINE_POSITIONS, DEFENSIVE_POSITIONS, TAB_DEFENSIVE_YIELD, abbr_to_pff_team, ODDS_API_PLAYER_PROP_MARKETS, AVAILABLE_SEASONS_WITH_UPCOMING
from data.transforms import (
    load_and_merge_data, precompute_league_percentiles, build_player_historical_summary,
    build_stat_allowed_matrix, build_alignment_multiplier, build_player_projection,
    build_market_projection, score_projected_stats, OFFENSE_PROJECTION_STATS,
    build_points_allowed_matrix,
)
from data.loaders import load_pff_data_with_fallback, load_schedule, load_saved_odds_api_key, fetch_nfl_player_props, load_team_pace, load_sharp_positional_coverage
from data.coverage_radar import list_receivers, list_coverage_teams, team_nickname
from data.utils import (
    calculate_exact_age, parse_height, parse_weight, clean_name_exact, clean_name_for_merge,
    build_last_name_index, match_abbreviated_name, calculate_percentile,
)
from ui.styling import get_pff_color, style_plain_dataframe, df_auto_height
from ui.components import (
    render_player_card, render_bio_strip, render_sticky_game_log, switch_tab, get_drafted_players_clean_keys,
    render_back_button, compute_bye_weeks, build_player_search_labels, render_stat_tiles, render_hero_tiles,
    render_fpts_week_strip, render_matchup_heat_strip, render_game_links, skeleton_loader,
)
from ui.player_snapshot import build_player_snapshot, render_percentile_bars_figure, render_percentile_radar_grid, split_snapshot_for_display


def _get_upcoming_opponent(team_abbr, year):
    """
    Earliest not-yet-played game for this team in the given schedule year -
    used for the Matchup Bridge button. Returns (opponent_abbr, week) or
    (None, None) if the team isn't found or the season's already complete
    (e.g. viewing a past/completed year has nothing "upcoming" left).
    """
    schedule_df = load_schedule(year)
    if schedule_df.empty:
        return None, None
    in_game = (schedule_df['home_team'] == team_abbr) | (schedule_df['away_team'] == team_abbr)
    upcoming = schedule_df[in_game & schedule_df['home_score'].isna()].sort_values('week')
    if upcoming.empty:
        return None, None
    game = upcoming.iloc[0]
    opponent = game['away_team'] if game['home_team'] == team_abbr else game['home_team']
    return opponent, int(game['week'])


def _get_remaining_schedule(team_abbr, year):
    """
    Every not-yet-played game for this team in the given schedule year, as
    (week, opponent_abbr) tuples sorted by week - same "not yet played"
    filter as _get_upcoming_opponent above (home_score.isna()), generalized
    to return the whole remaining slate instead of just the next game, for
    the matchup heat-strip.
    """
    schedule_df = load_schedule(year)
    if schedule_df.empty:
        return []
    in_game = (schedule_df['home_team'] == team_abbr) | (schedule_df['away_team'] == team_abbr)
    upcoming = schedule_df[in_game & schedule_df['home_score'].isna()].sort_values('week')
    return [
        (int(g['week']), g['away_team'] if g['home_team'] == team_abbr else g['home_team'])
        for _, g in upcoming.iterrows()
    ]


def _find_odds_event_id(odds_api_key, team_abbr, opponent_abbr):
    """
    Player props are fetched per Odds-API "event", not by team, so this
    finds THIS specific upcoming game's event id by matching full team
    names (what the bulk odds list uses) against TEAM_CONFIG's names for
    the two abbreviations already known from the schedule.
    """
    from data.loaders import fetch_nfl_odds
    team_name = TEAM_CONFIG.get(team_abbr, {}).get('name')
    opp_name = TEAM_CONFIG.get(opponent_abbr, {}).get('name')
    if not team_name or not opp_name:
        return None
    odds_data, err, _ = fetch_nfl_odds(odds_api_key)
    if err or not odds_data:
        return None
    for g in odds_data:
        teams = {g.get('home_team'), g.get('away_team')}
        if team_name in teams and opp_name in teams:
            return g.get('id')
    return None


def _build_schedule_only_log(team_abbr, year):
    """
    Full-season schedule for a team, shaped like the weekly game log table
    (week + opponent) but with no stat columns - used when a season has no
    real weekly player stats at all yet (e.g. before it kicks off - "week"
    isn't even a column in that year's merged stats frame, since
    load_year_data's roster-only outer-join fallback has nothing to put
    there). Shows "here's who they play and when" instead of a bare "no
    game logs" message for a season that just hasn't started.
    """
    schedule_df = load_schedule(year)
    if schedule_df.empty:
        return pd.DataFrame()
    in_game = (schedule_df['home_team'] == team_abbr) | (schedule_df['away_team'] == team_abbr)
    team_games = schedule_df[in_game].sort_values('week').copy()
    if team_games.empty:
        return pd.DataFrame()
    team_games['opponent_team'] = team_games.apply(
        lambda r: r['away_team'] if r['home_team'] == team_abbr else r['home_team'], axis=1
    )
    cols = ['week', 'opponent_team'] + (['gameday'] if 'gameday' in team_games.columns else [])
    return team_games[cols].reset_index(drop=True)


def _build_player_props_micro_table(props_data, player_name):
    """
    Same long-table shape as Live Odds' _build_props_long_table, filtered
    down to just this one player (that endpoint returns every player in the
    game, not one) - loose name match since sportsbook player-name spelling
    isn't guaranteed identical to nflverse's.
    """
    target_key = clean_name_for_merge(pd.Series([player_name])).iloc[0]
    rows = []
    for book in props_data.get('bookmakers', []):
        for market in book.get('markets', []):
            for o in market.get('outcomes', []):
                desc = o.get('description') or o.get('name') or ''
                if clean_name_for_merge(pd.Series([desc])).iloc[0] != target_key:
                    continue
                rows.append({
                    'Market': market.get('key', '').replace('player_', '').replace('_', ' ').title(),
                    'Selection': o.get('name'),
                    'Line': o.get('point'),
                    'Odds': o.get('price'),
                    'Book': book.get('title', book.get('key', '?')),
                })
    return pd.DataFrame(rows)


def render():
    render_back_button()

    # Restores the player/season/team-filter this tab was showing before the
    # user followed a game-log link out to its box score (Game Slate) and
    # came back. NOT a Streamlit quirk to work around defensively - confirmed
    # live (AppTest) that a keyed widget's own committed value is reset the
    # moment a script run doesn't instantiate it, which is exactly what
    # happens to every widget in this tab while Game Slate's body is the one
    # running. `ps_return_ctx` is stashed by render_game_links's `remember=`
    # at the moment a box-score chip is clicked (see below), and popped here
    # so it only fires once, on the very next render of this tab.
    return_ctx = st.session_state.pop('ps_return_ctx', None)

    # A cross-tab jump can request a specific season (e.g. clicking a name
    # on the 2022 Depth Chart should search 2022, not whatever year this tab
    # last happened to show) - must land in session_state BEFORE the year
    # selectbox below is instantiated this pass, same ordering rule as every
    # other cross-tab trigger in this file. Popped as a one-shot so it
    # doesn't keep overriding a manual year change on later visits.
    jump_to_year = st.session_state.pop('jump_to_year', None)
    if jump_to_year in AVAILABLE_SEASONS_WITH_UPCOMING and st.session_state.get('year_tab1') != jump_to_year:
        st.session_state['year_tab1'] = jump_to_year
    elif (not jump_to_year and return_ctx and return_ctx.get('year') in AVAILABLE_SEASONS_WITH_UPCOMING
          and st.session_state.get('year_tab1') != return_ctx['year']):
        st.session_state['year_tab1'] = return_ctx['year']

    # Optional team narrowing, for when you don't know (or don't want to
    # type) a player's exact name - "All Teams" (default) leaves the full
    # league-wide search below untouched. A cross-tab jump narrows this to
    # the clicked player's own team when the source tab knows it
    # (jump_to_team - Depth Charts always does), otherwise resets to "All
    # Teams" - otherwise a stale team filter left over from browsing one
    # team could silently hide the very player a jump was trying to reach.
    # Narrowing to the real team also disambiguates abbreviated "F.Last"
    # jumps (Depth Charts' own cell text): match_abbreviated_name matches
    # on (first-initial, last-name) alone, which is ambiguous LEAGUE-WIDE -
    # "Mi.Wilson" resolved to the wrong "Mack Wilson" instead of ARI's
    # Michael Wilson before this, since both are real "M. Wilson"s
    # somewhere in the league. Must be set BEFORE the
    # st.selectbox(key="player_search_team_filter") call below runs this
    # pass - moved up here (alongside jump_to_year, same reasoning) now
    # that Season and Filter-by-team share one row - same session_state-
    # before-widget-instantiation ordering rule as every other cross-tab
    # trigger in this file.
    jump_to_team = st.session_state.pop('jump_to_team', None)
    if st.session_state.get('jump_to_player'):
        if jump_to_team and jump_to_team in TEAM_CONFIG:
            st.session_state['player_search_team_filter'] = jump_to_team
        elif st.session_state.get('player_search_team_filter', 'All Teams') != 'All Teams':
            st.session_state['player_search_team_filter'] = 'All Teams'
    elif return_ctx and return_ctx.get('team_filter'):
        restored_filter = return_ctx['team_filter']
        if restored_filter == 'All Teams' or restored_filter in TEAM_CONFIG:
            st.session_state['player_search_team_filter'] = restored_filter

    # Scoring format is a segmented control in the global header now (same
    # key="score_tab1", instantiated once in ui.components.render_intro_and_
    # glossary before any tab's content runs) - read its value here instead
    # of instantiating a second widget under the same key, which Streamlit
    # would reject.
    t1_scoring_rule = st.session_state.get('score_tab1', 'Full PPR')

    team_options = ["All Teams"] + sorted(TEAM_CONFIG.keys(), key=lambda a: TEAM_CONFIG[a].get('name', a))
    team_option_labels = {"All Teams": "All Teams (search whole league)"}
    team_option_labels.update({abbr: f"{TEAM_CONFIG[abbr]['name']} ({abbr})" for abbr in TEAM_CONFIG})

    c_t1_year, c_t1_team = st.columns(2)
    with c_t1_year:
        t1_target_year = st.selectbox("Season", AVAILABLE_SEASONS_WITH_UPCOMING, index=1, key="year_tab1")
    with c_t1_team:
        team_filter = st.selectbox(
            "Team filter (optional)", team_options, index=0, key="player_search_team_filter",
            format_func=lambda abbr: team_option_labels.get(abbr, abbr),
            help="Don't know the exact name? Narrow the search & dropdown below to one team's roster instead.",
        )

    pff, pff_source_year = load_pff_data_with_fallback(t1_target_year)
    if pff_source_year != t1_target_year:
        st.caption(f"⚠️ No PFF grades uploaded yet for {t1_target_year} - showing {pff_source_year} grades until real data is added to pff_imports/{t1_target_year}/.")

    with skeleton_loader("tiles", n=10):
        df_t1_stats, t1_t_col, t1_n_col, global_rookie_names = load_and_merge_data(t1_target_year, t1_scoring_rule)
    df_t1_percentiles = precompute_league_percentiles(df_t1_stats, t1_n_col, t1_target_year)

    # Team-agnostic search: one league-wide, type-to-filter selectbox
    # instead of a mandatory "Select Team" dropdown gating player choice.
    # Position/team/bye are shown as metadata in the label itself (e.g.
    # "Justin Jefferson (WR - MIN) | Bye: 6") so that context is visible
    # in the dropdown before even selecting a player, resolved back to the
    # plain name (and that player's own team) after selection.
    bye_weeks = compute_bye_weeks(t1_target_year)
    labels, label_to_name = build_player_search_labels(df_t1_stats, t1_n_col, t1_t_col, bye_weeks, team_filter=team_filter)

    drafted_keys = get_drafted_players_clean_keys()
    if drafted_keys and labels != ["No players found"]:
        st.caption(f"🚫 {len(drafted_keys)} player(s) marked drafted on the VORP sheet - hidden below.")

    def _resolve_label(target_name):
        """
        Three-tier match from a plain player name to this year's actual
        dropdown label - shared by both trigger sources below, since a name
        arriving from another tab may use a different name format than this
        dropdown does.

        Tiers: exact (preserves suffixes) -> suffix/punctuation-stripped
        loose match -> abbreviated "F.Last" bridge. That last tier is the
        one that actually matters for jumps FROM Depth Charts specifically:
        its cells are built from roster_weekly's abbreviated player_name
        column (e.g. "P.Nacua"), which neither of the first two tiers can
        resolve on their own - clean_name_exact/clean_name_for_merge only
        lowercase and strip punctuation/suffixes, they don't expand an
        initial back into a first name. Confirmed this was a real bug: a
        Depth Chart click always failed to find the player before this
        tier was added, not merely for edge-case names.

        Vectorized over the whole label list once, not per-candidate - this
        dropdown covers the full league (~2,000 rows).
        """
        if not target_name or not labels:
            return None
        label_names = pd.Series([label_to_name[l] for l in labels])
        exact_keys = clean_name_exact(label_names)
        loose_keys = clean_name_for_merge(label_names)
        target_exact = clean_name_exact(pd.Series([target_name])).iloc[0]
        target_loose = clean_name_for_merge(pd.Series([target_name])).iloc[0]
        exact_hits = [l for l, k in zip(labels, exact_keys) if k == target_exact]
        loose_hits = [l for l, k in zip(labels, loose_keys) if k == target_loose]
        if exact_hits or loose_hits:
            return (exact_hits or loose_hits)[0]

        label_names_lower = label_names.str.lower()
        abbrev_index = build_last_name_index(label_names_lower)
        matched_lower = match_abbreviated_name(target_name, abbrev_index)
        if matched_lower:
            abbrev_hits = [l for l, n in zip(labels, label_names_lower) if n == matched_lower]
            if abbrev_hits:
                return abbrev_hits[0]
        return None

    # Two triggers resolve into the same dropdown label, both requiring the
    # session_state[key] write to happen BEFORE the st.selectbox(key=
    # 'player_sel_t1') call below runs this pass (Streamlit only allows
    # assigning to a keyed widget's state before that widget is
    # instantiated in the same script pass, or from a callback):
    #   1. jump_to_player - a cross-tab "teleport" (e.g. clicking a player
    #      row on Risers). Popped (not .get()) so this one-shot jump
    #      doesn't keep re-firing on every later visit to this tab.
    #   2. a year change while a player was already selected - carries the
    #      SAME player forward into the newly-selected year's label list
    #      instead of resetting to the placeholder, so switching "Season
    #      Data Framework" updates that player's stats in place rather than
    #      kicking you back to an empty state.
    jump_to_player = st.session_state.pop('jump_to_player', None)
    year_changed = st.session_state.get('_player_search_last_year') != t1_target_year
    carry_over_name = st.session_state.get('player_sel_t1_name') if year_changed else None
    st.session_state['_player_search_last_year'] = t1_target_year

    target_name = jump_to_player or (return_ctx.get('player') if return_ctx else None) or carry_over_name
    if target_name:
        match_label = _resolve_label(target_name)
        if match_label:
            st.session_state['player_sel_t1'] = match_label
        elif jump_to_player:
            st.warning(f"Couldn't find \"{jump_to_player}\" in {t1_target_year} data - they may not have played that season.")
        # A failed carry_over_name match falls through silently (no
        # warning) - entirely normal when the previously-selected player
        # simply didn't play in the newly-selected year.

    # index=None + placeholder=... (not a fake "— Select a player —" option
    # stuffed into the list at index 0) - a selected REAL option's text sits
    # in the input as literal, editable value text, which is why that old
    # approach required deleting the whole placeholder string before you
    # could type a search. A real placeholder (this widget's native
    # placeholder text, shown only when index/value is None) clears the
    # instant the box gets focus/input instead.
    selected_label = st.selectbox(
        "Search & select a player", labels,
        index=None, key="player_sel_t1", placeholder="— Select a player —",
        help="Type any part of a name to filter this list live, then click or arrow-down + Enter to select. Use the team filter above first if you don't know the exact name.",
    )
    selected_player = None if not selected_label or selected_label == "No players found" else label_to_name.get(selected_label, selected_label)
    if selected_player:
        # Stored under its own key (not player_sel_t1 itself) so the year-
        # change carry-over above has a stable plain name to resolve against
        # next rerun, independent of however this year's label happens to
        # be formatted (bye week text etc. varies year to year).
        st.session_state['player_sel_t1_name'] = selected_player

    if selected_player is None:
        return

    p_data = df_t1_stats[df_t1_stats[t1_n_col].astype(str) == selected_player].copy()

    if not p_data.empty and selected_player != "No players found":
        p_bio = p_data.iloc[0]
        pos = str(p_bio.get('position', 'N/A')).upper()
        if pos == 'NAN' or not pos: pos = 'N/A'

        filter_team = str(p_bio.get(t1_t_col, '')).upper()
        # THEME token rather than a literal cyan, so the "team unknown"
        # fallback tracks the palette instead of drifting from it.
        team_cfg = TEAM_CONFIG.get(filter_team, {'color': THEME['colors']['primary']})
        is_player_rookie = selected_player.lower() in global_rookie_names

        overall_grade = pff['pff_grades_map'].get(selected_player.lower(), 0.0)
        grade_color = get_pff_color(overall_grade, is_rookie=is_player_rookie, raw_grade=True)
        # Same "top PFF grade at this position, this season" league-leader
        # check as the depth chart's gold cell border (data.loaders.
        # _build_master_pff_grades computes league_gold_players once per
        # position/year) - direct .lower() lookup, not the abbreviated-name
        # bridge depth_charts.py needs, since this tab's n_col is already a
        # full display name in the same format PFF's own export uses.
        is_gold = overall_grade > 0 and not is_player_rookie and selected_player.lower() in pff['league_gold_players']

        jersey_val = str(p_bio.get('jersey_number', '--')).replace('.0', '')
        if jersey_val == 'nan' or not jersey_val: jersey_val = '--'

        p_age_exact = calculate_exact_age(p_bio.get('birth_date'), p_bio.get('age', 0))
        p_height = parse_height(p_bio.get('height', 0))
        p_weight = parse_weight(p_bio.get('weight', 0))

        # Computed here (ahead of the card) rather than down at the matrix
        # table, which used to be its only consumer - the scouting-card
        # overhaul embeds a percentile micro-radar directly on the card
        # face, so the snapshot needs to exist before render_player_card
        # runs. Every input here (p_data, p_bio, pff, overall_grade,
        # grade_color) is already available above; only games_played/
        # p_pct_row moved up with it, unchanged otherwise.
        games_played = p_data['week'].nunique() if 'week' in p_data.columns else 1
        games_played = max(games_played, 1)
        p_pct_row = df_t1_percentiles[(df_t1_percentiles[t1_n_col].astype(str) == selected_player) & (df_t1_percentiles['position'] == pos)]
        snapshot = build_player_snapshot(
            selected_player, pos, p_data, p_bio, pff, p_pct_row, games_played,
            overall_grade, grade_color, t1_target_year,
        )

        # Fantasy-points-by-week line chart FIRST - ahead of even the hero
        # tiles, per explicit feedback that this should be the very first
        # thing shown for a selected player, styled after the boards' own
        # "Last 5 Wks" sparkline column (the chart style users said reads
        # best "at a glance"). Passing the team's full schedule fixes the
        # x-axis to the whole season: a pre-kickoff 2026 view shows the
        # empty 18-week axis with opponents, and it fills in as games are
        # played rather than only spanning the weeks already played.
        _fpts_sched = _build_schedule_only_log(filter_team, t1_target_year)
        render_fpts_week_strip(p_data, t1_target_year, scoring_label=t1_scoring_rule,
                               schedule_df=_fpts_sched)

        # Rest-of-season matchup difficulty strip - additive companion to
        # the fpts-by-week strip above (that one looks back, this one looks
        # ahead). QB/RB/WR/TE only: build_points_allowed_matrix only ever
        # has those 4 position columns (same positions Defensive Yield's own
        # points-allowed table covers) - silently absent for any other
        # position, or a season with no remaining games left to play, same
        # "nothing to show" convention as the fpts strip.
        if pos in ('QB', 'RB', 'WR', 'TE'):
            remaining_games = _get_remaining_schedule(filter_team, t1_target_year)
            if remaining_games:
                def_matrix = build_points_allowed_matrix(df_t1_stats, t1_target_year)
                if not def_matrix.empty and pos in def_matrix.columns:
                    # ascending=True (default): a higher points-allowed
                    # number is a SOFTER matchup - same direction
                    # Defensive Yield's own strength-of-schedule table uses,
                    # so get_matchup_color reads identically here.
                    pct_by_team = dict(zip(def_matrix['Team'], calculate_percentile(def_matrix, pos)))
                    raw_by_team = dict(zip(def_matrix['Team'], def_matrix[pos]))
                    matchup_rows = [
                        {'week': wk, 'opponent': opp, 'pct': pct_by_team.get(opp), 'raw_pts': raw_by_team.get(opp)}
                        for wk, opp in remaining_games if opp in pct_by_team
                    ]
                    render_matchup_heat_strip(matchup_rows, pos, t1_target_year)

        # Hero stat band - the headline fantasy numbers (PPG, total points,
        # position rank, games) as large tiles spanning the page, the same
        # "identity + headline numbers first" hierarchy every pro player
        # page (ESPN/PFF/Sleeper) leads with. All values come from the
        # already-computed percentile summary; the only new math is the
        # position-rank count, a trivial comparison over the cached frame.
        if not p_pct_row.empty and 'fantasy_points_sum' in p_pct_row.columns and 'week' in p_data.columns:
            p_row = p_pct_row.iloc[0]
            total_fpts = float(p_row.get('fantasy_points_sum', 0) or 0)
            ppg = float(p_row.get('fantasy_points_mean', 0) or 0)
            ppg_pct = p_row.get('fantasy_points_mean_pct')
            pos_pool = df_t1_percentiles[df_t1_percentiles['position'] == pos]
            pos_rank = int((pos_pool['fantasy_points_sum'] > total_fpts).sum()) + 1 if not pos_pool.empty else None
            hero_items = [
                {'label': 'Fantasy PPG', 'value': f"{ppg:.1f}",
                 'sub': f"{ppg_pct:.0f}th percentile" if pd.notna(ppg_pct) else None,
                 'accent': get_pff_color(ppg_pct) if pd.notna(ppg_pct) else None},
                {'label': 'Total Points', 'value': f"{total_fpts:.1f}", 'sub': t1_scoring_rule},
            ]
            if pos_rank is not None:
                hero_items.append({'label': 'Position Rank', 'value': f"{pos} #{pos_rank}", 'sub': f"of {len(pos_pool)} {pos}s"})
            hero_items.append({'label': 'Games', 'value': f"{games_played}", 'sub': f"{t1_target_year} season"})
            render_hero_tiles(hero_items)

        c_left, c_right = st.columns([1, 1])

        with c_left:
            grade_str = 'ROOKIE' if is_player_rookie else (f"{overall_grade:.1f}" if overall_grade > 0 else 'N/A')
            render_player_card(
                selected_player, pos, grade_str, grade_color, filter_team, team_cfg['color'],
                jersey_val, p_bio['headshot_url'], is_gold=is_gold,
            )
            render_bio_strip(p_age_exact, p_height, p_weight)

            opponent_abbr, opponent_week = _get_upcoming_opponent(filter_team, t1_target_year)
            opponent_pff_code = abbr_to_pff_team(opponent_abbr) if opponent_abbr else None
            # Guarded by membership checks (both here and for radar_player
            # below) since st.selectbox raises if handed a session_state
            # value that isn't one of its own options - only offer the
            # button at all if PFF actually has coverage data for this
            # opponent, otherwise the destination tab would crash on load.
            if opponent_abbr and opponent_pff_code in list_coverage_teams(pff['def_coverage_scheme']):
                opponent_name = TEAM_CONFIG.get(opponent_abbr, {}).get('name', opponent_abbr)
                nav_context = {'radar_opponent': opponent_pff_code}
                # Best-effort: also pre-select this player in the radar if
                # PFF has man/zone coverage-scheme data for them (WR/TE only).
                if pos in ('WR', 'TE') and selected_player in list_receivers(pff['rec_scheme']):
                    nav_context['radar_player'] = selected_player
                # on_click, not `if st.button(...): switch_tab(...)` - by the
                # time this tab's render() body runs, app.py's st.tabs(key=
                # "active_tab") is already instantiated for this pass, and
                # Streamlit only allows assigning to a keyed widget's
                # session_state from a callback (a separate pre-script
                # phase), not later in the same normal script flow.
                st.button(
                    f"🏈 View Matchup vs {opponent_name} (Wk {opponent_week})", key="matchup_bridge_btn",
                    width="stretch", on_click=switch_tab, args=(TAB_DEFENSIVE_YIELD,), kwargs=nav_context,
                )

            # Live Odds micro-card - deliberately button-gated, not automatic.
            # Fetching props on every player viewed (Player Search is the
            # default/most-visited tab) would burn The Odds API's per-key
            # request quota fast, especially since player props cost more
            # credits per request than the bulk odds call - a manual trigger
            # plus the existing 15-min st.cache_data on both fetch_* calls
            # keeps this sane even with repeated clicks/reruns.
            if opponent_abbr:
                odds_api_key = st.session_state.get('odds_api_key') or load_saved_odds_api_key()
                opp_name_for_props = TEAM_CONFIG.get(opponent_abbr, {}).get('name', opponent_abbr)
                with st.expander(f"📊 {selected_player}'s prop lines vs {opp_name_for_props}", expanded=False):
                    if not odds_api_key:
                        st.caption("Enter your Odds API key in the Live Odds tab, then come back here.")
                    else:
                        if st.button("Load prop lines", key="ps_load_props_btn"):
                            event_id = _find_odds_event_id(odds_api_key, filter_team, opponent_abbr)
                            if event_id:
                                props_data, props_err = fetch_nfl_player_props(
                                    odds_api_key, event_id, markets=','.join(ODDS_API_PLAYER_PROP_MARKETS)
                                )
                            else:
                                props_data, props_err = None, "Couldn't find this game on the odds board yet (too far from kickoff, or not posted)."
                            st.session_state['ps_props_data'] = props_data
                            st.session_state['ps_props_err'] = props_err
                            st.session_state['ps_props_for_player'] = selected_player

                        # Only show cached results if they're for THIS
                        # player - otherwise switching players would keep
                        # showing whoever was last loaded's stale props.
                        if st.session_state.get('ps_props_for_player') == selected_player:
                            props_err = st.session_state.get('ps_props_err')
                            props_data = st.session_state.get('ps_props_data')
                            if props_err:
                                st.caption(f"⚠️ {props_err}")
                            elif props_data:
                                micro_df = _build_player_props_micro_table(props_data, selected_player)
                                if not micro_df.empty:
                                    st.dataframe(
                                        style_plain_dataframe(micro_df.set_index('Market')),
                                        width="stretch", height=df_auto_height(min(len(micro_df), 10)),
                                    )
                                else:
                                    st.caption("No props posted for this player yet by any tracked bookmaker.")

            st.markdown("<div class='custom-section-header'>SEASON PROFILE — LEAGUE PERCENTILES</div>", unsafe_allow_html=True)

            # The percentile-bar chart alone is now the primary view - it
            # used to sit directly below a stat-tile grid showing the exact
            # same headline stats a second time, which per explicit
            # feedback was redundant (the bar chart already reads clearly
            # on its own, especially now that it carries its own per-bar
            # hover detail). Everything else stays split into selectable
            # stat-family groups (slot / screen for WR-TE) behind a
            # segmented control - nothing is dropped, every stat is still
            # reachable, one group at a time.
            if snapshot:
                _, secondary_groups = split_snapshot_for_display(snapshot, pos)
                render_percentile_bars_figure(snapshot, selected_player, pos)

                if secondary_groups:
                    group_names = [g for g, _ in secondary_groups]
                    picked = st.segmented_control(
                        "More stats", group_names, key=f"profile_group_{pos}", default=None,
                        help="Each group shows a different slice of this player's advanced stats - pick one to expand it below.",
                    )
                    if picked:
                        rows = dict(secondary_groups)[picked]
                        render_stat_tiles(rows)
            else:
                st.info("No relative metrics available for this positional profile.")

        with c_right:
            st.markdown(f"<div class='custom-section-header'>GAME LOG — {t1_target_year}</div>", unsafe_allow_html=True)

            log_cols = []
            if pos == 'QB':
                log_cols = ['week', 'opponent_team', 'passing_attempts', 'passing_completions', 'passing_yards', 'passing_tds', 'passing_interceptions', 'rushing_attempts', 'rushing_yards', 'rushing_tds', 'fantasy_points']
            elif pos in ['WR', 'TE']:
                log_cols = ['week', 'opponent_team', 'targets', 'receptions', 'receiving_yards', 'y/c', 'receiving_tds', 'rushing_yards', 'rushing_tds', 'weekly_snap_pct', 'fantasy_points']
            elif pos == 'RB':
                log_cols = ['week', 'opponent_team', 'rushing_attempts', 'rushing_yards', 'ypc', 'rushing_tds', 'targets', 'receptions', 'receiving_yards', 'receiving_tds', 'weekly_snap_pct', 'fantasy_points']
            elif pos in OLINE_POSITIONS:
                log_cols = ['week', 'opponent_team', 'weekly_snap_pct']
            elif pos in DEFENSIVE_POSITIONS:
                log_cols = ['week', 'opponent_team', 'tackles', 'sacks', 'def_ints', 'forced_fumbles', 'weekly_snap_pct', 'fantasy_points']
            elif pos == 'K':
                # weekly_snap_pct is meaningless here (kickers aren't tracked
                # in the offensive/defensive snap-count export at all - same
                # gap the bio card's "N/A" already covers - see has_snap_match)
                # and showed a flat, misleading 0.0% every week instead.
                # Real per-kick weekly stats DO exist in the source file
                # though (fg_made/fg_att/fg_long/pat_made/pat_att) - using
                # those instead of a bogus snap % is both more accurate and
                # more useful for a position whose entire fantasy value is
                # these counting stats, not playing time.
                log_cols = ['week', 'opponent_team', 'fg_made', 'fg_att', 'fg_long', 'pat_made', 'pat_att', 'fantasy_points']
            else:
                log_cols = ['week', 'opponent_team', 'weekly_snap_pct']

            log_df = p_data[[c for c in log_cols if c in p_data.columns]].copy()
            if 'week' in log_df.columns:
                log_df = log_df[pd.to_numeric(log_df['week'], errors='coerce').fillna(0) > 0].sort_values('week')

            header_map = {
                'weekly_snap_pct': 'SNAP %', 'fantasy_points': 'FPTS', 'passing_attempts': 'PASS ATT',
                'passing_completions': 'PASS CMP', 'passing_yards': 'PASS YDS', 'passing_tds': 'PASS TD',
                'passing_interceptions': 'INT', 'rushing_attempts': 'RUSH ATT', 'rushing_yards': 'RUSH YDS',
                'rushing_tds': 'RUSH TD', 'targets': 'TGT', 'receptions': 'REC', 'receiving_yards': 'REC YDS',
                'receiving_tds': 'REC TD', 'y/c': 'Y/C', 'ypc': 'YPC', 'tackles': 'TKL', 'sacks': 'SCK',
                'def_ints': 'INT', 'forced_fumbles': 'FF', 'fg_made': 'FG MADE', 'fg_att': 'FG ATT',
                'fg_long': 'FG LONG', 'pat_made': 'PAT MADE', 'pat_att': 'PAT ATT',
            }

            # Season -> week is this app's one real nested-filtering drill-
            # down (the season itself is fixed by the selectbox up top,
            # already baked into this section's own header); a bordered
            # container makes that "you are now narrowing INSIDE the
            # {year} season" relationship read as one connected unit
            # instead of a slider floating loose above an unrelated-looking
            # table box.
            with st.container(border=True):
                # ONE full-season table in every case: the team's whole
                # schedule is the row set, and the player's real per-game
                # stats are merged onto the weeks he has played. An
                # unplayed week stays as a present-but-empty row rather
                # than vanishing, so the 2026 pre-kickoff view and a
                # mid-season view are the same styled table, just at
                # different fill levels.
                played_log = log_df if 'week' in log_df.columns else p_data.iloc[0:0]
                schedule_only = _build_schedule_only_log(filter_team, t1_target_year)

                if schedule_only.empty and played_log.empty:
                    st.info(f"No {t1_target_year} schedule or game logs available yet.")
                else:
                    if not schedule_only.empty:
                        sched = schedule_only.copy()
                        sched['week'] = pd.to_numeric(sched['week'], errors='coerce')
                        played_weeks = set()
                        merged = sched
                        if not played_log.empty:
                            stats_only = played_log.drop(
                                columns=[c for c in ('opponent_team',) if c in played_log.columns])
                            stats_only = stats_only.copy()
                            stats_only['week'] = pd.to_numeric(stats_only['week'], errors='coerce')
                            played_weeks = set(stats_only['week'].dropna().astype(int).tolist())
                            merged = sched.merge(stats_only, on='week', how='left')
                        merged['_unplayed'] = ~merged['week'].astype('Int64').isin(played_weeks)
                        full_log = merged.sort_values('week').reset_index(drop=True)
                    else:
                        full_log = played_log.copy()
                        full_log['_unplayed'] = False
                        full_log = full_log.sort_values('week').reset_index(drop=True)

                    weeks_available = sorted(
                        pd.to_numeric(full_log['week'], errors='coerce').dropna().astype(int).unique().tolist())
                    if len(weeks_available) > 1:
                        wk_lo, wk_hi = st.select_slider(
                            "Week range", options=weeks_available,
                            value=(weeks_available[0], weeks_available[-1]), key="t1_week_slider"
                        )
                        wk_num = pd.to_numeric(full_log['week'], errors='coerce')
                        log_df_view = full_log[(wk_num >= wk_lo) & (wk_num <= wk_hi)]
                    else:
                        log_df_view = full_log

                    played_in_view = log_df_view[~log_df_view['_unplayed']]
                    if played_in_view.empty:
                        st.caption(f"No games played yet for {t1_target_year} — the schedule below fills in with real per-game stats as the season goes.")
                    # avg_source_df = played rows only, so the sticky AVG
                    # footer keeps averaging real games, not empty ones.
                    render_sticky_game_log(log_df_view, played_in_view, log_cols, header_map)

                    # A chip strip, not a control inside the table: the game
                    # log is hand-rolled HTML (ui.styling.render_game_log_html_table),
                    # and HTML cannot fire a Python callback, so a per-row
                    # button has to live outside the table. game_link_rows
                    # already drops any row that doesn't resolve to a real
                    # completed game, so unplayed weeks produce no chip.
                    if not played_in_view.empty:
                        from data.box_score import game_link_rows
                        from data.game_slate import season_slate
                        slate, _slate_err = season_slate(t1_target_year)
                        links = game_link_rows(played_in_view, slate, team=filter_team)
                        render_game_links(
                            links, t1_target_year, key_prefix="ps_box",
                            caption="Open a game's full box score:",
                            remember=('ps_return_ctx', {
                                'player': selected_player, 'team_filter': team_filter,
                                'year': t1_target_year,
                            }),
                        )

            # Radar grid lives here (under the game log), not under the
            # matrix table in c_left - c_left was running much taller than
            # c_right with both the bars chart AND the radar grid stacked
            # on top of it, reading lop-sided. Only renders for positions
            # with a curated RADAR_GRID_GROUPS entry (see that constant's
            # docstring) - silently absent otherwise, same as before.
            radar_grid_fig = render_percentile_radar_grid(snapshot, selected_player, pos)
            if radar_grid_fig is not None:
                st.markdown("<div class='custom-section-header'>SKILL RADAR</div>", unsafe_allow_html=True)
                st.pyplot(radar_grid_fig, width="stretch")

        # AVAILABLE_SEASONS is [2025, 2024, ..., 2019] - excludes 2026
        # (never a "historical" season) automatically, and matches
        # local stats_player_week_{2019..2025}.csv coverage exactly. Full
        # width, below both columns - a multi-year totals table doesn't
        # belong squeezed into either the card/matrix or game-log column.
        hist_years = tuple(AVAILABLE_SEASONS_WITH_UPCOMING[1:])
        hist_grouped = build_player_historical_summary(selected_player, t1_scoring_rule, years=hist_years)
        if not hist_grouped.empty:
            st.markdown(f"<div class='custom-section-header'>CAREER TOTALS — {min(hist_years)} TO {max(hist_years)}</div>", unsafe_allow_html=True)
            # This table only ever has up to 7 rows (one per historical
            # season) and a handful of columns - stretching it edge-to-edge
            # across the same full-bleed width as the app's genuinely wide
            # tables (positional matrices, depth charts) left it looking
            # unanchored, with a wall of near-empty stretched cells instead
            # of a compact card. Same centered-column convention already
            # used for the Defensive Yield coverage radar (a narrower
            # visual inside a full-bleed page) - outer columns are pure
            # margin, not content.
            hist_l, hist_c, hist_r = st.columns([1, 5, 1])
            with hist_c:
                st.dataframe(style_plain_dataframe(hist_grouped.set_index('season')), width="stretch", height=df_auto_height(len(hist_grouped)))

        st.markdown("<div class='custom-section-header'>NEXT GAME PROJECTION</div>", unsafe_allow_html=True)
        if pos not in OFFENSE_PROJECTION_STATS:
            st.caption("Projections are only modeled for offensive skill positions (QB/RB/WR/TE).")
        elif not opponent_abbr:
            st.caption("No upcoming game found on the schedule to project against.")
        else:
            allowed_matrix = build_stat_allowed_matrix(df_t1_stats, position_filter=[pos])
            pace_df = load_team_pace(t1_target_year)

            alignment_mult = 1.0
            if pos in ('WR', 'TE') and not pff['rec'].empty:
                p_pff_rec = pff['rec'][pff['rec']['player'].str.lower() == selected_player.lower()]
                if not p_pff_rec.empty:
                    coverage_df = load_sharp_positional_coverage()
                    opp_nickname = team_nickname(TEAM_CONFIG.get(opponent_abbr, {}).get('name', ''))
                    rec_row = p_pff_rec.iloc[0]
                    alignment_mult = build_alignment_multiplier(
                        coverage_df, opp_nickname, rec_row.get('slot_rate', 0), rec_row.get('wide_rate', 0)
                    )

            model_proj, _ = build_player_projection(
                p_data, pos, opponent_abbr, t1_scoring_rule, allowed_matrix, pace_df, alignment_mult
            )

            if not model_proj:
                st.caption("Not enough game log data yet this season to build a projection.")
            else:
                # Reuse whatever prop lines are already loaded for this
                # player/opponent from the micro-card above (only if the
                # user clicked "Load prop lines" there) rather than firing a
                # second, separate odds fetch here - keeps this section from
                # burning extra Odds API quota on its own just by existing.
                market_stats = {}
                if st.session_state.get('ps_props_for_player') == selected_player and st.session_state.get('ps_props_data'):
                    market_stats = build_market_projection(st.session_state['ps_props_data'], selected_player)

                rows = []
                blended_stats = dict(model_proj)
                used_market = False
                for stat in OFFENSE_PROJECTION_STATS[pos]:
                    if stat not in model_proj:
                        continue
                    model_val = model_proj[stat]
                    market_val = market_stats.get(stat)
                    if market_val is not None:
                        # Weighted toward the market on purpose - live prop
                        # odds already price in injuries, weather, game
                        # script, and sharper opponent context than this
                        # model can reconstruct from box scores alone.
                        final_val = round(0.7 * market_val + 0.3 * model_val, 1)
                        blended_stats[stat] = final_val
                        used_market = True
                    else:
                        final_val = model_val
                    rows.append({
                        'Stat': stat.replace('_', ' ').title(),
                        'Model': model_val,
                        'Market Line': market_val if market_val is not None else '--',
                        'Projection': final_val,
                    })

                blended_fpts = score_projected_stats(blended_stats, t1_scoring_rule)
                anytime_td_prob = market_stats.get('anytime_td_prob')

                proj_df = pd.DataFrame(rows).set_index('Stat')
                st.dataframe(style_plain_dataframe(proj_df), width="stretch", height=df_auto_height(len(rows)))

                source_note = "blended with live sportsbook prop lines" if used_market else "model-only - load prop lines above to blend in live market odds"
                opp_display = TEAM_CONFIG.get(opponent_abbr, {}).get('name', opponent_abbr)
                st.caption(f"**{blended_fpts:.1f} projected fantasy points** vs {opp_display} (Wk {opponent_week}) - {source_note}.")
                if anytime_td_prob is not None:
                    st.caption(f"Market anytime-TD probability: {anytime_td_prob * 100:.0f}%")
