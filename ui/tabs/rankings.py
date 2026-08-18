"""
Weekly Rankings tab: this app's own week-by-week fantasy-points projection
model (data.weekly_projections), FantasyPros' live weekly projection pulled
straight from their API, a live player-prop-derived market projection
(data.odds_weekly), this app's own L5 recent-form ranking, and (still,
unchanged) an uploaded weekly-rankings export - all lined up on one table
by player.

A weekly ranking answers "who's producing right now" / "who should I start
this week" - a different question than the season-long draft-value ranking
Draft HQ compares against FantasyPros' DRAFT rankings. Keeping the two
comparisons on separate tabs (rather than one tab trying to serve both)
means each one's internal baseline actually matches what it's being
compared against.

The model, the live API pull, the live market projection, and the CSV
upload are four INDEPENDENT sources shown side by side, never blended into
one number - same "show market lines next to this board, don't merge them"
convention Draft HQ already uses.
"""
import pandas as pd
import streamlit as st

from config import AVAILABLE_SEASONS_WITH_UPCOMING
from data.draft_board import DEFAULT_SCORING, tier_by_position
from data.transforms import load_and_merge_data, build_recent_form_rank, build_recent_trend
from data.rankings import parse_fantasypros_upload, parse_custom_rankings, build_rankings_comparison
from data.utils import calculate_percentile, clean_name_exact, clean_name_for_merge
from data.weekly_projections import build_weekly_projections
from data.odds_weekly import weekly_props, weekly_market_projection
from ui.styling import style_plain_dataframe, df_auto_height, build_column_help_config
from ui.components import position_filter_multiselect, skeleton_loader, import_hint

_MODEL_STAT_COLS = ['passing_yards', 'passing_tds', 'rushing_attempts', 'rushing_yards',
                    'rushing_tds', 'targets', 'receptions', 'receiving_yards', 'receiving_tds']

# The stat line the model projects, shown alongside Model Proj Pts instead
# of just the point total - explicit request, so the number can be read
# rather than taken on faith. Ordered pass -> rush -> catch, the same
# volume-first sequencing the rest of the app's stat displays use. A
# position that doesn't carry a given raw stat (a WR has no passing_yards)
# just shows blank for it - the concat in build_weekly_projections already
# unions every position's columns, so these exist on the merged frame
# whenever ANY position projected that stat this week.
_STAT_DISPLAY_COLS = [
    ('passing_yards', 'Pass Yds'), ('passing_tds', 'Pass TDs'),
    ('rushing_attempts', 'Rush Att'), ('rushing_yards', 'Rush Yds'), ('rushing_tds', 'Rush TDs'),
    ('targets', 'Tgt'), ('receptions', 'Rec'), ('receiving_yards', 'Rec Yds'), ('receiving_tds', 'Rec TDs'),
]

# A fixed number of games, not a user-adjustable window - explicit request.
# "Recent" reads consistently across positions and weeks only if everyone's
# measured over the same trailing sample.
RECENT_FORM_GAMES = 5

# Rendering every skill-position player with a projection (300+ rows, each
# with several percentile-heatmapped columns) was the actual source of a
# real reported slowdown - the Styler recomputes its per-column heatmap over
# every visible row on every rerun. Defaulting to the top 50 (already-sorted
# order, so "top 50" means the 50 most relevant to whatever's currently
# filtered) with a dropdown to widen it keeps the common case fast without
# hiding the full pool from anyone who wants it.
_SHOW_N_OPTIONS = [25, 50, 100, 200, "All"]


def _limit_rows(df, key):
    choice = st.selectbox("Show", _SHOW_N_OPTIONS, index=1, key=key)
    return df if choice == "All" or len(df) <= choice else df.head(choice)


def _week_options(year):
    """
    [(week, label, is_next_incomplete)], from the real schedule
    (data.game_slate, already built for the Game Slate tab) - reused rather
    than a second week-numbering scheme. Falls back to a bare 1-18 range if
    the schedule feed is unreachable, so the tab still works offline.
    """
    try:
        from data.game_slate import slate_weeks
        weeks, _err = slate_weeks(year)
        reg = [w for w in weeks if w['season_type'] == 'REG']
        if reg:
            return reg
    except Exception:
        pass
    return [{'week': w, 'label': f'Week {w}', 'completed': False} for w in range(1, 19)]


def _default_week_index(weeks):
    """The next UNPLAYED week - a projection tool wants the upcoming week,
    the opposite default from Game Slate's own "last completed" one."""
    for i, w in enumerate(weeks):
        if not w['completed']:
            return i
    return max(len(weeks) - 1, 0)


def _render_fantasypros_weekly_pull(wk_year, wk_week, wk_scoring):
    """
    Button-triggered pull of FantasyPros' own weekly stat-line projection
    for the selected week, straight from their API - same secrets-aware key
    resolution and budget bookkeeping as Draft HQ's ECR/ADP pull
    (data.draft_sources.fetch_fantasypros_weekly_projections costs one call
    PER POSITION requested, see that function's own docstring for why).

    Always rendered, independent of whether this app's own model produced
    anything for the selected week - week 1 is exactly the case where the
    model CAN'T project (see build_weekly_projections' docstring) and
    FantasyPros' own number is the one worth having.

    Stamps a fetch time alongside the pull (`wr_fp_weekly_fetched_at`) so
    the table below can show a real "current to Week N, pulled at TIME"
    indicator instead of leaving whether the FantasyPros numbers are fresh
    or a stale holdover from an earlier week/scoring choice unstated.
    """
    import datetime
    from data.draft_sources import (
        get_fantasypros_api_key, save_fantasypros_api_key,
        fantasypros_api_calls_this_month, fantasypros_effective_limit,
        fetch_fantasypros_weekly_projections,
    )
    with st.expander("📡 FantasyPros weekly projection (live API)", expanded=False):
        st.caption(
            "Pulls FantasyPros' own projected stat line for QB/RB/WR/TE for the selected week - "
            "4 calls (one per position; the endpoint requires a position per request). Shown "
            "alongside this app's model below, never blended into it."
        )
        secret_key, key_source = get_fantasypros_api_key()
        if key_source == 'secrets':
            st.caption("🔑 Using the key from `.streamlit/secrets.toml`.")
            api_key = secret_key
        else:
            api_key = st.text_input("FantasyPros API key", type="password",
                                    key="wr_fp_api_key", value=secret_key)
            if api_key and api_key != secret_key:
                save_fantasypros_api_key(api_key)

        used = fantasypros_api_calls_this_month()
        limit = fantasypros_effective_limit()
        remaining = None if limit is None else limit - used
        cost = 4
        fetch = st.button("Fetch weekly projections", key="wr_fp_fetch",
                          disabled=not api_key or (remaining is not None and remaining < cost))
        if limit is not None:
            st.caption(f"{used} of {limit} calls used this month - this pull costs {cost}.")
        if fetch:
            with st.spinner("Calling the FantasyPros API…"):
                df, meta = fetch_fantasypros_weekly_projections(api_key, wk_year, wk_week)
            if meta.get('error') and df.empty:
                st.error(f"FantasyPros API: {meta['error']}")
            else:
                st.session_state['wr_fp_weekly_df'] = df
                st.session_state['wr_fp_weekly_meta'] = (wk_year, wk_week, wk_scoring)
                st.session_state['wr_fp_weekly_fetched_at'] = datetime.datetime.now()
                if meta.get('errors'):
                    st.warning(f"Some positions failed: {meta['errors']}")
                st.success(f"Pulled {len(df):,} FantasyPros weekly projections for week {wk_week}.")
                st.rerun()

    held = st.session_state.get('wr_fp_weekly_df')
    held_meta = st.session_state.get('wr_fp_weekly_meta')
    if held is not None and not held.empty and held_meta == (wk_year, wk_week, wk_scoring):
        return held
    return None


def _fantasypros_freshness_caption(wk_year, wk_week, wk_scoring):
    """The "has this actually been pulled, and when" indicator - explicit
    request. Distinguishes a live pull FOR this exact year/week/scoring
    selection from a held-over one for a different combination, since the
    session can carry a stale pull from before the selectors above changed."""
    held_meta = st.session_state.get('wr_fp_weekly_meta')
    fetched_at = st.session_state.get('wr_fp_weekly_fetched_at')
    if held_meta is None or fetched_at is None:
        st.caption("📡 FantasyPros: not pulled this session — open the expander above to fetch it.")
        return
    if held_meta == (wk_year, wk_week, wk_scoring):
        st.caption(
            f"📡 FantasyPros projections current to **Week {wk_week}, {wk_year}** "
            f"({wk_scoring}) — pulled {fetched_at:%b %d, %Y at %I:%M %p}."
        )
    else:
        held_year, held_week, held_scoring = held_meta
        st.caption(
            f"📡 FantasyPros: last pull was for Week {held_week}, {held_year} ({held_scoring}) at "
            f"{fetched_at:%b %d, %I:%M %p} — that doesn't match the selection above, so it isn't shown below. "
            "Re-fetch to pull the current selection."
        )


def _scoring_dict(scoring_mode):
    """This tab's simple 'Full PPR'/'Half-PPR'/'Standard' picker, expanded
    into the full per-stat scoring dict data.draft_board.score_stats (and
    therefore the market-projection path) needs. Every non-reception weight
    here already matches what data.transforms.score_projected_stats hardcodes
    for the model's own points, so this doesn't introduce a second scoring
    opinion - it's the same rules in the dict shape the market side reads."""
    rec = 1.0 if 'Full' in scoring_mode else (0.5 if 'Half' in scoring_mode else 0.0)
    return {**DEFAULT_SCORING, 'rec': rec}


def _render_weekly_market_pull(wk_scoring, name_pool):
    """
    Live player-prop lines -> a market-implied projected-points column,
    from the SAME free weekly board (PrizePicks/Underdog/DraftKings - no
    key, no quota) the Live Odds tab already pulls
    (data.odds_weekly.weekly_props), scored the way Draft HQ scores a
    season-long book line (data.odds_weekly.weekly_market_projection).

    This is the live CURRENT slate - books post the coming weekend's board
    Tuesday/Wednesday and there is no way to ask them for a past week's
    lines, so unlike the model and the FantasyPros pull this can't be
    retargeted at an arbitrary year/week from the selectors above. A
    caption says so plainly rather than silently mismatching.
    """
    with st.expander("📈 Player prop lines → market projection (live)", expanded=False):
        st.caption(
            "PrizePicks, Underdog and DraftKings — no key, no quota. The same live weekly board "
            "the Live Odds tab pulls, scored under the scoring mode selected above into a "
            "projected-points column here instead of a raw line list. This is always the CURRENT "
            "posted slate, not necessarily the season/week picked above."
        )
        force = st.button("🔄 Refresh player props", key="wr_market_refresh")
        with skeleton_loader("table", n_rows=5, n_cols=4):
            props, meta = weekly_props(force=force)
        stamp = meta.get('fetched_at')
        if stamp:
            age_bits = [f"Pulled {stamp:%b %d, %Y at %I:%M %p}"]
            if meta.get('stale'):
                age_bits.append("stale — a fresh pull didn't return anything")
            st.caption(" · ".join(age_bits))
        if props.empty:
            st.caption("No live player props available right now.")
            return pd.DataFrame()
        scored, mmeta = weekly_market_projection(props, _scoring_dict(wk_scoring), board=name_pool)
        if scored.empty:
            st.caption("Lines came back but none mapped to a stat this app scores.")
            return pd.DataFrame()
        st.success(f"{mmeta['players']} players priced from the live board.")
        return scored.rename(columns={'player': 'Player'})


def _attach_by_name(base_df, other_df, value_cols, prefix):
    """
    Merge `other_df`'s columns onto `base_df` by player name, two-tier
    (exact, then suffix-stripped fallback) - same matcher every cross-source
    join in this app uses, since no two sources spell every name identically.
    """
    if other_df is None or other_df.empty or 'Player' not in other_df.columns:
        return base_df
    exact_map = {clean_name_exact(pd.Series([p])).iloc[0]: p for p in other_df['Player']}
    loose_map = {clean_name_for_merge(pd.Series([p])).iloc[0]: p for p in other_df['Player']}
    lookup = other_df.set_index('Player')

    def _match(name):
        key = clean_name_exact(pd.Series([name])).iloc[0]
        if key in exact_map:
            return exact_map[key]
        key = clean_name_for_merge(pd.Series([name])).iloc[0]
        return loose_map.get(key)

    matched = base_df['Player'].map(_match)
    out = base_df.copy()
    for col in value_cols:
        if col in lookup.columns:
            out[f'{prefix}{col}'] = matched.map(lookup[col])
    return out


def _fantasypros_points_column(scoring_mode):
    return {'Full PPR': 'FP Proj Pts PPR', 'Half-PPR': 'FP Proj Pts Half',
           'Standard': 'FP Proj Pts'}[scoring_mode]


def _positional_rank_col(df, value_col):
    """
    A positional rank column ('RB4' style, matching the app's existing
    convention) from any points column - one per projection SOURCE (model,
    market, FantasyPros), so the three can be read and compared side by
    side rather than only this app's own model carrying a rank at all.

    Sentinel-safe the same way every other rank column in this app is
    (HANDOFF.md gotcha #5): a player this source has no number for gets a
    real blank, not a NaN that could sort to the top of the grid.
    """
    if value_col not in df.columns or 'Pos' not in df.columns:
        return pd.Series([None] * len(df), index=df.index, dtype=object)
    pos_rank = df.groupby('Pos')[value_col].rank(ascending=False, method='first')
    out = df['Pos'].astype(str) + pos_rank.astype('Int64').astype(str)
    return out.where(df[value_col].notna(), None)


def render():
    st.markdown("<div class='custom-section-header'>WEEKLY RANKINGS</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        # WITH_UPCOMING (not AVAILABLE_SEASONS) so the season about to start
        # is selectable and defaults FIRST (index=0) - unlike the tabs that
        # default to index=1 to skip a season with no weekly stats yet
        # (Player Search, Depth Charts, ...), this tab's whole job is "what
        # should I do this week", so once the new season's schedule is live
        # the always-useful default is the CURRENT/upcoming week, not a
        # stale prior season's final week (see _default_week_index below -
        # a season with no games played yet opens on its Week 1).
        wk_year = st.selectbox("Season", AVAILABLE_SEASONS_WITH_UPCOMING, index=0, key="weekly_rank_year")
    weeks = _week_options(wk_year)
    week_labels = [w['label'] for w in weeks]
    with c2:
        wk_week_idx = st.selectbox("Week", range(len(weeks)), index=_default_week_index(weeks),
                                   format_func=lambda i: week_labels[i], key="weekly_rank_week")
    wk_week = weeks[wk_week_idx]['week']
    wk_week_completed = bool(weeks[wk_week_idx].get('completed')) if weeks else False
    with c3:
        wk_scoring = st.selectbox("Scoring", ["Full PPR", "Half-PPR", "Standard"], key="weekly_rank_scoring")

    with skeleton_loader("table", n_rows=10, n_cols=7):
        df_stats, t_col, n_col, _ = load_and_merge_data(wk_year, wk_scoring)
        model_df, model_meta = build_weekly_projections(wk_year, wk_week, wk_scoring)

    form_df = build_recent_form_rank(df_stats, n_col, t_col, n_weeks=RECENT_FORM_GAMES)

    st.markdown("### This app's weekly model")
    fp_weekly = _render_fantasypros_weekly_pull(wk_year, wk_week, wk_scoring)
    # Sportsbooks only ever post the CURRENT live slate (see odds_weekly's
    # module docstring - there is no way to ask a book for a past week's
    # lines), so pulling it for an ALREADY-PLAYED week would silently
    # attach this week's live board to a table of players from a different
    # week entirely - the exact "totally looks off" mismatch reported
    # against this column. Only offered for the upcoming/in-progress week.
    if wk_week_completed:
        market_df = None
        st.caption(
            f"📈 Market Proj Pts isn't shown for {wk_year} Week {wk_week} — that week is already "
            "played, and sportsbooks only post the current live slate. Pick the upcoming week to see it."
        )
    else:
        market_df = _render_weekly_market_pull(wk_scoring, model_df[['Player']] if not model_df.empty else None)
    _fantasypros_freshness_caption(wk_year, wk_week, wk_scoring)

    if model_df.empty:
        st.info(f"This app's model has no projection for {wk_year} week {wk_week}: "
               f"{model_meta.get('reason', 'not enough data yet')}")
        if fp_weekly is not None:
            st.caption("Showing FantasyPros' live weekly projection instead:")
            fp_pts_col = _fantasypros_points_column(wk_scoring)
            cols = ['Player', 'Pos', 'Team'] + ([fp_pts_col] if fp_pts_col in fp_weekly.columns else [])
            display_df = position_filter_multiselect(fp_weekly[cols].rename(
                columns={fp_pts_col: 'FantasyPros Proj Pts'}), key="weekly_rank_pos_filter")
            display_df = _limit_rows(display_df, key="weekly_rank_show_n")
            indexed = display_df.set_index('Player')
            st.dataframe(style_plain_dataframe(indexed), width="stretch",
                        height=df_auto_height(min(len(display_df), 40)))
    else:
        merged_model = model_df.copy()
        if fp_weekly is not None:
            merged_model = _attach_by_name(
                merged_model, fp_weekly,
                ['FP Proj Pts', 'FP Proj Pts PPR', 'FP Proj Pts Half'] + _MODEL_STAT_COLS, 'FP ')
            src_col = 'FP ' + _fantasypros_points_column(wk_scoring)
            if src_col in merged_model.columns:
                merged_model = merged_model.rename(columns={src_col: 'FantasyPros Proj Pts'})
        if market_df is not None and not market_df.empty:
            merged_model = _attach_by_name(merged_model, market_df, ['Market Pts', 'Coverage'], 'Mkt ')
            merged_model = merged_model.rename(
                columns={'Mkt Market Pts': 'Market Proj Pts', 'Mkt Coverage': 'Market Coverage'})
            if 'Market Coverage' in merged_model.columns:
                # Missing stats price as ZERO in Market Proj Pts by design
                # (data.odds_projections.score_market_lines' own docstring -
                # keeping the market number independent of this app's model
                # means a gap can't be filled in), which is exactly why a
                # partial line reads "very low" with no explanation unless
                # this rides right alongside it - a book that only posted a
                # receptions prop for someone shows a real but partial
                # number, not a broken one.
                merged_model['Market Coverage'] = merged_model['Market Coverage'].map(
                    lambda v: f"{v * 100:.0f}%" if pd.notna(v) else None)
        if not form_df.empty:
            merged_model = _attach_by_name(merged_model, form_df, ['Recent Avg FPTS'], '')
            merged_model = merged_model.rename(columns={'Recent Avg FPTS': 'L5 Avg FPTS'})

        # Ranking column, directly after Opponent, colored by TIER rather
        # than a continuous scale - explicit request. Tiers are clustered
        # per position on Model Proj Pts wherever a significant cutoff
        # actually falls (data.draft_board.tier_by_position, the same
        # k-means-on-points technique Draft HQ's board tiers with), not a
        # fixed players-per-tier bucket.
        merged_model['_tier'] = tier_by_position(merged_model, 'Model Proj Pts', pos_col='Pos')
        merged_model['Model Rank'] = _positional_rank_col(merged_model, 'Model Proj Pts')
        # One rank column per projection SOURCE, not just this app's own
        # model - explicit request. Same positional-rank shape as Model
        # Rank so the three read consistently side by side; only shown when
        # that source actually produced a points column to rank.
        if 'Market Proj Pts' in merged_model.columns:
            merged_model['Market Rank'] = _positional_rank_col(merged_model, 'Market Proj Pts')
        if 'FantasyPros Proj Pts' in merged_model.columns:
            merged_model['FantasyPros Rank'] = _positional_rank_col(merged_model, 'FantasyPros Proj Pts')

        # Does our model's rank actually MATCH FantasyPros' own published
        # consensus rank (their real ECR, not the positional rank derived
        # above from their points projection - the two aren't guaranteed to
        # agree, since a projection reflects only the stat line while a
        # published ECR also folds in analyst judgment calls a raw point
        # total won't capture) - explicit request. Reads whatever weekly
        # FantasyPros export is already sitting in this session's uploader
        # (below) without requiring it to be re-uploaded once it's there;
        # see that uploader's own docstring for why a live per-week ECR
        # pull isn't wired in here (that endpoint exists on FantasyPros'
        # API - see draft_sources.py's "/rankings or /consensus-rankings"
        # note - but is unverified against a real response in this app and
        # a wrong guess would silently spend the call budget on garbage;
        # the CSV export is the same real ECR with zero guessing involved).
        _weekly_ecr_df = st.session_state.get('_weekly_rank_ecr_df')
        if _weekly_ecr_df is not None and not _weekly_ecr_df.empty:
            ecr_comparison = build_rankings_comparison(
                merged_model, value_col='Model Proj Pts', rank_label='Model Overall',
                fp_df=_weekly_ecr_df)
            if not ecr_comparison.empty and 'FantasyPros Rank' in ecr_comparison.columns:
                ecr_comparison = ecr_comparison.rename(columns={
                    'FantasyPros Rank': 'FantasyPros ECR',
                    'Model Overall vs FantasyPros': 'Model vs FantasyPros ECR',
                })
                keep = [c for c in ('Player', 'FantasyPros ECR', 'Model vs FantasyPros ECR')
                       if c in ecr_comparison.columns]
                merged_model = merged_model.merge(ecr_comparison[keep], on='Player', how='left')

        display_cols = ['Player', 'Pos', 'Team', 'Opponent', 'Model Rank']
        if 'Market Rank' in merged_model.columns:
            display_cols.append('Market Rank')
        if 'FantasyPros Rank' in merged_model.columns:
            display_cols.append('FantasyPros Rank')
        if 'FantasyPros ECR' in merged_model.columns:
            display_cols += ['FantasyPros ECR', 'Model vs FantasyPros ECR']
        display_cols += [label for col, label in _STAT_DISPLAY_COLS if col in merged_model.columns]
        merged_model = merged_model.rename(columns=dict(_STAT_DISPLAY_COLS))
        display_cols.append('Model Proj Pts')
        if 'Market Proj Pts' in merged_model.columns:
            display_cols.append('Market Proj Pts')
        if 'Market Coverage' in merged_model.columns:
            display_cols.append('Market Coverage')
        if 'FantasyPros Proj Pts' in merged_model.columns:
            display_cols.append('FantasyPros Proj Pts')
        if 'L5 Avg FPTS' in merged_model.columns:
            display_cols.append('L5 Avg FPTS')
        display_cols.append('Injury Status')

        keep_cols = [c for c in display_cols if c in merged_model.columns] + ['_tier']
        filtered_df = position_filter_multiselect(merged_model[keep_cols], key="weekly_rank_pos_filter")
        total_filtered = len(filtered_df)
        display_df = _limit_rows(filtered_df, key="weekly_rank_show_n")
        tier_values = display_df['_tier'].tolist()
        display_df = display_df.drop(columns=['_tier'])
        indexed = display_df.set_index('Player')

        # A sparkline next to the bare L5 average - same in-house pattern
        # Risers/Waiver Wire already uses (data.transforms.build_recent_trend
        # + st.column_config.LineChartColumn), reused rather than reinvented
        # so a real per-week trend is visible without opening Player Search.
        trend_map = build_recent_trend(df_stats, n_col, metric='fantasy_points', n_weeks=RECENT_FORM_GAMES)
        indexed['Last 5 Wks'] = [trend_map.get(name, []) for name in indexed.index]

        pct_cols = {}
        for c in ('Model Proj Pts', 'Market Proj Pts', 'FantasyPros Proj Pts', 'L5 Avg FPTS'):
            if c in indexed.columns and indexed[c].notna().any():
                pct_cols[c] = calculate_percentile(indexed.reset_index(), c)
        column_config = build_column_help_config(indexed, pinned_cols=['Team', 'Pos', 'Opponent', 'Model Rank'])
        column_config['Last 5 Wks'] = st.column_config.LineChartColumn(
            help="Fantasy points trend over this player's last 5 games played", width="small",
        )
        st.dataframe(
            style_plain_dataframe(indexed, numeric_pct_cols=pct_cols, tier_cols={'Model Rank': tier_values}),
            width="stretch", height=df_auto_height(min(len(display_df), 40)),
            column_config=column_config,
        )
        if total_filtered > len(display_df):
            st.caption(f"Showing {len(display_df)} of {total_filtered} players — widen \"Show\" above to see more.")
        st.caption(
            "**Model Rank** / **Market Rank** / **FantasyPros Rank** are each source's own "
            "positional rank (e.g. \"RB4\"), so the three can be read side by side rather than "
            "only this app's model carrying a rank at all — Model Rank is shaded by tier, a "
            "cluster break in Model Proj Pts at that position, not a fixed players-per-tier "
            "cutoff. **FantasyPros ECR** (when a weekly FantasyPros export is uploaded below) is "
            "their own REAL published consensus rank, not derived from the points projection above "
            "it — **Model vs FantasyPros ECR** shows how far apart the two actually are, which is "
            "not always the same story a proj-pts comparison alone would tell, since a published "
            "ECR also folds in analyst judgment a raw stat-line projection doesn't capture. "
            "**Model Proj Pts** is this app's own projection (usage blended toward the current "
            "season as it grows, opponent/pace/game-script adjusted - see "
            "docs/weekly_projections_methodology.md), with the stat line it's built from shown "
            "alongside it. **Market Proj Pts** is this week's live sportsbook player-prop lines "
            "re-scored under this league's settings — missing stats price as zero, so **Market "
            "Coverage** shows how much of a typical week's points the posted lines actually cover; "
            "a low number means a partial line, not a bad projection. **FantasyPros Proj Pts** is "
            "their analysts' number, pulled live above. Independent reads, shown side by side, "
            "never blended."
        )

    if form_df.empty:
        st.info(f"Not enough {wk_year} weekly data yet to build a recent-form baseline.")
        return

    st.markdown("---")
    st.markdown("**Upload a weekly rankings export** (FantasyPros weekly export, or any CSV with a player-name column)")
    st.caption(
        "A real FantasyPros weekly export (not a custom one) also feeds the **FantasyPros ECR** / "
        "**Model vs FantasyPros ECR** columns in the model table above, on the NEXT rerun after "
        "uploading — their actual published rank, not one derived from a points projection."
    )
    import_hint('fantasypros_weekly')
    weekly_upload = st.file_uploader("Weekly rankings CSV", type=["csv"], key="weekly_rank_upload")
    weekly_df = None
    if weekly_upload is not None:
        weekly_df = parse_fantasypros_upload(weekly_upload)
        # Only a REAL FantasyPros-shaped export counts as "ECR" for the top
        # table's comparison - a generic custom upload (parsed below as a
        # fallback) could be anyone's ranking, and labelling that "FantasyPros
        # ECR" would just be wrong.
        st.session_state['_weekly_rank_ecr_df'] = weekly_df if not weekly_df.empty else None
        if weekly_df.empty:
            weekly_df = parse_custom_rankings(weekly_upload)
        if weekly_df is None or weekly_df.empty:
            st.warning("Couldn't find a player-name column in that upload — expected a header like 'Player', 'Player Name', or 'Name'.")

    if weekly_df is None or weekly_df.empty:
        return

    comparison = build_rankings_comparison(form_df, value_col='Recent Avg FPTS', rank_label='Form', fp_df=weekly_df)
    extra_cols = [c for c in comparison.columns if c not in ('Player', 'Pos', 'Team')]
    merged = form_df.merge(comparison[['Player'] + extra_cols], on='Player', how='left')
    merged = merged.rename(columns={'Recent Avg FPTS': 'L5 Avg FPTS'})

    display_df = position_filter_multiselect(merged, key="weekly_rank_upload_pos_filter")
    display_df = _limit_rows(display_df, key="weekly_rank_upload_show_n")

    diverging_cols = {}
    delta_col = 'Form vs FantasyPros'
    if delta_col in display_df.columns and display_df[delta_col].notna().any():
        max_abs = display_df[delta_col].abs().max()
        if max_abs and max_abs > 0:
            diverging_cols[delta_col] = max_abs

    indexed = display_df.set_index('Player')
    st.dataframe(
        style_plain_dataframe(indexed, diverging_cols=diverging_cols),
        width="stretch", height=df_auto_height(min(len(display_df), 40)),
        column_config=build_column_help_config(indexed, pinned_cols=['Team', 'Pos']),
    )
