"""
Track Record sub-tab (Weekly Fantasy): the live record of the boards saved before each game (data/ledger), scored
against what happened - the model vs the market and vs FantasyPros, week by week, on the same players. Requested
2026-10-13 ("a track-record page in the app ... so you can see how much to trust a model call that bucks the market").
All of the scoring lives in data.ledger_scorecard; this module only lays it out.
"""
import glob
import os

import numpy as np
import pandas as pd
import streamlit as st

from data import ledger_scorecard as lsc
from data import prop_record as prec
from data import prop_tiers as ptiers
from data.loaders import load_schedule
from data.prediction_ledger import LEDGER_DIR
from data.transforms import load_and_merge_data
from ui.styling import df_auto_height, style_plain_dataframe
from ui.components import quiet_note  # noqa: E402

_STAT_LABELS = {
    'passing_attempts': 'Pass attempts', 'passing_completions': 'Completions', 'passing_yards': 'Passing yards',
    'passing_tds': 'Passing TDs', 'passing_interceptions': 'Interceptions', 'rushing_attempts': 'Carries',
    'rushing_yards': 'Rushing yards', 'rushing_tds': 'Rushing TDs', 'targets': 'Targets',
    'receptions': 'Receptions', 'receiving_yards': 'Receiving yards', 'receiving_tds': 'Receiving TDs',
}
_LEAN_MIN_GAP = {'receiving_yards': 10.0, 'rushing_yards': 10.0, 'passing_yards': 20.0, 'receptions': 0.75}


def _ledger_signature(year):
    """Changes whenever a ledger file for ``year`` is added or rewritten - the cache key for the scored frame."""
    paths = sorted(glob.glob(os.path.join(LEDGER_DIR, f'{year}_wk*_*.parquet')))
    return tuple((os.path.basename(p), int(os.path.getmtime(p))) for p in paths)


@st.cache_data(show_spinner=False, ttl=3600)
def _scored(year, _signature):
    stats_df, _team_col, name_col, _ = load_and_merge_data(year, 'Full PPR')
    if stats_df is None or stats_df.empty or 'week' not in stats_df.columns:
        return pd.DataFrame(), {'files': 0, 'scoring': None, 'mixed_scoring': False}
    return lsc.load_scored_ledger(year, stats_df, name_col, load_schedule(year))


def _archive_signature(year):
    """Changes whenever a props snapshot for ``year`` is added - with the ledger signature, the cache key for the prop record."""
    paths = sorted(glob.glob(os.path.join(prec.ARCHIVE, f'{year}_wk*.json')))
    return tuple((os.path.basename(p), int(os.path.getmtime(p))) for p in paths)


@st.cache_data(show_spinner=False, ttl=3600)
def _prop_rows(year, _ledger_sig, _archive_sig):
    stats_df, _team_col, name_col, _ = load_and_merge_data(year, 'Full PPR')
    try:
        prior, _t, prior_name_col, _ = load_and_merge_data(year - 1, 'Full PPR')
    except Exception:
        prior, prior_name_col = None, None
    rows, _info = prec.build_record(year, stats_df, name_col, prior, prior_name_col, load_schedule(year))
    return rows


_BLIND = ('always under', 'always over', 'book favourite')


def _render_prop_record(year, pos):
    """2026-11-11 (audit B3): the prop record against the lines the books posted, at the price of the side taken, beside
    the blind strategies that need no model. Scored on the line posted when the board was built (what could have been bet)."""
    st.markdown("**Props: the model's side against the posted line and price**")
    try:
        rows = _prop_rows(year, _ledger_signature(year), _archive_signature(year))
    except Exception:
        rows = pd.DataFrame()
    if rows.empty:
        st.caption("No archived priced lines for played weeks yet.")
        return
    if pos is not None:
        rows = rows[rows['Pos'].eq(pos)]
    all_rows = rows
    rows = rows[rows['timing'].eq('at board')]
    base = prec.baseline_table(rows)
    if base.empty:
        st.caption("Too few priced lines for this filter.")
        return
    allrow = base[base['stat'].eq('ALL')].iloc[0]
    best = max(_BLIND, key=lambda b: allrow[f'{b} roi'] if np.isfinite(allrow[f'{b} roi']) else -np.inf)
    c1, c2, c3 = st.columns(3)
    c1.metric("Model 5%+ edges", f"{100 * allrow['model 5%+ hit']:.1f}% hit",
              delta=f"{100 * allrow['model 5%+ roi']:+.1f}% ROI", help=f"{int(allrow['model 5%+ bets'])} bets at the posted price.")
    c2.metric("Best blind strategy", best.capitalize(), delta=f"{100 * allrow[f'{best} roi']:+.1f}% ROI",
              help="Betting one side of every line with no model at all.")
    c3.metric("Model lift", f"{100 * allrow['roi lift vs best blind']:+.1f}% ROI",
              help="The model's side on every line, ROI minus the best blind strategy's. Below zero: the model is not adding to it yet.")
    stat_rows = base[~base['stat'].eq('ALL')]
    shown = pd.DataFrame({
        'Stat': stat_rows['stat'].map(_STAT_LABELS).fillna(stat_rows['stat']), 'Lines': stat_rows['lines'],
        'Under ROI %': 100 * stat_rows['always under roi'], 'Over ROI %': 100 * stat_rows['always over roi'],
        'Favourite ROI %': 100 * stat_rows['book favourite roi'],
        'Model 5%+ bets': stat_rows['model 5%+ bets'], 'Model 5%+ hit %': 100 * stat_rows['model 5%+ hit'],
        'Model 5%+ ROI %': 100 * stat_rows['model 5%+ roi'], 'Model lift %': 100 * stat_rows['roi lift vs best blind']})
    st.dataframe(style_plain_dataframe(shown, diverging_cols={'Model lift %': 5.0, 'Model 5%+ ROI %': 5.0},
                                       decimals_by_col={c: 1 for c in shown.columns if c.endswith('%')}),
                 hide_index=True, width="stretch", height=df_auto_height(len(shown)))
    _render_tier_record(all_rows, year)
    books = prec.book_table(rows)
    sides = prec.side_table(rows)
    if not books.empty or not sides.empty:
        b1, b2 = st.columns(2)
        if not books.empty:
            bt = pd.DataFrame({'Book': books['book'], 'Bets': books['bets'], 'Hit %': 100 * books['hit'],
                               'ROI %': 100 * books['roi_price']})
            b1.dataframe(style_plain_dataframe(bt, decimals_by_col={'Hit %': 1, 'ROI %': 1}), hide_index=True,
                         width="stretch", height=df_auto_height(len(bt)))
        if not sides.empty:
            sd = sides[sides['edge_min'].eq(0.05)]
            stt = pd.DataFrame({'Model side': sd['side'].str.capitalize(), 'Bets': sd['bets'], 'Hit %': 100 * sd['hit'],
                                'ROI %': 100 * sd['roi_price']})
            b2.dataframe(style_plain_dataframe(stt, decimals_by_col={'Hit %': 1, 'ROI %': 1}), hide_index=True,
                         width="stretch", height=df_auto_height(len(stt)))
    quiet_note("Every priced line archived for a played week, at the line posted when the board was built. The model's side is "
               "the one where its P(over) beats the book's no-vig price; 5%+ = by at least 5 points. ROI is at the price "
               "actually posted for that side. Lift = the model's side on every line minus the best of the blind strategies "
               "(always under, always over, the side the price favours) - a record that only matches the blind under has found "
               "nothing. The stat table uses the sharpest book's line; the book table scores each book's own line and price; "
               "the side table is 5%+ edges. Full detail: scripts/prop_scorecard.py.", "Details")


def _render_tier_record(rows, year):
    """B2 (2026-11-11): the v0 tiers' record. The status reads only boards from data.prop_tiers.FORWARD_FROM on - the rules were
    chosen from the weeks before it, so those are shown as in-sample background and can never make a tier bettable."""
    fwd = ptiers.forward_rows(rows)
    if fwd.empty:
        return
    live = ptiers.tier_record(fwd, year, forward_only=True).set_index('tier')
    past = ptiers.tier_record(fwd[~ptiers.is_forward(year, fwd['week'])], year).set_index('tier')
    y, w = ptiers.FORWARD_FROM
    shown = pd.DataFrame({
        'Tier': list(ptiers.ORDER),
        'Bets': [int(live.at[t, 'bets']) for t in ptiers.ORDER],
        'Hit %': [f"{100 * live.at[t, 'hit shown']:.1f}" if live.at[t, 'bets'] else '-' for t in ptiers.ORDER],
        'ROI %': [f"{100 * live.at[t, 'roi_price']:+.1f}" if live.at[t, 'bets'] else '-' for t in ptiers.ORDER],
        'Status': [live.at[t, 'status'] for t in ptiers.ORDER],
        'Earlier bets': [int(past.at[t, 'bets']) for t in ptiers.ORDER],
        'Earlier hit %': [f"{100 * past.at[t, 'hit']:.1f}" if past.at[t, 'bets'] else '-' for t in ptiers.ORDER],
        'Earlier ROI %': [f"{100 * past.at[t, 'roi_price']:+.1f}" if past.at[t, 'bets'] else '-' for t in ptiers.ORDER]})
    st.dataframe(style_plain_dataframe(shown),
                 hide_index=True, width="stretch", height=df_auto_height(len(shown)))
    quiet_note(f"Tiers ({ptiers.VERSION}): A = receiving yards, receptions or a rushing-yards under, the model on the side the "
               "book's price leans to, 3-25 points from the book's probability; B = the same against the lean; No bet = "
               "passing stats, carries, or a gap over 25 points; Track = the rest. Bets, Hit % and ROI % count boards from "
               f"{y} week {w} on, when these rules were fixed; Hit % is pulled toward 50% until the sample is large. A tier "
               "reads bettable only after 300 such bets with ROI at the posted price clear of zero (90% lower bound). The "
               "earlier columns are the weeks the rules were chosen from, so they cannot prove them.", "Tiers")


def _weekly_table(comp, bench):
    out = pd.DataFrame({
        'Week': comp['week'].astype(str),
        'Players': comp['n'],
        'Model error': comp['model_rmse'],
        f'{bench} error': comp['bench_rmse'],
        'Model edge': comp['bench_rmse'] - comp['model_rmse'],
        'Model bias': comp['model_bias'],
        f'{bench} bias': comp['bench_bias'],
        'Model ranking %': 100 * comp['model_pairwise'],
        f'{bench} ranking %': 100 * comp['bench_pairwise'],
    })
    return out


def _render_summary(comp, bench):
    weekly = comp[comp['week'] != 'All']
    pooled = comp[comp['week'] == 'All']
    if pooled.empty:
        return
    p = pooled.iloc[0]
    better = int((weekly['model_rmse'] < weekly['bench_rmse']).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model error", f"{p['model_rmse']:.2f}",
              delta=f"{p['model_rmse'] - p['bench_rmse']:+.2f} vs {bench}", delta_color="inverse",
              help="Root-mean-square miss in fantasy points, all scored weeks pooled.")
    c2.metric(f"{bench} error", f"{p['bench_rmse']:.2f}")
    c3.metric("Weeks model closer", f"{better} of {len(weekly)}")
    c4.metric("Ranking accuracy", f"{100 * p['model_pairwise']:.1f}%",
              delta=f"{100 * (p['model_pairwise'] - p['bench_pairwise']):+.1f} vs {bench}",
              help="Share of player pairs in the same week the projection put in the right order.")


def _render_disagreements(frame, pos):
    st.markdown("**When the model and the market disagree**")
    table = lsc.disagreement_table(frame, positions=(None,) if pos is None else (pos,))
    table = table[table['n'] > 0]
    if table.empty:
        st.caption("No player-weeks yet with both a model and a full market projection.")
        return
    shown = pd.DataFrame({
        'Gap': table['Gap'], 'Model was': table['Side'].str.replace('Model ', ''), 'Players': table['n'],
        'Model closer %': 100 * table['Model closer'], 'Model avg': table['Model avg'],
        'Market avg': table['Market avg'], 'Actual avg': table['Actual avg']})
    st.dataframe(style_plain_dataframe(shown, decimals_by_col={'Model closer %': 0}), hide_index=True,
                 width="stretch", height=df_auto_height(len(shown)))
    quiet_note("Players whose model and market projections differ by at least the gap, scored on who ended closer to "
               "the actual points. Under 50% means the market was usually right when the two disagreed. Small samples "
               "swing a lot - read the player counts.", "Details")


def _render_stats(frame, bench):
    st.markdown(f"**Stat by stat: model vs {bench}**")
    comp = lsc.stat_comparison(frame, bench)
    if comp.empty:
        st.caption(f"No stats with both a model and a {bench} projection yet.")
        return
    shown = pd.DataFrame({
        'Stat': comp['stat'].map(_STAT_LABELS).fillna(comp['stat']), 'Players': comp['n'],
        'Model error': comp['model_rmse'], f'{bench} error': comp['bench_rmse'],
        'Model edge %': 100 * (comp['bench_rmse'] - comp['model_rmse']) / comp['bench_rmse'],
        'Model bias': comp['model_bias'], f'{bench} bias': comp['bench_bias']})
    st.dataframe(style_plain_dataframe(shown, diverging_cols={'Model edge %': 10.0},
                                       decimals_by_col={'Model edge %': 1, 'Model error': 2, f'{bench} error': 2,
                                                        'Model bias': 2, f'{bench} bias': 2}),
                 hide_index=True, width="stretch", height=df_auto_height(len(shown)))
    quiet_note("Error = root-mean-square miss in that stat's own units (yards, catches, TDs). Model edge % is how much "
               f"smaller the model's miss was than {bench}'s (green = model better). Bias = average projection minus "
               "actual (negative = projected too low).", "Details")


def _render_line_calls(frame):
    st.markdown("**Model leans against the market's number**")
    calls = lsc.line_calls(frame, min_gaps=_LEAN_MIN_GAP)
    if calls.empty:
        st.caption("No market stat numbers to compare against yet.")
        return
    shown = pd.DataFrame({
        'Stat': calls['stat'].map(_STAT_LABELS).fillna(calls['stat']),
        'Lean size': np.where(calls['min_gap'] > 0, 'over ' + calls['min_gap'].map('{:g}'.format), 'any'),
        'Calls': calls['n'], 'Overs': calls['overs'], 'Over hit %': 100 * calls['over_hit'],
        'Unders': calls['unders'], 'Under hit %': 100 * calls['under_hit'], 'Hit %': 100 * calls['hit']})
    st.dataframe(style_plain_dataframe(shown, decimals_by_col={'Over hit %': 0, 'Under hit %': 0, 'Hit %': 0}),
                 hide_index=True, width="stretch", height=df_auto_height(len(shown)))
    quiet_note("A lean is the side of the market's number the model sits on (model higher = over). Hit % is how often "
               "the actual result landed on that side. The market number is the de-vigged consensus the board shows, "
               "not one book's posted line, and a hit rate needs to clear roughly 52.4% to beat -110 pricing.", "Details")


def render():
    st.markdown("<div class='custom-section-header'>MODEL TRACK RECORD — LIVE BOARDS VS. WHAT HAPPENED</div>",
                unsafe_allow_html=True)
    years = lsc.ledger_years()
    if not years:
        st.info("No saved boards yet. Every 'Build board' on Weekly Rankings saves one; this page scores them once the "
                "games are played.")
        return
    c1, c2, c3, c4 = st.columns([1, 1.3, 2.2, 1.4])
    year = c1.selectbox("Season", years, index=0, key="track_year")
    bench = c2.radio("Compare against", ["Market", "FantasyPros"], horizontal=True, key="track_bench")
    pos_label = c3.radio("Position", ["All", "QB", "RB", "WR", "TE"], horizontal=True, key="track_pos")
    pool = c4.radio("Players", ["Startable", "Everyone"], horizontal=True, key="track_pool",
                    help="Startable = the top 24 QB / 40 RB / 55 WR / 20 TE by either projection each week, the "
                         "players a lineup decision is actually about. Everyone adds the deep bench, which is easy "
                         "to project and flatters every source.")
    pos = None if pos_label == "All" else pos_label

    with st.spinner("Scoring saved boards…"):
        frame, info = _scored(year, _ledger_signature(year))
    if frame.empty:
        st.info(f"No {year} boards can be scored yet: none of the weeks with a saved board has been played.")
        return
    if info.get('mixed_scoring'):
        st.caption(f"Boards were saved in more than one scoring format; only the {info['scoring']} ones are scored.")
    if info.get('unknown_scoring'):
        st.caption(f"{info['unknown_scoring']} saved board(s) did not match a scoring format and are scored as "
                   f"{info['scoring']}.")

    live = frame[frame['pre_kickoff']]
    comp = lsc.weekly_comparison(frame, bench, pos=pos, startable=(pool == "Startable"))
    if comp.empty:
        st.info(f"No played weeks yet where both the model and {bench} projected these players before kickoff.")
    else:
        _render_summary(comp, bench)
        table = _weekly_table(comp, bench)
        st.dataframe(
            style_plain_dataframe(table, diverging_cols={'Model edge': 0.5},
                                  decimals_by_col={'Model error': 2, f'{bench} error': 2, 'Model edge': 2,
                                                   'Model bias': 2, f'{bench} bias': 2, 'Model ranking %': 1,
                                                   f'{bench} ranking %': 1}),
            hide_index=True, width="stretch", height=df_auto_height(len(table)))
        quiet_note(f"Each player is scored from the last board saved before his game kicked off, on the players both "
                   f"the model and {bench} projected. Error = root-mean-square miss in fantasy points ({info['scoring']}); "
                   "Model edge = how many points smaller the model's miss was (green = model better). Bias = average "
                   "projection minus actual. Ranking % = share of player pairs in the same week put in the right order.", "Details")

    st.divider()
    _render_disagreements(frame, pos)
    st.divider()
    _render_stats(frame if pos is None else frame[frame['Pos'] == pos], bench)
    st.divider()
    _render_line_calls(frame if pos is None else frame[frame['Pos'] == pos])
    st.divider()
    _render_prop_record(year, pos)

    rebuilt = frame[~frame['pre_kickoff']]
    rebuilt_weeks = sorted(set(rebuilt['week']) - set(live['week']))
    with st.expander("How this is scored", expanded=False):
        st.markdown(
            "- **Which board counts:** for each player, the last board saved before his own game kicked off (boards "
            "are saved by every 'Build board' on Weekly Rankings, at most once an hour). A board built after kickoff "
            "never stands in for him.\n"
            "- **Same players on both sides:** every model-vs-benchmark number uses only the players both projected. "
            "The market only prices part of each week (mostly the bigger names) and a market total that covers less "
            "than 90% of a player's line is left out.\n"
            "- **The model as it was:** these are the live boards, so earlier weeks reflect the model before later "
            "changes, not today's model.\n"
            "- **Benchmarks only:** the market and FantasyPros numbers are never fed into the model.")
        if rebuilt_weeks:
            wk_text = ', '.join(str(w) for w in rebuilt_weeks)
            st.markdown(f"- **Not in the live record:** week(s) {wk_text} had no board saved before kickoff (rebuilt "
                        "afterwards from data up to that week), so they are left out above.")
