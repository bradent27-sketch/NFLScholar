"""
Per-play "big play" bonus pricing: +N points for a single reception, rush,
or completed pass that goes for at least Y yards - a real scoring rule
(Sleeper and ESPN both support it) that `data.draft_board.YARDAGE_BONUSES`
cannot price, because those threshold a GAME's summed yardage and a big
play is an event on ONE PLAY. No `PROJECTED_STATS` column carries play
length at all, so this is a genuinely new signal end to end: real per-play
gain lengths from play-by-play (`data.loaders.load_pbp`), which nothing in
the draft stack touched before this.

RULES ARE USER-DEFINED AND UNBOUNDED - any (category, yard threshold,
points) triple, as many as the user wants, added or removed live from Draft
HQ's League Scoring panel (`ui.tabs.draft_hq`'s Big Play Bonuses editor).
`YARDAGE_BONUSES` is a fixed, hardcoded list of twelve thresholds; this is
deliberately not, since a big-play rule can be set at any yardage line a
real league actually uses (30, 40, 50, ...).

WHY A BLENDED RATE, not a raw season count. A season's worth of 40+ yard
plays is a tiny, high-variance sample - one more or one fewer swings a
back's own rate by a third. Priced the same way `build_player_rates` prices
every other per-touch number in this stack: a recency-weighted OWN rate,
blended toward the POSITION's rate with the player's own qualifying-touch
count as the evidence, so a rookie or a 15-touch role player rides mostly
on the positional baseline instead of one lucky/unlucky play. Evidence and
the self-weight ceiling follow `draft_projections.STAT_SELF_WEIGHT`'s exact
shape (`weight = evidence * self_weight`, never letting even a fully-proven
veteran's own count fully override the positional read on a rare tail
event).

`_weekly_history` is imported from `data.draft_projections` rather than
reimplemented - it is the same multi-season weekly frame (+ its
`player_id`/name-column bridge) `build_player_rates` itself uses for every
other rate in the projection engine, so a big-play rate is keyed on exactly
the same player identity as everything it gets multiplied against.
"""
import numpy as np
import pandas as pd
import streamlit as st

from data.loaders import load_pbp
from data.draft_projections import _weekly_history, SEASON_RECENCY_WEIGHTS
from data.utils import clean_name_exact

BIG_PLAY_CATEGORIES = ('reception', 'rush', 'pass_completion')

CATEGORY_LABELS = {
    'reception': 'Reception', 'rush': 'Rush', 'pass_completion': 'Pass completion',
}
LABEL_TO_CATEGORY = {v: k for k, v in CATEGORY_LABELS.items()}

# Which already-projected PROJECTED_STATS column a category's play count is
# a rate OF. Pass completions are priced per ATTEMPT rather than per
# completion - PROJECTED_STATS has no separate completions column (only
# `attempts` is modeled) - and per-attempt already folds completion rate
# into the one number, which is the correct thing to multiply a QB's
# projected attempts by anyway.
BIG_PLAY_TOUCH_STAT = {
    'reception': 'receptions', 'rush': 'carries', 'pass_completion': 'attempts',
}

# Touches (not games) before a player's OWN big-play rate is trusted
# outright over the positional baseline. Set well above the ~24-game
# full-trust mark STAT_SELF_WEIGHT's evidence uses elsewhere, because a big
# play is a rare tail event - a real starting back might clear 40+ yards on
# under 5% of carries - so the same sample carries far fewer QUALIFYING
# events to learn a rate from than an every-touch stat like yards/carry.
FULL_TRUST_TOUCHES = 300.0
# Matches draft_projections.DEFAULT_SELF_WEIGHT - even a fully-proven
# veteran's own rate caps out contributing this much of the blend, same
# ceiling logic as every other per-touch rate in the draft stack.
BIG_PLAY_SELF_WEIGHT = 0.45

_REQUIRED_PBP_COLS = {'play_type', 'yards_gained', 'complete_pass', 'season_type',
                      'receiver_player_id', 'rusher_player_id', 'passer_player_id'}


@st.cache_data(show_spinner=False)
def _big_play_history(latest_season, n_seasons=5):
    """
    One row per qualifying play (reception / rush / completed pass) across
    up to n_seasons ending at latest_season: season, category, gsis_id,
    yards_gained. REG season only - load_pbp doesn't filter this itself,
    same fix build_redzone_usage already applies to the same raw source.
    """
    frames = []
    for yr in range(int(latest_season), int(latest_season) - int(n_seasons), -1):
        pbp = load_pbp(yr)
        if pbp is None or pbp.empty or not _REQUIRED_PBP_COLS.issubset(pbp.columns):
            continue
        reg = pbp[pbp['season_type'] == 'REG'] if 'season_type' in pbp.columns else pbp
        if reg.empty:
            continue
        gain = pd.to_numeric(reg['yards_gained'], errors='coerce')
        complete = reg['complete_pass'] == 1
        sacked = reg['sack'] == 1 if 'sack' in reg.columns else pd.Series(False, index=reg.index)

        # `complete` travels as its own column because 'pass_completion'
        # needs a TOUCH ROW for every attempt (to match PROJECTED_STATS'
        # `attempts`, official attempts EXCLUDE sacks by rule) but a HIT
        # only on the completed ones - reusing complete_pass==1 as the
        # inclusion filter itself (the earlier version of this function)
        # silently made the rate's denominator "completed passes" while the
        # caller multiplies it against PROJECTED ATTEMPTS, inflating every
        # QB's pass-completion bonus by roughly 1/completion-rate. Reception
        # and rush rows are already gated to real completed
        # catches/handoffs at inclusion time, so `complete` is trivially
        # True for both - only pass_completion rows carry a real mix.
        masks = (
            ('reception', 'receiver_player_id',
             reg['receiver_player_id'].notna() & complete, True),
            ('rush', 'rusher_player_id',
             (reg['play_type'] == 'run') & reg['rusher_player_id'].notna(), True),
            ('pass_completion', 'passer_player_id',
             reg['passer_player_id'].notna() & (reg['play_type'] == 'pass') & ~sacked, None),
        )
        for category, id_col, mask, forced_complete in masks:
            if not mask.any():
                continue
            sub = pd.DataFrame({
                'season': yr, 'category': category,
                'gsis_id': reg.loc[mask, id_col].to_numpy(),
                'yards_gained': gain.loc[mask].to_numpy(),
                'complete': (np.ones(int(mask.sum())) if forced_complete
                            else complete.loc[mask].astype(float).to_numpy()),
            })
            frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=['season', 'category', 'gsis_id', 'yards_gained', 'complete'])
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=['gsis_id', 'yards_gained'])


def _id_name_position_bridge(latest_season, n_seasons=5):
    """
    gsis_id -> (Player name, position), from the SAME weekly history
    build_player_rates itself keys on - one name format end to end, no
    third source to reconcile (see data.draft_projections._weekly_history).
    """
    df, name_col = _weekly_history(latest_season, n_seasons)
    if df is None or df.empty or 'player_id' not in df.columns:
        return pd.DataFrame(columns=['gsis_id', 'Player', 'position'])
    bridge = (df[['player_id', name_col, 'position']].dropna(subset=['player_id'])
              .drop_duplicates(subset=['player_id'], keep='last')
              .rename(columns={'player_id': 'gsis_id', name_col: 'Player'}))
    bridge['position'] = bridge['position'].astype(str).str.upper()
    return bridge


@st.cache_data(show_spinner=False)
def big_play_rate_table(category, threshold, latest_season, n_seasons=5):
    """
    Every player's blended big-play rate for one (category, threshold) rule:
    P(a single reception/rush/completed pass goes for >= threshold yards).

    Pooled at the PLAY level (weighted-hits / weighted-touches), not
    season-averaged, so a high-volume recent season isn't diluted by a
    shallow old one the way averaging two per-season rates would. Recency
    weights are the same SEASON_RECENCY_WEIGHTS ladder build_player_rates
    uses (most recent season 1.0, one year back 0.55, ...).

    Returns: Player, position, touches, own_rate, position_rate, evidence,
    rate (the blended number every caller should actually use).
    """
    plays = _big_play_history(latest_season, n_seasons)
    plays = plays[plays['category'] == category]
    if plays.empty:
        return pd.DataFrame(columns=['Player', 'position', 'touches', 'own_rate',
                                     'position_rate', 'evidence', 'rate'])

    seasons = sorted(plays['season'].unique(), reverse=True)
    weight_for = {s: (SEASON_RECENCY_WEIGHTS[i] if i < len(SEASON_RECENCY_WEIGHTS) else 0.05)
                  for i, s in enumerate(seasons)}
    plays = plays.copy()
    plays['_w'] = plays['season'].map(weight_for).fillna(0.05)
    # A hit needs BOTH the gain and completion - see the 'complete' column's
    # docstring in _big_play_history for why pass_completion rows include
    # incompletions/sacks-excluded attempts as touches that can never hit.
    hit = (plays['yards_gained'] >= float(threshold)) & (plays['complete'] == 1)
    plays['_wh'] = plays['_w'] * hit.astype(float)

    per_player = plays.groupby('gsis_id').agg(
        weighted_touches=('_w', 'sum'), weighted_hits=('_wh', 'sum'),
        touches=('_w', 'size'),
    ).reset_index()
    per_player['own_rate'] = (
        per_player['weighted_hits'] / per_player['weighted_touches'].replace(0, np.nan)
    ).fillna(0.0).clip(0.0, 1.0)

    bridge = _id_name_position_bridge(latest_season, n_seasons)
    per_player = per_player.merge(bridge, on='gsis_id', how='left').dropna(subset=['Player'])
    if per_player.empty:
        return pd.DataFrame(columns=['Player', 'position', 'touches', 'own_rate',
                                     'position_rate', 'evidence', 'rate'])

    # Positional baseline - pooled the same way (sum of weighted hits over
    # sum of weighted touches across the whole position), so a rookie or a
    # name-match miss still lands on a real, non-zero rate (see the
    # fallback in project_big_play_points below) instead of silently
    # scoring zero big-play bonus for players this table can't identify.
    pos_totals = per_player.groupby('position').agg(
        pos_hits=('weighted_hits', 'sum'), pos_touches=('weighted_touches', 'sum'))
    pos_totals['position_rate'] = (
        pos_totals['pos_hits'] / pos_totals['pos_touches'].replace(0, np.nan)
    ).fillna(0.0).clip(0.0, 1.0)
    per_player = per_player.merge(pos_totals[['position_rate']], on='position', how='left')
    per_player['position_rate'] = per_player['position_rate'].fillna(per_player['own_rate'])

    per_player['evidence'] = (per_player['weighted_touches'] / FULL_TRUST_TOUCHES).clip(upper=1.0)
    weight = per_player['evidence'] * BIG_PLAY_SELF_WEIGHT
    per_player['rate'] = (weight * per_player['own_rate'] +
                          (1 - weight) * per_player['position_rate']).clip(0.0, 1.0)
    return per_player[['Player', 'position', 'touches', 'own_rate', 'position_rate',
                       'evidence', 'rate']]


def _normalized_rules(scoring):
    """scoring['big_play_bonuses'] -> a clean list of (category, threshold, points)."""
    rules = scoring.get('big_play_bonuses') if isinstance(scoring, dict) else None
    out = []
    for rule in (rules or []):
        category = str((rule or {}).get('category', '')).strip().lower()
        threshold = pd.to_numeric((rule or {}).get('threshold'), errors='coerce')
        points = pd.to_numeric((rule or {}).get('points'), errors='coerce')
        if category not in BIG_PLAY_TOUCH_STAT or pd.isna(threshold) or pd.isna(points):
            continue
        if not points or threshold <= 0:
            continue
        out.append((category, float(threshold), float(points)))
    return out


def has_big_play_bonuses(scoring):
    return bool(_normalized_rules(scoring))


def project_big_play_points(board, scoring, latest_season, n_seasons=5):
    """
    Every active big-play rule, priced against a board that already carries
    projected touches (receptions/carries/attempts, from
    draft_projections.project_stat_lines) and summed into one points Series
    aligned to `board`'s index.

    A NAME NOT FOUND IN THE RATE TABLE (a rookie with no NFL history, or a
    name-format miss) FALLS BACK TO THE POSITION'S BASELINE RATE, not to
    zero - the same "no evidence -> lean on the position" logic the blend
    itself already applies to a proven veteran's thin sample, just taken to
    its limit for a player with none at all. Silently scoring zero would be
    wrong in a specific, misleading direction: it would look like "this
    player never has big plays" rather than "this app has no read on him
    yet."

    Returns (points, detail) - detail is a list of per-rule dicts (category,
    threshold, points, total) for a UI breakdown, since one summed number
    doesn't say which rule produced it.
    """
    total = pd.Series(0.0, index=board.index)
    detail = []
    rules = _normalized_rules(scoring)
    if not rules or board is None or board.empty or 'Player' not in board.columns:
        return total, detail

    board_keys = clean_name_exact(board['Player'])
    board_pos = board['Pos'].astype(str).str.upper() if 'Pos' in board.columns else None

    for category, threshold, points in rules:
        touch_stat = BIG_PLAY_TOUCH_STAT[category]
        if touch_stat not in board.columns:
            continue
        rates = big_play_rate_table(category, threshold, latest_season, n_seasons)
        if rates.empty:
            continue

        rate_by_key = dict(zip(clean_name_exact(rates['Player']), rates['rate']))
        rate = board_keys.map(rate_by_key)

        if board_pos is not None:
            pos_rate = (rates.drop_duplicates('position')
                        .set_index('position')['position_rate'].to_dict())
            rate = rate.where(rate.notna(), board_pos.map(pos_rate))
        rate = rate.fillna(0.0)

        touches = pd.to_numeric(board[touch_stat], errors='coerce').fillna(0.0)
        rule_points = touches * rate * points
        total = total + rule_points
        detail.append({
            'category': category, 'label': CATEGORY_LABELS.get(category, category),
            'threshold': threshold, 'points': points,
            'total': round(float(rule_points.sum()), 1),
        })
    return total.round(2), detail
