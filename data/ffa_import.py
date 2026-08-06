"""
Import a Fantasy Football Advice player export into this app's own board.

WHAT THIS IS FOR: you have an FFA subscription, and their player payload
carries things this app can't derive on its own - a hand-written scouting
note per player, their proprietary FFAValue score, Elo ratings, and their
analysts' projected stat lines. This reads a file YOU supply and folds it
into the board alongside the app's own projections.

WHAT THIS DELIBERATELY IS NOT: a scraper. There is no code here that talks
to their servers, polls their API, or automates a login. It reads a local
file you already have access to, once, when you hand it over. That keeps
this squarely a personal-use import of data you pay for, and it means a
draft-day outage on their end can't take your board down.

THE STAT LINE IS THE POINT. Their `projections` object is a full statistical
forecast - carries, rushing yards, receptions, passing attempts and so on -
not a finished point total. That matters because a point total is only
correct for the scoring settings it was computed under: their export is
half-PPR, and reading `proj` straight off it would be silently wrong in a
full-PPR or TE-premium league, and wrong by a lot once yardage bonuses are
switched on. Importing the STAT LINE and re-scoring it through this app's
own scoring engine keeps every league setting honest, and is the same path
the app's native projections already take.
"""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

# Where an import is persisted so it survives restarts. Dropping a file here
# by hand works exactly as well as uploading it through the UI.
FFA_IMPORT_PATH = os.path.join('external_data', 'ffa_players.json')

# Their stat keys -> this app's canonical column names. Their export is a
# flat object of season totals per player.
FFA_STAT_MAP = {
    'carries': 'carries',
    'rush_yds': 'rushing_yards',
    'rush_tds': 'rushing_tds',
    'receptions': 'receptions',
    'rec_yds': 'receiving_yards',
    'rec_tds': 'receiving_tds',
    'pass_attempts': 'attempts',
    'passing_yds': 'passing_yards',
    'passing_tds': 'passing_tds',
    'interceptions': 'passing_interceptions',
    # They publish one combined fumbles-lost figure; this app splits it by
    # rushing vs receiving. Assigned wholly to rushing rather than split
    # arbitrarily, since both carry the same scoring value in every league
    # this supports - the split exists in the schema, not in the scoring.
    'fumbles_lost': 'rushing_fumbles_lost',
}

# Columns carried across as-is, for display and for the value blend.
FFA_PASSTHROUGH = {
    'ffaValue': 'FFA Value',
    'proj': 'FFA Proj (their scoring)',
    'adp': 'FFA ADP',
    'adpComposite': 'FFA ADP Composite',
    'posRank': 'FFA Pos Rank',
    'elo': 'Elo',
    'pprElo': 'PPR Elo',
    'dynElo': 'Dynasty Elo',
    'age': 'Age',
    'rookie': 'Rookie',
    'byeWeek': 'FFA Bye',
    'notes': 'Notes',
}


def parse_ffa_export(raw):
    """
    Normalize an FFA player export into this app's columns.

    Accepts the shapes their endpoints actually return: a bare list of
    players, or an object wrapping one under `players` / `data` / `results`.
    Anything else returns empty rather than raising - an import is
    enrichment, and a malformed file should cost you the extra columns, not
    the draft board.
    """
    if isinstance(raw, dict):
        for key in ('players', 'data', 'results', 'items'):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list) or not raw:
        return pd.DataFrame()

    rows = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get('name') or entry.get('playerName') or entry.get('player_name')
        if not name:
            continue
        record = {
            'Player': str(name).strip(),
            'Pos': str(entry.get('pos') or entry.get('position') or '').upper().strip(),
            'Team': str(entry.get('team') or '').upper().strip(),
            'sleeper_id': entry.get('sleeperId') or entry.get('sleeper_id'),
        }
        for source_key, column in FFA_PASSTHROUGH.items():
            if source_key in entry:
                record[column] = entry[source_key]

        projections = entry.get('projections')
        if isinstance(projections, dict):
            for source_key, column in FFA_STAT_MAP.items():
                value = projections.get(source_key)
                if value is not None:
                    record[column] = pd.to_numeric(value, errors='coerce')
            record['has_ffa_stat_line'] = True
        else:
            record['has_ffa_stat_line'] = False
        rows.append(record)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df['Player'].ne('') & df['Player'].str.lower().ne('nan')]
    return df.drop_duplicates(subset=['Player'], keep='first').reset_index(drop=True)


def load_ffa_import(path=FFA_IMPORT_PATH):
    """Read a persisted import from disk. Returns (df, error)."""
    if not os.path.exists(path):
        return pd.DataFrame(), None
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            raw = json.load(fh)
    except Exception as exc:
        return pd.DataFrame(), f"couldn't read {path}: {exc}"
    df = parse_ffa_export(raw)
    if df.empty:
        return df, f"{path} didn't contain a recognisable player list"
    return df, None


def save_ffa_import(uploaded_file, path=FFA_IMPORT_PATH):
    """
    Persist an uploaded export so it survives a restart.

    Written to disk rather than held in session state because a draft board
    should still be there tomorrow: re-uploading the same file before every
    session is exactly the kind of friction that gets a feature abandoned
    two days before a draft.
    """
    try:
        uploaded_file.seek(0)
        raw = json.load(uploaded_file)
    except Exception as exc:
        return pd.DataFrame(), f"not valid JSON: {exc}"
    df = parse_ffa_export(raw)
    if df.empty:
        return df, "no recognisable player list in that file"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(raw, fh)
    except Exception as exc:
        return df, f"parsed OK but couldn't save to {path}: {exc}"
    return df, None


def merge_ffa_into_board(board, ffa_df, stat_line_weight=1.0, scoring=None):
    """
    Fold an FFA import into the board.

    `stat_line_weight` blends their projected stat line against this app's
    own, per stat, before anything is scored: 1.0 trusts their analysts
    completely, 0.0 ignores the stat line and keeps only their notes, value
    and Elo as extra columns. Blending on the STAT LINE rather than on final
    points is what keeps the result internally consistent - a blended
    projection still has a carry count that matches its rushing yards, so
    the yardage-bonus model and every scoring rule still apply to a coherent
    player rather than to an average of two different players' point totals.

    Rows they don't cover keep this app's own projection untouched, so a
    262-player export doesn't blank out the back half of a 700-player board.
    """
    if board.empty or ffa_df is None or ffa_df.empty:
        return board, {'matched': 0}

    from data.utils import clean_name_exact, clean_name_for_merge
    from data.draft_projections import PROJECTED_STATS, score_projected_lines

    out = board.copy()
    ffa = ffa_df.copy()
    ffa['_exact'] = clean_name_exact(ffa['Player'])
    ffa['_loose'] = clean_name_for_merge(ffa['Player'])
    exact_map = {k: i for i, k in enumerate(ffa['_exact'])}
    loose_map = {k: i for i, k in enumerate(ffa['_loose'])}

    board_exact = clean_name_exact(out['Player'])
    board_loose = clean_name_for_merge(out['Player'])
    positions = []
    for i in range(len(out)):
        idx = exact_map.get(board_exact.iloc[i])
        if idx is None:
            idx = loose_map.get(board_loose.iloc[i])
        positions.append(idx)
    out['_ffa_row'] = positions

    matched = int(sum(1 for p in positions if p is not None))
    for column in FFA_PASSTHROUGH.values():
        if column in ffa.columns:
            out[column] = [ffa[column].iloc[p] if p is not None else None for p in positions]

    weight = float(np.clip(stat_line_weight, 0.0, 1.0))
    blended = 0
    if weight > 0:
        for stat in PROJECTED_STATS:
            if stat not in ffa.columns:
                continue
            theirs = pd.Series(
                [ffa[stat].iloc[p] if p is not None else np.nan for p in positions],
                index=out.index, dtype=float)
            if stat not in out.columns:
                out[stat] = 0.0
            mine = pd.to_numeric(out[stat], errors='coerce').fillna(0.0)
            usable = theirs.notna()
            out.loc[usable, stat] = (weight * theirs[usable] + (1 - weight) * mine[usable]).round(2)
        blended = int(sum(1 for p in positions if p is not None))
        if scoring is not None:
            out = score_projected_lines(out, scoring)
            out['proj_basis'] = np.where(
                out['_ffa_row'].notna(),
                'FFA' if weight >= 0.999 else f'FFA blend ({int(weight * 100)}%)',
                out.get('proj_basis', 'rank curve'))

    out = out.drop(columns=['_ffa_row'])
    return out, {'matched': matched, 'blended': blended, 'imported': int(len(ffa))}


@st.cache_data(show_spinner=False)
def ffa_value_rank(ffa_df):
    """Their FFAValue turned into a rank, for comparison against this board."""
    if ffa_df is None or ffa_df.empty or 'FFA Value' not in ffa_df.columns:
        return pd.DataFrame()
    out = ffa_df[['Player', 'Pos', 'FFA Value']].copy()
    out['FFA Value Rank'] = pd.to_numeric(out['FFA Value'], errors='coerce').rank(
        ascending=False, method='first')
    return out.dropna(subset=['FFA Value Rank'])
