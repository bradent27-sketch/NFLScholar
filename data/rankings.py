"""
Fantasy Rankings ingestion: FantasyPros draft-rankings CSVs (PPR/Half/Standard,
refreshed periodically by the user in the same schema) plus arbitrary custom
ranking uploads, both aligned against this app's own VORP sheet via the
existing two-tier name matcher so minor naming differences across three
independent sources don't produce spurious "no match" rows.
"""
import os
import pandas as pd
import streamlit as st

from data.utils import clean_name_for_merge, clean_name_exact

# Keyed on scoring format, not a hardcoded filename - the user said they'll
# periodically refresh these with new FantasyPros exports in the same
# schema, so a drop-in replacement (same 3 filenames) works with no code
# changes.
FANTASYPROS_FILES = {
    'Full PPR': 'rankings/fantasypros_2026_draft_rankings_ppr.csv',
    'Half-PPR': 'rankings/fantasypros_2026_draft_rankings_half_ppr.csv',
    'Standard': 'rankings/fantasypros_2026_draft_rankings_standard.csv',
}


def _parse_fantasypros_dataframe(raw_df):
    """
    Shared column-mapping for FantasyPros' export schema: RK, TIERS,
    "PLAYER NAME", TEAM, POS, "BYE WEEK", UPSIDE, BUST, "SOS SEASON",
    "ECR VS. ADP". POS carries a positional-rank suffix (e.g. "WR1") -
    stripped here since the app's own Pos columns elsewhere are bare
    position codes. FantasyPros publishes both season-long DRAFT rankings
    and WEEKLY rankings in this identical shape (a different export, not a
    different schema) - shared so both load_fantasypros_rankings (fixed
    local draft-rankings path) and parse_fantasypros_upload (weekly, via
    file_uploader since that file changes every week) stay in sync.
    """
    raw_df = raw_df.copy()
    raw_df.columns = [c.strip() for c in raw_df.columns]
    out = pd.DataFrame({
        'Rank': pd.to_numeric(raw_df.get('RK'), errors='coerce'),
        'Tier': pd.to_numeric(raw_df.get('TIERS'), errors='coerce'),
        'Player': raw_df.get('PLAYER NAME', pd.Series(dtype=str)).astype(str).str.strip(),
        'Team': raw_df.get('TEAM', pd.Series(dtype=str)).astype(str).str.strip(),
        'Pos': raw_df.get('POS', pd.Series(dtype=str)).astype(str).str.replace(r'\d+$', '', regex=True),
        'Bye': raw_df.get('BYE WEEK'),
        'ECR vs ADP': raw_df.get('ECR VS. ADP'),
    })
    out = out.dropna(subset=['Rank', 'Player'])
    out = out[out['Player'].ne('') & out['Player'].str.lower().ne('nan')]
    return out.sort_values('Rank').reset_index(drop=True)


@st.cache_data
def load_fantasypros_rankings(scoring_format):
    path = FANTASYPROS_FILES.get(scoring_format)
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return _parse_fantasypros_dataframe(df)


def parse_fantasypros_upload(uploaded_file):
    """
    Same FantasyPros column schema as load_fantasypros_rankings, but read
    from an uploaded file instead of a fixed local path - for weekly
    rankings, which change every week, a file_uploader is a better fit than
    a filename the user has to rename each time (unlike the season-long
    draft rankings, which only need refreshing a few times a year).
    """
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        # seek(0) first - Streamlit's UploadedFile is a real seekable
        # BytesIO, but a caller that already tried parse_custom_rankings
        # (or vice versa) on this same object would otherwise hand this a
        # pointer sitting at EOF from that earlier read, which pd.read_csv
        # would silently turn into an empty/garbage frame instead of a
        # clear error - the weekly-rankings tab tries both parsers in
        # sequence on one upload, so this matters in practice, not just
        # in theory.
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
    except Exception:
        return pd.DataFrame()
    return _parse_fantasypros_dataframe(df)


def parse_custom_rankings(uploaded_file):
    """
    Generic custom-rankings CSV parser - any export with a player-name
    column and (optionally) a rank column. Column names are matched
    case-insensitively against a short list of common headers rather than
    requiring an exact FantasyPros-shaped schema, since a "custom" ranking
    could come from anywhere (a spreadsheet, a different site's export,
    hand-typed).
    """
    try:
        uploaded_file.seek(0)  # see parse_fantasypros_upload's docstring for why
        df = pd.read_csv(uploaded_file)
    except Exception:
        return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    lower_cols = {c.lower(): c for c in df.columns}

    name_col = next((lower_cols[c] for c in lower_cols if c in ('player', 'player name', 'name')), None)
    if not name_col:
        return pd.DataFrame()
    rank_col = next((lower_cols[c] for c in lower_cols if c in ('rk', 'rank', 'overall rank', 'ovr', 'ovr rank')), None)

    out = pd.DataFrame({'Player': df[name_col].astype(str).str.strip()})
    out['Rank'] = pd.to_numeric(df[rank_col], errors='coerce') if rank_col else range(1, len(df) + 1)
    out = out.dropna(subset=['Player'])
    out = out[out['Player'].ne('') & out['Player'].str.lower().ne('nan')]
    return out.sort_values('Rank').reset_index(drop=True)


def _match_one(name, board_exact, board_loose):
    exact_key = clean_name_exact(pd.Series([name])).iloc[0]
    if exact_key in board_exact:
        return board_exact[exact_key]
    loose_key = clean_name_for_merge(pd.Series([name])).iloc[0]
    if loose_key in board_loose:
        return board_loose[loose_key]
    return None


def build_rankings_comparison(value_df, value_col='VORP', rank_label='VORP', fp_df=None, custom_df=None):
    """
    One row per player (keyed to value_df's own player names - this app's
    canonical nflverse-derived names) with a {rank_label} Rank column
    alongside FantasyPros Rank and/or Custom Rank wherever a match is
    found, plus a delta column highlighting where this app's own ranking
    and the external one disagree most.

    Generalized over WHAT internal value is being ranked so the same
    comparison logic serves two different tabs: the VORP Draft Sheet
    (value_col='VORP', a season-long draft-value projection, compared
    against FantasyPros' DRAFT rankings) and the separate Weekly Rankings
    tab (value_col='Recent Avg FPTS', in-season recent form, compared
    against an uploaded WEEKLY rankings file) - a draft-value ranking and a
    recent-form ranking answer different questions and shouldn't be
    conflated into one comparison.

    The delta is computed against {rank_label} Rank RE-RANKED within just
    the matched subset for that source, not the full board's overall rank
    - value_df spans every player who cleared that board's own minimum-
    sample filter, which is usually far more players than a typical
    external ranking bothers to include at all. Diffing against the
    full-board rank means a fringe/deep-bench player the external source
    ranks 188th could show a "disagreement" of 1000+ purely because this
    app's board includes hundreds of irrelevant scrubs ranked worse than
    him - a pool-size artifact, not a real signal about model disagreement.
    Re-ranking within the matched subset puts both sides on the same
    numbering scale (1..N matched players), so the delta reflects actual
    relative disagreement. The full-board rank column itself is left as-is
    since it's still useful standalone context.
    """
    if value_df.empty:
        return pd.DataFrame()

    board = value_df['Player'].tolist()
    board_exact = {clean_name_exact(pd.Series([p])).iloc[0]: p for p in board}
    board_loose = {clean_name_for_merge(pd.Series([p])).iloc[0]: p for p in board}

    rank_col_name = f'{rank_label} Rank'
    out = value_df[['Player', 'Pos', 'Team']].copy()
    out[rank_col_name] = value_df[value_col].rank(ascending=False, method='first').astype(int)

    def attach(source_df, rank_col, delta_col):
        if source_df is None or source_df.empty:
            out[rank_col] = pd.NA
            return
        name_map = {}
        for _, r in source_df.iterrows():
            b = _match_one(r['Player'], board_exact, board_loose)
            if b and b not in name_map:
                name_map[b] = r['Rank']
        out[rank_col] = out['Player'].map(name_map)
        matched = out[rank_col].notna()
        if matched.sum() > 1:
            pool_rank = out.loc[matched, rank_col_name].rank(method='first')
            out.loc[matched, delta_col] = out.loc[matched, rank_col] - pool_rank
        # Unranked-by-this-source players get a sentinel worse than every
        # real rank instead of NaN - glide-data-grid's column sort doesn't
        # reliably push NaN to the bottom on an ascending click (confirmed:
        # it was landing ahead of rank 1), which made "sort to #1" and
        # "unranked players sort last" both unreliable at once. A real
        # number one worse than the max actual rank fixes both: ascending
        # still puts rank 1 first, and every unranked player now sorts
        # below the worst real rank rather than wherever NaN happened to land.
        if matched.any():
            out.loc[~matched, rank_col] = out.loc[matched, rank_col].max() + 1

    attach(fp_df, 'FantasyPros Rank', f'{rank_label} vs FantasyPros')
    attach(custom_df, 'Custom Rank', f'{rank_label} vs Custom')

    return out.sort_values(rank_col_name).reset_index(drop=True)
