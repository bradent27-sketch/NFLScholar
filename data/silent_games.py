"""
Silent games (2026-10-25, user: "Count silent games ... essential for making sure the model is accurate and that our backtests are
accurate ... especially for players who get less volume"; CFB Scholar's snap_appearances / silent_games_rate).

nflverse's weekly player stats are stat-triggered: a receiver who runs routes all game and is never targeted, a back who plays
special teams and a few offensive snaps without a touch, a blocking tight end - none of them gets a row that week. Every per-game
rate the model forms is then an average over the games a player DID record a stat, and every backtest that reads "played" off the
box score skips the games he played and scored nothing. Snap counts (nflverse, from PFR) list every snap-taker. Measured
2021-2025 (scripts/diag_silent_games.py): of offensive-snap games with no box-score row, TE 24-28%, WR 10-13%, RB 11-15%, QB 2%;
for low-volume players that inflates a per-game rate by about 33-44% (TE targets), 20% (WR targets), 6-15% (RB carries).

A silent game here is a regular-season game where a RB/WR/TE took an offensive snap (offense_snaps > 0) and has no box-score row
that week. QBs are left out: their silent games are one-snap kneel-downs and relief cameos (2%), and the QB1 rules read QB rows.

    silent_game_rows(year, stats_df, name_col, team_col)  -> box-score-shaped rows for those games, every stat 0
    with_silent_games(stats_df, year, name_col, team_col) -> stats_df with them appended (`_silent_game` marks them)
    played_players(year, week, stats_df, name_col)        -> names that played a regular-season week: a box-score row or a snap

Identity: a snap row carries a PFR id (pfr_player_id), as does a box-score row (pfr_id), so a silent game takes the player's own
identity columns from his box-score rows that season; a player with no box-score row at all that season keeps the snap table's name
(cleaned the same way every join in this app is).
"""
import numpy as np
import pandas as pd

from data.utils import clean_name_exact, clean_name_for_merge

SILENT_POSITIONS = {'RB': 'RB', 'FB': 'RB', 'WR': 'WR', 'TE': 'TE'}
SILENT_FLAG = '_silent_game'
# identity columns copied from the player's own box-score rows (the rest of a silent row is per game, or 0)
_IDENTITY_COLUMNS = ('player_id', 'player_name', 'position', 'position_group', 'headshot_url', 'gsis_id', 'pfr_id', 'full_name',
                     'roster_position_group', 'depth_chart_position', 'status', 'birth_date', 'draft_club', 'exact_name',
                     'clean_merge_name', 'years_exp', 'draft_number', 'snap_pct_avg', 'snap_games_played', 'has_snap_match',
                     'snap_data_year', 'team')


def _snap_games(year):
    """Regular-season offensive-snap games of RB/WR/TE: pfr id, name, position, week, team, opponent, game id, snap share 0-1."""
    from data.loaders import _load_snap_counts_raw
    raw = _load_snap_counts_raw(int(year))
    if raw is None or raw.empty or not {'pfr_player_id', 'player', 'position', 'week', 'offense_snaps'}.issubset(raw.columns):
        return pd.DataFrame()
    snaps = raw.copy()
    if 'game_type' in snaps.columns:
        snaps = snaps[snaps['game_type'].astype(str).str.upper().eq('REG')]
    snaps = snaps[pd.to_numeric(snaps['offense_snaps'], errors='coerce') > 0]
    snaps = snaps.assign(pos=snaps['position'].astype(str).str.upper().map(SILENT_POSITIONS))
    snaps = snaps[snaps['pos'].notna()]
    return pd.DataFrame({
        'pfr': snaps['pfr_player_id'].astype(str).to_numpy(), 'name': snaps['player'].astype(str).to_numpy(),
        'pos': snaps['pos'].to_numpy(), 'week': pd.to_numeric(snaps['week'], errors='coerce').to_numpy(),
        'team': snaps['team'].astype(str).str.upper().to_numpy(), 'opponent': snaps['opponent'].astype(str).str.upper().to_numpy(),
        'game_id': snaps['game_id'].astype(str).to_numpy() if 'game_id' in snaps.columns else '',
        'share': pd.to_numeric(snaps['offense_pct'], errors='coerce').fillna(0.0).to_numpy(),
    }).dropna(subset=['week'])


def _boxed(snaps, stats_df):
    """Whether each snap game has a box-score row that week (same PFR id, or the same cleaned name)."""
    weeks = pd.to_numeric(stats_df['week'], errors='coerce')
    by_id = set(zip(stats_df['pfr_id'].astype(str), weeks)) if 'pfr_id' in stats_df.columns else set()
    names = stats_df['exact_name'].astype(str) if 'exact_name' in stats_df.columns else clean_name_exact(stats_df['player_display_name'])
    by_name = set(zip(names, weeks))
    keys = clean_name_exact(pd.Series(snaps['name'].to_numpy()))
    return np.array([(p, w) in by_id or (k, w) in by_name for p, k, w in zip(snaps['pfr'], keys, snaps['week'])], dtype=bool)


def silent_game_rows(year, stats_df, name_col, team_col):
    """Box-score-shaped rows (every stat 0, `_silent_game` True) for the season's silent games (module docstring), in stats_df's
    columns and dtypes as far as they go. Empty when there are none or no snap counts."""
    if stats_df is None or stats_df.empty or 'week' not in stats_df.columns:
        return pd.DataFrame()
    snaps = _snap_games(year)
    if snaps.empty:
        return pd.DataFrame()
    silent = snaps[~_boxed(snaps, stats_df)].copy()
    if silent.empty:
        return pd.DataFrame()
    # the player's identity from his own box-score rows this season (latest row), by PFR id, else by cleaned name
    box = stats_df[pd.to_numeric(stats_df['week'], errors='coerce').notna()]
    box = box.sort_values('week')
    ident_cols = [c for c in (name_col, *_IDENTITY_COLUMNS) if c in box.columns]
    by_pfr = (box[box['pfr_id'].notna()].drop_duplicates('pfr_id', keep='last').assign(_k=lambda d: d['pfr_id'].astype(str))
              .set_index('_k')[ident_cols] if 'pfr_id' in box.columns else pd.DataFrame(columns=ident_cols))
    names = box['exact_name'].astype(str) if 'exact_name' in box.columns else clean_name_exact(box[name_col])
    by_name = box.assign(_k=names.to_numpy()).drop_duplicates('_k', keep='last').set_index('_k')[ident_cols]
    keys = clean_name_exact(pd.Series(silent['name'].to_numpy())).to_numpy()
    records = []
    for i, (pfr, key) in enumerate(zip(silent['pfr'], keys)):
        if pfr in by_pfr.index:
            ident = by_pfr.loc[pfr].to_dict()
        elif key in by_name.index:
            ident = by_name.loc[key].to_dict()
        else:
            name = silent['name'].iloc[i]
            ident = {name_col: name, 'player_name': name, 'full_name': name, 'exact_name': key,
                     'clean_merge_name': clean_name_for_merge(pd.Series([name])).iloc[0], 'pfr_id': pfr,
                     'position': silent['pos'].iloc[i], 'position_group': silent['pos'].iloc[i], 'has_snap_match': True}
        records.append(ident)
    rows = pd.DataFrame(records, index=silent.index)
    # the game itself, from the snap row (the box score has none to give)
    numeric = [c for c in stats_df.columns if pd.api.types.is_numeric_dtype(stats_df[c]) and not pd.api.types.is_bool_dtype(stats_df[c])
               and c not in rows.columns]
    rows = pd.concat([rows, pd.DataFrame(0.0, index=rows.index, columns=numeric)], axis=1)
    rows['season'] = float(year)
    rows['week'] = silent['week'].astype(float)
    rows['season_type'] = 'REG'
    rows['game_id'] = silent['game_id']
    rows['opponent_team'] = silent['opponent']
    rows['game_team'] = silent['team']
    rows['game_opponent'] = silent['opponent']
    rows[team_col] = silent['team']
    rows['weekly_snap_pct'] = (silent['share'] * 100.0).clip(0.0, 100.0)
    rows['has_snap_match'] = True
    rows[SILENT_FLAG] = True
    return rows.reset_index(drop=True)


def _align_categories(left, right):
    """Give `right`'s columns `left`'s categorical dtypes (categories unioned), so a concat keeps them categorical."""
    left, right = left.copy(), right.copy()
    for column in left.columns:
        if column in right.columns and isinstance(left[column].dtype, pd.CategoricalDtype):
            values = pd.Index(left[column].cat.categories).union(pd.Index(right[column].dropna().astype(str).unique()))
            dtype = pd.CategoricalDtype(values)
            left[column] = left[column].astype(str).where(left[column].notna()).astype(dtype)
            right[column] = right[column].astype(str).where(right[column].notna()).astype(dtype)
    return left, right


_MEMO = {}
_MEMO_MAX = 8


def _memo_key(stats_df, year, name_col, team_col):
    weeks = pd.to_numeric(stats_df.get('week'), errors='coerce')
    return (int(year), len(stats_df), float(weeks.max()) if weeks.notna().any() else -1.0, name_col, team_col)


def _silent_rows_cached(stats_df, year, name_col, team_col):
    key = _memo_key(stats_df, year, name_col, team_col)
    if key not in _MEMO:
        if len(_MEMO) >= _MEMO_MAX:
            _MEMO.pop(next(iter(_MEMO)))
        _MEMO[key] = silent_game_rows(year, stats_df, name_col, team_col)
    return _MEMO[key]


def with_silent_games(stats_df, year, name_col, team_col):
    """`stats_df` with the season's silent games appended (`_silent_game` True on them, False on box-score rows). A new frame; the
    loader's cached one is never changed. Unchanged but for the flag column when there are none."""
    if stats_df is None or stats_df.empty or 'week' not in stats_df.columns:
        return stats_df
    silent = _silent_rows_cached(stats_df, year, name_col, team_col)
    base = stats_df.assign(**{SILENT_FLAG: False})
    if silent.empty:
        return base
    left, right = _align_categories(base, silent[[c for c in silent.columns if c in base.columns]])
    return pd.concat([left, right], ignore_index=True, sort=False)


def played_players(year, week, stats_df, name_col, team_col='team'):
    """The names (stats_df's `name_col` spelling) of everyone who played regular-season `week` of `year`: a box-score row, or a
    silent game (a RB/WR/TE snap with no row). What a backtest should count as "played" (a silent game scores 0)."""
    weeks = pd.to_numeric(stats_df['week'], errors='coerce')
    played = set(stats_df.loc[weeks == week, name_col].astype(str))
    silent = _silent_rows_cached(stats_df, year, name_col, team_col)
    if not silent.empty:
        played |= set(silent.loc[pd.to_numeric(silent['week'], errors='coerce') == week, name_col].astype(str))
    return played
