"""Small, reviewable player-name aliases for sources that lack a stable ID.

Stable GSIS/PFF/player identifiers always take precedence in projection joins.
This file is the deliberately narrow fallback for a source display-name
difference such as ``Kenny Gainwell`` versus ``Kenneth Gainwell``.  Keeping
the mapping in CSV makes each exception inspectable and prevents a fuzzy-name
rule from silently joining two different players.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from data.utils import clean_name_exact


ALIAS_PATH = Path(__file__).with_name("player_aliases.csv")


# Keep the vocabulary small and explicit.  These are identifiers the app's
# roster / nflverse / PFF inputs actually expose; a source-specific field such
# as Ourlads' ``source_player_id`` is intentionally *not* included because it
# is an Ourlads web id, not a GSIS or PFF id.
STABLE_PLAYER_ID_COLUMNS = (
    "gsis_id",
    "player_id",
    "pff_id",
    "pff_player_id",
    "nflverse_id",
    "espn_id",
    "pfr_id",
    "sportradar_id",
)

# nflverse's historical box-score feed calls the GSIS value ``player_id``
# while roster exports call it ``gsis_id``. They are the same identity, so
# their keys must share a namespace; otherwise an offseason roster merge can
# quietly turn one person into two just because a source renamed the column.
# PFF aliases likewise share a namespace. Other provider IDs intentionally
# remain distinct because equal-looking values from different providers are
# not evidence of identity.
_IDENTIFIER_NAMESPACES = {
    "gsis_id": "gsis_id",
    "player_id": "gsis_id",
    "nflverse_id": "gsis_id",
    "pff_id": "pff_id",
    "pff_player_id": "pff_id",
    "espn_id": "espn_id",
    "pfr_id": "pfr_id",
    "sportradar_id": "sportradar_id",
}


def _name_keys(values) -> pd.Series:
    series = pd.Series(values)
    # Convert categoricals/nullables before filling; pandas otherwise refuses
    # to add the empty-string category during a normal roster lookup.
    series = series.astype(object).where(series.notna(), "")
    return clean_name_exact(series.astype(str))


def normalize_player_identifier(values) -> pd.Series:
    """Normalize stable external IDs without turning missing values into IDs.

    CSV readers often parse a numeric PFF/ESPN id as a float.  Removing a
    terminal ``.0`` makes that representation agree with the text form while
    retaining the id namespace for callers to record separately.  This is not
    a name-normalization fallback: an empty or malformed value stays empty.
    """
    series = pd.Series(values)
    series = series.astype(object).where(series.notna(), "").astype(str).str.strip()
    series = series.str.replace(r"\.0$", "", regex=True)
    invalid = series.str.lower().isin({"", "nan", "none", "<na>", "null", "0"})
    return series.mask(invalid, "")


def stable_roster_identity_keys(frame: pd.DataFrame, name_col: str,
                                id_columns: tuple[str, ...] = STABLE_PLAYER_ID_COLUMNS) -> pd.Series:
    """Return a deterministic per-row identity key for roster-like frames.

    A usable stable id wins over a reviewed name key.  Prefixing the id with
    its namespace prevents a numeric PFF id from ever colliding with a
    similarly shaped id from a different provider.  This helper is safe for
    cold-start pool deduplication, while source-to-roster joins should still
    use their source-aware resolver so that GSIS/PFF namespaces can be
    compared deliberately.
    """
    if frame is None or frame.empty:
        return pd.Series(dtype=str)
    if name_col not in frame.columns:
        fallback = pd.Series("", index=frame.index, dtype=object)
    else:
        fallback = "name:" + canonical_player_key(frame[name_col]).astype(str)
    result = fallback.astype(object).copy()
    # Earlier fields are stronger / more portable, so only fill rows that do
    # not yet have a stable identifier.
    has_stable = pd.Series(False, index=frame.index)
    for column in id_columns:
        if column not in frame.columns:
            continue
        raw = frame.loc[:, frame.columns == column]
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]
        values = normalize_player_identifier(raw)
        valid = values.ne("") & ~has_stable
        if valid.any():
            namespace = _IDENTIFIER_NAMESPACES.get(column, column)
            result.loc[valid] = f"{namespace}:" + values.loc[valid]
            has_stable.loc[valid] = True
    return result.astype(str)


@lru_cache(maxsize=4)
def load_player_aliases(path: str | Path = ALIAS_PATH) -> dict[str, str]:
    """Return a one-way source-key -> canonical-key map from the CSV.

    Malformed or ambiguous rows are ignored rather than guessed.  The app can
    continue without aliases; an alias only adds a conservative exact bridge.
    """
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        frame = pd.read_csv(target, dtype=str, keep_default_na=False)
    except Exception:
        return {}
    required = {"source_name", "canonical_name"}
    if not required.issubset(frame.columns):
        return {}
    source = _name_keys(frame["source_name"])
    canonical = _name_keys(frame["canonical_name"])
    pairs = pd.DataFrame({"source": source, "canonical": canonical})
    pairs = pairs[(pairs["source"] != "") & (pairs["canonical"] != "")]
    # A duplicate source alias with competing targets is not safe to use.
    ambiguous = pairs.groupby("source", observed=True)["canonical"].nunique()
    pairs = pairs[~pairs["source"].isin(ambiguous[ambiguous > 1].index)]
    return dict(zip(pairs["source"], pairs["canonical"]))


def canonical_player_key(values, path: str | Path = ALIAS_PATH) -> pd.Series:
    """Normalize source names through the reviewed alias map, index intact."""
    original = pd.Series(values)
    keys = _name_keys(original)
    aliases = load_player_aliases(path)
    if aliases:
        keys = keys.map(aliases).fillna(keys)
    keys.index = original.index
    return keys.astype(str)
