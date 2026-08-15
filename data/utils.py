"""
Pure, dependency-free helpers shared by data/loaders.py and data/transforms.py
(percentile ranking, name-matching, bio-field formatting). Kept in their own
module so loaders.py and transforms.py can each depend on this without a
circular import between the two of them.
"""
import datetime
import html
import unicodedata

import numpy as np
import pandas as pd


def calculate_percentile(df, col_name, ascending=True):
    if not df.empty and col_name in df.columns:
        return df[col_name].rank(pct=True, ascending=ascending) * 100
    return pd.Series([0]*len(df), index=df.index)


# Minimum share of team snaps a player must have played (this season, or
# this week - whichever granularity the caller's snap_col tracks) for their
# stat line to count toward the REFERENCE POOL other players get
# percentile-ranked against - see calculate_percentile_qualified below.
# Skill positions where a real committee/backup role is common get a lower
# bar than a starting QB, who's on the field for nearly every snap by
# definition. Without this, a pile of 10%-snap bench/garbage-time rows
# drags a position's whole comparison pool down, inflating every real
# starter's percentile against them - confirmed real: a clear WR1 reading
# as a 94th-percentile fantasy-PPG player purely because dozens of fringe
# receivers who barely play were sitting in the same denominator.
MIN_SNAP_PCT_FOR_PERCENTILE = {
    'QB': 80,
    'RB': 40, 'FB': 40, 'WR': 40, 'TE': 40,
}


def min_snap_pct_for_position(pos):
    """
    Returns None for a position with no defined participation cutoff (K/P/
    LS/O-line - snap % isn't a meaningful signal for a kicker, and this
    app's data doesn't track O-line snaps as a team-snap share at all).
    Callers should treat None as "don't filter, use the full pool", not as
    a 0% bar.
    """
    from config import DEFENSIVE_POSITIONS
    pos = str(pos).upper()
    if pos in MIN_SNAP_PCT_FOR_PERCENTILE:
        return MIN_SNAP_PCT_FOR_PERCENTILE[pos]
    if pos in DEFENSIVE_POSITIONS:
        return 60
    return None


def _rank_against_pool(values, pool, ascending):
    """
    Shared ECDF-style percentile core for calculate_percentile_qualified's
    two modes below: rank(pct=True)'s tie handling reduced to "fraction of
    `pool` this value beats" via a sorted-array searchsorted, which is
    equivalent for percentile purposes and lets both modes share one
    vectorized implementation.
    """
    sorted_vals = np.sort(pool.to_numpy())
    n_pool = len(sorted_vals)
    gv = values.to_numpy()
    valid = ~np.isnan(gv)
    if ascending:
        ranks = np.searchsorted(sorted_vals, gv, side='right')
    else:
        ranks = n_pool - np.searchsorted(sorted_vals, gv, side='left')
    return ranks / n_pool * 100, valid


def calculate_percentile_qualified(df, col_name, position_col='position', snap_col='snap_pct_avg', ascending=True, group_by_position=True):
    """
    Same result shape as calculate_percentile (one 0-100 percentile per row,
    aligned to df's index) but the REFERENCE POOL used to rank col_name is
    restricted to rows meeting THEIR OWN position's minimum snap-
    participation bar (min_snap_pct_for_position) - every row still gets a
    percentile, even a below-threshold bench/limited-snap row, it's just
    scored against the cleaned qualifying-peer distribution instead of
    being diluted by (and diluting everyone else's percentile against)
    fringe/inactive-role rows.

    group_by_position=True (the default, and the right choice whenever the
    caller was ALREADY ranking within each position separately - e.g. a
    season-long or per-week stat percentile computed per position) ranks
    each position group only against that position's own qualified pool -
    exactly calculate_percentile's existing per-position-group behavior,
    just with a cleaner denominator.

    group_by_position=False is for a caller that ranks across a MIXED-
    position pool as one flat leaderboard (e.g. a cross-position risers/
    rookie board) - every row still only QUALIFIES for the shared pool via
    its own position's bar, but the ranking itself stays one combined pool
    across every position, matching the caller's original flat-ranking
    behavior rather than silently splitting it into separate per-position
    leaderboards.

    Falls back to the FULL pool - i.e. behaves exactly like
    calculate_percentile - for any position with no defined cutoff, or if
    the qualifying pool happens to be empty (too small/thin a sample to
    trust, e.g. an early-season slice where almost nobody has logged enough
    snaps yet). Also falls back to plain calculate_percentile wholesale if
    position_col/snap_col aren't even present, so a caller can reach for
    this function without checking the columns exist first.
    """
    if df.empty or col_name not in df.columns or position_col not in df.columns or snap_col not in df.columns:
        return calculate_percentile(df, col_name, ascending=ascending)

    snaps = pd.to_numeric(df[snap_col], errors='coerce')
    values = pd.to_numeric(df[col_name], errors='coerce')
    positions = df[position_col]

    if not group_by_position:
        thresholds = positions.map(min_snap_pct_for_position)
        qualifies = thresholds.isna() | (snaps >= thresholds)
        pool = values[qualifies].dropna()
        if pool.empty:
            pool = values.dropna()
        if pool.empty:
            return pd.Series(0.0, index=df.index)
        pct_vals, valid = _rank_against_pool(values, pool, ascending)
        result = pd.Series(0.0, index=df.index)
        result.loc[df.index[valid]] = pct_vals[valid]
        return result

    result = pd.Series(0.0, index=df.index)
    for pos, group_idx in df.groupby(position_col, observed=True).groups.items():
        threshold = min_snap_pct_for_position(pos)
        group_vals = values.loc[group_idx]
        pool = pd.Series(dtype=float)
        if threshold is not None:
            qualified_idx = group_idx[snaps.loc[group_idx] >= threshold]
            pool = group_vals.loc[qualified_idx].dropna()
        if pool.empty:
            pool = group_vals.dropna()
        if pool.empty:
            continue
        pct_vals, valid = _rank_against_pool(group_vals, pool, ascending)
        result.loc[group_idx[valid]] = pct_vals[valid]
    return result


def get_val(row, col, fmt="{}"):
    if pd.isna(row.get(col)) or str(row.get(col)).strip() == '': return "--"
    try: return fmt.format(float(row[col]))
    except: return fmt.format(row[col])


def _normalize_name_text(name_series):
    """
    Repairs two encoding defects that survive all the way into the match keys
    below, BEFORE any lowercasing/punctuation-stripping happens. Both were
    found by scanning the real 2025 files, not reasoned about:

    1. HTML entities. snap_counts_2025.csv.csv stores apostrophes escaped -
       "Ja&apos;Marr Chase", "D&apos;Andre Swift", "Wan&apos;Dale Robinson",
       "Za&apos;Darius Smith", "Adoree&apos; Jackson", "L&apos;Jarius Sneed",
       "Dre&apos;Mont Jones" and 10 more. Stripping punctuation does NOT
       rescue these, it makes them worse: "D&apos;Andre Swift" collapses to
       "damposandreswift", which matches nothing on any other source, so
       every one of those players silently carried NO snap data at all.
    2. Accents. "Jevón Holland", "Rakeem Nuñez-Roches", "Audric Estimé" -
       the [^a-z] strip DELETES the accented letter rather than folding it,
       turning "jevón" into "jevn" instead of "jevon", so the same player
       spelled with and without the accent across two sources never met.

    Applied inside both match-key builders below, so every existing call
    site in the app picks the fix up without changing.
    """
    s = name_series.astype(str)
    # Both passes are gated on the defect actually being present, and both
    # use vectorized string ops when it is. This function sits on the hot
    # path of the draft board - profiling a live-draft pick found the naive
    # per-row .map(_strip_accents) making 16,542 Python-level calls per
    # board build, for ~0.13s of pure overhead on files that are almost
    # entirely ASCII.
    if s.str.contains('&', regex=False, na=False).any():
        s = s.map(html.unescape)
    if not s.str.isascii().all():
        # NFKD splits an accented character into base letter + combining
        # mark; encoding to ASCII and dropping what doesn't fit removes the
        # marks and leaves the base letter. Anything else non-ASCII is
        # dropped here rather than by the [^a-z] strip downstream, which is
        # the same end result.
        s = (s.str.normalize('NFKD')
              .str.encode('ascii', errors='ignore')
              .str.decode('ascii'))
    return s


def _strip_accents(value):
    """Scalar form, for the two single-string callers below
    (build_last_name_index / match_abbreviated_name). The Series path in
    _normalize_name_text uses a vectorized equivalent instead - see there.

    NFKD splits an accented character into base letter + combining mark, so
    dropping the marks leaves plain ASCII ("ó" -> "o") rather than deleting
    the letter outright."""
    return ''.join(c for c in unicodedata.normalize('NFKD', str(value)) if not unicodedata.combining(c))


# Cross-source first-name variants for the SAME real player, as
# (variant, canonical) pairs. This is a hand-curated list on purpose.
#
# The obvious general fix - match on first initial + last name whenever the
# exact and suffix-stripped keys both miss - was built and then measured
# against the real 2025 files, and it is NOT SAFE: of the 44 snap-count
# names it "uniquely" resolved, at least 10 resolved to a DIFFERENT REAL
# PLAYER, including "Landon Jackson" -> "Lamar Jackson", "Terrell Edmunds"
# -> "Tremaine Edmunds", "Jeff Wilson Jr." -> "Jared Wilson" and "Carlos
# Washington Jr." -> "Casey Washington". Silently attributing one player's
# snaps to another is far worse than the missing row it replaces, so the
# looseness is spent only where a human has confirmed the two names are one
# person.
#
# Kenneth Walker III is the case that prompted this: the snap-count export
# calls him "Ken Walker III" while the weekly stats, rosters and every PFF
# export call him "Kenneth Walker III", so his snap share resolved to no
# data at all. The rest of the list came from the same scan of names that
# match on neither existing tier.
#
# To extend: add a pair here. Both directions are registered, so it doesn't
# matter which spelling a future source happens to use.
PLAYER_NAME_ALIASES = [
    ("Ken Walker III", "Kenneth Walker III"),
    ("Zach Carter", "Zachary Carter"),
    ("Nathan Carter", "Nate Carter"),
    ("Jake Hummel", "Jacob Hummel"),
    ("Joshua Palmer", "Josh Palmer"),
    ("Mitch Tinsley", "Mitchell Tinsley"),
    ("Dax Hill", "Daxton Hill"),
    ("Patrick Surtain II", "Pat Surtain II"),
    ("Folorunso Fatukasi", "Foley Fatukasi"),
    ("Cam Bynum", "Camryn Bynum"),
    ("Foyesade Oluokun", "Foye Oluokun"),
    ("Mike Danna", "Michael Danna"),
    ("Chris Roland-Wallace", "Christian Roland-Wallace"),
    ("JuJu Brents", "Julius Brents"),
    ("Joshua Metellus", "Josh Metellus"),
    ("Josh Uche", "Joshua Uche"),
    ("Scotty Miller", "Scott Miller"),
    ("T.J. Slaton Jr.", "Tedarrell Slaton"),
    ("Christopher Edmonds", "Chris Edmonds"),
    ("Chig Okonkwo", "Chigoziem Okonkwo"),
    ("Gabe Davis", "Gabriel Davis"),
    ("Marquise Brown", "Hollywood Brown"),
    ("Cam Akers", "Cameron Akers"),
    ("Tank Bigsby", "Tarik Bigsby"),
    ("Nick Westbrook-Ikhine", "Nicholas Westbrook-Ikhine"),
]


def _base_clean(series):
    """Lowercase + drop everything that isn't a letter. The shared tail of
    both key builders, after _normalize_name_text has done the repairs."""
    return _normalize_name_text(series).str.lower().str.replace('[^a-z]', '', regex=True)


def _strip_suffix(series):
    return _normalize_name_text(series).str.lower().str.replace(
        r'\s+(jr|sr|ii|iii|iv|v)\.?\s*$', '', regex=True)


def _build_alias_maps():
    """
    Two lookup tables built from PLAYER_NAME_ALIASES - one keyed on the
    exact-match form, one on the suffix-stripped form - so each tier of the
    app's existing two-tier match can canonicalize with its own key shape.

    Every pair is registered in BOTH directions, collapsing onto whichever
    spelling sorts first. The alternative (treating the second element as
    "the" canonical name) only works if every source in the app agrees on
    which one that is, and they demonstrably don't - the snap file and the
    stats file disagree about Kenneth Walker in one direction and about
    Josh Palmer in the other. Collapsing to an arbitrary-but-stable
    representative means both spellings land on the same key regardless.
    """
    exact_map, loose_map = {}, {}
    for variant, canonical in PLAYER_NAME_ALIASES:
        pair = pd.Series([variant, canonical])
        ex_a, ex_b = _base_clean(pair).tolist()
        lo_a, lo_b = _base_clean(_strip_suffix(pair)).tolist()
        ex_rep, lo_rep = min(ex_a, ex_b), min(lo_a, lo_b)
        exact_map[ex_a] = exact_map[ex_b] = ex_rep
        loose_map[lo_a] = loose_map[lo_b] = lo_rep
    return exact_map, loose_map


_ALIAS_EXACT, _ALIAS_LOOSE = _build_alias_maps()


def clean_name_exact(name_series):
    """
    Same cleaning as clean_name_for_merge but WITHOUT stripping suffixes -
    just lowercase + strip punctuation/spacing. This is the first-choice
    match key (see clean_name_for_merge below for why a second, looser key
    also exists) so two different players who share a base name but are
    distinguished by a real suffix - e.g. "Byron Murphy" (Vikings CB) vs
    "Byron Murphy II" (Seahawks DT) - don't collide into the same key.

    Names are repaired (HTML entities, accents - see _normalize_name_text)
    and run through PLAYER_NAME_ALIASES before the key is returned, so a
    curated cross-source first-name variant like "Ken Walker III" vs
    "Kenneth Walker III" produces one shared key here at TIER ONE rather
    than falling through to the looser tier that exists for suffixes.
    """
    keys = _base_clean(name_series)
    return keys.map(lambda k: _ALIAS_EXACT.get(k, k))


def clean_name_for_merge(name_series):
    # Strip common suffixes before stripping punctuation. Your source files
    # are inconsistent about including these - confirmed snap_counts_2025.csv
    # has "A.J. Terrell Jr." while roster_weekly_2025.csv has "A.J. Terrell"
    # (and the reverse happens too, e.g. Marvin Harrison Jr. is suffixed on
    # both sides but Kyle Pitts / Jessie Bates / Ray-Ray McCloud are only
    # suffixed in snap_counts) - 144 of 1,828 players in snap_counts_2025
    # carry a suffix, so this was silently zeroing out snap % for a real
    # chunk of the league, not just one player.
    #
    # IMPORTANT: this loose/stripped key is only meant as a FALLBACK for
    # when an exact match (clean_name_exact above) fails - using it as the
    # only key caused a different bug: "Byron Murphy" and "Byron Murphy II"
    # are two different real players, but both collapse to "byronmurphy"
    # once suffixes are stripped, which silently merged one player's data
    # onto the other. See the two-tier matching in load_year_data.
    #
    # Also canonicalizes through PLAYER_NAME_ALIASES (suffix-stripped
    # variant of the same table clean_name_exact uses), which is what
    # rescues a source that drops the suffix as well as changing the first
    # name - "Ken Walker" for "Kenneth Walker III" hits neither tier
    # otherwise.
    cleaned = _base_clean(_strip_suffix(name_series))
    return cleaned.map(lambda k: _ALIAS_LOOSE.get(k, k))


_NAME_SUFFIXES = {'jr', 'jr.', 'sr', 'sr.', 'ii', 'iii', 'iv', 'v'}


def build_last_name_index(full_name_pool):
    """
    Groups a pool of lowercased "first last[ suffix]" names by last name
    (skipping a trailing suffix like Jr./II/III so it doesn't get mistaken
    for the last name) for fast abbreviated-name lookups - see
    match_abbreviated_name. Stores each candidate's FULL first name (not
    just its first letter) - see match_abbreviated_name's docstring for why
    that distinction matters. Build once per table render, not per cell -
    O(1) average lookup afterward instead of a linear scan of the whole
    pool for every single cell.
    """
    index = {}
    for full_name in full_name_pool:
        # Same entity/accent repair the match keys get - the index is keyed
        # on a raw last name here rather than on a cleaned key, so without
        # it "Nuñez-Roches" and "Nunez-Roches" land in two different buckets
        # and an F.Last lookup finds neither reliably.
        parts = _strip_accents(html.unescape(str(full_name))).split()
        if not parts:
            continue
        last_idx = -2 if (parts[-1].lower() in _NAME_SUFFIXES and len(parts) > 2) else -1
        index.setdefault(parts[last_idx], []).append((parts[0], full_name))
    return index


def match_abbreviated_name(abbrev_name, last_name_index):
    """
    Bridges a "F.Last"-abbreviated name (roster_weekly's own 'player_name'
    convention - confirmed real, e.g. "P.Mahomes" - which
    data.transforms.fetch_intelligent_depth_chart prefers over the full
    'player_display_name' column) to a full "first last" name pool indexed
    by build_last_name_index. A first-name PREFIX can't be reversed to a
    full first name, so this matches on (first-name prefix, last-name)
    instead of a full-name string - the same reason PFF grades never
    matched anything in the depth chart table before this existed
    (confirmed empirically: 0/49 cells matched a real PFF grade on a live
    KC roster; 47/49 matched once this bridge was added, the other 2 being
    real long-snapper/deep-roster players PFF genuinely doesn't grade at
    all).

    Matches on the FULL prefix before the period, not just its first
    letter - nflverse itself lengthens the abbreviation (e.g. "Mi." vs
    "Ma.") specifically when two same-last-name, same-initial teammates
    would otherwise collide (confirmed real: Michael Wilson and Mack
    Wilson, both ARI, 2025 - "Mi.Wilson" and "Ma.Wilson"). Collapsing to a
    single first letter throws away exactly the information nflverse added
    to disambiguate them, and silently resolved one to the other.
    """
    abbrev_name = _strip_accents(html.unescape(str(abbrev_name)))
    if '.' not in abbrev_name:
        return None
    prefix, _, last = abbrev_name.partition('.')
    if not prefix or not last:
        return None
    prefix = prefix.lower()
    last = last.lower().strip()
    candidates = [
        full_name for cand_first, full_name in last_name_index.get(last, [])
        if cand_first.lower().startswith(prefix)
    ]
    return candidates[0] if candidates else None


def american_odds_to_prob(odds):
    """
    American odds -> implied probability. Standard sportsbook conversion;
    includes the book's own vig (not de-vigged against the other side of
    the market), which is fine here since this only feeds a single-sided
    "anytime TD" probability read, not a two-way price comparison.
    """
    odds = float(odds)
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)


def calculate_exact_age(birth_date_str, backup_age):
    try:
        if pd.isna(birth_date_str) or not birth_date_str:
            if pd.notna(backup_age) and backup_age != 0: return f"{int(float(backup_age))}y"
            return "--"
        bday = pd.to_datetime(birth_date_str)
        # date.today(), NOT a hardcoded date - this was frozen at a literal
        # date once, which silently made every displayed age drift wrong as
        # the season went on (an age is a today-relative fact).
        today = datetime.date.today()
        years = today.year - bday.year
        months = today.month - bday.month
        if today.day < bday.day: months -= 1
        if months < 0:
            years -= 1
            months += 12
        return f"{years}y {months}m"
    except:
        if pd.notna(backup_age) and backup_age != 0: return f"{int(float(backup_age))}y"
        return "--"


def parse_height(inches_raw):
    try:
        if pd.isna(inches_raw) or inches_raw == 0: return "--"
        val = int(float(inches_raw))
        return f"{val // 12}' {val % 12}\""
    except: return "--"


def parse_weight(lbs_raw):
    try:
        if pd.isna(lbs_raw) or lbs_raw == 0: return "--"
        return str(int(float(lbs_raw)))
    except: return "--"
