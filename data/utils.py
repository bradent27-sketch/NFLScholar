"""
Pure, dependency-free helpers shared by data/loaders.py and data/transforms.py
(percentile ranking, name-matching, bio-field formatting). Kept in their own
module so loaders.py and transforms.py can each depend on this without a
circular import between the two of them.
"""
import datetime
import pandas as pd


def calculate_percentile(df, col_name, ascending=True):
    if not df.empty and col_name in df.columns:
        return df[col_name].rank(pct=True, ascending=ascending) * 100
    return pd.Series([0]*len(df), index=df.index)


def get_val(row, col, fmt="{}"):
    if pd.isna(row.get(col)) or str(row.get(col)).strip() == '': return "--"
    try: return fmt.format(float(row[col]))
    except: return fmt.format(row[col])


def clean_name_exact(name_series):
    """
    Same cleaning as clean_name_for_merge but WITHOUT stripping suffixes -
    just lowercase + strip punctuation/spacing. This is the first-choice
    match key (see clean_name_for_merge below for why a second, looser key
    also exists) so two different players who share a base name but are
    distinguished by a real suffix - e.g. "Byron Murphy" (Vikings CB) vs
    "Byron Murphy II" (Seahawks DT) - don't collide into the same key.
    """
    return name_series.astype(str).str.lower().str.replace('[^a-z]', '', regex=True)


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
    cleaned = name_series.astype(str).str.lower()
    cleaned = cleaned.str.replace(r'\s+(jr|sr|ii|iii|iv|v)\.?\s*$', '', regex=True)
    cleaned = cleaned.str.replace('[^a-z]', '', regex=True)
    return cleaned


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
        parts = str(full_name).split()
        if not parts:
            continue
        last_idx = -2 if (parts[-1] in _NAME_SUFFIXES and len(parts) > 2) else -1
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
