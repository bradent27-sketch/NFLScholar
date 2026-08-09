"""
Season-long market lines -> fantasy points, and the gap to our own model.

TWO THINGS ARE BUILT HERE, and they are the same arithmetic read in opposite
directions:

  A MARKET PROJECTION. A player's season-long over/unders are a stat line
  someone is willing to take money on. Scored under your league's settings by
  the same score_stats() every other number on the board goes through, that
  becomes a projected point total computed with no reference to this app's
  model at all - a genuinely independent second opinion, which is worth more
  than another opinion derived from the same rank curves.

  AN EDGE. Read the other way, the difference between our projection and the
  market's is a list of the players this model disagrees with the money
  about, sorted by how much.

WHAT A GAP ACTUALLY MEANS, because this is where a tool like this misleads
people. A large gap is NOT a bet. It is one of three things, and they are not
distinguishable from the number alone:

  1. The market knows something the model can't see - a camp report, a
     coordinator's comments, a snap-count trend from a joint practice.
     This is the common case, and it is why the board blends toward market
     ADP in the first place.
  2. The model is missing a stat the line depends on.
  3. A real disagreement worth acting on.

The ranking below is honest about which players it can even evaluate: a
market projection built from two of a receiver's five relevant stats is not
comparable to our full one, so coverage travels with every row and thin rows
are excluded from the edge list rather than quietly compared.

NOT A BET RECOMMENDER. This computes a difference between two projections.
It does not price a wager, account for the vig, or know your bankroll.
"""
import numpy as np
import pandas as pd

from data.draft_board import score_stats
from data.draft_projections import PROJECTED_STATS

# Stats that carry most of a season projection, per position. Used to measure
# COVERAGE - what share of the player's fantasy value the market lines
# actually span - so a player priced on receptions alone isn't compared
# against our full projection as though the two were the same quantity.
KEY_STATS = {
    'QB': ('passing_yards', 'passing_tds', 'passing_interceptions', 'rushing_yards'),
    'RB': ('rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards'),
    'WR': ('receiving_yards', 'receptions', 'receiving_tds'),
    'TE': ('receiving_yards', 'receptions', 'receiving_tds'),
}

# Below this share of a position's key stats, a market projection is too
# partial to compare. Two of a receiver's three, or three of a back's four.
MIN_COVERAGE = 0.6

# A market line is a MEDIAN (the point where the book wants equal money on
# both sides), while a fantasy projection is a MEAN. For right-skewed counting
# stats the mean sits above the median, so scoring the lines directly
# understates the expected total. Touchdowns are the worst offender - a
# roughly Poisson count where the tail is most of the fantasy value.
#
# These are deliberately modest. The honest version of this correction needs
# each stat's full distribution, which the lines don't carry; overreaching
# here would manufacture an edge out of an arithmetic assumption, which is
# exactly the failure this module is supposed to help detect.
MEDIAN_TO_MEAN = {
    'passing_tds': 1.02, 'rushing_tds': 1.05, 'receiving_tds': 1.05,
}


def market_stat_lines(props, season_only=True):
    """
    Normalized props -> one row per player carrying his scorable stats.

    Takes the MEDIAN line where a player has several for the same stat (two
    books, or a line that moved and was re-posted). Median rather than mean
    because a single stale or mistyped line shouldn't drag the number.
    """
    if props is None or props.empty:
        return pd.DataFrame()
    df = props.copy()
    if season_only:
        df = df[df['period'] == 'season']
    df = df[df['scorable'].fillna(False).astype(bool)]
    df = df[df['market'].isin(PROJECTED_STATS)]
    if df.empty:
        return pd.DataFrame()

    wide = (df.groupby(['player_key', 'market'])['line'].median().unstack('market'))
    meta = (df.sort_values('provider')
              .groupby('player_key')
              .agg(player=('player', 'first'), team=('team', 'first'),
                   position=('position', 'first'),
                   providers=('provider', lambda s: ', '.join(sorted(set(s)))),
                   n_lines=('line', 'size')))
    out = meta.join(wide).reset_index()
    for stat in PROJECTED_STATS:
        if stat not in out.columns:
            out[stat] = np.nan
    return out


def score_market_lines(market_rows, scoring, positions=None):
    """
    Market stat lines -> projected fantasy points, plus a coverage share.

    Missing stats are scored as ZERO, not imputed from our own model. That
    keeps the market projection independent - the moment a gap is filled with
    our number, comparing the two stops measuring anything. It also means an
    incomplete row reads LOW, which is exactly why coverage is computed
    alongside and why the edge list drops thin rows.
    """
    if market_rows is None or market_rows.empty:
        return pd.DataFrame()
    out = market_rows.copy()

    pos = out['position'].astype(str).str.upper()
    if positions is not None:
        # Position from the board where the provider didn't give one - books
        # are inconsistent about publishing it, and coverage can't be judged
        # without knowing which stats should have been there.
        filled = out['player_key'].map(positions)
        pos = pos.where(pos.isin(KEY_STATS), filled.astype(str).str.upper())
    out['position'] = pos

    scored = out.copy()
    for stat, factor in MEDIAN_TO_MEAN.items():
        if stat in scored.columns:
            scored[stat] = pd.to_numeric(scored[stat], errors='coerce') * factor
    scored['position'] = out['position']
    out['Market Pts'] = score_stats(scored.fillna(0.0), scoring,
                                    position_col='position').round(1)

    coverage = []
    for _, row in out.iterrows():
        keys = KEY_STATS.get(str(row['position']).upper())
        if not keys:
            coverage.append(np.nan)
            continue
        have = sum(1 for stat in keys if pd.notna(row.get(stat)))
        coverage.append(round(have / len(keys), 2))
    out['Coverage'] = coverage
    return out


def attach_board_player(market_scored, board):
    """
    Add a `board_player` column naming the board row each market line belongs
    to, matched in two tiers.

    THE EXACT KEY IS NOT ENOUGH, and this was caught by a real miss rather
    than anticipated: the board carries "Patrick Mahomes II" and the books
    write "Patrick Mahomes", so an exact-key join silently dropped him - and
    with him every Jr., Sr., II and III in the league, which is a large slice
    of the players anyone actually drafts.

    So: exact key first, then a suffix-stripped key for whatever is left.
    The loose pass is only allowed where the stripped key is UNIQUE ON BOTH
    SIDES, because that key is exactly what makes "Byron Murphy" and "Byron
    Murphy II" - two different real players - collide. An ambiguous fallback
    is left unmatched, which shows up as a visible miss instead of as one
    player's lines attached to another's projection.
    """
    from data.utils import clean_name_exact, clean_name_for_merge

    out = market_scored.copy()
    out['board_player'] = None
    if board is None or board.empty or out.empty:
        return out

    exact = dict(zip(clean_name_exact(board['Player']), board['Player']))
    out['board_player'] = out['player_key'].map(exact)

    missing = out['board_player'].isna()
    if not missing.any():
        return out

    board_loose = clean_name_for_merge(board['Player'])
    unique_loose = board_loose.value_counts()
    loose_map = {key: name for key, name in zip(board_loose, board['Player'])
                 if unique_loose.get(key, 0) == 1}

    market_loose = clean_name_for_merge(out['player'])
    market_counts = market_loose.value_counts()
    fallback = [
        loose_map.get(key) if market_counts.get(key, 0) == 1 else None
        for key in market_loose
    ]
    out.loc[missing, 'board_player'] = pd.Series(fallback, index=out.index)[missing]
    return out


def compare_to_board(board, market_scored, min_coverage=MIN_COVERAGE):
    """
    Join market projections onto the board and rank the disagreements.

    Returns (comparison, meta). `comparison` carries one row per player the
    market priced with enough coverage to judge, with:

        Proj Pts     our projection
        Market Pts   the same league's scoring applied to the market's lines
        Edge         ours minus theirs, in points
        Edge %       the same gap as a share of the market number, which is
                     what makes a 20-point gap on a QB comparable to a
                     20-point gap on a tight end

    Sorted by absolute Edge %, because the interesting rows are at BOTH ends:
    players we like far more than the market, and players we like far less.
    """
    meta = {'matched': 0, 'thin': 0, 'unmatched': 0}
    if board is None or board.empty or market_scored is None or market_scored.empty:
        return pd.DataFrame(), meta

    resolved = attach_board_player(market_scored, board)
    cols = [c for c in ('Player', 'Pos', 'Team', 'Proj Pts', 'ADP', 'ECR',
                        'Board Rank', 'Health') if c in board.columns]
    merged = resolved.dropna(subset=['board_player']).merge(
        board[cols], left_on='board_player', right_on='Player', how='inner')
    meta['unmatched'] = int(len(market_scored) - len(merged))
    if merged.empty:
        return pd.DataFrame(), meta

    thin = merged['Coverage'].fillna(0) < min_coverage
    meta['thin'] = int(thin.sum())
    merged = merged[~thin]
    meta['matched'] = int(len(merged))
    if merged.empty:
        return pd.DataFrame(), meta

    ours = pd.to_numeric(merged['Proj Pts'], errors='coerce')
    theirs = pd.to_numeric(merged['Market Pts'], errors='coerce')
    merged['Edge'] = (ours - theirs).round(1)
    merged['Edge %'] = np.where(theirs > 0, (ours - theirs) / theirs * 100, np.nan).round(1)

    merged = merged.sort_values('Edge %', key=lambda s: s.abs(), ascending=False)
    keep = ['Player', 'Pos', 'Team', 'Proj Pts', 'Market Pts', 'Edge', 'Edge %',
            'Coverage', 'ADP', 'ECR', 'Board Rank', 'providers', 'n_lines']
    return merged[[c for c in keep if c in merged.columns]].reset_index(drop=True), meta


def blend_market_into_projection(board, market_scored, weight=0.0,
                                 min_coverage=MIN_COVERAGE):
    """
    Move Proj Pts a chosen fraction of the way toward the market's number.

    DEFAULTS TO OFF (weight 0.0), and that is a considered default rather
    than caution. The board ALREADY blends toward the market once, through
    ADP and ECR in apply_market_blend. Folding book lines in here as well
    prices the same market opinion into the same board twice, and the second
    dose is invisible - it arrives inside the projection, upstream of the
    valuation, where nothing downstream can tell it apart from the model's
    own output.

    It is exposed because a market projection built from real stat lines is
    genuinely better evidence than a consensus RANK, and someone who
    understands the double-count may reasonably want some of it. Only players
    with enough coverage move; everyone else keeps our number, so turning
    this up never silently marks down a player the market simply didn't post
    lines for.
    """
    if not weight or board is None or board.empty or market_scored is None or market_scored.empty:
        return board, 0

    usable = market_scored[market_scored['Coverage'].fillna(0) >= min_coverage]
    if usable.empty:
        return board, 0

    # Same two-tier match the comparison uses, so the blend moves exactly the
    # players the comparison says it can see - if these two disagreed, the
    # Edge column would be damped for players the table claims are untouched.
    usable = attach_board_player(usable, board).dropna(subset=['board_player'])
    if usable.empty:
        return board, 0

    lookup = dict(zip(usable['board_player'], pd.to_numeric(usable['Market Pts'],
                                                            errors='coerce')))
    out = board.copy()
    market = out['Player'].map(lookup)
    ours = pd.to_numeric(out['Proj Pts'], errors='coerce')
    moved = market.notna() & ours.notna()
    weight = float(np.clip(weight, 0.0, 1.0))
    out.loc[moved, 'Proj Pts'] = (
        (1 - weight) * ours[moved] + weight * market[moved]).round(1)
    return out, int(moved.sum())
