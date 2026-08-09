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

# Stats that carry most of a season projection, per position, with a typical
# per-season quantity for each. The quantities turn coverage into a
# POINTS-WEIGHTED measure rather than a count, which matters more than it
# sounds - see below.
KEY_STATS = {
    'QB': {'passing_yards': 4000, 'passing_tds': 26, 'passing_interceptions': 11,
           'rushing_yards': 250, 'rushing_tds': 2},
    'RB': {'rushing_yards': 900, 'rushing_tds': 7, 'receptions': 45,
           'receiving_yards': 350, 'receiving_tds': 2},
    'WR': {'receptions': 70, 'receiving_yards': 950, 'receiving_tds': 6},
    'TE': {'receptions': 55, 'receiving_yards': 600, 'receiving_tds': 4},
}

# COVERAGE IS WEIGHTED BY FANTASY POINTS, NOT BY HOW MANY STATS ARE PRESENT,
# and the reason is a trap the first version walked straight into.
#
# Underdog's real season-long NFL board posts receiving yards and receiving
# TDs for a receiver but NOT receptions. Counted as stats that is 2 of 3 -
# 0.67 coverage, comfortably past any sane threshold - so the player would be
# compared against our full projection. But in a PPR league receptions are
# roughly 40% of a receiver's points, so his market total comes out ~40% low
# and he shows a huge "edge" that is pure arithmetic. Every receiver on the
# board, all in the same direction, which is exactly what a real edge is
# supposed to look like.
#
# Weighting by each stat's share of fantasy points under THIS league's
# scoring fixes it at the root and is league-aware for free: in a standard
# league a missing receptions line barely dents coverage, and in full PPR it
# halves it. The same rule handles a QB priced without his rushing line.
MIN_COVERAGE = 0.7

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


def market_stat_lines(props, season_only=True, board=None):
    """
    Normalized props -> one row per player carrying his scorable stats.

    Takes the MEDIAN line where a player has several for the same stat (two
    books, or a line that moved and was re-posted). Median rather than mean
    because a single stale or mistyped line shouldn't drag the number.
    """
    if props is None or props.empty:
        return pd.DataFrame()
    # Identity is settled BEFORE grouping, so two books spelling one player
    # differently become one row that averages across both.
    df = canonicalize_props(props, board) if board is not None else props.copy()
    if season_only:
        df = df[df['period'] == 'season']
    df = df[df['scorable'].fillna(False).astype(bool)]
    df = df[df['market'].isin(PROJECTED_STATS)]
    if df.empty:
        return pd.DataFrame()

    # MEDIAN ACROSS PROVIDERS, PER STAT - which is where the multi-book
    # averaging actually happens, and it happens at the STAT level rather
    # than the projection level on purpose. If Underdog prices receiving
    # yards and PrizePicks prices receptions, this keeps BOTH: the player
    # ends up with more of his line coming from the market instead of from
    # us. Averaging two finished projections would have thrown that away.
    # With two books the median is the mean; with one, it is that book.
    wide = (df.groupby(['player_key', 'market'])['line'].median().unstack('market'))
    meta = (df.sort_values('provider')
              .groupby('player_key')
              .agg(player=('player', 'first'), team=('team', 'first'),
                   position=('position', 'first'),
                   providers=('provider', lambda s: ', '.join(sorted(set(s)))),
                   Books=('provider', lambda s: len({p.split(' (')[0] for p in s})),
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

    out['Coverage'] = _points_weighted_coverage(out, scoring)
    return out


def _stat_point_weights(position, scoring):
    """
    {stat: share of a typical season's fantasy points} for a position.

    Built by scoring a typical season line one stat at a time under THIS
    league's settings, so the weights follow the rules rather than a
    hardcoded guess - full PPR makes receptions a third of a receiver's
    value, standard makes them nothing, and coverage should say so.
    """
    typical = KEY_STATS.get(str(position).upper())
    if not typical:
        return {}
    contributions = {}
    for stat, quantity in typical.items():
        row = pd.DataFrame([{**{s: 0.0 for s in typical}, stat: quantity,
                             'position': str(position).upper()}])
        contributions[stat] = abs(float(score_stats(row, scoring,
                                                    position_col='position').iloc[0]))
    total = sum(contributions.values())
    if total <= 0:
        return {stat: 1.0 / len(typical) for stat in typical}
    return {stat: value / total for stat, value in contributions.items()}


def _points_weighted_coverage(rows, scoring):
    """Share of a position's typical fantasy points the market actually priced."""
    weights_by_pos = {}
    values = []
    for _, row in rows.iterrows():
        position = str(row.get('position') or '').upper()
        if position not in KEY_STATS:
            values.append(np.nan)
            continue
        if position not in weights_by_pos:
            weights_by_pos[position] = _stat_point_weights(position, scoring)
        weights = weights_by_pos[position]
        covered = sum(weight for stat, weight in weights.items()
                      if pd.notna(row.get(stat)))
        values.append(round(covered, 2))
    return values


def build_book_projection(board, market_scored, scoring):
    """
    A COMPLETE projection: the book's number for every stat it priced, ours
    for everything it didn't.

    This is the one you can actually draft off. The other two market numbers
    in this module are deliberately partial and answer a narrower question:

        Market Pts       only the stats the book priced, nothing else. An
                         independent second opinion, and NOT comparable to a
                         full projection.
        Ours (matched)   our numbers over that same narrow set, so the two
                         can be differenced honestly.
        Book Proj        THIS. Book where the book spoke, us where it didn't,
                         so the total covers a whole player and sits on the
                         same scale as Proj Pts.

    WHERE THE BOOK SPOKE, IT WINS OUTRIGHT - no averaging with our number.
    A posted line is a price someone will take money on; splitting the
    difference with a model would throw away the thing that makes it worth
    having. Everything else falls back to our projection untouched, which is
    what makes the result complete: Underdog posts no receptions market at
    all, so a receiver keeps our reception estimate and his total stays a
    real full-PPR number instead of collapsing by a third.

    THE TWO BASES ARE NOT IDENTICAL, and it's worth knowing which way it
    leans. A season-long line embeds the chance the player misses games,
    while our projection assumes closer to a full slate - measured at about
    4% across every position. So a Book Proj runs slightly conservative in
    proportion to how much of the player the book actually priced.

    Returns a frame with player_key / board_player / Book Proj / Book Stats
    (how many stats came from the book) / Book Share (their share of the
    projected points).
    """
    if board is None or board.empty or market_scored is None or market_scored.empty:
        return pd.DataFrame()

    stat_cols = [c for c in PROJECTED_STATS if c in board.columns]
    if not stat_cols:
        return pd.DataFrame()

    resolved = attach_board_player(market_scored, board).dropna(subset=['board_player'])
    if resolved.empty:
        return pd.DataFrame()

    cols = ['Player', 'Pos', 'Proj Pts'] + stat_cols
    merged = resolved.merge(board[[c for c in cols if c in board.columns]],
                            left_on='board_player', right_on='Player', how='inner',
                            suffixes=('', '_ours'))
    if merged.empty:
        return pd.DataFrame()

    rows, book_counts, book_only = [], [], []
    for _, row in merged.iterrows():
        line = {'position': str(row.get('Pos') or '').upper()}
        used = 0
        priced_line = dict(line)
        for stat in PROJECTED_STATS:
            ours_col = f'{stat}_ours' if f'{stat}_ours' in merged.columns else stat
            our_value = pd.to_numeric(row.get(ours_col), errors='coerce')
            our_value = float(our_value) if pd.notna(our_value) else 0.0
            book_value = pd.to_numeric(row.get(stat), errors='coerce') if stat in merged.columns else np.nan

            if pd.notna(book_value):
                # A line is a MEDIAN; a projection is a MEAN. For the skewed
                # counting stats the mean sits above, and touchdowns are the
                # worst of them - see MEDIAN_TO_MEAN.
                value = float(book_value) * MEDIAN_TO_MEAN.get(stat, 1.0)
                used += 1
                priced_line[stat] = value
            else:
                value = our_value
                priced_line[stat] = 0.0
            line[stat] = value
        rows.append(line)
        book_counts.append(used)
        book_only.append(priced_line)

    frame = pd.DataFrame(rows)
    out = pd.DataFrame({
        'player_key': merged['player_key'].to_numpy(),
        'board_player': merged['board_player'].to_numpy(),
        'Pos': merged['Pos'].to_numpy(),
        'Proj Pts': pd.to_numeric(merged['Proj Pts'], errors='coerce').to_numpy(),
        'Book Proj': score_stats(frame, scoring, position_col='position').round(1).to_numpy(),
        'Book Stats': book_counts,
    })
    # How much of the projected total the book is actually responsible for -
    # the honest read on whether a Book Proj is a market number with a little
    # of ours in it, or ours with a little of the market in it.
    #
    # Clamped at 1.0. The raw ratio can exceed it legitimately: if the stats
    # we fill in are net NEGATIVE - interceptions and fumbles, which no book
    # posts a season-long line for - the book's contribution is larger than
    # the total it sits inside. Real arithmetic, but "103%" reads as a bug to
    # anyone glancing at the column, and everything above 1.0 means the same
    # thing anyway: essentially all of this number came from the market.
    from_book = score_stats(pd.DataFrame(book_only), scoring,
                            position_col='position').to_numpy()
    total = pd.to_numeric(out['Book Proj'], errors='coerce').to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        share = np.where(total > 0, from_book / total, np.nan)
    out['Book Share'] = np.round(np.clip(share, 0.0, 1.0), 2)
    out['Book Δ'] = (out['Book Proj'] - out['Proj Pts']).round(1)
    return out


def attach_book_projection(board, book_projection):
    """Put Book Proj and Book Δ on the board, blank where the book was silent."""
    if board is None or board.empty or book_projection is None or book_projection.empty:
        return board
    out = board.copy()
    for column in ('Book Proj', 'Book Δ', 'Book Share'):
        if column in book_projection.columns:
            out[column] = out['Player'].map(
                dict(zip(book_projection['board_player'], book_projection[column])))
    return out


def resolve_names_to_board(names, board):
    """
    Map any series of book-written player names onto the board's own spelling.

    Two tiers, same as load_year_data: exact key first, then a suffix-stripped
    key, and the loose pass only where that key is UNIQUE ON BOTH SIDES so
    "Byron Murphy" and "Byron Murphy II" never collapse into each other.
    Unresolvable names map to None rather than to a guess.
    """
    from data.utils import clean_name_exact, clean_name_for_merge

    names = pd.Series(list(names), dtype=object)
    if board is None or board.empty or names.empty:
        return pd.Series([None] * len(names), index=names.index, dtype=object)

    exact = dict(zip(clean_name_exact(board['Player']), board['Player']))
    resolved = clean_name_exact(names).map(exact)

    missing = resolved.isna()
    if missing.any():
        board_loose = clean_name_for_merge(board['Player'])
        board_counts = board_loose.value_counts()
        loose_map = {key: name for key, name in zip(board_loose, board['Player'])
                     if board_counts.get(key, 0) == 1}
        our_loose = clean_name_for_merge(names)

        # AMBIGUITY IS COUNTED ONLY AMONG NAMES STILL UNRESOLVED, which is
        # the whole subtlety. Counting across every name made the two books'
        # spellings of one player veto each other: PrizePicks writes "James
        # Cook III" and Underdog writes "James Cook", both strip to
        # "jamescook", so the key looked ambiguous and BOTH were refused -
        # even though the suffixed one had already matched the board exactly
        # and was therefore not in question at all.
        #
        # A name that already found its board row is not a competing
        # candidate. What still has to be refused is two UNRESOLVED names
        # sharing a stripped key, which is the real "Byron Murphy" vs "Byron
        # Murphy II" case.
        unresolved_counts = our_loose[missing].value_counts()
        fallback = pd.Series(
            [loose_map.get(key) if unresolved_counts.get(key, 0) == 1 else None
             for key in our_loose], index=names.index, dtype=object)
        resolved = resolved.where(~missing, fallback)
    return resolved


def canonicalize_props(props, board):
    """
    Rewrite each prop's player name to the board's spelling, so the same
    player written two ways by two books becomes ONE player.

    THIS HAS TO HAPPEN BEFORE ANY GROUPING. PrizePicks writes "James Cook
    III" and Underdog writes "James Cook"; left alone that is two rows, which
    means the books never average against each other, the Books count reads 1
    for a player both of them priced, and the loose board match then refuses
    BOTH of them as ambiguous because two market rows share one stripped key.
    Kyle Pitts and Omar Cooper failed the same way.

    Names the board doesn't know keep their original spelling - a punter the
    book prices and we don't rank should stay visible, not vanish.
    """
    if props is None or props.empty or board is None or board.empty:
        return props
    from data.utils import clean_name_exact

    out = props.copy()
    unique = pd.Series(sorted(set(out['player'].astype(str))), dtype=object)
    mapping = dict(zip(unique, resolve_names_to_board(unique, board)))
    canonical = out['player'].astype(str).map(mapping)
    out['player'] = canonical.where(canonical.notna(), out['player'])
    out['player_key'] = clean_name_exact(out['player'])
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


def compare_to_board(board, market_scored, scoring, min_coverage=MIN_COVERAGE):
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
    stat_cols = [c for c in PROJECTED_STATS if c in board.columns]
    merged = resolved.dropna(subset=['board_player']).merge(
        board[cols + stat_cols], left_on='board_player', right_on='Player',
        how='inner', suffixes=('', '_ours'))
    meta['unmatched'] = int(len(market_scored) - len(merged))
    if merged.empty:
        return pd.DataFrame(), meta

    thin = merged['Coverage'].fillna(0) < min_coverage
    meta['thin'] = int(thin.sum())
    merged = merged[~thin]
    meta['matched'] = int(len(merged))
    if merged.empty:
        return pd.DataFrame(), meta

    # LIKE FOR LIKE, and this is the whole correctness of the comparison.
    #
    # A market total only spans the stats that book actually posted a line
    # for. Underdog's real season-long NFL board carries receiving yards and
    # receiving TDs but NO receptions, so a receiver's market total is
    # missing roughly a fifth of his half-PPR value and a third of his full-
    # PPR value - through no error, that market simply doesn't exist.
    #
    # Compared against our FULL projection, every receiver then showed a +40%
    # to +60% "edge", all in the same direction. That is not an edge, it is a
    # units mismatch, and it is the most dangerous possible failure here
    # because a uniform positive bias is exactly what a genuine market
    # inefficiency would look like.
    #
    # So our side is re-scored across only the stats the market priced for
    # THAT player. Both numbers then describe the same quantity and the
    # difference means something.
    # Needs the board's raw stat columns to re-score against. Without them
    # every matched-scope total would come out 0 and every edge would read
    # hugely negative - so fall back to the whole-projection comparison and
    # SAY SO in meta, rather than reporting a silently broken number.
    if stat_cols:
        merged['Ours (matched)'] = _score_matched_scope(merged, scoring)
        meta['scope'] = 'matched'
    else:
        merged['Ours (matched)'] = pd.to_numeric(merged['Proj Pts'], errors='coerce')
        meta['scope'] = 'full projection (board carried no stat columns)'
    ours = pd.to_numeric(merged['Ours (matched)'], errors='coerce')
    theirs = pd.to_numeric(merged['Market Pts'], errors='coerce')
    merged['Edge'] = (ours - theirs).round(1)
    merged['Edge %'] = np.where(theirs > 0, (ours - theirs) / theirs * 100, np.nan).round(1)

    # The complete hybrid rides along, so one table carries both the honest
    # like-for-like difference AND the number you'd actually draft off.
    book = build_book_projection(board, market_scored, scoring)
    if not book.empty:
        for column in ('Book Proj', 'Book Δ', 'Book Share'):
            merged[column] = merged['board_player'].map(
                dict(zip(book['board_player'], book[column])))

    merged = merged.sort_values('Edge %', key=lambda s: s.abs(), ascending=False)
    keep = ['Player', 'Pos', 'Team', 'Proj Pts', 'Book Proj', 'Book Δ', 'Book Share',
            'Books', 'Ours (matched)', 'Market Pts', 'Edge', 'Edge %', 'Coverage',
            'ADP', 'ECR', 'Board Rank', 'providers', 'n_lines']
    return merged[[c for c in keep if c in merged.columns]].reset_index(drop=True), meta


def _score_matched_scope(merged, scoring):
    """
    Our projection re-scored over only the stats the market priced per player.

    The market frame and the board frame both carry the same stat column
    names, so the merge suffixed the board's copies with `_ours`. A stat is
    "priced" when the MARKET side is non-null; our value for it is then taken
    from the board side and everything else is zeroed.
    """
    values = []
    for _, row in merged.iterrows():
        line = {'position': str(row.get('Pos') or '').upper()}
        for stat in PROJECTED_STATS:
            ours_col = f'{stat}_ours' if f'{stat}_ours' in merged.columns else stat
            priced = stat in merged.columns and pd.notna(row.get(stat))
            our_value = pd.to_numeric(row.get(ours_col), errors='coerce')
            line[stat] = float(our_value) if (priced and pd.notna(our_value)) else 0.0
        values.append(round(float(score_stats(pd.DataFrame([line]), scoring,
                                              position_col='position').iloc[0]), 1))
    return values


def blend_market_into_projection(board, market_scored, weight=0.0,
                                 min_coverage=MIN_COVERAGE, scoring=None):
    """
    Move Proj Pts a chosen fraction of the way toward the BOOK PROJECTION.

    Toward Book Proj, not toward Market Pts, and the difference is not a
    detail. Market Pts covers only the stats the book priced, so blending
    toward it dragged every receiver down by roughly his whole receptions
    total - it was moving the board toward a number that was never a
    projection of a whole player. Book Proj is complete (book where the book
    spoke, ours elsewhere), so it sits on the same scale as Proj Pts and a
    50% blend means what it looks like.

    DEFAULTS TO OFF (weight 0.0), and that is a considered default rather
    than caution. The board ALREADY blends toward the market once, through
    ADP and ECR in apply_market_blend. Folding book lines in here as well
    prices the same market opinion into the same board twice, and the second
    dose is invisible - it arrives inside the projection, upstream of the
    valuation, where nothing downstream can tell it apart from the model's
    own output.

    Only players the book actually priced move; everyone else keeps our
    number, so turning this up never marks down a player the book was simply
    silent about.
    """
    if not weight or board is None or board.empty or market_scored is None or market_scored.empty:
        return board, 0
    if scoring is None:
        return board, 0

    usable = market_scored[market_scored['Coverage'].fillna(0) >= min_coverage]
    if usable.empty:
        return board, 0

    book = build_book_projection(board, usable, scoring)
    if book.empty:
        return board, 0

    lookup = dict(zip(book['board_player'], pd.to_numeric(book['Book Proj'],
                                                          errors='coerce')))
    out = board.copy()
    market = out['Player'].map(lookup)
    ours = pd.to_numeric(out['Proj Pts'], errors='coerce')
    moved = market.notna() & ours.notna()
    weight = float(np.clip(weight, 0.0, 1.0))
    out.loc[moved, 'Proj Pts'] = (
        (1 - weight) * ours[moved] + weight * market[moved]).round(1)
    return out, int(moved.sum())
